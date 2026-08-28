"""Relational, legal, and derived-value invariants for TremoraStore snapshots.

The stored alignment and window tables are caches, not authorities.  Their
complete contents are regenerated from immutable source tables and persisted
generation plans before a snapshot can be published or accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Mapping
from itertools import pairwise
from pathlib import PurePosixPath
from urllib.parse import urlparse

import pyarrow as pa

from .alignment_index import AlignmentError, build_frame_imu_index
from .clock_map import ClockMapError, ClockSegment, PiecewiseClockMap
from .schema import (
    SCHEMA_FACTORIES,
    QualityBits,
    logical_schema_contract,
)
from .window_index import (
    ContinuitySegment,
    WindowIndexError,
    build_window_index,
)


class StoreInvariantError(ValueError):
    """Raised when a snapshot violates the canonical storage contract."""


RESERVED_PROVENANCE_KEYS = frozenset({
    "snapshot_id",
    "recording_id",
    "schema_version",
    "clock_map_id",
    "window_policy_id",
    "creation_timestamp_utc",
})

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_ID = re.compile(r"[0-9a-f]{7,64}\Z")
_ALLOWED_ACCESS = frozenset({
    "PUBLIC",
    "PUBLIC_ZENODO_RECORD",
    "ACCESS_GRANTED",
    "GENERATED_LOCALLY",
})
_ALLOWED_SOURCE_KINDS = frozenset({
    "SYNTHETIC_GENERATED",
    "PUBLIC_DATASET",
    "CONTROLLED_PUBLIC_DERIVATIVE",
})
_ALLOWED_VALIDATION_ROLES = frozenset({
    "SYNTHETIC_REGRESSION_ONLY",
    "INVENTORY_ONLY_UNVERIFIED_SYNC",
    "SYSTEMS_CORRECTNESS",
    "SCALE_BENCHMARK",
    "IRREGULARITY_STRESS",
    "IMU_ONLY_SCHEMA_SPECTRAL",
})
_RELEASE_PERMITTED_USES = frozenset({
    "SOFTWARE_TESTING_ONLY",
    "RESEARCH_AND_DERIVED_BENCHMARK_ARTIFACTS_WITH_ATTRIBUTION",
    "RESEARCH_AND_DERIVED_BENCHMARK_ARTIFACTS",
})
_LOCAL_ONLY_PERMITTED_USES = frozenset({
    "LOCAL_RESEARCH_ANALYSIS_ONLY",
    "LOCAL_SYSTEMS_BENCHMARKING_ONLY",
})
_RELEASE_DERIVED_POLICIES = frozenset({
    "TEST_ARTIFACTS_MUST_NOT_BE_EMPIRICAL_EVIDENCE",
    "CC_BY_4_0_ATTRIBUTION_REQUIRED",
    "DERIVED_ARTIFACT_RELEASE_PERMITTED_WITHOUT_SOURCE_REDISTRIBUTION",
})
_LOCAL_ONLY_DERIVED_POLICIES = frozenset({
    "LOCAL_ANALYSIS_ONLY_NO_DERIVED_RELEASE",
})
_ALLOWED_MAPPING_STATUS = frozenset({"VALID", "UNRESOLVED", "REJECTED"})
_ALLOWED_DECODE_STATUS = frozenset({
    "OK",
    "DECODE_FAILURE",
    "MISSING_TIMESTAMP",
})
_KNOWN_QUALITY_MASK = sum(int(value) for value in QualityBits)
_VIDIMU_SOURCE_FILE_KEYS = frozenset({
    "video", "video_pose", "imu_quaternion",
})
_VIDIMU_SELECTION_FIELDS = frozenset({
    "release_version", "dataset_archive", "dataset_archive_sha256",
    "dataset_subtree", "video_archive", "video_archive_sha256",
    "video_subtree", "inventory_scope", "archive_digest_binding",
})
_VIDIMU_EXCLUSION_FIELDS = frozenset({
    "archive", "relative_path", "reason", "sha256",
})
_VIDIMU_NPOSE_REASON = "OFFICIAL_NPOSE_COMPANION_NOT_CANONICAL_PAIR_INPUT"
_VIDIMU_STO_REASON = (
    "OFFICIAL_STO_PROCESSED_COMPANION_NOT_CANONICAL_PAIR_INPUT"
)
_VIDIMU_MOT_REASON = (
    "OFFICIAL_MOT_PROCESSED_COMPANION_NOT_CANONICAL_PAIR_INPUT"
)
_VIDIMU_IK_MOT_REASON = (
    "OFFICIAL_IK_MOT_COMPANION_NOT_CANONICAL_PAIR_INPUT"
)
_VIDIMU_IK_ORIENTATION_REASON = (
    "OFFICIAL_IK_ORIENTATION_ERRORS_COMPANION_NOT_CANONICAL_PAIR_INPUT"
)
_VIDIMU_FULLSIZE_OUT_REASON = (
    "OFFICIAL_FULLSIZE_MP4_OUT_COMPANION_NOT_CANONICAL_PAIR_INPUT"
)
_VIDIMU_RELEASE_RESIDUE_REASON_PREFIX = (
    "OFFICIAL_V2_NONCANONICAL_RESIDUE_MAPPED_TO_"
)
_VIDIMU_S48_MOT_RECORDINGS = frozenset({
    "S48_A01_T01",
    "S48_A02_T02",
    "S48_A03_T02",
    "S48_A04_T01",
    "S48_A05_T02",
    "S48_A06_T02",
    "S48_A07_T02",
    "S48_A08_T01",
    "S48_A09_T01",
    "S48_A10_T02",
    "S48_A11_T02",
    "S48_A12_T02",
    "S48_A13_T02",
})
_VIDIMU_KNOWN_DATA_RESIDUES = {
    "S41_A03_T01": frozenset({
        "dataset/videoandimus/S41/S41_A03_P01.csv",
        "dataset/videoandimus/S41/S41_A03_P01_Npose.csv",
    }),
    "S49_A13_T01": frozenset({
        "dataset/videoandimus/S49/S49_A13_T01V2_Npose.csv",
    }),
}
_VIDIMU_KNOWN_VIDEO_RESIDUES = {
    "S49_A13_T01": {
        "videosbodytrack/S49/S49_A13_T01V2_Npose_pose.mp4": frozenset({
            "videosmallsize", "videosfullsize",
        }),
        "videosbodytrack/S49/S49_A13_T01V2_Npose.mp4.out": frozenset({
            "videosfullsize",
        }),
    },
    "S25_A02_T01": {
        "videosbodytrack/S24/S25_A02_T01_pose.mp4": frozenset({
            "videosmallsize", "videosfullsize",
        }),
        "videosbodytrack/S24/S25_A02_T01.mp4.out": frozenset({
            "videosfullsize",
        }),
    },
}
_REQUIRED_PROVENANCE_STRINGS = frozenset({
    "source_kind", "source_dataset", "source_dataset_version",
    "source_recording_id", "source_record_uri", "license_id", "license_uri",
    "license_terms_sha256", "source_access_status",
    "source_redistribution_status", "derived_artifact_policy",
    "permitted_use", "validation_role", "use_decision",
    "artifact_release_status", "ingestion_commit",
    "ingestion_software_version", "cv_estimator_version",
    "observability_policy_id",
})
_RECORDING_IDENTITY_FIELDS = frozenset({
    "stored_recording_id", "source_recording_id", "mapping_method",
})
_STREAM_SEMANTICS_FIELDS = frozenset({
    "schema_version", "video_streams", "imu_streams",
})
_VIDEO_SEMANTICS_FIELDS = frozenset({
    "recording_id", "video_stream_id",
    "source_keypoint_convention", "stored_keypoint_convention",
    "source_motion_vector_convention", "stored_motion_vector_convention",
    "source_palm_orientation_convention",
    "stored_palm_orientation_convention",
    "source_hand_scale_convention", "stored_hand_scale_convention",
    "canonicalization_transform_id", "canonicalization_software_version",
})
_IMU_SEMANTICS_FIELDS = frozenset({
    "recording_id", "stream_id", "body_location", "payload_kind",
    "source_acceleration_unit", "stored_acceleration_unit",
    "source_angular_velocity_unit", "stored_angular_velocity_unit",
    "source_quaternion_convention", "stored_quaternion_convention",
    "source_device_frame_convention", "stored_device_frame_convention",
    "canonicalization_transform_id", "canonicalization_software_version",
})
_NOT_PRESENT = "NOT_PRESENT"
_VIDIMU_EXTENSION_FIELDS = frozenset({
    "source_record_id", "source_record_url", "source_record_doi",
    "source_concept_doi", "source_files", "source_exclusion_ledger",
    "source_exclusion_ledger_sha256", "source_archive_selection",
    "source_archive_selection_sha256", "source_identity_sha256",
    "source_license", "source_license_spdx", "source_license_url",
    "source_terms_sha256", "source_terms_hash_origin",
    "redistribution_status", "recording_inventory_complete",
    "release_inventory_complete", "release_inventory_scope",
    "released_imu_payload", "video_nominal_fps", "imu_nominal_hz",
    "raw_accelerometer_axes_available", "raw_gyroscope_axes_available",
    "clock_truth_status", "allowed_validation_role",
    "prohibited_interpretation",
})
_COMMON_PROVENANCE_FIELDS = _REQUIRED_PROVENANCE_STRINGS | frozenset({
    "provenance_schema_version", "source_file_hashes",
    "local_analysis_allowed", "source_redistribution_allowed",
    "derived_artifact_release_allowed", "alignment_generation_parameters",
    "window_generation_parameters", "stream_semantics", "recording_identity",
})
_CANONICAL_ROW_SORT_KEYS = {
    "frame_index": ("recording_id", "video_stream_id", "canonical_ordinal"),
    "cv_estimates": ("recording_id", "video_stream_id", "canonical_ordinal"),
    "imu_samples": ("recording_id", "stream_id", "canonical_ordinal"),
    "clock_map": ("recording_id", "stream_id", "acquisition_ordinal"),
    "frame_imu_index": (
        "recording_id", "video_stream_id", "imu_stream_id",
        "frame_canonical_ordinal",
    ),
    "window_index": ("window_id",),
    "window_rejections": ("candidate_window_id",),
}


def canonical_schema(name: str) -> pa.Schema:
    """Return the one supported schema for a named snapshot table."""

    try:
        return SCHEMA_FACTORIES[name]()
    except KeyError as exc:
        raise StoreInvariantError(f"unknown TremoraStore table: {name}") from exc


def schemas_equal(left: pa.Schema, right: pa.Schema) -> bool:
    """Compare logical schemas, including Tremora metadata."""

    return logical_schema_contract(left) == logical_schema_contract(right)


def _required_string(value: Mapping[str, object], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result.strip():
        raise StoreInvariantError(f"provenance field {field!r} is required")
    return result


def _required_bool(value: Mapping[str, object], field: str) -> bool:
    result = value.get(field)
    if not isinstance(result, bool):
        raise StoreInvariantError(f"provenance field {field!r} must be boolean")
    return result


def _validate_uri(value: str, field: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https", "urn"}:
        raise StoreInvariantError(
            f"provenance field {field!r} must use http, https, or urn")
    if parsed.scheme in {"http", "https"} and not parsed.netloc:
        raise StoreInvariantError(f"provenance field {field!r} is not a valid URI")
    if parsed.scheme == "urn" and not parsed.path:
        raise StoreInvariantError(f"provenance field {field!r} is not a valid URN")


def _metadata_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StoreInvariantError("provenance identity metadata is not canonical JSON") \
            from exc
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise StoreInvariantError(f"{field} must be a relative POSIX path")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise StoreInvariantError(f"{field} must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise StoreInvariantError(f"{field} must be a safe relative POSIX path")
    return value


def _validate_stream_semantics(
    value: object,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Validate explicit source/stored units and transform provenance.

    The Arrow fields deliberately do not invent one universal physical frame:
    every stored stream must say what its source values meant and which
    versioned transform, including an explicit identity transform, produced the
    stored values.
    """

    if not isinstance(value, Mapping) or set(value) != _STREAM_SEMANTICS_FIELDS:
        raise StoreInvariantError(
            "stream_semantics must contain the complete canonical fields")
    if value.get("schema_version") != "1.0":
        raise StoreInvariantError("stream_semantics schema_version must be '1.0'")

    def validate_entries(
        raw_entries: object, *, fields: frozenset[str], stream_field: str,
    ) -> list[dict[str, str]]:
        if not isinstance(raw_entries, list) or not raw_entries:
            raise StoreInvariantError(
                f"stream_semantics {stream_field} entries must be non-empty")
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for index, raw in enumerate(raw_entries):
            if not isinstance(raw, Mapping) or set(raw) != fields:
                raise StoreInvariantError(
                    f"stream_semantics {stream_field} entry {index} is invalid")
            entry: dict[str, str] = {}
            for field in fields:
                item = raw[field]
                if not isinstance(item, str) or not item.strip():
                    raise StoreInvariantError(
                        f"stream_semantics {stream_field} {field} is required")
                entry[field] = item
            identity = (entry["recording_id"], entry[stream_field])
            if identity in seen:
                raise StoreInvariantError(
                    f"duplicate stream_semantics identity: {identity!r}")
            seen.add(identity)
            result.append(entry)
        return result

    videos = validate_entries(
        value.get("video_streams"), fields=_VIDEO_SEMANTICS_FIELDS,
        stream_field="video_stream_id")
    imus = validate_entries(
        value.get("imu_streams"), fields=_IMU_SEMANTICS_FIELDS,
        stream_field="stream_id")
    return videos, imus


def _validate_recording_identity(
    value: object, *, source_recording_id: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _RECORDING_IDENTITY_FIELDS:
        raise StoreInvariantError(
            "recording_identity must contain the complete canonical fields")
    result = {
        field: _required_string(value, field)
        for field in _RECORDING_IDENTITY_FIELDS
    }
    if result["source_recording_id"] != source_recording_id:
        raise StoreInvariantError(
            "recording_identity source does not match source_recording_id")
    exact = result["stored_recording_id"] == result["source_recording_id"]
    expected_method = (
        "EXACT_SOURCE_IDENTIFIER" if exact
        else "EXPLICIT_INTERNAL_TO_SOURCE_MAPPING")
    if result["mapping_method"] != expected_method:
        raise StoreInvariantError(
            "recording_identity mapping_method contradicts the identifiers")
    return result


def _validate_vidimu_provenance_extension(
    provenance: Mapping[str, object], source_hashes: Mapping[str, object],
    *, source_dataset_version: str,
) -> None:
    if set(provenance) != _COMMON_PROVENANCE_FIELDS | _VIDIMU_EXTENSION_FIELDS:
        raise StoreInvariantError(
            "VIDIMU provenance fields are incomplete or contain unsupported claims")
    recording_id = provenance.get("source_recording_id")
    if not isinstance(recording_id, str) \
            or not re.fullmatch(r"S[0-9]+_A[0-9]+_T[0-9]+", recording_id):
        raise StoreInvariantError("VIDIMU source_recording_id is not canonical")
    subject = recording_id.split("_", maxsplit=1)[0]
    if set(source_hashes) != _VIDIMU_SOURCE_FILE_KEYS:
        raise StoreInvariantError(
            "VIDIMU source_file_hashes must contain the three canonical inputs")
    source_files = provenance.get("source_files")
    if not isinstance(source_files, Mapping) \
            or set(source_files) != _VIDIMU_SOURCE_FILE_KEYS:
        raise StoreInvariantError(
            "VIDIMU source_files must contain the three canonical inputs")
    selection = provenance.get("source_archive_selection")
    if not isinstance(selection, Mapping) \
            or set(selection) != _VIDIMU_SELECTION_FIELDS:
        raise StoreInvariantError(
            "VIDIMU source_archive_selection is incomplete or unexpected")
    expected_selection = {
        "release_version": source_dataset_version,
        "dataset_archive": "dataset",
        "dataset_subtree": "dataset/videoandimus",
        "video_subtree": "videosbodytrack",
        "inventory_scope": "EXTRACTED_SUBSET_NO_RELEASE_COMPLETENESS_CLAIM",
        "archive_digest_binding": (
            "CALLER_SUPPLIED_DOWNLOAD_DIGEST_NOT_RECOMPUTED_FROM_EXTRACTION"),
    }
    if any(selection.get(key) != value
           for key, value in expected_selection.items()) \
            or selection.get("video_archive") not in {
                "videosmallsize", "videosfullsize",
            }:
        raise StoreInvariantError("VIDIMU archive/subtree selection is not canonical")
    for field in ("dataset_archive_sha256", "video_archive_sha256"):
        if not isinstance(selection.get(field), str) \
                or not _SHA256.fullmatch(selection[field]):
            raise StoreInvariantError(f"VIDIMU {field} must be a lowercase SHA-256")
    selection_digest = provenance.get("source_archive_selection_sha256")
    if selection_digest != _metadata_sha256(dict(selection)):
        raise StoreInvariantError(
            "VIDIMU source_archive_selection_sha256 is not derived")

    expected_archives = {
        "video": selection["video_archive"],
        "video_pose": selection["dataset_archive"],
        "imu_quaternion": selection["dataset_archive"],
    }
    expected_prefixes = {
        "video": f'{selection["video_subtree"]}/',
        "video_pose": f'{selection["dataset_subtree"]}/',
        "imu_quaternion": f'{selection["dataset_subtree"]}/',
    }
    expected_paths = {
        "video": f'{selection["video_subtree"]}/{subject}/{recording_id}_pose.mp4',
        "video_pose": (
            f'{selection["dataset_subtree"]}/{subject}/{recording_id}.csv'),
        "imu_quaternion": (
            f'{selection["dataset_subtree"]}/{subject}/{recording_id}.raw'),
    }
    canonical_paths: set[str] = set()
    for name in sorted(_VIDIMU_SOURCE_FILE_KEYS):
        entry = source_files[name]
        if not isinstance(entry, Mapping) \
                or set(entry) != {"archive", "relative_path", "sha256"}:
            raise StoreInvariantError(f"VIDIMU source_files[{name!r}] is invalid")
        relative = _safe_relative_path(
            entry.get("relative_path"), f"VIDIMU source_files[{name!r}]")
        if entry.get("archive") != expected_archives[name] \
                or not relative.startswith(expected_prefixes[name]) \
                or relative != expected_paths[name] \
                or entry.get("sha256") != source_hashes[name]:
            raise StoreInvariantError(
                f"VIDIMU source_files[{name!r}] contradicts source identity")
        canonical_paths.add(relative)

    ledger = provenance.get("source_exclusion_ledger")
    if not isinstance(ledger, list):
        raise StoreInvariantError("VIDIMU source_exclusion_ledger must be a list")
    allowed_exclusions = {
        (
            selection["dataset_archive"],
            (
                f'{selection["dataset_subtree"]}/{subject}/'
                f'{recording_id}_Npose.csv'
            ),
            _VIDIMU_NPOSE_REASON,
        ),
        (
            selection["video_archive"],
            (
                f'{selection["video_subtree"]}/{subject}/'
                f'{recording_id}_Npose_pose.mp4'
            ),
            _VIDIMU_NPOSE_REASON,
        ),
        (
            selection["dataset_archive"],
            f'{selection["dataset_subtree"]}/{subject}/{recording_id}.sto',
            _VIDIMU_STO_REASON,
        ),
        (
            selection["dataset_archive"],
            f'{selection["dataset_subtree"]}/{subject}/ik_{recording_id}.mot',
            _VIDIMU_IK_MOT_REASON,
        ),
        (
            selection["dataset_archive"],
            (
                f'{selection["dataset_subtree"]}/{subject}/'
                f'ik_{recording_id}_orientationErrors.sto'
            ),
            _VIDIMU_IK_ORIENTATION_REASON,
        ),
    }
    if recording_id in _VIDIMU_S48_MOT_RECORDINGS:
        allowed_exclusions.add((
            selection["dataset_archive"],
            f'{selection["dataset_subtree"]}/{subject}/{recording_id}.mot',
            _VIDIMU_MOT_REASON,
        ))
    if selection["video_archive"] == "videosfullsize":
        allowed_exclusions.update({
            (
                selection["video_archive"],
                (
                    f'{selection["video_subtree"]}/{subject}/'
                    f'{recording_id}.mp4.out'
                ),
                _VIDIMU_FULLSIZE_OUT_REASON,
            ),
            (
                selection["video_archive"],
                (
                    f'{selection["video_subtree"]}/{subject}/'
                    f'{recording_id}_Npose.mp4.out'
                ),
                _VIDIMU_FULLSIZE_OUT_REASON,
            ),
        })
    allowed_exclusions.update(
        (
            selection["dataset_archive"],
            relative,
            _VIDIMU_RELEASE_RESIDUE_REASON_PREFIX + recording_id,
        )
        for relative in _VIDIMU_KNOWN_DATA_RESIDUES.get(
            recording_id, frozenset())
    )
    allowed_exclusions.update(
        (
            selection["video_archive"],
            relative,
            _VIDIMU_RELEASE_RESIDUE_REASON_PREFIX + recording_id,
        )
        for relative, archives in _VIDIMU_KNOWN_VIDEO_RESIDUES.get(
            recording_id, {}).items()
        if selection["video_archive"] in archives
    )
    observed_exclusions: set[tuple[str, str]] = set()
    for index, entry in enumerate(ledger):
        if not isinstance(entry, Mapping) \
                or set(entry) != _VIDIMU_EXCLUSION_FIELDS:
            raise StoreInvariantError(f"VIDIMU exclusion {index} is invalid")
        archive = entry.get("archive")
        relative = _safe_relative_path(
            entry.get("relative_path"), f"VIDIMU exclusion {index}")
        identity = (archive, relative)
        if identity in observed_exclusions or relative in canonical_paths \
                or (archive, relative, entry.get("reason")) \
                not in allowed_exclusions \
                or not isinstance(entry.get("sha256"), str) \
                or not _SHA256.fullmatch(entry["sha256"]):
            raise StoreInvariantError(
                f"VIDIMU exclusion {index} contradicts the pinned policy")
        observed_exclusions.add(identity)
    if ledger != sorted(
            ledger, key=lambda item: (item["archive"], item["relative_path"])):
        raise StoreInvariantError("VIDIMU exclusion ledger is not canonical order")
    ledger_digest = provenance.get("source_exclusion_ledger_sha256")
    if ledger_digest != _metadata_sha256(ledger):
        raise StoreInvariantError(
            "VIDIMU source_exclusion_ledger_sha256 is not derived")
    source_identity = {
        "archive_selection": dict(selection),
        "excluded_files": ledger,
        "excluded_files_sha256": ledger_digest,
        "files": {name: dict(source_files[name]) for name in source_files},
    }
    if provenance.get("source_identity_sha256") != _metadata_sha256(source_identity):
        raise StoreInvariantError("VIDIMU source_identity_sha256 is not derived")

    expected_claims: dict[str, object] = {
        "source_record_id": 15_075_076,
        "source_record_url": "https://zenodo.org/records/15075076",
        "source_record_doi": "10.1038/s41597-023-02554-9",
        "source_concept_doi": "10.5281/zenodo.7681316",
        "source_license": "CC BY 4.0",
        "source_license_spdx": "CC-BY-4.0",
        "source_license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_terms_sha256": provenance.get("license_terms_sha256"),
        "redistribution_status": "PERMITTED_WITH_ATTRIBUTION",
        "recording_inventory_complete": True,
        "release_inventory_complete": False,
        "release_inventory_scope": (
            "EXTRACTED_SUBSET_NO_RELEASE_COMPLETENESS_CLAIM"),
        "released_imu_payload": "FIVE_50_HZ_QUATERNION_ORIENTATION_STREAMS",
        "video_nominal_fps": 30,
        "imu_nominal_hz": 50,
        "raw_accelerometer_axes_available": False,
        "raw_gyroscope_axes_available": False,
        "clock_truth_status": "UNVERIFIED_FROM_FILE_INVENTORY",
        "allowed_validation_role": "INVENTORY_ONLY_UNVERIFIED_SYNC",
        "prohibited_interpretation": (
            "RAW_ACCEL_GYRO_OR_INDEPENDENT_SYNC_OR_DRIFT_GROUND_TRUTH"),
    }
    if any(provenance.get(field) != expected
           for field, expected in expected_claims.items()):
        raise StoreInvariantError("VIDIMU scientific claim boundary is not pinned")
    if isinstance(provenance.get("source_record_id"), bool) \
            or not isinstance(provenance.get("source_record_id"), int):
        raise StoreInvariantError("VIDIMU source_record_id must be an integer")
    for field in (
        "recording_inventory_complete", "release_inventory_complete",
        "raw_accelerometer_axes_available", "raw_gyroscope_axes_available",
    ):
        if not isinstance(provenance.get(field), bool):
            raise StoreInvariantError(f"VIDIMU {field} must be boolean")
    if provenance.get("source_terms_hash_origin") not in {
        "SUPPLIED", "SUPPLIED_AND_VERIFIED_FROM_FILE", "COMPUTED_FROM_FILE",
    }:
        raise StoreInvariantError("VIDIMU source_terms_hash_origin is invalid")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, field: str, *, nullable: bool = False) -> int | None:
    if nullable and value is None:
        return None
    if not _is_int(value) or value <= 0:
        raise StoreInvariantError(f"{field} must be a positive integer")
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise StoreInvariantError(f"{field} must be finite")
    return float(value)


def _unit_interval(value: object, field: str) -> float:
    result = _finite_number(value, field)
    if not 0.0 <= result <= 1.0:
        raise StoreInvariantError(f"{field} must lie in [0,1]")
    return result


def validate_provenance(provenance: Mapping[str, object]) -> None:
    """Validate immutable source identity and separate analysis/release gates."""

    if not isinstance(provenance, Mapping):
        raise StoreInvariantError("provenance must be an object")
    if provenance.get("provenance_schema_version") != "1.0":
        raise StoreInvariantError("provenance schema version must be '1.0'")

    strings = {field: _required_string(provenance, field)
               for field in _REQUIRED_PROVENANCE_STRINGS}
    if strings["source_dataset"] != "VIDIMU" \
            and set(provenance).difference(_COMMON_PROVENANCE_FIELDS):
        raise StoreInvariantError(
            "provenance contains unsupported claims")
    _validate_recording_identity(
        provenance.get("recording_identity"),
        source_recording_id=strings["source_recording_id"],
    )
    _validate_uri(strings["source_record_uri"], "source_record_uri")
    _validate_uri(strings["license_uri"], "license_uri")
    if not _SHA256.fullmatch(strings["license_terms_sha256"]):
        raise StoreInvariantError("license_terms_sha256 must be a lowercase SHA-256")
    if not _COMMIT_ID.fullmatch(strings["ingestion_commit"]):
        raise StoreInvariantError("ingestion_commit must be a hexadecimal commit ID")

    source_hashes = provenance.get("source_file_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise StoreInvariantError("source_file_hashes must be a non-empty object")
    for source_name, digest in source_hashes.items():
        if not isinstance(source_name, str) or not source_name:
            raise StoreInvariantError("source_file_hashes keys must be non-empty strings")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise StoreInvariantError(
                f"source_file_hashes[{source_name!r}] must be a lowercase SHA-256")
    if strings["source_dataset"] == "VIDIMU":
        _validate_vidimu_provenance_extension(
            provenance, source_hashes,
            source_dataset_version=strings["source_dataset_version"],
        )

    if strings["source_access_status"] not in _ALLOWED_ACCESS:
        raise StoreInvariantError(
            "source_access_status does not authorize this local analysis")
    if strings["source_kind"] not in _ALLOWED_SOURCE_KINDS:
        raise StoreInvariantError("source_kind is not an allowed v0.1 source role")
    if strings["validation_role"] not in _ALLOWED_VALIDATION_ROLES:
        raise StoreInvariantError(
            "validation_role is not an allowed non-clinical systems role")
    local_allowed = _required_bool(provenance, "local_analysis_allowed")
    source_release = _required_bool(provenance, "source_redistribution_allowed")
    derived_release = _required_bool(provenance, "derived_artifact_release_allowed")
    if not local_allowed:
        raise StoreInvariantError("local_analysis_allowed must be true")

    source_kind = strings["source_kind"]
    synthetic_profile = {
        "source_access_status": "GENERATED_LOCALLY",
        "source_redistribution_status": "GENERATED_TEST_DATA_ONLY",
        "derived_artifact_policy": (
            "TEST_ARTIFACTS_MUST_NOT_BE_EMPIRICAL_EVIDENCE"),
        "permitted_use": "SOFTWARE_TESTING_ONLY",
        "validation_role": "SYNTHETIC_REGRESSION_ONLY",
    }
    if source_kind == "SYNTHETIC_GENERATED":
        if any(strings[field] != expected
               for field, expected in synthetic_profile.items()) \
                or not source_release or not derived_release:
            raise StoreInvariantError(
                "synthetic source_kind requires the complete synthetic-only profile")
    elif any(strings[field] == marker
             for field, marker in synthetic_profile.items()):
        raise StoreInvariantError(
            "synthetic-only legal/profile values require SYNTHETIC_GENERATED")

    redistribution = strings["source_redistribution_status"]
    allowed_redistribution = {
        "PERMITTED", "PERMITTED_WITH_ATTRIBUTION", "GENERATED_TEST_DATA_ONLY",
    }
    prohibited_redistribution = {"PROHIBITED_BY_TERMS", "LOCAL_ANALYSIS_ONLY"}
    if source_release and redistribution not in allowed_redistribution:
        raise StoreInvariantError(
            "source redistribution status contradicts source_redistribution_allowed")
    if not source_release and redistribution not in prohibited_redistribution:
        raise StoreInvariantError(
            "source redistribution status contradicts source_redistribution_allowed")

    expected_decision = (
        "ALLOW_ANALYSIS_AND_RELEASE" if derived_release
        else "ALLOW_LOCAL_ANALYSIS_ONLY"
    )
    expected_release_status = "RELEASABLE" if derived_release else "LOCAL_ONLY"
    if strings["use_decision"] != expected_decision:
        raise StoreInvariantError(
            "use_decision contradicts derived_artifact_release_allowed")
    if strings["artifact_release_status"] != expected_release_status:
        raise StoreInvariantError(
            "artifact_release_status contradicts derived_artifact_release_allowed")
    permitted_use = strings["permitted_use"]
    derived_policy = strings["derived_artifact_policy"]
    if derived_release:
        if permitted_use not in _RELEASE_PERMITTED_USES \
                or derived_policy not in _RELEASE_DERIVED_POLICIES:
            raise StoreInvariantError(
                "permitted_use or derived_artifact_policy does not authorize release")
    elif permitted_use not in _LOCAL_ONLY_PERMITTED_USES \
            or derived_policy not in _LOCAL_ONLY_DERIVED_POLICIES:
        raise StoreInvariantError(
            "local-only permitted_use and derived_artifact_policy must agree")

    if strings["source_dataset"] == "VIDIMU":
        vidimu_profile = {
            "source_kind": "PUBLIC_DATASET",
            "source_dataset_version": "2.0.0",
            "source_record_uri": "https://zenodo.org/records/15075076",
            "license_id": "CC-BY-4.0",
            "license_uri": "https://creativecommons.org/licenses/by/4.0/",
            "source_access_status": "PUBLIC_ZENODO_RECORD",
            "source_redistribution_status": "PERMITTED_WITH_ATTRIBUTION",
            "derived_artifact_policy": "CC_BY_4_0_ATTRIBUTION_REQUIRED",
            "permitted_use": (
                "RESEARCH_AND_DERIVED_BENCHMARK_ARTIFACTS_WITH_ATTRIBUTION"),
            "validation_role": "INVENTORY_ONLY_UNVERIFIED_SYNC",
            "use_decision": "ALLOW_ANALYSIS_AND_RELEASE",
            "artifact_release_status": "RELEASABLE",
        }
        if any(strings[field] != expected
               for field, expected in vidimu_profile.items()) \
                or not source_release or not derived_release:
            raise StoreInvariantError("VIDIMU legal/source profile is not pinned")

    pairs = _validate_alignment_plan(
        provenance.get("alignment_generation_parameters"))
    video_semantics, imu_semantics = _validate_stream_semantics(
        provenance.get("stream_semantics"))
    planned_video_streams = {
        (pair["recording_id"], pair["video_stream_id"])
        for pair in pairs
    }
    planned_imu_streams = {
        (pair["recording_id"], pair["imu_stream_id"])
        for pair in pairs
    }
    declared_video_streams = {
        (entry["recording_id"], entry["video_stream_id"])
        for entry in video_semantics
    }
    declared_imu_streams = {
        (entry["recording_id"], entry["stream_id"])
        for entry in imu_semantics
    }
    if declared_video_streams != planned_video_streams \
            or declared_imu_streams != planned_imu_streams:
        raise StoreInvariantError(
            "stream_semantics must exactly cover the alignment stream inventory")
    window_plan, segments = _validate_window_plan(
        provenance.get("window_generation_parameters"))
    alignment_pairs = {
        (pair["recording_id"], pair["video_stream_id"], pair["imu_stream_id"])
        for pair in pairs
    }
    continuity_pairs = {
        (segment.recording_id, segment.video_stream_id, segment.imu_stream_id)
        for segment in segments
    }
    if continuity_pairs != alignment_pairs:
        raise StoreInvariantError(
            "continuity-segment pairs must exactly match the alignment pair plan")
    signal_policy_pairs = {
        (policy["recording_id"], policy["video_stream_id"],
         policy["imu_stream_id"])
        for policy in _validate_frequency_signal_policies(
            window_plan["frequency_signal_policies"])
    }
    if signal_policy_pairs != alignment_pairs:
        raise StoreInvariantError(
            "frequency signal policies must exactly match alignment pairs")
    if strings["source_dataset"] == "VIDIMU":
        unresolved_bit = int(QualityBits.UNRESOLVED_CLOCK_MAP)
        source_subject = strings["source_recording_id"].split("_", maxsplit=1)[0]
        if {segment.split_group_id for segment in segments} != {source_subject}:
            raise StoreInvariantError(
                "VIDIMU split_group_id must equal the source subject ID")
        if any(segment.accepted or not segment.quality_bits & unresolved_bit
               for segment in segments):
            raise StoreInvariantError(
                "VIDIMU inventory-only provenance requires every continuity "
                "segment to remain rejected as UNRESOLVED_CLOCK_MAP")


def _validate_alignment_plan(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != {"pairs"}:
        raise StoreInvariantError(
            "alignment_generation_parameters must contain exactly 'pairs'")
    pairs = value.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise StoreInvariantError("alignment generation pairs must be non-empty")
    required = {
        "recording_id", "video_stream_id", "imu_stream_id", "video_end_ns",
        "max_imu_gap_ns", "min_coverage_fraction",
        "max_clock_residual_p95_ms",
    }
    result: list[dict[str, object]] = []
    identities: set[tuple[str, str, str]] = set()
    video_ends: dict[tuple[str, str], int] = {}
    for index, raw in enumerate(pairs):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise StoreInvariantError(
                f"alignment pair {index} must contain the complete canonical fields")
        pair = dict(raw)
        identity: list[str] = []
        for field in ("recording_id", "video_stream_id", "imu_stream_id"):
            item = pair[field]
            if not isinstance(item, str) or not item:
                raise StoreInvariantError(f"alignment pair {field} is required")
            identity.append(item)
        key = tuple(identity)
        if key in identities:
            raise StoreInvariantError(f"duplicate alignment pair plan: {key!r}")
        identities.add(key)
        _positive_int(pair["video_end_ns"], "alignment video_end_ns")
        video_key = (identity[0], identity[1])
        prior_video_end = video_ends.setdefault(
            video_key, int(pair["video_end_ns"]))
        if prior_video_end != pair["video_end_ns"]:
            raise StoreInvariantError(
                "alignment pairs for one video stream must share video_end_ns")
        _positive_int(
            pair["max_imu_gap_ns"], "alignment max_imu_gap_ns", nullable=True)
        _unit_interval(
            pair["min_coverage_fraction"], "alignment min_coverage_fraction")
        maximum_residual = _finite_number(
            pair["max_clock_residual_p95_ms"],
            "alignment max_clock_residual_p95_ms")
        if maximum_residual <= 0:
            raise StoreInvariantError(
                "alignment max_clock_residual_p95_ms must be positive")
        result.append(pair)
    return result


_WINDOW_PARAMETER_FIELDS = frozenset({
    "window_ns",
    "hop_ns",
    "tremor_band_low_hz",
    "tremor_band_high_hz",
    "min_video_coverage",
    "min_imu_coverage",
    "max_video_gap_ns",
    "max_imu_gap_ns",
    "video_observability_factor",
    "video_observability_cap_hz",
    "min_frequency_cycles",
    "max_cadence_deviation_fraction",
    "min_tracking_quality",
    "min_valid_keypoint_fraction",
    "frequency_signal_policies",
    "continuity_segments",
})
_SIGNAL_POLICY_FIELDS = frozenset({
    "recording_id", "video_stream_id", "imu_stream_id",
    "cv_motion_min_peak_to_peak_stored_units",
    "acceleration_min_peak_to_peak_stored_units",
    "angular_velocity_min_peak_to_peak_stored_units",
    "quaternion_min_angular_range_rad",
    "minimum_varying_cv_components", "minimum_varying_imu_channels",
})
_CONTINUITY_FIELDS = frozenset({
    "segment_id",
    "recording_id",
    "video_stream_id",
    "imu_stream_id",
    "start_time_ns",
    "end_time_ns",
    "split_group_id",
    "accepted",
    "quality_bits",
})


def _validate_frequency_signal_policies(
    value: object,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise StoreInvariantError(
            "frequency_signal_policies must be a non-empty list")
    result: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _SIGNAL_POLICY_FIELDS:
            raise StoreInvariantError(
                f"frequency signal policy {index} is incomplete or unexpected")
        policy = dict(raw)
        identity = []
        for field in ("recording_id", "video_stream_id", "imu_stream_id"):
            item = policy[field]
            if not isinstance(item, str) or not item:
                raise StoreInvariantError(
                    f"frequency signal policy {field} is required")
            identity.append(item)
        key = tuple(identity)
        if key in seen:
            raise StoreInvariantError(
                f"duplicate frequency signal policy: {key!r}")
        seen.add(key)
        for field in (
            "cv_motion_min_peak_to_peak_stored_units",
            "acceleration_min_peak_to_peak_stored_units",
            "angular_velocity_min_peak_to_peak_stored_units",
            "quaternion_min_angular_range_rad",
        ):
            item = policy[field]
            if item is not None and _finite_number(item, field) <= 0:
                raise StoreInvariantError(
                    f"frequency signal policy {field} must be positive or null")
        if policy["cv_motion_min_peak_to_peak_stored_units"] is None:
            raise StoreInvariantError("CV motion range threshold must be specified")
        quaternion = policy["quaternion_min_angular_range_rad"]
        if quaternion is not None and quaternion > math.pi:
            raise StoreInvariantError(
                "quaternion angular range threshold must not exceed pi")
        _positive_int(
            policy["minimum_varying_cv_components"],
            "minimum_varying_cv_components")
        _positive_int(
            policy["minimum_varying_imu_channels"],
            "minimum_varying_imu_channels")
        result.append(policy)
    return result


def _validate_window_plan(value: object) -> tuple[dict[str, object], list[ContinuitySegment]]:
    if not isinstance(value, Mapping) or set(value) != _WINDOW_PARAMETER_FIELDS:
        raise StoreInvariantError(
            "window_generation_parameters must contain the complete canonical policy")
    plan = dict(value)
    _positive_int(plan["window_ns"], "window_ns")
    _positive_int(plan["hop_ns"], "hop_ns")
    low = _finite_number(plan["tremor_band_low_hz"], "tremor_band_low_hz")
    high = _finite_number(plan["tremor_band_high_hz"], "tremor_band_high_hz")
    if low <= 0 or high <= low:
        raise StoreInvariantError("tremor band must have 0 < low < high")
    _unit_interval(plan["min_video_coverage"], "min_video_coverage")
    _unit_interval(plan["min_imu_coverage"], "min_imu_coverage")
    _positive_int(plan["max_video_gap_ns"], "max_video_gap_ns", nullable=True)
    _positive_int(plan["max_imu_gap_ns"], "max_imu_gap_ns", nullable=True)
    factor = _finite_number(
        plan["video_observability_factor"], "video_observability_factor")
    if not 0 < factor <= 0.5:
        raise StoreInvariantError("video_observability_factor must lie in (0,0.5]")
    if _finite_number(
            plan["video_observability_cap_hz"],
            "video_observability_cap_hz") <= 0:
        raise StoreInvariantError("video_observability_cap_hz must be positive")
    if _finite_number(plan["min_frequency_cycles"], "min_frequency_cycles") <= 0:
        raise StoreInvariantError("min_frequency_cycles must be positive")
    if _finite_number(
            plan["max_cadence_deviation_fraction"],
            "max_cadence_deviation_fraction") < 0:
        raise StoreInvariantError(
            "max_cadence_deviation_fraction must be non-negative")
    tracking = _unit_interval(
        plan["min_tracking_quality"], "min_tracking_quality")
    keypoints = _unit_interval(
        plan["min_valid_keypoint_fraction"], "min_valid_keypoint_fraction")
    if tracking == 0 or keypoints == 0:
        raise StoreInvariantError(
            "tracking/keypoint quality thresholds must be positive")
    _validate_frequency_signal_policies(plan["frequency_signal_policies"])

    raw_segments = plan["continuity_segments"]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise StoreInvariantError("continuity_segments must be a non-empty list")
    segments: list[ContinuitySegment] = []
    identities: set[tuple[str, str, str, str]] = set()
    split_group_by_recording: dict[str, str] = {}
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping) or set(raw) != _CONTINUITY_FIELDS:
            raise StoreInvariantError(
                f"continuity segment {index} must contain the complete canonical fields")
        if not isinstance(raw["accepted"], bool):
            raise StoreInvariantError("continuity accepted must be boolean")
        if not _is_int(raw["quality_bits"]) or not 0 <= raw["quality_bits"] < 2**32:
            raise StoreInvariantError("continuity quality_bits must be uint32")
        if raw["quality_bits"] & ~_KNOWN_QUALITY_MASK:
            raise StoreInvariantError("continuity quality_bits contains unknown flags")
        if not _is_int(raw["start_time_ns"]) or not _is_int(raw["end_time_ns"]):
            raise StoreInvariantError("continuity bounds must be integers")
        try:
            segment = ContinuitySegment(**dict(raw))
        except (TypeError, WindowIndexError) as exc:
            raise StoreInvariantError(f"invalid continuity segment {index}: {exc}") from exc
        identity = (
            segment.recording_id, segment.video_stream_id,
            segment.imu_stream_id, segment.segment_id,
        )
        if identity in identities:
            raise StoreInvariantError(
                f"duplicate continuity segment identity: {identity!r}")
        identities.add(identity)
        prior_split_group = split_group_by_recording.setdefault(
            segment.recording_id, segment.split_group_id)
        if segment.split_group_id != prior_split_group:
            raise StoreInvariantError(
                "one recording must use one immutable split_group_id")
        segments.append(segment)
    return plan, segments


def _rows_by_stream(table: pa.Table, stream_field: str) -> dict[tuple[str, str], list[dict]]:
    result: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in table.to_pylist():
        result[(row["recording_id"], row[stream_field])].append(row)
    return result


def _validate_stream_rows(
    table: pa.Table, *, stream_field: str, source_index_field: str,
    native_time_field: str, quality_field: str,
) -> None:
    for key, rows in _rows_by_stream(table, stream_field).items():
        rows.sort(key=lambda row: row["canonical_ordinal"])
        canonical = [row["canonical_ordinal"] for row in rows]
        if canonical != list(range(len(rows))):
            raise StoreInvariantError(
                f"canonical ordinals must be dense and zero-based for {key!r}")
        source_ordinals = [row["source_ordinal"] for row in rows]
        if any(not _is_int(value) or value < 0 for value in source_ordinals):
            raise StoreInvariantError(f"source ordinals must be non-negative for {key!r}")
        if len(source_ordinals) != len(set(source_ordinals)):
            raise StoreInvariantError(f"duplicate source ordinal for {key!r}")
        source_indexes = [row[source_index_field] for row in rows]
        if any(not _is_int(value) or value < 0 for value in source_indexes):
            raise StoreInvariantError(
                f"{source_index_field} values must be non-negative for {key!r}")
        if len(source_indexes) != len(set(source_indexes)):
            raise StoreInvariantError(
                f"duplicate {source_index_field} for {key!r}")
        by_epoch: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_epoch[row["clock_epoch_id"]].append(row)
        for epoch_rows in by_epoch.values():
            epoch_rows.sort(key=lambda row: row["source_ordinal"])
            for previous, current in pairwise(epoch_rows):
                if current[native_time_field] < previous[native_time_field]:
                    if not int(current[quality_field]) & int(
                            QualityBits.NON_MONOTONIC_TIMESTAMP):
                        raise StoreInvariantError(
                            "acquisition-order timestamp reversal requires the "
                            f"NON_MONOTONIC_TIMESTAMP bit for {key!r}")
                    raise StoreInvariantError(
                        "an acquisition-order timestamp reversal cannot reuse "
                        f"clock_epoch_id for {key!r}; declare a new epoch")
        order = [(row["canonical_time_ns"], row["source_ordinal"]) for row in rows]
        if order != sorted(order):
            raise StoreInvariantError(
                f"canonical ordering is not stable by time/source ordinal for {key!r}")
        for previous, current in pairwise(rows):
            if previous["canonical_time_ns"] == current["canonical_time_ns"]:
                duplicate_bit = int(QualityBits.DUPLICATE_TIMESTAMP)
                if not int(previous[quality_field]) & duplicate_bit \
                        or not int(current[quality_field]) & duplicate_bit:
                    raise StoreInvariantError(
                        f"duplicate timestamps require quality bits for {key!r}")
        for row in rows:
            if not _is_int(row[native_time_field]) \
                    or not _is_int(row["canonical_time_ns"]):
                raise StoreInvariantError(f"timestamps must be integer nanoseconds for {key!r}")
            if not _is_int(row[quality_field]) or not 0 <= row[quality_field] < 2**32:
                raise StoreInvariantError(f"{quality_field} must be uint32 for {key!r}")
            if row[quality_field] & ~_KNOWN_QUALITY_MASK:
                raise StoreInvariantError(
                    f"{quality_field} contains unknown quality flags for {key!r}")


def _validate_frames(
    frames: pa.Table,
    *,
    video_end_ns_by_stream: Mapping[tuple[str, str], int],
) -> None:
    _validate_stream_rows(
        frames, stream_field="video_stream_id", source_index_field="frame_index",
        native_time_field="video_pts_native_ns", quality_field="quality_bits")
    for stream_key, rows in _rows_by_stream(
            frames, "video_stream_id").items():
        rows.sort(key=lambda row: row["canonical_ordinal"])
        for index, row in enumerate(rows):
            expected_gap_ms = None if index == 0 else (
                row["canonical_time_ns"]
                - rows[index - 1]["canonical_time_ns"]
            ) / 1_000_000.0
            observed_gap_ms = row["gap_before_ms"]
            if expected_gap_ms is None:
                if observed_gap_ms is not None:
                    raise StoreInvariantError(
                        "the first frame gap_before_ms must be null")
            elif observed_gap_ms is None or not math.isclose(
                    observed_gap_ms, expected_gap_ms,
                    rel_tol=1e-9, abs_tol=1e-6):
                raise StoreInvariantError(
                    "frame gap_before_ms disagrees with canonical timestamps")
            effective_fps = row["effective_fps"]
            cadence_delta_ns = None
            if index + 1 < len(rows) and rows[index + 1]["clock_epoch_id"] \
                    == row["clock_epoch_id"]:
                cadence_delta_ns = (
                    rows[index + 1]["canonical_time_ns"]
                    - row["canonical_time_ns"])
            elif index == len(rows) - 1:
                video_end_ns = video_end_ns_by_stream.get(stream_key)
                if video_end_ns is None:
                    raise StoreInvariantError(
                        "alignment plan does not define the video tail")
                cadence_delta_ns = (
                    video_end_ns - row["canonical_time_ns"])
            elif index > 0 and rows[index - 1]["clock_epoch_id"] \
                    == row["clock_epoch_id"]:
                cadence_delta_ns = (
                    row["canonical_time_ns"]
                    - rows[index - 1]["canonical_time_ns"])
            if effective_fps is None:
                if cadence_delta_ns is not None and cadence_delta_ns > 0:
                    raise StoreInvariantError(
                        "frame effective_fps is required when canonical cadence "
                        "can be derived")
            else:
                if cadence_delta_ns is None or cadence_delta_ns <= 0:
                    raise StoreInvariantError(
                        "frame effective_fps must be null when canonical "
                        "cadence cannot be derived")
                if not math.isclose(
                            effective_fps,
                            1_000_000_000.0 / cadence_delta_ns,
                            rel_tol=1e-3, abs_tol=1e-6):
                    raise StoreInvariantError(
                        "frame effective_fps disagrees with canonical cadence")
    for row in frames.to_pylist():
        if row["width"] <= 0 or row["height"] <= 0:
            raise StoreInvariantError("frame dimensions must be positive")
        for field in ("effective_fps", "gap_before_ms"):
            value = row[field]
            if value is not None and (not math.isfinite(value) or value < 0):
                raise StoreInvariantError(f"frame {field} must be finite and non-negative")
        if row["effective_fps"] == 0:
            raise StoreInvariantError("frame effective_fps must be positive when present")
        status = row["decode_status"]
        if status not in _ALLOWED_DECODE_STATUS:
            raise StoreInvariantError(f"unknown frame decode_status: {status!r}")
        bits = int(row["quality_bits"])
        has_decode_failure = bool(bits & int(QualityBits.DECODE_FAILURE))
        has_missing_time = bool(bits & int(QualityBits.MISSING_TIMESTAMP))
        if (status == "DECODE_FAILURE") != has_decode_failure:
            raise StoreInvariantError("decode_status and DECODE_FAILURE bit disagree")
        if (status == "MISSING_TIMESTAMP") != has_missing_time:
            raise StoreInvariantError("decode_status and MISSING_TIMESTAMP bit disagree")


def _finite_payload(value: object, field: str) -> bool:
    """Validate populated numeric values and report nested/top-level nulls."""

    if value is None:
        return True
    if isinstance(value, list):
        has_null = False
        for item in value:
            if item is None:
                has_null = True
            elif isinstance(item, bool) or not isinstance(item, (int, float)) \
                    or not math.isfinite(item):
                raise StoreInvariantError(
                    f"CV payload {field} must contain finite numbers")
        return has_null
    elif isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise StoreInvariantError(f"CV payload {field} must be finite")
    return False


def _validate_cv(frames: pa.Table, estimates: pa.Table, provenance: Mapping[str, object]) -> None:
    frame_by_key = {
        (row["recording_id"], row["video_stream_id"], row["canonical_ordinal"]): row
        for row in frames.to_pylist()
    }
    estimate_rows = estimates.to_pylist()
    estimate_keys = [
        (row["recording_id"], row["video_stream_id"], row["canonical_ordinal"])
        for row in estimate_rows
    ]
    if len(estimate_keys) != len(set(estimate_keys)):
        raise StoreInvariantError("duplicate CV estimate key")
    if set(estimate_keys) != set(frame_by_key):
        raise StoreInvariantError("CV estimates must have exactly one row per frame")
    estimator_versions: set[str] = set()
    payload_fields = (
        "keypoints", "keypoint_validity", "motion_vector", "palm_orientation",
        "hand_scale", "estimated_frequency_hz", "tracking_quality",
    )
    numeric_fields = (
        "keypoints", "motion_vector", "palm_orientation", "hand_scale",
        "estimated_frequency_hz", "tracking_quality",
    )
    for row in estimate_rows:
        key = (row["recording_id"], row["video_stream_id"], row["canonical_ordinal"])
        frame = frame_by_key[key]
        if row["frame_index"] != frame["frame_index"] \
                or row["canonical_time_ns"] != frame["canonical_time_ns"]:
            raise StoreInvariantError("CV estimate foreign key disagrees with frame")
        estimator_versions.add(row["estimator_version"])
        has_null_payload = False
        for field in numeric_fields:
            has_null_payload = _finite_payload(
                row[field], field) or has_null_payload
        validity = row["keypoint_validity"]
        if validity is None:
            has_null_payload = True
        else:
            for item in validity:
                if item is None:
                    has_null_payload = True
                elif not isinstance(item, bool):
                    raise StoreInvariantError(
                        "CV keypoint_validity must contain booleans")
        tracking = row["tracking_quality"]
        if tracking is not None and not 0.0 <= tracking <= 1.0:
            raise StoreInvariantError("CV tracking_quality must lie in [0,1]")
        frequency = row["estimated_frequency_hz"]
        if frequency is not None and frequency < 0:
            raise StoreInvariantError(
                "CV estimated_frequency_hz must be non-negative")
        hand_scale = row["hand_scale"]
        if hand_scale is not None and hand_scale <= 0:
            raise StoreInvariantError("CV hand_scale must be positive")
        no_valid_keypoints = validity is not None and not any(validity)
        unusable_tracking = tracking is not None and tracking == 0.0
        if (no_valid_keypoints or unusable_tracking) \
                and not int(frame["quality_bits"]) & int(QualityBits.INVALID_CV):
            raise StoreInvariantError(
                "all-invalid keypoints or zero tracking require INVALID_CV")
        if (has_null_payload or any(row[field] is None for field in payload_fields)) \
                and not int(frame["quality_bits"]) & int(QualityBits.INVALID_CV):
            raise StoreInvariantError(
                "nullable CV payload requires the frame INVALID_CV quality bit")
    if estimator_versions != {provenance["cv_estimator_version"]}:
        raise StoreInvariantError(
            "CV estimator versions disagree with immutable provenance")


_ACCEL = frozenset({"ax", "ay", "az"})
_GYRO = frozenset({"gx", "gy", "gz"})
_QUAT = frozenset({"qw", "qx", "qy", "qz"})
_PAYLOAD_CHANNELS = {
    "ACCEL": _ACCEL,
    "GYRO": _GYRO,
    "ACCEL_GYRO": _ACCEL | _GYRO,
    "QUATERNION": _QUAT,
    "ACCEL_GYRO_QUATERNION": _ACCEL | _GYRO | _QUAT,
}
_QUATERNION_NORM_ABS_TOL = 1e-3


def _validate_stream_semantics_inventory(
    frames: pa.Table, samples: pa.Table, provenance: Mapping[str, object],
) -> None:
    video_entries, imu_entries = _validate_stream_semantics(
        provenance["stream_semantics"])
    expected_videos = set(_rows_by_stream(frames, "video_stream_id"))
    observed_videos = {
        (entry["recording_id"], entry["video_stream_id"])
        for entry in video_entries
    }
    if observed_videos != expected_videos:
        raise StoreInvariantError(
            "video stream_semantics inventory disagrees with frame_index")

    video_pairs = (
        ("source_keypoint_convention", "stored_keypoint_convention"),
        ("source_motion_vector_convention", "stored_motion_vector_convention"),
        ("source_palm_orientation_convention",
         "stored_palm_orientation_convention"),
        ("source_hand_scale_convention", "stored_hand_scale_convention"),
    )
    for entry in video_entries:
        if entry["canonicalization_transform_id"] == "IDENTITY_SOURCE_NATIVE" \
                and any(entry[source] != entry[stored]
                        for source, stored in video_pairs):
            raise StoreInvariantError(
                "identity video canonicalization must preserve source semantics")

    actual_imu_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in samples.select(
            ["recording_id", "stream_id", "payload_kind"]).to_pylist():
        actual_imu_kinds[(row["recording_id"], row["stream_id"])].add(
            row["payload_kind"])
    observed_imus = {
        (entry["recording_id"], entry["stream_id"])
        for entry in imu_entries
    }
    if observed_imus != set(actual_imu_kinds):
        raise StoreInvariantError(
            "IMU stream_semantics inventory disagrees with imu_samples")
    if provenance.get("source_dataset") == "VIDIMU":
        body_locations = {entry["body_location"] for entry in imu_entries}
        if len(actual_imu_kinds) != 5 or len(body_locations) != 5 \
                or any(kinds != {"QUATERNION"}
                       for kinds in actual_imu_kinds.values()):
            raise StoreInvariantError(
                "VIDIMU snapshots require five distinct released QUATERNION streams")
    for entry in imu_entries:
        key = (entry["recording_id"], entry["stream_id"])
        if actual_imu_kinds[key] != {entry["payload_kind"]}:
            raise StoreInvariantError(
                "IMU stream_semantics payload_kind disagrees with imu_samples")
        channels = _PAYLOAD_CHANNELS.get(entry["payload_kind"])
        if channels is None:
            raise StoreInvariantError(
                "IMU stream_semantics payload_kind is unsupported")
        semantic_groups = (
            (bool(channels & _ACCEL), "source_acceleration_unit",
             "stored_acceleration_unit"),
            (bool(channels & _GYRO), "source_angular_velocity_unit",
             "stored_angular_velocity_unit"),
            (bool(channels & _QUAT), "source_quaternion_convention",
             "stored_quaternion_convention"),
        )
        for present, source, stored in semantic_groups:
            values = (entry[source], entry[stored])
            if present and _NOT_PRESENT in values:
                raise StoreInvariantError(
                    f"present IMU channels require {source} and {stored}")
            if not present and values != (_NOT_PRESENT, _NOT_PRESENT):
                raise StoreInvariantError(
                    f"absent IMU channels require {source} and {stored} "
                    "to be NOT_PRESENT")
        if entry["canonicalization_transform_id"] == "IDENTITY_SOURCE_NATIVE":
            preserved_pairs = (
                ("source_acceleration_unit", "stored_acceleration_unit"),
                ("source_angular_velocity_unit", "stored_angular_velocity_unit"),
                ("source_quaternion_convention",
                 "stored_quaternion_convention"),
                ("source_device_frame_convention",
                 "stored_device_frame_convention"),
            )
            if any(entry[source] != entry[stored]
                   for source, stored in preserved_pairs):
                raise StoreInvariantError(
                    "identity IMU canonicalization must preserve source semantics")


def _validate_imu(samples: pa.Table) -> None:
    _validate_stream_rows(
        samples, stream_field="stream_id", source_index_field="sample_index",
        native_time_field="sensor_time_native_ns", quality_field="validity_bits")
    all_channels = _ACCEL | _GYRO | _QUAT
    rows = samples.to_pylist()
    payload_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        kind = row["payload_kind"]
        required = _PAYLOAD_CHANNELS.get(kind)
        if required is None:
            raise StoreInvariantError(f"unknown IMU payload_kind: {kind!r}")
        missing_required = any(row[channel] is None for channel in required)
        if missing_required \
                and not int(row["validity_bits"]) & int(
                    QualityBits.INVALID_IMU_PAYLOAD):
            raise StoreInvariantError(
                f"IMU payload_kind {kind} missing channels require "
                "INVALID_IMU_PAYLOAD")
        for channel in all_channels:
            value = row[channel]
            if channel in required:
                if value is not None and not math.isfinite(value):
                    raise StoreInvariantError(
                        f"IMU payload_kind {kind} requires finite-or-null {channel}")
            elif value is not None:
                raise StoreInvariantError(
                    f"IMU payload_kind {kind} prohibits channel {channel}")
        if required & _QUAT and not missing_required:
            norm = math.sqrt(sum(row[channel] ** 2 for channel in _QUAT))
            if not math.isclose(
                    norm, 1.0, rel_tol=0.0,
                    abs_tol=_QUATERNION_NORM_ABS_TOL):
                raise StoreInvariantError(
                    "IMU quaternion payload must be unit-normalized within 1e-3")
        payload_kinds[(row["recording_id"], row["stream_id"])].add(kind)
    if any(len(kinds) != 1 for kinds in payload_kinds.values()):
        raise StoreInvariantError(
            "one IMU stream must use one invariant payload_kind/channel layout")


def _clock_segments(clock_table: pa.Table) -> tuple[list[ClockSegment], PiecewiseClockMap]:
    segments: list[ClockSegment] = []
    for row in clock_table.to_pylist():
        if row["mapping_status"] not in _ALLOWED_MAPPING_STATUS:
            raise StoreInvariantError(
                f"unknown clock mapping_status: {row['mapping_status']!r}")
        for field in ("residual_p50_ms", "residual_p95_ms"):
            value = row[field]
            if value is not None and (not math.isfinite(value) or value < 0):
                raise StoreInvariantError(
                    f"clock {field} must be finite and non-negative")
        p50 = row["residual_p50_ms"]
        p95 = row["residual_p95_ms"]
        if (p50 is None) != (p95 is None):
            raise StoreInvariantError("clock residual quantiles must be jointly nullable")
        if p50 is not None and p95 < p50:
            raise StoreInvariantError("clock residual_p95_ms must be >= residual_p50_ms")
        values = dict(row)
        stored_drift = values.pop("drift_ppm_derived")
        try:
            segment = ClockSegment(**values)
        except (TypeError, ClockMapError) as exc:
            raise StoreInvariantError(f"invalid clock map segment: {exc}") from exc
        if not math.isfinite(stored_drift) or not math.isclose(
                stored_drift, segment.drift_ppm, rel_tol=1e-12, abs_tol=1e-9):
            raise StoreInvariantError("stored clock drift_ppm_derived is not derived")
        segments.append(segment)
    try:
        return segments, PiecewiseClockMap(segments)
    except ClockMapError as exc:
        raise StoreInvariantError(f"invalid piecewise clock map: {exc}") from exc


def _clock_component_intervals(
    segments: list[ClockSegment], *, max_residual_p95_ms: float,
) -> dict[tuple[str, str], list[tuple[int, int, str]]]:
    """Return contiguous clock runs that satisfy mapping and residual policy."""

    grouped: dict[tuple[str, str], list[ClockSegment]] = defaultdict(list)
    for segment in segments:
        grouped[(segment.recording_id, segment.stream_id)].append(segment)
    result: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    for stream_key, members in grouped.items():
        ordered = sorted(members, key=lambda item: item.acquisition_ordinal)
        target_intervals = result[stream_key]
        active_start: int | None = None
        active_end: int | None = None
        active_component: str | None = None
        active_first_epoch: str | None = None

        def flush(target=target_intervals) -> None:
            nonlocal active_start, active_end, active_component, active_first_epoch
            if active_start is not None and active_end is not None \
                    and active_component is not None \
                    and active_first_epoch is not None:
                run_id = f"{active_component}:{active_first_epoch}"
                target.append((active_start, active_end, run_id))
            active_start = active_end = None
            active_component = active_first_epoch = None

        for segment in ordered:
            residual = segment.residual_p95_ms
            usable = (
                segment.mapping_status == "VALID"
                and residual is not None
                and residual <= max_residual_p95_ms
            )
            if not usable:
                flush()
                continue
            start_ns = segment.canonical_boundary_ns(segment.native_start_ns)
            end_ns = segment.canonical_boundary_ns(segment.native_end_ns)
            if active_start is None:
                active_start = start_ns
                active_end = end_ns
                active_component = segment.continuity_component_id
                active_first_epoch = segment.clock_epoch_id
            elif segment.continuity_component_id == active_component \
                    and start_ns == active_end:
                active_end = end_ns
            else:
                flush()
                active_start = start_ns
                active_end = end_ns
                active_component = segment.continuity_component_id
                active_first_epoch = segment.clock_epoch_id
        flush()
    for intervals in result.values():
        intervals.sort()
    return dict(result)


def _clock_component_policy_cache(
    clock_table: pa.Table, pairs: list[dict[str, object]],
) -> dict[float, dict[tuple[str, str], list[tuple[int, int, str]]]]:
    """Parse the clock table once and materialize each distinct residual policy."""

    segments, _ = _clock_segments(clock_table)
    thresholds = {
        float(pair["max_clock_residual_p95_ms"])
        for pair in pairs
    }
    return {
        threshold: _clock_component_intervals(
            segments, max_residual_p95_ms=threshold)
        for threshold in thresholds
    }


def _pair_alignment_intervals(
    *, recording_id: str, video_stream_id: str, imu_stream_id: str,
    clock_components: Mapping[
        tuple[str, str], list[tuple[int, int, str]]
    ],
    continuity_intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Intersect clock components, then split at internal ledger boundaries.

    Clock-component bounds, rather than the video-domain ledger start, bound
    nearest-sample lookup.  This preserves a valid pre-video IMU predecessor
    while still preventing a nearest sample from crossing a clock reset.  The
    ledger boundaries further isolate task/rate/QC regimes.
    """

    video = clock_components.get((recording_id, video_stream_id), ())
    imu = clock_components.get((recording_id, imu_stream_id), ())
    common: list[tuple[int, int]] = []
    video_index = 0
    imu_index = 0
    while video_index < len(video) and imu_index < len(imu):
        video_start, video_end, _ = video[video_index]
        imu_start, imu_end, _ = imu[imu_index]
        start = max(video_start, imu_start)
        end = min(video_end, imu_end)
        if start < end:
            common.append((start, end))
        if video_end <= imu_end:
            video_index += 1
        if imu_end <= video_end:
            imu_index += 1

    ledger_start = min(start for start, _ in continuity_intervals)
    ledger_end = max(end for _, end in continuity_intervals)
    boundaries = sorted({
        boundary
        for start, end in continuity_intervals
        for boundary in (start, end)
        if ledger_start < boundary < ledger_end
    })
    result: list[tuple[int, int]] = []
    for start, end in common:
        first = bisect_right(boundaries, start)
        last = bisect_left(boundaries, end, lo=first)
        cuts = [start]
        cuts.extend(boundaries[first:last])
        cuts.append(end)
        result.extend(
            (left, right)
            for left, right in pairwise(cuts)
            if left < right
        )
    return result


def _validate_materialized_clocks(
    frames: pa.Table, samples: pa.Table, clock_table: pa.Table,
) -> None:
    segments, mapping = _clock_segments(clock_table)
    map_streams = {(segment.recording_id, segment.stream_id) for segment in segments}
    source_streams = set(_rows_by_stream(frames, "video_stream_id")) \
        | set(_rows_by_stream(samples, "stream_id"))
    if map_streams != source_streams:
        raise StoreInvariantError("clock-map stream inventory is incomplete or unexpected")
    by_key = {segment.key: segment for segment in segments}
    fatal_clock_bits = int(
        QualityBits.UNRESOLVED_CLOCK_MAP | QualityBits.CLOCK_RESET
        | QualityBits.MISSING_TIMESTAMP)
    specifications = (
        (frames, "video_stream_id", "video_pts_native_ns", "quality_bits"),
        (samples, "stream_id", "sensor_time_native_ns", "validity_bits"),
    )
    for table, stream_field, native_field, quality_field in specifications:
        for row in table.to_pylist():
            key = (row["recording_id"], row[stream_field], row["clock_epoch_id"])
            segment = by_key.get(key)
            if segment is None:
                raise StoreInvariantError(f"materialized sample has no clock epoch: {key!r}")
            if not segment.contains(row[native_field], row["source_ordinal"]):
                raise StoreInvariantError(
                    f"materialized sample lies outside clock segment: {key!r}")
            if segment.mapping_status != "VALID":
                if not int(row[quality_field]) & fatal_clock_bits:
                    raise StoreInvariantError(
                        "sample under an unusable clock map lacks a fatal clock bit")
                continue
            try:
                mapping.validate_materialized(
                    recording_id=row["recording_id"], stream_id=row[stream_field],
                    clock_epoch_id=row["clock_epoch_id"],
                    source_ordinal=row["source_ordinal"],
                    native_time_ns=row[native_field],
                    canonical_time_ns=row["canonical_time_ns"],
                )
            except ClockMapError as exc:
                raise StoreInvariantError(
                    f"materialized canonical timestamp is invalid: {exc}") from exc


def _validate_vidimu_inventory_clock_state(
    frames: pa.Table, samples: pa.Table, clock_table: pa.Table,
) -> None:
    unresolved_bit = int(QualityBits.UNRESOLVED_CLOCK_MAP)
    clock_rows = clock_table.to_pylist()
    if any(
        row["mapping_status"] != "UNRESOLVED"
        or row["residual_p50_ms"] is not None
        or row["residual_p95_ms"] is not None
        for row in clock_rows
    ):
        raise StoreInvariantError(
            "VIDIMU inventory-only clock rows must be UNRESOLVED with null "
            "residuals")
    if any(
        not int(bits) & unresolved_bit
        for bits in frames["quality_bits"].to_pylist()
    ) or any(
        not int(bits) & unresolved_bit
        for bits in samples["validity_bits"].to_pylist()
    ):
        raise StoreInvariantError(
            "VIDIMU inventory-only source rows require UNRESOLVED_CLOCK_MAP")


def _table_equal(left: pa.Table, right: pa.Table, sort_keys: tuple[str, ...]) -> bool:
    if sort_keys:
        ordering = [(key, "ascending") for key in sort_keys]
        left = left.sort_by(ordering)
        right = right.sort_by(ordering)
    return left.combine_chunks().equals(right.combine_chunks(), check_metadata=True)


def _validate_alignment(
    frames: pa.Table, samples: pa.Table, observed: pa.Table,
    provenance: Mapping[str, object], *,
    clock_component_cache: Mapping[
        float, dict[tuple[str, str], list[tuple[int, int, str]]]
    ],
) -> None:
    pairs = _validate_alignment_plan(provenance["alignment_generation_parameters"])
    expected_tables: list[pa.Table] = []
    expected_pair_keys: set[tuple[str, str, str]] = set()
    frame_streams = _rows_by_stream(frames, "video_stream_id")
    imu_streams = _rows_by_stream(samples, "stream_id")
    _, continuity_segments = _validate_window_plan(
        provenance["window_generation_parameters"])
    continuity_by_pair: dict[
        tuple[str, str, str], list[tuple[int, int]]
    ] = defaultdict(list)
    continuity_rows_by_pair: dict[
        tuple[str, str, str], list[ContinuitySegment]
    ] = defaultdict(list)
    for segment in continuity_segments:
        key = (
            segment.recording_id, segment.video_stream_id,
            segment.imu_stream_id)
        continuity_by_pair[key].append(
            (segment.start_time_ns, segment.end_time_ns))
        continuity_rows_by_pair[key].append(segment)
    for pair in pairs:
        recording_id = pair["recording_id"]
        frame_rows = list(frame_streams.get(
            (recording_id, pair["video_stream_id"]), ()))
        imu_rows = list(imu_streams.get(
            (recording_id, pair["imu_stream_id"]), ()))
        if not frame_rows or not imu_rows:
            raise StoreInvariantError("alignment plan references a missing source stream")
        frame_rows.sort(key=lambda row: row["canonical_ordinal"])
        imu_rows.sort(key=lambda row: row["canonical_ordinal"])
        continuity_key = (
            recording_id, pair["video_stream_id"], pair["imu_stream_id"])
        ledger_intervals = sorted(continuity_by_pair[continuity_key])
        if ledger_intervals[0][0] != frame_rows[0]["canonical_time_ns"] \
                or ledger_intervals[-1][1] != pair["video_end_ns"] \
                or any(left[1] != right[0]
                       for left, right in pairwise(ledger_intervals)):
            raise StoreInvariantError(
                "continuity segments must form a gap-free partition of the "
                f"complete video domain for {continuity_key!r}")
        clock_components = clock_component_cache[
            float(pair["max_clock_residual_p95_ms"])]
        alignment_intervals = _pair_alignment_intervals(
            recording_id=recording_id,
            video_stream_id=pair["video_stream_id"],
            imu_stream_id=pair["imu_stream_id"],
            clock_components=clock_components,
            continuity_intervals=ledger_intervals,
        )
        if not alignment_intervals:
            unusable_clock_bits = int(
                QualityBits.UNRESOLVED_CLOCK_MAP
                | QualityBits.SYNC_RESIDUAL_EXCEEDED
                | QualityBits.CLOCK_RESET)
            if any(
                segment.accepted
                or not segment.quality_bits & unusable_clock_bits
                for segment in continuity_rows_by_pair[continuity_key]
            ):
                raise StoreInvariantError(
                    "alignment pair has no common clock continuity and is not "
                    f"fully rejected with a clock-failure reason: {continuity_key!r}")
        try:
            expected_tables.append(build_frame_imu_index(
                recording_id=recording_id,
                video_stream_id=pair["video_stream_id"],
                imu_stream_id=pair["imu_stream_id"],
                frame_indices=[row["frame_index"] for row in frame_rows],
                frame_canonical_ordinals=[row["canonical_ordinal"] for row in frame_rows],
                frame_times_ns=[row["canonical_time_ns"] for row in frame_rows],
                imu_canonical_ordinals=[row["canonical_ordinal"] for row in imu_rows],
                imu_times_ns=[row["canonical_time_ns"] for row in imu_rows],
                video_end_ns=pair["video_end_ns"],
                continuity_intervals_ns=alignment_intervals,
                max_imu_gap_ns=pair["max_imu_gap_ns"],
                min_coverage_fraction=pair["min_coverage_fraction"],
            ))
        except AlignmentError as exc:
            raise StoreInvariantError(f"alignment generation plan is invalid: {exc}") from exc
        expected_pair_keys.add((
            recording_id, pair["video_stream_id"], pair["imu_stream_id"]))
    observed_pair_keys = {
        (row["recording_id"], row["video_stream_id"], row["imu_stream_id"])
        for row in observed.to_pylist()
    }
    if observed_pair_keys != expected_pair_keys:
        raise StoreInvariantError("alignment pair inventory is incomplete or unexpected")
    if {(key[0], key[1]) for key in expected_pair_keys} != set(frame_streams):
        raise StoreInvariantError("alignment plan does not cover every video stream")
    if {(key[0], key[2]) for key in expected_pair_keys} != set(imu_streams):
        raise StoreInvariantError("alignment plan does not cover every IMU stream")
    expected = pa.concat_tables(expected_tables) if len(expected_tables) > 1 \
        else expected_tables[0]
    keys = ("recording_id", "video_stream_id", "imu_stream_id",
            "frame_canonical_ordinal")
    if not _table_equal(expected, observed, keys):
        raise StoreInvariantError(
            "stored alignment table differs from complete regenerated alignment")


def _validate_windows(
    frames: pa.Table, estimates: pa.Table, samples: pa.Table,
    observed_valid: pa.Table,
    observed_rejected: pa.Table, observed_alignment: pa.Table,
    *, clock_component_cache: Mapping[
        float, dict[tuple[str, str], list[tuple[int, int, str]]]
    ],
    provenance: Mapping[str, object], window_policy_id: str,
) -> None:
    plan, segments = _validate_window_plan(provenance["window_generation_parameters"])
    pairs = _validate_alignment_plan(
        provenance["alignment_generation_parameters"])
    pair_policies = {
        (pair["recording_id"], pair["video_stream_id"], pair["imu_stream_id"]):
        pair["max_clock_residual_p95_ms"]
        for pair in pairs
    }
    for segment in segments:
        if not segment.accepted:
            continue
        pair_key = (
            segment.recording_id, segment.video_stream_id,
            segment.imu_stream_id,
        )
        maximum_residual = float(pair_policies[pair_key])
        clock_components = clock_component_cache[maximum_residual]
        for stream_id in (segment.video_stream_id, segment.imu_stream_id):
            containing = [
                interval for interval in clock_components.get(
                    (segment.recording_id, stream_id), ())
                if interval[0] <= segment.start_time_ns
                and segment.end_time_ns <= interval[1]
            ]
            if len(containing) != 1:
                raise StoreInvariantError(
                    "continuity segment crosses or falls outside a clock-map "
                    f"continuity component: {segment.segment_id!r}, {stream_id!r}")
    try:
        regenerated = build_window_index(
            frame_index=frames,
            cv_estimates=estimates,
            imu_samples=samples,
            frame_imu_index=observed_alignment,
            continuity_segments=segments,
            window_ns=plan["window_ns"],
            hop_ns=plan["hop_ns"],
            window_policy_id=window_policy_id,
            observability_policy_id=provenance["observability_policy_id"],
            tremor_band_low_hz=plan["tremor_band_low_hz"],
            tremor_band_high_hz=plan["tremor_band_high_hz"],
            frequency_signal_policies=plan["frequency_signal_policies"],
            min_video_coverage=plan["min_video_coverage"],
            min_imu_coverage=plan["min_imu_coverage"],
            max_video_gap_ns=plan["max_video_gap_ns"],
            max_imu_gap_ns=plan["max_imu_gap_ns"],
            video_observability_factor=plan["video_observability_factor"],
            video_observability_cap_hz=plan["video_observability_cap_hz"],
            min_frequency_cycles=plan["min_frequency_cycles"],
            max_cadence_deviation_fraction=plan[
                "max_cadence_deviation_fraction"],
            min_tracking_quality=plan["min_tracking_quality"],
            min_valid_keypoint_fraction=plan[
                "min_valid_keypoint_fraction"],
        )
    except WindowIndexError as exc:
        raise StoreInvariantError(f"window generation plan is invalid: {exc}") from exc
    if not _table_equal(regenerated.valid_index, observed_valid, ("window_id",)):
        raise StoreInvariantError(
            "stored valid-window index differs from complete regenerated index")
    if not _table_equal(
            regenerated.rejection_ledger, observed_rejected,
            ("candidate_window_id",)):
        raise StoreInvariantError(
            "stored window rejection ledger differs from complete regenerated ledger")


def validate_snapshot_tables(
    tables: Mapping[str, pa.Table], *, recording_id: str,
    window_policy_id: str, provenance: Mapping[str, object],
) -> None:
    """Validate all canonical tables and regenerate every derived index."""

    if not isinstance(tables, Mapping) or set(tables) != set(SCHEMA_FACTORIES):
        raise StoreInvariantError("snapshot must contain exactly the seven canonical tables")
    if not isinstance(recording_id, str) or not recording_id:
        raise StoreInvariantError("recording_id is required")
    if not isinstance(window_policy_id, str) or not window_policy_id:
        raise StoreInvariantError("window_policy_id is required")
    if not isinstance(provenance, Mapping):
        raise StoreInvariantError("provenance must be an object")
    base_provenance = {
        key: value for key, value in provenance.items()
        if key not in RESERVED_PROVENANCE_KEYS
    }
    validate_provenance(base_provenance)
    recording_identity = _validate_recording_identity(
        base_provenance["recording_identity"],
        source_recording_id=base_provenance["source_recording_id"],
    )
    if recording_identity["stored_recording_id"] != recording_id:
        raise StoreInvariantError(
            "recording_identity stored ID disagrees with snapshot recording_id")

    for name, table in tables.items():
        if not isinstance(table, pa.Table):
            raise StoreInvariantError(f"{name} must be an Arrow table")
        if not schemas_equal(table.schema, canonical_schema(name)):
            raise StoreInvariantError(f"{name} does not use its canonical schema")
        if table.num_rows and "recording_id" in table.column_names:
            identifiers = set(table["recording_id"].unique().to_pylist())
            if identifiers != {recording_id}:
                raise StoreInvariantError(
                    f"{name} contains a different or mixed recording ID")
        sort_keys = _CANONICAL_ROW_SORT_KEYS[name]
        observed_order = [
            tuple(row[key] for key in sort_keys)
            for row in table.select(sort_keys).to_pylist()
        ]
        if observed_order != sorted(observed_order):
            raise StoreInvariantError(
                f"{name} rows are not stored in canonical sort order")

    for required in ("frame_index", "cv_estimates", "imu_samples", "clock_map"):
        if tables[required].num_rows == 0:
            raise StoreInvariantError(f"{required} must not be empty")
    for name in ("window_index", "window_rejections"):
        table = tables[name]
        if table.num_rows and set(table["window_policy_id"].unique().to_pylist()) \
                != {window_policy_id}:
            raise StoreInvariantError(f"{name} contains another window policy")

    frames = tables["frame_index"]
    estimates = tables["cv_estimates"]
    samples = tables["imu_samples"]
    pairs = _validate_alignment_plan(
        base_provenance["alignment_generation_parameters"])
    video_end_ns_by_stream = {
        (pair["recording_id"], pair["video_stream_id"]):
            int(pair["video_end_ns"])
        for pair in pairs
    }
    _validate_frames(
        frames, video_end_ns_by_stream=video_end_ns_by_stream)
    _validate_stream_semantics_inventory(frames, samples, base_provenance)
    _validate_cv(frames, estimates, base_provenance)
    _validate_imu(samples)
    if base_provenance["source_dataset"] == "VIDIMU":
        _validate_vidimu_inventory_clock_state(
            frames, samples, tables["clock_map"])
    _validate_materialized_clocks(frames, samples, tables["clock_map"])
    clock_component_cache = _clock_component_policy_cache(
        tables["clock_map"], pairs)
    _validate_alignment(
        frames, samples, tables["frame_imu_index"], base_provenance,
        clock_component_cache=clock_component_cache)
    _validate_windows(
        frames, estimates, samples,
        tables["window_index"], tables["window_rejections"],
        tables["frame_imu_index"],
        clock_component_cache=clock_component_cache,
        provenance=base_provenance,
        window_policy_id=window_policy_id)
