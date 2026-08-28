"""Atomic, strictly audited v0.4 source-to-CV recording bundles.

This module is deliberately independent of the v0.3 bundle writer.  A v0.4
success outcome contains four frame/CV tables and exactly three canonical JSON
documents.  The terminal marker is written last inside a private staging
directory, and the complete directory is then published with a no-replace
rename.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..decode.display_transform import display_transform
from ..decode.frame_identity import FINALIZATION_SCHEMA_VERSION, stable_frame_id
from ..decode.pts_decoder import DecodeConfig
from ..finalize._bundle_io import (
    PARQUET_WRITER_POLICY_ID,
    ROW_GROUP_SIZE,
    FinalizationBundleError,
    _exclusive_descriptor,
    _fsync_directory,
    _rename_noreplace,
    canonical_json_bytes,
    read_json,
    read_verified_table,
    safe_component,
    sha256_bytes,
    sha256_file,
)
from ..parquet_writer import semantic_table_hash
from ..schema import QualityBits, schema_fingerprint
from .detection_contract import validate_detection_and_selection_rows
from .schemas import (
    V04_ASSOCIATION_CONTRACT_VERSION,
    V04_SORT_KEYS,
    cv_detections_v04_schema,
    cv_frame_results_v04_schema,
    primary_hand_selection_schema,
    video_frames_v04_schema,
)

V04_BUNDLE_SCHEMA_VERSION = "tremora-vidimu-v04-finalization-bundle-1.0.0"
V04_FINALIZATION_MANIFEST_VERSION = 1
V04_FINALIZATION_AUDIT_VERSION = 1
V04_SUCCESS_MARKER_VERSION = 1
V04_FINALIZATION_ARTIFACT_KIND = "TREMORA_VIDIMU_V04_FRAME_FINALIZATION"
V04_FINALIZATION_IDENTITY_DOMAIN = "tremora-vidimu-v04-finalization-identity-1"

TABLE_FILES = {
    "video_frames": "video_frames.parquet",
    "cv_frame_results": "cv_frame_results.parquet",
    "cv_detections": "cv_detections.parquet",
    "primary_hand_selection": "primary_hand_selection.parquet",
}
TABLE_SORT_KEYS = dict(V04_SORT_KEYS)
TABLE_SCHEMAS = {
    "video_frames": video_frames_v04_schema,
    "cv_frame_results": cv_frame_results_v04_schema,
    "cv_detections": cv_detections_v04_schema,
    "primary_hand_selection": primary_hand_selection_schema,
}
V04_SUCCESS_FILES = frozenset({
    *TABLE_FILES.values(),
    "finalization_manifest.json",
    "finalization_audit.json",
    "_SUCCESS",
})

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TABLE_CLAIM_FIELDS = frozenset({
    "path",
    "bytes",
    "rows",
    "sha256",
    "semantic_sha256",
    "schema_sha256",
    "sort_keys",
    "row_group_size",
    "writer_policy_id",
})
_MANIFEST_FIELDS = frozenset({
    "artifact_kind",
    "manifest_version",
    "bundle_schema_version",
    "association_contract_version",
    "finalization_id",
    "recording_id",
    "dataset",
    "source_snapshot",
    "source_assets",
    "source_video",
    "decoder",
    "model",
    "execution",
    "scope",
    "tables",
})
_SCOPE = {
    "camera_only_inference": True,
    "decoded_frames_complete": True,
    "all_detections_persisted": True,
    "imu_parsing_performed": False,
    "clock_or_sync_estimation_performed": False,
    "windowing_or_spectral_analysis_performed": False,
}
_RUNTIME_METADATA_FIELDS = frozenset({
    "active_recording_id",
    "deterministic_mode",
    "inference_delegate",
    "inference_delegate_thread_count",
    "model_manifest_sha256",
    "num_hands",
    "persisted_detection_coordinate_space",
    "persisted_detection_dtype",
    "raw_detection_coordinate_space",
    "recording_state_generation",
    "running_mode",
    "runtime_lock_sha256",
    "runtime_worker_concurrency_observed",
    "stateless_per_frame",
    "whole_process_thread_count",
})


class V04BundleError(RuntimeError):
    """Raised when a v0.4 bundle cannot be proven complete and immutable."""


@dataclass(frozen=True, slots=True)
class PublishedV04Bundle:
    path: Path
    finalization_id: str
    manifest_sha256: str
    audit: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class AuditedV04Bundle:
    path: Path
    manifest: Mapping[str, object]
    audit: Mapping[str, object]
    marker: Mapping[str, object]
    tables: Mapping[str, pa.Table]


def _fail(message: str, cause: BaseException | None = None) -> V04BundleError:
    error = V04BundleError(message)
    if cause is not None:
        error.__cause__ = cause
    return error


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _fail(f"{field} must be a lowercase SHA-256")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(f"{field} must be an integer >= {minimum}")
    return value


def _safe_relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise _fail(f"{field} must be a canonical safe relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed.as_posix() != value or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise _fail(f"{field} must be a canonical safe relative path")
    return value


def finalization_identity(inputs: Mapping[str, object]) -> str:
    """Return the immutable v0.4 source/decoder/model processing identity."""

    required = {
        "source_snapshot_sha256",
        "source_video_sha256",
        "decoder_version",
        "decoder_config_sha256",
        "model_manifest_sha256",
        "association_contract_version",
        "bundle_schema_version",
    }
    value = dict(inputs)
    if set(value) != required:
        raise _fail("finalization identity inputs are not the frozen contract")
    for name in (
        "source_snapshot_sha256",
        "source_video_sha256",
        "decoder_config_sha256",
        "model_manifest_sha256",
    ):
        _sha256(value[name], f"identity.{name}")
    if not isinstance(value["decoder_version"], str) or not value["decoder_version"]:
        raise _fail("identity.decoder_version must be nonempty")
    if value["association_contract_version"] != V04_ASSOCIATION_CONTRACT_VERSION:
        raise _fail("identity association contract is not frozen v0.4")
    if value["bundle_schema_version"] != V04_BUNDLE_SCHEMA_VERSION:
        raise _fail("identity bundle schema is not frozen v0.4")
    payload = canonical_json_bytes({
        "domain": V04_FINALIZATION_IDENTITY_DOMAIN,
        **value,
    })
    return hashlib.sha256(payload).hexdigest()


def _table_claim(path: Path, table: pa.Table, sort_keys: tuple[str, ...]) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "rows": table.num_rows,
        "sha256": sha256_file(path),
        "semantic_sha256": semantic_table_hash(table, sort_keys=sort_keys),
        "schema_sha256": schema_fingerprint(table.schema),
        "sort_keys": list(sort_keys),
        "row_group_size": ROW_GROUP_SIZE,
        "writer_policy_id": PARQUET_WRITER_POLICY_ID,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = _exclusive_descriptor(path)
    with os.fdopen(descriptor, "wb", buffering=0) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_table(path: Path, table: pa.Table) -> None:
    descriptor = _exclusive_descriptor(path)
    with os.fdopen(descriptor, "w+b", buffering=0) as handle:
        pq.write_table(
            table,
            handle,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            data_page_version="2.0",
            row_group_size=ROW_GROUP_SIZE,
        )
        handle.flush()
        os.fsync(handle.fileno())


def _expected_audit(
    manifest: Mapping[str, object],
    tables: Mapping[str, pa.Table],
) -> dict[str, object]:
    frame_ids = tables["video_frames"].column("frame_id").to_pylist()
    result_ids = tables["cv_frame_results"].column("frame_id").to_pylist()
    selection_ids = tables["primary_hand_selection"].column("frame_id").to_pylist()
    detection_ids = tables["cv_detections"].column("detection_id").to_pylist()
    return {
        "artifact_kind": V04_FINALIZATION_ARTIFACT_KIND,
        "audit_version": V04_FINALIZATION_AUDIT_VERSION,
        "finalization_id": manifest["finalization_id"],
        "recording_id": manifest["recording_id"],
        "decoded_frames": len(frame_ids),
        "cv_frame_results": len(result_ids),
        "cv_detections": len(detection_ids),
        "primary_hand_selections": len(selection_ids),
        "every_decoded_frame_has_cv_result": (
            len(result_ids) == len(frame_ids) and set(result_ids) == set(frame_ids)
        ),
        "every_decoded_frame_has_selection": (
            len(selection_ids) == len(frame_ids)
            and set(selection_ids) == set(frame_ids)
        ),
        "all_detection_ids_unique": len(detection_ids) == len(set(detection_ids)),
        "source_video_hash_verified": True,
        "artifact_hashes_verified": True,
        "scope_exclusions_verified": True,
        "overall_verdict": "PASS",
    }


def _validate_source_assets(value: object, recording_id: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 3:
        raise _fail("source_assets must contain exactly VIDEO, IMU, and ANNOTATION")
    roles: list[str] = []
    expected_modalities = {
        "VIDEO": "VISUAL",
        "IMU": "INERTIAL_QUATERNION",
        "ANNOTATION": "BODYTRACK_POSE",
    }
    normalized: list[dict[str, object]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {
            "recording_id",
            "source_object_id",
            "archive_member_path",
            "normalized_member_path",
            "asset_role",
            "modality",
            "bytes",
            "sha256",
        }:
            raise _fail("source asset evidence is not closed")
        if raw["recording_id"] != recording_id:
            raise _fail("source asset recording identity mismatch")
        role = raw["asset_role"]
        if role not in {"VIDEO", "IMU", "ANNOTATION"}:
            raise _fail("source asset role is invalid")
        if not isinstance(raw["source_object_id"], str) or not raw["source_object_id"]:
            raise _fail("source asset source_object_id is invalid")
        if raw["modality"] != expected_modalities[role]:
            raise _fail("source asset modality contradicts the frozen VIDIMU topology")
        _safe_relative_path(raw["archive_member_path"], "archive_member_path")
        _safe_relative_path(
            raw["normalized_member_path"], "normalized_member_path"
        )
        _integer(raw["bytes"], "source asset bytes")
        _sha256(raw["sha256"], "source asset sha256")
        roles.append(role)
        normalized.append(dict(raw))
    canonical = sorted(
        normalized,
        key=lambda item: (
            str(item["asset_role"]),
            str(item["source_object_id"]),
            str(item["archive_member_path"]),
        ),
    )
    if roles.count("VIDEO") != 1 or roles.count("IMU") != 1 \
            or roles.count("ANNOTATION") != 1 or normalized != canonical:
        raise _fail("source asset topology or canonical order is invalid")
    return normalized


def _validate_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    value = dict(manifest)
    if set(value) != _MANIFEST_FIELDS:
        raise _fail("finalization manifest fields are not the frozen contract")
    if value["artifact_kind"] != V04_FINALIZATION_ARTIFACT_KIND \
            or type(value["manifest_version"]) is not int \
            or value["manifest_version"] != V04_FINALIZATION_MANIFEST_VERSION \
            or value["bundle_schema_version"] != V04_BUNDLE_SCHEMA_VERSION \
            or value["association_contract_version"] \
            != V04_ASSOCIATION_CONTRACT_VERSION:
        raise _fail("finalization manifest version contract is invalid")
    try:
        recording_id = safe_component(value["recording_id"], field="recording_id")
    except FinalizationBundleError as exc:
        raise _fail("recording_id is unsafe", exc) from exc
    finalization_id_value = _sha256(value["finalization_id"], "finalization_id")
    dataset = value["dataset"]
    if not isinstance(dataset, dict) or set(dataset) != {
        "dataset_id", "dataset_version", "license_id", "citation_id"
    } or any(not isinstance(dataset[name], str) or not dataset[name] for name in dataset):
        raise _fail("dataset provenance is invalid")
    snapshot = value["source_snapshot"]
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "snapshot_manifest_sha256", "source_inventory_sha256"
    }:
        raise _fail("source snapshot evidence is invalid")
    snapshot_sha = _sha256(
        snapshot["snapshot_manifest_sha256"], "snapshot manifest sha256"
    )
    _sha256(snapshot["source_inventory_sha256"], "source inventory sha256")
    assets = _validate_source_assets(value["source_assets"], recording_id)
    video_assets = [item for item in assets if item["asset_role"] == "VIDEO"]
    source_video = value["source_video"]
    if not isinstance(source_video, dict) or set(source_video) != {
        "asset_relative_path",
        "source_object_id",
        "bytes",
        "sha256",
        "stream_index",
        "hash_verified_during_pinned_decode",
    }:
        raise _fail("source_video evidence is invalid")
    if source_video["asset_relative_path"] != (
        f"assets/{video_assets[0]['normalized_member_path']}"
    ) or source_video["source_object_id"] != video_assets[0]["source_object_id"] \
            or source_video["bytes"] != video_assets[0]["bytes"] \
            or source_video["sha256"] != video_assets[0]["sha256"]:
        raise _fail("source_video evidence contradicts the frozen asset")
    _safe_relative_path(source_video["asset_relative_path"], "source video path")
    _integer(source_video["stream_index"], "source stream index")
    if source_video["hash_verified_during_pinned_decode"] is not True:
        raise _fail("source video was not hash verified during pinned decode")
    decoder = value["decoder"]
    if not isinstance(decoder, dict) or set(decoder) != {
        "decoder_version", "decoder_config", "decoder_config_sha256"
    } or not isinstance(decoder["decoder_version"], str) \
            or not decoder["decoder_version"] or not isinstance(
                decoder["decoder_config"], dict
            ):
        raise _fail("decoder evidence is invalid")
    if set(decoder["decoder_config"]) != set(asdict(DecodeConfig())):
        raise _fail("decoder config fields are not exact")
    try:
        decoder_config = DecodeConfig(**decoder["decoder_config"])
    except (TypeError, ValueError) as exc:
        raise _fail("decoder config cannot be reconstructed", exc) from exc
    decoder_hash = _sha256(decoder["decoder_config_sha256"], "decoder config sha256")
    if decoder_hash != decoder_config.sha256 \
            or source_video["stream_index"] != decoder_config.stream_index:
        raise _fail("decoder config hash or stream claim does not recompute")
    model = value["model"]
    if not isinstance(model, dict) or set(model) != {
        "model_manifest_sha256",
        "model_id",
        "model_weights_sha256",
        "preprocessing_config_sha256",
        "runtime_lock_sha256",
        "inference_environment_id",
    }:
        raise _fail("model evidence is invalid")
    model_manifest_sha = _sha256(
        model["model_manifest_sha256"], "model manifest sha256"
    )
    for name in (
        "model_weights_sha256", "preprocessing_config_sha256", "runtime_lock_sha256"
    ):
        _sha256(model[name], f"model.{name}")
    if not isinstance(model["model_id"], str) or not model["model_id"] \
            or not isinstance(model["inference_environment_id"], str) \
            or not model["inference_environment_id"]:
        raise _fail("model identity strings are invalid")
    if model["inference_environment_id"] != (
        f"native-runtime-sha256:{model['runtime_lock_sha256']}"
    ):
        raise _fail("inference environment is not bound to the runtime lock")
    execution = value["execution"]
    if not isinstance(execution, dict) or set(execution) != {
        "execution_id", "execution_receipt_sha256"
    }:
        raise _fail("execution receipt binding is invalid")
    _sha256(execution["execution_id"], "execution_id")
    _sha256(execution["execution_receipt_sha256"], "execution receipt sha256")
    if value["scope"] != _SCOPE:
        raise _fail("scope is not the frozen v0.4 frame-only boundary")
    table_claims = value["tables"]
    if not isinstance(table_claims, dict) or set(table_claims) != set(TABLE_FILES):
        raise _fail("table inventory is not exact")
    for name, claim in table_claims.items():
        if not isinstance(claim, dict) or set(claim) != _TABLE_CLAIM_FIELDS \
                or claim["path"] != TABLE_FILES[name]:
            raise _fail(f"{name} artifact claim is invalid")
        _integer(claim["bytes"], f"{name} bytes")
        _integer(claim["rows"], f"{name} rows")
        _sha256(claim["sha256"], f"{name} sha256")
        _sha256(claim["semantic_sha256"], f"{name} semantic sha256")
        _sha256(claim["schema_sha256"], f"{name} schema sha256")
        if claim["sort_keys"] != list(TABLE_SORT_KEYS[name]) \
                or claim["row_group_size"] != ROW_GROUP_SIZE \
                or claim["writer_policy_id"] != PARQUET_WRITER_POLICY_ID:
            raise _fail(f"{name} writer contract is invalid")
    identity = {
        "source_snapshot_sha256": snapshot_sha,
        "source_video_sha256": source_video["sha256"],
        "decoder_version": decoder["decoder_version"],
        "decoder_config_sha256": decoder_hash,
        "model_manifest_sha256": model_manifest_sha,
        "association_contract_version": value["association_contract_version"],
        "bundle_schema_version": value["bundle_schema_version"],
    }
    if finalization_identity(identity) != finalization_id_value:
        raise _fail("finalization identity does not recompute")
    return value


def _round_fraction_ns(value: Fraction, field: str) -> int:
    scaled = value * 1_000_000_000
    if scaled < 0:
        raise _fail(f"{field} cannot be negative")
    return (scaled.numerator * 2 + scaled.denominator) // (
        2 * scaled.denominator
    )


def _validate_tables(
    tables: Mapping[str, pa.Table],
    manifest: Mapping[str, object],
) -> None:
    if set(tables) != set(TABLE_FILES):
        raise _fail("bundle table set is incomplete")
    for name, table in tables.items():
        if not isinstance(table, pa.Table):
            raise _fail(f"{name} is not an Arrow table")
        expected_schema = TABLE_SCHEMAS[name]()
        if schema_fingerprint(table.schema) != schema_fingerprint(expected_schema):
            raise _fail(f"{name} schema is not frozen v0.4")
        sorted_table = table.sort_by([
            (key, "ascending") for key in TABLE_SORT_KEYS[name]
        ])
        if not table.equals(sorted_table):
            raise _fail(f"{name} rows are not canonically sorted")
    frame_rows = tables["video_frames"].to_pylist()
    result_rows = tables["cv_frame_results"].to_pylist()
    frame_ids = [row["frame_id"] for row in frame_rows]
    if not frame_ids or len(frame_ids) != len(set(frame_ids)):
        raise _fail("video frame IDs must be nonempty and unique")
    if sorted(row["decode_ordinal"] for row in frame_rows) != list(range(len(frame_rows))):
        raise _fail("decoded frame ordinals must be contiguous")
    dataset = manifest["dataset"]
    source_video = manifest["source_video"]
    decoder = manifest["decoder"]
    model = manifest["model"]
    if not all(isinstance(item, Mapping) for item in (
        dataset, source_video, decoder, model
    )):
        raise _fail("manifest provenance objects changed after validation")
    pts_groups: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    frame_by_id: dict[str, dict[str, object]] = {}
    time_bases: set[tuple[int, int]] = set()
    frame_quality_mask = int(
        QualityBits.MISSING_TIMESTAMP
        | QualityBits.NON_MONOTONIC_TIMESTAMP
        | QualityBits.DUPLICATE_TIMESTAMP
        | QualityBits.STREAM_GAP
        | QualityBits.DECODE_FAILURE
    )
    for row in frame_rows:
        frame_id = _sha256(row["frame_id"], "video frame_id")
        source_hash = _sha256(
            row["source_video_sha256"], "frame source_video_sha256"
        )
        numerator = _integer(row["time_base_num"], "time_base_num", minimum=1)
        denominator = _integer(row["time_base_den"], "time_base_den", minimum=1)
        time_bases.add((numerator, denominator))
        pts = row["pts"]
        if pts is not None and (
            isinstance(pts, bool) or not isinstance(pts, int)
        ):
            raise _fail("frame PTS must be an integer or null")
        if row["dataset_id"] != dataset["dataset_id"] \
                or row["recording_id"] != manifest["recording_id"] \
                or source_hash != source_video["sha256"] \
                or row["stream_index"] != source_video["stream_index"] \
                or row["decoder_version"] != decoder["decoder_version"] \
                or row["schema_version"] != FINALIZATION_SCHEMA_VERSION:
            raise _fail("video frame provenance contradicts the manifest")
        if type(row["key_frame"]) is not bool \
                or not isinstance(row["picture_type"], str) \
                or not row["picture_type"] \
                or not isinstance(row["pixel_format"], str) \
                or not row["pixel_format"]:
            raise _fail("video frame decode-format evidence is incomplete")
        try:
            expected_id, expected_basis = stable_frame_id(
                source_video_sha256=source_hash,
                stream_index=row["stream_index"],
                pts=row["pts"],
                same_pts_rank=row["same_pts_rank"],
                decode_ordinal=row["decode_ordinal"],
            )
        except ValueError as exc:
            raise _fail("video frame identity inputs are invalid", exc) from exc
        if frame_id != expected_id or row["identity_basis"] != expected_basis:
            raise _fail("video frame identity does not recompute")
        quality = _integer(row["quality_bits"], "frame quality_bits")
        if quality & ~frame_quality_mask:
            raise _fail("video frame uses quality bits outside decoder scope")
        if (row["decode_status"] == "CORRUPT") != bool(
            quality & int(QualityBits.DECODE_FAILURE)
        ) or row["decode_status"] not in {"SUCCESS", "CORRUPT"}:
            raise _fail("frame decode status and quality bits disagree")
        duration_pts = row["duration_pts"]
        duration_ns = row["duration_ns"]
        if (duration_pts is None) != (duration_ns is None):
            raise _fail("frame duration is partially populated")
        if duration_pts is not None:
            duration_ticks = _integer(duration_pts, "duration_pts")
            expected_duration = _round_fraction_ns(
                Fraction(duration_ticks * numerator, denominator),
                "duration_ns",
            )
            if duration_ns != expected_duration:
                raise _fail("duration_ns is not rationally derived")
        try:
            expected_transform = display_transform(
                row["coded_width"], row["coded_height"], row["rotation_degrees"]
            )
        except ValueError as exc:
            raise _fail("video display transform inputs are invalid", exc) from exc
        matrix = np.asarray(row["source_to_display_transform"], dtype=np.float64)
        if matrix.shape != (9,) or not np.isfinite(matrix).all() \
                or not np.allclose(
                    matrix,
                    expected_transform.source_to_display,
                    rtol=0.0,
                    atol=1e-9,
                ) or row["display_width"] != expected_transform.display_width \
                or row["display_height"] != expected_transform.display_height:
            raise _fail("video source/display transform is not canonical")
        cv_fields = (
            row["display_to_cv_transform"],
            row["cv_to_source_transform"],
            row["cv_input_width"],
            row["cv_input_height"],
            row["cv_input_pixel_format"],
            row["cv_input_sha256"],
        )
        cv_present = [item is not None for item in cv_fields]
        if any(cv_present) != all(cv_present):
            raise _fail("video frame has a partial CV-input description")
        if all(cv_present):
            _sha256(row["cv_input_sha256"], "cv_input_sha256")
            if row["cv_input_width"] < 1 or row["cv_input_height"] < 1 \
                    or row["cv_input_pixel_format"] != "bgr24" \
                    or row["preprocessing_transform_invertible"] is not True:
                raise _fail("materialized CV input contract is invalid")
            try:
                display_to_cv = np.asarray(
                    row["display_to_cv_transform"], dtype=np.float64
                ).reshape(3, 3)
                cv_to_source = np.asarray(
                    row["cv_to_source_transform"], dtype=np.float64
                ).reshape(3, 3)
            except (TypeError, ValueError) as exc:
                raise _fail("CV/source transform shape is invalid", exc) from exc
            source_to_display = matrix.reshape(3, 3)
            source_to_cv = display_to_cv @ source_to_display
            if row["cv_input_width"] != row["display_width"] \
                    or row["cv_input_height"] != row["display_height"] \
                    or not np.array_equal(display_to_cv, np.eye(3)):
                raise _fail("frozen full-frame/no-resize preprocessing changed")
            if not np.isfinite(display_to_cv).all() \
                    or not np.isfinite(cv_to_source).all() \
                    or abs(float(np.linalg.det(source_to_cv))) <= 1e-12 \
                    or not np.allclose(
                        cv_to_source @ source_to_cv,
                        np.eye(3),
                        rtol=0.0,
                        atol=1e-8,
                    ):
                raise _fail("CV/source transforms do not form a canonical inverse")
        elif row["preprocessing_transform_invertible"] is not False:
            raise _fail("frame without CV input claims an invertible transform")
        if row["pts"] is None:
            if row["same_pts_rank"] != 0 or row["pts_status"] != "MISSING" \
                    or row["presentation_ordinal"] is not None \
                    or row["relative_pts_ns"] is not None \
                    or row["gap_before_ns"] is not None:
                raise _fail("missing-PTS frame contains fabricated timing")
        else:
            pts_groups[row["pts"]].append(row)
        frame_by_id[frame_id] = row
    if len(time_bases) != 1:
        raise _fail("one decoded stream must have one rational time base")
    ordered_decode = sorted(frame_rows, key=lambda item: item["decode_ordinal"])
    for rows in pts_groups.values():
        ordered = sorted(rows, key=lambda item: item["decode_ordinal"])
        if [row["same_pts_rank"] for row in ordered] != list(range(len(ordered))):
            raise _fail("same-PTS ranks are not canonical decode order")
    previous_timestamp: Fraction | None = None
    nonmonotonic_ids: set[str] = set()
    for row in ordered_decode:
        if row["pts"] is None:
            continue
        timestamp = Fraction(
            row["pts"] * row["time_base_num"], row["time_base_den"]
        )
        if previous_timestamp is not None and timestamp < previous_timestamp:
            nonmonotonic_ids.add(row["frame_id"])
        previous_timestamp = timestamp
    valid_rows = [row for row in frame_rows if row["pts"] is not None]
    presentation = sorted(valid_rows, key=lambda row: (
        Fraction(row["pts"] * row["time_base_num"], row["time_base_den"]),
        row["same_pts_rank"],
        row["decode_ordinal"],
    ))
    if [row["presentation_ordinal"] for row in presentation] != list(
        range(len(presentation))
    ):
        raise _fail("presentation ordinals do not follow rational PTS order")
    discontinuity_ids: set[str] = set()
    if presentation:
        origin = Fraction(
            presentation[0]["pts"] * presentation[0]["time_base_num"],
            presentation[0]["time_base_den"],
        )
        previous_timestamp = None
        threshold = decoder["decoder_config"]["discontinuity_threshold_ns"]
        for row in presentation:
            timestamp = Fraction(
                row["pts"] * row["time_base_num"], row["time_base_den"]
            )
            expected_relative = _round_fraction_ns(
                timestamp - origin, "relative_pts_ns"
            )
            expected_gap = None if previous_timestamp is None else (
                _round_fraction_ns(timestamp - previous_timestamp, "gap_before_ns")
            )
            if row["relative_pts_ns"] != expected_relative \
                    or row["gap_before_ns"] != expected_gap:
                raise _fail("relative PTS or presentation gap is not canonical")
            if expected_gap is not None and expected_gap > threshold:
                discontinuity_ids.add(row["frame_id"])
            previous_timestamp = timestamp
    for row in frame_rows:
        frame_id = row["frame_id"]
        missing = row["pts"] is None
        duplicate = not missing and len(pts_groups[row["pts"]]) > 1
        nonmonotonic = frame_id in nonmonotonic_ids
        discontinuity = frame_id in discontinuity_ids
        expected_timing_bits = 0
        if missing:
            expected_timing_bits |= int(QualityBits.MISSING_TIMESTAMP)
        if duplicate:
            expected_timing_bits |= int(QualityBits.DUPLICATE_TIMESTAMP)
        if nonmonotonic:
            expected_timing_bits |= int(QualityBits.NON_MONOTONIC_TIMESTAMP)
        if discontinuity:
            expected_timing_bits |= int(QualityBits.STREAM_GAP)
        observed_timing_bits = int(row["quality_bits"]) & int(
            QualityBits.MISSING_TIMESTAMP
            | QualityBits.NON_MONOTONIC_TIMESTAMP
            | QualityBits.DUPLICATE_TIMESTAMP
            | QualityBits.STREAM_GAP
        )
        if observed_timing_bits != expected_timing_bits:
            raise _fail("PTS timing quality bits are not canonically derived")
        expected_status = (
            "MISSING" if missing else
            "DUPLICATE" if duplicate else
            "NON_MONOTONIC" if nonmonotonic else
            "DISCONTINUITY" if discontinuity else
            "VALID"
        )
        if row["pts_status"] != expected_status:
            raise _fail("PTS status is not canonically derived")
    # CV rows have their own canonical sort order; compare exact coverage.
    if {row["frame_id"] for row in result_rows} != set(frame_ids) \
            or len(result_rows) != len(frame_ids):
        raise _fail("CV result rows do not cover decoded frames exactly")
    result_by_frame = {row["frame_id"]: row for row in result_rows}
    detections = tables["cv_detections"].to_pylist()
    detection_counts = Counter(row["frame_id"] for row in detections)
    result_quality_mask = frame_quality_mask | int(QualityBits.INVALID_CV)
    for row in result_rows:
        frame = frame_by_id[row["frame_id"]]
        status = row["inference_status"]
        if status not in {
            "SUCCESS",
            "NO_DETECTION",
            "DECODE_FAILURE",
            "PREPROCESS_FAILURE",
            "INFERENCE_FAILURE",
        }:
            raise _fail("CV frame result status is outside production v0.4")
        count = _integer(row["detection_count"], "detection_count")
        if row["recording_id"] != manifest["recording_id"] \
                or row["relative_pts_ns"] != frame["relative_pts_ns"] \
                or row["model_manifest_sha256"] \
                != model["model_manifest_sha256"] \
                or row["model_id"] != model["model_id"] \
                or row["model_weights_sha256"] != model["model_weights_sha256"] \
                or row["preprocessing_config_sha256"] \
                != model["preprocessing_config_sha256"] \
                or row["runtime_lock_sha256"] != model["runtime_lock_sha256"] \
                or row["association_contract_version"] \
                != V04_ASSOCIATION_CONTRACT_VERSION \
                or row["inference_environment_id"] \
                != model["inference_environment_id"]:
            raise _fail("CV frame result provenance contradicts the manifest/frame")
        if count != detection_counts[row["frame_id"]] \
                or (status == "SUCCESS") != (count > 0):
            raise _fail("CV result status/detection count is inconsistent")
        has_cv_input = frame["cv_input_sha256"] is not None
        if status in {"SUCCESS", "NO_DETECTION", "INFERENCE_FAILURE"} \
                and not has_cv_input \
                or status in {"PREPROCESS_FAILURE", "DECODE_FAILURE"} \
                and has_cv_input:
            raise _fail("CV result status contradicts preprocessing evidence")
        failure_reason = row["failure_reason"]
        if status in {"SUCCESS", "NO_DETECTION"}:
            if failure_reason is not None:
                raise _fail("successful CV result has a failure reason")
        elif not isinstance(failure_reason, str) or not failure_reason:
            raise _fail("failed CV result lacks a stable failure reason")
        if status == "DECODE_FAILURE" \
                and failure_reason != "DECODE_STATUS_CORRUPT":
            raise _fail("decode failure reason is not canonical")
        expected_quality = int(frame["quality_bits"])
        if status in {"PREPROCESS_FAILURE", "INFERENCE_FAILURE", "REJECTED_INPUT"}:
            expected_quality |= int(QualityBits.INVALID_CV)
        quality = _integer(row["frame_quality_bits"], "frame_quality_bits")
        if quality & ~result_quality_mask or quality != expected_quality:
            raise _fail("CV result quality bits are not canonically derived")
        if status == "DECODE_FAILURE" and frame["decode_status"] != "CORRUPT" \
                or frame["decode_status"] == "CORRUPT" \
                and status != "DECODE_FAILURE":
            raise _fail("decode failure status does not match the decoded frame")
        if row["runtime_ms"] is not None:
            raise _fail("binding v0.4 output may not persist runtime timing")
        if row["tracking_quality"] is not None:
            raise _fail("frozen IMAGE-mode inference may not claim tracking quality")
        transform = row["preprocessing_transform"]
        runtime_json = row["runtime_metadata_json"]
        if status in {"SUCCESS", "NO_DETECTION", "INFERENCE_FAILURE"}:
            if transform is None or runtime_json is None:
                raise _fail("inference result lacks transform/runtime provenance")
            if list(transform) != list(frame["display_to_cv_transform"]):
                raise _fail("result preprocessing transform contradicts frame input")
            if not isinstance(runtime_json, str):
                raise _fail("runtime metadata must be canonical JSON text")
            try:
                runtime_value = json.loads(runtime_json)
            except json.JSONDecodeError as exc:
                raise _fail("runtime metadata JSON is invalid", exc) from exc
            expected_runtime = {
                "active_recording_id": manifest["recording_id"],
                "deterministic_mode": False,
                "inference_delegate": "CPU",
                "inference_delegate_thread_count": None,
                "model_manifest_sha256": model["model_manifest_sha256"],
                "num_hands": 2,
                "persisted_detection_coordinate_space": "DISPLAY_PIXEL",
                "persisted_detection_dtype": "float32",
                "raw_detection_coordinate_space": "NORMALIZED_CV_INPUT",
                "recording_state_generation": 1,
                "running_mode": "IMAGE",
                "runtime_lock_sha256": model["runtime_lock_sha256"],
                "runtime_worker_concurrency_observed": 11,
                "stateless_per_frame": True,
                "whole_process_thread_count": None,
            }
            if not isinstance(runtime_value, dict) \
                    or set(runtime_value) != _RUNTIME_METADATA_FIELDS \
                    or canonical_json_bytes(runtime_value).decode("ascii") \
                    != runtime_json \
                    or canonical_json_bytes(runtime_value) \
                    != canonical_json_bytes(expected_runtime):
                raise _fail("runtime metadata contradicts frozen provenance")
        elif transform is not None or runtime_json is not None:
            raise _fail("pre-inference failure fabricates runtime provenance")
    selections = tables["primary_hand_selection"].to_pylist()
    for selection in selections:
        result = result_by_frame.get(selection["frame_id"])
        if result is None or selection["inference_status"] != result["inference_status"]:
            raise _fail("selection status contradicts CV frame result")
    try:
        validate_detection_and_selection_rows(
            tables["cv_detections"],
            tables["primary_hand_selection"],
            frame_ids=frame_ids,
        )
    except ValueError as exc:
        raise _fail("detection/selection contract failed", exc) from exc
    model_hash = model["model_manifest_sha256"]
    if any(
        row["model_manifest_sha256"] != model_hash
        for row in detections
    ) or any(
        row["model_manifest_sha256"] != model_hash
        for row in result_rows
    ):
        raise _fail("frame/detection rows mix a model manifest identity")


def _abort_owned(staging: Path, identity: tuple[int, int]) -> None:
    try:
        state = staging.stat(follow_symlinks=False)
    except OSError:
        return
    if not stat.S_ISDIR(state.st_mode) or (state.st_dev, state.st_ino) != identity:
        return
    marker = staging / "_SUCCESS"
    if marker.is_file() and not marker.is_symlink():
        marker.unlink(missing_ok=True)
    allowed = set(V04_SUCCESS_FILES)
    try:
        entries = list(staging.iterdir())
    except OSError:
        return
    if any(entry.name not in allowed or entry.is_symlink() or not entry.is_file()
           for entry in entries):
        return
    for entry in entries:
        entry.unlink()
    staging.rmdir()


def publish_v04_bundle(
    finalization_root: str | Path,
    *,
    recording_id: str,
    finalization_id: str,
    tables: Mapping[str, pa.Table],
    manifest: Mapping[str, object],
) -> PublishedV04Bundle:
    """Atomically publish one complete seven-file recording success bundle."""

    try:
        recording = safe_component(recording_id, field="recording_id")
        identity = safe_component(finalization_id, field="finalization_id")
    except FinalizationBundleError as exc:
        raise _fail("unsafe recording/finalization identity", exc) from exc
    if _SHA256_RE.fullmatch(identity) is None:
        raise _fail("finalization_id must be a lowercase SHA-256")
    root = Path(finalization_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise _fail("finalization root must be a real directory")
    root = root.resolve(strict=True)
    recording_dir = root / recording
    recording_dir.mkdir(exist_ok=True)
    if recording_dir.is_symlink() or recording_dir.resolve(strict=True).parent != root:
        raise _fail("recording directory escaped the finalization root")
    destination = recording_dir / identity
    if destination.exists() or destination.is_symlink():
        raise _fail("finalization identity already exists")
    staging = recording_dir / f".staging-{identity}-{uuid.uuid4().hex}"
    staging.mkdir()
    state = staging.stat()
    staging_identity = (state.st_dev, state.st_ino)
    published = False
    try:
        prepared_tables: dict[str, pa.Table] = {}
        claims: dict[str, dict[str, object]] = {}
        for name, filename in TABLE_FILES.items():
            if name not in tables:
                raise _fail("bundle table set is incomplete")
            table = tables[name].sort_by([
                (key, "ascending") for key in TABLE_SORT_KEYS[name]
            ])
            target = staging / filename
            _write_table(target, table)
            prepared_tables[name] = table
            claims[name] = _table_claim(target, table, TABLE_SORT_KEYS[name])
        manifest_value = dict(manifest)
        manifest_value["tables"] = claims
        if manifest_value.get("recording_id") != recording \
                or manifest_value.get("finalization_id") != identity:
            raise _fail("manifest identity does not match destination")
        validated_manifest = _validate_manifest(manifest_value)
        _validate_tables(prepared_tables, validated_manifest)
        manifest_bytes = canonical_json_bytes(validated_manifest)
        _write_exclusive(staging / "finalization_manifest.json", manifest_bytes)
        audit = _expected_audit(validated_manifest, prepared_tables)
        if not all((
            audit["every_decoded_frame_has_cv_result"],
            audit["every_decoded_frame_has_selection"],
            audit["all_detection_ids_unique"],
        )):
            raise _fail("recording reconciliation did not pass")
        audit_bytes = canonical_json_bytes(audit)
        _write_exclusive(staging / "finalization_audit.json", audit_bytes)
        marker = {
            "artifact_kind": V04_FINALIZATION_ARTIFACT_KIND,
            "success_marker_version": V04_SUCCESS_MARKER_VERSION,
            "finalization_id": identity,
            "manifest_sha256": sha256_bytes(manifest_bytes),
            "audit_sha256": sha256_bytes(audit_bytes),
        }
        _write_exclusive(staging / "_SUCCESS", canonical_json_bytes(marker))
        if {entry.name for entry in staging.iterdir()} != V04_SUCCESS_FILES:
            raise _fail("staged success file inventory is not exact")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW \
            | getattr(os, "O_CLOEXEC", 0)
        recording_descriptor = os.open(recording_dir, flags)
        staging_descriptor = -1
        try:
            staging_descriptor = os.open(staging.name, flags, dir_fd=recording_descriptor)
            pinned = os.fstat(staging_descriptor)
            if (pinned.st_dev, pinned.st_ino) != staging_identity \
                    or not stat.S_ISDIR(pinned.st_mode):
                raise _fail("staging identity changed before publication")
            for name in V04_SUCCESS_FILES:
                artifact = os.stat(name, dir_fd=staging_descriptor, follow_symlinks=False)
                if not stat.S_ISREG(artifact.st_mode):
                    raise _fail("staged artifacts must be regular files")
            os.fsync(staging_descriptor)
            os.fsync(recording_descriptor)
            try:
                _rename_noreplace(recording_descriptor, staging.name, identity)
            except FinalizationBundleError as exc:
                raise _fail("finalization publication collided", exc) from exc
            os.fsync(recording_descriptor)
        finally:
            if staging_descriptor >= 0:
                os.close(staging_descriptor)
            os.close(recording_descriptor)
        _fsync_directory(root)
        published = True
        audited = audit_v04_bundle(destination)
        return PublishedV04Bundle(
            path=destination.resolve(strict=True),
            finalization_id=identity,
            manifest_sha256=str(audited.marker["manifest_sha256"]),
            audit=audited.audit,
        )
    finally:
        if not published:
            _abort_owned(staging, staging_identity)


def audit_v04_bundle(path: str | Path) -> AuditedV04Bundle:
    """Strictly re-read, hash, and semantically reconcile one success bundle."""

    bundle = Path(path)
    if bundle.is_symlink() or not bundle.is_dir():
        raise _fail("bundle must be a real directory")
    bundle = bundle.resolve(strict=True)
    if _SHA256_RE.fullmatch(bundle.name) is None:
        raise _fail("bundle directory is not a finalization identity")
    actual_names = {entry.name for entry in bundle.iterdir()}
    if actual_names != V04_SUCCESS_FILES:
        raise _fail("bundle file inventory is not exact")
    for name in actual_names:
        state = (bundle / name).lstat()
        if not stat.S_ISREG(state.st_mode) or stat.S_ISLNK(state.st_mode):
            raise _fail("bundle artifacts must be non-symlink regular files")
    try:
        manifest, manifest_bytes = read_json(bundle / "finalization_manifest.json")
        audit, audit_bytes = read_json(bundle / "finalization_audit.json")
        marker, _marker_bytes = read_json(bundle / "_SUCCESS")
    except FinalizationBundleError as exc:
        raise _fail("bundle JSON verification failed", exc) from exc
    validated_manifest = _validate_manifest(manifest)
    if validated_manifest["finalization_id"] != bundle.name:
        raise _fail("bundle directory and manifest identity disagree")
    expected_marker = {
        "artifact_kind": V04_FINALIZATION_ARTIFACT_KIND,
        "success_marker_version": V04_SUCCESS_MARKER_VERSION,
        "finalization_id": bundle.name,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "audit_sha256": sha256_bytes(audit_bytes),
    }
    if marker != expected_marker:
        raise _fail("success marker is invalid")
    tables: dict[str, pa.Table] = {}
    claims = validated_manifest["tables"]
    if not isinstance(claims, dict):
        raise _fail("validated table claims changed type")
    for name, filename in TABLE_FILES.items():
        claim = claims[name]
        if not isinstance(claim, dict):
            raise _fail(f"validated {name} claim changed type")
        try:
            table, row_groups = read_verified_table(
                bundle / filename,
                expected_bytes=_integer(claim["bytes"], f"{name} bytes"),
                expected_sha256=_sha256(claim["sha256"], f"{name} sha256"),
            )
        except FinalizationBundleError as exc:
            raise _fail(f"{name} verified read failed", exc) from exc
        if claim["rows"] != table.num_rows \
                or claim["schema_sha256"] != schema_fingerprint(table.schema) \
                or claim["semantic_sha256"] != semantic_table_hash(
                    table, sort_keys=TABLE_SORT_KEYS[name]
                ):
            raise _fail(f"{name} artifact claims do not recompute")
        expected_row_groups = (
            [ROW_GROUP_SIZE] * (table.num_rows // ROW_GROUP_SIZE)
            + (
                [table.num_rows % ROW_GROUP_SIZE]
                if table.num_rows % ROW_GROUP_SIZE
                else []
            )
        )
        if table.num_rows == 0:
            # PyArrow writes one zero-row group for a schema-bearing empty
            # table. This is the same frozen policy used by Gate A.
            expected_row_groups = [0]
        if row_groups != expected_row_groups:
            raise _fail(f"{name} row-group policy changed")
        tables[name] = table
    _validate_tables(tables, validated_manifest)
    expected_audit = _expected_audit(validated_manifest, tables)
    if audit != expected_audit:
        raise _fail("finalization audit does not recompute")
    return AuditedV04Bundle(
        path=bundle,
        manifest=validated_manifest,
        audit=audit,
        marker=marker,
        tables=tables,
    )


__all__ = [
    "TABLE_FILES",
    "TABLE_SCHEMAS",
    "TABLE_SORT_KEYS",
    "V04_BUNDLE_SCHEMA_VERSION",
    "V04_FINALIZATION_ARTIFACT_KIND",
    "V04_SUCCESS_FILES",
    "AuditedV04Bundle",
    "PublishedV04Bundle",
    "V04BundleError",
    "audit_v04_bundle",
    "finalization_identity",
    "publish_v04_bundle",
]
