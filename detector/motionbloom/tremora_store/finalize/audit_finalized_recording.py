"""Strict independent audit for one immutable v0.3 frame-finalization bundle."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path, PurePosixPath

import pyarrow as pa

from ..cv.coordinate_mapping import CV_INPUT_HASH_VERSION, PIXEL_CONVENTION
from ..cv.offline_finalizer import SELECTION_POLICY_ID
from ..cv.pose_frame_contract import validate_pose_frame_association
from ..decode.frame_identity import (
    ASSOCIATION_SCHEMA_VERSION,
    FINALIZATION_SCHEMA_VERSION,
    finalization_identity,
)
from ..decode.pts_decoder import DecodeConfig
from ..parquet_writer import semantic_table_hash
from ..schema import QualityBits, schema_fingerprint
from ..schemas import FINALIZATION_SORT_KEYS, FINALIZATION_TABLE_SCHEMAS
from ._bundle_io import (
    FINALIZATION_FILES,
    PARQUET_WRITER_POLICY_ID,
    ROW_GROUP_SIZE,
    TABLE_FILES,
    FinalizationBundleError,
    canonical_json_bytes,
    read_json,
    read_verified_table,
    sha256_bytes,
)

FINALIZATION_MANIFEST_VERSION = 1
FINALIZATION_AUDIT_VERSION = 1
SUCCESS_MARKER_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_VIDIMU_ARCHIVE_ROLES = frozenset({"DATASET_ARCHIVE", "VIDEO_ARCHIVE"})
_MANIFEST_FIELDS = frozenset({
    "manifest_version",
    "finalization_id",
    "recording_id",
    "dataset_id",
    "finalization_schema_version",
    "association_schema_version",
    "validation_gate",
    "identity_inputs",
    "source",
    "decoder",
    "estimator",
    "coordinate_contract",
    "determinism_contract",
    "scope",
    "tables",
})
_SOURCE_FIELDS = frozenset({
    "dataset_id",
    "dataset_version",
    "recording_id",
    "source_kind",
    "source_original_path",
    "source_object_id",
    "materialization_date",
    "license_id",
    "license_record_sha256",
    "inventory_manifest_sha256",
    "source_video_bytes",
    "source_video_sha256",
    "stream_index",
    "source_hash_verified_during_pinned_decode",
})
_TABLE_METADATA_FIELDS = frozenset({
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


def _exact_fields(
    value: object,
    expected: frozenset[str],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise FinalizationBundleError(f"{field} fields are not the frozen contract")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FinalizationBundleError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FinalizationBundleError(f"{field} must be a lowercase SHA-256")
    return value


def _integer(
    value: object,
    field: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalizationBundleError(
            f"{field} must be an integer greater than or equal to {minimum}")
    return value


def _exact_integer_value(value: object, expected: int, field: str) -> int:
    actual = _integer(value, field)
    if actual != expected:
        raise FinalizationBundleError(f"{field} must equal {expected}")
    return actual


def _safe_relative_path(value: object, field: str) -> str:
    path_text = _nonempty(value, field)
    path = PurePosixPath(path_text)
    if path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts
    ) or path.as_posix() != path_text or "\\" in path_text or "\x00" in path_text:
        raise FinalizationBundleError(f"{field} must be a safe relative path")
    return path_text


def _canonical_source_archives(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise FinalizationBundleError("Gate B lacks source archive evidence")
    normalized: list[dict[str, str]] = []
    for item in value:
        asset = _exact_fields(item, frozenset({
            "original_path", "role", "sha256",
        }), "source archive asset")
        role = asset["role"]
        if not isinstance(role, str) or role not in _VIDIMU_ARCHIVE_ROLES:
            raise FinalizationBundleError("source archive asset role is invalid")
        normalized.append({
            "original_path": _safe_relative_path(
                asset["original_path"], "source archive original_path"),
            "role": str(role),
            "sha256": str(_sha256(
                asset["sha256"], "source archive sha256")),
        })
    canonical = sorted(normalized, key=lambda item: (
        item["original_path"], item["role"], item["sha256"],
    ))
    paths = [item["original_path"] for item in normalized]
    roles = [item["role"] for item in normalized]
    hashes = [item["sha256"] for item in normalized]
    if normalized != canonical \
            or set(roles) != _VIDIMU_ARCHIVE_ROLES \
            or len(normalized) != len(_VIDIMU_ARCHIVE_ROLES) \
            or len(paths) != len(set(paths)) \
            or len(roles) != len(set(roles)) \
            or len(hashes) != len(set(hashes)):
        raise FinalizationBundleError(
            "source archive evidence is not canonical and exact")
    return normalized


def _validate_manifest_semantics(
    manifest: dict[str, object],
    tables: dict[str, pa.Table],
) -> None:
    _exact_fields(manifest, _MANIFEST_FIELDS, "manifest")
    dataset_id = _nonempty(manifest["dataset_id"], "dataset_id")
    recording_id = _nonempty(manifest["recording_id"], "recording_id")
    finalization_id = _sha256(manifest["finalization_id"], "finalization_id")
    gate = manifest["validation_gate"]
    if gate not in {
        "GATE_A_SYNTHETIC", "GATE_A_REAL_VIDEO_PILOT", "GATE_B_VIDIMU",
    }:
        raise FinalizationBundleError("validation_gate is unsupported")

    identity = manifest["identity_inputs"]
    if not isinstance(identity, dict):
        raise FinalizationBundleError("identity_inputs must be an object")
    if finalization_identity(identity) != finalization_id:
        raise FinalizationBundleError("finalization identity does not recompute")

    source_fields = set(_SOURCE_FIELDS)
    source_value = manifest["source"]
    if isinstance(source_value, dict) and "paired_imu_assets" in source_value:
        source_fields.add("paired_imu_assets")
    if isinstance(source_value, dict) and "source_archives" in source_value:
        source_fields.add("source_archives")
    source = _exact_fields(
        source_value, frozenset(source_fields), "source provenance")
    if source["dataset_id"] != dataset_id \
            or source["recording_id"] != recording_id:
        raise FinalizationBundleError(
            "source provenance disagrees with manifest identity")
    _nonempty(source["dataset_version"], "source.dataset_version")
    source_kind = source["source_kind"]
    if source_kind not in {"SYNTHETIC_FIXTURE", "VIDIMU_PUBLIC"}:
        raise FinalizationBundleError("source_kind is unsupported")
    if gate == "GATE_A_SYNTHETIC" and source_kind != "SYNTHETIC_FIXTURE":
        raise FinalizationBundleError("synthetic Gate A contains a public source claim")
    if gate in {"GATE_A_REAL_VIDEO_PILOT", "GATE_B_VIDIMU"} \
            and source_kind != "VIDIMU_PUBLIC":
        raise FinalizationBundleError("real-video gate contains a substituted source")
    _safe_relative_path(
        source["source_original_path"], "source.source_original_path")
    _nonempty(source["source_object_id"], "source.source_object_id")
    try:
        materialization_date = date.fromisoformat(str(source["materialization_date"]))
    except ValueError as exc:
        raise FinalizationBundleError(
            "source.materialization_date must use YYYY-MM-DD") from exc
    if materialization_date.isoformat() != source["materialization_date"]:
        raise FinalizationBundleError(
            "source.materialization_date must be canonical")
    _nonempty(source["license_id"], "source.license_id")
    _sha256(source["license_record_sha256"], "source.license_record_sha256")
    inventory_hash = _sha256(
        source["inventory_manifest_sha256"],
        "source.inventory_manifest_sha256",
        nullable=True,
    )
    if source_kind == "VIDIMU_PUBLIC":
        if inventory_hash is None:
            raise FinalizationBundleError(
                "public source lacks frozen inventory evidence")
    elif inventory_hash is not None:
        raise FinalizationBundleError(
            "synthetic source contains unrelated public archive evidence")
    source_video_sha = _sha256(
        source["source_video_sha256"], "source.source_video_sha256")
    if source_video_sha != identity["source_video_sha256"]:
        raise FinalizationBundleError(
            "source video hash disagrees with finalization identity")
    _integer(source["source_video_bytes"], "source.source_video_bytes", minimum=1)
    stream_index = _integer(source["stream_index"], "source.stream_index")
    if source["source_hash_verified_during_pinned_decode"] is not True:
        raise FinalizationBundleError("source pinned-decode verification is not true")
    paired_imu_assets = source.get("paired_imu_assets")
    source_archives = source.get("source_archives")
    if gate == "GATE_B_VIDIMU":
        if not isinstance(paired_imu_assets, list) or not paired_imu_assets:
            raise FinalizationBundleError("Gate B lacks paired IMU evidence")
        normalized_imu_assets: list[dict[str, str]] = []
        for item in paired_imu_assets:
            asset = _exact_fields(item, frozenset({
                "original_path", "role", "sha256",
            }), "paired IMU asset")
            if asset["role"] != "IMU":
                raise FinalizationBundleError("paired IMU asset role is invalid")
            normalized_imu_assets.append({
                "original_path": _safe_relative_path(
                    asset["original_path"], "paired IMU original_path"),
                "role": "IMU",
                "sha256": str(_sha256(
                    asset["sha256"], "paired IMU sha256")),
            })
        canonical_imu_assets = sorted(
            normalized_imu_assets,
            key=lambda item: (
                item["original_path"], item["role"], item["sha256"],
            ),
        )
        paths = [item["original_path"] for item in normalized_imu_assets]
        if normalized_imu_assets != canonical_imu_assets \
                or len(paths) != len(set(paths)) \
                or source["source_original_path"] in paths:
            raise FinalizationBundleError(
                "paired IMU evidence is not canonical and unique")
        normalized_archives = _canonical_source_archives(source_archives)
        archive_paths = {
            item["original_path"] for item in normalized_archives
        }
        if source["source_original_path"] in archive_paths \
                or archive_paths.intersection(paths):
            raise FinalizationBundleError(
                "source archive evidence aliases a recording asset")
    elif paired_imu_assets is not None:
        raise FinalizationBundleError(
            "non-Gate-B manifest contains paired IMU evidence")
    elif source_archives is not None:
        raise FinalizationBundleError(
            "non-Gate-B manifest contains source archive evidence")

    decoder = _exact_fields(manifest["decoder"], frozenset({
        "decoder_version",
        "decoder_config",
        "decoder_config_sha256",
        "decode_ordinal_semantics",
        "timestamp_authority",
        "presentation_origin",
    }), "decoder")
    if decoder["decoder_version"] != identity["decoder_version"]:
        raise FinalizationBundleError("decoder version disagrees with identity")
    config_value = decoder["decoder_config"]
    if not isinstance(config_value, dict) or set(config_value) != set(
            asdict(DecodeConfig())):
        raise FinalizationBundleError("decoder_config fields are invalid")
    try:
        config = DecodeConfig(**config_value)
    except (TypeError, ValueError) as exc:
        raise FinalizationBundleError("decoder_config is invalid") from exc
    decoder_config_sha = _sha256(
        decoder["decoder_config_sha256"], "decoder.decoder_config_sha256")
    if decoder_config_sha != config.sha256 \
            or decoder_config_sha != identity["decoder_config_sha256"]:
        raise FinalizationBundleError("decoder config hash does not recompute")
    if config.stream_index != stream_index:
        raise FinalizationBundleError("decoder and source stream indices disagree")
    if decoder["decode_ordinal_semantics"] != "decoder_emission_ordinal" \
            or decoder["timestamp_authority"] != "raw_source_pts_and_time_base":
        raise FinalizationBundleError("decoder timing semantics are unsupported")

    estimator = _exact_fields(manifest["estimator"], frozenset({
        "model_id",
        "model_weights_sha256",
        "preprocessing_config_sha256",
        "inference_environment_id",
        "selection_policy_id",
        "inference_call_contract",
    }), "estimator")
    for field in ("model_id", "inference_environment_id"):
        if _nonempty(estimator[field], f"estimator.{field}") != identity[field]:
            raise FinalizationBundleError(f"estimator {field} disagrees with identity")
    for field in ("model_weights_sha256", "preprocessing_config_sha256"):
        if _sha256(estimator[field], f"estimator.{field}") != identity[field]:
            raise FinalizationBundleError(f"estimator {field} disagrees with identity")
    if estimator["selection_policy_id"] != SELECTION_POLICY_ID \
            or estimator["inference_call_contract"] \
            != "once_per_preprocessed_decoded_frame":
        raise FinalizationBundleError("estimator policy contract is unsupported")

    if manifest["coordinate_contract"] != {
        "persisted_landmark_space": "DISPLAY_PIXEL",
        "pixel_convention": PIXEL_CONVENTION,
        "cv_input_hash_version": CV_INPUT_HASH_VERSION,
        "container_rotation_applied_once": True,
        "preview_mirroring_applied": False,
    }:
        raise FinalizationBundleError("coordinate contract is unsupported")
    if manifest["determinism_contract"] != {
        "canonical_runtime_ms": None,
        "parquet_runtime_profiling_excluded": True,
        "cpu_software_decode": True,
    }:
        raise FinalizationBundleError("determinism contract is unsupported")
    if manifest["scope"] != {
        "video_imu_synchronization": False,
        "canonical_cross_modal_clock": False,
        "frame_imu_range_index": False,
        "window_generation": False,
        "performance_benchmark": False,
    }:
        raise FinalizationBundleError("scope contract was expanded")

    frame_rows = _table_rows(tables, "video_frames")
    if {row["dataset_id"] for row in frame_rows} != {dataset_id} \
            or {int(row["stream_index"]) for row in frame_rows} != {stream_index}:
        raise FinalizationBundleError(
            "frame table disagrees with dataset or selected stream")
    origins = [
        row for row in frame_rows if row["presentation_ordinal"] == 0
    ]
    expected_origin: dict[str, int] | None = None
    if origins:
        if len(origins) != 1:
            raise FinalizationBundleError("presentation origin is ambiguous")
        expected_origin = {
            "pts": int(origins[0]["pts"]),
            "time_base_num": int(origins[0]["time_base_num"]),
            "time_base_den": int(origins[0]["time_base_den"]),
        }
    if decoder["presentation_origin"] != expected_origin:
        raise FinalizationBundleError(
            "decoder presentation origin disagrees with frame evidence")


def _table_rows(tables: dict[str, pa.Table], name: str) -> list[dict[str, object]]:
    return tables[name].to_pylist()


def build_finalization_audit(
    *,
    manifest: dict[str, object],
    manifest_sha256: str,
    tables: dict[str, pa.Table],
) -> dict[str, object]:
    """Build the deterministic audit payload after all strict checks pass."""

    _validate_manifest_semantics(manifest, tables)
    identity = manifest["identity_inputs"]
    assert isinstance(identity, dict)
    decoder = manifest["decoder"]
    assert isinstance(decoder, dict)
    decoder_config = decoder["decoder_config"]
    assert isinstance(decoder_config, dict)
    discontinuity_threshold_ns = DecodeConfig(
        **decoder_config).discontinuity_threshold_ns
    association = validate_pose_frame_association(
        tables["video_frames"],
        tables["cv_frame_results"],
        tables["cv_detections"],
        finalization_id=str(manifest["finalization_id"]),
        expected_recording_id=str(manifest["recording_id"]),
        expected_source_video_sha256=str(identity["source_video_sha256"]),
        expected_decoder_version=str(identity["decoder_version"]),
        expected_model_id=str(identity["model_id"]),
        expected_model_weights_sha256=str(identity["model_weights_sha256"]),
        expected_preprocessing_config_sha256=str(
            identity["preprocessing_config_sha256"]),
        expected_inference_environment_id=str(
            identity["inference_environment_id"]),
        expected_discontinuity_threshold_ns=discontinuity_threshold_ns,
    )
    frames = _table_rows(tables, "video_frames")
    results = _table_rows(tables, "cv_frame_results")
    detections = _table_rows(tables, "cv_detections")
    if any(row["runtime_ms"] is not None for row in results):
        raise FinalizationBundleError(
            "canonical deterministic results must leave runtime_ms null")

    status_counts = Counter(str(row["inference_status"]) for row in results)
    frame_ids = {str(row["frame_id"]) for row in frames}
    result_ids = {str(row["frame_id"]) for row in results}
    detection_frame_ids = {str(row["frame_id"]) for row in detections}
    bits = [int(row["quality_bits"]) for row in frames]
    table_manifest = manifest["tables"]
    assert isinstance(table_manifest, dict)
    artifact_hashes = {
        name: str(metadata["sha256"])
        for name, metadata in sorted(table_manifest.items())
        if isinstance(metadata, dict)
    }

    return {
        "audit_version": FINALIZATION_AUDIT_VERSION,
        "association_schema_version": ASSOCIATION_SCHEMA_VERSION,
        "finalization_schema_version": FINALIZATION_SCHEMA_VERSION,
        "finalization_id": manifest["finalization_id"],
        "recording_id": manifest["recording_id"],
        "validation_gate": manifest["validation_gate"],
        "inventory_record_count": 1,
        "source_assets_present": "VERIFIED_DURING_FINALIZATION",
        "source_assets_embedded_in_bundle": 0,
        "source_hashes_verified": "VERIFIED_DURING_PINNED_DECODE",
        "videos_opened": 1,
        "videos_failed": 0,
        "decoded_frame_count": len(frames),
        "frames_with_valid_pts": sum(
            row["pts_status"] == "VALID" for row in frames),
        "frames_with_duplicate_pts": sum(
            bool(value & int(QualityBits.DUPLICATE_TIMESTAMP)) for value in bits),
        "frames_with_missing_pts": sum(
            bool(value & int(QualityBits.MISSING_TIMESTAMP)) for value in bits),
        "frames_with_nonmonotonic_pts": sum(
            bool(value & int(QualityBits.NON_MONOTONIC_TIMESTAMP))
            for value in bits
        ),
        "timestamp_discontinuities": sum(
            bool(value & int(QualityBits.STREAM_GAP)) for value in bits),
        "decoded_corrupt_frames": sum(
            bool(value & int(QualityBits.DECODE_FAILURE)) for value in bits),
        "cv_frame_result_count": len(results),
        "frames_with_detection": association.frames_with_detection,
        "frames_without_detection": association.frames_without_detection,
        "inference_failures": association.inference_failure_count,
        "inference_status_counts": dict(sorted(status_counts.items())),
        "detection_row_count": len(detections),
        "orphan_frame_results": len(result_ids.difference(frame_ids)),
        "orphan_detections": len(detection_frame_ids.difference(frame_ids)),
        "duplicate_frame_ids": len(frames) - len(frame_ids),
        "duplicate_detection_ids": len(detections) - len({
            str(row["detection_id"]) for row in detections
        }),
        "missing_frame_results": len(frame_ids.difference(result_ids)),
        "coordinate_transform_failures": 0,
        "source_hash_mismatches": 0,
        "model_hash_mismatches": 0,
        "artifact_hashes": artifact_hashes,
        "manifest_sha256": manifest_sha256,
        "deterministic_replay_status": "STORED_ARTIFACT_BYTES_VERIFIED",
        "overall_verdict": "PASS",
    }


def _manifest_tables(
    bundle_dir: Path,
    manifest: dict[str, object],
) -> dict[str, pa.Table]:
    table_metadata = manifest.get("tables")
    if not isinstance(table_metadata, dict) \
            or set(table_metadata) != set(TABLE_FILES):
        raise FinalizationBundleError("manifest table inventory is invalid")
    tables: dict[str, pa.Table] = {}
    for name, filename in TABLE_FILES.items():
        metadata = table_metadata.get(name)
        if not isinstance(metadata, dict) or set(metadata) != _TABLE_METADATA_FIELDS \
                or metadata.get("path") != filename:
            raise FinalizationBundleError(
                f"manifest metadata is invalid for {name}")
        _integer(metadata.get("bytes"), f"tables.{name}.bytes", minimum=1)
        _integer(metadata.get("rows"), f"tables.{name}.rows")
        for field in ("sha256", "semantic_sha256", "schema_sha256"):
            _sha256(metadata.get(field), f"tables.{name}.{field}")
        if metadata.get("writer_policy_id") != PARQUET_WRITER_POLICY_ID:
            raise FinalizationBundleError(
                f"Parquet writer policy mismatch: {name}")
        path = bundle_dir / filename
        if path.is_symlink() or not path.is_file():
            raise FinalizationBundleError(f"missing regular artifact: {filename}")
        try:
            table, row_groups = read_verified_table(
                path,
                expected_bytes=metadata.get("bytes"),
                expected_sha256=metadata.get("sha256"),
            )
        except (OSError, FinalizationBundleError) as exc:
            raise FinalizationBundleError(
                f"artifact verification failed: {name}") from exc
        expected_schema = FINALIZATION_TABLE_SCHEMAS[name]()
        if schema_fingerprint(table.schema) != schema_fingerprint(expected_schema) \
                or schema_fingerprint(table.schema) != metadata.get("schema_sha256"):
            raise FinalizationBundleError(f"schema hash mismatch: {name}")
        if table.num_rows != metadata.get("rows"):
            raise FinalizationBundleError(f"row count mismatch: {name}")
        sort_keys = metadata.get("sort_keys")
        expected_sort_keys = list(FINALIZATION_SORT_KEYS[name])
        if sort_keys != expected_sort_keys:
            raise FinalizationBundleError(f"sort policy mismatch: {name}")
        if semantic_table_hash(
            table, sort_keys=sort_keys,
        ) != metadata.get("semantic_sha256"):
            raise FinalizationBundleError(f"semantic hash mismatch: {name}")
        sorted_table = table.sort_by([
            (key, "ascending") for key in expected_sort_keys
        ])
        if not table.equals(sorted_table):
            raise FinalizationBundleError(f"physical row order mismatch: {name}")
        if metadata.get("row_group_size") != ROW_GROUP_SIZE:
            raise FinalizationBundleError(f"row-group policy mismatch: {name}")
        row_count = table.num_rows
        expected_row_groups = (
            [ROW_GROUP_SIZE] * (row_count // ROW_GROUP_SIZE)
            + ([row_count % ROW_GROUP_SIZE] if row_count % ROW_GROUP_SIZE else [])
        )
        if row_count == 0:
            expected_row_groups = [0]
        if row_groups != expected_row_groups:
            raise FinalizationBundleError(f"row-group layout mismatch: {name}")
        tables[name] = table
    return tables


def audit_finalized_recording(
    bundle_path: str | Path,
    *,
    _expected_finalization_id: str | None = None,
    _expected_recording_id: str | None = None,
) -> dict[str, object]:
    """Verify exact bytes, schemas, identities, FKs, transforms, and counts."""

    bundle_dir = Path(bundle_path)
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise FinalizationBundleError(
            "finalized recording path must be a real directory")
    finalization_id = _expected_finalization_id or bundle_dir.name
    present = {item.name for item in bundle_dir.iterdir()}
    if present != FINALIZATION_FILES:
        raise FinalizationBundleError(
            f"finalized artifact inventory mismatch: {sorted(present)}")
    if any(item.is_symlink() for item in bundle_dir.iterdir()):
        raise FinalizationBundleError("finalized artifacts may not be symlinks")

    manifest, manifest_bytes = read_json(
        bundle_dir / "finalization_manifest.json")
    manifest_sha = sha256_bytes(manifest_bytes)
    try:
        _exact_integer_value(
            manifest.get("manifest_version"),
            FINALIZATION_MANIFEST_VERSION,
            "manifest_version",
        )
    except FinalizationBundleError as exc:
        raise FinalizationBundleError(
            "unsupported finalization manifest version") from exc
    if manifest.get("finalization_schema_version") \
            != FINALIZATION_SCHEMA_VERSION \
            or manifest.get("association_schema_version") \
            != ASSOCIATION_SCHEMA_VERSION:
        raise FinalizationBundleError("unsupported finalization manifest version")
    if manifest.get("finalization_id") != finalization_id:
        raise FinalizationBundleError(
            "directory and manifest finalization identities disagree")
    identity_inputs = manifest.get("identity_inputs")
    if not isinstance(identity_inputs, dict) \
            or finalization_identity(identity_inputs) != finalization_id:
        raise FinalizationBundleError("finalization identity does not recompute")
    expected_recording_id = _expected_recording_id or bundle_dir.parent.name
    if manifest.get("recording_id") != expected_recording_id:
        raise FinalizationBundleError(
            "directory and manifest recording identities disagree")

    tables = _manifest_tables(bundle_dir, manifest)
    expected_audit = build_finalization_audit(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        tables=tables,
    )
    _stored_audit, audit_bytes = read_json(
        bundle_dir / "finalization_audit.json")
    if audit_bytes != canonical_json_bytes(expected_audit):
        raise FinalizationBundleError(
            "stored finalization audit does not reproduce")
    _success, success_bytes = read_json(bundle_dir / "_SUCCESS")
    expected_success = {
        "audit_sha256": sha256_bytes(audit_bytes),
        "finalization_id": finalization_id,
        "manifest_sha256": manifest_sha,
        "success_marker_version": SUCCESS_MARKER_VERSION,
    }
    if success_bytes != canonical_json_bytes(expected_success):
        raise FinalizationBundleError("_SUCCESS binding is invalid")
    return expected_audit


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit one immutable VIDIMU PTS/CV finalization bundle.")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        audit = audit_finalized_recording(args.bundle)
    except Exception as exc:  # noqa: BLE001 - CLI must emit a fail report
        report = {"overall_verdict": "FAIL", "error": str(exc)}
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 1
    encoded = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "FINALIZATION_AUDIT_VERSION",
    "FINALIZATION_MANIFEST_VERSION",
    "SUCCESS_MARKER_VERSION",
    "audit_finalized_recording",
    "build_finalization_audit",
]
