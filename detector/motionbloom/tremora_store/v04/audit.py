"""Independent strict audit and clean-root replay comparison for v0.4 Gate B."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..finalize._bundle_io import (
    FinalizationBundleError,
    _exclusive_descriptor,
    _rename_noreplace,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)
from ..schema import QualityBits
from ..source_snapshot import (
    SOURCE_INVENTORY_FILE,
    VIDIMU_V2_CITATION_ID,
    VIDIMU_V2_DATASET_ID,
    VIDIMU_V2_DATASET_VERSION,
    VIDIMU_V2_EXPECTED_ASSET_REFERENCE_COUNT,
    VIDIMU_V2_EXPECTED_RECORDING_COUNT,
    VIDIMU_V2_LICENSE_ID,
    SourceSnapshotError,
    verify_vidimu_source_snapshot,
)
from .bundle import (
    TABLE_FILES,
    V04_BUNDLE_SCHEMA_VERSION,
    AuditedV04Bundle,
    V04BundleError,
    audit_v04_bundle,
)
from .finalization import (
    EXECUTION_RECEIPT_FILE,
    RUN_AUDIT_FILE,
    RUN_MANIFEST_FILE,
    RUN_SUCCESS_FILE,
    V04_EXECUTION_RECEIPT_VERSION,
    V04_GATE_B_RUN_ARTIFACT_KIND,
    V04_GATE_B_RUN_AUDIT_VERSION,
    V04_GATE_B_RUN_IDENTITY_DOMAIN,
    V04_GATE_B_RUN_MANIFEST_VERSION,
    V04_GATE_B_RUN_SUCCESS_VERSION,
    GateBModelInputs,
)
from .model_manifest import ProductionModelError, load_and_verify_production_model
from .schemas import V04_ASSOCIATION_CONTRACT_VERSION

V04_GATE_B_RELEASE_AUDIT_KIND = "TREMORA_VIDIMU_V04_GATE_B_RELEASE_AUDIT"
V04_GATE_B_RELEASE_AUDIT_VERSION = 1

BYTE_IDENTICAL_SOURCE_TO_CV_PASS = "BYTE_IDENTICAL_SOURCE_TO_CV_PASS"
CANONICAL_CONTENT_IDENTICAL_PARQUET_BYTES_DIFFER_PASS = (
    "CANONICAL_CONTENT_IDENTICAL_PARQUET_BYTES_DIFFER_PASS"
)
FAIL = "FAIL"
RELEASE_STATUSES = frozenset({
    BYTE_IDENTICAL_SOURCE_TO_CV_PASS,
    CANONICAL_CONTENT_IDENTICAL_PARQUET_BYTES_DIFFER_PASS,
    FAIL,
})

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_UTC_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z"
)
_RUN_MANIFEST_FIELDS = frozenset({
    "artifact_kind",
    "run_manifest_version",
    "run_id",
    "source_snapshot_sha256",
    "source_inventory_sha256",
    "model_manifest_sha256",
    "execution_id",
    "execution_receipt_sha256",
    "recording_count",
    "recording_ids",
    "bundles",
    "scope",
})
_RUN_AUDIT_FIELDS = frozenset({
    "artifact_kind",
    "run_audit_version",
    "run_id",
    "recordings_expected",
    "recordings_finalized",
    "recordings_failed",
    "source_snapshot_verified_before_and_after",
    "model_manifest_verified",
    "execution_receipt_verified",
    "bundle_audits_passed",
    "overall_verdict",
})
_RUN_MARKER_FIELDS = frozenset({
    "artifact_kind",
    "run_success_version",
    "run_id",
    "run_manifest_sha256",
    "run_audit_sha256",
})


class GateBReleaseAuditError(RuntimeError):
    """Raised when a run or replay cannot be independently proven valid."""


@dataclass(frozen=True, slots=True)
class AuditedGateBRun:
    path: Path
    manifest: Mapping[str, object]
    audit: Mapping[str, object]
    marker: Mapping[str, object]
    execution_receipt: Mapping[str, object]
    bundles: Mapping[str, AuditedV04Bundle]
    file_identities: Mapping[str, tuple[int, int]]
    file_hashes: Mapping[str, str]
    processing_identity: Mapping[str, object]
    reconciliation: Mapping[str, int]
    canonical_content_sha256: str


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GateBReleaseAuditError(f"{field} must be a lowercase SHA-256")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateBReleaseAuditError(f"{field} must be an integer >= {minimum}")
    return value


def _run_id(snapshot_sha: str, model_sha: str) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "domain": V04_GATE_B_RUN_IDENTITY_DOMAIN,
        "source_snapshot_sha256": snapshot_sha,
        "model_manifest_sha256": model_sha,
        "association_contract_version": V04_ASSOCIATION_CONTRACT_VERSION,
        "bundle_schema_version": V04_BUNDLE_SCHEMA_VERSION,
    })).hexdigest()


def _safe_relative(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise GateBReleaseAuditError(f"{field} is not a safe relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed.as_posix() != value or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise GateBReleaseAuditError(f"{field} is not a safe relative path")
    return value


def _walk_regular_tree(root: Path) -> tuple[dict[str, tuple[int, int]], dict[str, str]]:
    identities: dict[str, tuple[int, int]] = {}
    hashes: dict[str, str] = {}
    for directory, child_directories, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        for name in child_directories:
            state = (directory_path / name).lstat()
            if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
                raise GateBReleaseAuditError("run tree contains a non-real directory")
        for name in file_names:
            path = directory_path / name
            state = path.lstat()
            if not stat.S_ISREG(state.st_mode) or stat.S_ISLNK(state.st_mode) \
                    or state.st_nlink != 1:
                raise GateBReleaseAuditError(
                    "run tree contains a non-regular or shared file"
                )
            relative = path.relative_to(root).as_posix()
            identities[relative] = (state.st_dev, state.st_ino)
            try:
                hashes[relative] = sha256_file(path)
            except FinalizationBundleError as exc:
                raise GateBReleaseAuditError("run artifact hashing failed") from exc
    return identities, hashes


def _load_source_authority(
    source_snapshot_path: str | Path,
    *,
    expected_snapshot_sha256: str,
    expected_inventory_sha256: str,
) -> tuple[
    object,
    dict[str, list[dict[str, object]]],
    dict[str, object],
]:
    try:
        source = verify_vidimu_source_snapshot(source_snapshot_path)
    except SourceSnapshotError as exc:
        raise GateBReleaseAuditError("source snapshot verification failed") from exc
    if source.snapshot_manifest_sha256 != expected_snapshot_sha256 \
            or source.source_inventory_sha256 != expected_inventory_sha256 \
            or source.extracted_asset_count \
            != VIDIMU_V2_EXPECTED_ASSET_REFERENCE_COUNT \
            or source.unavailable_asset_count != 0:
        raise GateBReleaseAuditError("source snapshot contradicts release anchors")
    try:
        inventory, inventory_bytes = read_json(
            source.path / SOURCE_INVENTORY_FILE,
            max_bytes=16 * 1024 * 1024,
        )
    except FinalizationBundleError as exc:
        raise GateBReleaseAuditError(
            "frozen source inventory cannot be read canonically"
        ) from exc
    if sha256_bytes(inventory_bytes) != expected_inventory_sha256:
        raise GateBReleaseAuditError("frozen source inventory hash changed")
    source_dataset = {
        "dataset_id": inventory.get("dataset_id"),
        "dataset_version": inventory.get("dataset_version"),
        "license_id": inventory.get("license_id"),
        "citation_id": inventory.get("citation_id"),
    }
    if source_dataset != {
        "dataset_id": VIDIMU_V2_DATASET_ID,
        "dataset_version": VIDIMU_V2_DATASET_VERSION,
        "license_id": VIDIMU_V2_LICENSE_ID,
        "citation_id": VIDIMU_V2_CITATION_ID,
    }:
        raise GateBReleaseAuditError(
            "frozen source inventory dataset provenance is invalid"
        )
    raw_assets = inventory.get("asset_references")
    if not isinstance(raw_assets, list) \
            or len(raw_assets) != VIDIMU_V2_EXPECTED_ASSET_REFERENCE_COUNT:
        raise GateBReleaseAuditError("frozen source inventory cardinality is invalid")
    asset_fields = {
        "source_object_id",
        "archive_member_path",
        "normalized_member_path",
        "recording_id",
        "asset_role",
        "modality",
        "expected_size_bytes",
        "expected_sha256",
        "availability",
        "unavailable_reason",
    }
    expected_modalities = {
        "VIDEO": "VISUAL",
        "IMU": "INERTIAL_QUATERNION",
        "ANNOTATION": "BODYTRACK_POSE",
    }
    grouped: dict[str, list[dict[str, object]]] = {}
    for raw in raw_assets:
        if not isinstance(raw, dict) or set(raw) != asset_fields \
                or raw["availability"] != "REQUIRED" \
                or raw["unavailable_reason"] is not None \
                or raw["asset_role"] not in expected_modalities \
                or raw["modality"] != expected_modalities[raw["asset_role"]]:
            raise GateBReleaseAuditError(
                "frozen source inventory asset topology is invalid"
            )
        for field in (
            "source_object_id",
            "archive_member_path",
            "normalized_member_path",
            "recording_id",
        ):
            if not isinstance(raw[field], str) or not raw[field]:
                raise GateBReleaseAuditError("source inventory asset text is invalid")
        _safe_relative(raw["archive_member_path"], "archive member path")
        _safe_relative(raw["normalized_member_path"], "normalized member path")
        size = _integer(raw["expected_size_bytes"], "source asset bytes")
        asset_sha = _sha256(raw["expected_sha256"], "source asset sha256")
        recording_id = str(raw["recording_id"])
        grouped.setdefault(recording_id, []).append({
            "recording_id": recording_id,
            "source_object_id": raw["source_object_id"],
            "archive_member_path": raw["archive_member_path"],
            "normalized_member_path": raw["normalized_member_path"],
            "asset_role": raw["asset_role"],
            "modality": raw["modality"],
            "bytes": size,
            "sha256": asset_sha,
        })
    if len(grouped) != VIDIMU_V2_EXPECTED_RECORDING_COUNT:
        raise GateBReleaseAuditError(
            "frozen source inventory recording set is not exact"
        )
    for recording_id, assets in grouped.items():
        assets.sort(key=lambda item: (
            str(item["asset_role"]),
            str(item["source_object_id"]),
            str(item["archive_member_path"]),
        ))
        if [item["asset_role"] for item in assets] != [
            "ANNOTATION", "IMU", "VIDEO"
        ]:
            raise GateBReleaseAuditError(
                f"source recording {recording_id} lacks exact asset roles"
            )
    return source, dict(sorted(grouped.items())), source_dataset


def _processing_identity(bundle: AuditedV04Bundle) -> dict[str, object]:
    manifest = bundle.manifest
    decoder = manifest["decoder"]
    model = manifest["model"]
    if not isinstance(decoder, Mapping) or not isinstance(model, Mapping):
        raise GateBReleaseAuditError("bundle processing identity is malformed")
    return {
        "decoder_version": decoder["decoder_version"],
        "decoder_config_sha256": decoder["decoder_config_sha256"],
        "model_manifest_sha256": model["model_manifest_sha256"],
        "model_id": model["model_id"],
        "model_weights_sha256": model["model_weights_sha256"],
        "preprocessing_config_sha256": model["preprocessing_config_sha256"],
        "runtime_lock_sha256": model["runtime_lock_sha256"],
        "inference_environment_id": model["inference_environment_id"],
        "association_contract_version": manifest["association_contract_version"],
        "bundle_schema_version": manifest["bundle_schema_version"],
    }


def _load_model_authority(
    model_inputs: GateBModelInputs,
    *,
    expected_model_manifest_sha256: str,
) -> dict[str, object]:
    """Independently verify every frozen production-model evidence edge."""

    if type(model_inputs) is not GateBModelInputs:
        raise GateBReleaseAuditError(
            "release audit model evidence must use exact GateBModelInputs"
        )
    if model_inputs.expected_manifest_sha256 != expected_model_manifest_sha256:
        raise GateBReleaseAuditError(
            "release audit model evidence contradicts its external anchor"
        )
    try:
        verified = load_and_verify_production_model(
            model_inputs.manifest_path,
            weights_path=model_inputs.weights_path,
            preprocessing_config_path=model_inputs.preprocessing_config_path,
            runtime_lock_path=model_inputs.runtime_lock_path,
            expected_manifest_sha256=expected_model_manifest_sha256,
            vendored_model_inventory=model_inputs.vendored_model_inventory_path,
        )
    except (OSError, ProductionModelError, RuntimeError, ValueError) as exc:
        raise GateBReleaseAuditError(
            "production model evidence failed independent verification"
        ) from exc
    manifest = verified.manifest
    return {
        "model_manifest_sha256": verified.manifest_sha256,
        "model_id": manifest["model_id"],
        "model_weights_sha256": manifest["model_weights_sha256"],
        "preprocessing_config_sha256": manifest["preprocessing_config_sha256"],
        "runtime_lock_sha256": manifest["runtime_lock_sha256"],
        "inference_environment_id": (
            f"native-runtime-sha256:{manifest['runtime_lock_sha256']}"
        ),
        "association_contract_version": manifest[
            "association_contract_version"
        ],
    }


def _run_reconciliation(
    bundles: Mapping[str, AuditedV04Bundle],
) -> dict[str, int]:
    metrics = {
        "recordings": len(bundles),
        "videos_expected": len(bundles),
        "videos_opened": len(bundles),
        "videos_failed": 0,
        "decoded_frames": 0,
        "frames_with_source_pts": 0,
        "valid_pts_frames": 0,
        "missing_pts_frames": 0,
        "duplicate_pts_frames": 0,
        "nonmonotonic_pts_frames": 0,
        "discontinuity_pts_frames": 0,
        "discontinuity_count": 0,
        "decoded_corrupt_frames": 0,
        "cv_frame_results": 0,
        "zero_detection_frames": 0,
        "one_detection_frames": 0,
        "multiple_detection_frames": 0,
        "total_detections": 0,
        "frames_with_zero_detections": 0,
        "frames_with_one_detection": 0,
        "frames_with_multiple_detections": 0,
        "total_detection_rows": 0,
        "decode_failures": 0,
        "preprocessing_failures": 0,
        "inference_failures": 0,
        "rejected_input_failures": 0,
        "cv_failures": 0,
        "foreign_key_violations": 0,
        "missing_frame_results": 0,
        "duplicate_frame_results": 0,
        "orphan_detections": 0,
        "duplicate_frame_ids": 0,
        "duplicate_detection_ids": 0,
        "selection_rows": 0,
        "selected_frames": 0,
        "abstained_frames": 0,
        "invalid_selection_references": 0,
        "invalid_selected_detection_ids": 0,
        "source_asset_claims": 0,
        "artifact_hashes_verified": 0,
    }
    for bundle in bundles.values():
        frames = bundle.tables["video_frames"].to_pylist()
        results = bundle.tables["cv_frame_results"].to_pylist()
        detections = bundle.tables["cv_detections"].to_pylist()
        selections = bundle.tables["primary_hand_selection"].to_pylist()
        frame_ids = [row["frame_id"] for row in frames]
        frame_id_set = set(frame_ids)
        result_ids = [row["frame_id"] for row in results]
        result_id_set = set(result_ids)
        detection_ids = [row["detection_id"] for row in detections]
        detection_id_set = set(detection_ids)
        orphan_detections = sum(
            row["frame_id"] not in frame_id_set for row in detections
        )
        invalid_selection_frames = sum(
            row["frame_id"] not in frame_id_set for row in selections
        )
        invalid_selected_ids = sum(
            row["selected_detection_id"] is not None
            and row["selected_detection_id"] not in detection_id_set
            for row in selections
        )
        metrics["decoded_frames"] += len(frames)
        metrics["frames_with_source_pts"] += sum(
            row["pts"] is not None for row in frames
        )
        metrics["valid_pts_frames"] += sum(
            row["pts_status"] == "VALID" for row in frames
        )
        metrics["missing_pts_frames"] += sum(
            bool(
                int(row["quality_bits"])
                & int(QualityBits.MISSING_TIMESTAMP)
            ) for row in frames
        )
        metrics["duplicate_pts_frames"] += sum(
            bool(
                int(row["quality_bits"])
                & int(QualityBits.DUPLICATE_TIMESTAMP)
            ) for row in frames
        )
        metrics["nonmonotonic_pts_frames"] += sum(
            bool(
                int(row["quality_bits"])
                & int(QualityBits.NON_MONOTONIC_TIMESTAMP)
            ) for row in frames
        )
        metrics["discontinuity_pts_frames"] += sum(
            bool(
                int(row["quality_bits"])
                & int(QualityBits.STREAM_GAP)
            ) for row in frames
        )
        metrics["discontinuity_count"] += sum(
            bool(
                int(row["quality_bits"])
                & int(QualityBits.STREAM_GAP)
            ) for row in frames
        )
        metrics["decoded_corrupt_frames"] += sum(
            bool(
                int(row["quality_bits"])
                & int(QualityBits.DECODE_FAILURE)
            ) for row in frames
        )
        metrics["cv_frame_results"] += len(results)
        metrics["zero_detection_frames"] += sum(
            row["detection_count"] == 0 for row in results
        )
        metrics["frames_with_zero_detections"] += sum(
            row["detection_count"] == 0 for row in results
        )
        metrics["one_detection_frames"] += sum(
            row["detection_count"] == 1 for row in results
        )
        metrics["frames_with_one_detection"] += sum(
            row["detection_count"] == 1 for row in results
        )
        metrics["multiple_detection_frames"] += sum(
            row["detection_count"] > 1 for row in results
        )
        metrics["frames_with_multiple_detections"] += sum(
            row["detection_count"] > 1 for row in results
        )
        metrics["total_detections"] += len(detections)
        metrics["total_detection_rows"] += len(detections)
        metrics["decode_failures"] += sum(
            row["inference_status"] == "DECODE_FAILURE" for row in results
        )
        metrics["preprocessing_failures"] += sum(
            row["inference_status"] == "PREPROCESS_FAILURE" for row in results
        )
        metrics["inference_failures"] += sum(
            row["inference_status"] == "INFERENCE_FAILURE" for row in results
        )
        metrics["rejected_input_failures"] += sum(
            row["inference_status"] == "REJECTED_INPUT" for row in results
        )
        metrics["cv_failures"] += sum(
            row["inference_status"] not in {"SUCCESS", "NO_DETECTION"}
            for row in results
        )
        missing_results = len(frame_id_set - result_id_set)
        duplicate_results = len(result_ids) - len(result_id_set)
        duplicate_frames = len(frame_ids) - len(frame_id_set)
        duplicate_detections = len(detection_ids) - len(detection_id_set)
        metrics["missing_frame_results"] += missing_results
        metrics["duplicate_frame_results"] += duplicate_results
        metrics["orphan_detections"] += orphan_detections
        metrics["duplicate_frame_ids"] += duplicate_frames
        metrics["duplicate_detection_ids"] += duplicate_detections
        metrics["selection_rows"] += len(selections)
        metrics["selected_frames"] += sum(
            row["selection_status"] == "SELECTED" for row in selections
        )
        metrics["abstained_frames"] += sum(
            row["selection_status"] == "ABSTAINED" for row in selections
        )
        metrics["invalid_selection_references"] += (
            invalid_selection_frames + invalid_selected_ids
        )
        metrics["invalid_selected_detection_ids"] += invalid_selected_ids
        metrics["foreign_key_violations"] += (
            orphan_detections
            + invalid_selection_frames
            + invalid_selected_ids
        )
        metrics["source_asset_claims"] += len(bundle.manifest["source_assets"])
        metrics["artifact_hashes_verified"] += len(TABLE_FILES) + 3
    return metrics


def _canonical_run_content(
    run_id: str,
    processing_identity: Mapping[str, object],
    bundles: Mapping[str, AuditedV04Bundle],
) -> str:
    payload = {
        "run_id": run_id,
        "processing_identity": dict(processing_identity),
        "recordings": [{
            "recording_id": recording_id,
            "finalization_id": bundle.manifest["finalization_id"],
            "table_semantic_sha256": {
                name: bundle.manifest["tables"][name]["semantic_sha256"]
                for name in TABLE_FILES
            },
        } for recording_id, bundle in sorted(bundles.items())],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def audit_gate_b_run(
    run_root: str | Path,
    *,
    source_snapshot_path: str | Path | None = None,
    expected_source_snapshot_sha256: str,
    expected_source_inventory_sha256: str,
    expected_model_manifest_sha256: str,
    _source_assets_by_recording: Mapping[
        str, list[dict[str, object]]
    ] | None = None,
    _source_dataset: Mapping[str, object] | None = None,
) -> AuditedGateBRun:
    """Strictly audit one complete run against independent hash anchors."""

    expected_snapshot = _sha256(
        expected_source_snapshot_sha256, "expected source snapshot"
    )
    expected_inventory = _sha256(
        expected_source_inventory_sha256, "expected source inventory"
    )
    expected_model = _sha256(expected_model_manifest_sha256, "expected model manifest")
    if (_source_assets_by_recording is None) != (_source_dataset is None):
        raise GateBReleaseAuditError(
            "source inventory assets and dataset provenance must be supplied together"
        )
    if _source_assets_by_recording is None:
        if source_snapshot_path is None:
            raise GateBReleaseAuditError(
                "strict run audit requires the verified source snapshot"
            )
        _source, loaded_assets, loaded_dataset = _load_source_authority(
            source_snapshot_path,
            expected_snapshot_sha256=expected_snapshot,
            expected_inventory_sha256=expected_inventory,
        )
        source_assets_by_recording = loaded_assets
        source_dataset = loaded_dataset
    else:
        source_assets_by_recording = {
            key: [dict(item) for item in value]
            for key, value in _source_assets_by_recording.items()
        }
        if _source_dataset is None:
            raise GateBReleaseAuditError(
                "source dataset provenance is missing"
            )
        source_dataset = dict(_source_dataset)
    root = Path(run_root)
    if root.is_symlink() or not root.is_dir():
        raise GateBReleaseAuditError("run root must be a real directory")
    root = root.resolve(strict=True)
    top_names = {entry.name for entry in root.iterdir()}
    if top_names != {
        "finalized",
        EXECUTION_RECEIPT_FILE,
        RUN_MANIFEST_FILE,
        RUN_AUDIT_FILE,
        RUN_SUCCESS_FILE,
    }:
        raise GateBReleaseAuditError("run root artifact inventory is not exact")
    finalized_dir = root / "finalized"
    if finalized_dir.is_symlink() or not finalized_dir.is_dir():
        raise GateBReleaseAuditError("run finalized hierarchy is invalid")
    try:
        manifest, manifest_bytes = read_json(root / RUN_MANIFEST_FILE)
        audit, audit_bytes = read_json(root / RUN_AUDIT_FILE)
        marker, _marker_bytes = read_json(root / RUN_SUCCESS_FILE)
        receipt, receipt_bytes = read_json(root / EXECUTION_RECEIPT_FILE)
    except FinalizationBundleError as exc:
        raise GateBReleaseAuditError("run JSON cannot be strictly read") from exc
    if set(manifest) != _RUN_MANIFEST_FIELDS \
            or manifest["artifact_kind"] != V04_GATE_B_RUN_ARTIFACT_KIND \
            or type(manifest["run_manifest_version"]) is not int \
            or manifest["run_manifest_version"] != V04_GATE_B_RUN_MANIFEST_VERSION:
        raise GateBReleaseAuditError("run manifest contract is invalid")
    if manifest["source_snapshot_sha256"] != expected_snapshot \
            or manifest["source_inventory_sha256"] != expected_inventory \
            or manifest["model_manifest_sha256"] != expected_model:
        raise GateBReleaseAuditError("run contradicts independent trust anchors")
    if set(receipt) != {
        "artifact_kind",
        "execution_receipt_version",
        "execution_id",
        "initial_process_id",
        "started_at_utc",
    } or receipt["artifact_kind"] != V04_GATE_B_RUN_ARTIFACT_KIND \
            or type(receipt["execution_receipt_version"]) is not int \
            or receipt["execution_receipt_version"] \
            != V04_EXECUTION_RECEIPT_VERSION \
            or isinstance(receipt["initial_process_id"], bool) \
            or not isinstance(receipt["initial_process_id"], int) \
            or receipt["initial_process_id"] <= 0 \
            or not isinstance(receipt["started_at_utc"], str) \
            or _UTC_RE.fullmatch(receipt["started_at_utc"]) is None:
        raise GateBReleaseAuditError("execution receipt contract is invalid")
    execution_id = _sha256(receipt["execution_id"], "execution_id")
    receipt_sha256 = sha256_bytes(receipt_bytes)
    if manifest["execution_id"] != execution_id \
            or manifest["execution_receipt_sha256"] != receipt_sha256:
        raise GateBReleaseAuditError("run manifest is not bound to its receipt")
    run_id = _sha256(manifest["run_id"], "run_id")
    if run_id != _run_id(expected_snapshot, expected_model):
        raise GateBReleaseAuditError("run identity does not recompute")
    expected_scope = {
        "camera_cv_only": True,
        "imu_and_annotation_payloads_unparsed": True,
        "sync_clock_window_spectrum_out_of_scope": True,
    }
    if manifest["scope"] != expected_scope:
        raise GateBReleaseAuditError("run scope boundary changed")
    count = _integer(manifest["recording_count"], "recording_count")
    if count != VIDIMU_V2_EXPECTED_RECORDING_COUNT:
        raise GateBReleaseAuditError("run does not contain exactly 208 recordings")
    recording_ids = manifest["recording_ids"]
    bundle_claims = manifest["bundles"]
    if not isinstance(recording_ids, list) or not isinstance(bundle_claims, list) \
            or len(recording_ids) != count or len(bundle_claims) != count \
            or recording_ids != sorted(recording_ids) \
            or len(set(recording_ids)) != count \
            or any(not isinstance(value, str) or not value for value in recording_ids):
        raise GateBReleaseAuditError("run recording inventory is invalid")
    if recording_ids != sorted(source_assets_by_recording):
        raise GateBReleaseAuditError(
            "run recording IDs do not equal the frozen source inventory"
        )
    snapshot_dir = finalized_dir / expected_snapshot
    model_dir = snapshot_dir / expected_model
    if {entry.name for entry in finalized_dir.iterdir()} != {expected_snapshot} \
            or snapshot_dir.is_symlink() or not snapshot_dir.is_dir() \
            or {entry.name for entry in snapshot_dir.iterdir()} != {expected_model} \
            or model_dir.is_symlink() or not model_dir.is_dir() \
            or {entry.name for entry in model_dir.iterdir()} != set(recording_ids):
        raise GateBReleaseAuditError("run directory topology is not exact")
    bundles: dict[str, AuditedV04Bundle] = {}
    recomputed_claims: list[dict[str, object]] = []
    for expected_recording, raw_claim in zip(recording_ids, bundle_claims, strict=True):
        if not isinstance(raw_claim, dict) or set(raw_claim) != {
            "recording_id", "finalization_id", "relative_path", "manifest_sha256"
        } or raw_claim["recording_id"] != expected_recording:
            raise GateBReleaseAuditError("run bundle claim is invalid")
        finalization_id = _sha256(raw_claim["finalization_id"], "finalization_id")
        manifest_sha = _sha256(raw_claim["manifest_sha256"], "manifest_sha256")
        relative = _safe_relative(raw_claim["relative_path"], "bundle relative_path")
        expected_relative = (
            f"finalized/{expected_snapshot}/{expected_model}/"
            f"{expected_recording}/{finalization_id}"
        )
        if relative != expected_relative:
            raise GateBReleaseAuditError("bundle path is outside frozen run hierarchy")
        recording_dir = model_dir / expected_recording
        if recording_dir.is_symlink() or not recording_dir.is_dir() \
                or {entry.name for entry in recording_dir.iterdir()} != {
                    finalization_id
                }:
            raise GateBReleaseAuditError(
                "recording outcome directory inventory is not exact"
            )
        try:
            bundle = audit_v04_bundle(root / relative)
        except V04BundleError as exc:
            raise GateBReleaseAuditError(
                f"bundle {expected_recording} failed strict audit"
            ) from exc
        bundle_manifest = bundle.manifest
        source_snapshot = bundle_manifest["source_snapshot"]
        model = bundle_manifest["model"]
        expected_assets = source_assets_by_recording[expected_recording]
        expected_video = next(
            item for item in expected_assets if item["asset_role"] == "VIDEO"
        )
        source_video = bundle_manifest["source_video"]
        if bundle_manifest["recording_id"] != expected_recording \
                or bundle_manifest["dataset"] != source_dataset \
                or source_snapshot["snapshot_manifest_sha256"] != expected_snapshot \
                or source_snapshot["source_inventory_sha256"] != expected_inventory \
                or model["model_manifest_sha256"] != expected_model \
                or bundle.marker["manifest_sha256"] != manifest_sha \
                or bundle_manifest["execution"] != {
                    "execution_id": execution_id,
                    "execution_receipt_sha256": receipt_sha256,
                } or bundle_manifest["source_assets"] != expected_assets \
                or source_video["asset_relative_path"] != (
                    f"assets/{expected_video['normalized_member_path']}"
                ) or source_video["source_object_id"] \
                != expected_video["source_object_id"] \
                or source_video["bytes"] != expected_video["bytes"] \
                or source_video["sha256"] != expected_video["sha256"]:
            raise GateBReleaseAuditError("bundle provenance contradicts its run")
        bundles[expected_recording] = bundle
        recomputed_claims.append({
            "recording_id": expected_recording,
            "finalization_id": finalization_id,
            "relative_path": relative,
            "manifest_sha256": manifest_sha,
        })
    if bundle_claims != recomputed_claims:
        raise GateBReleaseAuditError("run bundle claims are not canonical")
    if set(audit) != _RUN_AUDIT_FIELDS \
            or audit != {
                "artifact_kind": V04_GATE_B_RUN_ARTIFACT_KIND,
                "run_audit_version": V04_GATE_B_RUN_AUDIT_VERSION,
                "run_id": run_id,
                "recordings_expected": VIDIMU_V2_EXPECTED_RECORDING_COUNT,
                "recordings_finalized": count,
                "recordings_failed": 0,
                "source_snapshot_verified_before_and_after": True,
                "model_manifest_verified": True,
                "execution_receipt_verified": True,
                "bundle_audits_passed": count,
                "overall_verdict": "PASS",
            }:
        raise GateBReleaseAuditError("run audit does not recompute")
    if set(marker) != _RUN_MARKER_FIELDS \
            or marker != {
                "artifact_kind": V04_GATE_B_RUN_ARTIFACT_KIND,
                "run_success_version": V04_GATE_B_RUN_SUCCESS_VERSION,
                "run_id": run_id,
                "run_manifest_sha256": sha256_bytes(manifest_bytes),
                "run_audit_sha256": sha256_bytes(audit_bytes),
            }:
        raise GateBReleaseAuditError("run success marker is invalid")
    identities, hashes = _walk_regular_tree(root)
    expected_files = {
        EXECUTION_RECEIPT_FILE,
        RUN_MANIFEST_FILE,
        RUN_AUDIT_FILE,
        RUN_SUCCESS_FILE,
    }
    for claim in recomputed_claims:
        relative = str(claim["relative_path"])
        expected_files.update(f"{relative}/{name}" for name in (
            *TABLE_FILES.values(),
            "finalization_manifest.json",
            "finalization_audit.json",
            "_SUCCESS",
        ))
    if set(hashes) != expected_files:
        raise GateBReleaseAuditError("run tree contains missing or extra artifacts")
    processing_identities = [
        _processing_identity(bundle) for bundle in bundles.values()
    ]
    if not processing_identities or any(
        identity != processing_identities[0]
        for identity in processing_identities[1:]
    ):
        raise GateBReleaseAuditError(
            "run mixes decoder/model/preprocessing/runtime identities"
        )
    processing_identity = processing_identities[0]
    reconciliation = _run_reconciliation(bundles)
    if reconciliation["decoded_frames"] \
            != reconciliation["cv_frame_results"] \
            or reconciliation["selection_rows"] \
            != reconciliation["decoded_frames"] \
            or reconciliation["zero_detection_frames"] \
            + reconciliation["one_detection_frames"] \
            + reconciliation["multiple_detection_frames"] \
            != reconciliation["decoded_frames"]:
        raise GateBReleaseAuditError("run-wide frame/CV counts do not reconcile")
    return AuditedGateBRun(
        path=root,
        manifest=manifest,
        audit=audit,
        marker=marker,
        execution_receipt=receipt,
        bundles=bundles,
        file_identities=identities,
        file_hashes=hashes,
        processing_identity=processing_identity,
        reconciliation=reconciliation,
        canonical_content_sha256=_canonical_run_content(
            run_id,
            processing_identity,
            bundles,
        ),
    )


def _assert_inode_disjoint(
    primary: AuditedGateBRun,
    replay: AuditedGateBRun,
    *,
    source_path: Path,
) -> None:
    if primary.path == replay.path:
        raise GateBReleaseAuditError("primary and replay roots must be distinct")
    primary_root_state = primary.path.stat()
    replay_root_state = replay.path.stat()
    if (primary_root_state.st_dev, primary_root_state.st_ino) == (
        replay_root_state.st_dev, replay_root_state.st_ino
    ):
        raise GateBReleaseAuditError("primary and replay resolve to one root inode")
    if set(primary.file_identities) != set(replay.file_identities):
        raise GateBReleaseAuditError("primary/replay artifact topology differs")
    if set(primary.file_identities.values()).intersection(
        replay.file_identities.values()
    ):
        raise GateBReleaseAuditError(
            "primary/replay artifact inode sets are not disjoint"
        )
    source_identities, _source_hashes = _walk_regular_tree(source_path)
    source_inode_set = set(source_identities.values())
    if source_inode_set.intersection(primary.file_identities.values()) \
            or source_inode_set.intersection(replay.file_identities.values()):
        raise GateBReleaseAuditError(
            "derived run artifacts may not reuse source snapshot file inodes"
        )


def audit_vidimu_v04_gate_b_release(
    primary_run_root: str | Path,
    replay_run_root: str | Path,
    *,
    source_snapshot_path: str | Path,
    expected_source_snapshot_sha256: str,
    expected_source_inventory_sha256: str,
    expected_model_manifest_sha256: str,
    model_inputs: GateBModelInputs,
    numeric_tolerance: float | None = None,
) -> dict[str, object]:
    """Audit two clean-root runs and return one canonical fail-closed report."""

    try:
        expected_snapshot = _sha256(
            expected_source_snapshot_sha256, "expected source snapshot"
        )
        expected_inventory = _sha256(
            expected_source_inventory_sha256, "expected source inventory"
        )
        expected_model = _sha256(
            expected_model_manifest_sha256, "expected model manifest"
        )
        model_authority = _load_model_authority(
            model_inputs,
            expected_model_manifest_sha256=expected_model,
        )
        if numeric_tolerance is not None:
            raise GateBReleaseAuditError(
                "numeric re-inference cannot pass Gate B until a separately "
                "frozen authoritative snapshot has byte-identical replay proof"
            )
        source, source_assets_by_recording, source_dataset = (
            _load_source_authority(
            source_snapshot_path,
            expected_snapshot_sha256=expected_snapshot,
            expected_inventory_sha256=expected_inventory,
            )
        )
        primary = audit_gate_b_run(
            primary_run_root,
            expected_source_snapshot_sha256=expected_snapshot,
            expected_source_inventory_sha256=expected_inventory,
            expected_model_manifest_sha256=expected_model,
            _source_assets_by_recording=source_assets_by_recording,
            _source_dataset=source_dataset,
        )
        replay = audit_gate_b_run(
            replay_run_root,
            expected_source_snapshot_sha256=expected_snapshot,
            expected_source_inventory_sha256=expected_inventory,
            expected_model_manifest_sha256=expected_model,
            _source_assets_by_recording=source_assets_by_recording,
            _source_dataset=source_dataset,
        )
        _assert_inode_disjoint(primary, replay, source_path=source.path)
        if primary.manifest["recording_ids"] != replay.manifest["recording_ids"]:
            raise GateBReleaseAuditError("primary/replay recording inventories differ")
        if primary.processing_identity != replay.processing_identity:
            raise GateBReleaseAuditError(
                "primary/replay processing identities differ"
            )
        if any(
            primary.processing_identity[field] != expected_value
            for field, expected_value in model_authority.items()
        ):
            raise GateBReleaseAuditError(
                "run model provenance contradicts independently verified evidence"
            )
        if primary.execution_receipt["execution_id"] \
                == replay.execution_receipt["execution_id"] \
                or primary.execution_receipt["initial_process_id"] \
                == replay.execution_receipt["initial_process_id"]:
            raise GateBReleaseAuditError(
                "primary/replay lack distinct execution/process receipts"
            )
        per_record: list[dict[str, object]] = []
        byte_identical = True
        canonical_identical = True
        for recording_id in primary.manifest["recording_ids"]:
            left = primary.bundles[recording_id]
            right = replay.bundles[recording_id]
            if left.manifest["finalization_id"] != right.manifest["finalization_id"]:
                raise GateBReleaseAuditError("primary/replay finalization IDs differ")
            table_bytes_equal = all(
                left.manifest["tables"][name]["sha256"]
                == right.manifest["tables"][name]["sha256"]
                for name in TABLE_FILES
            )
            table_semantics_equal = all(
                left.manifest["tables"][name]["semantic_sha256"]
                == right.manifest["tables"][name]["semantic_sha256"]
                for name in TABLE_FILES
            )
            byte_identical = byte_identical and table_bytes_equal
            canonical_identical = canonical_identical and table_semantics_equal
            per_record.append({
                "recording_id": recording_id,
                "finalization_id": left.manifest["finalization_id"],
                "parquet_bytes_identical": table_bytes_equal,
                "canonical_content_identical": table_semantics_equal,
                "bounded_numeric_difference_observed": False,
                "maximum_absolute_numeric_difference": None,
                "primary_artifact_hashes": {
                    name: left.manifest["tables"][name]["sha256"]
                    for name in TABLE_FILES
                },
                "replay_artifact_hashes": {
                    name: right.manifest["tables"][name]["sha256"]
                    for name in TABLE_FILES
                },
                "primary_semantic_hashes": {
                    name: left.manifest["tables"][name]["semantic_sha256"]
                    for name in TABLE_FILES
                },
                "replay_semantic_hashes": {
                    name: right.manifest["tables"][name]["semantic_sha256"]
                    for name in TABLE_FILES
                },
                "decoded_frames": left.audit["decoded_frames"],
                "cv_frame_results": left.audit["cv_frame_results"],
                "cv_detections": left.audit["cv_detections"],
                "primary_hand_selections": left.audit[
                    "primary_hand_selections"
                ],
            })
        if byte_identical:
            status = BYTE_IDENTICAL_SOURCE_TO_CV_PASS
        elif canonical_identical:
            status = CANONICAL_CONTENT_IDENTICAL_PARQUET_BYTES_DIFFER_PASS
        else:
            raise GateBReleaseAuditError("replay differs outside the admitted contract")
        source_reconciliation = {
            "inventory_records": VIDIMU_V2_EXPECTED_RECORDING_COUNT,
            "source_objects_expected": source.source_object_count,
            "source_objects_present": source.source_object_count,
            "source_objects_downloaded": source.source_object_count,
            "source_objects_hash_verified": source.source_object_count,
            "source_objects_failed": 0,
            "asset_references_expected": VIDIMU_V2_EXPECTED_ASSET_REFERENCE_COUNT,
            "asset_references_resolved": source.extracted_asset_count,
            "asset_references_unavailable": source.unavailable_asset_count,
            "asset_references_ambiguous": 0,
            "unreferenced_extracted_assets": 0,
        }
        report = {
            "artifact_kind": V04_GATE_B_RELEASE_AUDIT_KIND,
            "release_audit_version": V04_GATE_B_RELEASE_AUDIT_VERSION,
            "status": status,
            "overall_verdict": "PASS",
            "independent_rerun_status": (
                "DISTINCT_EXECUTION_RECEIPTS_INITIAL_PROCESSES_ROOTS_"
                "AND_ARTIFACT_FILES_VERIFIED"
            ),
            "snapshot_replay_status": status,
            "source_snapshot_sha256": expected_snapshot,
            "source_inventory_sha256": expected_inventory,
            "model_manifest_sha256": expected_model,
            "recording_count": VIDIMU_V2_EXPECTED_RECORDING_COUNT,
            "source_assets_present": VIDIMU_V2_EXPECTED_ASSET_REFERENCE_COUNT,
            "source_hashes_verified": True,
            "videos_expected": VIDIMU_V2_EXPECTED_RECORDING_COUNT,
            "videos_opened": VIDIMU_V2_EXPECTED_RECORDING_COUNT,
            "videos_failed": 0,
            "source_reconciliation": source_reconciliation,
            "primary_run_reconciliation": dict(primary.reconciliation),
            "replay_run_reconciliation": dict(replay.reconciliation),
            "provenance_reconciliation": {
                **primary.processing_identity,
                "snapshot_manifest_sha256": expected_snapshot,
                "snapshot_wide_identity_uniform": True,
                "primary_replay_identity_equal": True,
                "model_evidence_independently_verified": True,
            },
            "primary_run_id": primary.manifest["run_id"],
            "replay_run_id": replay.manifest["run_id"],
            "primary_execution_id": primary.execution_receipt["execution_id"],
            "replay_execution_id": replay.execution_receipt["execution_id"],
            "primary_execution_receipt": dict(primary.execution_receipt),
            "replay_execution_receipt": dict(replay.execution_receipt),
            "distinct_roots_and_artifact_inodes_verified": True,
            "non_parquet_artifact_hashes_identical": all(
                primary.file_hashes[path] == replay.file_hashes[path]
                for path in primary.file_hashes
                if not path.endswith(".parquet")
            ),
            "numeric_tolerance": None,
            "maximum_absolute_numeric_difference": None,
            "artifact_hashes": {
                "primary_run_manifest": primary.file_hashes[RUN_MANIFEST_FILE],
                "primary_run_audit": primary.file_hashes[RUN_AUDIT_FILE],
                "primary_run_success": primary.file_hashes[RUN_SUCCESS_FILE],
                "primary_canonical_content": primary.canonical_content_sha256,
                "replay_run_manifest": replay.file_hashes[RUN_MANIFEST_FILE],
                "replay_run_audit": replay.file_hashes[RUN_AUDIT_FILE],
                "replay_run_success": replay.file_hashes[RUN_SUCCESS_FILE],
                "replay_canonical_content": replay.canonical_content_sha256,
            },
            "run_a_canonical_hash": primary.canonical_content_sha256,
            "run_b_canonical_hash": replay.canonical_content_sha256,
            "per_record": per_record,
        }
        return report
    except MemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - ordinary audit defects return FAIL
        return {
            "artifact_kind": V04_GATE_B_RELEASE_AUDIT_KIND,
            "release_audit_version": V04_GATE_B_RELEASE_AUDIT_VERSION,
            "status": FAIL,
            "overall_verdict": "FAIL",
            "independent_rerun_status": "NOT_VERIFIED",
            "snapshot_replay_status": FAIL,
            "failure_code": type(exc).__name__,
            "source_assets_present": 0,
            "source_hashes_verified": False,
            "videos_expected": VIDIMU_V2_EXPECTED_RECORDING_COUNT,
            "videos_opened": 0,
            "videos_failed": 0,
            "source_reconciliation": {},
            "primary_run_reconciliation": {},
            "replay_run_reconciliation": {},
            "provenance_reconciliation": {},
            "artifact_hashes": {},
            "run_a_canonical_hash": None,
            "run_b_canonical_hash": None,
            "per_record": [],
        }


def write_release_audit_report(
    path: str | Path,
    report: Mapping[str, object],
) -> str:
    """Write one canonical report without replacing an existing artifact."""

    if report.get("status") not in RELEASE_STATUSES:
        raise GateBReleaseAuditError("release report status is invalid")
    payload = canonical_json_bytes(dict(report))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.tmp-{uuid.uuid4().hex}"
    )
    descriptor = _exclusive_descriptor(temporary)
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW \
            | getattr(os, "O_CLOEXEC", 0)
        parent_descriptor = os.open(destination.parent, flags)
        try:
            _rename_noreplace(
                parent_descriptor,
                temporary.name,
                destination.name,
            )
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except (FinalizationBundleError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise GateBReleaseAuditError(
            "release report atomic publication failed"
        ) from exc
    return sha256_bytes(payload)


__all__ = [
    "BYTE_IDENTICAL_SOURCE_TO_CV_PASS",
    "CANONICAL_CONTENT_IDENTICAL_PARQUET_BYTES_DIFFER_PASS",
    "FAIL",
    "AuditedGateBRun",
    "GateBReleaseAuditError",
    "audit_gate_b_run",
    "audit_vidimu_v04_gate_b_release",
    "write_release_audit_report",
]
