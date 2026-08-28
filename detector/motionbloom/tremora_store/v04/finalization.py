"""Production Gate-B orchestration from a verified v0.4 VIDIMU snapshot.

The boundary intentionally stops at decoded-frame and camera-CV persistence.
IMU and annotation assets are required and provenance-bound, but their payloads
are not parsed, synchronized, windowed, or transformed here.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import stat
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..decode.pts_decoder import DecodeConfig, PTSDecoder
from ..finalize._bundle_io import (
    FinalizationBundleError,
    _exclusive_descriptor,
    _fsync_directory,
    _rename_noreplace,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
)
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
    V04_BUNDLE_SCHEMA_VERSION,
    V04_FINALIZATION_ARTIFACT_KIND,
    V04_SUCCESS_FILES,
    PublishedV04Bundle,
    V04BundleError,
    audit_v04_bundle,
    finalization_identity,
    publish_v04_bundle,
)
from .frame_finalizer import finalize_recording_frames
from .mediapipe_hand_landmarker import MediaPipeHandLandmarkerEstimator
from .model_manifest import (
    VerifiedProductionModel,
    load_and_verify_production_model,
)
from .schemas import V04_ASSOCIATION_CONTRACT_VERSION

V04_GATE_B_RUN_ARTIFACT_KIND = "TREMORA_VIDIMU_V04_GATE_B_RUN"
V04_GATE_B_RUN_MANIFEST_VERSION = 1
V04_GATE_B_RUN_AUDIT_VERSION = 1
V04_GATE_B_RUN_SUCCESS_VERSION = 1
V04_GATE_B_RUN_IDENTITY_DOMAIN = "tremora-vidimu-v04-gate-b-run-1"
RUN_MANIFEST_FILE = "run_manifest.json"
RUN_AUDIT_FILE = "run_audit.json"
RUN_SUCCESS_FILE = "_RUN_SUCCESS"
EXECUTION_RECEIPT_FILE = "execution_receipt.json"
V04_EXECUTION_RECEIPT_VERSION = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_INVENTORY_FIELDS = frozenset({
    "inventory_schema_version",
    "dataset_id",
    "dataset_version",
    "license_id",
    "citation_id",
    "terms_snapshot_id",
    "terms_source_object_id",
    "source_objects",
    "asset_references",
    "budgets",
})
_ASSET_FIELDS = frozenset({
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
})


class GateBFinalizationError(RuntimeError):
    """Raised when a production Gate-B run cannot be proven complete."""


@dataclass(frozen=True, slots=True)
class GateBModelInputs:
    """Independent caller-provided anchors for the production CV model."""

    manifest_path: Path
    weights_path: Path
    preprocessing_config_path: Path
    runtime_lock_path: Path
    expected_manifest_sha256: str
    vendored_model_inventory_path: Path | None = None


@dataclass(frozen=True, slots=True)
class FinalizedGateBRun:
    path: Path
    run_id: str
    source_snapshot_sha256: str
    model_manifest_sha256: str
    recording_count: int
    run_manifest_sha256: str
    execution_id: str


@dataclass(frozen=True, slots=True)
class _RecordingAsset:
    recording_id: str
    source_object_id: str
    archive_member_path: str
    normalized_member_path: str
    asset_role: str
    modality: str
    size_bytes: int
    sha256: str


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GateBFinalizationError(f"{field} must be a lowercase SHA-256")
    return value


def _real_directory(path: Path, field: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise GateBFinalizationError(f"{field} must be a real directory")
    return path.resolve(strict=True)


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_disjoint_roots(output: Path, source: Path, model: GateBModelInputs) -> None:
    paths = [
        source,
        Path(model.manifest_path).resolve(strict=True),
        Path(model.weights_path).resolve(strict=True),
        Path(model.preprocessing_config_path).resolve(strict=True),
        Path(model.runtime_lock_path).resolve(strict=True),
    ]
    if model.vendored_model_inventory_path is not None:
        paths.append(Path(model.vendored_model_inventory_path).resolve(strict=True))
    if any(_is_within(output, path) or _is_within(path, output) for path in paths):
        raise GateBFinalizationError(
            "output, source snapshot, and model evidence must be disjoint roots"
        )


def _open_output_root(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() or output.is_symlink():
        return _real_directory(output, "output root")
    parent = _real_directory(output.parent, "output parent")
    output.mkdir(mode=0o700)
    resolved = _real_directory(output, "output root")
    if resolved.parent != parent:
        raise GateBFinalizationError("output root escaped its requested parent")
    return resolved


def _load_inventory(
    snapshot_path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], dict[str, tuple[_RecordingAsset, ...]]]:
    try:
        inventory, inventory_bytes = read_json(
            snapshot_path / SOURCE_INVENTORY_FILE,
            max_bytes=16 * 1024 * 1024,
        )
    except FinalizationBundleError as exc:
        raise GateBFinalizationError("source inventory cannot be read canonically") from exc
    if hashlib.sha256(inventory_bytes).hexdigest() != expected_sha256:
        raise GateBFinalizationError("source inventory changed after strict verification")
    if set(inventory) != _INVENTORY_FIELDS \
            or inventory["dataset_id"] != VIDIMU_V2_DATASET_ID \
            or inventory["dataset_version"] != VIDIMU_V2_DATASET_VERSION \
            or inventory["license_id"] != VIDIMU_V2_LICENSE_ID \
            or inventory["citation_id"] != VIDIMU_V2_CITATION_ID:
        raise GateBFinalizationError("source inventory is not frozen VIDIMU v2")
    raw_assets = inventory["asset_references"]
    if not isinstance(raw_assets, list) \
            or len(raw_assets) != VIDIMU_V2_EXPECTED_ASSET_REFERENCE_COUNT:
        raise GateBFinalizationError("source inventory asset cardinality is invalid")
    grouped: defaultdict[str, list[_RecordingAsset]] = defaultdict(list)
    for raw in raw_assets:
        if not isinstance(raw, dict) or set(raw) != _ASSET_FIELDS \
                or raw["availability"] != "REQUIRED" \
                or raw["unavailable_reason"] is not None:
            raise GateBFinalizationError(
                "Gate B requires every frozen video/IMU/annotation asset"
            )
        recording_id = raw["recording_id"]
        source_object_id = raw["source_object_id"]
        archive_member_path = raw["archive_member_path"]
        normalized_member_path = raw["normalized_member_path"]
        role = raw["asset_role"]
        modality = raw["modality"]
        size = raw["expected_size_bytes"]
        sha = raw["expected_sha256"]
        if not all(isinstance(value, str) and value for value in (
            recording_id,
            source_object_id,
            archive_member_path,
            normalized_member_path,
            role,
            modality,
        )) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise GateBFinalizationError("source inventory asset entry is invalid")
        _sha256(sha, "asset sha256")
        grouped[recording_id].append(_RecordingAsset(
            recording_id=recording_id,
            source_object_id=source_object_id,
            archive_member_path=archive_member_path,
            normalized_member_path=normalized_member_path,
            asset_role=role,
            modality=modality,
            size_bytes=size,
            sha256=sha,
        ))
    if len(grouped) != VIDIMU_V2_EXPECTED_RECORDING_COUNT:
        raise GateBFinalizationError("Gate B requires exactly 208 recordings")
    result: dict[str, tuple[_RecordingAsset, ...]] = {}
    for recording_id, assets in grouped.items():
        roles = [asset.asset_role for asset in assets]
        if sorted(roles) != ["ANNOTATION", "IMU", "VIDEO"]:
            raise GateBFinalizationError(
                f"recording {recording_id} lacks exact video/IMU/annotation topology"
            )
        result[recording_id] = tuple(sorted(
            assets,
            key=lambda item: (
                item.asset_role,
                item.source_object_id,
                item.archive_member_path,
            ),
        ))
    return inventory, dict(sorted(result.items()))


def _load_model(inputs: GateBModelInputs) -> VerifiedProductionModel:
    if type(inputs) is not GateBModelInputs:
        raise GateBFinalizationError("model inputs must use the closed GateBModelInputs")
    expected = _sha256(inputs.expected_manifest_sha256, "model manifest anchor")
    try:
        return load_and_verify_production_model(
            inputs.manifest_path,
            weights_path=inputs.weights_path,
            preprocessing_config_path=inputs.preprocessing_config_path,
            runtime_lock_path=inputs.runtime_lock_path,
            expected_manifest_sha256=expected,
            vendored_model_inventory=inputs.vendored_model_inventory_path,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise GateBFinalizationError("production model verification failed") from exc


def _identity_inputs(
    *,
    source_snapshot_sha256: str,
    source_video_sha256: str,
    decoder_version: str,
    decoder_config_sha256: str,
    model_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_video_sha256": source_video_sha256,
        "decoder_version": decoder_version,
        "decoder_config_sha256": decoder_config_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "association_contract_version": V04_ASSOCIATION_CONTRACT_VERSION,
        "bundle_schema_version": V04_BUNDLE_SCHEMA_VERSION,
    }


def build_recording_manifest(
    *,
    inventory: Mapping[str, object],
    source_snapshot_sha256: str,
    source_inventory_sha256: str,
    assets: tuple[_RecordingAsset, ...],
    decoded_source_bytes: int,
    decoder_version: str,
    decoder_config: DecodeConfig,
    model: VerifiedProductionModel,
    finalization_id_value: str,
    execution_id: str,
    execution_receipt_sha256: str,
) -> dict[str, object]:
    """Build the closed recording manifest before table claims are injected."""

    if len(assets) != 3:
        raise GateBFinalizationError("recording manifest requires three assets")
    recording_id = assets[0].recording_id
    if any(asset.recording_id != recording_id for asset in assets):
        raise GateBFinalizationError("recording manifest assets are mixed")
    video = next(asset for asset in assets if asset.asset_role == "VIDEO")
    model_manifest = model.manifest
    return {
        "artifact_kind": V04_FINALIZATION_ARTIFACT_KIND,
        "manifest_version": 1,
        "bundle_schema_version": V04_BUNDLE_SCHEMA_VERSION,
        "association_contract_version": V04_ASSOCIATION_CONTRACT_VERSION,
        "finalization_id": finalization_id_value,
        "recording_id": recording_id,
        "dataset": {
            "dataset_id": inventory["dataset_id"],
            "dataset_version": inventory["dataset_version"],
            "license_id": inventory["license_id"],
            "citation_id": inventory["citation_id"],
        },
        "source_snapshot": {
            "snapshot_manifest_sha256": source_snapshot_sha256,
            "source_inventory_sha256": source_inventory_sha256,
        },
        "source_assets": [{
            "recording_id": asset.recording_id,
            "source_object_id": asset.source_object_id,
            "archive_member_path": asset.archive_member_path,
            "normalized_member_path": asset.normalized_member_path,
            "asset_role": asset.asset_role,
            "modality": asset.modality,
            "bytes": asset.size_bytes,
            "sha256": asset.sha256,
        } for asset in assets],
        "source_video": {
            "asset_relative_path": f"assets/{video.normalized_member_path}",
            "source_object_id": video.source_object_id,
            "bytes": decoded_source_bytes,
            "sha256": video.sha256,
            "stream_index": decoder_config.stream_index,
            "hash_verified_during_pinned_decode": True,
        },
        "decoder": {
            "decoder_version": decoder_version,
            "decoder_config": asdict(decoder_config),
            "decoder_config_sha256": decoder_config.sha256,
        },
        "model": {
            "model_manifest_sha256": model.manifest_sha256,
            "model_id": model_manifest["model_id"],
            "model_weights_sha256": model_manifest["model_weights_sha256"],
            "preprocessing_config_sha256": model_manifest[
                "preprocessing_config_sha256"
            ],
            "runtime_lock_sha256": model_manifest["runtime_lock_sha256"],
            "inference_environment_id": (
                "native-runtime-sha256:"
                f"{model_manifest['runtime_lock_sha256']}"
            ),
        },
        "execution": {
            "execution_id": _sha256(execution_id, "execution_id"),
            "execution_receipt_sha256": _sha256(
                execution_receipt_sha256,
                "execution_receipt_sha256",
            ),
        },
        "scope": {
            "camera_only_inference": True,
            "decoded_frames_complete": True,
            "all_detections_persisted": True,
            "imu_parsing_performed": False,
            "clock_or_sync_estimation_performed": False,
            "windowing_or_spectral_analysis_performed": False,
        },
        "tables": {},
    }


def _run_id(snapshot_sha256: str, model_sha256: str) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "domain": V04_GATE_B_RUN_IDENTITY_DOMAIN,
        "source_snapshot_sha256": snapshot_sha256,
        "model_manifest_sha256": model_sha256,
        "association_contract_version": V04_ASSOCIATION_CONTRACT_VERSION,
        "bundle_schema_version": V04_BUNDLE_SCHEMA_VERSION,
    })).hexdigest()


def _source_asset_evidence(
    assets: tuple[_RecordingAsset, ...],
) -> list[dict[str, object]]:
    return [{
        "recording_id": asset.recording_id,
        "source_object_id": asset.source_object_id,
        "archive_member_path": asset.archive_member_path,
        "normalized_member_path": asset.normalized_member_path,
        "asset_role": asset.asset_role,
        "modality": asset.modality,
        "bytes": asset.size_bytes,
        "sha256": asset.sha256,
    } for asset in assets]


def _execution_receipt(
    output: Path,
    *,
    initially_empty: bool,
) -> tuple[dict[str, object], str]:
    path = output / EXECUTION_RECEIPT_FILE
    if initially_empty:
        receipt = {
            "artifact_kind": V04_GATE_B_RUN_ARTIFACT_KIND,
            "execution_receipt_version": V04_EXECUTION_RECEIPT_VERSION,
            "execution_id": secrets.token_hex(32),
            "initial_process_id": os.getpid(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="microseconds"
            ).replace("+00:00", "Z"),
        }
        encoded = _write_run_json_exclusive(path, receipt)
        return receipt, sha256_bytes(encoded)
    try:
        receipt, encoded = read_json(path)
    except FinalizationBundleError as exc:
        raise GateBFinalizationError(
            "resumable run lacks a canonical execution receipt"
        ) from exc
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
            or not receipt["started_at_utc"].endswith("Z"):
        raise GateBFinalizationError("execution receipt contract is invalid")
    _sha256(receipt["execution_id"], "execution_id")
    return receipt, sha256_bytes(encoded)


def _cleanup_recording_staging(
    recording_dir: Path,
    expected_finalization_id: str,
) -> None:
    """Remove one exact unpublished owned staging remnant after a hard kill."""

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW \
        | getattr(os, "O_CLOEXEC", 0)
    recording_descriptor = os.open(recording_dir, flags)
    try:
        names = set(os.listdir(recording_descriptor))
        pattern = re.compile(
            rf"\.staging-{re.escape(expected_finalization_id)}-[0-9a-f]{{32}}\Z"
        )
        staging_names = sorted(name for name in names if pattern.fullmatch(name))
        unknown_hidden = {
            name for name in names
            if name.startswith(".staging-") and name not in staging_names
        }
        if unknown_hidden or len(staging_names) > 1:
            raise GateBFinalizationError(
                "recording contains an unowned or ambiguous staging remnant"
            )
        if not staging_names:
            return
        if expected_finalization_id in names:
            raise GateBFinalizationError(
                "published recording also contains a staging remnant"
            )
        staging_name = staging_names[0]
        staging_descriptor = os.open(
            staging_name,
            flags,
            dir_fd=recording_descriptor,
        )
        try:
            entries = set(os.listdir(staging_descriptor))
            if not entries.issubset(V04_SUCCESS_FILES):
                raise GateBFinalizationError(
                    "staging remnant is not a provably owned bundle"
                )
            for name in entries:
                state = os.stat(
                    name,
                    dir_fd=staging_descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(state.st_mode):
                    raise GateBFinalizationError(
                        "staging remnant contains a non-regular artifact"
                    )
            for name in entries:
                os.unlink(name, dir_fd=staging_descriptor)
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
        os.rmdir(staging_name, dir_fd=recording_descriptor)
        os.fsync(recording_descriptor)
    finally:
        os.close(recording_descriptor)


def _resume_verified_bundles(
    output: Path,
    *,
    finalized_root: Path,
    by_recording: Mapping[str, tuple[_RecordingAsset, ...]],
    source_snapshot_sha256: str,
    source_inventory_sha256: str,
    model_manifest_sha256: str,
    decoder_version: str,
    decoder_config: DecodeConfig,
    execution_id: str,
    execution_receipt_sha256: str,
) -> dict[str, PublishedV04Bundle]:
    """Admit only exact same-root immutable recording successes after a crash."""

    top_names = {entry.name for entry in output.iterdir()}
    if not top_names:
        return {}
    allowed_top = {
        "finalized",
        EXECUTION_RECEIPT_FILE,
        RUN_MANIFEST_FILE,
        RUN_AUDIT_FILE,
    }
    if RUN_SUCCESS_FILE in top_names:
        raise GateBFinalizationError(
            "completed run roots are immutable and cannot be reused"
        )
    if "finalized" not in top_names or not top_names.issubset(allowed_top):
        raise GateBFinalizationError(
            "nonempty output is not an exact resumable Gate-B run root"
        )
    finalized = output / "finalized"
    if finalized.is_symlink() or not finalized.is_dir():
        raise GateBFinalizationError("resumable finalized hierarchy is unsafe")
    if {entry.name for entry in finalized.iterdir()} != {
        source_snapshot_sha256
    }:
        raise GateBFinalizationError("resumable run contains another source snapshot")
    snapshot_dir = finalized / source_snapshot_sha256
    if snapshot_dir.is_symlink() or not snapshot_dir.is_dir() \
            or {entry.name for entry in snapshot_dir.iterdir()} != {
                model_manifest_sha256
            }:
        raise GateBFinalizationError("resumable run contains another model identity")
    if finalized_root.is_symlink() or not finalized_root.is_dir():
        raise GateBFinalizationError("resumable model hierarchy is unsafe")

    expected_recordings = set(by_recording)
    actual_recordings = {entry.name for entry in finalized_root.iterdir()}
    if not actual_recordings.issubset(expected_recordings):
        raise GateBFinalizationError("resumable run contains an unexpected recording")
    resumed: dict[str, PublishedV04Bundle] = {}
    for recording_id in sorted(actual_recordings):
        recording_dir = finalized_root / recording_id
        if recording_dir.is_symlink() or not recording_dir.is_dir():
            raise GateBFinalizationError("resumable recording path is unsafe")
        video = next(
            asset
            for asset in by_recording[recording_id]
            if asset.asset_role == "VIDEO"
        )
        expected_id = finalization_identity(_identity_inputs(
            source_snapshot_sha256=source_snapshot_sha256,
            source_video_sha256=video.sha256,
            decoder_version=decoder_version,
            decoder_config_sha256=decoder_config.sha256,
            model_manifest_sha256=model_manifest_sha256,
        ))
        _cleanup_recording_staging(recording_dir, expected_id)
        children = list(recording_dir.iterdir())
        if not children:
            # An owned empty recording directory is a normal pre-publication
            # crash remnant; the recording will be finalized below.
            continue
        if len(children) != 1 or children[0].name != expected_id \
                or children[0].is_symlink() or not children[0].is_dir():
            raise GateBFinalizationError(
                "resumable recording outcome inventory is not exact"
            )
        try:
            audited = audit_v04_bundle(children[0])
        except V04BundleError as exc:
            raise GateBFinalizationError(
                "resumable recording bundle failed strict audit"
            ) from exc
        manifest = audited.manifest
        if manifest["source_snapshot"] != {
            "snapshot_manifest_sha256": source_snapshot_sha256,
            "source_inventory_sha256": source_inventory_sha256,
        } or manifest["source_assets"] != _source_asset_evidence(
            by_recording[recording_id]
        ) or manifest["model"]["model_manifest_sha256"] \
                != model_manifest_sha256 \
                or manifest["decoder"]["decoder_version"] != decoder_version \
                or manifest["decoder"]["decoder_config_sha256"] \
                != decoder_config.sha256 \
                or manifest["execution"] != {
                    "execution_id": execution_id,
                    "execution_receipt_sha256": execution_receipt_sha256,
                }:
            raise GateBFinalizationError(
                "resumable bundle contradicts current frozen processing inputs"
            )
        for artifact in children[0].iterdir():
            state = artifact.lstat()
            if not stat.S_ISREG(state.st_mode) or stat.S_ISLNK(state.st_mode) \
                    or state.st_nlink != 1:
                raise GateBFinalizationError(
                    "resumable bundle artifacts must be unshared regular files"
                )
        resumed[recording_id] = PublishedV04Bundle(
            path=audited.path,
            finalization_id=expected_id,
            manifest_sha256=str(audited.marker["manifest_sha256"]),
            audit=audited.audit,
        )
    return resumed


def _write_run_json_exclusive(path: Path, value: Mapping[str, object]) -> bytes:
    """Atomically publish one run JSON file through a fixed owned temp name."""

    payload = canonical_json_bytes(dict(value))
    temporary = path.parent / f".{path.name}.tmp"
    descriptor = _exclusive_descriptor(temporary)
    with os.fdopen(descriptor, "wb", buffering=0) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW \
        | getattr(os, "O_CLOEXEC", 0)
    parent_descriptor = os.open(path.parent, flags)
    try:
        _rename_noreplace(
            parent_descriptor,
            temporary.name,
            path.name,
        )
        os.fsync(parent_descriptor)
    except FinalizationBundleError as exc:
        raise GateBFinalizationError(
            f"atomic run metadata publication collided: {path.name}"
        ) from exc
    finally:
        os.close(parent_descriptor)
    return payload


def _cleanup_run_temporaries(output: Path) -> None:
    names = {
        f".{name}.tmp" for name in (
            EXECUTION_RECEIPT_FILE,
            RUN_MANIFEST_FILE,
            RUN_AUDIT_FILE,
            RUN_SUCCESS_FILE,
        )
    }
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW \
        | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(output, flags)
    changed = False
    try:
        present = set(os.listdir(descriptor))
        for name in names.intersection(present):
            state = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(state.st_mode):
                raise GateBFinalizationError(
                    "run temporary artifact is not an owned regular file"
                )
            os.unlink(name, dir_fd=descriptor)
            changed = True
        if changed:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_or_verify_run_json(path: Path, value: Mapping[str, object]) -> bytes:
    payload = canonical_json_bytes(dict(value))
    if path.exists() or path.is_symlink():
        try:
            existing, existing_bytes = read_json(path)
        except FinalizationBundleError as exc:
            raise GateBFinalizationError(
                f"partial run metadata {path.name} is invalid"
            ) from exc
        if existing != dict(value) or existing_bytes != payload:
            raise GateBFinalizationError(
                f"partial run metadata {path.name} contradicts completed run"
            )
        return existing_bytes
    return _write_run_json_exclusive(path, value)


def finalize_vidimu_v04_gate_b(
    source_snapshot_path: str | Path,
    output_root: str | Path,
    *,
    model_inputs: GateBModelInputs,
    decoder_config: DecodeConfig | None = None,
) -> FinalizedGateBRun:
    """Finalize all 208 frozen VIDIMU recordings into one clean run root."""

    if decoder_config is None:
        decoder_config = DecodeConfig()
    if type(decoder_config) is not DecodeConfig:
        raise GateBFinalizationError("decoder_config must be an exact DecodeConfig")
    try:
        trusted_config = DecodeConfig(**asdict(decoder_config))
    except (TypeError, ValueError) as exc:
        raise GateBFinalizationError("decoder_config cannot be reconstructed") from exc
    try:
        source = verify_vidimu_source_snapshot(source_snapshot_path)
    except SourceSnapshotError as exc:
        raise GateBFinalizationError("source snapshot strict verification failed") from exc
    inventory, by_recording = _load_inventory(
        source.path,
        expected_sha256=source.source_inventory_sha256,
    )
    model = _load_model(model_inputs)
    output = _open_output_root(output_root)
    _require_disjoint_roots(output, source.path, model_inputs)
    _cleanup_run_temporaries(output)

    snapshot_sha = source.snapshot_manifest_sha256
    model_sha = model.manifest_sha256
    finalized_root = output / "finalized" / snapshot_sha / model_sha
    initially_empty = not any(output.iterdir())
    receipt, receipt_sha256 = _execution_receipt(
        output,
        initially_empty=initially_empty,
    )
    execution_id = str(receipt["execution_id"])
    current_decoder_version = PTSDecoder(
        DecodeConfig(**asdict(trusted_config))
    ).decoder_version
    receipt_only = {entry.name for entry in output.iterdir()} == {
        EXECUTION_RECEIPT_FILE
    }
    if initially_empty or receipt_only:
        finalized_root.mkdir(parents=True)
        resumed: dict[str, PublishedV04Bundle] = {}
    else:
        resumed = _resume_verified_bundles(
            output,
            finalized_root=finalized_root,
            by_recording=by_recording,
            source_snapshot_sha256=snapshot_sha,
            source_inventory_sha256=source.source_inventory_sha256,
            model_manifest_sha256=model_sha,
            decoder_version=current_decoder_version,
            decoder_config=trusted_config,
            execution_id=execution_id,
            execution_receipt_sha256=receipt_sha256,
        )
    if finalized_root.is_symlink() or not finalized_root.is_dir():
        raise GateBFinalizationError("finalized output hierarchy is unsafe")

    published_by_recording = dict(resumed)
    for recording_id, assets in by_recording.items():
        if recording_id in resumed:
            continue
        video = next(asset for asset in assets if asset.asset_role == "VIDEO")
        video_path = source.path / "assets" / video.normalized_member_path
        decoder = PTSDecoder(DecodeConfig(**asdict(trusted_config)))
        estimator = MediaPipeHandLandmarkerEstimator(model)
        try:
            decoded = decoder.decode(
                video_path,
                expected_source_sha256=video.sha256,
            )
            if decoded.source_bytes != video.size_bytes \
                    or decoded.source_video_sha256 != video.sha256 \
                    or decoded.decoder_config_sha256 != trusted_config.sha256 \
                    or decoded.decoder_version != current_decoder_version:
                raise GateBFinalizationError(
                    "pinned decoder result contradicts source inventory"
                )
            tables = finalize_recording_frames(
                decoded.frames,
                dataset_id=VIDIMU_V2_DATASET_ID,
                recording_id=recording_id,
                estimator=estimator,
            )
            identity_inputs = _identity_inputs(
                source_snapshot_sha256=snapshot_sha,
                source_video_sha256=decoded.source_video_sha256,
                decoder_version=decoded.decoder_version,
                decoder_config_sha256=decoded.decoder_config_sha256,
                model_manifest_sha256=model_sha,
            )
            finalization_id_value = finalization_identity(identity_inputs)
            manifest = build_recording_manifest(
                inventory=inventory,
                source_snapshot_sha256=snapshot_sha,
                source_inventory_sha256=source.source_inventory_sha256,
                assets=assets,
                decoded_source_bytes=decoded.source_bytes,
                decoder_version=decoded.decoder_version,
                decoder_config=trusted_config,
                model=model,
                finalization_id_value=finalization_id_value,
                execution_id=execution_id,
                execution_receipt_sha256=receipt_sha256,
            )
            published_by_recording[recording_id] = publish_v04_bundle(
                finalized_root,
                recording_id=recording_id,
                finalization_id=finalization_id_value,
                tables=tables.as_dict(),
                manifest=manifest,
            )
            del tables, decoded
        except (V04BundleError, OSError, RuntimeError, ValueError) as exc:
            raise GateBFinalizationError(
                f"recording {recording_id} did not produce a valid success bundle"
            ) from exc
        finally:
            estimator.close()

    if len(published_by_recording) != VIDIMU_V2_EXPECTED_RECORDING_COUNT:
        raise GateBFinalizationError("run did not finalize all 208 recordings")
    try:
        verified_again = verify_vidimu_source_snapshot(source.path)
    except SourceSnapshotError as exc:
        raise GateBFinalizationError(
            "source snapshot changed during finalization"
        ) from exc
    if verified_again.snapshot_manifest_sha256 != snapshot_sha:
        raise GateBFinalizationError("source snapshot identity changed during run")

    run_id = _run_id(snapshot_sha, model_sha)
    bundle_rows = [{
        "recording_id": item.path.parent.name,
        "finalization_id": item.finalization_id,
        "relative_path": item.path.relative_to(output).as_posix(),
        "manifest_sha256": item.manifest_sha256,
    } for item in published_by_recording.values()]
    bundle_rows.sort(key=lambda item: str(item["recording_id"]))
    run_manifest = {
        "artifact_kind": V04_GATE_B_RUN_ARTIFACT_KIND,
        "run_manifest_version": V04_GATE_B_RUN_MANIFEST_VERSION,
        "run_id": run_id,
        "source_snapshot_sha256": snapshot_sha,
        "source_inventory_sha256": source.source_inventory_sha256,
        "model_manifest_sha256": model_sha,
        "execution_id": execution_id,
        "execution_receipt_sha256": receipt_sha256,
        "recording_count": len(bundle_rows),
        "recording_ids": [item["recording_id"] for item in bundle_rows],
        "bundles": bundle_rows,
        "scope": {
            "camera_cv_only": True,
            "imu_and_annotation_payloads_unparsed": True,
            "sync_clock_window_spectrum_out_of_scope": True,
        },
    }
    run_manifest_bytes = _write_or_verify_run_json(
        output / RUN_MANIFEST_FILE, run_manifest
    )
    run_audit = {
        "artifact_kind": V04_GATE_B_RUN_ARTIFACT_KIND,
        "run_audit_version": V04_GATE_B_RUN_AUDIT_VERSION,
        "run_id": run_id,
        "recordings_expected": VIDIMU_V2_EXPECTED_RECORDING_COUNT,
        "recordings_finalized": len(bundle_rows),
        "recordings_failed": 0,
        "source_snapshot_verified_before_and_after": True,
        "model_manifest_verified": True,
        "execution_receipt_verified": True,
        "bundle_audits_passed": len(bundle_rows),
        "overall_verdict": "PASS",
    }
    run_audit_bytes = _write_or_verify_run_json(output / RUN_AUDIT_FILE, run_audit)
    marker = {
        "artifact_kind": V04_GATE_B_RUN_ARTIFACT_KIND,
        "run_success_version": V04_GATE_B_RUN_SUCCESS_VERSION,
        "run_id": run_id,
        "run_manifest_sha256": sha256_bytes(run_manifest_bytes),
        "run_audit_sha256": sha256_bytes(run_audit_bytes),
    }
    _write_run_json_exclusive(output / RUN_SUCCESS_FILE, marker)
    _fsync_directory(output)
    return FinalizedGateBRun(
        path=output,
        run_id=run_id,
        source_snapshot_sha256=snapshot_sha,
        model_manifest_sha256=model_sha,
        recording_count=len(bundle_rows),
        run_manifest_sha256=sha256_bytes(run_manifest_bytes),
        execution_id=execution_id,
    )


__all__ = [
    "RUN_AUDIT_FILE",
    "RUN_MANIFEST_FILE",
    "RUN_SUCCESS_FILE",
    "V04_GATE_B_RUN_ARTIFACT_KIND",
    "FinalizedGateBRun",
    "GateBFinalizationError",
    "GateBModelInputs",
    "build_recording_manifest",
    "finalize_vidimu_v04_gate_b",
]
