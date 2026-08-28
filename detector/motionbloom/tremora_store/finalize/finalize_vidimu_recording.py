"""Atomic one-record orchestration for VIDIMU PTS/CV finalization."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path, PurePosixPath

from ..cv.coordinate_mapping import CV_INPUT_HASH_VERSION, PIXEL_CONVENTION
from ..cv.offline_finalizer import (
    SELECTION_POLICY_ID,
    EstimatorProvenance,
    FinalizedFrameTables,
    OfflineFinalizationError,
    OfflineFrameEstimator,
    finalize_frames,
)
from ..cv.pose_frame_contract import validate_pose_frame_association
from ..decode.frame_identity import (
    ASSOCIATION_SCHEMA_VERSION,
    FINALIZATION_SCHEMA_VERSION,
    finalization_identity,
)
from ..decode.pts_decoder import (
    DecodeConfig,
    DecodedFrame,
    DecodedVideo,
    PTSDecoder,
    VerifiedSourceDecodeError,
)
from ..schemas import FINALIZATION_SORT_KEYS
from ._bundle_io import (
    FinalizationBundleError,
    FinalizationBundleWriter,
    read_json,
    safe_component,
)
from .audit_finalized_recording import (
    FINALIZATION_MANIFEST_VERSION,
    SUCCESS_MARKER_VERSION,
    audit_finalized_recording,
    build_finalization_audit,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_KINDS = frozenset({"SYNTHETIC_FIXTURE", "VIDIMU_PUBLIC"})
VALIDATION_GATES = frozenset({
    "GATE_A_SYNTHETIC",
    "GATE_A_REAL_VIDEO_PILOT",
    "GATE_B_VIDIMU",
})
VIDIMU_ARCHIVE_ROLES = frozenset({"DATASET_ARCHIVE", "VIDEO_ARCHIVE"})
_GATE_B_PREFLIGHT_TOKEN = object()


class RecordingFinalizationError(RuntimeError):
    """Raised when one recording cannot be finalized without substitution."""


def _safe_recording_id(value: object) -> str:
    try:
        return safe_component(value, field="recording_id")
    except FinalizationBundleError as exc:
        raise RecordingFinalizationError(
            "recording_id must be a safe single path component") from exc


def _sha256(value: object, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RecordingFinalizationError(
            f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecordingFinalizationError(f"{field} must be a non-empty string")
    return value


def _paired_imu_evidence(
    value: object,
    *,
    required: bool,
) -> list[dict[str, str]] | None:
    """Validate and canonicalize inventory-bound IMU provenance only."""

    if value is None:
        if required:
            raise RecordingFinalizationError(
                "Gate B requires paired IMU provenance from snapshot preflight")
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RecordingFinalizationError(
            "paired IMU provenance must be a sequence")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {
            "original_path", "role", "sha256",
        }:
            raise RecordingFinalizationError(
                f"paired IMU provenance entry {index} has invalid fields")
        original_path = item["original_path"]
        role = item["role"]
        sha256 = item["sha256"]
        if not isinstance(original_path, str):
            raise RecordingFinalizationError(
                "paired IMU original_path must be a string")
        parsed = PurePosixPath(original_path)
        if parsed.is_absolute() or not original_path or any(
            part in {"", ".", ".."} for part in parsed.parts
        ) or parsed.as_posix() != original_path or "\\" in original_path \
                or "\x00" in original_path:
            raise RecordingFinalizationError(
                "paired IMU original_path must be a safe relative path")
        if role != "IMU":
            raise RecordingFinalizationError(
                "paired IMU provenance role must be IMU")
        normalized_hash = _sha256(sha256, "paired IMU sha256")
        assert normalized_hash is not None
        normalized.append({
            "original_path": original_path,
            "role": role,
            "sha256": normalized_hash,
        })
    normalized.sort(key=lambda item: (
        item["original_path"], item["role"], item["sha256"],
    ))
    if required and not normalized:
        raise RecordingFinalizationError(
            "Gate B requires at least one paired IMU provenance entry")
    paths = [item["original_path"] for item in normalized]
    if len(paths) != len(set(paths)):
        raise RecordingFinalizationError(
            "paired IMU provenance contains duplicate original paths")
    return normalized


def _source_archive_evidence(
    value: object,
    *,
    required: bool,
) -> list[dict[str, str]] | None:
    """Validate the exact canonical archive topology established by preflight."""

    if value is None:
        if required:
            raise RecordingFinalizationError(
                "Gate B requires source archive provenance from snapshot preflight")
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RecordingFinalizationError(
            "source archive provenance must be a sequence")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {
            "original_path", "role", "sha256",
        }:
            raise RecordingFinalizationError(
                f"source archive provenance entry {index} has invalid fields")
        original_path = item["original_path"]
        if not isinstance(original_path, str):
            raise RecordingFinalizationError(
                "source archive original_path must be a string")
        parsed = PurePosixPath(original_path)
        if parsed.is_absolute() or not original_path or any(
            part in {"", ".", ".."} for part in parsed.parts
        ) or parsed.as_posix() != original_path or "\\" in original_path \
                or "\x00" in original_path:
            raise RecordingFinalizationError(
                "source archive original_path must be a safe relative path")
        role = item["role"]
        if not isinstance(role, str) or role not in VIDIMU_ARCHIVE_ROLES:
            raise RecordingFinalizationError(
                "source archive provenance role is invalid")
        normalized_hash = _sha256(
            item["sha256"], "source archive sha256")
        assert normalized_hash is not None
        normalized.append({
            "original_path": original_path,
            "role": role,
            "sha256": normalized_hash,
        })
    normalized.sort(key=lambda item: (
        item["original_path"], item["role"], item["sha256"],
    ))
    if required and {item["role"] for item in normalized} \
            != VIDIMU_ARCHIVE_ROLES:
        raise RecordingFinalizationError(
            "Gate B requires exactly one dataset and one video archive")
    paths = [item["original_path"] for item in normalized]
    roles = [item["role"] for item in normalized]
    hashes = [item["sha256"] for item in normalized]
    if len(normalized) != len(VIDIMU_ARCHIVE_ROLES) \
            or len(paths) != len(set(paths)) \
            or len(roles) != len(set(roles)) \
            or len(hashes) != len(set(hashes)):
        raise RecordingFinalizationError(
            "source archive provenance must have distinct paths, roles, and hashes")
    return normalized


@dataclass(frozen=True)
class RecordingProvenance:
    """Stable public/synthetic lineage recorded without local absolute paths."""

    dataset_id: str
    dataset_version: str
    recording_id: str
    source_kind: str
    source_original_path: str
    source_object_id: str
    materialization_date: str
    license_id: str
    license_record_sha256: str
    inventory_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "dataset_id", "dataset_version", "recording_id", "source_object_id",
            "license_id",
        ):
            _nonempty(getattr(self, field), field)
        _safe_recording_id(self.recording_id)
        if self.source_kind not in SOURCE_KINDS:
            raise RecordingFinalizationError(
                f"source_kind must be one of {sorted(SOURCE_KINDS)}")
        source_path = PurePosixPath(self.source_original_path)
        if source_path.is_absolute() or not self.source_original_path \
                or any(part in {"", ".", ".."} for part in source_path.parts):
            raise RecordingFinalizationError(
                "source_original_path must be a stable safe relative path")
        try:
            parsed_date = date.fromisoformat(self.materialization_date)
        except (TypeError, ValueError) as exc:
            raise RecordingFinalizationError(
                "materialization_date must use YYYY-MM-DD") from exc
        if parsed_date.isoformat() != self.materialization_date:
            raise RecordingFinalizationError(
                "materialization_date must use canonical YYYY-MM-DD")
        _sha256(self.license_record_sha256, "license_record_sha256")
        _sha256(
            self.inventory_manifest_sha256,
            "inventory_manifest_sha256",
            nullable=True,
        )
        if self.source_kind == "VIDIMU_PUBLIC" \
                and self.inventory_manifest_sha256 is None:
            raise RecordingFinalizationError(
                "public VIDIMU provenance requires its frozen inventory hash")
        if self.source_kind == "SYNTHETIC_FIXTURE" \
                and self.inventory_manifest_sha256 is not None:
            raise RecordingFinalizationError(
                "synthetic provenance cannot claim public archive evidence")

    def as_manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FinalizedRecording:
    path: Path
    finalization_id: str
    audit: dict[str, object]
    reused_existing: bool


@dataclass(frozen=True)
class _GateBVerifiedSourceDecodeOutcome:
    """Exact processing identity consumed before a trusted Gate-B rejection."""

    estimator_provenance: EstimatorProvenance
    decoder_version: str
    decoder_config: DecodeConfig


def _identity_inputs(
    *,
    source_video_sha256: str,
    decoder_version: str,
    decoder_config_sha256: str,
    estimator: EstimatorProvenance,
) -> dict[str, object]:
    return {
        "source_video_sha256": source_video_sha256,
        "decoder_version": decoder_version,
        "decoder_config_sha256": decoder_config_sha256,
        "model_id": estimator.model_id,
        "model_weights_sha256": estimator.model_weights_sha256,
        "preprocessing_config_sha256": estimator.preprocessing_config_sha256,
        "inference_environment_id": estimator.inference_environment_id,
        "association_schema_version": ASSOCIATION_SCHEMA_VERSION,
        "finalization_schema_version": FINALIZATION_SCHEMA_VERSION,
    }


def _presentation_origin(
    frames: tuple[DecodedFrame, ...],
) -> dict[str, int] | None:
    origins = [
        frame for frame in frames
        if frame.presentation_ordinal == 0
    ]
    if not origins:
        return None
    if len(origins) != 1:
        raise RecordingFinalizationError(
            "decoder produced an ambiguous presentation origin")
    frame = origins[0]
    return {
        "pts": int(frame.pts),
        "time_base_num": int(frame.time_base_num),
        "time_base_den": int(frame.time_base_den),
    }


def _validate_tables(
    tables: FinalizedFrameTables,
    *,
    finalization_id: str,
    provenance: RecordingProvenance,
    decoded: DecodedVideo,
    estimator: EstimatorProvenance,
    discontinuity_threshold_ns: int,
) -> None:
    validate_pose_frame_association(
        tables.video_frames,
        tables.cv_frame_results,
        tables.cv_detections,
        finalization_id=finalization_id,
        expected_recording_id=provenance.recording_id,
        expected_source_video_sha256=decoded.source_video_sha256,
        expected_decoder_version=decoded.decoder_version,
        expected_model_id=estimator.model_id,
        expected_model_weights_sha256=estimator.model_weights_sha256,
        expected_preprocessing_config_sha256=(
            estimator.preprocessing_config_sha256),
        expected_inference_environment_id=estimator.inference_environment_id,
        expected_discontinuity_threshold_ns=discontinuity_threshold_ns,
    )


def finalize_vidimu_recording(
    source_video_path: str | Path,
    output_root: str | Path,
    *,
    expected_source_video_sha256: str,
    provenance: RecordingProvenance,
    decoder: PTSDecoder,
    estimator: OfflineFrameEstimator,
    validation_gate: str = "GATE_A_SYNTHETIC",
    _gate_b_preflight_token: object | None = None,
    _paired_imu_assets: object = None,
    _source_archives: object = None,
) -> FinalizedRecording | _GateBVerifiedSourceDecodeOutcome:
    """Decode, infer, reconcile, hash, and atomically publish one recording.

    Local source paths are consumed but never persisted. The stable original
    path/object ID supplied by the frozen inventory is what enters provenance.
    """

    if validation_gate not in VALIDATION_GATES:
        raise RecordingFinalizationError(
            f"validation_gate must be one of {sorted(VALIDATION_GATES)}")
    expected_hash = _sha256(
        expected_source_video_sha256, "expected_source_video_sha256")
    assert expected_hash is not None
    if not isinstance(provenance, RecordingProvenance):
        raise RecordingFinalizationError(
            "provenance must be RecordingProvenance")
    recording_id = _safe_recording_id(provenance.recording_id)
    if not isinstance(decoder, PTSDecoder):
        raise RecordingFinalizationError("decoder must be PTSDecoder")
    estimator_provenance = estimator.provenance
    if not isinstance(estimator_provenance, EstimatorProvenance):
        raise RecordingFinalizationError(
            "estimator provenance must be EstimatorProvenance")
    if validation_gate == "GATE_A_SYNTHETIC" \
            and provenance.source_kind != "SYNTHETIC_FIXTURE":
        raise RecordingFinalizationError(
            "synthetic Gate A requires synthetic fixture provenance")
    if validation_gate == "GATE_A_REAL_VIDEO_PILOT" \
            and provenance.source_kind != "VIDIMU_PUBLIC":
        raise RecordingFinalizationError(
            "real-video Gate A requires public-source provenance")
    if validation_gate == "GATE_B_VIDIMU":
        if _gate_b_preflight_token is not _GATE_B_PREFLIGHT_TOKEN:
            raise RecordingFinalizationError(
                "Gate B must be run through complete snapshot preflight")
        if provenance.source_kind != "VIDIMU_PUBLIC":
            raise RecordingFinalizationError(
                "Gate B cannot be run with synthetic or substituted provenance")
        if type(estimator_provenance) is not EstimatorProvenance:
            raise RecordingFinalizationError(
                "Gate B requires exact estimator provenance")
        try:
            estimator_provenance = EstimatorProvenance(
                **asdict(estimator_provenance))
        except (TypeError, ValueError, OfflineFinalizationError) as exc:
            raise RecordingFinalizationError(
                "Gate B estimator provenance is invalid") from exc
        paired_imu_assets = _paired_imu_evidence(
            _paired_imu_assets, required=True)
        source_archives = _source_archive_evidence(
            _source_archives, required=True)
        if type(decoder) is not PTSDecoder \
                or type(decoder.config) is not DecodeConfig:
            raise RecordingFinalizationError(
                "Gate B requires an exact trusted PTSDecoder and DecodeConfig")
        try:
            gate_b_decoder_config = DecodeConfig(**asdict(decoder.config))
        except (TypeError, ValueError) as exc:
            raise RecordingFinalizationError(
                "Gate B decoder config is invalid") from exc
        gate_b_decoder_version = _nonempty(
            decoder.decoder_version, "decoder_version")
    else:
        if _gate_b_preflight_token is not None \
                or _paired_imu_assets is not None \
                or _source_archives is not None:
            raise RecordingFinalizationError(
                "Gate-B preflight evidence cannot be attached to another gate")
        paired_imu_assets = None
        source_archives = None
        gate_b_decoder_config = None
        gate_b_decoder_version = None

    try:
        decoded = decoder.decode(
            source_video_path,
            expected_source_sha256=expected_hash,
        )
    except VerifiedSourceDecodeError:
        if validation_gate == "GATE_B_VIDIMU" \
                and _gate_b_preflight_token is _GATE_B_PREFLIGHT_TOKEN:
            if gate_b_decoder_config is None \
                    or gate_b_decoder_version is None:
                raise RecordingFinalizationError(
                    "Gate B decoder identity was not resolved")
            return _GateBVerifiedSourceDecodeOutcome(
                estimator_provenance=estimator_provenance,
                decoder_version=gate_b_decoder_version,
                decoder_config=gate_b_decoder_config,
            )
        raise
    if validation_gate == "GATE_B_VIDIMU":
        if gate_b_decoder_config is None or gate_b_decoder_version is None:
            raise RecordingFinalizationError(
                "Gate B decoder identity was not resolved")
        if decoded.decoder_version != gate_b_decoder_version \
                or decoded.decoder_config_sha256 \
                != gate_b_decoder_config.sha256:
            raise RecordingFinalizationError(
                "Gate B decoded identity disagrees with trusted decoder inputs")
    identity_inputs = _identity_inputs(
        source_video_sha256=decoded.source_video_sha256,
        decoder_version=decoded.decoder_version,
        decoder_config_sha256=decoded.decoder_config_sha256,
        estimator=estimator_provenance,
    )
    finalization_id = finalization_identity(identity_inputs)
    bundle_path = (
        Path(output_root) / recording_id / finalization_id)
    if bundle_path.exists() or bundle_path.is_symlink():
        try:
            audit = audit_finalized_recording(bundle_path)
        except Exception as exc:
            raise RecordingFinalizationError(
                "existing finalization identity is incomplete or invalid") from exc
        stored_manifest, _ = read_json(
            bundle_path / "finalization_manifest.json")
        stored_source = stored_manifest.get("source")
        if not isinstance(stored_source, dict) or any(
            stored_source.get(key) != value
            for key, value in provenance.as_manifest().items()
        ) or stored_manifest.get("validation_gate") != validation_gate \
                or (
                    validation_gate == "GATE_B_VIDIMU"
                    and (
                        stored_source.get("paired_imu_assets")
                        != paired_imu_assets
                        or stored_source.get("source_archives")
                        != source_archives
                    )
                ):
            raise RecordingFinalizationError(
                "existing finalization identity has different provenance or gate")
        return FinalizedRecording(
            path=bundle_path.resolve(strict=True),
            finalization_id=finalization_id,
            audit=audit,
            reused_existing=True,
        )

    tables = finalize_frames(
        decoded.frames,
        dataset_id=provenance.dataset_id,
        recording_id=recording_id,
        finalization_id=finalization_id,
        estimator=estimator,
    )
    _validate_tables(
        tables,
        finalization_id=finalization_id,
        provenance=provenance,
        decoded=decoded,
        estimator=estimator_provenance,
        discontinuity_threshold_ns=decoder.config.discontinuity_threshold_ns,
    )

    writer: FinalizationBundleWriter | None = None
    try:
        writer = FinalizationBundleWriter(
            output_root,
            recording_id=recording_id,
            finalization_id=finalization_id,
        )
        sorted_tables = {
            name: writer.write_table(
                name,
                table,
                sort_keys=FINALIZATION_SORT_KEYS[name],
            )
            for name, table in tables.as_dict().items()
        }
        source_manifest: dict[str, object] = {
            **provenance.as_manifest(),
            "source_video_bytes": decoded.source_bytes,
            "source_video_sha256": decoded.source_video_sha256,
            "stream_index": decoded.stream_index,
            "source_hash_verified_during_pinned_decode": True,
        }
        if paired_imu_assets is not None:
            source_manifest["paired_imu_assets"] = paired_imu_assets
        if source_archives is not None:
            source_manifest["source_archives"] = source_archives
        manifest: dict[str, object] = {
            "manifest_version": FINALIZATION_MANIFEST_VERSION,
            "finalization_id": finalization_id,
            "recording_id": recording_id,
            "dataset_id": provenance.dataset_id,
            "finalization_schema_version": FINALIZATION_SCHEMA_VERSION,
            "association_schema_version": ASSOCIATION_SCHEMA_VERSION,
            "validation_gate": validation_gate,
            "identity_inputs": identity_inputs,
            "source": source_manifest,
            "decoder": {
                "decoder_version": decoded.decoder_version,
                "decoder_config": asdict(decoder.config),
                "decoder_config_sha256": decoded.decoder_config_sha256,
                "decode_ordinal_semantics": "decoder_emission_ordinal",
                "timestamp_authority": "raw_source_pts_and_time_base",
                "presentation_origin": _presentation_origin(decoded.frames),
            },
            "estimator": {
                **asdict(estimator_provenance),
                "selection_policy_id": SELECTION_POLICY_ID,
                "inference_call_contract": "once_per_preprocessed_decoded_frame",
            },
            "coordinate_contract": {
                "persisted_landmark_space": "DISPLAY_PIXEL",
                "pixel_convention": PIXEL_CONVENTION,
                "cv_input_hash_version": CV_INPUT_HASH_VERSION,
                "container_rotation_applied_once": True,
                "preview_mirroring_applied": False,
            },
            "determinism_contract": {
                "canonical_runtime_ms": None,
                "parquet_runtime_profiling_excluded": True,
                "cpu_software_decode": True,
            },
            "scope": {
                "video_imu_synchronization": False,
                "canonical_cross_modal_clock": False,
                "frame_imu_range_index": False,
                "window_generation": False,
                "performance_benchmark": False,
            },
            "tables": writer.artifacts,
        }
        manifest_artifact = writer.write_json(
            "finalization_manifest.json", manifest)
        manifest_sha = str(manifest_artifact["sha256"])
        audit = build_finalization_audit(
            manifest=manifest,
            manifest_sha256=manifest_sha,
            tables=sorted_tables,
        )
        audit_artifact = writer.write_json("finalization_audit.json", audit)
        writer.write_json("_SUCCESS", {
            "audit_sha256": audit_artifact["sha256"],
            "finalization_id": finalization_id,
            "manifest_sha256": manifest_sha,
            "success_marker_version": SUCCESS_MARKER_VERSION,
        })
        verified_audit = audit_finalized_recording(
            writer.staging_dir,
            _expected_finalization_id=finalization_id,
            _expected_recording_id=recording_id,
        )
        published = writer.publish()
    except BaseException as exc:
        if writer is not None:
            writer.abort()
        if isinstance(exc, (FinalizationBundleError, ValueError, OSError)):
            raise RecordingFinalizationError(
                "recording finalization failed before a valid publication") from exc
        raise
    return FinalizedRecording(
        path=published,
        finalization_id=finalization_id,
        audit=verified_audit,
        reused_existing=False,
    )


__all__ = [
    "VIDIMU_ARCHIVE_ROLES",
    "FinalizedRecording",
    "RecordingFinalizationError",
    "RecordingProvenance",
    "finalize_vidimu_recording",
]
