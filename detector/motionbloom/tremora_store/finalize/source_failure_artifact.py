"""Atomic, immutable evidence for a hash-verified Gate-B decode failure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from ..cv.offline_finalizer import EstimatorProvenance, OfflineFinalizationError
from ..decode.frame_identity import (
    ASSOCIATION_SCHEMA_VERSION,
    FINALIZATION_SCHEMA_VERSION,
    finalization_identity,
)
from ..decode.pts_decoder import DecodeConfig
from ._bundle_io import (
    SOURCE_FAILURE_FILES,
    FinalizationBundleError,
    SourceFailureBundleWriter,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
)
from .finalize_vidimu_recording import (
    _GATE_B_PREFLIGHT_TOKEN,
    VIDIMU_ARCHIVE_ROLES,
    RecordingFinalizationError,
    RecordingProvenance,
)

SOURCE_FAILURE_SCHEMA_VERSION = "tremora-source-decode-failure-1.0.0"
SOURCE_FAILURE_MANIFEST_VERSION = 1
SOURCE_FAILURE_AUDIT_VERSION = 1
FAILURE_MARKER_VERSION = 1
SOURCE_FAILURE_STAGE = "SOURCE_DECODE"
SOURCE_FAILURE_CATEGORY = "SOURCE_MEDIA_DECODE_FAILURE"
SOURCE_FAILURE_DETAIL_CODE = "PTS_DECODER_REJECTED_HASH_VERIFIED_SOURCE"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_SCOPE = {
    "video_imu_synchronization": False,
    "canonical_cross_modal_clock": False,
    "frame_imu_range_index": False,
    "window_generation": False,
    "performance_benchmark": False,
}
_IDENTITY_FIELDS = frozenset({
    "source_video_sha256",
    "decoder_version",
    "decoder_config_sha256",
    "model_id",
    "model_weights_sha256",
    "preprocessing_config_sha256",
    "inference_environment_id",
    "association_schema_version",
    "finalization_schema_version",
})
_MANIFEST_FIELDS = frozenset({
    "manifest_version",
    "failure_schema_version",
    "failure_id",
    "intended_finalization_id",
    "recording_id",
    "dataset_id",
    "validation_gate",
    "identity_inputs",
    "source",
    "decoder",
    "estimator",
    "failure",
    "scope",
})
_PROVENANCE_FIELDS = frozenset({
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
})
_SOURCE_FIELDS = _PROVENANCE_FIELDS | frozenset({
    "source_video_sha256",
    "source_video_bytes",
    "source_hash_verified_during_snapshot_preflight",
    "source_hash_reverified_after_decode_failure",
    "paired_imu_assets",
    "source_archives",
})
_AUDIT_COUNTER_FIELDS = (
    "decoded_frame_count",
    "frames_with_valid_pts",
    "frames_with_duplicate_pts",
    "frames_with_missing_pts",
    "frames_with_nonmonotonic_pts",
    "timestamp_discontinuities",
    "decoded_corrupt_frames",
    "cv_frame_result_count",
    "frames_with_detection",
    "frames_without_detection",
    "inference_failures",
    "detection_row_count",
    "orphan_frame_results",
    "orphan_detections",
    "duplicate_frame_ids",
    "duplicate_detection_ids",
    "missing_frame_results",
    "coordinate_transform_failures",
    "source_hash_mismatches",
    "model_hash_mismatches",
)


@dataclass(frozen=True)
class SourceFailureArtifact:
    """One published, strictly audited Gate-B source-decode outcome."""

    path: Path
    failure_id: str
    intended_finalization_id: str
    audit: dict[str, object]
    reused_existing: bool


def _exact_fields(
    value: object,
    expected: frozenset[str],
    *,
    field: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise FinalizationBundleError(f"{field} fields are not the frozen contract")
    return value


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FinalizationBundleError(f"{field} must be a non-empty string")
    return value


def _exact_version(value: object, expected: int, *, field: str) -> int:
    if type(value) is not int or value != expected:
        raise FinalizationBundleError(f"{field} version is unsupported")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in _SHA256_CHARACTERS for character in value
    ):
        raise FinalizationBundleError(f"{field} must be a lowercase SHA-256")
    return value


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FinalizationBundleError(f"{field} must be a non-negative integer")
    return value


def _safe_relative_path(value: object, *, field: str) -> str:
    text = _nonempty(value, field=field)
    parsed = PurePosixPath(text)
    if parsed.is_absolute() or any(
        part in {"", ".", ".."} for part in parsed.parts
    ) or parsed.as_posix() != text or "\\" in text or "\x00" in text:
        raise FinalizationBundleError(f"{field} must be a safe relative path")
    return text


def _canonical_assets(
    value: object,
    *,
    roles: frozenset[str],
    exact_count: int | None,
    field: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise FinalizationBundleError(f"{field} must be a nonempty list")
    normalized: list[dict[str, str]] = []
    for item in value:
        asset = _exact_fields(
            item,
            frozenset({"original_path", "role", "sha256"}),
            field=f"{field} asset",
        )
        role = asset["role"]
        if not isinstance(role, str) or role not in roles:
            raise FinalizationBundleError(f"{field} role is invalid")
        normalized.append({
            "original_path": _safe_relative_path(
                asset["original_path"], field=f"{field} original_path"),
            "role": role,
            "sha256": _sha256(asset["sha256"], field=f"{field} sha256"),
        })
    canonical = sorted(normalized, key=lambda item: (
        item["original_path"], item["role"], item["sha256"],
    ))
    paths = [item["original_path"] for item in normalized]
    observed_roles = [item["role"] for item in normalized]
    hashes = [item["sha256"] for item in normalized]
    if normalized != canonical \
            or len(paths) != len(set(paths)) \
            or (exact_count is not None and len(normalized) != exact_count) \
            or (exact_count is not None and set(observed_roles) != roles) \
            or (exact_count is not None and len(hashes) != len(set(hashes))):
        raise FinalizationBundleError(
            f"{field} is not canonical, exact, and unique")
    return normalized


def _validate_manifest_semantics(manifest: dict[str, object]) -> None:
    _exact_fields(manifest, _MANIFEST_FIELDS, field="source failure manifest")
    _exact_version(
        manifest["manifest_version"],
        SOURCE_FAILURE_MANIFEST_VERSION,
        field="source failure manifest",
    )
    if manifest["failure_schema_version"] != SOURCE_FAILURE_SCHEMA_VERSION:
        raise FinalizationBundleError("source failure manifest version is unsupported")
    failure_id = _sha256(manifest["failure_id"], field="failure_id")
    recording_id = _nonempty(manifest["recording_id"], field="recording_id")
    dataset_id = _nonempty(manifest["dataset_id"], field="dataset_id")
    if manifest["validation_gate"] != "GATE_B_VIDIMU":
        raise FinalizationBundleError("source failure evidence is Gate-B-only")

    identity = _exact_fields(
        manifest["identity_inputs"], _IDENTITY_FIELDS, field="identity_inputs")
    if identity["association_schema_version"] != ASSOCIATION_SCHEMA_VERSION \
            or identity["finalization_schema_version"] \
            != FINALIZATION_SCHEMA_VERSION:
        raise FinalizationBundleError(
            "source failure processing schema identity is unsupported")
    intended_finalization_id = _sha256(
        manifest["intended_finalization_id"], field="intended_finalization_id")
    try:
        recomputed_finalization_id = finalization_identity(identity)
    except ValueError as exc:
        raise FinalizationBundleError(
            "intended finalization identity inputs are invalid") from exc
    if recomputed_finalization_id != intended_finalization_id:
        raise FinalizationBundleError("intended finalization identity does not recompute")

    source = _exact_fields(
        manifest["source"], _SOURCE_FIELDS, field="source provenance")
    provenance_value = {
        field: source[field] for field in _PROVENANCE_FIELDS
    }
    try:
        provenance = RecordingProvenance(**provenance_value)
    except (TypeError, ValueError, RecordingFinalizationError) as exc:
        raise FinalizationBundleError("source provenance is invalid") from exc
    if provenance.source_kind != "VIDIMU_PUBLIC" \
            or provenance.recording_id != recording_id \
            or provenance.dataset_id != dataset_id:
        raise FinalizationBundleError(
            "source provenance disagrees with the failure identity")
    source_original_path = _safe_relative_path(
        source["source_original_path"], field="source original_path")
    source_video_sha256 = _sha256(
        source["source_video_sha256"], field="source_video_sha256")
    _nonnegative_integer(source["source_video_bytes"], field="source_video_bytes")
    if source_video_sha256 != identity["source_video_sha256"]:
        raise FinalizationBundleError(
            "source video hash disagrees with processing identity")
    if source["source_hash_verified_during_snapshot_preflight"] is not True \
            or source["source_hash_reverified_after_decode_failure"] is not True:
        raise FinalizationBundleError(
            "source failure lacks both pinned source-hash verifications")
    imu_assets = _canonical_assets(
        source["paired_imu_assets"],
        roles=frozenset({"IMU"}),
        exact_count=None,
        field="paired IMU evidence",
    )
    archives = _canonical_assets(
        source["source_archives"],
        roles=VIDIMU_ARCHIVE_ROLES,
        exact_count=len(VIDIMU_ARCHIVE_ROLES),
        field="source archive evidence",
    )
    source_paths = {
        source_original_path,
        *(item["original_path"] for item in imu_assets),
        *(item["original_path"] for item in archives),
    }
    if len(source_paths) != 1 + len(imu_assets) + len(archives):
        raise FinalizationBundleError("source failure evidence aliases source assets")

    decoder = _exact_fields(
        manifest["decoder"],
        frozenset({
            "decoder_version", "decoder_config", "decoder_config_sha256",
        }),
        field="decoder",
    )
    decoder_version = _nonempty(
        decoder["decoder_version"], field="decoder.decoder_version")
    config_value = decoder["decoder_config"]
    if not isinstance(config_value, dict) or set(config_value) != set(
        asdict(DecodeConfig())
    ):
        raise FinalizationBundleError("decoder config fields are invalid")
    try:
        config = DecodeConfig(**config_value)
    except (TypeError, ValueError) as exc:
        raise FinalizationBundleError("decoder config is invalid") from exc
    decoder_config_sha256 = _sha256(
        decoder["decoder_config_sha256"], field="decoder config sha256")
    if decoder_version != identity["decoder_version"] \
            or decoder_config_sha256 != config.sha256 \
            or decoder_config_sha256 != identity["decoder_config_sha256"]:
        raise FinalizationBundleError(
            "decoder evidence disagrees with processing identity")

    estimator = _exact_fields(
        manifest["estimator"],
        frozenset({
            "model_id",
            "model_weights_sha256",
            "preprocessing_config_sha256",
            "inference_environment_id",
        }),
        field="estimator",
    )
    try:
        estimator_provenance = EstimatorProvenance(**estimator)
    except (TypeError, ValueError, OfflineFinalizationError) as exc:
        raise FinalizationBundleError("estimator provenance is invalid") from exc
    for field, value in asdict(estimator_provenance).items():
        if identity[field] != value:
            raise FinalizationBundleError(
                "estimator evidence disagrees with processing identity")

    if manifest["failure"] != {
        "stage": SOURCE_FAILURE_STAGE,
        "category": SOURCE_FAILURE_CATEGORY,
        "detail_code": SOURCE_FAILURE_DETAIL_CODE,
    }:
        raise FinalizationBundleError("source failure classification is unsupported")
    scope = _exact_fields(
        manifest["scope"], frozenset(_SCOPE), field="source failure scope")
    if any(scope[field] is not False for field in _SCOPE):
        raise FinalizationBundleError("source failure scope was expanded")
    if failure_id != intended_finalization_id:
        raise FinalizationBundleError(
            "failure and intended finalization identities disagree")


def build_source_failure_audit(
    *,
    manifest: dict[str, object],
    manifest_sha256: str,
) -> dict[str, object]:
    """Build deterministic per-record audit evidence after semantic validation."""

    _validate_manifest_semantics(manifest)
    zero_counts = {field: 0 for field in _AUDIT_COUNTER_FIELDS}
    return {
        "audit_version": SOURCE_FAILURE_AUDIT_VERSION,
        "failure_schema_version": SOURCE_FAILURE_SCHEMA_VERSION,
        "failure_id": manifest["failure_id"],
        "intended_finalization_id": manifest["intended_finalization_id"],
        "recording_id": manifest["recording_id"],
        "validation_gate": manifest["validation_gate"],
        "recording_outcome": "SOURCE_DECODE_FAILURE",
        "failure_stage": SOURCE_FAILURE_STAGE,
        "failure_category": SOURCE_FAILURE_CATEGORY,
        "failure_detail_code": SOURCE_FAILURE_DETAIL_CODE,
        "inventory_record_count": 1,
        "source_assets_present": "VERIFIED_DURING_COMPLETE_GATE_B_PREFLIGHT",
        "source_assets_embedded_in_bundle": 0,
        "source_hashes_verified": (
            "VERIFIED_BEFORE_AND_AFTER_SOURCE_DECODE_FAILURE"
        ),
        "videos_opened": 1,
        "videos_failed": 1,
        **zero_counts,
        "inference_status_counts": {},
        "artifact_hashes": {
            "source_failure_manifest": manifest_sha256,
        },
        "manifest_sha256": manifest_sha256,
        "deterministic_replay_status": "STORED_ARTIFACT_BYTES_VERIFIED",
        "overall_verdict": "PASS",
    }


def audit_source_failure_artifact(
    bundle_path: str | Path,
    *,
    _expected_failure_id: str | None = None,
    _expected_recording_id: str | None = None,
) -> dict[str, object]:
    """Strictly verify one canonical, source-level Gate-B failure outcome."""

    bundle = Path(bundle_path)
    if bundle.is_symlink() or not bundle.is_dir():
        raise FinalizationBundleError(
            "source failure artifact path must be a real directory")
    failure_id = _expected_failure_id
    if failure_id is None:
        failure_id = bundle.name
    failure_id = _sha256(failure_id, field="failure_id")
    present = {item.name for item in bundle.iterdir()}
    if present != SOURCE_FAILURE_FILES:
        raise FinalizationBundleError(
            "source failure artifact inventory is not exact")
    for filename in present:
        path = bundle / filename
        if path.is_symlink() or not path.is_file():
            raise FinalizationBundleError(
                "source failure artifacts must be regular files")

    manifest, manifest_bytes = read_json(
        bundle / "source_failure_manifest.json")
    if manifest.get("failure_id") != failure_id:
        raise FinalizationBundleError(
            "source failure directory and manifest identities disagree")
    if _expected_recording_id is not None \
            and manifest.get("recording_id") != _expected_recording_id:
        raise FinalizationBundleError(
            "source failure recording identity disagrees")
    manifest_sha256 = sha256_bytes(manifest_bytes)
    expected_audit = build_source_failure_audit(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    stored_audit, audit_bytes = read_json(
        bundle / "source_failure_audit.json")
    _exact_version(
        stored_audit.get("audit_version"),
        SOURCE_FAILURE_AUDIT_VERSION,
        field="source failure audit",
    )
    if audit_bytes != canonical_json_bytes(expected_audit):
        raise FinalizationBundleError("source failure audit does not recompute")
    marker, marker_bytes = read_json(bundle / "_FAILURE")
    _exact_version(
        marker.get("failure_marker_version"),
        FAILURE_MARKER_VERSION,
        field="source failure marker",
    )
    expected_marker = {
        "audit_sha256": sha256_bytes(audit_bytes),
        "failure_id": failure_id,
        "failure_marker_version": FAILURE_MARKER_VERSION,
        "manifest_sha256": manifest_sha256,
    }
    if marker_bytes != canonical_json_bytes(expected_marker):
        raise FinalizationBundleError("_FAILURE binding is invalid")
    return expected_audit


def publish_source_decode_failure(
    output_root: str | Path,
    *,
    provenance: RecordingProvenance,
    expected_source_video_sha256: str,
    source_video_bytes: int,
    decoder_version: str,
    decoder_config: DecodeConfig,
    estimator_provenance: EstimatorProvenance,
    paired_imu_assets: Sequence[Mapping[str, str]],
    source_archives: Sequence[Mapping[str, str]],
    _gate_b_preflight_token: object | None,
) -> SourceFailureArtifact:
    """Publish stable evidence only after snapshot orchestration revalidates bytes."""

    if _gate_b_preflight_token is not _GATE_B_PREFLIGHT_TOKEN:
        raise RecordingFinalizationError(
            "source failure evidence requires complete Gate-B preflight")
    if not isinstance(provenance, RecordingProvenance) \
            or provenance.source_kind != "VIDIMU_PUBLIC":
        raise RecordingFinalizationError(
            "source failure evidence requires public VIDIMU provenance")
    try:
        expected_sha256 = _sha256(
            expected_source_video_sha256, field="source video sha256")
        source_video_bytes = _nonnegative_integer(
            source_video_bytes, field="source video bytes")
        if not isinstance(decoder_config, DecodeConfig):
            raise FinalizationBundleError("decoder config must be DecodeConfig")
        decoder_version = _nonempty(
            decoder_version, field="decoder version")
        if not isinstance(estimator_provenance, EstimatorProvenance):
            raise FinalizationBundleError(
                "estimator provenance must be EstimatorProvenance")
        normalized_imu = _canonical_assets(
            [dict(item) for item in paired_imu_assets],
            roles=frozenset({"IMU"}),
            exact_count=None,
            field="paired IMU evidence",
        )
        normalized_archives = _canonical_assets(
            [dict(item) for item in source_archives],
            roles=VIDIMU_ARCHIVE_ROLES,
            exact_count=len(VIDIMU_ARCHIVE_ROLES),
            field="source archive evidence",
        )
        identity_inputs: dict[str, object] = {
            "source_video_sha256": expected_sha256,
            "decoder_version": decoder_version,
            "decoder_config_sha256": decoder_config.sha256,
            **asdict(estimator_provenance),
            "association_schema_version": (
                ASSOCIATION_SCHEMA_VERSION
            ),
            "finalization_schema_version": FINALIZATION_SCHEMA_VERSION,
        }
        intended_finalization_id = finalization_identity(identity_inputs)
        source = {
            **provenance.as_manifest(),
            "source_video_sha256": expected_sha256,
            "source_video_bytes": source_video_bytes,
            "source_hash_verified_during_snapshot_preflight": True,
            "source_hash_reverified_after_decode_failure": True,
            "paired_imu_assets": normalized_imu,
            "source_archives": normalized_archives,
        }
        semantic_manifest: dict[str, object] = {
            "failure_schema_version": SOURCE_FAILURE_SCHEMA_VERSION,
            "intended_finalization_id": intended_finalization_id,
            "recording_id": provenance.recording_id,
            "dataset_id": provenance.dataset_id,
            "validation_gate": "GATE_B_VIDIMU",
            "identity_inputs": identity_inputs,
            "source": source,
            "decoder": {
                "decoder_version": decoder_version,
                "decoder_config": asdict(decoder_config),
                "decoder_config_sha256": decoder_config.sha256,
            },
            "estimator": asdict(estimator_provenance),
            "failure": {
                "stage": SOURCE_FAILURE_STAGE,
                "category": SOURCE_FAILURE_CATEGORY,
                "detail_code": SOURCE_FAILURE_DETAIL_CODE,
            },
            "scope": dict(_SCOPE),
        }
        failure_id = intended_finalization_id
        manifest = {
            "manifest_version": SOURCE_FAILURE_MANIFEST_VERSION,
            "failure_id": failure_id,
            **semantic_manifest,
        }
        _validate_manifest_semantics(manifest)
    except (FinalizationBundleError, TypeError, ValueError) as exc:
        raise RecordingFinalizationError(
            "source failure inputs do not satisfy the frozen contract") from exc

    final_path = (
        Path(output_root)
        / provenance.recording_id
        / failure_id
    )
    if final_path.exists() or final_path.is_symlink():
        try:
            audit = audit_source_failure_artifact(
                final_path,
                _expected_failure_id=failure_id,
                _expected_recording_id=provenance.recording_id,
            )
            stored_manifest, _ = read_json(
                final_path / "source_failure_manifest.json")
        except Exception as exc:
            raise RecordingFinalizationError(
                "existing source failure identity is incomplete or invalid") from exc
        if stored_manifest != manifest:
            raise RecordingFinalizationError(
                "existing source failure identity has different evidence")
        return SourceFailureArtifact(
            path=final_path.resolve(strict=True),
            failure_id=failure_id,
            intended_finalization_id=intended_finalization_id,
            audit=audit,
            reused_existing=True,
        )

    writer: SourceFailureBundleWriter | None = None
    try:
        writer = SourceFailureBundleWriter(
            output_root,
            recording_id=provenance.recording_id,
            failure_id=failure_id,
        )
        manifest_artifact = writer.write_json(
            "source_failure_manifest.json", manifest)
        manifest_sha256 = str(manifest_artifact["sha256"])
        audit = build_source_failure_audit(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
        )
        audit_artifact = writer.write_json(
            "source_failure_audit.json", audit)
        writer.write_json("_FAILURE", {
            "audit_sha256": audit_artifact["sha256"],
            "failure_id": failure_id,
            "failure_marker_version": FAILURE_MARKER_VERSION,
            "manifest_sha256": manifest_sha256,
        })
        verified_audit = audit_source_failure_artifact(
            writer.staging_dir,
            _expected_failure_id=failure_id,
            _expected_recording_id=provenance.recording_id,
        )
        published = writer.publish()
    except BaseException as exc:
        if writer is not None:
            writer.abort()
        if isinstance(exc, (FinalizationBundleError, ValueError, OSError)):
            raise RecordingFinalizationError(
                "source failure publication failed before an atomic outcome"
            ) from exc
        raise
    return SourceFailureArtifact(
        path=published,
        failure_id=failure_id,
        intended_finalization_id=intended_finalization_id,
        audit=verified_audit,
        reused_existing=False,
    )


__all__ = [
    "FAILURE_MARKER_VERSION",
    "SOURCE_FAILURE_AUDIT_VERSION",
    "SOURCE_FAILURE_CATEGORY",
    "SOURCE_FAILURE_DETAIL_CODE",
    "SOURCE_FAILURE_MANIFEST_VERSION",
    "SOURCE_FAILURE_SCHEMA_VERSION",
    "SOURCE_FAILURE_STAGE",
    "SourceFailureArtifact",
    "audit_source_failure_artifact",
    "build_source_failure_audit",
    "publish_source_decode_failure",
]
