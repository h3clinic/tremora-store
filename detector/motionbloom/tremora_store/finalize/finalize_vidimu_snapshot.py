"""Fail-closed Gate-B orchestration over a complete frozen VIDIMU inventory."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from ..cv.offline_finalizer import OfflineFrameEstimator
from ..decode.pts_decoder import DecodeConfig, PTSDecoder
from .finalize_vidimu_recording import (
    _GATE_B_PREFLIGHT_TOKEN,
    VIDIMU_ARCHIVE_ROLES,
    FinalizedRecording,
    RecordingFinalizationError,
    RecordingProvenance,
    _GateBVerifiedSourceDecodeOutcome,
    finalize_vidimu_recording,
)
from .source_failure_artifact import (
    SourceFailureArtifact,
    publish_source_decode_failure,
)

VIDIMU_INVENTORY_SCHEMA_VERSION = "tremora-vidimu-frozen-inventory-1.0.0"
_MAX_INVENTORY_BYTES = 16 * 1024 * 1024
_INVENTORY_ASSET_FIELDS = {"original_path", "role", "sha256"}
_INVENTORY_RECORD_FIELDS = {"imu_assets", "recording_id", "video"}


@dataclass(frozen=True)
class FrozenSourceAsset:
    """One local materialization bound to its inventory path and SHA-256."""

    original_path: str
    local_path: Path
    sha256: str
    role: str


@dataclass(frozen=True)
class VidimuSnapshotRecord:
    provenance: RecordingProvenance
    video: FrozenSourceAsset
    imu_assets: tuple[FrozenSourceAsset, ...]


@dataclass(frozen=True)
class VidimuSnapshotInputs:
    """All immutable inputs required before any Gate-B record may publish."""

    records: tuple[VidimuSnapshotRecord, ...]
    expected_recording_ids: tuple[str, ...]
    archive_assets: tuple[FrozenSourceAsset, ...]
    inventory_manifest_path: Path
    inventory_manifest_sha256: str
    license_record_path: Path
    license_record_sha256: str


def _verified_sha256(
    path: Path,
    expected_sha256: str,
    *,
    capture_limit: int | None = None,
) -> tuple[bytes | None, int]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 \
            or any(value not in "0123456789abcdef" for value in expected_sha256):
        raise RecordingFinalizationError(
            "snapshot asset hash must be a lowercase SHA-256")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RecordingFinalizationError(
            "Gate-B source verification requires O_NOFOLLOW and O_NONBLOCK")
    flags = (
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RecordingFinalizationError(
            f"required frozen source asset is unavailable: {path.name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RecordingFinalizationError(
                "frozen source asset must be a regular file")
        if capture_limit is not None and before.st_size > capture_limit:
            raise RecordingFinalizationError(
                "frozen inventory manifest exceeds its size limit")
        digest = hashlib.sha256()
        captured = bytearray() if capture_limit is not None else None
        offset = 0
        while offset < before.st_size:
            payload = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset)
            if not payload:
                raise RecordingFinalizationError(
                    "frozen source asset ended during verification")
            digest.update(payload)
            if captured is not None:
                captured.extend(payload)
            offset += len(payload)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        )
        if identity_before != identity_after \
                or digest.hexdigest() != expected_sha256:
            raise RecordingFinalizationError(
                f"frozen source asset hash mismatch: {path.name}")
        return (None if captured is None else bytes(captured), before.st_size)
    finally:
        os.close(descriptor)


def _safe_original_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise RecordingFinalizationError(f"{field} must be a string")
    original = PurePosixPath(value)
    if original.is_absolute() or not value or any(
        part in {"", ".", ".."} for part in original.parts
    ) or original.as_posix() != value or "\\" in value or "\x00" in value:
        raise RecordingFinalizationError(
            f"{field} must be a stable relative inventory path")
    return value


def _validate_asset(asset: FrozenSourceAsset, *, role: str | None = None) -> int:
    if not isinstance(asset, FrozenSourceAsset):
        raise RecordingFinalizationError("snapshot contains an invalid asset entry")
    _safe_original_path(asset.original_path, field="asset original_path")
    if role is not None and asset.role != role:
        raise RecordingFinalizationError(
            f"expected {role} asset, received {asset.role!r}")
    _payload, size = _verified_sha256(Path(asset.local_path), asset.sha256)
    return size


def _asset_evidence(asset: FrozenSourceAsset) -> dict[str, str]:
    return {
        "original_path": asset.original_path,
        "role": asset.role,
        "sha256": asset.sha256,
    }


def _canonical_archive_assets(
    value: object,
) -> tuple[dict[str, str], ...]:
    if not isinstance(value, tuple) or not value:
        raise RecordingFinalizationError(
            "Gate B archive_assets must be a nonempty tuple")
    evidence: list[dict[str, str]] = []
    for asset in value:
        _validate_asset(asset)
        if not isinstance(asset.role, str) \
                or asset.role not in VIDIMU_ARCHIVE_ROLES:
            raise RecordingFinalizationError(
                "Gate B archive role must be DATASET_ARCHIVE or VIDEO_ARCHIVE")
        evidence.append(_asset_evidence(asset))
    evidence.sort(key=lambda item: (
        item["original_path"], item["role"], item["sha256"],
    ))
    paths = [item["original_path"] for item in evidence]
    roles = [item["role"] for item in evidence]
    hashes = [item["sha256"] for item in evidence]
    if set(roles) != VIDIMU_ARCHIVE_ROLES \
            or len(evidence) != len(VIDIMU_ARCHIVE_ROLES):
        raise RecordingFinalizationError(
            "Gate B requires exactly one dataset and one video archive")
    if len(paths) != len(set(paths)) \
            or len(roles) != len(set(roles)) \
            or len(hashes) != len(set(hashes)):
        raise RecordingFinalizationError(
            "Gate B archives must have distinct paths, roles, and hashes")
    return tuple(evidence)


def _remember_fresh_instance(
    instance: object,
    seen: list[tuple[weakref.ReferenceType[object] | None, object | None]],
    *,
    factory_name: str,
) -> None:
    for reference, strong_reference in seen:
        prior = reference() if reference is not None else strong_reference
        if prior is instance:
            raise RecordingFinalizationError(
                f"{factory_name} must return a fresh instance per recording")
    try:
        seen.append((weakref.ref(instance), None))
    except TypeError:
        seen.append((None, instance))


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RecordingFinalizationError(
                f"frozen inventory contains duplicate JSON key: {key}")
        result[key] = value
    return result


def _inventory_asset(value: object, *, role: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _INVENTORY_ASSET_FIELDS:
        raise RecordingFinalizationError(
            "frozen inventory asset has invalid fields")
    original_path = _safe_original_path(
        value["original_path"], field="inventory asset original_path")
    if value["role"] != role:
        raise RecordingFinalizationError(
            f"frozen inventory asset role must be {role}")
    sha256 = value["sha256"]
    if not isinstance(sha256, str) or len(sha256) != 64 \
            or any(character not in "0123456789abcdef" for character in sha256):
        raise RecordingFinalizationError(
            "frozen inventory asset hash must be a lowercase SHA-256")
    return {
        "original_path": original_path,
        "role": role,
        "sha256": sha256,
    }


def _parse_frozen_inventory(
    payload: bytes,
) -> dict[str, dict[str, object]]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordingFinalizationError(
            "frozen inventory manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "inventory_schema_version", "records",
    }:
        raise RecordingFinalizationError(
            "frozen inventory manifest has invalid top-level fields")
    if value["inventory_schema_version"] != VIDIMU_INVENTORY_SCHEMA_VERSION:
        raise RecordingFinalizationError(
            "frozen inventory schema version is unsupported")
    records = value["records"]
    if not isinstance(records, list) or not records:
        raise RecordingFinalizationError(
            "frozen inventory records must be a nonempty list")

    records_by_id: dict[str, dict[str, object]] = {}
    claimed_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != _INVENTORY_RECORD_FIELDS:
            raise RecordingFinalizationError(
                "frozen inventory record has invalid fields")
        recording_id = record["recording_id"]
        if not isinstance(recording_id, str) or not recording_id \
                or PurePosixPath(recording_id).name != recording_id \
                or recording_id in {".", ".."}:
            raise RecordingFinalizationError(
                "frozen inventory recording_id must be nonempty")
        if recording_id in records_by_id:
            raise RecordingFinalizationError(
                "frozen inventory contains a duplicate recording ID")
        video = _inventory_asset(record["video"], role="ORIGINAL_VIDEO")
        raw_imu_assets = record["imu_assets"]
        if not isinstance(raw_imu_assets, list) or not raw_imu_assets:
            raise RecordingFinalizationError(
                "frozen inventory record requires paired IMU assets")
        imu_assets = [
            _inventory_asset(asset, role="IMU") for asset in raw_imu_assets
        ]
        imu_assets.sort(key=lambda asset: (
            asset["original_path"], asset["role"], asset["sha256"],
        ))
        paths = [video["original_path"], *(
            asset["original_path"] for asset in imu_assets
        )]
        if len(paths) != len(set(paths)) or claimed_paths.intersection(paths):
            raise RecordingFinalizationError(
                "frozen inventory contains duplicate source asset paths")
        claimed_paths.update(paths)
        records_by_id[recording_id] = {
            "recording_id": recording_id,
            "video": video,
            "imu_assets": imu_assets,
        }
    return records_by_id


def _load_frozen_inventory(
    path: Path,
    expected_sha256: str,
) -> dict[str, dict[str, object]]:
    payload, _size = _verified_sha256(
        path,
        expected_sha256,
        capture_limit=_MAX_INVENTORY_BYTES,
    )
    assert payload is not None
    return _parse_frozen_inventory(payload)


def preflight_vidimu_snapshot(inputs: VidimuSnapshotInputs) -> None:
    """Verify completeness and every immutable byte before processing starts."""

    if not isinstance(inputs, VidimuSnapshotInputs):
        raise RecordingFinalizationError(
            "snapshot inputs must be VidimuSnapshotInputs")
    archive_evidence = _canonical_archive_assets(inputs.archive_assets)
    inventory = _load_frozen_inventory(
        inputs.inventory_manifest_path, inputs.inventory_manifest_sha256)
    _verified_sha256(inputs.license_record_path, inputs.license_record_sha256)
    expected = tuple(inputs.expected_recording_ids)
    if not expected or len(expected) != len(set(expected)) or not all(
        isinstance(recording_id, str) and recording_id
        and PurePosixPath(recording_id).name == recording_id
        and recording_id not in {".", ".."}
        for recording_id in expected
    ):
        raise RecordingFinalizationError(
            "expected_recording_ids must be nonempty and unique")
    if set(inventory) != set(expected):
        missing = sorted(set(expected).difference(inventory))
        extra = sorted(set(inventory).difference(expected))
        raise RecordingFinalizationError(
            "frozen inventory recording mismatch; "
            f"missing={missing}, extra={extra}")
    inventory_paths = {
        str(record["video"]["original_path"])
        for record in inventory.values()
    }
    inventory_paths.update(
        str(asset["original_path"])
        for record in inventory.values()
        for asset in record["imu_assets"]
    )
    if inventory_paths.intersection(
        asset["original_path"] for asset in archive_evidence
    ):
        raise RecordingFinalizationError(
            "archive and recording asset paths must be distinct")
    records_by_id: dict[str, VidimuSnapshotRecord] = {}
    for record in inputs.records:
        if not isinstance(record, VidimuSnapshotRecord):
            raise RecordingFinalizationError(
                "snapshot contains an invalid recording entry")
        provenance = record.provenance
        if provenance.source_kind != "VIDIMU_PUBLIC":
            raise RecordingFinalizationError(
                "Gate B rejects synthetic or substituted recording provenance")
        if provenance.recording_id in records_by_id:
            raise RecordingFinalizationError(
                "snapshot contains a duplicate recording ID")
        if provenance.inventory_manifest_sha256 \
                != inputs.inventory_manifest_sha256 \
                or provenance.license_record_sha256 \
                != inputs.license_record_sha256:
            raise RecordingFinalizationError(
                "record provenance disagrees with snapshot-level frozen assets")
        if record.video.role != "ORIGINAL_VIDEO" \
                or record.video.original_path != provenance.source_original_path:
            raise RecordingFinalizationError(
                "recording video is not the frozen original-camera asset")
        _validate_asset(record.video, role="ORIGINAL_VIDEO")
        if not record.imu_assets:
            raise RecordingFinalizationError(
                "Gate B requires each recording's paired IMU asset")
        imu_evidence: list[dict[str, str]] = []
        for asset in record.imu_assets:
            _validate_asset(asset, role="IMU")
            imu_evidence.append(_asset_evidence(asset))
        imu_evidence.sort(key=lambda asset: (
            asset["original_path"], asset["role"], asset["sha256"],
        ))
        inventory_record = inventory.get(provenance.recording_id)
        if inventory_record is None \
                or inventory_record["video"] != _asset_evidence(record.video) \
                or inventory_record["imu_assets"] != imu_evidence:
            raise RecordingFinalizationError(
                "record source evidence disagrees with the frozen inventory")
        records_by_id[provenance.recording_id] = record
    if set(records_by_id) != set(expected):
        missing = sorted(set(expected).difference(records_by_id))
        extra = sorted(set(records_by_id).difference(expected))
        raise RecordingFinalizationError(
            f"snapshot recording inventory mismatch; missing={missing}, extra={extra}")


def finalize_vidimu_snapshot(
    inputs: VidimuSnapshotInputs,
    output_root: str | Path,
    *,
    decoder_factory: Callable[[], PTSDecoder],
    estimator_factory: Callable[[VidimuSnapshotRecord], OfflineFrameEstimator],
) -> tuple[FinalizedRecording | SourceFailureArtifact, ...]:
    """Finalize the complete inventory, resuming only at recording granularity.

    This function verifies IMU presence and hashes but deliberately does not
    parse or synchronize IMU data. Any preflight failure occurs before the first
    recording is finalized.
    """

    preflight_vidimu_snapshot(inputs)
    records = {record.provenance.recording_id: record for record in inputs.records}
    archive_evidence = _canonical_archive_assets(inputs.archive_assets)
    seen_decoders: list[
        tuple[weakref.ReferenceType[object] | None, object | None]
    ] = []
    seen_estimators: list[
        tuple[weakref.ReferenceType[object] | None, object | None]
    ] = []
    finalized: list[FinalizedRecording | SourceFailureArtifact] = []
    for recording_id in inputs.expected_recording_ids:
        record = records[recording_id]
        factory_decoder = decoder_factory()
        if type(factory_decoder) is not PTSDecoder \
                or type(factory_decoder.config) is not DecodeConfig:
            raise RecordingFinalizationError(
                "decoder_factory must return a new exact PTSDecoder and DecodeConfig")
        _remember_fresh_instance(
            factory_decoder, seen_decoders, factory_name="decoder_factory")
        # The factory selects only the frozen config. Gate B executes a new
        # trusted base-class instance so per-instance method shadowing cannot
        # forge the eligible post-verification decode-failure signal.
        try:
            trusted_config = DecodeConfig(**asdict(factory_decoder.config))
        except (TypeError, ValueError) as exc:
            raise RecordingFinalizationError(
                "decoder_factory returned an invalid frozen DecodeConfig") from exc
        decoder = PTSDecoder(trusted_config)
        estimator = estimator_factory(record)
        _remember_fresh_instance(
            estimator, seen_estimators, factory_name="estimator_factory")
        paired_imu_evidence = tuple(
            _asset_evidence(asset)
            for asset in sorted(
                record.imu_assets,
                key=lambda item: (
                    item.original_path, item.role, item.sha256,
                ),
            )
        )
        recording_outcome = finalize_vidimu_recording(
            record.video.local_path,
            output_root,
            expected_source_video_sha256=record.video.sha256,
            provenance=record.provenance,
            decoder=decoder,
            estimator=estimator,
            validation_gate="GATE_B_VIDIMU",
            _gate_b_preflight_token=_GATE_B_PREFLIGHT_TOKEN,
            _paired_imu_assets=paired_imu_evidence,
            _source_archives=archive_evidence,
        )
        if type(recording_outcome) is _GateBVerifiedSourceDecodeOutcome:
            # The private outcome can only be created at the trusted decoder
            # boundary and carries the exact processing identity consumed there.
            _payload, source_video_bytes = _verified_sha256(
                Path(record.video.local_path), record.video.sha256)
            outcome: FinalizedRecording | SourceFailureArtifact = (
                publish_source_decode_failure(
                    output_root,
                    provenance=record.provenance,
                    expected_source_video_sha256=record.video.sha256,
                    source_video_bytes=source_video_bytes,
                    decoder_version=recording_outcome.decoder_version,
                    decoder_config=recording_outcome.decoder_config,
                    estimator_provenance=(
                        recording_outcome.estimator_provenance
                    ),
                    paired_imu_assets=paired_imu_evidence,
                    source_archives=archive_evidence,
                    _gate_b_preflight_token=_GATE_B_PREFLIGHT_TOKEN,
                )
            )
        elif type(recording_outcome) is FinalizedRecording:
            outcome = recording_outcome
        else:
            raise RecordingFinalizationError(
                "recording finalizer returned an unexpected internal outcome")
        finalized.append(outcome)
    return tuple(finalized)


__all__ = [
    "VIDIMU_INVENTORY_SCHEMA_VERSION",
    "FrozenSourceAsset",
    "SourceFailureArtifact",
    "VidimuSnapshotInputs",
    "VidimuSnapshotRecord",
    "finalize_vidimu_snapshot",
    "preflight_vidimu_snapshot",
]
