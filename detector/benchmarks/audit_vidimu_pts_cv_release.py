"""Independent release audit for VIDIMU PTS/CV finalization bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from motionbloom.tremora_store.finalize._bundle_io import (
    FINALIZATION_FILES,
    SOURCE_FAILURE_FILES,
    FinalizationBundleError,
    read_json,
)
from motionbloom.tremora_store.finalize.audit_finalized_recording import (
    audit_finalized_recording,
)
from motionbloom.tremora_store.finalize.source_failure_artifact import (
    audit_source_failure_artifact,
)


class ReleaseAuditError(RuntimeError):
    """Raised when a claimed Gate-A/Gate-B release is incomplete or mixed."""


VIDIMU_INVENTORY_SCHEMA_VERSION = "tremora-vidimu-frozen-inventory-1.0.0"
_MAX_INVENTORY_BYTES = 16 * 1024 * 1024
_MAX_RECORDING_IDS_BYTES = 1024 * 1024
_INVENTORY_ASSET_FIELDS = {"original_path", "role", "sha256"}
_INVENTORY_RECORD_FIELDS = {"imu_assets", "recording_id", "video"}
_VIDIMU_ARCHIVE_ROLES = frozenset({"DATASET_ARCHIVE", "VIDEO_ARCHIVE"})
_PROCESSING_IDENTITY_FIELDS = (
    "model_id",
    "model_weights_sha256",
    "preprocessing_config_sha256",
    "inference_environment_id",
    "decoder_version",
    "decoder_config_sha256",
)
_AUDIT_COUNTER_FIELDS = (
    "videos_opened", "videos_failed",
    "decoded_frame_count", "frames_with_valid_pts",
    "frames_with_duplicate_pts", "frames_with_missing_pts",
    "frames_with_nonmonotonic_pts", "timestamp_discontinuities",
    "decoded_corrupt_frames", "cv_frame_result_count",
    "frames_with_detection", "frames_without_detection",
    "inference_failures", "detection_row_count", "orphan_frame_results",
    "orphan_detections", "duplicate_frame_ids", "duplicate_detection_ids",
    "missing_frame_results", "coordinate_transform_failures",
    "source_hash_mismatches", "model_hash_mismatches",
)


@dataclass(frozen=True)
class _RecordingOutcome:
    path: Path
    kind: str
    artifact_files: frozenset[str]


def _pinned_sha256(path: Path) -> tuple[int, int, int, str]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise ReleaseAuditError(
            "release audit requires O_NOFOLLOW and O_NONBLOCK")
    flags = (
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseAuditError(f"required source is unavailable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReleaseAuditError(f"required source is not regular: {path}")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            payload = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset)
            if not payload:
                raise ReleaseAuditError(f"source ended while hashing: {path}")
            digest.update(payload)
            offset += len(payload)
        after = os.fstat(descriptor)
        if (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        ) != (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise ReleaseAuditError(f"source changed while hashing: {path}")
        return before.st_dev, before.st_ino, before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _verified_sha256(path: Path, expected: str) -> int:
    _, _, observed_bytes, observed = _pinned_sha256(path)
    if observed != expected:
        raise ReleaseAuditError(f"source hash mismatch: {path}")
    return observed_bytes


def _pinned_bytes(path: Path, *, max_bytes: int) -> tuple[bytes, str]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise ReleaseAuditError(
            "release audit requires O_NOFOLLOW and O_NONBLOCK")
    flags = (
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseAuditError(f"required source is unavailable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ReleaseAuditError(
                "inventory manifest must be a bounded regular file")
        payload = bytearray()
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset)
            if not chunk:
                raise ReleaseAuditError("inventory ended while hashing")
            payload.extend(chunk)
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        ) != (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise ReleaseAuditError("inventory changed while reading")
        return bytes(payload), digest.hexdigest()
    finally:
        os.close(descriptor)


def _safe_original_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ReleaseAuditError(f"{field} must be a string")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or not value or any(
        part in {"", ".", ".."} for part in parsed.parts
    ) or parsed.as_posix() != value or "\\" in value or "\x00" in value:
        raise ReleaseAuditError(f"{field} must be a safe relative path")
    return value


def _lowercase_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ReleaseAuditError(f"{field} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class GateBArchiveAnchor:
    """One externally trusted VIDIMU archive path, role, and digest."""

    original_path: str
    role: str
    sha256: str

    def __post_init__(self) -> None:
        _safe_original_path(self.original_path, field="archive anchor path")
        if not isinstance(self.role, str) \
                or self.role not in _VIDIMU_ARCHIVE_ROLES:
            raise ReleaseAuditError("archive anchor role is invalid")
        _lowercase_sha256(self.sha256, field="archive anchor hash")


@dataclass(frozen=True)
class GateBTrustAnchors:
    """Caller-supplied identity, path, role, and digest Gate-B anchors."""

    expected_dataset_id: str
    expected_dataset_version: str
    inventory_manifest_sha256: str
    source_archives: tuple[GateBArchiveAnchor, ...]
    license_record_sha256: str

    def __post_init__(self) -> None:
        for field in ("expected_dataset_id", "expected_dataset_version"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ReleaseAuditError(f"{field} must be a nonempty string")
        for field in (
            "inventory_manifest_sha256",
            "license_record_sha256",
        ):
            _lowercase_sha256(getattr(self, field), field=field)
        if not isinstance(self.source_archives, tuple) or any(
            not isinstance(anchor, GateBArchiveAnchor)
            for anchor in self.source_archives
        ):
            raise ReleaseAuditError(
                "source_archives must be an immutable tuple of archive anchors")
        canonical = tuple(sorted(self.source_archives, key=lambda anchor: (
            anchor.original_path, anchor.role, anchor.sha256,
        )))
        paths = [anchor.original_path for anchor in self.source_archives]
        roles = [anchor.role for anchor in self.source_archives]
        hashes = [anchor.sha256 for anchor in self.source_archives]
        if self.source_archives != canonical \
                or set(roles) != _VIDIMU_ARCHIVE_ROLES \
                or len(self.source_archives) != len(_VIDIMU_ARCHIVE_ROLES) \
                or len(paths) != len(set(paths)) \
                or len(roles) != len(set(roles)) \
                or len(hashes) != len(set(hashes)):
            raise ReleaseAuditError(
                "source archive anchors must be canonical, exact, and distinct")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseAuditError(
                f"frozen inventory contains duplicate JSON key: {key}")
        result[key] = value
    return result


def _inventory_asset(value: object, *, role: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _INVENTORY_ASSET_FIELDS:
        raise ReleaseAuditError("frozen inventory asset has invalid fields")
    if value["role"] != role:
        raise ReleaseAuditError(f"frozen inventory asset role must be {role}")
    return {
        "original_path": _safe_original_path(
            value["original_path"], field="inventory asset original_path"),
        "role": role,
        "sha256": _lowercase_sha256(
            value["sha256"], field="inventory asset hash"),
    }


def _parse_frozen_inventory(payload: bytes) -> dict[str, dict[str, object]]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseAuditError(
            "frozen inventory manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "inventory_schema_version", "records",
    }:
        raise ReleaseAuditError(
            "frozen inventory manifest has invalid top-level fields")
    if value["inventory_schema_version"] != VIDIMU_INVENTORY_SCHEMA_VERSION:
        raise ReleaseAuditError("frozen inventory schema version is unsupported")
    records = value["records"]
    if not isinstance(records, list) or not records:
        raise ReleaseAuditError("frozen inventory records must be nonempty")
    result: dict[str, dict[str, object]] = {}
    claimed_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != _INVENTORY_RECORD_FIELDS:
            raise ReleaseAuditError("frozen inventory record has invalid fields")
        recording_id = record["recording_id"]
        if not isinstance(recording_id, str) or not recording_id \
                or PurePosixPath(recording_id).name != recording_id \
                or recording_id in {".", ".."} or recording_id in result:
            raise ReleaseAuditError(
                "frozen inventory recording IDs must be nonempty and unique")
        video = _inventory_asset(record["video"], role="ORIGINAL_VIDEO")
        raw_imu_assets = record["imu_assets"]
        if not isinstance(raw_imu_assets, list) or not raw_imu_assets:
            raise ReleaseAuditError(
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
            raise ReleaseAuditError(
                "frozen inventory contains duplicate source asset paths")
        claimed_paths.update(paths)
        result[recording_id] = {
            "recording_id": recording_id,
            "video": video,
            "imu_assets": imu_assets,
        }
    return result


def _load_frozen_inventory(
    path: Path,
    expected_sha256: str,
) -> dict[str, dict[str, object]]:
    payload, observed = _pinned_bytes(path, max_bytes=_MAX_INVENTORY_BYTES)
    if observed != expected_sha256:
        raise ReleaseAuditError("frozen inventory hash mismatch")
    return _parse_frozen_inventory(payload)


def _manifest_imu_assets(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ReleaseAuditError("Gate B source lacks paired IMU evidence")
    normalized = [_inventory_asset(item, role="IMU") for item in value]
    canonical = sorted(normalized, key=lambda asset: (
        asset["original_path"], asset["role"], asset["sha256"],
    ))
    if normalized != canonical or len({
        asset["original_path"] for asset in normalized
    }) != len(normalized):
        raise ReleaseAuditError(
            "Gate B paired IMU evidence is not canonical and unique")
    return normalized


def _manifest_archive_assets(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ReleaseAuditError("Gate B source lacks archive evidence")
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != _INVENTORY_ASSET_FIELDS:
            raise ReleaseAuditError("Gate B archive evidence has invalid fields")
        role = item["role"]
        if not isinstance(role, str) or role not in _VIDIMU_ARCHIVE_ROLES:
            raise ReleaseAuditError("Gate B archive evidence role is invalid")
        normalized.append({
            "original_path": _safe_original_path(
                item["original_path"], field="archive original_path"),
            "role": str(role),
            "sha256": _lowercase_sha256(
                item["sha256"], field="archive hash"),
        })
    canonical = sorted(normalized, key=lambda asset: (
        asset["original_path"], asset["role"], asset["sha256"],
    ))
    paths = [asset["original_path"] for asset in normalized]
    roles = [asset["role"] for asset in normalized]
    hashes = [asset["sha256"] for asset in normalized]
    if normalized != canonical \
            or set(roles) != _VIDIMU_ARCHIVE_ROLES \
            or len(normalized) != len(_VIDIMU_ARCHIVE_ROLES) \
            or len(paths) != len(set(paths)) \
            or len(roles) != len(set(roles)) \
            or len(hashes) != len(set(hashes)):
        raise ReleaseAuditError(
            "Gate B archive evidence is not canonical, exact, and distinct")
    return normalized


def _source_candidate(source_root: Path, original_path: object) -> Path:
    relative = _safe_original_path(original_path, field="source original path")
    candidate = source_root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ReleaseAuditError(
            f"required source is unavailable: {relative}") from exc
    if source_root not in resolved.parents:
        raise ReleaseAuditError(f"source asset escaped source root: {relative}")
    return resolved


def _resolved_release_root(path: str | Path, *, label: str) -> tuple[Path, tuple[int, int]]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ReleaseAuditError(f"{label} must be a real directory")
    try:
        resolved = candidate.resolve(strict=True)
        state = resolved.stat()
    except OSError as exc:
        raise ReleaseAuditError(f"{label} is unavailable") from exc
    return resolved, (state.st_dev, state.st_ino)


def _bundle_map(
    root: Path,
    expected_recording_ids: tuple[str, ...],
) -> dict[str, _RecordingOutcome]:
    if root.is_symlink() or not root.is_dir():
        raise ReleaseAuditError("finalized root must be a real directory")
    children = list(root.iterdir())
    if any(child.is_symlink() or not child.is_dir() for child in children):
        raise ReleaseAuditError(
            "finalized root contains a non-recording artifact")
    present_recordings = {child.name for child in children}
    if present_recordings != set(expected_recording_ids):
        raise ReleaseAuditError(
            "finalized recording inventory does not equal the frozen inventory")
    result: dict[str, _RecordingOutcome] = {}
    for recording_id in expected_recording_ids:
        recording_dir = root / recording_id
        candidates = list(recording_dir.iterdir())
        if len(candidates) != 1 or candidates[0].is_symlink() \
                or not candidates[0].is_dir():
            raise ReleaseAuditError(
                f"recording {recording_id} must have exactly one selected "
                "finalization outcome")
        candidate = candidates[0]
        present = {item.name for item in candidate.iterdir()}
        if "_SUCCESS" in present and "_FAILURE" not in present:
            result[recording_id] = _RecordingOutcome(
                path=candidate,
                kind="SUCCESS",
                artifact_files=FINALIZATION_FILES,
            )
        elif "_FAILURE" in present and "_SUCCESS" not in present:
            result[recording_id] = _RecordingOutcome(
                path=candidate,
                kind="SOURCE_DECODE_FAILURE",
                artifact_files=SOURCE_FAILURE_FILES,
            )
        else:
            raise ReleaseAuditError(
                f"recording {recording_id} lacks one unambiguous terminal outcome")
    return result


def _compare_bundle_bytes(
    left: Path,
    right: Path,
    artifact_files: frozenset[str],
) -> dict[str, object]:
    if {item.name for item in left.iterdir()} != artifact_files \
            or {item.name for item in right.iterdir()} != artifact_files:
        raise ReleaseAuditError("replay bundle inventory differs")
    artifact_sha256: dict[str, str] = {}
    for filename in sorted(artifact_files):
        left_fingerprint = _pinned_sha256(left / filename)
        right_fingerprint = _pinned_sha256(right / filename)
        if left_fingerprint[:2] == right_fingerprint[:2]:
            raise ReleaseAuditError(
                f"artifact trees share a file inode: "
                f"{left.parent.name}/{filename}")
        if left_fingerprint[2:] != right_fingerprint[2:]:
            raise ReleaseAuditError(
                f"replay is not byte-identical: {left.parent.name}/{filename}")
        artifact_sha256[filename] = left_fingerprint[3]
    return {
        "artifact_count": len(artifact_files),
        "artifact_sha256": artifact_sha256,
        "all_artifact_bytes_identical": True,
        "all_corresponding_artifact_files_distinct": True,
        "primary_finalization_id": left.name,
        "replay_finalization_id": right.name,
    }


def audit_vidimu_pts_cv_release(
    finalized_root: str | Path,
    *,
    expected_recording_ids: tuple[str, ...],
    required_gate: str,
    replay_root: str | Path | None,
    source_root: str | Path | None = None,
    inventory_manifest_path: str | Path | None = None,
    license_record_path: str | Path | None = None,
    gate_b_trust_anchors: GateBTrustAnchors | None = None,
) -> dict[str, object]:
    """Audit exact inventory, per-record bundles, sources, and byte replay."""

    if not expected_recording_ids or len(expected_recording_ids) != len(
            set(expected_recording_ids)) or not all(
                isinstance(recording_id, str) and recording_id
                and PurePosixPath(recording_id).name == recording_id
                and recording_id not in {".", ".."}
                for recording_id in expected_recording_ids
            ):
        raise ReleaseAuditError(
            "expected_recording_ids must be nonempty and unique")
    if required_gate not in {
        "GATE_A_SYNTHETIC", "GATE_A_REAL_VIDEO_PILOT", "GATE_B_VIDIMU",
    }:
        raise ReleaseAuditError("required_gate is invalid")
    if required_gate == "GATE_B_VIDIMU":
        if not isinstance(gate_b_trust_anchors, GateBTrustAnchors):
            raise ReleaseAuditError(
                "Gate B requires caller-provided frozen trust anchors")
    elif gate_b_trust_anchors is not None:
        raise ReleaseAuditError(
            "Gate-B trust anchors cannot attest a different validation gate")
    finalized_path, finalized_identity = _resolved_release_root(
        finalized_root, label="finalized root")
    bundles = _bundle_map(finalized_path, expected_recording_ids)
    replay_bundles = None
    if replay_root is not None:
        replay_path, replay_identity = _resolved_release_root(
            replay_root, label="replay root")
        if replay_identity == finalized_identity or replay_path == finalized_path:
            raise ReleaseAuditError(
                "artifact replay comparison requires a distinct finalized root")
        replay_bundles = _bundle_map(replay_path, expected_recording_ids)
        for recording_id in expected_recording_ids:
            if bundles[recording_id].kind != replay_bundles[recording_id].kind:
                raise ReleaseAuditError(
                    "primary and replay recording outcomes differ")
            left_state = bundles[recording_id].path.stat()
            right_state = replay_bundles[recording_id].path.stat()
            if (left_state.st_dev, left_state.st_ino) == (
                right_state.st_dev, right_state.st_ino,
            ):
                raise ReleaseAuditError(
                    "artifact replay comparison requires distinct bundle directories")
    else:
        raise ReleaseAuditError(
            "release PASS requires a distinct byte-identical artifact tree")

    per_recording: dict[str, dict[str, object]] = {}
    totals: dict[str, int] = {}
    inventory_hashes: set[str] = set()
    license_hashes: set[str] = set()
    gate_b_sources: dict[str, dict[str, object]] = {}
    processing_identities: set[tuple[object, ...]] = set()
    processing_identity: dict[str, object] | None = None
    artifact_hashes: dict[str, dict[str, str]] = {}
    successful_recordings: set[str] = set()
    source_failure_recordings: set[str] = set()
    for recording_id in expected_recording_ids:
        outcome = bundles[recording_id]
        bundle = outcome.path
        try:
            if outcome.kind == "SUCCESS":
                audit = audit_finalized_recording(bundle)
                manifest, _ = read_json(bundle / "finalization_manifest.json")
                successful_recordings.add(recording_id)
            else:
                if required_gate != "GATE_B_VIDIMU":
                    raise ReleaseAuditError(
                        "source failure outcomes are permitted only for Gate B")
                audit = audit_source_failure_artifact(bundle)
                manifest, _ = read_json(bundle / "source_failure_manifest.json")
                source_failure_recordings.add(recording_id)
        except (FinalizationBundleError, OSError) as exc:
            raise ReleaseAuditError(
                f"recording {recording_id} outcome failed strict audit") from exc
        if manifest.get("validation_gate") != required_gate:
            raise ReleaseAuditError(
                f"recording {recording_id} was finalized under a different gate")
        identity = manifest.get("identity_inputs")
        if not isinstance(identity, dict) or any(
            field not in identity for field in _PROCESSING_IDENTITY_FIELDS
        ):
            raise ReleaseAuditError(
                "recording processing identity is incomplete")
        selected_identity = {
            field: identity[field] for field in _PROCESSING_IDENTITY_FIELDS
        }
        processing_identities.add(tuple(
            selected_identity[field] for field in _PROCESSING_IDENTITY_FIELDS
        ))
        if processing_identity is None:
            processing_identity = selected_identity
        source = manifest.get("source")
        if not isinstance(source, dict):
            raise ReleaseAuditError("recording source provenance is missing")
        if required_gate == "GATE_B_VIDIMU":
            assert gate_b_trust_anchors is not None
            if source.get("source_kind") != "VIDIMU_PUBLIC":
                raise ReleaseAuditError("Gate B contains a substituted source")
            if source.get("dataset_id") \
                    != gate_b_trust_anchors.expected_dataset_id \
                    or source.get("dataset_version") \
                    != gate_b_trust_anchors.expected_dataset_version:
                raise ReleaseAuditError(
                    "Gate B dataset identity disagrees with caller trust anchors")
            for field, expected in (
                (
                    "inventory_manifest_sha256",
                    gate_b_trust_anchors.inventory_manifest_sha256,
                ),
                (
                    "license_record_sha256",
                    gate_b_trust_anchors.license_record_sha256,
                ),
            ):
                if source.get(field) != expected:
                    raise ReleaseAuditError(
                        f"Gate B {field} disagrees with caller trust anchors")
            for field, target in (
                ("inventory_manifest_sha256", inventory_hashes),
                ("license_record_sha256", license_hashes),
            ):
                value = source.get(field)
                if not isinstance(value, str) or not value:
                    raise ReleaseAuditError(f"Gate B source lacks {field}")
                target.add(value)
            if source.get("recording_id") != recording_id:
                raise ReleaseAuditError(
                    "Gate B source recording ID disagrees with bundle inventory")
            _manifest_imu_assets(source.get("paired_imu_assets"))
            source_archives = _manifest_archive_assets(
                source.get("source_archives"))
            trusted_archives = [
                asdict(anchor)
                for anchor in gate_b_trust_anchors.source_archives
            ]
            if source_archives != trusted_archives:
                raise ReleaseAuditError(
                    "Gate B archives disagree with caller trust anchors")
            gate_b_sources[recording_id] = source
        if replay_bundles is not None:
            replay_outcome = replay_bundles[recording_id]
            try:
                if outcome.kind == "SUCCESS":
                    replay_audit = audit_finalized_recording(
                        replay_outcome.path)
                else:
                    replay_audit = audit_source_failure_artifact(
                        replay_outcome.path)
            except (FinalizationBundleError, OSError) as exc:
                raise ReleaseAuditError(
                    f"recording {recording_id} replay failed strict audit") from exc
            if replay_audit != audit:
                raise ReleaseAuditError("replay audit payload differs")
            replay_evidence = _compare_bundle_bytes(
                bundle,
                replay_outcome.path,
                outcome.artifact_files,
            )
        else:  # pragma: no cover - replay absence fails above
            raise ReleaseAuditError("artifact replay tree is unavailable")
        hashes = replay_evidence["artifact_sha256"]
        assert isinstance(hashes, dict)
        artifact_hashes[recording_id] = {
            str(name): str(value) for name, value in hashes.items()
        }
        per_recording[recording_id] = {
            **audit,
            "recording_outcome": outcome.kind,
            "artifact_replay_evidence": replay_evidence,
        }
        for field in _AUDIT_COUNTER_FIELDS:
            totals[field] = totals.get(field, 0) + int(audit[field])

    if len(processing_identities) != 1 or processing_identity is None:
        raise ReleaseAuditError(
            "release contains mixed model, preprocessing, environment, or decoder identity")
    if totals.get("videos_opened") != len(expected_recording_ids) \
            or totals.get("videos_failed") != len(source_failure_recordings):
        raise ReleaseAuditError(
            "video outcome counts do not reconcile to the frozen inventory")
    if totals.get("decoded_frame_count") != totals.get("cv_frame_result_count"):
        raise ReleaseAuditError(
            "opened-video decoded frames do not reconcile to CV frame results")

    if required_gate == "GATE_B_VIDIMU":
        assert gate_b_trust_anchors is not None
        if any(len(values) != 1 for values in (
            inventory_hashes, license_hashes,
        )):
            raise ReleaseAuditError("Gate B snapshot provenance is mixed")
        if source_root is None or inventory_manifest_path is None \
                or license_record_path is None:
            raise ReleaseAuditError(
                "Gate B audit requires the frozen source, inventory, and license")
        source_base, _source_identity = _resolved_release_root(
            source_root, label="Gate B source root")
        inventory = _load_frozen_inventory(
            Path(inventory_manifest_path),
            gate_b_trust_anchors.inventory_manifest_sha256,
        )
        if set(inventory) != set(expected_recording_ids):
            raise ReleaseAuditError(
                "frozen inventory does not exactly match expected recordings")
        paired_imu_count = 0
        for recording_id in expected_recording_ids:
            source = gate_b_sources[recording_id]
            inventory_record = inventory[recording_id]
            video = inventory_record["video"]
            assert isinstance(video, dict)
            if source.get("source_original_path") != video["original_path"] \
                    or source.get("source_video_sha256") != video["sha256"]:
                raise ReleaseAuditError(
                    "Gate B original video disagrees with frozen inventory")
            paired_imu_assets = _manifest_imu_assets(
                source.get("paired_imu_assets"))
            if paired_imu_assets != inventory_record["imu_assets"]:
                raise ReleaseAuditError(
                    "Gate B paired IMU evidence disagrees with frozen inventory")
            video_path = _source_candidate(
                source_base, source["source_original_path"])
            source_video_bytes = _verified_sha256(
                video_path, str(source["source_video_sha256"]))
            claimed_source_video_bytes = source.get("source_video_bytes")
            if isinstance(claimed_source_video_bytes, bool) \
                    or not isinstance(claimed_source_video_bytes, int) \
                    or claimed_source_video_bytes != source_video_bytes:
                raise ReleaseAuditError(
                    "Gate B source byte count disagrees with pinned source")
            verified_imu_assets: list[dict[str, str]] = []
            for asset in paired_imu_assets:
                imu_path = _source_candidate(source_base, asset["original_path"])
                _verified_sha256(imu_path, asset["sha256"])
                verified_imu_assets.append(dict(asset))
                paired_imu_count += 1
            per_recording[recording_id]["source_asset_evidence"] = {
                "source_video": dict(video),
                "source_video_bytes": source_video_bytes,
                "recording_outcome": bundles[recording_id].kind,
                "paired_imu_assets": verified_imu_assets,
                "source_archives": [
                    asdict(anchor)
                    for anchor in gate_b_trust_anchors.source_archives
                ],
                "all_inventory_assets_present": True,
                "all_inventory_asset_hashes_verified": True,
            }
        for anchor in gate_b_trust_anchors.source_archives:
            archive = _source_candidate(source_base, anchor.original_path)
            _verified_sha256(archive, anchor.sha256)
        for recording_id in expected_recording_ids:
            evidence = per_recording[recording_id]["source_asset_evidence"]
            assert isinstance(evidence, dict)
            evidence["all_source_archives_present"] = True
            evidence["all_source_archive_hashes_verified"] = True
        archive_asset_count = len(gate_b_trust_anchors.source_archives)
        _verified_sha256(
            Path(license_record_path),
            gate_b_trust_anchors.license_record_sha256,
        )
        source_assets_present = "ALL_TRUST_ANCHORED_GATE_B_ASSETS_PRESENT"
        source_hashes_verified = (
            "ALL_TRUST_ANCHORED_GATE_B_ASSET_HASHES_VERIFIED"
        )
        serialized_anchors: dict[str, object] | None = asdict(
            gate_b_trust_anchors)
        serialized_anchors["source_archives"] = [
            asdict(anchor) for anchor in gate_b_trust_anchors.source_archives
        ]
    else:
        paired_imu_count = 0
        archive_asset_count = 0
        source_assets_present = "NOT_REVERIFIED_FOR_GATE_A_ARTIFACT_AUDIT"
        source_hashes_verified = (
            "PINNED_DECODE_EVIDENCE_VERIFIED_IN_BUNDLES_ONLY"
        )
        serialized_anchors = None

    return {
        "audit_version": 1,
        "required_gate": required_gate,
        "inventory_record_count": len(expected_recording_ids),
        "successful_recording_count": len(successful_recordings),
        "source_failure_recording_count": len(source_failure_recordings),
        "paired_imu_asset_count": paired_imu_count,
        "source_archive_asset_count": archive_asset_count,
        "source_assets_present": source_assets_present,
        "source_hashes_verified": source_hashes_verified,
        **totals,
        "artifact_hashes": artifact_hashes,
        "artifact_replay_status": (
            "BYTE_IDENTICAL_DISTINCT_FILES_AND_ROOTS_PASS"
        ),
        "deterministic_replay_status": (
            "STORED_ARTIFACT_BYTES_IDENTICAL_DISTINCT_FILES_AND_ROOTS_PASS"
        ),
        "independent_rerun_attestation": "NOT_PROVIDED",
        "frozen_processing_identity": processing_identity,
        "gate_b_trust_anchors": serialized_anchors,
        "per_recording": per_recording,
        "overall_verdict": "PASS",
    }


def _load_recording_ids(path: Path) -> tuple[str, ...]:
    try:
        payload, _sha256 = _pinned_bytes(
            path,
            max_bytes=_MAX_RECORDING_IDS_BYTES,
        )
        value = json.loads(payload)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ReleaseAuditError,
    ) as exc:
        raise ReleaseAuditError("recording ID manifest is invalid") from exc
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ReleaseAuditError("recording ID manifest must be a JSON string list")
    return tuple(value)


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict release audit for VIDIMU PTS/CV bundles.")
    parser.add_argument("--finalized-root", type=Path, required=True)
    parser.add_argument("--recording-ids", type=Path, required=True)
    parser.add_argument("--required-gate", required=True)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--inventory-manifest", type=Path)
    parser.add_argument("--license-record", type=Path)
    parser.add_argument("--expected-dataset-id")
    parser.add_argument("--expected-dataset-version")
    parser.add_argument("--inventory-sha256")
    parser.add_argument("--dataset-archive-path")
    parser.add_argument("--dataset-archive-sha256")
    parser.add_argument("--video-archive-path")
    parser.add_argument("--video-archive-sha256")
    parser.add_argument("--license-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        anchor_values = (
            args.expected_dataset_id,
            args.expected_dataset_version,
            args.inventory_sha256,
            args.dataset_archive_path,
            args.dataset_archive_sha256,
            args.video_archive_path,
            args.video_archive_sha256,
            args.license_sha256,
        )
        if any(value is not None for value in anchor_values):
            source_archives = tuple(sorted((
                GateBArchiveAnchor(
                    original_path=args.dataset_archive_path,
                    role="DATASET_ARCHIVE",
                    sha256=args.dataset_archive_sha256,
                ),
                GateBArchiveAnchor(
                    original_path=args.video_archive_path,
                    role="VIDEO_ARCHIVE",
                    sha256=args.video_archive_sha256,
                ),
            ), key=lambda anchor: (
                anchor.original_path, anchor.role, anchor.sha256,
            )))
            gate_b_trust_anchors = GateBTrustAnchors(
                expected_dataset_id=args.expected_dataset_id,
                expected_dataset_version=args.expected_dataset_version,
                inventory_manifest_sha256=args.inventory_sha256,
                source_archives=source_archives,
                license_record_sha256=args.license_sha256,
            )
        else:
            gate_b_trust_anchors = None
        report = audit_vidimu_pts_cv_release(
            args.finalized_root,
            expected_recording_ids=_load_recording_ids(args.recording_ids),
            required_gate=args.required_gate,
            replay_root=args.replay_root,
            source_root=args.source_root,
            inventory_manifest_path=args.inventory_manifest,
            license_record_path=args.license_record,
            gate_b_trust_anchors=gate_b_trust_anchors,
        )
        status = 0
    except Exception as exc:  # noqa: BLE001 - CLI must emit a fail report
        report = {"overall_verdict": "FAIL", "error": str(exc)}
        status = 1
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "GateBArchiveAnchor",
    "GateBTrustAnchors",
    "ReleaseAuditError",
    "audit_vidimu_pts_cv_release",
]
