"""Strict readers for the Ego4D metadata snapshot and asset manifest.

The audit does not reach into Ego4D itself: the licence is signed out of band
and the assets arrive as an operator-supplied, hash-pinned snapshot.  What this
module fixes is the *shape* that snapshot must have, so a malformed or
ambiguous release closes the gate instead of quietly reshaping itself.

Every rejection here is evidence about a release, never an availability
notice.  Absence of the file is the only thing that blocks.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

TIMELINE_PRESENT = "TIMELINE_PRESENT"
TIMELINE_NO_IMU_BY_SOURCE = "TIMELINE_NO_IMU_BY_SOURCE"
TIMELINE_INVALID_INTERVAL = "TIMELINE_INVALID_INTERVAL"

_VIDEO_KEYS = frozenset({
    "canonical_video_duration_ms",
    "capture_device_group",
    "components",
    "video_stream_end_ms",
    "video_stream_start_ms",
    "video_uid",
})
_COMPONENT_KEYS = frozenset({
    "component_end_in_canonical_ms",
    "component_idx",
    "component_start_in_canonical_ms",
    "has_imu",
})


class Ego4DMetadataError(ValueError):
    """Raised when a metadata snapshot or asset manifest is unusable."""


@dataclass(frozen=True, slots=True)
class ComponentTimeline:
    """One video component's placement on the canonical timeline."""

    component_idx: int
    component_start_in_canonical_ms: float
    component_end_in_canonical_ms: float
    has_imu: bool
    timeline_status: str


@dataclass(frozen=True, slots=True)
class VideoTimeline:
    """One canonical video and the components placed inside it."""

    video_uid: str
    canonical_video_duration_ms: float
    video_stream_start_ms: float
    video_stream_end_ms: float
    capture_device_group: str
    components: tuple[ComponentTimeline, ...]

    def component(self, component_idx: int) -> ComponentTimeline | None:
        for component in self.components:
            if component.component_idx == component_idx:
                return component
        return None

    @property
    def components_expected(self) -> int:
        return len(self.components)

    @property
    def components_with_imu(self) -> int:
        return sum(1 for component in self.components if component.has_imu)


@dataclass(frozen=True, slots=True)
class MetadataSnapshot:
    """The frozen metadata the subset selection is a pure function of."""

    videos: tuple[VideoTimeline, ...]
    metadata_snapshot_sha256: str

    def video(self, video_uid: str) -> VideoTimeline | None:
        for video in self.videos:
            if video.video_uid == video_uid:
                return video
        return None


def _require(mapping: Any, key: str, kinds: tuple[type, ...], where: str) -> Any:
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise Ego4DMetadataError(f"{where} is missing {key!r}")
    value = mapping[key]
    if isinstance(value, bool) and bool not in kinds:
        raise Ego4DMetadataError(f"{where}.{key} has the wrong type")
    if not isinstance(value, kinds):
        raise Ego4DMetadataError(f"{where}.{key} has the wrong type")
    return value


def _number(mapping: Any, key: str, where: str) -> float:
    value = _require(mapping, key, (int, float), where)
    number = float(value)
    if not math.isfinite(number):
        raise Ego4DMetadataError(f"{where}.{key} is not finite")
    return number


def _reject_unknown_keys(
    mapping: Mapping[str, Any], allowed: frozenset[str], where: str
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise Ego4DMetadataError(f"{where} carries unknown keys {unknown!r}")


def _component(payload: Any, where: str) -> ComponentTimeline:
    if not isinstance(payload, Mapping):
        raise Ego4DMetadataError(f"{where} is not an object")
    _reject_unknown_keys(payload, _COMPONENT_KEYS, where)
    component_idx = _require(payload, "component_idx", (int,), where)
    if component_idx < 0:
        raise Ego4DMetadataError(f"{where}.component_idx is negative")
    start = _number(payload, "component_start_in_canonical_ms", where)
    end = _number(payload, "component_end_in_canonical_ms", where)
    has_imu = bool(_require(payload, "has_imu", (bool,), where))
    if end < start:
        status = TIMELINE_INVALID_INTERVAL
    elif not has_imu:
        status = TIMELINE_NO_IMU_BY_SOURCE
    else:
        status = TIMELINE_PRESENT
    return ComponentTimeline(
        component_idx=int(component_idx),
        component_start_in_canonical_ms=start,
        component_end_in_canonical_ms=end,
        has_imu=has_imu,
        timeline_status=status,
    )


def _video(payload: Any, where: str) -> VideoTimeline:
    if not isinstance(payload, Mapping):
        raise Ego4DMetadataError(f"{where} is not an object")
    _reject_unknown_keys(payload, _VIDEO_KEYS, where)
    video_uid = _require(payload, "video_uid", (str,), where)
    if not video_uid:
        raise Ego4DMetadataError(f"{where}.video_uid is empty")
    duration = _number(payload, "canonical_video_duration_ms", where)
    if duration < 0:
        raise Ego4DMetadataError(f"{where}.canonical_video_duration_ms < 0")
    components_payload = _require(payload, "components", (list,), where)
    components = tuple(
        _component(item, f"{where}.components[{index}]")
        for index, item in enumerate(components_payload)
    )
    seen = [component.component_idx for component in components]
    if len(set(seen)) != len(seen):
        raise Ego4DMetadataError(f"{where} repeats a component_idx")
    return VideoTimeline(
        video_uid=str(video_uid),
        canonical_video_duration_ms=duration,
        video_stream_start_ms=_number(payload, "video_stream_start_ms", where),
        video_stream_end_ms=_number(payload, "video_stream_end_ms", where),
        capture_device_group=str(
            _require(payload, "capture_device_group", (str,), where)
        ),
        components=components,
    )


def parse_metadata_snapshot(
    payload: bytes, *, snapshot_sha256: str
) -> MetadataSnapshot:
    """Parse a metadata snapshot, refusing anything ambiguous."""

    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Ego4DMetadataError("metadata snapshot is not valid JSON") from exc
    if not isinstance(document, Mapping):
        raise Ego4DMetadataError("metadata snapshot is not an object")
    videos_payload = _require(document, "videos", (list,), "snapshot")
    videos = tuple(
        _video(item, f"snapshot.videos[{index}]")
        for index, item in enumerate(videos_payload)
    )
    uids = [video.video_uid for video in videos]
    if len(set(uids)) != len(uids):
        raise Ego4DMetadataError("metadata snapshot repeats a video_uid")
    return MetadataSnapshot(
        videos=videos, metadata_snapshot_sha256=snapshot_sha256
    )


@dataclass(frozen=True, slots=True)
class AssetEntry:
    """One ``<video_uid>:<component_idx>`` asset triple."""

    video_uid: str
    component_idx: int
    imu_relative_path: str
    canonical_video_relative_path: str
    imu_asset_sha256: str
    video_component_asset_sha256: str
    canonical_video_asset_sha256: str

    @property
    def key(self) -> str:
        return f"{self.video_uid}:{self.component_idx}"


_ASSET_KEYS = frozenset({
    "canonical_video_asset_sha256",
    "canonical_video_relative_path",
    "component_idx",
    "imu_asset_sha256",
    "imu_relative_path",
    "video_component_asset_sha256",
    "video_uid",
})
_SHA256_LENGTH = 64


def _sha256_field(mapping: Mapping[str, Any], key: str, where: str) -> str:
    value = str(_require(mapping, key, (str,), where))
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise Ego4DMetadataError(f"{where}.{key} is not a lowercase SHA-256")
    return value


def safe_relative_path(candidate: str, *, where: str) -> PurePosixPath:
    """Return ``candidate`` if it stays inside its root, else refuse it.

    An index entry that tries to escape its root is evidence about a release,
    so this raises and the gate closes; it never reports unavailability.
    """

    path = PurePosixPath(candidate)
    if not candidate or path.is_absolute() or path.parts == ():
        raise Ego4DMetadataError(f"{where} is not a relative path")
    if any(part in {"..", ""} for part in path.parts):
        raise Ego4DMetadataError(f"{where} escapes its root")
    return path


def parse_asset_manifest(payload: bytes) -> dict[str, AssetEntry]:
    """Parse the asset manifest keyed ``<video_uid>:<component_idx>``."""

    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Ego4DMetadataError("asset manifest is not valid JSON") from exc
    if not isinstance(document, Mapping):
        raise Ego4DMetadataError("asset manifest is not an object")
    assets_payload = _require(document, "assets", (list,), "manifest")
    entries: dict[str, AssetEntry] = {}
    for index, item in enumerate(assets_payload):
        where = f"manifest.assets[{index}]"
        if not isinstance(item, Mapping):
            raise Ego4DMetadataError(f"{where} is not an object")
        _reject_unknown_keys(item, _ASSET_KEYS, where)
        component_idx = _require(item, "component_idx", (int,), where)
        if component_idx < 0:
            raise Ego4DMetadataError(f"{where}.component_idx is negative")
        relative = str(_require(item, "imu_relative_path", (str,), where))
        safe_relative_path(relative, where=f"{where}.imu_relative_path")
        video_relative = str(
            _require(item, "canonical_video_relative_path", (str,), where)
        )
        safe_relative_path(
            video_relative,
            where=f"{where}.canonical_video_relative_path",
        )
        entry = AssetEntry(
            video_uid=str(_require(item, "video_uid", (str,), where)),
            component_idx=int(component_idx),
            imu_relative_path=relative,
            canonical_video_relative_path=video_relative,
            imu_asset_sha256=_sha256_field(item, "imu_asset_sha256", where),
            video_component_asset_sha256=_sha256_field(
                item, "video_component_asset_sha256", where
            ),
            canonical_video_asset_sha256=_sha256_field(
                item, "canonical_video_asset_sha256", where
            ),
        )
        if entry.key in entries:
            raise Ego4DMetadataError(f"asset manifest repeats {entry.key!r}")
        entries[entry.key] = entry
    return entries


def read_bytes_under_root(root: Path, relative: str, *, where: str) -> bytes:
    """Read one file that must resolve inside ``root``."""

    safe_relative_path(relative, where=where)
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise Ego4DMetadataError(f"{where} escapes its root")
    try:
        return candidate.read_bytes()
    except OSError as exc:
        raise Ego4DMetadataError(f"{where} could not be read") from exc


__all__ = [
    "TIMELINE_INVALID_INTERVAL",
    "TIMELINE_NO_IMU_BY_SOURCE",
    "TIMELINE_PRESENT",
    "AssetEntry",
    "ComponentTimeline",
    "Ego4DMetadataError",
    "MetadataSnapshot",
    "VideoTimeline",
    "parse_asset_manifest",
    "parse_metadata_snapshot",
    "read_bytes_under_root",
    "safe_relative_path",
]
