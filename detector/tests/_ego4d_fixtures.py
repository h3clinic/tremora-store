"""Synthetic Ego4D releases for E4D-P0.1 tests.

Every fixture writes a real metadata snapshot, asset manifest and normalized
IMU CSV, so the tests exercise the same readers the release path uses.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from motionbloom.tremora_store.ego4d.authority import EGO4D_IMU_COLUMNS

#: The interval in Ego4D's own documented example, about 201 Hz.
DOCUMENTED_DT_MS = 4.975124378109452


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def imu_csv(rows: Sequence[Sequence[str]], *, header: Sequence[str] | None = None,
            terminator: str = "\n", trailing: bool = True) -> bytes:
    columns = tuple(header) if header is not None else EGO4D_IMU_COLUMNS
    records = [",".join(columns)]
    records.extend(",".join(row) for row in rows)
    text = terminator.join(records)
    if trailing:
        text += terminator
    return text.encode("utf-8")


def clean_rows(
    *,
    component_idx: int = 0,
    count: int = 40,
    start_ms: float = 0.0,
    dt_ms: float = DOCUMENTED_DT_MS,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for index in range(count):
        time = start_ms + index * dt_ms
        rows.append([
            str(component_idx),
            repr(time),
            repr(time),
            "0.1", "0.2", "0.3",
            "9.8", "0.0", "0.1",
        ])
    return rows


@dataclass
class VideoSpec:
    """One synthetic video: its metadata and the IMU rows written for it."""

    video_uid: str
    duration_ms: float = 10_000.0
    capture_device_group: str = "GROUP_A"
    component_count: int = 1
    components_with_imu: int | None = None
    rows: list[list[str]] = field(default_factory=list)
    imu_payload: bytes | None = None
    declared_imu_sha256: str | None = None
    write_imu_file: bool = True
    imu_relative_path: str | None = None
    video_relative_path: str | None = None

    def payload(self) -> bytes:
        if self.imu_payload is not None:
            return self.imu_payload
        return imu_csv(self.rows or clean_rows())


def build_release(root: Path, specs: Sequence[VideoSpec]) -> tuple[Path, Path, Path]:
    """Write a synthetic release and return its metadata, IMU and video roots."""

    metadata_root = root / "metadata"
    imu_root = root / "imu"
    video_root = root / "video"
    for directory in (metadata_root, imu_root, video_root):
        directory.mkdir(parents=True, exist_ok=True)

    videos: list[dict[str, object]] = []
    assets: list[dict[str, object]] = []
    for spec in specs:
        with_imu = (
            spec.component_count
            if spec.components_with_imu is None
            else spec.components_with_imu
        )
        videos.append({
            "video_uid": spec.video_uid,
            "canonical_video_duration_ms": spec.duration_ms,
            "video_stream_start_ms": 0.0,
            "video_stream_end_ms": spec.duration_ms,
            "capture_device_group": spec.capture_device_group,
            "components": [
                {
                    "component_idx": index,
                    "component_start_in_canonical_ms": 0.0,
                    "component_end_in_canonical_ms": spec.duration_ms,
                    "has_imu": index < with_imu,
                }
                for index in range(spec.component_count)
            ],
        })
        payload = spec.payload()
        relative = spec.imu_relative_path or f"{spec.video_uid}.csv"
        if spec.write_imu_file:
            target = imu_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        assets.append({
            "video_uid": spec.video_uid,
            "component_idx": 0,
            "imu_relative_path": relative,
            "canonical_video_relative_path": (
                spec.video_relative_path or f"{spec.video_uid}.mp4"
            ),
            "imu_asset_sha256": (
                spec.declared_imu_sha256 or sha256_bytes(payload)
            ),
            "video_component_asset_sha256": sha256_bytes(
                f"component:{spec.video_uid}".encode()
            ),
            "canonical_video_asset_sha256": sha256_bytes(
                f"canonical:{spec.video_uid}".encode()
            ),
        })

    (metadata_root / "ego4d_metadata_snapshot.json").write_text(
        json.dumps({"videos": videos}, indent=2), encoding="utf-8"
    )
    (metadata_root / "ego4d_asset_manifest.json").write_text(
        json.dumps({"assets": assets}, indent=2), encoding="utf-8"
    )
    return metadata_root, imu_root, video_root


def frame_times_for(duration_ms: float, *, fps: float = 30.0) -> tuple[float, ...]:
    """A decoded timeline that starts at canonical zero and spans the video."""

    interval = 1000.0 / fps
    count = max(2, round(duration_ms / interval) + 1)
    return tuple(
        min(index * interval, duration_ms) for index in range(count)
    )


def decoder_for(specs: Sequence[VideoSpec]):
    """Return a decode callable that answers from the spec, not from media."""

    durations = {spec.video_uid: spec.duration_ms for spec in specs}
    by_video_path = {
        (spec.video_relative_path or f"{spec.video_uid}.mp4"): spec.video_uid
        for spec in specs
    }

    def decode(path: Path, expected_sha256: str) -> tuple[float, ...]:
        video_uid = by_video_path.get(path.name, path.stem)
        return frame_times_for(durations.get(video_uid, 0.0))

    return decode
