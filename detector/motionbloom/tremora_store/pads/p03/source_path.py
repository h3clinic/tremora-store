"""The independent source path: read the original file, not the store.

This deliberately does not call the P0.2 replay API, and it does not reuse the
P0.2 stream reader either.  It is a second, minimal implementation of "open
the device file the release published and pull the rows this window covers",
so the comparison tests source parsing against indexed replay rather than one
code path against itself.

The window's bounds are task-local, and this path derives the origin the same
way the release does -- the first row's own ``Time`` -- so it never consults
the store for it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..authority import CANONICAL_CHANNELS, CHANNEL_UNITS, TIME_CHANNEL
from .contract import FAMILY_AXES, SENSOR_FAMILY_ACCELEROMETER, SENSOR_FAMILY_GYROSCOPE

MOVEMENT_DIRECTORY = "movement"
_SCALE = Decimal(10) ** 12

SOURCE_READ_OK = "SOURCE_READ_OK"
SOURCE_FILE_ABSENT = "SOURCE_FILE_ABSENT"
SOURCE_HASH_MISMATCH = "SOURCE_HASH_MISMATCH"
SOURCE_DECLARATION_UNUSABLE = "SOURCE_DECLARATION_UNUSABLE"
SOURCE_ROW_MALFORMED = "SOURCE_ROW_MALFORMED"


class SourcePathError(ValueError):
    """Raised when the original file cannot be read as the release declares."""


@dataclass(frozen=True, slots=True)
class SourceWindow:
    """The rows the original file holds inside one window's bounds."""

    stream_id: str
    source_asset_sha256: str
    ordinals: tuple[int, ...]
    time_tokens: tuple[str, ...]
    times_ps: tuple[int, ...]
    channels: dict[str, tuple[float, ...]]
    status: str = SOURCE_READ_OK

    @property
    def row_count(self) -> int:
        return len(self.ordinals)

    def axes(self, family: str) -> list[tuple[float, ...]]:
        return [self.channels[name] for name in FAMILY_AXES[family]]

    def row_identity_sha256(self) -> str:
        """Bind the ordinals and the exact time tokens together."""

        digest = hashlib.sha256()
        digest.update(self.stream_id.encode("utf-8"))
        for ordinal, token in zip(
            self.ordinals, self.time_tokens, strict=True
        ):
            digest.update(b"\x1e")
            digest.update(str(ordinal).encode("ascii"))
            digest.update(b"\x1f")
            digest.update(token.encode("ascii"))
        return digest.hexdigest()


_CANONICAL_TO_COLUMN = {
    "Accelerometer_X": "accelerometer_x",
    "Accelerometer_Y": "accelerometer_y",
    "Accelerometer_Z": "accelerometer_z",
    "Gyroscope_X": "gyroscope_x",
    "Gyroscope_Y": "gyroscope_y",
    "Gyroscope_Z": "gyroscope_z",
}


def _exact_picoseconds(token: str) -> int:
    scaled = Decimal(token) * _SCALE
    if scaled != scaled.to_integral_value():
        raise SourcePathError(f"time token finer than the scale: {token!r}")
    return int(scaled)


def find_declaration(
    observation: Mapping[str, Any], task_name: str, device_location: str
) -> tuple[dict[str, Any], int]:
    """Locate the record and its declared row count in the observation."""

    for session in observation["session"]:
        if str(session["record_name"]) != task_name:
            continue
        for record in session["records"]:
            if str(record["device_location"]) == device_location:
                return dict(record), int(session["rows"])
    raise SourcePathError(
        f"no {device_location} record for {task_name} in the observation")


def read_source_window(
    *,
    release_root: Path,
    participant_id: str,
    task_name: str,
    device_location: str,
    stream_id: str,
    window_start_task_local_ps: int,
    window_end_task_local_ps: int,
    expected_asset_sha256: str | None = None,
) -> SourceWindow:
    """Open the original device file and take the rows this window covers."""

    movement = release_root / MOVEMENT_DIRECTORY
    observation_path = movement / f"observation_{participant_id}.json"
    try:
        observation = json.loads(observation_path.read_bytes().decode("utf-8"))
    except OSError as exc:
        raise SourcePathError(
            f"observation for {participant_id} unreadable") from exc
    record, declared_rows = find_declaration(
        observation, task_name, device_location
    )

    channels = tuple(str(name) for name in record["channels"])
    units = tuple(str(unit) for unit in record["units"])
    if len(channels) != len(units) or set(channels) != set(
        CANONICAL_CHANNELS
    ) or any(
        CHANNEL_UNITS[name] != unit
        for name, unit in zip(channels, units, strict=True)
    ):
        raise SourcePathError(SOURCE_DECLARATION_UNUSABLE)

    path = movement / str(record["file_name"])
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SourcePathError(SOURCE_FILE_ABSENT) from exc
    observed = hashlib.sha256(payload).hexdigest()
    if expected_asset_sha256 is not None and observed != expected_asset_sha256:
        raise SourcePathError(SOURCE_HASH_MISMATCH)

    lines = payload.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if len(lines) != declared_rows:
        raise SourcePathError(
            f"file holds {len(lines)} rows, metadata declares {declared_rows}")

    time_position = channels.index(TIME_CHANNEL)
    positions = {
        _CANONICAL_TO_COLUMN[name]: index
        for index, name in enumerate(channels)
        if name != TIME_CHANNEL
    }

    ordinals: list[int] = []
    tokens: list[str] = []
    times: list[int] = []
    values: dict[str, list[float]] = {name: [] for name in positions}
    origin: int | None = None

    for ordinal, line in enumerate(lines):
        cells = line.split(",")
        if len(cells) != len(channels):
            raise SourcePathError(SOURCE_ROW_MALFORMED)
        picoseconds = _exact_picoseconds(cells[time_position])
        if origin is None:
            origin = picoseconds
        task_local = picoseconds - origin
        if task_local < window_start_task_local_ps:
            continue
        if task_local >= window_end_task_local_ps:
            break
        ordinals.append(ordinal)
        tokens.append(cells[time_position])
        times.append(picoseconds)
        for name, index in positions.items():
            values[name].append(float(cells[index]))

    return SourceWindow(
        stream_id=stream_id,
        source_asset_sha256=observed,
        ordinals=tuple(ordinals),
        time_tokens=tuple(tokens),
        times_ps=tuple(times),
        channels={name: tuple(column) for name, column in values.items()},
    )


def replay_row_identity_sha256(
    stream_id: str, ordinals: Sequence[int], tokens: Sequence[str]
) -> str:
    """The same identity hash, computed from what replay returned."""

    digest = hashlib.sha256()
    digest.update(stream_id.encode("utf-8"))
    for ordinal, token in zip(ordinals, tokens, strict=True):
        digest.update(b"\x1e")
        digest.update(str(ordinal).encode("ascii"))
        digest.update(b"\x1f")
        digest.update(token.encode("ascii"))
    return digest.hexdigest()


FAMILY_NAMES = (SENSOR_FAMILY_GYROSCOPE, SENSOR_FAMILY_ACCELEROMETER)


__all__ = [
    "FAMILY_NAMES",
    "MOVEMENT_DIRECTORY",
    "SOURCE_DECLARATION_UNUSABLE",
    "SOURCE_FILE_ABSENT",
    "SOURCE_HASH_MISMATCH",
    "SOURCE_READ_OK",
    "SOURCE_ROW_MALFORMED",
    "SourcePathError",
    "SourceWindow",
    "find_declaration",
    "read_source_window",
    "replay_row_identity_sha256",
]
