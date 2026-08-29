"""Deterministic replay of stored PADS samples.

Replay returns exactly the rows the index names, in source order.  It never
interpolates, resamples, filters, normalizes, reorders by value, synthesizes a
timestamp or copies a neighbouring sample to fill a gap.

Task replay returns both wrists of one assessment together with the authority
that says they are protocol-paired and not clock-aligned.  It does not
sample-align them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from .contract import (
    BILATERAL_PAIRING_AUTHORITY,
    CROSS_WRIST_CLOCK_ALIGNMENT,
    SAMPLE_LEVEL_BILATERAL_FUSION_ALLOWED,
    TIMING_AUTHORITY,
)
from .exact_time import format_sensor_value
from .sample_store import read_stream_row_group

_SENSOR_COLUMNS = (
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
)


class ReplayError(LookupError):
    """Raised when the index names something the store does not hold."""


@dataclass(frozen=True, slots=True)
class ReplayedTask:
    """Both wrists of one assessment, retrieved but not aligned."""

    assessment_id: str
    participant_id: str
    task_name: str
    left: pa.Table
    right: pa.Table
    left_stream_id: str
    right_stream_id: str

    @property
    def authority(self) -> dict[str, Any]:
        return {
            "timing_authority": TIMING_AUTHORITY,
            "bilateral_pairing_authority": BILATERAL_PAIRING_AUTHORITY,
            "cross_wrist_clock_alignment": CROSS_WRIST_CLOCK_ALIGNMENT,
            "sample_level_fusion_allowed": (
                SAMPLE_LEVEL_BILATERAL_FUSION_ALLOWED
            ),
        }


def _entry(
    storage_index: Mapping[str, Mapping[str, Any]], stream_id: str
) -> Mapping[str, Any]:
    try:
        return storage_index[stream_id]
    except KeyError as exc:
        raise ReplayError(f"no stored row group for {stream_id!r}") from exc


def replay_stream(
    output_root: Path,
    storage_index: Mapping[str, Mapping[str, Any]],
    stream_id: str,
) -> pa.Table:
    """Return every stored sample of one stream, in source ordinal order."""

    entry = _entry(storage_index, stream_id)
    table = read_stream_row_group(output_root, dict(entry))
    ordinals = table.column("sample_ordinal").to_pylist()
    if ordinals != sorted(ordinals):
        raise ReplayError(f"{stream_id} is not stored in source order")
    if table.num_rows != int(entry["sample_count"]):
        raise ReplayError(f"{stream_id} row count disagrees with its index")
    return table


def replay_task(
    output_root: Path,
    storage_index: Mapping[str, Mapping[str, Any]],
    bilateral_task: Mapping[str, Any],
) -> ReplayedTask:
    """Return both wrists of one assessment plus its authority metadata."""

    left_id = str(bilateral_task["left_stream_id"])
    right_id = str(bilateral_task["right_stream_id"])
    return ReplayedTask(
        assessment_id=str(bilateral_task["assessment_id"]),
        participant_id=str(bilateral_task["participant_id"]),
        task_name=str(bilateral_task["task_name"]),
        left=replay_stream(output_root, storage_index, left_id),
        right=replay_stream(output_root, storage_index, right_id),
        left_stream_id=left_id,
        right_stream_id=right_id,
    )


def replay_window(
    output_root: Path,
    storage_index: Mapping[str, Mapping[str, Any]],
    window: Mapping[str, Any],
) -> pa.Table:
    """Return exactly the rows one window indexes -- no more, no fewer."""

    stream_id = str(window["stream_id"])
    table = replay_stream(output_root, storage_index, stream_id)
    first = int(window["first_sample_ordinal"])
    last = int(window["last_sample_ordinal"])
    entry = _entry(storage_index, stream_id)
    offset = first - int(entry["first_sample_ordinal"])
    length = last - first + 1
    if offset < 0 or offset + length > table.num_rows:
        raise ReplayError(
            f"{window['window_id']} names rows outside its stream")
    sliced = table.slice(offset, length)
    if sliced.num_rows != int(window["sample_count"]):
        raise ReplayError(
            f"{window['window_id']} indexes {window['sample_count']} samples "
            f"but replays {sliced.num_rows}")
    return sliced


def source_bytes_for(table: pa.Table, channel_order: Sequence[str]) -> bytes:
    """Rebuild the source text for the replayed rows, in declared order."""

    tokens = table.column("source_time_token").to_pylist()
    columns = {
        name: table.column(name).to_pylist() for name in _SENSOR_COLUMNS
    }
    canonical = {
        "Time": None,
        "Accelerometer_X": "accelerometer_x",
        "Accelerometer_Y": "accelerometer_y",
        "Accelerometer_Z": "accelerometer_z",
        "Gyroscope_X": "gyroscope_x",
        "Gyroscope_Y": "gyroscope_y",
        "Gyroscope_Z": "gyroscope_z",
    }
    lines: list[str] = []
    for row in range(table.num_rows):
        cells: list[str] = []
        for name in channel_order:
            column = canonical[name]
            cells.append(
                tokens[row] if column is None
                else format_sensor_value(columns[column][row])
            )
        lines.append(",".join(cells))
    return ("\n".join(lines) + "\n").encode("utf-8")


def replay_sha256(table: pa.Table, channel_order: Sequence[str]) -> str:
    """Hash the replayed source text; equals the source asset when exact."""

    return hashlib.sha256(source_bytes_for(table, channel_order)).hexdigest()


__all__ = [
    "ReplayError",
    "ReplayedTask",
    "replay_sha256",
    "replay_stream",
    "replay_task",
    "replay_window",
    "source_bytes_for",
]
