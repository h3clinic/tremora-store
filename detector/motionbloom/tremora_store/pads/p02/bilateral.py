"""Bilateral task and window co-indexing, without a synchronization claim.

The release establishes that two wrist streams belong to the same participant
and task.  It establishes nothing about a common hardware clock, so every row
these builders emit carries ``cross_wrist_clock_alignment = UNRESOLVED`` and
``sample_level_fusion_allowed = false``.

Windows are paired by their task-local grid offset, which is a shared
coordinate both streams were indexed on -- not by sample identity.  One query
can therefore retrieve both wrists for the same four seconds of a task without
asserting that left sample 400 is simultaneous with right sample 400.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .contract import (
    BILATERAL_PAIR_STATUS,
    BILATERAL_PAIRING_AUTHORITY,
    CROSS_WRIST_CLOCK_ALIGNMENT,
    SAMPLE_LEVEL_BILATERAL_FUSION_ALLOWED,
)
from .windows import Window

LEFT = "LeftWrist"
RIGHT = "RightWrist"

PAIR_COMPLETE = "PAIR_COMPLETE"
PAIR_MISSING_STREAM = "PAIR_MISSING_STREAM"


class BilateralError(ValueError):
    """Raised when a bilateral artifact would overstate its authority."""


@dataclass(frozen=True, slots=True)
class BilateralTask:
    """One assessment's two wrist streams."""

    assessment_id: str
    participant_id: str
    task_name: str
    left_stream_id: str
    right_stream_id: str
    pair_status: str = PAIR_COMPLETE

    def as_record(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "participant_id": self.participant_id,
            "task_name": self.task_name,
            "left_stream_id": self.left_stream_id,
            "right_stream_id": self.right_stream_id,
            "pairing_authority": BILATERAL_PAIRING_AUTHORITY,
            "cross_wrist_clock_alignment": CROSS_WRIST_CLOCK_ALIGNMENT,
            "sample_level_fusion_allowed": (
                SAMPLE_LEVEL_BILATERAL_FUSION_ALLOWED
            ),
            "pair_status": self.pair_status,
        }


@dataclass(frozen=True, slots=True)
class BilateralWindowPair:
    """Two windows at the same task-local offset, co-indexed not aligned."""

    bilateral_window_pair_id: str
    assessment_id: str
    participant_id: str
    task_name: str
    window_start_task_local_ps: int
    window_end_task_local_ps: int
    left_window_id: str
    right_window_id: str
    split_group_id: str
    outer_fold: int

    def as_record(self) -> dict[str, Any]:
        return {
            "bilateral_window_pair_id": self.bilateral_window_pair_id,
            "assessment_id": self.assessment_id,
            "participant_id": self.participant_id,
            "task_name": self.task_name,
            "window_start_task_local_ps": self.window_start_task_local_ps,
            "window_end_task_local_ps": self.window_end_task_local_ps,
            "left_window_id": self.left_window_id,
            "right_window_id": self.right_window_id,
            "pairing_status": BILATERAL_PAIR_STATUS,
            "pairing_authority": BILATERAL_PAIRING_AUTHORITY,
            # Stated on the row itself rather than left to a sibling table:
            # a published pair must carry the claim it does not make.
            "cross_wrist_clock_alignment": CROSS_WRIST_CLOCK_ALIGNMENT,
            "sample_level_fusion_allowed": (
                SAMPLE_LEVEL_BILATERAL_FUSION_ALLOWED
            ),
            "split_group_id": self.split_group_id,
            "outer_fold": self.outer_fold,
        }


def build_bilateral_tasks(
    streams_by_assessment: Mapping[str, Mapping[str, str]],
    *,
    participant_of: Mapping[str, str],
    task_of: Mapping[str, str],
) -> tuple[BilateralTask, ...]:
    """One row per assessment that declares both wrists."""

    tasks: list[BilateralTask] = []
    for assessment_id in sorted(streams_by_assessment):
        wrists = streams_by_assessment[assessment_id]
        left = wrists.get(LEFT)
        right = wrists.get(RIGHT)
        if left is None or right is None:
            tasks.append(BilateralTask(
                assessment_id=assessment_id,
                participant_id=participant_of[assessment_id],
                task_name=task_of[assessment_id],
                left_stream_id=left or "",
                right_stream_id=right or "",
                pair_status=PAIR_MISSING_STREAM,
            ))
            continue
        tasks.append(BilateralTask(
            assessment_id=assessment_id,
            participant_id=participant_of[assessment_id],
            task_name=task_of[assessment_id],
            left_stream_id=left,
            right_stream_id=right,
        ))
    return tuple(tasks)


def build_bilateral_window_pairs(
    windows: Iterable[Window],
) -> tuple[BilateralWindowPair, ...]:
    """Pair windows that share an assessment and a task-local offset."""

    by_offset: dict[tuple[str, int], dict[str, Window]] = defaultdict(dict)
    for window in windows:
        key = (window.assessment_id, window.window_start_task_local_ps)
        if window.device_location in by_offset[key]:
            raise BilateralError(
                f"two {window.device_location} windows at {key!r}")
        by_offset[key][window.device_location] = window

    pairs: list[BilateralWindowPair] = []
    for assessment_id, start in sorted(by_offset):
        sides = by_offset[(assessment_id, start)]
        left = sides.get(LEFT)
        right = sides.get(RIGHT)
        if left is None or right is None:
            # A window that exists on one wrist only is not a pair.  It stays
            # in the window index and simply has no partner.
            continue
        pairs.append(BilateralWindowPair(
            bilateral_window_pair_id=f"{assessment_id}@{start}",
            assessment_id=assessment_id,
            participant_id=left.participant_id,
            task_name=left.task_name,
            window_start_task_local_ps=start,
            window_end_task_local_ps=left.window_end_task_local_ps,
            left_window_id=left.window_id,
            right_window_id=right.window_id,
            split_group_id=left.split_group_id,
            outer_fold=left.outer_fold,
        ))
    return tuple(pairs)


def assert_no_sample_level_claim(records: Iterable[Mapping[str, Any]]) -> None:
    """Refuse any published bilateral row that claims sample-level fusion."""

    for record in records:
        if record.get("sample_level_fusion_allowed") is not False:
            raise BilateralError(
                "a bilateral row claims sample-level fusion")
        if record.get("cross_wrist_clock_alignment") != (
            CROSS_WRIST_CLOCK_ALIGNMENT
        ):
            raise BilateralError(
                "a bilateral row claims a resolved cross-wrist clock")


__all__ = [
    "LEFT",
    "PAIR_COMPLETE",
    "PAIR_MISSING_STREAM",
    "RIGHT",
    "BilateralError",
    "BilateralTask",
    "BilateralWindowPair",
    "assert_no_sample_level_claim",
    "build_bilateral_tasks",
    "build_bilateral_window_pairs",
]
