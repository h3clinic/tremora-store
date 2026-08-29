"""Frozen workload and audit-subset selection.

Two sets, both fixed before any spectrum is examined.

The **workload set** is one canonical window per stream that has at least one
valid P0.2.1 window: the window whose task-local midpoint lies closest to the
stream's own midpoint, ties broken by the earlier start.  358 of the 10,318
streams hold no valid window, so the set is 9,960 -- one per *eligible*
stream, not one per stream.

The **audit subset** is stratified by task, wrist, fold, sample count and
gap adjacency, ordered inside each stratum by a keyed SHA-256 digest rather
than an RNG, and capped per stratum.  On the materialized corpus the key
populates 862 strata, so a cap of ten selects 6,077 windows.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contract import (
    AUDIT_SELECTION_SEED,
    AUDIT_SELECTION_VERSION,
    AUDIT_WINDOWS_PER_STRATUM,
    GAP_ADJACENT,
    INTERIOR,
    REAL_BREAK_REASONS,
    WORKLOAD_SELECTION_VERSION,
)

_AUDIT_DOMAIN = "tremora-pads-p03-audit-subset-1"

SELECTED_CLOSEST_TO_STREAM_MIDPOINT = "CLOSEST_TO_STREAM_MIDPOINT"


class SelectionError(ValueError):
    """Raised when selection is asked of an inconsistent index."""


@dataclass(frozen=True, slots=True)
class WindowFacts:
    """The P0.2.1 window facts P0.3 selects and reasons over."""

    window_id: str
    stream_id: str
    participant_id: str
    assessment_id: str
    task_name: str
    device_location: str
    outer_fold: int
    segment_id: str
    window_start_task_local_ps: int
    window_end_task_local_ps: int
    first_sample_ordinal: int
    last_sample_ordinal: int
    sample_count: int
    dt_ref_ps: int
    coverage_fraction: float
    effective_rate_hz: float
    window_status: str
    gap_adjacent_status: str

    @property
    def midpoint_ps(self) -> int:
        return (
            self.window_start_task_local_ps + self.window_end_task_local_ps
        ) // 2

    @property
    def stratum_id(self) -> str:
        return "|".join((
            self.task_name,
            self.device_location,
            str(self.outer_fold),
            str(self.sample_count),
            self.gap_adjacent_status,
        ))


def gap_adjacency(
    windows: Sequence[Mapping[str, Any]],
    segments: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Label each window interior or adjacent to a real segment break.

    A window is adjacent when it is the first window of a segment that begins
    at a real break, or the last window of a segment that ends at one.  The
    stream simply starting or ending is not a break.
    """

    extremes: dict[str, list[int]] = {}
    for window in windows:
        segment_id = str(window["segment_id"])
        start = int(window["window_start_task_local_ps"])
        bounds = extremes.setdefault(segment_id, [start, start])
        bounds[0] = min(bounds[0], start)
        bounds[1] = max(bounds[1], start)

    labels: dict[str, str] = {}
    for window in windows:
        segment_id = str(window["segment_id"])
        segment = segments.get(segment_id)
        if segment is None:
            raise SelectionError(
                f"{window['window_id']} names no known segment")
        start = int(window["window_start_task_local_ps"])
        low, high = extremes[segment_id]
        adjacent = (
            str(segment["break_reason_before"]) in REAL_BREAK_REASONS
            and start == low
        ) or (
            str(segment["break_reason_after"]) in REAL_BREAK_REASONS
            and start == high
        )
        labels[str(window["window_id"])] = (
            GAP_ADJACENT if adjacent else INTERIOR
        )
    return labels


def window_facts(
    windows: Sequence[Mapping[str, Any]],
    segments: Mapping[str, Mapping[str, Any]],
) -> tuple[WindowFacts, ...]:
    """Attach gap adjacency and freeze the facts selection reasons over."""

    labels = gap_adjacency(windows, segments)
    return tuple(
        WindowFacts(
            window_id=str(window["window_id"]),
            stream_id=str(window["stream_id"]),
            participant_id=str(window["participant_id"]),
            assessment_id=str(window["assessment_id"]),
            task_name=str(window["task_name"]),
            device_location=str(window["device_location"]),
            outer_fold=int(window["outer_fold"]),
            segment_id=str(window["segment_id"]),
            window_start_task_local_ps=int(
                window["window_start_task_local_ps"]
            ),
            window_end_task_local_ps=int(window["window_end_task_local_ps"]),
            first_sample_ordinal=int(window["first_sample_ordinal"]),
            last_sample_ordinal=int(window["last_sample_ordinal"]),
            sample_count=int(window["sample_count"]),
            dt_ref_ps=int(window["dt_ref_ps"]),
            coverage_fraction=float(window["coverage_fraction"]),
            effective_rate_hz=float(window["effective_rate_hz"]),
            window_status=str(window["window_status"]),
            gap_adjacent_status=labels[str(window["window_id"])],
        )
        for window in windows
    )


def stream_midpoints_ps(
    streams: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Each stream's task-local midpoint, from its own stored extent."""

    midpoints: dict[str, int] = {}
    for stream in streams:
        origin = int(stream["source_time_origin_ps"])
        end = int(stream["source_time_end_ps"])
        midpoints[str(stream["stream_id"])] = (end - origin) // 2
    return midpoints


def select_workload(
    facts: Sequence[WindowFacts], midpoints: Mapping[str, int]
) -> tuple[WindowFacts, ...]:
    """One canonical window per stream that has a valid window."""

    by_stream: dict[str, list[WindowFacts]] = defaultdict(list)
    for window in facts:
        by_stream[window.stream_id].append(window)

    chosen: list[WindowFacts] = []
    for stream_id in sorted(by_stream):
        midpoint = midpoints.get(stream_id)
        if midpoint is None:
            raise SelectionError(f"{stream_id} has no stored extent")
        candidates = by_stream[stream_id]
        best = min(
            candidates,
            key=lambda window: (
                abs(window.midpoint_ps - midpoint),
                window.window_start_task_local_ps,
            ),
        )
        chosen.append(best)
    return tuple(chosen)


def audit_key(
    window_id: str,
    *,
    seed: int = AUDIT_SELECTION_SEED,
    algorithm_version: str = AUDIT_SELECTION_VERSION,
) -> str:
    """The keyed digest that orders one window inside its stratum."""

    payload = "\x1f".join(
        (_AUDIT_DOMAIN, algorithm_version, str(seed), window_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_audit_subset(
    facts: Sequence[WindowFacts],
    *,
    per_stratum: int = AUDIT_WINDOWS_PER_STRATUM,
    seed: int = AUDIT_SELECTION_SEED,
    algorithm_version: str = AUDIT_SELECTION_VERSION,
) -> tuple[WindowFacts, ...]:
    """Up to ``per_stratum`` windows from each populated stratum."""

    if per_stratum < 1:
        raise SelectionError("at least one window per stratum is required")
    strata: dict[str, list[WindowFacts]] = defaultdict(list)
    for window in facts:
        strata[window.stratum_id].append(window)

    chosen: list[WindowFacts] = []
    for stratum in sorted(strata):
        members = sorted(
            strata[stratum],
            key=lambda window: (
                audit_key(
                    window.window_id, seed=seed,
                    algorithm_version=algorithm_version,
                ),
                window.window_id,
            ),
        )
        chosen.extend(members[:per_stratum])
    chosen.sort(key=lambda window: window.window_id)
    return tuple(chosen)


def selection_coverage(
    subset: Sequence[WindowFacts],
) -> dict[str, Any]:
    """What the frozen subset actually covers, for the release report."""

    return {
        "tasks": sorted({window.task_name for window in subset}),
        "device_locations": sorted(
            {window.device_location for window in subset}
        ),
        "outer_folds": sorted({window.outer_fold for window in subset}),
        "sample_counts": sorted({window.sample_count for window in subset}),
        "gap_adjacent_windows": sum(
            1 for window in subset
            if window.gap_adjacent_status == GAP_ADJACENT
        ),
        "interior_windows": sum(
            1 for window in subset if window.gap_adjacent_status == INTERIOR
        ),
        "populated_strata": len({window.stratum_id for window in subset}),
        "selection_version": WORKLOAD_SELECTION_VERSION,
        "audit_selection_version": AUDIT_SELECTION_VERSION,
        "audit_selection_seed": AUDIT_SELECTION_SEED,
        "audit_windows_per_stratum": AUDIT_WINDOWS_PER_STRATUM,
    }


__all__ = [
    "SELECTED_CLOSEST_TO_STREAM_MIDPOINT",
    "SelectionError",
    "WindowFacts",
    "audit_key",
    "gap_adjacency",
    "select_audit_subset",
    "select_workload",
    "selection_coverage",
    "stream_midpoints_ps",
    "window_facts",
]
