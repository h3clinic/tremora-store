"""Cadence-relative coverage and continuity for Ego4D components.

Paired overlap is the canonical time *actually covered* by authority-eligible
IMU rows, clamped to the video, measured relative to each component's own
cadence.  Three weaker definitions were tried and all fail the same way:

* the video's duration lets a ten-minute video whose IMU covers two hundred
  milliseconds contribute ten minutes;
* the first-to-last span lets a two-row file -- one sample at the start, one at
  the end -- contribute a full hour;
* a flat 100 ms bridge is better but still too generous.  Ego4D's own
  documented example is spaced ~4.975 ms, about 201 Hz, so 100 ms spans roughly
  twenty expected sample intervals and a hole twenty samples wide would count
  as observed data.

Under sample support, two samples an hour apart contribute about two sample
intervals.  Continuity is a separate question from coverage and has its own
threshold.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import median

from .authority import (
    CONTINUITY_ABSOLUTE_CAP_MS,
    CONTINUITY_MULTIPLIER,
    MINIMUM_CADENCE_DELTAS,
)

MS_PER_HOUR = 3_600_000.0


class Ego4DCoverageError(ValueError):
    """Raised when coverage is asked of a sequence that is not ordered."""


@dataclass(frozen=True, slots=True)
class ComponentCoverage:
    """One component's cadence, covered time and contiguous segments."""

    component_idx: int
    eligible_sample_count: int
    reference_interval_ms: float | None
    coverage_ms: float
    segment_count: int
    continuity_threshold_ms: float | None
    intervals: tuple[tuple[float, float], ...]
    coverage_status: str


COVERAGE_MEASURED = "COVERAGE_MEASURED"
COVERAGE_NO_CADENCE = "COVERAGE_NO_CADENCE"
COVERAGE_NO_ELIGIBLE_ROWS = "COVERAGE_NO_ELIGIBLE_ROWS"


def assert_strictly_increasing(times: Sequence[float]) -> None:
    """Eligible canonical times are strictly increasing by construction."""

    for earlier, later in pairwise(times):
        if not later > earlier:
            raise Ego4DCoverageError(
                "eligible canonical timestamps are not strictly increasing")


def estimate_reference_interval(times: Sequence[float]) -> float | None:
    """Return the median positive interval, or ``None`` if it is not a cadence.

    Fewer than :data:`MINIMUM_CADENCE_DELTAS` positive finite deltas is not a
    cadence, and the component then reports no coverage at all rather than
    borrowing a rate from somewhere else.
    """

    deltas = [
        later - earlier
        for earlier, later in pairwise(times)
        if math.isfinite(later - earlier) and later - earlier > 0.0
    ]
    if len(deltas) < MINIMUM_CADENCE_DELTAS:
        return None
    reference = float(median(deltas))
    return reference if reference > 0.0 else None


def continuity_threshold_ms(
    reference_interval_ms: float,
    *,
    multiplier: float = CONTINUITY_MULTIPLIER,
    absolute_cap_ms: float = CONTINUITY_ABSOLUTE_CAP_MS,
) -> float:
    """Return the gap above which a contiguous segment breaks.

    The multiplier is project policy, not a fact Ego4D supplies.  The absolute
    cap survives only so a pathologically low-rate component cannot bridge an
    arbitrarily large interval.
    """

    return min(absolute_cap_ms, multiplier * reference_interval_ms)


def contiguous_segments(
    times: Sequence[float],
    *,
    reference_interval_ms: float,
    multiplier: float = CONTINUITY_MULTIPLIER,
    absolute_cap_ms: float = CONTINUITY_ABSOLUTE_CAP_MS,
) -> tuple[tuple[int, int], ...]:
    """Return ``(first, last)`` index pairs for each contiguous run."""

    if not times:
        return ()
    threshold = continuity_threshold_ms(
        reference_interval_ms,
        multiplier=multiplier,
        absolute_cap_ms=absolute_cap_ms,
    )
    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(times)):
        if times[index] - times[index - 1] > threshold:
            segments.append((start, index - 1))
            start = index
    segments.append((start, len(times) - 1))
    return tuple(segments)


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Merge overlapping half-open supports into a disjoint union."""

    ordered = sorted(
        (low, high) for low, high in intervals if high > low
    )
    merged: list[tuple[float, float]] = []
    for low, high in ordered:
        if merged and low <= merged[-1][1]:
            previous_low, previous_high = merged[-1]
            merged[-1] = (previous_low, max(previous_high, high))
        else:
            merged.append((low, high))
    return tuple(merged)


def union_length(intervals: Iterable[tuple[float, float]]) -> float:
    """Return the total length of a disjoint or overlapping interval set."""

    return sum(high - low for low, high in merge_intervals(intervals))


def sample_supports(
    times: Sequence[float],
    *,
    reference_interval_ms: float,
    clamp_low: float,
    clamp_high: float,
) -> tuple[tuple[float, float], ...]:
    """Return each sample's support, clamped to the video interval."""

    half = reference_interval_ms / 2.0
    supports: list[tuple[float, float]] = []
    for time in times:
        low = max(clamp_low, time - half)
        high = min(clamp_high, time + half)
        if high > low:
            supports.append((low, high))
    return merge_intervals(supports)


def component_coverage(
    component_idx: int,
    times: Sequence[float],
    *,
    clamp_low: float,
    clamp_high: float,
    multiplier: float = CONTINUITY_MULTIPLIER,
    absolute_cap_ms: float = CONTINUITY_ABSOLUTE_CAP_MS,
) -> ComponentCoverage:
    """Measure one component's covered canonical time and continuity."""

    assert_strictly_increasing(times)
    if not times:
        return ComponentCoverage(
            component_idx=component_idx,
            eligible_sample_count=0,
            reference_interval_ms=None,
            coverage_ms=0.0,
            segment_count=0,
            continuity_threshold_ms=None,
            intervals=(),
            coverage_status=COVERAGE_NO_ELIGIBLE_ROWS,
        )
    reference = estimate_reference_interval(times)
    if reference is None:
        return ComponentCoverage(
            component_idx=component_idx,
            eligible_sample_count=len(times),
            reference_interval_ms=None,
            coverage_ms=0.0,
            segment_count=0,
            continuity_threshold_ms=None,
            intervals=(),
            coverage_status=COVERAGE_NO_CADENCE,
        )
    supports = sample_supports(
        times,
        reference_interval_ms=reference,
        clamp_low=clamp_low,
        clamp_high=clamp_high,
    )
    segments = contiguous_segments(
        times,
        reference_interval_ms=reference,
        multiplier=multiplier,
        absolute_cap_ms=absolute_cap_ms,
    )
    return ComponentCoverage(
        component_idx=component_idx,
        eligible_sample_count=len(times),
        reference_interval_ms=reference,
        coverage_ms=union_length(supports),
        segment_count=len(segments),
        continuity_threshold_ms=continuity_threshold_ms(
            reference, multiplier=multiplier, absolute_cap_ms=absolute_cap_ms
        ),
        intervals=supports,
        coverage_status=COVERAGE_MEASURED,
    )


def video_coverage_ms(
    coverages: Iterable[ComponentCoverage],
) -> float:
    """Union component supports so overlapping components cannot double count."""

    intervals: list[tuple[float, float]] = []
    for coverage in coverages:
        intervals.extend(coverage.intervals)
    return union_length(intervals)


def hours(coverage_ms: float) -> float:
    return coverage_ms / MS_PER_HOUR


__all__ = [
    "COVERAGE_MEASURED",
    "COVERAGE_NO_CADENCE",
    "COVERAGE_NO_ELIGIBLE_ROWS",
    "MS_PER_HOUR",
    "ComponentCoverage",
    "Ego4DCoverageError",
    "assert_strictly_increasing",
    "component_coverage",
    "contiguous_segments",
    "continuity_threshold_ms",
    "estimate_reference_interval",
    "hours",
    "merge_intervals",
    "sample_supports",
    "union_length",
    "video_coverage_ms",
]
