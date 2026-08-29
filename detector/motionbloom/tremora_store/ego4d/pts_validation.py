"""In-memory reconciliation of Ego4D canonical time against decoded PTS.

Everything here is returned, never persisted: persisting a frame-to-IMU
relationship would be a P0.2 index, which P0.1 forbids and gate condition 11
checks for.

Origin agreement is two tests, not one, and the component's timeline status
names which of them failed.  The origin test asks whether the first decoded
frame sits at canonical zero; the span test asks whether the decoded timeline
is as long as the source says the canonical video is.  A span comparison alone
cannot see a shifted origin, because a constant shift cancels out of a span.

The span tolerance is one frame interval plus a slack, with the interval term
capped: uncapped, a two-frame timeline would grant itself a tolerance
proportional to its own length and a one-hour decode would "agree" with a
half-hour video.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import median

from .authority import (
    PTS_ORIGIN_TOLERANCE_MS,
    PTS_SPAN_FRAME_INTERVAL_CAP_MS,
    PTS_SPAN_SLACK_MS,
)

TIMELINE_RECONCILED = "TIMELINE_RECONCILED"
TIMELINE_ORIGIN_DISAGREEMENT = "TIMELINE_ORIGIN_DISAGREEMENT"
TIMELINE_SPAN_DISAGREEMENT = "TIMELINE_SPAN_DISAGREEMENT"
TIMELINE_ORIGIN_AND_SPAN_DISAGREEMENT = (
    "TIMELINE_ORIGIN_AND_SPAN_DISAGREEMENT"
)
TIMELINE_INSUFFICIENT_FRAMES = "TIMELINE_INSUFFICIENT_FRAMES"

_NS_PER_MS = 1_000_000.0


class Ego4DPtsValidationError(ValueError):
    """Raised when a decoded timeline cannot be reconciled at all."""


@dataclass(frozen=True, slots=True)
class PtsTimelineReconciliation:
    """One video's origin and span agreement, held in memory only."""

    frame_count: int
    first_frame_ms: float | None
    last_frame_ms: float | None
    decoded_span_ms: float | None
    canonical_video_duration_ms: float
    frame_interval_ms: float | None
    origin_offset_ms: float | None
    origin_tolerance_ms: float
    span_difference_ms: float | None
    span_tolerance_ms: float | None
    origin_agrees: bool
    span_agrees: bool
    timeline_status: str


@dataclass(frozen=True, slots=True)
class RowFrameRelationships:
    """Quantified, unpersisted frame relationships for eligible IMU rows."""

    eligible_row_count: int
    rows_inside_a_frame_interval: int
    rows_before_first_frame: int
    rows_after_last_frame: int
    max_nearest_frame_delta_ms: float | None
    median_nearest_frame_delta_ms: float | None


def frame_times_ms(relative_pts_ns: Sequence[int | None]) -> tuple[float, ...]:
    """Convert decoder output to milliseconds, dropping frames with no PTS."""

    return tuple(
        value / _NS_PER_MS for value in relative_pts_ns if value is not None
    )


def _frame_interval_ms(times: Sequence[float]) -> float | None:
    deltas = [
        later - earlier
        for earlier, later in pairwise(times)
        if math.isfinite(later - earlier) and later - earlier > 0.0
    ]
    if not deltas:
        return None
    return float(median(deltas))


def reconcile_pts_timeline(
    times: Sequence[float],
    *,
    canonical_video_duration_ms: float,
    origin_tolerance_ms: float = PTS_ORIGIN_TOLERANCE_MS,
    span_slack_ms: float = PTS_SPAN_SLACK_MS,
    span_frame_interval_cap_ms: float = PTS_SPAN_FRAME_INTERVAL_CAP_MS,
) -> PtsTimelineReconciliation:
    """Test the decoded timeline's origin and span against the source."""

    if not math.isfinite(canonical_video_duration_ms):
        raise Ego4DPtsValidationError(
            "canonical video duration is not finite")
    if len(times) < 2:
        return PtsTimelineReconciliation(
            frame_count=len(times),
            first_frame_ms=times[0] if times else None,
            last_frame_ms=times[-1] if times else None,
            decoded_span_ms=None,
            canonical_video_duration_ms=canonical_video_duration_ms,
            frame_interval_ms=None,
            origin_offset_ms=abs(times[0]) if times else None,
            origin_tolerance_ms=origin_tolerance_ms,
            span_difference_ms=None,
            span_tolerance_ms=None,
            origin_agrees=False,
            span_agrees=False,
            timeline_status=TIMELINE_INSUFFICIENT_FRAMES,
        )

    first = float(times[0])
    last = float(times[-1])
    span = last - first
    interval = _frame_interval_ms(times)
    origin_offset = abs(first)
    origin_agrees = origin_offset <= origin_tolerance_ms

    capped_interval = (
        min(interval, span_frame_interval_cap_ms) if interval is not None
        else 0.0
    )
    span_tolerance = capped_interval + span_slack_ms
    span_difference = abs(span - canonical_video_duration_ms)
    span_agrees = span_difference <= span_tolerance

    if origin_agrees and span_agrees:
        status = TIMELINE_RECONCILED
    elif origin_agrees:
        status = TIMELINE_SPAN_DISAGREEMENT
    elif span_agrees:
        status = TIMELINE_ORIGIN_DISAGREEMENT
    else:
        status = TIMELINE_ORIGIN_AND_SPAN_DISAGREEMENT

    return PtsTimelineReconciliation(
        frame_count=len(times),
        first_frame_ms=first,
        last_frame_ms=last,
        decoded_span_ms=span,
        canonical_video_duration_ms=canonical_video_duration_ms,
        frame_interval_ms=interval,
        origin_offset_ms=origin_offset,
        origin_tolerance_ms=origin_tolerance_ms,
        span_difference_ms=span_difference,
        span_tolerance_ms=span_tolerance,
        origin_agrees=origin_agrees,
        span_agrees=span_agrees,
        timeline_status=status,
    )


def quantify_row_relationships(
    frame_times: Sequence[float],
    canonical_times: Sequence[float],
) -> RowFrameRelationships:
    """Quantify nearest-frame and containing-interval relationships.

    The result is aggregate and returned only.  No per-row or per-frame
    mapping is produced, because that mapping is the P0.2 index.
    """

    if not canonical_times:
        return RowFrameRelationships(0, 0, 0, 0, None, None)
    if not frame_times:
        return RowFrameRelationships(
            len(canonical_times), 0, 0, 0, None, None)

    ordered = list(frame_times)
    first, last = ordered[0], ordered[-1]
    inside = before = after = 0
    deltas: list[float] = []
    for time in canonical_times:
        if time < first:
            before += 1
        elif time > last:
            after += 1
        else:
            inside += 1
        position = bisect.bisect_left(ordered, time)
        candidates = []
        if position < len(ordered):
            candidates.append(abs(ordered[position] - time))
        if position > 0:
            candidates.append(abs(time - ordered[position - 1]))
        deltas.append(min(candidates))

    return RowFrameRelationships(
        eligible_row_count=len(canonical_times),
        rows_inside_a_frame_interval=inside,
        rows_before_first_frame=before,
        rows_after_last_frame=after,
        max_nearest_frame_delta_ms=max(deltas) if deltas else None,
        median_nearest_frame_delta_ms=(
            float(median(deltas)) if deltas else None
        ),
    )


__all__ = [
    "TIMELINE_INSUFFICIENT_FRAMES",
    "TIMELINE_ORIGIN_AND_SPAN_DISAGREEMENT",
    "TIMELINE_ORIGIN_DISAGREEMENT",
    "TIMELINE_RECONCILED",
    "TIMELINE_SPAN_DISAGREEMENT",
    "Ego4DPtsValidationError",
    "PtsTimelineReconciliation",
    "RowFrameRelationships",
    "frame_times_ms",
    "quantify_row_relationships",
    "reconcile_pts_timeline",
]
