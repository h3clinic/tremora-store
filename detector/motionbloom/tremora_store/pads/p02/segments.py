"""Gap-aware contiguous segments over one stored stream.

P0.1 found no duplicate or non-monotonic Time in the release, but P0.2 does not
assume that a future or regenerated materialization is perfect: it recomputes
the boundaries from the stored samples every time.

A segment breaks at ``min(100 ms, 3 x dt_ref)``.  At the release's ~9.99 ms
median interval the cadence term binds, so the threshold is about 30 ms.  About
a fifth of the sampled corpus contains at least one such gap, so segments are
genuinely multi-part here rather than a formality.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from .contract import (
    GAP_ABSOLUTE_CAP_PS,
    GAP_MULTIPLIER,
    MINIMUM_CADENCE_DELTAS,
)
from .stream_reader import SAMPLE_VALID, StreamSamples

SEGMENT_VALID = "SEGMENT_VALID"

BREAK_STREAM_START = "STREAM_START"
BREAK_STREAM_END = "STREAM_END"
BREAK_TIME_GAP = "TIME_GAP"
BREAK_NONPOSITIVE_DELTA = "NONPOSITIVE_DELTA"
BREAK_ORDINAL_DISCONTINUITY = "ORDINAL_DISCONTINUITY"
BREAK_INVALID_SAMPLE = "INVALID_SAMPLE"


class SegmentError(ValueError):
    """Raised when segments would not partition their stream."""


@dataclass(frozen=True, slots=True)
class Segment:
    """One contiguous run of samples inside a stream."""

    segment_id: str
    stream_id: str
    segment_ordinal: int
    first_sample_ordinal: int
    last_sample_ordinal: int
    sample_count: int
    start_source_time_ps: int
    end_source_time_ps: int
    start_task_local_time_ps: int
    end_task_local_time_ps: int
    dt_ref_ps: int | None
    gap_threshold_ps: int | None
    break_reason_before: str
    break_reason_after: str
    segment_status: str = SEGMENT_VALID

    def as_record(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "stream_id": self.stream_id,
            "segment_ordinal": self.segment_ordinal,
            "first_sample_ordinal": self.first_sample_ordinal,
            "last_sample_ordinal": self.last_sample_ordinal,
            "sample_count": self.sample_count,
            "start_source_time_ps": self.start_source_time_ps,
            "end_source_time_ps": self.end_source_time_ps,
            "start_task_local_time_ps": self.start_task_local_time_ps,
            "end_task_local_time_ps": self.end_task_local_time_ps,
            "dt_ref_ps": self.dt_ref_ps,
            "gap_threshold_ps": self.gap_threshold_ps,
            "break_reason_before": self.break_reason_before,
            "break_reason_after": self.break_reason_after,
            "segment_status": self.segment_status,
        }


def integer_median(values: list[int]) -> int | None:
    """Exact integer median; the lower of the two middles when even.

    Deterministic and integral, which a floating average of the two middle
    values would not be.
    """

    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def reference_interval_ps(times: list[int]) -> int | None:
    """Median positive interval, or ``None`` when that is not a cadence."""

    deltas = [later - earlier for earlier, later in pairwise(times)]
    positive = [delta for delta in deltas if delta > 0]
    if len(positive) < MINIMUM_CADENCE_DELTAS:
        return None
    reference = integer_median(positive)
    return reference if reference and reference > 0 else None


def gap_threshold_ps(reference_ps: int) -> int:
    """``min(100 ms, 3 x dt_ref)`` in exact picoseconds."""

    return min(GAP_ABSOLUTE_CAP_PS, GAP_MULTIPLIER * reference_ps)


def build_segments(
    samples: StreamSamples,
    *,
    sample_statuses: list[str] | None = None,
) -> tuple[Segment, ...]:
    """Partition one stream into contiguous, non-overlapping segments."""

    count = samples.sample_count
    if count == 0:
        return ()
    statuses = sample_statuses or [SAMPLE_VALID] * count
    if len(statuses) != count:
        raise SegmentError("sample status list does not match the stream")

    reference = reference_interval_ps(samples.source_time_ps)
    threshold = gap_threshold_ps(reference) if reference is not None else None

    breaks: dict[int, str] = {}
    for index in range(1, count):
        delta = samples.source_time_ps[index] - samples.source_time_ps[index - 1]
        ordinal_step = (
            samples.source_row_ordinal[index]
            - samples.source_row_ordinal[index - 1]
        )
        if statuses[index] != SAMPLE_VALID:
            breaks[index] = BREAK_INVALID_SAMPLE
        elif ordinal_step != 1:
            breaks[index] = BREAK_ORDINAL_DISCONTINUITY
        elif delta <= 0:
            breaks[index] = BREAK_NONPOSITIVE_DELTA
        elif threshold is not None and delta > threshold:
            breaks[index] = BREAK_TIME_GAP

    boundaries = sorted(breaks)
    segments: list[Segment] = []
    start = 0
    for ordinal, end_exclusive in enumerate([*boundaries, count]):
        last = end_exclusive - 1
        before = (
            BREAK_STREAM_START if start == 0 else breaks[start]
        )
        after = (
            BREAK_STREAM_END if end_exclusive == count
            else breaks[end_exclusive]
        )
        segments.append(Segment(
            segment_id=f"{samples.stream_id}#{ordinal:04d}",
            stream_id=samples.stream_id,
            segment_ordinal=ordinal,
            first_sample_ordinal=samples.sample_ordinal[start],
            last_sample_ordinal=samples.sample_ordinal[last],
            sample_count=end_exclusive - start,
            start_source_time_ps=samples.source_time_ps[start],
            end_source_time_ps=samples.source_time_ps[last],
            start_task_local_time_ps=samples.task_local_time_ps[start],
            end_task_local_time_ps=samples.task_local_time_ps[last],
            dt_ref_ps=reference,
            gap_threshold_ps=threshold,
            break_reason_before=before,
            break_reason_after=after,
        ))
        start = end_exclusive
    return tuple(segments)


def assert_partitions_stream(
    segments: tuple[Segment, ...], samples: StreamSamples
) -> None:
    """Every sample in exactly one segment, contiguous and non-overlapping."""

    covered = 0
    expected_next = samples.sample_ordinal[0] if samples.sample_count else 0
    for segment in segments:
        if segment.first_sample_ordinal != expected_next:
            raise SegmentError(
                f"{segment.segment_id} does not resume where the previous "
                "segment ended")
        if segment.last_sample_ordinal < segment.first_sample_ordinal:
            raise SegmentError(f"{segment.segment_id} is inverted")
        span = segment.last_sample_ordinal - segment.first_sample_ordinal + 1
        if span != segment.sample_count:
            raise SegmentError(f"{segment.segment_id} is not contiguous")
        covered += segment.sample_count
        expected_next = segment.last_sample_ordinal + 1
    if covered != samples.sample_count:
        raise SegmentError(
            f"{samples.stream_id}: segments cover {covered} of "
            f"{samples.sample_count} samples")


__all__ = [
    "BREAK_INVALID_SAMPLE",
    "BREAK_NONPOSITIVE_DELTA",
    "BREAK_ORDINAL_DISCONTINUITY",
    "BREAK_STREAM_END",
    "BREAK_STREAM_START",
    "BREAK_TIME_GAP",
    "SEGMENT_VALID",
    "Segment",
    "SegmentError",
    "assert_partitions_stream",
    "build_segments",
    "gap_threshold_ps",
    "integer_median",
    "reference_interval_ps",
]
