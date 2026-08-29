"""Four-second analysis windows over gap-aware segments.

Windows sit on a grid anchored at task-local zero and stepping by the stride,
so the same offset means the same interval on every stream.  That is what makes
bilateral co-indexing possible without ever claiming sample-level alignment.

Membership is decided by source time, never by counting a fixed number of
samples forward.  The release's device clock jitters from 13.8 us to 58.8 ms
across the corpus, so ``first_sample + 400`` would silently mean a different
duration in every window.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

from .contract import (
    PICOSECONDS_PER_SECOND,
    TIMING_AUTHORITY,
    WINDOW_DURATION_PS,
    WINDOW_STRIDE_PS,
)
from .segments import Segment
from .stream_reader import StreamSamples

WINDOW_VALID = "WINDOW_VALID"

WINDOW_REJECTED_NOT_CONTAINED = "WINDOW_REJECTED_NOT_CONTAINED"
WINDOW_REJECTED_NO_SAMPLES = "WINDOW_REJECTED_NO_SAMPLES"


class WindowError(ValueError):
    """Raised when a window would cross a segment it does not belong to."""


@dataclass(frozen=True, slots=True)
class Window:
    """One four-second window, wholly inside one contiguous segment."""

    window_id: str
    stream_id: str
    participant_id: str
    assessment_id: str
    task_name: str
    device_location: str
    segment_id: str
    window_start_task_local_ps: int
    window_end_task_local_ps: int
    first_sample_ordinal: int
    last_sample_ordinal: int
    sample_count: int
    first_source_time_ps: int
    last_source_time_ps: int
    dt_ref_ps: int
    coverage_fraction: float
    effective_rate_hz: float
    split_group_id: str
    outer_fold: int
    window_status: str = WINDOW_VALID

    def as_record(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "stream_id": self.stream_id,
            "participant_id": self.participant_id,
            "assessment_id": self.assessment_id,
            "task_name": self.task_name,
            "device_location": self.device_location,
            "segment_id": self.segment_id,
            "window_start_task_local_ps": self.window_start_task_local_ps,
            "window_end_task_local_ps": self.window_end_task_local_ps,
            "first_sample_ordinal": self.first_sample_ordinal,
            "last_sample_ordinal": self.last_sample_ordinal,
            "sample_count": self.sample_count,
            "first_source_time_ps": self.first_source_time_ps,
            "last_source_time_ps": self.last_source_time_ps,
            "dt_ref_ps": self.dt_ref_ps,
            "coverage_fraction": self.coverage_fraction,
            "effective_rate_hz": self.effective_rate_hz,
            "split_group_id": self.split_group_id,
            "outer_fold": self.outer_fold,
            "window_status": self.window_status,
            "timing_authority": TIMING_AUTHORITY,
        }


def support_coverage_fraction(
    times: list[int], *, start_ps: int, end_ps: int, dt_ref_ps: int
) -> float:
    """Union of per-sample supports inside the window, over its duration.

    Each sample supports half a reference interval either side of itself; the
    union is clamped to the window, so a window whose samples cluster in one
    half reports about half coverage rather than full.
    """

    duration = end_ps - start_ps
    if duration <= 0 or dt_ref_ps <= 0 or not times:
        return 0.0
    half = dt_ref_ps // 2
    merged: list[tuple[int, int]] = []
    for time in times:
        low = max(start_ps, time - half)
        high = min(end_ps, time + half)
        if high <= low:
            continue
        if merged and low <= merged[-1][1]:
            previous_low, previous_high = merged[-1]
            merged[-1] = (previous_low, max(previous_high, high))
        else:
            merged.append((low, high))
    covered = sum(high - low for low, high in merged)
    return covered / duration


def build_windows(
    samples: StreamSamples,
    segments: tuple[Segment, ...],
    *,
    split_group_id: str,
    outer_fold: int,
    duration_ps: int = WINDOW_DURATION_PS,
    stride_ps: int = WINDOW_STRIDE_PS,
) -> tuple[Window, ...]:
    """Emit every grid window that fits wholly inside one segment."""

    times = samples.task_local_time_ps
    # Segment bounds are expressed as sample ordinals and the search below
    # indexes into ``times``; the two coincide only while ordinals are the
    # zero-anchored gapless sequence the reader produces.  Check it rather
    # than rely on it.
    if samples.sample_ordinal != list(range(samples.sample_count)):
        raise WindowError(
            f"{samples.stream_id} ordinals are not zero-anchored and gapless")
    windows: list[Window] = []
    for segment in segments:
        reference = segment.dt_ref_ps
        if reference is None or reference <= 0:
            continue
        segment_start = segment.start_task_local_time_ps
        segment_end = segment.end_task_local_time_ps
        first_grid = -(-segment_start // stride_ps) * stride_ps
        start = first_grid
        while start + duration_ps <= segment_end:
            end = start + duration_ps
            # Membership is a half-open time interval searched only
            # inside the segment that owns the window.
            lower_bound = segment.first_sample_ordinal
            upper_bound = segment.last_sample_ordinal + 1
            low = bisect.bisect_left(times, start, lower_bound, upper_bound)
            high = bisect.bisect_left(times, end, lower_bound, upper_bound)
            if high <= low:
                start += stride_ps
                continue
            member_times = times[low:high]
            coverage = support_coverage_fraction(
                member_times, start_ps=start, end_ps=end, dt_ref_ps=reference
            )
            count = high - low
            windows.append(Window(
                window_id=f"{samples.stream_id}@{start}",
                stream_id=samples.stream_id,
                participant_id=samples.participant_id,
                assessment_id=samples.assessment_id,
                task_name=samples.task_name,
                device_location=samples.device_location,
                segment_id=segment.segment_id,
                window_start_task_local_ps=start,
                window_end_task_local_ps=end,
                first_sample_ordinal=samples.sample_ordinal[low],
                last_sample_ordinal=samples.sample_ordinal[high - 1],
                sample_count=count,
                first_source_time_ps=samples.source_time_ps[low],
                last_source_time_ps=samples.source_time_ps[high - 1],
                dt_ref_ps=reference,
                coverage_fraction=coverage,
                effective_rate_hz=(
                    count * PICOSECONDS_PER_SECOND / duration_ps
                ),
                split_group_id=split_group_id,
                outer_fold=outer_fold,
            ))
            start += stride_ps
    return tuple(windows)


def assert_windows_inside_segments(
    windows: tuple[Window, ...], segments: tuple[Segment, ...]
) -> None:
    """Check the built windows, not merely the construction that made them."""

    by_id = {segment.segment_id: segment for segment in segments}
    for window in windows:
        segment = by_id.get(window.segment_id)
        if segment is None:
            raise WindowError(f"{window.window_id} names no known segment")
        if window.first_sample_ordinal < segment.first_sample_ordinal:
            raise WindowError(f"{window.window_id} starts before its segment")
        if window.last_sample_ordinal > segment.last_sample_ordinal:
            raise WindowError(f"{window.window_id} ends after its segment")
        if window.window_start_task_local_ps < (
            segment.start_task_local_time_ps
        ):
            raise WindowError(f"{window.window_id} opens before its segment")
        if window.window_end_task_local_ps > segment.end_task_local_time_ps:
            raise WindowError(f"{window.window_id} closes after its segment")
        if window.sample_count <= 0:
            raise WindowError(f"{window.window_id} indexes no sample")


__all__ = [
    "WINDOW_REJECTED_NOT_CONTAINED",
    "WINDOW_REJECTED_NO_SAMPLES",
    "WINDOW_VALID",
    "Window",
    "WindowError",
    "assert_windows_inside_segments",
    "build_windows",
    "support_coverage_fraction",
]
