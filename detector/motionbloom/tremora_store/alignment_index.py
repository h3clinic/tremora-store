"""Frame-to-IMU range indexes with exact half-open membership semantics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import numpy as np
import pyarrow as pa

from .schema import frame_imu_index_schema


class AlignmentError(ValueError):
    """Raised when source ordering or range-index inputs are inconsistent."""


def _int64(values: Sequence[int], name: str) -> np.ndarray:
    materialized = list(values)
    if any(
        not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
        or not -(2**63) <= int(value) <= 2**63 - 1
        for value in materialized
    ):
        raise AlignmentError(f"{name} values must be signed int64 integers")
    result = np.asarray(materialized, dtype=np.int64)
    if result.ndim != 1:
        raise AlignmentError(f"{name} must be one-dimensional")
    return result


def _require_dense_ordinals(values: np.ndarray, name: str) -> None:
    if len(values) and not np.array_equal(values, np.arange(len(values), dtype=np.int64)):
        raise AlignmentError(f"{name} must be dense zero-based canonical ordinals")


def _positive_median_period(times_ns: np.ndarray) -> float | None:
    if len(times_ns) < 2:
        return None
    positive = np.diff(times_ns)
    positive = positive[positive > 0]
    return None if not len(positive) else float(np.median(positive))


def _nearest_position(times_ns: np.ndarray, target_ns: int) -> int | None:
    """Choose nearest time; ties select earlier time then lower ordinal."""

    if not len(times_ns):
        return None
    upper = int(np.searchsorted(times_ns, target_ns, side="left"))
    candidates: list[int] = []
    if upper < len(times_ns):
        candidates.append(upper)
    if upper > 0:
        lower_time = int(times_ns[upper - 1])
        candidates.append(int(np.searchsorted(
            times_ns, lower_time, side="left")))
    return min(candidates, key=lambda pos: (
        abs(int(times_ns[pos]) - target_ns), int(times_ns[pos]), pos))


def build_frame_imu_index(
    *, recording_id: str, video_stream_id: str, imu_stream_id: str,
    frame_indices: Sequence[int], frame_canonical_ordinals: Sequence[int],
    frame_times_ns: Sequence[int], imu_canonical_ordinals: Sequence[int],
    imu_times_ns: Sequence[int], video_end_ns: int,
    continuity_intervals_ns: Sequence[tuple[int, int]],
    max_imu_gap_ns: int | None = None,
    min_coverage_fraction: float = 0.8,
) -> pa.Table:
    """Build one range row per frame for one video/IMU stream pair.

    Frame ``i`` owns ``[frame_time[i], frame_time[i+1])``; the final frame owns
    ``[frame_time[-1], video_end_ns)``. IMU start/stop ordinals are dense,
    canonical and stop-exclusive. Exact end-boundary samples belong to the next
    frame. The nearest sample may lie outside the frame-owned interval and its
    delta is signed as ``imu_time - frame_time``.

    An empty continuity inventory is an explicit all-invalid representation:
    every frame receives an empty IMU range, no nearest sample, zero coverage,
    and ``OUTSIDE_CONTINUITY``. This preserves inventory without fabricating an
    alignment when clocks are unresolved.
    """

    if any(
        not isinstance(value, str) or not value
        for value in (recording_id, video_stream_id, imu_stream_id)
    ):
        raise AlignmentError("recording and stream IDs must be non-empty strings")
    if isinstance(min_coverage_fraction, bool) \
            or not isinstance(min_coverage_fraction, (int, float)) \
            or not math.isfinite(min_coverage_fraction) \
            or not 0 <= min_coverage_fraction <= 1:
        raise AlignmentError("min_coverage_fraction must lie in [0,1]")
    frame_ids = _int64(frame_indices, "frame_indices")
    frame_ordinals = _int64(frame_canonical_ordinals, "frame_canonical_ordinals")
    frame_times = _int64(frame_times_ns, "frame_times_ns")
    imu_ordinals = _int64(imu_canonical_ordinals, "imu_canonical_ordinals")
    imu_times = _int64(imu_times_ns, "imu_times_ns")
    if not (len(frame_ids) == len(frame_ordinals) == len(frame_times)):
        raise AlignmentError("frame arrays must have equal length")
    if len(imu_ordinals) != len(imu_times):
        raise AlignmentError("IMU ordinal/time arrays must have equal length")
    _require_dense_ordinals(frame_ordinals, "frame_canonical_ordinals")
    _require_dense_ordinals(imu_ordinals, "imu_canonical_ordinals")
    if np.any(np.diff(frame_times) <= 0):
        raise AlignmentError("frame canonical timestamps must be strictly increasing")
    if np.any(np.diff(imu_times) < 0):
        raise AlignmentError("IMU canonical timestamps must be non-decreasing")
    if not isinstance(video_end_ns, int) or isinstance(video_end_ns, bool):
        raise AlignmentError("video_end_ns must be a signed int64 integer")
    if not -(2**63) <= video_end_ns <= 2**63 - 1:
        raise AlignmentError("video_end_ns must be a signed int64 integer")
    if len(frame_times) and video_end_ns <= int(frame_times[-1]):
        raise AlignmentError("video_end_ns must follow the final frame timestamp")
    if max_imu_gap_ns is not None and (
        not isinstance(max_imu_gap_ns, int) or isinstance(max_imu_gap_ns, bool)
        or max_imu_gap_ns <= 0
    ):
        raise AlignmentError("max_imu_gap_ns must be a positive integer")
    intervals: list[tuple[int, int, int, int, float | None]] = []
    prior_end: int | None = None
    for index, bounds in enumerate(continuity_intervals_ns):
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise AlignmentError(
                f"continuity interval {index} must contain start/end")
        start_value, end_value = bounds
        if not isinstance(start_value, int) or isinstance(start_value, bool) \
                or not isinstance(end_value, int) or isinstance(end_value, bool) \
                or end_value <= start_value:
            raise AlignmentError(
                "continuity intervals must be non-empty integer half-open ranges")
        if prior_end is not None and start_value < prior_end:
            raise AlignmentError("continuity intervals may not overlap")
        imu_lo = int(np.searchsorted(imu_times, start_value, side="left"))
        imu_hi = int(np.searchsorted(imu_times, end_value, side="left"))
        intervals.append((
            start_value, end_value, imu_lo, imu_hi,
            _positive_median_period(imu_times[imu_lo:imu_hi]),
        ))
        prior_end = end_value
    if not len(frame_times):
        return pa.Table.from_pylist([], schema=frame_imu_index_schema())
    rows: list[dict[str, object]] = []
    interval_ends = np.concatenate((frame_times[1:], np.asarray([video_end_ns])))
    # Frames and intervals are both ordered.  A monotone cursor keeps ownership
    # and boundary classification O(F + C), rather than rescanning every
    # continuity interval for every frame.
    interval_cursor = 0
    for frame_id, frame_ordinal, start_ns, end_ns in zip(
            frame_ids, frame_ordinals, frame_times, interval_ends, strict=True):
        start = int(start_ns)
        end = int(end_ns)
        while interval_cursor < len(intervals) \
                and intervals[interval_cursor][1] <= start:
            interval_cursor += 1
        candidate = (
            intervals[interval_cursor]
            if interval_cursor < len(intervals) else None
        )
        intersects = candidate is not None \
            and start < candidate[1] and end > candidate[0]
        continuity_status = None
        owner = None
        if candidate is not None \
                and candidate[0] <= start and end <= candidate[1]:
            owner = candidate
        elif intersects:
            continuity_status = "CONTINUITY_BOUNDARY"
        else:
            continuity_status = "OUTSIDE_CONTINUITY"
        period_ns = None if owner is None else owner[4]
        effective_gap_ns = max_imu_gap_ns
        if effective_gap_ns is None and period_ns is not None:
            effective_gap_ns = max(1, round(period_ns * 2.5))
        if intervals:
            lo = int(np.searchsorted(imu_times, start, side="left"))
            hi = int(np.searchsorted(imu_times, end, side="left"))
        else:
            lo = 0
            hi = 0
        count = hi - lo
        nearest_pos = None
        if owner is not None:
            owner_lo, owner_hi = owner[2], owner[3]
            local_nearest = _nearest_position(
                imu_times[owner_lo:owner_hi], start)
            if local_nearest is not None:
                nearest_pos = owner_lo + local_nearest
        nearest_ordinal = None if nearest_pos is None else int(imu_ordinals[nearest_pos])
        nearest_delta = None if nearest_pos is None else int(imu_times[nearest_pos]) - start
        if period_ns is None:
            coverage = 0.0
        else:
            # A half-open bin on a uniform lattice legitimately contains either
            # floor(D/T) or ceil(D/T) samples depending on phase.  Using the
            # fractional expectation falsely marks clean 30-FPS/50-Hz data as
            # partial whenever a frame owns one rather than two samples.
            expected = max(1, math.floor(
                (end - start) / period_ns + 1e-9))
            coverage = min(1.0, count / expected)
        selected = imu_times[lo:hi]
        duplicate = len(selected) > 1 and bool(np.any(np.diff(selected) == 0))
        if nearest_pos is not None:
            nearest_time = int(imu_times[nearest_pos])
            duplicate = duplicate or (
                int(np.searchsorted(imu_times, nearest_time, side="right"))
                - int(np.searchsorted(imu_times, nearest_time, side="left")) > 1)
        gap = False
        if count and effective_gap_ns is not None and owner is not None:
            spacings = [int(item) for item in np.diff(selected)]
            owner_lo, owner_hi = owner[2], owner[3]
            if lo > owner_lo:
                spacings.append(int(imu_times[lo]) - int(imu_times[lo - 1]))
            if hi < owner_hi:
                spacings.append(int(imu_times[hi]) - int(imu_times[hi - 1]))
            gap = bool(spacings) and max(spacings) > effective_gap_ns
        if continuity_status is not None:
            status = continuity_status
        elif count == 0:
            status = "NO_IMU"
        elif period_ns is None:
            status = "CADENCE_UNKNOWN"
        elif duplicate:
            status = "DUPLICATE_TIMESTAMP"
        elif gap or coverage < min_coverage_fraction:
            status = "GAP"
        else:
            status = "OK"
        rows.append({
            "recording_id": recording_id,
            "video_stream_id": video_stream_id,
            "imu_stream_id": imu_stream_id,
            "frame_index": int(frame_id),
            "frame_canonical_ordinal": int(frame_ordinal),
            "frame_time_ns": start,
            "frame_interval_end_ns": end,
            "imu_start_ordinal": lo,
            "imu_stop_ordinal": hi,
            "imu_nearest_ordinal": nearest_ordinal,
            "nearest_delta_ns": nearest_delta,
            "imu_sample_count": count,
            "imu_coverage_fraction": coverage,
            "alignment_status": status,
        })
    return pa.Table.from_pylist(rows, schema=frame_imu_index_schema())
