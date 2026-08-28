"""Gap-aware construction of deterministic multimodal analysis windows."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pyarrow as pa

from .schema import QualityBits, window_index_schema, window_rejections_schema

_KNOWN_QUALITY_MASK = sum(int(value) for value in QualityBits)
_IMU_SIGNAL_CHANNELS = {
    "ACCEL": ("ax", "ay", "az"),
    "GYRO": ("gx", "gy", "gz"),
    "ACCEL_GYRO": ("ax", "ay", "az", "gx", "gy", "gz"),
    "QUATERNION": ("qw", "qx", "qy", "qz"),
    "ACCEL_GYRO_QUATERNION": (
        "ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz"),
}
_SIGNAL_POLICY_FIELDS = frozenset({
    "recording_id", "video_stream_id", "imu_stream_id",
    "cv_motion_min_peak_to_peak_stored_units",
    "acceleration_min_peak_to_peak_stored_units",
    "angular_velocity_min_peak_to_peak_stored_units",
    "quaternion_min_angular_range_rad",
    "minimum_varying_cv_components", "minimum_varying_imu_channels",
})
_QUATERNION_NORM_ABS_TOL = 1e-3


class WindowIndexError(ValueError):
    """Raised when a window policy or continuity segment is invalid."""


@dataclass(frozen=True, slots=True)
class ContinuitySegment:
    """A prespecified half-open interval for one required stream pair."""

    segment_id: str
    recording_id: str
    video_stream_id: str
    imu_stream_id: str
    start_time_ns: int
    end_time_ns: int
    split_group_id: str
    accepted: bool = True
    quality_bits: int = 0

    def __post_init__(self) -> None:
        identifiers = (
            self.segment_id, self.recording_id, self.video_stream_id,
            self.imu_stream_id, self.split_group_id,
        )
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise WindowIndexError(
                "continuity-segment IDs must be non-empty strings")
        if not isinstance(self.accepted, bool):
            raise WindowIndexError("continuity accepted must be boolean")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            or not -(2**63) <= value <= 2**63 - 1
            for value in (self.start_time_ns, self.end_time_ns)
        ):
            raise WindowIndexError(
                "continuity bounds must be signed int64 nanoseconds")
        if self.end_time_ns <= self.start_time_ns:
            raise WindowIndexError("continuity segments must be non-empty and half-open")
        if not isinstance(self.quality_bits, int) \
                or isinstance(self.quality_bits, bool) \
                or not 0 <= self.quality_bits < 2**32 \
                or self.quality_bits & ~_KNOWN_QUALITY_MASK:
            raise WindowIndexError(
                "continuity quality_bits must contain only known uint32 flags")


@dataclass(frozen=True, slots=True)
class WindowIndexResult:
    valid_index: pa.Table
    rejection_ledger: pa.Table


def candidate_window_id(segment: ContinuitySegment, start_ns: int, end_ns: int,
                        window_policy_id: str) -> str:
    """Hash a typed, length-prefixed identity tuple without delimiter ambiguity."""

    digest = sha256(b"TREMORA_WINDOW_ID_V1\0")
    for value in (
        segment.recording_id, segment.video_stream_id, segment.imu_stream_id,
        segment.segment_id, window_policy_id, start_ns, end_ns,
    ):
        if isinstance(value, int):
            encoded = str(value).encode("ascii")
            digest.update(b"I")
        else:
            encoded = value.encode("utf-8")
            digest.update(b"S")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()[:24]


def _stream_array_cache(
    table: pa.Table, *, stream_field: str,
    ordinal_field: str, quality_field: str,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    required = {"recording_id", stream_field, ordinal_field,
                "canonical_time_ns", quality_field}
    missing = required.difference(table.column_names)
    if missing:
        raise WindowIndexError(f"table is missing fields: {sorted(missing)!r}")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in table.select(sorted(required)).to_pylist():
        grouped[(row["recording_id"], row[stream_field])].append(row)
    result: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: (row[ordinal_field], row["canonical_time_ns"]))
        ordinals = np.asarray([row[ordinal_field] for row in rows], dtype=np.int64)
        times = np.asarray([row["canonical_time_ns"] for row in rows], dtype=np.int64)
        qualities = np.asarray([row[quality_field] for row in rows], dtype=np.uint32)
        if len(ordinals) and not np.array_equal(
                ordinals, np.arange(len(ordinals), dtype=np.int64)):
            raise WindowIndexError(
                "canonical ordinals must be dense and zero-based per stream")
        if len(times) > 1 and np.any(np.diff(times) < 0):
            raise WindowIndexError("canonical times must be non-decreasing")
        if any(int(value) & ~_KNOWN_QUALITY_MASK for value in qualities):
            raise WindowIndexError("source quality bits contain unknown flags")
        result[key] = (ordinals, times, qualities)
    return result


def _alignment_array_cache(
    table: pa.Table,
) -> dict[
    tuple[str, str, str],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    required = {
        "recording_id", "video_stream_id", "imu_stream_id",
        "frame_canonical_ordinal", "frame_time_ns", "frame_interval_end_ns",
        "alignment_status", "imu_nearest_ordinal",
    }
    missing = required.difference(table.column_names)
    if missing:
        raise WindowIndexError(
            f"frame alignment is missing fields: {sorted(missing)!r}")
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in table.select(sorted(required)).to_pylist():
        grouped[(
            row["recording_id"], row["video_stream_id"], row["imu_stream_id"],
        )].append(row)
    result: dict[
        tuple[str, str, str],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: row["frame_canonical_ordinal"])
        ordinals = np.asarray(
            [row["frame_canonical_ordinal"] for row in rows], dtype=np.int64)
        times = np.asarray([row["frame_time_ns"] for row in rows], dtype=np.int64)
        ends = np.asarray(
            [row["frame_interval_end_ns"] for row in rows], dtype=np.int64)
        statuses = np.asarray(
            [row["alignment_status"] for row in rows], dtype=object)
        nearest_ordinals = np.asarray(
            [row["imu_nearest_ordinal"] for row in rows], dtype=object)
        if len(ordinals) and not np.array_equal(
                ordinals, np.arange(len(ordinals), dtype=np.int64)):
            raise WindowIndexError(
                "alignment canonical ordinals must be dense and zero-based")
        if np.any(ends <= times) or (len(ends) > 1 and np.any(np.diff(ends) <= 0)):
            raise WindowIndexError(
                "alignment frame-owned intervals must be positive and ordered")
        result[key] = (ordinals, times, ends, statuses, nearest_ordinals)
    return result


def _cv_signal_array_cache(
    table: pa.Table,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    required = {
        "recording_id", "video_stream_id", "canonical_ordinal",
        "canonical_time_ns", "tracking_quality", "keypoint_validity",
        "motion_vector",
    }
    missing = required.difference(table.column_names)
    if missing:
        raise WindowIndexError(
            f"CV estimates are missing fields: {sorted(missing)!r}")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in table.select(sorted(required)).to_pylist():
        grouped[(row["recording_id"], row["video_stream_id"])].append(row)
    result = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: row["canonical_ordinal"])
        ordinals = np.asarray(
            [row["canonical_ordinal"] for row in rows], dtype=np.int64)
        times = np.asarray(
            [row["canonical_time_ns"] for row in rows], dtype=np.int64)
        tracking = np.asarray([
            math.nan if row["tracking_quality"] is None
            else row["tracking_quality"]
            for row in rows
        ], dtype=np.float64)
        validity_fraction = np.asarray([
            0.0 if not row["keypoint_validity"]
            else sum(item is True for item in row["keypoint_validity"])
            / len(row["keypoint_validity"])
            for row in rows
        ], dtype=np.float64)
        motion = np.asarray([
            [math.nan, math.nan, math.nan]
            if row["motion_vector"] is None else row["motion_vector"]
            for row in rows
        ], dtype=np.float64)
        if len(ordinals) and not np.array_equal(
                ordinals, np.arange(len(ordinals), dtype=np.int64)):
            raise WindowIndexError(
                "CV canonical ordinals must be dense and zero-based")
        result[key] = (times, tracking, validity_fraction, motion)
    return result


def _imu_signal_array_cache(
    table: pa.Table,
) -> dict[tuple[str, str], tuple[str, np.ndarray]]:
    required = {
        "recording_id", "stream_id", "canonical_ordinal", "payload_kind",
        *(channel for channels in _IMU_SIGNAL_CHANNELS.values()
          for channel in channels),
    }
    missing = required.difference(table.column_names)
    if missing:
        raise WindowIndexError(
            f"IMU samples are missing fields: {sorted(missing)!r}")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in table.select(sorted(required)).to_pylist():
        grouped[(row["recording_id"], row["stream_id"])].append(row)
    result = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: row["canonical_ordinal"])
        kinds = {row["payload_kind"] for row in rows}
        if len(kinds) != 1 or next(iter(kinds), None) not in _IMU_SIGNAL_CHANNELS:
            raise WindowIndexError(
                "one supported IMU payload_kind is required per stream")
        channels = _IMU_SIGNAL_CHANNELS[next(iter(kinds))]
        kind = next(iter(kinds))
        result[key] = (kind, np.asarray([
            [math.nan if row[channel] is None else row[channel]
             for channel in channels]
            for row in rows
        ], dtype=np.float64))
    return result


def _varying_column_count(values: np.ndarray, *, minimum_range: float) -> int:
    if values.ndim != 2 or len(values) < 2:
        return 0
    count = 0
    for column in values.T:
        finite = column[np.isfinite(column)]
        if len(finite) >= 2 and float(np.max(finite) - np.min(finite)) \
                >= minimum_range:
            count += 1
    return count


def _signal_policy_map(
    policies: Iterable[dict[str, object]],
) -> dict[tuple[str, str, str], dict[str, object]]:
    result: dict[tuple[str, str, str], dict[str, object]] = {}
    for index, raw in enumerate(policies):
        if not isinstance(raw, dict) or set(raw) != _SIGNAL_POLICY_FIELDS:
            raise WindowIndexError(
                f"frequency signal policy {index} is incomplete or unexpected")
        policy = dict(raw)
        identity = []
        for field in ("recording_id", "video_stream_id", "imu_stream_id"):
            value = policy[field]
            if not isinstance(value, str) or not value:
                raise WindowIndexError(
                    f"frequency signal policy {field} is required")
            identity.append(value)
        key = tuple(identity)
        if key in result:
            raise WindowIndexError(f"duplicate frequency signal policy: {key!r}")
        for field in (
            "cv_motion_min_peak_to_peak_stored_units",
            "acceleration_min_peak_to_peak_stored_units",
            "angular_velocity_min_peak_to_peak_stored_units",
            "quaternion_min_angular_range_rad",
        ):
            value = policy[field]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
            ):
                raise WindowIndexError(
                    f"frequency signal policy {field} must be positive or null")
        if policy["cv_motion_min_peak_to_peak_stored_units"] is None:
            raise WindowIndexError("CV motion range threshold must be specified")
        quaternion_threshold = policy["quaternion_min_angular_range_rad"]
        if quaternion_threshold is not None and quaternion_threshold > math.pi:
            raise WindowIndexError(
                "quaternion angular range threshold must not exceed pi")
        for field in (
            "minimum_varying_cv_components", "minimum_varying_imu_channels",
        ):
            value = policy[field]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise WindowIndexError(
                    f"frequency signal policy {field} must be positive")
        result[key] = policy
    if not result:
        raise WindowIndexError("at least one frequency signal policy is required")
    return result


def _quaternion_angular_range(values: np.ndarray) -> float:
    if len(values) < 2 or values.shape[1] != 4 \
            or not np.all(np.isfinite(values)):
        return 0.0
    norms = np.linalg.norm(values, axis=1)
    if not np.all(np.isclose(
            norms, 1.0, rtol=0.0, atol=_QUATERNION_NORM_ABS_TOL)):
        raise WindowIndexError("quaternion signal values must be unit-normalized")
    normalized = values / norms[:, np.newaxis]
    minimum_abs_dot = 1.0
    for index, reference in enumerate(normalized[:-1]):
        dots = np.abs(normalized[index + 1:] @ reference)
        minimum_abs_dot = min(minimum_abs_dot, float(np.min(dots)))
        if minimum_abs_dot <= 0.0:
            return math.pi
    return float(2.0 * math.acos(min(1.0, max(0.0, minimum_abs_dot))))


def _imu_signal_range_gate(
    values: np.ndarray, *, payload_kind: str, policy: dict[str, object],
) -> bool:
    cursor = 0
    varying = 0
    if payload_kind in {"ACCEL", "ACCEL_GYRO", "ACCEL_GYRO_QUATERNION"}:
        threshold = policy["acceleration_min_peak_to_peak_stored_units"]
        if threshold is None:
            raise WindowIndexError(
                "acceleration payload requires a stored-unit range threshold")
        varying += _varying_column_count(
            values[:, cursor:cursor + 3], minimum_range=float(threshold))
        cursor += 3
    elif policy["acceleration_min_peak_to_peak_stored_units"] is not None:
        raise WindowIndexError(
            "acceleration threshold must be null when acceleration is absent")
    if payload_kind in {"GYRO", "ACCEL_GYRO", "ACCEL_GYRO_QUATERNION"}:
        threshold = policy["angular_velocity_min_peak_to_peak_stored_units"]
        if threshold is None:
            raise WindowIndexError(
                "gyroscope payload requires a stored-unit range threshold")
        varying += _varying_column_count(
            values[:, cursor:cursor + 3], minimum_range=float(threshold))
        cursor += 3
    elif policy["angular_velocity_min_peak_to_peak_stored_units"] is not None:
        raise WindowIndexError(
            "gyroscope threshold must be null when gyroscope is absent")
    if payload_kind in {"QUATERNION", "ACCEL_GYRO_QUATERNION"}:
        threshold = policy["quaternion_min_angular_range_rad"]
        if threshold is None:
            raise WindowIndexError(
                "quaternion payload requires an angular range threshold")
        if _quaternion_angular_range(values[:, cursor:cursor + 4]) \
                >= float(threshold):
            varying += 1
    elif policy["quaternion_min_angular_range_rad"] is not None:
        raise WindowIndexError(
            "quaternion threshold must be null when quaternion is absent")
    return varying >= policy["minimum_varying_imu_channels"]


def _period_ns(times: np.ndarray) -> float | None:
    if len(times) < 2:
        return None
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    return None if not len(diffs) else float(np.median(diffs))


def _sparse_max(values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Build an immutable range-maximum table for O(1) window queries."""

    if not len(values):
        return ()
    levels = [values]
    width = 2
    while width <= len(values):
        half = width // 2
        previous = levels[-1]
        levels.append(np.maximum(previous[:-half], previous[half:]))
        width *= 2
    return tuple(levels)


def _range_max(levels: tuple[np.ndarray, ...], lo: int, hi: int,
               *, default: float) -> float:
    if hi <= lo:
        return default
    width = hi - lo
    level = width.bit_length() - 1
    step = 1 << level
    return float(max(levels[level][lo], levels[level][hi - step]))


@dataclass(frozen=True, slots=True)
class _MetricCache:
    times: np.ndarray
    interval_ends: np.ndarray | None
    predecessor_time_ns: int | None
    successor_time_ns: int | None
    successor_interval_end_ns: int | None
    period_ns: float | None
    max_gap_ns: int | None
    max_cadence_deviation_fraction: float
    gap_levels: tuple[np.ndarray, ...]
    deviation_levels: tuple[np.ndarray, ...]
    duplicate_prefix: np.ndarray
    quality_prefixes: tuple[np.ndarray, ...]
    global_offset: int
    owns_intervals: bool


def _metric_cache(
    times: np.ndarray, qualities: np.ndarray, *, segment_lo: int,
    segment_hi: int, max_gap_ns: int | None,
    max_cadence_deviation_fraction: float,
    interval_ends: np.ndarray | None = None,
) -> _MetricCache:
    local_times = times[segment_lo:segment_hi]
    local_qualities = qualities[segment_lo:segment_hi]
    start_diffs = np.diff(local_times)
    owns_intervals = interval_ends is not None
    if owns_intervals:
        if len(interval_ends) != len(times):
            raise WindowIndexError(
                "frame-owned interval ends must exactly cover frame timestamps")
        metric_differences = (
            interval_ends[segment_lo:segment_hi] - local_times)
        if np.any(metric_differences <= 0):
            raise WindowIndexError("frame-owned intervals must be positive")
        owned_boundaries = np.concatenate((
            local_times,
            interval_ends[segment_hi - 1:segment_hi],
        ))
        period_ns = _period_ns(owned_boundaries)
    else:
        metric_differences = start_diffs
        period_ns = _period_ns(local_times)
    resolved_gap = max_gap_ns
    if resolved_gap is None and period_ns is not None:
        resolved_gap = max(1, round(period_ns * 2.5))
    deviations = np.full(
        len(metric_differences), -math.inf, dtype=np.float64)
    if period_ns is not None:
        positive = metric_differences > 0
        deviations[positive] = np.abs(
            metric_differences[positive].astype(np.float64) - period_ns
        ) / period_ns
    duplicate_prefix = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.cumsum(start_diffs == 0, dtype=np.int64),
    ))
    quality_prefixes = tuple(
        np.concatenate((
            np.asarray([0], dtype=np.int64),
            np.cumsum(
                (local_qualities & np.uint32(1 << bit)) != 0,
                dtype=np.int64),
        ))
        for bit in range(32)
    )
    return _MetricCache(
        times=local_times,
        interval_ends=(
            interval_ends[segment_lo:segment_hi]
            if interval_ends is not None else None),
        predecessor_time_ns=(
            int(times[segment_lo - 1]) if segment_lo > 0 else None),
        successor_time_ns=(
            int(times[segment_hi]) if segment_hi < len(times) else None),
        successor_interval_end_ns=(
            int(interval_ends[segment_hi])
            if interval_ends is not None and segment_hi < len(times)
            else None),
        period_ns=period_ns,
        max_gap_ns=resolved_gap,
        max_cadence_deviation_fraction=max_cadence_deviation_fraction,
        gap_levels=_sparse_max(metric_differences),
        deviation_levels=_sparse_max(deviations),
        duplicate_prefix=duplicate_prefix,
        quality_prefixes=quality_prefixes,
        global_offset=segment_lo,
        owns_intervals=owns_intervals,
    )


def _range_metrics(
    cache: _MetricCache, start_ns: int, end_ns: int,
    *, global_hi_limit: int | None = None,
):
    times = cache.times
    local_lo = int(np.searchsorted(times, start_ns, side="left"))
    local_hi = int(np.searchsorted(times, end_ns, side="left"))
    if global_hi_limit is not None:
        local_hi = max(local_lo, min(
            local_hi,
            max(0, min(len(times), global_hi_limit - cache.global_offset)),
        ))
    lo = cache.global_offset + local_lo
    hi = cache.global_offset + local_hi
    count = local_hi - local_lo
    period_ns = cache.period_ns
    expected = 0.0 if period_ns is None else (end_ns - start_ns) / period_ns
    coverage = 0.0 if expected <= 0 else min(1.0, count / expected)
    duplicate = local_hi - local_lo > 1 and (
        cache.duplicate_prefix[local_hi - 1]
        - cache.duplicate_prefix[local_lo] > 0)
    metric_hi = local_hi if cache.owns_intervals else max(
        local_lo, local_hi - 1)
    max_gap_value = _range_max(
        cache.gap_levels, local_lo, metric_hi, default=-math.inf)
    cadence_deviation = _range_max(
        cache.deviation_levels, local_lo, metric_hi, default=-math.inf)
    relevant_diff_count = count if cache.owns_intervals else max(0, count - 1)
    boundary_gap = False
    if count:
        boundary_tolerance_ns = 0.0 if period_ns is None else max(
            1.0, period_ns * 1e-6)
        if not cache.owns_intervals and local_lo > 0 \
                and period_ns is not None \
                and int(times[local_lo]) - period_ns \
                >= start_ns - boundary_tolerance_ns:
            difference = int(times[local_lo]) - int(times[local_lo - 1])
            max_gap_value = max(max_gap_value, difference)
            if difference > 0:
                cadence_deviation = max(
                    cadence_deviation, abs(difference - period_ns) / period_ns)
            relevant_diff_count += 1
            boundary_gap = boundary_gap or (
                difference > period_ns + boundary_tolerance_ns)
        elif not cache.owns_intervals and local_lo == 0 \
                and period_ns is not None:
            start_offset_ns = int(times[0]) - start_ns
            if start_offset_ns >= period_ns - boundary_tolerance_ns:
                difference = (
                    None if cache.predecessor_time_ns is None
                    else int(times[0]) - cache.predecessor_time_ns)
                boundary_gap = boundary_gap or difference is None or (
                    difference > period_ns + boundary_tolerance_ns)
                max_gap_value = max(
                    max_gap_value,
                    start_offset_ns if difference is None else difference,
                )
                cadence_deviation = max(cadence_deviation, (
                    start_offset_ns / period_ns if difference is None
                    else abs(difference - period_ns) / period_ns
                ))
                relevant_diff_count += 1
        if cache.owns_intervals and period_ns is not None:
            interval_ends = cache.interval_ends
            if interval_ends is None:  # pragma: no cover - cache invariant
                raise WindowIndexError("owned intervals are missing from cache")
            start_offset_ns = int(times[local_lo]) - start_ns
            if start_offset_ns >= period_ns - boundary_tolerance_ns:
                difference = (
                    None if local_lo == 0
                    and cache.predecessor_time_ns is None
                    else int(times[local_lo]) - (
                        cache.predecessor_time_ns
                        if local_lo == 0
                        else int(times[local_lo - 1])
                    )
                )
                boundary_gap = boundary_gap or difference is None or (
                    difference > period_ns + boundary_tolerance_ns)
                max_gap_value = max(
                    max_gap_value,
                    start_offset_ns if difference is None else difference,
                )
                cadence_deviation = max(cadence_deviation, (
                    start_offset_ns / period_ns if difference is None
                    else abs(difference - period_ns) / period_ns
                ))
                relevant_diff_count += 1
            end_offset_ns = end_ns - int(interval_ends[local_hi - 1])
            if local_hi < len(times):
                straddler_start_ns = int(times[local_hi])
                straddler_end_ns = int(interval_ends[local_hi])
            else:
                straddler_start_ns = cache.successor_time_ns
                straddler_end_ns = cache.successor_interval_end_ns
            long_gap_straddler = (
                straddler_start_ns is not None
                and straddler_end_ns is not None
                and straddler_start_ns < end_ns < straddler_end_ns
                and cache.max_gap_ns is not None
                and straddler_end_ns - straddler_start_ns
                > cache.max_gap_ns + boundary_tolerance_ns
            )
            if end_offset_ns >= period_ns - boundary_tolerance_ns \
                    and not long_gap_straddler:
                boundary_gap = True
                max_gap_value = max(max_gap_value, end_offset_ns)
                cadence_deviation = max(
                    cadence_deviation, end_offset_ns / period_ns)
                relevant_diff_count += 1
        if not cache.owns_intervals and local_hi < len(times) \
                and period_ns is not None \
                and int(times[local_hi - 1]) + period_ns \
                < end_ns - boundary_tolerance_ns:
            difference = int(times[local_hi]) - int(times[local_hi - 1])
            max_gap_value = max(max_gap_value, difference)
            if difference > 0:
                cadence_deviation = max(
                    cadence_deviation, abs(difference - period_ns) / period_ns)
            relevant_diff_count += 1
            boundary_gap = boundary_gap or (
                difference > period_ns + boundary_tolerance_ns)
        elif not cache.owns_intervals and local_hi == len(times) \
                and period_ns is not None:
            end_offset_ns = end_ns - int(times[local_hi - 1])
            if end_offset_ns > period_ns + boundary_tolerance_ns:
                difference = (
                    None if cache.successor_time_ns is None
                    else cache.successor_time_ns - int(times[local_hi - 1]))
                boundary_gap = boundary_gap or difference is None or (
                    difference > period_ns + boundary_tolerance_ns)
                max_gap_value = max(
                    max_gap_value,
                    end_offset_ns if difference is None else difference,
                )
                cadence_deviation = max(cadence_deviation, (
                    (end_offset_ns - period_ns) / period_ns
                    if difference is None
                    else abs(difference - period_ns) / period_ns
                ))
                relevant_diff_count += 1
    gap = boundary_gap or (
        cache.max_gap_ns is not None and max_gap_value > cache.max_gap_ns)
    cadence_regular = period_ns is not None and relevant_diff_count > 0 \
        and cadence_deviation > -math.inf
    if cadence_regular:
        cadence_regular = (
            not boundary_gap
            and cadence_deviation <= cache.max_cadence_deviation_fraction)
    else:
        cadence_deviation = None
    aggregate_quality = 0
    for bit, prefix in enumerate(cache.quality_prefixes):
        if prefix[local_hi] - prefix[local_lo] > 0:
            aggregate_quality |= 1 << bit
    return (lo, hi, count, coverage, duplicate, gap, aggregate_quality,
            cadence_regular, cadence_deviation)


def build_window_index(
    *, frame_index: pa.Table, cv_estimates: pa.Table, imu_samples: pa.Table,
    frame_imu_index: pa.Table,
    continuity_segments: Iterable[ContinuitySegment], window_ns: int,
    hop_ns: int, window_policy_id: str, observability_policy_id: str,
    tremor_band_low_hz: float, tremor_band_high_hz: float,
    frequency_signal_policies: Iterable[dict[str, object]],
    min_video_coverage: float = 0.9, min_imu_coverage: float = 0.9,
    max_video_gap_ns: int | None = None, max_imu_gap_ns: int | None = None,
    video_observability_factor: float = 0.4,
    video_observability_cap_hz: float = 12.0,
    min_frequency_cycles: float = 3.0,
    max_cadence_deviation_fraction: float = 0.2,
    min_tracking_quality: float = 0.1,
    min_valid_keypoint_fraction: float = 0.5,
) -> WindowIndexResult:
    """Build a valid-only temporal index plus a rejected-candidate ledger.

    Candidate windows are generated independently within each declared
    continuity segment, so no candidate can bridge a task, clock, gap or
    CV-validity boundary represented by that partition.
    """

    if not isinstance(window_ns, int) or isinstance(window_ns, bool) \
            or not isinstance(hop_ns, int) or isinstance(hop_ns, bool) \
            or window_ns <= 0 or hop_ns <= 0:
        raise WindowIndexError("window_ns and hop_ns must be positive integers")
    if any(
        not isinstance(value, str) or not value
        for value in (window_policy_id, observability_policy_id)
    ):
        raise WindowIndexError(
            "versioned window and observability policies must be non-empty strings")

    def finite_number(value: object) -> bool:
        return not isinstance(value, bool) \
            and isinstance(value, (int, float)) and math.isfinite(value)

    if not finite_number(tremor_band_low_hz) \
            or not finite_number(tremor_band_high_hz) \
            or tremor_band_low_hz <= 0 \
            or tremor_band_high_hz <= tremor_band_low_hz:
        raise WindowIndexError("tremor band must have 0 < low < high")
    if not finite_number(min_video_coverage) \
            or not finite_number(min_imu_coverage) \
            or not 0 <= min_video_coverage <= 1 \
            or not 0 <= min_imu_coverage <= 1:
        raise WindowIndexError("coverage thresholds must lie in [0,1]")
    if not finite_number(video_observability_factor) \
            or not 0 < video_observability_factor <= 0.5:
        raise WindowIndexError("video observability factor must lie in (0,0.5]")
    if not finite_number(video_observability_cap_hz) \
            or video_observability_cap_hz <= 0:
        raise WindowIndexError("video observability cap must be finite and positive")
    if not finite_number(min_frequency_cycles) or min_frequency_cycles <= 0:
        raise WindowIndexError("minimum frequency cycles must be finite and positive")
    if not finite_number(max_cadence_deviation_fraction) \
            or max_cadence_deviation_fraction < 0:
        raise WindowIndexError(
            "maximum cadence deviation must be finite and non-negative")
    if not finite_number(min_tracking_quality) \
            or not 0 < min_tracking_quality <= 1:
        raise WindowIndexError("minimum tracking quality must lie in (0,1]")
    if not finite_number(min_valid_keypoint_fraction) \
            or not 0 < min_valid_keypoint_fraction <= 1:
        raise WindowIndexError(
            "minimum valid-keypoint fraction must lie in (0,1]")
    for name, value in (("max_video_gap_ns", max_video_gap_ns),
                        ("max_imu_gap_ns", max_imu_gap_ns)):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise WindowIndexError(f"{name} must be a positive integer")

    segments = sorted(continuity_segments, key=lambda segment: (
        segment.recording_id, segment.video_stream_id, segment.imu_stream_id,
        segment.start_time_ns))
    valid_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    previous_end: dict[tuple[str, str, str], int] = {}
    stream_cache: dict[tuple[str, str, str], tuple[np.ndarray, ...]] = {}
    seen_candidate_ids: set[str] = set()
    frame_streams = _stream_array_cache(
        frame_index, stream_field="video_stream_id",
        ordinal_field="canonical_ordinal", quality_field="quality_bits")
    imu_streams = _stream_array_cache(
        imu_samples, stream_field="stream_id",
        ordinal_field="canonical_ordinal", quality_field="validity_bits")
    alignment_streams = _alignment_array_cache(frame_imu_index)
    cv_signal_streams = _cv_signal_array_cache(cv_estimates)
    imu_signal_streams = _imu_signal_array_cache(imu_samples)
    signal_policies = _signal_policy_map(frequency_signal_policies)
    segment_pairs = {
        (segment.recording_id, segment.video_stream_id, segment.imu_stream_id)
        for segment in segments
    }
    if set(signal_policies) != segment_pairs:
        raise WindowIndexError(
            "frequency signal policies must exactly match continuity pairs")
    empty_int64 = np.asarray([], dtype=np.int64)
    empty_uint32 = np.asarray([], dtype=np.uint32)
    empty_stream = (empty_int64, empty_int64, empty_uint32)

    for segment in segments:
        pair_key = (segment.recording_id, segment.video_stream_id,
                    segment.imu_stream_id)
        prior_end = previous_end.get(pair_key)
        if prior_end is not None and segment.start_time_ns < prior_end:
            raise WindowIndexError(f"overlapping continuity segments for {pair_key!r}")
        previous_end[pair_key] = segment.end_time_ns
        if pair_key not in stream_cache:
            frame_values = frame_streams.get(
                (segment.recording_id, segment.video_stream_id), empty_stream)
            imu_values = imu_streams.get(
                (segment.recording_id, segment.imu_stream_id), empty_stream)
            stream_cache[pair_key] = (*frame_values, *imu_values)
        f_ord, f_time, f_quality, i_ord, i_time, i_quality = stream_cache[pair_key]
        del f_ord, i_ord
        alignment_values = alignment_streams.get(pair_key)
        if alignment_values is None:
            raise WindowIndexError(f"missing frame alignment for {pair_key!r}")
        a_ord, a_time, a_end, a_status, a_nearest = alignment_values
        expected_frame_ordinals = np.arange(len(f_time), dtype=np.int64)
        if not np.array_equal(a_ord, expected_frame_ordinals) \
                or not np.array_equal(a_time, f_time):
            raise WindowIndexError(
                f"frame alignment does not exactly cover {pair_key!r}")
        cv_signal_values = cv_signal_streams.get(
            (segment.recording_id, segment.video_stream_id))
        if cv_signal_values is None:
            raise WindowIndexError(f"missing CV estimates for {pair_key!r}")
        cv_time, cv_tracking, cv_validity_fraction, cv_motion = cv_signal_values
        if not np.array_equal(cv_time, f_time):
            raise WindowIndexError(
                f"CV estimates do not exactly cover {pair_key!r}")
        imu_signal_values = imu_signal_streams.get(
            (segment.recording_id, segment.imu_stream_id))
        if imu_signal_values is None or len(imu_signal_values[1]) != len(i_time):
            raise WindowIndexError(
                f"IMU signal payload does not exactly cover {pair_key!r}")
        imu_payload_kind, imu_signal = imu_signal_values
        signal_policy = signal_policies.get(pair_key)
        if signal_policy is None:
            raise WindowIndexError(
                f"missing frequency signal policy for {pair_key!r}")
        f_segment_lo = int(np.searchsorted(
            f_time, segment.start_time_ns, side="left"))
        f_segment_hi = int(np.searchsorted(
            f_time, segment.end_time_ns, side="left"))
        # A frame belongs to this segment's cadence model only when its complete
        # owned interval is contained by the segment.  Otherwise the straddler
        # can distort the reference period even though no candidate may select it.
        f_segment_hi = min(
            f_segment_hi,
            int(np.searchsorted(a_end, segment.end_time_ns, side="right")),
        )
        i_segment_lo = int(np.searchsorted(
            i_time, segment.start_time_ns, side="left"))
        i_segment_hi = int(np.searchsorted(
            i_time, segment.end_time_ns, side="left"))
        f_cache = _metric_cache(
            f_time, f_quality, segment_lo=f_segment_lo, segment_hi=f_segment_hi,
            max_gap_ns=max_video_gap_ns,
            max_cadence_deviation_fraction=max_cadence_deviation_fraction,
            interval_ends=a_end)
        i_cache = _metric_cache(
            i_time, i_quality, segment_lo=i_segment_lo, segment_hi=i_segment_hi,
            max_gap_ns=max_imu_gap_ns,
            max_cadence_deviation_fraction=max_cadence_deviation_fraction)

        start = segment.start_time_ns
        while start + window_ns <= segment.end_time_ns:
            end = start + window_ns
            window_id = candidate_window_id(
                segment, start, end, window_policy_id)
            if window_id in seen_candidate_ids:
                raise WindowIndexError(f"duplicate candidate window ID: {window_id}")
            seen_candidate_ids.add(window_id)
            # Exclude a frame whose owned interval straddles the candidate's
            # right boundary.  Timestamp-only selection would otherwise admit
            # motion/IMU assigned from across a task, reset, or QC boundary.
            owned_hi = int(np.searchsorted(a_end, end, side="right"))
            f_values = _range_metrics(
                f_cache, start, end, global_hi_limit=owned_hi)
            i_values = _range_metrics(
                i_cache, start, end)
            (f_lo, f_hi, f_count, f_cov, f_dup, f_gap, f_bits,
             f_cadence_regular, f_cadence_deviation) = f_values
            (i_lo, i_hi, i_count, i_cov, i_dup, i_gap, i_bits,
             i_cadence_regular, i_cadence_deviation) = i_values
            reasons: list[str] = []
            reason_bits = int(segment.quality_bits) | f_bits | i_bits
            for nearest_ordinal in a_nearest[f_lo:f_hi]:
                if nearest_ordinal is None:
                    continue
                if isinstance(nearest_ordinal, bool) \
                        or not isinstance(nearest_ordinal, int) \
                        or not 0 <= nearest_ordinal < len(i_quality):
                    raise WindowIndexError(
                        "alignment nearest ordinal is outside its IMU stream")
                reason_bits |= int(i_quality[nearest_ordinal])
            if not segment.accepted:
                reasons.append("REJECTED_CONTINUITY_SEGMENT")
            if not f_count:
                reasons.append("NO_VIDEO")
                reason_bits |= int(QualityBits.PARTIAL_COVERAGE)
            if not i_count:
                reasons.append("NO_IMU")
                reason_bits |= int(QualityBits.PARTIAL_COVERAGE)
            if f_dup or i_dup:
                reasons.append("DUPLICATE_TIMESTAMP")
                reason_bits |= int(QualityBits.DUPLICATE_TIMESTAMP)
            if f_gap or i_gap or reason_bits & int(QualityBits.STREAM_GAP):
                reasons.append("STREAM_GAP")
                reason_bits |= int(QualityBits.STREAM_GAP)
            if f_cov < min_video_coverage or i_cov < min_imu_coverage:
                reasons.append("INSUFFICIENT_COVERAGE")
                reason_bits |= int(QualityBits.PARTIAL_COVERAGE)
            selected_alignment = a_status[f_lo:f_hi]
            if any(status in {"CONTINUITY_BOUNDARY", "OUTSIDE_CONTINUITY"}
                   for status in selected_alignment):
                reasons.append("INVALID_ALIGNMENT")
                reason_bits |= int(QualityBits.PARTIAL_COVERAGE)
            fatal_mask = int(
                QualityBits.MISSING_TIMESTAMP | QualityBits.NON_MONOTONIC_TIMESTAMP
                | QualityBits.DUPLICATE_TIMESTAMP | QualityBits.STREAM_GAP
                | QualityBits.DECODE_FAILURE | QualityBits.INVALID_CV
                | QualityBits.CLOCK_RESET | QualityBits.UNRESOLVED_CLOCK_MAP
                | QualityBits.PARTIAL_COVERAGE
                | QualityBits.SYNC_RESIDUAL_EXCEEDED
                | QualityBits.INVALID_IMU_PAYLOAD)
            if reason_bits & fatal_mask:
                reasons.append("FATAL_QUALITY_BIT")
            if reasons:
                rejected_rows.append({
                    "candidate_window_id": window_id,
                    "recording_id": segment.recording_id,
                    "video_stream_id": segment.video_stream_id,
                    "imu_stream_id": segment.imu_stream_id,
                    "continuity_segment_id": segment.segment_id,
                    "window_policy_id": window_policy_id,
                    "start_time_ns": start,
                    "end_time_ns": end,
                    "reason_bits": reason_bits,
                    "reason_codes": "|".join(sorted(set(reasons))),
                })
                start += hop_ns
                continue

            duration_s = window_ns / 1_000_000_000.0
            video_fps = f_count / duration_s
            imu_hz = i_count / duration_s
            video_rate_ceiling = 0.5 * video_fps if f_cadence_regular else None
            imu_rate_ceiling = 0.5 * imu_hz if i_cadence_regular else None
            video_observable = (min(
                video_observability_cap_hz,
                video_observability_factor * video_fps,
            ) if f_cadence_regular else 0.0)
            imu_observable = 0.5 * imu_hz if i_cadence_regular else 0.0
            enough_cycles = tremor_band_low_hz * duration_s >= min_frequency_cycles
            band_supported = (
                f_cadence_regular and i_cadence_regular
                and tremor_band_high_hz < video_observable
                and tremor_band_high_hz < imu_observable and enough_cycles
                and not reason_bits & int(QualityBits.FREQUENCY_UNSUPPORTED))
            selected_tracking = cv_tracking[f_lo:f_hi]
            selected_validity = cv_validity_fraction[f_lo:f_hi]
            tracking_valid = bool(
                len(selected_tracking)
                and np.all(np.isfinite(selected_tracking))
                and np.all(selected_tracking >= min_tracking_quality)
                and np.all(selected_validity >= min_valid_keypoint_fraction)
            )
            cv_motion_range_gate = (
                _varying_column_count(
                    cv_motion[f_lo:f_hi],
                    minimum_range=float(signal_policy[
                        "cv_motion_min_peak_to_peak_stored_units"]),
                )
                >= signal_policy["minimum_varying_cv_components"]
            )
            imu_signal_range_gate = _imu_signal_range_gate(
                imu_signal[i_lo:i_hi],
                payload_kind=imu_payload_kind,
                policy=signal_policy,
            )
            frequency_allowed = (
                band_supported and tracking_valid and cv_motion_range_gate)
            valid_for_frequency = (
                frequency_allowed and imu_signal_range_gate)
            output_bits = reason_bits
            if not valid_for_frequency:
                output_bits |= int(QualityBits.FREQUENCY_UNSUPPORTED)
            valid_rows.append({
                "window_id": window_id,
                "recording_id": segment.recording_id,
                "video_stream_id": segment.video_stream_id,
                "imu_stream_id": segment.imu_stream_id,
                "continuity_segment_id": segment.segment_id,
                "window_policy_id": window_policy_id,
                "observability_policy_id": observability_policy_id,
                "start_time_ns": start,
                "end_time_ns": end,
                "frame_start_ordinal": f_lo,
                "frame_stop_ordinal": f_hi,
                "imu_start_ordinal": i_lo,
                "imu_stop_ordinal": i_hi,
                "frame_count": f_count,
                "imu_sample_count": i_count,
                "effective_video_fps": video_fps,
                "effective_imu_hz": imu_hz,
                "video_coverage": f_cov,
                "imu_coverage": i_cov,
                "video_cadence_regular": f_cadence_regular,
                "imu_cadence_regular": i_cadence_regular,
                "video_max_cadence_deviation_fraction": f_cadence_deviation,
                "imu_max_cadence_deviation_fraction": i_cadence_deviation,
                "video_rate_based_nyquist_hz": video_rate_ceiling,
                "imu_rate_based_nyquist_hz": imu_rate_ceiling,
                "video_observable_max_hz": video_observable,
                "imu_observable_max_hz": imu_observable,
                "tremor_band_supported": band_supported,
                "cv_tracking_valid": tracking_valid,
                "cv_motion_range_gate_passed": cv_motion_range_gate,
                "imu_signal_range_gate_passed": imu_signal_range_gate,
                "frequency_estimation_allowed": frequency_allowed,
                "quality_bits": output_bits,
                "valid_for_frequency": valid_for_frequency,
                "split_group_id": segment.split_group_id,
            })
            start += hop_ns

    return WindowIndexResult(
        valid_index=pa.Table.from_pylist(valid_rows, schema=window_index_schema()),
        rejection_ledger=pa.Table.from_pylist(
            rejected_rows, schema=window_rejections_schema()))
