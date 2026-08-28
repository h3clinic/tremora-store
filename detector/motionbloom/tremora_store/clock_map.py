"""Piecewise rational clock mapping with explicit reset epochs."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from math import gcd

import pyarrow as pa

from .schema import clock_map_schema


class ClockMapError(ValueError):
    """Raised when a clock map is invalid, missing, or ambiguous."""


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_MAPPING_STATUSES = frozenset({"VALID", "UNRESOLVED", "REJECTED"})


def _require_int64(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) \
            or not _INT64_MIN <= value <= _INT64_MAX:
        raise ClockMapError(f"{field} must be a signed int64 integer")
    return value


def round_div_nearest_even(numerator: int, denominator: int) -> int:
    """Return numerator/denominator rounded to nearest, with ties to even."""

    if not isinstance(numerator, int) or isinstance(numerator, bool):
        raise ClockMapError("rounding numerator must be an integer")
    if not isinstance(denominator, int) or isinstance(denominator, bool) \
            or denominator <= 0:
        raise ClockMapError("scale denominator must be a positive integer")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient % 2 == 1):
        quotient += 1
    return sign * quotient


@dataclass(frozen=True, slots=True)
class ClockSegment:
    recording_id: str
    stream_id: str
    clock_epoch_id: str
    continuity_component_id: str
    acquisition_ordinal: int
    source_start_ordinal: int
    source_stop_ordinal: int
    native_start_ns: int
    native_end_ns: int
    native_anchor_ns: int
    canonical_anchor_ns: int
    scale_numerator: int
    scale_denominator: int
    residual_p50_ms: float | None = None
    residual_p95_ms: float | None = None
    mapping_status: str = "VALID"

    def __post_init__(self) -> None:
        identifiers = (
            self.recording_id, self.stream_id, self.clock_epoch_id,
            self.continuity_component_id,
        )
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ClockMapError(
                "recording, stream, epoch and continuity IDs must be strings")
        for field in (
            "acquisition_ordinal", "source_start_ordinal", "source_stop_ordinal",
            "native_start_ns", "native_end_ns", "native_anchor_ns",
            "canonical_anchor_ns", "scale_numerator", "scale_denominator",
        ):
            _require_int64(getattr(self, field), field)
        if not 0 <= self.acquisition_ordinal <= 2**31 - 1:
            raise ClockMapError("acquisition_ordinal must be a non-negative int32")
        if self.source_start_ordinal < 0:
            raise ClockMapError("source_start_ordinal must be non-negative")
        if self.source_stop_ordinal <= self.source_start_ordinal:
            raise ClockMapError("source ordinal ranges must be non-empty and half-open")
        if self.native_end_ns <= self.native_start_ns:
            raise ClockMapError("native ranges must be non-empty and half-open")
        if not self.native_start_ns <= self.native_anchor_ns < self.native_end_ns:
            raise ClockMapError("native anchor must lie inside its clock segment")
        if self.scale_numerator <= 0 or self.scale_denominator <= 0:
            raise ClockMapError("clock-map rational scale must be positive")
        if gcd(self.scale_numerator, self.scale_denominator) != 1:
            raise ClockMapError("clock-map rational scale must be normalized")
        if self.mapping_status not in _MAPPING_STATUSES:
            raise ClockMapError("mapping_status is not supported")
        for field in ("residual_p50_ms", "residual_p95_ms"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
            ):
                raise ClockMapError(
                    f"{field} must be finite and non-negative when present")
        if (self.residual_p50_ms is None) != (self.residual_p95_ms is None):
            raise ClockMapError("clock residual quantiles must be jointly nullable")
        if self.residual_p50_ms is not None \
                and self.residual_p95_ms < self.residual_p50_ms:
            raise ClockMapError("residual_p95_ms must be >= residual_p50_ms")
        for boundary in (self.native_start_ns, self.native_end_ns):
            mapped = self._canonical_boundary_unchecked(boundary)
            _require_int64(mapped, "mapped canonical boundary")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.recording_id, self.stream_id, self.clock_epoch_id)

    @property
    def drift_ppm(self) -> float:
        """Forward-map scale error: ``(canonical/native - 1) * 1e6`` ppm.

        This is a property of the fitted native-to-canonical map, not a claim
        about the oscillator's conventional fast/slow error sign.
        """

        return ((self.scale_numerator - self.scale_denominator) * 1_000_000.0
                / self.scale_denominator)

    def _canonical_boundary_unchecked(self, native_time_ns: int) -> int:
        delta_native = native_time_ns - self.native_anchor_ns
        mapped_delta = round_div_nearest_even(
            delta_native * self.scale_numerator, self.scale_denominator)
        return self.canonical_anchor_ns + mapped_delta

    def canonical_boundary_ns(self, native_time_ns: int) -> int:
        """Map a segment boundary exactly, including the exclusive end."""

        _require_int64(native_time_ns, "native_time_ns")
        if not self.native_start_ns <= native_time_ns <= self.native_end_ns:
            raise ClockMapError("native boundary lies outside the clock segment")
        result = self._canonical_boundary_unchecked(native_time_ns)
        return _require_int64(result, "mapped canonical timestamp")

    def contains(self, native_time_ns: int,
                 source_ordinal: int | None = None) -> bool:
        _require_int64(native_time_ns, "native_time_ns")
        if source_ordinal is not None:
            _require_int64(source_ordinal, "source_ordinal")
            if source_ordinal < 0:
                raise ClockMapError("source_ordinal must be non-negative")
        native_ok = self.native_start_ns <= native_time_ns < self.native_end_ns
        ordinal_ok = source_ordinal is None or (
            self.source_start_ordinal <= source_ordinal < self.source_stop_ordinal
        )
        return native_ok and ordinal_ok

    def map_ns(self, native_time_ns: int,
               *, source_ordinal: int | None = None) -> int:
        if self.mapping_status != "VALID":
            raise ClockMapError(f"clock segment {self.key!r} is not usable: "
                                f"{self.mapping_status}")
        if not self.contains(native_time_ns, source_ordinal):
            raise ClockMapError(f"timestamp/ordinal lies outside segment {self.key!r}")
        return self.canonical_boundary_ns(native_time_ns)

    def as_dict(self) -> dict[str, object]:
        return {
            "recording_id": self.recording_id,
            "stream_id": self.stream_id,
            "clock_epoch_id": self.clock_epoch_id,
            "continuity_component_id": self.continuity_component_id,
            "acquisition_ordinal": self.acquisition_ordinal,
            "source_start_ordinal": self.source_start_ordinal,
            "source_stop_ordinal": self.source_stop_ordinal,
            "native_start_ns": self.native_start_ns,
            "native_end_ns": self.native_end_ns,
            "native_anchor_ns": self.native_anchor_ns,
            "canonical_anchor_ns": self.canonical_anchor_ns,
            "scale_numerator": self.scale_numerator,
            "scale_denominator": self.scale_denominator,
            "drift_ppm_derived": self.drift_ppm,
            "residual_p50_ms": self.residual_p50_ms,
            "residual_p95_ms": self.residual_p95_ms,
            "mapping_status": self.mapping_status,
        }


class PiecewiseClockMap:
    """Immutable affine segments keyed by stream and explicit clock epoch."""

    def __init__(self, segments: Iterable[ClockSegment],
                 *, continuity_tolerance_ns: int = 1):
        if not isinstance(continuity_tolerance_ns, int) \
                or isinstance(continuity_tolerance_ns, bool) \
                or continuity_tolerance_ns < 0:
            raise ClockMapError("continuity tolerance must be a non-negative integer")
        ordered = sorted(
            segments,
            key=lambda item: (item.recording_id, item.stream_id,
                              item.acquisition_ordinal, item.clock_epoch_id),
        )
        if not ordered:
            raise ClockMapError("at least one clock segment is required")
        seen_keys: set[tuple[str, str, str]] = set()
        seen_ordinals: set[tuple[str, str, int]] = set()
        seen_components: dict[tuple[str, str], set[str]] = {}
        previous_stop: dict[tuple[str, str], int] = {}
        previous_segment: dict[tuple[str, str], ClockSegment] = {}
        for segment in ordered:
            if segment.key in seen_keys:
                raise ClockMapError(f"duplicate clock epoch key: {segment.key!r}")
            ordinal_key = (segment.recording_id, segment.stream_id,
                           segment.acquisition_ordinal)
            if ordinal_key in seen_ordinals:
                raise ClockMapError(f"duplicate acquisition ordinal: {ordinal_key!r}")
            stream_key = (segment.recording_id, segment.stream_id)
            prior_stop = previous_stop.get(stream_key)
            if prior_stop is not None and segment.source_start_ordinal < prior_stop:
                raise ClockMapError(f"overlapping source-ordinal ranges for {stream_key!r}")
            prior = previous_segment.get(stream_key)
            if prior is not None:
                prior_end = prior.canonical_boundary_ns(prior.native_end_ns)
                current_start = segment.canonical_boundary_ns(segment.native_start_ns)
                difference = current_start - prior_end
                same_component = (
                    segment.continuity_component_id
                    == prior.continuity_component_id
                )
                if not same_component and segment.continuity_component_id in \
                        seen_components.get(stream_key, set()):
                    raise ClockMapError(
                        "one continuity component cannot be reused noncontiguously")
                if same_component:
                    if segment.source_start_ordinal != prior.source_stop_ordinal:
                        raise ClockMapError(
                            "one continuity component cannot skip source ordinals")
                    if segment.native_start_ns != prior.native_end_ns:
                        raise ClockMapError(
                            "one continuity component requires adjacent native "
                            "clock domains; a reset or gap starts a new component")
                    if abs(difference) > continuity_tolerance_ns:
                        raise ClockMapError(
                            "clock segments in one continuity component are not "
                            "canonical-time continuous")
                elif difference < -continuity_tolerance_ns:
                    raise ClockMapError(
                        "later acquisition segment moves backward in canonical time")
            seen_keys.add(segment.key)
            seen_ordinals.add(ordinal_key)
            seen_components.setdefault(stream_key, set()).add(
                segment.continuity_component_id)
            previous_stop[stream_key] = segment.source_stop_ordinal
            previous_segment[stream_key] = segment
        self._segments = tuple(ordered)
        self._by_key = {segment.key: segment for segment in self._segments}

    @property
    def segments(self) -> tuple[ClockSegment, ...]:
        return self._segments

    def map_ns(self, recording_id: str, stream_id: str, native_time_ns: int,
               *, clock_epoch_id: str | None = None,
               source_ordinal: int | None = None) -> int:
        if not isinstance(recording_id, str) or not recording_id \
                or not isinstance(stream_id, str) or not stream_id:
            raise ClockMapError("recording_id and stream_id must be strings")
        if clock_epoch_id is not None \
                and (not isinstance(clock_epoch_id, str) or not clock_epoch_id):
            raise ClockMapError("clock_epoch_id must be a string when supplied")
        if clock_epoch_id is not None:
            key = (recording_id, stream_id, clock_epoch_id)
            try:
                return self._by_key[key].map_ns(
                    native_time_ns, source_ordinal=source_ordinal)
            except KeyError as exc:
                raise ClockMapError(f"unknown clock epoch: {key!r}") from exc
        candidates = [
            segment for segment in self._segments
            if segment.recording_id == recording_id
            and segment.stream_id == stream_id
            and segment.contains(native_time_ns, source_ordinal)
        ]
        if not candidates:
            raise ClockMapError("no clock segment contains the timestamp/ordinal")
        if len(candidates) > 1:
            raise ClockMapError("native timestamp is ambiguous after a reset; "
                                "supply clock_epoch_id")
        return candidates[0].map_ns(native_time_ns, source_ordinal=source_ordinal)

    def validate_materialized(self, *, recording_id: str, stream_id: str,
                              clock_epoch_id: str, source_ordinal: int,
                              native_time_ns: int,
                              canonical_time_ns: int) -> None:
        expected = self.map_ns(
            recording_id, stream_id, native_time_ns,
            clock_epoch_id=clock_epoch_id, source_ordinal=source_ordinal)
        if expected != canonical_time_ns:
            raise ClockMapError(
                f"materialized canonical timestamp {canonical_time_ns} != {expected}")

    def to_table(self) -> pa.Table:
        return pa.Table.from_pylist(
            [segment.as_dict() for segment in self._segments],
            schema=clock_map_schema(),
        )
