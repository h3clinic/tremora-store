"""The two-stage resampler, and the support both stages must agree on.

The valid support of a derived segment is the intersection of the two stages:

    S_derived = S_100Hz_bracketable  intersect  S_FIR_valid

The FIR guard alone is not sufficient.  If a parent grid point at either end
cannot be bracketed by real source samples, that parent interval never exists,
and its absence has to reach the derived eligibility mask *before* the filter
guard is applied -- otherwise the guard would be measured against a parent that
was never built.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .contract import PARENT_RATE_HZ, WINDOW_ELIGIBILITY
from .filters import design
from .rational_time import grid_for, polyphase_anchor, supported_output_ordinals
from .stage_a import ParentRange, bracketable_parent_range, build_parent
from .stage_b import filter_to_rate

DERIVED_SUPPORTED = "DERIVED_SUPPORTED"
DERIVED_NO_PARENT = "DERIVED_NO_PARENT"
DERIVED_NO_FILTER_SUPPORT = "DERIVED_NO_FILTER_SUPPORT"

WINDOW_ELIGIBLE = "WINDOW_ELIGIBLE"
WINDOW_OUTSIDE_SUPPORT = "WINDOW_OUTSIDE_SUPPORT"
WINDOW_NO_SAMPLES = "WINDOW_NO_SAMPLES"


class ResampleError(ValueError):
    """Raised when a derived signal is asked for where none is supported."""


@dataclass(frozen=True, slots=True)
class DerivedSupport:
    """What one segment can support at one derived rate."""

    rate_hz: int
    parent: ParentRange
    supported: range
    status: str

    @property
    def first_time_ps(self) -> int | None:
        if not self.supported:
            return None
        return int(
            grid_for(self.rate_hz).sample_picoseconds(self.supported.start)
        )

    @property
    def last_time_ps(self) -> int | None:
        if not self.supported:
            return None
        return int(
            grid_for(self.rate_hz).sample_picoseconds(self.supported.stop - 1)
        )


def derive_support(
    times_ps: Sequence[int], rate_hz: int
) -> DerivedSupport:
    """Intersect the bracketable parent with the filter's valid region."""

    parent = bracketable_parent_range(times_ps)
    if parent.empty:
        return DerivedSupport(rate_hz, parent, range(0), DERIVED_NO_PARENT)
    if rate_hz == PARENT_RATE_HZ:
        # The parent carries no filter, so its support is the bracketable
        # region itself.
        return DerivedSupport(
            rate_hz, parent, parent.as_range(), DERIVED_SUPPORTED
        )
    supported = supported_output_ordinals(
        rate_hz,
        taps=design(rate_hz).size,
        parent_first=parent.first_ordinal,
        parent_last=parent.last_ordinal,
    )
    status = DERIVED_SUPPORTED if supported else DERIVED_NO_FILTER_SUPPORT
    return DerivedSupport(rate_hz, parent, supported, status)


def window_output_ordinals(
    rate_hz: int, start_ps: int, end_ps: int
) -> range:
    """Derived ordinals whose exact times fall in ``[start, end)``."""

    grid = grid_for(rate_hz)
    covering = grid.ordinals_covering(start_ps, end_ps)
    if not covering:
        return range(0)
    last = covering.stop - 1
    if grid.sample_picoseconds(last) >= end_ps:
        last -= 1
    return range(covering.start, last + 1) if last >= covering.start else range(0)


def window_eligibility(
    support: DerivedSupport, start_ps: int, end_ps: int
) -> tuple[str, range]:
    """Whether a window's whole interval lies inside supported output."""

    ordinals = window_output_ordinals(support.rate_hz, start_ps, end_ps)
    if not ordinals:
        return WINDOW_NO_SAMPLES, range(0)
    first_time = support.first_time_ps
    last_time = support.last_time_ps
    if first_time is None or last_time is None:
        return WINDOW_OUTSIDE_SUPPORT, range(0)
    if start_ps < first_time or end_ps > last_time:
        return WINDOW_OUTSIDE_SUPPORT, range(0)
    if ordinals.start < support.supported.start or (
        ordinals.stop - 1 >= support.supported.stop
    ):
        return WINDOW_OUTSIDE_SUPPORT, range(0)
    return WINDOW_ELIGIBLE, ordinals


def derive_window(
    times_ps: Sequence[int],
    channels: Sequence[Sequence[float]],
    *,
    rate_hz: int,
    support: DerivedSupport,
    ordinals: range,
) -> np.ndarray:
    """Build only the parent slice this window needs, then filter it."""

    if not ordinals:
        raise ResampleError("no output ordinals were requested")
    if rate_hz == PARENT_RATE_HZ:
        return build_parent(
            times_ps, channels,
            first_ordinal=ordinals.start, last_ordinal=ordinals.stop - 1,
        )
    taps = design(rate_hz).size
    _, first_anchor, branch = polyphase_anchor(
        rate_hz, ordinals.start, taps=taps
    )
    _, last_anchor, _ = polyphase_anchor(
        rate_hz, ordinals.stop - 1, taps=taps
    )
    needed_first = first_anchor - branch + 1
    needed_last = last_anchor
    for ordinal in ordinals:
        _, anchor, width = polyphase_anchor(rate_hz, ordinal, taps=taps)
        needed_first = min(needed_first, anchor - width + 1)
        needed_last = max(needed_last, anchor)
    if (
        needed_first < support.parent.first_ordinal
        or needed_last > support.parent.last_ordinal
    ):
        raise ResampleError(
            f"{rate_hz} Hz window needs parent outside the bracketable range")
    parent = build_parent(
        times_ps, channels,
        first_ordinal=needed_first, last_ordinal=needed_last,
    )
    return filter_to_rate(
        parent, rate_hz=rate_hz,
        parent_first_ordinal=needed_first, output_ordinals=ordinals,
    )


def window_times_seconds(rate_hz: int, ordinals: range) -> list[float]:
    """Exact grid times, as float seconds, for a derived window."""

    grid = grid_for(rate_hz)
    return [float(grid.sample_seconds(ordinal)) for ordinal in ordinals]


def window_times_picoseconds(rate_hz: int, ordinals: range) -> list[int]:
    """Reported integer picoseconds; the exact times remain rational."""

    grid = grid_for(rate_hz)
    return [grid.sample_picoseconds_rounded(ordinal) for ordinal in ordinals]


ELIGIBILITY_POLICY = WINDOW_ELIGIBILITY


__all__ = [
    "DERIVED_NO_FILTER_SUPPORT",
    "DERIVED_NO_PARENT",
    "DERIVED_SUPPORTED",
    "ELIGIBILITY_POLICY",
    "WINDOW_ELIGIBLE",
    "WINDOW_NO_SAMPLES",
    "WINDOW_OUTSIDE_SUPPORT",
    "DerivedSupport",
    "ResampleError",
    "derive_support",
    "derive_window",
    "window_eligibility",
    "window_output_ordinals",
    "window_times_picoseconds",
    "window_times_seconds",
]
