"""Stage A: irregular source time to an exact 100 Hz parent grid.

Deterministic linear interpolation between the two source samples that bracket
each parent grid point, inside one P0.2.1 segment.  Never extrapolation: a
parent point outside the segment's own first and last sample is simply not
produced, and that unsupported interval propagates into every derived rate's
eligibility before any filter guard is applied.

Linear interpolation is not transparent -- on an ideal uniform grid its
response is sinc^2, about -0.41 dB at 12 Hz -- which is precisely why 100 Hz is
an ablation in its own right rather than a pass-through.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .contract import PARENT_RATE_HZ
from .rational_time import PICOSECONDS_PER_SECOND, grid_for

PARENT_PERIOD_PS = PICOSECONDS_PER_SECOND // PARENT_RATE_HZ

PARENT_BUILT = "PARENT_BUILT"
PARENT_NO_BRACKETABLE_POINT = "PARENT_NO_BRACKETABLE_POINT"


class StageAError(ValueError):
    """Raised when the parent grid cannot be built as specified."""


@dataclass(frozen=True, slots=True)
class ParentRange:
    """The parent ordinals a segment's own samples can bracket."""

    first_ordinal: int
    last_ordinal: int

    @property
    def count(self) -> int:
        return max(0, self.last_ordinal - self.first_ordinal + 1)

    @property
    def empty(self) -> bool:
        return self.count == 0

    def as_range(self) -> range:
        return range(self.first_ordinal, self.last_ordinal + 1)


def bracketable_parent_range(times_ps: Sequence[int]) -> ParentRange:
    """Parent ordinals lying between a segment's first and last sample.

    Exact integer arithmetic on the 100 Hz grid, whose period is exactly
    10,000,000,000 ps.  Interior points are always bracketed because a P0.2.1
    segment holds no gap larger than its own threshold; only the two ends can
    fall outside.
    """

    if len(times_ps) < 2:
        return ParentRange(0, -1)
    covering = grid_for(PARENT_RATE_HZ).ordinals_covering(
        int(times_ps[0]), int(times_ps[-1])
    )
    if not covering:
        return ParentRange(0, -1)
    return ParentRange(covering.start, covering.stop - 1)


def build_parent(
    times_ps: Sequence[int],
    channels: Sequence[Sequence[float]],
    *,
    first_ordinal: int,
    last_ordinal: int,
) -> np.ndarray:
    """Interpolate each channel onto parent ordinals ``[first, last]``.

    Returns an array of shape ``(channels, samples)``.  Every target time is
    bracketed by construction; the caller establishes that with
    :func:`bracketable_parent_range` before asking.
    """

    if last_ordinal < first_ordinal:
        return np.zeros((len(channels), 0), dtype=np.float64)
    source_times = np.asarray(times_ps, dtype=np.int64)
    targets = (
        np.arange(first_ordinal, last_ordinal + 1, dtype=np.int64)
        * PARENT_PERIOD_PS
    )
    if targets[0] < source_times[0] or targets[-1] > source_times[-1]:
        raise StageAError(
            "a parent ordinal falls outside the segment; no extrapolation")

    upper = np.searchsorted(source_times, targets, side="right")
    lower = np.clip(upper - 1, 0, source_times.size - 2)
    left = source_times[lower]
    right = source_times[lower + 1]
    span = (right - left).astype(np.float64)
    if np.any(span <= 0.0):
        raise StageAError("a bracketing interval is not positive")
    weight = (targets - left).astype(np.float64) / span

    parent = np.empty((len(channels), targets.size), dtype=np.float64)
    for index, values in enumerate(channels):
        series = np.asarray(values, dtype=np.float64)
        if series.size != source_times.size:
            raise StageAError("a channel does not match the segment times")
        low = series[lower]
        parent[index] = low + weight * (series[lower + 1] - low)
    return parent


__all__ = [
    "PARENT_BUILT",
    "PARENT_NO_BRACKETABLE_POINT",
    "PARENT_PERIOD_PS",
    "ParentRange",
    "StageAError",
    "bracketable_parent_range",
    "build_parent",
]
