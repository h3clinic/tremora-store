"""Exact rational timing for the derived uniform grids.

Three of the four derived rates have an exact picosecond period: 100 Hz is
10,000,000,000 ps, 50 Hz is 20,000,000,000 ps and 25 Hz is 40,000,000,000 ps.
30 Hz does not -- 1/30 s is 100,000,000,000/3 ps -- and no decimal scale can
fix that, because 30 carries a factor of three.

So grid time is carried as an exact :class:`~fractions.Fraction` throughout and
the picosecond form is derived, with its exactness stated per rate rather than
assumed.  A grid ordinal's time is ``ordinal / rate`` seconds exactly, anchored
at task-local zero so a grid point means the same instant on every segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .contract import (
    DERIVED_RATES_HZ,
    GRID_ORIGIN,
    PARENT_RATE_HZ,
    RATES_WITH_EXACT_PICOSECOND_PERIOD,
    RESAMPLING_RATIOS,
)

PICOSECONDS_PER_SECOND = 10**12


class RationalTimeError(ValueError):
    """Raised when a grid would be built on an unusable rate."""


@dataclass(frozen=True, slots=True)
class RateGrid:
    """One derived uniform grid, timed exactly."""

    rate_hz: Fraction
    origin: str = GRID_ORIGIN

    @property
    def period_seconds(self) -> Fraction:
        return 1 / self.rate_hz

    @property
    def period_picoseconds(self) -> Fraction:
        return Fraction(PICOSECONDS_PER_SECOND) / self.rate_hz

    @property
    def exact_in_picoseconds(self) -> bool:
        """Whether the period is a whole number of picoseconds."""

        return self.period_picoseconds.denominator == 1

    def sample_seconds(self, ordinal: int) -> Fraction:
        """The exact time of one grid ordinal, in seconds."""

        return Fraction(ordinal) * self.period_seconds

    def sample_picoseconds(self, ordinal: int) -> Fraction:
        """The exact time of one grid ordinal, in picoseconds."""

        return Fraction(ordinal) * self.period_picoseconds

    def sample_picoseconds_exact(self, ordinal: int) -> int | None:
        """The integer picosecond time, or ``None`` when it is not exact."""

        value = self.sample_picoseconds(ordinal)
        return int(value) if value.denominator == 1 else None

    def sample_picoseconds_rounded(self, ordinal: int) -> int:
        """A reported picosecond time; never used for the transform itself."""

        value = self.sample_picoseconds(ordinal)
        return round(value.numerator / value.denominator)

    def rounding_residual_picoseconds(self, ordinal: int) -> Fraction:
        """How far the reported integer sits from the exact rational time."""

        return self.sample_picoseconds(ordinal) - Fraction(
            self.sample_picoseconds_rounded(ordinal)
        )

    def ordinals_covering(
        self, start_picoseconds: int, end_picoseconds: int
    ) -> range:
        """Grid ordinals whose exact times lie within ``[start, end]``.

        The comparison is exact rational arithmetic, so a 30 Hz ordinal is
        never admitted or excluded by a rounding artefact.
        """

        if end_picoseconds < start_picoseconds:
            raise RationalTimeError("the interval is inverted")
        period = self.period_picoseconds
        first = math.ceil(Fraction(start_picoseconds) / period)
        last = math.floor(Fraction(end_picoseconds) / period)
        return range(first, last + 1) if last >= first else range(0)

    def as_record(self) -> dict[str, object]:
        return {
            "rate_hz_num": self.rate_hz.numerator,
            "rate_hz_den": self.rate_hz.denominator,
            "period_picoseconds_num": self.period_picoseconds.numerator,
            "period_picoseconds_den": self.period_picoseconds.denominator,
            "exact_in_picoseconds": self.exact_in_picoseconds,
            "grid_origin": self.origin,
        }


def grid_for(rate_hz: int | Fraction) -> RateGrid:
    """Return the frozen grid for one derived rate."""

    rate = Fraction(rate_hz)
    if rate <= 0:
        raise RationalTimeError("a derived rate must be positive")
    return RateGrid(rate_hz=rate)


def parent_grid() -> RateGrid:
    """The uniform 100 Hz parent every derived rate is built from."""

    return grid_for(PARENT_RATE_HZ)


def parent_span_for_output(
    rate_hz: int, ordinal: int, *, taps: int
) -> tuple[int, int]:
    """The inclusive parent indices one derived output sample needs.

    ``y[k] = sum_n h[M*k + D - L*n] x[n]`` for a symmetric filter of odd
    length with group delay ``D``, so the required parent range is
    ``[ceil((M*k - D)/L), floor((M*k + D)/L)]``.  For an integer decimation
    (``L == 1``) that is simply ``M*k +/- D``.
    """

    upsample, decimate = RESAMPLING_RATIOS[rate_hz]
    if taps % 2 == 0:
        raise RationalTimeError("a linear-phase Type I filter has odd length")
    delay = (taps - 1) // 2
    working = decimate * ordinal
    first = -((-(working - delay)) // upsample)
    last = (working + delay) // upsample
    return first, last


def supported_output_ordinals(
    rate_hz: int,
    *,
    taps: int,
    parent_first: int,
    parent_last: int,
) -> range:
    """Output ordinals whose whole kernel support lies inside the parent.

    Nothing is padded, reflected, repeated or renormalized: an ordinal whose
    kernel would run off the end simply produces no output.
    """

    if parent_last < parent_first:
        return range(0)
    upsample, decimate = RESAMPLING_RATIOS[rate_hz]
    delay = (taps - 1) // 2
    first = -((-(upsample * parent_first + delay)) // decimate)
    last = (upsample * parent_last - delay) // decimate
    first = max(first, 0)
    return range(first, last + 1) if last >= first else range(0)


def assert_declared_exactness() -> None:
    """The contract's exactness claim must match the arithmetic.

    Import-time in the tests rather than here, so a wrong constant fails
    loudly with a message rather than silently shaping a run.
    """

    for rate in DERIVED_RATES_HZ:
        expected = rate in RATES_WITH_EXACT_PICOSECOND_PERIOD
        if grid_for(rate).exact_in_picoseconds != expected:
            raise RationalTimeError(
                f"{rate} Hz exactness disagrees with the contract")


__all__ = [
    "PICOSECONDS_PER_SECOND",
    "RateGrid",
    "RationalTimeError",
    "assert_declared_exactness",
    "grid_for",
    "parent_grid",
    "parent_span_for_output",
    "supported_output_ordinals",
]
