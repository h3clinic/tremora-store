"""Frozen, hashed anti-alias coefficients.

One rate-independent kernel, expressed in zero-crossings, applied with a
per-rate cutoff.  Writing it that way means there is a single coefficient table
to freeze and hash rather than four, and the only thing that varies with rate
is the mapping from elapsed time to kernel argument.

The tabulated coefficients *are* the filter.  The analytic Kaiser-windowed sinc
is how the table was produced; it is not consulted when resampling, so the
frozen bytes fully determine the result.

Applied weights are normalized to unit sum per output sample.  With irregular
input the local sample density varies, and without normalization the passband
would ripple with the input spacing rather than with the filter.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from functools import lru_cache
from typing import Any

import numpy as np

from .contract import (
    ANTI_ALIAS_KERNEL,
    COEFFICIENT_TABLE_LENGTH,
    CORE_BAND_MAX_HZ,
    CORE_BAND_MIN_HZ,
    CUTOFF_FRACTION,
    DERIVED_RATES_HZ,
    EDGE_BAND_MAX_HZ,
    HALF_WIDTH_ZERO_CROSSINGS,
    KAISER_BETA,
    TAPS_PER_ZERO_CROSSING,
    WEIGHT_NORMALIZATION,
    cutoff_hz,
)


class AntiAliasError(ValueError):
    """Raised when the frozen filter is asked for something it cannot give."""


@lru_cache(maxsize=1)
def coefficient_table() -> np.ndarray:
    """The frozen half-kernel, sampled every 1/128 of a zero crossing.

    Index ``j`` holds the kernel at ``u = j / TAPS_PER_ZERO_CROSSING`` zero
    crossings from the centre.  The kernel is symmetric, so only ``u >= 0`` is
    stored.
    """

    indices = np.arange(COEFFICIENT_TABLE_LENGTH, dtype=np.float64)
    u = indices / TAPS_PER_ZERO_CROSSING
    sinc = np.sinc(u)
    window = np.i0(
        KAISER_BETA * np.sqrt(
            np.maximum(0.0, 1.0 - (u / HALF_WIDTH_ZERO_CROSSINGS) ** 2)
        )
    ) / np.i0(KAISER_BETA)
    table = sinc * window
    table.setflags(write=False)
    return table


def coefficients_sha256() -> str:
    """Hash the exact float bytes of the frozen table."""

    return hashlib.sha256(
        np.ascontiguousarray(coefficient_table(), dtype=np.float64).tobytes()
    ).hexdigest()


def kernel_weight(u: float) -> float:
    """The frozen kernel at ``u`` zero crossings, by table lookup.

    Lookup, not interpolation: the table is the filter, and the quantization
    of ``u`` to 1/128 of a zero crossing is part of its definition.
    """

    index = round(float(abs(u)) * TAPS_PER_ZERO_CROSSING)
    if index >= COEFFICIENT_TABLE_LENGTH:
        return 0.0
    return float(coefficient_table()[index])


def kernel_weights(u: np.ndarray) -> np.ndarray:
    """Vectorized table lookup for many kernel arguments at once."""

    indices = np.rint(np.abs(u) * TAPS_PER_ZERO_CROSSING).astype(np.int64)
    inside = indices < COEFFICIENT_TABLE_LENGTH
    weights = np.zeros(indices.shape, dtype=np.float64)
    weights[inside] = coefficient_table()[indices[inside]]
    return weights


def support_seconds(rate_hz: int | Fraction) -> Fraction:
    """Half-width of the kernel in seconds at one derived rate.

    ``u = 2 f_c dt``, so the kernel reaches zero at
    ``dt = HALF_WIDTH / (2 f_c)``.
    """

    return Fraction(HALF_WIDTH_ZERO_CROSSINGS, 2) / cutoff_hz(int(rate_hz))


def kernel_argument(
    delta_seconds: np.ndarray, rate_hz: int | Fraction
) -> np.ndarray:
    """Convert elapsed seconds to kernel zero-crossings for one rate."""

    return 2.0 * float(cutoff_hz(int(rate_hz))) * delta_seconds


def frequency_response(
    frequencies_hz: np.ndarray, rate_hz: int | Fraction
) -> np.ndarray:
    """The frozen filter's magnitude response, for reporting and controls."""

    table = coefficient_table()
    offsets = np.arange(
        -(COEFFICIENT_TABLE_LENGTH - 1), COEFFICIENT_TABLE_LENGTH
    ) / TAPS_PER_ZERO_CROSSING
    taps = np.concatenate([table[:0:-1], table])
    seconds = offsets / (2.0 * float(cutoff_hz(int(rate_hz))))
    phase = np.exp(
        -2j * np.pi * np.outer(np.asarray(frequencies_hz, np.float64), seconds)
    )
    response = np.abs(np.sum(phase * taps, axis=1))
    return response / np.sum(taps)


def declared_band_response() -> dict[str, dict[str, float]]:
    """The frozen filter's response at the band edges, per derived rate.

    Published as measured evidence rather than prose.  At 25 Hz the cutoff
    coincides with the 10 Hz core/edge split, so the topmost core bin is
    already 6 dB down and the edge band runs from -6 to -28 dB; at 30 Hz the
    top of the analysis grid sits at the cutoff.  Both are properties of the
    frozen coefficients, and a control checks them rather than trusting this
    comment.
    """

    edges = np.array(
        [CORE_BAND_MIN_HZ, CORE_BAND_MAX_HZ, EDGE_BAND_MAX_HZ],
        dtype=np.float64,
    )
    labels = ("core_min_hz", "core_max_hz", "edge_max_hz")
    response: dict[str, dict[str, float]] = {}
    for rate in DERIVED_RATES_HZ:
        magnitudes = frequency_response(edges, rate)
        response[str(rate)] = {
            label: float(20.0 * np.log10(max(float(value), 1e-12)))
            for label, value in zip(labels, magnitudes, strict=True)
        }
        response[str(rate)]["own_nyquist_hz"] = float(rate) / 2.0
        response[str(rate)]["own_nyquist_db"] = float(
            20.0 * np.log10(
                max(float(frequency_response(
                    np.array([rate / 2.0]), rate
                )[0]), 1e-12)
            )
        )
    return response


def anti_alias_manifest() -> dict[str, Any]:
    """The frozen filter definition published in every P0.4 record."""

    return {
        "kernel": ANTI_ALIAS_KERNEL,
        "kaiser_beta": KAISER_BETA,
        "half_width_zero_crossings": HALF_WIDTH_ZERO_CROSSINGS,
        "taps_per_zero_crossing": TAPS_PER_ZERO_CROSSING,
        "coefficient_table_length": COEFFICIENT_TABLE_LENGTH,
        "coefficients_sha256": coefficients_sha256(),
        "cutoff_fraction_num": CUTOFF_FRACTION.numerator,
        "cutoff_fraction_den": CUTOFF_FRACTION.denominator,
        "weight_normalization": WEIGHT_NORMALIZATION,
        "cutoff_hz": {
            str(rate): float(cutoff_hz(rate)) for rate in DERIVED_RATES_HZ
        },
        "support_seconds": {
            str(rate): float(support_seconds(rate))
            for rate in DERIVED_RATES_HZ
        },
        "declared_band_response_db": declared_band_response(),
    }


__all__ = [
    "AntiAliasError",
    "anti_alias_manifest",
    "coefficient_table",
    "coefficients_sha256",
    "declared_band_response",
    "frequency_response",
    "kernel_argument",
    "kernel_weight",
    "kernel_weights",
    "support_seconds",
]
