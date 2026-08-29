"""Deterministic nonuniform tremor-band spectral kernel.

A conventional FFT assumes uniformly spaced samples.  PADS does not have them:
reference intervals run from 9.9199 ms to 10.0800 ms across the corpus and
individual deltas from 13.8 us to 58.8 ms, so the kernel evaluates the
transform at the *actual* sample times.

Per axis, in this order:

1. centre time on the window start;
2. fit and remove a linear trend using the actual times;
3. apply a continuous-time Hann weight defined on those times;
4. evaluate the nonuniform transform on the frozen frequency grid.

Everything is float64 and every reduction is a fixed-shape numpy sum, so two
executions of the same window agree bit for bit.  No BLAS call is made -- the
transform is an elementwise product and a reduction, not a matrix multiply --
which keeps the result independent of threading.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contract import FREQUENCY_BIN_COUNT
from .grid import frequency_values

_TWO_PI = 2.0 * math.pi


class SpectralKernelError(ValueError):
    """Raised when a window cannot be transformed as written."""


@dataclass(frozen=True, slots=True)
class AxisSpectrum:
    """One axis's power over the frozen grid."""

    power: np.ndarray

    def as_list(self) -> list[float]:
        return [float(value) for value in self.power]


@dataclass(frozen=True, slots=True)
class FamilySpectrum:
    """Three axes and their summed aggregate for one sensor family."""

    family: str
    axis_x: np.ndarray
    axis_y: np.ndarray
    axis_z: np.ndarray
    aggregate: np.ndarray
    normalized_aggregate: np.ndarray
    dominant_frequency_hz: float
    band_power: float
    spectral_entropy: float
    peak_to_median_ratio: float

    def content_sha256(self) -> str:
        """Hash the exact float bytes, so equality here is bit equality."""

        digest = hashlib.sha256()
        digest.update(self.family.encode("ascii"))
        for array in (
            self.axis_x, self.axis_y, self.axis_z,
            self.aggregate, self.normalized_aggregate,
        ):
            digest.update(b"\x1f")
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def as_record(self) -> dict[str, Any]:
        return {
            "axis_x_power": [float(v) for v in self.axis_x],
            "axis_y_power": [float(v) for v in self.axis_y],
            "axis_z_power": [float(v) for v in self.axis_z],
            "aggregate_power": [float(v) for v in self.aggregate],
            "normalized_aggregate_power": [
                float(v) for v in self.normalized_aggregate
            ],
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "band_power": self.band_power,
            "spectral_entropy": self.spectral_entropy,
            "peak_to_median_ratio": self.peak_to_median_ratio,
        }


def relative_seconds(times_ps: Sequence[int]) -> np.ndarray:
    """Centre picosecond timestamps on the first sample, in seconds."""

    if len(times_ps) < 2:
        raise SpectralKernelError("a spectrum needs at least two samples")
    origin = times_ps[0]
    return np.asarray(
        [(value - origin) / 1e12 for value in times_ps], dtype=np.float64
    )


def assert_strictly_increasing(tau: np.ndarray) -> None:
    if not np.all(np.diff(tau) > 0.0):
        raise SpectralKernelError("sample times are not strictly increasing")


def detrend_linear(tau: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Remove the least-squares line fitted against the actual times."""

    count = tau.shape[0]
    sum_t = float(np.sum(tau))
    sum_tt = float(np.sum(tau * tau))
    sum_x = float(np.sum(values))
    sum_tx = float(np.sum(tau * values))
    denominator = count * sum_tt - sum_t * sum_t
    if denominator == 0.0:
        return values - (sum_x / count)
    slope = (count * sum_tx - sum_t * sum_x) / denominator
    intercept = (sum_x - slope * sum_t) / count
    return values - (intercept + slope * tau)


def hann_weights(tau: np.ndarray, duration_s: float) -> np.ndarray:
    """A continuous-time Hann weight evaluated at the actual sample times."""

    if duration_s <= 0.0:
        raise SpectralKernelError("window duration must be positive")
    return 0.5 - 0.5 * np.cos(_TWO_PI * (tau / duration_s))


def _phase_matrix(tau: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    # exp(-2*pi*i*f*tau) for every (frequency, sample) pair.  Built once per
    # window and reused by all six axes.
    return np.exp(-1j * _TWO_PI * np.outer(frequencies, tau))


def axis_power(
    weighted_signal: np.ndarray,
    phase: np.ndarray,
    weight_energy: float,
) -> np.ndarray:
    """``|X(f)|^2 / sum(w^2)`` for one axis."""

    transform = np.sum(phase * weighted_signal, axis=1)
    return (transform.real**2 + transform.imag**2) / weight_energy


def spectral_metrics(
    power: np.ndarray, frequencies: np.ndarray, step_hz: float
) -> tuple[float, float, float, float]:
    """Dominant frequency, band power, entropy and peak-to-median ratio."""

    total = float(np.sum(power))
    dominant = float(frequencies[int(np.argmax(power))])
    band_power = total * step_hz
    if total > 0.0:
        density = power / total
        positive = density[density > 0.0]
        entropy = float(
            -np.sum(positive * np.log(positive)) / math.log(power.shape[0])
        )
    else:
        entropy = 0.0
    median = float(np.median(power))
    ratio = float(np.max(power) / median) if median > 0.0 else 0.0
    return dominant, band_power, entropy, ratio


def family_spectrum(
    family: str,
    times_ps: Sequence[int],
    axes: Sequence[Sequence[float]],
    *,
    duration_s: float,
    frequencies: np.ndarray | None = None,
    step_hz: float = 0.25,
) -> FamilySpectrum:
    """Transform three raw axes and aggregate them within one family."""

    if len(axes) != 3:
        raise SpectralKernelError("a sensor family has exactly three axes")
    grid = (
        np.asarray(frequency_values(), dtype=np.float64)
        if frequencies is None else frequencies
    )
    if grid.shape[0] != FREQUENCY_BIN_COUNT:
        raise SpectralKernelError(
            f"the frozen grid has {FREQUENCY_BIN_COUNT} bins")

    tau = relative_seconds(times_ps)
    assert_strictly_increasing(tau)
    weights = hann_weights(tau, duration_s)
    weight_energy = float(np.sum(weights * weights))
    if weight_energy <= 0.0:
        raise SpectralKernelError("window weights carry no energy")
    phase = _phase_matrix(tau, grid)

    powers: list[np.ndarray] = []
    for axis in axes:
        values = np.asarray(axis, dtype=np.float64)
        if values.shape[0] != tau.shape[0]:
            raise SpectralKernelError("axis length does not match the times")
        if not np.all(np.isfinite(values)):
            raise SpectralKernelError("axis carries a non-finite value")
        detrended = detrend_linear(tau, values)
        powers.append(axis_power(weights * detrended, phase, weight_energy))

    aggregate = powers[0] + powers[1] + powers[2]
    total = float(np.sum(aggregate))
    normalized = (
        aggregate / total if total > 0.0 else np.zeros_like(aggregate)
    )
    dominant, band_power, entropy, ratio = spectral_metrics(
        aggregate, grid, step_hz
    )
    return FamilySpectrum(
        family=family,
        axis_x=powers[0],
        axis_y=powers[1],
        axis_z=powers[2],
        aggregate=aggregate,
        normalized_aggregate=normalized,
        dominant_frequency_hz=dominant,
        band_power=band_power,
        spectral_entropy=entropy,
        peak_to_median_ratio=ratio,
    )


def input_content_sha256(
    times_ps: Sequence[int], axes: Sequence[Sequence[float]]
) -> str:
    """Hash the exact inputs a spectrum was computed from."""

    digest = hashlib.sha256()
    digest.update(
        np.ascontiguousarray(times_ps, dtype=np.int64).tobytes()
    )
    for axis in axes:
        digest.update(b"\x1f")
        digest.update(
            np.ascontiguousarray(axis, dtype=np.float64).tobytes()
        )
    return digest.hexdigest()


__all__ = [
    "AxisSpectrum",
    "FamilySpectrum",
    "SpectralKernelError",
    "assert_strictly_increasing",
    "axis_power",
    "detrend_linear",
    "family_spectrum",
    "hann_weights",
    "input_content_sha256",
    "relative_seconds",
    "spectral_metrics",
]
