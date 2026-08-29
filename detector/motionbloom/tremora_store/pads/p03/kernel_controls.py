"""In-process synthetic controls for the spectral kernel.

The gate condition ``SYNTHETIC_KERNEL_CONTROLS_PASS`` is decided by running
these, not by deferring to a test suite that may not have run.  Timings come
from a fixed arithmetic pattern rather than an RNG, so the controls give the
same answer on every execution.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contract import FREQUENCY_BIN_COUNT, WINDOW_DURATION_S
from .grid import frequency_values
from .kernel import (
    SpectralKernelError,
    assert_no_gap,
    family_spectrum,
)

CONTROLS_PASS = "SYNTHETIC_KERNEL_CONTROLS_PASS"
CONTROLS_FAIL = "SYNTHETIC_KERNEL_CONTROLS_FAIL"

_GRID = np.asarray(frequency_values(), dtype=np.float64)


def _times(count: int, *, base: float = 0.0100, jitter: float = 0.0018) -> np.ndarray:
    # Deterministic irregular spacing: no RNG, so the control is reproducible.
    deltas = np.asarray(
        [base + jitter * math.sin(0.7 * index) for index in range(count)],
        dtype=np.float64,
    )
    return np.concatenate([[0.0], np.cumsum(deltas)[:-1]])


def _ps(times: np.ndarray) -> list[int]:
    return [round(float(value) * 1e12) for value in times]


def _tone(times: np.ndarray, hz: float, phase: float = 0.0) -> np.ndarray:
    return np.sin(2.0 * np.pi * hz * times + phase)


def _spectrum(times: np.ndarray, signal: np.ndarray):
    zero = np.zeros_like(signal)
    return family_spectrum(
        "GYROSCOPE", _ps(times), [signal, zero, zero],
        duration_s=WINDOW_DURATION_S, frequencies=_GRID,
    )


def run_controls() -> dict[str, Any]:
    """Run every control and report each outcome individually."""

    outcomes: dict[str, bool] = {}
    times = _times(400)

    five = _spectrum(times, _tone(times, 5.0))
    eight = _spectrum(times, _tone(times, 8.0))
    outcomes["tone_5hz_recovered"] = five.dominant_frequency_hz == 5.0
    outcomes["tone_8hz_recovered"] = eight.dominant_frequency_hz == 8.0
    outcomes["grid_is_thirty_seven_bins"] = (
        five.aggregate.shape == (FREQUENCY_BIN_COUNT,)
    )

    reference = five.aggregate
    shifted = _spectrum(times, _tone(times, 5.0, 1.9)).aggregate
    outcomes["phase_shift_preserves_spectrum"] = bool(
        np.max(np.abs(shifted - reference)) / np.max(reference) < 0.05
    )

    x, y = _tone(times, 5.0), _tone(times, 5.0, math.pi / 2)
    zero = np.zeros_like(x)
    angle = 0.7
    before = family_spectrum(
        "GYROSCOPE", _ps(times), [x, y, zero],
        duration_s=WINDOW_DURATION_S, frequencies=_GRID,
    ).aggregate
    after = family_spectrum(
        "GYROSCOPE", _ps(times),
        [math.cos(angle) * x - math.sin(angle) * y,
         math.sin(angle) * x + math.cos(angle) * y, zero],
        duration_s=WINDOW_DURATION_S, frequencies=_GRID,
    ).aggregate
    outcomes["axis_rotation_preserves_summed_power"] = bool(
        np.max(np.abs(after - before)) / np.max(before) < 1e-11
    )

    clean = _tone(times, 5.0)
    ramped = _spectrum(times, clean + 3.0 * times).aggregate
    outcomes["linear_trend_removed"] = bool(
        np.max(np.abs(ramped - reference)) / np.max(reference) < 1e-9
    )

    outcomes["every_observed_length_accepted"] = all(
        _spectrum(
            _times(count), _tone(_times(count), 6.0)
        ).dominant_frequency_hz == 6.0
        for count in range(395, 406)
    )

    slow = np.arange(360) / 90.0
    outcomes["source_time_not_ordinal_over_rate"] = _spectrum(
        slow, _tone(slow, 5.0)
    ).dominant_frequency_hz == 5.0

    fast = np.arange(400) / 100.0
    slower = np.arange(400) / 80.0
    values = _tone(fast, 5.0)
    outcomes["sample_count_is_not_the_clock"] = (
        _spectrum(fast, values).dominant_frequency_hz
        != _spectrum(slower, values).dominant_frequency_hz
    )

    magnitude = _spectrum(times, np.abs(clean))
    outcomes["vector_magnitude_would_double_frequency"] = (
        five.dominant_frequency_hz == 5.0
        and magnitude.dominant_frequency_hz == 10.0
    )

    swapped = times.copy()
    swapped[10], swapped[11] = swapped[11], swapped[10]
    try:
        _spectrum(swapped, clean)
        outcomes["non_monotonic_refused"] = False
    except SpectralKernelError:
        outcomes["non_monotonic_refused"] = True

    gapped = times.copy()
    gapped[200:] += 0.05
    try:
        assert_no_gap(_ps(gapped), 30_000_000_000)
        outcomes["gap_crossing_refused"] = False
    except SpectralKernelError:
        outcomes["gap_crossing_refused"] = True

    passed = all(outcomes.values())
    return {
        "status": CONTROLS_PASS if passed else CONTROLS_FAIL,
        "controls_total": len(outcomes),
        "controls_passed": sum(1 for value in outcomes.values() if value),
        "controls": dict(sorted(outcomes.items())),
    }


__all__ = ["CONTROLS_FAIL", "CONTROLS_PASS", "run_controls"]
