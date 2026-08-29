"""Synthetic controls for the nonuniform tremor-band spectral kernel.

These validate the spectral implementation itself.  The source-versus-replay
audit tests a different boundary -- source parsing against indexed replay --
and legitimately uses this same kernel on both sides.
"""

from __future__ import annotations

import numpy as np
import pytest
from motionbloom.tremora_store.pads.p03.contract import FREQUENCY_BIN_COUNT
from motionbloom.tremora_store.pads.p03.grid import frequency_values
from motionbloom.tremora_store.pads.p03.kernel import (
    SpectralKernelError,
    assert_no_gap,
    detrend_linear,
    family_spectrum,
    hann_weights,
    input_content_sha256,
    relative_seconds,
)

GRID = np.asarray(frequency_values(), dtype=np.float64)
DURATION = 4.0


def jittered_times(
    count: int, *, base: float = 0.01, jitter: float = 0.002, seed: int = 11
) -> np.ndarray:
    """Irregular sample times, like the release's own device clock."""

    rng = np.random.default_rng(seed)
    deltas = rng.uniform(base - jitter, base + jitter, count)
    return np.concatenate([[0.0], np.cumsum(deltas)[:-1]])


def picoseconds(times: np.ndarray) -> list[int]:
    return [round(float(value) * 1e12) for value in times]


def tone(times: np.ndarray, hz: float, phase: float = 0.0) -> np.ndarray:
    return np.sin(2.0 * np.pi * hz * times + phase)


def spectrum(times: np.ndarray, signal: np.ndarray, **kwargs):
    zero = np.zeros_like(signal)
    return family_spectrum(
        "GYROSCOPE", picoseconds(times), [signal, zero, zero],
        duration_s=DURATION, **kwargs,
    )


# --- the grid -------------------------------------------------------------


def test_the_frequency_grid_remains_exactly_thirty_seven_bins() -> None:
    times = jittered_times(400)
    result = spectrum(times, tone(times, 5.0))
    assert result.aggregate.shape == (FREQUENCY_BIN_COUNT,)
    assert result.axis_x.shape == (37,)
    assert result.normalized_aggregate.shape == (37,)
    with pytest.raises(SpectralKernelError):
        spectrum(times, tone(times, 5.0), frequencies=GRID[:10])


# --- recovering a known tone ---------------------------------------------


@pytest.mark.parametrize("hz", (5.0, 8.0))
def test_an_irregularly_sampled_tone_peaks_at_its_own_frequency(
    hz: float,
) -> None:
    times = jittered_times(400)
    result = spectrum(times, tone(times, hz))
    assert result.dominant_frequency_hz == pytest.approx(hz, abs=1e-9)
    assert result.peak_to_median_ratio > 100.0


@pytest.mark.parametrize("phase", (0.0, 0.7, 1.9, 3.0))
def test_a_phase_shift_preserves_the_power_spectrum(phase: float) -> None:
    times = jittered_times(400)
    reference = spectrum(times, tone(times, 5.0)).aggregate
    shifted = spectrum(times, tone(times, 5.0, phase)).aggregate
    assert np.max(np.abs(shifted - reference)) / np.max(reference) < 0.05
    assert np.argmax(shifted) == np.argmax(reference)


def test_an_axis_rotation_preserves_summed_three_axis_power() -> None:
    times = jittered_times(400)
    x, y = tone(times, 5.0), tone(times, 5.0, np.pi / 2)
    z = np.zeros_like(x)
    angle = 0.7
    before = family_spectrum(
        "GYROSCOPE", picoseconds(times), [x, y, z], duration_s=DURATION
    ).aggregate
    after = family_spectrum(
        "GYROSCOPE", picoseconds(times),
        [np.cos(angle) * x - np.sin(angle) * y,
         np.sin(angle) * x + np.cos(angle) * y, z],
        duration_s=DURATION,
    ).aggregate
    # An orthonormal rotation of the sensor axes leaves the summed power
    # invariant to floating-point roundoff.
    assert np.max(np.abs(after - before)) / np.max(before) < 1e-12


def test_linear_trend_removal_suppresses_low_frequency_leakage() -> None:
    times = jittered_times(400)
    clean = tone(times, 5.0)
    ramped = clean + 3.0 * times
    without = spectrum(times, clean).aggregate
    with_ramp = spectrum(times, ramped).aggregate
    assert np.max(np.abs(with_ramp - without)) / np.max(without) < 1e-9
    # The 3 Hz bin does not inflate: the ramp is gone, not smeared.
    assert with_ramp[0] == pytest.approx(without[0], rel=1e-6)


# --- the clock ------------------------------------------------------------


@pytest.mark.parametrize("count", tuple(range(395, 406)))
def test_every_observed_window_length_is_accepted(count: int) -> None:
    # The materialized corpus carries 395 to 405 samples per four-second
    # window; 400 is common but never assumed.
    times = jittered_times(count)
    result = spectrum(times, tone(times, 6.0))
    assert result.dominant_frequency_hz == pytest.approx(6.0, abs=1e-9)


def test_source_timestamps_not_ordinal_over_rate_control_phase() -> None:
    # A 5 Hz tone sampled on a 90 Hz clock.  Reading the ordinal as i/100
    # would report 5.56 Hz; the actual timestamps report 5 Hz.
    times = np.arange(360) / 90.0
    result = spectrum(times, tone(times, 5.0))
    assert result.dominant_frequency_hz == pytest.approx(5.0, abs=1e-9)
    assert result.dominant_frequency_hz != pytest.approx(5.5, abs=0.1)


def test_the_sample_count_is_never_used_as_the_clock() -> None:
    # Identical values, two different time vectors: the spectra must differ,
    # which they cannot do if the count is driving the transform.
    fast = np.arange(400) / 100.0
    slow = np.arange(400) / 80.0
    values = tone(fast, 5.0)
    assert spectrum(fast, values).dominant_frequency_hz != (
        spectrum(slow, values).dominant_frequency_hz
    )


def test_non_monotonic_input_is_refused() -> None:
    times = jittered_times(400)
    swapped = times.copy()
    swapped[10], swapped[11] = swapped[11], swapped[10]
    with pytest.raises(SpectralKernelError):
        spectrum(swapped, tone(times, 5.0))
    duplicated = times.copy()
    duplicated[10] = duplicated[9]
    with pytest.raises(SpectralKernelError):
        spectrum(duplicated, tone(times, 5.0))


def test_gap_crossing_input_is_refused() -> None:
    times = jittered_times(400)
    threshold = 30_000_000_000  # 30 ms, the corpus policy
    assert_no_gap(picoseconds(times), threshold)
    gapped = times.copy()
    gapped[200:] += 0.05
    with pytest.raises(SpectralKernelError):
        assert_no_gap(picoseconds(gapped), threshold)
    with pytest.raises(SpectralKernelError):
        assert_no_gap(picoseconds(times), 0)


# --- raw axes -------------------------------------------------------------


def test_vector_magnitude_is_not_used_as_the_primary_spectrum() -> None:
    times = jittered_times(400)
    signal = tone(times, 5.0)
    raw = spectrum(times, signal)
    magnitude = spectrum(times, np.abs(signal))
    # |sin(2*pi*f*t)| has a fundamental at 2f: taking magnitude first would
    # report 10 Hz for a 5 Hz tremor.
    assert raw.dominant_frequency_hz == pytest.approx(5.0, abs=1e-9)
    assert magnitude.dominant_frequency_hz == pytest.approx(10.0, abs=1e-9)


def test_the_three_axes_are_summed_not_combined_in_quadrature() -> None:
    times = jittered_times(400)
    x, y, z = tone(times, 5.0), tone(times, 7.0), np.zeros(400)
    result = family_spectrum(
        "GYROSCOPE", picoseconds(times), [x, y, z], duration_s=DURATION
    )
    assert np.allclose(
        result.aggregate, result.axis_x + result.axis_y + result.axis_z
    )
    assert result.axis_x[int(np.argmax(result.axis_x))] > 0.0
    assert GRID[int(np.argmax(result.axis_x))] == pytest.approx(5.0)
    assert GRID[int(np.argmax(result.axis_y))] == pytest.approx(7.0)


# --- determinism and helpers ---------------------------------------------


def test_the_same_input_hashes_to_the_same_spectrum() -> None:
    times = jittered_times(400)
    signal = tone(times, 5.0)
    first = spectrum(times, signal)
    second = spectrum(times, signal)
    assert first.content_sha256() == second.content_sha256()
    assert input_content_sha256(
        picoseconds(times), [signal, np.zeros(400), np.zeros(400)]
    ) == input_content_sha256(
        picoseconds(times), [signal, np.zeros(400), np.zeros(400)]
    )
    changed = spectrum(times, tone(times, 6.0))
    assert changed.content_sha256() != first.content_sha256()


def test_the_hann_weight_is_defined_on_the_actual_times() -> None:
    times = jittered_times(400)
    tau = relative_seconds(picoseconds(times))
    weights = hann_weights(tau, DURATION)
    assert weights[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(weights >= -1e-12)
    assert np.all(weights <= 1.0 + 1e-12)
    with pytest.raises(SpectralKernelError):
        hann_weights(tau, 0.0)


def test_detrending_removes_exactly_the_fitted_line() -> None:
    tau = np.linspace(0.0, 4.0, 400)
    line = 2.0 + 0.5 * tau
    assert np.allclose(detrend_linear(tau, line), 0.0, atol=1e-12)
    assert np.allclose(
        detrend_linear(tau, line + np.sin(tau)),
        detrend_linear(tau, np.sin(tau)),
        atol=1e-12,
    )


def test_a_window_shorter_than_two_samples_is_refused() -> None:
    with pytest.raises(SpectralKernelError):
        relative_seconds([0])
    with pytest.raises(SpectralKernelError):
        family_spectrum(
            "GYROSCOPE", [0, 1], [[1.0, 2.0], [1.0, 2.0]],
            duration_s=DURATION,
        )


def test_a_non_finite_axis_value_is_refused() -> None:
    times = jittered_times(400)
    signal = tone(times, 5.0)
    signal[7] = np.nan
    with pytest.raises(SpectralKernelError):
        spectrum(times, signal)
