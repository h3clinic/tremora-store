"""In-process synthetic controls for the two-stage resampler.

Run by the audit itself rather than deferred to a test suite that may not have
executed.  Timings come from a fixed arithmetic pattern, not an RNG, so the
controls answer the same way on every execution.

The constant-input control is the load-bearing one: it shows that the realized
per-output DC gain is exactly the published polyphase branch gain, which is
only true if no phase-specific normalization is being applied.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contract import (
    DERIVED_RATES_HZ,
    PARENT_RATE_HZ,
    PASSBAND_MAX_HZ,
    PASSBAND_RIPPLE_MAX_DB,
    STOPBAND_ATTENUATION_MIN_DB,
    STOPBAND_START_HZ,
)
from .filters import (
    FILTER_SPECS,
    frequency_response,
    polyphase_dc_gains,
)
from .rational_time import grid_for
from .resample import (
    ResampleError,
    derive_support,
    derive_window,
    window_eligibility,
    window_times_seconds,
)

CONTROLS_PASS = "RESAMPLING_CONTROLS_PASS"
CONTROLS_FAIL = "RESAMPLING_CONTROLS_FAIL"

_SECONDS = 40.0
_GUARD_S = 2.0


def _segment(duration_s: float = _SECONDS) -> np.ndarray:
    """Deterministic irregular source times, in picoseconds."""

    count = int(duration_s * PARENT_RATE_HZ) + 1
    deltas = np.asarray(
        [
            1.0e10 + 4.0e8 * math.sin(0.37 * index)
            for index in range(count)
        ],
        dtype=np.float64,
    )
    times = np.concatenate([[0.0], np.cumsum(deltas)[:-1]])
    return np.rint(times).astype(np.int64)


def _amplitude(times_s: list[float], values: np.ndarray, hz: float) -> float:
    design = np.column_stack([
        np.sin(2.0 * np.pi * hz * np.asarray(times_s)),
        np.cos(2.0 * np.pi * hz * np.asarray(times_s)),
    ])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(np.hypot(*coefficients))


def _derive(times_ps: np.ndarray, signal: np.ndarray, rate_hz: int):
    support = derive_support(times_ps, rate_hz)
    if not support.supported:
        return None, None
    guard = int(_GUARD_S * 1e12)
    start = (support.first_time_ps or 0) + guard
    end = (support.last_time_ps or 0) - guard
    status, ordinals = window_eligibility(support, start, end)
    if status != "WINDOW_ELIGIBLE":
        return None, None
    values = derive_window(
        times_ps, [signal], rate_hz=rate_hz, support=support,
        ordinals=ordinals,
    )[0]
    return window_times_seconds(rate_hz, ordinals), values


def run_controls() -> dict[str, Any]:
    """Run every resampling control and report each outcome individually."""

    outcomes: dict[str, bool] = {}
    measured: dict[str, Any] = {}
    times_ps = _segment()
    seconds = times_ps / 1e12

    # --- constant input: realized gain is the published branch gain --------
    constant_ok = True
    ripple_match = True
    for rate in DERIVED_RATES_HZ:
        _, values = _derive(times_ps, np.ones(times_ps.size), rate)
        if values is None:
            constant_ok = False
            continue
        published = polyphase_dc_gains(rate) if rate in FILTER_SPECS else [1.0]
        observed_ripple = float(values.max() - values.min())
        published_ripple = max(published) - min(published)
        measured[f"constant_input_{rate}"] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "observed_ripple": observed_ripple,
            "published_branch_ripple": published_ripple,
        }
        constant_ok &= abs(float(values.mean()) - 1.0) < 1e-6
        ripple_match &= abs(observed_ripple - published_ripple) < 1e-9
        # No accumulated drift, stated exactly: for a constant input every
        # output of a given phase must carry the identical branch gain.  A
        # half-versus-half mean would instead measure how the phases happen
        # to divide between the halves, which is the published ripple rather
        # than drift.
        upsample = (
            FILTER_SPECS[rate].upsample if rate in FILTER_SPECS else 1
        )
        within_phase = max(
            float(np.ptp(values[phase::upsample]))
            for phase in range(upsample)
            if values[phase::upsample].size
        )
        measured[f"constant_input_{rate}"]["within_phase_spread"] = (
            within_phase
        )
        constant_ok &= within_phase < 1e-12
    outcomes["constant_input_unit_gain"] = constant_ok
    outcomes["phase_ripple_matches_published_branch_gains"] = ripple_match
    outcomes["no_per_phase_normalization"] = ripple_match and any(
        max(polyphase_dc_gains(rate)) - min(polyphase_dc_gains(rate)) > 0.0
        for rate in FILTER_SPECS
    )

    # --- preservation band survives ---------------------------------------
    preserved = True
    for rate in DERIVED_RATES_HZ:
        probe = PASSBAND_MAX_HZ.get(rate, 12.0)
        tone = np.sin(2.0 * np.pi * probe * seconds)
        grid_times, values = _derive(times_ps, tone, rate)
        if values is None:
            preserved = False
            continue
        amplitude = _amplitude(grid_times, values, probe)
        measured[f"preserved_{rate}"] = {
            "hz": probe, "amplitude": amplitude,
            "db": 20.0 * math.log10(max(amplitude, 1e-12)),
        }
        # Stage A's own sinc^2 loss is expected and is why the tolerance is
        # not the filter's ripple budget alone.
        preserved &= amplitude > 0.9
    outcomes["preservation_band_survives"] = preserved

    # --- an out-of-band tone that would fold into the band is suppressed ---
    alias = True
    for rate, intruder in ((30, 18.0), (25, 15.0), (50, 38.0)):
        tone = np.sin(2.0 * np.pi * intruder * seconds)
        grid_times, values = _derive(times_ps, tone, rate)
        if values is None:
            alias = False
            continue
        folded = abs(intruder - rate * round(intruder / rate))
        amplitude = _amplitude(grid_times, values, folded)
        measured[f"alias_{rate}"] = {
            "intruder_hz": intruder, "folds_to_hz": folded,
            "amplitude": amplitude,
            "db": 20.0 * math.log10(max(amplitude, 1e-12)),
        }
        alias &= amplitude < 1e-2
    outcomes["out_of_band_tone_does_not_fold_into_the_band"] = alias

    # --- declared filter specifications ------------------------------------
    specification = True
    for rate in sorted(FILTER_SPECS):
        edge = frequency_response(
            np.array([PASSBAND_MAX_HZ[rate]]), rate
        )[0]
        working = FILTER_SPECS[rate].working_rate_hz
        stopband = frequency_response(
            np.linspace(STOPBAND_START_HZ[rate], working / 2.0, 2048), rate
        )
        specification &= abs(20.0 * math.log10(edge)) <= (
            PASSBAND_RIPPLE_MAX_DB
        )
        specification &= -20.0 * math.log10(
            max(float(stopband.max()), 1e-16)
        ) >= STOPBAND_ATTENUATION_MIN_DB
    outcomes["filters_meet_their_declared_specification"] = specification

    # --- exact derived timing ----------------------------------------------
    exact = True
    for rate in DERIVED_RATES_HZ:
        grid = grid_for(rate)
        steps = {
            grid.sample_seconds(k + 1) - grid.sample_seconds(k)
            for k in (0, 7, 999)
        }
        exact &= steps == {1 / grid.rate_hz}
    outcomes["derived_grids_have_no_cumulative_drift"] = exact
    outcomes["thirty_hertz_period_is_not_integer_picoseconds"] = (
        grid_for(30).sample_picoseconds_exact(1) is None
    )

    # --- edges refuse rather than pad --------------------------------------
    short = _segment(duration_s=1.0)
    refused = all(
        not derive_support(short, rate).supported
        for rate in (30, 25)
    )
    outcomes["a_segment_shorter_than_the_kernel_yields_nothing"] = refused

    padding_refused = False
    support = derive_support(times_ps, 25)
    try:
        derive_window(
            times_ps, [np.ones(times_ps.size)], rate_hz=25,
            support=support, ordinals=range(4),
        )
    except (ResampleError, Exception):  # noqa: BLE001 - any refusal counts
        padding_refused = True
    outcomes["unsupported_output_is_refused_not_padded"] = padding_refused

    passed = all(outcomes.values())
    return {
        "status": CONTROLS_PASS if passed else CONTROLS_FAIL,
        "controls_total": len(outcomes),
        "controls_passed": sum(1 for value in outcomes.values() if value),
        "controls": dict(sorted(outcomes.items())),
        "measured": {key: measured[key] for key in sorted(measured)},
    }


__all__ = ["CONTROLS_FAIL", "CONTROLS_PASS", "run_controls"]
