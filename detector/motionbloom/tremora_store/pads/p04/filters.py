"""Frozen, hashed anti-alias filters, one per derived rate.

Each filter is specified by what its output rate must *preserve*, never by a
universal cutoff fraction:

    50 Hz   pass 0-12 Hz,  stop from 25 Hz    (decimate 2 at 100 Hz)
    30 Hz   pass 0-12 Hz,  stop from 15 Hz    (3/10 polyphase at 300 Hz)
    25 Hz   pass 0-10 Hz,  stop from 12.5 Hz  (decimate 4 at 100 Hz)

The stopband starts at the output Nyquist, so nothing above it survives to fold
anywhere in the output band.  Each is a linear-phase Type I FIR designed by the
Kaiser window method with unit DC gain, so the coefficients are symmetric, the
group delay is an exact integer, and the transfer function is fixed for every
output sample.  Nothing is renormalized at run time.

100 Hz has no filter here: it is the uniformized parent, and nothing is being
decimated into it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from .contract import (
    FILTER_DESIGN,
    FILTER_DESIGN_ATTENUATION_TARGET_DB,
    PARENT_RATE_HZ,
    PASSBAND_MAX_HZ,
    PASSBAND_RIPPLE_MAX_DB,
    RESAMPLING_RATIOS,
    STOPBAND_ATTENUATION_MIN_DB,
    STOPBAND_START_HZ,
    STRESS_BAND_HZ,
)


class AntiAliasError(ValueError):
    """Raised when a frozen filter would not meet its own specification."""


@dataclass(frozen=True, slots=True)
class FilterSpec:
    """What one derived rate's filter has to do."""

    rate_hz: int
    upsample: int
    decimate: int
    passband_hz: float
    stopband_start_hz: float
    attenuation_target_db: float = FILTER_DESIGN_ATTENUATION_TARGET_DB

    @property
    def working_rate_hz(self) -> int:
        """The rate the filter actually runs at, after any upsampling."""

        return PARENT_RATE_HZ * self.upsample

    @property
    def cutoff_hz(self) -> float:
        """Midpoint of the transition band; a design detail, not a target."""

        return (self.passband_hz + self.stopband_start_hz) / 2.0

    @property
    def transition_hz(self) -> float:
        return self.stopband_start_hz - self.passband_hz


FILTER_SPECS: dict[int, FilterSpec] = {
    rate: FilterSpec(
        rate_hz=rate,
        upsample=RESAMPLING_RATIOS[rate][0],
        decimate=RESAMPLING_RATIOS[rate][1],
        passband_hz=PASSBAND_MAX_HZ[rate],
        stopband_start_hz=STOPBAND_START_HZ[rate],
    )
    for rate in sorted(RESAMPLING_RATIOS, reverse=True)
}


def _kaiser_beta(attenuation_db: float) -> float:
    if attenuation_db > 50.0:
        return 0.1102 * (attenuation_db - 8.7)
    if attenuation_db >= 21.0:  # pragma: no cover - designs here exceed 50 dB
        return (
            0.5842 * (attenuation_db - 21.0) ** 0.4
            + 0.07886 * (attenuation_db - 21.0)
        )
    return 0.0  # pragma: no cover


def _tap_count(spec: FilterSpec, beta_attenuation: float) -> int:
    transition = 2.0 * np.pi * spec.transition_hz / spec.working_rate_hz
    count = int(np.ceil((beta_attenuation - 8.0) / (2.285 * transition))) + 1
    return count + 1 if count % 2 == 0 else count


@cache
def design(rate_hz: int) -> np.ndarray:
    """The frozen coefficients for one derived rate.

    Unit DC gain before the upsampling gain, then scaled by the interpolation
    factor so the whole chain has unit DC gain.
    """

    try:
        spec = FILTER_SPECS[rate_hz]
    except KeyError as exc:
        raise AntiAliasError(f"{rate_hz} Hz has no frozen filter") from exc
    beta = _kaiser_beta(spec.attenuation_target_db)
    count = _tap_count(spec, spec.attenuation_target_db)
    half = (count - 1) // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    taps = np.sinc(
        2.0 * spec.cutoff_hz * offsets / spec.working_rate_hz
    ) * np.kaiser(count, beta)
    taps = taps / taps.sum() * spec.upsample
    taps.setflags(write=False)
    return taps


def group_delay_taps(rate_hz: int) -> int:
    """Exact integer group delay, in working-rate samples."""

    return (design(rate_hz).size - 1) // 2


def filter_sha256(rate_hz: int) -> str:
    """Hash the exact float bytes of one frozen coefficient set."""

    return hashlib.sha256(
        np.ascontiguousarray(design(rate_hz), dtype=np.float64).tobytes()
    ).hexdigest()


def coefficients_sha256() -> str:
    """One hash binding every frozen coefficient set."""

    digest = hashlib.sha256()
    for rate in sorted(FILTER_SPECS):
        digest.update(str(rate).encode("ascii"))
        digest.update(b"\x1f")
        digest.update(filter_sha256(rate).encode("ascii"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def frequency_response(
    frequencies_hz: np.ndarray, rate_hz: int
) -> np.ndarray:
    """Magnitude response of one frozen filter at its working rate."""

    taps = design(rate_hz)
    spec = FILTER_SPECS[rate_hz]
    offsets = np.arange(taps.size) - (taps.size - 1) // 2
    phase = np.exp(
        -2j * np.pi
        * np.outer(
            np.asarray(frequencies_hz, dtype=np.float64)
            / spec.working_rate_hz,
            offsets,
        )
    )
    return np.abs(phase @ taps) / spec.upsample


def measured_specification(rate_hz: int) -> dict[str, float]:
    """What the frozen coefficients actually do, measured not asserted."""

    spec = FILTER_SPECS[rate_hz]
    passband = frequency_response(
        np.linspace(0.0, spec.passband_hz, 512), rate_hz
    )
    stopband = frequency_response(
        np.linspace(
            spec.stopband_start_hz, spec.working_rate_hz / 2.0, 4096
        ),
        rate_hz,
    )
    taps = design(rate_hz)
    return {
        "taps": int(taps.size),
        "group_delay_taps": float(group_delay_taps(rate_hz)),
        "dc_gain": float(frequency_response(np.array([0.0]), rate_hz)[0]),
        "passband_ripple_db": float(
            20.0 * np.log10(passband.max() / passband.min())
        ),
        "passband_edge_db": float(
            20.0 * np.log10(
                frequency_response(
                    np.array([spec.passband_hz]), rate_hz
                )[0]
            )
        ),
        "stopband_attenuation_db": float(
            -20.0 * np.log10(max(float(stopband.max()), 1e-16))
        ),
        "symmetric": bool(np.array_equal(taps, taps[::-1])),
    }


def assert_meets_specification(rate_hz: int) -> None:
    """Refuse a filter that does not meet the frozen contract."""

    measured = measured_specification(rate_hz)
    if abs(measured["dc_gain"] - 1.0) > 1e-9:
        raise AntiAliasError(f"{rate_hz} Hz filter has no unit DC gain")
    if not measured["symmetric"]:
        raise AntiAliasError(f"{rate_hz} Hz filter is not symmetric")
    if measured["passband_ripple_db"] > PASSBAND_RIPPLE_MAX_DB:
        raise AntiAliasError(
            f"{rate_hz} Hz passband ripple exceeds "
            f"{PASSBAND_RIPPLE_MAX_DB} dB")
    if measured["stopband_attenuation_db"] < STOPBAND_ATTENUATION_MIN_DB:
        raise AntiAliasError(
            f"{rate_hz} Hz stopband is under "
            f"{STOPBAND_ATTENUATION_MIN_DB} dB")


def stage_a_reference_response(
    frequencies_hz: np.ndarray | None = None,
) -> dict[str, float]:
    """Linear interpolation's response on an ideal uniform parent grid.

    Published as a labelled reference, not as a claim about the irregular
    case: it is why 100 Hz is an ablation in its own right rather than a
    transparent pass-through.
    """

    probe = (
        np.array([3.0, 10.0, 12.0])
        if frequencies_hz is None else np.asarray(frequencies_hz)
    )
    magnitude = np.sinc(probe / PARENT_RATE_HZ) ** 2
    return {
        f"{float(frequency):g}_hz_db": float(20.0 * np.log10(value))
        for frequency, value in zip(probe, magnitude, strict=True)
    }


def polyphase_dc_gains(rate_hz: int) -> list[float]:
    """Each polyphase branch's DC sum.

    They are not identical: at 30 Hz they differ by about 5.6e-6, which is
    0.000049 dB and far inside the ripple budget.  Reported rather than
    normalized away, because normalizing each branch would replace one frozen
    transfer function with three.
    """

    taps = design(rate_hz)
    upsample = FILTER_SPECS[rate_hz].upsample
    return [float(taps[phase::upsample].sum()) for phase in range(upsample)]


def filter_manifest() -> dict[str, Any]:
    """The frozen filter definitions published in every P0.4 record."""

    entries: dict[str, Any] = {}
    for rate, spec in sorted(FILTER_SPECS.items()):
        measured = measured_specification(rate)
        entries[str(rate)] = {
            "upsample": spec.upsample,
            "decimate": spec.decimate,
            "working_rate_hz": spec.working_rate_hz,
            "passband_hz": spec.passband_hz,
            "stopband_start_hz": spec.stopband_start_hz,
            "cutoff_hz": spec.cutoff_hz,
            "coefficients_sha256": filter_sha256(rate),
            "polyphase_dc_gains": polyphase_dc_gains(rate),
            **measured,
        }
    return {
        "design": FILTER_DESIGN,
        "attenuation_target_db": FILTER_DESIGN_ATTENUATION_TARGET_DB,
        "passband_ripple_max_db": PASSBAND_RIPPLE_MAX_DB,
        "stopband_attenuation_min_db": STOPBAND_ATTENUATION_MIN_DB,
        "coefficients_sha256": coefficients_sha256(),
        "stress_band_hz": {
            str(rate): list(band) for rate, band in STRESS_BAND_HZ.items()
        },
        "stage_a_reference_response_db": stage_a_reference_response(),
        "filters": entries,
    }


__all__ = [
    "FILTER_SPECS",
    "AntiAliasError",
    "FilterSpec",
    "assert_meets_specification",
    "coefficients_sha256",
    "design",
    "filter_manifest",
    "filter_sha256",
    "frequency_response",
    "group_delay_taps",
    "measured_specification",
    "polyphase_dc_gains",
    "stage_a_reference_response",
]
