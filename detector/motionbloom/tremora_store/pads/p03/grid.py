"""The frozen tremor-band frequency grid.

3.0 Hz to 12.0 Hz in 0.25 Hz steps: 37 bins, matching the Rayleigh resolution
of a four-second window exactly.  Values are generated from integer
millihertz, so each is an exact binary fraction and two roots agree to the
last bit rather than accumulating a stride.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .contract import (
    FREQUENCY_BIN_COUNT,
    FREQUENCY_MAX_HZ,
    FREQUENCY_MIN_HZ,
    FREQUENCY_STEP_HZ,
    RAYLEIGH_RESOLUTION_HZ,
    SPECTRAL_CONTRACT_VERSION,
    WINDOW_DURATION_S,
)

_MILLIHERTZ_MIN = round(FREQUENCY_MIN_HZ * 1000)
_MILLIHERTZ_STEP = round(FREQUENCY_STEP_HZ * 1000)

WINDOW_DURATION_PS = round(WINDOW_DURATION_S * 10**12)


def frequency_values() -> tuple[float, ...]:
    """The 37 grid frequencies, in hertz."""

    values = tuple(
        (_MILLIHERTZ_MIN + _MILLIHERTZ_STEP * index) / 1000.0
        for index in range(FREQUENCY_BIN_COUNT)
    )
    if values[0] != FREQUENCY_MIN_HZ or values[-1] != FREQUENCY_MAX_HZ:
        raise AssertionError("frequency grid endpoints drifted")
    return values


def grid_record() -> dict[str, Any]:
    """The canonical content of ``pads_p03_frequency_grid.json``."""

    values = frequency_values()
    return {
        "grid_id": grid_id(),
        "frequency_min_hz": FREQUENCY_MIN_HZ,
        "frequency_max_hz": FREQUENCY_MAX_HZ,
        "frequency_step_hz": FREQUENCY_STEP_HZ,
        "frequency_bin_count": FREQUENCY_BIN_COUNT,
        "frequency_values": list(values),
        "window_duration_ns": round(WINDOW_DURATION_S * 10**9),
        "window_duration_ps": WINDOW_DURATION_PS,
        "rayleigh_resolution_hz": RAYLEIGH_RESOLUTION_HZ,
        "spectral_contract_version": SPECTRAL_CONTRACT_VERSION,
    }


def grid_hash() -> str:
    """Hash the grid definition, not the file it is written to."""

    payload = {
        "frequency_values": list(frequency_values()),
        "window_duration_ps": WINDOW_DURATION_PS,
        "spectral_contract_version": SPECTRAL_CONTRACT_VERSION,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def grid_id() -> str:
    return f"tremor-band-3to12hz-0p25-{grid_hash()[:16]}"


def nyquist_hz(dt_ref_ps: int) -> float:
    """The cadence-supported Nyquist limit for one stream."""

    if dt_ref_ps <= 0:
        raise ValueError("dt_ref must be positive")
    return 10**12 / (2.0 * dt_ref_ps)


def grid_within_nyquist(dt_ref_ps: int) -> bool:
    """Whether the frozen grid stays under a stream's own Nyquist limit."""

    return FREQUENCY_MAX_HZ < nyquist_hz(dt_ref_ps)


__all__ = [
    "WINDOW_DURATION_PS",
    "frequency_values",
    "grid_hash",
    "grid_id",
    "grid_record",
    "grid_within_nyquist",
    "nyquist_hz",
]
