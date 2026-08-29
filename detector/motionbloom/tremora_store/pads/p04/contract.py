"""Frozen PADS-P0.4 rate-ablation and anti-aliasing contract.

P0.4 asks one question: how much of the tremor-band spectral content that P0.3
established from irregular source time survives a deliberate reduction to
uniform 100, 50, 30 and 25 Hz grids, each behind a declared anti-alias filter?

The P0.3 native irregular source-time spectrum is the reference.  P0.4 never
modifies it, never recomputes it from a nominal grid, and never treats a
derived rate as the truth against which the native signal is judged.

Boundaries that are part of the contract rather than of the implementation:

* derived signals come from whole P0.2.1 contiguous segments, never from
  individual windows -- a filter run inside a four-second window would see its
  own edges rather than the recording's;
* 30 Hz has no exact picosecond period (1/30 s is 100000000000/3 ps), so grid
  timing is carried as exact rationals and the exactness is stated per rate
  rather than assumed;
* anti-alias coefficients are tabulated, frozen and hashed;
* at 25 Hz the 3-10 Hz core band and the 10-12 Hz edge-stress band are
  reported separately, because the cutoff falls exactly at 10 Hz and the edge
  band sits inside the filter transition;
* source-direct and replay-derived outputs must agree exactly;
* summaries are participant-level.

P0.4 emits no classification, no video association and no P0.5 storage or
retrieval benchmark result.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from ..authority import PADS_DATASET_ID, RELATIVE_TIME_BASIS, VIDEO_PAIRING
from ..p02.contract import (
    CROSS_WRIST_CLOCK_ALIGNMENT,
    SAMPLE_LEVEL_BILATERAL_FUSION_ALLOWED,
    TIMING_AUTHORITY,
)
from ..p03.contract import (
    FREQUENCY_BIN_COUNT,
    FREQUENCY_MAX_HZ,
    FREQUENCY_MIN_HZ,
    FREQUENCY_STEP_HZ,
    SENSOR_FAMILIES,
    WINDOW_DURATION_S,
)

P04_CONTRACT_VERSION = "tremora-pads-rate-ablation-and-anti-aliasing-0.4.0"
P04_SCHEMA_VERSION = "pads-p0.4.0"
P04_IMPLEMENTATION_VERSION = "pads-p04-rate-ablation-1.0.0"
P04_ARTIFACT_KIND = "TREMORA_PADS_P04_RATE_ABLATION_RELEASE_AUDIT"
RESAMPLING_CONTRACT_VERSION = "tremora-pads-bandlimited-nonuniform-resample-0.4.0"

GATE_PASS = "PASS_PADS_RATE_ABLATION_AND_ANTI_ALIASING"
GATE_NO_GO = "NO_GO_PADS_RATE_ABLATION"
BLOCKED_DEPENDENCY = "BLOCKED_P03_DEPENDENCY_UNAVAILABLE"

SUCCESS_MARKER = "_PADS_P04_RATE_ABLATION_SUCCESS"
GENERIC_SUCCESS_MARKER = "_SUCCESS"

# --- the reference --------------------------------------------------------

NATIVE_RATE_LABEL = "NATIVE_IRREGULAR_SOURCE_TIME"

#: The reference is P0.3's spectrum computed at the actual sample times.  A
#: derived rate is compared *to* it; it is never recomputed to suit one.
REFERENCE_MILESTONE = "PADS_P0_3"

# --- derived rates --------------------------------------------------------

DERIVED_RATES_HZ: tuple[int, ...] = (100, 50, 30, 25)

#: Only 30 Hz lacks an exact picosecond period, and that is why grid timing is
#: rational rather than integer-picosecond throughout.
RATES_WITH_EXACT_PICOSECOND_PERIOD: tuple[int, ...] = (100, 50, 25)

#: The uniform grid is anchored at task-local zero, the same origin P0.2.1
#: anchors its window grid on, so a grid point means the same instant on every
#: segment and every rate.
GRID_ORIGIN = "TASK_LOCAL_ZERO"

#: Derived signals are built from a whole contiguous segment.  Resampling
#: inside a window would filter the window's own edges.
RESAMPLING_DOMAIN = "P02_CONTIGUOUS_SEGMENT"

# --- anti-alias filter ----------------------------------------------------

#: One rate-independent kernel expressed in zero-crossings, applied with a
#: per-rate cutoff.  The tabulated coefficients are the filter; the analytic
#: form is how they were produced, not how they are used.
ANTI_ALIAS_KERNEL = "KAISER_WINDOWED_SINC"
KAISER_BETA = 8.6
HALF_WIDTH_ZERO_CROSSINGS = 8
TAPS_PER_ZERO_CROSSING = 128
COEFFICIENT_TABLE_LENGTH = HALF_WIDTH_ZERO_CROSSINGS * TAPS_PER_ZERO_CROSSING + 1

#: ``f_cutoff = CUTOFF_FRACTION * rate / 2``.  At 0.8 this puts the 25 Hz
#: cutoff at exactly 10.0 Hz -- the core/edge boundary -- and the 30 Hz cutoff
#: at exactly 12.0 Hz, the top of the analysis grid.  Both facts are declared
#: rather than discovered later.
CUTOFF_FRACTION = Fraction(4, 5)

#: Irregular input means local sample density varies, so applied weights are
#: normalized to unit sum per output sample.  Without it the passband would
#: ripple with the input spacing rather than with the filter.
WEIGHT_NORMALIZATION = "UNIT_SUM_PER_OUTPUT_SAMPLE"

#: A derived sample is refused rather than invented when its support holds too
#: few input samples to normalize meaningfully.
MINIMUM_TAPS_PER_OUTPUT_SAMPLE = 4


def cutoff_hz(rate_hz: int) -> Fraction:
    """The exact anti-alias cutoff for one derived rate."""

    return CUTOFF_FRACTION * Fraction(rate_hz, 2)


# --- analysis bands -------------------------------------------------------

CORE_BAND_MIN_HZ = FREQUENCY_MIN_HZ
CORE_BAND_MAX_HZ = 10.0
EDGE_BAND_MIN_HZ = 10.0
EDGE_BAND_MAX_HZ = FREQUENCY_MAX_HZ

CORE_BAND = "CORE_3_TO_10_HZ"
EDGE_BAND = "EDGE_STRESS_10_TO_12_HZ"
BANDS: tuple[str, ...] = (CORE_BAND, EDGE_BAND)


def band_of(frequency_hz: float) -> str:
    """Which band one grid frequency belongs to; the split is at 10 Hz."""

    return CORE_BAND if frequency_hz <= CORE_BAND_MAX_HZ else EDGE_BAND


#: 29 core bins (3.00-10.00) and 8 edge bins (10.25-12.00) partition the 37.
CORE_BIN_COUNT = 29
EDGE_BIN_COUNT = 8

# --- claim boundary -------------------------------------------------------

WITHHELD_P04_ARTIFACTS: dict[str, int] = {
    "classification_tables": 0,
    "diagnosis_tables": 0,
    "generic_success_markers": 0,
    "retrieval_latency_tables": 0,
    "severity_tables": 0,
    "storage_benchmark_tables": 0,
    "tremor_detection_tables": 0,
    "video_association_tables": 0,
}

#: P0.4 legitimately uses resampling and anti-aliasing vocabulary, which P0.3
#: forbade.  What stays forbidden is the clinical and P0.5 vocabulary.
FORBIDDEN_P04_SUBSTRINGS: tuple[str, ...] = (
    "classification",
    "diagnosis",
    "hdf5",
    "latency",
    "severity",
    "throughput",
)


class PadsP04ContractError(ValueError):
    """Raised when a P0.4 artifact would exceed its milestone."""


def assert_no_clinical_or_benchmark_claim(names: object) -> None:
    """Refuse a name that implies a clinical result or a P0.5 benchmark."""

    offending = sorted({
        str(name)
        for name in names  # type: ignore[union-attr]
        if any(
            token in str(name).casefold()
            for token in FORBIDDEN_P04_SUBSTRINGS
        )
    })
    if offending:
        raise PadsP04ContractError(
            f"P0.4 ablates sampling rate only; {offending!r} implies a "
            "clinical or storage-benchmark milestone"
        )


def assert_p04_names(names: object) -> None:
    """Apply the inherited video screen and the P0.4 milestone screen."""

    from ..authority import assert_no_paired_claim

    materialized = list(names)  # type: ignore[call-overload]
    assert_no_paired_claim(materialized)
    assert_no_clinical_or_benchmark_claim(materialized)


def authority_block() -> dict[str, Any]:
    """The authority every P0.4 record carries."""

    return {
        "dataset_id": PADS_DATASET_ID,
        "timing_authority": TIMING_AUTHORITY,
        "time_basis": RELATIVE_TIME_BASIS,
        "video_pairing": VIDEO_PAIRING,
        "hardware_sync_claim": False,
        "cross_wrist_clock_alignment": CROSS_WRIST_CLOCK_ALIGNMENT,
        "sample_level_bilateral_fusion_allowed": (
            SAMPLE_LEVEL_BILATERAL_FUSION_ALLOWED
        ),
        "reference": NATIVE_RATE_LABEL,
        "reference_milestone": REFERENCE_MILESTONE,
        "resampling_domain": RESAMPLING_DOMAIN,
        "grid_origin": GRID_ORIGIN,
        "anti_alias_kernel": ANTI_ALIAS_KERNEL,
        "weight_normalization": WEIGHT_NORMALIZATION,
        "derived_rates_hz": list(DERIVED_RATES_HZ),
        "analysis_bands": list(BANDS),
        "contract_version": P04_CONTRACT_VERSION,
        "resampling_contract_version": RESAMPLING_CONTRACT_VERSION,
    }


__all__ = [
    "ANTI_ALIAS_KERNEL",
    "BANDS",
    "BLOCKED_DEPENDENCY",
    "COEFFICIENT_TABLE_LENGTH",
    "CORE_BAND",
    "CORE_BAND_MAX_HZ",
    "CORE_BAND_MIN_HZ",
    "CORE_BIN_COUNT",
    "CUTOFF_FRACTION",
    "DERIVED_RATES_HZ",
    "EDGE_BAND",
    "EDGE_BAND_MAX_HZ",
    "EDGE_BAND_MIN_HZ",
    "EDGE_BIN_COUNT",
    "FORBIDDEN_P04_SUBSTRINGS",
    "FREQUENCY_BIN_COUNT",
    "FREQUENCY_MAX_HZ",
    "FREQUENCY_MIN_HZ",
    "FREQUENCY_STEP_HZ",
    "GATE_NO_GO",
    "GATE_PASS",
    "GENERIC_SUCCESS_MARKER",
    "GRID_ORIGIN",
    "HALF_WIDTH_ZERO_CROSSINGS",
    "KAISER_BETA",
    "MINIMUM_TAPS_PER_OUTPUT_SAMPLE",
    "NATIVE_RATE_LABEL",
    "P04_ARTIFACT_KIND",
    "P04_CONTRACT_VERSION",
    "P04_IMPLEMENTATION_VERSION",
    "P04_SCHEMA_VERSION",
    "RATES_WITH_EXACT_PICOSECOND_PERIOD",
    "REFERENCE_MILESTONE",
    "RESAMPLING_CONTRACT_VERSION",
    "RESAMPLING_DOMAIN",
    "SENSOR_FAMILIES",
    "SUCCESS_MARKER",
    "TAPS_PER_ZERO_CROSSING",
    "WEIGHT_NORMALIZATION",
    "WINDOW_DURATION_S",
    "WITHHELD_P04_ARTIFACTS",
    "PadsP04ContractError",
    "assert_no_clinical_or_benchmark_claim",
    "assert_p04_names",
    "authority_block",
    "band_of",
    "cutoff_hz",
]
