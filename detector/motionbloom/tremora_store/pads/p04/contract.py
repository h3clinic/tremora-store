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
* resampling is two-stage: irregular source to an exact 100 Hz parent by
  deterministic linear interpolation, then parent to 50/30/25 Hz by a frozen
  linear time-invariant polyphase FIR.  A per-output normalized irregular-sinc
  would have a transfer function that changes with the local timestamp
  configuration, so the published response would not be the one the samples
  experienced;
* filters are specified by what each output rate must *preserve*, never by a
  universal cutoff fraction.  A universal fraction attenuated 12 Hz by 6 dB at
  30 Hz, which would have made "30 Hz loses 12 Hz content" partly an artefact
  of the filter rather than of the rate;
* at 25 Hz the 3-10 Hz preservation band and the 10-12 Hz edge-stress band are
  reported separately, because 25 Hz is the rate at which full-band
  preservation stops being physically available;
* source-direct and replay-derived outputs must agree exactly;
* summaries are participant-level.

P0.4 emits no classification, no video association and no P0.5 storage or
retrieval benchmark result.
"""

from __future__ import annotations

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

# --- stage A: irregular source to a uniform parent ------------------------

#: Deterministic linear interpolation between the two source samples that
#: bracket the target time, inside one P0.2.1 segment.  Never extrapolation.
SOURCE_TO_PARENT = "SOURCE_TIME_LINEAR_INTERPOLATION"
PARENT_RATE_HZ = 100

#: 100 Hz carries no anti-alias filter: nothing is being decimated.  What it
#: measures instead is the cost of uniformizing an irregular clock, which is a
#: distinct ablation and is why it is a rate in its own right.
PARENT_HAS_ANTI_ALIAS_FILTER = False

#: Linear interpolation is not transparent.  On an ideal uniform grid its
#: reference response is sinc^2(f / f_parent): about -0.03 dB at 3 Hz, -0.29 dB
#: at 10 Hz and -0.41 dB at 12 Hz.  The irregular case differs; the figure is
#: published as a labelled reference so the 100 Hz result is read as the
#: uniformization ablation it is.
STAGE_A_REFERENCE_RESPONSE = "SINC_SQUARED_IDEAL_UNIFORM"

# --- stage B: parent to derived rate --------------------------------------

#: One frozen linear time-invariant transfer function per derived rate.  100 to
#: 50 and 100 to 25 are integer decimations; 100 to 30 is a rational 3/10
#: polyphase conversion, which puts every 30 Hz output exactly on a 300 Hz grid
#: point and so gives that path a single explicit transfer function rather than
#: a filter followed by a second interpolation.
DERIVED_RATE_METHOD = "FROZEN_POLYPHASE_FIR"

#: Rejected: the effective filter would change with the local timestamp
#: configuration, so a published fixed response would not be the response the
#: samples experienced, and density correction would be mixed into the
#: anti-alias claim.
PER_OUTPUT_WEIGHT_NORMALIZATION = False

#: ``rate -> (upsample, decimate)`` from the 100 Hz parent.
RESAMPLING_RATIOS: dict[int, tuple[int, int]] = {
    50: (1, 2),
    30: (3, 10),
    25: (1, 4),
}

#: Each derived rate's preservation band and where its stopband begins.  The
#: stopband starts at the output Nyquist, so nothing above it survives to fold
#: anywhere in the output band -- stricter than merely protecting the
#: passband, and simpler to state.
PASSBAND_MAX_HZ: dict[int, float] = {50: 12.0, 30: 12.0, 25: 10.0}
STOPBAND_START_HZ: dict[int, float] = {50: 25.0, 30: 15.0, 25: 12.5}

#: 25 Hz is the rate at which the analysis band stops fitting inside the
#: preservation band; 10-12 Hz is then explicitly stressed rather than claimed.
STRESS_BAND_HZ: dict[int, tuple[float, float]] = {25: (10.0, 12.0)}

PASSBAND_RIPPLE_MAX_DB = 0.25
STOPBAND_ATTENUATION_MIN_DB = 60.0
FILTER_DESIGN_ATTENUATION_TARGET_DB = 65.0
FILTER_DESIGN = "KAISER_WINDOW_METHOD_LINEAR_PHASE_TYPE_I"

# --- edges ----------------------------------------------------------------

#: A symmetric kernel that lacks its context at a segment edge produces no
#: output there.  Padding, reflection, endpoint repetition or renormalizing the
#: truncated taps would all keep the sample at the cost of changing the
#: transfer function, which is the one thing this milestone cannot afford.
EDGE_POLICY = "REFUSE_UNSUPPORTED_OUTPUT_SAMPLES"
EDGE_PADDING_ALLOWED = False
TRUNCATED_KERNEL_RENORMALIZATION_ALLOWED = False

#: A derived window is eligible only when its whole four-second interval lies
#: inside supported filtered output.
WINDOW_ELIGIBILITY = "FULLY_INSIDE_SUPPORTED_OUTPUT"


def passband_hz(rate_hz: int) -> float:
    """The band this derived rate is required to preserve."""

    return PASSBAND_MAX_HZ[rate_hz]


def stopband_start_hz(rate_hz: int) -> float:
    """Where this derived rate's stopband begins: its own Nyquist."""

    return STOPBAND_START_HZ[rate_hz]


# --- analysis bands -------------------------------------------------------

CORE_BAND_MIN_HZ = FREQUENCY_MIN_HZ
CORE_BAND_MAX_HZ = 10.0
EDGE_BAND_MIN_HZ = 10.0
EDGE_BAND_MAX_HZ = FREQUENCY_MAX_HZ

#: The reporting split.  At 50 and 30 Hz both bands sit inside the
#: preservation band, so the split is descriptive; at 25 Hz it is physical,
#: because only the core band is preserved.
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
        "source_to_parent": SOURCE_TO_PARENT,
        "parent_rate_hz": PARENT_RATE_HZ,
        "derived_rate_method": DERIVED_RATE_METHOD,
        "per_output_weight_normalization": (
            PER_OUTPUT_WEIGHT_NORMALIZATION
        ),
        "edge_policy": EDGE_POLICY,
        "window_eligibility": WINDOW_ELIGIBILITY,
        "derived_rates_hz": list(DERIVED_RATES_HZ),
        "analysis_bands": list(BANDS),
        "contract_version": P04_CONTRACT_VERSION,
        "resampling_contract_version": RESAMPLING_CONTRACT_VERSION,
    }


__all__ = [
    "BANDS",
    "BLOCKED_DEPENDENCY",
    "CORE_BAND",
    "CORE_BAND_MAX_HZ",
    "CORE_BAND_MIN_HZ",
    "CORE_BIN_COUNT",
    "DERIVED_RATES_HZ",
    "DERIVED_RATE_METHOD",
    "EDGE_BAND",
    "EDGE_BAND_MAX_HZ",
    "EDGE_BAND_MIN_HZ",
    "EDGE_BIN_COUNT",
    "EDGE_PADDING_ALLOWED",
    "EDGE_POLICY",
    "FILTER_DESIGN",
    "FILTER_DESIGN_ATTENUATION_TARGET_DB",
    "FORBIDDEN_P04_SUBSTRINGS",
    "FREQUENCY_BIN_COUNT",
    "FREQUENCY_MAX_HZ",
    "FREQUENCY_MIN_HZ",
    "FREQUENCY_STEP_HZ",
    "GATE_NO_GO",
    "GATE_PASS",
    "GENERIC_SUCCESS_MARKER",
    "GRID_ORIGIN",
    "NATIVE_RATE_LABEL",
    "P04_ARTIFACT_KIND",
    "P04_CONTRACT_VERSION",
    "P04_IMPLEMENTATION_VERSION",
    "P04_SCHEMA_VERSION",
    "PARENT_HAS_ANTI_ALIAS_FILTER",
    "PARENT_RATE_HZ",
    "PASSBAND_MAX_HZ",
    "PASSBAND_RIPPLE_MAX_DB",
    "PER_OUTPUT_WEIGHT_NORMALIZATION",
    "RATES_WITH_EXACT_PICOSECOND_PERIOD",
    "REFERENCE_MILESTONE",
    "RESAMPLING_CONTRACT_VERSION",
    "RESAMPLING_DOMAIN",
    "RESAMPLING_RATIOS",
    "SENSOR_FAMILIES",
    "SOURCE_TO_PARENT",
    "STAGE_A_REFERENCE_RESPONSE",
    "STOPBAND_ATTENUATION_MIN_DB",
    "STOPBAND_START_HZ",
    "STRESS_BAND_HZ",
    "SUCCESS_MARKER",
    "TRUNCATED_KERNEL_RENORMALIZATION_ALLOWED",
    "WINDOW_DURATION_S",
    "WINDOW_ELIGIBILITY",
    "WITHHELD_P04_ARTIFACTS",
    "PadsP04ContractError",
    "assert_no_clinical_or_benchmark_claim",
    "assert_p04_names",
    "authority_block",
    "band_of",
    "passband_hz",
    "stopband_start_hz",
]
