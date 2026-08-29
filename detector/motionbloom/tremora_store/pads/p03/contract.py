"""Frozen PADS-P0.3 spectral-preservation contract.

P0.3 proves one thing: tremor-band spectral quantities computed directly from
the original PADS device files are preserved exactly when the same windows are
retrieved through the P0.2.1 indexed store.

It is a storage-and-signal-integrity result.  It is not a disease
classification, a tremor-detection accuracy, a video-IMU, a bilateral
sample-fusion or a sampling-rate-ablation result, and
:data:`WITHHELD_P03_ARTIFACTS` publishes a zero count for each so the boundary
is visible rather than inferred from an absence.

Two facts about the materialized corpus drive the numeric policy.

*Sample counts are not fixed.*  The 50,676 P0.2.1 windows carry between 395
and 405 samples, most often 400, 398 or 397.  Nothing here may assume a count,
and a test walks the whole observed range.

*Cadence is per stream.*  Reference intervals run from 9.9199 ms to 10.0800 ms
across the corpus, so Nyquist eligibility is decided from each stream's stored
``dt_ref_ps`` and never from the declared 100 Hz.
"""

from __future__ import annotations

from typing import Any

from ..authority import PADS_DATASET_ID, RELATIVE_TIME_BASIS, VIDEO_PAIRING
from ..p02.contract import (
    CROSS_WRIST_CLOCK_ALIGNMENT,
    SAMPLE_LEVEL_BILATERAL_FUSION_ALLOWED,
    TIMING_AUTHORITY,
)

P03_CONTRACT_VERSION = "tremora-pads-source-time-spectral-preservation-0.3.0"
P03_SCHEMA_VERSION = "pads-p0.3.0"
P03_IMPLEMENTATION_VERSION = "pads-p03-spectral-preservation-1.0.0"
P03_ARTIFACT_KIND = "TREMORA_PADS_P03_SPECTRAL_RELEASE_AUDIT"
SPECTRAL_CONTRACT_VERSION = "tremora-pads-nudft-tremor-band-0.3.0"

GATE_PASS = "PASS_PADS_SOURCE_TIME_SPECTRAL_PRESERVATION"
GATE_NO_GO = "NO_GO_PADS_SPECTRAL_PRESERVATION"
BLOCKED_DEPENDENCY = "BLOCKED_P02_DEPENDENCY_UNAVAILABLE"

SUCCESS_MARKER = "_PADS_P03_SPECTRAL_SUCCESS"
GENERIC_SUCCESS_MARKER = "_SUCCESS"

# --- frequency grid -------------------------------------------------------

FREQUENCY_MIN_HZ = 3.0
FREQUENCY_MAX_HZ = 12.0
FREQUENCY_STEP_HZ = 0.25
FREQUENCY_BIN_COUNT = 37
WINDOW_DURATION_S = 4.0

#: 1 / 4 s.  The grid matches the window's Rayleigh resolution exactly; no
#: zero-padding or oversampling, which would imply physical resolution this
#: milestone has not earned.
RAYLEIGH_RESOLUTION_HZ = 1.0 / WINDOW_DURATION_S

# --- signal policy --------------------------------------------------------

SENSOR_FAMILY_ACCELEROMETER = "ACCELEROMETER"
SENSOR_FAMILY_GYROSCOPE = "GYROSCOPE"

#: Gyroscope is the primary tremor-frequency workload; accelerometer is
#: corroborative.  Absolute power is never compared between the two: their
#: units differ.
SENSOR_FAMILIES: tuple[str, ...] = (
    SENSOR_FAMILY_GYROSCOPE,
    SENSOR_FAMILY_ACCELEROMETER,
)

FAMILY_AXES: dict[str, tuple[str, str, str]] = {
    SENSOR_FAMILY_ACCELEROMETER: (
        "accelerometer_x", "accelerometer_y", "accelerometer_z",
    ),
    SENSOR_FAMILY_GYROSCOPE: (
        "gyroscope_x", "gyroscope_y", "gyroscope_z",
    ),
}

#: Vector magnitude is never the primary spectral input: it can suppress or
#: distort oscillatory components and can double an apparent frequency.
VECTOR_MAGNITUDE_ALLOWED = False

DETREND_POLICY = "LINEAR_IN_SOURCE_TIME"
WINDOW_FUNCTION = "CONTINUOUS_TIME_HANN"
TRANSFORM = "NONUNIFORM_DISCRETE_FOURIER_TRANSFORM"
NUMERIC_DTYPE = "float64"

# --- selection policy -----------------------------------------------------

WORKLOAD_SELECTION_VERSION = "pads-p03-canonical-midpoint-window-1.0.0"
AUDIT_SELECTION_VERSION = "pads-p03-stratified-source-replay-audit-1.0.0"
AUDIT_SELECTION_SEED = 20260829

#: Up to this many windows per populated stratum.  On the materialized corpus
#: the five-part key populates 862 strata, so this cap selects 6,077 windows.
AUDIT_WINDOWS_PER_STRATUM = 10

AUDIT_STRATUM_FIELDS: tuple[str, ...] = (
    "task_name",
    "device_location",
    "outer_fold",
    "sample_count_class",
    "gap_adjacent_status",
)

GAP_ADJACENT = "GAP_ADJACENT"
INTERIOR = "INTERIOR"

#: Segment break reasons that represent a real discontinuity rather than the
#: stream simply beginning or ending.
REAL_BREAK_REASONS: frozenset[str] = frozenset({
    "TIME_GAP",
    "NONPOSITIVE_DELTA",
    "ORDINAL_DISCONTINUITY",
    "INVALID_SAMPLE",
})

# --- eligibility ----------------------------------------------------------

SPECTRALLY_ELIGIBLE = "SPECTRALLY_ELIGIBLE"
INELIGIBLE_WINDOW_NOT_VALID = "INELIGIBLE_WINDOW_NOT_VALID"
INELIGIBLE_TIME_NOT_INCREASING = "INELIGIBLE_TIME_NOT_INCREASING"
INELIGIBLE_NO_CADENCE = "INELIGIBLE_NO_CADENCE"
INELIGIBLE_COVERAGE = "INELIGIBLE_COVERAGE"
INELIGIBLE_SEGMENT_CROSSING = "INELIGIBLE_SEGMENT_CROSSING"
INELIGIBLE_ABOVE_NYQUIST = "INELIGIBLE_ABOVE_NYQUIST"
INELIGIBLE_UNUSABLE_CHANNEL = "INELIGIBLE_UNUSABLE_CHANNEL"

#: The coverage floor P0.2.1 already applied to admit a window at all.
MINIMUM_COVERAGE_FRACTION = 0.5

# --- claim boundary -------------------------------------------------------

WITHHELD_P03_ARTIFACTS: dict[str, int] = {
    "anti_alias_filter_outputs": 0,
    "bilateral_fusion_tables": 0,
    "classification_tables": 0,
    "comparative_benchmark_tables": 0,
    "derived_rate_tables": 0,
    "generic_success_markers": 0,
    "resampled_signal_tables": 0,
    "tremor_detection_tables": 0,
    "video_association_tables": 0,
}

#: P0.3 computes spectra, so the P0.2 analysis screen cannot be reused as-is.
#: What stays forbidden is the *next* milestones' vocabulary.
FORBIDDEN_P03_SUBSTRINGS: tuple[str, ...] = (
    "anti_alias",
    "classification",
    "decimat",
    "diagnosis",
    "downsample",
    "resample",
    "severity",
)


class PadsP03ContractError(ValueError):
    """Raised when a P0.3 artifact would exceed its milestone."""


def assert_no_derived_rate_claim(names: object) -> None:
    """Refuse a name that implies P0.4 resampling or a clinical claim."""

    offending = sorted({
        str(name)
        for name in names  # type: ignore[union-attr]
        if any(
            token in str(name).casefold()
            for token in FORBIDDEN_P03_SUBSTRINGS
        )
    })
    if offending:
        raise PadsP03ContractError(
            f"P0.3 preserves spectra only; {offending!r} implies a "
            "resampling, ablation or clinical milestone"
        )


def assert_p03_names(names: object) -> None:
    """Apply the inherited video screen and the P0.3 milestone screen."""

    materialized = list(names)  # type: ignore[call-overload]
    # The P0.2 screen forbids spectral vocabulary, which P0.3 legitimately
    # uses; only its video half applies here.
    from ..authority import assert_no_paired_claim

    assert_no_paired_claim(materialized)
    assert_no_derived_rate_claim(materialized)


def authority_block() -> dict[str, Any]:
    """The authority every P0.3 record carries."""

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
        "spectral_input": "RAW_AXES_PER_SENSOR_FAMILY",
        "vector_magnitude_primary_signal": VECTOR_MAGNITUDE_ALLOWED,
        "transform": TRANSFORM,
        "detrend": DETREND_POLICY,
        "window_function": WINDOW_FUNCTION,
        "numeric_dtype": NUMERIC_DTYPE,
        "contract_version": P03_CONTRACT_VERSION,
        "spectral_contract_version": SPECTRAL_CONTRACT_VERSION,
    }


__all__ = [
    "AUDIT_SELECTION_SEED",
    "AUDIT_SELECTION_VERSION",
    "AUDIT_STRATUM_FIELDS",
    "AUDIT_WINDOWS_PER_STRATUM",
    "BLOCKED_DEPENDENCY",
    "DETREND_POLICY",
    "FAMILY_AXES",
    "FORBIDDEN_P03_SUBSTRINGS",
    "FREQUENCY_BIN_COUNT",
    "FREQUENCY_MAX_HZ",
    "FREQUENCY_MIN_HZ",
    "FREQUENCY_STEP_HZ",
    "GAP_ADJACENT",
    "GATE_NO_GO",
    "GATE_PASS",
    "GENERIC_SUCCESS_MARKER",
    "INELIGIBLE_ABOVE_NYQUIST",
    "INELIGIBLE_COVERAGE",
    "INELIGIBLE_NO_CADENCE",
    "INELIGIBLE_SEGMENT_CROSSING",
    "INELIGIBLE_TIME_NOT_INCREASING",
    "INELIGIBLE_UNUSABLE_CHANNEL",
    "INELIGIBLE_WINDOW_NOT_VALID",
    "INTERIOR",
    "MINIMUM_COVERAGE_FRACTION",
    "NUMERIC_DTYPE",
    "P03_ARTIFACT_KIND",
    "P03_CONTRACT_VERSION",
    "P03_IMPLEMENTATION_VERSION",
    "P03_SCHEMA_VERSION",
    "RAYLEIGH_RESOLUTION_HZ",
    "REAL_BREAK_REASONS",
    "SENSOR_FAMILIES",
    "SENSOR_FAMILY_ACCELEROMETER",
    "SENSOR_FAMILY_GYROSCOPE",
    "SPECTRALLY_ELIGIBLE",
    "SPECTRAL_CONTRACT_VERSION",
    "SUCCESS_MARKER",
    "TRANSFORM",
    "VECTOR_MAGNITUDE_ALLOWED",
    "WINDOW_DURATION_S",
    "WINDOW_FUNCTION",
    "WITHHELD_P03_ARTIFACTS",
    "WORKLOAD_SELECTION_VERSION",
    "PadsP03ContractError",
    "assert_no_derived_rate_claim",
    "assert_p03_names",
    "authority_block",
]
