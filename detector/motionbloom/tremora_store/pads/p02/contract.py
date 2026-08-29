"""Frozen PADS-P0.2 storage, indexing and replay contract.

P0.2 materializes a compact deterministic representation of the PADS release:
participant, assessment and stream indexes, an exact source-time sample store,
gap-aware contiguous segments, four-second window indexes, bilateral task
co-indexing and participant-disjoint folds.

It calculates no spectrum, no tremor frequency, no band power, no resampled
signal, no anti-aliasing output, no classification, no video association and no
comparative benchmark result.  Those are separate milestones and
:data:`WITHHELD_P02_ARTIFACTS` is published so a reader can see the boundary
held rather than infer it from an absence.

Two facts about the published release drive the numeric policy here.

*Time is exact only in picoseconds.*  Every ``Time`` token in the release
carries exactly ten decimal places, so its resolution is 1e-10 s and a
nanosecond integer cannot represent it without rounding.  The store therefore
keeps source time as an int64 picosecond count, which is exact for ten
decimals, alongside the preserved token.  Rounding to nanoseconds would have
been a silent loss of a digit the source actually wrote.

*Sensor values reconstruct exactly through a declared format.*  Every sampled
value round-trips through ``{:.10f}``, so the store keeps float64 and rebuilds
the source token on replay -- but it verifies that round-trip for every value
as it materializes, and fails closed on the first value that does not.
"""

from __future__ import annotations

from typing import Any

from ..authority import (
    PADS_DATASET_ID,
    RELATIVE_TIME_BASIS,
    VIDEO_PAIRING,
    assert_no_paired_claim,
)

P02_CONTRACT_VERSION = "tremora-pads-index-and-window-authority-0.2.1"
P02_SCHEMA_VERSION = "pads-p0.2.0"
P02_IMPLEMENTATION_VERSION = "pads-p02-index-materialization-1.1.0"
P02_ARTIFACT_KIND = "TREMORA_PADS_P02_INDEX_RELEASE_AUDIT"

GATE_PASS = "PASS_PADS_INDEX_AND_WINDOW_AUTHORITY"
GATE_NO_GO = "NO_GO_PADS_INDEX_AND_WINDOW_MATERIALIZATION"
BLOCKED_DEPENDENCY = "BLOCKED_P01_DEPENDENCY_UNAVAILABLE"

#: P0.2 uses its own marker.  A generic ``_SUCCESS`` would let a reader mistake
#: an index materialization for a synchronization result.
SUCCESS_MARKER = "_PADS_P02_INDEX_SUCCESS"
GENERIC_SUCCESS_MARKER = "_SUCCESS"

# --- timing boundary ------------------------------------------------------

TIMING_AUTHORITY = "SOURCE_RELATIVE_UNIMODAL_CLOCK"
TIME_BASIS = RELATIVE_TIME_BASIS
HARDWARE_SYNC_CLAIM = False

#: The release establishes that two wrist streams belong to the same
#: participant and task.  It does not establish a common hardware clock precise
#: enough for sample-to-sample fusion, so P0.2 may retrieve both wrists for one
#: task and may never claim that left sample 400 is simultaneous with right
#: sample 400.
BILATERAL_PAIRING_AUTHORITY = "SOURCE_PROTOCOL_PAIR"
CROSS_WRIST_CLOCK_ALIGNMENT = "UNRESOLVED"
SAMPLE_LEVEL_BILATERAL_FUSION_ALLOWED = False
BILATERAL_PAIR_STATUS = "PROTOCOL_COINDEXED"

# --- exact numeric policy -------------------------------------------------

#: Ten decimal places in the source means 1e-10 s; picoseconds represent that
#: exactly, nanoseconds do not.
TIME_SCALE_DECIMALS = 12
PICOSECONDS_PER_SECOND = 10**TIME_SCALE_DECIMALS
SOURCE_TIME_DECIMALS = 10

#: The format the release wrote its values in.  Replay rebuilds source tokens
#: through it, and materialization verifies every value against its own token.
SENSOR_VALUE_FORMAT = "{:.10f}"

# --- segment and window policy -------------------------------------------

#: A segment breaks at ``min(100 ms, 3 x dt_ref)``.  At the release's ~9.99 ms
#: median interval that is about 30 ms.  The multiplier is project policy,
#: frozen at 3; the absolute cap survives only so a pathologically low-rate
#: stream cannot bridge an arbitrarily large interval.
GAP_MULTIPLIER = 3
GAP_ABSOLUTE_CAP_PS = 100 * 10**9
MINIMUM_CADENCE_DELTAS = 8

WINDOW_DURATION_PS = 4 * PICOSECONDS_PER_SECOND
WINDOW_STRIDE_PS = 2 * PICOSECONDS_PER_SECOND

#: Window membership is decided by source time, never by counting a fixed
#: number of samples forward.  The real device clock jitters from 13.8 us to
#: 58.8 ms across the corpus, so ``first_sample + 400`` would silently mean a
#: different duration in every window.
WINDOW_MEMBERSHIP = "SOURCE_TIME_HALF_OPEN_INTERVAL"

# --- packing policy -------------------------------------------------------

#: One row group per stream, a fixed number of streams per file, streams in
#: stable ``stream_id`` order.  10,318 tiny text files are not a runtime
#: representation, and duplicating samples inside task or window records would
#: store the corpus several times over.
STREAMS_PER_PART = 256
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 9
PARQUET_VERSION = "2.6"
PARQUET_WRITER_POLICY_ID = (
    "tremora-pads-p02-parquet-zstd9-v2.6-one-row-group-per-stream-1.0.0"
)
SAMPLE_PART_PREFIX = "part-"

# --- fold policy ----------------------------------------------------------

N_OUTER_FOLDS = 5
SPLIT_SEED = 20260829
SPLIT_ALGORITHM_VERSION = "pads-participant-grouped-folds-1.0.0"

# --- claim boundary -------------------------------------------------------

WITHHELD_P02_ARTIFACTS: dict[str, int] = {
    "anti_alias_filter_outputs": 0,
    "band_power_tables": 0,
    "classification_tables": 0,
    "comparative_benchmark_tables": 0,
    "generic_success_markers": 0,
    "resampled_signal_tables": 0,
    "spectral_feature_tables": 0,
    "tremor_frequency_tables": 0,
    "video_association_tables": 0,
}

#: Substrings that would betray a milestone P0.2 has not opened.  Screened the
#: same way the video screen is: on substrings, because a deny-list of exact
#: names would refuse ``spectrum`` while admitting ``spectrum_ref``.
FORBIDDEN_ANALYSIS_SUBSTRINGS: tuple[str, ...] = (
    "anti_alias",
    "band_power",
    "bandpower",
    "downsample",
    "fft",
    "psd",
    "resample",
    "spectral",
    "spectrum",
    "tremor_frequency",
    "welch",
)


class PadsP02ContractError(ValueError):
    """Raised when a P0.2 artifact would exceed its milestone."""


def assert_no_analysis_claim(names: object) -> None:
    """Refuse a table or field name that implies P0.3, P0.4 or P0.5 work."""

    offending = sorted({
        str(name)
        for name in names  # type: ignore[union-attr]
        if any(
            token in str(name).casefold()
            for token in FORBIDDEN_ANALYSIS_SUBSTRINGS
        )
    })
    if offending:
        raise PadsP02ContractError(
            f"P0.2 materializes indexes only; {offending!r} implies a "
            "spectral, resampling or classification milestone"
        )


def assert_p02_names(names: object) -> None:
    """Apply both the inherited video screen and the P0.2 analysis screen."""

    materialized = list(names)  # type: ignore[call-overload]
    assert_no_paired_claim(materialized)
    assert_no_analysis_claim(materialized)


def authority_block() -> dict[str, Any]:
    """The authority every P0.2 record carries."""

    return {
        "dataset_id": PADS_DATASET_ID,
        "timing_authority": TIMING_AUTHORITY,
        "time_basis": TIME_BASIS,
        "video_pairing": VIDEO_PAIRING,
        "hardware_sync_claim": HARDWARE_SYNC_CLAIM,
        "bilateral_pairing_authority": BILATERAL_PAIRING_AUTHORITY,
        "cross_wrist_clock_alignment": CROSS_WRIST_CLOCK_ALIGNMENT,
        "sample_level_bilateral_fusion_allowed": (
            SAMPLE_LEVEL_BILATERAL_FUSION_ALLOWED
        ),
        "time_scale_decimals": TIME_SCALE_DECIMALS,
        "source_time_decimals": SOURCE_TIME_DECIMALS,
        "contract_version": P02_CONTRACT_VERSION,
    }


__all__ = [
    "BILATERAL_PAIRING_AUTHORITY",
    "BILATERAL_PAIR_STATUS",
    "BLOCKED_DEPENDENCY",
    "CROSS_WRIST_CLOCK_ALIGNMENT",
    "FORBIDDEN_ANALYSIS_SUBSTRINGS",
    "GAP_ABSOLUTE_CAP_PS",
    "GAP_MULTIPLIER",
    "GATE_NO_GO",
    "GATE_PASS",
    "GENERIC_SUCCESS_MARKER",
    "HARDWARE_SYNC_CLAIM",
    "MINIMUM_CADENCE_DELTAS",
    "N_OUTER_FOLDS",
    "P02_ARTIFACT_KIND",
    "P02_CONTRACT_VERSION",
    "P02_IMPLEMENTATION_VERSION",
    "P02_SCHEMA_VERSION",
    "PARQUET_COMPRESSION",
    "PARQUET_COMPRESSION_LEVEL",
    "PARQUET_VERSION",
    "PARQUET_WRITER_POLICY_ID",
    "PICOSECONDS_PER_SECOND",
    "SAMPLE_LEVEL_BILATERAL_FUSION_ALLOWED",
    "SAMPLE_PART_PREFIX",
    "SENSOR_VALUE_FORMAT",
    "SOURCE_TIME_DECIMALS",
    "SPLIT_ALGORITHM_VERSION",
    "SPLIT_SEED",
    "STREAMS_PER_PART",
    "SUCCESS_MARKER",
    "TIME_BASIS",
    "TIME_SCALE_DECIMALS",
    "TIMING_AUTHORITY",
    "WINDOW_DURATION_PS",
    "WINDOW_MEMBERSHIP",
    "WINDOW_STRIDE_PS",
    "WITHHELD_P02_ARTIFACTS",
    "PadsP02ContractError",
    "assert_no_analysis_claim",
    "assert_p02_names",
    "authority_block",
]
