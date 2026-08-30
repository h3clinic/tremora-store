"""The frozen PADS-P0.5 comparative storage and retrieval benchmark contract.

Four representations answer the same queries over the same frozen source, and
the milestone asks whether authority-preserving storage costs anything to
retrieve.  It does not ask whether TremoraStore wins.  No gate condition
mentions who was fastest or smallest, and a run in which a baseline beats M1
is a completed benchmark, not a failure -- which is why the gate is named for
the experiment rather than for its outcome.

Two fairness rules do real work here and are stated as constants rather than
as prose.  The HDF5 baseline is given genuine per-stream and per-window offset
indexes, because a comparison against a representation forced to scan whole
files measures the crippling, not the architecture.  The duplicated-window
baseline physically duplicates the samples its overlapping windows share,
because a duplication-overhead claim against a representation that quietly
deduplicated would be measuring nothing at all.
"""

from __future__ import annotations

from typing import Any

# --- identity -------------------------------------------------------------

P05_CONTRACT_VERSION = "tremora-pads-comparative-systems-benchmark-0.5.0"
P05_SCHEMA_VERSION = "pads-p0.5.0"
P05_IMPLEMENTATION_VERSION = "pads-p05-comparative-benchmark-1.0.0"
P05_ARTIFACT_KIND = "TREMORA_PADS_P05_COMPARATIVE_BENCHMARK_RELEASE_AUDIT"

GATE_PASS = "PASS_PADS_COMPARATIVE_SYSTEMS_BENCHMARK"
GATE_NO_GO = "NO_GO_PADS_COMPARATIVE_SYSTEMS_BENCHMARK"
BLOCKED_DEPENDENCY = "BLOCKED_P04_DEPENDENCY_UNAVAILABLE"

SUCCESS_MARKER = "_PADS_P05_BENCHMARK_SUCCESS"
GENERIC_SUCCESS_MARKER = "_SUCCESS"

DATASET_ID = "PADS"
REFERENCE_MILESTONE = "PADS_P0_2_1"

# --- the four representations ---------------------------------------------

B0 = "B0_SOURCE_TEXT_RUNTIME_PARSE"
B1 = "B1_WINDOW_MATERIALIZED_DUPLICATED"
B2 = "B2_HDF5_RANGE_INDEXED"
M1 = "M1_TREMORA_PARQUET_INDEXED"

REPRESENTATIONS: tuple[str, ...] = (B0, B1, B2, M1)

REPRESENTATION_LABELS: dict[str, str] = {
    B0: "Original TXT + runtime parsing",
    B1: "Window-materialized, duplicated",
    B2: "HDF5 columnar + range index",
    M1: "TremoraStore Parquet + indexes",
}

#: M1 is the system under test; the other three are the comparison. Naming it
#: here is bookkeeping, not a prediction: no gate condition reads this.
SYSTEM_UNDER_TEST = M1

# --- what every representation is given -----------------------------------

#: Frozen from the P0.2.1 release audit.  A representation that holds a
#: different number of anything is not answering the same question.
SOURCE_SAMPLES = 13_447_168
SOURCE_STREAMS = 10_318
SOURCE_ASSESSMENTS = 5_159
SOURCE_PARTICIPANTS = 469
WINDOWS = 50_676

#: The P0.2.1 windows overlap by half, so their sample instances exceed the
#: samples they cover, and they cover neither every sample nor every stream:
#: 12,150,522 distinct samples over 9,960 streams.  A representation that
#: stored only window content could not replay the other 358 streams at all,
#: which is why B1 holds the full sample set *and* its duplicated windows.
WINDOW_SAMPLE_INSTANCES = 20_225_340
WINDOW_COVERED_SAMPLES = 12_150_522
WINDOW_COVERED_STREAMS = 9_960

# --- fairness -------------------------------------------------------------

#: Every representation is built from the same source manifest, answers the
#: same frozen query ids, and returns the same source-time semantics.
FAIRNESS_RULES: tuple[str, ...] = (
    "IDENTICAL_SOURCE_MANIFEST",
    "IDENTICAL_QUERY_IDENTIFIERS",
    "IDENTICAL_SOURCE_TIME_SEMANTICS",
    "EQUIVALENT_PARTICIPANT_TASK_STREAM_METADATA",
    "EQUIVALENT_PRECOMPUTED_INDEXES",
    "MATCHED_COMPRESSION_CODEC_AND_LEVEL",
    "NO_REPRESENTATION_PRIVATE_RESULT_CACHE",
    "FRESH_PROCESS_PER_COLD_MEASUREMENT",
)

#: Matched across every representation that compresses at all, so a storage
#: difference is a layout difference and not a codec difference.  B0 is the
#: release's own uncompressed text and is reported as such.
COMPRESSION_CODEC = "zstd"
COMPRESSION_LEVEL = 9
COMPRESSION_POLICY: dict[str, dict[str, Any]] = {
    B0: {"codec": "none", "level": None, "note": "the published release text"},
    B1: {"codec": COMPRESSION_CODEC, "level": COMPRESSION_LEVEL},
    B2: {"codec": COMPRESSION_CODEC, "level": COMPRESSION_LEVEL},
    M1: {"codec": COMPRESSION_CODEC, "level": COMPRESSION_LEVEL},
}

#: HDF5 gets real indexes.  Both are published so the claim that it was
#: treated fairly can be checked rather than taken on trust.
HDF5_REQUIRED_INDEXES: tuple[str, ...] = (
    "stream_offset_index",
    "window_offset_index",
)
HDF5_CHUNK_ROWS = 8_192

#: B1 must physically duplicate what its overlapping windows share.
B1_DUPLICATES_OVERLAPPING_SAMPLES = True
B1_EXPECTED_STORED_INSTANCES = SOURCE_SAMPLES + WINDOW_SAMPLE_INSTANCES

#: M1's window index references samples; it never copies them.
M1_WINDOW_INDEX_COPIES_SAMPLES = False

# --- the query workload ---------------------------------------------------

Q1 = "Q1_SINGLE_STREAM_REPLAY"
Q2 = "Q2_SINGLE_WINDOW"
Q3 = "Q3_PARTICIPANT_TASK_RETRIEVAL"
Q4 = "Q4_BATCH_WINDOWS"

QUERY_CLASSES: tuple[str, ...] = (Q1, Q2, Q3, Q4)

#: Q2 is the principal latency workload: every P0.2.1 window, once per round.
QUERY_COUNTS: dict[str, int] = {
    Q1: SOURCE_STREAMS,
    Q2: WINDOWS,
    Q3: SOURCE_ASSESSMENTS,
}

BATCH_SIZES: tuple[int, ...] = (8, 32, 64, 256)
BATCHES_PER_SIZE = 64
PRIMARY_BATCH_SIZE = 64

#: Protocol-paired, never sample-synchronized.  Q3 returns both wrists'
#: metadata and content; it does not claim they share a clock.
Q3_PAIRING = "PROTOCOL_PAIRED_NOT_SAMPLE_SYNCHRONIZED"

#: The workload is generated and hashed before any timing runs, from keyed
#: SHA-256 rather than a random number generator, so the same order can be
#: rebuilt anywhere without carrying a seed's implementation with it.
WORKLOAD_SELECTION = "KEYED_SHA256_DETERMINISTIC"
WORKLOAD_FROZEN_BEFORE_TIMING = True

# --- timing methodology ---------------------------------------------------

WARMUP_ROUNDS = 1
MEASURED_ROUNDS = 10
TOTAL_ROUNDS = WARMUP_ROUNDS + MEASURED_ROUNDS

#: Q2 is the principal latency workload and gets the full ten measured
#: rounds.  Q1 and Q3 each re-read the entire corpus once per representation
#: per round -- they are whole-stream and both-wrist replays -- and are
#: secondary outcomes, so they get four.  This is published rather than
#: applied quietly: a reader comparing a Q1 interval against a Q2 interval is
#: entitled to know one rests on fewer rounds.
MEASURED_ROUNDS_BY_QUERY_CLASS: dict[str, int] = {
    "Q1_SINGLE_STREAM_REPLAY": 4,
    "Q2_SINGLE_WINDOW": MEASURED_ROUNDS,
    "Q3_PARTICIPANT_TASK_RETRIEVAL": 4,
    "Q4_BATCH_WINDOWS": 4,
}

#: Each round shuffles the frozen query order independently, every
#: representation sees that same order within the round, and the order the
#: representations run in rotates between rounds.  That is what keeps cache,
#: thermal and ordering drift from being read as an architectural result.
PER_ROUND_SHUFFLE = True
REPRESENTATION_ORDER_ROTATES = True

COLD = "COLD"
WARM = "WARM"
MEASUREMENT_PHASES: tuple[str, ...] = (COLD, WARM)

#: Cold means a new process against an unopened representation.  Page caches
#: are not dropped: doing that identically and safely across filesystems is
#: not something this environment can promise, and fresh processes with
#: repeated randomized rounds reproduce more honestly than a cache-drop that
#: might silently no-op.
COLD_METHOD = "FRESH_PROCESS_UNOPENED_REPRESENTATION"
PAGE_CACHE_DROPPED = False

LATENCY_PERCENTILES: tuple[int, ...] = (50, 95, 99)

#: Speed ratios bootstrap over query identifiers, never over repeated timings
#: of the same query, so ten rounds of one query cannot pose as ten
#: independent workloads.
BOOTSTRAP_UNIT = "QUERY_ID"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95

# --- published metrics ----------------------------------------------------

STORAGE_METRICS: tuple[str, ...] = (
    "source_payload_bytes",
    "physical_storage_bytes",
    "metadata_bytes",
    "index_bytes",
    "unique_samples",
    "stored_sample_instances",
    "duplicate_sample_instances",
    "duplication_factor",
    "bytes_per_unique_sample",
    "bytes_per_stream",
    "bytes_per_window",
    "compression_ratio_vs_original_source",
    "file_count",
)

RETRIEVAL_METRICS: tuple[str, ...] = (
    "representation",
    "query_class",
    "query_id",
    "round_id",
    "latency_ns",
    "cpu_time_ns",
    "rows_returned",
    "bytes_returned",
    "peak_rss_delta",
    "content_sha256",
    "status",
)

PRIMARY_OUTCOMES: tuple[str, ...] = (
    "q2_window_latency_p50",
    "q2_window_latency_p95",
    "q2_window_latency_p99",
    "q4_batch_64_throughput",
    "physical_storage_bytes",
    "duplication_factor",
)

SECONDARY_OUTCOMES: tuple[str, ...] = (
    "initialization_latency",
    "q1_sequential_stream_throughput",
    "peak_rss",
    "file_count",
    "metadata_and_index_overhead",
)

# --- content equivalence --------------------------------------------------

#: The comparison is over what a query returns, not over the bytes a container
#: happens to use to hold it.  Two representations agree when these five
#: things agree, and nothing else is part of the hash.
CONTENT_HASH_FIELDS: tuple[str, ...] = (
    "stream_id",
    "source_row_ordinal",
    "source_time_token",
    "source_time_ps",
    "sensor_values",
)
CONTENT_HASH_BASIS = "CANONICAL_ROW_IDENTITY_NOT_CONTAINER_BYTES"
SENSOR_VALUE_COUNT = 6

#: A faster representation that returns different rows is not a benchmark
#: result; it is an invalid baseline, and these must all be zero.
REQUIRED_ZERO_COUNTS: tuple[str, ...] = (
    "content_mismatches",
    "row_count_mismatches",
    "time_mismatches",
    "sensor_value_mismatches",
    "failed_queries",
)

# --- derived-rate stores stay out of the primary comparison ---------------

#: P0.4's derived stores are reported in one secondary table.  Folding them
#: into the primary comparison would silently change the question from how
#: efficiently one authoritative representation is stored and retrieved to
#: how efficiently a family of derived signals is stored, so their samples
#: never enter the duplication accounting.
DERIVED_STORES_IN_PRIMARY_COMPARISON = False
DERIVED_STORE_RATES_HZ: tuple[int, ...] = (100, 50, 30, 25)

# --- claim boundary -------------------------------------------------------

#: P0.5 legitimately uses storage-benchmark vocabulary, which every earlier
#: PADS milestone forbade.  What stays forbidden is the clinical and
#: multimodal vocabulary, and any name that would claim the benchmark was won.
FORBIDDEN_P05_SUBSTRINGS: tuple[str, ...] = (
    "classification",
    "diagnosis",
    "outperform",
    "severity",
    "superior",
    "video",
    "wins",
)

WITHHELD_P05_ARTIFACTS: dict[str, int] = {
    "classification_tables": 0,
    "derived_rate_signal_tables": 0,
    "diagnosis_tables": 0,
    "generic_success_markers": 0,
    "new_signal_processing_outputs": 0,
    "severity_tables": 0,
    "superiority_claims": 0,
    "tremor_detection_tables": 0,
    "video_association_tables": 0,
}

#: P0.5 adds no signal processing.  It resamples nothing, filters nothing and
#: transforms nothing; it moves bytes and compares what came back.
NEW_SIGNAL_PROCESSING = False


class PadsP05ContractError(ValueError):
    """Raised when a P0.5 artifact would exceed its milestone."""


def assert_no_clinical_or_superiority_claim(names: object) -> None:
    """Refuse a name that implies a clinical result or a declared winner."""

    offending = sorted({
        str(name)
        for name in names  # type: ignore[union-attr]
        if any(
            token in str(name).casefold()
            for token in FORBIDDEN_P05_SUBSTRINGS
        )
    })
    if offending:
        raise PadsP05ContractError(
            f"P0.5 compares representations only; {offending!r} implies a "
            "clinical result, a later milestone, or a winner"
        )


def authority_block() -> dict[str, Any]:
    """What P0.5 claims, and what it refuses to claim."""

    return {
        "dataset_id": DATASET_ID,
        "contract_version": P05_CONTRACT_VERSION,
        "reference_milestone": REFERENCE_MILESTONE,
        "representations": list(REPRESENTATIONS),
        "system_under_test": SYSTEM_UNDER_TEST,
        "query_classes": list(QUERY_CLASSES),
        "compression_policy": {
            name: dict(policy)
            for name, policy in sorted(COMPRESSION_POLICY.items())
        },
        "fairness_rules": list(FAIRNESS_RULES),
        "content_hash_basis": CONTENT_HASH_BASIS,
        "content_hash_fields": list(CONTENT_HASH_FIELDS),
        "measured_rounds": MEASURED_ROUNDS,
        "warmup_rounds_discarded": WARMUP_ROUNDS,
        "page_cache_dropped": PAGE_CACHE_DROPPED,
        "cold_method": COLD_METHOD,
        "bootstrap_unit": BOOTSTRAP_UNIT,
        "derived_stores_in_primary_comparison": (
            DERIVED_STORES_IN_PRIMARY_COMPARISON
        ),
        "new_signal_processing": NEW_SIGNAL_PROCESSING,
        "outcome_is_not_gated": (
            "no condition requires M1 to be fastest or smallest"
        ),
    }


__all__ = [
    "B0",
    "B1",
    "B1_DUPLICATES_OVERLAPPING_SAMPLES",
    "B1_EXPECTED_STORED_INSTANCES",
    "B2",
    "BATCHES_PER_SIZE",
    "BATCH_SIZES",
    "BLOCKED_DEPENDENCY",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_UNIT",
    "COLD",
    "COLD_METHOD",
    "COMPRESSION_CODEC",
    "COMPRESSION_LEVEL",
    "COMPRESSION_POLICY",
    "CONTENT_HASH_BASIS",
    "CONTENT_HASH_FIELDS",
    "DATASET_ID",
    "DERIVED_STORES_IN_PRIMARY_COMPARISON",
    "DERIVED_STORE_RATES_HZ",
    "FAIRNESS_RULES",
    "FORBIDDEN_P05_SUBSTRINGS",
    "GATE_NO_GO",
    "GATE_PASS",
    "GENERIC_SUCCESS_MARKER",
    "HDF5_CHUNK_ROWS",
    "HDF5_REQUIRED_INDEXES",
    "LATENCY_PERCENTILES",
    "M1",
    "M1_WINDOW_INDEX_COPIES_SAMPLES",
    "MEASURED_ROUNDS",
    "MEASURED_ROUNDS_BY_QUERY_CLASS",
    "MEASUREMENT_PHASES",
    "NEW_SIGNAL_PROCESSING",
    "P05_ARTIFACT_KIND",
    "P05_CONTRACT_VERSION",
    "P05_IMPLEMENTATION_VERSION",
    "P05_SCHEMA_VERSION",
    "PAGE_CACHE_DROPPED",
    "PER_ROUND_SHUFFLE",
    "PRIMARY_BATCH_SIZE",
    "PRIMARY_OUTCOMES",
    "Q1",
    "Q2",
    "Q3",
    "Q3_PAIRING",
    "Q4",
    "QUERY_CLASSES",
    "QUERY_COUNTS",
    "REFERENCE_MILESTONE",
    "REPRESENTATIONS",
    "REPRESENTATION_LABELS",
    "REPRESENTATION_ORDER_ROTATES",
    "REQUIRED_ZERO_COUNTS",
    "RETRIEVAL_METRICS",
    "SECONDARY_OUTCOMES",
    "SENSOR_VALUE_COUNT",
    "SOURCE_ASSESSMENTS",
    "SOURCE_PARTICIPANTS",
    "SOURCE_SAMPLES",
    "SOURCE_STREAMS",
    "STORAGE_METRICS",
    "SUCCESS_MARKER",
    "SYSTEM_UNDER_TEST",
    "TOTAL_ROUNDS",
    "WARM",
    "WARMUP_ROUNDS",
    "WINDOWS",
    "WINDOW_COVERED_SAMPLES",
    "WINDOW_COVERED_STREAMS",
    "WINDOW_SAMPLE_INSTANCES",
    "WITHHELD_P05_ARTIFACTS",
    "WORKLOAD_FROZEN_BEFORE_TIMING",
    "WORKLOAD_SELECTION",
    "PadsP05ContractError",
    "assert_no_clinical_or_superiority_claim",
    "authority_block",
]
