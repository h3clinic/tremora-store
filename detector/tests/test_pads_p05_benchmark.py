"""The timing methodology, and the twenty-five gate conditions."""

from __future__ import annotations

from dataclasses import replace

import pytest
from motionbloom.tremora_store.pads.p05.benchmark import (
    QUERY_OK,
    WARMUP_ROUND_ID,
    Measurement,
    speed_ratios,
)
from motionbloom.tremora_store.pads.p05.contract import (
    B0,
    B1,
    B2,
    GATE_NO_GO,
    GATE_PASS,
    M1,
    MEASURED_ROUNDS_BY_QUERY_CLASS,
    Q1,
    Q2,
    Q3,
    Q4,
    QUERY_CLASSES,
    REPRESENTATIONS,
    WITHHELD_P05_ARTIFACTS,
)
from motionbloom.tremora_store.pads.p05.gate import (
    GATE_CONDITIONS,
    NEVER_GATE_CONDITIONS,
    REPRODUCTION_VERIFIED,
    PadsP05GateFacts,
    evaluate_gate,
    failing_conditions,
)
from motionbloom.tremora_store.pads.p05.sink import (
    MeasurementSink,
    batch_throughput_table,
    per_query_medians,
    summarize_table,
    warmup_rows_present,
)
from motionbloom.tremora_store.pads.p05.workload import representation_order


def _measurement(
    name: str, query_class: str, query_id: str, round_id: int, latency: int
) -> Measurement:
    return Measurement(
        representation=name, query_class=query_class, query_id=query_id,
        round_id=round_id, latency_ns=latency, cpu_time_ns=latency // 2,
        rows_returned=400, bytes_returned=4_000, peak_rss_delta=0,
        content_sha256="a" * 64, status=QUERY_OK,
    )


@pytest.fixture
def table(tmp_path):
    """A real written timing table; nothing is summarized from memory."""

    path = tmp_path / "retrieval.parquet"
    with MeasurementSink(path, flush_rows=256) as sink:
        for round_id in (WARMUP_ROUND_ID, 0, 1, 2):
            for index, name in enumerate(REPRESENTATIONS):
                for query in range(6):
                    sink.add(_measurement(
                        name, Q2, f"w{query}", round_id,
                        1_000_000 * (index + 1) + query * 1_000,
                    ).as_record())
                sink.add(_measurement(
                    name, Q4, "batch:64:0000", round_id,
                    10_000_000 * (index + 1),
                ).as_record())
    return path


# --- what is measured -----------------------------------------------------


def test_the_warmup_round_never_reaches_a_summary(table) -> None:
    # Dropped at the sink, so it cannot reach a summary by being forgotten
    # about somewhere downstream.
    assert warmup_rows_present(table) == 0
    summary = summarize_table(table)
    for classes in summary.values():
        # Three measured rounds of six Q2 queries.
        assert classes[Q2]["queries"] == 18


def test_a_failed_query_is_counted_at_the_sink(tmp_path) -> None:
    path = tmp_path / "f.parquet"
    with MeasurementSink(path) as sink:
        item = _measurement(M1, Q2, "bad", 0, 5)
        record = item.as_record()
        record["status"] = "QUERY_FAILED"
        sink.add(record)
        assert sink.counters.failed_queries == 1


def test_percentiles_are_reported_for_every_representation(table) -> None:
    summary = summarize_table(table)
    assert set(summary) == set(REPRESENTATIONS)
    for name, classes in summary.items():
        entry = classes[Q2]
        for key in (
            "p50_latency_ns", "p95_latency_ns", "p99_latency_ns",
            "mean_latency_ns", "rows_per_second", "queries_per_second",
        ):
            assert key in entry, (name, key)
        assert entry["p50_latency_ns"] <= entry["p95_latency_ns"]
        assert entry["p95_latency_ns"] <= entry["p99_latency_ns"]


def test_batch_throughput_counts_only_the_named_size(table) -> None:
    throughput = batch_throughput_table(table, size=64)
    assert set(throughput) == set(REPRESENTATIONS)
    for entry in throughput.values():
        assert entry["batches"] == 3
        assert entry["windows"] == 3 * 64
        assert entry["windows_per_second"] > 0
    assert batch_throughput_table(table, size=8) == {}


# --- the statistical unit -------------------------------------------------


def test_the_bootstrap_resamples_queries_not_repeated_timings(
    table,
) -> None:
    ratios = speed_ratios(
        per_query_medians(table), baseline=B0, system=M1,
        query_class=Q2, resamples=200,
    )
    assert ratios["bootstrap_unit"] == "QUERY_ID"
    # Six distinct queries, not eighteen observations of them.
    assert ratios["queries"] == 6
    assert ratios["confidence_low"] <= ratios["median_ratio"]
    assert ratios["median_ratio"] <= ratios["confidence_high"]


def test_a_slower_baseline_gives_a_ratio_above_one(table) -> None:
    # B0 is the fastest in the fixture, M1 the slowest, so B0 over M1 < 1.
    medians = per_query_medians(table)
    assert speed_ratios(
        medians, baseline=B0, system=M1, resamples=200
    )["median_ratio"] < 1.0
    assert speed_ratios(
        medians, baseline=M1, system=B0, resamples=200
    )["median_ratio"] > 1.0


def test_the_bootstrap_is_deterministic(table) -> None:
    medians = per_query_medians(table)
    first = speed_ratios(medians, baseline=B1, system=M1, resamples=200)
    again = speed_ratios(medians, baseline=B1, system=M1, resamples=200)
    assert first == again


def test_no_shared_query_gives_an_empty_ratio() -> None:
    result = speed_ratios({}, baseline=B0, system=M1, resamples=10)
    assert result["queries"] == 0
    assert result["median_ratio"] == 0.0


# --- the gate -------------------------------------------------------------


@pytest.fixture
def passing() -> PadsP05GateFacts:
    return PadsP05GateFacts(
        dependency_status="P02_DEPENDENCY_VERIFIED",
        p03_evidence_sha256="c" * 64, p04_evidence_sha256="d" * 64,
        pinned_p03_evidence_sha256="c" * 64,
        pinned_p04_evidence_sha256="d" * 64,
        source_manifest_sha256="m" * 64,
        manifest_by_representation={n: "m" * 64 for n in REPRESENTATIONS},
        query_classes_supported={n: 4 for n in REPRESENTATIONS},
        windows_reconciled=50_676, expected_windows=50_676,
        streams_reconciled=10_318, assessments_reconciled=5_159,
        per_representation_mismatches={n: 0 for n in REPRESENTATIONS},
        b1_stored_instances=33_672_508, b1_unique_samples=13_447_168,
        m1_stored_instances=13_447_168, m1_unique_samples=13_447_168,
        hdf5_indexes_present=(
            "stream_offset_index", "window_offset_index",
        ),
        hdf5_indexes_required=(
            "stream_offset_index", "window_offset_index",
        ),
        hdf5_window_index_entries=50_676, hdf5_chunked=True,
        compression_declared={
            n: {"codec": "zstd", "level": 9} for n in REPRESENTATIONS
        },
        workload_hash_before_timing="f" * 64,
        workload_hash_after_timing="f" * 64,
        warmup_rounds_discarded=1, warmup_records_in_summary=0,
        representation_orders=tuple(
            representation_order(index) for index in range(11)
        ),
        rounds_completed_by_class=dict(MEASURED_ROUNDS_BY_QUERY_CLASS),
        timing_executions_completed=2, timing_executions_required=2,
        storage_reconciled=True, storage_problems=(),
        latency_records=1_000, expected_latency_records=1_000,
        failed_queries=0, reproduction_status=REPRODUCTION_VERIFIED,
        signal_processing_declared=True, new_signal_processing_outputs=0,
        emitted_forbidden_artifacts=dict(WITHHELD_P05_ARTIFACTS),
    )


def test_the_contract_names_twenty_five_conditions() -> None:
    assert len(GATE_CONDITIONS) == 25
    assert len(set(GATE_CONDITIONS)) == 25


def test_winning_is_not_a_condition() -> None:
    # The research question is comparative performance under an
    # authority-preserving architecture, not manufacturing a victory.
    assert NEVER_GATE_CONDITIONS == (
        "M1_MUST_BE_FASTEST", "M1_MUST_BE_SMALLEST",
    )
    for absent in NEVER_GATE_CONDITIONS:
        assert absent not in GATE_CONDITIONS
    joined = " ".join(GATE_CONDITIONS).casefold()
    for token in ("fastest", "smallest", "outperform", "superior", "wins"):
        assert token not in joined


def test_a_complete_set_of_facts_passes(passing) -> None:
    result = evaluate_gate(passing)
    assert result.gate_status == GATE_PASS
    assert failing_conditions(result.as_record()) == ()
    assert result.as_record()["gate_conditions_satisfied"] == 25


def test_the_gate_passes_even_when_every_baseline_wins(passing) -> None:
    # The whole point: a run TremoraStore loses is still a valid benchmark.
    assert evaluate_gate(passing).satisfied


def test_nothing_passes_on_an_empty_record() -> None:
    result = evaluate_gate(PadsP05GateFacts())
    assert result.gate_status == GATE_NO_GO
    assert len(failing_conditions(result.as_record())) == 25


@pytest.mark.parametrize(
    ("field", "value", "condition"),
    (
        ("dependency_status", "P02_REPORT_ABSENT", "P02_1_DEPENDENCY_VERIFIED"),
        ("p03_evidence_sha256", "x" * 64, "P03_AND_P04_EVIDENCE_UNCHANGED"),
        ("p04_evidence_sha256", "x" * 64, "P03_AND_P04_EVIDENCE_UNCHANGED"),
        ("windows_reconciled", 50_675, "ALL_50676_WINDOWS_RECONCILED"),
        ("content_mismatches", 1,
         "ZERO_CROSS_REPRESENTATION_CONTENT_MISMATCHES"),
        ("time_mismatches", 1,
         "ZERO_CROSS_REPRESENTATION_CONTENT_MISMATCHES"),
        ("b1_stored_instances", 13_447_168,
         "B1_PHYSICALLY_DUPLICATES_OVERLAPPING_WINDOW_SAMPLES"),
        ("m1_stored_instances", 13_447_169,
         "M1_WINDOW_INDEX_DOES_NOT_DUPLICATE_SOURCE_SAMPLES"),
        ("hdf5_indexes_present", ("stream_offset_index",),
         "HDF5_HAS_FAIR_RANGE_INDEX"),
        ("hdf5_chunked", False, "HDF5_HAS_FAIR_RANGE_INDEX"),
        ("hdf5_window_index_entries", 5, "HDF5_HAS_FAIR_RANGE_INDEX"),
        ("workload_hash_after_timing", "e" * 64,
         "QUERY_WORKLOAD_FROZEN_BEFORE_TIMING"),
        ("warmup_rounds_discarded", 0, "WARMUP_EXCLUDED"),
        ("warmup_records_in_summary", 1, "WARMUP_EXCLUDED"),
        ("storage_reconciled", False, "STORAGE_COUNTS_RECONCILED"),
        ("latency_records", 999, "LATENCY_RECORD_COUNTS_RECONCILED"),
        ("failed_queries", 1, "NO_FAILED_BENCHMARK_QUERIES"),
        ("reproduction_status", "REPRODUCTION_NOT_ATTEMPTED",
         "INDEPENDENT_BENCHMARK_REPRODUCTION_VERIFIED"),
        ("timing_executions_completed", 1,
         "INDEPENDENT_BENCHMARK_REPRODUCTION_VERIFIED"),
        ("new_signal_processing_outputs", 1, "NO_NEW_SIGNAL_PROCESSING"),
        ("signal_processing_declared", False, "NO_NEW_SIGNAL_PROCESSING"),
    ),
)
def test_each_fact_closes_the_condition_it_belongs_to(
    passing, field: str, value: object, condition: str
) -> None:
    result = evaluate_gate(replace(passing, **{field: value}))
    assert result.gate_status == GATE_NO_GO
    assert condition in failing_conditions(result.as_record())


def test_a_representation_reading_a_different_source_closes_the_gate(
    passing,
) -> None:
    manifests = dict(passing.manifest_by_representation)
    manifests[B2] = "z" * 64
    result = evaluate_gate(
        replace(passing, manifest_by_representation=manifests)
    )
    assert "ALL_BASELINES_USE_IDENTICAL_SOURCE_MANIFEST" in (
        failing_conditions(result.as_record())
    )


def test_a_representation_missing_a_query_class_closes_the_gate(
    passing,
) -> None:
    supported = dict(passing.query_classes_supported)
    supported[B1] = 3
    result = evaluate_gate(
        replace(passing, query_classes_supported=supported)
    )
    failures = failing_conditions(result.as_record())
    assert "ALL_BASELINES_SUPPORT_EQUIVALENT_QUERY_SEMANTICS" in failures
    assert "B1_CONTENT_EQUIVALENT" in failures


@pytest.mark.parametrize(
    ("name", "condition"),
    (
        (B0, "B0_CONTENT_EQUIVALENT"),
        (B1, "B1_CONTENT_EQUIVALENT"),
        (B2, "B2_CONTENT_EQUIVALENT"),
        (M1, "M1_CONTENT_EQUIVALENT"),
    ),
)
def test_each_representation_has_its_own_equivalence_condition(
    passing, name: str, condition: str
) -> None:
    mismatches = dict(passing.per_representation_mismatches)
    mismatches[name] = 1
    result = evaluate_gate(
        replace(passing, per_representation_mismatches=mismatches)
    )
    assert condition in failing_conditions(result.as_record())


def test_an_incomplete_round_budget_closes_the_gate(passing) -> None:
    short = dict(MEASURED_ROUNDS_BY_QUERY_CLASS)
    short[Q2] = short[Q2] - 1
    result = evaluate_gate(
        replace(passing, rounds_completed_by_class=short)
    )
    assert "MEASURED_ROUNDS_COMPLETE" in failing_conditions(
        result.as_record()
    )


def test_an_unrotated_representation_order_closes_the_gate(
    passing,
) -> None:
    fixed = tuple(REPRESENTATIONS for _ in range(11))
    result = evaluate_gate(replace(passing, representation_orders=fixed))
    assert "REPRESENTATION_ORDER_ROTATED" in failing_conditions(
        result.as_record()
    )


@pytest.mark.parametrize(
    ("artifact", "condition"),
    (
        ("classification_tables", "NO_CLASSIFICATION"),
        ("diagnosis_tables", "NO_CLASSIFICATION"),
        ("severity_tables", "NO_CLASSIFICATION"),
        ("video_association_tables", "NO_VIDEO_ASSOCIATION"),
    ),
)
def test_a_later_milestone_artifact_closes_the_gate(
    passing, artifact: str, condition: str
) -> None:
    withheld = dict(WITHHELD_P05_ARTIFACTS)
    withheld[artifact] = 1
    result = evaluate_gate(
        replace(passing, emitted_forbidden_artifacts=withheld)
    )
    assert condition in failing_conditions(result.as_record())


def test_the_query_classes_are_the_four_that_were_frozen() -> None:
    assert QUERY_CLASSES == (Q1, Q2, Q3, Q4)
