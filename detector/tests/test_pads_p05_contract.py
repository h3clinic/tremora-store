"""The frozen P0.5 benchmark contract, workload and row identity."""

from __future__ import annotations

import pytest
from motionbloom.tremora_store.pads.p05.contract import (
    B0,
    B1,
    B1_EXPECTED_STORED_INSTANCES,
    B2,
    BATCH_SIZES,
    BATCHES_PER_SIZE,
    COMPRESSION_CODEC,
    COMPRESSION_LEVEL,
    COMPRESSION_POLICY,
    CONTENT_HASH_BASIS,
    DERIVED_STORES_IN_PRIMARY_COMPARISON,
    GATE_PASS,
    GENERIC_SUCCESS_MARKER,
    HDF5_REQUIRED_INDEXES,
    M1,
    M1_WINDOW_INDEX_COPIES_SAMPLES,
    MEASURED_ROUNDS,
    NEW_SIGNAL_PROCESSING,
    PAGE_CACHE_DROPPED,
    Q1,
    Q2,
    Q3,
    Q4,
    QUERY_CLASSES,
    QUERY_COUNTS,
    REPRESENTATIONS,
    SENSOR_VALUE_COUNT,
    SOURCE_ASSESSMENTS,
    SOURCE_SAMPLES,
    SOURCE_STREAMS,
    SUCCESS_MARKER,
    TOTAL_ROUNDS,
    WARMUP_ROUNDS,
    WINDOW_COVERED_SAMPLES,
    WINDOW_COVERED_STREAMS,
    WINDOW_SAMPLE_INSTANCES,
    WINDOWS,
    WITHHELD_P05_ARTIFACTS,
    PadsP05ContractError,
    assert_no_clinical_or_superiority_claim,
    authority_block,
)
from motionbloom.tremora_store.pads.p05.rows import (
    SENSOR_ORDER,
    RowIdentityError,
    compare,
    result_from_rows,
)
from motionbloom.tremora_store.pads.p05.workload import (
    build_workload,
    deterministic_order,
    representation_order,
    rotation_is_complete,
    round_order,
)

STREAMS = tuple(f"s{index:05d}" for index in range(SOURCE_STREAMS))
WINDOW_IDS = tuple(f"w{index:06d}" for index in range(WINDOWS))
ASSESSMENTS = tuple(f"a{index:05d}" for index in range(SOURCE_ASSESSMENTS))


@pytest.fixture(scope="module")
def workload():
    return build_workload(
        stream_ids=STREAMS,
        window_ids=WINDOW_IDS,
        assessment_ids=ASSESSMENTS,
    )


# --- the contract ---------------------------------------------------------


def test_the_gate_is_named_for_the_experiment_not_its_outcome() -> None:
    assert GATE_PASS == "PASS_PADS_COMPARATIVE_SYSTEMS_BENCHMARK"
    for token in ("outperform", "superior", "wins", "fastest", "smallest"):
        assert token not in GATE_PASS.casefold()


def test_the_four_representations_are_frozen() -> None:
    assert REPRESENTATIONS == (B0, B1, B2, M1)
    assert len(set(REPRESENTATIONS)) == 4


def test_compression_is_matched_across_everything_that_compresses() -> None:
    # A storage difference has to be a layout difference, not a codec one.
    for name in (B1, B2, M1):
        assert COMPRESSION_POLICY[name]["codec"] == COMPRESSION_CODEC
        assert COMPRESSION_POLICY[name]["level"] == COMPRESSION_LEVEL
    # B0 is the release's own text and is reported as uncompressed.
    assert COMPRESSION_POLICY[B0]["codec"] == "none"


def test_hdf5_is_promised_real_indexes() -> None:
    # A comparison against a representation forced to scan whole files
    # measures the crippling, not the architecture.
    assert HDF5_REQUIRED_INDEXES == (
        "stream_offset_index", "window_offset_index",
    )


def test_the_duplication_baseline_actually_duplicates() -> None:
    # Windows overlap by half, and cover neither every sample nor every
    # stream, so B1 must hold the full sample set as well as its copies.
    assert WINDOW_SAMPLE_INSTANCES > WINDOW_COVERED_SAMPLES
    assert WINDOW_COVERED_SAMPLES < SOURCE_SAMPLES
    assert WINDOW_COVERED_STREAMS < SOURCE_STREAMS
    assert B1_EXPECTED_STORED_INSTANCES == (
        SOURCE_SAMPLES + WINDOW_SAMPLE_INSTANCES
    )
    assert B1_EXPECTED_STORED_INSTANCES / SOURCE_SAMPLES > 2.0
    # M1's window index references samples rather than copying them.
    assert M1_WINDOW_INDEX_COPIES_SAMPLES is False


def test_the_milestone_adds_no_signal_processing() -> None:
    assert NEW_SIGNAL_PROCESSING is False
    assert DERIVED_STORES_IN_PRIMARY_COMPARISON is False


def test_the_timing_method_is_declared_including_what_it_does_not_do(
) -> None:
    assert WARMUP_ROUNDS == 1
    assert MEASURED_ROUNDS == 10
    assert TOTAL_ROUNDS == 11
    # Not dropping page caches is a published fact, not an omission.
    assert PAGE_CACHE_DROPPED is False


def test_the_content_hash_is_over_rows_not_container_bytes() -> None:
    assert CONTENT_HASH_BASIS == (
        "CANONICAL_ROW_IDENTITY_NOT_CONTAINER_BYTES"
    )
    assert SENSOR_VALUE_COUNT == len(SENSOR_ORDER) == 6


def test_the_marker_is_specific_to_this_milestone() -> None:
    assert SUCCESS_MARKER == "_PADS_P05_BENCHMARK_SUCCESS"
    assert SUCCESS_MARKER != GENERIC_SUCCESS_MARKER


def test_a_superiority_or_clinical_name_is_refused() -> None:
    assert_no_clinical_or_superiority_claim(
        ["q2_window_latency_p50", "duplication_factor", "physical_bytes"]
    )
    for name in (
        "m1_outperforms_baselines", "tremor_classification",
        "severity_table", "video_association", "superior_throughput",
    ):
        with pytest.raises(PadsP05ContractError):
            assert_no_clinical_or_superiority_claim([name])


def test_the_authority_block_says_the_outcome_is_not_gated() -> None:
    block = authority_block()
    assert block["system_under_test"] == M1
    assert block["outcome_is_not_gated"] == (
        "no condition requires M1 to be fastest or smallest"
    )
    assert block["page_cache_dropped"] is False
    assert block["new_signal_processing"] is False


def test_the_withheld_milestones_start_at_zero() -> None:
    assert set(WITHHELD_P05_ARTIFACTS.values()) == {0}
    for name in (
        "classification_tables", "video_association_tables",
        "superiority_claims", "new_signal_processing_outputs",
        "derived_rate_signal_tables", "generic_success_markers",
    ):
        assert WITHHELD_P05_ARTIFACTS[name] == 0


# --- the workload ---------------------------------------------------------


def test_every_query_class_is_populated(workload) -> None:
    counts = workload.counts()
    assert counts[Q1] == QUERY_COUNTS[Q1] == SOURCE_STREAMS
    assert counts[Q2] == QUERY_COUNTS[Q2] == WINDOWS
    assert counts[Q3] == QUERY_COUNTS[Q3] == SOURCE_ASSESSMENTS
    assert counts[Q4] == len(BATCH_SIZES) * BATCHES_PER_SIZE
    assert set(counts) == set(QUERY_CLASSES)


def test_the_workload_is_a_permutation_not_a_sample(workload) -> None:
    # Q2 is the principal latency workload and uses every window, once.
    assert sorted(workload.window_ids) == sorted(WINDOW_IDS)
    assert len(set(workload.window_ids)) == WINDOWS
    assert sorted(workload.stream_ids) == sorted(STREAMS)


def test_the_frozen_order_is_not_the_input_order(workload) -> None:
    assert workload.window_ids != WINDOW_IDS
    assert workload.stream_ids != STREAMS


def test_the_workload_hash_is_stable_and_content_sensitive(
    workload,
) -> None:
    again = build_workload(
        stream_ids=STREAMS, window_ids=WINDOW_IDS,
        assessment_ids=ASSESSMENTS,
    )
    assert workload.content_sha256() == again.content_sha256()
    different = build_workload(
        stream_ids=STREAMS, window_ids=WINDOW_IDS[:-1],
        assessment_ids=ASSESSMENTS,
    )
    assert workload.content_sha256() != different.content_sha256()


def test_batches_are_scattered_rather_than_contiguous(workload) -> None:
    sizes = [len(batch) for batch in workload.batches]
    for size in BATCH_SIZES:
        assert sizes.count(size) == BATCHES_PER_SIZE
    # Drawn from their own ordering, so a batch is not a contiguous run that
    # a range-indexed representation would answer unrealistically well.
    first = workload.batches[0]
    positions = [workload.window_ids.index(w) for w in first]
    assert max(positions) - min(positions) > len(first)


def test_every_round_reorders_independently(workload) -> None:
    orders = [
        round_order(workload.window_ids, query_class=Q2, round_id=index)
        for index in range(TOTAL_ROUNDS)
    ]
    assert len({tuple(order) for order in orders}) == TOTAL_ROUNDS
    for order in orders:
        assert sorted(order) == sorted(WINDOW_IDS)


def test_a_round_order_is_reproducible(workload) -> None:
    first = round_order(workload.window_ids, query_class=Q2, round_id=3)
    again = round_order(workload.window_ids, query_class=Q2, round_id=3)
    assert first == again


def test_the_representation_order_rotates_and_completes() -> None:
    assert representation_order(0) == (B0, B1, B2, M1)
    assert representation_order(1) == (B1, B2, M1, B0)
    assert representation_order(2) == (B2, M1, B0, B1)
    assert representation_order(3) == (M1, B0, B1, B2)
    for index in range(TOTAL_ROUNDS):
        assert sorted(representation_order(index)) == sorted(REPRESENTATIONS)
    # Across the rounds every representation leads at least once.
    assert rotation_is_complete()


def test_the_shuffle_needs_no_seeded_generator() -> None:
    # Keyed SHA-256, so the order rebuilds from the contract alone.
    once = deterministic_order(["b", "a", "c"], purpose="probe")
    assert once == deterministic_order(["c", "b", "a"], purpose="probe")
    assert once != deterministic_order(["a", "b", "c"], purpose="other")
    assert sorted(once) == ["a", "b", "c"]


# --- the row identity -----------------------------------------------------


def _row(token: str = "4.0082712173", value: float = -0.0160145294) -> dict:
    return {
        "stream_id": "001:CrossArms:LeftWrist",
        "source_row_ordinal": 7,
        "source_time_token": token,
        "source_time_ps": 4_008_271_217_300,
        "accelerometer_x": value, "accelerometer_y": 0.1,
        "accelerometer_z": 0.2, "gyroscope_x": 0.3,
        "gyroscope_y": 0.4, "gyroscope_z": 0.5,
    }


def test_identical_rows_agree_on_every_required_equality() -> None:
    left = result_from_rows("q", [_row()])
    right = result_from_rows("q", [_row()])
    assert all(compare(left, right).values())
    assert left.rows == 1


def test_a_rounded_time_token_is_caught_and_localized() -> None:
    # The release's ten-decimal token is compared as the string it wrote.
    verdict = compare(
        result_from_rows("q", [_row()]),
        result_from_rows("q", [_row(token="4.008271217")]),
    )
    assert verdict["content_match"] is False
    assert verdict["time_match"] is False
    # The separate digests say which half disagreed.
    assert verdict["sensor_value_match"] is True
    assert verdict["row_count_match"] is True


def test_a_perturbed_sensor_value_is_caught_and_localized() -> None:
    verdict = compare(
        result_from_rows("q", [_row()]),
        result_from_rows("q", [_row(value=-0.01601452941)]),
    )
    assert verdict["content_match"] is False
    assert verdict["sensor_value_match"] is False
    assert verdict["time_match"] is True


def test_a_missing_row_is_caught() -> None:
    verdict = compare(
        result_from_rows("q", [_row()]), result_from_rows("q", [])
    )
    assert verdict["row_count_match"] is False
    assert verdict["content_match"] is False


def test_the_column_order_a_container_uses_does_not_matter() -> None:
    # Same measurements, different dict insertion order.
    shuffled = dict(reversed(list(_row().items())))
    assert result_from_rows("q", [shuffled]).content_sha256 == (
        result_from_rows("q", [_row()]).content_sha256
    )


def test_a_row_missing_a_sensor_axis_is_refused() -> None:
    broken = _row()
    del broken["gyroscope_z"]
    with pytest.raises((RowIdentityError, KeyError)):
        result_from_rows("q", [broken])
