from __future__ import annotations

import random

import pytest
from _ego4d_fixtures import DOCUMENTED_DT_MS, clean_rows, imu_csv
from motionbloom.tremora_store.ego4d import coverage as coverage_module
from motionbloom.tremora_store.ego4d.authority import (
    EGO4D_IMU_COLUMNS,
    Ego4DAuthorityError,
    assert_not_relabelled,
    authority_contract,
)
from motionbloom.tremora_store.ego4d.imu_parser import (
    Ego4DParseError,
    parse_normalized_imu_csv,
    split_records,
)
from motionbloom.tremora_store.ego4d.metadata import (
    Ego4DMetadataError,
    parse_asset_manifest,
    parse_metadata_snapshot,
)
from motionbloom.tremora_store.ego4d.pts_validation import (
    TIMELINE_INSUFFICIENT_FRAMES,
    TIMELINE_ORIGIN_DISAGREEMENT,
    TIMELINE_RECONCILED,
    TIMELINE_SPAN_DISAGREEMENT,
    quantify_row_relationships,
    reconcile_pts_timeline,
)
from motionbloom.tremora_store.ego4d.row_status import (
    STATUS_PRECEDENCE,
    IssueBit,
    issue_bit_names,
    resolve_status,
)
from motionbloom.tremora_store.ego4d.selection import (
    STRATA,
    STRATUM_CLEAN_MONOTONIC,
    STRATUM_EXTREME_TIMESTAMP,
    VideoCandidate,
    select_subset,
    selection_key,
)
from motionbloom.tremora_store.ego4d.tokens import TokenKind, classify

SNAPSHOT = {
    "videos": [{
        "video_uid": "V1",
        "canonical_video_duration_ms": 10_000.0,
        "video_stream_start_ms": 0.0,
        "video_stream_end_ms": 10_000.0,
        "capture_device_group": "GROUP_A",
        "components": [{
            "component_idx": 0,
            "component_start_in_canonical_ms": 0.0,
            "component_end_in_canonical_ms": 10_000.0,
            "has_imu": True,
        }],
    }]
}


def _timeline():
    import json
    snapshot = parse_metadata_snapshot(
        json.dumps(SNAPSHOT).encode("utf-8"), snapshot_sha256="0" * 64
    )
    return snapshot.video("V1")


def _parse(rows, **kwargs):
    return parse_normalized_imu_csv(
        imu_csv(rows, **kwargs),
        video_uid="V1",
        source_asset_sha256="a" * 64,
        timeline=_timeline(),
    )


# --- authority -------------------------------------------------------------


def test_ego4d_cannot_be_relabelled_a_raw_clock() -> None:
    for tier in ("RAW_SHARED_CLOCK", "RAW_MAPPED_CLOCK"):
        with pytest.raises(Ego4DAuthorityError):
            assert_not_relabelled(tier)
    assert assert_not_relabelled(
        "SOURCE_CANONICAL_TIMESTAMP"
    ).value == "SOURCE_CANONICAL_TIMESTAMP"


def test_authority_contract_declares_a_derived_origin_assumption() -> None:
    contract = authority_contract()
    assert contract["timing_authority"] == "SOURCE_CANONICAL_TIMESTAMP"
    assert contract["raw_shared_clock"] is False
    assert contract["hardware_sync_claim"] is False
    assert contract["derived_under_assumption"] is True
    assert "t=0" in contract["assumption"]


def test_source_column_spelling_is_preserved() -> None:
    assert "accl_x" in EGO4D_IMU_COLUMNS
    assert "accel_x" not in EGO4D_IMU_COLUMNS


# --- tokens ---------------------------------------------------------------


@pytest.mark.parametrize("token", ("1_000", "0x10", " 5", "5 ", "abc", "1,0"))
def test_python_float_spellings_the_source_never_wrote_are_unparseable(
    token: str,
) -> None:
    assert classify(token)[0] is TokenKind.UNPARSEABLE


@pytest.mark.parametrize("token", ("nan", "NaN", "NAN", "-inf", "Infinity"))
def test_non_finite_spellings_land_in_one_bucket(token: str) -> None:
    assert classify(token)[0] is TokenKind.NONFINITE


@pytest.mark.parametrize("token", ("", "null", "NULL", "na", "n/a"))
def test_null_spellings_are_not_non_finite(token: str) -> None:
    assert classify(token)[0] is TokenKind.NULL


def test_a_decimal_token_keeps_its_exact_value() -> None:
    kind, value = classify("4.975124378109452")
    assert kind is TokenKind.DECIMAL
    assert value == 4.975124378109452


# --- record splitting -----------------------------------------------------


def test_a_single_trailing_terminator_is_not_a_record() -> None:
    records, terminator = split_records(b"a\nb\n")
    assert records == ("a", "b")
    assert terminator == "\n"


def test_a_blank_interior_record_is_a_parse_failure() -> None:
    with pytest.raises(Ego4DParseError):
        split_records(b"a\n\nb\n")


def test_crlf_is_accepted_and_mixed_terminators_are_not() -> None:
    records, terminator = split_records(b"a\r\nb\r\n")
    assert records == ("a", "b")
    assert terminator == "\r\n"
    with pytest.raises(Ego4DParseError):
        split_records(b"a\r\nb\n")


# --- parser ---------------------------------------------------------------


def test_header_must_be_the_normalized_ego4d_column_order() -> None:
    shuffled = list(EGO4D_IMU_COLUMNS)
    shuffled[7], shuffled[8] = shuffled[8], shuffled[7]
    with pytest.raises(Ego4DParseError):
        _parse(clean_rows(count=10), header=shuffled)


def test_a_short_row_is_a_parse_failure() -> None:
    rows = clean_rows(count=3)
    rows[1] = rows[1][:-1]
    with pytest.raises(Ego4DParseError):
        _parse(rows)


def test_a_malformed_component_index_is_a_parse_failure() -> None:
    rows = clean_rows(count=3)
    rows[1][0] = "one"
    with pytest.raises(Ego4DParseError):
        _parse(rows)


def test_source_order_and_ordinals_survive_parsing() -> None:
    parsed = _parse(clean_rows(count=25))
    assert parsed.data_line_count == 25
    assert [row.source_row_ordinal for row in parsed.rows] == list(range(25))
    assert all(row.eligible for row in parsed.rows)


def test_tokens_are_preserved_exactly_as_written() -> None:
    rows = clean_rows(count=2)
    rows[0][2] = "1.500"
    parsed = _parse(rows)
    assert parsed.rows[0].canonical_timestamp_token == "1.500"
    assert parsed.rows[0].canonical_timestamp_ms == 1.5


def test_a_null_canonical_timestamp_is_never_inferred() -> None:
    rows = clean_rows(count=2)
    rows[1][2] = ""
    parsed = _parse(rows)
    row = parsed.rows[1]
    assert row.canonical_timestamp_ms is None
    assert row.canonical_authority_status == "SOURCE_CANONICAL_NULL_AFTER_TRIM"


def test_a_non_finite_canonical_timestamp_is_distinct_from_null() -> None:
    rows = clean_rows(count=2)
    rows[1][2] = "NaN"
    parsed = _parse(rows)
    assert parsed.rows[1].canonical_authority_status == (
        "SOURCE_CANONICAL_NONFINITE"
    )


def test_an_unparseable_token_is_not_silently_coerced() -> None:
    rows = clean_rows(count=2)
    rows[1][2] = "1_000"
    parsed = _parse(rows)
    assert parsed.rows[1].canonical_authority_status == (
        "SOURCE_CANONICAL_UNPARSEABLE_TOKEN"
    )
    assert parsed.rows[1].canonical_timestamp_ms is None
    assert parsed.rows[1].canonical_timestamp_token == "1_000"


def test_a_timestamp_outside_the_video_stays_visible() -> None:
    rows = clean_rows(count=2)
    rows[1][2] = "50000.0"
    parsed = _parse(rows)
    assert parsed.rows[1].canonical_authority_status == (
        "SOURCE_CANONICAL_OUTSIDE_VIDEO"
    )
    assert parsed.rows[1].canonical_timestamp_ms == 50000.0


def test_an_extreme_timestamp_is_recorded_not_repaired() -> None:
    rows = clean_rows(count=2)
    rows[1][2] = "999999999999.0"
    parsed = _parse(rows)
    bits = IssueBit(parsed.rows[1].issue_bits)
    assert IssueBit.SOURCE_CANONICAL_EXTREME_MAGNITUDE & bits
    assert parsed.rows[1].canonical_timestamp_ms == 999999999999.0


def test_a_duplicate_timestamp_is_classified_and_kept() -> None:
    rows = clean_rows(count=3)
    rows[2][2] = rows[1][2]
    parsed = _parse(rows)
    assert len(parsed.rows) == 3
    assert parsed.rows[2].canonical_authority_status == (
        "SOURCE_CANONICAL_DUPLICATE"
    )


def test_non_monotonic_is_measured_against_the_frontier_not_the_previous_row(
) -> None:
    rows = clean_rows(count=3)
    rows[0][2] = "100.0"
    rows[1][2] = "50.0"
    rows[2][2] = "70.0"
    parsed = _parse(rows)
    # 70 is above 50 but below the 100 already established in source order, so
    # it stays out of the eligible set and the eligible subsequence keeps
    # increasing.
    assert [row.canonical_authority_status for row in parsed.rows] == [
        "SOURCE_CANONICAL_VALID",
        "SOURCE_CANONICAL_NONMONOTONIC",
        "SOURCE_CANONICAL_NONMONOTONIC",
    ]
    eligible = [row.canonical_timestamp_ms for row in parsed.rows if row.eligible]
    assert eligible == [100.0]


def test_no_row_is_dropped_for_non_monotonicity() -> None:
    rows = clean_rows(count=6)
    rows[3][2] = "1.0"
    parsed = _parse(rows)
    assert len(parsed.rows) == 6
    assert parsed.data_line_count == 6


def test_missing_sensor_axes_are_separated_by_sensor() -> None:
    rows = clean_rows(count=3)
    rows[1][6] = ""
    rows[2][3] = "NaN"
    parsed = _parse(rows)
    assert parsed.rows[1].canonical_authority_status == "MISSING_ACCELERATION"
    assert parsed.rows[1].accl_x is None
    assert parsed.rows[2].canonical_authority_status == "MISSING_GYROSCOPE"


def test_a_row_outside_the_metadata_component_is_not_covered() -> None:
    rows = clean_rows(count=2, component_idx=7)
    parsed = _parse(rows)
    assert parsed.rows[0].canonical_authority_status == "COMPONENT_NOT_COVERED"


# --- precedence -----------------------------------------------------------


def test_precedence_names_one_verdict_but_records_every_condition() -> None:
    bits = (
        IssueBit.MISSING_GYROSCOPE
        | IssueBit.SOURCE_CANONICAL_DUPLICATE
        | IssueBit.SOURCE_CANONICAL_OUTSIDE_VIDEO
    )
    assert resolve_status(bits) == "SOURCE_CANONICAL_OUTSIDE_VIDEO"
    assert issue_bit_names(bits) == (
        "SOURCE_CANONICAL_OUTSIDE_VIDEO",
        "SOURCE_CANONICAL_DUPLICATE",
        "MISSING_GYROSCOPE",
    )


def test_every_disqualifying_bit_has_a_precedence_entry() -> None:
    assert set(STATUS_PRECEDENCE) == set(IssueBit)


@pytest.mark.parametrize("bit", list(IssueBit))
def test_no_disqualifying_bit_can_be_called_valid(bit: IssueBit) -> None:
    assert resolve_status(bit) != "SOURCE_CANONICAL_VALID"


def test_only_a_clean_row_is_valid() -> None:
    assert resolve_status(0) == "SOURCE_CANONICAL_VALID"


# --- coverage -------------------------------------------------------------


def test_coverage_is_sample_support_not_first_to_last_span() -> None:
    times = [index * DOCUMENTED_DT_MS for index in range(200)]
    measured = coverage_module.component_coverage(
        0, times, clamp_low=0.0, clamp_high=3_600_000.0
    )
    assert measured.coverage_status == coverage_module.COVERAGE_MEASURED
    assert measured.reference_interval_ms == pytest.approx(DOCUMENTED_DT_MS)
    # 200 supports of one interval each, less the half-interval the first
    # sample loses to the clamp at canonical zero.
    assert measured.coverage_ms == pytest.approx(
        199.5 * DOCUMENTED_DT_MS, rel=1e-9
    )


def test_two_endpoints_an_hour_apart_are_not_an_hour_of_coverage() -> None:
    measured = coverage_module.component_coverage(
        0, [0.0, 3_600_000.0], clamp_low=0.0, clamp_high=3_600_000.0
    )
    assert measured.coverage_status == coverage_module.COVERAGE_NO_CADENCE
    assert measured.coverage_ms == 0.0


def test_fewer_than_eight_deltas_is_not_a_cadence() -> None:
    times = [index * DOCUMENTED_DT_MS for index in range(8)]
    assert coverage_module.estimate_reference_interval(times) is None
    times = [index * DOCUMENTED_DT_MS for index in range(9)]
    assert coverage_module.estimate_reference_interval(times) is not None


def test_a_ninety_five_millisecond_hole_at_two_hundred_hertz_breaks() -> None:
    times = [index * DOCUMENTED_DT_MS for index in range(20)]
    times += [times[-1] + 95.0 + index * DOCUMENTED_DT_MS for index in range(20)]
    measured = coverage_module.component_coverage(
        0, times, clamp_low=0.0, clamp_high=3_600_000.0
    )
    assert measured.segment_count == 2
    assert measured.continuity_threshold_ms == pytest.approx(
        3 * DOCUMENTED_DT_MS
    )


@pytest.mark.parametrize(
    ("multiplier", "expected_segments"), ((2.0, 2), (3.0, 2), (5.0, 1))
)
def test_continuity_multiplier_sensitivity(
    multiplier: float, expected_segments: int
) -> None:
    times = [index * DOCUMENTED_DT_MS for index in range(20)]
    times += [
        times[-1] + 4 * DOCUMENTED_DT_MS + index * DOCUMENTED_DT_MS
        for index in range(20)
    ]
    measured = coverage_module.component_coverage(
        0,
        times,
        clamp_low=0.0,
        clamp_high=3_600_000.0,
        multiplier=multiplier,
    )
    assert measured.segment_count == expected_segments


def test_a_low_rate_component_is_still_capped_at_one_hundred_ms() -> None:
    assert coverage_module.continuity_threshold_ms(1000.0) == 100.0


def test_coverage_refuses_a_sequence_that_is_not_increasing() -> None:
    with pytest.raises(coverage_module.Ego4DCoverageError):
        coverage_module.component_coverage(
            0, [5.0, 4.0], clamp_low=0.0, clamp_high=10.0
        )


def test_overlapping_components_cannot_double_count() -> None:
    times = [index * DOCUMENTED_DT_MS for index in range(50)]
    first = coverage_module.component_coverage(
        0, times, clamp_low=0.0, clamp_high=10_000.0
    )
    second = coverage_module.component_coverage(
        1, times, clamp_low=0.0, clamp_high=10_000.0
    )
    assert coverage_module.video_coverage_ms(
        [first, second]
    ) == pytest.approx(first.coverage_ms)


# --- selection ------------------------------------------------------------


def _candidates(count: int, *, coverage_ms: float = 3_600_000.0):
    return [
        VideoCandidate(
            video_uid=f"V{index:04d}",
            strata=frozenset({STRATUM_CLEAN_MONOTONIC}),
            paired_coverage_ms=coverage_ms,
            capture_device_group="GROUP_A" if index % 2 else "GROUP_B",
        )
        for index in range(count)
    ]


def test_selection_does_not_depend_on_input_order() -> None:
    candidates = _candidates(40)
    shuffled = list(candidates)
    random.Random(7).shuffle(shuffled)
    first = select_subset(
        candidates, metadata_snapshot_sha256="a" * 64, minimum_videos=10,
        minimum_coverage_hours=1.0,
    )
    second = select_subset(
        shuffled, metadata_snapshot_sha256="a" * 64, minimum_videos=10,
        minimum_coverage_hours=1.0,
    )
    assert first.selected_video_uids == second.selected_video_uids


def test_selection_changes_with_the_metadata_snapshot() -> None:
    candidates = _candidates(40)
    first = select_subset(
        candidates, metadata_snapshot_sha256="a" * 64, minimum_videos=10,
        minimum_coverage_hours=1.0,
    )
    second = select_subset(
        candidates, metadata_snapshot_sha256="b" * 64, minimum_videos=10,
        minimum_coverage_hours=1.0,
    )
    assert first.selected_video_uids != second.selected_video_uids


def test_selection_key_is_a_pure_function_of_the_frozen_inputs() -> None:
    assert selection_key("V1", metadata_snapshot_sha256="a" * 64) == (
        selection_key("V1", metadata_snapshot_sha256="a" * 64)
    )
    assert selection_key("V1", metadata_snapshot_sha256="a" * 64) != (
        selection_key("V2", metadata_snapshot_sha256="a" * 64)
    )


def test_an_absent_stratum_is_named_and_never_topped_up() -> None:
    selection = select_subset(
        _candidates(120),
        metadata_snapshot_sha256="a" * 64,
    )
    assert STRATUM_CLEAN_MONOTONIC in selection.strata_present_in_source
    assert STRATUM_EXTREME_TIMESTAMP in selection.strata_absent_in_source
    assert selection.shortfalls[f"stratum:{STRATUM_EXTREME_TIMESTAMP}"] == 1.0
    assert selection.floors_satisfied is False


def test_every_available_stratum_is_represented() -> None:
    candidates = [
        VideoCandidate(
            video_uid=f"V{index:04d}",
            strata=frozenset({STRATA[index % len(STRATA)]}),
            paired_coverage_ms=3_600_000.0,
            capture_device_group="GROUP_A" if index % 2 else "GROUP_B",
        )
        for index in range(120)
    ]
    selection = select_subset(candidates, metadata_snapshot_sha256="a" * 64)
    assert set(selection.strata_represented) == set(STRATA)
    assert selection.floors_satisfied is True
    assert selection.selected_video_count >= 100


def test_the_frozen_floors_are_published_in_the_record() -> None:
    record = select_subset(
        [], metadata_snapshot_sha256="a" * 64
    ).as_record()
    assert record["floors"] == {
        "minimum_videos": 100,
        "minimum_coverage_hours": 10.0,
        "minimum_capture_device_groups": 2,
    }
    assert record["selection_seed"] == 20260828


# --- PTS reconciliation ---------------------------------------------------


def test_a_shifted_origin_is_caught_even_when_the_span_agrees() -> None:
    times = [1000.0 + index * 33.3667 for index in range(1000)]
    result = reconcile_pts_timeline(
        times, canonical_video_duration_ms=times[-1] - times[0]
    )
    assert result.span_agrees is True
    assert result.origin_agrees is False
    assert result.timeline_status == TIMELINE_ORIGIN_DISAGREEMENT


def test_a_two_frame_hour_does_not_agree_with_a_half_hour_video() -> None:
    result = reconcile_pts_timeline(
        [0.0, 3_600_000.0], canonical_video_duration_ms=1_800_000.0
    )
    assert result.span_tolerance_ms == 150.0
    assert result.timeline_status == TIMELINE_SPAN_DISAGREEMENT


def test_a_clean_timeline_reconciles() -> None:
    times = [index * 33.3667 for index in range(900)]
    result = reconcile_pts_timeline(
        times, canonical_video_duration_ms=times[-1]
    )
    assert result.timeline_status == TIMELINE_RECONCILED


def test_a_single_frame_cannot_reconcile() -> None:
    result = reconcile_pts_timeline([0.0], canonical_video_duration_ms=100.0)
    assert result.timeline_status == TIMELINE_INSUFFICIENT_FRAMES


def test_row_relationships_are_aggregate_only() -> None:
    frames = [0.0, 33.0, 66.0]
    result = quantify_row_relationships(frames, [10.0, 100.0, -5.0])
    assert result.rows_inside_a_frame_interval == 1
    assert result.rows_after_last_frame == 1
    assert result.rows_before_first_frame == 1
    assert result.max_nearest_frame_delta_ms == pytest.approx(34.0)


# --- metadata -------------------------------------------------------------


def test_a_snapshot_with_unknown_keys_is_refused() -> None:
    import json
    payload = json.loads(json.dumps(SNAPSHOT))
    payload["videos"][0]["surprise"] = 1
    with pytest.raises(Ego4DMetadataError):
        parse_metadata_snapshot(
            json.dumps(payload).encode("utf-8"), snapshot_sha256="0" * 64
        )


def test_a_manifest_entry_that_escapes_its_root_is_refused() -> None:
    import json
    payload = {
        "assets": [{
            "video_uid": "V1",
            "component_idx": 0,
            "imu_relative_path": "../../etc/passwd",
            "canonical_video_relative_path": "V1.mp4",
            "imu_asset_sha256": "a" * 64,
            "video_component_asset_sha256": "b" * 64,
            "canonical_video_asset_sha256": "c" * 64,
        }]
    }
    with pytest.raises(Ego4DMetadataError):
        parse_asset_manifest(json.dumps(payload).encode("utf-8"))


def test_a_manifest_with_a_bad_hash_field_is_refused() -> None:
    import json
    payload = {
        "assets": [{
            "video_uid": "V1",
            "component_idx": 0,
            "imu_relative_path": "V1.csv",
            "canonical_video_relative_path": "V1.mp4",
            "imu_asset_sha256": "not-a-hash",
            "video_component_asset_sha256": "b" * 64,
            "canonical_video_asset_sha256": "c" * 64,
        }]
    }
    with pytest.raises(Ego4DMetadataError):
        parse_asset_manifest(json.dumps(payload).encode("utf-8"))
