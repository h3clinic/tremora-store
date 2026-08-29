from __future__ import annotations

import dataclasses
import hashlib
from fractions import Fraction
from pathlib import Path

import pytest
from _pads_fixtures import timeseries_bytes
from motionbloom.tremora_store.pads.authority import (
    CANONICAL_CHANNELS,
    CHANNEL_UNITS,
)
from motionbloom.tremora_store.pads.movement import StreamDeclaration
from motionbloom.tremora_store.pads.p02 import segments as segments_module
from motionbloom.tremora_store.pads.p02.bilateral import (
    BilateralError,
    assert_no_sample_level_claim,
    build_bilateral_tasks,
    build_bilateral_window_pairs,
)
from motionbloom.tremora_store.pads.p02.contract import (
    GAP_ABSOLUTE_CAP_PS,
    WINDOW_DURATION_PS,
    WINDOW_STRIDE_PS,
)
from motionbloom.tremora_store.pads.p02.folds import (
    FoldError,
    assert_participant_disjoint,
    assign_folds,
    fold_sizes,
)
from motionbloom.tremora_store.pads.p02.replay import (
    ReplayError,
    replay_sha256,
    replay_stream,
    replay_task,
    replay_window,
)
from motionbloom.tremora_store.pads.p02.sample_store import SampleStoreWriter
from motionbloom.tremora_store.pads.p02.segments import (
    SegmentError,
    assert_partitions_stream,
    build_segments,
    gap_threshold_ps,
    integer_median,
)
from motionbloom.tremora_store.pads.p02.stream_reader import read_stream
from motionbloom.tremora_store.pads.p02.windows import (
    WindowError,
    assert_windows_inside_segments,
    build_windows,
    support_coverage_fraction,
)

RATE = Fraction(100)
INTERVAL = 0.0099946
EVIDENCE = "e" * 64


def _declaration(device_location: str = "LeftWrist") -> StreamDeclaration:
    return StreamDeclaration(
        device_location=device_location,
        channels=CANONICAL_CHANNELS,
        units=tuple(CHANNEL_UNITS[name] for name in CANONICAL_CHANNELS),
        file_name=f"timeseries/001_Relaxed_{device_location}.txt",
    )


def _stream(
    payload: bytes,
    *,
    rows: int,
    device_location: str = "LeftWrist",
    task: str = "Relaxed",
):
    return read_stream(
        payload,
        declaration=_declaration(device_location),
        declared_rows=rows,
        sampling_rate=RATE,
        stream_id=f"001:{task}:{device_location}",
        participant_id="001",
        assessment_id=f"001:{task}",
        task_name=task,
        source_asset_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _with_gap(rows: int, *, at: int, gap_seconds: float) -> bytes:
    lines = []
    time = 0.0
    for index in range(rows):
        if index == at:
            time += gap_seconds
        else:
            time += INTERVAL if index else 0.0
        lines.append(
            f"{time:.10f}," + ",".join(["0.0010000000"] * 6)
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


# --- segments -------------------------------------------------------------


def test_a_clean_stream_is_one_segment() -> None:
    samples = _stream(timeseries_bytes(512), rows=512)
    built = build_segments(samples)
    assert len(built) == 1
    assert built[0].break_reason_before == "STREAM_START"
    assert built[0].break_reason_after == "STREAM_END"
    assert built[0].sample_count == 512
    assert_partitions_stream(built, samples)


def test_a_gap_above_the_threshold_starts_a_new_segment() -> None:
    payload = _with_gap(512, at=200, gap_seconds=0.05)
    samples = _stream(payload, rows=512)
    built = build_segments(samples)
    assert len(built) == 2
    assert built[0].break_reason_after == "TIME_GAP"
    assert built[1].break_reason_before == "TIME_GAP"
    assert built[0].sample_count + built[1].sample_count == 512
    assert_partitions_stream(built, samples)


def test_a_gap_below_the_threshold_does_not_split() -> None:
    payload = _with_gap(512, at=200, gap_seconds=0.025)
    samples = _stream(payload, rows=512)
    assert len(build_segments(samples)) == 1


def test_the_threshold_is_cadence_relative_with_an_absolute_cap() -> None:
    assert gap_threshold_ps(9_994_600_000) == 3 * 9_994_600_000
    # A pathologically slow stream is still capped at 100 ms.
    assert gap_threshold_ps(500_000_000_000) == GAP_ABSOLUTE_CAP_PS


def test_a_non_positive_delta_breaks_the_segment() -> None:
    payload = timeseries_bytes(
        64, time_override={30: "0.0010000000"}
    )
    samples = _stream(payload, rows=64)
    built = build_segments(samples)
    assert len(built) > 1
    assert "NONPOSITIVE_DELTA" in {
        segment.break_reason_before for segment in built
    }
    assert_partitions_stream(built, samples)


def test_an_invalid_sample_breaks_the_segment() -> None:
    samples = _stream(timeseries_bytes(64), rows=64)
    statuses = ["SAMPLE_VALID"] * 64
    statuses[20] = "SAMPLE_REFUSED"
    built = build_segments(samples, sample_statuses=statuses)
    assert built[1].break_reason_before == "INVALID_SAMPLE"
    assert_partitions_stream(built, samples)


def test_a_stream_without_a_cadence_is_not_split() -> None:
    samples = _stream(timeseries_bytes(4), rows=4)
    built = build_segments(samples)
    assert len(built) == 1
    assert built[0].dt_ref_ps is None
    assert built[0].gap_threshold_ps is None


def test_the_median_interval_is_an_exact_integer() -> None:
    assert integer_median([3, 1, 2]) == 2
    assert integer_median([4, 1, 2, 3]) == 2
    assert integer_median([]) is None


def test_segments_that_do_not_partition_are_refused() -> None:
    samples = _stream(timeseries_bytes(64), rows=64)
    built = build_segments(samples)
    broken = (dataclasses.replace(built[0], sample_count=10),)
    with pytest.raises(SegmentError):
        assert_partitions_stream(broken, samples)


# --- windows --------------------------------------------------------------


def _windowed(payload: bytes, rows: int, **kwargs):
    samples = _stream(payload, rows=rows, **kwargs)
    built = build_segments(samples)
    windows = build_windows(
        samples, built, split_group_id="001", outer_fold=2
    )
    return samples, built, windows


def test_windows_sit_on_a_stride_grid_anchored_at_task_local_zero() -> None:
    _, _, windows = _windowed(timeseries_bytes(2048), 2048)
    starts = [window.window_start_task_local_ps for window in windows]
    assert starts[0] == 0
    assert all(start % WINDOW_STRIDE_PS == 0 for start in starts)
    assert starts == sorted(starts)
    for window in windows:
        assert window.window_end_task_local_ps - (
            window.window_start_task_local_ps
        ) == WINDOW_DURATION_PS


def test_window_membership_is_by_time_not_by_counting_samples() -> None:
    _, _, windows = _windowed(timeseries_bytes(2048), 2048)
    counts = {window.sample_count for window in windows}
    # A jittering clock does not deliver exactly 400 samples per window.
    assert counts != {400}
    assert all(395 <= count <= 405 for count in counts)


def test_a_gap_costs_the_windows_that_would_have_crossed_it() -> None:
    _, _, clean = _windowed(timeseries_bytes(1024), 1024)
    _, built, gapped = _windowed(
        _with_gap(1024, at=400, gap_seconds=0.05), 1024
    )
    assert len(built) == 2
    assert len(gapped) < len(clean)
    for window in gapped:
        segment = next(
            item for item in built if item.segment_id == window.segment_id
        )
        assert window.first_sample_ordinal >= segment.first_sample_ordinal
        assert window.last_sample_ordinal <= segment.last_sample_ordinal


def test_built_windows_are_checked_against_their_segments() -> None:
    _, built, windows = _windowed(timeseries_bytes(1024), 1024)
    assert_windows_inside_segments(windows, built)
    tampered = (
        dataclasses.replace(windows[0], last_sample_ordinal=10_000),
    )
    with pytest.raises(WindowError):
        assert_windows_inside_segments(tampered, built)


def test_a_window_naming_an_unknown_segment_is_refused() -> None:
    _, built, windows = _windowed(timeseries_bytes(1024), 1024)
    tampered = (dataclasses.replace(windows[0], segment_id="nowhere"),)
    with pytest.raises(WindowError):
        assert_windows_inside_segments(tampered, built)


def test_coverage_is_the_union_of_sample_supports() -> None:
    # Two samples one interval apart cover two intervals inside a window that
    # is twenty intervals long.
    dt = 1_000_000_000
    fraction = support_coverage_fraction(
        [5 * dt, 6 * dt], start_ps=0, end_ps=20 * dt, dt_ref_ps=dt
    )
    assert fraction == pytest.approx(2 / 20)


def test_a_dense_window_reports_nearly_full_coverage() -> None:
    _, _, windows = _windowed(timeseries_bytes(1024), 1024)
    assert all(window.coverage_fraction > 0.99 for window in windows)
    assert all(window.coverage_fraction <= 1.0 for window in windows)


def test_windows_carry_the_participant_fold_and_authority() -> None:
    _, _, windows = _windowed(timeseries_bytes(1024), 1024)
    record = windows[0].as_record()
    assert record["outer_fold"] == 2
    assert record["split_group_id"] == "001"
    assert record["timing_authority"] == "SOURCE_RELATIVE_UNIMODAL_CLOCK"


def test_a_stream_with_non_gapless_ordinals_is_refused() -> None:
    samples = _stream(timeseries_bytes(64), rows=64)
    built = build_segments(samples)
    samples.sample_ordinal[10] = 999
    with pytest.raises(WindowError):
        build_windows(samples, built, split_group_id="001", outer_fold=0)


# --- folds ----------------------------------------------------------------


CONDITIONS = {
    **{f"P{i:03d}": "Parkinson's" for i in range(276)},
    **{f"H{i:03d}": "Healthy" for i in range(79)},
    **{f"M{i:03d}": "Multiple Sclerosis" for i in range(11)},
}


def test_fold_assignment_is_deterministic() -> None:
    assert assign_folds(CONDITIONS) == assign_folds(CONDITIONS)


def test_folds_are_participant_disjoint() -> None:
    assignment = assign_folds(CONDITIONS)
    assert set(assignment) == set(CONDITIONS)
    assert set(assignment.values()) == {0, 1, 2, 3, 4}
    assert_participant_disjoint(
        assignment, {f"{p}:Relaxed:LeftWrist": p for p in CONDITIONS}
    )


def test_each_condition_is_spread_across_the_folds() -> None:
    assignment = assign_folds(CONDITIONS)
    for condition in ("Parkinson's", "Healthy", "Multiple Sclerosis"):
        members = [
            assignment[p] for p, c in CONDITIONS.items() if c == condition
        ]
        assert len(set(members)) == 5


def test_fold_sizes_are_balanced() -> None:
    sizes = fold_sizes(assign_folds(CONDITIONS))
    assert max(sizes.values()) - min(sizes.values()) <= 3


def test_a_different_seed_gives_a_different_assignment() -> None:
    assert assign_folds(CONDITIONS) != assign_folds(
        CONDITIONS, split_seed=1
    )


def test_a_stream_without_a_fold_is_refused() -> None:
    assignment = assign_folds({"A": "Healthy"})
    with pytest.raises(FoldError):
        assert_participant_disjoint(assignment, {"B:Relaxed:LeftWrist": "B"})


# --- bilateral ------------------------------------------------------------


def _bilateral_windows():
    left = _windowed(timeseries_bytes(1024), 1024)[2]
    right = _windowed(
        timeseries_bytes(1024), 1024, device_location="RightWrist"
    )[2]
    return left, right


def test_windows_pair_by_task_local_offset() -> None:
    left, right = _bilateral_windows()
    pairs = build_bilateral_window_pairs([*left, *right])
    assert len(pairs) == min(len(left), len(right))
    for pair in pairs:
        assert pair.left_window_id.endswith(
            str(pair.window_start_task_local_ps)
        )
        assert pair.right_window_id.endswith(
            str(pair.window_start_task_local_ps)
        )


def test_a_one_sided_window_is_not_paired() -> None:
    left, right = _bilateral_windows()
    pairs = build_bilateral_window_pairs([*left, *right[:-1]])
    assert len(pairs) == len(right) - 1


def test_every_bilateral_row_refuses_sample_level_fusion() -> None:
    left, right = _bilateral_windows()
    pairs = build_bilateral_window_pairs([*left, *right])
    records = [pair.as_record() for pair in pairs]
    assert_no_sample_level_claim(records)
    for record in records:
        assert record["cross_wrist_clock_alignment"] == "UNRESOLVED"
        assert record["sample_level_fusion_allowed"] is False
        assert record["pairing_authority"] == "SOURCE_PROTOCOL_PAIR"
        assert record["pairing_status"] == "PROTOCOL_COINDEXED"


def test_a_row_claiming_fusion_is_refused() -> None:
    with pytest.raises(BilateralError):
        assert_no_sample_level_claim([{
            "sample_level_fusion_allowed": True,
            "cross_wrist_clock_alignment": "UNRESOLVED",
        }])
    with pytest.raises(BilateralError):
        assert_no_sample_level_claim([{
            "sample_level_fusion_allowed": False,
            "cross_wrist_clock_alignment": "RESOLVED",
        }])


def test_a_task_missing_a_wrist_is_marked_incomplete() -> None:
    tasks = build_bilateral_tasks(
        {"001:Relaxed": {"LeftWrist": "001:Relaxed:LeftWrist"}},
        participant_of={"001:Relaxed": "001"},
        task_of={"001:Relaxed": "Relaxed"},
    )
    assert tasks[0].pair_status == "PAIR_MISSING_STREAM"
    assert tasks[0].as_record()["sample_level_fusion_allowed"] is False


def test_two_windows_for_one_wrist_at_one_offset_are_refused() -> None:
    left, _ = _bilateral_windows()
    with pytest.raises(BilateralError):
        build_bilateral_window_pairs([left[0], left[0]])


# --- replay ---------------------------------------------------------------


@pytest.fixture
def stored(tmp_path: Path):
    payload_left = timeseries_bytes(1024)
    payload_right = timeseries_bytes(1024)
    left = _stream(payload_left, rows=1024)
    right = _stream(payload_right, rows=1024, device_location="RightWrist")
    index = {}
    with SampleStoreWriter(tmp_path, p01_evidence_sha256=EVIDENCE) as writer:
        for samples in sorted((left, right), key=lambda s: s.stream_id):
            entry = writer.add(samples)
            index[entry.stream_id] = entry.as_record()
    return tmp_path, index, left, right


def test_stream_replay_returns_every_sample_in_source_order(stored) -> None:
    root, index, left, _ = stored
    table = replay_stream(root, index, left.stream_id)
    assert table.num_rows == 1024
    assert table.column("sample_ordinal").to_pylist() == list(range(1024))
    assert table.column("source_time_token").to_pylist() == (
        left.source_time_token
    )


def test_stream_replay_rebuilds_the_source_bytes(stored) -> None:
    root, index, left, _ = stored
    table = replay_stream(root, index, left.stream_id)
    assert replay_sha256(table, left.source_channel_order) == (
        left.source_asset_sha256
    )


def test_window_replay_returns_exactly_the_indexed_rows(stored) -> None:
    root, index, left, _ = stored
    built = build_segments(left)
    windows = build_windows(
        left, built, split_group_id="001", outer_fold=0
    )
    window = windows[1].as_record()
    table = replay_window(root, index, window)
    assert table.num_rows == window["sample_count"]
    assert table.column("sample_ordinal")[0].as_py() == (
        window["first_sample_ordinal"]
    )
    assert table.column("sample_ordinal")[-1].as_py() == (
        window["last_sample_ordinal"]
    )
    assert table.column("source_time_ps")[0].as_py() == (
        window["first_source_time_ps"]
    )


def test_window_replay_refuses_a_disagreeing_index(stored) -> None:
    root, index, left, _ = stored
    built = build_segments(left)
    window = build_windows(
        left, built, split_group_id="001", outer_fold=0
    )[0].as_record()
    window["sample_count"] = window["sample_count"] + 1
    with pytest.raises(ReplayError):
        replay_window(root, index, window)


def test_task_replay_returns_both_wrists_and_refuses_to_align_them(
    stored,
) -> None:
    root, index, left, right = stored
    task = replay_task(root, index, {
        "assessment_id": "001:Relaxed",
        "participant_id": "001",
        "task_name": "Relaxed",
        "left_stream_id": left.stream_id,
        "right_stream_id": right.stream_id,
    })
    assert task.left.num_rows == task.right.num_rows == 1024
    assert task.authority["cross_wrist_clock_alignment"] == "UNRESOLVED"
    assert task.authority["sample_level_fusion_allowed"] is False
    assert task.authority["bilateral_pairing_authority"] == (
        "SOURCE_PROTOCOL_PAIR"
    )


def test_replay_of_an_unstored_stream_is_refused(stored) -> None:
    root, index, _, _ = stored
    with pytest.raises(ReplayError):
        replay_stream(root, index, "999:Nothing:LeftWrist")


def test_segment_module_exports_its_break_reasons() -> None:
    assert segments_module.BREAK_TIME_GAP == "TIME_GAP"
    assert segments_module.BREAK_STREAM_START == "STREAM_START"
