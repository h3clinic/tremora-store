from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from _pads_fixtures import build_release
from motionbloom.tremora_store.pads.p02.materialize import (
    materialize as p02_materialize,
)
from motionbloom.tremora_store.pads.p03 import selection
from motionbloom.tremora_store.pads.p03.contract import (
    GAP_ADJACENT,
    INTERIOR,
    SENSOR_FAMILIES,
    SPECTRALLY_ELIGIBLE,
)
from motionbloom.tremora_store.pads.p03.materialize import (
    PRESERVED,
    materialize,
    read_store,
)
from motionbloom.tremora_store.pads.p03.selection import (
    SelectionError,
    audit_key,
    select_audit_subset,
    select_workload,
    stream_midpoints_ps,
    window_facts,
)
from motionbloom.tremora_store.pads.p03.source_path import (
    SourcePathError,
    read_source_window,
    replay_row_identity_sha256,
)

PARTICIPANTS = ("001", "002", "003")


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("p03")
    release_root = build_release(root, participants=PARTICIPANTS)
    store_root = root / "store"
    p02_materialize(
        release_root=release_root,
        output_root=store_root,
        p01_evidence_sha256="e" * 64,
        expected_samples=0,
    )
    return {
        "root": root,
        "release_root": release_root,
        "store_root": store_root,
        "tables": read_store(store_root),
    }


def _facts(store: dict[str, Any]):
    tables = store["tables"]
    segments = {
        str(row["segment_id"]): row
        for row in tables["pads_segments.parquet"]
    }
    return window_facts(tables["pads_windows.parquet"], segments), tables


# --- selection ------------------------------------------------------------


def test_gap_adjacency_labels_only_real_breaks(store) -> None:
    facts, _ = _facts(store)
    labels = {window.gap_adjacent_status for window in facts}
    assert labels <= {GAP_ADJACENT, INTERIOR}
    # A clean synthetic release has no induced gap, so nothing is adjacent.
    assert labels == {INTERIOR}


def test_gap_adjacency_refuses_a_window_with_no_segment(store) -> None:
    _, tables = _facts(store)
    with pytest.raises(SelectionError):
        window_facts(tables["pads_windows.parquet"], {})


def test_the_workload_takes_one_window_per_eligible_stream(store) -> None:
    facts, tables = _facts(store)
    workload = select_workload(
        facts, stream_midpoints_ps(tables["pads_streams.parquet"])
    )
    chosen = {window.stream_id for window in workload}
    assert len(workload) == len(chosen)
    assert chosen == {window.stream_id for window in facts}
    # Streams with no valid window are simply absent, not invented.
    assert len(chosen) <= len(tables["pads_streams.parquet"])


def test_the_workload_window_is_closest_to_the_stream_midpoint(
    store,
) -> None:
    facts, tables = _facts(store)
    midpoints = stream_midpoints_ps(tables["pads_streams.parquet"])
    workload = {w.stream_id: w for w in select_workload(facts, midpoints)}
    by_stream: dict[str, list] = {}
    for window in facts:
        by_stream.setdefault(window.stream_id, []).append(window)
    for stream_id, chosen in workload.items():
        distances = [
            (abs(w.midpoint_ps - midpoints[stream_id]),
             w.window_start_task_local_ps)
            for w in by_stream[stream_id]
        ]
        assert (
            abs(chosen.midpoint_ps - midpoints[stream_id]),
            chosen.window_start_task_local_ps,
        ) == min(distances)


def test_a_tie_is_broken_by_the_earlier_window(store) -> None:
    facts, tables = _facts(store)
    midpoints = stream_midpoints_ps(tables["pads_streams.parquet"])
    # Force an exact tie by moving the midpoint between two windows.
    stream_id = facts[0].stream_id
    candidates = sorted(
        (w for w in facts if w.stream_id == stream_id),
        key=lambda w: w.window_start_task_local_ps,
    )
    if len(candidates) < 2:
        pytest.skip("stream has a single window")
    tied = (candidates[0].midpoint_ps + candidates[1].midpoint_ps) // 2
    forced = dict(midpoints)
    forced[stream_id] = tied
    chosen = {w.stream_id: w for w in select_workload(facts, forced)}
    assert chosen[stream_id].window_start_task_local_ps <= (
        candidates[1].window_start_task_local_ps
    )


def test_workload_selection_needs_a_stored_extent(store) -> None:
    facts, _ = _facts(store)
    with pytest.raises(SelectionError):
        select_workload(facts, {})


def test_the_audit_subset_is_deterministic_and_capped(store) -> None:
    facts, _ = _facts(store)
    first = select_audit_subset(facts, per_stratum=2)
    second = select_audit_subset(facts, per_stratum=2)
    assert [w.window_id for w in first] == [w.window_id for w in second]
    strata: dict[str, int] = {}
    for window in first:
        strata[window.stratum_id] = strata.get(window.stratum_id, 0) + 1
    assert max(strata.values()) <= 2


def test_a_different_seed_reorders_the_audit_subset(store) -> None:
    facts, _ = _facts(store)
    default = select_audit_subset(facts, per_stratum=1)
    other = select_audit_subset(facts, per_stratum=1, seed=1)
    assert [w.window_id for w in default] != [w.window_id for w in other]


def test_the_audit_key_is_a_pure_function_of_the_frozen_inputs() -> None:
    assert audit_key("w") == audit_key("w")
    assert audit_key("w") != audit_key("x")
    assert audit_key("w") != audit_key("w", seed=1)


def test_the_stratum_key_carries_all_five_fields(store) -> None:
    facts, _ = _facts(store)
    window = facts[0]
    parts = window.stratum_id.split("|")
    assert parts == [
        window.task_name, window.device_location, str(window.outer_fold),
        str(window.sample_count), window.gap_adjacent_status,
    ]


def test_at_least_one_window_per_stratum_is_required(store) -> None:
    facts, _ = _facts(store)
    with pytest.raises(SelectionError):
        select_audit_subset(facts, per_stratum=0)


def test_the_coverage_summary_reports_what_was_frozen(store) -> None:
    facts, _ = _facts(store)
    coverage = selection.selection_coverage(
        select_audit_subset(facts, per_stratum=2)
    )
    assert coverage["audit_selection_seed"] == 20260829
    assert coverage["device_locations"] == ["LeftWrist", "RightWrist"]
    assert len(coverage["tasks"]) == 11
    assert coverage["populated_strata"] > 0


# --- the independent source path -----------------------------------------


def test_the_source_path_returns_the_rows_replay_returns(store) -> None:
    facts, tables = _facts(store)
    index = {
        str(row["stream_id"]): row
        for row in tables["pads_stream_storage_index.parquet"]
    }
    from motionbloom.tremora_store.pads.p02.replay import replay_window

    for window in facts[:6]:
        record = {
            "window_id": window.window_id,
            "stream_id": window.stream_id,
            "first_sample_ordinal": window.first_sample_ordinal,
            "last_sample_ordinal": window.last_sample_ordinal,
            "sample_count": window.sample_count,
        }
        replayed = replay_window(store["store_root"], index, record)
        ordinals = replayed.column("sample_ordinal").to_pylist()
        tokens = replayed.column("source_time_token").to_pylist()
        source = read_source_window(
            release_root=store["release_root"],
            participant_id=window.participant_id,
            task_name=window.task_name,
            device_location=window.device_location,
            stream_id=window.stream_id,
            window_start_task_local_ps=window.window_start_task_local_ps,
            window_end_task_local_ps=window.window_end_task_local_ps,
            expected_asset_sha256=str(
                index[window.stream_id]["source_asset_sha256"]
            ),
        )
        assert list(source.ordinals) == ordinals
        assert list(source.time_tokens) == tokens
        assert source.row_identity_sha256() == replay_row_identity_sha256(
            window.stream_id, ordinals, tokens
        )


def test_the_source_path_refuses_a_hash_that_does_not_match(store) -> None:
    facts, _ = _facts(store)
    window = facts[0]
    with pytest.raises(SourcePathError):
        read_source_window(
            release_root=store["release_root"],
            participant_id=window.participant_id,
            task_name=window.task_name,
            device_location=window.device_location,
            stream_id=window.stream_id,
            window_start_task_local_ps=window.window_start_task_local_ps,
            window_end_task_local_ps=window.window_end_task_local_ps,
            expected_asset_sha256=hashlib.sha256(b"wrong").hexdigest(),
        )


def test_the_source_path_refuses_an_unknown_record(store) -> None:
    facts, _ = _facts(store)
    window = facts[0]
    with pytest.raises(SourcePathError):
        read_source_window(
            release_root=store["release_root"],
            participant_id=window.participant_id,
            task_name="NoSuchTask",
            device_location=window.device_location,
            stream_id=window.stream_id,
            window_start_task_local_ps=0,
            window_end_task_local_ps=4_000_000_000_000,
        )


def test_the_source_path_derives_its_own_origin(store) -> None:
    # It never consults the store: the origin is the file's own first Time.
    facts, _ = _facts(store)
    window = facts[0]
    source = read_source_window(
        release_root=store["release_root"],
        participant_id=window.participant_id,
        task_name=window.task_name,
        device_location=window.device_location,
        stream_id=window.stream_id,
        window_start_task_local_ps=0,
        window_end_task_local_ps=4_000_000_000_000,
    )
    assert source.ordinals[0] == 0
    assert source.times_ps[0] == 0


# --- materialization ------------------------------------------------------


@pytest.fixture(scope="module")
def result(store, tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("p03_out")
    produced = materialize(
        release_root=store["release_root"],
        store_root=store["store_root"],
        output_root=output,
    )
    return produced, output


def test_every_workload_window_is_eligible_and_spectral(result) -> None:
    produced, _ = result
    assert produced.workload_windows_selected > 0
    assert produced.workload_windows_eligible == (
        produced.workload_windows_selected
    )
    assert produced.workload_windows_ineligible == 0
    assert produced.gyro_spectral_rows == produced.workload_windows_eligible
    assert produced.accel_spectral_rows == produced.workload_windows_eligible


def test_source_and_replay_agree_exactly(result) -> None:
    produced, _ = result
    assert produced.audit_windows_selected > 0
    assert produced.source_replay_row_mismatches == 0
    assert produced.source_replay_input_hash_mismatches == 0
    assert produced.source_replay_spectral_hash_mismatches == 0
    assert produced.dominant_frequency_mismatches == 0
    assert produced.maximum_observed_bin_error == 0.0
    assert produced.source_unreadable == 0


def test_the_three_tables_are_written(result) -> None:
    import pyarrow.parquet as pq

    produced, output = result
    workload = pq.read_table(output / "pads_p03_workload_windows.parquet")
    spectra = pq.read_table(output / "pads_p03_spectra.parquet")
    audit = pq.read_table(output / "pads_p03_source_replay_audit.parquet")
    assert workload.num_rows == produced.workload_windows_selected
    assert spectra.num_rows == produced.workload_windows_eligible * len(
        SENSOR_FAMILIES
    )
    assert audit.num_rows == produced.audit_windows_selected
    assert set(
        audit.column("preservation_status").to_pylist()
    ) == {PRESERVED}
    assert set(
        workload.column("spectral_eligibility").to_pylist()
    ) == {SPECTRALLY_ELIGIBLE}


def test_power_vectors_are_thirty_seven_bins(result) -> None:
    import pyarrow.parquet as pq

    _, output = result
    spectra = pq.read_table(output / "pads_p03_spectra.parquet")
    for name in ("aggregate_power", "normalized_aggregate_power"):
        assert spectra.column(name)[0].as_py().__len__() == 37
    normalized = spectra.column("normalized_aggregate_power")[0].as_py()
    assert sum(normalized) == pytest.approx(1.0)


def test_nyquist_is_reported_per_stream_not_from_the_declared_rate(
    result,
) -> None:
    import pyarrow.parquet as pq

    _, output = result
    workload = pq.read_table(output / "pads_p03_workload_windows.parquet")
    nyquist = workload.column("nyquist_hz").to_pylist()
    dt_ref = workload.column("dt_ref_ps").to_pylist()
    for limit, interval in zip(nyquist, dt_ref, strict=True):
        assert limit == pytest.approx(1e12 / (2.0 * interval))
        assert limit > 12.0


def test_materialization_is_deterministic(store, tmp_path: Path) -> None:
    first = materialize(
        release_root=store["release_root"],
        store_root=store["store_root"],
        output_root=tmp_path / "a",
    )
    second = materialize(
        release_root=store["release_root"],
        store_root=store["store_root"],
        output_root=tmp_path / "b",
    )
    assert first.spectral_table_content_sha256 == (
        second.spectral_table_content_sha256
    )
    assert first.as_record() == second.as_record()
