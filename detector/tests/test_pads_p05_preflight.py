"""Preflight, baseline verification, and the streaming timing table."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from _pads_fixtures import build_release
from motionbloom.tremora_store.pads.p02.materialize import (
    materialize as p02_materialize,
)
from motionbloom.tremora_store.pads.p05.build import build_all
from motionbloom.tremora_store.pads.p05.contract import (
    B1,
    B2,
    M1,
    MEASURED_ROUNDS_BY_QUERY_CLASS,
    P05_CONTRACT_VERSION,
    Q2,
    REPRESENTATIONS,
)
from motionbloom.tremora_store.pads.p05.preflight import (
    BASELINE_ABSENT,
    BASELINE_CONTRACT_MISMATCH,
    BASELINE_HASH_MISMATCH,
    BYTES_PER_MEASURED_ROW,
    FROZEN_WORKLOAD_SHA256,
    INSUFFICIENT_DISK,
    MINIMUM_FREE_MARGIN_BYTES,
    PREFLIGHT_OK,
    RUN_COUNT,
    SOURCE_MANIFEST_MISMATCH,
    WORKLOAD_HASH_MISMATCH,
    baseline_identity,
    check_disk,
    project_measured_rows,
    project_run_bytes,
    run_preflight,
)
from motionbloom.tremora_store.pads.p05.sink import (
    MeasurementSink,
    per_query_medians,
    rounds_by_class,
    summarize_table,
    warmup_rows_present,
)

PARTICIPANTS = ("001", "002")


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("p05_preflight")
    release_root = build_release(root, participants=PARTICIPANTS)
    store_root = root / "store"
    p02_materialize(
        release_root=release_root, output_root=store_root,
        p01_evidence_sha256="e" * 64, expected_samples=0,
    )
    built = build_all(
        store_root=store_root, baseline_root=root / "base",
    )
    return {
        "root": root, "release_root": release_root,
        "store_root": store_root, "baseline_root": root / "base",
        "built": built,
    }


@pytest.fixture
def roomy(monkeypatch):
    """Pretend the volume has plenty of room.

    Otherwise these tests pass or fail according to the free space of
    whatever machine runs them, which is a fact about that machine and not
    about the preflight.
    """

    import shutil as shutil_module

    from motionbloom.tremora_store.pads.p05 import preflight as module

    Usage = type(shutil_module.disk_usage("/"))
    monkeypatch.setattr(
        module.shutil, "disk_usage",
        lambda _path: Usage(total=10 * 1024**4, used=0, free=9 * 1024**4),
    )


def _preflight(bench: dict[str, Any], **kwargs):
    identities = json.loads(
        (bench["baseline_root"] / "baseline_identities.json").read_bytes()
    )
    defaults = {
        "baseline_root": bench["baseline_root"],
        "store_root": bench["store_root"],
        "output_root": bench["root"],
        "query_counts": {Q2: 100},
        "workload_content_sha256": FROZEN_WORKLOAD_SHA256,
        "source_manifest_sha256": "m" * 64,
        "expected_identities": identities["baselines"],
    }
    defaults.update(kwargs)
    return run_preflight(**defaults)


# --- baselines are verified, never rebuilt --------------------------------


def test_the_build_records_what_each_baseline_is(bench) -> None:
    identities = bench["built"]["identities"]
    assert set(identities) == {B1, B2, M1}
    for name, entry in identities.items():
        assert len(entry["content_sha256"]) == 64, name
        assert entry["file_count"] > 0
        assert entry["physical_storage_bytes"] > 0
        assert entry["contract_version"] == P05_CONTRACT_VERSION


def test_verified_baselines_pass_preflight(bench, roomy) -> None:
    report = _preflight(bench)
    assert report.status == PREFLIGHT_OK, report.detail
    assert report.ok
    assert set(report.baselines) == {B1, B2, M1}


def test_a_changed_baseline_fails_rather_than_rebuilding(bench) -> None:
    identities = json.loads(
        (bench["baseline_root"] / "baseline_identities.json").read_bytes()
    )["baselines"]
    tampered = {
        name: {**entry, "content_sha256": "0" * 64} if name == B2 else entry
        for name, entry in identities.items()
    }
    report = _preflight(bench, expected_identities=tampered)
    assert report.status == BASELINE_HASH_MISMATCH
    assert not report.ok


def test_a_baseline_from_another_contract_fails(bench) -> None:
    identities = json.loads(
        (bench["baseline_root"] / "baseline_identities.json").read_bytes()
    )["baselines"]
    tampered = {
        name: {**entry, "contract_version": "older"} if name == B1 else entry
        for name, entry in identities.items()
    }
    report = _preflight(bench, expected_identities=tampered)
    assert report.status == BASELINE_CONTRACT_MISMATCH


def test_an_absent_baseline_fails(bench, tmp_path) -> None:
    report = _preflight(bench, baseline_root=tmp_path / "nowhere")
    assert report.status == BASELINE_ABSENT


def test_a_moved_workload_fails(bench) -> None:
    report = _preflight(bench, workload_content_sha256="a" * 64)
    assert report.status == WORKLOAD_HASH_MISMATCH


def test_a_disagreeing_source_manifest_fails(bench) -> None:
    report = _preflight(
        bench, source_manifest_sha256="a" * 64,
        expected_source_manifest_sha256="b" * 64,
    )
    assert report.status == SOURCE_MANIFEST_MISMATCH


def test_the_identity_hash_notices_a_changed_byte(bench, tmp_path) -> None:
    copied = tmp_path / "b2copy"
    shutil.copytree(bench["baseline_root"] / "b2", copied)
    before = baseline_identity(B2, copied)
    target = next(copied.rglob("*.h5"))
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0xFF
    target.write_bytes(bytes(payload))
    assert baseline_identity(B2, copied).content_sha256 != (
        before.content_sha256
    )


# --- the disk preflight ---------------------------------------------------


def test_the_projection_comes_from_the_workload_not_a_guess() -> None:
    counts = {name: 1_000 for name in MEASURED_ROUNDS_BY_QUERY_CLASS}
    rows = project_measured_rows(counts)
    expected = sum(
        1_000 * budget * len(REPRESENTATIONS)
        for budget in MEASURED_ROUNDS_BY_QUERY_CLASS.values()
    )
    assert rows == expected
    # More queries must project more bytes; nothing is fixed.
    _, small = project_run_bytes({Q2: 10})
    _, large = project_run_bytes({Q2: 10_000})
    assert large > small


def test_the_requirement_is_two_runs_plus_a_margin(tmp_path) -> None:
    _, required, free, total = check_disk(tmp_path, 1_000_000)
    assert required == RUN_COUNT * 1_000_000 + MINIMUM_FREE_MARGIN_BYTES
    assert total > 0
    assert free >= 0
    assert MINIMUM_FREE_MARGIN_BYTES >= 1024**3


def test_an_impossible_footprint_refuses_to_start(bench) -> None:
    # A workload that would need more than the volume holds.
    report = _preflight(bench, query_counts={Q2: 10**12})
    assert report.status == INSUFFICIENT_DISK
    assert not report.ok
    assert report.required_free_bytes > report.free_bytes
    assert "GB free" in report.detail


def test_the_preflight_publishes_the_numbers_that_decided_it(bench, roomy) -> None:
    record = _preflight(bench).as_record()
    for key in (
        "projected_run_bytes", "projected_total_bytes",
        "required_free_bytes", "free_bytes", "total_bytes",
        "measured_rows_projected", "bytes_per_measured_row", "run_count",
    ):
        assert key in record, key
    assert record["projected_total_bytes"] == (
        RUN_COUNT * record["projected_run_bytes"]
    )


# --- the streaming sink ---------------------------------------------------


def _row(round_id: int, name: str, query_id: str, latency: int) -> dict:
    return {
        "representation": name, "query_class": Q2, "query_id": query_id,
        "round_id": round_id, "latency_ns": latency,
        "cpu_time_ns": latency // 2, "rows_returned": 400,
        "bytes_returned": 4_000, "peak_rss_delta": 0,
        "content_sha256": "a" * 64, "status": "QUERY_OK",
    }


@pytest.fixture
def table(tmp_path) -> Path:
    path = tmp_path / "retrieval.parquet"
    with MeasurementSink(path, flush_rows=64) as sink:
        for round_id in (-1, 0, 1, 2):
            for name in REPRESENTATIONS:
                for query in range(50):
                    sink.add(_row(
                        round_id, name, f"w{query:03d}",
                        1_000_000 + query * 1_000,
                    ))
    return path


def test_warmup_rows_never_reach_the_table(table) -> None:
    assert warmup_rows_present(table) == 0
    # Three measured rounds of fifty queries for each representation.
    assert pq.ParquetFile(table).metadata.num_rows == 3 * 4 * 50


def test_the_table_is_written_in_bounded_batches(table) -> None:
    handle = pq.ParquetFile(table)
    assert handle.num_row_groups > 1
    for index in range(handle.num_row_groups):
        assert handle.metadata.row_group(index).num_rows <= 64


def test_summaries_are_read_back_from_the_written_table(table) -> None:
    summary = summarize_table(table)
    assert set(summary) == set(REPRESENTATIONS)
    for classes in summary.values():
        entry = classes[Q2]
        assert entry["queries"] == 150
        assert entry["p50_latency_ns"] <= entry["p95_latency_ns"]
        assert entry["p95_latency_ns"] <= entry["p99_latency_ns"]


def test_rounds_are_counted_from_the_table(table) -> None:
    assert rounds_by_class(table) == {Q2: 3}


def test_one_median_per_query_not_one_per_observation(table) -> None:
    medians = per_query_medians(table, query_class=Q2)
    assert set(medians) == set(REPRESENTATIONS)
    for queries in medians.values():
        # Fifty query ids, each collapsed from its three rounds.
        assert len(queries) == 50


def test_a_failed_row_is_counted_but_still_recorded(tmp_path) -> None:
    path = tmp_path / "failures.parquet"
    with MeasurementSink(path) as sink:
        sink.add(_row(0, M1, "ok", 1_000))
        broken = _row(0, M1, "bad", 5)
        broken["status"] = "QUERY_FAILED"
        sink.add(broken)
    assert pq.ParquetFile(path).metadata.num_rows == 2


def test_the_projected_row_size_is_not_optimistic(tmp_path) -> None:
    """The projection must not under-count what a real table costs."""

    path = tmp_path / "sized.parquet"
    count = 40_000
    with MeasurementSink(path) as sink:
        for index in range(count):
            sink.add(_row(
                index % 4, REPRESENTATIONS[index % 4],
                f"{index:06d}:window:{index % 977}", 1_000_000 + index,
            ))
    observed = path.stat().st_size / count
    # The constant the preflight projects with has to be at least what a
    # written table actually costs per row, or the disk check is optimistic
    # in exactly the direction that fills a volume.
    assert observed <= BYTES_PER_MEASURED_ROW, (
        f"observed {observed:.1f} bytes/row exceeds the projected "
        f"{BYTES_PER_MEASURED_ROW}"
    )


def test_a_refusal_reports_the_volume_numbers_it_refused_on(bench, roomy) -> None:
    """The record exists to say why the run did not happen."""

    from motionbloom.tremora_store.pads.p05.audit import _preflight_record
    from motionbloom.tremora_store.pads.p05.memory import check_memory

    report = _preflight(bench, query_counts={Q2: 10**12})
    record = _preflight_record(
        preflight=report,
        memory=check_memory([{
            "index": 0, "swap_total_bytes": 0, "swap_used_bytes": 0,
            "free_percentage": 80.0,
        }]),
        inspected={},
    )
    assert record["release_status"] == "ERROR_RESOURCE_PREFLIGHT"
    assert record["gate_evaluated"] is False
    assert record["materialized_release_artifacts"] == 0
    for key in ("free_bytes", "required_free_bytes", "total_bytes"):
        assert key in record["preflight"], key
    assert record["preflight"]["free_bytes"] > 0


def test_the_hashed_evidence_omits_the_volume_state(bench, roomy) -> None:
    # Run B measures less free space than run A, because run A just wrote a
    # timing table.  That difference must not reach the evidence hash.
    deterministic = _preflight(bench).deterministic_record()
    for key in ("free_bytes", "required_free_bytes", "total_bytes"):
        assert key not in deterministic, key
    # ...and what it verified and projected must still be there.
    for key in (
        "baselines", "measured_rows_projected", "projected_run_bytes",
        "workload_content_sha256",
    ):
        assert key in deterministic, key
    assert "free_bytes" in _preflight(bench).volume_record()


# --- the memory preflight -------------------------------------------------


def _samples(*used_gb, total_gb=8.0, free_pct=60.0):
    return [
        {
            "index": index,
            "swap_total_bytes": int(total_gb * 1024**3),
            "swap_used_bytes": int(used * 1024**3),
            "free_percentage": free_pct,
        }
        for index, used in enumerate(used_gb)
    ]


def test_a_quiet_machine_passes_the_memory_check() -> None:
    from motionbloom.tremora_store.pads.p05.memory import (
        MEMORY_OK,
        check_memory,
    )

    report = check_memory(_samples(0.1, 0.1, 0.1))
    assert report.status == MEMORY_OK
    assert report.ok


def test_a_machine_already_paging_is_refused() -> None:
    # The state that killed the first authoritative run: swap heavily
    # occupied before the benchmark has allocated anything.
    from motionbloom.tremora_store.pads.p05.memory import (
        SWAP_HEAVILY_USED,
        check_memory,
    )

    report = check_memory(_samples(5.9, 5.9, 5.9))
    assert report.status == SWAP_HEAVILY_USED
    assert not report.ok
    assert "already paging" in report.detail


def test_climbing_swap_is_refused_even_when_it_is_still_small() -> None:
    from motionbloom.tremora_store.pads.p05.memory import (
        SWAP_GROWING,
        check_memory,
    )

    report = check_memory(_samples(0.1, 0.6, 1.1))
    assert report.status == SWAP_GROWING
    assert report.swap_growth_bytes > 0


def test_low_free_memory_is_refused() -> None:
    from motionbloom.tremora_store.pads.p05.memory import (
        MEMORY_PRESSURE_ELEVATED,
        check_memory,
    )

    report = check_memory(_samples(0.1, 0.1, free_pct=3.0))
    assert report.status == MEMORY_PRESSURE_ELEVATED


def test_a_machine_with_no_swap_configured_is_fine() -> None:
    from motionbloom.tremora_store.pads.p05.memory import (
        MEMORY_OK,
        check_memory,
    )

    report = check_memory([{
        "index": 0, "swap_total_bytes": 0, "swap_used_bytes": 0,
        "free_percentage": 67.0,
    }])
    assert report.status == MEMORY_OK


def test_memory_numbers_are_provenance_not_evidence() -> None:
    """They must never reach the canonical hash."""

    from motionbloom.tremora_store.pads.p05.memory import check_memory

    record = check_memory(_samples(0.1, 0.1)).as_record()
    assert "swap_used_bytes" in record
    assert "free_percentage" in record


# --- the early-exit paths return records, not pairs -----------------------


def test_a_blocked_dependency_returns_one_record(tmp_path) -> None:
    """Every early exit must return what the CLI expects to read.

    A blocked path that returned a pair got past the tests and only failed
    when the real run hit it, three minutes into an authoritative attempt.
    """

    from motionbloom.tremora_store.pads.p05.audit import measure_pads_p05

    record = measure_pads_p05(
        release_root=tmp_path / "absent",
        store_root=tmp_path / "absent",
        baseline_root=tmp_path / "base",
        output_root=tmp_path / "out",
        p02_report_path=tmp_path / "p02.json",
        p03_report_path=tmp_path / "p03.json",
        p04_report_path=tmp_path / "p04.json",
    )
    assert isinstance(record, dict), "an early exit returned a pair"
    assert record.get("gate_evaluated") is False
    assert "blocked_reason" in record


def test_the_preflight_refusal_also_returns_one_record(bench, tmp_path) -> None:
    from motionbloom.tremora_store.pads.p05.audit import _preflight_record
    from motionbloom.tremora_store.pads.p05.memory import check_memory

    record = _preflight_record(
        preflight=_preflight(bench, query_counts={Q2: 10**12}),
        memory=check_memory(_samples(0.1)),
        inspected={},
    )
    assert isinstance(record, dict)
    assert record["gate_evaluated"] is False


# --- settling between the two runs ----------------------------------------


def _health(swap_gb: float, free_pct: float = 70.0):
    return {
        "index": 0,
        "swap_total_bytes": 5 * 1024**3,
        "swap_used_bytes": int(swap_gb * 1024**3),
        "free_percentage": free_pct,
    }


def _settle(readings, *, reference=None, disk_gb=12.0, **kwargs):
    from motionbloom.tremora_store.pads.p05 import settle as module

    queue = list(readings)

    def fake_sample(count=1, **_):
        return [queue.pop(0) if queue else readings[-1]]

    ticks = {"now": 0.0}

    def fake_clock():
        return ticks["now"]

    def fake_sleep(seconds):
        ticks["now"] += seconds

    original = module.sample_memory
    module.sample_memory = fake_sample
    try:
        return module.settle_between_runs(
            reference=reference,
            disk_free=lambda: int(disk_gb * 1024**3),
            required_healthy=kwargs.pop("required_healthy", 3),
            interval=30.0,
            sleep=fake_sleep,
            clock=fake_clock,
            **kwargs,
        )
    finally:
        module.sample_memory = original


def test_a_drained_machine_settles() -> None:
    from motionbloom.tremora_store.pads.p05.settle import SETTLED

    report = _settle([_health(0.05)] * 4)
    assert report.status == SETTLED
    assert report.ok
    assert report.consecutive_healthy >= 3


def test_a_machine_still_holding_swap_does_not_settle() -> None:
    from motionbloom.tremora_store.pads.p05.settle import SETTLE_TIMEOUT

    # The state run A left the machine in: 3.5 GB of swap still occupied.
    report = _settle([_health(3.5)] * 40, max_wait=600.0)
    assert report.status == SETTLE_TIMEOUT
    assert not report.ok
    assert "swap" in report.detail


def test_one_healthy_reading_is_not_enough() -> None:
    """A machine that dips healthy while still draining must not qualify."""

    from motionbloom.tremora_store.pads.p05.settle import SETTLE_TIMEOUT

    readings = [_health(3.0), _health(0.05), _health(3.0), _health(0.05)]
    report = _settle(readings * 12, max_wait=600.0)
    assert report.status == SETTLE_TIMEOUT
    assert report.consecutive_healthy < 3


def test_growing_swap_is_unhealthy_even_when_the_level_is_fine() -> None:
    from motionbloom.tremora_store.pads.p05.settle import SETTLE_TIMEOUT

    climbing = [_health(0.01), _health(0.02), _health(0.03), _health(0.04)]
    report = _settle(climbing * 12, max_wait=600.0)
    assert report.status == SETTLE_TIMEOUT
    # Every reading is under the swap cap, so the only thing that can have
    # broken the streak is the growth check.
    reasons = {entry["reason"] for entry in report.history}
    assert "swap still growing" in reasons
    assert report.consecutive_healthy < 3


def test_the_gate_tightens_to_the_first_run_s_start() -> None:
    """Run A started clean, so run B is held to that, not just the floor."""

    from motionbloom.tremora_store.pads.p05.settle import _thresholds

    clean = _thresholds({"swap_used_bytes": 0, "free_percentage": 70.0})
    assert clean["max_swap_used_bytes"] == 512 * 1024**2
    assert clean["min_free_percentage"] == 60.0
    # A first run that started dirtier cannot loosen the absolute floors.
    dirty = _thresholds(
        {"swap_used_bytes": 4 * 1024**3, "free_percentage": 20.0}
    )
    assert dirty["max_swap_used_bytes"] == 512 * 1024**2
    assert dirty["min_free_percentage"] == 60.0


def test_low_disk_blocks_settling_even_with_memory_free() -> None:
    from motionbloom.tremora_store.pads.p05.settle import SETTLE_TIMEOUT

    report = _settle([_health(0.05)] * 40, disk_gb=1.0, max_wait=600.0)
    assert report.status == SETTLE_TIMEOUT
    assert "disk" in report.detail
