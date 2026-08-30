"""Representation equivalence and storage accounting."""

from __future__ import annotations

from typing import Any

import pytest
from _pads_fixtures import build_release
from motionbloom.tremora_store.pads.p02.materialize import (
    materialize as p02_materialize,
)
from motionbloom.tremora_store.pads.p05.build import build_b1, build_b2
from motionbloom.tremora_store.pads.p05.contract import (
    B0,
    B1,
    B2,
    M1,
    REPRESENTATIONS,
)
from motionbloom.tremora_store.pads.p05.equivalence import (
    EQUIVALENT,
    NOT_EQUIVALENT,
    REFERENCE,
    compare_all,
)
from motionbloom.tremora_store.pads.p05.representations import (
    DuplicatedWindowRepresentation,
    Hdf5RangeIndexedRepresentation,
    SourceTextRepresentation,
    TremoraParquetRepresentation,
)
from motionbloom.tremora_store.pads.p05.storage import (
    StorageAccount,
    StorageAccountingError,
    account_b0,
    account_b1,
    account_b2,
    account_m1,
    derived_store_account,
    reconcile,
    storage_tables,
)

PARTICIPANTS = ("001", "002", "003")


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("p05_equivalence")
    release_root = build_release(root, participants=PARTICIPANTS)
    store_root = root / "store"
    p02_materialize(
        release_root=release_root, output_root=store_root,
        p01_evidence_sha256="e" * 64, expected_samples=0,
    )
    b1 = build_b1(store_root=store_root, output_root=root / "b1")
    b2 = build_b2(store_root=store_root, output_root=root / "b2")
    reps = {
        B0: SourceTextRepresentation(release_root, store_root),
        B1: DuplicatedWindowRepresentation(root / "b1"),
        B2: Hdf5RangeIndexedRepresentation(root / "b2"),
        M1: TremoraParquetRepresentation(store_root),
    }
    for representation in reps.values():
        representation.open()
    yield {
        "root": root, "release_root": release_root, "store_root": store_root,
        "reps": reps, "b1": b1, "b2": b2,
    }
    for representation in reps.values():
        representation.close()


@pytest.fixture(scope="module")
def report(bench: dict[str, Any]):
    reps = bench["reps"]
    return compare_all(
        reps,
        window_ids=list(reps[M1].windows),
        stream_ids=list(reps[M1].index),
        assessment_ids=list(reps[M1].assessments),
    )


# --- equivalence ----------------------------------------------------------


def test_the_reference_is_the_published_source_not_the_system_under_test(
) -> None:
    # Agreement means agreement with the release, not with TremoraStore.
    assert REFERENCE == REPRESENTATIONS[0] == B0
    assert REFERENCE != M1


def test_every_representation_agrees_on_everything(report) -> None:
    record = report.as_record()
    assert record["status"] == EQUIVALENT
    assert record["content_mismatches"] == 0
    assert record["row_count_mismatches"] == 0
    assert record["time_mismatches"] == 0
    assert record["sensor_value_mismatches"] == 0
    assert record["failed_queries"] == 0


def test_the_comparison_actually_covered_all_three_query_classes(
    report, bench
) -> None:
    reps = bench["reps"]
    assert report.windows_compared == len(reps[M1].windows) > 0
    assert report.streams_compared == len(reps[M1].index) > 0
    assert report.assessments_compared == len(reps[M1].assessments) > 0
    # Three baselines against the reference, for every identifier.
    expected = 3 * (
        report.windows_compared + report.streams_compared
        + report.assessments_compared
    )
    assert report.comparisons == expected
    assert report.rows_reconciled > 0


def test_every_representation_answered_every_query(report) -> None:
    per = report.as_record()["per_representation"]
    assert set(per) == set(REPRESENTATIONS)
    counts = {name: entry["queries"] for name, entry in per.items()}
    assert len(set(counts.values())) == 1, counts
    for entry in per.values():
        assert entry["failed_queries"] == 0


def test_a_disagreeing_representation_is_caught_and_named(bench) -> None:
    reps = dict(bench["reps"])

    class Wrong(TremoraParquetRepresentation):
        def window_rows(self, window_id: str):
            rows = super().window_rows(window_id)
            if rows:
                rows[0] = dict(rows[0])
                rows[0]["gyroscope_z"] = rows[0]["gyroscope_z"] + 1e-9
            return rows

    broken = Wrong(bench["store_root"])
    broken.open()
    reps[M1] = broken
    report = compare_all(
        reps,
        window_ids=list(broken.windows)[:5],
        stream_ids=[], assessment_ids=[],
    )
    record = report.as_record()
    assert record["status"] == NOT_EQUIVALENT
    assert record["content_mismatches"] == 5
    assert record["sensor_value_mismatches"] == 5
    # And the time half is untouched, so the failure is localized.
    assert record["time_mismatches"] == 0
    assert record["per_representation"][M1]["content_mismatches"] == 5
    assert record["per_representation"][B2]["content_mismatches"] == 0


def test_a_failing_representation_is_recorded_not_silently_skipped(
    bench,
) -> None:
    reps = dict(bench["reps"])

    class Broken(TremoraParquetRepresentation):
        def window_rows(self, window_id: str):
            raise RuntimeError("no")

    broken = Broken(bench["store_root"])
    broken.open()
    reps[M1] = broken
    report = compare_all(
        reps, window_ids=list(broken.windows)[:3],
        stream_ids=[], assessment_ids=[],
    )
    record = report.as_record()
    assert record["failed_queries"] == 3
    assert record["status"] == NOT_EQUIVALENT
    assert record["failure_count"] >= 3


def test_the_window_hash_is_stable_and_content_sensitive(
    report, bench
) -> None:
    reps = bench["reps"]
    again = compare_all(
        reps, window_ids=list(reps[M1].windows),
        stream_ids=[], assessment_ids=[],
    )
    assert again.window_content_sha256 == report.window_content_sha256
    fewer = compare_all(
        reps, window_ids=list(reps[M1].windows)[:-1],
        stream_ids=[], assessment_ids=[],
    )
    assert fewer.window_content_sha256 != report.window_content_sha256


# --- storage accounting ---------------------------------------------------


@pytest.fixture(scope="module")
def accounts(bench: dict[str, Any]) -> dict[str, StorageAccount]:
    return {
        B0: account_b0(bench["release_root"], bench["store_root"]),
        B1: account_b1(bench["root"] / "b1", bench["b1"].as_record()),
        B2: account_b2(bench["root"] / "b2", bench["b2"].as_record()),
        M1: account_m1(bench["store_root"]),
    }


def test_the_sample_counts_are_read_rather_than_assumed(accounts) -> None:
    # Every representation reports the corpus it was actually given, not the
    # frozen constant for the corpus the code was written against.
    counts = {name: account.unique_samples for name, account in accounts.items()}
    assert len(set(counts.values())) == 1, counts
    from motionbloom.tremora_store.pads.p05.contract import SOURCE_SAMPLES

    assert accounts[B0].unique_samples != SOURCE_SAMPLES


def test_bytes_are_measured_from_disk_not_reported_by_the_writer(
    accounts, bench
) -> None:
    for name, account in accounts.items():
        assert account.physical_storage_bytes > 0, name
        assert account.file_count > 0, name
    # B0's device files live under movement/timeseries, not beside the
    # observations; counting only the JSON would have called 6 MB the corpus.
    assert accounts[B0].source_payload_bytes > accounts[B0].metadata_bytes


def test_only_the_duplicating_baseline_duplicates(accounts) -> None:
    assert accounts[B1].duplication_factor > 1.0
    assert accounts[B1].duplicate_sample_instances > 0
    for name in (B0, B2, M1):
        assert accounts[name].duplication_factor == 1.0
        assert accounts[name].duplicate_sample_instances == 0


def test_index_and_payload_are_reported_separately(accounts) -> None:
    # "TremoraStore is bigger" and "TremoraStore carries indexes the
    # baselines do not" are different statements; the reader gets both.
    for name, account in accounts.items():
        record = account.as_record(original_source_bytes=1_000)
        assert record["index_bytes"] >= 0, name
        assert record["metadata_bytes"] >= 0, name
        assert record["source_payload_bytes"] > 0, name
    assert accounts[M1].index_bytes > 0


def test_every_published_storage_metric_is_present(accounts) -> None:
    from motionbloom.tremora_store.pads.p05.contract import STORAGE_METRICS

    record = accounts[M1].as_record(original_source_bytes=1_000)
    for metric in STORAGE_METRICS:
        assert metric in record, metric


def test_the_accounting_refuses_a_release_with_no_device_files(
    tmp_path,
) -> None:
    empty = tmp_path / "release"
    (empty / "movement").mkdir(parents=True)
    with pytest.raises(StorageAccountingError):
        account_b0(empty, tmp_path)


def test_the_reconciliation_catches_a_miscounted_baseline(
    accounts,
) -> None:
    tampered = dict(accounts)
    broken = StorageAccount(
        representation=B1,
        physical_storage_bytes=1,
        unique_samples=accounts[B1].unique_samples,
        stored_sample_instances=accounts[B1].unique_samples,
    )
    tampered[B1] = broken
    verdict = reconcile(
        tampered,
        expected_unique_samples=accounts[B0].unique_samples,
        expected_window_instances=(
            accounts[B1].stored_sample_instances - accounts[B0].unique_samples
        ),
    )
    assert verdict["reconciled"] is False
    assert any("instances" in problem for problem in verdict["problems"])


def test_the_derived_stores_stay_out_of_the_primary_comparison(
    accounts, bench, tmp_path
) -> None:
    tables = storage_tables(
        accounts=accounts, original_source_bytes=1_000_000,
        p04_root=tmp_path,
        expected_unique_samples=accounts[B0].unique_samples,
        expected_window_instances=(
            accounts[B1].stored_sample_instances - accounts[B0].unique_samples
        ),
    )
    assert tables["derived_stores"]["included_in_primary_comparison"] is False
    # Their samples never enter any representation's duplication maths.
    for record in tables["accounts"]:
        assert record["unique_samples"] == accounts[B0].unique_samples
    assert derived_store_account(tmp_path)["derived_store_bytes"] == 0


def test_the_storage_tables_list_every_representation(accounts) -> None:
    tables = storage_tables(
        accounts=accounts, original_source_bytes=1_000_000,
        expected_unique_samples=accounts[B0].unique_samples,
        expected_window_instances=(
            accounts[B1].stored_sample_instances - accounts[B0].unique_samples
        ),
    )
    assert [record["representation"] for record in tables["accounts"]] == [
        B0, B1, B2, M1
    ]
    assert tables["reconciliation"]["reconciled"] is True
