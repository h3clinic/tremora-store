"""The four representations, and whether any of them was built unfairly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _pads_fixtures import build_release
from motionbloom.tremora_store.pads.p02.materialize import (
    materialize as p02_materialize,
)
from motionbloom.tremora_store.pads.p05.build import (
    build_b1,
    build_b2,
)
from motionbloom.tremora_store.pads.p05.contract import (
    COMPRESSION_CODEC,
    COMPRESSION_LEVEL,
    HDF5_REQUIRED_INDEXES,
    Q1,
    Q2,
    Q3,
    Q4,
)
from motionbloom.tremora_store.pads.p05.representations import (
    DuplicatedWindowRepresentation,
    Hdf5RangeIndexedRepresentation,
    RepresentationError,
    SourceTextRepresentation,
    TremoraParquetRepresentation,
    _token_picoseconds,
)
from motionbloom.tremora_store.pads.p05.rows import (
    SENSOR_ORDER,
    compare,
    result_from_rows,
)

PARTICIPANTS = ("001", "002", "003")


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("p05_baselines")
    release_root = build_release(root, participants=PARTICIPANTS)
    store_root = root / "store"
    p02_materialize(
        release_root=release_root, output_root=store_root,
        p01_evidence_sha256="e" * 64, expected_samples=0,
    )
    b1 = build_b1(store_root=store_root, output_root=root / "b1")
    b2 = build_b2(store_root=store_root, output_root=root / "b2")
    return {
        "root": root, "release_root": release_root, "store_root": store_root,
        "b1_root": root / "b1", "b2_root": root / "b2",
        "b1_report": b1, "b2_report": b2,
    }


@pytest.fixture(scope="module")
def reps(bench: dict[str, Any]) -> dict[str, Any]:
    built = {
        "B0": SourceTextRepresentation(
            bench["release_root"], bench["store_root"]
        ),
        "B1": DuplicatedWindowRepresentation(bench["b1_root"]),
        "B2": Hdf5RangeIndexedRepresentation(bench["b2_root"]),
        "M1": TremoraParquetRepresentation(bench["store_root"]),
    }
    for representation in built.values():
        representation.open()
    yield built
    for representation in built.values():
        representation.close()


# --- equivalence ----------------------------------------------------------


def test_every_representation_returns_the_same_windows(reps) -> None:
    reference = reps["M1"]
    for window_id in reference.windows:
        expected = result_from_rows(
            window_id, reference.window_rows(window_id)
        )
        for name in ("B0", "B1", "B2"):
            got = result_from_rows(
                window_id, reps[name].window_rows(window_id)
            )
            assert all(compare(expected, got).values()), (name, window_id)


def test_every_representation_replays_the_same_streams(reps) -> None:
    reference = reps["M1"]
    for stream_id in reference.index:
        expected = result_from_rows(
            stream_id, reference.stream_rows(stream_id)
        )
        for name in ("B0", "B1", "B2"):
            got = result_from_rows(
                stream_id, reps[name].stream_rows(stream_id)
            )
            assert all(compare(expected, got).values()), (name, stream_id)


def test_every_representation_returns_the_same_assessments(reps) -> None:
    reference = reps["M1"]
    for assessment_id in reference.assessments:
        expected = result_from_rows(
            assessment_id, reference.assessment_rows(assessment_id)
        )
        for name in ("B0", "B1", "B2"):
            got = result_from_rows(
                assessment_id, reps[name].assessment_rows(assessment_id)
            )
            assert all(compare(expected, got).values()), name


def test_a_batch_returns_its_windows_concatenated(reps) -> None:
    window_ids = list(reps["M1"].windows)[:4]
    for name, representation in reps.items():
        batched = representation.batch_rows(window_ids)
        single = [
            row for window_id in window_ids
            for row in representation.window_rows(window_id)
        ]
        assert batched == single, name


def test_the_query_dispatch_covers_every_class(reps) -> None:
    representation = reps["M1"]
    window_ids = list(representation.windows)[:3]
    assert representation.query(Q1, next(iter(representation.index))).rows > 0
    assert representation.query(Q2, window_ids[0]).rows > 0
    assert representation.query(
        Q3, next(iter(representation.assessments))
    ).rows > 0
    assert representation.query(
        Q4, "batch:3:0000", window_ids=window_ids
    ).rows > 0


# --- fairness -------------------------------------------------------------


def test_b1_physically_duplicates_its_overlapping_windows(
    bench: dict[str, Any],
) -> None:
    report = bench["b1_report"].as_record()
    # Not a claim about the contract's numbers: what this build wrote.
    assert report["duplicate_sample_instances"] > 0
    assert report["duplication_factor"] > 1.0
    assert report["stored_sample_instances"] > report["unique_samples"]


def test_m1_and_b2_store_each_sample_once(
    bench: dict[str, Any], reps
) -> None:
    assert bench["b2_report"].as_record()["duplication_factor"] == 1.0
    # M1's window index carries ordinal ranges, never sample copies.
    window = next(iter(reps["M1"].windows.values()))
    assert "first_sample_ordinal" in window
    assert not any(
        sensor in window for sensor in SENSOR_ORDER
    )


def test_hdf5_was_given_the_indexes_it_was_promised(
    bench: dict[str, Any],
) -> None:
    import h5py
    import hdf5plugin  # noqa: F401

    path = bench["b2_root"] / Hdf5RangeIndexedRepresentation.FILENAME
    with h5py.File(path, "r") as handle:
        for name in HDF5_REQUIRED_INDEXES:
            assert name in handle, name
        assert json.loads(handle.attrs["indexes"]) == list(
            HDF5_REQUIRED_INDEXES
        )
        assert handle.attrs["compression_codec"] == COMPRESSION_CODEC
        assert handle.attrs["compression_level"] == COMPRESSION_LEVEL
        # Chunked, so a window is a slice and not a whole-file scan.
        assert handle["samples"]["gyroscope_z"].chunks is not None


def test_a_window_read_does_not_scan_the_whole_hdf5_file(reps) -> None:
    representation = reps["B2"]
    window_id = next(iter(representation.window_offsets))
    _, start, stop = representation.window_offsets[window_id]
    total = representation._columns["source_time_ps"].shape[0]
    # The offset index resolves the window to a bounded slice.
    assert 0 <= start < stop <= total
    assert stop - start == len(representation.window_rows(window_id))


def test_m1_reads_the_same_columns_the_baselines_were_built_with() -> None:
    from motionbloom.tremora_store.pads.p05.build import CARRIED_COLUMNS

    # Charging M1 for the eleven provenance columns no baseline carries
    # would be crippling it the way the contract forbids crippling HDF5.
    assert set(TremoraParquetRepresentation.PROJECTION) == set(
        CARRIED_COLUMNS
    )


def test_compression_is_matched_between_the_built_baselines(
    bench: dict[str, Any],
) -> None:
    for key in ("b1_report", "b2_report"):
        compression = bench[key].as_record()["compression"]
        assert compression["codec"] == COMPRESSION_CODEC
        assert compression["level"] == COMPRESSION_LEVEL


def test_no_representation_caches_a_query_result(reps) -> None:
    # Repeating a query must recompute it: a private result cache would make
    # the comparison meaningless.
    for name, representation in reps.items():
        window_id = next(iter(reps["M1"].windows))
        first = representation.window_rows(window_id)
        second = representation.window_rows(window_id)
        assert first == second, name
        assert first is not second, name


# --- the source path ------------------------------------------------------


def test_the_time_token_is_converted_without_a_float(bench) -> None:
    # Ten decimal places of seconds do not fit in a double, so the source
    # path does integer arithmetic on the digits.
    assert _token_picoseconds("4.0082712173") == 4_008_271_217_300
    assert _token_picoseconds("0.0000000000") == 0
    assert _token_picoseconds("10.5") == 10_500_000_000_000
    assert _token_picoseconds("-1.25") == -1_250_000_000_000


def test_an_unknown_identifier_is_refused_not_answered(reps) -> None:
    for name in ("B1", "B2"):
        with pytest.raises(RepresentationError):
            reps[name].window_rows("no-such-window")
    with pytest.raises((RepresentationError, KeyError)):
        reps["B0"].window_rows("no-such-window")
    with pytest.raises((RepresentationError, KeyError)):
        reps["M1"].window_rows("no-such-window")


def test_the_builders_report_what_they_actually_wrote(
    bench: dict[str, Any],
) -> None:
    for key, root in (("b1_report", "b1_root"), ("b2_report", "b2_root")):
        record = bench[key].as_record()
        assert record["physical_storage_bytes"] > 0
        assert record["file_count"] >= 1
        assert record["streams"] > 0
        assert record["windows"] > 0
        assert Path(bench[root]).exists()
