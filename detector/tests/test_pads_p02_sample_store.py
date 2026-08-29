from __future__ import annotations

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
from motionbloom.tremora_store.pads.p02 import stream_reader
from motionbloom.tremora_store.pads.p02.sample_store import (
    SampleStoreError,
    SampleStoreWriter,
    part_relative_path,
    read_stream_row_group,
)
from motionbloom.tremora_store.pads.p02.stream_reader import read_stream

RATE = Fraction(100)
EVIDENCE = "e" * 64


def _declaration(
    *, channels=None, device_location="LeftWrist"
) -> StreamDeclaration:
    names = tuple(channels or CANONICAL_CHANNELS)
    return StreamDeclaration(
        device_location=device_location,
        channels=names,
        units=tuple(CHANNEL_UNITS.get(name, "?") for name in names),
        file_name=f"timeseries/001_Relaxed_{device_location}.txt",
    )


def _read(payload: bytes, *, rows: int, declaration=None, stream_id="001:Relaxed:LeftWrist"):
    return read_stream(
        payload,
        declaration=declaration or _declaration(),
        declared_rows=rows,
        sampling_rate=RATE,
        stream_id=stream_id,
        participant_id="001",
        assessment_id="001:Relaxed",
        task_name="Relaxed",
        source_asset_sha256=hashlib.sha256(payload).hexdigest(),
    )


# --- reading --------------------------------------------------------------


def test_a_clean_stream_reads_into_exact_picoseconds() -> None:
    payload = timeseries_bytes(64)
    samples = _read(payload, rows=64)
    assert samples.ok
    assert samples.sample_count == 64
    assert samples.source_time_origin_ps == 0
    assert samples.source_time_ps[1] == 9_994_600_000
    assert samples.task_local_time_ps[0] == 0


def test_task_local_time_is_measured_from_the_stored_origin() -> None:
    payload = timeseries_bytes(16)
    shifted = b"\n".join(
        line.replace(line.split(b",")[0], b"%.10f" % (
            float(line.split(b",")[0]) + 5.0
        ), 1)
        for line in payload.rstrip(b"\n").split(b"\n")
    ) + b"\n"
    samples = _read(shifted, rows=16)
    assert samples.ok
    assert samples.source_time_origin_ps == 5_000_000_000_000
    assert samples.task_local_time_ps[0] == 0
    assert samples.source_time_ps[0] == 5_000_000_000_000


def test_replay_rebuilds_the_source_bytes_exactly() -> None:
    payload = timeseries_bytes(128)
    samples = _read(payload, rows=128)
    assert samples.replay_source_bytes() == payload
    assert hashlib.sha256(
        samples.replay_source_bytes()
    ).hexdigest() == samples.source_asset_sha256


def test_a_permuted_declaration_still_replays_in_source_order() -> None:
    permuted = (
        "Gyroscope_X", "Time", "Accelerometer_X", "Accelerometer_Y",
        "Accelerometer_Z", "Gyroscope_Y", "Gyroscope_Z",
    )
    rows = [
        f"0.5000000000,{index * 0.0099946:.10f},0.1000000000,0.2000000000,"
        f"0.3000000000,0.4000000000,0.5000000000"
        for index in range(32)
    ]
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    samples = _read(payload, rows=32, declaration=_declaration(channels=permuted))
    assert samples.ok
    assert samples.canonicalization_permutation[0] == 1
    assert samples.source_time_ps[1] == 9_994_600_000
    assert samples.channel("Gyroscope_X")[0] == 0.5
    assert samples.replay_source_bytes() == payload


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        (timeseries_bytes(8, blank_row_at=3), stream_reader.STREAM_BLANK_ROW),
        (timeseries_bytes(8, columns=6),
         stream_reader.STREAM_COLUMN_COUNT_MISMATCH),
        (timeseries_bytes(7), stream_reader.STREAM_ROW_COUNT_MISMATCH),
        (timeseries_bytes(8, time_override={2: "NaN"}),
         stream_reader.STREAM_TIME_NOT_EXACT),
        (timeseries_bytes(8, time_override={2: "0.00990295410001"}),
         stream_reader.STREAM_TIME_NOT_EXACT),
        (timeseries_bytes(8, value_override="0.001"),
         stream_reader.STREAM_VALUE_DOES_NOT_ROUND_TRIP),
    ),
)
def test_a_stream_that_cannot_be_stored_exactly_is_refused(
    payload: bytes, expected: str
) -> None:
    samples = _read(payload, rows=8)
    assert samples.stream_status == expected
    assert samples.ok is False


def test_carriage_returns_are_refused_rather_than_stripped() -> None:
    payload = timeseries_bytes(8).replace(b"\n", b"\r\n")
    samples = _read(payload, rows=8)
    assert samples.stream_status == (
        stream_reader.STREAM_UNEXPECTED_LINE_TERMINATOR
    )


def test_an_ambiguous_declaration_is_refused_by_the_p01_contract() -> None:
    bad = StreamDeclaration(
        device_location="Ankle",
        channels=CANONICAL_CHANNELS,
        units=tuple(CHANNEL_UNITS[name] for name in CANONICAL_CHANNELS),
        file_name="timeseries/001_Relaxed_Ankle.txt",
    )
    samples = _read(timeseries_bytes(8), rows=8, declaration=bad)
    assert samples.stream_status == stream_reader.STREAM_DECLARATION_REFUSED


# --- content hash ---------------------------------------------------------


def test_the_content_hash_is_stable_and_sensitive() -> None:
    first = _read(timeseries_bytes(32), rows=32)
    second = _read(timeseries_bytes(32), rows=32)
    assert first.content_sha256() == second.content_sha256()
    other = _read(
        timeseries_bytes(32), rows=32, stream_id="001:Relaxed:RightWrist"
    )
    assert other.content_sha256() != first.content_sha256()
    changed = _read(
        timeseries_bytes(32, time_override={5: "9.9999999999"}), rows=32
    )
    assert changed.content_sha256() != first.content_sha256()


# --- packing --------------------------------------------------------------


def _streams(count: int):
    for index in range(count):
        payload = timeseries_bytes(16)
        yield read_stream(
            payload,
            declaration=_declaration(),
            declared_rows=16,
            sampling_rate=RATE,
            stream_id=f"S{index:04d}",
            participant_id=f"P{index:03d}",
            assessment_id=f"P{index:03d}:Relaxed",
            task_name="Relaxed",
            source_asset_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_streams_must_be_packed_in_ascending_identifier_order(
    tmp_path: Path,
) -> None:
    streams = list(_streams(2))
    with SampleStoreWriter(
        tmp_path, p01_evidence_sha256=EVIDENCE
    ) as writer:
        writer.add(streams[1])
        with pytest.raises(SampleStoreError):
            writer.add(streams[0])


def test_each_stream_becomes_exactly_one_row_group(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    entries = []
    with SampleStoreWriter(
        tmp_path, p01_evidence_sha256=EVIDENCE, streams_per_part=3
    ) as writer:
        for samples in _streams(7):
            entries.append(writer.add(samples))

    parts = sorted((tmp_path / "samples").glob("*.parquet"))
    assert [path.name for path in parts] == [
        "part-00000.parquet", "part-00001.parquet", "part-00002.parquet",
    ]
    assert pq.ParquetFile(parts[0]).num_row_groups == 3
    assert pq.ParquetFile(parts[2]).num_row_groups == 1
    assert [entry.row_group_index for entry in entries] == [
        0, 1, 2, 0, 1, 2, 0
    ]
    assert entries[3].parquet_relative_path == part_relative_path(1)


def test_a_stored_stream_reads_back_identically(tmp_path: Path) -> None:
    with SampleStoreWriter(
        tmp_path, p01_evidence_sha256=EVIDENCE
    ) as writer:
        samples = next(iter(_streams(1)))
        entry = writer.add(samples)
    table = read_stream_row_group(tmp_path, entry)
    assert table.num_rows == samples.sample_count
    assert table.column("source_time_token").to_pylist() == (
        samples.source_time_token
    )
    assert table.column("source_time_ps").to_pylist() == samples.source_time_ps
    assert table.column("accelerometer_x").to_pylist() == samples.values[0]
    assert set(table.column("p01_evidence_sha256").to_pylist()) == {EVIDENCE}


def test_two_writes_of_the_same_streams_are_byte_identical(
    tmp_path: Path,
) -> None:
    digests = []
    for run in ("a", "b"):
        root = tmp_path / run
        root.mkdir()
        with SampleStoreWriter(
            root, p01_evidence_sha256=EVIDENCE, streams_per_part=4
        ) as writer:
            for samples in _streams(6):
                writer.add(samples)
        digests.append([
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "samples").glob("*.parquet"))
        ])
    assert digests[0] == digests[1]


def test_an_empty_stream_cannot_be_stored(tmp_path: Path) -> None:
    samples = _read(timeseries_bytes(4), rows=4)
    samples.sample_ordinal.clear()
    with SampleStoreWriter(
        tmp_path, p01_evidence_sha256=EVIDENCE
    ) as writer, pytest.raises(SampleStoreError):
        writer.add(samples)
