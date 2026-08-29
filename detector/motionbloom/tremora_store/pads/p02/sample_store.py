"""Deterministic Parquet packing for the PADS source-time sample store.

10,318 tiny text files are not a runtime representation, and copying samples
into task or window records would store the corpus several times over.  The
store instead holds every sample exactly once, packed by stream:

* streams in stable ``stream_id`` order;
* a fixed number of streams per part file;
* exactly one row group per stream, so a stream is one contiguous read;
* one fixed writer configuration, pinned by ``PARQUET_WRITER_POLICY_ID``.

The storage index records where each stream lives and a content hash of what
was stored.  That hash is over the logical content rather than the container
bytes, so two runs agree even if a Parquet writer detail changes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq

from .contract import (
    P02_SCHEMA_VERSION,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_VERSION,
    SAMPLE_PART_PREFIX,
    STREAMS_PER_PART,
)
from .schemas import pads_samples_schema
from .stream_reader import SAMPLE_VALID, StreamSamples

SAMPLES_DIRECTORY = "samples"

#: Dictionary-encoded because they are constant inside a row group.  The time
#: token is deliberately excluded: it is high-cardinality and a dictionary
#: would be larger than the values.
_DICTIONARY_COLUMNS = (
    "participant_id",
    "assessment_id",
    "task_name",
    "device_location",
    "stream_id",
    "sample_status",
    "source_asset_sha256",
    "p01_evidence_sha256",
    "schema_version",
)


class SampleStoreError(RuntimeError):
    """Raised when the sample store would be written non-deterministically."""


@dataclass(frozen=True, slots=True)
class StorageIndexEntry:
    """Where one stream physically lives, and what was stored for it."""

    stream_id: str
    participant_id: str
    assessment_id: str
    task_name: str
    device_location: str
    parquet_relative_path: str
    row_group_index: int
    first_sample_ordinal: int
    last_sample_ordinal: int
    sample_count: int
    first_source_time_ps: int
    last_source_time_ps: int
    first_task_local_time_ps: int
    last_task_local_time_ps: int
    source_asset_sha256: str
    row_group_content_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "participant_id": self.participant_id,
            "assessment_id": self.assessment_id,
            "task_name": self.task_name,
            "device_location": self.device_location,
            "parquet_relative_path": self.parquet_relative_path,
            "row_group_index": self.row_group_index,
            "first_sample_ordinal": self.first_sample_ordinal,
            "last_sample_ordinal": self.last_sample_ordinal,
            "sample_count": self.sample_count,
            "first_source_time_ps": self.first_source_time_ps,
            "last_source_time_ps": self.last_source_time_ps,
            "first_task_local_time_ps": self.first_task_local_time_ps,
            "last_task_local_time_ps": self.last_task_local_time_ps,
            "source_asset_sha256": self.source_asset_sha256,
            "row_group_content_sha256": self.row_group_content_sha256,
        }


def stream_table(
    samples: StreamSamples, *, p01_evidence_sha256: str
) -> pa.Table:
    """Build one stream's rows in the frozen sample schema."""

    count = samples.sample_count
    if count == 0:
        raise SampleStoreError(f"{samples.stream_id} has no samples to store")
    schema = pads_samples_schema()
    columns = {
        "participant_id": [samples.participant_id] * count,
        "assessment_id": [samples.assessment_id] * count,
        "task_name": [samples.task_name] * count,
        "device_location": [samples.device_location] * count,
        "stream_id": [samples.stream_id] * count,
        "sample_ordinal": samples.sample_ordinal,
        "source_row_ordinal": samples.source_row_ordinal,
        "source_time_token": samples.source_time_token,
        "source_time_ps": samples.source_time_ps,
        "task_local_time_ps": samples.task_local_time_ps,
        "accelerometer_x": samples.values[0],
        "accelerometer_y": samples.values[1],
        "accelerometer_z": samples.values[2],
        "gyroscope_x": samples.values[3],
        "gyroscope_y": samples.values[4],
        "gyroscope_z": samples.values[5],
        "sample_status": [SAMPLE_VALID] * count,
        "source_asset_sha256": [samples.source_asset_sha256] * count,
        "p01_evidence_sha256": [p01_evidence_sha256] * count,
        "schema_version": [P02_SCHEMA_VERSION] * count,
    }
    return pa.Table.from_pydict(
        {field.name: columns[field.name] for field in schema},
        schema=schema,
    )


def part_relative_path(part_index: int) -> str:
    return f"{SAMPLES_DIRECTORY}/{SAMPLE_PART_PREFIX}{part_index:05d}.parquet"


class SampleStoreWriter:
    """Pack streams into fixed-size parts, one row group per stream."""

    def __init__(
        self,
        output_root: Path,
        *,
        p01_evidence_sha256: str,
        streams_per_part: int = STREAMS_PER_PART,
    ) -> None:
        self._root = output_root
        self._evidence = p01_evidence_sha256
        self._streams_per_part = streams_per_part
        self._schema = pads_samples_schema()
        self._writer: pq.ParquetWriter | None = None
        self._part_index = -1
        self._row_group_index = 0
        self._streams_in_part = 0
        self._previous_stream_id: str | None = None
        (output_root / SAMPLES_DIRECTORY).mkdir(parents=True, exist_ok=True)

    def _roll(self) -> None:
        self.close()
        self._part_index += 1
        self._row_group_index = 0
        self._streams_in_part = 0
        path = self._root / part_relative_path(self._part_index)
        self._writer = pq.ParquetWriter(
            path,
            self._schema,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            version=PARQUET_VERSION,
            use_dictionary=list(_DICTIONARY_COLUMNS),
            write_statistics=True,
            store_schema=True,
        )

    def add(self, samples: StreamSamples) -> StorageIndexEntry:
        """Write one stream as exactly one row group."""

        if self._previous_stream_id is not None and (
            samples.stream_id <= self._previous_stream_id
        ):
            raise SampleStoreError(
                "streams must be packed in ascending stream_id order")
        self._previous_stream_id = samples.stream_id
        if self._writer is None or (
            self._streams_in_part >= self._streams_per_part
        ):
            self._roll()
        assert self._writer is not None

        table = stream_table(samples, p01_evidence_sha256=self._evidence)
        self._writer.write_table(table, row_group_size=table.num_rows)
        entry = StorageIndexEntry(
            stream_id=samples.stream_id,
            participant_id=samples.participant_id,
            assessment_id=samples.assessment_id,
            task_name=samples.task_name,
            device_location=samples.device_location,
            parquet_relative_path=part_relative_path(self._part_index),
            row_group_index=self._row_group_index,
            first_sample_ordinal=samples.sample_ordinal[0],
            last_sample_ordinal=samples.sample_ordinal[-1],
            sample_count=samples.sample_count,
            first_source_time_ps=samples.source_time_ps[0],
            last_source_time_ps=samples.source_time_ps[-1],
            first_task_local_time_ps=samples.task_local_time_ps[0],
            last_task_local_time_ps=samples.task_local_time_ps[-1],
            source_asset_sha256=samples.source_asset_sha256,
            row_group_content_sha256=samples.content_sha256(),
        )
        self._row_group_index += 1
        self._streams_in_part += 1
        return entry

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    @property
    def part_count(self) -> int:
        return self._part_index + 1


def read_stream_row_group(
    output_root: Path, entry: StorageIndexEntry | dict[str, Any]
) -> pa.Table:
    """Read back exactly the row group one stream was stored as."""

    record = entry.as_record() if isinstance(entry, StorageIndexEntry) else entry
    path = output_root / str(record["parquet_relative_path"])
    handle = pq.ParquetFile(path)
    return handle.read_row_group(int(record["row_group_index"]))


def iter_storage_index(
    entries: Sequence[StorageIndexEntry],
) -> Iterator[dict[str, Any]]:
    for entry in entries:
        yield entry.as_record()


__all__ = [
    "SAMPLES_DIRECTORY",
    "SampleStoreError",
    "SampleStoreWriter",
    "StorageIndexEntry",
    "iter_storage_index",
    "part_relative_path",
    "read_stream_row_group",
    "stream_table",
]
