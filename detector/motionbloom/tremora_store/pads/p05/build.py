"""Build the two baselines that have to be materialized before they can run.

B0 is the release as published and B1 needs nothing built.  B1 and B2 do, and
how they are built is where a comparison like this is usually decided, so both
constructions are deliberate and both are checkable afterwards.

B1 physically duplicates the samples its overlapping windows share, because a
duplication claim against a baseline that quietly deduplicated would measure
nothing.  B2 is given per-stream and per-window offset indexes and chunked
columnar datasets, because a comparison against an HDF5 file forced to scan
would measure the crippling.  Both compress with the same codec and level the
Parquet store uses, so a size difference is a layout difference.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..p02.replay import read_stream_row_group
from .contract import (
    COMPRESSION_CODEC,
    COMPRESSION_LEVEL,
    HDF5_CHUNK_ROWS,
    HDF5_REQUIRED_INDEXES,
)
from .representations import (
    DuplicatedWindowRepresentation,
    Hdf5RangeIndexedRepresentation,
)
from .rows import SENSOR_ORDER

PARQUET_VERSION = "2.6"
CARRIED_COLUMNS: tuple[str, ...] = (
    "source_row_ordinal",
    "source_time_token",
    "source_time_ps",
    *SENSOR_ORDER,
)


class BuildError(RuntimeError):
    """Raised when a baseline cannot be materialized as the contract says."""


@dataclass(slots=True)
class BuildReport:
    """What was actually written, so the fairness rules can be checked."""

    representation: str
    stored_sample_instances: int = 0
    unique_samples: int = 0
    streams: int = 0
    windows: int = 0
    file_count: int = 0
    physical_storage_bytes: int = 0
    index_bytes: int = 0
    metadata_bytes: int = 0
    compression: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "stored_sample_instances": self.stored_sample_instances,
            "unique_samples": self.unique_samples,
            "duplicate_sample_instances": (
                self.stored_sample_instances - self.unique_samples
            ),
            "duplication_factor": (
                self.stored_sample_instances / self.unique_samples
                if self.unique_samples else 0.0
            ),
            "streams": self.streams,
            "windows": self.windows,
            "file_count": self.file_count,
            "physical_storage_bytes": self.physical_storage_bytes,
            "index_bytes": self.index_bytes,
            "metadata_bytes": self.metadata_bytes,
            "compression": dict(sorted(self.compression.items())),
            **{key: self.detail[key] for key in sorted(self.detail)},
        }


def _store_tables(store_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        name: pq.read_table(store_root / f"{name}.parquet").to_pylist()
        for name in (
            "pads_streams", "pads_windows", "pads_assessments",
            "pads_stream_storage_index",
        )
    }


def _stream_tables(
    store_root: Path, index: Mapping[str, Mapping[str, Any]]
) -> Iterator[tuple[str, pa.Table]]:
    """Every stream's samples, once, in index order."""

    for stream_id in sorted(index):
        table = read_stream_row_group(store_root, dict(index[stream_id]))
        yield stream_id, table.select(list(CARRIED_COLUMNS))


def _directory_bytes(root: Path, pattern: str = "**/*") -> tuple[int, int]:
    total = 0
    count = 0
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            total += path.stat().st_size
            count += 1
    return total, count


# --- B1 -------------------------------------------------------------------


def build_b1(
    *, store_root: Path, output_root: Path, progress: bool = False
) -> BuildReport:
    """Materialize the duplicated-window baseline."""

    tables = _store_tables(store_root)
    index = {
        str(row["stream_id"]): row
        for row in tables["pads_stream_storage_index"]
    }
    windows = tables["pads_windows"]
    output_root.mkdir(parents=True, exist_ok=True)
    stream_dir = output_root / DuplicatedWindowRepresentation.STREAM_DIRECTORY
    window_dir = output_root / DuplicatedWindowRepresentation.WINDOW_DIRECTORY
    stream_dir.mkdir(exist_ok=True)
    window_dir.mkdir(exist_ok=True)

    by_stream: dict[str, list[dict[str, Any]]] = {}
    for row in windows:
        by_stream.setdefault(str(row["stream_id"]), []).append(row)

    report = BuildReport(representation=DuplicatedWindowRepresentation.name)
    stream_entries: dict[str, dict[str, Any]] = {}
    window_entries: dict[str, dict[str, Any]] = {}

    schema = None
    stream_path = stream_dir / "streams.parquet"
    window_path = window_dir / "windows.parquet"
    stream_writer = None
    window_writer = None
    stream_group = 0
    window_group = 0
    try:
        for position, (stream_id, table) in enumerate(
            _stream_tables(store_root, index)
        ):
            if schema is None:
                schema = table.schema
                stream_writer = pq.ParquetWriter(
                    stream_path, schema, compression=COMPRESSION_CODEC,
                    compression_level=COMPRESSION_LEVEL,
                    version=PARQUET_VERSION,
                )
                window_writer = pq.ParquetWriter(
                    window_path, schema, compression=COMPRESSION_CODEC,
                    compression_level=COMPRESSION_LEVEL,
                    version=PARQUET_VERSION,
                )
            # The full stream, so Q1 and Q3 can be answered at all.
            stream_writer.write_table(table, row_group_size=table.num_rows)
            stream_entries[stream_id] = {
                "path": f"{DuplicatedWindowRepresentation.STREAM_DIRECTORY}"
                        "/streams.parquet",
                "row_group": stream_group,
                "stream_id": stream_id,
            }
            stream_group += 1
            report.unique_samples += table.num_rows
            report.stored_sample_instances += table.num_rows
            report.streams += 1

            ordinals = table.column("source_row_ordinal").to_numpy()
            for window in sorted(
                by_stream.get(stream_id, ()),
                key=lambda row: int(row["first_sample_ordinal"]),
            ):
                lo = int(np.searchsorted(
                    ordinals, int(window["first_sample_ordinal"]), "left"
                ))
                hi = int(np.searchsorted(
                    ordinals, int(window["last_sample_ordinal"]), "right"
                ))
                block = table.slice(lo, hi - lo)
                # A physical copy, overlap and all.  That is the baseline.
                window_writer.write_table(
                    block, row_group_size=max(block.num_rows, 1)
                )
                window_entries[str(window["window_id"])] = {
                    "path": (
                        f"{DuplicatedWindowRepresentation.WINDOW_DIRECTORY}"
                        "/windows.parquet"
                    ),
                    "row_group": window_group,
                    "stream_id": stream_id,
                }
                window_group += 1
                report.stored_sample_instances += block.num_rows
                report.windows += 1
            if progress and position % 1000 == 0:
                print(f"  b1 {position}/{len(index)}", flush=True)
    finally:
        if stream_writer is not None:
            stream_writer.close()
        if window_writer is not None:
            window_writer.close()

    manifest = {
        "streams": stream_entries,
        "windows": window_entries,
        "assessments": {
            str(row["assessment_id"]): {
                "left": row.get("left_stream_id"),
                "right": row.get("right_stream_id"),
            }
            for row in tables["pads_assessments"]
        },
        "compression": {
            "codec": COMPRESSION_CODEC, "level": COMPRESSION_LEVEL,
        },
    }
    manifest_path = output_root / "b1_manifest.json"
    manifest_path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )
    report.physical_storage_bytes, report.file_count = _directory_bytes(
        output_root
    )
    report.index_bytes = manifest_path.stat().st_size
    report.metadata_bytes = report.index_bytes
    report.compression = {
        "codec": COMPRESSION_CODEC, "level": COMPRESSION_LEVEL,
    }
    report.detail = {
        "window_row_groups": window_group,
        "stream_row_groups": stream_group,
    }
    return report


# --- B2 -------------------------------------------------------------------


def build_b2(
    *, store_root: Path, output_root: Path, progress: bool = False
) -> BuildReport:
    """Materialize the HDF5 baseline, with the indexes it is owed."""

    import h5py
    import hdf5plugin

    tables = _store_tables(store_root)
    index = {
        str(row["stream_id"]): row
        for row in tables["pads_stream_storage_index"]
    }
    total = sum(int(row["sample_count"]) for row in index.values())
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / Hdf5RangeIndexedRepresentation.FILENAME

    report = BuildReport(representation=Hdf5RangeIndexedRepresentation.name)
    compression = hdf5plugin.Zstd(clevel=COMPRESSION_LEVEL)
    chunk = (min(HDF5_CHUNK_ROWS, total),)

    with h5py.File(path, "w") as handle:
        samples = handle.create_group("samples")
        columns = {
            name: samples.create_dataset(
                name, shape=(total,), dtype="f8", chunks=chunk, **compression
            )
            for name in SENSOR_ORDER
        }
        columns["source_row_ordinal"] = samples.create_dataset(
            "source_row_ordinal", shape=(total,), dtype="i4",
            chunks=chunk, **compression,
        )
        columns["source_time_ps"] = samples.create_dataset(
            "source_time_ps", shape=(total,), dtype="i8",
            chunks=chunk, **compression,
        )
        tokens = samples.create_dataset(
            "source_time_token", shape=(total,), dtype="S16",
            chunks=chunk, **compression,
        )

        offsets: list[tuple[str, int, int]] = []
        cursor = 0
        ordinal_lookup: dict[str, np.ndarray] = {}
        for position, (stream_id, table) in enumerate(
            _stream_tables(store_root, index)
        ):
            rows = table.num_rows
            stop = cursor + rows
            for name in SENSOR_ORDER:
                columns[name][cursor:stop] = table.column(name).to_numpy()
            ordinals = table.column("source_row_ordinal").to_numpy()
            columns["source_row_ordinal"][cursor:stop] = ordinals
            columns["source_time_ps"][cursor:stop] = (
                table.column("source_time_ps").to_numpy()
            )
            tokens[cursor:stop] = [
                value.encode("ascii")
                for value in table.column("source_time_token").to_pylist()
            ]
            offsets.append((stream_id, cursor, stop))
            ordinal_lookup[stream_id] = ordinals
            cursor = stop
            report.streams += 1
            report.stored_sample_instances += rows
            report.unique_samples += rows
            if progress and position % 1000 == 0:
                print(f"  b2 {position}/{len(index)}", flush=True)

        if cursor != total:  # pragma: no cover - accounting guard
            raise BuildError(f"wrote {cursor} of {total} samples")

        # The two indexes the contract promises this baseline.
        string_type = h5py.string_dtype()
        stream_index = handle.create_group("stream_offset_index")
        stream_index.create_dataset(
            "stream_id", data=[name for name, _, _ in offsets],
            dtype=string_type,
        )
        stream_index.create_dataset(
            "start", data=np.array([lo for _, lo, _ in offsets], dtype="i8")
        )
        stream_index.create_dataset(
            "stop", data=np.array([hi for _, _, hi in offsets], dtype="i8")
        )

        base = {name: lo for name, lo, _ in offsets}
        window_ids, window_streams, starts, stops = [], [], [], []
        for row in sorted(
            tables["pads_windows"], key=lambda item: str(item["window_id"])
        ):
            stream_id = str(row["stream_id"])
            ordinals = ordinal_lookup[stream_id]
            lo = int(np.searchsorted(
                ordinals, int(row["first_sample_ordinal"]), "left"
            ))
            hi = int(np.searchsorted(
                ordinals, int(row["last_sample_ordinal"]), "right"
            ))
            window_ids.append(str(row["window_id"]))
            window_streams.append(stream_id)
            starts.append(base[stream_id] + lo)
            stops.append(base[stream_id] + hi)
            report.windows += 1
        window_index = handle.create_group("window_offset_index")
        window_index.create_dataset(
            "window_id", data=window_ids, dtype=string_type
        )
        window_index.create_dataset(
            "stream_id", data=window_streams, dtype=string_type
        )
        window_index.create_dataset(
            "start", data=np.array(starts, dtype="i8")
        )
        window_index.create_dataset("stop", data=np.array(stops, dtype="i8"))

        handle.attrs["assessments"] = json.dumps({
            str(row["assessment_id"]): {
                "left": row.get("left_stream_id"),
                "right": row.get("right_stream_id"),
            }
            for row in tables["pads_assessments"]
        }, sort_keys=True)
        handle.attrs["compression_codec"] = COMPRESSION_CODEC
        handle.attrs["compression_level"] = COMPRESSION_LEVEL
        handle.attrs["indexes"] = json.dumps(list(HDF5_REQUIRED_INDEXES))

    report.physical_storage_bytes = path.stat().st_size
    report.file_count = 1
    # The index datasets are inside the same file; their size is measured by
    # what they hold rather than by a separate file.
    report.index_bytes = (
        len(window_ids) * 16 + report.streams * 16
    )
    report.metadata_bytes = len(
        json.dumps(list(HDF5_REQUIRED_INDEXES)).encode()
    )
    report.compression = {
        "codec": COMPRESSION_CODEC, "level": COMPRESSION_LEVEL,
    }
    report.detail = {
        "chunk_rows": int(chunk[0]),
        "indexes": list(HDF5_REQUIRED_INDEXES),
    }
    return report


__all__ = [
    "CARRIED_COLUMNS",
    "BuildError",
    "BuildReport",
    "build_b1",
    "build_b2",
]
