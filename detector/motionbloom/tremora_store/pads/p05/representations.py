"""The four representations, behind one interface.

Each answers the same four query classes and returns rows in the same
canonical form, so what the benchmark times is the retrieval and not four
different notions of what a result is.  Building the query result is
deliberately identical across all four -- the same ``result_from_rows`` call
on the same field names -- because otherwise a difference in how a
representation assembles its answer would be read as a difference in how fast
it retrieves one.

Nothing here caches a query's result.  Indexes and open file handles are built
once at initialization, which every representation gets and which the cold
measurement charges for; a private result cache available to only one of them
would make the comparison meaningless.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .contract import B0, B1, B2, M1
from .rows import SENSOR_ORDER, QueryResult, result_from_rows

MOVEMENT_DIRECTORY = "movement"
TIME_DECIMALS = 10


class RepresentationError(RuntimeError):
    """Raised when a representation cannot answer a query it was given."""


class Representation(ABC):
    """One way of holding the corpus, and four ways of asking it for rows."""

    name: str

    def __init__(self) -> None:
        self.initialized = False

    # --- lifecycle --------------------------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Load whatever indexes and handles this representation needs."""

    def close(self) -> None:
        return None

    # --- the four query classes -------------------------------------------

    @abstractmethod
    def stream_rows(self, stream_id: str) -> list[dict[str, Any]]:
        """Every stored sample of one stream, in source order."""

    @abstractmethod
    def window_rows(self, window_id: str) -> list[dict[str, Any]]:
        """The rows one four-second window covers."""

    @abstractmethod
    def assessment_rows(self, assessment_id: str) -> list[dict[str, Any]]:
        """Both wrists of one assessment: protocol-paired, not synchronized."""

    def batch_rows(
        self, window_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        """A scattered batch of windows.

        The default is the honest one -- fetch each window -- and a
        representation overrides it only if it can genuinely do better, not to
        skip work the others do.
        """

        out: list[dict[str, Any]] = []
        for window_id in window_ids:
            out.extend(self.window_rows(window_id))
        return out

    # --- the uniform result -----------------------------------------------

    def query(self, query_class: str, query_id: str, **kwargs: Any) -> (
        QueryResult
    ):
        from .contract import Q1, Q2, Q3, Q4

        if query_class == Q1:
            rows = self.stream_rows(query_id)
        elif query_class == Q2:
            rows = self.window_rows(query_id)
        elif query_class == Q3:
            rows = self.assessment_rows(query_id)
        elif query_class == Q4:
            rows = self.batch_rows(kwargs["window_ids"])
        else:  # pragma: no cover - contract guard
            raise RepresentationError(f"unknown query class {query_class!r}")
        return result_from_rows(query_id, rows)


# --- shared index loading -------------------------------------------------


def _index_tables(store_root: Path) -> dict[str, list[dict[str, Any]]]:
    """The P0.2.1 metadata every representation is given equally."""

    out = {}
    for name in (
        "pads_streams", "pads_windows", "pads_assessments",
        "pads_stream_storage_index",
    ):
        out[name] = pq.read_table(store_root / f"{name}.parquet").to_pylist()
    return out


def _rows_from_arrays(
    stream_id: str,
    ordinals: np.ndarray,
    tokens: Sequence[str],
    times_ps: np.ndarray,
    sensors: np.ndarray,
) -> list[dict[str, Any]]:
    """Assemble canonical rows once, the same way for every representation."""

    out = []
    for index in range(len(ordinals)):
        row = {
            "stream_id": stream_id,
            "source_row_ordinal": int(ordinals[index]),
            "source_time_token": tokens[index],
            "source_time_ps": int(times_ps[index]),
        }
        for position, name in enumerate(SENSOR_ORDER):
            row[name] = float(sensors[index, position])
        out.append(row)
    return out


# --- B0: the published text, parsed at query time -------------------------


class SourceTextRepresentation(Representation):
    """The release's own TXT files, parsed when a query asks for them.

    It gets the same resolved path index the others get, built once at open.
    What it does not get is any pre-parsing: that is the representation.
    """

    name = B0

    def __init__(self, release_root: Path, store_root: Path) -> None:
        super().__init__()
        self.movement = release_root / MOVEMENT_DIRECTORY
        self.store_root = store_root
        self.paths: dict[str, str] = {}
        self.windows: dict[str, dict[str, Any]] = {}
        self.assessments: dict[str, dict[str, Any]] = {}
        self.streams: dict[str, dict[str, Any]] = {}

    def open(self) -> None:
        tables = _index_tables(self.store_root)
        self.streams = {
            str(row["stream_id"]): row for row in tables["pads_streams"]
        }
        self.windows = {
            str(row["window_id"]): row for row in tables["pads_windows"]
        }
        self.assessments = {
            str(row["assessment_id"]): row
            for row in tables["pads_assessments"]
        }
        for participant in sorted({
            str(row["participant_id"]) for row in tables["pads_streams"]
        }):
            path = self.movement / f"observation_{participant}.json"
            observation = json.loads(path.read_bytes().decode("utf-8"))
            for session in observation["session"]:
                task = str(session["record_name"])
                for record in session["records"]:
                    location = str(record["device_location"])
                    key = f"{participant}:{task}:{location}"
                    self.paths[key] = str(record["file_name"])
        self.initialized = True

    def _parse(
        self, stream_id: str, first: int, last: int
    ) -> list[dict[str, Any]]:
        try:
            name = self.paths[stream_id]
        except KeyError as exc:
            raise RepresentationError(f"no source file for {stream_id}") from exc
        lines = (self.movement / name).read_text().split("\n")
        out = []
        for ordinal in range(first, last + 1):
            fields = lines[ordinal].split(",")
            token = fields[0]
            row = {
                "stream_id": stream_id,
                "source_row_ordinal": ordinal,
                "source_time_token": token,
                "source_time_ps": _token_picoseconds(token),
            }
            for position, sensor in enumerate(SENSOR_ORDER):
                row[sensor] = float(fields[1 + position])
            out.append(row)
        return out

    def stream_rows(self, stream_id: str) -> list[dict[str, Any]]:
        stream = self.streams[stream_id]
        return self._parse(
            stream_id, 0, int(stream["source_row_count"]) - 1
        )

    def window_rows(self, window_id: str) -> list[dict[str, Any]]:
        window = self.windows[window_id]
        return self._parse(
            str(window["stream_id"]),
            int(window["first_sample_ordinal"]),
            int(window["last_sample_ordinal"]),
        )

    def assessment_rows(self, assessment_id: str) -> list[dict[str, Any]]:
        assessment = self.assessments[assessment_id]
        out: list[dict[str, Any]] = []
        for side in ("left_stream_id", "right_stream_id"):
            stream_id = assessment.get(side)
            if stream_id:
                out.extend(self.stream_rows(str(stream_id)))
        return out


def _token_picoseconds(token: str) -> int:
    """The release's ten-decimal seconds token, exactly, in picoseconds.

    Done with integer arithmetic on the digits rather than through a float,
    because a float cannot hold ten decimal places of seconds and this is the
    value the equivalence check compares.
    """

    negative = token.startswith("-")
    body = token[1:] if negative else token
    if "." in body:
        whole, _, fraction = body.partition(".")
    else:
        whole, fraction = body, ""
    fraction = (fraction + "0" * TIME_DECIMALS)[:TIME_DECIMALS]
    value = int(whole or "0") * 10**12 + int(fraction or "0") * 100
    return -value if negative else value



# --- M1: the P0.2.1 Parquet store and its immutable indexes ---------------


class TremoraParquetRepresentation(Representation):
    """The system under test: one row group per stream, indexes beside it.

    Its window index carries ordinal ranges, not sample copies, so a window is
    a slice of the stream's row group rather than a stored object.  That is
    the property the duplication accounting is about.
    """

    name = M1

    #: The nine columns a query needs.  The store carries eleven more --
    #: participant, task, provenance hashes, schema version -- which are what
    #: make it auditable and which no baseline was asked to carry either.
    #: Reading them here would charge M1 for columns B1 and B2 were built
    #: without, so the projection is symmetry rather than special pleading.
    PROJECTION: tuple[str, ...] = (
        "source_row_ordinal",
        "source_time_token",
        "source_time_ps",
        *SENSOR_ORDER,
    )

    def __init__(self, store_root: Path) -> None:
        super().__init__()
        self.store_root = store_root
        self.index: dict[str, dict[str, Any]] = {}
        self.windows: dict[str, dict[str, Any]] = {}
        self.assessments: dict[str, dict[str, Any]] = {}
        self._files: dict[str, pq.ParquetFile] = {}

    def open(self) -> None:
        tables = _index_tables(self.store_root)
        self.index = {
            str(row["stream_id"]): row
            for row in tables["pads_stream_storage_index"]
        }
        self.windows = {
            str(row["window_id"]): row for row in tables["pads_windows"]
        }
        self.assessments = {
            str(row["assessment_id"]): row
            for row in tables["pads_assessments"]
        }
        self.initialized = True

    def close(self) -> None:
        self._files.clear()

    def _file(self, relative: str) -> pq.ParquetFile:
        # An open handle, exactly as B1 keeps and as B2's HDF5 file is.  This
        # is a file handle, not a result cache: no query's rows are retained.
        handle = self._files.get(relative)
        if handle is None:
            handle = pq.ParquetFile(self.store_root / relative)
            self._files[relative] = handle
        return handle

    def _table(self, stream_id: str):
        try:
            entry = self.index[stream_id]
        except KeyError as exc:
            raise RepresentationError(f"{stream_id} is not indexed") from exc
        handle = self._file(str(entry["parquet_relative_path"]))
        return handle.read_row_group(
            int(entry["row_group_index"]), columns=list(self.PROJECTION)
        )

    def _rows(self, stream_id: str, first: int | None, last: int | None):
        table = self._table(stream_id)
        ordinals = table.column("source_row_ordinal").to_numpy()
        if first is None:
            lo, hi = 0, table.num_rows
        else:
            lo = int(np.searchsorted(ordinals, first, side="left"))
            hi = int(np.searchsorted(ordinals, last, side="right"))
        sliced = table.slice(lo, hi - lo)
        return _rows_from_arrays(
            stream_id,
            sliced.column("source_row_ordinal").to_numpy(),
            sliced.column("source_time_token").to_pylist(),
            sliced.column("source_time_ps").to_numpy(),
            np.column_stack([
                sliced.column(name).to_numpy() for name in SENSOR_ORDER
            ]),
        )

    def stream_rows(self, stream_id: str) -> list[dict[str, Any]]:
        return self._rows(stream_id, None, None)

    def window_rows(self, window_id: str) -> list[dict[str, Any]]:
        window = self.windows[window_id]
        return self._rows(
            str(window["stream_id"]),
            int(window["first_sample_ordinal"]),
            int(window["last_sample_ordinal"]),
        )

    def assessment_rows(self, assessment_id: str) -> list[dict[str, Any]]:
        assessment = self.assessments[assessment_id]
        out: list[dict[str, Any]] = []
        for side in ("left_stream_id", "right_stream_id"):
            stream_id = assessment.get(side)
            if stream_id:
                out.extend(self.stream_rows(str(stream_id)))
        return out


# --- B1: window-materialized, physically duplicating the overlap ----------


class DuplicatedWindowRepresentation(Representation):
    """A window store that materializes each window as its own block.

    This is the architecture a training pipeline reaches for when it wants a
    window to be one contiguous read.  The cost is that neighbouring windows
    overlap by half and each keeps its own copy of the shared samples.

    It also holds the full per-stream sample set, because the windows cover
    only 12,150,522 of the 13,447,168 samples across 9,960 of the 10,318
    streams.  Without that it could not replay the other 358 streams at all,
    and would fail the equivalence check rather than lose a race -- which
    would say nothing about duplication.
    """

    name = B1

    STREAM_DIRECTORY = "streams"
    WINDOW_DIRECTORY = "windows"

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.stream_index: dict[str, dict[str, Any]] = {}
        self.window_index: dict[str, dict[str, Any]] = {}
        self.assessments: dict[str, dict[str, Any]] = {}
        self._files: dict[str, pq.ParquetFile] = {}

    def open(self) -> None:
        manifest = json.loads(
            (self.root / "b1_manifest.json").read_bytes().decode("utf-8")
        )
        self.stream_index = manifest["streams"]
        self.window_index = manifest["windows"]
        self.assessments = manifest["assessments"]
        self.initialized = True

    def close(self) -> None:
        self._files.clear()

    def _file(self, relative: str) -> pq.ParquetFile:
        handle = self._files.get(relative)
        if handle is None:
            handle = pq.ParquetFile(self.root / relative)
            self._files[relative] = handle
        return handle

    def _read(self, entry: Mapping[str, Any], stream_id: str):
        table = self._file(str(entry["path"])).read_row_group(
            int(entry["row_group"])
        )
        return _rows_from_arrays(
            stream_id,
            table.column("source_row_ordinal").to_numpy(),
            table.column("source_time_token").to_pylist(),
            table.column("source_time_ps").to_numpy(),
            np.column_stack([
                table.column(name).to_numpy() for name in SENSOR_ORDER
            ]),
        )

    def stream_rows(self, stream_id: str) -> list[dict[str, Any]]:
        try:
            entry = self.stream_index[stream_id]
        except KeyError as exc:
            raise RepresentationError(f"{stream_id} not materialized") from exc
        return self._read(entry, stream_id)

    def window_rows(self, window_id: str) -> list[dict[str, Any]]:
        try:
            entry = self.window_index[window_id]
        except KeyError as exc:
            raise RepresentationError(f"{window_id} not materialized") from exc
        # One contiguous read: this is exactly what the duplication buys.
        return self._read(entry, str(entry["stream_id"]))

    def assessment_rows(self, assessment_id: str) -> list[dict[str, Any]]:
        assessment = self.assessments[assessment_id]
        out: list[dict[str, Any]] = []
        for side in ("left", "right"):
            stream_id = assessment.get(side)
            if stream_id:
                out.extend(self.stream_rows(str(stream_id)))
        return out


# --- B2: HDF5, columnar, with real range indexes --------------------------


class Hdf5RangeIndexedRepresentation(Representation):
    """One columnar HDF5 file with genuine per-stream and per-window offsets.

    It is given the indexes it needs to answer a window with a single chunked
    slice.  A version made to scan whole files would lose this benchmark and
    the result would mean nothing, so it is not built that way.
    """

    name = B2

    FILENAME = "pads_b2.h5"

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.handle: Any = None
        self.stream_offsets: dict[str, tuple[int, int]] = {}
        self.window_offsets: dict[str, tuple[str, int, int]] = {}
        self.assessments: dict[str, dict[str, Any]] = {}
        self._columns: dict[str, Any] = {}
        self._tokens: Any = None

    def open(self) -> None:
        import h5py
        import hdf5plugin  # noqa: F401 - registers the zstd filter for reads

        self.handle = h5py.File(self.root / self.FILENAME, "r")
        index = self.handle["stream_offset_index"]
        for name, start, stop in zip(
            index["stream_id"].asstr()[:],
            index["start"][:], index["stop"][:], strict=True,
        ):
            self.stream_offsets[str(name)] = (int(start), int(stop))
        windows = self.handle["window_offset_index"]
        for name, stream_id, start, stop in zip(
            windows["window_id"].asstr()[:],
            windows["stream_id"].asstr()[:],
            windows["start"][:], windows["stop"][:], strict=True,
        ):
            self.window_offsets[str(name)] = (
                str(stream_id), int(start), int(stop)
            )
        self.assessments = json.loads(
            self.handle.attrs["assessments"]
        )
        samples = self.handle["samples"]
        self._columns = {name: samples[name] for name in SENSOR_ORDER}
        self._columns["source_row_ordinal"] = samples["source_row_ordinal"]
        self._columns["source_time_ps"] = samples["source_time_ps"]
        self._tokens = samples["source_time_token"]
        self.initialized = True

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def _slice(
        self, stream_id: str, start: int, stop: int
    ) -> list[dict[str, Any]]:
        sensors = np.column_stack([
            self._columns[name][start:stop] for name in SENSOR_ORDER
        ])
        return _rows_from_arrays(
            stream_id,
            self._columns["source_row_ordinal"][start:stop],
            [token.decode("ascii") for token in self._tokens[start:stop]],
            self._columns["source_time_ps"][start:stop],
            sensors,
        )

    def stream_rows(self, stream_id: str) -> list[dict[str, Any]]:
        try:
            start, stop = self.stream_offsets[stream_id]
        except KeyError as exc:
            raise RepresentationError(f"{stream_id} not stored") from exc
        return self._slice(stream_id, start, stop)

    def window_rows(self, window_id: str) -> list[dict[str, Any]]:
        try:
            stream_id, start, stop = self.window_offsets[window_id]
        except KeyError as exc:
            raise RepresentationError(f"{window_id} not indexed") from exc
        return self._slice(stream_id, start, stop)

    def assessment_rows(self, assessment_id: str) -> list[dict[str, Any]]:
        assessment = self.assessments[assessment_id]
        out: list[dict[str, Any]] = []
        for side in ("left", "right"):
            stream_id = assessment.get(side)
            if stream_id:
                out.extend(self.stream_rows(str(stream_id)))
        return out


__all__ = [
    "B0",
    "B1",
    "B2",
    "M1",
    "MOVEMENT_DIRECTORY",
    "DuplicatedWindowRepresentation",
    "Hdf5RangeIndexedRepresentation",
    "Representation",
    "RepresentationError",
    "SourceTextRepresentation",
    "TremoraParquetRepresentation",
]
