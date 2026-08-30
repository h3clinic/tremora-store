"""Streaming writer for the timing table, and the summaries read back from it.

At roughly 2.3 million measured queries the timing table is itself a
significant artifact, so rows are written to compressed Parquet in bounded
batches and never accumulated in memory.  Only running counters live in the
process.

The summaries are then computed by reading that table back column by column,
which keeps live memory flat regardless of how many rounds were measured and
means the published numbers are derived from the published table rather than
from a parallel in-memory copy of it that could drift from it.
"""

from __future__ import annotations

import statistics
from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq

from ..p02.contract import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_VERSION,
)
from .contract import (
    LATENCY_PERCENTILES,
    MEASURED_ROUNDS_BY_QUERY_CLASS,
    PRIMARY_BATCH_SIZE,
    Q2,
    REPRESENTATIONS,
)
from .schemas import pads_p05_retrieval_schema

#: Rows per Parquet row group.  Large enough that the file is not a pile of
#: tiny groups, small enough that the buffer is a few megabytes.
FLUSH_ROWS = 65_536

QUERY_OK = "QUERY_OK"


class SinkError(RuntimeError):
    """Raised when the timing table cannot be written or read back."""


@dataclass(slots=True)
class SinkCounters:
    """What the sink knows without holding a single row."""

    rows_written: int = 0
    rows_offered: int = 0
    failed_queries: int = 0
    warmup_rows_discarded: int = 0
    bytes_returned: int = 0
    rounds_seen_by_class: dict[str, set] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "rows_written": self.rows_written,
            "rows_offered": self.rows_offered,
            "failed_queries": self.failed_queries,
            "warmup_rows_discarded": self.warmup_rows_discarded,
            "bytes_returned": self.bytes_returned,
            "rounds_by_query_class": {
                name: sorted(rounds)
                for name, rounds in sorted(self.rounds_seen_by_class.items())
            },
        }


class MeasurementSink:
    """Append measured rows to Parquet in bounded batches.

    Warm-up rows are dropped here rather than filtered later, so a warm-up
    measurement cannot reach a summary by being forgotten about downstream.
    """

    def __init__(self, path: Path, *, flush_rows: int = FLUSH_ROWS) -> None:
        self.path = path
        self.flush_rows = flush_rows
        self.schema = pads_p05_retrieval_schema()
        self.counters = SinkCounters()
        self._buffer: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None

    def __enter__(self) -> Self:
        self._writer = pq.ParquetWriter(
            self.path, self.schema,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            version=PARQUET_VERSION,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add(self, record: Mapping[str, Any]) -> None:
        self.counters.rows_offered += 1
        if str(record["status"]) != QUERY_OK:
            self.counters.failed_queries += 1
        round_id = int(record["round_id"])
        if round_id < 0:
            self.counters.warmup_rows_discarded += 1
            return
        self.counters.rounds_seen_by_class.setdefault(
            str(record["query_class"]), set()
        ).add(round_id)
        self.counters.bytes_returned += int(record["bytes_returned"])
        self._buffer.append(
            {name: record[name] for name in self.schema.names}
        )
        if len(self._buffer) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer or self._writer is None:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        self._writer.write_table(table)
        self.counters.rows_written += table.num_rows
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# --- reading the table back ----------------------------------------------


def _batches(
    path: Path, columns: Sequence[str], batch_rows: int = 262_144
) -> Iterator[pa.RecordBatch]:
    handle = pq.ParquetFile(path)
    yield from handle.iter_batches(
        batch_size=batch_rows, columns=list(columns)
    )


def _percentile(values: Sequence[int], percentile: int) -> float:
    if not values:
        return 0.0
    position = (percentile / 100.0) * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1.0 - weight) + values[high] * weight


def summarize_table(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Per representation and query class, the published latency summary."""

    latencies: dict[tuple[str, str], array] = {}
    totals: dict[tuple[str, str], dict[str, int]] = {}
    for batch in _batches(path, (
        "representation", "query_class", "latency_ns", "cpu_time_ns",
        "rows_returned", "peak_rss_delta",
    )):
        names = batch.column("representation").to_pylist()
        classes = batch.column("query_class").to_pylist()
        lat = batch.column("latency_ns").to_pylist()
        cpu = batch.column("cpu_time_ns").to_pylist()
        rows = batch.column("rows_returned").to_pylist()
        rss = batch.column("peak_rss_delta").to_pylist()
        for index in range(batch.num_rows):
            key = (names[index], classes[index])
            latencies.setdefault(key, array("q")).append(lat[index])
            entry = totals.setdefault(
                key, {"cpu": 0, "rows": 0, "rss": 0, "elapsed": 0}
            )
            entry["cpu"] += cpu[index]
            entry["rows"] += rows[index]
            entry["elapsed"] += lat[index]
            entry["rss"] = max(entry["rss"], rss[index])

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (name, query_class), values in sorted(latencies.items()):
        ordered = sorted(values)
        entry = totals[(name, query_class)]
        summary = {
            "queries": len(ordered),
            "rows_returned": entry["rows"],
            "mean_latency_ns": statistics.fmean(ordered),
            "cpu_time_ns_total": entry["cpu"],
            "peak_rss_delta_max": entry["rss"],
            "rows_per_second": (
                entry["rows"] / (entry["elapsed"] / 1e9)
                if entry["elapsed"] else 0.0
            ),
            "queries_per_second": (
                len(ordered) / (entry["elapsed"] / 1e9)
                if entry["elapsed"] else 0.0
            ),
            "measured_rounds": MEASURED_ROUNDS_BY_QUERY_CLASS.get(
                query_class, 0
            ),
        }
        for percentile in LATENCY_PERCENTILES:
            summary[f"p{percentile}_latency_ns"] = _percentile(
                ordered, percentile
            )
        out.setdefault(name, {})[query_class] = summary
    return out


def batch_throughput_table(
    path: Path, *, size: int = PRIMARY_BATCH_SIZE
) -> dict[str, dict[str, float]]:
    """Windows per second for one batch size, read back from the table."""

    out: dict[str, dict[str, float]] = {}
    prefix = f"batch:{size}:"
    for batch in _batches(path, (
        "representation", "query_class", "query_id", "latency_ns",
    )):
        names = batch.column("representation").to_pylist()
        ids = batch.column("query_id").to_pylist()
        lat = batch.column("latency_ns").to_pylist()
        for index in range(batch.num_rows):
            if not ids[index].startswith(prefix):
                continue
            entry = out.setdefault(
                names[index],
                {"windows": 0.0, "elapsed_ns": 0.0, "batches": 0.0},
            )
            entry["windows"] += size
            entry["elapsed_ns"] += lat[index]
            entry["batches"] += 1
    for entry in out.values():
        entry["windows_per_second"] = (
            entry["windows"] / (entry["elapsed_ns"] / 1e9)
            if entry["elapsed_ns"] else 0.0
        )
    return out


def per_query_medians(
    path: Path, *, query_class: str = Q2
) -> dict[str, dict[str, float]]:
    """One number per (representation, query id): its median over rounds.

    This is the unit the speed ratios bootstrap over.  Collapsing a query's
    rounds to a single median here is what stops ten repetitions of one query
    posing as ten independent workload items.
    """

    gathered: dict[str, dict[str, array]] = {}
    for batch in _batches(path, (
        "representation", "query_class", "query_id", "latency_ns",
    )):
        names = batch.column("representation").to_pylist()
        classes = batch.column("query_class").to_pylist()
        ids = batch.column("query_id").to_pylist()
        lat = batch.column("latency_ns").to_pylist()
        for index in range(batch.num_rows):
            if classes[index] != query_class:
                continue
            gathered.setdefault(names[index], {}).setdefault(
                ids[index], array("q")
            ).append(lat[index])
    return {
        name: {
            query_id: float(statistics.median(values))
            for query_id, values in queries.items()
        }
        for name, queries in gathered.items()
    }


def participant_rows_from_table(
    path: Path,
    *,
    window_participants: Mapping[str, str],
    condition_groups: Mapping[str, str],
    query_class: str = Q2,
) -> list[dict[str, Any]]:
    """Per-participant latency, read back from the table."""

    buckets: dict[tuple[str, str], array] = {}
    for batch in _batches(path, (
        "representation", "query_class", "query_id", "latency_ns",
    )):
        names = batch.column("representation").to_pylist()
        classes = batch.column("query_class").to_pylist()
        ids = batch.column("query_id").to_pylist()
        lat = batch.column("latency_ns").to_pylist()
        for index in range(batch.num_rows):
            if classes[index] != query_class:
                continue
            participant = window_participants.get(ids[index])
            if participant is None:
                continue
            buckets.setdefault(
                (participant, names[index]), array("q")
            ).append(lat[index])
    out = []
    for (participant, name), values in sorted(buckets.items()):
        ordered = sorted(values)
        out.append({
            "participant_id": participant,
            "condition_group": condition_groups.get(participant, "UNKNOWN"),
            "representation": name,
            "query_class": query_class,
            "queries": len(ordered),
            "median_latency_ns": float(statistics.median(ordered)),
            "p95_latency_ns": _percentile(ordered, 95),
        })
    return out


def rounds_by_class(path: Path) -> dict[str, int]:
    """How many distinct measured rounds each query class actually got."""

    seen: dict[str, set] = {}
    for batch in _batches(path, ("query_class", "round_id")):
        classes = batch.column("query_class").to_pylist()
        rounds = batch.column("round_id").to_pylist()
        for index in range(batch.num_rows):
            seen.setdefault(classes[index], set()).add(rounds[index])
    return {name: len(rounds) for name, rounds in sorted(seen.items())}


def warmup_rows_present(path: Path) -> int:
    """Warm-up rows that reached the table.  Must be zero."""

    total = 0
    for batch in _batches(path, ("round_id",)):
        total += sum(1 for value in batch.column("round_id").to_pylist()
                     if value < 0)
    return total


def representations_present(path: Path) -> tuple[str, ...]:
    seen: set[str] = set()
    for batch in _batches(path, ("representation",)):
        seen.update(batch.column("representation").to_pylist())
    return tuple(sorted(seen & set(REPRESENTATIONS)))


__all__ = [
    "FLUSH_ROWS",
    "MeasurementSink",
    "SinkCounters",
    "SinkError",
    "batch_throughput_table",
    "participant_rows_from_table",
    "per_query_medians",
    "representations_present",
    "rounds_by_class",
    "summarize_table",
    "warmup_rows_present",
]
