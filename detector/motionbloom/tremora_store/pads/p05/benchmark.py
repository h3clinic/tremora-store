"""Running the frozen workload and recording what it cost.

Timings are not reproducible and this module does not pretend otherwise.  The
canonical evidence covers the deterministic half -- which rows came back, what
is on disk, which queries ran in which order -- and the measured latencies are
published beside it as their own table.  Two honest executions therefore agree
byte-for-byte on the evidence while disagreeing about nanoseconds, which is
the truthful version of "reproduced".

Every representation sees the same query order within a round, and which
representation runs first rotates between rounds, so an ordering or thermal
effect lands on all four or on none.
"""

from __future__ import annotations

import hashlib
import resource
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contract import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    COLD,
    MEASURED_ROUNDS,
    MEASURED_ROUNDS_BY_QUERY_CLASS,
    Q1,
    Q2,
    Q3,
    Q4,
    REPRESENTATIONS,
    TOTAL_ROUNDS,
    WARM,
    WARMUP_ROUNDS,
)
from .representations import Representation
from .rows import result_from_rows
from .workload import QueryWorkload, representation_order, round_order

QUERY_OK = "QUERY_OK"
QUERY_FAILED = "QUERY_FAILED"

WARMUP_ROUND_ID = -1


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


@dataclass(slots=True)
class Measurement:
    """One timed query, in the eleven fields the contract publishes."""

    representation: str
    query_class: str
    query_id: str
    round_id: int
    latency_ns: int
    cpu_time_ns: int
    rows_returned: int
    bytes_returned: int
    peak_rss_delta: int
    content_sha256: str
    status: str

    def as_record(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "query_class": self.query_class,
            "query_id": self.query_id,
            "round_id": self.round_id,
            "latency_ns": self.latency_ns,
            "cpu_time_ns": self.cpu_time_ns,
            "rows_returned": self.rows_returned,
            "bytes_returned": self.bytes_returned,
            "peak_rss_delta": self.peak_rss_delta,
            "content_sha256": self.content_sha256,
            "status": self.status,
        }


@dataclass(slots=True)
class ColdMeasurement:
    """What a representation cost before it had been opened."""

    representation: str
    open_latency_ns: int
    first_query_latency_ns: int
    metadata_bytes_loaded: int
    peak_rss_kib: int

    def as_record(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "phase": COLD,
            "open_latency_ns": self.open_latency_ns,
            "first_query_latency_ns": self.first_query_latency_ns,
            "metadata_bytes_loaded": self.metadata_bytes_loaded,
            "peak_rss_kib": self.peak_rss_kib,
        }


@dataclass(slots=True)
class BenchmarkReport:
    """What the timing half produced, without holding a single row.

    The rows themselves went straight to Parquet through the sink; what is
    kept here is counters and the per-round bookkeeping the gate reads.
    """

    cold: list[ColdMeasurement] = field(default_factory=list)
    rounds_completed: int = 0
    warmup_rounds_discarded: int = 0
    failed_queries: int = 0
    rows_written: int = 0
    rows_offered: int = 0
    round_orders: list[dict[str, Any]] = field(default_factory=list)
    order_digest: str = ""
    rounds_by_query_class: dict[str, int] = field(default_factory=dict)
    table_path: str = ""

    def as_record(self) -> dict[str, Any]:
        return {
            "phase": WARM,
            "rounds_completed": self.rounds_completed,
            "measured_rounds": MEASURED_ROUNDS,
            "measured_rounds_by_query_class": dict(
                sorted(self.rounds_by_query_class.items())
            ),
            "warmup_rounds_discarded": self.warmup_rounds_discarded,
            "rows_written": self.rows_written,
            "rows_offered": self.rows_offered,
            "failed_queries": self.failed_queries,
            "round_order_digest": self.order_digest,
            "round_orders": self.round_orders,
            "cold": [item.as_record() for item in self.cold],
        }


def time_query(
    representation: Representation,
    *,
    query_class: str,
    query_id: str,
    round_id: int,
    **kwargs: Any,
) -> Measurement:
    """Run one query and record what it cost."""

    before = _rss_kib()
    cpu0 = time.process_time_ns()
    start = time.perf_counter_ns()
    try:
        fetched = representation.fetch(query_class, query_id, **kwargs)
        # The clock stops here.  What follows is verification, and timing it
        # would add the same ~1.7 ms to every representation and compress the
        # differences the benchmark exists to measure.
        latency = time.perf_counter_ns() - start
        cpu = time.process_time_ns() - cpu0
        status = QUERY_OK
        result = result_from_rows(query_id, fetched)
        rows, payload, digest = (
            result.rows, result.bytes_returned, result.content_sha256
        )
    except Exception:  # noqa: BLE001 - a refusal is a recorded outcome
        latency = time.perf_counter_ns() - start
        cpu = time.process_time_ns() - cpu0
        status, rows, payload, digest = QUERY_FAILED, 0, 0, ""
    return Measurement(
        representation=representation.name,
        query_class=query_class,
        query_id=query_id,
        round_id=round_id,
        latency_ns=latency,
        cpu_time_ns=cpu,
        rows_returned=rows,
        bytes_returned=payload,
        peak_rss_delta=max(0, _rss_kib() - before),
        content_sha256=digest,
        status=status,
    )


def measure_cold(
    factory,
    *,
    name: str,
    first_query: tuple[str, str],
) -> ColdMeasurement:
    """Open an unopened representation in this process and time it.

    A fresh process is what the contract calls cold; this is the part of it
    that can be measured from inside one, and the driver supplies the fresh
    process around it.
    """

    before = _rss_kib()
    representation = factory()
    start = time.perf_counter_ns()
    representation.open()
    opened = time.perf_counter_ns() - start
    query_class, query_id = first_query
    start = time.perf_counter_ns()
    representation.query(query_class, query_id)
    first = time.perf_counter_ns() - start
    peak = _rss_kib()
    representation.close()
    return ColdMeasurement(
        representation=name,
        open_latency_ns=opened,
        first_query_latency_ns=first,
        metadata_bytes_loaded=0,
        peak_rss_kib=max(0, peak - before),
    )


def run_rounds(
    representations: Mapping[str, Representation],
    workload: QueryWorkload,
    sink: Any,
    *,
    query_classes: Sequence[str] = (Q1, Q2, Q3, Q4),
    rounds: int = TOTAL_ROUNDS,
    progress: bool = False,
) -> BenchmarkReport:
    """Run every round, rotating who goes first and reshuffling each time.

    Each measurement goes to the sink as it is taken.  Nothing accumulates:
    at roughly 2.3 million rows, holding them would cost more memory than the
    representations being measured.
    """

    report = BenchmarkReport()
    digest = hashlib.sha256()
    for index in range(rounds):
        round_id = WARMUP_ROUND_ID if index < WARMUP_ROUNDS else (
            index - WARMUP_ROUNDS
        )
        order = representation_order(index)
        # One order per round, handed to every representation, so an
        # ordering effect cannot land on one of them alone.
        orders = {
            Q1: round_order(
                workload.stream_ids, query_class=Q1, round_id=index
            ),
            Q2: round_order(
                workload.window_ids, query_class=Q2, round_id=index
            ),
            Q3: round_order(
                workload.assessment_ids, query_class=Q3, round_id=index
            ),
        }
        digest.update(f"{index}:{','.join(order)}".encode())
        for query_class in query_classes:
            if query_class in orders:
                digest.update(orders[query_class][0].encode("utf-8"))
        report.round_orders.append({
            "round_index": index,
            "round_id": round_id,
            "is_warmup": round_id < 0,
            "representation_order": list(order),
            "first_query_by_class": {
                name: orders[name][0] for name in orders
            },
        })

        for name in order:
            representation = representations[name]
            for query_class in query_classes:
                budget = MEASURED_ROUNDS_BY_QUERY_CLASS.get(
                    query_class, MEASURED_ROUNDS
                )
                if round_id >= budget:
                    continue
                if query_class == Q4:
                    for position, batch in enumerate(workload.batches):
                        size = len(batch)
                        item = time_query(
                            representation, query_class=Q4,
                            query_id=f"batch:{size}:{position:04d}",
                            round_id=round_id, window_ids=batch,
                        )
                        sink.add(item.as_record())
                        if item.status != QUERY_OK:
                            report.failed_queries += 1
                    continue
                for query_id in orders[query_class]:
                    item = time_query(
                        representation, query_class=query_class,
                        query_id=query_id, round_id=round_id,
                    )
                    sink.add(item.as_record())
                    if item.status != QUERY_OK:
                        report.failed_queries += 1
            if progress:
                print(f"  round {index}/{rounds} {name} done", flush=True)
        if round_id < 0:
            report.warmup_rounds_discarded += 1
        else:
            report.rounds_completed += 1
    report.order_digest = digest.hexdigest()
    sink.flush()
    report.rows_written = sink.counters.rows_written
    report.rows_offered = sink.counters.rows_offered
    report.rounds_by_query_class = {
        name: len(rounds_seen)
        for name, rounds_seen in sorted(
            sink.counters.rounds_seen_by_class.items()
        )
    }
    return report


# --- speed ratios ---------------------------------------------------------


def _bootstrap_indices(count: int, draw: int) -> list[list[int]]:
    """Deterministic resamples from keyed SHA-256, never an RNG."""

    out = []
    for index in range(draw):
        picks = []
        stream = hashlib.sha256(f"p05-bootstrap-{index}".encode()).digest()
        cursor = 0
        while len(picks) < count:
            if cursor + 4 > len(stream):
                stream = hashlib.sha256(stream).digest()
                cursor = 0
            value = int.from_bytes(stream[cursor:cursor + 4], "big")
            cursor += 4
            picks.append(value % count)
        out.append(picks)
    return out


def speed_ratios(
    medians: Mapping[str, Mapping[str, float]],
    *,
    baseline: str,
    system: str,
    query_class: str = Q2,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Baseline over system, bootstrapped across query ids.

    ``medians`` carries one number per (representation, query id): that
    query's median across its rounds.  The resampling unit is therefore the
    query identifier, not the individual timing -- ten rounds of one query are
    one workload item observed ten times, and resampling the observations
    would let them pose as ten independent queries and shrink the interval to
    something unearned.
    """

    left = medians.get(baseline, {})
    right = medians.get(system, {})
    ratios: list[float] = []
    for query_id in sorted(set(left) & set(right)):
        bottom = right[query_id]
        if bottom <= 0:
            continue
        ratios.append(left[query_id] / bottom)

    if not ratios:
        return {
            "baseline": baseline, "system": system,
            "query_class": query_class, "queries": 0,
            "median_ratio": 0.0, "confidence_low": 0.0,
            "confidence_high": 0.0, "resamples": 0,
            "bootstrap_unit": "QUERY_ID",
        }

    drawn = []
    for picks in _bootstrap_indices(len(ratios), resamples):
        drawn.append(statistics.median([ratios[index] for index in picks]))
    drawn.sort()
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    low = drawn[int(tail * (len(drawn) - 1))]
    high = drawn[int((1.0 - tail) * (len(drawn) - 1))]
    return {
        "baseline": baseline,
        "system": system,
        "query_class": query_class,
        "queries": len(ratios),
        "median_ratio": statistics.median(ratios),
        "confidence_low": low,
        "confidence_high": high,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "resamples": resamples,
        "bootstrap_unit": "QUERY_ID",
        "note": (
            "ratio above 1.0 means the baseline took longer than "
            f"{system}; below 1.0 means it was faster"
        ),
    }


def all_speed_ratios(
    medians: Mapping[str, Mapping[str, float]],
    *,
    system: str,
    query_class: str = Q2,
) -> list[dict[str, Any]]:
    return [
        speed_ratios(
            medians, baseline=name, system=system, query_class=query_class
        )
        for name in REPRESENTATIONS if name != system
    ]


__all__ = [
    "QUERY_FAILED",
    "QUERY_OK",
    "WARMUP_ROUND_ID",
    "BenchmarkReport",
    "ColdMeasurement",
    "Measurement",
    "all_speed_ratios",
    "measure_cold",
    "run_rounds",
    "speed_ratios",
    "time_query",
]
