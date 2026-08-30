"""The frozen query workload, generated before anything is timed.

Order matters to a latency benchmark, so it is decided here and hashed, not
decided by whichever representation happens to run first.  Selection and
shuffling use keyed SHA-256 rather than a random number generator: the order
is then rebuildable from the contract alone, without carrying a particular
interpreter's generator along with it.

Every representation sees the same order within a round, and the order the
representations themselves run in rotates between rounds.  Both facts are
produced here and checked by the gate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .contract import (
    BATCH_SIZES,
    BATCHES_PER_SIZE,
    Q1,
    Q2,
    Q3,
    Q4,
    QUERY_CLASSES,
    REPRESENTATIONS,
    TOTAL_ROUNDS,
    WORKLOAD_SELECTION,
)

WORKLOAD_KEY = b"tremora-pads-p05-workload-v1"


class WorkloadError(ValueError):
    """Raised when the workload cannot be built as the contract declares."""


def _rank(key: bytes, token: str) -> bytes:
    """A stable sort key for one identifier under one purpose."""

    return hashlib.sha256(key + b"\x1f" + token.encode("utf-8")).digest()


def deterministic_order(
    identifiers: Sequence[str], *, purpose: str
) -> tuple[str, ...]:
    """Shuffle deterministically by keyed hash, not by a seeded generator."""

    key = WORKLOAD_KEY + b"\x1e" + purpose.encode("utf-8")
    return tuple(
        sorted(identifiers, key=lambda token: (_rank(key, token), token))
    )


@dataclass(frozen=True, slots=True)
class QueryWorkload:
    """Every query this benchmark will issue, in its frozen base order."""

    stream_ids: tuple[str, ...]
    window_ids: tuple[str, ...]
    assessment_ids: tuple[str, ...]
    batches: tuple[tuple[str, ...], ...]
    batch_sizes: tuple[int, ...]
    #: Batch sizes larger than the corpus, skipped rather than truncated.
    skipped_batch_sizes: tuple[int, ...] = ()

    @property
    def query_ids(self) -> dict[str, tuple[str, ...]]:
        return {
            Q1: self.stream_ids,
            Q2: self.window_ids,
            Q3: self.assessment_ids,
            Q4: tuple(
                f"batch:{size}:{index:04d}"
                for size, index in zip(
                    self.batch_sizes,
                    _batch_indexes(self.batch_sizes),
                    strict=True,
                )
            ),
        }

    def counts(self) -> dict[str, int]:
        return {name: len(ids) for name, ids in self.query_ids.items()}

    def content_sha256(self) -> str:
        """One hash over the whole workload, taken before any timing runs."""

        digest = hashlib.sha256()
        digest.update(WORKLOAD_SELECTION.encode("ascii"))
        for name in QUERY_CLASSES:
            digest.update(b"\x1e")
            digest.update(name.encode("ascii"))
            for identifier in self.query_ids[name]:
                digest.update(b"\x1f")
                digest.update(identifier.encode("utf-8"))
        for batch in self.batches:
            digest.update(b"\x1d")
            for window_id in batch:
                digest.update(b"\x1f")
                digest.update(window_id.encode("utf-8"))
        return digest.hexdigest()

    def as_record(self) -> dict[str, Any]:
        return {
            "workload_selection": WORKLOAD_SELECTION,
            "workload_content_sha256": self.content_sha256(),
            "query_counts": self.counts(),
            "batch_sizes": list(BATCH_SIZES),
            "batches_per_size": BATCHES_PER_SIZE,
            "batch_count": len(self.batches),
            "batched_window_fetches": sum(
                len(batch) for batch in self.batches
            ),
            "skipped_batch_sizes": list(self.skipped_batch_sizes),
        }


def _batch_indexes(sizes: Sequence[int]) -> list[int]:
    seen: dict[int, int] = {}
    out: list[int] = []
    for size in sizes:
        out.append(seen.get(size, 0))
        seen[size] = seen.get(size, 0) + 1
    return out


def build_workload(
    *,
    stream_ids: Sequence[str],
    window_ids: Sequence[str],
    assessment_ids: Sequence[str],
) -> QueryWorkload:
    """Freeze the workload from the P0.2.1 identifiers."""

    if not stream_ids or not window_ids or not assessment_ids:
        raise WorkloadError("the workload needs streams, windows and pairs")

    streams = deterministic_order(stream_ids, purpose=Q1)
    windows = deterministic_order(window_ids, purpose=Q2)
    assessments = deterministic_order(assessment_ids, purpose=Q3)

    # Batches are drawn from their own deterministic ordering, so a batch is
    # a genuine scatter across the corpus rather than a contiguous run that
    # any range-indexed representation would answer unrealistically well.
    pool = deterministic_order(window_ids, purpose=Q4)
    batches: list[tuple[str, ...]] = []
    sizes: list[int] = []
    skipped: list[int] = []
    cursor = 0
    for size in BATCH_SIZES:
        if size > len(pool):
            # A batch bigger than the corpus would be silently truncated and
            # then reported as if it were that size.  It is skipped and the
            # skip is published; on the real corpus nothing is skipped.
            skipped.append(size)
            continue
        for _ in range(BATCHES_PER_SIZE):
            if cursor + size > len(pool):
                cursor = 0
            batches.append(tuple(pool[cursor:cursor + size]))
            sizes.append(size)
            cursor += size
    return QueryWorkload(
        stream_ids=streams,
        window_ids=windows,
        assessment_ids=assessments,
        batches=tuple(batches),
        batch_sizes=tuple(sizes),
        skipped_batch_sizes=tuple(skipped),
    )


def round_order(
    identifiers: Sequence[str], *, query_class: str, round_id: int
) -> tuple[str, ...]:
    """This round's independently shuffled order for one query class.

    Every representation is handed this same order inside the round, so an
    ordering effect lands on all four or on none.
    """

    return deterministic_order(
        identifiers, purpose=f"{query_class}:round:{round_id}"
    )


def representation_order(round_id: int) -> tuple[str, ...]:
    """Rotate which representation runs first, one step per round."""

    count = len(REPRESENTATIONS)
    offset = round_id % count
    return tuple(REPRESENTATIONS[(offset + i) % count] for i in range(count))


def rotation_is_complete(rounds: int = TOTAL_ROUNDS) -> bool:
    """Every representation must lead at least once across the rounds."""

    leaders = {representation_order(index)[0] for index in range(rounds)}
    return leaders == set(REPRESENTATIONS)


__all__ = [
    "WORKLOAD_KEY",
    "QueryWorkload",
    "WorkloadError",
    "build_workload",
    "deterministic_order",
    "representation_order",
    "rotation_is_complete",
    "round_order",
]
