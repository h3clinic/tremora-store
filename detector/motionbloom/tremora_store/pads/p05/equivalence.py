"""Whether the four representations are answering the same question at all.

This runs before any timing and is worth more than the timing.  A faster
representation that returns different rows is not a benchmark result, so the
comparison is settled first: every one of the 50,676 windows, every stream and
every assessment, hashed to canonical row identity and required to agree.

The reference is chosen by position rather than by preference -- whichever
representation the contract lists first -- so the comparison is not anchored
on the system under test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contract import (
    Q1,
    Q2,
    Q3,
    REPRESENTATIONS,
)
from .representations import Representation
from .rows import compare, result_from_rows

EQUIVALENT = "REPRESENTATIONS_EQUIVALENT"
NOT_EQUIVALENT = "REPRESENTATIONS_DISAGREE"
QUERY_OK = "QUERY_OK"
QUERY_FAILED = "QUERY_FAILED"

#: The reference every other representation is compared against.  It is the
#: first in the contract's own ordering, which is B0 -- the published source
#: text -- so agreement is agreement with the release rather than with the
#: system under test.
REFERENCE = REPRESENTATIONS[0]


@dataclass(slots=True)
class EquivalenceReport:
    """What agreed, what did not, and where."""

    reference: str = REFERENCE
    windows_compared: int = 0
    streams_compared: int = 0
    assessments_compared: int = 0
    comparisons: int = 0
    content_mismatches: int = 0
    row_count_mismatches: int = 0
    time_mismatches: int = 0
    sensor_value_mismatches: int = 0
    failed_queries: int = 0
    rows_reconciled: int = 0
    per_representation: dict[str, dict[str, int]] = field(
        default_factory=dict
    )
    window_content_sha256: str = ""
    failures: list[str] = field(default_factory=list)

    @property
    def equivalent(self) -> bool:
        return (
            self.comparisons > 0
            and self.content_mismatches == 0
            and self.row_count_mismatches == 0
            and self.time_mismatches == 0
            and self.sensor_value_mismatches == 0
            and self.failed_queries == 0
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "status": EQUIVALENT if self.equivalent else NOT_EQUIVALENT,
            "reference_representation": self.reference,
            "windows_compared": self.windows_compared,
            "streams_compared": self.streams_compared,
            "assessments_compared": self.assessments_compared,
            "comparisons": self.comparisons,
            "rows_reconciled": self.rows_reconciled,
            "content_mismatches": self.content_mismatches,
            "row_count_mismatches": self.row_count_mismatches,
            "time_mismatches": self.time_mismatches,
            "sensor_value_mismatches": self.sensor_value_mismatches,
            "failed_queries": self.failed_queries,
            "window_content_sha256": self.window_content_sha256,
            "per_representation": {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(self.per_representation.items())
            },
            "failure_count": len(self.failures),
            "failures": sorted(self.failures)[:64],
        }


def _fold(report: EquivalenceReport, name: str, key: str) -> None:
    counts = report.per_representation.setdefault(
        name, {
            "content_mismatches": 0, "row_count_mismatches": 0,
            "time_mismatches": 0, "sensor_value_mismatches": 0,
            "failed_queries": 0, "queries": 0,
        }
    )
    counts[key] += 1


def compare_all(
    representations: Mapping[str, Representation],
    *,
    window_ids: Sequence[str],
    stream_ids: Sequence[str],
    assessment_ids: Sequence[str],
    progress: bool = False,
) -> EquivalenceReport:
    """Reconcile every query across every representation."""

    import hashlib

    report = EquivalenceReport()
    if REFERENCE not in representations:
        raise KeyError(f"the reference {REFERENCE} was not built")
    reference = representations[REFERENCE]
    others = [
        (name, representations[name])
        for name in REPRESENTATIONS if name != REFERENCE
    ]

    windows_digest = hashlib.sha256()
    for query_class, identifiers, method, counter in (
        (Q2, window_ids, "window_rows", "windows_compared"),
        (Q1, stream_ids, "stream_rows", "streams_compared"),
        (Q3, assessment_ids, "assessment_rows", "assessments_compared"),
    ):
        for position, identifier in enumerate(identifiers):
            try:
                expected = result_from_rows(
                    identifier, getattr(reference, method)(identifier)
                )
            except Exception as exc:  # noqa: BLE001 - any refusal counts
                report.failed_queries += 1
                _fold(report, REFERENCE, "failed_queries")
                report.failures.append(
                    f"{REFERENCE}:{query_class}:{identifier}: {exc}"
                )
                continue
            _fold(report, REFERENCE, "queries")
            setattr(report, counter, getattr(report, counter) + 1)
            report.rows_reconciled += expected.rows
            if query_class == Q2:
                windows_digest.update(identifier.encode("utf-8"))
                windows_digest.update(b"\x1f")
                windows_digest.update(expected.content_sha256.encode("ascii"))
                windows_digest.update(b"\x1e")

            for name, representation in others:
                try:
                    got = result_from_rows(
                        identifier, getattr(representation, method)(identifier)
                    )
                except Exception as exc:  # noqa: BLE001
                    report.failed_queries += 1
                    _fold(report, name, "failed_queries")
                    report.failures.append(
                        f"{name}:{query_class}:{identifier}: {exc}"
                    )
                    continue
                _fold(report, name, "queries")
                report.comparisons += 1
                verdict = compare(expected, got)
                if not verdict["content_match"]:
                    report.content_mismatches += 1
                    _fold(report, name, "content_mismatches")
                    report.failures.append(
                        f"{name}:{query_class}:{identifier}: content"
                    )
                if not verdict["row_count_match"]:
                    report.row_count_mismatches += 1
                    _fold(report, name, "row_count_mismatches")
                if not verdict["time_match"]:
                    report.time_mismatches += 1
                    _fold(report, name, "time_mismatches")
                if not verdict["sensor_value_match"]:
                    report.sensor_value_mismatches += 1
                    _fold(report, name, "sensor_value_mismatches")
            if progress and position % 5000 == 0:
                print(
                    f"  {query_class} {position}/{len(identifiers)}",
                    flush=True,
                )
    report.window_content_sha256 = windows_digest.hexdigest()
    return report


__all__ = [
    "EQUIVALENT",
    "NOT_EQUIVALENT",
    "QUERY_FAILED",
    "QUERY_OK",
    "REFERENCE",
    "EquivalenceReport",
    "compare_all",
]
