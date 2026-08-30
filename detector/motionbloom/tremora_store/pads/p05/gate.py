"""The twenty-five PADS-P0.5 hard-gate conditions.

Notice what is absent.  There is no condition requiring TremoraStore to be
fastest or smallest, and none is implied by any other.  A run in which a
baseline wins on every primary outcome satisfies all twenty-five, because the
question is whether the comparison was conducted honestly, not who won it.

The two fairness conditions are decided by inspecting what was built rather
than by trusting that it was built correctly: the HDF5 file is opened and its
indexes read back, and B1's duplication is counted from what it actually
wrote.  An implementation that promised both and delivered neither would pass
a contract check and fail these.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .contract import (
    B0,
    B1,
    B2,
    GATE_NO_GO,
    GATE_PASS,
    M1,
    MEASURED_ROUNDS_BY_QUERY_CLASS,
    REPRESENTATIONS,
    WARMUP_ROUNDS,
    WINDOWS,
)

P02_1_DEPENDENCY_VERIFIED = "P02_1_DEPENDENCY_VERIFIED"
P03_AND_P04_EVIDENCE_UNCHANGED = "P03_AND_P04_EVIDENCE_UNCHANGED"
ALL_BASELINES_USE_IDENTICAL_SOURCE_MANIFEST = (
    "ALL_BASELINES_USE_IDENTICAL_SOURCE_MANIFEST"
)
ALL_BASELINES_SUPPORT_EQUIVALENT_QUERY_SEMANTICS = (
    "ALL_BASELINES_SUPPORT_EQUIVALENT_QUERY_SEMANTICS"
)
ALL_50676_WINDOWS_RECONCILED = "ALL_50676_WINDOWS_RECONCILED"
B0_CONTENT_EQUIVALENT = "B0_CONTENT_EQUIVALENT"
B1_CONTENT_EQUIVALENT = "B1_CONTENT_EQUIVALENT"
B2_CONTENT_EQUIVALENT = "B2_CONTENT_EQUIVALENT"
M1_CONTENT_EQUIVALENT = "M1_CONTENT_EQUIVALENT"
ZERO_CROSS_REPRESENTATION_CONTENT_MISMATCHES = (
    "ZERO_CROSS_REPRESENTATION_CONTENT_MISMATCHES"
)
B1_PHYSICALLY_DUPLICATES_OVERLAPPING_WINDOW_SAMPLES = (
    "B1_PHYSICALLY_DUPLICATES_OVERLAPPING_WINDOW_SAMPLES"
)
M1_WINDOW_INDEX_DOES_NOT_DUPLICATE_SOURCE_SAMPLES = (
    "M1_WINDOW_INDEX_DOES_NOT_DUPLICATE_SOURCE_SAMPLES"
)
HDF5_HAS_FAIR_RANGE_INDEX = "HDF5_HAS_FAIR_RANGE_INDEX"
COMPRESSION_POLICIES_DECLARED = "COMPRESSION_POLICIES_DECLARED"
QUERY_WORKLOAD_FROZEN_BEFORE_TIMING = "QUERY_WORKLOAD_FROZEN_BEFORE_TIMING"
WARMUP_EXCLUDED = "WARMUP_EXCLUDED"
REPRESENTATION_ORDER_ROTATED = "REPRESENTATION_ORDER_ROTATED"
MEASURED_ROUNDS_COMPLETE = "MEASURED_ROUNDS_COMPLETE"
STORAGE_COUNTS_RECONCILED = "STORAGE_COUNTS_RECONCILED"
LATENCY_RECORD_COUNTS_RECONCILED = "LATENCY_RECORD_COUNTS_RECONCILED"
NO_FAILED_BENCHMARK_QUERIES = "NO_FAILED_BENCHMARK_QUERIES"
INDEPENDENT_BENCHMARK_REPRODUCTION_VERIFIED = (
    "INDEPENDENT_BENCHMARK_REPRODUCTION_VERIFIED"
)
NO_NEW_SIGNAL_PROCESSING = "NO_NEW_SIGNAL_PROCESSING"
NO_CLASSIFICATION = "NO_CLASSIFICATION"
NO_VIDEO_ASSOCIATION = "NO_VIDEO_ASSOCIATION"

GATE_CONDITIONS: tuple[str, ...] = (
    P02_1_DEPENDENCY_VERIFIED,
    P03_AND_P04_EVIDENCE_UNCHANGED,
    ALL_BASELINES_USE_IDENTICAL_SOURCE_MANIFEST,
    ALL_BASELINES_SUPPORT_EQUIVALENT_QUERY_SEMANTICS,
    ALL_50676_WINDOWS_RECONCILED,
    B0_CONTENT_EQUIVALENT,
    B1_CONTENT_EQUIVALENT,
    B2_CONTENT_EQUIVALENT,
    M1_CONTENT_EQUIVALENT,
    ZERO_CROSS_REPRESENTATION_CONTENT_MISMATCHES,
    B1_PHYSICALLY_DUPLICATES_OVERLAPPING_WINDOW_SAMPLES,
    M1_WINDOW_INDEX_DOES_NOT_DUPLICATE_SOURCE_SAMPLES,
    HDF5_HAS_FAIR_RANGE_INDEX,
    COMPRESSION_POLICIES_DECLARED,
    QUERY_WORKLOAD_FROZEN_BEFORE_TIMING,
    WARMUP_EXCLUDED,
    REPRESENTATION_ORDER_ROTATED,
    MEASURED_ROUNDS_COMPLETE,
    STORAGE_COUNTS_RECONCILED,
    LATENCY_RECORD_COUNTS_RECONCILED,
    NO_FAILED_BENCHMARK_QUERIES,
    INDEPENDENT_BENCHMARK_REPRODUCTION_VERIFIED,
    NO_NEW_SIGNAL_PROCESSING,
    NO_CLASSIFICATION,
    NO_VIDEO_ASSOCIATION,
)

#: Deliberately not conditions.  Written down so that adding one later is a
#: visible change to this file rather than a quiet drift in what the gate
#: means.
NEVER_GATE_CONDITIONS: tuple[str, ...] = (
    "M1_MUST_BE_FASTEST",
    "M1_MUST_BE_SMALLEST",
)

#: What reproduction means for a benchmark.  Two honest executions agree
#: exactly on the deterministic half -- workload, storage accounting, result
#: content hashes, counts, baseline identities, gate facts -- and each
#: completes its own timing.  Requiring identical latency tables would be
#: requiring the machine not to be a machine.
REPRODUCTION_VERIFIED = "DETERMINISTIC_EVIDENCE_IDENTICAL_PADS_P05_PASS"
REPRODUCTION_NOT_ATTEMPTED = "REPRODUCTION_NOT_ATTEMPTED"


class PadsP05GateError(ValueError):
    """Raised when the gate is asked to evaluate incoherent facts."""


@dataclass(frozen=True, slots=True)
class GateCondition:
    name: str
    satisfied: bool
    detail: str

    def as_record(self) -> dict[str, Any]:
        return {
            "condition": self.name,
            "satisfied": self.satisfied,
            "detail": self.detail,
        }


@dataclass(slots=True)
class PadsP05GateFacts:
    """Everything the twenty-five conditions are decided from."""

    dependency_status: str = "P02_DEPENDENCY_NOT_EVALUATED"
    p03_evidence_sha256: str = ""
    p04_evidence_sha256: str = ""
    pinned_p03_evidence_sha256: str = ""
    pinned_p04_evidence_sha256: str = ""
    source_manifest_sha256: str = ""
    manifest_by_representation: dict[str, str] = field(default_factory=dict)
    query_classes_supported: dict[str, int] = field(default_factory=dict)
    expected_query_classes: int = 4
    windows_reconciled: int = 0
    expected_windows: int = WINDOWS
    streams_reconciled: int = 0
    assessments_reconciled: int = 0
    per_representation_mismatches: dict[str, int] = field(
        default_factory=dict
    )
    content_mismatches: int = 0
    row_count_mismatches: int = 0
    time_mismatches: int = 0
    sensor_value_mismatches: int = 0
    b1_stored_instances: int = 0
    b1_unique_samples: int = 0
    m1_stored_instances: int = 0
    m1_unique_samples: int = 0
    hdf5_indexes_present: tuple[str, ...] = ()
    hdf5_indexes_required: tuple[str, ...] = ()
    hdf5_window_index_entries: int = 0
    hdf5_chunked: bool = False
    compression_declared: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    workload_hash_before_timing: str = ""
    workload_hash_after_timing: str = ""
    warmup_rounds_discarded: int = 0
    warmup_records_in_summary: int = 0
    representation_orders: tuple[tuple[str, ...], ...] = ()
    rounds_completed_by_class: dict[str, int] = field(default_factory=dict)
    expected_rounds_by_class: dict[str, int] = field(
        default_factory=lambda: dict(MEASURED_ROUNDS_BY_QUERY_CLASS)
    )
    storage_reconciled: bool = False
    storage_problems: tuple[str, ...] = ()
    latency_records: int = 0
    expected_latency_records: int = 0
    failed_queries: int = 0
    reproduction_status: str = REPRODUCTION_NOT_ATTEMPTED
    timing_executions_completed: int = 0
    timing_executions_required: int = 2
    new_signal_processing_outputs: int = 0
    signal_processing_declared: bool = False
    emitted_forbidden_artifacts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateResult:
    conditions: tuple[GateCondition, ...]
    gate_status: str

    @property
    def satisfied(self) -> bool:
        return self.gate_status == GATE_PASS

    def as_record(self) -> dict[str, Any]:
        return {
            "gate_status": self.gate_status,
            "gate_conditions": [
                condition.as_record() for condition in self.conditions
            ],
            "gate_conditions_satisfied": sum(
                1 for condition in self.conditions if condition.satisfied
            ),
            "gate_conditions_total": len(self.conditions),
            "conditions_deliberately_absent": list(NEVER_GATE_CONDITIONS),
        }


def _equivalent(facts: PadsP05GateFacts, name: str) -> GateCondition:
    """One representation answered everything, and agreed on all of it."""

    mismatches = facts.per_representation_mismatches.get(name)
    answered = facts.query_classes_supported.get(name, 0)
    return GateCondition(
        f"{name.split('_')[0]}_CONTENT_EQUIVALENT",
        mismatches == 0
        and answered == facts.expected_query_classes
        and facts.windows_reconciled > 0,
        f"{mismatches} mismatches over {answered} query classes"
        if mismatches is not None
        else "not compared",
    )


def evaluate_gate(facts: PadsP05GateFacts) -> GateResult:
    """Evaluate every condition; all twenty-five are required."""

    manifests = set(facts.manifest_by_representation.values())
    rounds_complete = bool(facts.rounds_completed_by_class) and all(
        facts.rounds_completed_by_class.get(name, 0) >= expected
        for name, expected in facts.expected_rounds_by_class.items()
    )
    conditions = (
        GateCondition(
            P02_1_DEPENDENCY_VERIFIED,
            facts.dependency_status == "P02_DEPENDENCY_VERIFIED",
            facts.dependency_status,
        ),
        GateCondition(
            P03_AND_P04_EVIDENCE_UNCHANGED,
            bool(facts.p03_evidence_sha256)
            and bool(facts.p04_evidence_sha256)
            and facts.p03_evidence_sha256 == facts.pinned_p03_evidence_sha256
            and facts.p04_evidence_sha256 == facts.pinned_p04_evidence_sha256,
            f"P0.3 {facts.p03_evidence_sha256[:12]}, "
            f"P0.4 {facts.p04_evidence_sha256[:12]}",
        ),
        GateCondition(
            ALL_BASELINES_USE_IDENTICAL_SOURCE_MANIFEST,
            len(facts.manifest_by_representation) == len(REPRESENTATIONS)
            and len(manifests) == 1
            and bool(facts.source_manifest_sha256)
            and manifests == {facts.source_manifest_sha256},
            f"{len(manifests)} distinct manifests over "
            f"{len(facts.manifest_by_representation)} representations",
        ),
        GateCondition(
            ALL_BASELINES_SUPPORT_EQUIVALENT_QUERY_SEMANTICS,
            len(facts.query_classes_supported) == len(REPRESENTATIONS)
            and all(
                count == facts.expected_query_classes
                for count in facts.query_classes_supported.values()
            ),
            f"{sorted(facts.query_classes_supported.values())} of "
            f"{facts.expected_query_classes} query classes answered",
        ),
        GateCondition(
            ALL_50676_WINDOWS_RECONCILED,
            facts.windows_reconciled == facts.expected_windows
            and facts.expected_windows > 0,
            f"{facts.windows_reconciled} of {facts.expected_windows} windows, "
            f"{facts.streams_reconciled} streams, "
            f"{facts.assessments_reconciled} assessments",
        ),
        _equivalent(facts, B0),
        _equivalent(facts, B1),
        _equivalent(facts, B2),
        _equivalent(facts, M1),
        GateCondition(
            ZERO_CROSS_REPRESENTATION_CONTENT_MISMATCHES,
            facts.windows_reconciled > 0
            and facts.content_mismatches == 0
            and facts.row_count_mismatches == 0
            and facts.time_mismatches == 0
            and facts.sensor_value_mismatches == 0,
            f"{facts.content_mismatches} content, "
            f"{facts.row_count_mismatches} row-count, "
            f"{facts.time_mismatches} time, "
            f"{facts.sensor_value_mismatches} sensor-value",
        ),
        GateCondition(
            B1_PHYSICALLY_DUPLICATES_OVERLAPPING_WINDOW_SAMPLES,
            # Counted from what B1 wrote.  A baseline that quietly
            # deduplicated would make the duplication claim measure nothing.
            facts.b1_unique_samples > 0
            and facts.b1_stored_instances > facts.b1_unique_samples,
            f"{facts.b1_stored_instances} instances for "
            f"{facts.b1_unique_samples} unique samples",
        ),
        GateCondition(
            M1_WINDOW_INDEX_DOES_NOT_DUPLICATE_SOURCE_SAMPLES,
            facts.m1_unique_samples > 0
            and facts.m1_stored_instances == facts.m1_unique_samples,
            f"{facts.m1_stored_instances} instances for "
            f"{facts.m1_unique_samples} unique samples",
        ),
        GateCondition(
            HDF5_HAS_FAIR_RANGE_INDEX,
            # Read back out of the file, not taken from the build's word.
            bool(facts.hdf5_indexes_required)
            and set(facts.hdf5_indexes_required).issubset(
                set(facts.hdf5_indexes_present)
            )
            and facts.hdf5_window_index_entries == facts.expected_windows
            and facts.hdf5_chunked,
            f"{sorted(facts.hdf5_indexes_present)} present, "
            f"{facts.hdf5_window_index_entries} window offsets, "
            f"chunked={facts.hdf5_chunked}",
        ),
        GateCondition(
            COMPRESSION_POLICIES_DECLARED,
            len(facts.compression_declared) == len(REPRESENTATIONS)
            and all(
                "codec" in policy
                for policy in facts.compression_declared.values()
            ),
            f"{len(facts.compression_declared)} policies declared",
        ),
        GateCondition(
            QUERY_WORKLOAD_FROZEN_BEFORE_TIMING,
            bool(facts.workload_hash_before_timing)
            and facts.workload_hash_before_timing
            == facts.workload_hash_after_timing,
            f"{facts.workload_hash_before_timing[:16]} before, "
            f"{facts.workload_hash_after_timing[:16]} after",
        ),
        GateCondition(
            WARMUP_EXCLUDED,
            facts.warmup_rounds_discarded == WARMUP_ROUNDS
            and facts.warmup_records_in_summary == 0,
            f"{facts.warmup_rounds_discarded} discarded, "
            f"{facts.warmup_records_in_summary} warm-up records reached the "
            "summary",
        ),
        GateCondition(
            REPRESENTATION_ORDER_ROTATED,
            len(facts.representation_orders) > 1
            and len({
                order[0] for order in facts.representation_orders
            }) == len(REPRESENTATIONS)
            and all(
                sorted(order) == sorted(REPRESENTATIONS)
                for order in facts.representation_orders
            ),
            f"{len({order[0] for order in facts.representation_orders})} "
            f"distinct leaders over {len(facts.representation_orders)} rounds",
        ),
        GateCondition(
            MEASURED_ROUNDS_COMPLETE,
            rounds_complete,
            f"{dict(sorted(facts.rounds_completed_by_class.items()))} against "
            f"{dict(sorted(facts.expected_rounds_by_class.items()))}",
        ),
        GateCondition(
            STORAGE_COUNTS_RECONCILED,
            facts.storage_reconciled and not facts.storage_problems,
            "reconciled" if facts.storage_reconciled
            else f"{list(facts.storage_problems)[:3]}",
        ),
        GateCondition(
            LATENCY_RECORD_COUNTS_RECONCILED,
            facts.expected_latency_records > 0
            and facts.latency_records == facts.expected_latency_records,
            f"{facts.latency_records} records against "
            f"{facts.expected_latency_records} expected",
        ),
        GateCondition(
            NO_FAILED_BENCHMARK_QUERIES,
            facts.latency_records > 0 and facts.failed_queries == 0,
            f"{facts.failed_queries} failed queries",
        ),
        GateCondition(
            INDEPENDENT_BENCHMARK_REPRODUCTION_VERIFIED,
            # Two things, not one: the deterministic evidence is identical
            # across the two processes, and both actually completed their own
            # timing.  Latencies differ between them and are published rather
            # than hashed, so identity of the timing tables is neither
            # required nor meaningful.
            facts.reproduction_status == REPRODUCTION_VERIFIED
            and facts.timing_executions_completed
            >= facts.timing_executions_required,
            f"{facts.reproduction_status}; "
            f"{facts.timing_executions_completed} of "
            f"{facts.timing_executions_required} timing executions complete",
        ),
        GateCondition(
            NO_NEW_SIGNAL_PROCESSING,
            facts.signal_processing_declared
            and facts.new_signal_processing_outputs == 0,
            f"{facts.new_signal_processing_outputs} signal-processing outputs",
        ),
        GateCondition(
            NO_CLASSIFICATION,
            bool(facts.emitted_forbidden_artifacts)
            and facts.emitted_forbidden_artifacts.get(
                "classification_tables", 1
            ) == 0
            and facts.emitted_forbidden_artifacts.get(
                "diagnosis_tables", 1
            ) == 0
            and facts.emitted_forbidden_artifacts.get(
                "severity_tables", 1
            ) == 0,
            "no classification, diagnosis or severity tables"
            if facts.emitted_forbidden_artifacts else "the screen did not run",
        ),
        GateCondition(
            NO_VIDEO_ASSOCIATION,
            bool(facts.emitted_forbidden_artifacts)
            and facts.emitted_forbidden_artifacts.get(
                "video_association_tables", 1
            ) == 0,
            "no video-association tables"
            if facts.emitted_forbidden_artifacts else "the screen did not run",
        ),
    )
    names = tuple(condition.name for condition in conditions)
    if names != GATE_CONDITIONS:  # pragma: no cover - contract guard
        raise PadsP05GateError("gate condition set drifted from the contract")
    status = (
        GATE_PASS
        if all(condition.satisfied for condition in conditions)
        else GATE_NO_GO
    )
    return GateResult(conditions=conditions, gate_status=status)


def failing_conditions(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(item["condition"])
        for item in record.get("gate_conditions", ())
        if not item.get("satisfied")
    )


__all__ = [
    "GATE_CONDITIONS",
    "NEVER_GATE_CONDITIONS",
    "REPRODUCTION_NOT_ATTEMPTED",
    "REPRODUCTION_VERIFIED",
    "GateCondition",
    "GateResult",
    "PadsP05GateError",
    "PadsP05GateFacts",
    "evaluate_gate",
    "failing_conditions",
]
