"""The sixteen PADS-P0.3 hard-gate conditions.

Several are decided by positive probes rather than by the absence of a
complaint: the nominal-grid condition counts windows whose stored timestamps
*would* have differed from an ordinal/rate substitution, the Nyquist condition
recomputes the limit from each stream's own cadence, and the kernel controls
are run in this process rather than deferred to a test suite that may not have
been executed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .contract import GATE_NO_GO, GATE_PASS
from .dependency import DEPENDENCY_VERIFIED
from .kernel_controls import CONTROLS_PASS

P02_1_DEPENDENCY_VERIFIED = "P02_1_DEPENDENCY_VERIFIED"
FREQUENCY_GRID_FROZEN = "FREQUENCY_GRID_FROZEN"
WORKLOAD_SELECTION_DETERMINISTIC = "WORKLOAD_SELECTION_DETERMINISTIC"
ONE_CANONICAL_WINDOW_SELECTED_PER_ELIGIBLE_STREAM = (
    "ONE_CANONICAL_WINDOW_SELECTED_PER_ELIGIBLE_STREAM"
)
SOURCE_TIME_USED_FOR_EVERY_SPECTRUM = "SOURCE_TIME_USED_FOR_EVERY_SPECTRUM"
NO_NOMINAL_GRID_TIMESTAMP_SUBSTITUTION = (
    "NO_NOMINAL_GRID_TIMESTAMP_SUBSTITUTION"
)
DT_REF_USED_FOR_CADENCE_AND_NYQUIST = "DT_REF_USED_FOR_CADENCE_AND_NYQUIST"
NO_FIXED_SAMPLE_COUNT_ASSUMPTION = "NO_FIXED_SAMPLE_COUNT_ASSUMPTION"
RAW_AXES_PRESERVED = "RAW_AXES_PRESERVED"
NO_VECTOR_MAGNITUDE_PRIMARY_SIGNAL = "NO_VECTOR_MAGNITUDE_PRIMARY_SIGNAL"
SOURCE_AND_REPLAY_ROWS_IDENTICAL = "SOURCE_AND_REPLAY_ROWS_IDENTICAL"
SOURCE_AND_REPLAY_SPECTRA_IDENTICAL = "SOURCE_AND_REPLAY_SPECTRA_IDENTICAL"
ALL_SPECTRAL_OUTPUT_ROWS_RECONCILED = "ALL_SPECTRAL_OUTPUT_ROWS_RECONCILED"
SYNTHETIC_KERNEL_CONTROLS_PASS = "SYNTHETIC_KERNEL_CONTROLS_PASS"
INDEPENDENT_MATERIALIZATION_REPRODUCED = (
    "INDEPENDENT_MATERIALIZATION_REPRODUCED"
)
NO_RESAMPLING_RATE_ABLATION_OR_VIDEO_ARTIFACTS = (
    "NO_RESAMPLING_RATE_ABLATION_OR_VIDEO_ARTIFACTS"
)

GATE_CONDITIONS: tuple[str, ...] = (
    P02_1_DEPENDENCY_VERIFIED,
    FREQUENCY_GRID_FROZEN,
    WORKLOAD_SELECTION_DETERMINISTIC,
    ONE_CANONICAL_WINDOW_SELECTED_PER_ELIGIBLE_STREAM,
    SOURCE_TIME_USED_FOR_EVERY_SPECTRUM,
    NO_NOMINAL_GRID_TIMESTAMP_SUBSTITUTION,
    DT_REF_USED_FOR_CADENCE_AND_NYQUIST,
    NO_FIXED_SAMPLE_COUNT_ASSUMPTION,
    RAW_AXES_PRESERVED,
    NO_VECTOR_MAGNITUDE_PRIMARY_SIGNAL,
    SOURCE_AND_REPLAY_ROWS_IDENTICAL,
    SOURCE_AND_REPLAY_SPECTRA_IDENTICAL,
    ALL_SPECTRAL_OUTPUT_ROWS_RECONCILED,
    SYNTHETIC_KERNEL_CONTROLS_PASS,
    INDEPENDENT_MATERIALIZATION_REPRODUCED,
    NO_RESAMPLING_RATE_ABLATION_OR_VIDEO_ARTIFACTS,
)

REPRODUCTION_VERIFIED = "BYTE_IDENTICAL_PADS_P03_PASS"
REPRODUCTION_NOT_ATTEMPTED = "REPRODUCTION_NOT_ATTEMPTED"


class PadsP03GateError(ValueError):
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
class PadsP03GateFacts:
    """Everything the sixteen conditions are decided from."""

    dependency_status: str = "P02_DEPENDENCY_NOT_EVALUATED"
    grid_hash: str = ""
    pinned_grid_hash: str = ""
    grid_bin_count: int = 0
    workload_selection_stable: bool = False
    streams_with_valid_windows: int = 0
    workload_windows_selected: int = 0
    workload_distinct_streams: int = 0
    workload_windows_eligible: int = 0
    windows_differing_from_nominal_grid: int = 0
    nominal_grid_substitutions: int = 0
    nyquist_derived_from_dt_ref_rows: int = 0
    declared_rate_nyquist_rows: int = 0
    distinct_sample_counts: int = 0
    every_length_accepted: bool = False
    windows_refused_for_length: int = 0
    raw_axis_sum_mismatches: int = 0
    vector_magnitude_uses: int = 0
    audit_windows_selected: int = 0
    source_replay_row_mismatches: int = 0
    source_replay_input_hash_mismatches: int = 0
    source_replay_spectral_hash_mismatches: int = 0
    dominant_frequency_mismatches: int = 0
    maximum_observed_bin_error: float = 0.0
    source_unreadable: int = 0
    spectral_rows_written: int = 0
    sensor_family_count: int = 2
    kernel_controls_status: str = "CONTROLS_NOT_RUN"
    reproduction_status: str = REPRODUCTION_NOT_ATTEMPTED
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
        }


def evaluate_gate(facts: PadsP03GateFacts) -> GateResult:
    """Evaluate every condition; all sixteen are required."""

    expected_spectral_rows = (
        facts.workload_windows_eligible * facts.sensor_family_count
    )
    conditions = (
        GateCondition(
            P02_1_DEPENDENCY_VERIFIED,
            facts.dependency_status == DEPENDENCY_VERIFIED,
            facts.dependency_status,
        ),
        GateCondition(
            FREQUENCY_GRID_FROZEN,
            bool(facts.grid_hash)
            and facts.grid_hash == facts.pinned_grid_hash
            and facts.grid_bin_count == 37,
            f"{facts.grid_bin_count} bins, hash {facts.grid_hash[:16]}",
        ),
        GateCondition(
            WORKLOAD_SELECTION_DETERMINISTIC,
            facts.workload_selection_stable,
            "reselection reproduced the same windows"
            if facts.workload_selection_stable
            else "reselection produced a different set",
        ),
        GateCondition(
            ONE_CANONICAL_WINDOW_SELECTED_PER_ELIGIBLE_STREAM,
            facts.streams_with_valid_windows > 0
            and facts.workload_windows_selected
            == facts.streams_with_valid_windows
            and facts.workload_distinct_streams
            == facts.workload_windows_selected,
            f"{facts.workload_windows_selected} windows over "
            f"{facts.workload_distinct_streams} streams, "
            f"{facts.streams_with_valid_windows} eligible",
        ),
        GateCondition(
            SOURCE_TIME_USED_FOR_EVERY_SPECTRUM,
            facts.source_replay_input_hash_mismatches == 0
            and facts.audit_windows_selected > 0
            and facts.source_unreadable == 0,
            f"{facts.source_replay_input_hash_mismatches} input-hash "
            f"mismatches over {facts.audit_windows_selected} audited windows",
        ),
        GateCondition(
            NO_NOMINAL_GRID_TIMESTAMP_SUBSTITUTION,
            facts.nominal_grid_substitutions == 0
            and facts.windows_differing_from_nominal_grid
            == facts.workload_windows_eligible
            and facts.workload_windows_eligible > 0,
            f"{facts.nominal_grid_substitutions} substitutions; "
            f"{facts.windows_differing_from_nominal_grid} windows differ "
            "from an ordinal/rate grid",
        ),
        GateCondition(
            DT_REF_USED_FOR_CADENCE_AND_NYQUIST,
            facts.nyquist_derived_from_dt_ref_rows
            == facts.workload_windows_eligible
            and facts.workload_windows_eligible > 0,
            f"{facts.nyquist_derived_from_dt_ref_rows} rows derive Nyquist "
            f"from dt_ref; {facts.declared_rate_nyquist_rows} report the "
            "declared-rate limit",
        ),
        GateCondition(
            NO_FIXED_SAMPLE_COUNT_ASSUMPTION,
            # A property of the implementation, not of the corpus: the kernel
            # accepts every length the materialization produces, and no window
            # was refused for its length.  How many distinct lengths the
            # corpus happens to offer is reported, not required.
            facts.every_length_accepted
            and facts.windows_refused_for_length == 0
            and facts.workload_windows_eligible > 0,
            f"every observed length accepted={facts.every_length_accepted}, "
            f"{facts.windows_refused_for_length} refused for length, "
            f"{facts.distinct_sample_counts} distinct lengths carried a "
            "spectrum",
        ),
        GateCondition(
            RAW_AXES_PRESERVED,
            facts.raw_axis_sum_mismatches == 0
            and facts.spectral_rows_written > 0,
            f"{facts.raw_axis_sum_mismatches} rows where the aggregate is "
            "not the exact sum of its three axes",
        ),
        GateCondition(
            NO_VECTOR_MAGNITUDE_PRIMARY_SIGNAL,
            facts.vector_magnitude_uses == 0,
            f"{facts.vector_magnitude_uses} vector-magnitude inputs",
        ),
        GateCondition(
            SOURCE_AND_REPLAY_ROWS_IDENTICAL,
            facts.audit_windows_selected > 0
            and facts.source_replay_row_mismatches == 0,
            f"{facts.source_replay_row_mismatches} row mismatches",
        ),
        GateCondition(
            SOURCE_AND_REPLAY_SPECTRA_IDENTICAL,
            facts.audit_windows_selected > 0
            and facts.source_replay_spectral_hash_mismatches == 0
            and facts.dominant_frequency_mismatches == 0
            and facts.maximum_observed_bin_error == 0.0,
            f"{facts.source_replay_spectral_hash_mismatches} hash "
            f"mismatches, maximum bin error "
            f"{facts.maximum_observed_bin_error!r}",
        ),
        GateCondition(
            ALL_SPECTRAL_OUTPUT_ROWS_RECONCILED,
            expected_spectral_rows > 0
            and facts.spectral_rows_written == expected_spectral_rows,
            f"{facts.spectral_rows_written} rows against "
            f"{expected_spectral_rows} expected",
        ),
        GateCondition(
            SYNTHETIC_KERNEL_CONTROLS_PASS,
            facts.kernel_controls_status == CONTROLS_PASS,
            facts.kernel_controls_status,
        ),
        GateCondition(
            INDEPENDENT_MATERIALIZATION_REPRODUCED,
            facts.reproduction_status == REPRODUCTION_VERIFIED,
            facts.reproduction_status,
        ),
        GateCondition(
            NO_RESAMPLING_RATE_ABLATION_OR_VIDEO_ARTIFACTS,
            not any(facts.emitted_forbidden_artifacts.values()),
            f"{sum(facts.emitted_forbidden_artifacts.values())} forbidden "
            "artifacts emitted",
        ),
    )
    names = tuple(condition.name for condition in conditions)
    if names != GATE_CONDITIONS:  # pragma: no cover - contract guard
        raise PadsP03GateError("gate condition set drifted from the contract")
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
    "ALL_SPECTRAL_OUTPUT_ROWS_RECONCILED",
    "DT_REF_USED_FOR_CADENCE_AND_NYQUIST",
    "FREQUENCY_GRID_FROZEN",
    "GATE_CONDITIONS",
    "INDEPENDENT_MATERIALIZATION_REPRODUCED",
    "NO_FIXED_SAMPLE_COUNT_ASSUMPTION",
    "NO_NOMINAL_GRID_TIMESTAMP_SUBSTITUTION",
    "NO_RESAMPLING_RATE_ABLATION_OR_VIDEO_ARTIFACTS",
    "NO_VECTOR_MAGNITUDE_PRIMARY_SIGNAL",
    "ONE_CANONICAL_WINDOW_SELECTED_PER_ELIGIBLE_STREAM",
    "P02_1_DEPENDENCY_VERIFIED",
    "RAW_AXES_PRESERVED",
    "REPRODUCTION_NOT_ATTEMPTED",
    "REPRODUCTION_VERIFIED",
    "SOURCE_AND_REPLAY_ROWS_IDENTICAL",
    "SOURCE_AND_REPLAY_SPECTRA_IDENTICAL",
    "SOURCE_TIME_USED_FOR_EVERY_SPECTRUM",
    "SYNTHETIC_KERNEL_CONTROLS_PASS",
    "WORKLOAD_SELECTION_DETERMINISTIC",
    "GateCondition",
    "GateResult",
    "PadsP03GateError",
    "PadsP03GateFacts",
    "evaluate_gate",
    "failing_conditions",
]
