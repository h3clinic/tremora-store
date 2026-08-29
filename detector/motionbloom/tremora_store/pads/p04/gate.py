"""The eighteen PADS-P0.4 hard-gate conditions.

Two of them are the ones the design review turned on, so they are decided by
positive probes rather than by the absence of a complaint.

The support condition is decided in two parts, because whether a corpus
happens to contain an unbracketable interval is a property of that corpus and
not of the code.  A control constructs an offset segment in this process and
requires that the parent stage removed outputs the FIR guard alone would have
admitted; every derived ordinal in the real run is then re-checked against its
own kernel support.  An implementation that applied only the FIR guard would
satisfy the filter arithmetic and fail the control.

The gain condition recomputes the branch sums from the frozen coefficients and
requires that the executed per-output gain still varies by the published
spread, which is what distinguishes a published imbalance from a concealed
one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .contract import (
    DERIVED_RATES_HZ,
    EDGE_PADDING_ALLOWED,
    GATE_NO_GO,
    GATE_PASS,
    PASSBAND_RIPPLE_MAX_DB,
    PER_OUTPUT_WEIGHT_NORMALIZATION,
    RATES_WITH_EXACT_PICOSECOND_PERIOD,
    RESAMPLING_DOMAIN,
    STOPBAND_ATTENUATION_MIN_DB,
    TRUNCATED_KERNEL_RENORMALIZATION_ALLOWED,
)
from .controls import CONTROLS_PASS
from .dependency import DEPENDENCY_VERIFIED

P03_DEPENDENCY_VERIFIED = "P03_DEPENDENCY_VERIFIED"
ANTI_ALIAS_COEFFICIENTS_FROZEN = "ANTI_ALIAS_COEFFICIENTS_FROZEN"
FILTER_RESPONSE_MEETS_SPECIFICATION = "FILTER_RESPONSE_MEETS_SPECIFICATION"
DERIVED_FROM_WHOLE_P02_1_SEGMENTS = "DERIVED_FROM_WHOLE_P02_1_SEGMENTS"
NATIVE_SPECTRA_ARE_THE_P03_REFERENCE = "NATIVE_SPECTRA_ARE_THE_P03_REFERENCE"
ALL_FOUR_RATES_MATERIALIZED = "ALL_FOUR_RATES_MATERIALIZED"
EXACT_RATIONAL_TIMING_PRESERVED = "EXACT_RATIONAL_TIMING_PRESERVED"
TWO_STAGE_SUPPORT_INTERSECTION_ENFORCED = (
    "TWO_STAGE_SUPPORT_INTERSECTION_ENFORCED"
)
NO_EDGE_PADDING_OR_KERNEL_RENORMALIZATION = (
    "NO_EDGE_PADDING_OR_KERNEL_RENORMALIZATION"
)
NO_PER_PHASE_GAIN_NORMALIZATION = "NO_PER_PHASE_GAIN_NORMALIZATION"
DERIVED_SAMPLE_COUNTS_EXACT = "DERIVED_SAMPLE_COUNTS_EXACT"
CORE_AND_EDGE_BANDS_REPORTED_SEPARATELY = (
    "CORE_AND_EDGE_BANDS_REPORTED_SEPARATELY"
)
SOURCE_AND_REPLAY_DERIVED_IDENTICAL = "SOURCE_AND_REPLAY_DERIVED_IDENTICAL"
ALL_DERIVED_OUTPUT_ROWS_RECONCILED = "ALL_DERIVED_OUTPUT_ROWS_RECONCILED"
PARTICIPANT_LEVEL_SUMMARIES_PRESENT = "PARTICIPANT_LEVEL_SUMMARIES_PRESENT"
SYNTHETIC_RESAMPLING_CONTROLS_PASS = "SYNTHETIC_RESAMPLING_CONTROLS_PASS"
INDEPENDENT_MATERIALIZATION_REPRODUCED = (
    "INDEPENDENT_MATERIALIZATION_REPRODUCED"
)
NO_CLASSIFICATION_VIDEO_OR_P05_ARTIFACTS = (
    "NO_CLASSIFICATION_VIDEO_OR_P05_ARTIFACTS"
)

GATE_CONDITIONS: tuple[str, ...] = (
    P03_DEPENDENCY_VERIFIED,
    ANTI_ALIAS_COEFFICIENTS_FROZEN,
    FILTER_RESPONSE_MEETS_SPECIFICATION,
    DERIVED_FROM_WHOLE_P02_1_SEGMENTS,
    NATIVE_SPECTRA_ARE_THE_P03_REFERENCE,
    ALL_FOUR_RATES_MATERIALIZED,
    EXACT_RATIONAL_TIMING_PRESERVED,
    TWO_STAGE_SUPPORT_INTERSECTION_ENFORCED,
    NO_EDGE_PADDING_OR_KERNEL_RENORMALIZATION,
    NO_PER_PHASE_GAIN_NORMALIZATION,
    DERIVED_SAMPLE_COUNTS_EXACT,
    CORE_AND_EDGE_BANDS_REPORTED_SEPARATELY,
    SOURCE_AND_REPLAY_DERIVED_IDENTICAL,
    ALL_DERIVED_OUTPUT_ROWS_RECONCILED,
    PARTICIPANT_LEVEL_SUMMARIES_PRESENT,
    SYNTHETIC_RESAMPLING_CONTROLS_PASS,
    INDEPENDENT_MATERIALIZATION_REPRODUCED,
    NO_CLASSIFICATION_VIDEO_OR_P05_ARTIFACTS,
)

#: How closely the realized per-output ripple must match the published branch
#: spread.  The two are computed by different routes -- one by filtering a
#: constant, one by summing coefficients -- so they agree to about 4e-15 dB
#: rather than exactly.  Per-phase normalization would collapse the realized
#: ripple to zero, a discrepancy of 4.9e-05 dB: ten million times this bound.
BRANCH_GAIN_AGREEMENT_TOLERANCE_DB = 1e-9

REPRODUCTION_VERIFIED = "BYTE_IDENTICAL_PADS_P04_PASS"
REPRODUCTION_NOT_ATTEMPTED = "REPRODUCTION_NOT_ATTEMPTED"


class PadsP04GateError(ValueError):
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
class PadsP04GateFacts:
    """Everything the eighteen conditions are decided from."""

    dependency_status: str = "P03_DEPENDENCY_NOT_EVALUATED"
    coefficients_hash: str = ""
    pinned_coefficients_hash: str = ""
    filter_rates_measured: int = 0
    worst_passband_ripple_db: float = 0.0
    worst_stopband_attenuation_db: float = 0.0
    resampling_domain: str = ""
    segments_derived: int = 0
    windows_used_as_resampling_domain: int = 0
    reference_spectral_table_hash: str = ""
    pinned_spectral_table_hash: str = ""
    reference_milestone: str = ""
    rates_materialized: tuple[int, ...] = ()
    rates_with_exact_picosecond_period: tuple[int, ...] = ()
    rational_timing_ordinals_checked: int = 0
    rational_timing_mismatches: int = 0
    rounded_thirty_hz_ordinals: int = 0
    intersection_control_passed: bool = False
    ordinals_checked_against_parent_support: int = 0
    ordinals_admitted_by_filter_guard_alone: int = 0
    ordinals_removed_by_parent_stage: int = 0
    ordinals_admitted_over_unbracketed_parent: int = 0
    derived_samples_written: int = 0
    edge_padded_samples: int = 0
    renormalized_kernels: int = 0
    per_phase_normalization: bool = False
    branch_gain_rates_measured: int = 0
    published_branch_gain_spread_db: float = 0.0
    observed_branch_gain_spread_db: float = 0.0
    within_phase_gain_spread: float = 0.0
    derived_sample_count_mismatches: int = 0
    core_band_rows: int = 0
    edge_band_rows: int = 0
    merged_band_rows: int = 0
    audit_comparisons: int = 0
    derived_value_mismatches: int = 0
    derived_spectral_mismatches: int = 0
    maximum_observed_bin_error: float = 0.0
    source_unreadable: int = 0
    spectral_rows_written: int = 0
    eligible_rate_windows: int = 0
    sensor_family_count: int = 2
    participant_summary_rows: int = 0
    participants_covered: int = 0
    controls_status: str = "CONTROLS_NOT_RUN"
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


def evaluate_gate(facts: PadsP04GateFacts) -> GateResult:
    """Evaluate every condition; all eighteen are required."""

    expected_spectral_rows = (
        facts.eligible_rate_windows * facts.sensor_family_count
    )
    conditions = (
        GateCondition(
            P03_DEPENDENCY_VERIFIED,
            facts.dependency_status == DEPENDENCY_VERIFIED,
            facts.dependency_status,
        ),
        GateCondition(
            ANTI_ALIAS_COEFFICIENTS_FROZEN,
            bool(facts.coefficients_hash)
            and facts.coefficients_hash == facts.pinned_coefficients_hash,
            f"coefficients {facts.coefficients_hash[:16]} against pinned "
            f"{facts.pinned_coefficients_hash[:16]}",
        ),
        GateCondition(
            FILTER_RESPONSE_MEETS_SPECIFICATION,
            facts.filter_rates_measured == 3
            and facts.worst_passband_ripple_db <= PASSBAND_RIPPLE_MAX_DB
            and facts.worst_stopband_attenuation_db
            >= STOPBAND_ATTENUATION_MIN_DB,
            f"worst ripple {facts.worst_passband_ripple_db:.4f} dB of "
            f"{PASSBAND_RIPPLE_MAX_DB} allowed, worst stopband "
            f"{facts.worst_stopband_attenuation_db:.2f} dB of "
            f"{STOPBAND_ATTENUATION_MIN_DB} required, over "
            f"{facts.filter_rates_measured} rates",
        ),
        GateCondition(
            DERIVED_FROM_WHOLE_P02_1_SEGMENTS,
            facts.resampling_domain == RESAMPLING_DOMAIN
            and facts.segments_derived > 0
            and facts.windows_used_as_resampling_domain == 0,
            f"domain {facts.resampling_domain!r} over "
            f"{facts.segments_derived} segments; "
            f"{facts.windows_used_as_resampling_domain} windows resampled "
            "in isolation",
        ),
        GateCondition(
            NATIVE_SPECTRA_ARE_THE_P03_REFERENCE,
            bool(facts.reference_spectral_table_hash)
            and facts.reference_spectral_table_hash
            == facts.pinned_spectral_table_hash
            and facts.reference_milestone == "PADS_P0_3",
            f"reference {facts.reference_milestone} table "
            f"{facts.reference_spectral_table_hash[:16]}",
        ),
        GateCondition(
            ALL_FOUR_RATES_MATERIALIZED,
            tuple(sorted(facts.rates_materialized))
            == tuple(sorted(DERIVED_RATES_HZ)),
            f"{sorted(facts.rates_materialized)} materialized against "
            f"{sorted(DERIVED_RATES_HZ)} required",
        ),
        GateCondition(
            EXACT_RATIONAL_TIMING_PRESERVED,
            # Decided over ordinals actually checked: 30 Hz has no exact
            # picosecond period, so it must be absent from the exact set and
            # still land on k/30 s in the rational representation.
            facts.rational_timing_ordinals_checked > 0
            and facts.rational_timing_mismatches == 0
            and facts.rounded_thirty_hz_ordinals == 0
            and tuple(sorted(facts.rates_with_exact_picosecond_period))
            == tuple(sorted(RATES_WITH_EXACT_PICOSECOND_PERIOD)),
            f"{facts.rational_timing_mismatches} of "
            f"{facts.rational_timing_ordinals_checked} ordinals off their "
            f"exact rational time; {facts.rounded_thirty_hz_ordinals} "
            "rounded at 30 Hz"
        ),
        GateCondition(
            TWO_STAGE_SUPPORT_INTERSECTION_ENFORCED,
            # S_derived = S_100Hz_bracketable INTERSECT S_FIR_valid, in that
            # order.  An ordinal the parent could not bracket must already be
            # gone before the filter guard runs; the filter guard alone would
            # not have removed it.
            facts.intersection_control_passed
            and facts.ordinals_checked_against_parent_support > 0
            and facts.ordinals_admitted_over_unbracketed_parent == 0,
            "control "
            f"{'passed' if facts.intersection_control_passed else 'failed'}; "
            f"{facts.ordinals_admitted_over_unbracketed_parent} of "
            f"{facts.ordinals_checked_against_parent_support} derived "
            "ordinals reach into an unbracketable parent interval "
            f"({facts.ordinals_removed_by_parent_stage} of "
            f"{facts.ordinals_admitted_by_filter_guard_alone} were removed "
            "by the parent stage in this corpus)",
        ),
        GateCondition(
            NO_EDGE_PADDING_OR_KERNEL_RENORMALIZATION,
            not EDGE_PADDING_ALLOWED
            and not TRUNCATED_KERNEL_RENORMALIZATION_ALLOWED
            and facts.derived_samples_written > 0
            and facts.edge_padded_samples == 0
            and facts.renormalized_kernels == 0,
            f"{facts.edge_padded_samples} padded samples over "
            f"{facts.derived_samples_written} derived, "
            f"{facts.renormalized_kernels} renormalized kernels",
        ),
        GateCondition(
            NO_PER_PHASE_GAIN_NORMALIZATION,
            # The branch imbalance is published, so the realized per-output
            # ripple must equal it exactly and the gain within one phase must
            # not move at all.  Both would be hidden by per-phase scaling.
            not PER_OUTPUT_WEIGHT_NORMALIZATION
            and not facts.per_phase_normalization
            and facts.branch_gain_rates_measured > 0
            and facts.published_branch_gain_spread_db > 0.0
            and abs(
                facts.observed_branch_gain_spread_db
                - facts.published_branch_gain_spread_db
            )
            <= BRANCH_GAIN_AGREEMENT_TOLERANCE_DB
            and facts.within_phase_gain_spread == 0.0,
            f"observed spread {facts.observed_branch_gain_spread_db:.6f} dB "
            f"against published "
            f"{facts.published_branch_gain_spread_db:.6f} dB; within-phase "
            f"spread {facts.within_phase_gain_spread!r}",
        ),
        GateCondition(
            DERIVED_SAMPLE_COUNTS_EXACT,
            facts.derived_sample_count_mismatches == 0
            and facts.eligible_rate_windows > 0,
            f"{facts.derived_sample_count_mismatches} windows whose derived "
            f"length is not rate times duration, over "
            f"{facts.eligible_rate_windows} eligible rate-windows",
        ),
        GateCondition(
            CORE_AND_EDGE_BANDS_REPORTED_SEPARATELY,
            facts.core_band_rows > 0
            and facts.edge_band_rows > 0
            and facts.merged_band_rows == 0,
            f"{facts.core_band_rows} core rows, {facts.edge_band_rows} edge "
            f"rows, {facts.merged_band_rows} merged",
        ),
        GateCondition(
            SOURCE_AND_REPLAY_DERIVED_IDENTICAL,
            facts.audit_comparisons > 0
            and facts.derived_value_mismatches == 0
            and facts.derived_spectral_mismatches == 0
            and facts.maximum_observed_bin_error == 0.0
            and facts.source_unreadable == 0,
            f"{facts.derived_value_mismatches} value and "
            f"{facts.derived_spectral_mismatches} spectral mismatches over "
            f"{facts.audit_comparisons} comparisons, maximum bin error "
            f"{facts.maximum_observed_bin_error!r}",
        ),
        GateCondition(
            ALL_DERIVED_OUTPUT_ROWS_RECONCILED,
            expected_spectral_rows > 0
            and facts.spectral_rows_written == expected_spectral_rows,
            f"{facts.spectral_rows_written} rows against "
            f"{expected_spectral_rows} expected",
        ),
        GateCondition(
            PARTICIPANT_LEVEL_SUMMARIES_PRESENT,
            facts.participants_covered > 0
            and facts.participant_summary_rows > 0
            and facts.participant_summary_rows
            % facts.participants_covered
            == 0,
            f"{facts.participant_summary_rows} rows over "
            f"{facts.participants_covered} participants",
        ),
        GateCondition(
            SYNTHETIC_RESAMPLING_CONTROLS_PASS,
            facts.controls_status == CONTROLS_PASS,
            facts.controls_status,
        ),
        GateCondition(
            INDEPENDENT_MATERIALIZATION_REPRODUCED,
            facts.reproduction_status == REPRODUCTION_VERIFIED,
            facts.reproduction_status,
        ),
        GateCondition(
            NO_CLASSIFICATION_VIDEO_OR_P05_ARTIFACTS,
            bool(facts.emitted_forbidden_artifacts)
            and not any(facts.emitted_forbidden_artifacts.values()),
            f"{sum(facts.emitted_forbidden_artifacts.values())} forbidden "
            f"artifacts over {len(facts.emitted_forbidden_artifacts)} "
            "screened categories"
            if facts.emitted_forbidden_artifacts
            else "the screen did not run",
        ),
    )
    names = tuple(condition.name for condition in conditions)
    if names != GATE_CONDITIONS:  # pragma: no cover - contract guard
        raise PadsP04GateError("gate condition set drifted from the contract")
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
    "ALL_DERIVED_OUTPUT_ROWS_RECONCILED",
    "ALL_FOUR_RATES_MATERIALIZED",
    "ANTI_ALIAS_COEFFICIENTS_FROZEN",
    "BRANCH_GAIN_AGREEMENT_TOLERANCE_DB",
    "CORE_AND_EDGE_BANDS_REPORTED_SEPARATELY",
    "DERIVED_FROM_WHOLE_P02_1_SEGMENTS",
    "DERIVED_SAMPLE_COUNTS_EXACT",
    "EXACT_RATIONAL_TIMING_PRESERVED",
    "FILTER_RESPONSE_MEETS_SPECIFICATION",
    "GATE_CONDITIONS",
    "INDEPENDENT_MATERIALIZATION_REPRODUCED",
    "NATIVE_SPECTRA_ARE_THE_P03_REFERENCE",
    "NO_CLASSIFICATION_VIDEO_OR_P05_ARTIFACTS",
    "NO_EDGE_PADDING_OR_KERNEL_RENORMALIZATION",
    "NO_PER_PHASE_GAIN_NORMALIZATION",
    "P03_DEPENDENCY_VERIFIED",
    "PARTICIPANT_LEVEL_SUMMARIES_PRESENT",
    "REPRODUCTION_NOT_ATTEMPTED",
    "REPRODUCTION_VERIFIED",
    "SOURCE_AND_REPLAY_DERIVED_IDENTICAL",
    "SYNTHETIC_RESAMPLING_CONTROLS_PASS",
    "TWO_STAGE_SUPPORT_INTERSECTION_ENFORCED",
    "GateCondition",
    "GateResult",
    "PadsP04GateError",
    "PadsP04GateFacts",
    "evaluate_gate",
    "failing_conditions",
]
