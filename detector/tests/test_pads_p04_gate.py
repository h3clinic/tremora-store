"""The eighteen P0.4 gate conditions, and what each of them refuses."""

from __future__ import annotations

from dataclasses import replace

import pytest
from motionbloom.tremora_store.pads.p04.contract import (
    DERIVED_RATES_HZ,
    GATE_NO_GO,
    GATE_PASS,
    RATES_WITH_EXACT_PICOSECOND_PERIOD,
    RESAMPLING_DOMAIN,
    WITHHELD_P04_ARTIFACTS,
)
from motionbloom.tremora_store.pads.p04.controls import (
    CONTROLS_PASS,
    run_controls,
)
from motionbloom.tremora_store.pads.p04.dependency import DEPENDENCY_VERIFIED
from motionbloom.tremora_store.pads.p04.filters import (
    coefficients_sha256,
    dc_terminology,
)
from motionbloom.tremora_store.pads.p04.gate import (
    GATE_CONDITIONS,
    REPRODUCTION_VERIFIED,
    GateResult,
    PadsP04GateFacts,
    evaluate_gate,
    failing_conditions,
)

PUBLISHED_SPREAD_DB = max(
    dc_terminology(rate)["polyphase_dc_gain_spread_db"] for rate in (50, 30, 25)
)


@pytest.fixture(scope="module")
def controls() -> dict:
    return run_controls()


@pytest.fixture
def passing(controls: dict) -> PadsP04GateFacts:
    """Facts that satisfy every condition, built from real measurements."""

    constant = controls["measured"]["constant_input_30"]
    return PadsP04GateFacts(
        dependency_status=DEPENDENCY_VERIFIED,
        coefficients_hash=coefficients_sha256(),
        pinned_coefficients_hash=coefficients_sha256(),
        filter_rates_measured=3,
        worst_passband_ripple_db=0.0093,
        worst_stopband_attenuation_db=63.80,
        resampling_domain=RESAMPLING_DOMAIN,
        segments_derived=160,
        windows_used_as_resampling_domain=0,
        reference_spectral_table_hash="c" * 64,
        pinned_spectral_table_hash="c" * 64,
        reference_milestone="PADS_P0_3",
        rates_materialized=DERIVED_RATES_HZ,
        rates_with_exact_picosecond_period=RATES_WITH_EXACT_PICOSECOND_PERIOD,
        rational_timing_ordinals_checked=31_280,
        rational_timing_mismatches=0,
        rounded_thirty_hz_ordinals=0,
        intersection_control_passed=True,
        ordinals_checked_against_parent_support=31_280,
        ordinals_admitted_by_filter_guard_alone=103_316,
        ordinals_removed_by_parent_stage=3_751,
        ordinals_admitted_over_unbracketed_parent=0,
        derived_samples_written=31_280,
        edge_padded_samples=0,
        renormalized_kernels=0,
        per_phase_normalization=False,
        branch_gain_rates_measured=3,
        published_branch_gain_spread_db=PUBLISHED_SPREAD_DB,
        observed_branch_gain_spread_db=constant["observed_ripple_db"],
        within_phase_gain_spread=constant["within_phase_spread"],
        derived_sample_count_mismatches=0,
        core_band_rows=16,
        edge_band_rows=16,
        merged_band_rows=0,
        audit_comparisons=7,
        derived_value_mismatches=0,
        derived_spectral_mismatches=0,
        maximum_observed_bin_error=0.0,
        source_unreadable=0,
        spectral_rows_written=294,
        eligible_rate_windows=147,
        sensor_family_count=2,
        participant_summary_rows=32,
        participants_covered=16,
        controls_status=CONTROLS_PASS,
        reproduction_status=REPRODUCTION_VERIFIED,
        emitted_forbidden_artifacts=dict(WITHHELD_P04_ARTIFACTS),
    )


def _status(facts: PadsP04GateFacts) -> GateResult:
    return evaluate_gate(facts)


def test_the_contract_names_eighteen_conditions() -> None:
    assert len(GATE_CONDITIONS) == 18
    assert len(set(GATE_CONDITIONS)) == 18


def test_a_complete_set_of_facts_passes(passing: PadsP04GateFacts) -> None:
    result = _status(passing)
    assert result.gate_status == GATE_PASS
    assert result.satisfied
    assert failing_conditions(result.as_record()) == ()


def test_nothing_passes_on_an_empty_record() -> None:
    result = _status(PadsP04GateFacts())
    assert result.gate_status == GATE_NO_GO
    # Absence blocks: no condition may be satisfied by having measured
    # nothing at all.
    assert len(failing_conditions(result.as_record())) == 18


@pytest.mark.parametrize(
    ("field", "value", "condition"),
    (
        ("dependency_status", "P03_REPORT_HASH_MISMATCH",
         "P03_DEPENDENCY_VERIFIED"),
        ("coefficients_hash", "a" * 64, "ANTI_ALIAS_COEFFICIENTS_FROZEN"),
        ("worst_passband_ripple_db", 0.26,
         "FILTER_RESPONSE_MEETS_SPECIFICATION"),
        ("worst_stopband_attenuation_db", 59.9,
         "FILTER_RESPONSE_MEETS_SPECIFICATION"),
        ("windows_used_as_resampling_domain", 1,
         "DERIVED_FROM_WHOLE_P02_1_SEGMENTS"),
        ("reference_spectral_table_hash", "d" * 64,
         "NATIVE_SPECTRA_ARE_THE_P03_REFERENCE"),
        ("rates_materialized", (100, 50, 30), "ALL_FOUR_RATES_MATERIALIZED"),
        ("rational_timing_mismatches", 1,
         "EXACT_RATIONAL_TIMING_PRESERVED"),
        ("rounded_thirty_hz_ordinals", 1, "EXACT_RATIONAL_TIMING_PRESERVED"),
        ("ordinals_admitted_over_unbracketed_parent", 1,
         "TWO_STAGE_SUPPORT_INTERSECTION_ENFORCED"),
        ("intersection_control_passed", False,
         "TWO_STAGE_SUPPORT_INTERSECTION_ENFORCED"),
        ("ordinals_checked_against_parent_support", 0,
         "TWO_STAGE_SUPPORT_INTERSECTION_ENFORCED"),
        ("edge_padded_samples", 1,
         "NO_EDGE_PADDING_OR_KERNEL_RENORMALIZATION"),
        ("renormalized_kernels", 1,
         "NO_EDGE_PADDING_OR_KERNEL_RENORMALIZATION"),
        ("per_phase_normalization", True,
         "NO_PER_PHASE_GAIN_NORMALIZATION"),
        ("within_phase_gain_spread", 1e-18,
         "NO_PER_PHASE_GAIN_NORMALIZATION"),
        ("derived_sample_count_mismatches", 1, "DERIVED_SAMPLE_COUNTS_EXACT"),
        ("merged_band_rows", 1, "CORE_AND_EDGE_BANDS_REPORTED_SEPARATELY"),
        ("edge_band_rows", 0, "CORE_AND_EDGE_BANDS_REPORTED_SEPARATELY"),
        ("derived_value_mismatches", 1,
         "SOURCE_AND_REPLAY_DERIVED_IDENTICAL"),
        ("derived_spectral_mismatches", 1,
         "SOURCE_AND_REPLAY_DERIVED_IDENTICAL"),
        ("maximum_observed_bin_error", 1e-18,
         "SOURCE_AND_REPLAY_DERIVED_IDENTICAL"),
        ("source_unreadable", 1, "SOURCE_AND_REPLAY_DERIVED_IDENTICAL"),
        ("spectral_rows_written", 293,
         "ALL_DERIVED_OUTPUT_ROWS_RECONCILED"),
        ("participants_covered", 0,
         "PARTICIPANT_LEVEL_SUMMARIES_PRESENT"),
        ("controls_status", "RESAMPLING_CONTROLS_FAIL",
         "SYNTHETIC_RESAMPLING_CONTROLS_PASS"),
        ("reproduction_status", "REPRODUCTION_NOT_ATTEMPTED",
         "INDEPENDENT_MATERIALIZATION_REPRODUCED"),
    ),
)
def test_each_fact_closes_the_condition_it_belongs_to(
    passing: PadsP04GateFacts, field: str, value: object, condition: str
) -> None:
    result = _status(replace(passing, **{field: value}))
    assert result.gate_status == GATE_NO_GO
    assert condition in failing_conditions(result.as_record())


def test_a_forbidden_artifact_closes_the_gate(
    passing: PadsP04GateFacts,
) -> None:
    withheld = dict(WITHHELD_P04_ARTIFACTS)
    withheld["classification_tables"] = 1
    result = _status(replace(passing, emitted_forbidden_artifacts=withheld))
    assert "NO_CLASSIFICATION_VIDEO_OR_P05_ARTIFACTS" in failing_conditions(
        result.as_record()
    )


def test_per_phase_normalization_cannot_hide_behind_a_zero_ripple(
    passing: PadsP04GateFacts,
) -> None:
    # Normalizing each branch drives the realized ripple to zero.  The gate
    # requires the realized ripple to still be the published one, so the
    # concealment is what closes it.
    result = _status(replace(passing, observed_branch_gain_spread_db=0.0))
    assert "NO_PER_PHASE_GAIN_NORMALIZATION" in failing_conditions(
        result.as_record()
    )


def test_the_realized_and_published_spreads_need_not_be_bit_identical(
    passing: PadsP04GateFacts,
) -> None:
    # They are computed by different routes -- filtering a constant, versus
    # summing coefficients -- so they agree to about 4e-15 dB, not exactly.
    assert (
        passing.observed_branch_gain_spread_db
        != passing.published_branch_gain_spread_db
    )
    assert _status(passing).satisfied


def test_thirty_hz_may_not_be_declared_exact_in_picoseconds(
    passing: PadsP04GateFacts,
) -> None:
    result = _status(
        replace(passing, rates_with_exact_picosecond_period=(100, 50, 30, 25))
    )
    assert "EXACT_RATIONAL_TIMING_PRESERVED" in failing_conditions(
        result.as_record()
    )
