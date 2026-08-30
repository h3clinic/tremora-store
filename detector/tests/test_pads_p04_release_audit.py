"""Checks on the published PADS-P0.4 release audit.

The report is hash-pinned.  Editing it to change a verdict, or regenerating it
from a different store, dependency, filter set or kernel, fails here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from motionbloom.tremora_store.pads.p04.contract import (
    DERIVED_RATES_HZ,
    GATE_PASS,
    GENERIC_SUCCESS_MARKER,
    SUCCESS_MARKER,
    assert_no_clinical_or_benchmark_claim,
)
from motionbloom.tremora_store.pads.p04.dependency import FROZEN_DEPENDENCY
from motionbloom.tremora_store.pads.p04.filters import coefficients_sha256
from motionbloom.tremora_store.pads.p04.gate import (
    GATE_CONDITIONS,
    failing_conditions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "detector/benchmarks/pads_p04_release_audit.json"
REPORT_SHA256 = (
    "45bfb308c6db053c48a1abf7231615d395a8b5039999a99f91991aecbce379d3"
)
EVIDENCE_SHA256 = (
    "2aaebd342902c9e2d7d4f3900be92bfd59f52d299c320aa47a7677dc476979b0"
)
SPECTRAL_TABLE_SHA256 = (
    "a66785ded71a33d5f09469a14c600bd7f98edf960918f549f3df0bff8f56d32c"
)
ANTI_ALIAS_SHA256 = (
    "976957f77d3ba0edbe72507bb32617751bbf1f3c1f38e299c5ce5e4120163d81"
)

WORKLOAD_WINDOWS = 9_960
ATTEMPTED = 39_840
ELIGIBLE = 38_316
DERIVED_SAMPLES = 7_981_740
ELIGIBLE_BY_RATE = {"100": 9_960, "50": 9_756, "30": 9_327, "25": 9_273}
SAMPLES_PER_WINDOW = {"100": "400", "50": "200", "30": "120", "25": "100"}


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_bytes())


@pytest.fixture(scope="module")
def produced(report: dict) -> dict:
    return report["materialization"]


def test_the_published_report_is_hash_pinned() -> None:
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_SHA256


def test_the_gate_passed_on_every_condition(report: dict) -> None:
    assert report["gate_status"] == GATE_PASS
    assert failing_conditions(report) == ()
    assert report["gate_conditions_satisfied"] == len(GATE_CONDITIONS) == 18
    assert report["audit_execution_status"] == "PASS"


def test_it_was_authorized_by_the_pinned_p03_pass(report: dict) -> None:
    dependency = report["p03_dependency"]
    assert dependency["dependency_status"] == "P03_DEPENDENCY_VERIFIED"
    assert dependency["pinned"]["p03_evidence_sha256"] == (
        FROZEN_DEPENDENCY.p03_evidence_sha256
    )
    assert dependency["observed_report_sha256"] == (
        FROZEN_DEPENDENCY.p03_report_sha256
    )
    assert dependency["pinned"]["p03_gate_status"] == (
        "PASS_PADS_SOURCE_TIME_SPECTRAL_PRESERVATION"
    )


def test_the_reference_spectra_are_the_frozen_p03_table(report: dict) -> None:
    # P0.4 never recomputes a native spectrum; the P0.3 table is the
    # reference, and the audit rehashed it rather than taking the pin's word.
    dependency = report["p03_dependency"]
    assert dependency["observed_spectral_table_sha256"] == (
        FROZEN_DEPENDENCY.p03_spectral_table_sha256
    )
    assert dependency["pinned"]["p03_spectral_table_sha256"] == (
        FROZEN_DEPENDENCY.p03_spectral_table_sha256
    )


# --- the filters ----------------------------------------------------------


def test_the_anti_alias_coefficients_are_the_frozen_ones(
    report: dict,
) -> None:
    filters = report["anti_alias_filters"]
    assert filters["coefficients_sha256"] == ANTI_ALIAS_SHA256
    assert coefficients_sha256() == ANTI_ALIAS_SHA256


def test_each_rate_has_its_own_filter_not_one_universal_cutoff(
    report: dict,
) -> None:
    measured = report["anti_alias_filters"]["measured"]
    assert set(measured) == {"50", "30", "25"}
    # 25 Hz preserves to 10 Hz where 50 and 30 Hz preserve to 12; a universal
    # fraction-of-Nyquist cutoff would have put a transition band across the
    # top of the tremor band at 25 Hz.
    manifest = report["anti_alias_filters"]["manifest"]["filters"]
    assert manifest["25"]["passband_hz"] == 10.0
    assert manifest["50"]["passband_hz"] == 12.0
    assert manifest["30"]["passband_hz"] == 12.0


def test_every_filter_meets_the_published_specification(report: dict) -> None:
    filters = report["anti_alias_filters"]
    assert filters["worst_passband_ripple_db"] <= 0.25
    assert filters["worst_stopband_attenuation_db"] >= 60.0
    for entry in filters["measured"].values():
        assert entry["dc_gain"] == pytest.approx(1.0, abs=1e-9)
        assert entry["symmetric"] == 1.0


def test_the_branch_gains_are_published_rather_than_normalized(
    report: dict,
) -> None:
    terms = report["anti_alias_filters"]["dc_terminology"]
    assert set(terms) == {"50", "30", "25"}
    thirty = terms["30"]
    assert thirty["per_phase_normalization"] is False
    assert thirty["upsampling_factor"] == 3
    assert thirty["prototype_coefficient_sum"] == pytest.approx(3.0, abs=1e-12)
    assert thirty["effective_dc_gain"] == pytest.approx(1.0, abs=1e-12)
    gains = thirty["polyphase_dc_gains"]
    assert len(gains) == 3
    # The imbalance is real, published, and far inside the ripple budget.
    assert len(set(gains)) > 1
    assert 0.0 < thirty["polyphase_dc_gain_spread_db"] < 1e-3


# --- the resampler --------------------------------------------------------


def test_all_four_rates_were_materialized(
    report: dict, produced: dict
) -> None:
    assert report["derived_rates_hz"] == list(DERIVED_RATES_HZ)
    assert produced["rates_materialized"] == [25, 30, 50, 100]


def test_eligibility_falls_monotonically_with_the_rate(
    produced: dict,
) -> None:
    assert produced["workload_windows"] == WORKLOAD_WINDOWS
    assert produced["derived_rate_windows_attempted"] == ATTEMPTED
    assert produced["derived_rate_windows_eligible"] == ELIGIBLE
    assert produced["eligible_by_rate"] == ELIGIBLE_BY_RATE
    # A longer kernel reaches further past each segment's ends, and those
    # outputs are refused rather than padded.
    counts = [ELIGIBLE_BY_RATE[str(rate)] for rate in (100, 50, 30, 25)]
    assert counts == sorted(counts, reverse=True)


def test_every_window_carries_exactly_rate_times_duration_samples(
    produced: dict,
) -> None:
    assert produced["derived_samples_written"] == DERIVED_SAMPLES
    assert produced["derived_sample_count_mismatches"] == 0
    for rate, expected in SAMPLES_PER_WINDOW.items():
        counts = produced["sample_count_by_rate"][rate]
        # One length per rate, no exceptions.
        assert list(counts) == [expected]
        assert counts[expected] == ELIGIBLE_BY_RATE[rate]


def test_every_ordinal_landed_on_its_exact_rational_time(
    produced: dict,
) -> None:
    assert produced["rational_timing_ordinals_checked"] == DERIVED_SAMPLES
    assert produced["rational_timing_mismatches"] == 0
    # 30 Hz has no exact picosecond period; none of its ordinals was rounded.
    assert produced["rounded_thirty_hz_ordinals"] == 0


def test_support_is_the_intersection_of_both_stages(produced: dict) -> None:
    # The parent-bracketing stage removed outputs the FIR guard alone would
    # have admitted, and nothing survived over an unbracketable interval.
    assert produced["ordinals_removed_by_parent_stage"] > 0
    assert produced["ordinals_removed_by_parent_stage"] < produced[
        "ordinals_admitted_by_filter_guard_alone"
    ]
    assert produced["ordinals_admitted_over_unbracketed_parent"] == 0
    assert produced["parent_ordinals_unbracketed"] > 0


# --- source against replay ------------------------------------------------


def test_source_and_replay_derive_identical_values(produced: dict) -> None:
    assert produced["audit_comparisons"] > 0
    assert produced["source_replay_sample_mismatches"] == 0
    assert produced["source_replay_derived_mismatches"] == 0
    assert produced["source_replay_spectral_mismatches"] == 0
    assert produced["source_unreadable"] == 0
    # Bit equality, not a tolerance.  A future non-zero is a reproducibility
    # incident to diagnose, not grounds to loosen the comparison.
    assert produced["maximum_bin_absolute_error"] == 0.0


def test_every_eligible_window_carried_both_family_spectra(
    produced: dict,
) -> None:
    assert produced["spectral_rows"] == ELIGIBLE * 2
    assert produced["spectral_table_content_sha256"] == SPECTRAL_TABLE_SHA256
    assert produced["failure_count"] == 0


def test_the_two_bands_are_summarized_separately(produced: dict) -> None:
    assert produced["core_summary_rows"] == produced["edge_summary_rows"]
    assert produced["core_summary_rows"] > 0
    assert (
        produced["core_summary_rows"] + produced["edge_summary_rows"]
        == produced["participant_summary_rows"]
    )
    assert produced["participants"] == 469


# --- the controls ---------------------------------------------------------


def test_the_resampling_controls_ran_in_the_audit_process(
    report: dict,
) -> None:
    controls = report["resampling_controls"]
    assert controls["status"] == "RESAMPLING_CONTROLS_PASS"
    assert controls["controls_passed"] == controls["controls_total"] == 12
    assert all(controls["controls"].values())


def test_the_constant_input_control_shows_no_per_output_normalization(
    report: dict,
) -> None:
    constant = report["resampling_controls"]["measured"]["constant_input_30"]
    assert constant["mean"] == pytest.approx(1.0, abs=1e-6)
    assert constant["observed_ripple"] == pytest.approx(
        constant["published_branch_ripple"], abs=1e-12
    )
    # Nothing is renormalized per output, so within one phase the gain does
    # not move at all.
    assert constant["within_phase_spread"] == 0.0


def test_a_band_power_ratio_does_not_measure_the_sample_count(
    report: dict,
) -> None:
    ratios = report["resampling_controls"]["measured"][
        "core_band_power_ratio_by_rate"
    ]
    assert set(ratios) == {"100", "50", "30", "25"}
    # Unnormalized these would read 1.00, 0.50, 0.30, 0.25.
    for rate, ratio in ratios.items():
        assert 0.95 <= ratio <= 1.05, (rate, ratio)


# --- the claim boundary ---------------------------------------------------


def test_the_run_was_single_threaded_and_used_no_blas(report: dict) -> None:
    assert report["numeric_execution"]["blas_used"] is False
    assert set(report["numeric_execution"]["threading"].values()) == {"1"}


def test_two_separate_processes_produced_the_same_evidence(
    report: dict,
) -> None:
    assert report["canonical_evidence_sha256"] == EVIDENCE_SHA256
    assert report["independent_reproduction_status"] == (
        "BYTE_IDENTICAL_PADS_P04_PASS"
    )
    receipts = report["reproduction_receipts"]
    assert receipts["run_a_canonical_hash"] == EVIDENCE_SHA256
    assert receipts["run_b_canonical_hash"] == EVIDENCE_SHA256
    # The proof is that the two runs disagree about everything except the
    # evidence: different run id, process and output-root device/inode.
    assert receipts["run_a"]["run_id"] != receipts["run_b"]["run_id"]
    assert receipts["run_a"]["process_id"] != receipts["run_b"]["process_id"]
    assert receipts["run_a"]["output_root_identity"] != (
        receipts["run_b"]["output_root_identity"]
    )


def test_no_later_milestone_was_claimed(report: dict) -> None:
    withheld = report["withheld_artifacts"]
    assert withheld
    assert set(withheld.values()) == {0}
    for name in (
        "classification_tables", "diagnosis_tables", "severity_tables",
        "video_association_tables", "storage_benchmark_tables",
        "tremor_detection_tables", "retrieval_latency_tables",
        "generic_success_markers",
    ):
        assert withheld[name] == 0


def test_the_report_names_nothing_beyond_this_milestone(report: dict) -> None:
    assert_no_clinical_or_benchmark_claim(report.keys())
    assert_no_clinical_or_benchmark_claim(report["materialization"].keys())
    assert SUCCESS_MARKER == "_PADS_P04_RATE_ABLATION_SUCCESS"
    # Specific to this milestone, never the bare generic marker.
    assert SUCCESS_MARKER != GENERIC_SUCCESS_MARKER
