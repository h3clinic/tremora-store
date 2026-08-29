"""Checks on the published PADS-P0.3 release audit.

The report is hash-pinned.  Editing it to change a verdict, or regenerating it
from a different store, dependency or kernel, fails here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from motionbloom.tremora_store.pads.p03.contract import (
    FREQUENCY_BIN_COUNT,
    GATE_PASS,
    SUCCESS_MARKER,
    assert_p03_names,
)
from motionbloom.tremora_store.pads.p03.dependency import FROZEN_DEPENDENCY
from motionbloom.tremora_store.pads.p03.gate import (
    GATE_CONDITIONS,
    failing_conditions,
)
from motionbloom.tremora_store.pads.p03.grid import grid_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "detector/benchmarks/pads_p03_release_audit.json"
REPORT_SHA256 = (
    "a2b6dfa3f598dfe7e2821285c3262dd1f817b168cb81e62683ad796445faf615"
)
EVIDENCE_SHA256 = (
    "a0be87d48c1146862aa83a2a1238d3df50fc344781878b6ee0a03f738548df17"
)
SPECTRAL_TABLE_SHA256 = (
    "27bb6444bdfcab77911134b1c4671f563c51084f69e6d587e339c0a00d76c97e"
)

STREAMS = 10_318
ELIGIBLE_STREAMS = 9_960
AUDIT_WINDOWS = 6_077


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_bytes())


def test_the_published_report_is_hash_pinned() -> None:
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_SHA256


def test_the_gate_passed_on_every_condition(report: dict) -> None:
    assert report["gate_status"] == GATE_PASS
    assert failing_conditions(report) == ()
    assert report["gate_conditions_satisfied"] == len(GATE_CONDITIONS) == 16


def test_it_was_authorized_by_the_pinned_p02_pass(report: dict) -> None:
    dependency = report["p02_dependency"]
    assert dependency["dependency_status"] == "P02_1_DEPENDENCY_VERIFIED"
    assert dependency["pinned"] == FROZEN_DEPENDENCY.as_record()
    assert dependency["observed_storage_index_sha256"] == (
        FROZEN_DEPENDENCY.storage_index_content_sha256
    )
    assert dependency["observed_report_sha256"] == (
        FROZEN_DEPENDENCY.p02_report_sha256
    )


def test_one_canonical_window_per_eligible_stream(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["streams_total"] == STREAMS
    assert materialization["streams_with_valid_windows"] == ELIGIBLE_STREAMS
    assert materialization["workload_windows_selected"] == ELIGIBLE_STREAMS
    assert materialization["workload_distinct_streams"] == ELIGIBLE_STREAMS
    assert materialization["workload_selection_stable"] is True
    # 358 streams hold no valid window and are absent, not invented.
    assert STREAMS - ELIGIBLE_STREAMS == 358


def test_every_workload_window_carried_a_spectrum(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["workload_windows_eligible"] == ELIGIBLE_STREAMS
    assert materialization["workload_windows_ineligible"] == 0
    assert materialization["gyro_spectral_rows"] == ELIGIBLE_STREAMS
    assert materialization["accel_spectral_rows"] == ELIGIBLE_STREAMS
    assert report["materialized_release_artifacts"] == ELIGIBLE_STREAMS * 2
    assert materialization["spectral_table_content_sha256"] == (
        SPECTRAL_TABLE_SHA256
    )


def test_source_and_replay_are_bit_identical(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["audit_windows_selected"] == AUDIT_WINDOWS
    assert materialization["source_replay_row_mismatches"] == 0
    assert materialization["source_replay_input_hash_mismatches"] == 0
    assert materialization["source_replay_spectral_hash_mismatches"] == 0
    assert materialization["dominant_frequency_mismatches"] == 0
    assert materialization["source_unreadable"] == 0
    # Exactly zero, not a tolerance: both paths feed identical float64 inputs
    # into the identical kernel.
    assert materialization["maximum_observed_bin_error"] == 0.0


def test_no_nominal_grid_was_substituted(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["nominal_grid_substitutions"] == 0
    # The probe has teeth: every one of the windows genuinely differs from an
    # ordinal/rate grid, so the zero above is not vacuous.
    assert materialization["windows_differing_from_nominal_grid"] == (
        ELIGIBLE_STREAMS
    )


def test_nyquist_came_from_each_stream_cadence(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["nyquist_derived_from_dt_ref_rows"] == (
        ELIGIBLE_STREAMS
    )
    # No stream's dt_ref is exactly 10 ms, so nothing reports the declared
    # 100 Hz rate's 50 Hz limit.
    assert materialization["declared_rate_nyquist_rows"] == 0


def test_no_fixed_sample_count_was_assumed(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["windows_refused_for_length"] == 0
    histogram = materialization["sample_count_histogram"]
    assert materialization["distinct_sample_counts"] == len(histogram) == 9
    assert set(histogram) == {str(n) for n in range(397, 406)}
    # 400 is the most common length but far from the only one.
    assert max(histogram, key=lambda key: histogram[key]) == "400"
    assert sum(histogram.values()) == ELIGIBLE_STREAMS


def test_the_audit_subset_covers_every_stratum_dimension(
    report: dict,
) -> None:
    coverage = report["materialization"]["selection_coverage"]
    assert coverage["populated_strata"] == 862
    assert len(coverage["tasks"]) == 11
    assert coverage["device_locations"] == ["LeftWrist", "RightWrist"]
    assert coverage["outer_folds"] == [0, 1, 2, 3, 4]
    # All eleven observed window lengths appear, including the 395- and
    # 396-sample cases the workload's one-per-stream rule never reaches.
    assert coverage["sample_counts"] == list(range(395, 406))
    assert coverage["gap_adjacent_windows"] == 1_497
    assert coverage["interior_windows"] == 4_580
    assert coverage["audit_windows_per_stratum"] == 10
    assert coverage["audit_selection_seed"] == 20260829


def test_raw_axes_were_preserved_and_summed(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["raw_axis_sum_mismatches"] == 0
    assert materialization["vector_magnitude_uses"] == 0
    assert report["authority"]["spectral_input"] == (
        "RAW_AXES_PER_SENSOR_FAMILY"
    )
    assert report["authority"]["vector_magnitude_primary_signal"] is False


def test_the_kernel_controls_ran_and_passed(report: dict) -> None:
    controls = report["kernel_controls"]
    assert controls["status"] == "SYNTHETIC_KERNEL_CONTROLS_PASS"
    assert controls["controls_passed"] == controls["controls_total"] == 12
    for name in (
        "tone_5hz_recovered", "tone_8hz_recovered",
        "phase_shift_preserves_spectrum",
        "axis_rotation_preserves_summed_power", "linear_trend_removed",
        "every_observed_length_accepted",
        "source_time_not_ordinal_over_rate",
        "sample_count_is_not_the_clock",
        "vector_magnitude_would_double_frequency",
        "non_monotonic_refused", "gap_crossing_refused",
        "grid_is_thirty_seven_bins",
    ):
        assert controls["controls"][name] is True


def test_the_frequency_grid_is_the_frozen_one(report: dict) -> None:
    grid = report["frequency_grid"]
    assert grid["frequency_bin_count"] == FREQUENCY_BIN_COUNT == 37
    assert grid["frequency_grid_hash"] == grid_hash()


def test_the_run_was_single_threaded_and_used_no_blas(report: dict) -> None:
    execution = report["numeric_execution"]
    assert execution["blas_used"] is False
    assert set(execution["threading"].values()) == {"1"}


def test_two_separate_processes_produced_the_same_evidence(
    report: dict,
) -> None:
    receipts = report["reproduction_receipts"]
    assert report["canonical_evidence_sha256"] == EVIDENCE_SHA256
    assert receipts["run_a_canonical_hash"] == EVIDENCE_SHA256
    assert receipts["run_b_canonical_hash"] == EVIDENCE_SHA256
    assert receipts["run_a"]["run_id"] != receipts["run_b"]["run_id"]
    assert receipts["run_a"]["process_id"] != receipts["run_b"]["process_id"]
    assert receipts["run_a"]["output_root_identity"] != (
        receipts["run_b"]["output_root_identity"]
    )
    assert report["independent_reproduction_status"] == (
        "BYTE_IDENTICAL_PADS_P03_PASS"
    )


def test_the_unimodal_boundary_is_still_declared(report: dict) -> None:
    authority = report["authority"]
    assert authority["timing_authority"] == "SOURCE_RELATIVE_UNIMODAL_CLOCK"
    assert authority["time_basis"] == "SOURCE_TIME_COLUMN"
    assert authority["video_pairing"] == "NOT_APPLICABLE"
    assert authority["hardware_sync_claim"] is False
    assert authority["cross_wrist_clock_alignment"] == "UNRESOLVED"
    assert authority["sample_level_bilateral_fusion_allowed"] is False


def test_the_next_milestones_are_published_as_withheld(report: dict) -> None:
    assert set(report["withheld_artifacts"].values()) == {0}
    for name in (
        "resampled_signal_tables", "anti_alias_filter_outputs",
        "derived_rate_tables", "classification_tables",
        "tremor_detection_tables", "bilateral_fusion_tables",
        "video_association_tables", "comparative_benchmark_tables",
        "generic_success_markers",
    ):
        assert report["withheld_artifacts"][name] == 0


def test_the_marker_is_specific_to_this_milestone() -> None:
    assert SUCCESS_MARKER == "_PADS_P03_SPECTRAL_SUCCESS"


def test_the_report_names_no_forbidden_artifact_of_its_own(
    report: dict,
) -> None:
    body = dict(report)
    body.pop("withheld_artifacts")
    body.pop("gate_conditions")
    assert_p03_names(sorted(body))
    assert_p03_names(sorted(body["materialization"]))


def test_the_report_carries_no_absolute_path() -> None:
    text = REPORT.read_text(encoding="ascii")
    assert "/Users/" not in text
    assert "/home/" not in text
