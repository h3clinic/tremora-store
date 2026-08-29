"""Checks on the published PADS-P0.2 release audit.

The report is hash-pinned.  Editing it to change a verdict, or regenerating it
from a different release, dependency or implementation, fails here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from motionbloom.tremora_store.pads.p02.contract import (
    GATE_PASS,
    SUCCESS_MARKER,
    assert_p02_names,
)
from motionbloom.tremora_store.pads.p02.dependency import FROZEN_DEPENDENCY
from motionbloom.tremora_store.pads.p02.gate import (
    GATE_CONDITIONS,
    failing_conditions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "detector/benchmarks/pads_p02_release_audit.json"
REPORT_SHA256 = (
    "8e5eb21cf8ecafcadc26a5a0bcdb37a4bd5bad0088a33bf42d1939b45b1f41eb"
)
EVIDENCE_SHA256 = (
    "7ca16981b3bce63c5b8262fc0efe8855a73b9a8435e2aa485dc6e79ae898a139"
)
STORAGE_INDEX_SHA256 = (
    "22aeeb036cfe2cc6e1e0cc63d2142f75c5754b3bcf2b545aa5a77340f76420f1"
)

STREAMS = 10_318
SAMPLES = 13_447_168


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_bytes())


def test_the_published_report_is_hash_pinned() -> None:
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_SHA256


def test_the_gate_passed_on_every_condition(report: dict) -> None:
    assert report["gate_status"] == GATE_PASS
    assert failing_conditions(report) == ()
    assert report["gate_conditions_satisfied"] == len(GATE_CONDITIONS) == 16


def test_it_was_authorized_by_the_pinned_p01_pass(report: dict) -> None:
    dependency = report["p01_dependency"]
    assert dependency["dependency_status"] == (
        "P01_AUTHORITY_DEPENDENCY_VERIFIED"
    )
    assert dependency["pinned"] == FROZEN_DEPENDENCY.as_record()
    assert dependency["observed_evidence_sha256"] == (
        FROZEN_DEPENDENCY.p01_evidence_sha256
    )
    assert dependency["observed_report_sha256"] == (
        FROZEN_DEPENDENCY.p01_report_sha256
    )


#: The release-audit counts the milestone specification requires.  Encoded
#: as a test so a future report cannot quietly stop publishing one.
REQUIRED_COUNTS = (
    "participants_expected",
    "participants_materialized",
    "assessments_expected",
    "assessments_materialized",
    "streams_expected",
    "streams_materialized",
    "samples_expected",
    "samples_materialized",
    "duplicate_materialized_samples",
    "missing_materialized_samples",
    "parquet_files",
    "row_groups",
    "streams_with_exactly_one_row_group",
    "segments",
    "segment_partition_failures",
    "detected_time_gaps",
    "windows",
    "windows_crossing_segments",
    "window_replay_failures",
    "bilateral_task_pairs",
    "bilateral_window_pairs",
    "sample_level_alignment_claims",
    "fold_count",
    "participants_in_multiple_folds",
    "run_a_canonical_hash",
    "run_b_canonical_hash",
    "independent_reproduction_status",
)


def test_every_required_release_count_is_published(report: dict) -> None:
    published = (
        set(report)
        | set(report["materialization"])
        | set(report["replay_verification"])
        | set(report["reproduction_receipts"])
        | {f"{name}_expected" for name in report["expected"]}
    )
    missing = [name for name in REQUIRED_COUNTS if name not in published]
    assert missing == []


def test_the_headline_reconciliation_holds_in_all_three_terms(
    report: dict,
) -> None:
    # 13,447,168 = source = stored = replayed.  The third term is the samples
    # the store actually handed back, not the samples it was told to hold.
    assert report["expected"]["samples"] == SAMPLES
    assert report["materialization"]["samples_materialized"] == SAMPLES
    assert report["replay_verification"]["samples_replayed"] == SAMPLES
    assert report["materialization"]["missing_materialized_samples"] == 0


def test_every_source_sample_is_stored_exactly_once(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["samples_materialized"] == SAMPLES
    assert materialization["duplicate_materialized_samples"] == 0
    assert materialization["missing_materialized_samples"] == 0
    assert materialization["streams_materialized"] == STREAMS
    assert materialization["streams_refused"] == 0
    assert report["expected"]["samples"] == SAMPLES


def test_each_stream_is_exactly_one_row_group(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["row_groups"] == STREAMS
    assert materialization["streams_with_exactly_one_row_group"] == STREAMS
    # 10,318 streams packed 256 to a part.
    assert materialization["parquet_files"] == -(-STREAMS // 256) == 41


def test_the_whole_corpus_replays_byte_exactly(report: dict) -> None:
    verification = report["replay_verification"]
    assert verification["streams_checked"] == STREAMS
    assert verification["streams_byte_exact"] == STREAMS
    assert verification["streams_failed"] == 0
    assert verification["source_time_token_failures"] == 0
    assert verification["window_replay_failures"] == 0
    assert verification["storage_index_content_sha256"] == (
        STORAGE_INDEX_SHA256
    )


def test_the_gaps_the_release_actually_contains(report: dict) -> None:
    materialization = report["materialization"]
    # 14,729 segments over 10,318 streams: 4,411 real gaps above the
    # cadence-relative threshold.  Gap-aware windowing is load-bearing here.
    assert materialization["segments"] == 14_729
    assert materialization["detected_time_gaps"] == 4_411
    assert materialization["segment_partition_failures"] == 0


def test_no_window_crosses_a_segment(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["windows"] == 50_676
    assert materialization["windows_crossing_segments"] == 0
    assert report["replay_verification"]["windows_checked"] == 50_676


def test_bilateral_pairs_are_complete_but_never_sample_aligned(
    report: dict,
) -> None:
    materialization = report["materialization"]
    assert materialization["bilateral_task_pairs"] == 5_159
    # Fewer pairs than half the windows: a window whose partner fell inside a
    # gap on the other wrist has no partner, and is not invented.
    assert materialization["bilateral_window_pairs"] == 23_928
    assert materialization["bilateral_window_pairs"] < (
        materialization["windows"] // 2
    )
    assert materialization["sample_level_alignment_claims"] == 0
    authority = report["authority"]
    assert authority["bilateral_pairing_authority"] == "SOURCE_PROTOCOL_PAIR"
    assert authority["cross_wrist_clock_alignment"] == "UNRESOLVED"
    assert authority["sample_level_bilateral_fusion_allowed"] is False


def test_folds_are_participant_disjoint_and_balanced(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["fold_count"] == 5
    assert materialization["participants_in_multiple_folds"] == 0
    assert materialization["participants_without_fold"] == 0
    sizes = materialization["fold_sizes"]
    assert sum(sizes.values()) == 469
    assert max(sizes.values()) - min(sizes.values()) <= 4


def test_every_referenced_source_file_was_verified_again(report: dict) -> None:
    materialization = report["materialization"]
    assert materialization["source_files_hash_verified"] == 11_256
    assert materialization["source_files_failed"] == 0
    assert materialization["failure_count"] == 0


def test_two_separate_processes_produced_the_same_evidence(
    report: dict,
) -> None:
    receipts = report["reproduction_receipts"]
    run_a, run_b = receipts["run_a"], receipts["run_b"]
    assert report["canonical_evidence_sha256"] == EVIDENCE_SHA256
    assert receipts["run_a_canonical_hash"] == EVIDENCE_SHA256
    assert receipts["run_b_canonical_hash"] == EVIDENCE_SHA256
    assert run_a["canonical_evidence_sha256"] == EVIDENCE_SHA256
    assert run_b["canonical_evidence_sha256"] == EVIDENCE_SHA256
    assert run_a["run_id"] != run_b["run_id"]
    assert run_a["process_id"] != run_b["process_id"]
    assert run_a["output_root_identity"] != run_b["output_root_identity"]
    assert report["independent_reproduction_status"] == (
        "BYTE_IDENTICAL_PADS_P02_PASS"
    )


def test_source_time_is_published_in_picoseconds(report: dict) -> None:
    authority = report["authority"]
    assert authority["time_scale_decimals"] == 12
    assert authority["source_time_decimals"] == 10
    assert authority["time_basis"] == "SOURCE_TIME_COLUMN"
    assert authority["timing_authority"] == "SOURCE_RELATIVE_UNIMODAL_CLOCK"


def test_the_withheld_milestones_are_published_as_zero(report: dict) -> None:
    assert set(report["withheld_artifacts"].values()) == {0}
    for name in (
        "spectral_feature_tables", "tremor_frequency_tables",
        "band_power_tables", "resampled_signal_tables",
        "anti_alias_filter_outputs", "classification_tables",
        "video_association_tables", "comparative_benchmark_tables",
        "generic_success_markers",
    ):
        assert report["withheld_artifacts"][name] == 0


def test_the_report_names_no_forbidden_artifact_of_its_own(
    report: dict,
) -> None:
    body = dict(report)
    body.pop("withheld_artifacts")
    body.pop("gate_conditions")
    assert_p02_names(sorted(body))
    assert_p02_names(sorted(body["materialization"]))
    assert_p02_names(sorted(body["replay_verification"]))


def test_the_marker_is_specific_to_this_milestone() -> None:
    assert SUCCESS_MARKER == "_PADS_P02_INDEX_SUCCESS"


def test_the_report_carries_no_absolute_path() -> None:
    text = REPORT.read_text(encoding="ascii")
    assert "/Users/" not in text
    assert "/home/" not in text
