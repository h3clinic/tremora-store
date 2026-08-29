"""Checks on the published PADS-P0.1 release audit.

The report is hash-pinned.  Editing it to change a verdict, or regenerating it
from a different release or implementation, fails here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from motionbloom.tremora_store.pads.authority import GATE_PASS
from motionbloom.tremora_store.pads.gate import GATE_CONDITIONS, failing_conditions
from motionbloom.tremora_store.pads.release_structure import (
    PADS_EXPECTED_ASSESSMENTS,
    PADS_EXPECTED_PARTICIPANTS,
    PADS_EXPECTED_STREAMS,
)
from motionbloom.tremora_store.release_gate import (
    EXIT_PASS,
    canonical_json_bytes,
    exit_code_for,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "detector/benchmarks/pads_p01_release_audit.json"
REPORT_SHA256 = (
    "6d2e0fab4bbcc3762e70c95b30b48293c17d785d3db9877288a4efa75f03a749"
)
EVIDENCE_SHA256 = (
    "e25ce02f7cc023061f5840e314564b153a7829ff487c996b59b25798cf4c801a"
)
SOURCE_MANIFEST_SHA256 = (
    "514cd95405a12afcdfb126d47d1f559e2e8a744f03e586c3088d5e4fd7b02c46"
)


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_bytes())


def test_the_published_report_is_hash_pinned() -> None:
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_SHA256


def test_the_report_is_canonical_json(report: dict) -> None:
    assert REPORT.read_bytes() == canonical_json_bytes(report)


def test_the_gate_passed_on_every_condition(report: dict) -> None:
    assert report["gate_status"] == GATE_PASS
    assert failing_conditions(report) == ()
    assert report["gate_conditions_satisfied"] == len(GATE_CONDITIONS) == 14
    assert exit_code_for(report) == EXIT_PASS


def test_the_release_reconciled_exactly(report: dict) -> None:
    structure = report["release_structure"]
    assert structure["release_structure_status"] == (
        "PADS_RELEASE_STRUCTURE_RECONCILED"
    )
    assert structure["observed_participants"] == PADS_EXPECTED_PARTICIPANTS
    assert structure["observed_assessments"] == PADS_EXPECTED_ASSESSMENTS
    assert structure["observed_streams"] == PADS_EXPECTED_STREAMS
    assert structure["failure_count"] == 0


def test_every_referenced_source_file_was_hash_verified(report: dict) -> None:
    # 10,318 timeseries files plus 469 observation and 469 patient records.
    assert report["streams"]["hash_verified"] == 11_256
    assert report["streams"]["hash_failed"] == 0
    assert report["source_failure_count"] == 0
    assert report["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256


def test_every_declared_stream_parsed(report: dict) -> None:
    assert report["streams"]["declared"] == PADS_EXPECTED_STREAMS
    assert report["streams"]["parsed"] == PADS_EXPECTED_STREAMS
    assert report["streams"]["refused"] == 0
    assert report["streams"]["failure_count"] == 0


def test_the_source_time_column_carried_the_whole_corpus(report: dict) -> None:
    assert report["authority"]["relative_time_basis"] == "SOURCE_TIME_COLUMN"
    assert report["samples"]["total"] == 13_447_168
    # Six sensor channels per sample, every one usable.
    assert report["samples"]["usable_sensor_values"] == (
        report["samples"]["total"] * 6
    )
    assert report["samples"]["duplicate_time"] == 0
    assert report["samples"]["nonmonotonic_time"] == 0


def test_the_published_cadence_agrees_with_the_declared_rate(
    report: dict,
) -> None:
    assert report["streams"]["cadence_deviating"] == 0
    assert report["streams"]["span_deviating"] == 0
    assert report["streams"]["noncanonical_source_order"] == 0


def test_the_row_counts_came_from_the_release_metadata(report: dict) -> None:
    per_task = report["per_task"]
    assert len(per_task) == 11
    long_tasks = {
        task for task, counts in per_task.items()
        if counts["declared_rows"] == 2048
    }
    assert long_tasks == {"Relaxed", "RelaxedTask", "Entrainment"}
    for counts in per_task.values():
        assert counts["declared_rows"] in {1024, 2048}
        assert counts["streams"] == counts["parsed"] == 938
        assert counts["refused"] == 0


def test_support_and_span_stayed_one_sample_period_apart(
    report: dict,
) -> None:
    relaxed = report["per_task"]["Relaxed"]
    assert relaxed["expected_sample_support_seconds"] == 20.48
    assert relaxed["expected_first_to_last_span_seconds"] == 20.47
    stretch = report["per_task"]["StretchHold"]
    assert stretch["expected_sample_support_seconds"] == 10.24
    assert stretch["expected_first_to_last_span_seconds"] == 10.23


def test_two_separate_processes_produced_the_same_evidence(
    report: dict,
) -> None:
    receipts = report["reproduction_receipts"]
    run_a, run_b = receipts["run_a"], receipts["run_b"]
    assert run_a["canonical_evidence_sha256"] == EVIDENCE_SHA256
    assert run_b["canonical_evidence_sha256"] == EVIDENCE_SHA256
    assert report["canonical_evidence_sha256"] == EVIDENCE_SHA256
    assert run_a["run_id"] != run_b["run_id"]
    assert run_a["process_id"] != run_b["process_id"]
    assert run_a["output_root_identity"] != run_b["output_root_identity"]
    assert run_a["source_manifest_sha256"] == run_b["source_manifest_sha256"]


def test_the_report_claims_no_paired_or_derived_artifact(
    report: dict,
) -> None:
    assert report["authority"]["video_pairing"] == "NOT_APPLICABLE"
    assert report["authority"]["modality"] == "INERTIAL_ONLY_NO_VIDEO"
    assert report["authority"]["hardware_sync_claim"] is False
    assert report["authority"]["timing_authority"] == (
        "SOURCE_RELATIVE_UNIMODAL_CLOCK"
    )
    assert report["materialized_release_artifacts"] == 0
    assert set(report["withheld_artifacts"].values()) == {0}


def test_the_report_carries_no_absolute_path() -> None:
    text = REPORT.read_text(encoding="ascii")
    assert "/Users/" not in text
    assert "/home/" not in text
