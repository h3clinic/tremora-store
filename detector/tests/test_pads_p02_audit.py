from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from _pads_fixtures import build_release
from motionbloom.tremora_store.pads.p02 import audit as audit_module
from motionbloom.tremora_store.pads.p02.audit import (
    EVIDENCE_FILENAME,
    EXIT_BLOCKED,
    EXIT_NO_GO,
    EXIT_PASS,
    RECEIPT_FILENAME,
    audit_pads_p02,
)
from motionbloom.tremora_store.pads.p02.contract import (
    GATE_PASS,
    SUCCESS_MARKER,
    assert_p02_names,
)
from motionbloom.tremora_store.pads.p02.dependency import dependency_record
from motionbloom.tremora_store.pads.p02.gate import (
    GATE_CONDITIONS,
    P01_AUTHORITY_DEPENDENCY_VERIFIED,
    failing_conditions,
)
from motionbloom.tremora_store.pads.p02.schemas import P02_INDEX_FILES
from motionbloom.tremora_store.release_gate import canonical_json_bytes

PARTICIPANTS = ("001", "002")
EXPECTED_SAMPLES = len(PARTICIPANTS) * 2 * (3 * 2048 + 8 * 1024)
EXPECTED_STREAMS = len(PARTICIPANTS) * 11 * 2
EXPECTED_ASSESSMENTS = len(PARTICIPANTS) * 11


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("p02_release")
    release_root = build_release(root, participants=PARTICIPANTS)
    manifest = hashlib.sha256(
        (release_root / "SHA256SUMS.txt").read_bytes()
    ).hexdigest()
    report = {
        "gate_status": "PASS_SOURCE_RELATIVE_UNIMODAL_CLOCK",
        "canonical_evidence_sha256": "e" * 64,
        "contract_version": (
            "tremora-pads-source-relative-unimodal-clock-0.1.0"
        ),
        "source_manifest_sha256": manifest,
        "release_structure": {
            "observed_participants": len(PARTICIPANTS),
            "observed_assessments": EXPECTED_ASSESSMENTS,
            "observed_streams": EXPECTED_STREAMS,
        },
        "samples": {"total": EXPECTED_SAMPLES},
    }
    report_path = root / "report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    pin = dependency_record()
    pin["pinned"].update({
        "p01_report_sha256": hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest(),
        "p01_evidence_sha256": "e" * 64,
        "source_manifest_sha256": manifest,
        "expected_participants": len(PARTICIPANTS),
        "expected_assessments": EXPECTED_ASSESSMENTS,
        "expected_streams": EXPECTED_STREAMS,
        "expected_samples": EXPECTED_SAMPLES,
    })
    dependency_path = root / "dependency.json"
    dependency_path.write_bytes(canonical_json_bytes(pin))
    return {
        "root": root,
        "release_root": release_root,
        "report_path": report_path,
        "dependency_path": dependency_path,
    }


def _audit(release: dict[str, Any], name: str, **kwargs):
    return audit_pads_p02(
        release_root=release["release_root"],
        output_root=release["root"] / name,
        dependency_path=kwargs.pop("dependency_path",
                                   release["dependency_path"]),
        p01_report_path=kwargs.pop("p01_report_path",
                                   release["report_path"]),
        **kwargs,
    )


@pytest.fixture(scope="module")
def two_runs(release: dict[str, Any]):
    first, receipt = _audit(release, "run_a", run_id="r1", process_id=101)
    second, _ = _audit(
        release, "run_b", run_id="r2", process_id=202,
        reproduction_receipt=receipt,
    )
    return first, second


# --- blocked means the authority is absent --------------------------------


def test_an_absent_dependency_file_blocks(
    release: dict[str, Any], tmp_path: Path
) -> None:
    record, receipt = _audit(
        release, "blocked_dep",
        dependency_path=tmp_path / "absent.json",
    )
    assert record["release_status"] == "BLOCKED_P01_DEPENDENCY_UNAVAILABLE"
    assert record["gate_evaluated"] is False
    assert "gate_status" not in record
    assert "canonical_evidence_sha256" not in record
    assert receipt is None


def test_an_absent_p01_report_blocks(
    release: dict[str, Any], tmp_path: Path
) -> None:
    record, _ = _audit(
        release, "blocked_report", p01_report_path=tmp_path / "absent.json"
    )
    assert record["release_status"] == "BLOCKED_P01_DEPENDENCY_UNAVAILABLE"
    assert record["gate_evaluated"] is False


def test_an_absent_release_root_blocks(
    release: dict[str, Any], tmp_path: Path
) -> None:
    record, _ = audit_pads_p02(
        release_root=tmp_path / "absent_release",
        output_root=release["root"] / "blocked_release",
        dependency_path=release["dependency_path"],
        p01_report_path=release["report_path"],
    )
    assert record["release_status"] == "BLOCKED_P01_DEPENDENCY_UNAVAILABLE"


# --- disagreement closes the gate instead ---------------------------------


def test_a_tampered_p01_report_closes_the_gate_and_materializes_nothing(
    release: dict[str, Any], tmp_path: Path
) -> None:
    tampered = tmp_path / "report.json"
    tampered.write_bytes(canonical_json_bytes({"gate_status": "NO_GO"}))
    output = release["root"] / "mismatch"
    record, _ = _audit(
        release, "mismatch", p01_report_path=tampered
    )
    assert record["release_status"] == "EVALUATED"
    assert P01_AUTHORITY_DEPENDENCY_VERIFIED in failing_conditions(record)
    assert record["materialized_release_artifacts"] == 0
    assert record["materialization"] == {"materialized": False}
    assert not (output / "samples").exists()
    assert not (output / SUCCESS_MARKER).exists()


# --- the materialized result ----------------------------------------------


def test_run_a_closes_only_on_reproduction(two_runs) -> None:
    first, _ = two_runs
    assert failing_conditions(first) == (
        "INDEPENDENT_MATERIALIZATION_REPRODUCED",
    )


def test_run_b_passes_every_condition(two_runs) -> None:
    _, second = two_runs
    assert failing_conditions(second) == ()
    assert second["gate_status"] == GATE_PASS
    assert second["gate_conditions_total"] == len(GATE_CONDITIONS) == 16


def test_the_two_runs_agree_on_the_evidence_hash(two_runs) -> None:
    first, second = two_runs
    assert first["canonical_evidence_sha256"] == (
        second["canonical_evidence_sha256"]
    )
    assert first["run_receipt"]["run_id"] != second["run_receipt"]["run_id"]
    assert first["run_receipt"]["output_root_identity"] != (
        second["run_receipt"]["output_root_identity"]
    )


def test_every_sample_is_stored_exactly_once(two_runs) -> None:
    _, second = two_runs
    materialization = second["materialization"]
    assert materialization["samples_materialized"] == EXPECTED_SAMPLES
    assert materialization["duplicate_materialized_samples"] == 0
    assert materialization["streams_materialized"] == EXPECTED_STREAMS
    assert materialization["streams_refused"] == 0


def test_each_stream_is_exactly_one_row_group(
    release: dict[str, Any], two_runs
) -> None:
    _, second = two_runs
    materialization = second["materialization"]
    assert materialization["row_groups"] == EXPECTED_STREAMS
    assert materialization["streams_with_exactly_one_row_group"] == (
        EXPECTED_STREAMS
    )
    parts = sorted((release["root"] / "run_b" / "samples").glob("*.parquet"))
    assert sum(
        pq.ParquetFile(path).num_row_groups for path in parts
    ) == EXPECTED_STREAMS


def test_every_stream_replays_byte_exactly(two_runs) -> None:
    _, second = two_runs
    verification = second["replay_verification"]
    assert verification["streams_checked"] == EXPECTED_STREAMS
    assert verification["streams_byte_exact"] == EXPECTED_STREAMS
    assert verification["streams_failed"] == 0
    assert verification["source_time_token_failures"] == 0
    assert verification["window_replay_failures"] == 0
    assert verification["windows_checked"] == (
        second["materialization"]["windows"]
    )


def test_every_index_table_is_written(release: dict[str, Any], two_runs) -> None:
    root = release["root"] / "run_b"
    for filename in P02_INDEX_FILES.values():
        assert (root / filename).is_file()
    windows = pq.read_table(root / "pads_windows.parquet")
    assert windows.num_rows == two_runs[1]["materialization"]["windows"]
    assert set(
        windows.column("timing_authority").to_pylist()
    ) == {"SOURCE_RELATIVE_UNIMODAL_CLOCK"}


def test_no_bilateral_row_claims_sample_level_alignment(
    release: dict[str, Any], two_runs
) -> None:
    root = release["root"] / "run_b"
    for filename in (
        "pads_bilateral_tasks.parquet", "pads_bilateral_window_pairs.parquet"
    ):
        table = pq.read_table(root / filename)
        assert set(
            table.column("sample_level_fusion_allowed").to_pylist()
        ) == {False}
        assert set(
            table.column("cross_wrist_clock_alignment").to_pylist()
        ) == {"UNRESOLVED"}
    assert two_runs[1]["materialization"][
        "sample_level_alignment_claims"
    ] == 0


def test_folds_are_participant_disjoint_in_the_published_index(
    release: dict[str, Any], two_runs
) -> None:
    table = pq.read_table(
        release["root"] / "run_b" / "pads_participants.parquet"
    )
    folds = dict(zip(
        table.column("participant_id").to_pylist(),
        table.column("outer_fold").to_pylist(),
        strict=True,
    ))
    assert set(folds) == set(PARTICIPANTS)
    assert all(fold >= 0 for fold in folds.values())
    assert two_runs[1]["materialization"]["participants_without_fold"] == 0


def test_the_success_marker_is_specific_and_only_on_pass(
    release: dict[str, Any], two_runs
) -> None:
    assert (release["root"] / "run_b" / SUCCESS_MARKER).is_file()
    assert not (release["root"] / "run_b" / "_SUCCESS").exists()
    assert not (release["root"] / "run_a" / SUCCESS_MARKER).exists()


def test_the_claim_boundary_is_published(two_runs) -> None:
    _, second = two_runs
    assert set(second["withheld_artifacts"].values()) == {0}
    assert second["materialized_release_artifacts"] == EXPECTED_STREAMS


def test_no_emitted_artifact_names_a_withheld_milestone(
    release: dict[str, Any], two_runs
) -> None:
    """The boundary is checked on what was written, not on the prose."""

    root = release["root"] / "run_b"
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        assert_p02_names([path.name])
        if path.suffix == ".parquet":
            schema = pq.read_schema(path)
            assert_p02_names(schema.names)


# --- CLI ------------------------------------------------------------------


def test_cli_exits_three_on_a_single_run(release: dict[str, Any]) -> None:
    output = release["root"] / "cli_a"
    code = audit_module.main([
        "--release-root", str(release["release_root"]),
        "--output-root", str(output),
        "--dependency", str(release["dependency_path"]),
        "--p01-report", str(release["report_path"]),
    ])
    assert code == EXIT_NO_GO
    assert (output / EVIDENCE_FILENAME).is_file()
    assert (output / RECEIPT_FILENAME).is_file()
    assert not (output / SUCCESS_MARKER).exists()


def test_a_second_run_in_the_same_process_is_not_a_reproduction(
    release: dict[str, Any],
) -> None:
    # The CLI cannot reproduce itself in one process: the receipts would share
    # a PID.  This is the check working, not a limitation to route around.
    output = release["root"] / "cli_same_process"
    code = audit_module.main([
        "--release-root", str(release["release_root"]),
        "--output-root", str(output),
        "--dependency", str(release["dependency_path"]),
        "--p01-report", str(release["report_path"]),
        "--reproduction-receipt",
        str(release["root"] / "cli_a" / RECEIPT_FILENAME),
    ])
    assert code == EXIT_NO_GO
    assert not (output / SUCCESS_MARKER).exists()


def test_the_release_driver_spawns_two_processes_and_passes(
    release: dict[str, Any],
) -> None:
    import json
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    output = release["root"] / "driver"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "detector/benchmarks/audit_pads_p02_release.py"),
            "--release-root", str(release["release_root"]),
            "--output-root", str(output),
            "--dependency", str(release["dependency_path"]),
            "--p01-report", str(release["report_path"]),
        ],
        check=False,
        capture_output=True,
        cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "detector"), "PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == EXIT_PASS, completed.stderr.decode()
    summary = json.loads(
        (output / "pads_p02_release_summary.json").read_bytes()
    )
    assert summary["gate_status"] == GATE_PASS
    assert summary["failing_conditions"] == []
    assert summary["run_a_exit_code"] == EXIT_NO_GO
    assert summary["run_b_exit_code"] == EXIT_PASS
    assert summary["run_a_canonical_hash"] == summary["run_b_canonical_hash"]
    assert summary["success_marker_present"] is True
    assert summary["generic_success_marker_present"] is False


def test_cli_exits_four_when_the_dependency_is_absent(
    release: dict[str, Any], tmp_path: Path
) -> None:
    output = release["root"] / "cli_blocked"
    code = audit_module.main([
        "--release-root", str(release["release_root"]),
        "--output-root", str(output),
        "--dependency", str(tmp_path / "absent.json"),
        "--p01-report", str(release["report_path"]),
    ])
    assert code == EXIT_BLOCKED
    assert not (output / RECEIPT_FILENAME).exists()


def test_cli_refuses_to_overwrite_an_existing_evidence_record(
    release: dict[str, Any],
) -> None:
    output = release["root"] / "cli_a"
    code = audit_module.main([
        "--release-root", str(release["release_root"]),
        "--output-root", str(output),
        "--dependency", str(release["dependency_path"]),
        "--p01-report", str(release["report_path"]),
    ])
    assert code == 2
