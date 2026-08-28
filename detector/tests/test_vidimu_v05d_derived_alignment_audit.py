from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from benchmarks import audit_vidimu_v05d_derived_alignment as audit
from motionbloom.tremora_store.v05d.source_transform import canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "detector/benchmarks/audit_vidimu_v05d_derived_alignment.py"
CHECKED_REPORT = (
    REPO_ROOT
    / "detector/benchmarks/vidimu_v05d_derived_alignment_release_audit.json"
)
SNAPSHOT = (
    REPO_ROOT
    / "data/snapshots/vidimu"
    / "a6e2194aee5478718e6f92cf9306214e361b08bb61363998f1e6e59e7378f1eb"
)
V05_SCRIPT = REPO_ROOT / "detector/benchmarks/audit_vidimu_v05_sync_authority.py"
V05_REPORT = (
    REPO_ROOT / "detector/benchmarks/vidimu_v05_sync_authority_audit.json"
)
EXPECTED_REPORT_SHA256 = (
    "131a6110d699ed8d0ebd7611c820112f1fe6af5c0e44f116181bd4a8495ac1b0"
)
EXPECTED_SCRIPT_SHA256 = (
    "fe2c6c2e9193cc33d187443a3d884c663a58a6e63933a2aa224e872d1c5fb1d0"
)


def _checked() -> dict[str, object]:
    payload = CHECKED_REPORT.read_bytes()
    value = json.loads(payload)
    assert canonical_json_bytes(value) == payload
    return value


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(child) for child in value))
    return set()


def test_checked_release_audit_has_exact_nogo_headline() -> None:
    value = _checked()
    assert hashlib.sha256(CHECKED_REPORT.read_bytes()).hexdigest() == (
        EXPECTED_REPORT_SHA256
    )
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == EXPECTED_SCRIPT_SHA256
    assert value["audit_execution_status"] == "PASS"
    assert value["raw_native_clock_gate"] == "NO_GO_RAW_NATIVE_CLOCK_AUTHORITY"
    assert value["source_derived_alignment_gate"] == "NO_GO"
    assert value["source_derived_alignment_gate_reason"] == (
        "NO_GO_V05D_CONTRACT_SEMANTICS"
    )
    assert value["overrides_expected"] == 217
    assert value["overrides_parsed"] == 217
    assert value["overrides_bound"] == 217
    assert value["overrides_reproduced"] == 217
    assert value["overrides_unreproduced"] == 0
    assert value["source_derived_pairs_eligible"] == 0
    assert value["source_derived_candidate_pairs_upper_bound"] == 206


def test_s53_a13_t03_remains_ambiguous() -> None:
    value = _checked()
    pair = next(
        row
        for row in value["ambiguous_pair_contracts"]
        if row["recording_id"] == "S53_A13_T03"
    )
    assert pair["alignment_authority"] == "AMBIGUOUS_SOURCE_ALIGNMENT"
    assert pair["eligibility_status"] == "EXCLUDED"


def test_s57_a07_t01_remains_ambiguous() -> None:
    value = _checked()
    pair = next(
        row
        for row in value["ambiguous_pair_contracts"]
        if row["recording_id"] == "S57_A07_T01"
    )
    assert pair["alignment_authority"] == "AMBIGUOUS_SOURCE_ALIGNMENT"
    assert pair["eligibility_status"] == "EXCLUDED"


def test_ambiguous_recordings_emit_no_mapping() -> None:
    value = _checked()
    assert value["ambiguous_source_pairs"] == 2
    assert value["ambiguous_pair_ids"] == ["S53_A13_T03", "S57_A07_T01"]
    for pair in value["ambiguous_pair_contracts"]:
        assert pair["chosen_mapping"] is None
        assert pair["range_index_created"] is False
        assert pair["window_created"] is False
        assert pair["eligibility_override_applied"] is False


def test_no_canonical_time_fields_emitted() -> None:
    keys = _all_keys(_checked())
    assert "canonical_time_ns" not in keys
    assert "clock_offset_ns" not in keys
    assert "sync_residual_ns" not in keys
    assert _checked()["canonical_clocks_created"] == 0


def test_no_clock_drift_fields_emitted() -> None:
    keys = _all_keys(_checked())
    assert "drift_ppm" not in keys
    assert "clock_scale" not in keys
    assert _checked()["clock_segments_created"] == 0


def test_no_generic_success_marker_emitted() -> None:
    value = _checked()
    assert value["generic_success_markers_created"] == 0
    assert value["sto_success_markers_created"] == 0
    assert value["withheld_artifact_counts"]["generic_success_markers"] == 0
    assert not (CHECKED_REPORT.parent / "_SUCCESS").exists()
    assert not (
        CHECKED_REPORT.parent / "_STO_DERIVED_ALIGNMENT_SUCCESS"
    ).exists()


def test_sto_success_marker_requires_full_reconciliation() -> None:
    value = _checked()
    assert audit.sto_success_marker_allowed(value) is False
    partially_overridden = dict(value)
    partially_overridden["source_derived_alignment_gate"] = "PASS"
    assert audit.sto_success_marker_allowed(partially_overridden) is False

    superficial_pass = copy.deepcopy(value)
    superficial_pass.update({
        "derived_rate_contracts_created": 1,
        "imu_tick_groups_created": 1,
        "source_derived_alignment_gate": "PASS",
        "sto_alignment_contracts_created": 1,
    })
    assert audit.sto_success_marker_allowed(superficial_pass) is False

    fully_reconciled = copy.deepcopy(value)
    fully_reconciled.update({
        "blockers": [],
        "byte_identical_sto_materialization": True,
        "derived_rate_contracts_created": 1,
        "imu_tick_groups_created": 1,
        "run_canonical_hash_basis": "BYTE_IDENTICAL_STO_MATERIALIZATION",
        "source_derived_alignment_gate": "PASS",
        "source_derived_alignment_gate_reason": "PASS",
        "source_derived_pairs_eligible": 206,
        "source_trim_overlays_created": 1,
        "sto_alignment_contracts_created": 1,
        "sto_alignment_validation_created": 1,
    })
    fully_reconciled["materialization_claim_boundary"][
        "derived_alignment_parquet_emitted"
    ] = True
    fully_reconciled.pop("withheld_artifact_counts")
    assert audit.sto_success_marker_allowed(fully_reconciled) is False
    for forbidden_claim in (
        "canonical_clock_claimed",
        "canonical_time_fields_emitted",
        "exact_physical_sample_times_claimed",
        "frame_to_imu_index_emitted",
        "hardware_synchronization_claimed",
        "measured_drift_claimed",
        "native_clock_mapping_claimed",
        "sto_success_marker_emitted",
        "timestamp_level_alignment_accuracy_claimed",
        "windows_emitted",
    ):
        forged = copy.deepcopy(fully_reconciled)
        forged["materialization_claim_boundary"][forbidden_claim] = True
        assert audit.sto_success_marker_allowed(forged) is False


def test_release_report_binds_all_v05d_implementations() -> None:
    value = _checked()
    pins = value["source_evidence"]["frozen_inputs"]
    assert value["implementation"]["authority_contract_sha256"] == (
        pins["v05d_authority_module_sha256"]
    )
    assert value["implementation"]["schema_contract_sha256"] == (
        pins["v05d_schema_module_sha256"]
    )
    assert value["implementation"]["source_transform_sha256"] == (
        pins["v05d_source_transform_module_sha256"]
    )


def test_sto_release_audit_fails_closed() -> None:
    value = _checked()
    evidence = copy.deepcopy(value["source_evidence"])
    evidence["raw_poll_to_sto_mot_reconciliation"][
        "raw_groups_are_nominal_50hz_ticks"
    ] = True
    with pytest.raises(
        audit.DerivedAlignmentAuditError,
        match="does not satisfy the frozen v0.5D audit basis",
    ):
        audit._assert_exact_source_evidence(evidence)

    missing_override = copy.deepcopy(value["source_evidence"])
    missing_override["source_trim_reproduction"]["overrides_bound"] = 216
    with pytest.raises(audit.DerivedAlignmentAuditError):
        audit._assert_exact_source_evidence(missing_override)


def test_no_materialization_artifact_claimed_on_nogo() -> None:
    value = _checked()
    assert value["sto_alignment_contracts_created"] == 0
    assert value["imu_tick_groups_created"] == 0
    assert value["derived_rate_contracts_created"] == 0
    assert value["source_trim_overlays_created"] == 0
    assert value["sto_alignment_validation_created"] == 0
    assert value["byte_identical_sto_materialization"] is False
    assert value["materialization_claim_boundary"] == {
        "canonical_clock_claimed": False,
        "canonical_time_fields_emitted": False,
        "derived_alignment_parquet_emitted": False,
        "exact_physical_sample_times_claimed": False,
        "frame_to_imu_index_emitted": False,
        "hardware_synchronization_claimed": False,
        "measured_drift_claimed": False,
        "native_clock_mapping_claimed": False,
        "sto_success_marker_emitted": False,
        "timestamp_level_alignment_accuracy_claimed": False,
        "windows_emitted": False,
    }


@pytest.mark.skipif(os.name != "posix", reason="secure publication requires POSIX")
def test_audit_publication_is_atomic_and_no_replace() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "audit.json"
        audit._write_exclusive(destination, b"complete\n")
        assert destination.read_bytes() == b"complete\n"
        assert list(Path(temporary).glob(".audit.json.tmp-*")) == []
        with pytest.raises(audit.DerivedAlignmentAuditError, match="already exists"):
            audit._write_exclusive(destination, b"replacement\n")
        assert destination.read_bytes() == b"complete\n"


@pytest.mark.skipif(os.name != "posix", reason="secure publication requires POSIX")
def test_fsync_failure_publishes_no_audit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "audit.json"
        with (
            mock.patch.object(audit.os, "fsync", side_effect=OSError("injected")),
            pytest.raises(OSError, match="injected"),
        ):
            audit._write_exclusive(destination, b"partial")
        assert not destination.exists()
        assert list(Path(temporary).glob(".audit.json.tmp-*")) == []


@pytest.mark.skipif(os.name != "posix", reason="secure publication requires POSIX")
def test_short_writes_cannot_publish_a_truncated_audit() -> None:
    real_write = os.write

    def short_write(descriptor: int, payload: bytes | memoryview) -> int:
        limited = payload[:max(1, len(payload) // 2)]
        return real_write(descriptor, limited)

    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "audit.json"
        payload = b"complete-audit-payload\n"
        with mock.patch.object(audit.os, "write", side_effect=short_write):
            audit._write_exclusive(destination, payload)
        assert destination.read_bytes() == payload
        assert list(Path(temporary).glob(".audit.json.tmp-*")) == []


@pytest.mark.skipif(os.name != "posix", reason="secure publication requires POSIX")
def test_parent_directory_swap_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        parent = root / "publish"
        moved = root / "moved"
        parent.mkdir()
        destination = parent / "audit.json"
        real_open_chain = audit._open_directory_chain
        parent_opens = 0

        def racing_open_chain(
            path: Path,
            flags: int,
        ) -> int:
            nonlocal parent_opens
            parent_opens += 1
            if parent_opens == 2:
                os.rename(parent, moved)
                parent.mkdir()
            return real_open_chain(path, flags)

        with (
            mock.patch.object(
                audit,
                "_open_directory_chain",
                side_effect=racing_open_chain,
            ),
            pytest.raises(
                audit.DerivedAlignmentAuditError,
                match="output parent changed during publication",
            ),
        ):
            audit._write_exclusive(destination, b"complete\n")
        assert not (parent / "audit.json").exists()
        assert not (moved / "audit.json").exists()
        assert list(moved.glob(".audit.json.tmp-*")) == []


@pytest.mark.skipif(os.name != "posix", reason="secure publication requires POSIX")
def test_ancestor_symlink_swap_cannot_redirect_into_frozen_input() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        outer = root / "outer"
        parent = outer / "publish"
        moved = root / "moved"
        frozen = root / "frozen"
        parent.mkdir(parents=True)
        (frozen / "publish").mkdir(parents=True)
        destination = parent / "audit.json"
        real_open_chain = audit._open_directory_chain
        raced = False

        def racing_open_chain(path: Path, flags: int) -> int:
            nonlocal raced
            if not raced:
                raced = True
                os.rename(outer, moved)
                outer.symlink_to(frozen, target_is_directory=True)
            return real_open_chain(path, flags)

        with (
            mock.patch.object(
                audit,
                "_open_directory_chain",
                side_effect=racing_open_chain,
            ),
            pytest.raises(
                audit.DerivedAlignmentAuditError,
                match="cannot pin output parent directory",
            ),
        ):
            audit._write_exclusive(destination, b"complete\n", forbidden=(frozen,))
        assert not (moved / "publish/audit.json").exists()
        assert not (frozen / "publish/audit.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="secure publication requires POSIX")
def test_output_cannot_alias_or_enter_a_frozen_input() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        frozen = Path(temporary) / "frozen"
        frozen.mkdir()
        destination = frozen / "audit.json"
        with pytest.raises(
            audit.DerivedAlignmentAuditError,
            match="output must not alias or enter an input",
        ):
            audit._write_exclusive(destination, b"audit\n", (frozen,))
        assert not destination.exists()


def test_cli_execution_error_publishes_no_report() -> None:
    arguments = [
        "--snapshot-root", "snapshot",
        "--analysis-archive", "analysis",
        "--tools-archive", "tools",
        "--v05-authority-script", "script",
        "--v05-authority-report", "report",
    ]
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "output.json"
        with (
            mock.patch.object(
                audit,
                "audit_vidimu_v05d_derived_alignment",
                side_effect=audit.DerivedAlignmentAuditError("injected"),
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            assert audit.main([*arguments, "--output", str(output)]) == 1
        assert not output.exists()


@pytest.mark.skipif(
    os.environ.get("VIDIMU_V05D_REAL_INPUTS") != "1",
    reason="set VIDIMU_V05D_REAL_INPUTS=1 for pinned release proof",
)
def test_two_process_withheld_materialization_ledger_identical() -> None:
    analysis = os.environ["VIDIMU_V05D_ANALYSIS_ARCHIVE"]
    tools = os.environ["VIDIMU_V05D_TOOLS_ARCHIVE"]

    def command(output: Path) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT),
            "--snapshot-root", str(SNAPSHOT),
            "--analysis-archive", analysis,
            "--tools-archive", tools,
            "--v05-authority-script", str(V05_SCRIPT),
            "--v05-authority-report", str(V05_REPORT),
            "--output", str(output),
        ]

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_a = root / "run-a"
        run_b = root / "run-b"
        run_a.mkdir()
        run_b.mkdir()
        outputs = (run_a / "audit.json", run_b / "audit.json")
        processes = [
            subprocess.Popen(
                command(output),
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for output in outputs
        ]
        results = [process.communicate(timeout=90) for process in processes]
        assert processes[0].pid != processes[1].pid
        for process, (stdout, stderr), output in zip(
            processes, results, outputs, strict=True
        ):
            assert process.returncode == audit.NO_GO_EXIT_CODE, stderr.decode()
            assert stdout == output.read_bytes()
            assert sorted(path.name for path in output.parent.iterdir()) == [
                "audit.json"
            ]
        assert outputs[0].stat().st_ino != outputs[1].stat().st_ino
        assert outputs[0].read_bytes() == outputs[1].read_bytes()
        assert outputs[0].read_bytes() == CHECKED_REPORT.read_bytes()
        assert hashlib.sha256(outputs[0].read_bytes()).hexdigest() == (
            EXPECTED_REPORT_SHA256
        )
        value = json.loads(outputs[0].read_bytes())
        assert value["run_a_canonical_hash"] == value["run_b_canonical_hash"]
        assert value["byte_identical_sto_materialization"] is False
