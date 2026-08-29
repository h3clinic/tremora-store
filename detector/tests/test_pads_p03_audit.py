from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from _pads_fixtures import build_release
from motionbloom.tremora_store.pads.p02.contract import (
    SUCCESS_MARKER as P02_SUCCESS_MARKER,
)
from motionbloom.tremora_store.pads.p02.materialize import (
    materialize as p02_materialize,
)
from motionbloom.tremora_store.pads.p03 import audit as audit_module
from motionbloom.tremora_store.pads.p03.audit import (
    EVIDENCE_FILENAME,
    EXIT_BLOCKED,
    EXIT_NO_GO,
    RECEIPT_FILENAME,
    audit_pads_p03,
    pin_single_thread,
)
from motionbloom.tremora_store.pads.p03.contract import (
    GATE_PASS,
    SUCCESS_MARKER,
    assert_p03_names,
)
from motionbloom.tremora_store.pads.p03.dependency import (
    dependency_record,
    observed_storage_index_hash,
)
from motionbloom.tremora_store.pads.p03.gate import (
    GATE_CONDITIONS,
    P02_1_DEPENDENCY_VERIFIED,
    failing_conditions,
)
from motionbloom.tremora_store.pads.p03.schemas import (
    FREQUENCY_GRID_FILENAME,
    P03_TABLE_FILES,
)
from motionbloom.tremora_store.release_gate import canonical_json_bytes

PARTICIPANTS = ("001", "002", "003")


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("p03_audit")
    release_root = build_release(root, participants=PARTICIPANTS)
    store_root = root / "store"
    p02_materialize(
        release_root=release_root,
        output_root=store_root,
        p01_evidence_sha256="e" * 64,
        expected_samples=0,
    )
    (store_root / P02_SUCCESS_MARKER).write_bytes(b"")

    manifest = hashlib.sha256(
        (release_root / "SHA256SUMS.txt").read_bytes()
    ).hexdigest()
    report = {
        "gate_status": "PASS_PADS_INDEX_AND_WINDOW_AUTHORITY",
        "canonical_evidence_sha256": "b" * 64,
        "p01_dependency": {"pinned": {
            "p01_evidence_sha256": "e" * 64,
            "source_manifest_sha256": manifest,
        }},
        "replay_verification": {
            "storage_index_content_sha256": observed_storage_index_hash(
                store_root
            ),
        },
    }
    report_path = root / "p02_report.json"
    report_path.write_bytes(canonical_json_bytes(report))

    pin = dependency_record()
    pin["pinned"].update({
        "p01_evidence_sha256": "e" * 64,
        "p02_evidence_sha256": "b" * 64,
        "p02_report_sha256": hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest(),
        "source_manifest_sha256": manifest,
        "storage_index_content_sha256": observed_storage_index_hash(
            store_root
        ),
    })
    dependency_path = root / "dependency.json"
    dependency_path.write_bytes(canonical_json_bytes(pin))
    return {
        "root": root,
        "release_root": release_root,
        "store_root": store_root,
        "report_path": report_path,
        "dependency_path": dependency_path,
    }


def _audit(bench: dict[str, Any], name: str, **kwargs):
    return audit_pads_p03(
        release_root=bench["release_root"],
        store_root=kwargs.pop("store_root", bench["store_root"]),
        output_root=bench["root"] / name,
        dependency_path=kwargs.pop(
            "dependency_path", bench["dependency_path"]
        ),
        p02_report_path=kwargs.pop("p02_report_path", bench["report_path"]),
        **kwargs,
    )


@pytest.fixture(scope="module")
def two_runs(bench: dict[str, Any]):
    first, receipt = _audit(bench, "run_a", run_id="r1", process_id=101)
    second, _ = _audit(
        bench, "run_b", run_id="r2", process_id=202,
        reproduction_receipt=receipt,
    )
    return first, second


# --- blocked means the authority is absent --------------------------------


@pytest.mark.parametrize(
    ("key", "name"),
    (
        ("dependency_path", "blocked_dep"),
        ("p02_report_path", "blocked_report"),
        ("store_root", "blocked_store"),
    ),
)
def test_an_absent_authority_blocks(
    bench: dict[str, Any], tmp_path: Path, key: str, name: str
) -> None:
    record, receipt = _audit(bench, name, **{key: tmp_path / "absent"})
    assert record["release_status"] == "BLOCKED_P02_DEPENDENCY_UNAVAILABLE"
    assert record["gate_evaluated"] is False
    assert "gate_status" not in record
    assert "canonical_evidence_sha256" not in record
    assert receipt is None


def test_a_substituted_store_closes_the_gate_and_materializes_nothing(
    bench: dict[str, Any], tmp_path: Path
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    other = tmp_path / "other_store"
    other.mkdir()
    pq.write_table(
        pa.table({
            "stream_id": ["001:Relaxed:LeftWrist"],
            "row_group_content_sha256": ["f" * 64],
        }),
        other / "pads_stream_storage_index.parquet",
    )
    (other / P02_SUCCESS_MARKER).write_bytes(b"")
    output = bench["root"] / "substituted"
    record, _ = _audit(bench, "substituted", store_root=other)
    assert record["release_status"] == "EVALUATED"
    assert P02_1_DEPENDENCY_VERIFIED in failing_conditions(record)
    assert record["materialization"] == {"materialized": False}
    assert record["materialized_release_artifacts"] == 0
    assert not (output / SUCCESS_MARKER).exists()
    assert not (output / P03_TABLE_FILES["pads_p03_spectra"]).exists()


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
    assert second["independent_reproduction_status"] == (
        "BYTE_IDENTICAL_PADS_P03_PASS"
    )


def test_the_two_runs_agree_on_the_evidence_hash(two_runs) -> None:
    first, second = two_runs
    assert first["canonical_evidence_sha256"] == (
        second["canonical_evidence_sha256"]
    )
    assert first["run_receipt"]["run_id"] != second["run_receipt"]["run_id"]


def test_source_and_replay_are_bit_identical(two_runs) -> None:
    _, second = two_runs
    materialization = second["materialization"]
    assert materialization["source_replay_row_mismatches"] == 0
    assert materialization["source_replay_input_hash_mismatches"] == 0
    assert materialization["source_replay_spectral_hash_mismatches"] == 0
    assert materialization["dominant_frequency_mismatches"] == 0
    assert materialization["maximum_observed_bin_error"] == 0.0
    assert materialization["source_unreadable"] == 0


def test_no_nominal_grid_was_substituted(two_runs) -> None:
    _, second = two_runs
    materialization = second["materialization"]
    assert materialization["nominal_grid_substitutions"] == 0
    # The probe has teeth: every window's stored times genuinely differ from
    # an ordinal/rate grid.
    assert materialization["windows_differing_from_nominal_grid"] == (
        materialization["workload_windows_eligible"]
    )


def test_nyquist_came_from_the_stream_cadence(two_runs) -> None:
    _, second = two_runs
    materialization = second["materialization"]
    assert materialization["nyquist_derived_from_dt_ref_rows"] == (
        materialization["workload_windows_eligible"]
    )
    assert materialization["declared_rate_nyquist_rows"] == 0


def test_raw_axes_are_preserved_and_summed(two_runs) -> None:
    _, second = two_runs
    assert second["materialization"]["raw_axis_sum_mismatches"] == 0
    assert second["materialization"]["vector_magnitude_uses"] == 0
    assert second["authority"]["vector_magnitude_primary_signal"] is False


def test_the_kernel_controls_ran_in_this_process(two_runs) -> None:
    _, second = two_runs
    controls = second["kernel_controls"]
    assert controls["status"] == "SYNTHETIC_KERNEL_CONTROLS_PASS"
    assert controls["controls_passed"] == controls["controls_total"] == 12
    assert controls["controls"]["vector_magnitude_would_double_frequency"]
    assert controls["controls"]["source_time_not_ordinal_over_rate"]


def test_the_frequency_grid_is_published_and_frozen(
    bench: dict[str, Any], two_runs
) -> None:
    _, second = two_runs
    assert second["frequency_grid"]["frequency_bin_count"] == 37
    grid = json.loads(
        (bench["root"] / "run_b" / FREQUENCY_GRID_FILENAME).read_bytes()
    )
    assert len(grid["frequency_values"]) == 37
    assert grid["frequency_values"][0] == 3.0
    assert grid["frequency_values"][-1] == 12.0
    assert grid["rayleigh_resolution_hz"] == 0.25


def test_every_table_is_written_and_named_within_the_milestone(
    bench: dict[str, Any], two_runs
) -> None:
    import pyarrow.parquet as pq

    root = bench["root"] / "run_b"
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        assert_p03_names([path.name])
        if path.suffix == ".parquet":
            assert_p03_names(pq.read_schema(path).names)
    spectra = pq.read_table(root / P03_TABLE_FILES["pads_p03_spectra"])
    assert spectra.num_rows == (
        two_runs[1]["materialization"]["workload_windows_eligible"] * 2
    )


def test_the_marker_is_specific_and_only_on_pass(
    bench: dict[str, Any], two_runs
) -> None:
    assert (bench["root"] / "run_b" / SUCCESS_MARKER).is_file()
    assert not (bench["root"] / "run_b" / "_SUCCESS").exists()
    assert not (bench["root"] / "run_a" / SUCCESS_MARKER).exists()


def test_the_run_pins_single_threaded_numerics(two_runs) -> None:
    _, second = two_runs
    execution = second["numeric_execution"]
    assert execution["blas_used"] is False
    assert set(execution["threading"].values()) == {"1"}
    assert pin_single_thread()["OMP_NUM_THREADS"] == "1"


def test_the_withheld_milestones_are_published_as_zero(two_runs) -> None:
    _, second = two_runs
    assert set(second["withheld_artifacts"].values()) == {0}
    for name in (
        "resampled_signal_tables", "anti_alias_filter_outputs",
        "derived_rate_tables", "classification_tables",
        "tremor_detection_tables", "video_association_tables",
    ):
        assert second["withheld_artifacts"][name] == 0


# --- CLI ------------------------------------------------------------------


def test_cli_exits_three_on_a_single_run(bench: dict[str, Any]) -> None:
    output = bench["root"] / "cli_a"
    code = audit_module.main([
        "--release-root", str(bench["release_root"]),
        "--store-root", str(bench["store_root"]),
        "--output-root", str(output),
        "--dependency", str(bench["dependency_path"]),
        "--p02-report", str(bench["report_path"]),
    ])
    assert code == EXIT_NO_GO
    assert (output / EVIDENCE_FILENAME).is_file()
    assert (output / RECEIPT_FILENAME).is_file()
    assert not (output / SUCCESS_MARKER).exists()


def test_cli_exits_four_when_the_dependency_is_absent(
    bench: dict[str, Any], tmp_path: Path
) -> None:
    code = audit_module.main([
        "--release-root", str(bench["release_root"]),
        "--store-root", str(bench["store_root"]),
        "--output-root", str(bench["root"] / "cli_blocked"),
        "--dependency", str(tmp_path / "absent.json"),
        "--p02-report", str(bench["report_path"]),
    ])
    assert code == EXIT_BLOCKED


def test_the_release_driver_spawns_two_processes_and_passes(
    bench: dict[str, Any],
) -> None:
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    output = bench["root"] / "driver"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "detector/benchmarks/audit_pads_p03_release.py"),
            "--release-root", str(bench["release_root"]),
            "--store-root", str(bench["store_root"]),
            "--output-root", str(output),
            "--dependency", str(bench["dependency_path"]),
            "--p02-report", str(bench["report_path"]),
        ],
        check=False, capture_output=True, cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "detector"), "PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr.decode()
    summary = json.loads(
        (output / "pads_p03_release_summary.json").read_bytes()
    )
    assert summary["gate_status"] == GATE_PASS
    assert summary["failing_conditions"] == []
    assert summary["run_a_exit_code"] == EXIT_NO_GO
    assert summary["run_a_canonical_hash"] == summary["run_b_canonical_hash"]
    assert summary["independent_reproduction_status"] == (
        "BYTE_IDENTICAL_PADS_P03_PASS"
    )
    assert summary["success_marker_present"] is True
    assert summary["generic_success_marker_present"] is False
