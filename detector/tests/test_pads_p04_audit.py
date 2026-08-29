"""The P0.4 audit end to end, on a synthetic release with a real P0.3 root."""

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
from motionbloom.tremora_store.pads.p03.contract import (
    SUCCESS_MARKER as P03_SUCCESS_MARKER,
)
from motionbloom.tremora_store.pads.p03.dependency import (
    observed_storage_index_hash,
)
from motionbloom.tremora_store.pads.p03.grid import grid_hash
from motionbloom.tremora_store.pads.p03.materialize import (
    materialize as p03_materialize,
)
from motionbloom.tremora_store.pads.p04.audit import (
    EVIDENCE_FILENAME,
    EXIT_BLOCKED,
    EXIT_NO_GO,
    RECEIPT_FILENAME,
    audit_pads_p04,
    main,
    measure_filters,
    pin_single_thread,
)
from motionbloom.tremora_store.pads.p04.contract import (
    GATE_PASS,
    SUCCESS_MARKER,
    assert_no_clinical_or_benchmark_claim,
)
from motionbloom.tremora_store.pads.p04.dependency import (
    dependency_record,
    observed_spectral_table_hash,
)
from motionbloom.tremora_store.pads.p04.filters import coefficients_sha256
from motionbloom.tremora_store.pads.p04.gate import (
    GATE_CONDITIONS,
    failing_conditions,
)
from motionbloom.tremora_store.pads.p04.schemas import P04_TABLE_FILES
from motionbloom.tremora_store.release_gate import canonical_json_bytes

PARTICIPANTS = ("001", "002", "003")


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("p04_audit")
    release_root = build_release(root, participants=PARTICIPANTS)

    store_root = root / "store"
    p02_materialize(
        release_root=release_root,
        output_root=store_root,
        p01_evidence_sha256="e" * 64,
        expected_samples=0,
    )
    (store_root / P02_SUCCESS_MARKER).write_bytes(b"")

    p03_root = root / "p03"
    p03_materialize(
        release_root=release_root,
        store_root=store_root,
        output_root=p03_root,
    )
    (p03_root / P03_SUCCESS_MARKER).write_bytes(b"")

    manifest = hashlib.sha256(
        (release_root / "SHA256SUMS.txt").read_bytes()
    ).hexdigest()
    index_hash = observed_storage_index_hash(store_root)
    report = {
        "gate_status": "PASS_PADS_SOURCE_TIME_SPECTRAL_PRESERVATION",
        "canonical_evidence_sha256": "a" * 64,
        "frequency_grid": {"frequency_grid_hash": grid_hash()},
        "p02_dependency": {"pinned": {
            "p02_evidence_sha256": "b" * 64,
            "storage_index_content_sha256": index_hash,
            "source_manifest_sha256": manifest,
        }},
    }
    report_path = root / "p03_report.json"
    report_path.write_bytes(canonical_json_bytes(report))

    pin = dependency_record()
    pin["pinned"].update({
        "p03_evidence_sha256": "a" * 64,
        "p03_report_sha256": hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest(),
        "p03_spectral_table_sha256": observed_spectral_table_hash(p03_root),
        "frequency_grid_sha256": grid_hash(),
        "p02_evidence_sha256": "b" * 64,
        "storage_index_content_sha256": index_hash,
        "source_manifest_sha256": manifest,
        "anti_alias_coefficients_sha256": coefficients_sha256(),
    })
    dependency_path = root / "dependency.json"
    dependency_path.write_bytes(canonical_json_bytes(pin))
    return {
        "root": root,
        "release_root": release_root,
        "store_root": store_root,
        "p03_root": p03_root,
        "report_path": report_path,
        "dependency_path": dependency_path,
    }


def _audit(bench: dict[str, Any], name: str, **kwargs):
    return audit_pads_p04(
        release_root=bench["release_root"],
        store_root=kwargs.pop("store_root", bench["store_root"]),
        p03_root=kwargs.pop("p03_root", bench["p03_root"]),
        output_root=bench["root"] / name,
        dependency_path=kwargs.pop(
            "dependency_path", bench["dependency_path"]
        ),
        p03_report_path=kwargs.pop("p03_report_path", bench["report_path"]),
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


# --- absence blocks, disagreement closes ----------------------------------


@pytest.mark.parametrize(
    ("key", "name"),
    (
        ("dependency_path", "blocked_dep"),
        ("p03_report_path", "blocked_report"),
        ("p03_root", "blocked_p03"),
        ("store_root", "blocked_store"),
    ),
)
def test_an_absent_input_blocks_rather_than_failing(
    bench: dict[str, Any], key: str, name: str
) -> None:
    record, receipt = _audit(bench, name, **{key: bench["root"] / "nowhere"})
    assert record["gate_evaluated"] is False
    assert record["release_status"] == "BLOCKED_P03_DEPENDENCY_UNAVAILABLE"
    assert record["materialized_release_artifacts"] == 0
    assert receipt is None


def test_a_moved_coefficient_set_closes_the_gate_and_materializes_nothing(
    bench: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Disagreement is not absence: the chain is present, so this is a
    # verdict rather than a block.
    pin = json.loads(bench["dependency_path"].read_bytes())
    pin["pinned"]["anti_alias_coefficients_sha256"] = "f" * 64
    moved = bench["root"] / "moved_dependency.json"
    moved.write_bytes(canonical_json_bytes(pin))
    report = json.loads(bench["report_path"].read_bytes())
    report_path = bench["root"] / "moved_report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    pin["pinned"]["p03_report_sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    moved.write_bytes(canonical_json_bytes(pin))

    record, _ = _audit(
        bench, "moved", dependency_path=moved, p03_report_path=report_path
    )
    assert record["gate_evaluated"] is True
    assert record["gate_status"] != GATE_PASS
    assert record["materialized_release_artifacts"] == 0
    assert "P03_DEPENDENCY_VERIFIED" in failing_conditions(record)


# --- the two runs ---------------------------------------------------------


def test_run_a_closes_only_on_reproduction(two_runs) -> None:
    first, _ = two_runs
    assert failing_conditions(first) == (
        "INDEPENDENT_MATERIALIZATION_REPRODUCED",
    )


def test_run_b_passes_every_condition(two_runs) -> None:
    _, second = two_runs
    assert failing_conditions(second) == ()
    assert second["gate_status"] == GATE_PASS
    assert second["gate_conditions_satisfied"] == len(GATE_CONDITIONS) == 18


def test_the_two_runs_agree_on_the_evidence_hash(two_runs) -> None:
    first, second = two_runs
    assert (
        first["canonical_evidence_sha256"]
        == second["canonical_evidence_sha256"]
    )
    # ...and disagree about where and by whom they ran.
    assert first["run_receipt"]["run_id"] != second["run_receipt"]["run_id"]
    assert (
        first["run_receipt"]["process_id"]
        != second["run_receipt"]["process_id"]
    )


def test_source_and_replay_derive_bit_identical_values(two_runs) -> None:
    _, second = two_runs
    produced = second["materialization"]
    assert produced["audit_comparisons"] > 0
    assert produced["source_replay_derived_mismatches"] == 0
    assert produced["source_replay_sample_mismatches"] == 0
    assert produced["source_replay_spectral_mismatches"] == 0
    assert produced["maximum_bin_absolute_error"] == 0.0


def test_the_parent_stage_removed_ordinals_the_filter_guard_would_admit(
    two_runs,
) -> None:
    _, second = two_runs
    produced = second["materialization"]
    # Nothing survived over an interval the parent could not bracket.
    assert produced["ordinals_admitted_over_unbracketed_parent"] == 0
    assert produced["rational_timing_ordinals_checked"] > 0
    # Whether this corpus's segments actually start off the parent grid is a
    # property of the corpus, so the removal itself is proved by the control,
    # on a segment built for the purpose.
    control = second["resampling_controls"]
    assert control["controls"][
        "support_intersection_precedes_the_filter_guard"
    ]
    removed = control["measured"]["support_intersection"][
        "ordinals_removed_by_parent_stage"
    ]
    assert set(removed) == {"100", "50", "30", "25"}
    assert all(count > 0 for count in removed.values())


def test_every_derived_ordinal_landed_on_its_exact_rational_time(
    two_runs,
) -> None:
    _, second = two_runs
    produced = second["materialization"]
    assert produced["rational_timing_ordinals_checked"] > 0
    assert produced["rational_timing_mismatches"] == 0
    assert produced["rounded_thirty_hz_ordinals"] == 0


def test_all_four_rates_were_materialized(two_runs) -> None:
    _, second = two_runs
    assert second["materialization"]["rates_materialized"] == [25, 30, 50, 100]
    assert second["derived_rates_hz"] == [100, 50, 30, 25]


def test_the_filters_were_measured_in_this_process(two_runs) -> None:
    _, second = two_runs
    filters = second["anti_alias_filters"]
    assert filters["coefficients_sha256"] == coefficients_sha256()
    assert filters["worst_passband_ripple_db"] <= 0.25
    assert filters["worst_stopband_attenuation_db"] >= 60.0
    assert set(filters["measured"]) == {"25", "30", "50"}


def test_the_branch_gains_are_published_unnormalized(two_runs) -> None:
    _, second = two_runs
    terms = second["anti_alias_filters"]["dc_terminology"]["30"]
    assert terms["per_phase_normalization"] is False
    assert terms["upsampling_factor"] == 3
    assert terms["effective_dc_gain"] == pytest.approx(1.0, abs=1e-12)
    assert len(terms["polyphase_dc_gains"]) == 3
    assert terms["polyphase_dc_gain_spread_db"] > 0.0
    assert len(set(terms["polyphase_dc_gains"])) > 1


def test_the_resampling_controls_ran_in_this_process(two_runs) -> None:
    _, second = two_runs
    controls = second["resampling_controls"]
    assert controls["status"] == "RESAMPLING_CONTROLS_PASS"
    assert controls["controls_passed"] == controls["controls_total"] == 11
    assert (
        controls["measured"]["constant_input_30"]["within_phase_spread"] == 0.0
    )


def test_the_core_and_edge_bands_are_summarized_separately(two_runs) -> None:
    _, second = two_runs
    produced = second["materialization"]
    assert produced["core_summary_rows"] > 0
    assert produced["edge_summary_rows"] == produced["core_summary_rows"]
    assert (
        produced["core_summary_rows"] + produced["edge_summary_rows"]
        == produced["participant_summary_rows"]
    )


def test_every_table_is_written_and_named_within_the_milestone(
    bench: dict[str, Any], two_runs
) -> None:
    run_b = bench["root"] / "run_b"
    for filename in P04_TABLE_FILES.values():
        assert (run_b / filename).is_file()
    assert_no_clinical_or_benchmark_claim(P04_TABLE_FILES.values())
    assert_no_clinical_or_benchmark_claim(
        path.name for path in run_b.iterdir()
    )


def test_the_marker_is_specific_and_only_on_pass(
    bench: dict[str, Any], two_runs
) -> None:
    assert (bench["root"] / "run_b" / SUCCESS_MARKER).is_file()
    assert not (bench["root"] / "run_b" / "_SUCCESS").exists()
    # Run A did not reproduce, so it carries no marker at all.
    assert not (bench["root"] / "run_a" / SUCCESS_MARKER).exists()


def test_the_run_pins_single_threaded_numerics(two_runs) -> None:
    _, second = two_runs
    assert second["numeric_execution"]["blas_used"] is False
    assert set(pin_single_thread().values()) == {"1"}


def test_the_withheld_milestones_are_published_as_zero(two_runs) -> None:
    _, second = two_runs
    withheld = second["withheld_artifacts"]
    assert withheld
    assert set(withheld.values()) == {0}
    for name in (
        "classification_tables", "video_association_tables",
        "storage_benchmark_tables", "generic_success_markers",
    ):
        assert withheld[name] == 0


def test_measure_filters_reports_all_three_rates() -> None:
    filters = measure_filters()
    assert set(filters["measured"]) == {"25", "30", "50"}
    assert set(filters["dc_terminology"]) == {"25", "30", "50"}


# --- the CLI --------------------------------------------------------------


def _cli(bench: dict[str, Any], name: str, extra: list[str]) -> int:
    return main([
        "--release-root", str(bench["release_root"]),
        "--store-root", str(bench["store_root"]),
        "--p03-root", str(bench["p03_root"]),
        "--output-root", str(bench["root"] / name),
        "--dependency", str(bench["dependency_path"]),
        "--p03-report", str(bench["report_path"]),
        *extra,
    ])


def test_cli_exits_three_on_a_single_run(bench: dict[str, Any]) -> None:
    assert _cli(bench, "cli_single", []) == EXIT_NO_GO
    record = json.loads(
        (bench["root"] / "cli_single" / EVIDENCE_FILENAME).read_bytes()
    )
    assert failing_conditions(record) == (
        "INDEPENDENT_MATERIALIZATION_REPRODUCED",
    )
    assert (bench["root"] / "cli_single" / RECEIPT_FILENAME).is_file()


def test_cli_exits_four_when_the_dependency_is_absent(
    bench: dict[str, Any],
) -> None:
    code = main([
        "--release-root", str(bench["release_root"]),
        "--store-root", str(bench["store_root"]),
        "--p03-root", str(bench["p03_root"]),
        "--output-root", str(bench["root"] / "cli_blocked"),
        "--dependency", str(bench["root"] / "nowhere.json"),
        "--p03-report", str(bench["report_path"]),
    ])
    assert code == EXIT_BLOCKED


def test_the_release_driver_spawns_two_processes_and_passes(
    bench: dict[str, Any],
) -> None:
    import subprocess
    import sys

    driver = (
        Path(__file__).resolve().parents[1]
        / "benchmarks" / "audit_pads_p04_release.py"
    )
    output_root = bench["root"] / "driver"
    completed = subprocess.run(
        [
            sys.executable, str(driver),
            "--release-root", str(bench["release_root"]),
            "--store-root", str(bench["store_root"]),
            "--p03-root", str(bench["p03_root"]),
            "--output-root", str(output_root),
            "--dependency", str(bench["dependency_path"]),
            "--p03-report", str(bench["report_path"]),
        ],
        check=False, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    summary = json.loads(completed.stdout)
    assert summary["gate_status"] == GATE_PASS
    assert summary["failing_conditions"] == []
    assert summary["run_a_canonical_hash"] == summary["run_b_canonical_hash"]
    assert summary["success_marker_present"] is True
    assert summary["generic_success_marker_present"] is False
