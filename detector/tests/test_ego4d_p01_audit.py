from __future__ import annotations

import json
from pathlib import Path

import pytest
from _ego4d_fixtures import (
    VideoSpec,
    build_release,
    clean_rows,
    decoder_for,
    imu_csv,
    sha256_bytes,
)
from motionbloom.tremora_store.ego4d import audit as audit_module
from motionbloom.tremora_store.ego4d.audit import audit_ego4d_p01
from motionbloom.tremora_store.ego4d.authority import GATE_NO_GO, GATE_PASS
from motionbloom.tremora_store.ego4d.gate import (
    ALL_ASSETS_HASH_VERIFIED,
    AUDIT_REPRODUCES_BYTE_IDENTICALLY,
    EVERY_IMU_ROW_REPRESENTED,
    EVERY_SELECTED_VIDEO_HAS_PTS_TIMELINE,
    GATE_CONDITIONS,
    NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED,
    SUBSET_FLOORS_SATISFIED,
)
from motionbloom.tremora_store.release_gate import (
    EXIT_BLOCKED,
    EXIT_NO_GO,
    EXIT_PASS,
    canonical_json_bytes,
    exit_code_for,
)

SMALL_FLOORS = {
    "minimum_videos": 6,
    "minimum_coverage_hours": 0.0001,
    "minimum_capture_device_groups": 2,
}


def _all_strata_specs() -> list[VideoSpec]:
    """Six videos, one per stratum, so the floors can actually be met."""

    null_rows = clean_rows(count=40)
    for row in null_rows[10:20]:
        row[2] = ""

    nonmonotonic_rows = clean_rows(count=40)
    nonmonotonic_rows[30][2] = "1.0"

    missing_accl_rows = clean_rows(count=40)
    for row in missing_accl_rows[5:9]:
        row[6] = ""

    extreme_rows = clean_rows(count=40)
    extreme_rows[35][2] = "999999999999.0"

    return [
        VideoSpec("V_CLEAN", rows=clean_rows(count=40),
                  capture_device_group="GROUP_A"),
        VideoSpec("V_NULL", rows=null_rows, capture_device_group="GROUP_B"),
        VideoSpec("V_NONMONO", rows=nonmonotonic_rows,
                  capture_device_group="GROUP_A"),
        VideoSpec("V_ACCL", rows=missing_accl_rows,
                  capture_device_group="GROUP_B"),
        VideoSpec("V_EXTREME", rows=extreme_rows,
                  capture_device_group="GROUP_A"),
        VideoSpec("V_PARTIAL", rows=clean_rows(count=40),
                  component_count=2, components_with_imu=2,
                  capture_device_group="GROUP_B"),
    ]


def _audit(root: Path, specs, **kwargs):
    metadata_root, imu_root, video_root = build_release(root, specs)
    return audit_ego4d_p01(
        metadata_root=metadata_root,
        imu_root=imu_root,
        video_root=kwargs.pop("video_root", video_root),
        publication_destination=kwargs.pop(
            "publication_destination", str(root / "run_a.json")
        ),
        decode_frame_times=decoder_for(specs),
        **{**SMALL_FLOORS, **kwargs},
    )


def _condition(record, name: str) -> bool:
    for condition in record["gate_conditions"]:
        if condition["condition"] == name:
            return bool(condition["satisfied"])
    raise AssertionError(f"no condition named {name}")


# --- blocked means absent, never malformed --------------------------------


def test_an_absent_metadata_root_blocks(tmp_path: Path) -> None:
    record = audit_ego4d_p01(
        metadata_root=tmp_path / "missing",
        imu_root=tmp_path / "imu",
        video_root=None,
        publication_destination=str(tmp_path / "out.json"),
    )
    assert record["release_status"] == "BLOCKED_INPUT_DATA_UNAVAILABLE"
    assert record["gate_evaluated"] is False
    assert "gate_status" not in record
    assert "canonical_evidence_sha256" not in record
    assert exit_code_for(record) == EXIT_BLOCKED


def test_an_empty_asset_manifest_closes_the_gate_rather_than_blocking(
    tmp_path: Path,
) -> None:
    metadata_root, imu_root, video_root = build_release(tmp_path, [])
    record = audit_ego4d_p01(
        metadata_root=metadata_root,
        imu_root=imu_root,
        video_root=video_root,
        publication_destination=str(tmp_path / "out.json"),
    )
    assert record["release_status"] == "EVALUATED"
    assert record["gate_status"] == GATE_NO_GO
    assert exit_code_for(record) == EXIT_NO_GO


def test_an_absent_video_root_closes_the_gate_rather_than_blocking(
    tmp_path: Path,
) -> None:
    record = _audit(tmp_path, _all_strata_specs(), video_root=None)
    assert record["release_status"] == "EVALUATED"
    assert _condition(record, EVERY_SELECTED_VIDEO_HAS_PTS_TIMELINE) is False
    assert record["gate_status"] == GATE_NO_GO


def test_an_unparseable_asset_closes_the_gate_rather_than_blocking(
    tmp_path: Path,
) -> None:
    specs = _all_strata_specs()
    broken = imu_csv(clean_rows(count=4)).replace(b"\n", b"\n\n", 1)
    specs[0] = VideoSpec(
        "V_CLEAN", imu_payload=broken, capture_device_group="GROUP_A"
    )
    record = _audit(tmp_path, specs)
    assert record["release_status"] == "EVALUATED"
    assert record["assets"]["failed"] >= 1
    assert _condition(record, ALL_ASSETS_HASH_VERIFIED) is False
    assert record["gate_status"] == GATE_NO_GO


def test_a_traversing_manifest_entry_closes_the_gate(tmp_path: Path) -> None:
    metadata_root, imu_root, video_root = build_release(
        tmp_path, _all_strata_specs()
    )
    manifest_path = metadata_root / "ego4d_asset_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["assets"][0]["imu_relative_path"] = "../../escape.csv"
    manifest_path.write_text(json.dumps(payload))
    record = audit_ego4d_p01(
        metadata_root=metadata_root,
        imu_root=imu_root,
        video_root=video_root,
        publication_destination=str(tmp_path / "out.json"),
    )
    assert record["release_status"] == "EVALUATED"
    assert record["source_failures"]
    assert record["gate_status"] == GATE_NO_GO


def test_a_hash_mismatch_is_a_failed_asset(tmp_path: Path) -> None:
    specs = _all_strata_specs()
    specs[0] = VideoSpec(
        "V_CLEAN",
        rows=clean_rows(count=40),
        declared_imu_sha256=sha256_bytes(b"different"),
        capture_device_group="GROUP_A",
    )
    record = _audit(tmp_path, specs)
    statuses = {
        item["video_uid"]: item["asset_status"]
        for item in record["assets"]["records"]
    }
    assert statuses["V_CLEAN"] == "ASSET_HASH_MISMATCH"
    assert record["gate_status"] == GATE_NO_GO


def test_a_missing_asset_file_is_a_failed_asset(tmp_path: Path) -> None:
    specs = _all_strata_specs()
    specs[0] = VideoSpec(
        "V_CLEAN",
        rows=clean_rows(count=40),
        write_imu_file=False,
        capture_device_group="GROUP_A",
    )
    record = _audit(tmp_path, specs)
    statuses = {
        item["video_uid"]: item["asset_status"]
        for item in record["assets"]["records"]
    }
    assert statuses["V_CLEAN"] == "ASSET_MISSING"
    assert record["gate_status"] == GATE_NO_GO


# --- accounting -----------------------------------------------------------


def test_rows_are_counted_against_an_independently_derived_line_count(
    tmp_path: Path,
) -> None:
    record = _audit(tmp_path, _all_strata_specs())
    assert record["rows"]["authority_rows"] == 240
    assert record["rows"]["source_data_lines"] == 240
    assert _condition(record, EVERY_IMU_ROW_REPRESENTED) is True


def test_no_timestamp_is_inferred_and_every_issue_row_is_classified(
    tmp_path: Path,
) -> None:
    record = _audit(tmp_path, _all_strata_specs())
    assert record["rows"]["inferred_timestamps"] == 0
    assert record["rows"]["unclassified_issue_rows"] == 0
    assert record["rows"]["token_preservation_failures"] == 0
    assert record["rows"]["valid_rows_outside_video_interval"] == 0
    assert record["rows"]["files_with_ordinal_gaps"] == 0


def test_the_summary_accounts_for_every_row(tmp_path: Path) -> None:
    record = _audit(tmp_path, _all_strata_specs())
    summaries = {
        item["video_uid"]: item
        for item in record["timing_authority_summary"]
    }
    assert summaries["V_NULL"]["canonical_rows_null"] == 10
    assert summaries["V_NULL"]["canonical_rows_valid"] == 30
    assert summaries["V_ACCL"]["rows_missing_acceleration"] == 4
    assert summaries["V_EXTREME"]["canonical_rows_extreme"] == 1
    assert summaries["V_NONMONO"]["canonical_rows_nonmonotonic_source_order"] == 1


def test_an_all_unusable_corpus_cannot_pass(tmp_path: Path) -> None:
    rows = clean_rows(count=40)
    for row in rows:
        row[2] = ""
    specs = [
        VideoSpec(f"V{index}", rows=[list(row) for row in rows],
                  capture_device_group="GROUP_A" if index % 2 else "GROUP_B")
        for index in range(6)
    ]
    record = _audit(tmp_path, specs)
    assert record["gate_status"] == GATE_NO_GO
    assert _condition(record, SUBSET_FLOORS_SATISFIED) is False
    assert record["subset_selection"]["paired_coverage_hours"] == 0.0


# --- the claim boundary ---------------------------------------------------


def test_no_index_or_window_artifact_is_emitted(tmp_path: Path) -> None:
    record = _audit(tmp_path, _all_strata_specs())
    assert record["withheld_artifacts"] == {
        "contiguous_window_tables": 0,
        "frame_imu_index_tables": 0,
        "spectral_feature_tables": 0,
        "storage_benchmark_result_tables": 0,
        "success_markers": 0,
    }
    assert record["materialized_release_artifacts"] == 0
    assert _condition(record, NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED) is True


def test_row_frame_relationships_are_aggregate_only(tmp_path: Path) -> None:
    record = _audit(tmp_path, _all_strata_specs())
    for item in record["row_frame_relationships"]:
        assert set(item) == {
            "video_uid",
            "eligible_row_count",
            "rows_inside_a_frame_interval",
            "rows_before_first_frame",
            "rows_after_last_frame",
            "max_nearest_frame_delta_ms",
            "median_nearest_frame_delta_ms",
        }


def test_the_gate_condition_set_is_frozen(tmp_path: Path) -> None:
    record = _audit(tmp_path, _all_strata_specs())
    assert tuple(
        item["condition"] for item in record["gate_conditions"]
    ) == GATE_CONDITIONS
    assert record["gate_conditions_total"] == 11


# --- reproduction ---------------------------------------------------------


def test_one_execution_cannot_satisfy_the_reproduction_condition(
    tmp_path: Path,
) -> None:
    record = _audit(tmp_path, _all_strata_specs())
    assert _condition(record, AUDIT_REPRODUCES_BYTE_IDENTICALLY) is False
    assert record["gate_status"] == GATE_NO_GO


def test_two_clean_roots_agree_on_the_evidence_hash(tmp_path: Path) -> None:
    specs = _all_strata_specs()
    run_a = _audit(
        tmp_path / "a", specs, publication_destination="/roots/a/report.json"
    )
    run_b = _audit(
        tmp_path / "b", specs, publication_destination="/roots/b/report.json"
    )
    assert (
        run_a["canonical_evidence_sha256"]
        == run_b["canonical_evidence_sha256"]
    )


def test_a_copied_report_does_not_prove_a_second_execution(
    tmp_path: Path,
) -> None:
    specs = _all_strata_specs()
    run_a = _audit(
        tmp_path / "a", specs, publication_destination="/roots/a/report.json"
    )
    copied = json.loads(canonical_json_bytes(run_a).decode("ascii"))
    verified = _audit(
        tmp_path / "a2",
        specs,
        publication_destination="/roots/a/report.json",
        reproduction_record=copied,
    )
    assert _condition(verified, AUDIT_REPRODUCES_BYTE_IDENTICALLY) is False


def test_a_clean_corpus_passes_with_a_genuine_second_execution(
    tmp_path: Path,
) -> None:
    specs = _all_strata_specs()
    run_a = _audit(
        tmp_path / "a", specs, publication_destination="/roots/a/report.json"
    )
    run_b = _audit(
        tmp_path / "b",
        specs,
        publication_destination="/roots/b/report.json",
        reproduction_record=run_a,
    )
    failing = [
        item["condition"]
        for item in run_b["gate_conditions"]
        if not item["satisfied"]
    ]
    assert failing == []
    assert run_b["gate_status"] == GATE_PASS
    assert exit_code_for(run_b) == EXIT_PASS


# --- CLI ------------------------------------------------------------------


def test_cli_blocks_with_exit_four_and_writes_a_canonical_record(
    tmp_path: Path, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    output = tmp_path / "report.json"
    code = audit_module.main([
        "--metadata-root", str(tmp_path / "missing"),
        "--imu-root", str(tmp_path / "imu"),
        "--output", str(output),
    ])
    assert code == EXIT_BLOCKED
    record = json.loads(output.read_text())
    assert record["release_status"] == "BLOCKED_INPUT_DATA_UNAVAILABLE"
    assert output.read_bytes() == canonical_json_bytes(record)
    assert not (tmp_path / "_SUCCESS").exists()


def test_cli_refuses_to_overwrite_an_existing_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("{}")
    code = audit_module.main([
        "--metadata-root", str(tmp_path / "missing"),
        "--imu-root", str(tmp_path / "imu"),
        "--output", str(output),
    ])
    assert code == 2
    assert output.read_text() == "{}"
