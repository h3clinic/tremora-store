from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from _pads_fixtures import (
    OBSERVED_INTERVAL,
    SAMPLING_RATE,
    build_release,
    timeseries_bytes,
)
from motionbloom.tremora_store.pads import audit as audit_module
from motionbloom.tremora_store.pads import movement
from motionbloom.tremora_store.pads.audit import audit_pads_p01
from motionbloom.tremora_store.pads.authority import (
    CANONICAL_CHANNELS,
    GATE_NO_GO,
    GATE_PASS,
    PadsAuthorityError,
    assert_no_paired_claim,
    assert_unimodal_authority,
    authority_contract,
    first_to_last_span_seconds,
    sample_support_seconds,
)
from motionbloom.tremora_store.pads.gate import (
    GATE_CONDITIONS,
    PADS_ALL_SOURCE_FILES_HASH_VERIFIED,
    PADS_INDEPENDENT_REPRODUCTION_VERIFIED,
    PADS_NO_VIDEO_ASSOCIATION_EMITTED,
    PADS_RELEASE_STRUCTURE_RECONCILED,
    failing_conditions,
)
from motionbloom.tremora_store.pads.movement import (
    StreamDeclaration,
    parse_timeseries,
    validate_declaration,
)
from motionbloom.tremora_store.pads.release_structure import (
    PADS_EXPECTED_ASSESSMENTS,
    PADS_EXPECTED_PARTICIPANTS,
    PADS_EXPECTED_STREAMS,
    PADS_EXPECTED_TASKS,
)
from motionbloom.tremora_store.pads.schemas import PADS_TABLE_SCHEMAS
from motionbloom.tremora_store.release_gate import (
    EXIT_BLOCKED,
    EXIT_NO_GO,
    EXIT_PASS,
    exit_code_for,
)

RATE = Fraction(SAMPLING_RATE)


def _declaration(
    *, channels=None, units=None, device_location="LeftWrist",
    file_name="timeseries/001_Relaxed_LeftWrist.txt",
) -> StreamDeclaration:
    from motionbloom.tremora_store.pads.authority import CHANNEL_UNITS

    names = tuple(channels or CANONICAL_CHANNELS)
    return StreamDeclaration(
        device_location=device_location,
        channels=names,
        units=tuple(units or [CHANNEL_UNITS.get(n, "?") for n in names]),
        file_name=file_name,
    )


def _audit(dataset_root: Path, output_root: Path, **kwargs):
    output_root.mkdir(parents=True, exist_ok=True)
    return audit_pads_p01(
        dataset_root=dataset_root,
        output_root=output_root,
        expected_participants=kwargs.pop("expected_participants", 1),
        **kwargs,
    )


# --- authority ------------------------------------------------------------


def test_pads_cannot_be_bound_to_a_paired_tier() -> None:
    for tier in ("RAW_SHARED_CLOCK", "SOURCE_CANONICAL_TIMESTAMP"):
        with pytest.raises(PadsAuthorityError):
            assert_unimodal_authority(tier)


def test_the_contract_names_the_source_time_column() -> None:
    contract = authority_contract()
    assert contract["relative_time_basis"] == "SOURCE_TIME_COLUMN"
    assert contract["video_pairing"] == "NOT_APPLICABLE"
    assert contract["modality"] == "INERTIAL_ONLY_NO_VIDEO"
    assert contract["hardware_sync_claim"] is False


@pytest.mark.parametrize(
    "name",
    ("video_uid", "video_uid_ref", "camera_stream_uid", "frame_id",
     "rgb_path", "pts_ns", "pixel_scale"),
)
def test_video_bearing_names_are_screened_by_substring(name: str) -> None:
    with pytest.raises(PadsAuthorityError):
        assert_no_paired_claim(["participant_id", name])


@pytest.mark.parametrize(
    "name",
    ("pts", "pts_ns", "video_pts", "frame_pts0", "PTS", "imu_pts_ms"),
)
def test_presentation_timestamp_names_are_refused(name: str) -> None:
    with pytest.raises(PadsAuthorityError):
        assert_no_paired_claim(["participant_id", name])


@pytest.mark.parametrize(
    "name", ("reproduction_receipts", "run_receipt", "receipts", "scripts")
)
def test_pts_matches_on_letter_boundaries_not_as_a_bare_infix(
    name: str,
) -> None:
    # A bare substring rule refuses "receipts", which is about execution
    # provenance and has nothing to do with presentation timestamps.  The
    # substring rule exists to catch extensions of a video-bearing name, not
    # arbitrary infixes.
    assert_no_paired_claim(["participant_id", name])


def test_support_and_span_differ_by_one_sample_period() -> None:
    assert float(sample_support_seconds(2048, RATE)) == 20.48
    assert float(first_to_last_span_seconds(2048, RATE)) == 20.47
    assert float(sample_support_seconds(1024, RATE)) == 10.24
    assert float(first_to_last_span_seconds(1024, RATE)) == 10.23


def test_a_recording_that_spans_the_support_has_one_sample_too_many() -> None:
    # A 2048-sample file whose timestamps span 20.48 s carries 2049 intervals.
    assert float(first_to_last_span_seconds(2049, RATE)) == 20.48


# --- declarations ---------------------------------------------------------


def test_the_canonical_declaration_needs_no_permutation() -> None:
    permutation, issues = validate_declaration(_declaration())
    assert permutation == (0, 1, 2, 3, 4, 5, 6)
    assert issues == ()


def test_a_permuted_declaration_parses_and_is_reported_not_refused() -> None:
    from motionbloom.tremora_store.pads.authority import CHANNEL_UNITS

    permuted = (
        "Gyroscope_X", "Time", "Accelerometer_X", "Accelerometer_Y",
        "Accelerometer_Z", "Gyroscope_Y", "Gyroscope_Z",
    )
    declaration = _declaration(
        channels=permuted,
        units=[CHANNEL_UNITS[name] for name in permuted],
    )
    permutation, issues = validate_declaration(declaration)
    assert issues == ("NONCANONICAL_SOURCE_ORDER",)
    # permutation[i] is the physical column supplying canonical channel i.
    assert permutation[0] == 1
    assert tuple(permuted[position] for position in permutation) == (
        CANONICAL_CHANNELS
    )


def test_a_permuted_declaration_reads_time_from_its_declared_column() -> None:
    from motionbloom.tremora_store.pads.authority import CHANNEL_UNITS

    permuted = (
        "Gyroscope_X", "Time", "Accelerometer_X", "Accelerometer_Y",
        "Accelerometer_Z", "Gyroscope_Y", "Gyroscope_Z",
    )
    declaration = _declaration(
        channels=permuted,
        units=[CHANNEL_UNITS[name] for name in permuted],
    )
    rows = [
        f"0.5,{index * OBSERVED_INTERVAL:.10f},0.1,0.2,0.3,0.4,0.5"
        for index in range(64)
    ]
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    stream = parse_timeseries(
        payload, declaration=declaration, declared_rows=64, sampling_rate=RATE
    )
    assert stream.stream_status == movement.STREAM_PARSED
    assert stream.times[0] == 0.0
    assert stream.observed_median_interval_seconds == pytest.approx(
        OBSERVED_INTERVAL, rel=1e-6
    )


@pytest.mark.parametrize(
    ("channels", "units", "expected"),
    (
        (
            ("Time", "Time", "Accelerometer_Y", "Accelerometer_Z",
             "Gyroscope_X", "Gyroscope_Y", "Gyroscope_Z"),
            ("s", "s", "g", "g", "rad/s", "rad/s", "rad/s"),
            movement.DUPLICATE_CHANNEL_NAME,
        ),
        (
            ("Time", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z",
             "Gyroscope_X", "Gyroscope_Y"),
            ("s", "g", "g", "g", "rad/s", "rad/s"),
            movement.MISSING_CHANNEL,
        ),
        (
            ("Time", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z",
             "Gyroscope_X", "Gyroscope_Y", "Magnetometer_X"),
            ("s", "g", "g", "g", "rad/s", "rad/s", "uT"),
            movement.UNKNOWN_CHANNEL,
        ),
        (
            CANONICAL_CHANNELS,
            ("s", "g", "g", "g", "rad/s", "rad/s", "deg/s"),
            movement.CHANNEL_UNIT_MISMATCH,
        ),
        (
            CANONICAL_CHANNELS,
            ("s", "g", "g", "g", "rad/s", "rad/s"),
            movement.CHANNEL_UNIT_LENGTH_MISMATCH,
        ),
    ),
)
def test_an_ambiguous_declaration_closes_the_records_gate(
    channels, units, expected: str
) -> None:
    stream = parse_timeseries(
        timeseries_bytes(8),
        declaration=_declaration(channels=channels, units=units),
        declared_rows=8,
        sampling_rate=RATE,
    )
    assert stream.stream_status == expected


def test_an_unrecognized_device_location_closes_the_records_gate() -> None:
    stream = parse_timeseries(
        timeseries_bytes(8),
        declaration=_declaration(device_location="Ankle"),
        declared_rows=8,
        sampling_rate=RATE,
    )
    assert stream.stream_status == movement.UNRECOGNIZED_DEVICE_LOCATION


def test_a_missing_device_location_closes_the_records_gate() -> None:
    stream = parse_timeseries(
        timeseries_bytes(8),
        declaration=_declaration(device_location=""),
        declared_rows=8,
        sampling_rate=RATE,
    )
    assert stream.stream_status == movement.MISSING_DEVICE_LOCATION


# --- timeseries -----------------------------------------------------------


def test_the_source_time_column_is_the_timeline() -> None:
    stream = parse_timeseries(
        timeseries_bytes(64),
        declaration=_declaration(), declared_rows=64, sampling_rate=RATE,
    )
    assert stream.stream_status == movement.STREAM_PARSED
    # Ordinal / rate would put sample 63 at 0.63 exactly; the source does not.
    assert stream.times[63] != 0.63
    assert stream.times[63] == pytest.approx(63 * OBSERVED_INTERVAL, abs=1e-9)


def test_an_unusable_time_value_closes_the_record(
) -> None:
    stream = parse_timeseries(
        timeseries_bytes(64, time_override={10: ""}),
        declaration=_declaration(), declared_rows=64, sampling_rate=RATE,
    )
    assert stream.stream_status == movement.INVALID_TIME
    assert stream.invalid_time_count == 1


def test_an_unparseable_time_token_is_never_replaced_by_the_rate() -> None:
    stream = parse_timeseries(
        timeseries_bytes(64, time_override={10: "1_000"}),
        declaration=_declaration(), declared_rows=64, sampling_rate=RATE,
    )
    assert stream.stream_status == movement.INVALID_TIME


def test_duplicate_time_stays_visible() -> None:
    stream = parse_timeseries(
        timeseries_bytes(64, time_override={10: f"{9 * OBSERVED_INTERVAL:.10f}"}),
        declaration=_declaration(), declared_rows=64, sampling_rate=RATE,
    )
    assert stream.stream_status == movement.STREAM_PARSED
    assert stream.duplicate_time_count == 1
    assert movement.DUPLICATE_TIME in stream.issue_codes


def test_non_monotonic_time_stays_visible() -> None:
    stream = parse_timeseries(
        timeseries_bytes(64, time_override={30: "0.0010000000"}),
        declaration=_declaration(), declared_rows=64, sampling_rate=RATE,
    )
    assert stream.stream_status == movement.STREAM_PARSED
    assert stream.nonmonotonic_time_count == 1
    assert movement.NONMONOTONIC_TIME in stream.issue_codes


def test_a_blank_source_row_is_a_parse_failure() -> None:
    stream = parse_timeseries(
        timeseries_bytes(64, blank_row_at=10),
        declaration=_declaration(), declared_rows=64, sampling_rate=RATE,
    )
    assert stream.stream_status == movement.BLANK_SOURCE_ROW


def test_a_row_count_disagreement_closes_the_record() -> None:
    stream = parse_timeseries(
        timeseries_bytes(63),
        declaration=_declaration(), declared_rows=64, sampling_rate=RATE,
    )
    assert stream.stream_status == movement.ROW_COUNT_MISMATCH


def test_a_column_count_disagreement_closes_the_record() -> None:
    stream = parse_timeseries(
        timeseries_bytes(64, columns=6),
        declaration=_declaration(), declared_rows=64, sampling_rate=RATE,
    )
    assert stream.stream_status == movement.ROW_COLUMN_COUNT_MISMATCH


def test_a_corpus_with_no_usable_value_is_refused() -> None:
    stream = parse_timeseries(
        timeseries_bytes(64, value_override="NaN"),
        declaration=_declaration(), declared_rows=64, sampling_rate=RATE,
    )
    assert stream.stream_status == movement.NO_USABLE_VALUES


def test_the_published_jitter_does_not_deviate_from_the_declared_rate(
) -> None:
    stream = parse_timeseries(
        timeseries_bytes(2048),
        declaration=_declaration(), declared_rows=2048, sampling_rate=RATE,
    )
    assert stream.issue_codes == ()


def test_a_halved_rate_does_deviate_from_the_declared_rate() -> None:
    stream = parse_timeseries(
        timeseries_bytes(2048, interval=0.02),
        declaration=_declaration(), declared_rows=2048, sampling_rate=RATE,
    )
    assert movement.CADENCE_DEVIATES_FROM_DECLARED_RATE in stream.issue_codes
    assert movement.SPAN_DEVIATES_FROM_DECLARED_RATE in stream.issue_codes


# --- release structure ----------------------------------------------------


def test_the_frozen_release_totals_follow_from_the_task_and_device_lists(
) -> None:
    assert len(PADS_EXPECTED_TASKS) == 11
    assert PADS_EXPECTED_ASSESSMENTS == PADS_EXPECTED_PARTICIPANTS * 11
    assert PADS_EXPECTED_STREAMS == PADS_EXPECTED_ASSESSMENTS * 2
    assert (PADS_EXPECTED_PARTICIPANTS, PADS_EXPECTED_ASSESSMENTS,
            PADS_EXPECTED_STREAMS) == (469, 5159, 10318)


def test_a_complete_synthetic_release_reconciles(tmp_path: Path) -> None:
    dataset_root = build_release(tmp_path)
    record, _ = _audit(dataset_root, tmp_path / "out")
    assert record["release_structure"]["release_structure_status"] == (
        "PADS_RELEASE_STRUCTURE_RECONCILED"
    )
    assert record["release_structure"]["observed_assessments"] == 11
    assert record["release_structure"]["observed_streams"] == 22


def test_a_participant_missing_one_task_closes_the_gate(
    tmp_path: Path,
) -> None:
    dataset_root = build_release(tmp_path, tasks=PADS_EXPECTED_TASKS[:-1])
    record, _ = _audit(dataset_root, tmp_path / "out")
    assert record["gate_status"] == GATE_NO_GO
    codes = {
        item["code"] for item in record["release_structure"]["failures"]
    }
    assert "MISSING_TASK" in codes


def test_a_duplicated_task_closes_the_gate(tmp_path: Path) -> None:
    dataset_root = build_release(
        tmp_path, tasks=(*PADS_EXPECTED_TASKS, "Relaxed")
    )
    record, _ = _audit(dataset_root, tmp_path / "out")
    codes = {
        item["code"] for item in record["release_structure"]["failures"]
    }
    assert "DUPLICATE_TASK" in codes
    assert record["gate_status"] == GATE_NO_GO


def test_an_unknown_extra_task_closes_the_gate(tmp_path: Path) -> None:
    dataset_root = build_release(
        tmp_path, tasks=(*PADS_EXPECTED_TASKS, "SurpriseTask")
    )
    record, _ = _audit(dataset_root, tmp_path / "out")
    codes = {
        item["code"] for item in record["release_structure"]["failures"]
    }
    assert "UNKNOWN_EXTRA_TASK" in codes
    assert record["gate_status"] == GATE_NO_GO


@pytest.mark.parametrize("dropped", ("LeftWrist", "RightWrist"))
def test_a_missing_wrist_record_closes_the_gate(
    tmp_path: Path, dropped: str
) -> None:
    kept = tuple(
        device for device in ("LeftWrist", "RightWrist") if device != dropped
    )
    dataset_root = build_release(tmp_path, devices=kept)
    record, _ = _audit(dataset_root, tmp_path / "out")
    codes = {
        item["code"] for item in record["release_structure"]["failures"]
    }
    assert "MISSING_DEVICE_RECORD" in codes
    assert record["gate_status"] == GATE_NO_GO


def test_a_missing_referenced_file_closes_the_gate(tmp_path: Path) -> None:
    dataset_root = build_release(
        tmp_path, drop_files=("timeseries/001_Relaxed_LeftWrist.txt",)
    )
    record, _ = _audit(dataset_root, tmp_path / "out")
    codes = {
        item["code"] for item in record["release_structure"]["failures"]
    }
    assert "MISSING_REFERENCED_FILE" in codes
    assert record["gate_status"] == GATE_NO_GO


def test_a_path_traversing_reference_closes_the_gate(tmp_path: Path) -> None:
    def escaping(participant: str, task: str, device: str) -> str:
        if task == "Relaxed" and device == "LeftWrist":
            return "../../escape.txt"
        return f"timeseries/{participant}_{task}_{device}.txt"

    dataset_root = build_release(tmp_path, file_name_for=escaping)
    record, _ = _audit(dataset_root, tmp_path / "out")
    assert record["release_status"] == "EVALUATED"
    assert record["gate_status"] == GATE_NO_GO
    assert record["source_failure_count"] >= 1


def test_a_mislabelled_reference_closes_the_gate(tmp_path: Path) -> None:
    def wrong(participant: str, task: str, device: str) -> str:
        if task == "Relaxed" and device == "LeftWrist":
            return "timeseries/001_Relaxed_RightWrist.txt"
        return f"timeseries/{participant}_{task}_{device}.txt"

    dataset_root = build_release(tmp_path, file_name_for=wrong)
    record, _ = _audit(dataset_root, tmp_path / "out")
    codes = {
        item["code"] for item in record["release_structure"]["failures"]
    }
    assert "PARTICIPANT_TASK_DEVICE_MISMATCH" in codes


def test_row_counts_come_from_metadata_not_a_task_allowlist(
    tmp_path: Path,
) -> None:
    # A 777-row declaration is honoured when the file matches it; a task-name
    # allowlist or a hardcoded 2048 would reject this release.
    dataset_root = build_release(tmp_path, rows_for={"Relaxed": 777})
    record, _ = _audit(dataset_root, tmp_path / "out")
    assert record["per_task"]["Relaxed"]["declared_rows"] == 777
    assert record["streams"]["refused"] == 0


def test_entrainment_uses_its_own_declared_two_thousand_forty_eight_rows(
    tmp_path: Path,
) -> None:
    dataset_root = build_release(tmp_path)
    record, _ = _audit(dataset_root, tmp_path / "out")
    per_task = record["per_task"]
    assert per_task["Entrainment"]["declared_rows"] == 2048
    assert per_task["Relaxed"]["declared_rows"] == 2048
    assert per_task["RelaxedTask"]["declared_rows"] == 2048
    for task in PADS_EXPECTED_TASKS:
        if task not in {"Relaxed", "RelaxedTask", "Entrainment"}:
            assert per_task[task]["declared_rows"] == 1024


def test_no_row_count_allowlist_exists_in_the_implementation() -> None:
    import inspect

    for module in (movement, audit_module):
        source = inspect.getsource(module)
        assert "2048" not in source
        assert "1024" not in source


def test_every_published_task_structure_is_exercised(tmp_path: Path) -> None:
    dataset_root = build_release(tmp_path)
    record, _ = _audit(dataset_root, tmp_path / "out")
    assert set(record["per_task"]) == set(PADS_EXPECTED_TASKS)
    assert record["release_structure"]["tasks_observed"] == sorted(
        PADS_EXPECTED_TASKS
    )


# --- audit and gate -------------------------------------------------------


def test_an_absent_release_blocks(tmp_path: Path) -> None:
    record, receipt = audit_pads_p01(
        dataset_root=tmp_path / "missing", output_root=tmp_path
    )
    assert record["release_status"] == "BLOCKED_INPUT_DATA_UNAVAILABLE"
    assert record["gate_evaluated"] is False
    assert "gate_status" not in record
    assert receipt is None
    assert exit_code_for(record) == EXIT_BLOCKED


def test_an_absent_checksum_list_closes_the_gate_rather_than_blocking(
    tmp_path: Path,
) -> None:
    dataset_root = build_release(tmp_path, write_checksums=False)
    record, _ = _audit(dataset_root, tmp_path / "out")
    assert record["release_status"] == "EVALUATED"
    assert PADS_ALL_SOURCE_FILES_HASH_VERIFIED in failing_conditions(record)
    assert record["gate_status"] == GATE_NO_GO


def test_a_hash_mismatch_closes_the_gate(tmp_path: Path) -> None:
    dataset_root = build_release(tmp_path)
    sums = dataset_root / "SHA256SUMS.txt"
    lines = sums.read_text().splitlines()
    # Corrupt one timeseries digest, leaving its path intact.
    for index, line in enumerate(lines):
        if "timeseries/" in line:
            digest, _, relative = line.partition(" ")
            replacement = ("0" if digest[0] != "0" else "1") + digest[1:]
            lines[index] = f"{replacement} {relative}"
            break
    sums.write_text("\n".join(lines) + "\n")
    record, _ = _audit(dataset_root, tmp_path / "out")
    assert PADS_ALL_SOURCE_FILES_HASH_VERIFIED in failing_conditions(record)


def test_the_gate_condition_set_is_frozen(tmp_path: Path) -> None:
    dataset_root = build_release(tmp_path)
    record, _ = _audit(dataset_root, tmp_path / "out")
    assert tuple(
        item["condition"] for item in record["gate_conditions"]
    ) == GATE_CONDITIONS
    assert PADS_RELEASE_STRUCTURE_RECONCILED in GATE_CONDITIONS
    assert PADS_INDEPENDENT_REPRODUCTION_VERIFIED in GATE_CONDITIONS


def test_one_execution_cannot_satisfy_the_reproduction_condition(
    tmp_path: Path,
) -> None:
    dataset_root = build_release(tmp_path)
    record, _ = _audit(dataset_root, tmp_path / "out")
    assert PADS_INDEPENDENT_REPRODUCTION_VERIFIED in failing_conditions(record)
    assert record["gate_status"] == GATE_NO_GO


def test_two_genuine_executions_agree_and_pass(tmp_path: Path) -> None:
    dataset_root = build_release(tmp_path)
    first, receipt_a = _audit(
        dataset_root, tmp_path / "a", run_id="r1", process_id=101
    )
    second, receipt_b = _audit(
        dataset_root, tmp_path / "b", run_id="r2", process_id=202,
        reproduction_receipt=receipt_a,
    )
    assert (
        first["canonical_evidence_sha256"]
        == second["canonical_evidence_sha256"]
    )
    assert receipt_a["output_root"] != receipt_b["output_root"]
    assert failing_conditions(second) == ()
    assert second["gate_status"] == GATE_PASS
    assert exit_code_for(second) == EXIT_PASS


def test_a_copied_receipt_is_not_a_second_execution(tmp_path: Path) -> None:
    dataset_root = build_release(tmp_path)
    _, receipt_a = _audit(
        dataset_root, tmp_path / "a", run_id="r1", process_id=101
    )
    copied = json.loads(json.dumps(receipt_a))
    second, _ = _audit(
        dataset_root, tmp_path / "b", run_id="r1", process_id=101,
        reproduction_receipt=copied,
    )
    assert PADS_INDEPENDENT_REPRODUCTION_VERIFIED in failing_conditions(second)


def test_no_window_spectrum_or_video_artifact_is_emitted(
    tmp_path: Path,
) -> None:
    dataset_root = build_release(tmp_path)
    record, _ = _audit(dataset_root, tmp_path / "out")
    assert record["withheld_artifacts"] == {
        "contiguous_window_tables": 0,
        "frame_imu_index_tables": 0,
        "spectral_feature_tables": 0,
        "storage_benchmark_result_tables": 0,
        "success_markers": 0,
    }
    assert record["materialized_release_artifacts"] == 0
    # The only mentions of video in the record are the declarations that there
    # is none, and the schema screen must confirm no field name carries one.
    assert record["authority"]["video_pairing"] == "NOT_APPLICABLE"
    assert record["authority"]["modality"] == "INERTIAL_ONLY_NO_VIDEO"
    assert PADS_NO_VIDEO_ASSOCIATION_EMITTED not in failing_conditions(record)
    for table, factory in PADS_TABLE_SCHEMAS.items():
        assert_no_paired_claim([table, *factory().names])


# --- CLI ------------------------------------------------------------------


def test_cli_blocks_with_exit_four(tmp_path: Path) -> None:
    output = tmp_path / "out"
    code = audit_module.main([
        "--dataset-root", str(tmp_path / "missing"),
        "--output-root", str(output),
    ])
    assert code == EXIT_BLOCKED
    record = json.loads((output / "pads_p01_evidence.json").read_text())
    assert record["release_status"] == "BLOCKED_INPUT_DATA_UNAVAILABLE"
    assert not (output / "pads_p01_run_receipt.json").exists()
    assert not (output / "_SUCCESS").exists()


def test_cli_writes_evidence_and_receipt_and_exits_three(
    tmp_path: Path,
) -> None:
    dataset_root = build_release(tmp_path)
    output = tmp_path / "out"
    code = audit_module.main([
        "--dataset-root", str(dataset_root),
        "--output-root", str(output),
    ])
    assert code == EXIT_NO_GO
    evidence = json.loads((output / "pads_p01_evidence.json").read_text())
    receipt = json.loads((output / "pads_p01_run_receipt.json").read_text())
    assert evidence["canonical_evidence_sha256"] == (
        receipt["canonical_evidence_sha256"]
    )
    assert not (output / "_SUCCESS").exists()


def test_cli_refuses_to_overwrite_an_existing_evidence_record(
    tmp_path: Path,
) -> None:
    dataset_root = build_release(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "pads_p01_evidence.json").write_text("{}")
    code = audit_module.main([
        "--dataset-root", str(dataset_root),
        "--output-root", str(output),
    ])
    assert code == 2
    assert (output / "pads_p01_evidence.json").read_text() == "{}"
