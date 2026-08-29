from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from motionbloom.tremora_store.pads.authority import PadsAuthorityError
from motionbloom.tremora_store.pads.p02 import contract, dependency, exact_time
from motionbloom.tremora_store.pads.p02.dependency import (
    DEPENDENCY_VERIFIED,
    FROZEN_DEPENDENCY,
    dependency_record,
    load_dependency,
    verify_dependency,
)
from motionbloom.tremora_store.pads.p02.exact_time import (
    ExactTimeError,
    exact_picoseconds,
    format_sensor_value,
    picoseconds_to_seconds_token,
    sensor_value_round_trips,
)
from motionbloom.tremora_store.pads.p02.schemas import (
    P02_INDEX_FILES,
    P02_TABLE_SCHEMAS,
)
from motionbloom.tremora_store.release_gate import canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO_ROOT / "detector/benchmarks"
DEPENDENCY_FILE = BENCHMARKS / "pads_p01_dependency.json"
P01_REPORT = BENCHMARKS / "pads_p01_release_audit.json"


# --- contract -------------------------------------------------------------


def test_the_milestone_boundary_is_published_not_implied() -> None:
    assert set(contract.WITHHELD_P02_ARTIFACTS.values()) == {0}
    for name in (
        "spectral_feature_tables",
        "tremor_frequency_tables",
        "band_power_tables",
        "resampled_signal_tables",
        "anti_alias_filter_outputs",
        "classification_tables",
        "video_association_tables",
        "comparative_benchmark_tables",
        "generic_success_markers",
    ):
        assert name in contract.WITHHELD_P02_ARTIFACTS


@pytest.mark.parametrize(
    "name",
    ("psd_estimate", "spectrum_ref", "band_power", "resampled_50hz",
     "anti_alias_filter", "tremor_frequency_hz", "welch_window", "fft_bins"),
)
def test_analysis_names_are_screened_by_substring(name: str) -> None:
    with pytest.raises(contract.PadsP02ContractError):
        contract.assert_p02_names(["stream_id", name])


@pytest.mark.parametrize("name", ("video_uid_ref", "camera_id", "frame_index"))
def test_the_inherited_video_screen_still_applies(name: str) -> None:
    with pytest.raises(PadsAuthorityError):
        contract.assert_p02_names(["stream_id", name])


def test_the_window_policy_is_exact_in_picoseconds() -> None:
    assert contract.WINDOW_DURATION_PS == 4_000_000_000_000
    assert contract.WINDOW_STRIDE_PS == 2_000_000_000_000
    assert contract.PICOSECONDS_PER_SECOND == 10**12
    assert contract.WINDOW_MEMBERSHIP == "SOURCE_TIME_HALF_OPEN_INTERVAL"


def test_the_gap_policy_is_cadence_relative_with_an_absolute_cap() -> None:
    assert contract.GAP_MULTIPLIER == 3
    assert contract.GAP_ABSOLUTE_CAP_PS == 100 * 10**9
    # At the release's ~9.99 ms median interval the cadence term binds, not
    # the cap: about 30 ms.
    median_ps = 9_994_506_900
    threshold = min(
        contract.GAP_ABSOLUTE_CAP_PS, contract.GAP_MULTIPLIER * median_ps
    )
    assert threshold == 3 * median_ps
    assert 29e9 < threshold < 31e9


def test_the_success_marker_is_specific_to_this_milestone() -> None:
    assert contract.SUCCESS_MARKER == "_PADS_P02_INDEX_SUCCESS"
    assert contract.SUCCESS_MARKER != contract.GENERIC_SUCCESS_MARKER


def test_the_authority_block_refuses_sample_level_fusion() -> None:
    block = contract.authority_block()
    assert block["timing_authority"] == "SOURCE_RELATIVE_UNIMODAL_CLOCK"
    assert block["time_basis"] == "SOURCE_TIME_COLUMN"
    assert block["video_pairing"] == "NOT_APPLICABLE"
    assert block["hardware_sync_claim"] is False
    assert block["bilateral_pairing_authority"] == "SOURCE_PROTOCOL_PAIR"
    assert block["cross_wrist_clock_alignment"] == "UNRESOLVED"
    assert block["sample_level_bilateral_fusion_allowed"] is False


# --- exact time -----------------------------------------------------------


def test_the_release_resolution_is_finer_than_a_nanosecond() -> None:
    token = "0.0099029541"
    assert exact_picoseconds(token) == 9_902_954_100
    # The same token in nanoseconds is 9,902,954.1 -- not an integer, so a
    # nanosecond store would have rounded a digit the source wrote.
    assert (Decimal(token) * 10**9) % 1 != 0


def test_a_token_finer_than_the_declared_scale_is_refused() -> None:
    with pytest.raises(ExactTimeError) as caught:
        exact_picoseconds("0.00990295410001")
    assert caught.value.code == "TIME_TOKEN_NOT_EXACT_AT_SCALE"


@pytest.mark.parametrize("token", ("", "NaN", "1_000", "abc", " 0.5"))
def test_an_unparseable_time_token_is_refused(token: str) -> None:
    with pytest.raises(ExactTimeError) as caught:
        exact_picoseconds(token)
    assert caught.value.code == "TIME_TOKEN_UNPARSEABLE"


@pytest.mark.parametrize(
    "token",
    ("0.0000000000", "20.4590511322", "-0.0000017271", "0.0099029541"),
)
def test_picoseconds_round_trip_to_the_source_width(token: str) -> None:
    assert picoseconds_to_seconds_token(exact_picoseconds(token)) == token


@pytest.mark.parametrize(
    "token",
    ("-0.0039583882", "0.0023599046", "-0.0000017271", "9.8000000000"),
)
def test_sensor_values_rebuild_their_source_token(token: str) -> None:
    assert sensor_value_round_trips(token, float(token))
    assert format_sensor_value(float(token)) == token


def test_a_value_that_does_not_rebuild_its_token_is_detected() -> None:
    # Three decimals is not the format the release wrote.
    assert not sensor_value_round_trips("0.001", 0.001)
    with pytest.raises(ExactTimeError):
        exact_time.assert_sensor_round_trip("0.001", 0.001)


# --- schemas --------------------------------------------------------------


def test_every_schema_passes_both_screens() -> None:
    for name, factory in P02_TABLE_SCHEMAS.items():
        contract.assert_p02_names([name, *factory().names])


def test_source_derived_times_are_picoseconds_everywhere() -> None:
    for factory in P02_TABLE_SCHEMAS.values():
        schema = factory()
        for field in schema:
            if field.name.endswith("_ns"):
                raise AssertionError(f"{field.name} would round the source")
            if field.name.endswith("_ps"):
                assert field.type == __import__("pyarrow").int64()
        assert schema.metadata[b"tremora.time_unit"] == b"PICOSECOND"


def test_the_sample_store_is_the_only_unindexed_table() -> None:
    assert "pads_samples" not in P02_INDEX_FILES
    assert set(P02_INDEX_FILES) | {"pads_samples"} == set(P02_TABLE_SCHEMAS)


# --- dependency -----------------------------------------------------------


def test_the_dependency_file_is_generated_from_the_frozen_code_pin() -> None:
    assert DEPENDENCY_FILE.read_bytes() == canonical_json_bytes(
        dependency_record()
    )
    assert load_dependency(DEPENDENCY_FILE) == FROZEN_DEPENDENCY


def test_the_pin_names_the_p01_pass_and_its_counts() -> None:
    assert FROZEN_DEPENDENCY.p01_gate_status == (
        "PASS_SOURCE_RELATIVE_UNIMODAL_CLOCK"
    )
    assert FROZEN_DEPENDENCY.expected_participants == 469
    assert FROZEN_DEPENDENCY.expected_assessments == 5159
    assert FROZEN_DEPENDENCY.expected_streams == 10318
    assert FROZEN_DEPENDENCY.expected_samples == 13_447_168


def test_the_pinned_report_is_the_published_one(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    report = json.loads(P01_REPORT.read_bytes())
    (release / "SHA256SUMS.txt").write_bytes(b"")
    result = verify_dependency(
        dependency_path=DEPENDENCY_FILE,
        report_path=P01_REPORT,
        release_root=release,
    )
    # Everything up to the release manifest reconciles; the synthetic release
    # root is what stops it here.
    assert result.status == "SOURCE_MANIFEST_MISMATCH"
    assert result.observed_report_sha256 == FROZEN_DEPENDENCY.p01_report_sha256
    assert result.observed_evidence_sha256 == report[
        "canonical_evidence_sha256"
    ]
    assert not result.blocks


def _release_with_manifest(tmp_path: Path) -> Path:
    """A release root whose checksum list hashes to the pinned manifest."""

    source = Path("/Users/aharshi/Developer/tremora-data/pads/release")
    del source  # the fixtures below never touch the real release
    release = tmp_path / "release"
    release.mkdir()
    return release


@pytest.mark.parametrize(
    ("missing", "expected"),
    (
        ("dependency", "P01_DEPENDENCY_FILE_ABSENT"),
        ("report", "P01_REPORT_ABSENT"),
    ),
)
def test_an_absent_dependency_or_report_blocks(
    tmp_path: Path, missing: str, expected: str
) -> None:
    result = verify_dependency(
        dependency_path=(
            tmp_path / "absent.json" if missing == "dependency"
            else DEPENDENCY_FILE
        ),
        report_path=(
            tmp_path / "absent_report.json" if missing == "report"
            else P01_REPORT
        ),
        release_root=_release_with_manifest(tmp_path),
    )
    assert result.status == expected
    assert result.blocks is True


def test_an_absent_release_root_blocks(tmp_path: Path) -> None:
    result = verify_dependency(
        dependency_path=DEPENDENCY_FILE,
        report_path=P01_REPORT,
        release_root=tmp_path / "absent_release",
    )
    assert result.status == "RELEASE_ROOT_ABSENT"
    assert result.blocks is True


def test_a_changed_report_does_not_block_it_closes_the_gate(
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "report.json"
    report = json.loads(P01_REPORT.read_bytes())
    report["gate_status"] = "NO_GO_PADS_UNIMODAL_INGEST"
    tampered.write_bytes(canonical_json_bytes(report))
    result = verify_dependency(
        dependency_path=DEPENDENCY_FILE,
        report_path=tampered,
        release_root=_release_with_manifest(tmp_path),
    )
    assert result.status == "P01_REPORT_HASH_MISMATCH"
    assert result.blocks is False
    assert result.verified is False


def test_a_p01_no_go_can_never_authorize_p02(tmp_path: Path) -> None:
    report = json.loads(P01_REPORT.read_bytes())
    report["gate_status"] = "NO_GO_PADS_UNIMODAL_INGEST"
    tampered = tmp_path / "report.json"
    tampered.write_bytes(canonical_json_bytes(report))
    pin = json.loads(DEPENDENCY_FILE.read_bytes())
    import hashlib
    pin["pinned"]["p01_report_sha256"] = hashlib.sha256(
        tampered.read_bytes()
    ).hexdigest()
    pinned_path = tmp_path / "dependency.json"
    pinned_path.write_bytes(canonical_json_bytes(pin))
    result = verify_dependency(
        dependency_path=pinned_path,
        report_path=tampered,
        release_root=_release_with_manifest(tmp_path),
    )
    assert result.status == "P01_GATE_NOT_PASS"
    assert result.blocks is False


def test_a_malformed_dependency_file_does_not_block(tmp_path: Path) -> None:
    broken = tmp_path / "dependency.json"
    broken.write_text("{}")
    result = verify_dependency(
        dependency_path=broken,
        report_path=P01_REPORT,
        release_root=_release_with_manifest(tmp_path),
    )
    assert result.status == "P01_DEPENDENCY_FILE_MALFORMED"
    assert result.blocks is False


def test_dependency_statuses_are_all_named() -> None:
    assert DEPENDENCY_VERIFIED == "P01_AUTHORITY_DEPENDENCY_VERIFIED"
    assert dependency.ABSENCE_STATUSES == frozenset({
        "P01_DEPENDENCY_FILE_ABSENT",
        "P01_REPORT_ABSENT",
        "RELEASE_ROOT_ABSENT",
    })
