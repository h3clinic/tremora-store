from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from motionbloom.tremora_store.pads.authority import PadsAuthorityError
from motionbloom.tremora_store.pads.p02.contract import (
    SUCCESS_MARKER as P02_SUCCESS_MARKER,
)
from motionbloom.tremora_store.pads.p03 import contract, dependency, grid
from motionbloom.tremora_store.pads.p03.dependency import (
    FROZEN_DEPENDENCY,
    dependency_record,
    load_dependency,
    observed_storage_index_hash,
    verify_dependency,
)
from motionbloom.tremora_store.pads.p03.schemas import (
    FREQUENCY_GRID_FILENAME,
    P03_TABLE_FILES,
    P03_TABLE_SCHEMAS,
)
from motionbloom.tremora_store.release_gate import canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO_ROOT / "detector/benchmarks"
DEPENDENCY_FILE = BENCHMARKS / "pads_p02_dependency.json"
P02_REPORT = BENCHMARKS / "pads_p02_release_audit.json"


# --- frequency grid -------------------------------------------------------


def test_the_grid_is_thirty_seven_quarter_hertz_bins() -> None:
    values = grid.frequency_values()
    assert len(values) == contract.FREQUENCY_BIN_COUNT == 37
    assert values[0] == 3.0
    assert values[-1] == 12.0
    # Generated from integer millihertz, so every step is exact rather than
    # an accumulated stride.
    for index, value in enumerate(values):
        assert value == 3.0 + index * 0.25


def test_the_grid_matches_the_window_rayleigh_resolution() -> None:
    assert contract.RAYLEIGH_RESOLUTION_HZ == 1.0 / contract.WINDOW_DURATION_S
    assert contract.FREQUENCY_STEP_HZ == contract.RAYLEIGH_RESOLUTION_HZ
    record = grid.grid_record()
    assert record["rayleigh_resolution_hz"] == 0.25
    assert record["window_duration_ps"] == 4_000_000_000_000


def test_the_grid_hash_is_stable_and_identifies_the_grid() -> None:
    assert grid.grid_hash() == grid.grid_hash()
    assert grid.grid_id().startswith("tremor-band-3to12hz-0p25-")
    assert grid.grid_record()["grid_id"] == grid.grid_id()


def test_nyquist_comes_from_the_stream_cadence() -> None:
    # The corpus median interval, not the declared 100 Hz.
    assert grid.nyquist_hz(10_005_950_900) == pytest.approx(49.97, abs=0.01)
    assert grid.grid_within_nyquist(10_005_950_900)
    # A stream sampled just above the band cannot carry the grid.
    assert not grid.grid_within_nyquist(50_000_000_000)
    with pytest.raises(ValueError):
        grid.nyquist_hz(0)


# --- contract -------------------------------------------------------------


def test_the_milestone_boundary_is_published() -> None:
    assert set(contract.WITHHELD_P03_ARTIFACTS.values()) == {0}
    for name in (
        "resampled_signal_tables", "anti_alias_filter_outputs",
        "derived_rate_tables", "classification_tables",
        "tremor_detection_tables", "bilateral_fusion_tables",
        "video_association_tables", "comparative_benchmark_tables",
        "generic_success_markers",
    ):
        assert name in contract.WITHHELD_P03_ARTIFACTS


@pytest.mark.parametrize(
    "name",
    ("resampled_50hz", "anti_alias_filter", "downsample_ratio",
     "decimation_factor", "classification_label", "severity_score",
     "diagnosis_group"),
)
def test_next_milestone_vocabulary_is_refused(name: str) -> None:
    with pytest.raises(contract.PadsP03ContractError):
        contract.assert_p03_names(["window_id", name])


@pytest.mark.parametrize("name", ("video_uid_ref", "camera_id", "frame_index"))
def test_the_video_screen_still_applies(name: str) -> None:
    with pytest.raises(PadsAuthorityError):
        contract.assert_p03_names(["window_id", name])


@pytest.mark.parametrize(
    "name", ("spectral_content_sha256", "aggregate_power", "band_power")
)
def test_spectral_vocabulary_is_allowed_here(name: str) -> None:
    # P0.2 forbade these; P0.3 is the milestone that legitimately uses them.
    contract.assert_p03_names(["window_id", name])


def test_gyroscope_is_the_primary_family_and_magnitude_is_refused() -> None:
    assert contract.SENSOR_FAMILIES[0] == "GYROSCOPE"
    assert contract.VECTOR_MAGNITUDE_ALLOWED is False
    block = contract.authority_block()
    assert block["spectral_input"] == "RAW_AXES_PER_SENSOR_FAMILY"
    assert block["vector_magnitude_primary_signal"] is False
    assert block["transform"] == "NONUNIFORM_DISCRETE_FOURIER_TRANSFORM"
    assert block["numeric_dtype"] == "float64"


def test_the_authority_inherits_the_unimodal_boundary() -> None:
    block = contract.authority_block()
    assert block["timing_authority"] == "SOURCE_RELATIVE_UNIMODAL_CLOCK"
    assert block["time_basis"] == "SOURCE_TIME_COLUMN"
    assert block["video_pairing"] == "NOT_APPLICABLE"
    assert block["hardware_sync_claim"] is False
    assert block["cross_wrist_clock_alignment"] == "UNRESOLVED"
    assert block["sample_level_bilateral_fusion_allowed"] is False


def test_the_success_marker_is_specific_to_this_milestone() -> None:
    assert contract.SUCCESS_MARKER == "_PADS_P03_SPECTRAL_SUCCESS"
    assert contract.SUCCESS_MARKER != contract.GENERIC_SUCCESS_MARKER


# --- schemas --------------------------------------------------------------


def test_power_vectors_are_fixed_at_the_grid_width() -> None:
    spectra = P03_TABLE_SCHEMAS["pads_p03_spectra"]()
    for name in (
        "axis_x_power", "axis_y_power", "axis_z_power",
        "aggregate_power", "normalized_aggregate_power",
    ):
        field = spectra.field(name)
        assert field.type.list_size == contract.FREQUENCY_BIN_COUNT
        assert field.type.value_type == pa.float64()


def test_every_schema_passes_both_screens() -> None:
    for name, factory in P03_TABLE_SCHEMAS.items():
        contract.assert_p03_names([name, *factory().names])
    assert set(P03_TABLE_FILES) == set(P03_TABLE_SCHEMAS)
    assert FREQUENCY_GRID_FILENAME == "pads_p03_frequency_grid.json"


# --- dependency -----------------------------------------------------------


def test_the_dependency_file_is_generated_from_the_frozen_code_pin() -> None:
    assert DEPENDENCY_FILE.read_bytes() == canonical_json_bytes(
        dependency_record()
    )
    assert load_dependency(DEPENDENCY_FILE) == FROZEN_DEPENDENCY


def test_the_pin_names_the_published_p02_evidence_and_report() -> None:
    published = json.loads(P02_REPORT.read_bytes())
    assert FROZEN_DEPENDENCY.p02_evidence_sha256 == (
        published["canonical_evidence_sha256"]
    )
    assert FROZEN_DEPENDENCY.p02_report_sha256 == hashlib.sha256(
        P02_REPORT.read_bytes()
    ).hexdigest()
    assert FROZEN_DEPENDENCY.p02_gate_status == published["gate_status"]
    assert FROZEN_DEPENDENCY.storage_index_content_sha256 == (
        published["replay_verification"]["storage_index_content_sha256"]
    )
    assert FROZEN_DEPENDENCY.expected_windows == (
        published["materialization"]["windows"]
    )
    assert FROZEN_DEPENDENCY.expected_samples == (
        published["materialization"]["samples_materialized"]
    )


def _store(tmp_path: Path, *, content_hashes: dict[str, str],
           marker: bool = True) -> Path:
    root = tmp_path / "store"
    root.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "stream_id": list(content_hashes),
        "row_group_content_sha256": list(content_hashes.values()),
    })
    pq.write_table(table, root / "pads_stream_storage_index.parquet")
    if marker:
        (root / P02_SUCCESS_MARKER).write_bytes(b"")
    return root


def test_an_absent_dependency_or_report_or_store_blocks(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, content_hashes={"a": "0" * 64})
    absent = tmp_path / "absent.json"
    for kwargs, expected in (
        ({"dependency_path": absent}, "P02_DEPENDENCY_FILE_ABSENT"),
        ({"p02_report_path": absent}, "P02_REPORT_ABSENT"),
        ({"store_root": tmp_path / "no_store"}, "P02_STORE_ROOT_ABSENT"),
    ):
        result = verify_dependency(**{
            "dependency_path": DEPENDENCY_FILE,
            "p02_report_path": P02_REPORT,
            "store_root": store,
            **kwargs,
        })
        assert result.status == expected
        assert result.blocks is True


def test_a_store_with_a_different_content_hash_is_refused(
    tmp_path: Path,
) -> None:
    # Same shape, same row count, different content: exactly the substitution
    # a row-count check would miss.
    store = _store(tmp_path, content_hashes={"001:Relaxed:LeftWrist": "f" * 64})
    result = verify_dependency(
        dependency_path=DEPENDENCY_FILE,
        p02_report_path=P02_REPORT,
        store_root=store,
    )
    assert result.status == "STORAGE_INDEX_HASH_MISMATCH"
    assert result.blocks is False
    assert result.verified is False
    assert result.observed_storage_index_sha256 == (
        observed_storage_index_hash(store)
    )


def test_a_store_without_its_success_marker_is_refused(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, content_hashes={"a": "0" * 64}, marker=False)
    result = verify_dependency(
        dependency_path=DEPENDENCY_FILE,
        p02_report_path=P02_REPORT,
        store_root=store,
    )
    assert result.status == "P02_STORE_SUCCESS_MARKER_ABSENT"
    assert result.blocks is False


def test_a_tampered_p02_report_does_not_block_it_closes_the_gate(
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "report.json"
    report = json.loads(P02_REPORT.read_bytes())
    report["gate_status"] = "NO_GO_PADS_INDEX_AND_WINDOW_MATERIALIZATION"
    tampered.write_bytes(canonical_json_bytes(report))
    result = verify_dependency(
        dependency_path=DEPENDENCY_FILE,
        p02_report_path=tampered,
        store_root=_store(tmp_path, content_hashes={"a": "0" * 64}),
    )
    assert result.status == "P02_REPORT_HASH_MISMATCH"
    assert result.blocks is False


def test_dependency_absence_statuses_are_named() -> None:
    assert dependency.ABSENCE_STATUSES == frozenset({
        "P02_DEPENDENCY_FILE_ABSENT",
        "P02_REPORT_ABSENT",
        "P02_STORE_ROOT_ABSENT",
    })
    assert dependency.DEPENDENCY_VERIFIED == "P02_1_DEPENDENCY_VERIFIED"
