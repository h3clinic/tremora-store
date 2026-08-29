from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from motionbloom.tremora_store.pads.authority import PadsAuthorityError
from motionbloom.tremora_store.pads.p03.contract import (
    FREQUENCY_BIN_COUNT,
)
from motionbloom.tremora_store.pads.p03.contract import (
    SUCCESS_MARKER as P03_SUCCESS_MARKER,
)
from motionbloom.tremora_store.pads.p03.grid import frequency_values, grid_hash
from motionbloom.tremora_store.pads.p04 import contract, dependency, filters
from motionbloom.tremora_store.pads.p04.contract import (
    CORE_BIN_COUNT,
    CUTOFF_FRACTION,
    DERIVED_RATES_HZ,
    EDGE_BIN_COUNT,
    band_of,
    cutoff_hz,
)
from motionbloom.tremora_store.pads.p04.dependency import (
    FROZEN_DEPENDENCY,
    dependency_record,
    load_dependency,
    verify_dependency,
)
from motionbloom.tremora_store.pads.p04.rational_time import (
    RationalTimeError,
    assert_declared_exactness,
    grid_for,
)
from motionbloom.tremora_store.pads.p04.schemas import (
    P04_TABLE_FILES,
    P04_TABLE_SCHEMAS,
)
from motionbloom.tremora_store.release_gate import canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO_ROOT / "detector/benchmarks"
DEPENDENCY_FILE = BENCHMARKS / "pads_p03_dependency.json"
P03_REPORT = BENCHMARKS / "pads_p03_release_audit.json"


# --- rates and bands ------------------------------------------------------


def test_the_four_derived_rates_are_frozen() -> None:
    assert DERIVED_RATES_HZ == (100, 50, 30, 25)
    assert contract.NATIVE_RATE_LABEL == "NATIVE_IRREGULAR_SOURCE_TIME"
    assert contract.REFERENCE_MILESTONE == "PADS_P0_3"


def test_only_thirty_hertz_lacks_an_exact_picosecond_period() -> None:
    assert_declared_exactness()
    assert grid_for(30).exact_in_picoseconds is False
    assert grid_for(30).period_picoseconds == Fraction(100_000_000_000, 3)
    for rate, period in ((100, 10**10), (50, 2 * 10**10), (25, 4 * 10**10)):
        grid = grid_for(rate)
        assert grid.exact_in_picoseconds is True
        assert grid.period_picoseconds == period
        assert grid.sample_picoseconds_exact(7) == 7 * period


def test_thirty_hertz_time_stays_rational_and_reports_its_residual() -> None:
    grid = grid_for(30)
    assert grid.sample_seconds(7) == Fraction(7, 30)
    assert grid.sample_picoseconds(7) == Fraction(700_000_000_000, 3)
    assert grid.sample_picoseconds_exact(7) is None
    assert grid.sample_picoseconds_rounded(7) == 233_333_333_333
    assert grid.rounding_residual_picoseconds(7) == Fraction(1, 3)


def test_grid_ordinals_are_chosen_by_exact_comparison() -> None:
    grid = grid_for(30)
    # One second holds ordinals 0..30 at 30 Hz; ordinal 30 lands exactly on
    # the boundary and is included by exact rational comparison.
    assert list(grid.ordinals_covering(0, 10**12)) == list(range(31))
    assert grid.sample_picoseconds(30) == 10**12
    assert list(grid.ordinals_covering(10**12, 10**12)) == [30]
    with pytest.raises(RationalTimeError):
        grid.ordinals_covering(10, 0)
    with pytest.raises(RationalTimeError):
        grid_for(0)


def test_the_grid_is_anchored_at_task_local_zero() -> None:
    assert contract.GRID_ORIGIN == "TASK_LOCAL_ZERO"
    assert grid_for(100).sample_seconds(0) == 0
    assert contract.RESAMPLING_DOMAIN == "P02_CONTIGUOUS_SEGMENT"


def test_the_bands_partition_the_analysis_grid_at_ten_hertz() -> None:
    values = frequency_values()
    core = [value for value in values if band_of(value) == contract.CORE_BAND]
    edge = [value for value in values if band_of(value) == contract.EDGE_BAND]
    assert len(core) == CORE_BIN_COUNT == 29
    assert len(edge) == EDGE_BIN_COUNT == 8
    assert len(core) + len(edge) == FREQUENCY_BIN_COUNT == 37
    assert core[0] == 3.0
    assert core[-1] == 10.0
    assert edge[0] == 10.25
    assert edge[-1] == 12.0


# --- the frozen filter ----------------------------------------------------


def test_the_cutoff_is_exact_and_lands_on_the_band_split_at_25_hz() -> None:
    assert CUTOFF_FRACTION == Fraction(4, 5)
    assert cutoff_hz(25) == Fraction(10)
    assert cutoff_hz(25) == Fraction(str(contract.CORE_BAND_MAX_HZ))
    assert cutoff_hz(30) == Fraction(12)
    assert cutoff_hz(50) == Fraction(20)
    assert cutoff_hz(100) == Fraction(40)


def test_the_coefficient_table_is_frozen_and_hashed() -> None:
    table = filters.coefficient_table()
    assert table.shape == (contract.COEFFICIENT_TABLE_LENGTH,) == (1025,)
    assert table.dtype == np.float64
    assert table[0] == pytest.approx(1.0)
    assert filters.coefficients_sha256() == (
        "ad041db76fa87977f10fcc355ddfba9f4e1a556966afb049d585d0b8d9236f35"
    )
    # The table is the filter: the same bytes on every call.
    assert hashlib.sha256(
        np.ascontiguousarray(filters.coefficient_table()).tobytes()
    ).hexdigest() == filters.coefficients_sha256()


def test_the_kernel_is_used_by_lookup_not_interpolation() -> None:
    table = filters.coefficient_table()
    taps = contract.TAPS_PER_ZERO_CROSSING
    assert filters.kernel_weight(0.0) == table[0]
    assert filters.kernel_weight(1.0) == table[taps]
    assert filters.kernel_weight(-1.0) == table[taps]
    # Beyond the half width the kernel is exactly zero, not extrapolated.
    assert filters.kernel_weight(
        contract.HALF_WIDTH_ZERO_CROSSINGS + 0.5
    ) == 0.0
    assert np.array_equal(
        filters.kernel_weights(np.array([0.0, 1.0, -1.0, 99.0])),
        np.array([table[0], table[taps], table[taps], 0.0]),
    )


def test_the_kernel_support_shrinks_as_the_rate_falls() -> None:
    supports = [float(filters.support_seconds(r)) for r in DERIVED_RATES_HZ]
    assert supports == pytest.approx([0.1, 0.2, 1 / 3, 0.4])
    # Lower rate, lower cutoff, wider kernel in time.
    assert supports == sorted(supports)


@pytest.mark.parametrize("rate", DERIVED_RATES_HZ)
def test_the_filter_passes_dc_and_stops_above_its_cutoff(rate: int) -> None:
    cutoff = float(cutoff_hz(rate))
    response = filters.frequency_response(
        np.array([0.0, cutoff, 1.5 * cutoff, 2.0 * cutoff]), rate
    )
    assert response[0] == pytest.approx(1.0, abs=1e-9)
    # A windowed sinc is 6 dB down at its own cutoff, by construction.
    assert response[1] == pytest.approx(0.5, rel=0.02)
    assert 20 * np.log10(response[2]) < -80.0
    assert 20 * np.log10(response[3]) < -100.0


def test_the_band_edge_response_is_published_as_measured_evidence() -> None:
    response = filters.declared_band_response()
    assert set(response) == {str(rate) for rate in DERIVED_RATES_HZ}
    # 100 and 50 Hz carry the whole analysis band unattenuated.
    for rate in ("100", "50"):
        assert response[rate]["edge_max_hz"] == pytest.approx(0.0, abs=0.01)
    # At 30 Hz the top of the grid sits at the cutoff.
    assert response["30"]["edge_max_hz"] == pytest.approx(-6.02, abs=0.05)
    # At 25 Hz the cutoff is the core/edge split itself, so the topmost core
    # bin is already 6 dB down and the edge band is deep in the transition.
    assert response["25"]["core_max_hz"] == pytest.approx(-6.02, abs=0.05)
    assert response["25"]["edge_max_hz"] == pytest.approx(-28.36, abs=0.1)
    assert response["25"]["core_min_hz"] == pytest.approx(0.0, abs=0.01)


def test_the_manifest_publishes_the_whole_filter_definition() -> None:
    manifest = filters.anti_alias_manifest()
    assert manifest["kernel"] == "KAISER_WINDOWED_SINC"
    assert manifest["kaiser_beta"] == 8.6
    assert manifest["half_width_zero_crossings"] == 8
    assert manifest["taps_per_zero_crossing"] == 128
    assert manifest["coefficient_table_length"] == 1025
    assert manifest["weight_normalization"] == "UNIT_SUM_PER_OUTPUT_SAMPLE"
    assert manifest["coefficients_sha256"] == filters.coefficients_sha256()
    assert manifest["cutoff_hz"] == {
        "100": 40.0, "50": 20.0, "30": 12.0, "25": 10.0,
    }


# --- screens and claim boundary ------------------------------------------


@pytest.mark.parametrize(
    "name",
    ("classification_label", "diagnosis_group", "severity_score",
     "retrieval_latency_ms", "read_throughput", "hdf5_baseline"),
)
def test_clinical_and_benchmark_vocabulary_is_refused(name: str) -> None:
    with pytest.raises(contract.PadsP04ContractError):
        contract.assert_p04_names(["window_id", name])


@pytest.mark.parametrize("name", ("video_uid_ref", "camera_id", "frame_index"))
def test_the_video_screen_still_applies(name: str) -> None:
    with pytest.raises(PadsAuthorityError):
        contract.assert_p04_names(["window_id", name])


@pytest.mark.parametrize(
    "name",
    ("resampled_power", "anti_alias_coefficients_sha256", "derived_rate_hz",
     "aggregate_power"),
)
def test_resampling_vocabulary_is_allowed_here(name: str) -> None:
    # P0.3 forbade these; P0.4 is the milestone that legitimately uses them.
    contract.assert_p04_names(["window_id", name])


def test_the_milestone_boundary_is_published() -> None:
    assert set(contract.WITHHELD_P04_ARTIFACTS.values()) == {0}
    for name in (
        "classification_tables", "diagnosis_tables", "severity_tables",
        "tremor_detection_tables", "video_association_tables",
        "storage_benchmark_tables", "retrieval_latency_tables",
        "generic_success_markers",
    ):
        assert name in contract.WITHHELD_P04_ARTIFACTS


def test_the_success_marker_is_specific_to_this_milestone() -> None:
    assert contract.SUCCESS_MARKER == "_PADS_P04_RATE_ABLATION_SUCCESS"
    assert contract.SUCCESS_MARKER != contract.GENERIC_SUCCESS_MARKER


def test_the_authority_keeps_the_unimodal_boundary() -> None:
    block = contract.authority_block()
    assert block["timing_authority"] == "SOURCE_RELATIVE_UNIMODAL_CLOCK"
    assert block["video_pairing"] == "NOT_APPLICABLE"
    assert block["hardware_sync_claim"] is False
    assert block["cross_wrist_clock_alignment"] == "UNRESOLVED"
    assert block["sample_level_bilateral_fusion_allowed"] is False
    assert block["reference"] == "NATIVE_IRREGULAR_SOURCE_TIME"
    assert block["resampling_domain"] == "P02_CONTIGUOUS_SEGMENT"


# --- schemas --------------------------------------------------------------


def test_every_schema_passes_both_screens() -> None:
    for name, factory in P04_TABLE_SCHEMAS.items():
        contract.assert_p04_names([name, *factory().names])
    assert set(P04_TABLE_FILES) == set(P04_TABLE_SCHEMAS)


def test_grid_timing_is_carried_as_exact_rationals() -> None:
    grids = P04_TABLE_SCHEMAS["pads_p04_rate_grids"]()
    for name in (
        "rate_hz_num", "rate_hz_den",
        "period_picoseconds_num", "period_picoseconds_den",
    ):
        assert grids.field(name).type == pa.int64()
    assert grids.field("exact_in_picoseconds").type == pa.bool_()
    assert grids.metadata[b"tremora.grid_timing"] == b"EXACT_RATIONAL"


def test_power_vectors_are_fixed_at_the_analysis_grid_width() -> None:
    spectra = P04_TABLE_SCHEMAS["pads_p04_rate_spectra"]()
    for name in ("aggregate_power", "normalized_aggregate_power"):
        assert spectra.field(name).type.list_size == FREQUENCY_BIN_COUNT


def test_the_spectra_table_carries_its_native_reference(
) -> None:
    spectra = P04_TABLE_SCHEMAS["pads_p04_rate_spectra"]()
    for name in (
        "native_dominant_frequency_hz", "native_core_band_power",
        "native_edge_band_power", "native_spectral_content_sha256",
    ):
        assert name in spectra.names


def test_summaries_are_participant_level_and_per_band() -> None:
    summary = P04_TABLE_SCHEMAS["pads_p04_participant_summary"]()
    for name in (
        "participant_id", "rate_hz", "sensor_family", "band", "windows",
    ):
        assert name in summary.names
    assert "window_id" not in summary.names


# --- dependency -----------------------------------------------------------


def test_the_dependency_file_is_generated_from_the_frozen_code_pin() -> None:
    assert DEPENDENCY_FILE.read_bytes() == canonical_json_bytes(
        dependency_record()
    )
    assert load_dependency(DEPENDENCY_FILE) == FROZEN_DEPENDENCY


def test_the_pin_names_the_published_p03_evidence() -> None:
    published = json.loads(P03_REPORT.read_bytes())
    assert FROZEN_DEPENDENCY.p03_evidence_sha256 == (
        published["canonical_evidence_sha256"]
    )
    assert FROZEN_DEPENDENCY.p03_report_sha256 == hashlib.sha256(
        P03_REPORT.read_bytes()
    ).hexdigest()
    assert FROZEN_DEPENDENCY.p03_gate_status == published["gate_status"]
    assert FROZEN_DEPENDENCY.p03_spectral_table_sha256 == (
        published["materialization"]["spectral_table_content_sha256"]
    )
    assert FROZEN_DEPENDENCY.frequency_grid_sha256 == grid_hash()
    assert FROZEN_DEPENDENCY.anti_alias_coefficients_sha256 == (
        filters.coefficients_sha256()
    )


def test_the_pin_carries_the_whole_p02_chain() -> None:
    published = json.loads(P03_REPORT.read_bytes())
    inherited = published["p02_dependency"]["pinned"]
    assert FROZEN_DEPENDENCY.p02_evidence_sha256 == (
        inherited["p02_evidence_sha256"]
    )
    assert FROZEN_DEPENDENCY.storage_index_content_sha256 == (
        inherited["storage_index_content_sha256"]
    )
    assert FROZEN_DEPENDENCY.source_manifest_sha256 == (
        inherited["source_manifest_sha256"]
    )


def _p03_root(tmp_path: Path, *, content: str, marker: bool = True) -> Path:
    root = tmp_path / "p03"
    root.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table({
            "spectral_record_id": ["w#GYROSCOPE"],
            "spectral_content_sha256": [content],
        }),
        root / "pads_p03_spectra.parquet",
    )
    if marker:
        (root / P03_SUCCESS_MARKER).write_bytes(b"")
    return root


def _store(tmp_path: Path) -> Path:
    import pyarrow.parquet as pq
    from motionbloom.tremora_store.pads.p02.contract import (
        SUCCESS_MARKER as P02_SUCCESS_MARKER,
    )

    root = tmp_path / "store"
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"stream_id": ["s"], "row_group_content_sha256": ["0" * 64]}),
        root / "pads_stream_storage_index.parquet",
    )
    (root / P02_SUCCESS_MARKER).write_bytes(b"")
    return root


@pytest.mark.parametrize(
    ("key", "expected"),
    (
        ("dependency_path", "P03_DEPENDENCY_FILE_ABSENT"),
        ("p03_report_path", "P03_REPORT_ABSENT"),
        ("p03_root", "P03_ROOT_ABSENT"),
        ("store_root", "P02_STORE_ROOT_ABSENT"),
    ),
)
def test_an_absent_authority_blocks(
    tmp_path: Path, key: str, expected: str
) -> None:
    arguments = {
        "dependency_path": DEPENDENCY_FILE,
        "p03_report_path": P03_REPORT,
        "p03_root": _p03_root(
            tmp_path, content=FROZEN_DEPENDENCY.p03_spectral_table_sha256
        ),
        "store_root": _store(tmp_path),
    }
    arguments[key] = tmp_path / "absent"
    result = verify_dependency(**arguments)
    assert result.status == expected
    assert result.blocks is True


def test_a_substituted_spectra_table_is_refused(tmp_path: Path) -> None:
    # Same row count, different content: the substitution a count check misses.
    result = verify_dependency(
        dependency_path=DEPENDENCY_FILE,
        p03_report_path=P03_REPORT,
        p03_root=_p03_root(tmp_path, content="f" * 64),
        store_root=_store(tmp_path),
    )
    assert result.status == "SPECTRAL_TABLE_HASH_MISMATCH"
    assert result.blocks is False
    assert result.verified is False


def test_p03_outputs_without_their_marker_are_refused(
    tmp_path: Path,
) -> None:
    result = verify_dependency(
        dependency_path=DEPENDENCY_FILE,
        p03_report_path=P03_REPORT,
        p03_root=_p03_root(tmp_path, content="0" * 64, marker=False),
        store_root=_store(tmp_path),
    )
    assert result.status == "P03_SUCCESS_MARKER_ABSENT"
    assert result.blocks is False


def test_a_tampered_p03_report_closes_the_gate(tmp_path: Path) -> None:
    tampered = tmp_path / "report.json"
    report = json.loads(P03_REPORT.read_bytes())
    report["gate_status"] = "NO_GO_PADS_SPECTRAL_PRESERVATION"
    tampered.write_bytes(canonical_json_bytes(report))
    result = verify_dependency(
        dependency_path=DEPENDENCY_FILE,
        p03_report_path=tampered,
        p03_root=_p03_root(tmp_path, content="0" * 64),
        store_root=_store(tmp_path),
    )
    assert result.status == "P03_REPORT_HASH_MISMATCH"
    assert result.blocks is False


def test_dependency_absence_statuses_are_named() -> None:
    assert dependency.ABSENCE_STATUSES == frozenset({
        "P03_DEPENDENCY_FILE_ABSENT",
        "P03_REPORT_ABSENT",
        "P03_ROOT_ABSENT",
        "P02_STORE_ROOT_ABSENT",
    })
    assert dependency.DEPENDENCY_VERIFIED == "P03_DEPENDENCY_VERIFIED"
