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
    DERIVED_RATES_HZ,
    EDGE_BIN_COUNT,
    band_of,
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
    parent_grid,
    parent_span_for_output,
    supported_output_ordinals,
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


# --- the frozen filters ---------------------------------------------------


def test_the_filters_are_specified_by_what_each_rate_preserves() -> None:
    # No universal cutoff fraction: a shared 4/5 attenuated 12 Hz by 6 dB at
    # 30 Hz, which would have made "30 Hz loses 12 Hz" partly an artefact of
    # the filter rather than of the rate.
    assert not hasattr(contract, "CUTOFF_FRACTION")
    assert not hasattr(contract, "WEIGHT_NORMALIZATION")
    assert contract.PER_OUTPUT_WEIGHT_NORMALIZATION is False
    assert contract.PASSBAND_MAX_HZ == {50: 12.0, 30: 12.0, 25: 10.0}
    assert contract.STOPBAND_START_HZ == {50: 25.0, 30: 15.0, 25: 12.5}
    for rate, start in contract.STOPBAND_START_HZ.items():
        # The stopband begins at the output Nyquist, so nothing above it
        # survives to fold anywhere in the output band.
        assert start == rate / 2.0


@pytest.mark.parametrize(
    ("rate", "preserved_hz", "stopband_hz"),
    ((50, 12.0, 25.0), (30, 12.0, 15.0), (25, 10.0, 12.5)),
)
def test_each_rate_preserves_its_band_and_stops_its_stopband(
    rate: int, preserved_hz: float, stopband_hz: float
) -> None:
    attenuation_at_edge = -20.0 * np.log10(
        filters.frequency_response(np.array([preserved_hz]), rate)[0]
    )
    assert abs(attenuation_at_edge) <= contract.PASSBAND_RIPPLE_MAX_DB
    working = filters.FILTER_SPECS[rate].working_rate_hz
    stopband = filters.frequency_response(
        np.linspace(stopband_hz, working / 2.0, 4096), rate
    )
    assert -20.0 * np.log10(stopband.max()) >= (
        contract.STOPBAND_ATTENUATION_MIN_DB
    )


@pytest.mark.parametrize("rate", (50, 30, 25))
def test_every_filter_meets_the_frozen_specification(rate: int) -> None:
    filters.assert_meets_specification(rate)
    measured = filters.measured_specification(rate)
    assert measured["dc_gain"] == pytest.approx(1.0, abs=1e-9)
    assert measured["symmetric"] is True
    assert measured["taps"] % 2 == 1
    assert measured["passband_ripple_db"] <= contract.PASSBAND_RIPPLE_MAX_DB
    assert measured["stopband_attenuation_db"] >= (
        contract.STOPBAND_ATTENUATION_MIN_DB
    )


def test_twelve_hertz_at_thirty_is_no_longer_attenuated_by_design() -> None:
    # The whole point of the correction: -6.02 dB became -0.003 dB.
    edge = filters.measured_specification(30)["passband_edge_db"]
    assert abs(edge) < 0.01
    ten = filters.measured_specification(25)["passband_edge_db"]
    assert abs(ten) < 0.01


@pytest.mark.parametrize("rate", (50, 30, 25))
def test_the_coefficient_hashes_are_pinned(rate: int) -> None:
    taps = filters.design(rate)
    assert hashlib.sha256(
        np.ascontiguousarray(taps, dtype=np.float64).tobytes()
    ).hexdigest() == filters.filter_sha256(rate)
    assert filters.design(rate) is filters.design(rate)


def test_the_combined_coefficient_hash_is_pinned() -> None:
    assert filters.coefficients_sha256() == (
        "976957f77d3ba0edbe72507bb32617751bbf1f3c1f38e299c5ce5e4120163d81"
    )
    assert FROZEN_DEPENDENCY.anti_alias_coefficients_sha256 == (
        filters.coefficients_sha256()
    )


def test_the_polyphase_branch_gains_are_reported_not_normalized() -> None:
    gains = filters.polyphase_dc_gains(30)
    assert len(gains) == 3
    spread = max(gains) - min(gains)
    # Reported: normalizing each branch would replace one frozen transfer
    # function with three.
    assert spread < 1e-5
    assert 20.0 * np.log10(max(gains) / min(gains)) < 0.001
    assert filters.polyphase_dc_gains(50) == [pytest.approx(1.0)]


def test_the_parent_rate_carries_no_anti_alias_filter() -> None:
    assert contract.PARENT_RATE_HZ == 100
    assert contract.PARENT_HAS_ANTI_ALIAS_FILTER is False
    assert 100 not in filters.FILTER_SPECS
    with pytest.raises(filters.AntiAliasError):
        filters.design(100)


def test_linear_interpolation_is_declared_as_not_transparent() -> None:
    assert contract.SOURCE_TO_PARENT == "SOURCE_TIME_LINEAR_INTERPOLATION"
    reference = filters.stage_a_reference_response()
    # 100 Hz is an ablation in its own right; the reference says why.
    assert reference["12_hz_db"] == pytest.approx(-0.4135, abs=0.001)
    assert reference["10_hz_db"] == pytest.approx(-0.2867, abs=0.001)
    assert reference["3_hz_db"] == pytest.approx(-0.0257, abs=0.001)


def test_the_stress_band_is_structurally_labelled() -> None:
    assert contract.STRESS_BAND_HZ == {25: (10.0, 12.0)}
    assert contract.EDGE_BAND == "EDGE_STRESS_10_TO_12_HZ"
    assert band_of(10.25) == contract.EDGE_BAND
    assert band_of(12.0) == contract.EDGE_BAND
    assert band_of(10.0) == contract.CORE_BAND
    # Only 25 Hz has a stress band: at 50 and 30 Hz the whole analysis grid
    # sits inside the preservation band.
    assert 50 not in contract.STRESS_BAND_HZ
    assert 30 not in contract.STRESS_BAND_HZ


# --- exact derived timing -------------------------------------------------


@pytest.mark.parametrize(("rate", "decimate"), ((50, 2), (25, 4)))
def test_integer_decimation_lands_on_exact_parent_samples(
    rate: int, decimate: int
) -> None:
    parent = parent_grid()
    derived = grid_for(rate)
    assert contract.RESAMPLING_RATIOS[rate] == (1, decimate)
    for ordinal in (0, 1, 97, 1000):
        assert derived.sample_seconds(ordinal) == Fraction(ordinal, rate)
        assert derived.sample_picoseconds_exact(ordinal) == (
            parent.sample_picoseconds_exact(decimate * ordinal)
        )


def test_thirty_hertz_lands_on_an_exact_rational_grid_without_drift() -> None:
    derived = grid_for(30)
    upsample, decimate = contract.RESAMPLING_RATIOS[30]
    assert (upsample, decimate) == (3, 10)
    working = Fraction(contract.PARENT_RATE_HZ * upsample)
    for ordinal in (0, 1, 7, 599, 10_000):
        # Ordinal k sits on working index 10k, which is exactly k/30 s.
        assert Fraction(decimate * ordinal) / working == Fraction(ordinal, 30)
        assert derived.sample_seconds(ordinal) == Fraction(ordinal, 30)
    # No cumulative drift: the exact time is a single multiplication, and the
    # step between consecutive ordinals never varies.
    steps = {
        derived.sample_seconds(k + 1) - derived.sample_seconds(k)
        for k in range(0, 10_000, 997)
    }
    assert steps == {Fraction(1, 30)}
    assert derived.sample_picoseconds_exact(1) is None


# --- segment edges --------------------------------------------------------


@pytest.mark.parametrize("rate", (50, 30, 25))
def test_insufficient_filter_support_refuses_output(rate: int) -> None:
    taps = filters.design(rate).size
    supported = supported_output_ordinals(
        rate, taps=taps, parent_first=0, parent_last=1999
    )
    assert supported.start > 0
    assert supported.stop - 1 < (1999 * rate) // 100
    for ordinal in supported:
        first, last = parent_span_for_output(rate, ordinal, taps=taps)
        assert first >= 0
        assert last <= 1999
    # The ordinal just before the first supported one would need parent
    # samples that do not exist, so it produces nothing rather than padding.
    first, _ = parent_span_for_output(rate, supported.start - 1, taps=taps)
    assert first < 0


@pytest.mark.parametrize("rate", (50, 30, 25))
def test_a_segment_shorter_than_the_kernel_yields_no_output(
    rate: int,
) -> None:
    taps = filters.design(rate).size
    assert supported_output_ordinals(
        rate, taps=taps, parent_first=0, parent_last=10
    ) == range(0)
    assert supported_output_ordinals(
        rate, taps=taps, parent_first=5, parent_last=4
    ) == range(0)


def test_no_padding_or_renormalization_is_permitted() -> None:
    assert contract.EDGE_POLICY == "REFUSE_UNSUPPORTED_OUTPUT_SAMPLES"
    assert contract.EDGE_PADDING_ALLOWED is False
    assert contract.TRUNCATED_KERNEL_RENORMALIZATION_ALLOWED is False
    assert contract.WINDOW_ELIGIBILITY == "FULLY_INSIDE_SUPPORTED_OUTPUT"


def test_an_even_length_kernel_is_refused() -> None:
    with pytest.raises(RationalTimeError):
        parent_span_for_output(50, 0, taps=32)


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
