"""Closed Arrow schemas for PADS-P0.4 rate-ablation evidence.

Grid timing is carried as exact rational numerator/denominator pairs rather
than a single integer, because 30 Hz has no exact picosecond period.  The
rounded picosecond form appears only where a human needs to read it, and never
as the value a transform is computed from.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

from ..p03.contract import FREQUENCY_BIN_COUNT
from .contract import (
    P04_CONTRACT_VERSION,
    P04_SCHEMA_VERSION,
    RESAMPLING_CONTRACT_VERSION,
    assert_p04_names,
)

POWER_VECTOR = pa.list_(pa.float64(), FREQUENCY_BIN_COUNT)


def _schema(fields: list[pa.Field], table_name: str) -> pa.Schema:
    assert_p04_names([table_name, *(field.name for field in fields)])
    return pa.schema(
        fields,
        metadata={
            b"tremora.schema_version": P04_SCHEMA_VERSION.encode("ascii"),
            b"tremora.table": table_name.encode("ascii"),
            b"tremora.contract_version": (
                P04_CONTRACT_VERSION.encode("ascii")
            ),
            b"tremora.resampling_contract_version": (
                RESAMPLING_CONTRACT_VERSION.encode("ascii")
            ),
            b"tremora.grid_timing": b"EXACT_RATIONAL",
        },
    )


def pads_p04_rate_grids_schema() -> pa.Schema:
    """One row per contiguous segment and derived rate."""

    return _schema([
        pa.field("segment_id", pa.string(), nullable=False),
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("device_location", pa.string(), nullable=False),
        pa.field("rate_hz", pa.int32(), nullable=False),
        pa.field("rate_hz_num", pa.int64(), nullable=False),
        pa.field("rate_hz_den", pa.int64(), nullable=False),
        pa.field("period_picoseconds_num", pa.int64(), nullable=False),
        pa.field("period_picoseconds_den", pa.int64(), nullable=False),
        pa.field("exact_in_picoseconds", pa.bool_(), nullable=False),
        pa.field("grid_origin", pa.string(), nullable=False),
        pa.field("first_ordinal", pa.int64(), nullable=False),
        pa.field("last_ordinal", pa.int64(), nullable=False),
        pa.field("derived_sample_count", pa.int32(), nullable=False),
        pa.field(
            "segment_start_task_local_ps", pa.int64(), nullable=False
        ),
        pa.field("segment_end_task_local_ps", pa.int64(), nullable=False),
        pa.field("cutoff_hz", pa.float64(), nullable=False),
        pa.field("support_seconds", pa.float64(), nullable=False),
        pa.field("minimum_taps_observed", pa.int32(), nullable=False),
        pa.field("samples_refused_for_support", pa.int32(), nullable=False),
        pa.field(
            "anti_alias_coefficients_sha256", pa.string(), nullable=False
        ),
        pa.field("grid_status", pa.string(), nullable=False),
    ], "pads_p04_rate_grids")


def pads_p04_rate_spectra_schema() -> pa.Schema:
    """One row per workload window, derived rate and sensor family."""

    return _schema([
        pa.field("spectral_record_id", pa.string(), nullable=False),
        pa.field("window_id", pa.string(), nullable=False),
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("device_location", pa.string(), nullable=False),
        pa.field("outer_fold", pa.int32(), nullable=False),
        pa.field("rate_hz", pa.int32(), nullable=False),
        pa.field("sensor_family", pa.string(), nullable=False),
        pa.field("aggregate_power", POWER_VECTOR, nullable=False),
        pa.field("normalized_aggregate_power", POWER_VECTOR, nullable=False),
        pa.field("dominant_frequency_hz", pa.float64(), nullable=False),
        pa.field("band_power", pa.float64(), nullable=False),
        pa.field("core_band_power", pa.float64(), nullable=False),
        pa.field("edge_band_power", pa.float64(), nullable=False),
        # The native P0.3 reference this row is compared against.
        pa.field(
            "native_dominant_frequency_hz", pa.float64(), nullable=False
        ),
        pa.field("native_core_band_power", pa.float64(), nullable=False),
        pa.field("native_edge_band_power", pa.float64(), nullable=False),
        pa.field(
            "dominant_frequency_shift_hz", pa.float64(), nullable=False
        ),
        pa.field("core_band_power_ratio", pa.float64(), nullable=False),
        pa.field("edge_band_power_ratio", pa.float64(), nullable=False),
        pa.field(
            "core_normalized_spectral_distance", pa.float64(), nullable=False
        ),
        pa.field(
            "edge_normalized_spectral_distance", pa.float64(), nullable=False
        ),
        pa.field("derived_sample_count", pa.int32(), nullable=False),
        pa.field("native_sample_count", pa.int32(), nullable=False),
        pa.field("input_content_sha256", pa.string(), nullable=False),
        pa.field("spectral_content_sha256", pa.string(), nullable=False),
        pa.field("native_spectral_content_sha256", pa.string(), nullable=False),
        pa.field("spectral_status", pa.string(), nullable=False),
    ], "pads_p04_rate_spectra")


def pads_p04_source_replay_audit_schema() -> pa.Schema:
    """Source-direct against replay-derived, for one window and rate."""

    return _schema([
        pa.field("window_id", pa.string(), nullable=False),
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("rate_hz", pa.int32(), nullable=False),
        pa.field("stratum_id", pa.string(), nullable=False),
        pa.field("source_asset_sha256", pa.string(), nullable=False),
        pa.field("source_derived_sample_count", pa.int32(), nullable=False),
        pa.field("replay_derived_sample_count", pa.int32(), nullable=False),
        pa.field("source_derived_sha256", pa.string(), nullable=False),
        pa.field("replay_derived_sha256", pa.string(), nullable=False),
        pa.field(
            "source_gyro_spectrum_sha256", pa.string(), nullable=False
        ),
        pa.field(
            "replay_gyro_spectrum_sha256", pa.string(), nullable=False
        ),
        pa.field(
            "source_accel_spectrum_sha256", pa.string(), nullable=False
        ),
        pa.field(
            "replay_accel_spectrum_sha256", pa.string(), nullable=False
        ),
        pa.field("maximum_bin_absolute_error", pa.float64(), nullable=False),
        pa.field("dominant_frequency_match", pa.bool_(), nullable=False),
        pa.field("preservation_status", pa.string(), nullable=False),
    ], "pads_p04_source_replay_audit")


def pads_p04_participant_summary_schema() -> pa.Schema:
    """One row per participant, derived rate, sensor family and band."""

    return _schema([
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("condition_group", pa.string(), nullable=False),
        pa.field("outer_fold", pa.int32(), nullable=False),
        pa.field("rate_hz", pa.int32(), nullable=False),
        pa.field("sensor_family", pa.string(), nullable=False),
        pa.field("band", pa.string(), nullable=False),
        pa.field("windows", pa.int32(), nullable=False),
        pa.field("median_band_power_ratio", pa.float64(), nullable=False),
        pa.field(
            "median_normalized_spectral_distance",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "median_dominant_frequency_shift_hz",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "windows_with_preserved_dominant_frequency",
            pa.int32(),
            nullable=False,
        ),
        pa.field("summary_status", pa.string(), nullable=False),
    ], "pads_p04_participant_summary")


P04_TABLE_SCHEMAS: dict[str, Callable[[], pa.Schema]] = {
    "pads_p04_rate_grids": pads_p04_rate_grids_schema,
    "pads_p04_rate_spectra": pads_p04_rate_spectra_schema,
    "pads_p04_source_replay_audit": pads_p04_source_replay_audit_schema,
    "pads_p04_participant_summary": pads_p04_participant_summary_schema,
}

P04_TABLE_FILES: dict[str, str] = {
    "pads_p04_rate_grids": "pads_p04_rate_grids.parquet",
    "pads_p04_rate_spectra": "pads_p04_rate_spectra.parquet",
    "pads_p04_source_replay_audit": (
        "pads_p04_source_replay_audit.parquet"
    ),
    "pads_p04_participant_summary": (
        "pads_p04_participant_summary.parquet"
    ),
}

P04_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "pads_p04_rate_grids": ("segment_id", "rate_hz"),
    "pads_p04_rate_spectra": ("window_id", "rate_hz", "sensor_family"),
    "pads_p04_source_replay_audit": ("window_id", "rate_hz"),
    "pads_p04_participant_summary": (
        "participant_id", "rate_hz", "sensor_family", "band",
    ),
}

ANTI_ALIAS_MANIFEST_FILENAME = "pads_p04_anti_alias_manifest.json"


__all__ = [
    "ANTI_ALIAS_MANIFEST_FILENAME",
    "P04_SORT_KEYS",
    "P04_TABLE_FILES",
    "P04_TABLE_SCHEMAS",
    "POWER_VECTOR",
    "pads_p04_participant_summary_schema",
    "pads_p04_rate_grids_schema",
    "pads_p04_rate_spectra_schema",
    "pads_p04_source_replay_audit_schema",
]
