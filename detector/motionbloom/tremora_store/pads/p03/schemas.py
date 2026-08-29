"""Closed Arrow schemas for PADS-P0.3 spectral-preservation evidence.

Power vectors are fixed-size 37-element float64 lists, so the grid width is a
property of the schema rather than a convention the writer has to remember.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

from .contract import (
    FREQUENCY_BIN_COUNT,
    P03_CONTRACT_VERSION,
    P03_SCHEMA_VERSION,
    SPECTRAL_CONTRACT_VERSION,
    assert_p03_names,
)

POWER_VECTOR = pa.list_(pa.float64(), FREQUENCY_BIN_COUNT)


def _schema(fields: list[pa.Field], table_name: str) -> pa.Schema:
    assert_p03_names([table_name, *(field.name for field in fields)])
    return pa.schema(
        fields,
        metadata={
            b"tremora.schema_version": P03_SCHEMA_VERSION.encode("ascii"),
            b"tremora.table": table_name.encode("ascii"),
            b"tremora.contract_version": (
                P03_CONTRACT_VERSION.encode("ascii")
            ),
            b"tremora.spectral_contract_version": (
                SPECTRAL_CONTRACT_VERSION.encode("ascii")
            ),
            b"tremora.time_unit": b"PICOSECOND",
        },
    )


def pads_p03_workload_windows_schema() -> pa.Schema:
    """One canonical window per stream that has a valid P0.2.1 window."""

    return _schema([
        pa.field("window_id", pa.string(), nullable=False),
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("assessment_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("device_location", pa.string(), nullable=False),
        pa.field("outer_fold", pa.int32(), nullable=False),
        pa.field("sample_count", pa.int32(), nullable=False),
        pa.field("dt_ref_ps", pa.int64(), nullable=False),
        pa.field("coverage_fraction", pa.float64(), nullable=False),
        pa.field("effective_rate_hz", pa.float64(), nullable=False),
        pa.field("nyquist_hz", pa.float64(), nullable=False),
        pa.field("gap_adjacent_status", pa.string(), nullable=False),
        pa.field("workload_selection_reason", pa.string(), nullable=False),
        pa.field("spectral_eligibility", pa.string(), nullable=False),
        pa.field("ineligibility_reason", pa.string()),
    ], "pads_p03_workload_windows")


def pads_p03_spectra_schema() -> pa.Schema:
    """One row per workload window and sensor family."""

    return _schema([
        pa.field("spectral_record_id", pa.string(), nullable=False),
        pa.field("window_id", pa.string(), nullable=False),
        pa.field("sensor_family", pa.string(), nullable=False),
        pa.field("axis_x_power", POWER_VECTOR, nullable=False),
        pa.field("axis_y_power", POWER_VECTOR, nullable=False),
        pa.field("axis_z_power", POWER_VECTOR, nullable=False),
        pa.field("aggregate_power", POWER_VECTOR, nullable=False),
        pa.field("normalized_aggregate_power", POWER_VECTOR, nullable=False),
        pa.field("dominant_frequency_hz", pa.float64(), nullable=False),
        pa.field("band_power", pa.float64(), nullable=False),
        pa.field("spectral_entropy", pa.float64(), nullable=False),
        pa.field("peak_to_median_ratio", pa.float64(), nullable=False),
        pa.field("sample_count", pa.int32(), nullable=False),
        pa.field("dt_ref_ps", pa.int64(), nullable=False),
        pa.field("frequency_grid_id", pa.string(), nullable=False),
        pa.field("input_content_sha256", pa.string(), nullable=False),
        pa.field("spectral_content_sha256", pa.string(), nullable=False),
        pa.field("spectral_status", pa.string(), nullable=False),
    ], "pads_p03_spectra")


def pads_p03_source_replay_audit_schema() -> pa.Schema:
    """One row per independently audited window."""

    return _schema([
        pa.field("window_id", pa.string(), nullable=False),
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("stratum_id", pa.string(), nullable=False),
        pa.field("source_asset_sha256", pa.string(), nullable=False),
        pa.field("source_row_count", pa.int32(), nullable=False),
        pa.field("replay_row_count", pa.int32(), nullable=False),
        pa.field("source_row_identity_sha256", pa.string(), nullable=False),
        pa.field("replay_row_identity_sha256", pa.string(), nullable=False),
        pa.field("source_input_sha256", pa.string(), nullable=False),
        pa.field("replay_input_sha256", pa.string(), nullable=False),
        pa.field(
            "source_accel_spectrum_sha256", pa.string(), nullable=False
        ),
        pa.field(
            "replay_accel_spectrum_sha256", pa.string(), nullable=False
        ),
        pa.field("source_gyro_spectrum_sha256", pa.string(), nullable=False),
        pa.field("replay_gyro_spectrum_sha256", pa.string(), nullable=False),
        pa.field("maximum_bin_absolute_error", pa.float64(), nullable=False),
        pa.field("dominant_frequency_match", pa.bool_(), nullable=False),
        pa.field("preservation_status", pa.string(), nullable=False),
    ], "pads_p03_source_replay_audit")


P03_TABLE_SCHEMAS: dict[str, Callable[[], pa.Schema]] = {
    "pads_p03_workload_windows": pads_p03_workload_windows_schema,
    "pads_p03_spectra": pads_p03_spectra_schema,
    "pads_p03_source_replay_audit": pads_p03_source_replay_audit_schema,
}

P03_TABLE_FILES: dict[str, str] = {
    "pads_p03_workload_windows": "pads_p03_workload_windows.parquet",
    "pads_p03_spectra": "pads_p03_spectra.parquet",
    "pads_p03_source_replay_audit": "pads_p03_source_replay_audit.parquet",
}

P03_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "pads_p03_workload_windows": ("window_id",),
    "pads_p03_spectra": ("window_id", "sensor_family"),
    "pads_p03_source_replay_audit": ("window_id",),
}

FREQUENCY_GRID_FILENAME = "pads_p03_frequency_grid.json"


__all__ = [
    "FREQUENCY_GRID_FILENAME",
    "P03_SORT_KEYS",
    "P03_TABLE_FILES",
    "P03_TABLE_SCHEMAS",
    "POWER_VECTOR",
    "pads_p03_source_replay_audit_schema",
    "pads_p03_spectra_schema",
    "pads_p03_workload_windows_schema",
]
