"""Closed Arrow schemas for VIDIMU v0.5D ordinal alignment evidence."""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

from .authority import (
    ALIGNMENT_CONTRACT_VERSION,
    assert_no_forbidden_clock_fields,
)

V05D_SCHEMA_VERSION = "0.5d.0"


def _schema(fields: list[pa.Field], table_name: str) -> pa.Schema:
    assert_no_forbidden_clock_fields(field.name for field in fields)
    return pa.schema(
        fields,
        metadata={
            b"tremora.schema_version": V05D_SCHEMA_VERSION.encode("ascii"),
            b"tremora.table": table_name.encode("ascii"),
            b"tremora.alignment_contract_version": (
                ALIGNMENT_CONTRACT_VERSION.encode("ascii")
            ),
            b"tremora.temporal_domain": b"ORDINAL_ONLY_NO_CANONICAL_CLOCK",
        },
    )


def sto_alignment_contracts_schema() -> pa.Schema:
    """One closed source-derived alignment decision per VIDIMU pair."""

    return _schema([
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("pair_id", pa.string(), nullable=False),
        pa.field("video_asset_sha256", pa.string(), nullable=False),
        pa.field("imu_raw_asset_sha256", pa.string(), nullable=False),
        pa.field("imu_mot_asset_sha256", pa.string(), nullable=False),
        pa.field("video_csv_asset_sha256", pa.string(), nullable=False),
        pa.field("source_tools_repository_url", pa.string(), nullable=False),
        pa.field("source_tools_commit", pa.string(), nullable=False),
        pa.field("estimate_notebook_sha256", pa.string(), nullable=False),
        pa.field("modify_notebook_sha256", pa.string(), nullable=False),
        pa.field("sync_utility_sha256", pa.string(), nullable=False),
        pa.field("override_source_asset_sha256", pa.string()),
        pa.field("override_row_ordinal", pa.int32()),
        pa.field("override_row_sha256", pa.string()),
        pa.field("alignment_authority", pa.string(), nullable=False),
        pa.field("alignment_method", pa.string(), nullable=False),
        pa.field("alignment_contract_version", pa.string(), nullable=False),
        pa.field("source_subject", pa.string(), nullable=False),
        pa.field("source_activity", pa.string(), nullable=False),
        pa.field("source_trial", pa.string(), nullable=False),
        pa.field("cut_side", pa.string()),
        pa.field("video_cut_frames", pa.int32()),
        pa.field("imu_cut_ticks", pa.int32()),
        pa.field("imu_raw_cut_rows", pa.int32()),
        pa.field("imu_mot_cut_rows", pa.int32()),
        pa.field("video_csv_cut_rows", pa.int32()),
        pa.field("nominal_video_rate_num", pa.int32(), nullable=False),
        pa.field("nominal_video_rate_den", pa.int32(), nullable=False),
        pa.field("nominal_imu_rate_num", pa.int32(), nullable=False),
        pa.field("nominal_imu_rate_den", pa.int32(), nullable=False),
        pa.field("alignment_status", pa.string(), nullable=False),
        pa.field("ambiguity_status", pa.string(), nullable=False),
        pa.field("eligibility_status", pa.string(), nullable=False),
        pa.field("exclusion_reason", pa.string()),
    ], "sto_alignment_contracts")


def imu_tick_groups_schema() -> pa.Schema:
    """Requested RAW-row group shape; population still requires tick authority."""

    return _schema([
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("imu_tick_ordinal", pa.int64(), nullable=False),
        pa.field("first_raw_row_ordinal", pa.int64(), nullable=False),
        pa.field("last_raw_row_ordinal", pa.int64(), nullable=False),
        pa.field("sensor_row_count", pa.int32(), nullable=False),
        pa.field("expected_sensor_count", pa.int32(), nullable=False),
        pa.field("observed_sensor_count", pa.int32(), nullable=False),
        pa.field("group_status", pa.string(), nullable=False),
    ], "imu_tick_groups")


def derived_rate_contract_schema() -> pa.Schema:
    """Nominal-rate relationship in an aligned ordinal domain only."""

    return _schema([
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("video_rate_num", pa.int32(), nullable=False),
        pa.field("video_rate_den", pa.int32(), nullable=False),
        pa.field("imu_rate_num", pa.int32(), nullable=False),
        pa.field("imu_rate_den", pa.int32(), nullable=False),
        pa.field("rate_ratio_num", pa.int32(), nullable=False),
        pa.field("rate_ratio_den", pa.int32(), nullable=False),
        pa.field("video_origin_ordinal", pa.int64(), nullable=False),
        pa.field("imu_origin_ordinal", pa.int64(), nullable=False),
        pa.field("phase_assumption", pa.string(), nullable=False),
        pa.field("timing_authority", pa.string(), nullable=False),
        pa.field("uncertainty_status", pa.string(), nullable=False),
    ], "derived_rate_contract")


def sto_alignment_validation_schema() -> pa.Schema:
    """Non-authoritative reproduction diagnostics for the upstream RMSE step."""

    return _schema([
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("source_cut_frames", pa.int32()),
        pa.field("reproduced_best_cut_frames", pa.int32()),
        pa.field("source_best_rmse", pa.float64()),
        pa.field("reproduced_best_rmse", pa.float64()),
        pa.field("second_best_cut_frames", pa.int32()),
        pa.field("second_best_rmse", pa.float64()),
        pa.field("best_second_margin", pa.float64()),
        pa.field("candidate_count_within_tolerance", pa.int32()),
        pa.field("validation_status", pa.string(), nullable=False),
    ], "sto_alignment_validation")


def source_trim_overlays_schema() -> pa.Schema:
    """All 217 non-MP4 source instructions bound to immutable byte assets."""

    return _schema([
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("pair_id", pa.string(), nullable=False),
        pa.field("override_type", pa.string(), nullable=False),
        pa.field("override_source_asset_sha256", pa.string(), nullable=False),
        pa.field("override_row_ordinal", pa.int32(), nullable=False),
        pa.field("override_row_sha256", pa.string(), nullable=False),
        pa.field("source_member_path", pa.string(), nullable=False),
        pa.field("source_asset_sha256", pa.string(), nullable=False),
        pa.field("published_member_path", pa.string(), nullable=False),
        pa.field("published_asset_sha256", pa.string(), nullable=False),
        pa.field("cut_frames", pa.int32(), nullable=False),
        pa.field("retained_header_rows", pa.int32(), nullable=False),
        pa.field("trim_start_data_row_ordinal", pa.int64(), nullable=False),
        pa.field("trim_stop_data_row_ordinal", pa.int64(), nullable=False),
        pa.field("expected_removed_rows", pa.int32(), nullable=False),
        pa.field("observed_removed_rows", pa.int32(), nullable=False),
        pa.field("generated_derivative_sha256", pa.string(), nullable=False),
        pa.field("comparison_status", pa.string(), nullable=False),
    ], "source_trim_overlays")


V05D_TABLE_SCHEMAS: dict[str, Callable[[], pa.Schema]] = {
    "sto_alignment_contracts": sto_alignment_contracts_schema,
    "imu_tick_groups": imu_tick_groups_schema,
    "derived_rate_contract": derived_rate_contract_schema,
    "sto_alignment_validation": sto_alignment_validation_schema,
    "source_trim_overlays": source_trim_overlays_schema,
}

V05D_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "sto_alignment_contracts": ("recording_id",),
    "imu_tick_groups": ("recording_id", "imu_tick_ordinal"),
    "derived_rate_contract": ("recording_id",),
    "sto_alignment_validation": ("recording_id",),
    "source_trim_overlays": ("override_row_ordinal",),
}


__all__ = [
    "V05D_SCHEMA_VERSION",
    "V05D_SORT_KEYS",
    "V05D_TABLE_SCHEMAS",
    "derived_rate_contract_schema",
    "imu_tick_groups_schema",
    "source_trim_overlays_schema",
    "sto_alignment_contracts_schema",
    "sto_alignment_validation_schema",
]
