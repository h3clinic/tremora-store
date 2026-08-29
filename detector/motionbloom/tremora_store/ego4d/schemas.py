"""Closed Arrow schemas for E4D-P0.1 timing-authority evidence.

Four tables: one row per considered asset triple, one row per source IMU row in
source order, one row per component placement, and one additive accounting row
per video.  The acceleration columns keep the source's ``accl_*`` spelling.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

from .authority import EGO4D_CONTRACT_VERSION, EGO4D_SCHEMA_VERSION

#: The row table stores canonical time but never a frame association, so the
#: P0.2 index cannot be smuggled into a P0.1 artifact.
FORBIDDEN_INDEX_FIELDS = frozenset({
    "coverage_fraction",
    "first_imu_canonical_ordinal",
    "frame_end_ms",
    "frame_id",
    "frame_start_ms",
    "last_imu_canonical_ordinal",
    "window_id",
})


class Ego4DSchemaError(ValueError):
    """Raised when a P0.1 schema would carry a P0.2 index field."""


def _schema(fields: list[pa.Field], table_name: str) -> pa.Schema:
    present = FORBIDDEN_INDEX_FIELDS.intersection(
        field.name for field in fields
    )
    if present:
        raise Ego4DSchemaError(
            f"{table_name} carries index fields {sorted(present)!r}")
    return pa.schema(
        fields,
        metadata={
            b"tremora.schema_version": EGO4D_SCHEMA_VERSION.encode("ascii"),
            b"tremora.table": table_name.encode("ascii"),
            b"tremora.timing_authority": b"SOURCE_CANONICAL_TIMESTAMP",
            b"tremora.contract_version": (
                EGO4D_CONTRACT_VERSION.encode("ascii")
            ),
        },
    )


def ego4d_imu_assets_schema() -> pa.Schema:
    """One row per considered ``(video, component)`` asset triple."""

    return _schema([
        pa.field("video_uid", pa.string(), nullable=False),
        pa.field("component_idx", pa.int32(), nullable=False),
        pa.field("imu_asset_sha256", pa.string(), nullable=False),
        pa.field("video_component_asset_sha256", pa.string(), nullable=False),
        pa.field("canonical_video_asset_sha256", pa.string(), nullable=False),
        pa.field("imu_row_count", pa.int64(), nullable=False),
        pa.field("source_component_start_ms", pa.float64()),
        pa.field("source_component_end_ms", pa.float64()),
        pa.field("asset_status", pa.string(), nullable=False),
        pa.field("failure_reason", pa.string()),
    ], "ego4d_imu_assets")


def ego4d_imu_authority_rows_schema() -> pa.Schema:
    """One row per source IMU row, in source order, tokens preserved."""

    return _schema([
        pa.field("video_uid", pa.string(), nullable=False),
        pa.field("component_idx", pa.int32(), nullable=False),
        pa.field("source_row_ordinal", pa.int64(), nullable=False),
        pa.field("component_timestamp_token", pa.string(), nullable=False),
        pa.field("component_timestamp_ms", pa.float64()),
        pa.field("canonical_timestamp_token", pa.string(), nullable=False),
        pa.field("canonical_timestamp_ms", pa.float64()),
        pa.field("gyro_x", pa.float64()),
        pa.field("gyro_y", pa.float64()),
        pa.field("gyro_z", pa.float64()),
        pa.field("accl_x", pa.float64()),
        pa.field("accl_y", pa.float64()),
        pa.field("accl_z", pa.float64()),
        pa.field("canonical_authority_status", pa.string(), nullable=False),
        pa.field("issue_bits", pa.int32(), nullable=False),
        pa.field("source_asset_sha256", pa.string(), nullable=False),
        pa.field("parser_version", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ], "ego4d_imu_authority_rows")


def ego4d_video_timeline_authority_schema() -> pa.Schema:
    """Component placement plus the PTS timeline it is checked against."""

    return _schema([
        pa.field("video_uid", pa.string(), nullable=False),
        pa.field("component_idx", pa.int32(), nullable=False),
        pa.field("component_start_in_canonical_ms", pa.float64()),
        pa.field("component_end_in_canonical_ms", pa.float64()),
        pa.field("canonical_video_duration_ms", pa.float64()),
        pa.field("video_stream_start_ms", pa.float64()),
        pa.field("video_stream_end_ms", pa.float64()),
        pa.field("metadata_source_sha256", pa.string(), nullable=False),
        pa.field("timeline_status", pa.string(), nullable=False),
    ], "ego4d_video_timeline_authority")


def ego4d_timing_authority_summary_schema() -> pa.Schema:
    """One additive accounting row per video."""

    return _schema([
        pa.field("video_uid", pa.string(), nullable=False),
        pa.field("imu_rows_total", pa.int64(), nullable=False),
        pa.field("canonical_rows_valid", pa.int64(), nullable=False),
        pa.field("canonical_rows_null", pa.int64(), nullable=False),
        pa.field("canonical_rows_nonfinite", pa.int64(), nullable=False),
        pa.field(
            "canonical_rows_nonmonotonic_source_order",
            pa.int64(),
            nullable=False,
        ),
        pa.field("canonical_rows_outside_video", pa.int64(), nullable=False),
        pa.field("canonical_rows_duplicate", pa.int64(), nullable=False),
        pa.field("canonical_rows_extreme", pa.int64(), nullable=False),
        pa.field("canonical_rows_unparseable", pa.int64(), nullable=False),
        pa.field("rows_missing_acceleration", pa.int64(), nullable=False),
        pa.field("rows_missing_gyroscope", pa.int64(), nullable=False),
        pa.field("components_expected", pa.int32(), nullable=False),
        pa.field("components_present", pa.int32(), nullable=False),
        pa.field("components_with_imu", pa.int32(), nullable=False),
        pa.field("components_without_imu", pa.int32(), nullable=False),
        pa.field("canonical_coverage_start_ms", pa.float64()),
        pa.field("canonical_coverage_end_ms", pa.float64()),
        pa.field("canonical_coverage_duration_ms", pa.float64()),
        pa.field("authority_eligible", pa.bool_(), nullable=False),
        pa.field("ineligibility_reason", pa.string()),
    ], "ego4d_timing_authority_summary")


EGO4D_TABLE_SCHEMAS: dict[str, Callable[[], pa.Schema]] = {
    "ego4d_imu_assets": ego4d_imu_assets_schema,
    "ego4d_imu_authority_rows": ego4d_imu_authority_rows_schema,
    "ego4d_video_timeline_authority": (
        ego4d_video_timeline_authority_schema
    ),
    "ego4d_timing_authority_summary": (
        ego4d_timing_authority_summary_schema
    ),
}

EGO4D_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "ego4d_imu_assets": ("video_uid", "component_idx"),
    "ego4d_imu_authority_rows": ("video_uid", "source_row_ordinal"),
    "ego4d_video_timeline_authority": ("video_uid", "component_idx"),
    "ego4d_timing_authority_summary": ("video_uid",),
}


__all__ = [
    "EGO4D_SORT_KEYS",
    "EGO4D_TABLE_SCHEMAS",
    "FORBIDDEN_INDEX_FIELDS",
    "Ego4DSchemaError",
    "ego4d_imu_assets_schema",
    "ego4d_imu_authority_rows_schema",
    "ego4d_timing_authority_summary_schema",
    "ego4d_video_timeline_authority_schema",
]
