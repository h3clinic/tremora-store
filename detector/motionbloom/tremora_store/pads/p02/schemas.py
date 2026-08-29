"""Closed Arrow schemas for PADS-P0.2 indexes, sample store and windows.

Every table and field name passes two screens at construction: the inherited
video screen, and a P0.2 screen for spectral, resampling and classification
names.  Both work on substrings, so ``spectrum_ref`` cannot slip past a
deny-list of exact names.

Source-derived times are int64 picoseconds.  See :mod:`.exact_time` for why
nanoseconds would round a digit the release actually wrote.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

from .contract import (
    P02_CONTRACT_VERSION,
    P02_SCHEMA_VERSION,
    TIMING_AUTHORITY,
    assert_p02_names,
)


def _schema(fields: list[pa.Field], table_name: str) -> pa.Schema:
    assert_p02_names([table_name, *(field.name for field in fields)])
    return pa.schema(
        fields,
        metadata={
            b"tremora.schema_version": P02_SCHEMA_VERSION.encode("ascii"),
            b"tremora.table": table_name.encode("ascii"),
            b"tremora.timing_authority": TIMING_AUTHORITY.encode("ascii"),
            b"tremora.contract_version": (
                P02_CONTRACT_VERSION.encode("ascii")
            ),
            b"tremora.time_unit": b"PICOSECOND",
        },
    )


def pads_samples_schema() -> pa.Schema:
    """One row per source sample, in source order, packed by stream."""

    return _schema([
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("assessment_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("device_location", pa.string(), nullable=False),
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("sample_ordinal", pa.int32(), nullable=False),
        pa.field("source_row_ordinal", pa.int32(), nullable=False),
        pa.field("source_time_token", pa.string(), nullable=False),
        pa.field("source_time_ps", pa.int64(), nullable=False),
        pa.field("task_local_time_ps", pa.int64(), nullable=False),
        pa.field("accelerometer_x", pa.float64(), nullable=False),
        pa.field("accelerometer_y", pa.float64(), nullable=False),
        pa.field("accelerometer_z", pa.float64(), nullable=False),
        pa.field("gyroscope_x", pa.float64(), nullable=False),
        pa.field("gyroscope_y", pa.float64(), nullable=False),
        pa.field("gyroscope_z", pa.float64(), nullable=False),
        pa.field("sample_status", pa.string(), nullable=False),
        pa.field("source_asset_sha256", pa.string(), nullable=False),
        pa.field("p01_evidence_sha256", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ], "pads_samples")


def pads_stream_storage_index_schema() -> pa.Schema:
    """Where each stream physically lives; exactly one row group per stream."""

    return _schema([
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("assessment_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("device_location", pa.string(), nullable=False),
        pa.field("parquet_relative_path", pa.string(), nullable=False),
        pa.field("row_group_index", pa.int32(), nullable=False),
        pa.field("first_sample_ordinal", pa.int32(), nullable=False),
        pa.field("last_sample_ordinal", pa.int32(), nullable=False),
        pa.field("sample_count", pa.int32(), nullable=False),
        pa.field("first_source_time_ps", pa.int64(), nullable=False),
        pa.field("last_source_time_ps", pa.int64(), nullable=False),
        pa.field("first_task_local_time_ps", pa.int64(), nullable=False),
        pa.field("last_task_local_time_ps", pa.int64(), nullable=False),
        pa.field("source_asset_sha256", pa.string(), nullable=False),
        pa.field("row_group_content_sha256", pa.string(), nullable=False),
    ], "pads_stream_storage_index")


def pads_participants_schema() -> pa.Schema:
    """One row per participant; grouping metadata only."""

    return _schema([
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("source_patient_asset_sha256", pa.string(), nullable=False),
        pa.field("condition_group", pa.string(), nullable=False),
        pa.field("split_group_id", pa.string(), nullable=False),
        pa.field("outer_fold", pa.int32(), nullable=False),
        pa.field("participant_status", pa.string(), nullable=False),
    ], "pads_participants")


def pads_assessments_schema() -> pa.Schema:
    """One row per participant and task, with its two wrist streams."""

    return _schema([
        pa.field("assessment_id", pa.string(), nullable=False),
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("task_ordinal", pa.int32(), nullable=False),
        pa.field("declared_sampling_rate_hz", pa.float64(), nullable=False),
        pa.field("declared_row_count", pa.int32(), nullable=False),
        pa.field(
            "expected_sample_support_seconds", pa.float64(), nullable=False
        ),
        pa.field("left_stream_id", pa.string(), nullable=False),
        pa.field("right_stream_id", pa.string(), nullable=False),
        pa.field("bilateral_pair_status", pa.string(), nullable=False),
        pa.field("cross_wrist_clock_alignment", pa.string(), nullable=False),
        pa.field("sample_level_fusion_allowed", pa.bool_(), nullable=False),
    ], "pads_assessments")


def pads_streams_schema() -> pa.Schema:
    """One row per device stream, with its observed cadence."""

    return _schema([
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("assessment_id", pa.string(), nullable=False),
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("device_location", pa.string(), nullable=False),
        pa.field("source_asset_sha256", pa.string(), nullable=False),
        pa.field("source_row_count", pa.int32(), nullable=False),
        pa.field("stored_row_count", pa.int32(), nullable=False),
        pa.field("declared_sampling_rate_hz", pa.float64(), nullable=False),
        pa.field("median_interval_ps", pa.int64()),
        pa.field("minimum_interval_ps", pa.int64()),
        pa.field("maximum_interval_ps", pa.int64()),
        pa.field("cadence_mad_ps", pa.int64()),
        pa.field("source_time_start_ps", pa.int64(), nullable=False),
        pa.field("source_time_end_ps", pa.int64(), nullable=False),
        pa.field("source_time_origin_token", pa.string(), nullable=False),
        pa.field("source_time_origin_ps", pa.int64(), nullable=False),
        pa.field("sample_support_seconds", pa.float64(), nullable=False),
        pa.field(
            "first_to_last_span_seconds", pa.float64(), nullable=False
        ),
        pa.field("segment_count", pa.int32(), nullable=False),
        pa.field("stream_status", pa.string(), nullable=False),
    ], "pads_streams")


def pads_segments_schema() -> pa.Schema:
    """Contiguous runs of one stream, partitioning it exactly."""

    return _schema([
        pa.field("segment_id", pa.string(), nullable=False),
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("segment_ordinal", pa.int32(), nullable=False),
        pa.field("first_sample_ordinal", pa.int32(), nullable=False),
        pa.field("last_sample_ordinal", pa.int32(), nullable=False),
        pa.field("sample_count", pa.int32(), nullable=False),
        pa.field("start_source_time_ps", pa.int64(), nullable=False),
        pa.field("end_source_time_ps", pa.int64(), nullable=False),
        pa.field("start_task_local_time_ps", pa.int64(), nullable=False),
        pa.field("end_task_local_time_ps", pa.int64(), nullable=False),
        pa.field("dt_ref_ps", pa.int64()),
        pa.field("gap_threshold_ps", pa.int64()),
        pa.field("break_reason_before", pa.string(), nullable=False),
        pa.field("break_reason_after", pa.string(), nullable=False),
        pa.field("segment_status", pa.string(), nullable=False),
    ], "pads_segments")


def pads_windows_schema() -> pa.Schema:
    """Four-second windows, each contained in one contiguous segment."""

    return _schema([
        pa.field("window_id", pa.string(), nullable=False),
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("assessment_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("device_location", pa.string(), nullable=False),
        pa.field("segment_id", pa.string(), nullable=False),
        pa.field("window_start_task_local_ps", pa.int64(), nullable=False),
        pa.field("window_end_task_local_ps", pa.int64(), nullable=False),
        pa.field("first_sample_ordinal", pa.int32(), nullable=False),
        pa.field("last_sample_ordinal", pa.int32(), nullable=False),
        pa.field("sample_count", pa.int32(), nullable=False),
        pa.field("first_source_time_ps", pa.int64(), nullable=False),
        pa.field("last_source_time_ps", pa.int64(), nullable=False),
        pa.field("dt_ref_ps", pa.int64(), nullable=False),
        pa.field("coverage_fraction", pa.float64(), nullable=False),
        pa.field("effective_rate_hz", pa.float64(), nullable=False),
        pa.field("split_group_id", pa.string(), nullable=False),
        pa.field("outer_fold", pa.int32(), nullable=False),
        pa.field("window_status", pa.string(), nullable=False),
        pa.field("timing_authority", pa.string(), nullable=False),
    ], "pads_windows")


def pads_bilateral_tasks_schema() -> pa.Schema:
    """Task-level wrist pairing, with the alignment claim it does not make."""

    return _schema([
        pa.field("assessment_id", pa.string(), nullable=False),
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("left_stream_id", pa.string(), nullable=False),
        pa.field("right_stream_id", pa.string(), nullable=False),
        pa.field("pairing_authority", pa.string(), nullable=False),
        pa.field("cross_wrist_clock_alignment", pa.string(), nullable=False),
        pa.field("sample_level_fusion_allowed", pa.bool_(), nullable=False),
        pa.field("pair_status", pa.string(), nullable=False),
    ], "pads_bilateral_tasks")


def pads_bilateral_window_pairs_schema() -> pa.Schema:
    """Windows co-indexed by task-local offset, not by sample identity."""

    return _schema([
        pa.field("bilateral_window_pair_id", pa.string(), nullable=False),
        pa.field("assessment_id", pa.string(), nullable=False),
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("window_start_task_local_ps", pa.int64(), nullable=False),
        pa.field("window_end_task_local_ps", pa.int64(), nullable=False),
        pa.field("left_window_id", pa.string(), nullable=False),
        pa.field("right_window_id", pa.string(), nullable=False),
        pa.field("pairing_status", pa.string(), nullable=False),
        pa.field("pairing_authority", pa.string(), nullable=False),
        pa.field("cross_wrist_clock_alignment", pa.string(), nullable=False),
        pa.field("sample_level_fusion_allowed", pa.bool_(), nullable=False),
        pa.field("split_group_id", pa.string(), nullable=False),
        pa.field("outer_fold", pa.int32(), nullable=False),
    ], "pads_bilateral_window_pairs")


P02_TABLE_SCHEMAS: dict[str, Callable[[], pa.Schema]] = {
    "pads_samples": pads_samples_schema,
    "pads_stream_storage_index": pads_stream_storage_index_schema,
    "pads_participants": pads_participants_schema,
    "pads_assessments": pads_assessments_schema,
    "pads_streams": pads_streams_schema,
    "pads_segments": pads_segments_schema,
    "pads_windows": pads_windows_schema,
    "pads_bilateral_tasks": pads_bilateral_tasks_schema,
    "pads_bilateral_window_pairs": pads_bilateral_window_pairs_schema,
}

P02_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "pads_samples": ("stream_id", "sample_ordinal"),
    "pads_stream_storage_index": ("stream_id",),
    "pads_participants": ("participant_id",),
    "pads_assessments": ("participant_id", "task_ordinal"),
    "pads_streams": ("stream_id",),
    "pads_segments": ("stream_id", "segment_ordinal"),
    "pads_windows": ("stream_id", "window_start_task_local_ps"),
    "pads_bilateral_tasks": ("assessment_id",),
    "pads_bilateral_window_pairs": (
        "assessment_id", "window_start_task_local_ps",
    ),
}

#: Index tables are written whole; only the sample store is packed per stream.
P02_INDEX_FILES: dict[str, str] = {
    "pads_stream_storage_index": "pads_stream_storage_index.parquet",
    "pads_participants": "pads_participants.parquet",
    "pads_assessments": "pads_assessments.parquet",
    "pads_streams": "pads_streams.parquet",
    "pads_segments": "pads_segments.parquet",
    "pads_windows": "pads_windows.parquet",
    "pads_bilateral_tasks": "pads_bilateral_tasks.parquet",
    "pads_bilateral_window_pairs": "pads_bilateral_window_pairs.parquet",
}


__all__ = [
    "P02_INDEX_FILES",
    "P02_SORT_KEYS",
    "P02_TABLE_SCHEMAS",
    "pads_assessments_schema",
    "pads_bilateral_tasks_schema",
    "pads_bilateral_window_pairs_schema",
    "pads_participants_schema",
    "pads_samples_schema",
    "pads_segments_schema",
    "pads_stream_storage_index_schema",
    "pads_streams_schema",
    "pads_windows_schema",
]
