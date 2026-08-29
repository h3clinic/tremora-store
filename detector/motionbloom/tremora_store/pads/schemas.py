"""Closed Arrow schemas for PADS-P0.1 ingest evidence.

Every table and field name is screened for video-bearing substrings at schema
construction, so an inertial-only dataset cannot acquire a cross-modal field by
accident.  The screen runs on substrings rather than exact names.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

from .authority import (
    MODALITY_STAMP,
    PADS_CONTRACT_VERSION,
    PADS_SCHEMA_VERSION,
    assert_no_paired_claim,
)


class PadsSchemaError(ValueError):
    """Raised when a PADS schema would imply a video association."""


def _schema(fields: list[pa.Field], table_name: str) -> pa.Schema:
    assert_no_paired_claim([table_name, *(field.name for field in fields)])
    return pa.schema(
        fields,
        metadata={
            b"tremora.schema_version": PADS_SCHEMA_VERSION.encode("ascii"),
            b"tremora.table": table_name.encode("ascii"),
            b"tremora.timing_authority": (
                b"SOURCE_RELATIVE_UNIMODAL_CLOCK"
            ),
            b"tremora.modality": MODALITY_STAMP.encode("ascii"),
            b"tremora.contract_version": (
                PADS_CONTRACT_VERSION.encode("ascii")
            ),
        },
    )


def pads_participants_schema() -> pa.Schema:
    """One row per participant."""

    return _schema([
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("condition", pa.string()),
        pa.field("diagnosis_detail", pa.string()),
        pa.field("handedness", pa.string()),
        pa.field("source_patient_sha256", pa.string()),
        pa.field("metadata_status", pa.string(), nullable=False),
    ], "pads_participants")


def pads_assessments_schema() -> pa.Schema:
    """One row per participant and published task."""

    return _schema([
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("assessment_id", pa.string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("task_ordinal", pa.int32(), nullable=False),
        pa.field("declared_row_count", pa.int32(), nullable=False),
        pa.field("declared_sampling_rate_num", pa.int32(), nullable=False),
        pa.field("declared_sampling_rate_den", pa.int32(), nullable=False),
        pa.field("expected_sample_support_seconds", pa.float64()),
        pa.field("expected_first_to_last_span_seconds", pa.float64()),
        pa.field("assessment_status", pa.string(), nullable=False),
        pa.field("exclusion_reason", pa.string()),
    ], "pads_assessments")


def pads_streams_schema() -> pa.Schema:
    """One row per device file, with the source declaration preserved."""

    return _schema([
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("assessment_id", pa.string(), nullable=False),
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("device_location", pa.string(), nullable=False),
        pa.field("source_asset_sha256", pa.string(), nullable=False),
        pa.field("source_relative_path", pa.string(), nullable=False),
        pa.field("declared_row_count", pa.int32(), nullable=False),
        pa.field("parsed_row_count", pa.int32(), nullable=False),
        pa.field(
            "source_channel_order", pa.list_(pa.string()), nullable=False
        ),
        pa.field(
            "source_units_order", pa.list_(pa.string()), nullable=False
        ),
        pa.field(
            "canonicalization_permutation",
            pa.list_(pa.int32()),
            nullable=False,
        ),
        pa.field("observed_median_interval_seconds", pa.float64()),
        pa.field("observed_first_to_last_span_seconds", pa.float64()),
        pa.field("stream_status", pa.string(), nullable=False),
        pa.field("issue_codes", pa.list_(pa.string()), nullable=False),
        pa.field("exclusion_reason", pa.string()),
    ], "pads_streams")


def pads_samples_schema() -> pa.Schema:
    """One row per source sample, in source order, canonically ordered."""

    return _schema([
        pa.field("stream_id", pa.string(), nullable=False),
        pa.field("sample_ordinal", pa.int32(), nullable=False),
        pa.field("source_time_token", pa.string(), nullable=False),
        pa.field("relative_time_seconds", pa.float64()),
        pa.field("accelerometer_x_g", pa.float64()),
        pa.field("accelerometer_y_g", pa.float64()),
        pa.field("accelerometer_z_g", pa.float64()),
        pa.field("gyroscope_x_rad_s", pa.float64()),
        pa.field("gyroscope_y_rad_s", pa.float64()),
        pa.field("gyroscope_z_rad_s", pa.float64()),
        pa.field("time_status", pa.string(), nullable=False),
        pa.field("sample_status", pa.string(), nullable=False),
        pa.field("source_asset_sha256", pa.string(), nullable=False),
        pa.field("parser_version", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ], "pads_samples")


PADS_TABLE_SCHEMAS: dict[str, Callable[[], pa.Schema]] = {
    "pads_participants": pads_participants_schema,
    "pads_assessments": pads_assessments_schema,
    "pads_streams": pads_streams_schema,
    "pads_samples": pads_samples_schema,
}

PADS_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "pads_participants": ("participant_id",),
    "pads_assessments": ("participant_id", "task_ordinal"),
    "pads_streams": ("participant_id", "assessment_id", "device_location"),
    "pads_samples": ("stream_id", "sample_ordinal"),
}


__all__ = [
    "PADS_SORT_KEYS",
    "PADS_TABLE_SCHEMAS",
    "PadsSchemaError",
    "pads_assessments_schema",
    "pads_participants_schema",
    "pads_samples_schema",
    "pads_streams_schema",
]
