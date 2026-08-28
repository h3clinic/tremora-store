"""Canonical Arrow schemas for a TremoraStore snapshot.

The stream, epoch and ordinal keys are part of the storage contract. Native
timestamps alone cannot identify a sample after a clock reset, and bare sample
indexes cannot identify a range in a recording with multiple IMUs or views.
"""

from __future__ import annotations

import hashlib
import json
from enum import IntFlag

import pyarrow as pa

SCHEMA_VERSION = "0.1.0"


class QualityBits(IntFlag):
    """Cross-table quality flags; zero means no condition was asserted."""

    NONE = 0
    MISSING_TIMESTAMP = 1 << 0
    NON_MONOTONIC_TIMESTAMP = 1 << 1
    DUPLICATE_TIMESTAMP = 1 << 2
    STREAM_GAP = 1 << 3
    DECODE_FAILURE = 1 << 4
    INVALID_CV = 1 << 5
    CLOCK_RESET = 1 << 6
    UNRESOLVED_CLOCK_MAP = 1 << 7
    PARTIAL_COVERAGE = 1 << 8
    FREQUENCY_UNSUPPORTED = 1 << 9
    SYNC_RESIDUAL_EXCEEDED = 1 << 10
    INVALID_IMU_PAYLOAD = 1 << 11


def _schema(fields: list[pa.Field], table_name: str) -> pa.Schema:
    metadata = {
        b"tremora.schema_version": SCHEMA_VERSION.encode(),
        b"tremora.table": table_name.encode(),
        b"tremora.interval_semantics": b"half-open [start_ns,end_ns)",
    }
    return pa.schema(
        fields,
        metadata=metadata,
    )


def _metadata_contract(metadata: dict[bytes, bytes] | None) -> list[tuple[str, str]]:
    return sorted(
        (key.decode("utf-8", "surrogateescape"),
         value.decode("utf-8", "surrogateescape"))
        for key, value in (metadata or {}).items()
    )


def _field_contract(field: pa.Field, *, include_name: bool) -> dict[str, object]:
    contract: dict[str, object] = {
        "type": logical_type_contract(field.type),
        "nullable": field.nullable,
        "metadata": _metadata_contract(field.metadata),
    }
    if include_name:
        contract["name"] = field.name
    return contract


def logical_type_contract(data_type: pa.DataType) -> object:
    """Normalize Arrow types across lossless Parquet round trips.

    Arrow may rename a fixed-list child from ``item`` to ``element`` when it
    reads Parquet. That name is not part of Tremora's logical contract.
    """

    if pa.types.is_fixed_size_list(data_type):
        return {"kind": "fixed_size_list", "size": data_type.list_size,
                "value": _field_contract(
                    data_type.value_field, include_name=False)}
    if pa.types.is_list(data_type):
        return {"kind": "list", "value": _field_contract(
            data_type.value_field, include_name=False)}
    if pa.types.is_large_list(data_type):
        return {"kind": "large_list", "value": _field_contract(
            data_type.value_field, include_name=False)}
    return str(data_type)


def logical_schema_contract(schema: pa.Schema) -> dict[str, object]:
    return {
        "fields": [
            _field_contract(field, include_name=True) for field in schema],
        "metadata": _metadata_contract(schema.metadata),
    }


def schema_fingerprint(schema: pa.Schema) -> str:
    """Hash the canonical logical schema, including Tremora metadata."""

    payload = json.dumps(logical_schema_contract(schema), sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def frame_index_schema() -> pa.Schema:
    return _schema(
        [
            pa.field("recording_id", pa.string(), nullable=False),
            pa.field("video_stream_id", pa.string(), nullable=False),
            pa.field("clock_epoch_id", pa.string(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("source_ordinal", pa.int64(), nullable=False),
            pa.field("canonical_ordinal", pa.int64(), nullable=False),
            pa.field("video_pts_native_ns", pa.int64(), nullable=False),
            pa.field("canonical_time_ns", pa.int64(), nullable=False),
            pa.field("decode_status", pa.string(), nullable=False),
            pa.field("width", pa.int32(), nullable=False),
            pa.field("height", pa.int32(), nullable=False),
            pa.field("effective_fps", pa.float64()),
            pa.field("gap_before_ms", pa.float64()),
            pa.field("quality_bits", pa.uint32(), nullable=False),
        ],
        "frame_index",
    )


def cv_estimates_schema(
    *, keypoint_count: int = 21, keypoint_dimensions: int = 3,
    motion_vector_size: int = 3,
) -> pa.Schema:
    dimensions = (keypoint_count, keypoint_dimensions, motion_vector_size)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in dimensions
    ):
        raise ValueError(
            "fixed-size CV array dimensions must be positive integers")
    return _schema(
        [
            pa.field("recording_id", pa.string(), nullable=False),
            pa.field("video_stream_id", pa.string(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("canonical_ordinal", pa.int64(), nullable=False),
            pa.field("canonical_time_ns", pa.int64(), nullable=False),
            pa.field(
                "keypoints",
                pa.list_(pa.float32(), list_size=keypoint_count * keypoint_dimensions),
            ),
            pa.field(
                "keypoint_validity",
                pa.list_(pa.bool_(), list_size=keypoint_count),
            ),
            pa.field(
                "motion_vector",
                pa.list_(pa.float32(), list_size=motion_vector_size),
            ),
            pa.field("palm_orientation", pa.list_(pa.float32(), list_size=4)),
            pa.field("hand_scale", pa.float32()),
            pa.field("estimated_frequency_hz", pa.float32()),
            pa.field("tracking_quality", pa.float32()),
            pa.field("estimator_version", pa.string(), nullable=False),
        ],
        "cv_estimates",
    )


def imu_samples_schema() -> pa.Schema:
    """Raw inertial payload with honest nullable quaternion support.

    VIDIMU supplies 50 Hz quaternions, not accelerometer/gyroscope XYZ. Those
    channels remain null for VIDIMU and are never synthesized from orientation.
    """

    return _schema(
        [
            pa.field("recording_id", pa.string(), nullable=False),
            pa.field("stream_id", pa.string(), nullable=False),
            pa.field("clock_epoch_id", pa.string(), nullable=False),
            pa.field("sample_index", pa.int64(), nullable=False),
            pa.field("source_ordinal", pa.int64(), nullable=False),
            pa.field("canonical_ordinal", pa.int64(), nullable=False),
            pa.field("sensor_time_native_ns", pa.int64(), nullable=False),
            pa.field("canonical_time_ns", pa.int64(), nullable=False),
            pa.field("payload_kind", pa.string(), nullable=False),
            pa.field("ax", pa.float64()),
            pa.field("ay", pa.float64()),
            pa.field("az", pa.float64()),
            pa.field("gx", pa.float64()),
            pa.field("gy", pa.float64()),
            pa.field("gz", pa.float64()),
            pa.field("qw", pa.float64()),
            pa.field("qx", pa.float64()),
            pa.field("qy", pa.float64()),
            pa.field("qz", pa.float64()),
            pa.field("validity_bits", pa.uint32(), nullable=False),
        ],
        "imu_samples",
    )


def clock_map_schema() -> pa.Schema:
    return _schema(
        [
            pa.field("recording_id", pa.string(), nullable=False),
            pa.field("stream_id", pa.string(), nullable=False),
            pa.field("clock_epoch_id", pa.string(), nullable=False),
            pa.field("continuity_component_id", pa.string(), nullable=False),
            pa.field("acquisition_ordinal", pa.int32(), nullable=False),
            pa.field("source_start_ordinal", pa.int64(), nullable=False),
            pa.field("source_stop_ordinal", pa.int64(), nullable=False),
            pa.field("native_start_ns", pa.int64(), nullable=False),
            pa.field("native_end_ns", pa.int64(), nullable=False),
            pa.field("native_anchor_ns", pa.int64(), nullable=False),
            pa.field("canonical_anchor_ns", pa.int64(), nullable=False),
            pa.field("scale_numerator", pa.int64(), nullable=False),
            pa.field("scale_denominator", pa.int64(), nullable=False),
            pa.field("drift_ppm_derived", pa.float64(), nullable=False),
            pa.field("residual_p50_ms", pa.float64()),
            pa.field("residual_p95_ms", pa.float64()),
            pa.field("mapping_status", pa.string(), nullable=False),
        ],
        "clock_map",
    )


def frame_imu_index_schema() -> pa.Schema:
    return _schema(
        [
            pa.field("recording_id", pa.string(), nullable=False),
            pa.field("video_stream_id", pa.string(), nullable=False),
            pa.field("imu_stream_id", pa.string(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("frame_canonical_ordinal", pa.int64(), nullable=False),
            pa.field("frame_time_ns", pa.int64(), nullable=False),
            pa.field("frame_interval_end_ns", pa.int64(), nullable=False),
            pa.field("imu_start_ordinal", pa.int64(), nullable=False),
            pa.field("imu_stop_ordinal", pa.int64(), nullable=False),
            pa.field("imu_nearest_ordinal", pa.int64()),
            pa.field("nearest_delta_ns", pa.int64()),
            pa.field("imu_sample_count", pa.int32(), nullable=False),
            pa.field("imu_coverage_fraction", pa.float64(), nullable=False),
            pa.field("alignment_status", pa.string(), nullable=False),
        ],
        "frame_imu_index",
    )


def window_index_schema() -> pa.Schema:
    return _schema(
        [
            pa.field("window_id", pa.string(), nullable=False),
            pa.field("recording_id", pa.string(), nullable=False),
            pa.field("video_stream_id", pa.string(), nullable=False),
            pa.field("imu_stream_id", pa.string(), nullable=False),
            pa.field("continuity_segment_id", pa.string(), nullable=False),
            pa.field("window_policy_id", pa.string(), nullable=False),
            pa.field("observability_policy_id", pa.string(), nullable=False),
            pa.field("start_time_ns", pa.int64(), nullable=False),
            pa.field("end_time_ns", pa.int64(), nullable=False),
            pa.field("frame_start_ordinal", pa.int64(), nullable=False),
            pa.field("frame_stop_ordinal", pa.int64(), nullable=False),
            pa.field("imu_start_ordinal", pa.int64(), nullable=False),
            pa.field("imu_stop_ordinal", pa.int64(), nullable=False),
            pa.field("frame_count", pa.int32(), nullable=False),
            pa.field("imu_sample_count", pa.int32(), nullable=False),
            pa.field("effective_video_fps", pa.float64(), nullable=False),
            pa.field("effective_imu_hz", pa.float64(), nullable=False),
            pa.field("video_coverage", pa.float64(), nullable=False),
            pa.field("imu_coverage", pa.float64(), nullable=False),
            pa.field("video_cadence_regular", pa.bool_(), nullable=False),
            pa.field("imu_cadence_regular", pa.bool_(), nullable=False),
            pa.field("video_max_cadence_deviation_fraction", pa.float64()),
            pa.field("imu_max_cadence_deviation_fraction", pa.float64()),
            pa.field("video_rate_based_nyquist_hz", pa.float64()),
            pa.field("imu_rate_based_nyquist_hz", pa.float64()),
            pa.field("video_observable_max_hz", pa.float64(), nullable=False),
            pa.field("imu_observable_max_hz", pa.float64(), nullable=False),
            pa.field("tremor_band_supported", pa.bool_(), nullable=False),
            pa.field("cv_tracking_valid", pa.bool_(), nullable=False),
            pa.field("cv_motion_range_gate_passed", pa.bool_(), nullable=False),
            pa.field("imu_signal_range_gate_passed", pa.bool_(), nullable=False),
            pa.field("frequency_estimation_allowed", pa.bool_(), nullable=False),
            pa.field("quality_bits", pa.uint32(), nullable=False),
            pa.field("valid_for_frequency", pa.bool_(), nullable=False),
            pa.field("split_group_id", pa.string(), nullable=False),
        ],
        "window_index",
    )


def window_rejections_schema() -> pa.Schema:
    return _schema(
        [
            pa.field("candidate_window_id", pa.string(), nullable=False),
            pa.field("recording_id", pa.string(), nullable=False),
            pa.field("video_stream_id", pa.string(), nullable=False),
            pa.field("imu_stream_id", pa.string(), nullable=False),
            pa.field("continuity_segment_id", pa.string(), nullable=False),
            pa.field("window_policy_id", pa.string(), nullable=False),
            pa.field("start_time_ns", pa.int64(), nullable=False),
            pa.field("end_time_ns", pa.int64(), nullable=False),
            pa.field("reason_bits", pa.uint32(), nullable=False),
            pa.field("reason_codes", pa.string(), nullable=False),
        ],
        "window_rejections",
    )


SCHEMA_FACTORIES = {
    "frame_index": frame_index_schema,
    "cv_estimates": cv_estimates_schema,
    "imu_samples": imu_samples_schema,
    "clock_map": clock_map_schema,
    "frame_imu_index": frame_imu_index_schema,
    "window_index": window_index_schema,
    "window_rejections": window_rejections_schema,
}
