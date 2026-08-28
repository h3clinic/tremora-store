"""Arrow schemas for v0.4 frame inference and separate hand selection."""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

V04_SCHEMA_VERSION = "0.4.0"
V04_ASSOCIATION_CONTRACT_VERSION = "tremora-pose-frame-association-1.0.0"
PRIMARY_HAND_SELECTION_CONTRACT_VERSION = (
    "tremora-primary-hand-unambiguous-only-0.4.0"
)
HAND_LANDMARK_COUNT = 21
HOMOGRAPHY_ELEMENT_COUNT = 9


def _fixed_size_list(
    value_type: pa.DataType,
    size: int,
    *,
    elements_nullable: bool = False,
) -> pa.FixedSizeListType:
    return pa.list_(
        pa.field("item", value_type, nullable=elements_nullable),
        list_size=size,
    )


def _schema(
    fields: list[pa.Field],
    table_name: str,
    *,
    extra_metadata: dict[bytes, bytes] | None = None,
) -> pa.Schema:
    metadata = {
        b"tremora.schema_version": V04_SCHEMA_VERSION.encode("ascii"),
        b"tremora.table": table_name.encode("ascii"),
        b"tremora.association_schema_version": (
            V04_ASSOCIATION_CONTRACT_VERSION.encode("ascii")
        ),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return pa.schema(fields, metadata=metadata)


def _transform_field(
    name: str,
    *,
    input_space: str,
    output_space: str,
    nullable: bool,
) -> pa.Field:
    return pa.field(
        name,
        _fixed_size_list(pa.float64(), HOMOGRAPHY_ELEMENT_COUNT),
        nullable=nullable,
        metadata={
            b"tremora.matrix_convention": b"row_major_homogeneous_3x3",
            b"tremora.input_coordinate_space": input_space.encode("ascii"),
            b"tremora.output_coordinate_space": output_space.encode("ascii"),
        },
    )


def video_frames_v04_schema() -> pa.Schema:
    """One source-PTS-preserving row per v0.3 decoder-emitted frame."""

    return _schema([
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("source_video_sha256", pa.string(), nullable=False),
        pa.field("stream_index", pa.int32(), nullable=False),
        pa.field("frame_id", pa.string(), nullable=False),
        pa.field("identity_basis", pa.string(), nullable=False),
        pa.field("decode_ordinal", pa.int64(), nullable=False),
        pa.field("presentation_ordinal", pa.int64()),
        pa.field("pts", pa.int64()),
        pa.field("time_base_num", pa.int64(), nullable=False),
        pa.field("time_base_den", pa.int64(), nullable=False),
        pa.field("relative_pts_ns", pa.int64()),
        pa.field("same_pts_rank", pa.int32(), nullable=False),
        pa.field("duration_pts", pa.int64()),
        pa.field("duration_ns", pa.int64()),
        pa.field("gap_before_ns", pa.int64()),
        pa.field("coded_width", pa.int32(), nullable=False),
        pa.field("coded_height", pa.int32(), nullable=False),
        pa.field("display_width", pa.int32(), nullable=False),
        pa.field("display_height", pa.int32(), nullable=False),
        pa.field("rotation_degrees", pa.int16(), nullable=False),
        pa.field("pixel_format", pa.string(), nullable=False),
        pa.field("key_frame", pa.bool_()),
        pa.field("picture_type", pa.string()),
        pa.field("pts_status", pa.string(), nullable=False),
        pa.field("decode_status", pa.string(), nullable=False),
        pa.field("quality_bits", pa.uint32(), nullable=False),
        _transform_field(
            "source_to_display_transform",
            input_space="SOURCE_PIXEL",
            output_space="DISPLAY_PIXEL",
            nullable=False,
        ),
        _transform_field(
            "display_to_cv_transform",
            input_space="DISPLAY_PIXEL",
            output_space="CV_INPUT_PIXEL",
            nullable=True,
        ),
        _transform_field(
            "cv_to_source_transform",
            input_space="CV_INPUT_PIXEL",
            output_space="SOURCE_PIXEL",
            nullable=True,
        ),
        pa.field(
            "preprocessing_transform_invertible",
            pa.bool_(),
            nullable=False,
        ),
        pa.field("cv_input_width", pa.int32()),
        pa.field("cv_input_height", pa.int32()),
        pa.field("cv_input_pixel_format", pa.string()),
        pa.field("cv_input_sha256", pa.string()),
        pa.field("decoder_version", pa.string(), nullable=False),
        # This is the decoder-emitted schema version, not the surrounding
        # v0.4 table version carried in Arrow schema metadata.
        pa.field("schema_version", pa.string(), nullable=False),
    ], "video_frames", extra_metadata={
        b"tremora.cardinality": b"exactly_one_row_per_decoder_emitted_frame",
        b"tremora.frame_identity": (
            b"source_video_sha256,stream_index,pts,same_pts_rank"
        ),
        b"tremora.missing_pts_identity": (
            b"source_video_sha256,stream_index,decode_ordinal"
        ),
        b"tremora.presentation_origin": b"first_accepted_presentation_frame",
        b"tremora.timing_authority": b"source_pts_and_time_base",
    })


def cv_frame_results_v04_schema() -> pa.Schema:
    """Exactly one explicit production-CV outcome per decoded frame."""

    return _schema([
        pa.field("frame_id", pa.string(), nullable=False),
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("relative_pts_ns", pa.int64()),
        pa.field("model_manifest_sha256", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_weights_sha256", pa.string(), nullable=False),
        pa.field(
            "preprocessing_config_sha256",
            pa.string(),
            nullable=False,
        ),
        pa.field("runtime_lock_sha256", pa.string(), nullable=False),
        pa.field("association_contract_version", pa.string(), nullable=False),
        pa.field("inference_environment_id", pa.string(), nullable=False),
        pa.field("inference_status", pa.string(), nullable=False),
        pa.field("detection_count", pa.int32(), nullable=False),
        pa.field("failure_reason", pa.string()),
        _transform_field(
            "preprocessing_transform",
            input_space="DISPLAY_PIXEL",
            output_space="CV_INPUT_PIXEL",
            nullable=True,
        ),
        pa.field("runtime_metadata_json", pa.string()),
        # Wall-clock timing remains deliberately non-binding and nullable.
        pa.field("runtime_ms", pa.float64()),
        pa.field("tracking_quality", pa.float32()),
        pa.field("frame_quality_bits", pa.uint32(), nullable=False),
    ], "cv_frame_results", extra_metadata={
        b"tremora.cardinality": b"exactly_one_row_per_video_frames.frame_id",
        b"tremora.primary_selection": b"separate_primary_hand_selection_table",
        b"tremora.runtime_metadata": b"closed_canonical_json",
        b"tremora.runtime_semantics": (
            b"nullable_non_binding_observation_not_identity_input"
        ),
        b"tremora.timing_authority": b"source_pts_and_time_base",
    })


def cv_detections_v04_schema() -> pa.Schema:
    """All detector outputs; no primary-hand selection fields are embedded."""

    return _schema([
        pa.field("detection_id", pa.string(), nullable=False),
        pa.field("frame_id", pa.string(), nullable=False),
        pa.field("model_manifest_sha256", pa.string(), nullable=False),
        pa.field("detection_rank", pa.int32(), nullable=False),
        pa.field("same_payload_rank", pa.int32(), nullable=False),
        pa.field("handedness", pa.string()),
        pa.field("handedness_confidence", pa.float32()),
        pa.field("detection_confidence", pa.float32()),
        pa.field(
            "bbox_xyxy_display",
            _fixed_size_list(pa.float32(), 4),
            nullable=False,
            metadata={b"tremora.coordinate_space": b"DISPLAY_PIXEL"},
        ),
        pa.field(
            "landmarks_xy_display",
            _fixed_size_list(pa.float32(), HAND_LANDMARK_COUNT * 2),
            nullable=False,
            metadata={b"tremora.coordinate_space": b"DISPLAY_PIXEL"},
        ),
        pa.field(
            "landmarks_z_model",
            _fixed_size_list(
                pa.float32(), HAND_LANDMARK_COUNT, elements_nullable=True
            ),
            nullable=False,
            metadata={b"tremora.coordinate_space": b"MODEL_RELATIVE"},
        ),
        pa.field(
            "landmark_confidence",
            _fixed_size_list(
                pa.float32(), HAND_LANDMARK_COUNT, elements_nullable=True
            ),
            nullable=False,
        ),
        pa.field(
            "landmark_validity_mask",
            _fixed_size_list(pa.bool_(), HAND_LANDMARK_COUNT),
            nullable=False,
        ),
    ], "cv_detections")


def primary_hand_selection_schema() -> pa.Schema:
    """One independently versioned primary-selection outcome per frame."""

    return _schema([
        pa.field("frame_id", pa.string(), nullable=False),
        pa.field(
            "primary_hand_selection_contract_version",
            pa.string(),
            nullable=False,
        ),
        pa.field("inference_status", pa.string(), nullable=False),
        pa.field("selection_status", pa.string(), nullable=False),
        pa.field("selected_detection_id", pa.string()),
        pa.field("selection_reason", pa.string(), nullable=False),
    ], "primary_hand_selection")


V04_TABLE_SCHEMAS: dict[str, Callable[[], pa.Schema]] = {
    "video_frames": video_frames_v04_schema,
    "cv_frame_results": cv_frame_results_v04_schema,
    "cv_detections": cv_detections_v04_schema,
    "primary_hand_selection": primary_hand_selection_schema,
}

V04_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "video_frames": (
        "recording_id",
        "stream_index",
        "presentation_ordinal",
        "decode_ordinal",
    ),
    "cv_frame_results": ("recording_id", "relative_pts_ns", "frame_id"),
    "cv_detections": ("frame_id", "detection_rank"),
    "primary_hand_selection": ("frame_id",),
}


__all__ = [
    "HAND_LANDMARK_COUNT",
    "HOMOGRAPHY_ELEMENT_COUNT",
    "PRIMARY_HAND_SELECTION_CONTRACT_VERSION",
    "V04_ASSOCIATION_CONTRACT_VERSION",
    "V04_SCHEMA_VERSION",
    "V04_SORT_KEYS",
    "V04_TABLE_SCHEMAS",
    "cv_detections_v04_schema",
    "cv_frame_results_v04_schema",
    "primary_hand_selection_schema",
    "video_frames_v04_schema",
]
