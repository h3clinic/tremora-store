"""Arrow schema for source-PTS-preserving decoded video frames."""

from __future__ import annotations

import pyarrow as pa

from . import HOMOGRAPHY_ELEMENT_COUNT, _finalization_schema, _fixed_size_list

PTS_STATUSES = frozenset({
    "VALID",
    "DUPLICATE",
    "NON_MONOTONIC",
    "MISSING",
    "DISCONTINUITY",
})

DECODE_STATUSES = frozenset({
    "SUCCESS",
    "CORRUPT",
})

FRAME_IDENTITY_BASES = frozenset({
    "SOURCE_PTS_SAME_PTS_RANK",
    "MISSING_PTS_DECODE_ORDINAL",
})


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


def video_frames_schema() -> pa.Schema:
    """Return the v0.3 one-row-per-decoder-emitted-frame schema.

    PTS-derived fields remain nullable so missing-PTS evidence can be retained,
    but every emitted frame retains a non-null rational time base. A missing
    PTS is never replaced with nominal-FPS-derived time. ``CORRUPT`` denotes an
    emitted frame carrying decoder corruption evidence; a hard decoder failure
    aborts atomic finalization and therefore has no frame row.
    """

    return _finalization_schema(
        [
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
            pa.field("schema_version", pa.string(), nullable=False),
        ],
        "video_frames",
        extra_metadata={
            b"tremora.frame_identity": (
                b"source_video_sha256,stream_index,pts,same_pts_rank"
            ),
            b"tremora.missing_pts_identity": (
                b"source_video_sha256,stream_index,decode_ordinal"
            ),
            b"tremora.presentation_origin": (
                b"first_accepted_presentation_frame"
            ),
        },
    )


__all__ = [
    "DECODE_STATUSES",
    "FRAME_IDENTITY_BASES",
    "PTS_STATUSES",
    "video_frames_schema",
]
