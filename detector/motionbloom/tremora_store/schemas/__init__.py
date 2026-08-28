"""Arrow schemas for the immutable VIDIMU PTS/CV finalization artifact.

These tables are deliberately separate from the canonical cross-modal
TremoraStore schemas.  Version 0.3 records source video timing and binds CV
outputs to decoder-created frame identities; it does not define an IMU clock
map or a canonical cross-modal timeline.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

FINALIZATION_SCHEMA_VERSION = "0.3.0"
ASSOCIATION_SCHEMA_VERSION = "tremora-pose-frame-association-1.0.0"
HAND_LANDMARK_COUNT = 21
HOMOGRAPHY_ELEMENT_COUNT = 9


def _fixed_size_list(
    value_type: pa.DataType,
    size: int,
    *,
    elements_nullable: bool = False,
) -> pa.FixedSizeListType:
    """Return a fixed-size list with an explicit child-nullability contract."""

    return pa.list_(
        pa.field("item", value_type, nullable=elements_nullable),
        list_size=size,
    )


def _finalization_schema(
    fields: list[pa.Field],
    table_name: str,
    *,
    extra_metadata: dict[bytes, bytes] | None = None,
) -> pa.Schema:
    metadata = {
        b"tremora.schema_version": FINALIZATION_SCHEMA_VERSION.encode("ascii"),
        b"tremora.table": table_name.encode("ascii"),
        b"tremora.association_schema_version": (
            ASSOCIATION_SCHEMA_VERSION.encode("ascii")
        ),
        b"tremora.timing_authority": b"source_pts_and_time_base",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return pa.schema(fields, metadata=metadata)


from .cv_detections import cv_detections_schema
from .cv_frame_results import cv_frame_results_schema
from .video_frames import video_frames_schema

FINALIZATION_TABLE_SCHEMAS: dict[str, Callable[[], pa.Schema]] = {
    "video_frames": video_frames_schema,
    "cv_frame_results": cv_frame_results_schema,
    "cv_detections": cv_detections_schema,
}

FINALIZATION_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "video_frames": (
        "recording_id",
        "stream_index",
        "presentation_ordinal",
        "decode_ordinal",
    ),
    "cv_frame_results": ("recording_id", "relative_pts_ns", "frame_id"),
    "cv_detections": ("frame_id", "detection_rank"),
}

__all__ = [
    "ASSOCIATION_SCHEMA_VERSION",
    "FINALIZATION_SCHEMA_VERSION",
    "FINALIZATION_SORT_KEYS",
    "FINALIZATION_TABLE_SCHEMAS",
    "HAND_LANDMARK_COUNT",
    "HOMOGRAPHY_ELEMENT_COUNT",
    "cv_detections_schema",
    "cv_frame_results_schema",
    "video_frames_schema",
]
