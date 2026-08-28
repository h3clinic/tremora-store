"""Arrow schema for zero-to-many hand detections bound to each frame."""

from __future__ import annotations

import pyarrow as pa

from . import HAND_LANDMARK_COUNT, _finalization_schema, _fixed_size_list

LANDMARK_XY_COORDINATE_SPACE = "DISPLAY_PIXEL"
LANDMARK_Z_COORDINATE_SPACE = "MODEL_RELATIVE"
BBOX_COORDINATE_SPACE = "DISPLAY_PIXEL"


def cv_detections_schema() -> pa.Schema:
    """Return the fixed-size v0.3 hand-detection schema.

    Landmark arrays are flattened in landmark-major order: ``x0,y0,x1,y1``
    for XY and ``z0,z1`` for model-relative depth.
    """

    return _finalization_schema(
        [
            pa.field("detection_id", pa.string(), nullable=False),
            pa.field("frame_id", pa.string(), nullable=False),
            pa.field("detection_rank", pa.int32(), nullable=False),
            pa.field("handedness", pa.string()),
            pa.field("handedness_confidence", pa.float32()),
            pa.field("detection_confidence", pa.float32()),
            pa.field(
                "bbox_xyxy_display",
                _fixed_size_list(pa.float32(), 4),
                nullable=False,
                metadata={
                    b"tremora.coordinate_space": b"DISPLAY_PIXEL",
                    b"tremora.layout": b"xmin,ymin,xmax,ymax",
                },
            ),
            pa.field("bbox_coordinate_space", pa.string(), nullable=False),
            pa.field(
                "landmarks_xy_display",
                _fixed_size_list(pa.float32(), HAND_LANDMARK_COUNT * 2),
                nullable=False,
                metadata={
                    b"tremora.coordinate_space": b"DISPLAY_PIXEL",
                    b"tremora.layout": b"x0,y0,x1,y1,...",
                    b"tremora.landmark_count": str(
                        HAND_LANDMARK_COUNT
                    ).encode("ascii"),
                },
            ),
            pa.field(
                "landmarks_xy_coordinate_space",
                pa.string(),
                nullable=False,
            ),
            pa.field(
                "landmarks_xy_normalized_cv_input",
                _fixed_size_list(
                    pa.float32(),
                    HAND_LANDMARK_COUNT * 2,
                    elements_nullable=True,
                ),
                nullable=False,
                metadata={
                    b"tremora.coordinate_space": b"NORMALIZED_CV_INPUT",
                    b"tremora.layout": b"x0,y0,x1,y1,...",
                    b"tremora.landmark_count": str(
                        HAND_LANDMARK_COUNT
                    ).encode("ascii"),
                },
            ),
            pa.field(
                "normalized_landmarks_coordinate_space",
                pa.string(),
            ),
            pa.field(
                "landmarks_z_model",
                _fixed_size_list(
                    pa.float32(),
                    HAND_LANDMARK_COUNT,
                    elements_nullable=True,
                ),
                nullable=False,
                metadata={
                    b"tremora.coordinate_space": b"MODEL_RELATIVE",
                    b"tremora.layout": b"z0,z1,...",
                    b"tremora.landmark_count": str(
                        HAND_LANDMARK_COUNT
                    ).encode("ascii"),
                },
            ),
            pa.field(
                "landmarks_z_coordinate_space",
                pa.string(),
            ),
            pa.field(
                "landmark_confidence",
                _fixed_size_list(
                    pa.float32(),
                    HAND_LANDMARK_COUNT,
                    elements_nullable=True,
                ),
                nullable=False,
            ),
            pa.field(
                "landmark_validity_mask",
                _fixed_size_list(pa.bool_(), HAND_LANDMARK_COUNT),
                nullable=False,
            ),
            pa.field("selection_score", pa.float64(), nullable=False),
            pa.field("selected_for_primary_track", pa.bool_(), nullable=False),
        ],
        "cv_detections",
        extra_metadata={
            b"tremora.cardinality": b"zero_or_more_rows_per_frame_id",
            b"tremora.detection_rank": b"zero_based_deterministic_per_frame",
        },
    )


__all__ = [
    "BBOX_COORDINATE_SPACE",
    "LANDMARK_XY_COORDINATE_SPACE",
    "LANDMARK_Z_COORDINATE_SPACE",
    "cv_detections_schema",
]
