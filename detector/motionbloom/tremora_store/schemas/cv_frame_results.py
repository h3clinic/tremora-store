"""Arrow schema for exactly one CV outcome per decoded frame."""

from __future__ import annotations

import pyarrow as pa

from . import _finalization_schema

INFERENCE_STATUSES = frozenset({
    "SUCCESS",
    "NO_DETECTION",
    "DECODE_FAILURE",
    "PREPROCESS_FAILURE",
    "INFERENCE_FAILURE",
    "REJECTED_INPUT",
})


def cv_frame_results_schema() -> pa.Schema:
    """Return the v0.3 per-frame inference-outcome schema."""

    return _finalization_schema(
        [
            pa.field("frame_id", pa.string(), nullable=False),
            pa.field("recording_id", pa.string(), nullable=False),
            pa.field("relative_pts_ns", pa.int64()),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("model_weights_sha256", pa.string(), nullable=False),
            pa.field(
                "preprocessing_config_sha256",
                pa.string(),
                nullable=False,
            ),
            pa.field("inference_environment_id", pa.string(), nullable=False),
            pa.field("inference_status", pa.string(), nullable=False),
            pa.field("detection_count", pa.int32(), nullable=False),
            pa.field("selected_detection_id", pa.string()),
            pa.field("runtime_ms", pa.float64()),
            pa.field("tracking_quality", pa.float32()),
            pa.field("frame_quality_bits", pa.uint32(), nullable=False),
        ],
        "cv_frame_results",
        extra_metadata={
            b"tremora.cardinality": b"exactly_one_row_per_video_frames.frame_id",
            b"tremora.runtime_semantics": (
                b"nullable_non_binding_observation_not_identity_input"
            ),
        },
    )


__all__ = ["INFERENCE_STATUSES", "cv_frame_results_schema"]
