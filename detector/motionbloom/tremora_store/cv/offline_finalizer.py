"""Bind deterministic offline CV inference to decoder-owned frame identities."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa

from ..decode.frame_identity import canonical_sha256
from ..decode.pts_decoder import DecodedFrame
from ..schemas.cv_detections import cv_detections_schema
from ..schemas.cv_frame_results import cv_frame_results_schema
from ..schemas.video_frames import video_frames_schema
from .coordinate_mapping import (
    CV_INPUT_PIXEL,
    DISPLAY_PIXEL,
    NORMALIZED_CV_INPUT,
    PreparedCVInput,
    canonical_cv_input_sha256,
    compose_transforms,
    identity_transform,
    invert_transform,
    map_bbox_xyxy,
    map_points,
    normalized_to_pixels,
)
from .pose_frame_contract import stable_detection_id

INFERENCE_STATUSES = frozenset({
    "SUCCESS",
    "NO_DETECTION",
    "DECODE_FAILURE",
    "PREPROCESS_FAILURE",
    "INFERENCE_FAILURE",
    "REJECTED_INPUT",
})
SELECTION_POLICY_ID = "highest-selection-score-then-content-hash-1.0.0"
MODEL_RELATIVE = "MODEL_RELATIVE"
CV_FAILURE_QUALITY_BIT = 1 << 5
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class OfflineFinalizationError(ValueError):
    """Raised when estimator output cannot be represented without guessing."""


@dataclass(frozen=True)
class EstimatorProvenance:
    model_id: str
    model_weights_sha256: str
    preprocessing_config_sha256: str
    inference_environment_id: str

    def __post_init__(self) -> None:
        for field in ("model_id", "inference_environment_id"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise OfflineFinalizationError(f"{field} must be a non-empty string")
        for field in ("model_weights_sha256", "preprocessing_config_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise OfflineFinalizationError(
                    f"{field} must be a lowercase SHA-256")


@dataclass(frozen=True)
class RawDetection:
    """One estimator detection before conversion to display-pixel space."""

    landmarks_xy: np.ndarray
    coordinate_space: str
    bbox_xyxy: tuple[float, float, float, float] | None = None
    landmarks_z_model: np.ndarray | None = None
    landmark_confidence: np.ndarray | None = None
    landmark_validity_mask: np.ndarray | None = None
    handedness: str | None = None
    handedness_confidence: float | None = None
    detection_confidence: float | None = None
    selection_score: float = 0.0


@dataclass(frozen=True)
class InferenceOutput:
    detections: tuple[RawDetection, ...] = ()
    tracking_quality: float | None = None


@runtime_checkable
class OfflineFrameEstimator(Protocol):
    """Minimal frame API; it receives identity but never creates numbering."""

    @property
    def provenance(self) -> EstimatorProvenance: ...

    def prepare(self, display_bgr: np.ndarray) -> PreparedCVInput: ...

    def infer(self, frame_id: str, cv_input_bgr: np.ndarray) -> InferenceOutput: ...


@dataclass(frozen=True)
class FinalizedFrameTables:
    video_frames: pa.Table
    cv_frame_results: pa.Table
    cv_detections: pa.Table

    def as_dict(self) -> dict[str, pa.Table]:
        return {
            "video_frames": self.video_frames,
            "cv_frame_results": self.cv_frame_results,
            "cv_detections": self.cv_detections,
        }


class IdentityPreprocessor:
    """Reusable no-resize BGR24 preprocessor for deterministic Gate-A tests."""

    def __call__(self, display_bgr: np.ndarray) -> PreparedCVInput:
        return PreparedCVInput(
            pixels=np.ascontiguousarray(display_bgr),
            display_to_cv_transform=identity_transform(),
        )


def _optional_probability(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise OfflineFinalizationError(f"{field} must be in [0,1] or null")
    return result


def _optional_finite(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise OfflineFinalizationError(f"{field} must be finite or null")
    return result


def _float_array(
    value: np.ndarray | None,
    shape: tuple[int, ...],
    field: str,
    *,
    nullable: bool,
) -> np.ndarray | None:
    if value is None:
        if nullable:
            return None
        raise OfflineFinalizationError(f"{field} is required")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise OfflineFinalizationError(
            f"{field} must be a finite array with shape {shape}")
    return result


def _canonical_detection(
    detection: RawDetection,
    *,
    prepared: PreparedCVInput,
) -> dict[str, object]:
    points = _float_array(
        detection.landmarks_xy, (21, 2), "landmarks_xy", nullable=False)
    assert points is not None
    normalized_points: np.ndarray | None = None
    if detection.coordinate_space == DISPLAY_PIXEL:
        display_points = points
        coordinate_to_display = identity_transform()
    elif detection.coordinate_space == CV_INPUT_PIXEL:
        coordinate_to_display = invert_transform(
            prepared.display_to_cv_transform)
        display_points = map_points(points, coordinate_to_display)
    elif detection.coordinate_space == NORMALIZED_CV_INPUT:
        normalized_points = points
        cv_points = normalized_to_pixels(
            points,
            width=int(prepared.pixels.shape[1]),
            height=int(prepared.pixels.shape[0]),
        )
        coordinate_to_display = invert_transform(
            prepared.display_to_cv_transform)
        display_points = map_points(cv_points, coordinate_to_display)
    else:
        raise OfflineFinalizationError(
            "estimator detection coordinate_space must be DISPLAY_PIXEL, "
            "CV_INPUT_PIXEL, or NORMALIZED_CV_INPUT")

    validity = detection.landmark_validity_mask
    if validity is None:
        valid_mask = np.ones(21, dtype=np.bool_)
    else:
        valid_mask = np.asarray(validity)
        if valid_mask.shape != (21,) or valid_mask.dtype.kind != "b":
            raise OfflineFinalizationError(
                "landmark_validity_mask must be a 21-element bool array")
        valid_mask = valid_mask.astype(np.bool_, copy=False)
    if not np.any(valid_mask):
        raise OfflineFinalizationError(
            "a detection must contain at least one valid landmark")

    z_values = _float_array(
        detection.landmarks_z_model,
        (21,),
        "landmarks_z_model",
        nullable=True,
    )
    confidences = _float_array(
        detection.landmark_confidence,
        (21,),
        "landmark_confidence",
        nullable=True,
    )
    if confidences is not None and (
        np.any(confidences < 0.0) or np.any(confidences > 1.0)
    ):
        raise OfflineFinalizationError(
            "landmark_confidence values must be in [0,1]")

    if detection.bbox_xyxy is None:
        selected = display_points[valid_mask]
        bbox_display = (
            float(np.min(selected[:, 0])), float(np.min(selected[:, 1])),
            float(np.max(selected[:, 0])), float(np.max(selected[:, 1])),
        )
    else:
        bbox = tuple(float(value) for value in detection.bbox_xyxy)
        if detection.coordinate_space == NORMALIZED_CV_INPUT:
            width = max(int(prepared.pixels.shape[1]) - 1, 0)
            height = max(int(prepared.pixels.shape[0]) - 1, 0)
            bbox = (
                bbox[0] * width, bbox[1] * height,
                bbox[2] * width, bbox[3] * height,
            )
        bbox_display = map_bbox_xyxy(bbox, coordinate_to_display)

    score = _optional_finite(detection.selection_score, "selection_score")
    assert score is not None
    handedness_confidence = _optional_probability(
        detection.handedness_confidence, "handedness_confidence")
    detection_confidence = _optional_probability(
        detection.detection_confidence, "detection_confidence")
    row: dict[str, object] = {
        "handedness": detection.handedness,
        "handedness_confidence": (
            None if handedness_confidence is None
            else float(np.float32(handedness_confidence))
        ),
        "detection_confidence": (
            None if detection_confidence is None
            else float(np.float32(detection_confidence))
        ),
        "bbox_xyxy_display": [
            float(value) for value in np.asarray(
                bbox_display, dtype=np.float32)
        ],
        "bbox_coordinate_space": DISPLAY_PIXEL,
        "landmarks_xy_display": [
            float(value) for value in display_points.astype(
                np.float32).reshape(-1)],
        "landmarks_xy_coordinate_space": DISPLAY_PIXEL,
        "landmarks_xy_normalized_cv_input": (
            [None] * 42 if normalized_points is None else [
                float(value) for value in normalized_points.astype(
                    np.float32).reshape(-1)
            ]
        ),
        "normalized_landmarks_coordinate_space": (
            None if normalized_points is None else NORMALIZED_CV_INPUT),
        "landmarks_z_model": [None] * 21 if z_values is None else [
            float(value) for value in z_values.astype(np.float32)],
        "landmarks_z_coordinate_space": None if z_values is None else MODEL_RELATIVE,
        "landmark_confidence": [None] * 21 if confidences is None else [
            float(value) for value in confidences.astype(np.float32)],
        "landmark_validity_mask": [bool(value) for value in valid_mask],
        "selection_score": float(np.float32(score)),
    }
    content = {
        key: value for key, value in row.items()
        if key not in {"selection_score"}
    }
    row["_sort_key"] = (
        -float(row["selection_score"]),
        canonical_sha256(content),
    )
    return row


def _frame_row(frame: object, *, dataset_id: str, recording_id: str) -> dict[str, object]:
    names = (
        "source_video_sha256", "stream_index", "frame_id", "identity_basis",
        "decode_ordinal", "presentation_ordinal", "pts", "time_base_num",
        "time_base_den", "relative_pts_ns", "same_pts_rank", "duration_pts",
        "duration_ns", "gap_before_ns", "coded_width", "coded_height",
        "display_width", "display_height", "rotation_degrees", "pixel_format",
        "key_frame", "picture_type", "pts_status", "decode_status",
        "quality_bits", "source_to_display_transform", "decoder_version",
        "schema_version",
    )
    row = {name: getattr(frame, name) for name in names}
    row.update({
        "dataset_id": dataset_id,
        "recording_id": recording_id,
        "display_to_cv_transform": None,
        "cv_to_source_transform": None,
        "preprocessing_transform_invertible": False,
        "cv_input_width": None,
        "cv_input_height": None,
        "cv_input_pixel_format": None,
        "cv_input_sha256": None,
    })
    return row


def finalize_frames(
    decoded_frames: Sequence[DecodedFrame],
    *,
    dataset_id: str,
    recording_id: str,
    finalization_id: str,
    estimator: OfflineFrameEstimator,
) -> FinalizedFrameTables:
    """Run one inference per successfully preprocessed decoded frame."""

    if not isinstance(dataset_id, str) or not dataset_id \
            or not isinstance(recording_id, str) or not recording_id:
        raise OfflineFinalizationError(
            "dataset_id and recording_id must be non-empty strings")
    if not isinstance(finalization_id, str) \
            or _SHA256_RE.fullmatch(finalization_id) is None:
        raise OfflineFinalizationError(
            "finalization_id must be a lowercase SHA-256")
    if not isinstance(estimator, OfflineFrameEstimator):
        raise OfflineFinalizationError(
            "estimator does not implement the offline frame protocol")
    provenance = estimator.provenance
    if not isinstance(provenance, EstimatorProvenance):
        raise OfflineFinalizationError(
            "estimator provenance must be EstimatorProvenance")

    frame_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    seen_frames: set[str] = set()

    for frame in decoded_frames:
        frame_id = frame.frame_id
        if frame_id in seen_frames:
            raise OfflineFinalizationError("decoder emitted a duplicate frame_id")
        seen_frames.add(frame_id)
        frame_row = _frame_row(
            frame, dataset_id=dataset_id, recording_id=recording_id)
        status = (
            "DECODE_FAILURE"
            if frame.decode_status != "SUCCESS"
            else "NO_DETECTION"
        )
        tracking_quality: float | None = None
        normalized: list[dict[str, object]] = []
        prepared = None
        if status != "DECODE_FAILURE":
            try:
                prepared = estimator.prepare(frame.display_bgr)
                if not isinstance(prepared, PreparedCVInput):
                    raise OfflineFinalizationError(
                        "estimator.prepare must return PreparedCVInput")
                # Own the exact estimator buffer so preprocessing cannot retain
                # an alias into decoder pixels. Start inference read-only and
                # verify the same bytes afterward in case an estimator forces
                # the NumPy write flag back on.
                pixels = np.array(
                    prepared.pixels,
                    dtype=np.uint8,
                    order="C",
                    copy=True,
                )
                pixels.setflags(write=False)
                prepared = PreparedCVInput(
                    pixels=pixels,
                    display_to_cv_transform=prepared.display_to_cv_transform,
                    pixel_format=prepared.pixel_format,
                )
                cv_input_sha256 = canonical_cv_input_sha256(prepared)
                source_to_cv = compose_transforms(
                    frame.source_to_display_transform,
                    prepared.display_to_cv_transform,
                )
                cv_to_source = invert_transform(source_to_cv)
                frame_row.update({
                    "display_to_cv_transform": list(
                        prepared.display_to_cv_transform),
                    "cv_to_source_transform": list(cv_to_source),
                    "preprocessing_transform_invertible": True,
                    "cv_input_width": int(pixels.shape[1]),
                    "cv_input_height": int(pixels.shape[0]),
                    "cv_input_pixel_format": prepared.pixel_format,
                    "cv_input_sha256": cv_input_sha256,
                })
            except Exception:  # noqa: BLE001 - retain per-frame preprocess failure
                status = "PREPROCESS_FAILURE"
                prepared = None

        if prepared is not None:
            try:
                output = estimator.infer(frame_id, prepared.pixels)
                if canonical_cv_input_sha256(prepared) != frame_row[
                    "cv_input_sha256"
                ]:
                    raise OfflineFinalizationError(
                        "estimator mutated the frozen CV input buffer")
                if not isinstance(output, InferenceOutput):
                    raise OfflineFinalizationError(
                        "estimator.infer must return InferenceOutput")
                tracking_quality = _optional_probability(
                    output.tracking_quality, "tracking_quality")
                normalized = [
                    _canonical_detection(detection, prepared=prepared)
                    for detection in output.detections
                ]
                normalized.sort(key=lambda row: row["_sort_key"])
                status = "SUCCESS" if normalized else "NO_DETECTION"
            except Exception:  # noqa: BLE001 - retain per-frame inference failure
                status = "INFERENCE_FAILURE"
                normalized = []
                tracking_quality = None

        selected_detection_id: str | None = None
        for rank, detection in enumerate(normalized):
            detection.pop("_sort_key", None)
            detection_id = stable_detection_id(
                finalization_id=finalization_id,
                frame_id=frame_id,
                detection_rank=rank,
            )
            selected = rank == 0
            if selected:
                selected_detection_id = detection_id
            detection_rows.append({
                "detection_id": detection_id,
                "frame_id": frame_id,
                "detection_rank": rank,
                **detection,
                "selected_for_primary_track": selected,
            })

        frame_quality = int(frame.quality_bits)
        if status in {"PREPROCESS_FAILURE", "INFERENCE_FAILURE", "REJECTED_INPUT"}:
            frame_quality |= CV_FAILURE_QUALITY_BIT
        frame_rows.append(frame_row)
        result_rows.append({
            "frame_id": frame_id,
            "recording_id": recording_id,
            "relative_pts_ns": frame.relative_pts_ns,
            "model_id": provenance.model_id,
            "model_weights_sha256": provenance.model_weights_sha256,
            "preprocessing_config_sha256": provenance.preprocessing_config_sha256,
            "inference_environment_id": provenance.inference_environment_id,
            "inference_status": status,
            "detection_count": len(normalized),
            "selected_detection_id": selected_detection_id,
            # Wall time is deliberately outside the deterministic artifact.
            "runtime_ms": None,
            "tracking_quality": tracking_quality,
            "frame_quality_bits": frame_quality,
        })

    frame_table = pa.Table.from_pylist(frame_rows, schema=video_frames_schema())
    result_table = pa.Table.from_pylist(
        result_rows, schema=cv_frame_results_schema())
    detection_table = pa.Table.from_pylist(
        detection_rows, schema=cv_detections_schema())
    return FinalizedFrameTables(
        video_frames=frame_table,
        cv_frame_results=result_table,
        cv_detections=detection_table,
    )
