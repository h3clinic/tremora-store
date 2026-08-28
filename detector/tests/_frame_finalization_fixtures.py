"""Deterministic CV fixtures for PTS/frame-association Gate-A tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from motionbloom.tremora_store.cv.coordinate_mapping import (
    DISPLAY_PIXEL,
    PreparedCVInput,
    canonical_cv_input_sha256,
    identity_transform,
)
from motionbloom.tremora_store.cv.offline_finalizer import (
    EstimatorProvenance,
    InferenceOutput,
    RawDetection,
)

DEFAULT_ESTIMATOR_PROVENANCE = EstimatorProvenance(
    model_id="gate-a-deterministic-pose-v1",
    model_weights_sha256="1" * 64,
    preprocessing_config_sha256="2" * 64,
    inference_environment_id="gate-a-cpu-software-fixture-v1",
)
PIXEL_MARKER_ESTIMATOR_PROVENANCE = EstimatorProvenance(
    model_id="gate-a-pixel-marker-estimator-v1",
    model_weights_sha256=hashlib.sha256(
        b"NO_WEIGHTS:gate-a-pixel-marker-estimator-v1"
    ).hexdigest(),
    preprocessing_config_sha256=hashlib.sha256(
        b"identity-bgr24-preprocessing-v1"
    ).hexdigest(),
    inference_environment_id="gate-a-cpu-numpy-pixel-fixture-v1",
)


class FatalInferenceInterruption(BaseException):
    """Simulate a process-level interruption that must abort publication."""


def fixture_detection(
    marker: int,
    *,
    selection_score: float | None = None,
) -> RawDetection:
    """Return one finite, visibly distinct 21-landmark hand detection."""

    x = np.linspace(3.0, 23.0, 21, dtype=np.float64) + marker
    y = np.linspace(4.0, 24.0, 21, dtype=np.float64) + marker * 0.5
    return RawDetection(
        landmarks_xy=np.column_stack((x, y)),
        coordinate_space=DISPLAY_PIXEL,
        handedness="Left" if marker % 2 == 0 else "Right",
        handedness_confidence=0.9,
        detection_confidence=0.8,
        landmark_confidence=np.full(21, 0.75, dtype=np.float64),
        selection_score=(
            float(100 - marker)
            if selection_score is None
            else float(selection_score)
        ),
    )


class DeterministicPoseEstimator:
    """Small estimator double with observable decoder-owned identity calls."""

    def __init__(
        self,
        *,
        detection_counts: Sequence[int] = (0, 1, 2, 0, 1),
        provenance: EstimatorProvenance = DEFAULT_ESTIMATOR_PROVENANCE,
        reverse_detections: bool = False,
        tied_selection_scores: bool = False,
        recoverable_failure_call: int | None = None,
        fatal_failure_call: int | None = None,
    ) -> None:
        if not detection_counts or any(count < 0 for count in detection_counts):
            raise ValueError("detection_counts must contain non-negative values")
        self._provenance = provenance
        self._detection_counts = tuple(int(count) for count in detection_counts)
        self._reverse_detections = reverse_detections
        self._tied_selection_scores = tied_selection_scores
        self._recoverable_failure_call = recoverable_failure_call
        self._fatal_failure_call = fatal_failure_call
        self.prepare_call_count = 0
        self.inference_frame_ids: list[str] = []

    @property
    def provenance(self) -> EstimatorProvenance:
        return self._provenance

    def prepare(self, display_bgr: np.ndarray) -> PreparedCVInput:
        self.prepare_call_count += 1
        return PreparedCVInput(
            pixels=np.ascontiguousarray(display_bgr),
            display_to_cv_transform=identity_transform(),
        )

    def infer(self, frame_id: str, cv_input_bgr: np.ndarray) -> InferenceOutput:
        del cv_input_bgr
        call_index = len(self.inference_frame_ids)
        self.inference_frame_ids.append(frame_id)
        if call_index == self._fatal_failure_call:
            raise FatalInferenceInterruption("injected fatal inference interruption")
        if call_index == self._recoverable_failure_call:
            raise RuntimeError("injected recoverable inference failure")
        count = self._detection_counts[call_index % len(self._detection_counts)]
        detections = [
            fixture_detection(
                marker,
                selection_score=0.5 if self._tied_selection_scores else None,
            )
            for marker in range(count)
        ]
        if self._reverse_detections:
            detections.reverse()
        return InferenceOutput(
            detections=tuple(detections),
            tracking_quality=0.95,
        )


def _bright_components(pixels: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return 8-connected bright-pixel components in deterministic scan order."""

    frame = np.asarray(pixels)
    mask = np.all(frame >= 250, axis=2)
    visited = np.zeros(mask.shape, dtype=np.bool_)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for seed_y, seed_x in np.argwhere(mask):
        if visited[seed_y, seed_x]:
            continue
        pending = [(int(seed_y), int(seed_x))]
        visited[seed_y, seed_x] = True
        points: list[tuple[int, int]] = []
        while pending:
            y, x = pending.pop()
            points.append((y, x))
            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    if mask[neighbor_y, neighbor_x] \
                            and not visited[neighbor_y, neighbor_x]:
                        visited[neighbor_y, neighbor_x] = True
                        pending.append((neighbor_y, neighbor_x))
        component = np.asarray(points, dtype=np.int64)
        if len(component) >= 16:
            components.append(component)
    components.sort(key=lambda item: (
        int(np.min(item[:, 1])), int(np.min(item[:, 0]))
    ))
    return tuple(components)


class PixelMarkerPoseEstimator:
    """Fixture estimator whose detections are derived only from input pixels."""

    def __init__(self) -> None:
        self.prepare_call_count = 0
        self.inference_frame_ids: list[str] = []
        self.input_sha256_by_frame_id: dict[str, str] = {}

    @property
    def provenance(self) -> EstimatorProvenance:
        return PIXEL_MARKER_ESTIMATOR_PROVENANCE

    def prepare(self, display_bgr: np.ndarray) -> PreparedCVInput:
        self.prepare_call_count += 1
        return PreparedCVInput(
            pixels=np.ascontiguousarray(display_bgr),
            display_to_cv_transform=identity_transform(),
        )

    def infer(self, frame_id: str, cv_input_bgr: np.ndarray) -> InferenceOutput:
        self.inference_frame_ids.append(frame_id)
        prepared = PreparedCVInput(
            pixels=np.ascontiguousarray(cv_input_bgr),
            display_to_cv_transform=identity_transform(),
        )
        self.input_sha256_by_frame_id[frame_id] = canonical_cv_input_sha256(
            prepared
        )
        detections: list[RawDetection] = []
        for component in _bright_components(prepared.pixels):
            y_min = int(np.min(component[:, 0]))
            y_max = int(np.max(component[:, 0]))
            x_min = int(np.min(component[:, 1]))
            x_max = int(np.max(component[:, 1]))
            landmarks = np.column_stack((
                np.linspace(x_min, x_max, 21, dtype=np.float64),
                np.linspace(y_min, y_max, 21, dtype=np.float64),
            ))
            detections.append(RawDetection(
                landmarks_xy=landmarks,
                coordinate_space=DISPLAY_PIXEL,
                bbox_xyxy=(
                    float(x_min), float(y_min), float(x_max), float(y_max)
                ),
                handedness="Left" if x_min < prepared.pixels.shape[1] / 2 else "Right",
                handedness_confidence=1.0,
                detection_confidence=1.0,
                landmark_confidence=np.ones(21, dtype=np.float64),
                selection_score=float(prepared.pixels.shape[1] - x_min),
            ))
        return InferenceOutput(
            detections=tuple(detections),
            tracking_quality=1.0,
        )


__all__ = [
    "DEFAULT_ESTIMATOR_PROVENANCE",
    "PIXEL_MARKER_ESTIMATOR_PROVENANCE",
    "DeterministicPoseEstimator",
    "FatalInferenceInterruption",
    "PixelMarkerPoseEstimator",
    "fixture_detection",
]
