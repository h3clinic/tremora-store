"""Frozen CPU/IMAGE MediaPipe HandLandmarker adapter for v0.4.

MediaPipe is imported only when the real backend is created.  Contract and
manifest tests therefore remain runnable without the optional production
runtime installed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, Self

import numpy as np

from ..cv.coordinate_mapping import (
    NORMALIZED_CV_INPUT,
    PreparedCVInput,
    identity_transform,
    invert_transform,
    map_bbox_xyxy,
    map_points,
    normalized_to_pixels,
)
from ..cv.offline_finalizer import (
    EstimatorProvenance,
    InferenceOutput,
    RawDetection,
)
from .model_manifest import (
    PERSISTED_DETECTION_COORDINATE_SPACE,
    PERSISTED_DETECTION_DTYPE,
    RAW_DETECTION_COORDINATE_SPACE,
    ProductionModelError,
    VerifiedProductionModel,
    _is_loader_verified_production_model,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ProductionInferenceError(RuntimeError):
    """Raised for lifecycle or result-shape failures at inference time."""


class FrameLocalBackendError(RuntimeError):
    """A backend-declared recoverable error isolated to one input frame."""


class _Backend(Protocol):
    def detect(self, rgb: np.ndarray) -> object: ...

    def close(self) -> None: ...


BackendFactory = Callable[[bytes, Mapping[str, object]], _Backend]
_TEST_BACKEND_FACTORY_CAPABILITY = object()


@dataclass(frozen=True)
class ProductionInferenceOutput(InferenceOutput):
    """Inference output with explicit status and immutable runtime metadata."""

    inference_status: str = "NO_DETECTION"
    preprocessing_transform: tuple[float, ...] = (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    )
    runtime_metadata: Mapping[str, object] | None = None
    failure_reason: str | None = None


class _MediaPipeBackend:
    def __init__(self, weights_bytes: bytes, config: Mapping[str, object]) -> None:
        # Optional runtime boundary: do not move these imports to module scope.
        import mediapipe as mp  # type: ignore[import-not-found]
        from mediapipe.tasks.python import (  # type: ignore[import-not-found]
            BaseOptions,
            vision,
        )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_buffer=weights_bytes,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=int(config["num_hands"]),
            min_hand_detection_confidence=float(
                config["min_hand_detection_confidence"]
            ),
            min_hand_presence_confidence=float(
                config["min_hand_presence_confidence"]
            ),
            min_tracking_confidence=float(config["min_tracking_confidence"]),
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray) -> object:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        return self._landmarker.detect(image)

    def close(self) -> None:
        self._landmarker.close()


def _default_backend_factory(
    weights_bytes: bytes,
    config: Mapping[str, object],
) -> _Backend:
    return _MediaPipeBackend(weights_bytes, config)


def raw_detection_to_display_payload(
    detection: RawDetection,
    prepared: PreparedCVInput,
) -> dict[str, object]:
    """Map raw normalized estimator output to persisted display pixels.

    The HandLandmarker result remains ``NORMALIZED_CV_INPUT`` until this
    explicit persistence boundary.  The returned float32 payload is suitable
    for :func:`build_detection_rows`, whose schema is ``DISPLAY_PIXEL``.
    """

    if detection.coordinate_space != RAW_DETECTION_COORDINATE_SPACE:
        raise ProductionInferenceError(
            "production raw detection must be NORMALIZED_CV_INPUT"
        )
    normalized = np.asarray(detection.landmarks_xy, dtype=np.float64)
    if normalized.shape != (21, 2) or not np.isfinite(normalized).all():
        raise ProductionInferenceError(
            "raw normalized landmarks must be finite with shape (21, 2)"
        )
    validity_value = detection.landmark_validity_mask
    if validity_value is None:
        validity = np.ones(21, dtype=np.bool_)
    else:
        validity = np.asarray(validity_value)
        if validity.shape != (21,) or validity.dtype.kind != "b":
            raise ProductionInferenceError(
                "raw landmark validity must contain 21 booleans"
            )
        validity = validity.astype(np.bool_, copy=False)
    if not np.any(validity):
        raise ProductionInferenceError("raw detection has no valid landmarks")

    cv_points = normalized_to_pixels(
        normalized,
        width=int(prepared.pixels.shape[1]),
        height=int(prepared.pixels.shape[0]),
    )
    cv_to_display = invert_transform(prepared.display_to_cv_transform)
    display_points = map_points(cv_points, cv_to_display)
    if detection.bbox_xyxy is None:
        selected = display_points[validity]
        bbox_display = (
            float(np.min(selected[:, 0])),
            float(np.min(selected[:, 1])),
            float(np.max(selected[:, 0])),
            float(np.max(selected[:, 1])),
        )
    else:
        bbox = np.asarray(detection.bbox_xyxy, dtype=np.float64)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ProductionInferenceError("raw normalized bbox is invalid")
        scale_x = max(int(prepared.pixels.shape[1]) - 1, 0)
        scale_y = max(int(prepared.pixels.shape[0]) - 1, 0)
        bbox_display = map_bbox_xyxy(
            (
                float(bbox[0] * scale_x),
                float(bbox[1] * scale_y),
                float(bbox[2] * scale_x),
                float(bbox[3] * scale_y),
            ),
            cv_to_display,
        )

    def optional_vector(value: np.ndarray | None, field: str) -> np.ndarray | None:
        if value is None:
            return None
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (21,) or not np.isfinite(result).all():
            raise ProductionInferenceError(f"{field} must contain 21 finite values")
        return result

    return {
        "handedness": detection.handedness,
        "handedness_confidence": detection.handedness_confidence,
        "detection_confidence": detection.detection_confidence,
        "bbox_xyxy_display": np.asarray(bbox_display, dtype=np.float32),
        "landmarks_xy_display": np.asarray(display_points, dtype=np.float32),
        "landmarks_z_model": optional_vector(
            detection.landmarks_z_model, "landmarks_z_model"
        ),
        "landmark_confidence": optional_vector(
            detection.landmark_confidence, "landmark_confidence"
        ),
        "landmark_validity_mask": validity,
    }


class MediaPipeHandLandmarkerEstimator:
    """Full-frame BGR-to-RGB, all-detection, per-recording-reset adapter."""

    tracker_version: None = None
    stateless_per_frame = True

    def __init__(
        self,
        verified_model: VerifiedProductionModel,
    ) -> None:
        if not _is_loader_verified_production_model(verified_model):
            raise ProductionModelError(
                "loader-verified production model is required"
            )
        self._model = verified_model
        self._backend_factory = _default_backend_factory
        self._backend: _Backend | None = None
        self._active_recording_id: str | None = None
        self._recording_state_generation = 0
        self._closed = False

    @classmethod
    def _with_test_backend(
        cls,
        verified_model: VerifiedProductionModel,
        backend_factory: BackendFactory,
        capability: object,
    ) -> MediaPipeHandLandmarkerEstimator:
        if capability is not _TEST_BACKEND_FACTORY_CAPABILITY:
            raise ProductionModelError("test backend capability is invalid")
        estimator = cls(verified_model)
        estimator._backend_factory = backend_factory
        return estimator

    @property
    def provenance(self) -> EstimatorProvenance:
        manifest = self._model.manifest
        return EstimatorProvenance(
            model_id=str(manifest["model_id"]),
            model_weights_sha256=str(manifest["model_weights_sha256"]),
            preprocessing_config_sha256=str(
                manifest["preprocessing_config_sha256"]
            ),
            inference_environment_id=(
                "native-runtime-sha256:"
                f"{manifest['runtime_lock_sha256']}"
            ),
        )

    @property
    def model_manifest_sha256(self) -> str:
        return self._model.manifest_sha256

    @property
    def active_recording_id(self) -> str | None:
        return self._active_recording_id

    @property
    def recording_state_generation(self) -> int:
        return self._recording_state_generation

    def prepare(self, display_bgr: np.ndarray) -> PreparedCVInput:
        pixels = np.asarray(display_bgr)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ProductionInferenceError(
                "production input must be a full-frame HxWx3 uint8 BGR image"
            )
        if pixels.shape[0] < 1 or pixels.shape[1] < 1:
            raise ProductionInferenceError("production input frame must be non-empty")
        return PreparedCVInput(
            pixels=np.array(pixels, dtype=np.uint8, order="C", copy=True),
            display_to_cv_transform=identity_transform(),
            pixel_format="bgr24",
        )

    def _release_backend(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.close()

    def reset_for_recording(self, recording_id: str) -> None:
        """Close any prior backend and create a clean IMAGE-mode instance."""

        if self._closed:
            raise ProductionInferenceError("estimator is closed")
        if not isinstance(recording_id, str) or not recording_id:
            raise ProductionInferenceError("recording_id must be non-empty")
        self._release_backend()
        self._active_recording_id = None
        self._backend = self._backend_factory(
            self._model.weights_bytes,
            self._model.preprocessing_config,
        )
        self._active_recording_id = recording_id
        self._recording_state_generation += 1

    def end_recording(self, recording_id: str) -> None:
        if recording_id != self._active_recording_id:
            raise ProductionInferenceError("cannot end a recording that is not active")
        self._release_backend()
        self._active_recording_id = None

    def _runtime_metadata(self) -> Mapping[str, object]:
        return MappingProxyType({
            "active_recording_id": self._active_recording_id,
            "deterministic_mode": False,
            "inference_delegate": "CPU",
            "inference_delegate_thread_count": None,
            "model_manifest_sha256": self._model.manifest_sha256,
            "num_hands": 2,
            "persisted_detection_coordinate_space": (
                PERSISTED_DETECTION_COORDINATE_SPACE
            ),
            "persisted_detection_dtype": PERSISTED_DETECTION_DTYPE,
            "raw_detection_coordinate_space": RAW_DETECTION_COORDINATE_SPACE,
            "recording_state_generation": self._recording_state_generation,
            "running_mode": "IMAGE",
            "runtime_lock_sha256": self._model.manifest["runtime_lock_sha256"],
            "runtime_worker_concurrency_observed": self._model.manifest[
                "runtime_worker_concurrency_observed"
            ],
            "stateless_per_frame": True,
            "whole_process_thread_count": None,
        })

    @staticmethod
    def _category(value: object) -> tuple[str | None, float | None]:
        if value is None:
            return None, None
        name = getattr(value, "category_name", None)
        score = getattr(value, "score", None)
        if name is not None and (not isinstance(name, str) or not name):
            raise ProductionInferenceError("invalid handedness category name")
        if score is not None:
            score = float(np.float32(score))
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ProductionInferenceError("invalid handedness confidence")
        return name, score

    @staticmethod
    def _raw_detections(result: object) -> tuple[RawDetection, ...]:
        missing = object()
        landmarks_value = getattr(result, "hand_landmarks", missing)
        handedness_value = getattr(result, "handedness", missing)
        if landmarks_value is missing or handedness_value is missing \
                or landmarks_value is None or handedness_value is None:
            raise ProductionInferenceError(
                "HandLandmarker result requires hand_landmarks and handedness"
            )
        try:
            landmarks_groups = tuple(landmarks_value)
            handedness_groups = tuple(handedness_value)
        except TypeError as exc:
            raise ProductionInferenceError(
                "HandLandmarker result groups must be iterable"
            ) from exc
        if len(handedness_groups) != len(landmarks_groups):
            raise ProductionInferenceError(
                "HandLandmarker handedness group count must equal landmarks"
            )
        if len(landmarks_groups) > 2:
            raise ProductionInferenceError(
                "HandLandmarker exceeded the frozen num_hands=2 contract"
            )
        detections: list[RawDetection] = []
        for index, landmarks in enumerate(landmarks_groups):
            if len(landmarks) != 21:
                raise ProductionInferenceError(
                    "HandLandmarker detection must contain 21 landmarks"
                )
            xyz = np.asarray([
                (float(point.x), float(point.y), float(point.z))
                for point in landmarks
            ], dtype=np.float64)
            if xyz.shape != (21, 3) or not np.isfinite(xyz).all():
                raise ProductionInferenceError(
                    "HandLandmarker landmarks must be finite"
                )
            categories = (
                tuple(handedness_groups[index])
                if index < len(handedness_groups)
                else ()
            )
            handedness, handedness_confidence = (
                MediaPipeHandLandmarkerEstimator._category(categories[0])
                if categories else (None, None)
            )
            bbox = (
                float(np.min(xyz[:, 0])),
                float(np.min(xyz[:, 1])),
                float(np.max(xyz[:, 0])),
                float(np.max(xyz[:, 1])),
            )
            detections.append(RawDetection(
                landmarks_xy=xyz[:, :2],
                coordinate_space=NORMALIZED_CV_INPUT,
                bbox_xyxy=bbox,
                landmarks_z_model=xyz[:, 2],
                landmark_confidence=None,
                landmark_validity_mask=np.ones(21, dtype=np.bool_),
                handedness=handedness,
                handedness_confidence=handedness_confidence,
                # Tasks HandLandmarker does not expose the detector score for
                # accepted results.  Do not relabel handedness as detection confidence.
                detection_confidence=None,
                selection_score=0.0,
            ))
        return tuple(detections)

    def infer(self, frame_id: str, cv_input_bgr: np.ndarray) -> InferenceOutput:
        if not isinstance(frame_id, str) or _SHA256_RE.fullmatch(frame_id) is None:
            raise ProductionInferenceError("frame_id must be a lowercase SHA-256")
        if self._backend is None or self._active_recording_id is None:
            raise ProductionInferenceError(
                "reset_for_recording must precede production inference"
            )
        pixels = np.asarray(cv_input_bgr)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ProductionInferenceError("inference input must be uint8 BGR24")
        rgb = np.ascontiguousarray(pixels[:, :, ::-1])
        result = self._backend.detect(rgb)
        return InferenceOutput(
            detections=self._raw_detections(result),
            tracking_quality=None,
        )

    def infer_with_status(
        self,
        frame_id: str,
        cv_input_bgr: np.ndarray,
    ) -> ProductionInferenceOutput:
        """Return an explicit zero/one/many outcome without hiding failures."""

        try:
            output = self.infer(frame_id, cv_input_bgr)
        except FrameLocalBackendError as exc:
            return ProductionInferenceOutput(
                detections=(),
                tracking_quality=None,
                inference_status="INFERENCE_FAILURE",
                preprocessing_transform=identity_transform(),
                runtime_metadata=self._runtime_metadata(),
                failure_reason=f"{type(exc).__module__}.{type(exc).__qualname__}",
            )
        return ProductionInferenceOutput(
            detections=output.detections,
            tracking_quality=None,
            inference_status="SUCCESS" if output.detections else "NO_DETECTION",
            preprocessing_transform=identity_transform(),
            runtime_metadata=self._runtime_metadata(),
            failure_reason=None,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._release_backend()
        self._active_recording_id = None
        self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise ProductionInferenceError("estimator is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


def _test_media_pipe_hand_landmarker_estimator(
    verified_model: VerifiedProductionModel,
    backend_factory: BackendFactory,
) -> MediaPipeHandLandmarkerEstimator:
    """Construct a fake-backed estimator for in-module contract tests only."""

    return MediaPipeHandLandmarkerEstimator._with_test_backend(
        verified_model,
        backend_factory,
        _TEST_BACKEND_FACTORY_CAPABILITY,
    )


__all__ = [
    "BackendFactory",
    "FrameLocalBackendError",
    "MediaPipeHandLandmarkerEstimator",
    "ProductionInferenceError",
    "ProductionInferenceOutput",
    "raw_detection_to_display_payload",
]
