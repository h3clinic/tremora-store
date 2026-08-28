"""Bind v0.4 production hand inference to v0.3 decoder frame identities."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from ..cv.coordinate_mapping import (
    PreparedCVInput,
    canonical_cv_input_sha256,
    compose_transforms,
    invert_transform,
)
from ..cv.offline_finalizer import EstimatorProvenance
from ..decode.pts_decoder import DecodedFrame
from ..finalize._bundle_io import canonical_json_bytes
from ..schema import QualityBits
from .detection_contract import (
    INFERENCE_STATUSES,
    build_detection_rows,
    build_primary_hand_selection,
    detection_rows_table,
    primary_selection_table,
    validate_detection_and_selection_rows,
)
from .mediapipe_hand_landmarker import (
    MediaPipeHandLandmarkerEstimator,
    ProductionInferenceError,
    ProductionInferenceOutput,
    raw_detection_to_display_payload,
)
from .model_manifest import (
    VerifiedProductionModel,
    _is_loader_verified_production_model,
)
from .schemas import (
    V04_ASSOCIATION_CONTRACT_VERSION,
    cv_frame_results_v04_schema,
    video_frames_v04_schema,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CV_FAILURE_QUALITY_BIT = int(QualityBits.INVALID_CV)
_INFERENCE_OUTPUT_STATUSES = frozenset({
    "SUCCESS",
    "NO_DETECTION",
    "INFERENCE_FAILURE",
})
_RUNTIME_METADATA_FIELDS = frozenset({
    "active_recording_id",
    "deterministic_mode",
    "inference_delegate",
    "inference_delegate_thread_count",
    "model_manifest_sha256",
    "num_hands",
    "persisted_detection_coordinate_space",
    "persisted_detection_dtype",
    "raw_detection_coordinate_space",
    "recording_state_generation",
    "running_mode",
    "runtime_lock_sha256",
    "runtime_worker_concurrency_observed",
    "stateless_per_frame",
    "whole_process_thread_count",
})


class V04FrameFinalizationError(ValueError):
    """Raised when v0.4 frame inference cannot be recorded unambiguously."""


@dataclass(frozen=True)
class V04FinalizedFrameTables:
    """The four per-recording Arrow tables produced by v0.4 inference."""

    video_frames: pa.Table
    cv_frame_results: pa.Table
    cv_detections: pa.Table
    primary_hand_selection: pa.Table

    def as_dict(self) -> dict[str, pa.Table]:
        return {
            "video_frames": self.video_frames,
            "cv_frame_results": self.cv_frame_results,
            "cv_detections": self.cv_detections,
            "primary_hand_selection": self.primary_hand_selection,
        }


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise V04FrameFinalizationError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise V04FrameFinalizationError(f"{field} must be non-empty text")
    return value


def _transform(value: object, field: str) -> list[float]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise V04FrameFinalizationError(
            f"{field} must contain 9 finite numeric values"
        ) from exc
    if result.shape != (9,) or not np.isfinite(result).all():
        raise V04FrameFinalizationError(
            f"{field} must contain 9 finite numeric values"
        )
    result = np.array(result, dtype=np.float64, order="C", copy=True)
    result[result == 0] = 0.0
    return [float(item) for item in result]


def _optional_probability(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V04FrameFinalizationError(f"{field} must be numeric or null")
    result = float(np.float32(value))
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise V04FrameFinalizationError(f"{field} must be in [0,1] or null")
    return result


def _exception_identity(error: BaseException) -> str:
    error_type = type(error)
    return f"{error_type.__module__}.{error_type.__qualname__}"


def _verified_model(
    estimator: MediaPipeHandLandmarkerEstimator,
) -> VerifiedProductionModel:
    if type(estimator) is not MediaPipeHandLandmarkerEstimator:
        raise V04FrameFinalizationError(
            "estimator must be the frozen v0.4 production estimator"
        )
    model = getattr(estimator, "_model", None)
    if not _is_loader_verified_production_model(model):
        raise V04FrameFinalizationError(
            "estimator model must retain loader attestation"
        )
    assert isinstance(model, VerifiedProductionModel)
    return model


def _validate_provenance(
    estimator: MediaPipeHandLandmarkerEstimator,
    model: VerifiedProductionModel,
) -> EstimatorProvenance:
    provenance = estimator.provenance
    if not isinstance(provenance, EstimatorProvenance):
        raise V04FrameFinalizationError(
            "estimator provenance must be EstimatorProvenance"
        )
    manifest = model.manifest
    expected = {
        "model_id": manifest["model_id"],
        "model_weights_sha256": manifest["model_weights_sha256"],
        "preprocessing_config_sha256": manifest[
            "preprocessing_config_sha256"
        ],
        "inference_environment_id": (
            f"native-runtime-sha256:{manifest['runtime_lock_sha256']}"
        ),
    }
    if any(getattr(provenance, field) != value for field, value in expected.items()):
        raise V04FrameFinalizationError(
            "estimator provenance does not match its verified model manifest"
        )
    if estimator.model_manifest_sha256 != model.manifest_sha256:
        raise V04FrameFinalizationError(
            "estimator model-manifest identity is inconsistent"
        )
    _sha256(model.manifest_sha256, "model_manifest_sha256")
    if manifest["association_contract_version"] != (
        V04_ASSOCIATION_CONTRACT_VERSION
    ):
        raise V04FrameFinalizationError(
            "model association contract does not match the v0.4 schema"
        )
    return provenance


def _runtime_metadata_json(
    value: object,
    *,
    estimator: MediaPipeHandLandmarkerEstimator,
    model: VerifiedProductionModel,
    recording_id: str,
) -> str:
    if not isinstance(value, Mapping) or set(value) != _RUNTIME_METADATA_FIELDS:
        raise V04FrameFinalizationError(
            "runtime metadata fields do not match the frozen v0.4 contract"
        )
    manifest = model.manifest
    runtime_lock = model.runtime_lock
    expected: dict[str, object] = {
        "active_recording_id": recording_id,
        "deterministic_mode": manifest["deterministic_mode"],
        "inference_delegate": runtime_lock["inference_delegate"],
        "inference_delegate_thread_count": manifest[
            "inference_delegate_thread_count"
        ],
        "model_manifest_sha256": model.manifest_sha256,
        "num_hands": model.preprocessing_config["num_hands"],
        "persisted_detection_coordinate_space": manifest[
            "persisted_detection_coordinate_space"
        ],
        "persisted_detection_dtype": manifest["persisted_detection_dtype"],
        "raw_detection_coordinate_space": manifest[
            "raw_detection_coordinate_space"
        ],
        "recording_state_generation": estimator.recording_state_generation,
        "running_mode": model.preprocessing_config["running_mode"],
        "runtime_lock_sha256": manifest["runtime_lock_sha256"],
        "runtime_worker_concurrency_observed": manifest[
            "runtime_worker_concurrency_observed"
        ],
        "stateless_per_frame": True,
        "whole_process_thread_count": runtime_lock[
            "whole_process_thread_count"
        ],
    }
    actual = dict(value)
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if expected_value is None:
            valid = actual_value is None
        else:
            valid = (
                type(actual_value) is type(expected_value)
                and actual_value == expected_value
            )
        if not valid:
            raise V04FrameFinalizationError(
                f"runtime metadata {field} does not match verified provenance"
            )
    return canonical_json_bytes(actual).decode("ascii")


def _frame_row(
    frame: DecodedFrame,
    *,
    dataset_id: str,
    recording_id: str,
) -> dict[str, object]:
    decoder_fields = (
        "source_video_sha256",
        "stream_index",
        "frame_id",
        "identity_basis",
        "decode_ordinal",
        "presentation_ordinal",
        "pts",
        "time_base_num",
        "time_base_den",
        "relative_pts_ns",
        "same_pts_rank",
        "duration_pts",
        "duration_ns",
        "gap_before_ns",
        "coded_width",
        "coded_height",
        "display_width",
        "display_height",
        "rotation_degrees",
        "pixel_format",
        "key_frame",
        "picture_type",
        "pts_status",
        "decode_status",
        "quality_bits",
        "source_to_display_transform",
        "decoder_version",
        "schema_version",
    )
    row = {field: getattr(frame, field) for field in decoder_fields}
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


def _frozen_prepared_input(
    prepared: PreparedCVInput,
    *,
    frame: DecodedFrame,
    frame_row: dict[str, object],
) -> tuple[PreparedCVInput, str]:
    if type(prepared) is not PreparedCVInput:
        raise V04FrameFinalizationError(
            "production prepare must return PreparedCVInput"
        )
    pixels = np.array(prepared.pixels, dtype=np.uint8, order="C", copy=True)
    pixels.setflags(write=False)
    frozen = PreparedCVInput(
        pixels=pixels,
        display_to_cv_transform=prepared.display_to_cv_transform,
        pixel_format=prepared.pixel_format,
    )
    display_to_cv = _transform(
        frozen.display_to_cv_transform,
        "display_to_cv_transform",
    )
    source_to_cv = compose_transforms(
        frame.source_to_display_transform,
        tuple(display_to_cv),
    )
    cv_to_source = invert_transform(source_to_cv)
    input_sha256 = canonical_cv_input_sha256(frozen)
    frame_row.update({
        "display_to_cv_transform": display_to_cv,
        "cv_to_source_transform": list(cv_to_source),
        "preprocessing_transform_invertible": True,
        "cv_input_width": int(pixels.shape[1]),
        "cv_input_height": int(pixels.shape[0]),
        "cv_input_pixel_format": frozen.pixel_format,
        "cv_input_sha256": input_sha256,
    })
    return frozen, input_sha256


def _validated_output(
    output: object,
    *,
    prepared: PreparedCVInput,
    input_sha256: str,
    estimator: MediaPipeHandLandmarkerEstimator,
    model: VerifiedProductionModel,
    recording_id: str,
) -> tuple[
    ProductionInferenceOutput,
    str,
    list[float],
    str,
    float | None,
]:
    if type(output) is not ProductionInferenceOutput:
        raise V04FrameFinalizationError(
            "infer_with_status must return ProductionInferenceOutput"
        )
    assert isinstance(output, ProductionInferenceOutput)
    if canonical_cv_input_sha256(prepared) != input_sha256:
        raise V04FrameFinalizationError(
            "production estimator mutated the frozen CV input buffer"
        )
    status = output.inference_status
    if status not in _INFERENCE_OUTPUT_STATUSES or status not in INFERENCE_STATUSES:
        raise V04FrameFinalizationError(
            "production estimator returned an invalid inference status"
        )
    if type(output.detections) is not tuple:
        raise V04FrameFinalizationError(
            "production detections must be an immutable tuple"
        )
    if (status == "SUCCESS") != bool(output.detections):
        raise V04FrameFinalizationError(
            "SUCCESS must describe exactly a non-empty detection result"
        )
    failure_reason = output.failure_reason
    if status == "INFERENCE_FAILURE":
        _nonempty_text(failure_reason, "failure_reason")
    elif failure_reason is not None:
        raise V04FrameFinalizationError(
            "successful inference cannot carry a failure reason"
        )
    preprocessing_transform = _transform(
        output.preprocessing_transform,
        "preprocessing_transform",
    )
    expected_transform = _transform(
        prepared.display_to_cv_transform,
        "prepared display_to_cv_transform",
    )
    if preprocessing_transform != expected_transform:
        raise V04FrameFinalizationError(
            "reported preprocessing transform does not match estimator input"
        )
    runtime_json = _runtime_metadata_json(
        output.runtime_metadata,
        estimator=estimator,
        model=model,
        recording_id=recording_id,
    )
    tracking_quality = _optional_probability(
        output.tracking_quality,
        "tracking_quality",
    )
    return (
        output,
        status,
        preprocessing_transform,
        runtime_json,
        tracking_quality,
    )


def _validate_decoded_frames(
    decoded_frames: Sequence[DecodedFrame],
) -> tuple[DecodedFrame, ...]:
    if isinstance(decoded_frames, (str, bytes)) or not isinstance(
        decoded_frames, Sequence
    ):
        raise V04FrameFinalizationError(
            "decoded_frames must be a sequence of DecodedFrame values"
        )
    frames = tuple(decoded_frames)
    seen_ids: set[str] = set()
    source_streams: set[tuple[str, int]] = set()
    for frame in frames:
        if type(frame) is not DecodedFrame:
            raise V04FrameFinalizationError(
                "decoded_frames must contain v0.3 DecodedFrame values"
            )
        frame_id = _sha256(frame.frame_id, "frame_id")
        _sha256(frame.source_video_sha256, "source_video_sha256")
        if frame_id in seen_ids:
            raise V04FrameFinalizationError(
                "decoder emitted a duplicate frame_id"
            )
        seen_ids.add(frame_id)
        source_streams.add((frame.source_video_sha256, frame.stream_index))
        if frame.decode_status not in {"SUCCESS", "CORRUPT"}:
            raise V04FrameFinalizationError(
                "decoded frame status must be SUCCESS or CORRUPT"
            )
    if len(source_streams) > 1:
        raise V04FrameFinalizationError(
            "one recording cannot mix decoded video streams"
        )
    return frames


def finalize_recording_frames(
    decoded_frames: Sequence[DecodedFrame],
    *,
    dataset_id: str,
    recording_id: str,
    estimator: MediaPipeHandLandmarkerEstimator,
) -> V04FinalizedFrameTables:
    """Finalize every decoded frame with one reset v0.4 estimator lifecycle.

    Unexpected preprocessing/result-contract/backend exceptions abort the
    recording. Only decoder corruption, an explicit preprocessing rejection,
    or the estimator's explicit frame-local failure output becomes a status
    row. No detection is discarded, and primary selection is independently
    recomputed from the canonical persisted detection rows.
    """

    dataset = _nonempty_text(dataset_id, "dataset_id")
    recording = _nonempty_text(recording_id, "recording_id")
    frames = _validate_decoded_frames(decoded_frames)
    model = _verified_model(estimator)
    provenance = _validate_provenance(estimator, model)
    manifest = model.manifest
    runtime_lock_sha256 = _sha256(
        manifest["runtime_lock_sha256"],
        "runtime_lock_sha256",
    )
    association_contract = _nonempty_text(
        manifest["association_contract_version"],
        "association_contract_version",
    )

    frame_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    reset_succeeded = False
    estimator.reset_for_recording(recording)
    reset_succeeded = True
    try:
        for frame in frames:
            frame_row = _frame_row(
                frame,
                dataset_id=dataset,
                recording_id=recording,
            )
            status = "DECODE_FAILURE"
            failure_reason: str | None = "DECODE_STATUS_CORRUPT"
            preprocessing_transform: list[float] | None = None
            runtime_metadata_json: str | None = None
            tracking_quality: float | None = None
            frame_detection_rows: list[dict[str, object]] = []

            if frame.decode_status == "SUCCESS":
                try:
                    candidate = estimator.prepare(frame.display_bgr)
                except ProductionInferenceError as exc:
                    status = "PREPROCESS_FAILURE"
                    failure_reason = _exception_identity(exc)
                else:
                    prepared, input_sha256 = _frozen_prepared_input(
                        candidate,
                        frame=frame,
                        frame_row=frame_row,
                    )
                    output_value = estimator.infer_with_status(
                        frame.frame_id,
                        prepared.pixels,
                    )
                    (
                        output,
                        status,
                        preprocessing_transform,
                        runtime_metadata_json,
                        tracking_quality,
                    ) = _validated_output(
                        output_value,
                        prepared=prepared,
                        input_sha256=input_sha256,
                        estimator=estimator,
                        model=model,
                        recording_id=recording,
                    )
                    failure_reason = output.failure_reason
                    payloads = [
                        raw_detection_to_display_payload(detection, prepared)
                        for detection in output.detections
                    ]
                    frame_detection_rows = build_detection_rows(
                        frame_id=frame.frame_id,
                        model_manifest_sha256=model.manifest_sha256,
                        detections=payloads,
                    )

            selection = build_primary_hand_selection(
                frame_id=frame.frame_id,
                detections=frame_detection_rows,
                inference_status=status,
            )
            frame_quality = int(frame.quality_bits)
            if status in {
                "PREPROCESS_FAILURE",
                "INFERENCE_FAILURE",
                "REJECTED_INPUT",
            }:
                frame_quality |= _CV_FAILURE_QUALITY_BIT
            frame_rows.append(frame_row)
            detection_rows.extend(frame_detection_rows)
            selection_rows.append(selection)
            result_rows.append({
                "frame_id": frame.frame_id,
                "recording_id": recording,
                "relative_pts_ns": frame.relative_pts_ns,
                "model_manifest_sha256": model.manifest_sha256,
                "model_id": provenance.model_id,
                "model_weights_sha256": provenance.model_weights_sha256,
                "preprocessing_config_sha256": (
                    provenance.preprocessing_config_sha256
                ),
                "runtime_lock_sha256": runtime_lock_sha256,
                "association_contract_version": association_contract,
                "inference_environment_id": provenance.inference_environment_id,
                "inference_status": status,
                "detection_count": len(frame_detection_rows),
                "failure_reason": failure_reason,
                "preprocessing_transform": preprocessing_transform,
                "runtime_metadata_json": runtime_metadata_json,
                "runtime_ms": None,
                "tracking_quality": tracking_quality,
                "frame_quality_bits": frame_quality,
            })

        frame_ids = [frame.frame_id for frame in frames]
        validate_detection_and_selection_rows(
            detection_rows,
            selection_rows,
            frame_ids=frame_ids,
        )
        tables = V04FinalizedFrameTables(
            video_frames=pa.Table.from_pylist(
                frame_rows,
                schema=video_frames_v04_schema(),
            ),
            cv_frame_results=pa.Table.from_pylist(
                result_rows,
                schema=cv_frame_results_v04_schema(),
            ),
            cv_detections=detection_rows_table(detection_rows),
            primary_hand_selection=primary_selection_table(selection_rows),
        )
    finally:
        if reset_succeeded:
            estimator.end_recording(recording)
    return tables


__all__ = [
    "V04FinalizedFrameTables",
    "V04FrameFinalizationError",
    "finalize_recording_frames",
]
