"""Parallel v0.4 production-CV contracts.

Version 0.3 remains the frozen Gate-A association implementation.  Nothing in
this package changes its schemas or rank-derived fixture identities.
"""

from .audit import (
    BYTE_IDENTICAL_SOURCE_TO_CV_PASS,
    CANONICAL_CONTENT_IDENTICAL_PARQUET_BYTES_DIFFER_PASS,
    FAIL,
    audit_gate_b_run,
    audit_vidimu_v04_gate_b_release,
)
from .bundle import (
    V04_SUCCESS_FILES,
    audit_v04_bundle,
    publish_v04_bundle,
)
from .detection_contract import (
    PRIMARY_HAND_SELECTION_CONTRACT_VERSION,
    V04_ASSOCIATION_CONTRACT_VERSION,
    DetectionContractError,
    build_detection_rows,
    build_primary_hand_selection,
    stable_payload_detection_id,
    validate_detection_and_selection_rows,
)
from .finalization import (
    FinalizedGateBRun,
    GateBFinalizationError,
    GateBModelInputs,
    finalize_vidimu_v04_gate_b,
)
from .frame_finalizer import (
    V04FinalizedFrameTables,
    V04FrameFinalizationError,
    finalize_recording_frames,
)
from .mediapipe_hand_landmarker import (
    FrameLocalBackendError,
    MediaPipeHandLandmarkerEstimator,
    raw_detection_to_display_payload,
)
from .model_manifest import (
    VENDORED_MODEL_ID,
    VENDORED_MODEL_SHA256,
    VENDORED_MODEL_SIZE_BYTES,
    ProductionModelError,
    VerifiedProductionModel,
    build_production_model_manifest,
    load_and_verify_production_model,
    write_production_model_manifest,
)

__all__ = [
    "BYTE_IDENTICAL_SOURCE_TO_CV_PASS",
    "CANONICAL_CONTENT_IDENTICAL_PARQUET_BYTES_DIFFER_PASS",
    "FAIL",
    "PRIMARY_HAND_SELECTION_CONTRACT_VERSION",
    "V04_ASSOCIATION_CONTRACT_VERSION",
    "V04_SUCCESS_FILES",
    "VENDORED_MODEL_ID",
    "VENDORED_MODEL_SHA256",
    "VENDORED_MODEL_SIZE_BYTES",
    "DetectionContractError",
    "FinalizedGateBRun",
    "FrameLocalBackendError",
    "GateBFinalizationError",
    "GateBModelInputs",
    "MediaPipeHandLandmarkerEstimator",
    "ProductionModelError",
    "V04FinalizedFrameTables",
    "V04FrameFinalizationError",
    "VerifiedProductionModel",
    "audit_gate_b_run",
    "audit_v04_bundle",
    "audit_vidimu_v04_gate_b_release",
    "build_detection_rows",
    "build_primary_hand_selection",
    "build_production_model_manifest",
    "finalize_recording_frames",
    "finalize_vidimu_v04_gate_b",
    "load_and_verify_production_model",
    "publish_v04_bundle",
    "raw_detection_to_display_payload",
    "stable_payload_detection_id",
    "validate_detection_and_selection_rows",
    "write_production_model_manifest",
]
