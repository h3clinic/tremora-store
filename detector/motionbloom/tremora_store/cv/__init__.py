"""Offline CV finalization bound to decoder-owned frame identities."""

from .offline_finalizer import (
    EstimatorProvenance,
    FinalizedFrameTables,
    InferenceOutput,
    OfflineFinalizationError,
    OfflineFrameEstimator,
    RawDetection,
    finalize_frames,
)
from .pose_frame_contract import (
    ASSOCIATION_SCHEMA_VERSION,
    PoseFrameAssociationAudit,
    PoseFrameContractError,
    stable_detection_id,
    validate_pose_frame_association,
)

__all__ = [
    "ASSOCIATION_SCHEMA_VERSION",
    "EstimatorProvenance",
    "FinalizedFrameTables",
    "InferenceOutput",
    "OfflineFinalizationError",
    "OfflineFrameEstimator",
    "PoseFrameAssociationAudit",
    "PoseFrameContractError",
    "RawDetection",
    "finalize_frames",
    "stable_detection_id",
    "validate_pose_frame_association",
]
