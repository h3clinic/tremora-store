"""Immutable PTS/CV finalization bundles, separate from cross-modal snapshots."""

from .audit_finalized_recording import audit_finalized_recording
from .finalize_vidimu_recording import (
    FinalizedRecording,
    RecordingFinalizationError,
    RecordingProvenance,
    finalize_vidimu_recording,
)
from .finalize_vidimu_snapshot import (
    FrozenSourceAsset,
    VidimuSnapshotInputs,
    VidimuSnapshotRecord,
    finalize_vidimu_snapshot,
    preflight_vidimu_snapshot,
)
from .source_failure_artifact import (
    SourceFailureArtifact,
    audit_source_failure_artifact,
)

__all__ = [
    "FinalizedRecording",
    "FrozenSourceAsset",
    "RecordingFinalizationError",
    "RecordingProvenance",
    "SourceFailureArtifact",
    "VidimuSnapshotInputs",
    "VidimuSnapshotRecord",
    "audit_finalized_recording",
    "audit_source_failure_artifact",
    "finalize_vidimu_recording",
    "finalize_vidimu_snapshot",
    "preflight_vidimu_snapshot",
]
