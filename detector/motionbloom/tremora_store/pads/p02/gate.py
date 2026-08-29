"""The sixteen PADS-P0.2 hard-gate conditions.

Several are checked against something the materializer did not produce: the
sample count against the pinned P0.1 total, replay against the release's own
asset hashes read back from disk, and window ranges against rows actually
returned by the store rather than the arrays that built them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .contract import GATE_NO_GO, GATE_PASS
from .dependency import DEPENDENCY_VERIFIED

P01_AUTHORITY_DEPENDENCY_VERIFIED = "P01_AUTHORITY_DEPENDENCY_VERIFIED"
ALL_SOURCE_ASSETS_HASH_VERIFIED = "ALL_SOURCE_ASSETS_HASH_VERIFIED"
PARTICIPANT_INDEX_RECONCILED = "PARTICIPANT_INDEX_RECONCILED"
ASSESSMENT_INDEX_RECONCILED = "ASSESSMENT_INDEX_RECONCILED"
STREAM_INDEX_RECONCILED = "STREAM_INDEX_RECONCILED"
ALL_SOURCE_SAMPLES_STORED_EXACTLY_ONCE = (
    "ALL_SOURCE_SAMPLES_STORED_EXACTLY_ONCE"
)
SOURCE_TIME_TOKENS_PRESERVED = "SOURCE_TIME_TOKENS_PRESERVED"
STREAM_ROW_GROUP_INDEX_COMPLETE = "STREAM_ROW_GROUP_INDEX_COMPLETE"
SEGMENTS_PARTITION_STREAMS_EXACTLY = "SEGMENTS_PARTITION_STREAMS_EXACTLY"
WINDOWS_NEVER_CROSS_SEGMENT_BOUNDARIES = (
    "WINDOWS_NEVER_CROSS_SEGMENT_BOUNDARIES"
)
WINDOW_SAMPLE_RANGES_REPLAY_EXACTLY = "WINDOW_SAMPLE_RANGES_REPLAY_EXACTLY"
BILATERAL_TASK_PAIRS_COMPLETE = "BILATERAL_TASK_PAIRS_COMPLETE"
NO_SAMPLE_LEVEL_BILATERAL_SYNC_CLAIM = (
    "NO_SAMPLE_LEVEL_BILATERAL_SYNC_CLAIM"
)
PARTICIPANT_FOLDS_DISJOINT = "PARTICIPANT_FOLDS_DISJOINT"
INDEPENDENT_MATERIALIZATION_REPRODUCED = (
    "INDEPENDENT_MATERIALIZATION_REPRODUCED"
)
NO_VIDEO_SPECTRAL_OR_RESAMPLING_ARTIFACTS = (
    "NO_VIDEO_SPECTRAL_OR_RESAMPLING_ARTIFACTS"
)

GATE_CONDITIONS: tuple[str, ...] = (
    P01_AUTHORITY_DEPENDENCY_VERIFIED,
    ALL_SOURCE_ASSETS_HASH_VERIFIED,
    PARTICIPANT_INDEX_RECONCILED,
    ASSESSMENT_INDEX_RECONCILED,
    STREAM_INDEX_RECONCILED,
    ALL_SOURCE_SAMPLES_STORED_EXACTLY_ONCE,
    SOURCE_TIME_TOKENS_PRESERVED,
    STREAM_ROW_GROUP_INDEX_COMPLETE,
    SEGMENTS_PARTITION_STREAMS_EXACTLY,
    WINDOWS_NEVER_CROSS_SEGMENT_BOUNDARIES,
    WINDOW_SAMPLE_RANGES_REPLAY_EXACTLY,
    BILATERAL_TASK_PAIRS_COMPLETE,
    NO_SAMPLE_LEVEL_BILATERAL_SYNC_CLAIM,
    PARTICIPANT_FOLDS_DISJOINT,
    INDEPENDENT_MATERIALIZATION_REPRODUCED,
    NO_VIDEO_SPECTRAL_OR_RESAMPLING_ARTIFACTS,
)

REPRODUCTION_VERIFIED = "BYTE_IDENTICAL_PADS_P02_PASS"
REPRODUCTION_NOT_ATTEMPTED = "REPRODUCTION_NOT_ATTEMPTED"


class PadsP02GateError(ValueError):
    """Raised when the gate is asked to evaluate incoherent facts."""


@dataclass(frozen=True, slots=True)
class GateCondition:
    name: str
    satisfied: bool
    detail: str

    def as_record(self) -> dict[str, Any]:
        return {
            "condition": self.name,
            "satisfied": self.satisfied,
            "detail": self.detail,
        }


@dataclass(slots=True)
class PadsP02GateFacts:
    """Everything the sixteen conditions are decided from."""

    dependency_status: str = "P01_DEPENDENCY_NOT_EVALUATED"
    source_files_expected: int = 0
    source_files_hash_verified: int = 0
    source_files_failed: int = 0
    participants_expected: int = 0
    participants_materialized: int = 0
    assessments_expected: int = 0
    assessments_materialized: int = 0
    streams_expected: int = 0
    streams_materialized: int = 0
    streams_refused: int = 0
    samples_expected: int = 0
    samples_materialized: int = 0
    samples_replayed: int = 0
    duplicate_materialized_samples: int = 0
    source_time_token_failures: int = 0
    row_groups: int = 0
    streams_with_exactly_one_row_group: int = 0
    segment_partition_failures: int = 0
    windows: int = 0
    windows_crossing_segments: int = 0
    windows_checked: int = 0
    window_replay_failures: int = 0
    replay_streams_checked: int = 0
    replay_byte_exact_streams: int = 0
    bilateral_task_pairs: int = 0
    bilateral_task_pairs_expected: int = 0
    sample_level_alignment_claims: int = 0
    fold_count: int = 0
    participants_in_multiple_folds: int = 0
    participants_without_fold: int = 0
    reproduction_status: str = REPRODUCTION_NOT_ATTEMPTED
    emitted_forbidden_artifacts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateResult:
    conditions: tuple[GateCondition, ...]
    gate_status: str

    @property
    def satisfied(self) -> bool:
        return self.gate_status == GATE_PASS

    def as_record(self) -> dict[str, Any]:
        return {
            "gate_status": self.gate_status,
            "gate_conditions": [
                condition.as_record() for condition in self.conditions
            ],
            "gate_conditions_satisfied": sum(
                1 for condition in self.conditions if condition.satisfied
            ),
            "gate_conditions_total": len(self.conditions),
        }


def evaluate_gate(facts: PadsP02GateFacts) -> GateResult:
    """Evaluate every condition; all sixteen are required."""

    conditions = (
        GateCondition(
            P01_AUTHORITY_DEPENDENCY_VERIFIED,
            facts.dependency_status == DEPENDENCY_VERIFIED,
            facts.dependency_status,
        ),
        GateCondition(
            ALL_SOURCE_ASSETS_HASH_VERIFIED,
            facts.source_files_expected > 0
            and facts.source_files_hash_verified
            == facts.source_files_expected
            and facts.source_files_failed == 0,
            f"{facts.source_files_hash_verified}/"
            f"{facts.source_files_expected} verified, "
            f"{facts.source_files_failed} failed",
        ),
        GateCondition(
            PARTICIPANT_INDEX_RECONCILED,
            facts.participants_expected > 0
            and facts.participants_materialized == facts.participants_expected,
            f"{facts.participants_materialized}/"
            f"{facts.participants_expected}",
        ),
        GateCondition(
            ASSESSMENT_INDEX_RECONCILED,
            facts.assessments_expected > 0
            and facts.assessments_materialized == facts.assessments_expected,
            f"{facts.assessments_materialized}/{facts.assessments_expected}",
        ),
        GateCondition(
            STREAM_INDEX_RECONCILED,
            facts.streams_expected > 0
            and facts.streams_materialized == facts.streams_expected
            and facts.streams_refused == 0,
            f"{facts.streams_materialized}/{facts.streams_expected}, "
            f"{facts.streams_refused} refused",
        ),
        GateCondition(
            ALL_SOURCE_SAMPLES_STORED_EXACTLY_ONCE,
            facts.samples_expected > 0
            and facts.samples_materialized == facts.samples_expected
            and facts.samples_replayed == facts.samples_expected
            and facts.duplicate_materialized_samples == 0,
            f"{facts.samples_materialized}/{facts.samples_expected} stored, "
            f"{facts.samples_replayed} replayed, "
            f"{facts.duplicate_materialized_samples} duplicated",
        ),
        GateCondition(
            SOURCE_TIME_TOKENS_PRESERVED,
            facts.source_time_token_failures == 0
            and facts.replay_streams_checked > 0
            and facts.replay_byte_exact_streams
            == facts.replay_streams_checked,
            f"{facts.replay_byte_exact_streams}/"
            f"{facts.replay_streams_checked} streams replay byte exactly, "
            f"{facts.source_time_token_failures} token disagreements",
        ),
        GateCondition(
            STREAM_ROW_GROUP_INDEX_COMPLETE,
            facts.streams_materialized > 0
            and facts.row_groups == facts.streams_materialized
            and facts.streams_with_exactly_one_row_group
            == facts.streams_materialized,
            f"{facts.row_groups} row groups for "
            f"{facts.streams_materialized} streams",
        ),
        GateCondition(
            SEGMENTS_PARTITION_STREAMS_EXACTLY,
            facts.segment_partition_failures == 0,
            f"{facts.segment_partition_failures} partition failures",
        ),
        GateCondition(
            WINDOWS_NEVER_CROSS_SEGMENT_BOUNDARIES,
            facts.windows > 0 and facts.windows_crossing_segments == 0,
            f"{facts.windows_crossing_segments} of {facts.windows} windows "
            "cross a segment",
        ),
        GateCondition(
            WINDOW_SAMPLE_RANGES_REPLAY_EXACTLY,
            facts.windows_checked == facts.windows
            and facts.windows_checked > 0
            and facts.window_replay_failures == 0,
            f"{facts.windows_checked}/{facts.windows} windows replayed, "
            f"{facts.window_replay_failures} failures",
        ),
        GateCondition(
            BILATERAL_TASK_PAIRS_COMPLETE,
            facts.bilateral_task_pairs_expected > 0
            and facts.bilateral_task_pairs
            == facts.bilateral_task_pairs_expected,
            f"{facts.bilateral_task_pairs}/"
            f"{facts.bilateral_task_pairs_expected}",
        ),
        GateCondition(
            NO_SAMPLE_LEVEL_BILATERAL_SYNC_CLAIM,
            facts.sample_level_alignment_claims == 0,
            f"{facts.sample_level_alignment_claims} rows claim sample-level "
            "alignment",
        ),
        GateCondition(
            PARTICIPANT_FOLDS_DISJOINT,
            facts.fold_count > 1
            and facts.participants_in_multiple_folds == 0
            and facts.participants_without_fold == 0,
            f"{facts.fold_count} folds, "
            f"{facts.participants_in_multiple_folds} participants in more "
            f"than one, {facts.participants_without_fold} unassigned",
        ),
        GateCondition(
            INDEPENDENT_MATERIALIZATION_REPRODUCED,
            facts.reproduction_status == REPRODUCTION_VERIFIED,
            facts.reproduction_status,
        ),
        GateCondition(
            NO_VIDEO_SPECTRAL_OR_RESAMPLING_ARTIFACTS,
            not any(facts.emitted_forbidden_artifacts.values()),
            f"{sum(facts.emitted_forbidden_artifacts.values())} forbidden "
            "artifacts emitted",
        ),
    )
    names = tuple(condition.name for condition in conditions)
    if names != GATE_CONDITIONS:  # pragma: no cover - contract guard
        raise PadsP02GateError("gate condition set drifted from the contract")
    status = (
        GATE_PASS
        if all(condition.satisfied for condition in conditions)
        else GATE_NO_GO
    )
    return GateResult(conditions=conditions, gate_status=status)


def failing_conditions(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(item["condition"])
        for item in record.get("gate_conditions", ())
        if not item.get("satisfied")
    )


__all__ = [
    "ALL_SOURCE_ASSETS_HASH_VERIFIED",
    "ALL_SOURCE_SAMPLES_STORED_EXACTLY_ONCE",
    "ASSESSMENT_INDEX_RECONCILED",
    "BILATERAL_TASK_PAIRS_COMPLETE",
    "GATE_CONDITIONS",
    "INDEPENDENT_MATERIALIZATION_REPRODUCED",
    "NO_SAMPLE_LEVEL_BILATERAL_SYNC_CLAIM",
    "NO_VIDEO_SPECTRAL_OR_RESAMPLING_ARTIFACTS",
    "P01_AUTHORITY_DEPENDENCY_VERIFIED",
    "PARTICIPANT_FOLDS_DISJOINT",
    "PARTICIPANT_INDEX_RECONCILED",
    "REPRODUCTION_NOT_ATTEMPTED",
    "REPRODUCTION_VERIFIED",
    "SEGMENTS_PARTITION_STREAMS_EXACTLY",
    "SOURCE_TIME_TOKENS_PRESERVED",
    "STREAM_INDEX_RECONCILED",
    "STREAM_ROW_GROUP_INDEX_COMPLETE",
    "WINDOWS_NEVER_CROSS_SEGMENT_BOUNDARIES",
    "WINDOW_SAMPLE_RANGES_REPLAY_EXACTLY",
    "GateCondition",
    "GateResult",
    "PadsP02GateError",
    "PadsP02GateFacts",
    "evaluate_gate",
    "failing_conditions",
]
