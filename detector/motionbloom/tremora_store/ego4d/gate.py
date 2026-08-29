"""The eleven E4D-P0.1 hard-gate conditions.

Three of these are only as strong as what they are measured against, and each
is checked against something the audit did not derive from the thing under
test:

* row representation compares the authority rows against a data-line count
  taken from the split records before any row object exists.  A check that
  compares a list against its own length can only ever pass;
* row loss is checked as a gapless zero-anchored ordinal sequence over the
  whole file.  Anchoring on the first surviving ordinal, or grouping by
  component, hides a lost prefix, a lost suffix or a lost whole component --
  and, because a normalized Ego4D CSV may interleave components, grouping by
  component also raises a false no-go on a perfectly intact file;
* reproducibility requires a second report that identifies itself as a P0.1
  record of the same schema, produced by the same implementation hashes, over
  the same metadata snapshot, *and* naming a different publication
  destination.  A bare evidence hash is not evidence that another root ran.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .authority import GATE_NO_GO, GATE_PASS

ALL_ASSETS_HASH_VERIFIED = "ALL_ASSETS_HASH_VERIFIED"
EVERY_IMU_ROW_REPRESENTED = "EVERY_IMU_ROW_REPRESENTED"
SOURCE_TIMESTAMP_TOKENS_PRESERVED = "SOURCE_TIMESTAMP_TOKENS_PRESERVED"
NO_MISSING_TIMESTAMP_INFERRED = "NO_MISSING_TIMESTAMP_INFERRED"
EVERY_SELECTED_VIDEO_HAS_PTS_TIMELINE = (
    "EVERY_SELECTED_VIDEO_HAS_PTS_TIMELINE"
)
VALID_CANONICAL_ROWS_INSIDE_VIDEO_INTERVAL = (
    "VALID_CANONICAL_ROWS_INSIDE_VIDEO_INTERVAL"
)
KNOWN_ISSUE_ROWS_VISIBLE_AND_CLASSIFIED = (
    "KNOWN_ISSUE_ROWS_VISIBLE_AND_CLASSIFIED"
)
NO_ROW_DROPPED_FOR_NON_MONOTONICITY = "NO_ROW_DROPPED_FOR_NON_MONOTONICITY"
AUDIT_REPRODUCES_BYTE_IDENTICALLY = "AUDIT_REPRODUCES_BYTE_IDENTICALLY"
SUBSET_FLOORS_SATISFIED = "SUBSET_FLOORS_SATISFIED"
NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED = "NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED"

#: All eleven are required.  Order is frozen so a published report can be
#: compared condition by condition across runs.
GATE_CONDITIONS: tuple[str, ...] = (
    ALL_ASSETS_HASH_VERIFIED,
    EVERY_IMU_ROW_REPRESENTED,
    SOURCE_TIMESTAMP_TOKENS_PRESERVED,
    NO_MISSING_TIMESTAMP_INFERRED,
    EVERY_SELECTED_VIDEO_HAS_PTS_TIMELINE,
    VALID_CANONICAL_ROWS_INSIDE_VIDEO_INTERVAL,
    KNOWN_ISSUE_ROWS_VISIBLE_AND_CLASSIFIED,
    NO_ROW_DROPPED_FOR_NON_MONOTONICITY,
    AUDIT_REPRODUCES_BYTE_IDENTICALLY,
    SUBSET_FLOORS_SATISFIED,
    NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED,
)

REPRODUCTION_VERIFIED = "REPRODUCTION_VERIFIED"
REPRODUCTION_NOT_ATTEMPTED = "REPRODUCTION_NOT_ATTEMPTED"
REPRODUCTION_SAME_DESTINATION = "REPRODUCTION_SAME_DESTINATION"
REPRODUCTION_IDENTITY_MISMATCH = "REPRODUCTION_IDENTITY_MISMATCH"
REPRODUCTION_EVIDENCE_MISMATCH = "REPRODUCTION_EVIDENCE_MISMATCH"

_REPRODUCTION_IDENTITY_FIELDS = (
    "artifact_kind",
    "contract_version",
    "implementation_version",
    "metadata_snapshot_sha256",
    "schema_version",
)


class Ego4DGateError(ValueError):
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
class Ego4DGateFacts:
    """Everything the eleven conditions are decided from."""

    assets_expected: int = 0
    assets_hash_verified: int = 0
    assets_failed: int = 0
    data_line_total: int = 0
    authority_row_total: int = 0
    token_preservation_failures: int = 0
    inferred_timestamp_count: int = 0
    selected_video_count: int = 0
    videos_with_pts_timeline: int = 0
    videos_with_timeline_disagreement: int = 0
    valid_rows_outside_video_interval: int = 0
    unclassified_issue_rows: int = 0
    files_with_ordinal_gaps: int = 0
    reproduction_status: str = REPRODUCTION_NOT_ATTEMPTED
    subset_floors_satisfied: bool = False
    subset_shortfalls: dict[str, float] = field(default_factory=dict)
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


def verify_reproduction(
    record: Mapping[str, Any],
    other: Mapping[str, Any] | None,
) -> str:
    """Decide whether ``other`` is a genuine second execution of ``record``.

    Identity fields alone do not distinguish a second execution from this run's
    own record copied to a new filename, so a different publication destination
    is also required.  Like the v0.5D execution receipts, this establishes two
    executions under the trusted procedure; it is not cryptographic remote
    attestation and does not defend against a deliberately forged record.
    """

    if other is None:
        return REPRODUCTION_NOT_ATTEMPTED
    for key in _REPRODUCTION_IDENTITY_FIELDS:
        if record.get(key) != other.get(key) or record.get(key) is None:
            return REPRODUCTION_IDENTITY_MISMATCH
    evidence = record.get("canonical_evidence_sha256")
    if not evidence or evidence != other.get("canonical_evidence_sha256"):
        return REPRODUCTION_EVIDENCE_MISMATCH
    destination = record.get("publication_destination")
    if not destination or destination == other.get("publication_destination"):
        return REPRODUCTION_SAME_DESTINATION
    return REPRODUCTION_VERIFIED


def evaluate_gate(facts: Ego4DGateFacts) -> GateResult:
    """Evaluate all eleven conditions; every one is required."""

    if facts.assets_expected < 0 or facts.authority_row_total < 0:
        raise Ego4DGateError("gate facts carry negative counts")

    conditions = (
        GateCondition(
            ALL_ASSETS_HASH_VERIFIED,
            facts.assets_expected > 0
            and facts.assets_hash_verified == facts.assets_expected
            and facts.assets_failed == 0,
            f"{facts.assets_hash_verified}/{facts.assets_expected} verified, "
            f"{facts.assets_failed} failed",
        ),
        GateCondition(
            EVERY_IMU_ROW_REPRESENTED,
            facts.authority_row_total == facts.data_line_total,
            f"{facts.authority_row_total} rows against "
            f"{facts.data_line_total} source data lines",
        ),
        GateCondition(
            SOURCE_TIMESTAMP_TOKENS_PRESERVED,
            facts.token_preservation_failures == 0,
            f"{facts.token_preservation_failures} tokens not preserved",
        ),
        GateCondition(
            NO_MISSING_TIMESTAMP_INFERRED,
            facts.inferred_timestamp_count == 0,
            f"{facts.inferred_timestamp_count} inferred timestamps",
        ),
        GateCondition(
            EVERY_SELECTED_VIDEO_HAS_PTS_TIMELINE,
            facts.selected_video_count > 0
            and facts.videos_with_pts_timeline == facts.selected_video_count
            and facts.videos_with_timeline_disagreement == 0,
            f"{facts.videos_with_pts_timeline}/{facts.selected_video_count} "
            f"decoded, {facts.videos_with_timeline_disagreement} disagree",
        ),
        GateCondition(
            VALID_CANONICAL_ROWS_INSIDE_VIDEO_INTERVAL,
            facts.valid_rows_outside_video_interval == 0,
            f"{facts.valid_rows_outside_video_interval} valid rows outside "
            "the documented video interval",
        ),
        GateCondition(
            KNOWN_ISSUE_ROWS_VISIBLE_AND_CLASSIFIED,
            facts.unclassified_issue_rows == 0,
            f"{facts.unclassified_issue_rows} issue rows unclassified",
        ),
        GateCondition(
            NO_ROW_DROPPED_FOR_NON_MONOTONICITY,
            facts.files_with_ordinal_gaps == 0,
            f"{facts.files_with_ordinal_gaps} files with a gapped or "
            "non-zero-anchored source ordinal sequence",
        ),
        GateCondition(
            AUDIT_REPRODUCES_BYTE_IDENTICALLY,
            facts.reproduction_status == REPRODUCTION_VERIFIED,
            facts.reproduction_status,
        ),
        GateCondition(
            SUBSET_FLOORS_SATISFIED,
            facts.subset_floors_satisfied and not facts.subset_shortfalls,
            "shortfalls: "
            f"{sorted(facts.subset_shortfalls)!r}"
            if facts.subset_shortfalls else "no shortfall",
        ),
        GateCondition(
            NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED,
            not any(facts.emitted_forbidden_artifacts.values()),
            f"{sum(facts.emitted_forbidden_artifacts.values())} forbidden "
            "artifacts emitted",
        ),
    )
    names = tuple(condition.name for condition in conditions)
    if names != GATE_CONDITIONS:  # pragma: no cover - contract guard
        raise Ego4DGateError("gate condition set drifted from the contract")
    status = (
        GATE_PASS
        if all(condition.satisfied for condition in conditions)
        else GATE_NO_GO
    )
    return GateResult(conditions=conditions, gate_status=status)


def ordinal_sequence_is_intact(ordinals: Sequence[int], expected: int) -> bool:
    """Zero-anchored, gapless, and exactly as long as the source file.

    Loss at the end leaves a gapless sequence, which is why the row-count
    condition exists alongside this one.
    """

    if len(ordinals) != expected:
        return False
    return list(ordinals) == list(range(expected))


__all__ = [
    "ALL_ASSETS_HASH_VERIFIED",
    "AUDIT_REPRODUCES_BYTE_IDENTICALLY",
    "EVERY_IMU_ROW_REPRESENTED",
    "EVERY_SELECTED_VIDEO_HAS_PTS_TIMELINE",
    "GATE_CONDITIONS",
    "KNOWN_ISSUE_ROWS_VISIBLE_AND_CLASSIFIED",
    "NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED",
    "NO_MISSING_TIMESTAMP_INFERRED",
    "NO_ROW_DROPPED_FOR_NON_MONOTONICITY",
    "REPRODUCTION_EVIDENCE_MISMATCH",
    "REPRODUCTION_IDENTITY_MISMATCH",
    "REPRODUCTION_NOT_ATTEMPTED",
    "REPRODUCTION_SAME_DESTINATION",
    "REPRODUCTION_VERIFIED",
    "SOURCE_TIMESTAMP_TOKENS_PRESERVED",
    "SUBSET_FLOORS_SATISFIED",
    "VALID_CANONICAL_ROWS_INSIDE_VIDEO_INTERVAL",
    "Ego4DGateError",
    "Ego4DGateFacts",
    "GateCondition",
    "GateResult",
    "evaluate_gate",
    "ordinal_sequence_is_intact",
    "verify_reproduction",
]
