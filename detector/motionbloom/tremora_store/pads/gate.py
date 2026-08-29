"""The PADS-P0.1 hard-gate conditions.

Two of these bind the release rather than a single file, and both were added
before the first authoritative run rather than checked by eye afterwards:
``PADS_RELEASE_STRUCTURE_RECONCILED`` proves 469 x 11 x 2 from the source
metadata, and ``PADS_INDEPENDENT_REPRODUCTION_VERIFIED`` requires two receipts
that disagree about where and by whom they ran while agreeing about what they
audited.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .authority import GATE_NO_GO, GATE_PASS, RELATIVE_TIME_BASIS
from .release_structure import RELEASE_STRUCTURE_RECONCILED
from .reproduction import REPRODUCTION_VERIFIED

PADS_ALL_SOURCE_FILES_HASH_VERIFIED = "PADS_ALL_SOURCE_FILES_HASH_VERIFIED"
PADS_RELEASE_STRUCTURE_RECONCILED = "PADS_RELEASE_STRUCTURE_RECONCILED"
PADS_EVERY_DECLARED_STREAM_PARSED = "PADS_EVERY_DECLARED_STREAM_PARSED"
PADS_ROW_COUNTS_METADATA_DIRECTED = "PADS_ROW_COUNTS_METADATA_DIRECTED"
PADS_SOURCE_TIME_IS_THE_TIMELINE = "PADS_SOURCE_TIME_IS_THE_TIMELINE"
PADS_CADENCE_AGREES_WITH_DECLARED_RATE = (
    "PADS_CADENCE_AGREES_WITH_DECLARED_RATE"
)
PADS_SOURCE_ORDER_AND_UNITS_PRESERVED = (
    "PADS_SOURCE_ORDER_AND_UNITS_PRESERVED"
)
PADS_NO_AMBIGUOUS_DECLARATION = "PADS_NO_AMBIGUOUS_DECLARATION"
PADS_DEVICE_LOCATIONS_RECOGNIZED = "PADS_DEVICE_LOCATIONS_RECOGNIZED"
PADS_NO_BLANK_ROW_DISCARDED = "PADS_NO_BLANK_ROW_DISCARDED"
PADS_USABLE_VALUES_PRESENT = "PADS_USABLE_VALUES_PRESENT"
PADS_NO_VIDEO_ASSOCIATION_EMITTED = "PADS_NO_VIDEO_ASSOCIATION_EMITTED"
PADS_INDEPENDENT_REPRODUCTION_VERIFIED = (
    "PADS_INDEPENDENT_REPRODUCTION_VERIFIED"
)
PADS_NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED = (
    "PADS_NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED"
)

GATE_CONDITIONS: tuple[str, ...] = (
    PADS_ALL_SOURCE_FILES_HASH_VERIFIED,
    PADS_RELEASE_STRUCTURE_RECONCILED,
    PADS_EVERY_DECLARED_STREAM_PARSED,
    PADS_ROW_COUNTS_METADATA_DIRECTED,
    PADS_SOURCE_TIME_IS_THE_TIMELINE,
    PADS_CADENCE_AGREES_WITH_DECLARED_RATE,
    PADS_SOURCE_ORDER_AND_UNITS_PRESERVED,
    PADS_NO_AMBIGUOUS_DECLARATION,
    PADS_DEVICE_LOCATIONS_RECOGNIZED,
    PADS_NO_BLANK_ROW_DISCARDED,
    PADS_USABLE_VALUES_PRESENT,
    PADS_NO_VIDEO_ASSOCIATION_EMITTED,
    PADS_INDEPENDENT_REPRODUCTION_VERIFIED,
    PADS_NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED,
)


class PadsGateError(ValueError):
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
class PadsGateFacts:
    """Everything the conditions are decided from."""

    source_files_expected: int = 0
    source_files_hash_verified: int = 0
    source_files_failed: int = 0
    release_structure_status: str = "RELEASE_STRUCTURE_NOT_EVALUATED"
    streams_declared: int = 0
    streams_parsed: int = 0
    streams_refused: int = 0
    row_count_mismatch_streams: int = 0
    invalid_time_streams: int = 0
    no_usable_value_streams: int = 0
    cadence_deviating_streams: int = 0
    span_deviating_streams: int = 0
    ambiguous_declaration_streams: int = 0
    unrecognized_device_location_streams: int = 0
    blank_row_streams: int = 0
    canonicalization_failures: int = 0
    video_bearing_field_count: int = 0
    relative_time_basis: str = RELATIVE_TIME_BASIS
    reproduction_status: str = "REPRODUCTION_NOT_ATTEMPTED"
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


def evaluate_gate(facts: PadsGateFacts) -> GateResult:
    """Evaluate every condition; all are required."""

    if facts.source_files_expected < 0 or facts.streams_declared < 0:
        raise PadsGateError("gate facts carry negative counts")

    conditions = (
        GateCondition(
            PADS_ALL_SOURCE_FILES_HASH_VERIFIED,
            facts.source_files_expected > 0
            and facts.source_files_hash_verified
            == facts.source_files_expected
            and facts.source_files_failed == 0,
            f"{facts.source_files_hash_verified}/"
            f"{facts.source_files_expected} verified, "
            f"{facts.source_files_failed} failed",
        ),
        GateCondition(
            PADS_RELEASE_STRUCTURE_RECONCILED,
            facts.release_structure_status == RELEASE_STRUCTURE_RECONCILED,
            facts.release_structure_status,
        ),
        GateCondition(
            PADS_EVERY_DECLARED_STREAM_PARSED,
            facts.streams_declared > 0
            and facts.streams_parsed == facts.streams_declared
            and facts.streams_refused == 0,
            f"{facts.streams_parsed}/{facts.streams_declared} parsed, "
            f"{facts.streams_refused} refused",
        ),
        GateCondition(
            PADS_ROW_COUNTS_METADATA_DIRECTED,
            facts.row_count_mismatch_streams == 0,
            f"{facts.row_count_mismatch_streams} streams disagree with their "
            "own declared row count",
        ),
        GateCondition(
            PADS_SOURCE_TIME_IS_THE_TIMELINE,
            facts.relative_time_basis == RELATIVE_TIME_BASIS
            and facts.invalid_time_streams == 0,
            f"basis {facts.relative_time_basis}, "
            f"{facts.invalid_time_streams} streams with unusable Time",
        ),
        GateCondition(
            PADS_CADENCE_AGREES_WITH_DECLARED_RATE,
            facts.cadence_deviating_streams == 0
            and facts.span_deviating_streams == 0,
            f"{facts.cadence_deviating_streams} cadence and "
            f"{facts.span_deviating_streams} span deviations",
        ),
        GateCondition(
            PADS_SOURCE_ORDER_AND_UNITS_PRESERVED,
            facts.canonicalization_failures == 0,
            f"{facts.canonicalization_failures} canonicalization failures",
        ),
        GateCondition(
            PADS_NO_AMBIGUOUS_DECLARATION,
            facts.ambiguous_declaration_streams == 0,
            f"{facts.ambiguous_declaration_streams} ambiguous declarations",
        ),
        GateCondition(
            PADS_DEVICE_LOCATIONS_RECOGNIZED,
            facts.unrecognized_device_location_streams == 0,
            f"{facts.unrecognized_device_location_streams} unrecognized "
            "device locations",
        ),
        GateCondition(
            PADS_NO_BLANK_ROW_DISCARDED,
            facts.blank_row_streams == 0,
            f"{facts.blank_row_streams} streams with a blank source row",
        ),
        GateCondition(
            PADS_USABLE_VALUES_PRESENT,
            facts.no_usable_value_streams == 0,
            f"{facts.no_usable_value_streams} streams with no usable value",
        ),
        GateCondition(
            PADS_NO_VIDEO_ASSOCIATION_EMITTED,
            facts.video_bearing_field_count == 0,
            f"{facts.video_bearing_field_count} video-bearing field names",
        ),
        GateCondition(
            PADS_INDEPENDENT_REPRODUCTION_VERIFIED,
            facts.reproduction_status == REPRODUCTION_VERIFIED,
            facts.reproduction_status,
        ),
        GateCondition(
            PADS_NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED,
            not any(facts.emitted_forbidden_artifacts.values()),
            f"{sum(facts.emitted_forbidden_artifacts.values())} forbidden "
            "artifacts emitted",
        ),
    )
    names = tuple(condition.name for condition in conditions)
    if names != GATE_CONDITIONS:  # pragma: no cover - contract guard
        raise PadsGateError("gate condition set drifted from the contract")
    status = (
        GATE_PASS
        if all(condition.satisfied for condition in conditions)
        else GATE_NO_GO
    )
    return GateResult(conditions=conditions, gate_status=status)


def failing_conditions(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the conditions a published record reports as unsatisfied."""

    return tuple(
        str(item["condition"])
        for item in record.get("gate_conditions", ())
        if not item.get("satisfied")
    )


__all__ = [
    "GATE_CONDITIONS",
    "PADS_ALL_SOURCE_FILES_HASH_VERIFIED",
    "PADS_CADENCE_AGREES_WITH_DECLARED_RATE",
    "PADS_DEVICE_LOCATIONS_RECOGNIZED",
    "PADS_EVERY_DECLARED_STREAM_PARSED",
    "PADS_INDEPENDENT_REPRODUCTION_VERIFIED",
    "PADS_NO_AMBIGUOUS_DECLARATION",
    "PADS_NO_BLANK_ROW_DISCARDED",
    "PADS_NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED",
    "PADS_NO_VIDEO_ASSOCIATION_EMITTED",
    "PADS_RELEASE_STRUCTURE_RECONCILED",
    "PADS_ROW_COUNTS_METADATA_DIRECTED",
    "PADS_SOURCE_ORDER_AND_UNITS_PRESERVED",
    "PADS_SOURCE_TIME_IS_THE_TIMELINE",
    "PADS_USABLE_VALUES_PRESENT",
    "GateCondition",
    "GateResult",
    "PadsGateError",
    "PadsGateFacts",
    "evaluate_gate",
    "failing_conditions",
]
