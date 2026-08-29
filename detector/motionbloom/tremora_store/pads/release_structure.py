"""Release-level structural reconciliation for PADS-P0.1.

The gate binds the *whole release*, not one file at a time.  PADS 1.0.0
publishes 469 participants and eleven neurologist-designed tasks recorded at
both wrists, and the release's own checksum list carries 469 observation
records, 469 patient records and 10,318 timeseries files.  This module proves
that relationship from the source metadata rather than assuming it:

    469 participants x 11 tasks           = 5,159 assessment steps
    5,159 assessment steps x 2 wrists     = 10,318 device files

The expected task *names* and the release-level totals are frozen.  Individual
sample counts are not: every session is validated against its own declared
``rows``, and there is no task-name row-count allowlist anywhere in the
implementation.  A malformed or incomplete release is evidence for NO-GO, never
for BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .movement import Observation, Participant

PADS_RELEASE_VERSION = "1.0.0"
PADS_RELEASE_CONTRACT_VERSION = "tremora-pads-release-structure-1.0.0"

#: Frozen task names, in the order the release declares them.
PADS_EXPECTED_TASKS: tuple[str, ...] = (
    "Relaxed",
    "RelaxedTask",
    "StretchHold",
    "LiftHold",
    "HoldWeight",
    "PointFinger",
    "DrinkGlas",
    "CrossArms",
    "TouchIndex",
    "TouchNose",
    "Entrainment",
)

PADS_EXPECTED_DEVICE_LOCATIONS: tuple[str, ...] = ("LeftWrist", "RightWrist")

PADS_EXPECTED_PARTICIPANTS = 469
PADS_EXPECTED_ASSESSMENTS = (
    PADS_EXPECTED_PARTICIPANTS * len(PADS_EXPECTED_TASKS)
)
PADS_EXPECTED_STREAMS = (
    PADS_EXPECTED_ASSESSMENTS * len(PADS_EXPECTED_DEVICE_LOCATIONS)
)

RELEASE_STRUCTURE_RECONCILED = "PADS_RELEASE_STRUCTURE_RECONCILED"

MISSING_PARTICIPANT_METADATA = "MISSING_PARTICIPANT_METADATA"
MISSING_TASK = "MISSING_TASK"
DUPLICATE_TASK = "DUPLICATE_TASK"
UNKNOWN_EXTRA_TASK = "UNKNOWN_EXTRA_TASK"
MISSING_DEVICE_RECORD = "MISSING_DEVICE_RECORD"
DUPLICATE_DEVICE_RECORD = "DUPLICATE_DEVICE_RECORD"
MISSING_REFERENCED_FILE = "MISSING_REFERENCED_FILE"
PATH_TRAVERSAL = "PATH_TRAVERSAL"
PARTICIPANT_TASK_DEVICE_MISMATCH = "PARTICIPANT_TASK_DEVICE_MISMATCH"
PARTICIPANT_COUNT_MISMATCH = "PARTICIPANT_COUNT_MISMATCH"
ASSESSMENT_COUNT_MISMATCH = "ASSESSMENT_COUNT_MISMATCH"
STREAM_COUNT_MISMATCH = "STREAM_COUNT_MISMATCH"


@dataclass(frozen=True, slots=True)
class StructureFailure:
    """One structural defect, named and located."""

    code: str
    subject: str
    detail: str

    def as_record(self) -> dict[str, Any]:
        return {"code": self.code, "subject": self.subject,
                "detail": self.detail}


@dataclass(slots=True)
class ReleaseStructureResult:
    """What the release actually contains, against what it must contain."""

    observed_participants: int = 0
    observed_assessments: int = 0
    observed_streams: int = 0
    expected_participants: int = PADS_EXPECTED_PARTICIPANTS
    expected_assessments: int = PADS_EXPECTED_ASSESSMENTS
    expected_streams: int = PADS_EXPECTED_STREAMS
    tasks_observed: tuple[str, ...] = ()
    failures: list[StructureFailure] = field(default_factory=list)

    @property
    def reconciled(self) -> bool:
        return not self.failures

    @property
    def status(self) -> str:
        return (
            RELEASE_STRUCTURE_RECONCILED
            if self.reconciled
            else self.failures[0].code
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "release_version": PADS_RELEASE_VERSION,
            "release_contract_version": PADS_RELEASE_CONTRACT_VERSION,
            "release_structure_status": self.status,
            "expected_participants": self.expected_participants,
            "observed_participants": self.observed_participants,
            "expected_assessments": self.expected_assessments,
            "observed_assessments": self.observed_assessments,
            "expected_streams": self.expected_streams,
            "observed_streams": self.observed_streams,
            "expected_tasks": list(PADS_EXPECTED_TASKS),
            "tasks_observed": list(self.tasks_observed),
            "failure_count": len(self.failures),
            "failures": [failure.as_record() for failure in self.failures[:64]],
        }


def expected_stream_stem(
    participant_id: str, task_name: str, device_location: str
) -> str:
    """The release names each device file for exactly what it contains."""

    return f"{participant_id}_{task_name}_{device_location}"


def reconcile_release_structure(
    observations: Mapping[str, Observation],
    participants: Mapping[str, Participant],
    *,
    file_exists: Callable[[str], bool],
    expected_participants: int = PADS_EXPECTED_PARTICIPANTS,
    expected_tasks: tuple[str, ...] = PADS_EXPECTED_TASKS,
    expected_device_locations: tuple[str, ...] = (
        PADS_EXPECTED_DEVICE_LOCATIONS
    ),
) -> ReleaseStructureResult:
    """Reconcile the whole release against its frozen structure."""

    expected_assessments = expected_participants * len(expected_tasks)
    expected_streams = expected_assessments * len(expected_device_locations)
    result = ReleaseStructureResult(
        observed_participants=len(observations),
        expected_participants=expected_participants,
        expected_assessments=expected_assessments,
        expected_streams=expected_streams,
    )
    observed_tasks: set[str] = set()

    for participant_id in sorted(observations):
        observation = observations[participant_id]
        if participant_id not in participants:
            result.failures.append(StructureFailure(
                MISSING_PARTICIPANT_METADATA,
                participant_id,
                "no patient record for this participant",
            ))

        seen_tasks: dict[str, int] = {}
        for session in observation.sessions:
            observed_tasks.add(session.record_name)
            seen_tasks[session.record_name] = (
                seen_tasks.get(session.record_name, 0) + 1
            )
            if session.record_name not in expected_tasks:
                result.failures.append(StructureFailure(
                    UNKNOWN_EXTRA_TASK,
                    f"{participant_id}:{session.record_name}",
                    "task is not one of the eleven published structures",
                ))
                continue
            result.observed_assessments += 1

            seen_devices: dict[str, int] = {}
            for stream in session.streams:
                seen_devices[stream.device_location] = (
                    seen_devices.get(stream.device_location, 0) + 1
                )
                result.observed_streams += 1
                subject = (
                    f"{participant_id}:{session.record_name}:"
                    f"{stream.device_location}"
                )
                path = PurePosixPath(stream.file_name)
                if any(part in {"..", ""} for part in path.parts) or (
                    path.is_absolute()
                ):
                    result.failures.append(StructureFailure(
                        PATH_TRAVERSAL, subject, stream.file_name))
                    continue
                if path.stem != expected_stream_stem(
                    participant_id, session.record_name,
                    stream.device_location,
                ):
                    result.failures.append(StructureFailure(
                        PARTICIPANT_TASK_DEVICE_MISMATCH,
                        subject,
                        f"{stream.file_name} does not name this record",
                    ))
                    continue
                if not file_exists(stream.file_name):
                    result.failures.append(StructureFailure(
                        MISSING_REFERENCED_FILE, subject, stream.file_name))

            for device in expected_device_locations:
                count = seen_devices.get(device, 0)
                if count == 0:
                    result.failures.append(StructureFailure(
                        MISSING_DEVICE_RECORD,
                        f"{participant_id}:{session.record_name}:{device}",
                        "the session declares no record for this wrist",
                    ))
                elif count > 1:
                    result.failures.append(StructureFailure(
                        DUPLICATE_DEVICE_RECORD,
                        f"{participant_id}:{session.record_name}:{device}",
                        f"{count} records for one wrist",
                    ))
            for device in sorted(set(seen_devices) - set(
                expected_device_locations
            )):
                result.failures.append(StructureFailure(
                    MISSING_DEVICE_RECORD,
                    f"{participant_id}:{session.record_name}:{device}",
                    "unrecognized device location in the session",
                ))

        for task, count in sorted(seen_tasks.items()):
            if count > 1 and task in expected_tasks:
                result.failures.append(StructureFailure(
                    DUPLICATE_TASK,
                    f"{participant_id}:{task}",
                    f"{count} sessions declare this task",
                ))
        for task in expected_tasks:
            if task not in seen_tasks:
                result.failures.append(StructureFailure(
                    MISSING_TASK,
                    f"{participant_id}:{task}",
                    "the participant has no session for this task",
                ))

    result.tasks_observed = tuple(sorted(observed_tasks))

    if result.observed_participants != expected_participants:
        result.failures.insert(0, StructureFailure(
            PARTICIPANT_COUNT_MISMATCH,
            "release",
            f"{result.observed_participants} of {expected_participants}",
        ))
    if result.observed_assessments != expected_assessments:
        result.failures.insert(0, StructureFailure(
            ASSESSMENT_COUNT_MISMATCH,
            "release",
            f"{result.observed_assessments} of {expected_assessments}",
        ))
    if result.observed_streams != expected_streams:
        result.failures.insert(0, StructureFailure(
            STREAM_COUNT_MISMATCH,
            "release",
            f"{result.observed_streams} of {expected_streams}",
        ))
    return result


__all__ = [
    "ASSESSMENT_COUNT_MISMATCH",
    "DUPLICATE_DEVICE_RECORD",
    "DUPLICATE_TASK",
    "MISSING_DEVICE_RECORD",
    "MISSING_PARTICIPANT_METADATA",
    "MISSING_REFERENCED_FILE",
    "MISSING_TASK",
    "PADS_EXPECTED_ASSESSMENTS",
    "PADS_EXPECTED_DEVICE_LOCATIONS",
    "PADS_EXPECTED_PARTICIPANTS",
    "PADS_EXPECTED_STREAMS",
    "PADS_EXPECTED_TASKS",
    "PADS_RELEASE_CONTRACT_VERSION",
    "PADS_RELEASE_VERSION",
    "PARTICIPANT_COUNT_MISMATCH",
    "PARTICIPANT_TASK_DEVICE_MISMATCH",
    "PATH_TRAVERSAL",
    "RELEASE_STRUCTURE_RECONCILED",
    "STREAM_COUNT_MISMATCH",
    "UNKNOWN_EXTRA_TASK",
    "ReleaseStructureResult",
    "StructureFailure",
    "expected_stream_stem",
    "reconcile_release_structure",
]
