"""Deterministic stratified selection of the E4D-P0.1 benchmark subset.

Auditing all available normalized IMU metadata is reasonable; decoding the
whole video collection is not.  The subset is frozen before any storage
performance is measured, from metadata alone, as a pure function of the
selection seed, the algorithm version and the metadata-snapshot hash.

No random number generator is involved: candidate order comes from a keyed
SHA-256 digest, so two clean roots agree without shared state and input order
cannot change the result.  A stratum the source cannot supply is named in
``strata_absent_in_source`` and given a non-zero shortfall; it is never topped
up from another stratum, because a fabricated stratum destroys the meaning of
the stratification.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .authority import (
    MINIMUM_CAPTURE_DEVICE_GROUPS,
    MINIMUM_PAIRED_COVERAGE_HOURS,
    MINIMUM_SELECTED_VIDEOS,
    SELECTION_ALGORITHM_VERSION,
    SELECTION_SEED,
)
from .coverage import hours

STRATUM_CLEAN_MONOTONIC = "CLEAN_MONOTONIC"
STRATUM_NONMONOTONIC_SOURCE_ORDER = "NONMONOTONIC_SOURCE_ORDER"
STRATUM_NULL_CANONICAL_TIMES = "NULL_CANONICAL_TIMES"
STRATUM_PARTIAL_COMPONENT_COVERAGE = "PARTIAL_COMPONENT_COVERAGE"
STRATUM_MISSING_ACCELERATION = "MISSING_ACCELERATION"
STRATUM_EXTREME_TIMESTAMP = "EXTREME_TIMESTAMP"

#: Frozen stratum order.  It fixes which stratum claims a video that carries
#: several, so selection stays a pure function of the snapshot.
STRATA: tuple[str, ...] = (
    STRATUM_CLEAN_MONOTONIC,
    STRATUM_NONMONOTONIC_SOURCE_ORDER,
    STRATUM_NULL_CANONICAL_TIMES,
    STRATUM_PARTIAL_COMPONENT_COVERAGE,
    STRATUM_MISSING_ACCELERATION,
    STRATUM_EXTREME_TIMESTAMP,
)

_SELECTION_DOMAIN = "tremora-ego4d-subset-selection-1"


class Ego4DSelectionError(ValueError):
    """Raised when a subset would be selected from unusable inputs."""


@dataclass(frozen=True, slots=True)
class VideoCandidate:
    """One video's selection facts, derived from metadata and parsed rows."""

    video_uid: str
    strata: frozenset[str]
    paired_coverage_ms: float
    capture_device_group: str


@dataclass(frozen=True, slots=True)
class SubsetSelection:
    """The frozen subset and the floors it did or did not satisfy."""

    selection_seed: int
    selection_algorithm_version: str
    metadata_snapshot_sha256: str
    selected_video_uids: tuple[str, ...]
    selected_video_count: int
    paired_coverage_hours: float
    capture_device_groups: tuple[str, ...]
    strata_present_in_source: tuple[str, ...]
    strata_absent_in_source: tuple[str, ...]
    strata_represented: tuple[str, ...]
    shortfalls: dict[str, float]
    floors_satisfied: bool
    #: The floors actually applied.  They are published inside the evidence
    #: block, so a run that lowered them cannot look like one that did not.
    minimum_videos: int
    minimum_coverage_hours: float
    minimum_capture_device_groups: int

    def as_record(self) -> dict[str, Any]:
        return {
            "selection_seed": self.selection_seed,
            "selection_algorithm_version": self.selection_algorithm_version,
            "metadata_snapshot_sha256": self.metadata_snapshot_sha256,
            "selected_video_count": self.selected_video_count,
            "selected_video_uids": list(self.selected_video_uids),
            "paired_coverage_hours": self.paired_coverage_hours,
            "capture_device_groups": list(self.capture_device_groups),
            "strata_present_in_source": list(self.strata_present_in_source),
            "strata_absent_in_source": list(self.strata_absent_in_source),
            "strata_represented": list(self.strata_represented),
            "shortfalls": dict(sorted(self.shortfalls.items())),
            "floors_satisfied": self.floors_satisfied,
            "floors": {
                "minimum_videos": self.minimum_videos,
                "minimum_coverage_hours": self.minimum_coverage_hours,
                "minimum_capture_device_groups": (
                    self.minimum_capture_device_groups
                ),
            },
        }


def selection_key(
    video_uid: str,
    *,
    metadata_snapshot_sha256: str,
    selection_seed: int = SELECTION_SEED,
    selection_algorithm_version: str = SELECTION_ALGORITHM_VERSION,
) -> str:
    """Return the keyed digest that orders one candidate."""

    payload = "\x1f".join((
        _SELECTION_DOMAIN,
        selection_algorithm_version,
        str(selection_seed),
        metadata_snapshot_sha256,
        video_uid,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ordered(
    candidates: Iterable[VideoCandidate],
    *,
    metadata_snapshot_sha256: str,
    selection_seed: int,
    selection_algorithm_version: str,
) -> tuple[VideoCandidate, ...]:
    keyed = [
        (
            selection_key(
                candidate.video_uid,
                metadata_snapshot_sha256=metadata_snapshot_sha256,
                selection_seed=selection_seed,
                selection_algorithm_version=selection_algorithm_version,
            ),
            candidate.video_uid,
            candidate,
        )
        for candidate in candidates
    ]
    keyed.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in keyed)


def select_subset(
    candidates: Sequence[VideoCandidate],
    *,
    metadata_snapshot_sha256: str,
    selection_seed: int = SELECTION_SEED,
    selection_algorithm_version: str = SELECTION_ALGORITHM_VERSION,
    minimum_videos: int = MINIMUM_SELECTED_VIDEOS,
    minimum_coverage_hours: float = MINIMUM_PAIRED_COVERAGE_HOURS,
    minimum_capture_device_groups: int = MINIMUM_CAPTURE_DEVICE_GROUPS,
) -> SubsetSelection:
    """Freeze the benchmark subset from metadata alone."""

    uids = [candidate.video_uid for candidate in candidates]
    if len(set(uids)) != len(uids):
        raise Ego4DSelectionError("candidate list repeats a video_uid")
    unknown = sorted({
        stratum
        for candidate in candidates
        for stratum in candidate.strata
        if stratum not in STRATA
    })
    if unknown:
        raise Ego4DSelectionError(f"unknown strata {unknown!r}")

    ordered = _ordered(
        candidates,
        metadata_snapshot_sha256=metadata_snapshot_sha256,
        selection_seed=selection_seed,
        selection_algorithm_version=selection_algorithm_version,
    )
    present = tuple(
        stratum
        for stratum in STRATA
        if any(stratum in candidate.strata for candidate in ordered)
    )
    absent = tuple(stratum for stratum in STRATA if stratum not in present)

    selected: list[VideoCandidate] = []
    chosen: set[str] = set()
    # Represent every stratum the source can supply before extending, so a
    # rare stratum cannot be crowded out by the coverage floor.
    for stratum in present:
        for candidate in ordered:
            if candidate.video_uid in chosen:
                continue
            if stratum in candidate.strata:
                selected.append(candidate)
                chosen.add(candidate.video_uid)
                break

    def totals() -> tuple[int, float, set[str]]:
        return (
            len(selected),
            hours(sum(item.paired_coverage_ms for item in selected)),
            {item.capture_device_group for item in selected},
        )

    for candidate in ordered:
        count, coverage_hours, groups = totals()
        if (
            count >= minimum_videos
            and coverage_hours >= minimum_coverage_hours
            and len(groups) >= minimum_capture_device_groups
        ):
            break
        if candidate.video_uid in chosen:
            continue
        selected.append(candidate)
        chosen.add(candidate.video_uid)

    count, coverage_hours, groups = totals()
    represented = tuple(
        stratum
        for stratum in STRATA
        if any(stratum in candidate.strata for candidate in selected)
    )
    shortfalls: dict[str, float] = {}
    if count < minimum_videos:
        shortfalls["selected_videos"] = float(minimum_videos - count)
    if coverage_hours < minimum_coverage_hours:
        shortfalls["paired_coverage_hours"] = (
            minimum_coverage_hours - coverage_hours
        )
    if len(groups) < minimum_capture_device_groups:
        shortfalls["capture_device_groups"] = float(
            minimum_capture_device_groups - len(groups)
        )
    for stratum in absent:
        # A stratum the source cannot supply is a non-zero shortfall, never a
        # substitution from another stratum.
        shortfalls[f"stratum:{stratum}"] = 1.0
    for stratum in present:
        if stratum not in represented:  # pragma: no cover - defensive
            shortfalls[f"stratum:{stratum}"] = 1.0

    return SubsetSelection(
        selection_seed=selection_seed,
        selection_algorithm_version=selection_algorithm_version,
        metadata_snapshot_sha256=metadata_snapshot_sha256,
        selected_video_uids=tuple(item.video_uid for item in selected),
        selected_video_count=count,
        paired_coverage_hours=coverage_hours,
        capture_device_groups=tuple(sorted(groups)),
        strata_present_in_source=present,
        strata_absent_in_source=absent,
        strata_represented=represented,
        shortfalls=shortfalls,
        floors_satisfied=not shortfalls,
        minimum_videos=minimum_videos,
        minimum_coverage_hours=minimum_coverage_hours,
        minimum_capture_device_groups=minimum_capture_device_groups,
    )


__all__ = [
    "STRATA",
    "STRATUM_CLEAN_MONOTONIC",
    "STRATUM_EXTREME_TIMESTAMP",
    "STRATUM_MISSING_ACCELERATION",
    "STRATUM_NONMONOTONIC_SOURCE_ORDER",
    "STRATUM_NULL_CANONICAL_TIMES",
    "STRATUM_PARTIAL_COMPONENT_COVERAGE",
    "Ego4DSelectionError",
    "SubsetSelection",
    "VideoCandidate",
    "select_subset",
    "selection_key",
]
