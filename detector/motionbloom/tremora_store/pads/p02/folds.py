"""Participant-disjoint outer folds, frozen before any downstream work.

The split group is the participant, so all 22 of a participant's device streams
land in one fold and no window from a participant can appear on both sides of a
future evaluation.

Assignment is deterministic and stratified: participants are grouped by the
source condition category, ordered inside each group by a keyed digest of the
seed and the participant id, then dealt round-robin across the folds.  No random
number generator is involved, so two clean roots agree without shared state.

P0.2 assigns outer-fold identity only.  Train, validation and test labels belong
to whatever milestone actually trains something.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping

from .contract import N_OUTER_FOLDS, SPLIT_ALGORITHM_VERSION, SPLIT_SEED

_FOLD_DOMAIN = "tremora-pads-p02-participant-fold-1"

PARTICIPANT_ASSIGNED = "PARTICIPANT_ASSIGNED"


class FoldError(ValueError):
    """Raised when folds would not be participant-disjoint."""


def fold_key(
    participant_id: str,
    *,
    split_seed: int = SPLIT_SEED,
    algorithm_version: str = SPLIT_ALGORITHM_VERSION,
) -> str:
    """The keyed digest that orders one participant inside its group."""

    payload = "\x1f".join((
        _FOLD_DOMAIN, algorithm_version, str(split_seed), participant_id
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_folds(
    condition_by_participant: Mapping[str, str],
    *,
    n_outer_folds: int = N_OUTER_FOLDS,
    split_seed: int = SPLIT_SEED,
    algorithm_version: str = SPLIT_ALGORITHM_VERSION,
) -> dict[str, int]:
    """Deal participants round-robin across folds, stratified by condition."""

    if n_outer_folds < 2:
        raise FoldError("at least two outer folds are required")
    grouped: dict[str, list[str]] = defaultdict(list)
    for participant_id, condition in condition_by_participant.items():
        grouped[str(condition)].append(participant_id)

    assignment: dict[str, int] = {}
    for condition in sorted(grouped):
        members = sorted(
            grouped[condition],
            key=lambda participant: (
                fold_key(
                    participant,
                    split_seed=split_seed,
                    algorithm_version=algorithm_version,
                ),
                participant,
            ),
        )
        for index, participant_id in enumerate(members):
            assignment[participant_id] = index % n_outer_folds
    return assignment


def assert_participant_disjoint(
    assignment: Mapping[str, int],
    stream_participants: Mapping[str, str],
) -> None:
    """No participant in two folds, and no stream split across folds."""

    folds_by_participant: dict[str, set[int]] = defaultdict(set)
    for participant_id, fold in assignment.items():
        folds_by_participant[participant_id].add(int(fold))
    offending = sorted(
        participant for participant, folds in folds_by_participant.items()
        if len(folds) != 1
    )
    if offending:
        raise FoldError(f"participants in multiple folds: {offending!r}")
    missing = sorted(
        participant for participant in set(stream_participants.values())
        if participant not in assignment
    )
    if missing:
        raise FoldError(f"streams without a fold: {missing!r}")


def fold_sizes(assignment: Mapping[str, int]) -> dict[int, int]:
    sizes: dict[int, int] = defaultdict(int)
    for fold in assignment.values():
        sizes[int(fold)] += 1
    return dict(sorted(sizes.items()))


__all__ = [
    "PARTICIPANT_ASSIGNED",
    "FoldError",
    "assert_participant_disjoint",
    "assign_folds",
    "fold_key",
    "fold_sizes",
]
