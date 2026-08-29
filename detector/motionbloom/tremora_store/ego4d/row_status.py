"""Row verdicts and issue bits for Ego4D authority rows.

``canonical_authority_status`` is a single-valued verdict resolved by a frozen
precedence chain.  Because one row can exhibit several conditions and a single
verdict can show only one, every observed condition is additionally retained in
``issue_bits``: precedence decides what a row is *called*, never what is
*recorded*.

VALID and eligible are the same set by construction.  Every disqualifying issue
bit has an entry in the precedence chain, and the import-time assertion below
fails the module if one does not -- so a row the audit will not build an index
on can never be labelled valid or counted in ``canonical_rows_valid``.
"""

from __future__ import annotations

from enum import IntFlag

SOURCE_CANONICAL_VALID = "SOURCE_CANONICAL_VALID"


class IssueBit(IntFlag):
    """Every condition observed on one source row."""

    COMPONENT_NOT_COVERED = 1 << 0
    SOURCE_CANONICAL_NONFINITE = 1 << 1
    SOURCE_CANONICAL_NULL_AFTER_TRIM = 1 << 2
    SOURCE_CANONICAL_OUTSIDE_VIDEO = 1 << 3
    SOURCE_CANONICAL_NONMONOTONIC = 1 << 4
    SOURCE_CANONICAL_DUPLICATE = 1 << 5
    SOURCE_CANONICAL_EXTREME_MAGNITUDE = 1 << 6
    SOURCE_CANONICAL_UNPARSEABLE_TOKEN = 1 << 7
    MISSING_ACCELERATION = 1 << 8
    MISSING_GYROSCOPE = 1 << 9


#: Highest priority first.  Frozen: reordering it changes what every row in
#: every published report is called.
STATUS_PRECEDENCE: tuple[IssueBit, ...] = (
    IssueBit.COMPONENT_NOT_COVERED,
    IssueBit.SOURCE_CANONICAL_NONFINITE,
    IssueBit.SOURCE_CANONICAL_NULL_AFTER_TRIM,
    IssueBit.SOURCE_CANONICAL_OUTSIDE_VIDEO,
    IssueBit.SOURCE_CANONICAL_NONMONOTONIC,
    IssueBit.SOURCE_CANONICAL_DUPLICATE,
    IssueBit.SOURCE_CANONICAL_EXTREME_MAGNITUDE,
    IssueBit.SOURCE_CANONICAL_UNPARSEABLE_TOKEN,
    IssueBit.MISSING_ACCELERATION,
    IssueBit.MISSING_GYROSCOPE,
)

STATUS_NAMES: tuple[str, ...] = (
    *(bit.name or "" for bit in STATUS_PRECEDENCE),
    SOURCE_CANONICAL_VALID,
)

_MISSING_FROM_PRECEDENCE = sorted(
    bit.name or "" for bit in IssueBit if bit not in STATUS_PRECEDENCE
)
if _MISSING_FROM_PRECEDENCE:  # pragma: no cover - import-time contract
    raise AssertionError(
        "every disqualifying issue bit needs a precedence entry; missing "
        f"{_MISSING_FROM_PRECEDENCE!r}"
    )
del _MISSING_FROM_PRECEDENCE


def resolve_status(bits: IssueBit | int) -> str:
    """Return the single verdict for one row's observed conditions."""

    observed = IssueBit(bits)
    for bit in STATUS_PRECEDENCE:
        if bit & observed:
            return bit.name or ""
    return SOURCE_CANONICAL_VALID


def is_eligible(status: str) -> bool:
    """Authority-eligible rows are exactly the valid ones."""

    return status == SOURCE_CANONICAL_VALID


def issue_bit_names(bits: IssueBit | int) -> tuple[str, ...]:
    """Return every observed condition, not merely the winning verdict."""

    observed = IssueBit(bits)
    return tuple(
        bit.name or "" for bit in STATUS_PRECEDENCE if bit & observed
    )


__all__ = [
    "SOURCE_CANONICAL_VALID",
    "STATUS_NAMES",
    "STATUS_PRECEDENCE",
    "IssueBit",
    "is_eligible",
    "issue_bit_names",
    "resolve_status",
]
