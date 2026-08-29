"""Exact numeric-token classification shared by the dataset adapters.

The preserved source token and the parsed number are different facts, and every
adapter keeps both.  Python's :func:`float` would accept ``1_000`` and store a
number the source never wrote while the preserved token still said ``1_000``,
so classification is done against an explicit decimal grammar instead.

Null and non-finite are also different facts and are never merged.  Both are
matched case-insensitively so ``NaN`` and ``NAN`` cannot land in different
buckets.
"""

from __future__ import annotations

import math
import re
from enum import StrEnum

_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_INTEGER = re.compile(r"[+-]?\d+\Z")

NULL_TOKENS = frozenset({"", "null", "none", "na", "n/a"})
NONFINITE_TOKENS = frozenset({
    "-inf",
    "-infinity",
    "-nan",
    "+inf",
    "+infinity",
    "+nan",
    "inf",
    "infinity",
    "nan",
})


class TokenKind(StrEnum):
    """What one source token is, before any numeric interpretation."""

    DECIMAL = "DECIMAL"
    NULL = "NULL"
    NONFINITE = "NONFINITE"
    UNPARSEABLE = "UNPARSEABLE"


def classify(token: str) -> tuple[TokenKind, float | None]:
    """Classify one exact source token.

    Surrounding whitespace is not stripped: these releases are machine
    written, and quietly trimming a field would normalize away evidence about
    the release.
    """

    folded = token.casefold()
    if folded in NULL_TOKENS:
        return TokenKind.NULL, None
    if folded in NONFINITE_TOKENS:
        value = math.inf if "inf" in folded else math.nan
        if folded.startswith("-"):
            value = -value
        return TokenKind.NONFINITE, value
    if _DECIMAL.fullmatch(token) is None:
        return TokenKind.UNPARSEABLE, None
    try:
        value = float(token)
    except ValueError:  # pragma: no cover - grammar already rejected these
        return TokenKind.UNPARSEABLE, None
    if not math.isfinite(value):
        # A decimal token whose magnitude overflows to infinity is not a
        # non-finite spelling; it is a token this parser cannot represent.
        return TokenKind.UNPARSEABLE, None
    return TokenKind.DECIMAL, value


def parse_index(token: str) -> int | None:
    """Return a non-negative integer index, or ``None`` if malformed."""

    if _INTEGER.fullmatch(token) is None:
        return None
    value = int(token)
    return value if value >= 0 else None


__all__ = [
    "NONFINITE_TOKENS",
    "NULL_TOKENS",
    "TokenKind",
    "classify",
    "parse_index",
]
