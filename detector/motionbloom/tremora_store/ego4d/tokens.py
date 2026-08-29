"""Token classification for Ego4D normalized IMU CSVs.

Ego4D shares the project's exact-decimal grammar with the other adapters; see
:mod:`..source_tokens` for why ``float()`` is not used directly.
"""

from __future__ import annotations

from ..source_tokens import (
    NONFINITE_TOKENS,
    NULL_TOKENS,
    TokenKind,
    classify,
    parse_index,
)

__all__ = [
    "NONFINITE_TOKENS",
    "NULL_TOKENS",
    "TokenKind",
    "classify",
    "parse_index",
]
