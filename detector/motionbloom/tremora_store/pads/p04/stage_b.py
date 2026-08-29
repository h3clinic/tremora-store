"""Stage B: uniform 100 Hz parent to a derived rate.

One frozen linear time-invariant transfer function per rate, evaluated in its
polyphase form:

    y[k] = sum_q h[p + L*q] * x[n0 - q]

where ``p`` is the output's phase, ``n0`` its anchor and ``q`` runs over that
branch's taps.  Nothing is renormalized: the executed filter is the frozen one
for every output sample, and the three 30 Hz branch gains are published as
measured rather than corrected.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .contract import RESAMPLING_RATIOS
from .filters import design
from .rational_time import polyphase_anchor

FILTERED = "FILTERED"
UNSUPPORTED_AT_EDGE = "UNSUPPORTED_AT_EDGE"


class StageBError(ValueError):
    """Raised when filtering would need samples that do not exist."""


def filter_to_rate(
    parent: np.ndarray,
    *,
    rate_hz: int,
    parent_first_ordinal: int,
    output_ordinals: range,
) -> np.ndarray:
    """Filter and resample a parent block to ``rate_hz``.

    ``parent`` has shape ``(channels, samples)`` and starts at
    ``parent_first_ordinal``.  Every requested output ordinal must already be
    supported; this raises rather than padding if one is not.
    """

    taps = design(rate_hz)
    upsample, _ = RESAMPLING_RATIOS[rate_hz]
    channels, available = parent.shape
    last_ordinal = parent_first_ordinal + available - 1

    by_phase: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for position, ordinal in enumerate(output_ordinals):
        phase, anchor, branch = polyphase_anchor(
            rate_hz, ordinal, taps=taps.size
        )
        if (
            anchor - branch + 1 < parent_first_ordinal
            or anchor > last_ordinal
        ):
            raise StageBError(
                f"{rate_hz} Hz ordinal {ordinal} lacks its kernel context")
        by_phase[phase].append((position, anchor))

    output = np.empty((channels, len(output_ordinals)), dtype=np.float64)
    for phase, entries in by_phase.items():
        branch = taps[phase::upsample]
        positions = np.array([item[0] for item in entries], dtype=np.int64)
        anchors = np.array([item[1] for item in entries], dtype=np.int64)
        # x[n0 - q] for q = 0 .. branch-1, gathered as one matrix.
        offsets = np.arange(branch.size, dtype=np.int64)
        indices = (
            anchors[:, None] - offsets[None, :] - parent_first_ordinal
        )
        for channel in range(channels):
            gathered = parent[channel][indices]
            output[channel, positions] = gathered @ branch
    return output


__all__ = [
    "FILTERED",
    "UNSUPPORTED_AT_EDGE",
    "StageBError",
    "filter_to_rate",
]
