"""VIDIMU source-tool-output derived-alignment contracts.

Version 0.5D is deliberately separate from the canonical-clock store.  It can
describe source-authored ordinal cuts, but it cannot create native clocks,
clock segments, or timestamp-level video--IMU correspondence.
"""

from .authority import (
    ALIGNMENT_CONTRACT_VERSION,
    ALIGNMENT_METHOD,
    AMBIGUOUS_RECORDING_IDS,
    AlignmentAuthority,
    V05DContractError,
    assert_benchmark_eligible_authority,
    assert_no_forbidden_clock_fields,
)

__all__ = [
    "ALIGNMENT_CONTRACT_VERSION",
    "ALIGNMENT_METHOD",
    "AMBIGUOUS_RECORDING_IDS",
    "AlignmentAuthority",
    "V05DContractError",
    "assert_benchmark_eligible_authority",
    "assert_no_forbidden_clock_fields",
]
