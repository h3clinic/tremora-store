"""Frozen alignment-authority taxonomy and VIDIMU v0.5D claim boundary."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

ALIGNMENT_CONTRACT_VERSION = "tremora-vidimu-sto-derived-alignment-0.5d.0"
ALIGNMENT_METHOD = "VIDIMU_RMSE_SHIFT_AND_TRIM"
SOURCE_TOOLS_REPOSITORY_URL = "https://github.com/twyncoder/vidimu-tools"
SOURCE_TOOLS_COMMIT = "19beec4156f0109d46341a08f06b035d772afaec"

RAW_NATIVE_CLOCK_GATE = "NO_GO_RAW_NATIVE_CLOCK_AUTHORITY"
SOURCE_DERIVED_ALIGNMENT_GATE_PASS = "PASS"
SOURCE_DERIVED_ALIGNMENT_GATE_NO_GO = "NO_GO"

V05_AUTHORITY_SCRIPT_SHA256 = (
    "abbd59097a16f767729521b4968ac997a55506e467621b997b1894d45334d65a"
)
V05_AUTHORITY_REPORT_SHA256 = (
    "3d4492f984ddffaed579da2e107aaf9f7d1e9cdae1ddc83629f8708d8e75bdec"
)

ESTIMATE_NOTEBOOK_SHA256 = (
    "a88c3bb86a27587ca30d99f820643bb48e27500e58f62b035174143e9c1e4865"
)
MODIFY_NOTEBOOK_SHA256 = (
    "5b719a4e6b80419df18f0711dc62f44e0d7cbdb6bb4337847f3281be097d5fbf"
)
SYNC_UTILITY_SHA256 = (
    "f2674ded71f19a837c9e7cb5f6678ae7944ad8f09932e029845d0766b55f139d"
)

AMBIGUOUS_RECORDING_IDS = frozenset({"S53_A13_T03", "S57_A07_T01"})

FORBIDDEN_CLOCK_FIELDS = frozenset({
    "canonical_time_ns",
    "clock_offset_ns",
    "clock_scale",
    "drift_ppm",
    "sync_residual_ns",
})


class AlignmentAuthority(StrEnum):
    """Frozen, mutually exclusive alignment-authority tiers."""

    RAW_SHARED_CLOCK = "RAW_SHARED_CLOCK"
    RAW_MAPPED_CLOCK = "RAW_MAPPED_CLOCK"
    SOURCE_DERIVED_ALIGNMENT = "SOURCE_DERIVED_ALIGNMENT"
    HEURISTIC_ALIGNMENT = "HEURISTIC_ALIGNMENT"
    AMBIGUOUS_SOURCE_ALIGNMENT = "AMBIGUOUS_SOURCE_ALIGNMENT"
    UNRESOLVED = "UNRESOLVED"


class V05DContractError(ValueError):
    """Raised when a v0.5D artifact would exceed its source authority."""


def assert_no_forbidden_clock_fields(field_names: Iterable[str]) -> None:
    """Reject schemas that imply a native or canonical clock."""

    observed = set(field_names).intersection(FORBIDDEN_CLOCK_FIELDS)
    if observed:
        raise V05DContractError(
            "source-derived alignment cannot contain clock fields: "
            f"{sorted(observed)!r}"
        )


def assert_benchmark_eligible_authority(
    authority: AlignmentAuthority | str,
) -> None:
    """Allow only evidence-backed authority tiers into paper benchmarks."""

    try:
        normalized = AlignmentAuthority(authority)
    except ValueError as exc:
        raise V05DContractError("unknown alignment authority") from exc
    if normalized not in {
        AlignmentAuthority.RAW_SHARED_CLOCK,
        AlignmentAuthority.RAW_MAPPED_CLOCK,
        AlignmentAuthority.SOURCE_DERIVED_ALIGNMENT,
    }:
        raise V05DContractError(
            f"{normalized.value} is not eligible for paper benchmarks"
        )


__all__ = [
    "ALIGNMENT_CONTRACT_VERSION",
    "ALIGNMENT_METHOD",
    "AMBIGUOUS_RECORDING_IDS",
    "ESTIMATE_NOTEBOOK_SHA256",
    "FORBIDDEN_CLOCK_FIELDS",
    "MODIFY_NOTEBOOK_SHA256",
    "RAW_NATIVE_CLOCK_GATE",
    "SOURCE_DERIVED_ALIGNMENT_GATE_NO_GO",
    "SOURCE_DERIVED_ALIGNMENT_GATE_PASS",
    "SOURCE_TOOLS_COMMIT",
    "SOURCE_TOOLS_REPOSITORY_URL",
    "SYNC_UTILITY_SHA256",
    "V05_AUTHORITY_REPORT_SHA256",
    "V05_AUTHORITY_SCRIPT_SHA256",
    "AlignmentAuthority",
    "V05DContractError",
    "assert_benchmark_eligible_authority",
    "assert_no_forbidden_clock_fields",
]
