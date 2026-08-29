"""Frozen cross-dataset timing-authority taxonomy for TremoraStore.

VIDIMU v0.5 and v0.5D established that reproducing a publisher-provided
transformation is not sufficient to establish temporal authority.  This module
generalizes the decision that finding forced: every dataset binding names the
tier of timing evidence its source actually supplies, and the tier — not the
prose around it — decides which artifacts may be materialized.

The VIDIMU-specific ``AlignmentAuthority`` enum in :mod:`..v05d.authority` is
frozen evidence and is not edited to accommodate another dataset.  The tiers
here are a superset taxonomy; they neither revise nor reinterpret the v0.5 or
v0.5D reports, whose checked JSON stays hash-pinned by the contract tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

TIMING_AUTHORITY_CONTRACT_VERSION = (
    "tremora-cross-dataset-timing-authority-0.1.0"
)


class TimingAuthority(StrEnum):
    """Frozen, mutually exclusive timing-authority tiers.

    Declaration order is strongest to weakest and is itself part of the
    contract; :data:`TIMING_AUTHORITY_ORDER` pins it against accidental
    reordering.
    """

    RAW_SHARED_CLOCK = "RAW_SHARED_CLOCK"
    RAW_MAPPED_CLOCK = "RAW_MAPPED_CLOCK"
    SOURCE_CANONICAL_TIMESTAMP = "SOURCE_CANONICAL_TIMESTAMP"
    SOURCE_ALIGNED_RELATIVE_TIME = "SOURCE_ALIGNED_RELATIVE_TIME"
    SOURCE_DERIVED_ALIGNMENT = "SOURCE_DERIVED_ALIGNMENT"
    SOURCE_RELATIVE_UNIMODAL_CLOCK = "SOURCE_RELATIVE_UNIMODAL_CLOCK"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


TIMING_AUTHORITY_ORDER: tuple[TimingAuthority, ...] = (
    TimingAuthority.RAW_SHARED_CLOCK,
    TimingAuthority.RAW_MAPPED_CLOCK,
    TimingAuthority.SOURCE_CANONICAL_TIMESTAMP,
    TimingAuthority.SOURCE_ALIGNED_RELATIVE_TIME,
    TimingAuthority.SOURCE_DERIVED_ALIGNMENT,
    TimingAuthority.SOURCE_RELATIVE_UNIMODAL_CLOCK,
    TimingAuthority.AMBIGUOUS,
    TimingAuthority.UNRESOLVED,
)

#: A frame-to-IMU index requires a tier that places both modalities on one
#: source-supplied timeline.  ``SOURCE_RELATIVE_UNIMODAL_CLOCK`` is excluded by
#: construction: a unimodal clock has no second modality to index against.
PAIRED_INDEX_ELIGIBLE_TIERS: frozenset[TimingAuthority] = frozenset(
    TIMING_AUTHORITY_ORDER[:5]
)

#: A storage or replay benchmark additionally admits a unimodal source clock,
#: because it measures the store rather than a cross-modal relationship.
STORAGE_BENCHMARK_ELIGIBLE_TIERS: frozenset[TimingAuthority] = frozenset(
    (*TIMING_AUTHORITY_ORDER[:5], TimingAuthority.SOURCE_RELATIVE_UNIMODAL_CLOCK)
)

#: Tiers whose timeline is reconstructed from something the source published
#: rather than read from a common hardware clock.  Each must name the
#: assumption it derives under.
DERIVED_TIERS: frozenset[TimingAuthority] = frozenset({
    TimingAuthority.SOURCE_CANONICAL_TIMESTAMP,
    TimingAuthority.SOURCE_ALIGNED_RELATIVE_TIME,
    TimingAuthority.SOURCE_DERIVED_ALIGNMENT,
    TimingAuthority.SOURCE_RELATIVE_UNIMODAL_CLOCK,
})

#: Tiers that resolve no timeline at all.  They cannot derive under an
#: assumption, because there is no derivation to qualify.
UNRESOLVING_TIERS: frozenset[TimingAuthority] = frozenset({
    TimingAuthority.AMBIGUOUS,
    TimingAuthority.UNRESOLVED,
})


class TimingAuthorityError(ValueError):
    """Raised when an artifact would exceed its source timing authority."""


def normalize(authority: TimingAuthority | str) -> TimingAuthority:
    """Return ``authority`` as a tier, rejecting unknown spellings."""

    try:
        return TimingAuthority(authority)
    except ValueError as exc:
        raise TimingAuthorityError(
            f"unknown timing authority {authority!r}") from exc


def assert_frame_imu_index_allowed(
    authority: TimingAuthority | str,
    *,
    context: str = "frame-to-IMU index",
) -> TimingAuthority:
    """Refuse a paired index the source timing evidence cannot support."""

    tier = normalize(authority)
    if tier not in PAIRED_INDEX_ELIGIBLE_TIERS:
        raise TimingAuthorityError(
            f"{tier.value} cannot support a {context}")
    return tier


def assert_storage_benchmark_allowed(
    authority: TimingAuthority | str,
    *,
    context: str = "storage benchmark",
) -> TimingAuthority:
    """Refuse a benchmark whose dataset resolves no timeline."""

    tier = normalize(authority)
    if tier not in STORAGE_BENCHMARK_ELIGIBLE_TIERS:
        raise TimingAuthorityError(
            f"{tier.value} cannot support a {context}")
    return tier


def assert_hardware_sync_claim(
    authority: TimingAuthority | str,
) -> TimingAuthority:
    """Only a shared raw clock may back a hardware-synchronization claim."""

    tier = normalize(authority)
    if tier is not TimingAuthority.RAW_SHARED_CLOCK:
        raise TimingAuthorityError(
            f"{tier.value} cannot support a hardware-synchronization claim")
    return tier


@dataclass(frozen=True, slots=True)
class DatasetTimingBinding:
    """One dataset's declared timing authority and its stated assumption."""

    dataset_id: str
    timing_authority: TimingAuthority
    derived_under_assumption: str | None = None
    hardware_sync_claim: bool = False
    paired_modalities: bool = False

    def __post_init__(self) -> None:
        tier = normalize(self.timing_authority)
        if tier is not self.timing_authority:
            object.__setattr__(self, "timing_authority", tier)
        assumption = self.derived_under_assumption
        if tier in DERIVED_TIERS and not assumption:
            raise TimingAuthorityError(
                f"{self.dataset_id} binds {tier.value} without stating the "
                "assumption it derives under")
        if tier in UNRESOLVING_TIERS and assumption:
            raise TimingAuthorityError(
                f"{self.dataset_id} binds {tier.value}, which derives nothing "
                "and cannot state a derivation assumption")
        if self.hardware_sync_claim:
            assert_hardware_sync_claim(tier)
        if self.paired_modalities and tier is (
            TimingAuthority.SOURCE_RELATIVE_UNIMODAL_CLOCK
        ):
            raise TimingAuthorityError(
                f"{self.dataset_id} declares paired modalities under a "
                "unimodal source clock")

    def assert_frame_imu_index_allowed(self) -> TimingAuthority:
        if not self.paired_modalities:
            raise TimingAuthorityError(
                f"{self.dataset_id} has no paired modality to index")
        return assert_frame_imu_index_allowed(
            self.timing_authority,
            context=f"{self.dataset_id} frame-to-IMU index",
        )

    def assert_storage_benchmark_allowed(self) -> TimingAuthority:
        return assert_storage_benchmark_allowed(
            self.timing_authority,
            context=f"{self.dataset_id} storage benchmark",
        )


VIDIMU_BINDING = DatasetTimingBinding(
    dataset_id="VIDIMU",
    timing_authority=TimingAuthority.UNRESOLVED,
    paired_modalities=True,
)

EGO4D_BINDING = DatasetTimingBinding(
    dataset_id="EGO4D",
    timing_authority=TimingAuthority.SOURCE_CANONICAL_TIMESTAMP,
    derived_under_assumption=(
        "the first IMU timestamp of a component is aligned to the original "
        "container's t=0, per the Ego4D IMU documentation"
    ),
    paired_modalities=True,
)

PADS_BINDING = DatasetTimingBinding(
    dataset_id="PADS",
    timing_authority=TimingAuthority.SOURCE_RELATIVE_UNIMODAL_CLOCK,
    derived_under_assumption=(
        "the published per-sample Time column is the task-local timeline; the "
        "declared sampling rate validates it and never generates it"
    ),
)

DATASET_BINDINGS: Mapping[str, DatasetTimingBinding] = {
    binding.dataset_id: binding
    for binding in (VIDIMU_BINDING, EGO4D_BINDING, PADS_BINDING)
}


def binding_for(dataset_id: str) -> DatasetTimingBinding:
    """Return the frozen binding for ``dataset_id``."""

    try:
        return DATASET_BINDINGS[dataset_id]
    except KeyError as exc:
        raise TimingAuthorityError(
            f"no timing-authority binding for dataset {dataset_id!r}"
        ) from exc


def assert_tiers_exhaustive(tiers: Iterable[TimingAuthority | str]) -> None:
    """Reject a precedence or status table that omits a declared tier."""

    observed = {normalize(tier) for tier in tiers}
    missing = set(TIMING_AUTHORITY_ORDER) - observed
    if missing:
        raise TimingAuthorityError(
            "timing-authority table omits "
            f"{sorted(tier.value for tier in missing)!r}")


__all__ = [
    "DATASET_BINDINGS",
    "DERIVED_TIERS",
    "EGO4D_BINDING",
    "PADS_BINDING",
    "PAIRED_INDEX_ELIGIBLE_TIERS",
    "STORAGE_BENCHMARK_ELIGIBLE_TIERS",
    "TIMING_AUTHORITY_CONTRACT_VERSION",
    "TIMING_AUTHORITY_ORDER",
    "UNRESOLVING_TIERS",
    "VIDIMU_BINDING",
    "DatasetTimingBinding",
    "TimingAuthority",
    "TimingAuthorityError",
    "assert_frame_imu_index_allowed",
    "assert_hardware_sync_claim",
    "assert_storage_benchmark_allowed",
    "assert_tiers_exhaustive",
    "binding_for",
    "normalize",
]
