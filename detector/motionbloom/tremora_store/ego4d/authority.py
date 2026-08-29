"""Ego4D E4D-P0.1 source-canonical timing-authority contract.

Ego4D publishes normalized IMU CSVs whose ``canonical_timestamp_ms`` column is
already expressed as an offset into the canonical video, which makes it
directly usable for frame-range indexing.  The same documentation discloses
what that costs: the first IMU timestamp is *assumed* aligned to the original
container's ``t = 0``, some canonical timestamps are null because canonical
videos are trimmed to the video-stream region, some components carry no IMU at
all, some files lack complete acceleration, some timestamps are significantly
large, small or non-monotonic, and no IMU calibration was performed.

That combination makes Ego4D suitable for an authority-aware, gap-aware store
and unsuitable for a raw hardware-clock claim.  Ego4D binds at
``SOURCE_CANONICAL_TIMESTAMP`` and :func:`assert_not_relabelled` refuses any
attempt to promote it.
"""

from __future__ import annotations

from typing import Any

from ..timing_authority import (
    EGO4D_BINDING,
    TimingAuthority,
    TimingAuthorityError,
    normalize,
)

EGO4D_DATASET_ID = "EGO4D"
EGO4D_CONTRACT_VERSION = "tremora-ego4d-source-canonical-timestamp-0.1.0"
EGO4D_SCHEMA_VERSION = "e4d-p0.1.0"
EGO4D_PARSER_VERSION = "tremora-ego4d-normalized-imu-parser-0.1.0"
EGO4D_IMPLEMENTATION_VERSION = "ego4d-p01-timing-authority-audit-1.0.0"
EGO4D_ARTIFACT_KIND = "TREMORA_EGO4D_P01_TIMING_AUTHORITY_AUDIT"

GATE_PASS = "PASS_SOURCE_CANONICAL_TIMESTAMP_AUTHORITY"
GATE_NO_GO = "NO_GO_EGO4D_CANONICAL_TIMESTAMP_AUTHORITY"

#: The source spells acceleration ``accl_*``.  TremoraStore preserves that
#: spelling: renaming a source column is a silent claim that the two names are
#: interchangeable.
EGO4D_IMU_COLUMNS: tuple[str, ...] = (
    "component_idx",
    "component_timestamp_ms",
    "canonical_timestamp_ms",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accl_x",
    "accl_y",
    "accl_z",
)

#: Project policy, not an Ego4D fact: a canonical offset beyond a day is
#: recorded as an extreme magnitude rather than silently trusted or repaired.
EXTREME_CANONICAL_MAGNITUDE_MS = 86_400_000.0

#: Deterministic subset selection.  No random number generator is involved;
#: candidate order comes from a keyed digest, so two clean roots agree without
#: shared state and input order cannot change the result.
SELECTION_SEED = 20260828
SELECTION_ALGORITHM_VERSION = "tremora-ego4d-stratified-selection-0.1.0"

MINIMUM_SELECTED_VIDEOS = 100
MINIMUM_PAIRED_COVERAGE_HOURS = 10.0
MINIMUM_CAPTURE_DEVICE_GROUPS = 2

#: Cadence policy.  A component's reference interval is estimated from its own
#: eligible samples; fewer than this many positive deltas is not a cadence, and
#: the component then reports no coverage at all.
MINIMUM_CADENCE_DELTAS = 8

#: A segment breaks at ``min(CONTINUITY_ABSOLUTE_CAP_MS, MULTIPLIER * dt_ref)``.
#: The multiplier is project policy, frozen at 3 and exercised at 2x, 3x and 5x
#: by the test suite.  The absolute cap survives only so a pathologically
#: low-rate component cannot bridge an arbitrarily large interval.
CONTINUITY_MULTIPLIER = 3.0
CONTINUITY_ABSOLUTE_CAP_MS = 100.0

#: PTS reconciliation tolerances.  The span term is capped: uncapped, a
#: two-frame timeline would grant itself a tolerance proportional to its own
#: length and a one-hour decode would "agree" with a half-hour video.
PTS_ORIGIN_TOLERANCE_MS = 1.0
PTS_SPAN_SLACK_MS = 50.0
PTS_SPAN_FRAME_INTERVAL_CAP_MS = 100.0


class Ego4DAuthorityError(ValueError):
    """Raised when an Ego4D artifact would exceed its source authority."""


def assert_not_relabelled(authority: TimingAuthority | str) -> TimingAuthority:
    """Refuse any promotion of Ego4D above its documented tier.

    The canonical timestamp is derived under a documented assumption about the
    first IMU sample.  Calling that a mapped or shared raw clock would claim
    evidence Ego4D does not publish.
    """

    try:
        tier = normalize(authority)
    except TimingAuthorityError as exc:
        raise Ego4DAuthorityError(str(exc)) from exc
    if tier is not TimingAuthority.SOURCE_CANONICAL_TIMESTAMP:
        raise Ego4DAuthorityError(
            f"Ego4D cannot be relabelled {tier.value}; its canonical "
            "timestamps are derived under a documented origin assumption")
    return tier


def authority_contract() -> dict[str, Any]:
    """Return the frozen P0.1 authority block published in every record."""

    binding = EGO4D_BINDING
    assert_not_relabelled(binding.timing_authority)
    return {
        "dataset_id": EGO4D_DATASET_ID,
        "timing_authority": binding.timing_authority.value,
        "raw_shared_clock": False,
        "hardware_sync_claim": False,
        "derived_under_assumption": True,
        "assumption": binding.derived_under_assumption,
        "contract_version": EGO4D_CONTRACT_VERSION,
    }


__all__ = [
    "CONTINUITY_ABSOLUTE_CAP_MS",
    "CONTINUITY_MULTIPLIER",
    "EGO4D_ARTIFACT_KIND",
    "EGO4D_CONTRACT_VERSION",
    "EGO4D_DATASET_ID",
    "EGO4D_IMPLEMENTATION_VERSION",
    "EGO4D_IMU_COLUMNS",
    "EGO4D_PARSER_VERSION",
    "EGO4D_SCHEMA_VERSION",
    "EXTREME_CANONICAL_MAGNITUDE_MS",
    "GATE_NO_GO",
    "GATE_PASS",
    "MINIMUM_CADENCE_DELTAS",
    "MINIMUM_CAPTURE_DEVICE_GROUPS",
    "MINIMUM_PAIRED_COVERAGE_HOURS",
    "MINIMUM_SELECTED_VIDEOS",
    "PTS_ORIGIN_TOLERANCE_MS",
    "PTS_SPAN_FRAME_INTERVAL_CAP_MS",
    "PTS_SPAN_SLACK_MS",
    "SELECTION_ALGORITHM_VERSION",
    "SELECTION_SEED",
    "Ego4DAuthorityError",
    "assert_not_relabelled",
    "authority_contract",
]
