"""PADS-P0.1 source-relative unimodal-clock contract.

PADS publishes a per-sample ``Time`` channel.  Each observation record declares
its columns -- ``Time`` in seconds followed by acceleration in g and rotation
in rad/s -- alongside a separately declared sampling rate.  The source time
column is therefore the timeline, and the declared rate is a validation
constraint on it, never a generator for it: an invalid time value is never
replaced by ``sample_ordinal / rate``.  The record's gate closes instead.

PADS has no paired video.  That is made structural rather than advisory: every
table and field name is screened for video-bearing substrings rather than
checked against a list of exact names, because a deny-list would refuse
``video_uid`` while admitting ``video_uid_ref``.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Any

from ..timing_authority import (
    PADS_BINDING,
    TimingAuthority,
    TimingAuthorityError,
    normalize,
)

PADS_DATASET_ID = "PADS"
PADS_CONTRACT_VERSION = "tremora-pads-source-relative-unimodal-clock-0.1.0"
PADS_SCHEMA_VERSION = "pads-p0.1.0"
PADS_PARSER_VERSION = "tremora-pads-movement-parser-0.1.0"
PADS_IMPLEMENTATION_VERSION = "pads-p01-unimodal-ingest-audit-1.0.0"
PADS_ARTIFACT_KIND = "TREMORA_PADS_P01_INGEST_AUDIT"

GATE_PASS = "PASS_SOURCE_RELATIVE_UNIMODAL_CLOCK"
GATE_NO_GO = "NO_GO_PADS_UNIMODAL_INGEST"

RELATIVE_TIME_BASIS = "SOURCE_TIME_COLUMN"
TIMELINE_ORIGIN = "TASK_LOCAL_ZERO"
CROSS_WRIST_ALIGNMENT = "SOURCE_DECLARED_TASK_SIMULTANEITY"
CROSS_WRIST_UNCERTAINTY = "NOT_QUANTIFIED"
VIDEO_PAIRING = "NOT_APPLICABLE"
MODALITY_STAMP = "INERTIAL_ONLY_NO_VIDEO"

TIME_CHANNEL = "Time"

#: The internal canonical order.  Source order is preserved separately, and the
#: permutation relating the two travels with every record.
CANONICAL_CHANNELS: tuple[str, ...] = (
    "Time",
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "Gyroscope_X",
    "Gyroscope_Y",
    "Gyroscope_Z",
)

#: The unit each named channel must declare.  Checked per channel *name*, so a
#: reordered declaration is still verified.
CHANNEL_UNITS: dict[str, str] = {
    "Time": "s",
    "Accelerometer_X": "g",
    "Accelerometer_Y": "g",
    "Accelerometer_Z": "g",
    "Gyroscope_X": "rad/s",
    "Gyroscope_Y": "rad/s",
    "Gyroscope_Z": "rad/s",
}

#: Taken from the release's own observation metadata.  A third location closes
#: the affected records' gate as UNRECOGNIZED_DEVICE_LOCATION rather than
#: parsing; widening this set is a separately versioned contract change.
RECOGNIZED_DEVICE_LOCATIONS: frozenset[str] = frozenset({
    "LeftWrist",
    "RightWrist",
})

#: Substrings, not exact names: a deny-list of exact names would refuse
#: ``video_uid`` while admitting ``video_uid_ref`` or ``camera_stream_uid``.
VIDEO_BEARING_SUBSTRINGS: tuple[str, ...] = (
    "camera",
    "frame",
    "pixel",
    "pts",
    "rgb",
    "video",
)

NONCANONICAL_SOURCE_ORDER = "NONCANONICAL_SOURCE_ORDER"


class PadsAuthorityError(ValueError):
    """Raised when a PADS artifact would exceed its source authority."""


def assert_unimodal_authority(
    authority: TimingAuthority | str,
) -> TimingAuthority:
    """Refuse any tier other than the one PADS actually supplies."""

    try:
        tier = normalize(authority)
    except TimingAuthorityError as exc:
        raise PadsAuthorityError(str(exc)) from exc
    if tier is not TimingAuthority.SOURCE_RELATIVE_UNIMODAL_CLOCK:
        raise PadsAuthorityError(
            f"PADS cannot be bound to {tier.value}; it publishes a task-local "
            "source time column and no second modality")
    return tier


def assert_no_paired_claim(names: Iterable[str]) -> None:
    """Refuse any table or field name that implies a video association."""

    offending = sorted({
        name
        for name in names
        if any(token in name.casefold() for token in VIDEO_BEARING_SUBSTRINGS)
    })
    if offending:
        raise PadsAuthorityError(
            f"PADS is inertial-only; {offending!r} implies a video association"
        )


def sample_support_seconds(rows: int, rate_hz: Fraction | int) -> Fraction:
    """Return ``rows / rate``: the time the samples support."""

    return Fraction(rows, 1) / Fraction(rate_hz)


def first_to_last_span_seconds(rows: int, rate_hz: Fraction | int) -> Fraction:
    """Return ``(rows - 1) / rate``: the span the timestamps cover.

    The two differ by exactly one sample period.  2048 samples at 100 Hz
    support 20.48 s but span 20.47 s; requiring a 20.48 s span would be an
    off-by-one-sample error, and a recording that does span 20.48 s has one
    sample too many.
    """

    if rows <= 0:
        return Fraction(0)
    return Fraction(rows - 1, 1) / Fraction(rate_hz)


def authority_contract() -> dict[str, Any]:
    """Return the frozen P0.1 authority block published in every record."""

    binding = PADS_BINDING
    assert_unimodal_authority(binding.timing_authority)
    return {
        "dataset_id": PADS_DATASET_ID,
        "timing_authority": binding.timing_authority.value,
        "relative_time_basis": RELATIVE_TIME_BASIS,
        "timeline_origin": TIMELINE_ORIGIN,
        "cross_wrist_alignment": CROSS_WRIST_ALIGNMENT,
        "cross_wrist_uncertainty": CROSS_WRIST_UNCERTAINTY,
        "video_pairing": VIDEO_PAIRING,
        "modality": MODALITY_STAMP,
        "raw_shared_clock": False,
        "hardware_sync_claim": False,
        "derived_under_assumption": True,
        "assumption": binding.derived_under_assumption,
        "contract_version": PADS_CONTRACT_VERSION,
    }


__all__ = [
    "CANONICAL_CHANNELS",
    "CHANNEL_UNITS",
    "CROSS_WRIST_ALIGNMENT",
    "CROSS_WRIST_UNCERTAINTY",
    "GATE_NO_GO",
    "GATE_PASS",
    "MODALITY_STAMP",
    "NONCANONICAL_SOURCE_ORDER",
    "PADS_ARTIFACT_KIND",
    "PADS_CONTRACT_VERSION",
    "PADS_DATASET_ID",
    "PADS_IMPLEMENTATION_VERSION",
    "PADS_PARSER_VERSION",
    "PADS_SCHEMA_VERSION",
    "RECOGNIZED_DEVICE_LOCATIONS",
    "RELATIVE_TIME_BASIS",
    "TIMELINE_ORIGIN",
    "TIME_CHANNEL",
    "VIDEO_BEARING_SUBSTRINGS",
    "VIDEO_PAIRING",
    "PadsAuthorityError",
    "assert_no_paired_claim",
    "assert_unimodal_authority",
    "authority_contract",
    "first_to_last_span_seconds",
    "sample_support_seconds",
]
