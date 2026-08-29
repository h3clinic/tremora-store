"""Readers for the PADS movement records, patients and timeseries files.

Channel order, units, row count and device location all come from each
observation record; nothing is assumed from a task name.  Physical column *i*
is read as declared channel *i* and the result is persisted in TremoraStore's
internal canonical order alongside the permutation that relates the two, so the
reordering is explicit and reversible.

A permuted declaration is therefore an issue to report, not a refusal: the limb
is identified at record level by ``device_location``, so a permutation cannot
relabel a limb, and it could only mislabel a sensor if the parser ignored the
declaration and assumed fixed physical columns -- which is exactly what it does
not do.

The gate closes only on genuine ambiguity, where no reading of the file follows
from what the metadata says.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any

from ..source_tokens import TokenKind, classify
from .authority import (
    CANONICAL_CHANNELS,
    CHANNEL_UNITS,
    NONCANONICAL_SOURCE_ORDER,
    RECOGNIZED_DEVICE_LOCATIONS,
    TIME_CHANNEL,
    first_to_last_span_seconds,
    sample_support_seconds,
)

# --- ambiguity failures ---------------------------------------------------

DUPLICATE_CHANNEL_NAME = "DUPLICATE_CHANNEL_NAME"
MISSING_CHANNEL = "MISSING_CHANNEL"
UNKNOWN_CHANNEL = "UNKNOWN_CHANNEL"
CHANNEL_UNIT_MISMATCH = "CHANNEL_UNIT_MISMATCH"
CHANNEL_UNIT_LENGTH_MISMATCH = "CHANNEL_UNIT_LENGTH_MISMATCH"
MISSING_DEVICE_LOCATION = "MISSING_DEVICE_LOCATION"
UNRECOGNIZED_DEVICE_LOCATION = "UNRECOGNIZED_DEVICE_LOCATION"
ROW_COLUMN_COUNT_MISMATCH = "ROW_COLUMN_COUNT_MISMATCH"

AMBIGUITY_FAILURES: tuple[str, ...] = (
    DUPLICATE_CHANNEL_NAME,
    MISSING_CHANNEL,
    UNKNOWN_CHANNEL,
    CHANNEL_UNIT_MISMATCH,
    CHANNEL_UNIT_LENGTH_MISMATCH,
    MISSING_DEVICE_LOCATION,
    UNRECOGNIZED_DEVICE_LOCATION,
)

# --- stream outcomes ------------------------------------------------------

STREAM_PARSED = "STREAM_PARSED"
BLANK_SOURCE_ROW = "BLANK_SOURCE_ROW"
ROW_COUNT_MISMATCH = "ROW_COUNT_MISMATCH"
NO_USABLE_VALUES = "NO_USABLE_VALUES"
INVALID_TIME = "INVALID_TIME"
FILE_MISSING = "FILE_MISSING"
HASH_MISMATCH = "HASH_MISMATCH"
PATH_ESCAPES_ROOT = "PATH_ESCAPES_ROOT"

# --- non-fatal issue codes ------------------------------------------------

NONMONOTONIC_TIME = "NONMONOTONIC_TIME"
DUPLICATE_TIME = "DUPLICATE_TIME"
CADENCE_DEVIATES_FROM_DECLARED_RATE = "CADENCE_DEVIATES_FROM_DECLARED_RATE"
SPAN_DEVIATES_FROM_DECLARED_RATE = "SPAN_DEVIATES_FROM_DECLARED_RATE"

#: Frozen policy.  The release's own timing is a real device clock: the first
#: file carries intervals from 7.13 ms to 12.90 ms around a 9.99 ms median at a
#: declared 100 Hz.  Only the median can be checked against the declared
#: period, and the span accumulates that jitter over the whole recording.
CADENCE_RELATIVE_TOLERANCE = 0.10
SPAN_RELATIVE_TOLERANCE = 0.05

TIME_VALID = "TIME_VALID"
TIME_DUPLICATE = "TIME_DUPLICATE"
TIME_NONMONOTONIC = "TIME_NONMONOTONIC"
TIME_INVALID = "TIME_INVALID"


class PadsSourceError(ValueError):
    """Raised when a PADS source record cannot be read as written."""


@dataclass(frozen=True, slots=True)
class StreamDeclaration:
    """One device file as the observation record declares it."""

    device_location: str
    channels: tuple[str, ...]
    units: tuple[str, ...]
    file_name: str


@dataclass(frozen=True, slots=True)
class SessionDeclaration:
    """One published task and the device files recorded for it."""

    record_name: str
    rows: int
    task_ordinal: int
    streams: tuple[StreamDeclaration, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    """One participant's movement record."""

    subject_id: str
    study_id: str
    device_id: str | None
    sampling_rate: Fraction
    sessions: tuple[SessionDeclaration, ...]
    source_relative_path: str


@dataclass(frozen=True, slots=True)
class Participant:
    """The participant facts the benchmark tables actually need."""

    participant_id: str
    condition: str | None
    diagnosis_detail: str | None
    handedness: str | None


@dataclass(slots=True)
class ParsedStream:
    """One parsed device file and everything observed about its timeline."""

    device_location: str
    declared_row_count: int
    parsed_row_count: int
    source_channel_order: tuple[str, ...]
    source_units_order: tuple[str, ...]
    canonicalization_permutation: tuple[int, ...]
    times: list[float] = field(default_factory=list)
    time_statuses: list[str] = field(default_factory=list)
    usable_value_count: int = 0
    observed_median_interval_seconds: float | None = None
    observed_first_to_last_span_seconds: float | None = None
    duplicate_time_count: int = 0
    nonmonotonic_time_count: int = 0
    invalid_time_count: int = 0
    stream_status: str = STREAM_PARSED
    issue_codes: tuple[str, ...] = ()
    exclusion_reason: str | None = None

    @property
    def parsed(self) -> bool:
        return self.stream_status == STREAM_PARSED


def _require(mapping: Any, key: str, kinds: tuple[type, ...], where: str) -> Any:
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise PadsSourceError(f"{where} is missing {key!r}")
    value = mapping[key]
    if isinstance(value, bool) and bool not in kinds:
        raise PadsSourceError(f"{where}.{key} has the wrong type")
    if not isinstance(value, kinds):
        raise PadsSourceError(f"{where}.{key} has the wrong type")
    return value


def safe_relative_path(candidate: str, *, where: str) -> PurePosixPath:
    """Return ``candidate`` if it stays inside the movement root."""

    path = PurePosixPath(candidate)
    if not candidate or path.is_absolute():
        raise PadsSourceError(f"{where} is not a relative path")
    if any(part in {"..", ""} for part in path.parts):
        raise PadsSourceError(f"{where} escapes the movement root")
    return path


def parse_observation(
    document: Mapping[str, Any], *, source_relative_path: str
) -> Observation:
    """Read one ``movement/observation_*.json`` exactly as declared."""

    where = source_relative_path
    if _require(document, "resource_type", (str,), where) != "observation":
        raise PadsSourceError(f"{where} is not an observation record")
    rate = _require(document, "sampling_rate", (int, float), where)
    if not isinstance(rate, int) and not float(rate).is_integer():
        sampling_rate = Fraction(str(rate))
    else:
        sampling_rate = Fraction(int(rate))
    if sampling_rate <= 0:
        raise PadsSourceError(f"{where}.sampling_rate is not positive")

    sessions_payload = _require(document, "session", (list,), where)
    sessions: list[SessionDeclaration] = []
    for ordinal, session in enumerate(sessions_payload):
        session_where = f"{where}.session[{ordinal}]"
        record_name = str(
            _require(session, "record_name", (str,), session_where)
        )
        rows = _require(session, "rows", (int,), session_where)
        if rows <= 0:
            raise PadsSourceError(f"{session_where}.rows is not positive")
        records_payload = _require(session, "records", (list,), session_where)
        streams: list[StreamDeclaration] = []
        for index, record in enumerate(records_payload):
            record_where = f"{session_where}.records[{index}]"
            file_name = str(
                _require(record, "file_name", (str,), record_where)
            )
            safe_relative_path(file_name, where=f"{record_where}.file_name")
            channels = tuple(
                str(name)
                for name in _require(record, "channels", (list,), record_where)
            )
            units = tuple(
                str(unit)
                for unit in _require(record, "units", (list,), record_where)
            )
            streams.append(StreamDeclaration(
                device_location=str(
                    _require(record, "device_location", (str,), record_where)
                ),
                channels=channels,
                units=units,
                file_name=file_name,
            ))
        sessions.append(SessionDeclaration(
            record_name=record_name,
            rows=int(rows),
            task_ordinal=ordinal,
            streams=tuple(streams),
        ))

    return Observation(
        subject_id=str(_require(document, "subject_id", (str,), where)),
        study_id=str(_require(document, "study_id", (str,), where)),
        device_id=(
            str(document["device_id"]) if "device_id" in document else None
        ),
        sampling_rate=sampling_rate,
        sessions=tuple(sessions),
        source_relative_path=source_relative_path,
    )


def parse_patient(
    document: Mapping[str, Any], *, source_relative_path: str
) -> Participant:
    """Read one ``patients/patient_*.json``.

    Only the facts the benchmark tables need are carried forward; the release's
    demographic fields stay in the source.
    """

    where = source_relative_path
    if _require(document, "resource_type", (str,), where) != "patient":
        raise PadsSourceError(f"{where} is not a patient record")
    return Participant(
        participant_id=str(_require(document, "id", (str,), where)),
        condition=(
            str(document["condition"]) if "condition" in document else None
        ),
        diagnosis_detail=(
            str(document["disease_comment"])
            if "disease_comment" in document else None
        ),
        handedness=(
            str(document["handedness"]) if "handedness" in document else None
        ),
    )


def validate_declaration(
    declaration: StreamDeclaration,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Check one declaration and return its canonicalization permutation.

    ``permutation[i]`` is the physical column that supplies canonical channel
    ``i``.  A permuted but complete declaration yields an issue code, never a
    refusal.
    """

    if declaration.device_location == "":
        raise PadsSourceError(MISSING_DEVICE_LOCATION)
    if declaration.device_location not in RECOGNIZED_DEVICE_LOCATIONS:
        raise PadsSourceError(UNRECOGNIZED_DEVICE_LOCATION)
    if len(declaration.channels) != len(declaration.units):
        raise PadsSourceError(CHANNEL_UNIT_LENGTH_MISMATCH)
    if len(set(declaration.channels)) != len(declaration.channels):
        raise PadsSourceError(DUPLICATE_CHANNEL_NAME)
    unknown = set(declaration.channels) - set(CANONICAL_CHANNELS)
    if unknown:
        raise PadsSourceError(UNKNOWN_CHANNEL)
    missing = set(CANONICAL_CHANNELS) - set(declaration.channels)
    if missing:
        raise PadsSourceError(MISSING_CHANNEL)
    for name, unit in zip(declaration.channels, declaration.units, strict=True):
        # Units are checked per channel *name*, so a reordered declaration is
        # still verified.
        if CHANNEL_UNITS[name] != unit:
            raise PadsSourceError(CHANNEL_UNIT_MISMATCH)

    positions = {name: index for index, name in enumerate(declaration.channels)}
    permutation = tuple(positions[name] for name in CANONICAL_CHANNELS)
    issues: tuple[str, ...] = (
        () if declaration.channels == CANONICAL_CHANNELS
        else (NONCANONICAL_SOURCE_ORDER,)
    )
    return permutation, issues


def parse_timeseries(
    payload: bytes,
    *,
    declaration: StreamDeclaration,
    declared_rows: int,
    sampling_rate: Fraction,
) -> ParsedStream:
    """Parse one device file against its own declaration."""

    try:
        permutation, issues = validate_declaration(declaration)
    except PadsSourceError as exc:
        return ParsedStream(
            device_location=declaration.device_location,
            declared_row_count=declared_rows,
            parsed_row_count=0,
            source_channel_order=declaration.channels,
            source_units_order=declaration.units,
            canonicalization_permutation=(),
            stream_status=str(exc),
            exclusion_reason=str(exc),
        )

    stream = ParsedStream(
        device_location=declaration.device_location,
        declared_row_count=declared_rows,
        parsed_row_count=0,
        source_channel_order=declaration.channels,
        source_units_order=declaration.units,
        canonicalization_permutation=permutation,
    )
    issue_codes = list(issues)

    text = payload.decode("utf-8", errors="strict")
    records = text.split("\n")
    if records and records[-1] == "":
        records = records[:-1]
    for index, record in enumerate(records):
        if record.strip("\r") == "":
            # A blank source row is a parse failure, not a skipped line.
            stream.stream_status = BLANK_SOURCE_ROW
            stream.exclusion_reason = f"blank record at line {index + 1}"
            return stream

    time_position = permutation[CANONICAL_CHANNELS.index(TIME_CHANNEL)]
    highest_time: float | None = None
    seen_times: set[float] = set()
    for record in records:
        cells = record.rstrip("\r").split(",")
        if len(cells) != len(declaration.channels):
            stream.stream_status = ROW_COLUMN_COUNT_MISMATCH
            stream.exclusion_reason = (
                f"row has {len(cells)} columns, declaration has "
                f"{len(declaration.channels)}"
            )
            return stream
        stream.parsed_row_count += 1

        kind, value = classify(cells[time_position])
        if kind is not TokenKind.DECIMAL or value is None:
            stream.invalid_time_count += 1
            stream.time_statuses.append(TIME_INVALID)
        else:
            status = TIME_VALID
            if value in seen_times:
                stream.duplicate_time_count += 1
                status = TIME_DUPLICATE
            elif highest_time is not None and value < highest_time:
                stream.nonmonotonic_time_count += 1
                status = TIME_NONMONOTONIC
            seen_times.add(value)
            if highest_time is None or value > highest_time:
                highest_time = value
            stream.times.append(value)
            stream.time_statuses.append(status)

        for position in permutation[1:]:
            sensor_kind, _ = classify(cells[position])
            if sensor_kind is TokenKind.DECIMAL:
                stream.usable_value_count += 1

    if stream.parsed_row_count != declared_rows:
        stream.stream_status = ROW_COUNT_MISMATCH
        stream.exclusion_reason = (
            f"parsed {stream.parsed_row_count} rows, metadata declares "
            f"{declared_rows}"
        )
        return stream
    if stream.invalid_time_count:
        # The declared rate validates the timeline; it never generates one.
        stream.stream_status = INVALID_TIME
        stream.exclusion_reason = (
            f"{stream.invalid_time_count} unusable Time values"
        )
        return stream
    if stream.usable_value_count == 0:
        stream.stream_status = NO_USABLE_VALUES
        stream.exclusion_reason = "no usable sensor value in the record"
        return stream

    if len(stream.times) >= 2:
        deltas = [
            later - earlier for earlier, later in pairwise(stream.times)
        ]
        positive = [delta for delta in deltas if delta > 0.0]
        if positive:
            stream.observed_median_interval_seconds = float(median(positive))
        stream.observed_first_to_last_span_seconds = (
            stream.times[-1] - stream.times[0]
        )

    declared_period = float(1 / sampling_rate)
    observed = stream.observed_median_interval_seconds
    if observed is None or not math.isclose(
        observed, declared_period, rel_tol=CADENCE_RELATIVE_TOLERANCE
    ):
        issue_codes.append(CADENCE_DEVIATES_FROM_DECLARED_RATE)
    expected_span = float(
        first_to_last_span_seconds(declared_rows, sampling_rate)
    )
    span = stream.observed_first_to_last_span_seconds
    if span is None or not math.isclose(
        span, expected_span, rel_tol=SPAN_RELATIVE_TOLERANCE
    ):
        issue_codes.append(SPAN_DEVIATES_FROM_DECLARED_RATE)
    if stream.duplicate_time_count:
        issue_codes.append(DUPLICATE_TIME)
    if stream.nonmonotonic_time_count:
        issue_codes.append(NONMONOTONIC_TIME)

    stream.issue_codes = tuple(sorted(set(issue_codes)))
    return stream


def expected_durations(
    rows: int, sampling_rate: Fraction
) -> tuple[float, float]:
    """Return ``(sample support, first-to-last span)`` in seconds."""

    return (
        float(sample_support_seconds(rows, sampling_rate)),
        float(first_to_last_span_seconds(rows, sampling_rate)),
    )


def stream_id_for(
    participant_id: str, task_name: str, device_location: str
) -> str:
    return f"{participant_id}:{task_name}:{device_location}"


def read_json_under_root(
    root: Path, relative: str, *, where: str
) -> Mapping[str, Any]:
    """Read one JSON document that must resolve inside ``root``."""

    import json

    safe_relative_path(relative, where=where)
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if resolved_root not in candidate.parents:
        raise PadsSourceError(f"{where} escapes its root")
    try:
        document = json.loads(candidate.read_bytes().decode("utf-8"))
    except OSError as exc:
        raise PadsSourceError(f"{where} could not be read") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise PadsSourceError(f"{where} is not valid JSON") from exc
    if not isinstance(document, Mapping):
        raise PadsSourceError(f"{where} is not an object")
    return document


def canonical_channel_order() -> Sequence[str]:
    return CANONICAL_CHANNELS


__all__ = [
    "AMBIGUITY_FAILURES",
    "BLANK_SOURCE_ROW",
    "CADENCE_DEVIATES_FROM_DECLARED_RATE",
    "CADENCE_RELATIVE_TOLERANCE",
    "CHANNEL_UNIT_LENGTH_MISMATCH",
    "CHANNEL_UNIT_MISMATCH",
    "DUPLICATE_CHANNEL_NAME",
    "DUPLICATE_TIME",
    "FILE_MISSING",
    "HASH_MISMATCH",
    "INVALID_TIME",
    "MISSING_CHANNEL",
    "MISSING_DEVICE_LOCATION",
    "NONMONOTONIC_TIME",
    "NO_USABLE_VALUES",
    "PATH_ESCAPES_ROOT",
    "ROW_COLUMN_COUNT_MISMATCH",
    "ROW_COUNT_MISMATCH",
    "SPAN_DEVIATES_FROM_DECLARED_RATE",
    "SPAN_RELATIVE_TOLERANCE",
    "STREAM_PARSED",
    "TIME_DUPLICATE",
    "TIME_INVALID",
    "TIME_NONMONOTONIC",
    "TIME_VALID",
    "UNKNOWN_CHANNEL",
    "UNRECOGNIZED_DEVICE_LOCATION",
    "Observation",
    "PadsSourceError",
    "ParsedStream",
    "Participant",
    "SessionDeclaration",
    "StreamDeclaration",
    "expected_durations",
    "parse_observation",
    "parse_patient",
    "parse_timeseries",
    "read_json_under_root",
    "safe_relative_path",
    "stream_id_for",
    "validate_declaration",
]
