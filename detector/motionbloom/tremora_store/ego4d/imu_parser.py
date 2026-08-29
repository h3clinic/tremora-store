"""Strict parser for Ego4D normalized IMU CSVs.

The parser preserves what the source wrote and classifies what it finds.  It
does not:

* infer a null canonical timestamp;
* replace a timestamp using the sample rate;
* repair an extreme timestamp;
* discard a non-monotonic row;
* overwrite source order with canonical-time order.

A blank record anywhere but a single trailing terminator is a parse failure.
Skipping one would delete a source row and shift every ordinal after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .authority import (
    EGO4D_IMU_COLUMNS,
    EGO4D_PARSER_VERSION,
    EGO4D_SCHEMA_VERSION,
    EXTREME_CANONICAL_MAGNITUDE_MS,
)
from .metadata import TIMELINE_PRESENT, VideoTimeline
from .row_status import IssueBit, resolve_status
from .tokens import TokenKind, classify, parse_index

_GYRO_COLUMNS = ("gyro_x", "gyro_y", "gyro_z")
_ACCL_COLUMNS = ("accl_x", "accl_y", "accl_z")


class Ego4DParseError(ValueError):
    """Raised when a normalized IMU CSV cannot be read as written."""


@dataclass(frozen=True, slots=True)
class AuthorityRow:
    """One source IMU row, in source order, tokens and values both kept."""

    video_uid: str
    component_idx: int
    source_row_ordinal: int
    component_timestamp_token: str
    component_timestamp_ms: float | None
    canonical_timestamp_token: str
    canonical_timestamp_ms: float | None
    gyro_x: float | None
    gyro_y: float | None
    gyro_z: float | None
    accl_x: float | None
    accl_y: float | None
    accl_z: float | None
    canonical_authority_status: str
    issue_bits: int
    source_asset_sha256: str

    @property
    def eligible(self) -> bool:
        return self.issue_bits == 0


@dataclass(frozen=True, slots=True)
class ParsedImuFile:
    """One parsed asset plus the accounting the gate checks it against."""

    video_uid: str
    source_asset_sha256: str
    header: tuple[str, ...]
    data_line_count: int
    line_terminator: str
    rows: tuple[AuthorityRow, ...]

    @property
    def eligible_rows(self) -> tuple[AuthorityRow, ...]:
        return tuple(row for row in self.rows if row.eligible)


def split_records(payload: bytes) -> tuple[tuple[str, ...], str]:
    """Split a CSV payload into records without deleting any of them.

    Returns the records and the detected line terminator.  The record count is
    established here, before any row object exists, so the row-representation
    gate check compares two independently derived numbers rather than a list
    against its own length.
    """

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Ego4DParseError("IMU asset is not UTF-8") from exc
    if not text:
        raise Ego4DParseError("IMU asset is empty")
    records = text.split("\n")
    if records and records[-1] == "":
        records = records[:-1]
    if not records:
        raise Ego4DParseError("IMU asset carries no records")
    carriage = [record.endswith("\r") for record in records]
    if all(carriage):
        terminator = "\r\n"
        records = [record[:-1] for record in records]
    elif any(carriage):
        raise Ego4DParseError("IMU asset mixes line terminators")
    else:
        terminator = "\n"
    for index, record in enumerate(records):
        if record == "":
            raise Ego4DParseError(
                f"IMU asset has a blank record at line {index + 1}")
    return tuple(records), terminator


def _sensor_values(
    fields: dict[str, str], columns: tuple[str, str, str]
) -> tuple[list[float | None], bool]:
    values: list[float | None] = []
    complete = True
    for column in columns:
        kind, value = classify(fields[column])
        if kind is not TokenKind.DECIMAL:
            complete = False
            values.append(None)
        else:
            values.append(value)
    return values, complete


def parse_normalized_imu_csv(
    payload: bytes,
    *,
    video_uid: str,
    source_asset_sha256: str,
    timeline: VideoTimeline | None,
) -> ParsedImuFile:
    """Parse one normalized IMU asset into authority rows."""

    records, terminator = split_records(payload)
    header = tuple(records[0].split(","))
    if header != EGO4D_IMU_COLUMNS:
        raise Ego4DParseError(
            "IMU asset header is not the normalized Ego4D column order")
    data_records = records[1:]
    data_line_count = len(data_records)

    duration_ms = (
        timeline.canonical_video_duration_ms if timeline is not None else None
    )
    highest_canonical: dict[int, float] = {}
    seen_canonical: dict[int, set[float]] = {}
    rows: list[AuthorityRow] = []

    for ordinal, record in enumerate(data_records):
        cells = record.split(",")
        if len(cells) != len(EGO4D_IMU_COLUMNS):
            raise Ego4DParseError(
                f"row {ordinal} has {len(cells)} fields, expected "
                f"{len(EGO4D_IMU_COLUMNS)}")
        fields = dict(zip(EGO4D_IMU_COLUMNS, cells, strict=True))
        component_idx = parse_index(fields["component_idx"])
        if component_idx is None:
            raise Ego4DParseError(
                f"row {ordinal} has a malformed component_idx")

        bits = IssueBit(0)
        component = (
            timeline.component(component_idx) if timeline is not None else None
        )
        if component is None or component.timeline_status != TIMELINE_PRESENT:
            bits |= IssueBit.COMPONENT_NOT_COVERED

        component_kind, component_ms = classify(
            fields["component_timestamp_ms"])
        if component_kind is not TokenKind.DECIMAL:
            component_ms = None

        canonical_token = fields["canonical_timestamp_ms"]
        canonical_kind, canonical_ms = classify(canonical_token)
        if canonical_kind is TokenKind.NULL:
            bits |= IssueBit.SOURCE_CANONICAL_NULL_AFTER_TRIM
            canonical_ms = None
        elif canonical_kind is TokenKind.NONFINITE:
            bits |= IssueBit.SOURCE_CANONICAL_NONFINITE
            canonical_ms = None
        elif canonical_kind is TokenKind.UNPARSEABLE:
            bits |= IssueBit.SOURCE_CANONICAL_UNPARSEABLE_TOKEN
            canonical_ms = None
        else:
            assert canonical_ms is not None
            if abs(canonical_ms) > EXTREME_CANONICAL_MAGNITUDE_MS:
                bits |= IssueBit.SOURCE_CANONICAL_EXTREME_MAGNITUDE
            if duration_ms is not None and (
                canonical_ms < 0.0 or canonical_ms > duration_ms
            ):
                bits |= IssueBit.SOURCE_CANONICAL_OUTSIDE_VIDEO
            # Non-monotonic is measured against the monotone frontier
            # already established in source order, not merely the previous
            # row.  Comparing against the previous row would let a value that
            # is below the frontier re-enter the eligible set as soon as one
            # lower value preceded it, and the eligible subsequence would stop
            # being increasing.
            highest = highest_canonical.get(component_idx)
            if highest is not None and canonical_ms < highest:
                bits |= IssueBit.SOURCE_CANONICAL_NONMONOTONIC
            observed = seen_canonical.setdefault(component_idx, set())
            if canonical_ms in observed:
                bits |= IssueBit.SOURCE_CANONICAL_DUPLICATE
            observed.add(canonical_ms)
            if highest is None or canonical_ms > highest:
                highest_canonical[component_idx] = canonical_ms

        gyro, gyro_complete = _sensor_values(fields, _GYRO_COLUMNS)
        accl, accl_complete = _sensor_values(fields, _ACCL_COLUMNS)
        if not gyro_complete:
            bits |= IssueBit.MISSING_GYROSCOPE
        if not accl_complete:
            bits |= IssueBit.MISSING_ACCELERATION

        rows.append(AuthorityRow(
            video_uid=video_uid,
            component_idx=component_idx,
            source_row_ordinal=ordinal,
            component_timestamp_token=fields["component_timestamp_ms"],
            component_timestamp_ms=component_ms,
            canonical_timestamp_token=canonical_token,
            canonical_timestamp_ms=canonical_ms,
            gyro_x=gyro[0],
            gyro_y=gyro[1],
            gyro_z=gyro[2],
            accl_x=accl[0],
            accl_y=accl[1],
            accl_z=accl[2],
            canonical_authority_status=resolve_status(bits),
            issue_bits=int(bits),
            source_asset_sha256=source_asset_sha256,
        ))

    return ParsedImuFile(
        video_uid=video_uid,
        source_asset_sha256=source_asset_sha256,
        header=header,
        data_line_count=data_line_count,
        line_terminator=terminator,
        rows=tuple(rows),
    )


def row_record(row: AuthorityRow) -> dict[str, Any]:
    """Return one authority row as a schema-shaped mapping."""

    return {
        "video_uid": row.video_uid,
        "component_idx": row.component_idx,
        "source_row_ordinal": row.source_row_ordinal,
        "component_timestamp_token": row.component_timestamp_token,
        "component_timestamp_ms": row.component_timestamp_ms,
        "canonical_timestamp_token": row.canonical_timestamp_token,
        "canonical_timestamp_ms": row.canonical_timestamp_ms,
        "gyro_x": row.gyro_x,
        "gyro_y": row.gyro_y,
        "gyro_z": row.gyro_z,
        "accl_x": row.accl_x,
        "accl_y": row.accl_y,
        "accl_z": row.accl_z,
        "canonical_authority_status": row.canonical_authority_status,
        "issue_bits": row.issue_bits,
        "source_asset_sha256": row.source_asset_sha256,
        "parser_version": EGO4D_PARSER_VERSION,
        "schema_version": EGO4D_SCHEMA_VERSION,
    }


__all__ = [
    "AuthorityRow",
    "Ego4DParseError",
    "ParsedImuFile",
    "parse_normalized_imu_csv",
    "row_record",
    "split_records",
]
