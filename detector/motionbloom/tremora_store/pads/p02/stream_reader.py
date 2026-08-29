"""Read one PADS device file into the exact representation P0.2 stores.

The declaration contract is P0.1's: physical column *i* is read as declared
channel *i*, units are checked per channel name, and the limb comes from
``device_location`` rather than from column order.  What P0.2 adds is exactness
-- an integer picosecond time and a proven round-trip for every sensor value --
and the bytes needed to reconstruct the source file exactly on replay.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from fractions import Fraction

from ..authority import CANONICAL_CHANNELS, TIME_CHANNEL
from ..movement import (
    PadsSourceError,
    StreamDeclaration,
    validate_declaration,
)
from .contract import P02_SCHEMA_VERSION
from .exact_time import (
    ExactTimeError,
    exact_picoseconds,
    format_sensor_value,
)

STREAM_READ_OK = "STREAM_READ_OK"
STREAM_DECLARATION_REFUSED = "STREAM_DECLARATION_REFUSED"
STREAM_BLANK_ROW = "STREAM_BLANK_ROW"
STREAM_COLUMN_COUNT_MISMATCH = "STREAM_COLUMN_COUNT_MISMATCH"
STREAM_ROW_COUNT_MISMATCH = "STREAM_ROW_COUNT_MISMATCH"
STREAM_TIME_NOT_EXACT = "STREAM_TIME_NOT_EXACT"
STREAM_VALUE_DOES_NOT_ROUND_TRIP = "STREAM_VALUE_DOES_NOT_ROUND_TRIP"
STREAM_UNEXPECTED_LINE_TERMINATOR = "STREAM_UNEXPECTED_LINE_TERMINATOR"

SAMPLE_VALID = "SAMPLE_VALID"

_LINE_TERMINATOR = "\n"
_SENSOR_CHANNELS = CANONICAL_CHANNELS[1:]


@dataclass(slots=True)
class StreamSamples:
    """One stream's samples in TremoraStore's canonical channel order."""

    stream_id: str
    participant_id: str
    assessment_id: str
    task_name: str
    device_location: str
    source_asset_sha256: str
    declared_row_count: int
    declared_sampling_rate_hz: Fraction
    source_channel_order: tuple[str, ...]
    source_units_order: tuple[str, ...]
    canonicalization_permutation: tuple[int, ...]
    sample_ordinal: list[int] = field(default_factory=list)
    source_row_ordinal: list[int] = field(default_factory=list)
    source_time_token: list[str] = field(default_factory=list)
    source_time_ps: list[int] = field(default_factory=list)
    task_local_time_ps: list[int] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)
    source_time_origin_token: str = ""
    source_time_origin_ps: int = 0
    stream_status: str = STREAM_READ_OK
    exclusion_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.stream_status == STREAM_READ_OK

    @property
    def sample_count(self) -> int:
        return len(self.sample_ordinal)

    def channel(self, name: str) -> list[float]:
        return self.values[_SENSOR_CHANNELS.index(name)]

    def replay_source_bytes(self) -> bytes:
        """Rebuild the source file exactly as the release wrote it."""

        columns = self.source_channel_order
        time_position = columns.index(TIME_CHANNEL)
        lines: list[str] = []
        for row in range(self.sample_count):
            cells: list[str] = []
            for position, name in enumerate(columns):
                if position == time_position:
                    cells.append(self.source_time_token[row])
                else:
                    cells.append(format_sensor_value(
                        self.values[_SENSOR_CHANNELS.index(name)][row]
                    ))
            lines.append(",".join(cells))
        return (
            _LINE_TERMINATOR.join(lines) + _LINE_TERMINATOR
        ).encode("utf-8")

    def content_sha256(self) -> str:
        """A hash of the stored logical content, independent of encoding.

        Two runs that materialize the same stream agree here even if a Parquet
        writer detail changes, so the reproduction check compares content
        rather than container bytes.
        """

        digest = hashlib.sha256()
        digest.update(self.stream_id.encode("utf-8"))
        digest.update(b"\x1e")
        digest.update(P02_SCHEMA_VERSION.encode("ascii"))
        for row in range(self.sample_count):
            digest.update(b"\x1e")
            digest.update(str(self.sample_ordinal[row]).encode("ascii"))
            digest.update(b"\x1f")
            digest.update(self.source_time_token[row].encode("ascii"))
            digest.update(b"\x1f")
            digest.update(str(self.source_time_ps[row]).encode("ascii"))
            digest.update(b"\x1f")
            digest.update(str(self.task_local_time_ps[row]).encode("ascii"))
            for channel in self.values:
                digest.update(b"\x1f")
                digest.update(
                    format_sensor_value(channel[row]).encode("ascii")
                )
        return digest.hexdigest()


def _refused(
    declaration: StreamDeclaration,
    *,
    stream_id: str,
    participant_id: str,
    assessment_id: str,
    task_name: str,
    source_asset_sha256: str,
    declared_rows: int,
    rate: Fraction,
    status: str,
    reason: str,
) -> StreamSamples:
    return StreamSamples(
        stream_id=stream_id,
        participant_id=participant_id,
        assessment_id=assessment_id,
        task_name=task_name,
        device_location=declaration.device_location,
        source_asset_sha256=source_asset_sha256,
        declared_row_count=declared_rows,
        declared_sampling_rate_hz=rate,
        source_channel_order=declaration.channels,
        source_units_order=declaration.units,
        canonicalization_permutation=(),
        stream_status=status,
        exclusion_reason=reason,
    )


def read_stream(
    payload: bytes,
    *,
    declaration: StreamDeclaration,
    declared_rows: int,
    sampling_rate: Fraction,
    stream_id: str,
    participant_id: str,
    assessment_id: str,
    task_name: str,
    source_asset_sha256: str,
) -> StreamSamples:
    """Read one device file into exact picoseconds and canonical order."""

    refuse = {
        "stream_id": stream_id,
        "participant_id": participant_id,
        "assessment_id": assessment_id,
        "task_name": task_name,
        "source_asset_sha256": source_asset_sha256,
        "declared_rows": declared_rows,
        "rate": sampling_rate,
    }
    try:
        permutation, _ = validate_declaration(declaration)
    except PadsSourceError as exc:
        return _refused(
            declaration, status=STREAM_DECLARATION_REFUSED,
            reason=str(exc), **refuse,
        )

    text = payload.decode("utf-8")
    if "\r" in text:
        return _refused(
            declaration, status=STREAM_UNEXPECTED_LINE_TERMINATOR,
            reason="the file does not use bare LF terminators", **refuse,
        )
    records = text.split(_LINE_TERMINATOR)
    if records and records[-1] == "":
        records = records[:-1]
    for index, record in enumerate(records):
        if record == "":
            return _refused(
                declaration, status=STREAM_BLANK_ROW,
                reason=f"blank record at line {index + 1}", **refuse,
            )

    samples = StreamSamples(
        stream_id=stream_id,
        participant_id=participant_id,
        assessment_id=assessment_id,
        task_name=task_name,
        device_location=declaration.device_location,
        source_asset_sha256=source_asset_sha256,
        declared_row_count=declared_rows,
        declared_sampling_rate_hz=sampling_rate,
        source_channel_order=declaration.channels,
        source_units_order=declaration.units,
        canonicalization_permutation=permutation,
        values=[[] for _ in _SENSOR_CHANNELS],
    )

    time_position = permutation[CANONICAL_CHANNELS.index(TIME_CHANNEL)]
    sensor_positions = permutation[1:]
    origin_ps: int | None = None
    width = len(declaration.channels)

    for ordinal, record in enumerate(records):
        cells = record.split(",")
        if len(cells) != width:
            return _refused(
                declaration, status=STREAM_COLUMN_COUNT_MISMATCH,
                reason=(
                    f"row {ordinal} has {len(cells)} columns, declaration has "
                    f"{width}"
                ),
                **refuse,
            )
        time_token = cells[time_position]
        try:
            picoseconds = exact_picoseconds(time_token)
        except ExactTimeError as exc:
            return _refused(
                declaration, status=STREAM_TIME_NOT_EXACT,
                reason=f"row {ordinal}: {exc.code}", **refuse,
            )
        if origin_ps is None:
            origin_ps = picoseconds
            samples.source_time_origin_token = time_token
            samples.source_time_origin_ps = picoseconds

        for channel_index, position in enumerate(sensor_positions):
            token = cells[position]
            value = float(token)
            if format_sensor_value(value) != token:
                # Proven, not assumed: the store may only drop a token if the
                # value rebuilds it exactly.
                return _refused(
                    declaration, status=STREAM_VALUE_DOES_NOT_ROUND_TRIP,
                    reason=f"row {ordinal} column {position}", **refuse,
                )
            samples.values[channel_index].append(value)

        samples.sample_ordinal.append(ordinal)
        samples.source_row_ordinal.append(ordinal)
        samples.source_time_token.append(time_token)
        samples.source_time_ps.append(picoseconds)
        samples.task_local_time_ps.append(picoseconds - origin_ps)

    if samples.sample_count != declared_rows:
        return _refused(
            declaration, status=STREAM_ROW_COUNT_MISMATCH,
            reason=(
                f"read {samples.sample_count} rows, metadata declares "
                f"{declared_rows}"
            ),
            **refuse,
        )
    return samples


__all__ = [
    "SAMPLE_VALID",
    "STREAM_BLANK_ROW",
    "STREAM_COLUMN_COUNT_MISMATCH",
    "STREAM_DECLARATION_REFUSED",
    "STREAM_READ_OK",
    "STREAM_ROW_COUNT_MISMATCH",
    "STREAM_TIME_NOT_EXACT",
    "STREAM_UNEXPECTED_LINE_TERMINATOR",
    "STREAM_VALUE_DOES_NOT_ROUND_TRIP",
    "StreamSamples",
    "read_stream",
]
