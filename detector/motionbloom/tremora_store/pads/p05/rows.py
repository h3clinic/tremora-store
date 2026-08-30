"""The canonical row identity every representation is compared on.

Four containers hold the same measurements four different ways.  Comparing
their bytes would compare Parquet against HDF5 against text, which is a fact
about encodings and not about whether a query returned the right rows.  So the
comparison is defined on the rows themselves: which stream, which source row,
what the release's own ``Time`` token said, and the six sensor values.

The time token is carried as the release wrote it -- ten decimal places -- and
compared as a string, because that string is the source's own statement of
when the sample was taken.  A representation that rounded it to a float and
back would differ here, which is the point.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .contract import SENSOR_VALUE_COUNT

#: The order the six sensor values are hashed in, fixed once so that a
#: representation which happens to store its columns differently still
#: produces the same identity.
SENSOR_ORDER: tuple[str, ...] = (
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
)

ROW_IDENTITY_VERSION = "pads-p05-row-identity-1"


class RowIdentityError(ValueError):
    """Raised when a row cannot be reduced to its canonical identity."""


@dataclass(frozen=True, slots=True)
class QueryResult:
    """What one query returned, in the only form the comparison sees."""

    query_id: str
    rows: int
    content_sha256: str
    bytes_returned: int
    time_token_sha256: str
    sensor_value_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "rows_returned": self.rows,
            "content_sha256": self.content_sha256,
            "bytes_returned": self.bytes_returned,
            "time_token_sha256": self.time_token_sha256,
            "sensor_value_sha256": self.sensor_value_sha256,
        }


def _value_token(value: float) -> bytes:
    """A float's exact decimal form, so two containers cannot disagree.

    ``repr`` is the shortest string that round-trips to the identical double,
    which makes this an exact comparison of the value rather than a rounded
    one.  A representation that lost a bit in storage produces a different
    token here.
    """

    return repr(float(value)).encode("ascii")


def row_digest_update(
    digest: hashlib._Hash,
    *,
    stream_id: str,
    source_row_ordinal: int,
    source_time_token: str,
    source_time_ps: int,
    sensor_values: Sequence[float],
) -> None:
    """Fold one row into a running canonical digest."""

    if len(sensor_values) != SENSOR_VALUE_COUNT:
        raise RowIdentityError(
            f"a row carries {SENSOR_VALUE_COUNT} sensor values, "
            f"got {len(sensor_values)}"
        )
    digest.update(stream_id.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(str(int(source_row_ordinal)).encode("ascii"))
    digest.update(b"\x1f")
    digest.update(source_time_token.encode("ascii"))
    digest.update(b"\x1f")
    digest.update(str(int(source_time_ps)).encode("ascii"))
    for value in sensor_values:
        digest.update(b"\x1f")
        digest.update(_value_token(value))
    digest.update(b"\x1e")


def result_from_rows(query_id: str, rows: Iterable[dict[str, Any]]) -> (
    QueryResult
):
    """Reduce a representation's returned rows to their canonical identity.

    Three digests come back rather than one.  A single content hash says only
    that two representations disagreed; the separate time-token and
    sensor-value hashes say which half disagreed, which is the difference
    between a diagnosable defect and a mystery.
    """

    content = hashlib.sha256(ROW_IDENTITY_VERSION.encode("ascii"))
    times = hashlib.sha256()
    sensors = hashlib.sha256()
    count = 0
    payload = 0
    for row in rows:
        values = [float(row[name]) for name in SENSOR_ORDER]
        token = str(row["source_time_token"])
        row_digest_update(
            content,
            stream_id=str(row["stream_id"]),
            source_row_ordinal=int(row["source_row_ordinal"]),
            source_time_token=token,
            source_time_ps=int(row["source_time_ps"]),
            sensor_values=values,
        )
        times.update(token.encode("ascii"))
        times.update(b"\x1f")
        times.update(str(int(row["source_time_ps"])).encode("ascii"))
        times.update(b"\x1e")
        for value in values:
            sensors.update(_value_token(value))
            sensors.update(b"\x1f")
        sensors.update(b"\x1e")
        count += 1
        # The bytes a caller actually receives: eight per double, plus the
        # time token's own characters and the ordinal.
        payload += 8 * SENSOR_VALUE_COUNT + len(token) + 8 + 4
    return QueryResult(
        query_id=query_id,
        rows=count,
        content_sha256=content.hexdigest(),
        bytes_returned=payload,
        time_token_sha256=times.hexdigest(),
        sensor_value_sha256=sensors.hexdigest(),
    )


def compare(
    reference: QueryResult, other: QueryResult
) -> dict[str, bool]:
    """Which of the four required equalities hold between two results."""

    return {
        "content_match": reference.content_sha256 == other.content_sha256,
        "row_count_match": reference.rows == other.rows,
        "time_match": reference.time_token_sha256 == other.time_token_sha256,
        "sensor_value_match": (
            reference.sensor_value_sha256 == other.sensor_value_sha256
        ),
    }


__all__ = [
    "ROW_IDENTITY_VERSION",
    "SENSOR_ORDER",
    "QueryResult",
    "RowIdentityError",
    "compare",
    "result_from_rows",
    "row_digest_update",
]
