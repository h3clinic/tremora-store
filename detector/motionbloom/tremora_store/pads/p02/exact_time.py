"""Exact numeric conversion for the PADS source-time store.

Every ``Time`` token in the release carries exactly ten decimal places, so its
resolution is 1e-10 s.  An integer nanosecond count cannot represent that: the
first sample interval of the first published file, ``0.0099029541`` s, is
9,902,954.1 ns.  The store therefore scales to picoseconds, where the same
token is exactly 9,902,954,100 ps.

Conversion never rounds.  A token that does not scale to an exact integer at
the declared scale raises, and the caller closes that stream's gate.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from ...source_tokens import TokenKind, classify
from .contract import (
    PICOSECONDS_PER_SECOND,
    SENSOR_VALUE_FORMAT,
    SOURCE_TIME_DECIMALS,
    TIME_SCALE_DECIMALS,
)

TIME_TOKEN_UNPARSEABLE = "TIME_TOKEN_UNPARSEABLE"
TIME_TOKEN_NOT_EXACT_AT_SCALE = "TIME_TOKEN_NOT_EXACT_AT_SCALE"
SENSOR_TOKEN_DOES_NOT_ROUND_TRIP = "SENSOR_TOKEN_DOES_NOT_ROUND_TRIP"

_SCALE = Decimal(10) ** TIME_SCALE_DECIMALS


class ExactTimeError(ValueError):
    """Raised when a source token cannot be represented without rounding."""

    def __init__(self, code: str, token: str) -> None:
        super().__init__(f"{code}: {token!r}")
        self.code = code
        self.token = token


def exact_picoseconds(token: str) -> int:
    """Return ``token`` seconds as an exact integer picosecond count."""

    kind, _ = classify(token)
    if kind is not TokenKind.DECIMAL:
        raise ExactTimeError(TIME_TOKEN_UNPARSEABLE, token)
    try:
        scaled = Decimal(token) * _SCALE
    except InvalidOperation as exc:  # pragma: no cover - grammar rejected it
        raise ExactTimeError(TIME_TOKEN_UNPARSEABLE, token) from exc
    if scaled != scaled.to_integral_value():
        # The source wrote a digit finer than the declared scale.  Rounding it
        # away here would silently change the timeline.
        raise ExactTimeError(TIME_TOKEN_NOT_EXACT_AT_SCALE, token)
    return int(scaled)


def picoseconds_to_seconds_token(picoseconds: int) -> str:
    """Render a picosecond count back at the source's decimal width."""

    value = Decimal(picoseconds) / _SCALE
    return f"{value:.{SOURCE_TIME_DECIMALS}f}"


def format_sensor_value(value: float) -> str:
    """Render one sensor value in the format the release wrote."""

    return SENSOR_VALUE_FORMAT.format(value)


def sensor_value_round_trips(token: str, value: float) -> bool:
    """Whether ``value`` rebuilds ``token`` exactly through the format.

    Checked for every value as it is materialized rather than assumed, so
    byte-exact replay is a proven property of this corpus and not a hope.
    """

    return format_sensor_value(value) == token


def assert_sensor_round_trip(token: str, value: float) -> None:
    if not sensor_value_round_trips(token, value):
        raise ExactTimeError(SENSOR_TOKEN_DOES_NOT_ROUND_TRIP, token)


def seconds_to_picoseconds(seconds: float | Decimal) -> int:
    """Scale a policy constant expressed in seconds; exactness required."""

    scaled = Decimal(str(seconds)) * _SCALE
    if scaled != scaled.to_integral_value():
        raise ExactTimeError(TIME_TOKEN_NOT_EXACT_AT_SCALE, str(seconds))
    return int(scaled)


def picoseconds_per_second() -> int:
    return PICOSECONDS_PER_SECOND


__all__ = [
    "SENSOR_TOKEN_DOES_NOT_ROUND_TRIP",
    "TIME_TOKEN_NOT_EXACT_AT_SCALE",
    "TIME_TOKEN_UNPARSEABLE",
    "ExactTimeError",
    "assert_sensor_round_trip",
    "exact_picoseconds",
    "format_sensor_value",
    "picoseconds_per_second",
    "picoseconds_to_seconds_token",
    "seconds_to_picoseconds",
    "sensor_value_round_trips",
]
