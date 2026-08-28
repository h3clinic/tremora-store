"""Strict source parsers for the pinned VIDIMU v2.0.0 release.

The released BodyTrack CSV has row ordinals but no timestamps or frame IDs.
The released RAW file has a decimal ``timestamp`` field, but the authors do not
document its unit, clock source, or relation to video PTS. RAW rows also occur
much faster than the documented 50 Hz sensor output and commonly hold a repeated
quaternion. Consequently this module preserves every source observation in an
explicitly unit-unknown source table and defers canonical IMU/clock
materialization. It never invents timestamp units or video times, deduplicates
held values, or promotes RAW row cadence to sensor information bandwidth.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import stat
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final

import pyarrow as pa

from ..schema import QualityBits
from .vidimu import (
    VIDIMU_DATASET_SUBTREE,
    VIDIMU_INVENTORY_SCOPE,
    VIDIMU_VIDEO_ARCHIVES,
    VidimuAdapter,
    VidimuAdapterError,
    parse_recording_id,
)

VIDIMU_SOURCE_PARSER_VERSION = "vidimu-native-source-v0.1.0"
VIDIMU_NOMINAL_IMU_HZ = 50
VIDIMU_RAW_HEADER: Final = ("QUAT", "w", "x", "y", "z", "timestamp")
VIDIMU_QUATERNION_NORM_ABS_TOL = 1e-3
VIDIMU_RAW_TIMESTAMP_INTERPRETATION = (
    "EXACT_RELEASE_DECIMAL_TOKEN;UNIT_CLOCK_SOURCE_AND_VIDEO_RELATION_UNDOCUMENTED"
)
VIDIMU_RAW_OBSERVATION_KIND = (
    "TIMESTAMPED_RAW_SOURCE_ROW_OBSERVATIONS_WITH_NOMINAL_50_HZ_SENSOR_OUTPUT;"
    "CONSECUTIVE_SOURCE_PAYLOAD_REPETITIONS_PRESERVED"
)

VIDIMU_POSE_JOINTS: Final = (
    "pelvis",
    "left_hip",
    "right_hip",
    "torso",
    "left_knee",
    "right_knee",
    "neck",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky_knuckle",
    "right_pinky_knuckle",
    "left_middle_tip",
    "right_middle_tip",
    "left_index_knuckle",
    "right_index_knuckle",
    "left_thumb_tip",
    "right_thumb_tip",
)
_NORMALIZED_POSE_HEADER: Final = tuple(
    f"{joint}_{axis}" for joint in VIDIMU_POSE_JOINTS for axis in ("x", "y", "z")
)
# All v2.0.0 pose CSVs contain one leading ASCII space on the final three
# tokens.  Validate that exact source defect before normalization.
VIDIMU_POSE_SOURCE_HEADER: Final = (
    *_NORMALIZED_POSE_HEADER[:-3],
    *(f" {token}" for token in _NORMALIZED_POSE_HEADER[-3:]),
)

VIDIMU_LOWER_BODY_LAYOUT: Final = (
    "qsHIPS",
    "qsRUL",
    "qsRLL",
    "qsLUL",
    "qsLLL",
)
VIDIMU_UPPER_BODY_LAYOUT: Final = (
    "qsBACK",
    "qsRUA",
    "qsRLA",
    "qsLUA",
    "qsLLA",
)
VIDIMU_SENSOR_BODY_LOCATIONS: Final = {
    "qsHIPS": "LOWER_BACK_L3_L5",
    "qsRUL": "RIGHT_UPPER_LEG_LATERAL_MID_THIGH",
    "qsRLL": "RIGHT_LOWER_LEG_LATERAL_CRANIAL",
    "qsLUL": "LEFT_UPPER_LEG_LATERAL_MID_THIGH",
    "qsLLL": "LEFT_LOWER_LEG_LATERAL_CRANIAL",
    "qsBACK": "UPPER_BACK_T5_T7",
    "qsRUA": "RIGHT_UPPER_ARM_LATERAL_MID",
    "qsRLA": "RIGHT_LOWER_ARM_POSTERIOR_WRIST",
    "qsLUA": "LEFT_UPPER_ARM_LATERAL_MID",
    "qsLLA": "LEFT_LOWER_ARM_POSTERIOR_WRIST",
}

_RECORDING_RE = re.compile(r"^S[0-9]{2}_A(?P<activity>0[1-9]|1[0-3])_T[0-9]{2}$")
_DECIMAL_TIMESTAMP_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,9})?$")
_MAX_SOURCE_TIMESTAMP_TOKEN_LENGTH = 32
_MAX_RAW_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_POSE_SOURCE_BYTES = 4 * 1024 * 1024
_ORIGINAL_VIDEO_SUBTREE = "videosoriginal"
_BODYTRACK_VIDEO_SUBTREE = "videosbodytrack"
_FILE_IDENTITY_FIELDS: Final = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_ORIGINAL_NPOSE_RE = re.compile(
    r"^(?P<recording>S[0-9]{2}_A(?:0[1-9]|1[0-3])_T[0-9]{2})_Npose\.mp4$"
)


class VidimuSourceError(ValueError):
    """Raised when a VIDIMU source file violates the pinned v2 format."""


@dataclass(frozen=True, slots=True)
class VidimuCalibrationQuaternion:
    source_ordinal: int
    stream_id: str
    sensor_label: str
    body_location: str
    source_timestamp_token: str
    qw: float
    qx: float
    qy: float
    qz: float


@dataclass(frozen=True, slots=True)
class VidimuInvalidQuaternion:
    source_ordinal: int
    source_line_number: int
    stream_id: str
    sensor_label: str
    source_timestamp_token: str
    source_values: tuple[str, str, str, str]
    norm: float | None


@dataclass(frozen=True, slots=True)
class VidimuRawStreamStatistics:
    stream_id: str
    sensor_label: str
    body_location: str
    observation_count: int
    consecutive_distinct_payload_count: int
    held_payload_observation_count: int
    first_source_timestamp_token: str
    last_source_timestamp_token: str
    first_source_ordinal: int
    last_source_ordinal: int


@dataclass(frozen=True, slots=True)
class VidimuRawCapture:
    """Parsed RAW observations without a claimed timestamp unit or clock."""

    source_recording_id: str
    stored_recording_id: str
    source_sha256: str
    sensor_layout: tuple[str, ...]
    calibration: tuple[VidimuCalibrationQuaternion, ...]
    invalid_quaternions: tuple[VidimuInvalidQuaternion, ...]
    stream_statistics: tuple[VidimuRawStreamStatistics, ...]
    source_rows: pa.Table
    source_row_count: int
    nominal_sensor_hz: int = VIDIMU_NOMINAL_IMU_HZ
    timestamp_interpretation: str = VIDIMU_RAW_TIMESTAMP_INTERPRETATION
    source_observation_kind: str = VIDIMU_RAW_OBSERVATION_KIND
    clock_truth_status: str = (
        "UNRESOLVED_TIMESTAMP_UNIT_CLOCK_SOURCE_AND_VIDEO_RELATION"
    )
    canonical_materialization_status: str = "DEFERRED"

    def stream_semantics(self) -> list[dict[str, str]]:
        """Return closed-schema provenance semantics for the five streams."""

        return [
            {
                "recording_id": self.stored_recording_id,
                "stream_id": item.stream_id,
                "body_location": item.body_location,
                "payload_kind": "QUATERNION",
                "source_acceleration_unit": "NOT_PRESENT",
                "stored_acceleration_unit": "NOT_PRESENT",
                "source_angular_velocity_unit": "NOT_PRESENT",
                "stored_angular_velocity_unit": "NOT_PRESENT",
                "source_quaternion_convention": (
                    "WXYZ_DIMENSIONLESS_RIGHT_HANDED_ENU;"
                    "ROTATION_DIRECTION_UNSPECIFIED_BY_RELEASE"
                ),
                "stored_quaternion_convention": (
                    "WXYZ_DIMENSIONLESS_RIGHT_HANDED_ENU;"
                    "ROTATION_DIRECTION_UNSPECIFIED_BY_RELEASE"
                ),
                "source_device_frame_convention": (
                    "VIDIMU_SENSOR_NATIVE;HEADING_RESET_LOCAL_X_FACES_"
                    "CAMERA_FRONTAL_PLANE"
                ),
                "stored_device_frame_convention": (
                    "VIDIMU_SENSOR_NATIVE;HEADING_RESET_LOCAL_X_FACES_"
                    "CAMERA_FRONTAL_PLANE"
                ),
                "canonicalization_transform_id": (
                    "SOURCE_ONLY_NO_TIME_CANONICALIZATION;"
                    "IDENTITY_VALID_QUATERNIONS_INVALID_SOURCE_ROWS_TO_NULL"
                ),
                "canonicalization_software_version": (VIDIMU_SOURCE_PARSER_VERSION),
            }
            for item in self.stream_statistics
        ]


@dataclass(frozen=True, slots=True)
class VidimuPoseCapture:
    """BodyTrack source rows that remain unbound to decoded video PTS."""

    source_recording_id: str
    source_sha256: str
    source_rows: pa.Table
    row_count: int
    rows_with_zero_triplets: int
    timing_status: str = "UNBOUND_NO_SOURCE_TIMESTAMPS_OR_FRAME_IDS"
    coordinate_convention: str = (
        "BODYTRACK_NATIVE_ABSOLUTE_3D_MILLIMETRES;AXIS_DIRECTIONS_UNSPECIFIED"
    )


@dataclass(frozen=True, slots=True)
class VidimuSourceRecording:
    """Explicit original-subtree candidate for one paired VIDIMU record."""

    recording_id: str
    subject_id: str
    activity_id: str
    trial_id: str
    camera_video_path: Path | None
    bodytrack_pose_csv_path: Path | None
    quaternion_raw_path: Path | None
    bodytrack_qa_video_path: Path | None

    @property
    def inventory_complete(self) -> bool:
        """Whether the three model-input/source files are present.

        The optional BodyTrack-rendered QA video is deliberately excluded from
        this predicate and can never substitute for the original video path.
        """

        return all(
            (
                self.camera_video_path,
                self.bodytrack_pose_csv_path,
                self.quaternion_raw_path,
            )
        )


@dataclass(frozen=True, slots=True)
class _VidimuSourceBinding:
    recording: VidimuSourceRecording
    dataset_subject_identity: tuple[int, int]
    camera_subject_identity: tuple[int, int] | None
    bodytrack_subject_identity: tuple[int, int] | None
    camera_file_identity: tuple[int, ...] | None
    pose_file_identity: tuple[int, ...]
    raw_file_identity: tuple[int, ...]
    qa_file_identity: tuple[int, ...] | None


def vidimu_raw_source_schema() -> pa.Schema:
    """Unit-unknown RAW source rows; never a canonical IMU sample table."""

    return pa.schema(
        [
            pa.field("source_recording_id", pa.string(), nullable=False),
            pa.field("stream_id", pa.string(), nullable=False),
            pa.field("sensor_label", pa.string(), nullable=False),
            pa.field("body_location", pa.string(), nullable=False),
            pa.field("source_epoch_kind", pa.string(), nullable=False),
            pa.field("sample_index", pa.int64(), nullable=False),
            pa.field("source_ordinal", pa.int64(), nullable=False),
            pa.field("source_line_number", pa.int64(), nullable=False),
            pa.field("source_timestamp_token", pa.string(), nullable=False),
            pa.field("payload_kind", pa.string(), nullable=False),
            pa.field("qw", pa.float64()),
            pa.field("qx", pa.float64()),
            pa.field("qy", pa.float64()),
            pa.field("qz", pa.float64()),
            pa.field("source_quality_bits", pa.uint32(), nullable=False),
        ],
        metadata={
            b"tremora.source_adapter": b"VIDIMU_RAW_V2",
            b"tremora.source_parser_version": (
                VIDIMU_SOURCE_PARSER_VERSION.encode("ascii")
            ),
            b"tremora.source_timestamp_semantics": (
                VIDIMU_RAW_TIMESTAMP_INTERPRETATION.encode("ascii")
            ),
            b"tremora.source_quaternion_convention": (
                b"WXYZ_DIMENSIONLESS_RIGHT_HANDED_ENU;"
                b"ROTATION_DIRECTION_UNSPECIFIED_BY_RELEASE"
            ),
            b"tremora.canonical_materialization_status": b"DEFERRED",
        },
    )


def vidimu_pose_source_schema() -> pa.Schema:
    """Intermediate source schema; it is not a canonical CV estimate table."""

    return pa.schema(
        [
            pa.field("source_recording_id", pa.string(), nullable=False),
            pa.field("source_row_ordinal", pa.int64(), nullable=False),
            pa.field(
                "positions_mm",
                pa.list_(pa.float64(), list_size=len(VIDIMU_POSE_JOINTS) * 3),
                nullable=False,
            ),
            pa.field(
                "zero_triplet_mask",
                pa.list_(pa.bool_(), list_size=len(VIDIMU_POSE_JOINTS)),
                nullable=False,
            ),
        ],
        metadata={
            b"tremora.source_adapter": b"VIDIMU_BODYTRACK_CSV_V2",
            b"tremora.source_parser_version": (
                VIDIMU_SOURCE_PARSER_VERSION.encode("ascii")
            ),
            b"tremora.source_unit": b"millimetres",
            b"tremora.timing_status": (b"UNBOUND_NO_SOURCE_TIMESTAMPS_OR_FRAME_IDS"),
            b"tremora.coordinate_convention": (
                b"BODYTRACK_NATIVE_ABSOLUTE_3D_MILLIMETRES;AXIS_DIRECTIONS_UNSPECIFIED"
            ),
            b"tremora.ordered_joint_layout": (
                ",".join(VIDIMU_POSE_JOINTS).encode("ascii")
            ),
            b"tremora.ordered_joint_layout_sha256": hashlib.sha256(
                ",".join(VIDIMU_POSE_JOINTS).encode("ascii")
            )
            .hexdigest()
            .encode("ascii"),
        },
    )


def _recording_layout(recording_id: str) -> tuple[str, ...]:
    if not isinstance(recording_id, str):
        raise VidimuSourceError("source_recording_id must be a canonical VIDIMU string")
    match = _RECORDING_RE.fullmatch(recording_id)
    if match is None:
        raise VidimuSourceError(
            f"not a canonical VIDIMU source recording ID: {recording_id!r}"
        )
    activity = int(match.group("activity"))
    if 1 <= activity <= 4:
        return VIDIMU_LOWER_BODY_LAYOUT
    if 5 <= activity <= 13:
        return VIDIMU_UPPER_BODY_LAYOUT
    raise VidimuSourceError(
        f"VIDIMU activity lies outside the pinned A01-A13 range: {recording_id}"
    )


def _required_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise VidimuSourceError(f"{field} must be a non-empty string")
    return value


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(metadata, field) for field in _FILE_IDENTITY_FIELDS)


def _open_source_descriptor(
    path: Path,
    *,
    flags: int,
    trusted_root: Path | None,
    trusted_root_identity: tuple[int, int] | None,
    trusted_ancestor_identities: tuple[tuple[int, int], ...] = (),
) -> int:
    """Open a source, optionally through one pinned directory hierarchy."""

    if trusted_root is None:
        try:
            return os.open(path, flags)
        except OSError as exc:
            raise VidimuSourceError(f"cannot open VIDIMU source file: {path}") from exc
    if trusted_root_identity is None:
        raise VidimuSourceError("trusted source root identity is missing")
    if (
        os.open not in getattr(os, "supports_dir_fd", ())
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        raise VidimuSourceError(
            "trusted VIDIMU source traversal requires no-follow openat support"
        )
    try:
        relative = path.relative_to(trusted_root)
    except ValueError as exc:
        raise VidimuSourceError(
            "VIDIMU source path escapes its trusted extraction root"
        ) from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise VidimuSourceError("VIDIMU source relative path is invalid")
    if len(trusted_ancestor_identities) != len(relative.parts) - 1:
        raise VidimuSourceError(
            "trusted VIDIMU source ancestor inventory is incomplete"
        )

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_NONBLOCK", 0)
    descriptors: list[int] = []
    try:
        root_descriptor = os.open(trusted_root, directory_flags)
        descriptors.append(root_descriptor)
        root_stat = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or (
                root_stat.st_dev,
                root_stat.st_ino,
            )
            != trusted_root_identity
        ):
            raise VidimuSourceError(
                "trusted VIDIMU extraction root changed after discovery"
            )
        parent_descriptor = root_descriptor
        for component, expected_identity in zip(
            relative.parts[:-1],
            trusted_ancestor_identities,
        ):
            child_descriptor = os.open(
                component, directory_flags, dir_fd=parent_descriptor
            )
            descriptors.append(child_descriptor)
            child_stat = os.fstat(child_descriptor)
            if not stat.S_ISDIR(child_stat.st_mode):
                raise VidimuSourceError("VIDIMU source ancestor is not a directory")
            if (child_stat.st_dev, child_stat.st_ino) != expected_identity:
                raise VidimuSourceError(
                    "trusted VIDIMU source ancestor changed after discovery"
                )
            parent_descriptor = child_descriptor
        descriptor = os.open(relative.parts[-1], flags, dir_fd=parent_descriptor)
    except VidimuSourceError:
        raise
    except OSError as exc:
        raise VidimuSourceError(
            f"cannot open VIDIMU source file safely: {path}"
        ) from exc
    finally:
        for directory_descriptor in reversed(descriptors):
            os.close(directory_descriptor)
    return descriptor


def _read_regular_source(
    path: Path,
    *,
    expected_name: str,
    max_bytes: int,
    trusted_root: Path | None = None,
    trusted_root_identity: tuple[int, int] | None = None,
    trusted_ancestor_identities: tuple[tuple[int, int], ...] = (),
    trusted_file_identity: tuple[int, ...] | None = None,
) -> tuple[bytes, str]:
    if path.name != expected_name:
        raise VidimuSourceError(
            f"VIDIMU source filename must be exactly {expected_name}"
        )
    if path.is_symlink():
        raise VidimuSourceError("VIDIMU source file must not be a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = _open_source_descriptor(
        path,
        flags=flags,
        trusted_root=trusted_root,
        trusted_root_identity=trusted_root_identity,
        trusted_ancestor_identities=trusted_ancestor_identities,
    )
    try:
        try:
            before = os.fstat(descriptor)
        except OSError as exc:
            raise VidimuSourceError(
                f"cannot inspect VIDIMU source file: {path}"
            ) from exc
        if not stat.S_ISREG(before.st_mode):
            raise VidimuSourceError("VIDIMU source path is not a regular file")
        if (
            trusted_file_identity is not None
            and _stat_identity(before) != trusted_file_identity
        ):
            raise VidimuSourceError(
                "trusted VIDIMU source file changed after discovery"
            )
        if before.st_size > max_bytes:
            raise VidimuSourceError(
                f"VIDIMU source file exceeds the pinned {max_bytes}-byte limit"
            )
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            try:
                chunk = os.read(descriptor, 1024 * 1024)
            except OSError as exc:
                raise VidimuSourceError(
                    f"cannot read VIDIMU source file: {path}"
                ) from exc
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > max_bytes:
                raise VidimuSourceError(
                    "VIDIMU source grew beyond its pinned byte limit while reading"
                )
            chunks.append(chunk)
        try:
            after = os.fstat(descriptor)
        except OSError as exc:
            raise VidimuSourceError(
                f"cannot revalidate VIDIMU source file: {path}"
            ) from exc
        if observed_size != before.st_size or _stat_identity(before) != _stat_identity(
            after
        ):
            raise VidimuSourceError(
                "VIDIMU source file changed while it was being read"
            )
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    return payload, hashlib.sha256(payload).hexdigest()


def _csv_rows(payload: bytes, *, source: str) -> list[list[str]]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise VidimuSourceError(f"{source} must be strict UTF-8 without a BOM") from exc
    if text.startswith("\ufeff"):
        raise VidimuSourceError(f"{source} must not contain a UTF-8 BOM")
    try:
        return list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise VidimuSourceError(f"{source} is not valid strict CSV: {exc}") from exc


def _timestamp_order_key(token: str, *, line_number: int) -> tuple[int, int]:
    """Return an exact unitless ordering key without decimal arithmetic."""

    if (
        len(token) > _MAX_SOURCE_TIMESTAMP_TOKEN_LENGTH
        or _DECIMAL_TIMESTAMP_RE.fullmatch(token) is None
    ):
        raise VidimuSourceError(
            f"RAW timestamp on line {line_number} is not a plain decimal with "
            "at most nine fractional decimal places"
        )
    whole, separator, fraction = token.partition(".")
    return int(whole), int((fraction if separator else "").ljust(9, "0") or "0")


def _quaternion(
    tokens: list[str],
    *,
    line_number: int,
) -> tuple[tuple[float, float, float, float] | None, float | None]:
    if any(token != token.strip() or not token for token in tokens):
        raise VidimuSourceError(
            f"RAW quaternion on line {line_number} contains whitespace or empties"
        )
    try:
        values = tuple(float(token) for token in tokens)
    except ValueError:
        return None, None
    if not all(math.isfinite(value) for value in values):
        return None, None
    quaternion = (values[0], values[1], values[2], values[3])
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isclose(
        norm,
        1.0,
        rel_tol=0.0,
        abs_tol=VIDIMU_QUATERNION_NORM_ABS_TOL,
    ):
        return None, norm
    return quaternion, norm


def _stream_id(sensor_label: str) -> str:
    return f"vidimu-{sensor_label}"


def parse_vidimu_raw(
    path: str | Path,
    *,
    source_recording_id: str,
    stored_recording_id: str | None = None,
    _trusted_root: Path | None = None,
    _trusted_root_identity: tuple[int, int] | None = None,
    _trusted_ancestor_identities: tuple[tuple[int, int], ...] = (),
    _trusted_file_identity: tuple[int, ...] | None = None,
) -> VidimuRawCapture:
    """Parse one canonical RAW source without assigning a time unit or map."""

    layout = _recording_layout(source_recording_id)
    stored = (
        source_recording_id
        if stored_recording_id is None
        else _required_identifier(stored_recording_id, "stored_recording_id")
    )
    source_path = Path(path)
    payload, digest = _read_regular_source(
        source_path,
        expected_name=f"{source_recording_id}.raw",
        max_bytes=_MAX_RAW_SOURCE_BYTES,
        trusted_root=_trusted_root,
        trusted_root_identity=_trusted_root_identity,
        trusted_ancestor_identities=_trusted_ancestor_identities,
        trusted_file_identity=_trusted_file_identity,
    )
    rows = _csv_rows(payload, source="VIDIMU RAW")
    if not rows or tuple(rows[0]) != VIDIMU_RAW_HEADER:
        raise VidimuSourceError(
            f"VIDIMU RAW header must be exactly {','.join(VIDIMU_RAW_HEADER)}"
        )
    data_rows = rows[1:]
    if len(data_rows) < len(layout):
        raise VidimuSourceError("VIDIMU RAW is missing the five N-pose rows")
    if any(not row for row in data_rows):
        raise VidimuSourceError("VIDIMU RAW must not contain blank data rows")
    if any(len(row) != len(VIDIMU_RAW_HEADER) for row in data_rows):
        raise VidimuSourceError("every VIDIMU RAW data row must have six fields")

    calibration: list[VidimuCalibrationQuaternion] = []
    source_rows: list[dict[str, object]] = []
    calibration_order_keys: list[tuple[int, int]] = []
    for ordinal, (expected_sensor, row) in enumerate(zip(layout, data_rows[:5])):
        line_number = ordinal + 2
        if row[0] != expected_sensor:
            raise VidimuSourceError(
                "the first five VIDIMU RAW rows must be the positional N-pose "
                f"layout; line {line_number} expected {expected_sensor}"
            )
        timestamp_token = row[5]
        calibration_order_keys.append(
            _timestamp_order_key(timestamp_token, line_number=line_number)
        )
        quaternion, _ = _quaternion(row[1:5], line_number=line_number)
        if quaternion is None:
            raise VidimuSourceError(
                f"N-pose quaternion on line {line_number} is invalid"
            )
        stream_id = _stream_id(expected_sensor)
        calibration.append(
            VidimuCalibrationQuaternion(
                source_ordinal=ordinal,
                stream_id=stream_id,
                sensor_label=expected_sensor,
                body_location=VIDIMU_SENSOR_BODY_LOCATIONS[expected_sensor],
                source_timestamp_token=timestamp_token,
                qw=quaternion[0],
                qx=quaternion[1],
                qy=quaternion[2],
                qz=quaternion[3],
            )
        )
        source_rows.append(
            {
                "source_recording_id": source_recording_id,
                "stream_id": stream_id,
                "sensor_label": expected_sensor,
                "body_location": VIDIMU_SENSOR_BODY_LOCATIONS[expected_sensor],
                "source_epoch_kind": "NPOSE_CALIBRATION",
                "sample_index": 0,
                "source_ordinal": ordinal,
                "source_line_number": line_number,
                "source_timestamp_token": timestamp_token,
                "payload_kind": "QUATERNION",
                "qw": quaternion[0],
                "qx": quaternion[1],
                "qy": quaternion[2],
                "qz": quaternion[3],
                "source_quality_bits": 0,
            }
        )
    if len(set(calibration_order_keys)) != 1:
        raise VidimuSourceError(
            "the five positional N-pose rows must share one source timestamp"
        )
    calibration_order_key = calibration_order_keys[0]

    measurement_rows = data_rows[5:]
    if not measurement_rows:
        raise VidimuSourceError("VIDIMU RAW contains no post-calibration observations")
    if len(measurement_rows) % len(layout):
        raise VidimuSourceError(
            "post-calibration VIDIMU RAW rows must contain complete five-row cycles"
        )

    by_sensor: dict[str, list[dict[str, object]]] = {sensor: [] for sensor in layout}
    timestamp_keys_by_sensor: dict[str, list[tuple[int, int]]] = {
        sensor: [] for sensor in layout
    }
    raw_payload_by_sensor: dict[str, list[tuple[str, str, str, str]]] = {
        sensor: [] for sensor in layout
    }
    invalid: list[VidimuInvalidQuaternion] = []
    for measurement_offset, row in enumerate(measurement_rows):
        source_ordinal = measurement_offset + len(layout)
        line_number = source_ordinal + 2
        expected_sensor = layout[measurement_offset % len(layout)]
        if row[0] != expected_sensor:
            raise VidimuSourceError(
                "post-calibration VIDIMU RAW sensor order is not the pinned "
                f"five-row cycle; line {line_number} expected {expected_sensor}"
            )
        timestamp_token = row[5]
        timestamp_key = _timestamp_order_key(timestamp_token, line_number=line_number)
        quaternion_tokens = (row[1], row[2], row[3], row[4])
        quaternion, norm = _quaternion(row[1:5], line_number=line_number)
        stream_rows = by_sensor[expected_sensor]
        quality_bits = 0
        if quaternion is None:
            quality_bits |= int(QualityBits.INVALID_IMU_PAYLOAD)
            invalid.append(
                VidimuInvalidQuaternion(
                    source_ordinal=source_ordinal,
                    source_line_number=line_number,
                    stream_id=_stream_id(expected_sensor),
                    sensor_label=expected_sensor,
                    source_timestamp_token=timestamp_token,
                    source_values=quaternion_tokens,
                    norm=norm,
                )
            )
        stream_rows.append(
            {
                "source_recording_id": source_recording_id,
                "stream_id": _stream_id(expected_sensor),
                "sensor_label": expected_sensor,
                "body_location": VIDIMU_SENSOR_BODY_LOCATIONS[expected_sensor],
                "source_epoch_kind": "DYNAMIC_OBSERVATION",
                "sample_index": len(stream_rows) + 1,
                "source_ordinal": source_ordinal,
                "source_line_number": line_number,
                "source_timestamp_token": timestamp_token,
                "payload_kind": "QUATERNION",
                "qw": None if quaternion is None else quaternion[0],
                "qx": None if quaternion is None else quaternion[1],
                "qy": None if quaternion is None else quaternion[2],
                "qz": None if quaternion is None else quaternion[3],
                "source_quality_bits": quality_bits,
            }
        )
        timestamp_keys_by_sensor[expected_sensor].append(timestamp_key)
        raw_payload_by_sensor[expected_sensor].append(quaternion_tokens)

    statistics: list[VidimuRawStreamStatistics] = []
    for sensor in layout:
        stream_rows = by_sensor[sensor]
        timestamp_keys = timestamp_keys_by_sensor[sensor]
        for row_index, (previous_key, current_key) in enumerate(
            pairwise(timestamp_keys),
            start=1,
        ):
            if current_key < previous_key:
                raise VidimuSourceError(
                    f"VIDIMU RAW timestamp reverses within {sensor}; an explicit "
                    "clock-reset policy is required"
                )
            if current_key == previous_key:
                duplicate = int(QualityBits.DUPLICATE_TIMESTAMP)
                previous = stream_rows[row_index - 1]
                current = stream_rows[row_index]
                previous["source_quality_bits"] = (
                    int(previous["source_quality_bits"]) | duplicate
                )
                current["source_quality_bits"] = (
                    int(current["source_quality_bits"]) | duplicate
                )
        payloads = raw_payload_by_sensor[sensor]
        distinct = 1 + sum(
            current != previous for previous, current in pairwise(payloads)
        )
        first = stream_rows[0]
        last = stream_rows[-1]
        if timestamp_keys[0] <= calibration_order_key:
            raise VidimuSourceError(
                f"VIDIMU RAW dynamic observations for {sensor} must begin after "
                "its positional N-pose row"
            )
        stream_id = _stream_id(sensor)
        statistics.append(
            VidimuRawStreamStatistics(
                stream_id=stream_id,
                sensor_label=sensor,
                body_location=VIDIMU_SENSOR_BODY_LOCATIONS[sensor],
                observation_count=len(stream_rows),
                consecutive_distinct_payload_count=distinct,
                held_payload_observation_count=len(stream_rows) - distinct,
                first_source_timestamp_token=str(first["source_timestamp_token"]),
                last_source_timestamp_token=str(last["source_timestamp_token"]),
                first_source_ordinal=int(first["source_ordinal"]),
                last_source_ordinal=int(last["source_ordinal"]),
            )
        )
        source_rows.extend(stream_rows)

    ordered_rows = sorted(source_rows, key=lambda row: int(row["source_ordinal"]))
    source_table = pa.Table.from_pylist(ordered_rows, schema=vidimu_raw_source_schema())
    return VidimuRawCapture(
        source_recording_id=source_recording_id,
        stored_recording_id=stored,
        source_sha256=digest,
        sensor_layout=layout,
        calibration=tuple(calibration),
        invalid_quaternions=tuple(invalid),
        stream_statistics=tuple(statistics),
        source_rows=source_table,
        source_row_count=len(data_rows),
    )


def parse_vidimu_pose_csv(
    path: str | Path,
    *,
    source_recording_id: str,
    _trusted_root: Path | None = None,
    _trusted_root_identity: tuple[int, int] | None = None,
    _trusted_ancestor_identities: tuple[tuple[int, int], ...] = (),
    _trusted_file_identity: tuple[int, ...] | None = None,
) -> VidimuPoseCapture:
    """Parse BodyTrack positions without manufacturing a frame-time binding."""

    _recording_layout(source_recording_id)
    source_path = Path(path)
    payload, digest = _read_regular_source(
        source_path,
        expected_name=f"{source_recording_id}.csv",
        max_bytes=_MAX_POSE_SOURCE_BYTES,
        trusted_root=_trusted_root,
        trusted_root_identity=_trusted_root_identity,
        trusted_ancestor_identities=_trusted_ancestor_identities,
        trusted_file_identity=_trusted_file_identity,
    )
    rows = _csv_rows(payload, source="VIDIMU BodyTrack CSV")
    if not rows or tuple(rows[0]) != VIDIMU_POSE_SOURCE_HEADER:
        raise VidimuSourceError(
            "VIDIMU BodyTrack CSV header does not match the exact v2 source schema"
        )
    normalized = tuple(token.strip(" ") for token in rows[0])
    if normalized != _NORMALIZED_POSE_HEADER or len(normalized) != len(set(normalized)):
        raise VidimuSourceError(
            "VIDIMU BodyTrack CSV header normalization is not one-to-one"
        )
    data_rows = rows[1:]
    if not data_rows:
        raise VidimuSourceError("VIDIMU BodyTrack CSV contains no pose rows")
    parsed: list[dict[str, object]] = []
    rows_with_zero_triplets = 0
    for ordinal, row in enumerate(data_rows):
        line_number = ordinal + 2
        if len(row) != len(_NORMALIZED_POSE_HEADER):
            raise VidimuSourceError(
                f"BodyTrack row {line_number} must contain 102 coordinates"
            )
        if any(token != token.strip() or not token for token in row):
            raise VidimuSourceError(
                f"BodyTrack row {line_number} contains whitespace or empty values"
            )
        try:
            values = [float(token) for token in row]
        except ValueError as exc:
            raise VidimuSourceError(
                f"BodyTrack row {line_number} contains a non-numeric coordinate"
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise VidimuSourceError(
                f"BodyTrack row {line_number} contains a non-finite coordinate"
            )
        zero_mask = [
            values[index : index + 3] == [0.0, 0.0, 0.0]
            for index in range(0, len(values), 3)
        ]
        rows_with_zero_triplets += int(any(zero_mask))
        parsed.append(
            {
                "source_recording_id": source_recording_id,
                "source_row_ordinal": ordinal,
                "positions_mm": values,
                # This is an observed sentinel pattern, not a visibility or
                # tracking-validity claim.
                "zero_triplet_mask": zero_mask,
            }
        )
    table = pa.Table.from_pylist(parsed, schema=vidimu_pose_source_schema())
    return VidimuPoseCapture(
        source_recording_id=source_recording_id,
        source_sha256=digest,
        source_rows=table,
        row_count=len(parsed),
        rows_with_zero_triplets=rows_with_zero_triplets,
    )


class VidimuCameraSourceAdapter:
    """Discover original-video candidates + pose/RAW from official v2 extracts.

    ``video_archive_root`` is the extraction root containing the archive wrapper
    directory (``videosmallsize`` or ``videosfullsize``), which in turn contains
    both ``videosoriginal`` and ``videosbodytrack``.  The latter is optional QA
    only.  This explicit API prevents the legacy inventory adapter's rendered
    ``*_pose.mp4`` from being mistaken for camera input.

    The class deliberately has no snapshot publication method. RAW timing
    remains unresolved and pose rows remain unbound until a PTS-aware decoder is
    implemented and exact frame association succeeds. Adapter parsing also
    fails closed when the platform cannot provide no-follow directory-relative
    opens; Python's Windows filesystem API cannot promise the same ancestor-swap
    resistance as POSIX ``openat``.
    """

    def __init__(
        self,
        dataset_archive_root: str | Path,
        *,
        video_archive_root: str | Path,
        video_archive: str,
        dataset_archive_sha256: str,
        video_archive_sha256: str,
        inventory_scope: str,
        terms_sha256: str | None = None,
        terms_path: str | Path | None = None,
    ):
        if video_archive not in VIDIMU_VIDEO_ARCHIVES:
            raise VidimuAdapterError(
                "video_archive must be exactly videosmallsize or videosfullsize"
            )
        if inventory_scope != VIDIMU_INVENTORY_SCOPE:
            raise VidimuAdapterError(
                "inventory_scope must explicitly acknowledge extracted-subset "
                "inventory without release-completeness evidence"
            )
        self.video_archive = video_archive
        self.video_archive_root = Path(video_archive_root).resolve()
        wrapper_candidate = self.video_archive_root / video_archive
        if wrapper_candidate.is_symlink() or not wrapper_candidate.is_dir():
            raise VidimuAdapterError(
                "VIDIMU video extraction must contain its exact archive wrapper"
            )
        wrapper = wrapper_candidate.resolve(strict=True)
        if not wrapper.is_relative_to(self.video_archive_root):
            raise VidimuAdapterError("VIDIMU video wrapper escapes its extraction root")
        self.video_wrapper = wrapper
        self.camera_subtree = self._required_subtree(wrapper, _ORIGINAL_VIDEO_SUBTREE)
        self.bodytrack_subtree = self._required_subtree(
            wrapper, _BODYTRACK_VIDEO_SUBTREE
        )
        # Reuse the frozen v0.1 data/bodytrack grammar only for validation and
        # paired CSV/RAW discovery. Its rendered video is retained as optional
        # QA metadata and never promoted to camera_video_path.
        self._legacy_inventory = VidimuAdapter(
            dataset_archive_root,
            video_archive_root=wrapper,
            video_archive=video_archive,
            dataset_archive_sha256=dataset_archive_sha256,
            video_archive_sha256=video_archive_sha256,
            inventory_scope=inventory_scope,
            terms_sha256=terms_sha256,
            terms_path=terms_path,
        )
        self._video_archive_root_identity = self._directory_identity(
            self.video_archive_root,
            label="video archive root",
        )
        self._dataset_archive_root_identity = self._directory_identity(
            self._legacy_inventory.dataset_archive_root,
            label="dataset archive root",
        )
        self._wrapper_identity = self._trusted_directory_identity(
            self.video_wrapper,
            root=self.video_archive_root,
            root_identity=self._video_archive_root_identity,
            ancestor_identities=(),
            label="video archive wrapper",
        )
        self._camera_subtree_identity = self._trusted_directory_identity(
            self.camera_subtree,
            root=self.video_archive_root,
            root_identity=self._video_archive_root_identity,
            ancestor_identities=(self._wrapper_identity,),
            label="original-camera subtree",
        )
        self._bodytrack_subtree_identity = self._trusted_directory_identity(
            self.bodytrack_subtree,
            root=self.video_archive_root,
            root_identity=self._video_archive_root_identity,
            ancestor_identities=(self._wrapper_identity,),
            label="BodyTrack-QA subtree",
        )
        dataset_wrapper = (
            self._legacy_inventory.dataset_archive_root
            / Path(VIDIMU_DATASET_SUBTREE).parts[0]
        )
        self._dataset_wrapper_identity = self._trusted_directory_identity(
            dataset_wrapper,
            root=self._legacy_inventory.dataset_archive_root,
            root_identity=self._dataset_archive_root_identity,
            ancestor_identities=(),
            label="dataset wrapper",
        )
        self._dataset_subtree_identity = self._trusted_directory_identity(
            self._legacy_inventory.dataset_subtree,
            root=self._legacy_inventory.dataset_archive_root,
            root_identity=self._dataset_archive_root_identity,
            ancestor_identities=(self._dataset_wrapper_identity,),
            label="dataset subtree",
        )
        self._bindings_by_id: dict[str, _VidimuSourceBinding] = {}

    @staticmethod
    def _required_subtree(wrapper: Path, name: str) -> Path:
        candidate = wrapper / name
        if candidate.is_symlink() or not candidate.is_dir():
            raise VidimuAdapterError(
                f"VIDIMU selected video wrapper must contain {name}"
            )
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(wrapper):
            raise VidimuAdapterError(
                "VIDIMU selected video subtree escapes its archive"
            )
        return resolved

    @staticmethod
    def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise VidimuAdapterError(f"VIDIMU {label} changed") from exc
        try:
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise VidimuAdapterError(f"VIDIMU {label} is not a real directory")
        return metadata.st_dev, metadata.st_ino

    @staticmethod
    def _trusted_metadata(
        path: Path,
        *,
        root: Path,
        root_identity: tuple[int, int],
        ancestor_identities: tuple[tuple[int, int], ...],
        directory: bool,
        label: str,
    ) -> os.stat_result:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        if directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = _open_source_descriptor(
                path,
                flags=flags,
                trusted_root=root,
                trusted_root_identity=root_identity,
                trusted_ancestor_identities=ancestor_identities,
            )
            try:
                metadata = os.fstat(descriptor)
            finally:
                os.close(descriptor)
        except (OSError, VidimuSourceError) as exc:
            raise VidimuAdapterError(f"VIDIMU {label} changed") from exc
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_type(metadata.st_mode):
            raise VidimuAdapterError(f"VIDIMU {label} changed")
        return metadata

    @classmethod
    def _trusted_directory_identity(
        cls,
        path: Path,
        *,
        root: Path,
        root_identity: tuple[int, int],
        ancestor_identities: tuple[tuple[int, int], ...],
        label: str,
        expected_identity: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        metadata = cls._trusted_metadata(
            path,
            root=root,
            root_identity=root_identity,
            ancestor_identities=ancestor_identities,
            directory=True,
            label=label,
        )
        identity = metadata.st_dev, metadata.st_ino
        if expected_identity is not None and identity != expected_identity:
            raise VidimuAdapterError(f"VIDIMU {label} changed")
        return identity

    def _revalidate_roots(self) -> None:
        if (
            self._directory_identity(
                self.video_archive_root,
                label="video archive root",
            )
            != self._video_archive_root_identity
        ):
            raise VidimuAdapterError("VIDIMU video archive root changed")
        if (
            self._directory_identity(
                self._legacy_inventory.dataset_archive_root,
                label="dataset archive root",
            )
            != self._dataset_archive_root_identity
        ):
            raise VidimuAdapterError("VIDIMU dataset archive root changed")
        self._trusted_directory_identity(
            self.video_wrapper,
            root=self.video_archive_root,
            root_identity=self._video_archive_root_identity,
            ancestor_identities=(),
            expected_identity=self._wrapper_identity,
            label="video archive wrapper",
        )
        self._trusted_directory_identity(
            self.camera_subtree,
            root=self.video_archive_root,
            root_identity=self._video_archive_root_identity,
            ancestor_identities=(self._wrapper_identity,),
            expected_identity=self._camera_subtree_identity,
            label="original-camera subtree",
        )
        self._trusted_directory_identity(
            self.bodytrack_subtree,
            root=self.video_archive_root,
            root_identity=self._video_archive_root_identity,
            ancestor_identities=(self._wrapper_identity,),
            expected_identity=self._bodytrack_subtree_identity,
            label="BodyTrack-QA subtree",
        )
        dataset_wrapper = (
            self._legacy_inventory.dataset_archive_root
            / Path(VIDIMU_DATASET_SUBTREE).parts[0]
        )
        self._trusted_directory_identity(
            dataset_wrapper,
            root=self._legacy_inventory.dataset_archive_root,
            root_identity=self._dataset_archive_root_identity,
            ancestor_identities=(),
            expected_identity=self._dataset_wrapper_identity,
            label="dataset wrapper",
        )
        self._trusted_directory_identity(
            self._legacy_inventory.dataset_subtree,
            root=self._legacy_inventory.dataset_archive_root,
            root_identity=self._dataset_archive_root_identity,
            ancestor_identities=(self._dataset_wrapper_identity,),
            expected_identity=self._dataset_subtree_identity,
            label="dataset subtree",
        )

    @classmethod
    def _revalidate_member(
        cls,
        path: Path,
        *,
        root: Path,
        root_identity: tuple[int, int],
        ancestor_identities: tuple[tuple[int, int], ...],
        expected_name: str,
        label: str,
        expected_identity: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        if path.name != expected_name:
            raise VidimuAdapterError(f"VIDIMU {label} filename changed")
        metadata = cls._trusted_metadata(
            path,
            root=root,
            root_identity=root_identity,
            ancestor_identities=ancestor_identities,
            directory=False,
            label=label,
        )
        identity = _stat_identity(metadata)
        if expected_identity is not None and identity != expected_identity:
            raise VidimuAdapterError(f"VIDIMU {label} changed")
        return identity

    def _bind_recording(
        self,
        recording: VidimuSourceRecording,
    ) -> _VidimuSourceBinding:
        assert recording.bodytrack_pose_csv_path is not None
        assert recording.quaternion_raw_path is not None
        dataset_subject = self._legacy_inventory.dataset_subtree / recording.subject_id
        dataset_subject_identity = self._trusted_directory_identity(
            dataset_subject,
            root=self._legacy_inventory.dataset_subtree,
            root_identity=self._dataset_subtree_identity,
            ancestor_identities=(),
            label="dataset subject directory",
        )
        pose_identity = self._revalidate_member(
            recording.bodytrack_pose_csv_path,
            root=self._legacy_inventory.dataset_subtree,
            root_identity=self._dataset_subtree_identity,
            ancestor_identities=(dataset_subject_identity,),
            expected_name=f"{recording.recording_id}.csv",
            label="BodyTrack CSV source",
        )
        raw_identity = self._revalidate_member(
            recording.quaternion_raw_path,
            root=self._legacy_inventory.dataset_subtree,
            root_identity=self._dataset_subtree_identity,
            ancestor_identities=(dataset_subject_identity,),
            expected_name=f"{recording.recording_id}.raw",
            label="RAW source",
        )
        camera_subject_identity: tuple[int, int] | None = None
        camera_identity: tuple[int, ...] | None = None
        if recording.camera_video_path is not None:
            camera_subject_identity = self._trusted_directory_identity(
                self.camera_subtree / recording.subject_id,
                root=self.camera_subtree,
                root_identity=self._camera_subtree_identity,
                ancestor_identities=(),
                label="original-camera subject directory",
            )
            camera_identity = self._revalidate_member(
                recording.camera_video_path,
                root=self.camera_subtree,
                root_identity=self._camera_subtree_identity,
                ancestor_identities=(camera_subject_identity,),
                expected_name=f"{recording.recording_id}.mp4",
                label="original-camera candidate",
            )
        bodytrack_subject_identity: tuple[int, int] | None = None
        qa_identity: tuple[int, ...] | None = None
        if recording.bodytrack_qa_video_path is not None:
            bodytrack_subject_identity = self._trusted_directory_identity(
                self.bodytrack_subtree / recording.subject_id,
                root=self.bodytrack_subtree,
                root_identity=self._bodytrack_subtree_identity,
                ancestor_identities=(),
                label="BodyTrack-QA subject directory",
            )
            qa_identity = self._revalidate_member(
                recording.bodytrack_qa_video_path,
                root=self.bodytrack_subtree,
                root_identity=self._bodytrack_subtree_identity,
                ancestor_identities=(bodytrack_subject_identity,),
                expected_name=f"{recording.recording_id}_pose.mp4",
                label="BodyTrack-QA video",
            )
        if (
            self._trusted_directory_identity(
                dataset_subject,
                root=self._legacy_inventory.dataset_subtree,
                root_identity=self._dataset_subtree_identity,
                ancestor_identities=(),
                label="dataset subject directory",
            )
            != dataset_subject_identity
        ):
            raise VidimuAdapterError("VIDIMU dataset subject directory changed")
        if (
            camera_subject_identity is not None
            and self._trusted_directory_identity(
                self.camera_subtree / recording.subject_id,
                root=self.camera_subtree,
                root_identity=self._camera_subtree_identity,
                ancestor_identities=(),
                label="original-camera subject directory",
            )
            != camera_subject_identity
        ):
            raise VidimuAdapterError("VIDIMU original-camera subject directory changed")
        if (
            bodytrack_subject_identity is not None
            and self._trusted_directory_identity(
                self.bodytrack_subtree / recording.subject_id,
                root=self.bodytrack_subtree,
                root_identity=self._bodytrack_subtree_identity,
                ancestor_identities=(),
                label="BodyTrack-QA subject directory",
            )
            != bodytrack_subject_identity
        ):
            raise VidimuAdapterError("VIDIMU BodyTrack-QA subject directory changed")
        return _VidimuSourceBinding(
            recording=recording,
            dataset_subject_identity=dataset_subject_identity,
            camera_subject_identity=camera_subject_identity,
            bodytrack_subject_identity=bodytrack_subject_identity,
            camera_file_identity=camera_identity,
            pose_file_identity=pose_identity,
            raw_file_identity=raw_identity,
            qa_file_identity=qa_identity,
        )

    def _original_residues(self) -> dict[str, str]:
        prefix = f"{self.video_archive}/{_ORIGINAL_VIDEO_SUBTREE}"
        return {
            f"{prefix}/S24/S25_A02_T01.mp4": "S25_A02_T01",
            f"{prefix}/S49/S49_A13_T01V2_Npose.mp4": "S49_A13_T01",
        }

    def _discover_original_videos(self) -> dict[str, Path]:
        result: dict[str, Path] = {}
        residues = self._original_residues()
        ignored_metadata = set()
        if self.video_archive == "videosfullsize":
            ignored_metadata.add(
                f"{self.video_archive}/{_ORIGINAL_VIDEO_SUBTREE}/toBodyTrack.txt"
            )
        for path in sorted(self.camera_subtree.rglob("*")):
            if path.is_symlink():
                raise VidimuAdapterError(
                    "VIDIMU original-camera subtree must not contain symlinks"
                )
            if not path.is_file():
                continue
            archive_relative = path.relative_to(self.video_archive_root).as_posix()
            if archive_relative in ignored_metadata or archive_relative in residues:
                continue
            relative = path.relative_to(self.camera_subtree)
            if len(relative.parts) != 2 or path.suffix != ".mp4":
                raise VidimuAdapterError(
                    "unexpected VIDIMU original-video filename or location: "
                    f"{relative.as_posix()}"
                )
            npose = _ORIGINAL_NPOSE_RE.fullmatch(path.name)
            if npose is not None:
                recording_id, subject, _, _ = parse_recording_id(
                    npose.group("recording")
                )
                if relative.parts[0] != subject:
                    raise VidimuAdapterError(
                        f"{recording_id} N-pose video is outside its subject subtree"
                    )
                continue
            try:
                _recording_layout(path.stem)
                recording_id, subject, _, _ = parse_recording_id(path.stem)
            except VidimuAdapterError as exc:
                raise VidimuAdapterError(
                    f"unexpected VIDIMU original-video filename: {path.name}"
                ) from exc
            except VidimuSourceError as exc:
                raise VidimuAdapterError(
                    f"unexpected VIDIMU original-video filename: {path.name}"
                ) from exc
            if relative.parts[0] != subject:
                raise VidimuAdapterError(
                    f"{recording_id} camera video is outside its subject subtree"
                )
            if recording_id in result:
                raise VidimuAdapterError(
                    f"multiple original camera videos for {recording_id}"
                )
            result[recording_id] = path.resolve(strict=True)
        return result

    def discover(self) -> tuple[VidimuSourceRecording, ...]:
        """Return paired data records with explicit original-video paths."""

        self._revalidate_roots()
        originals = self._discover_original_videos()
        legacy = self._legacy_inventory.discover()
        records: list[VidimuSourceRecording] = []
        for item in legacy:
            # VIDIMU's paired public subset is defined by same-stem CSV+RAW.
            # Render-only and video-only inventory entries are not source pairs.
            if item.pose_path is None or item.quaternion_path is None:
                continue
            try:
                _recording_layout(item.recording_id)
            except VidimuSourceError as exc:
                raise VidimuAdapterError(
                    "paired VIDIMU dataset source ID must use exact "
                    "Sxx_A01-A13_Txx case and padding"
                ) from exc
            records.append(
                VidimuSourceRecording(
                    recording_id=item.recording_id,
                    subject_id=item.subject_id,
                    activity_id=item.activity_id,
                    trial_id=item.trial_id,
                    camera_video_path=originals.get(item.recording_id),
                    bodytrack_pose_csv_path=item.pose_path,
                    quaternion_raw_path=item.quaternion_path,
                    bodytrack_qa_video_path=item.video_path,
                )
            )
        result = tuple(sorted(records, key=lambda item: item.recording_id))
        self._revalidate_roots()
        bindings = {item.recording_id: self._bind_recording(item) for item in result}
        self._revalidate_roots()
        self._bindings_by_id = bindings
        return result

    def _current_recording(
        self,
        recording: VidimuSourceRecording,
    ) -> _VidimuSourceBinding:
        if not isinstance(recording, VidimuSourceRecording):
            raise VidimuAdapterError("recording must be a VidimuSourceRecording")
        self._revalidate_roots()
        if not self._bindings_by_id:
            self.discover()
        binding = self._bindings_by_id.get(recording.recording_id)
        if binding is None or binding.recording != recording:
            raise VidimuAdapterError(
                "VIDIMU source recording changed or is not current inventory"
            )
        current = binding.recording
        if not current.inventory_complete:
            raise VidimuAdapterError(
                f"cannot parse incomplete VIDIMU source: {current.recording_id}"
            )
        assert current.camera_video_path is not None
        assert current.bodytrack_pose_csv_path is not None
        assert current.quaternion_raw_path is not None
        assert binding.camera_subject_identity is not None
        assert binding.camera_file_identity is not None
        self._revalidate_member(
            current.camera_video_path,
            root=self.camera_subtree,
            root_identity=self._camera_subtree_identity,
            ancestor_identities=(binding.camera_subject_identity,),
            expected_name=f"{current.recording_id}.mp4",
            label="original-camera candidate",
            expected_identity=binding.camera_file_identity,
        )
        self._revalidate_member(
            current.bodytrack_pose_csv_path,
            root=self._legacy_inventory.dataset_subtree,
            root_identity=self._dataset_subtree_identity,
            ancestor_identities=(binding.dataset_subject_identity,),
            expected_name=f"{current.recording_id}.csv",
            label="BodyTrack CSV source",
            expected_identity=binding.pose_file_identity,
        )
        self._revalidate_member(
            current.quaternion_raw_path,
            root=self._legacy_inventory.dataset_subtree,
            root_identity=self._dataset_subtree_identity,
            ancestor_identities=(binding.dataset_subject_identity,),
            expected_name=f"{current.recording_id}.raw",
            label="RAW source",
            expected_identity=binding.raw_file_identity,
        )
        if current.bodytrack_qa_video_path is not None:
            assert binding.bodytrack_subject_identity is not None
            assert binding.qa_file_identity is not None
            self._revalidate_member(
                current.bodytrack_qa_video_path,
                root=self.bodytrack_subtree,
                root_identity=self._bodytrack_subtree_identity,
                ancestor_identities=(binding.bodytrack_subject_identity,),
                expected_name=f"{current.recording_id}_pose.mp4",
                label="BodyTrack-QA video",
                expected_identity=binding.qa_file_identity,
            )
        if current.camera_video_path.name.endswith("_pose.mp4"):
            raise VidimuAdapterError(
                "BodyTrack-rendered video cannot be used as camera input"
            )
        return binding

    def parse_raw(
        self,
        recording: VidimuSourceRecording,
        *,
        stored_recording_id: str | None = None,
    ) -> VidimuRawCapture:
        binding = self._current_recording(recording)
        current = binding.recording
        assert current.quaternion_raw_path is not None
        capture = parse_vidimu_raw(
            current.quaternion_raw_path,
            source_recording_id=current.recording_id,
            stored_recording_id=stored_recording_id,
            _trusted_root=self._legacy_inventory.dataset_subtree,
            _trusted_root_identity=self._dataset_subtree_identity,
            _trusted_ancestor_identities=(binding.dataset_subject_identity,),
            _trusted_file_identity=binding.raw_file_identity,
        )
        self._current_recording(recording)
        return capture

    def parse_pose(
        self,
        recording: VidimuSourceRecording,
    ) -> VidimuPoseCapture:
        binding = self._current_recording(recording)
        current = binding.recording
        assert current.bodytrack_pose_csv_path is not None
        capture = parse_vidimu_pose_csv(
            current.bodytrack_pose_csv_path,
            source_recording_id=current.recording_id,
            _trusted_root=self._legacy_inventory.dataset_subtree,
            _trusted_root_identity=self._dataset_subtree_identity,
            _trusted_ancestor_identities=(binding.dataset_subject_identity,),
            _trusted_file_identity=binding.pose_file_identity,
        )
        self._current_recording(recording)
        return capture


__all__ = [
    "VIDIMU_LOWER_BODY_LAYOUT",
    "VIDIMU_NOMINAL_IMU_HZ",
    "VIDIMU_POSE_JOINTS",
    "VIDIMU_POSE_SOURCE_HEADER",
    "VIDIMU_QUATERNION_NORM_ABS_TOL",
    "VIDIMU_RAW_HEADER",
    "VIDIMU_RAW_OBSERVATION_KIND",
    "VIDIMU_RAW_TIMESTAMP_INTERPRETATION",
    "VIDIMU_SENSOR_BODY_LOCATIONS",
    "VIDIMU_SOURCE_PARSER_VERSION",
    "VIDIMU_UPPER_BODY_LAYOUT",
    "VidimuCalibrationQuaternion",
    "VidimuCameraSourceAdapter",
    "VidimuInvalidQuaternion",
    "VidimuPoseCapture",
    "VidimuRawCapture",
    "VidimuRawStreamStatistics",
    "VidimuSourceError",
    "VidimuSourceRecording",
    "parse_vidimu_pose_csv",
    "parse_vidimu_raw",
    "vidimu_pose_source_schema",
    "vidimu_raw_source_schema",
]
