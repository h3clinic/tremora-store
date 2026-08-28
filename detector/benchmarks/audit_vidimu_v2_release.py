"""Reproducible release audit for the pinned public VIDIMU v2.0.0 data.

The 336.8 MB ``dataset.zip`` is read in full, matched to Zenodo's published
size/MD5, assigned a local SHA-256, and its canonical CSV/RAW pairs are parsed
with MotionBloom's versioned source-only parser.  The two video archives are
not downloaded: one bounded HTTP range from the tail of each archive is enough
to validate and hash its ZIP/ZIP64 central directory.

Central-directory evidence proves member names and ZIP metadata only.  It does
not recompute either video archive's published MD5, hash video member bytes,
decode video, or establish synchronization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import struct
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final

from motionbloom.tremora_store.adapters import vidimu_source
from motionbloom.tremora_store.adapters.vidimu import (
    VIDIMU_CONCEPT_DOI,
    VIDIMU_LICENSE_SPDX,
    VIDIMU_RECORD_DOI,
    VIDIMU_RECORD_URL,
    VIDIMU_RELEASE_VERSION,
    VIDIMU_ZENODO_RECORD_ID,
)
from motionbloom.tremora_store.adapters.vidimu_source import (
    VIDIMU_QUATERNION_NORM_ABS_TOL,
    VIDIMU_SOURCE_PARSER_VERSION,
    parse_vidimu_pose_csv,
    parse_vidimu_raw,
)
from motionbloom.tremora_store.schema import SCHEMA_VERSION

AUDIT_SCHEMA_VERSION = "1.0"
AUDIT_IMPLEMENTATION_VERSION = "vidimu-v2-release-audit-v1.0.0"
AUDIT_ARTIFACT_KIND = "VIDIMU_V2_RELEASE_AUDIT"
ZENODO_RECORD_API_URL = "https://zenodo.org/api/records/15075076"
REMOTE_TAIL_BYTES = 1024 * 1024
HTTP_TIMEOUT_SECONDS = 120
MAX_RECORD_METADATA_BYTES = 2 * 1024 * 1024
MAX_RAW_MEMBER_BYTES = 16 * 1024 * 1024
MAX_POSE_MEMBER_BYTES = 4 * 1024 * 1024

_ARCHIVE_PINS: Final = {
    "dataset.zip": {
        "size_bytes": 336_819_642,
        "md5": "368d34d13651b44e6d4444c4a6c41380",
        "content_url": (
            "https://zenodo.org/api/records/15075076/files/dataset.zip/content"
        ),
    },
    "videosmallsize.zip": {
        "size_bytes": 1_095_515_926,
        "md5": "b3cbfadf8cf719f47a748a6b5eeb3f2b",
        "content_url": (
            "https://zenodo.org/api/records/15075076/files/videosmallsize.zip/content"
        ),
    },
    "videosfullsize.zip": {
        "size_bytes": 41_304_165_581,
        "md5": "c6e48c41f7b6d071ceb050e9c13c3205",
        "content_url": (
            "https://zenodo.org/api/records/15075076/files/videosfullsize.zip/content"
        ),
    },
}

_EXPECTED_CSV_ONLY: Final = (
    "S54_A13_T01",
    "S56_A01_T02",
    "S56_A02_T01",
    "S56_A03_T02",
    "S56_A04_T02",
    "S57_A04_T02",
    "S57_A11_T02",
)
_EXPECTED_NONZERO_NPOSE: Final = (
    "S41_A02_T01",
    "S41_A03_T01",
    "S41_A04_T01",
    "S41_A05_T03",
    "S41_A06_T01",
    "S41_A07_T01",
    "S41_A08_T01",
    "S41_A09_T01",
    "S41_A10_T01",
    "S41_A12_T02",
    "S41_A13_T02",
)
_EXPECTED_QA_MISSING: Final = (
    "S57_A04_T01",
    "S57_A11_T01",
)

_DATA_MEMBER_RE = re.compile(
    r"^dataset/videoandimus/(?P<subject>S[0-9]{2})/"
    r"(?P<recording>(?P=subject)_A(?:0[1-9]|1[0-3])_T[0-9]{2})"
    r"\.(?P<extension>csv|raw)$"
)
_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_EOCD = struct.Struct("<4s4H2LH")
_ZIP64_LOCATOR = struct.Struct("<4sLQL")
_ZIP64_EOCD = struct.Struct("<4sQ2H2L4Q")
_CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
_FILE_IDENTITY_FIELDS: Final = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


class AuditError(RuntimeError):
    """Raised when release evidence does not satisfy the pinned audit."""


@dataclass(frozen=True, slots=True)
class CentralEntry:
    """Normalized semantic fields from one ZIP central-directory record."""

    path: str
    crc32_hex: str
    compressed_size: int
    uncompressed_size: int
    compression_method: int
    general_purpose_bit_flags: int
    external_attributes: int
    local_header_offset: int
    is_directory: bool

    def public_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "crc32_hex": self.crc32_hex,
            "compressed_size": self.compressed_size,
            "uncompressed_size": self.uncompressed_size,
            "compression_method": self.compression_method,
            "general_purpose_bit_flags": self.general_purpose_bit_flags,
            "external_attributes": self.external_attributes,
            "local_header_offset": self.local_header_offset,
            "is_directory": self.is_directory,
        }


@dataclass(frozen=True, slots=True)
class CentralDirectoryAudit:
    entries: tuple[CentralEntry, ...]
    central_directory_offset: int
    central_directory_size_bytes: int
    central_directory_sha256: str
    zip64: bool
    archive_comment_length: int


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    archive: dict[str, object]
    records: tuple[dict[str, object], ...]
    recording_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VideoAudit:
    archive: dict[str, object]
    selected_members: Mapping[str, dict[str, object]]


def _compact_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_bytes(value: object) -> bytes:
    """Canonical checked-artifact encoding used by this audit."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            separators=(",", ": "),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> dict[str, object]:
    md5 = hashlib.md5(usedforsecurity=False)  # Zenodo's published algorithm.
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {"size_bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _normalized_source_sha256(path: Path) -> str:
    """Hash UTF-8 implementation text after canonical LF normalization."""

    payload = path.read_bytes()
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AuditError("implementation source must be strict UTF-8") from exc
    normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return _sha256_bytes(normalized)


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(metadata, field) for field in _FILE_IDENTITY_FIELDS)


def _open_pinned_dataset_archive(path: Path) -> tuple[BinaryIO, os.stat_result]:
    """Open one no-follow regular-file descriptor and bind its path identity."""

    try:
        path_metadata = path.lstat()
    except OSError as exc:
        raise AuditError("cannot inspect local dataset ZIP") from exc
    if not stat.S_ISREG(path_metadata.st_mode):
        raise AuditError("dataset ZIP must be a non-symlink regular file")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AuditError("cannot open local dataset ZIP safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _file_identity(opened) != _file_identity(
            path_metadata
        ):
            raise AuditError("local dataset ZIP changed while it was opened")
        return os.fdopen(descriptor, "rb"), opened
    except Exception:
        os.close(descriptor)
        raise


def _hash_stream(handle: BinaryIO) -> dict[str, object]:
    """Hash all bytes from a seekable descriptor without changing identity."""

    md5 = hashlib.md5(usedforsecurity=False)  # Zenodo's published algorithm.
    sha256 = hashlib.sha256()
    size = 0
    handle.seek(0)
    while chunk := handle.read(1024 * 1024):
        size += len(chunk)
        md5.update(chunk)
        sha256.update(chunk)
    return {"size_bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _revalidate_pinned_dataset_archive(
    path: Path,
    handle: BinaryIO,
    *,
    opened: os.stat_result,
    observed_size: int,
) -> None:
    """Reject path replacement or descriptor mutation during the audit."""

    try:
        after = os.fstat(handle.fileno())
        path_after = path.lstat()
    except OSError as exc:
        raise AuditError("local dataset ZIP changed during the audit") from exc
    expected = _file_identity(opened)
    if (
        observed_size != opened.st_size
        or _file_identity(after) != expected
        or _file_identity(path_after) != expected
    ):
        raise AuditError("local dataset ZIP changed during the audit")


def _atomic_write(
    path: Path,
    payload: bytes,
    *,
    forbidden_identity: tuple[int, int],
) -> None:
    """Publish through one pinned parent descriptor without following aliases."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.name or path.name in {".", ".."}:
        raise AuditError("audit output filename is invalid")
    if (
        os.open not in getattr(os, "supports_dir_fd", ())
        or os.stat not in getattr(os, "supports_dir_fd", ())
        or os.unlink not in getattr(os, "supports_dir_fd", ())
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        raise AuditError(
            "secure audit publication requires POSIX no-follow dir_fd support"
        )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        parent_descriptor = os.open(path.parent, directory_flags)
    except OSError as exc:
        raise AuditError("cannot pin audit output parent directory") from exc
    parent_metadata = os.fstat(parent_descriptor)
    parent_identity = parent_metadata.st_dev, parent_metadata.st_ino
    temporary_name: str | None = None
    temporary_identity: tuple[int, int] | None = None
    descriptor = -1
    try:
        try:
            destination = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            destination = None
        except OSError as exc:
            raise AuditError("cannot inspect audit output destination") from exc
        if destination is not None:
            if stat.S_ISLNK(destination.st_mode):
                raise AuditError("audit output destination must not be a symlink")
            if (destination.st_dev, destination.st_ino) == forbidden_identity:
                raise AuditError("audit output destination aliases dataset ZIP")
            if not stat.S_ISREG(destination.st_mode):
                raise AuditError("audit output destination must be a regular file")
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        open_flags |= getattr(os, "O_CLOEXEC", 0)
        for _attempt in range(128):
            candidate = f".tremora-audit-{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    open_flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            except OSError as exc:
                raise AuditError("cannot create audit output temporary file") from exc
            temporary_name = candidate
            temporary_metadata = os.fstat(descriptor)
            temporary_identity = (
                temporary_metadata.st_dev,
                temporary_metadata.st_ino,
            )
            break
        if temporary_name is None or temporary_identity is None or descriptor < 0:
            raise AuditError("cannot allocate a unique audit output temporary file")
        try:
            os.fchmod(descriptor, 0o644)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(
                    temporary_name,
                    path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
            except (NotImplementedError, OSError, TypeError) as exc:
                raise AuditError(
                    "cannot atomically publish audit through pinned parent"
                ) from exc
            temporary_name = None
            os.fsync(parent_descriptor)
            try:
                published = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise AuditError("cannot verify published audit destination") from exc
            if (
                not stat.S_ISREG(published.st_mode)
                or (published.st_dev, published.st_ino) != temporary_identity
                or published.st_size != len(payload)
            ):
                raise AuditError("published audit destination identity is inconsistent")
            try:
                current_parent_descriptor = os.open(path.parent, directory_flags)
            except OSError as exc:
                raise AuditError(
                    "audit output parent changed during publication"
                ) from exc
            try:
                current_parent = os.fstat(current_parent_descriptor)
            finally:
                os.close(current_parent_descriptor)
            if (current_parent.st_dev, current_parent.st_ino) != parent_identity:
                raise AuditError("audit output parent changed during publication")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        status = getcode() if getcode is not None else None
    if not isinstance(status, int):
        raise AuditError("HTTP response has no numeric status")
    return status


def fetch_record_metadata(
    url: str = ZENODO_RECORD_API_URL,
    *,
    opener: Callable[..., object] = urllib.request.urlopen,
    timeout_seconds: int = HTTP_TIMEOUT_SECONDS,
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": f"TremoraStore/{AUDIT_IMPLEMENTATION_VERSION}",
        },
    )
    with opener(request, timeout=timeout_seconds) as response:  # type: ignore[attr-defined]
        if _response_status(response) != 200:
            raise AuditError("Zenodo record API did not return HTTP 200")
        payload = response.read(MAX_RECORD_METADATA_BYTES + 1)  # type: ignore[attr-defined]
    if len(payload) > MAX_RECORD_METADATA_BYTES:
        raise AuditError("Zenodo record metadata exceeds the audit byte limit")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError("Zenodo record metadata is not valid JSON") from exc
    if not isinstance(value, dict):
        raise AuditError("Zenodo record metadata must be a JSON object")
    return value


def _fetch_tail_range(
    url: str,
    *,
    archive_size: int,
    opener: Callable[..., object],
    timeout_seconds: int,
) -> tuple[bytes, dict[str, object]]:
    response_size = min(REMOTE_TAIL_BYTES, archive_size)
    range_start = archive_size - response_size
    range_end = archive_size - 1
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={range_start}-{range_end}",
            "User-Agent": f"TremoraStore/{AUDIT_IMPLEMENTATION_VERSION}",
        },
    )
    with opener(request, timeout=timeout_seconds) as response:  # type: ignore[attr-defined]
        if _response_status(response) != 206:
            raise AuditError("video archive range request did not return HTTP 206")
        headers = response.headers  # type: ignore[attr-defined]
        expected_range = f"bytes {range_start}-{range_end}/{archive_size}"
        if headers.get("Content-Range") != expected_range:
            raise AuditError("video archive returned an unexpected Content-Range")
        if headers.get("Content-Encoding") not in (None, "identity"):
            raise AuditError("video archive range response was transparently encoded")
        content_length = headers.get("Content-Length")
        if content_length is not None and content_length != str(response_size):
            raise AuditError("video archive range Content-Length is inconsistent")
        payload = response.read(response_size + 1)  # type: ignore[attr-defined]
    if len(payload) != response_size:
        raise AuditError("video archive range body length is inconsistent")
    return payload, {
        "range_start": range_start,
        "range_end_inclusive": range_end,
        "response_byte_size": len(payload),
        "request_count": 1,
    }


def _find_eocd(tail: bytes) -> int:
    cursor = len(tail)
    while True:
        index = tail.rfind(_EOCD_SIGNATURE, 0, cursor)
        if index < 0:
            raise AuditError("ZIP end-of-central-directory record is absent")
        if index + _EOCD.size <= len(tail):
            values = _EOCD.unpack_from(tail, index)
            comment_length = values[-1]
            if index + _EOCD.size + comment_length == len(tail):
                return index
        cursor = index


def _zip64_value(
    payload: bytes,
    cursor: int,
    width: int,
    *,
    label: str,
) -> tuple[int, int]:
    if cursor + width > len(payload):
        raise AuditError(f"ZIP64 extra field omits {label}")
    format_string = "<Q" if width == 8 else "<L"
    return struct.unpack_from(format_string, payload, cursor)[0], cursor + width


def _resolve_zip64_fields(
    extra: bytes,
    *,
    uncompressed_size: int,
    compressed_size: int,
    local_header_offset: int,
    disk_start: int,
) -> tuple[int, int, int, int]:
    zip64_payload: bytes | None = None
    cursor = 0
    while cursor < len(extra):
        if cursor + 4 > len(extra):
            raise AuditError("ZIP central extra field is truncated")
        identifier, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        if cursor + size > len(extra):
            raise AuditError("ZIP central extra field payload is truncated")
        if identifier == 0x0001:
            if zip64_payload is not None:
                raise AuditError("ZIP central entry has duplicate ZIP64 extras")
            zip64_payload = extra[cursor : cursor + size]
        cursor += size
    needs_zip64 = (
        uncompressed_size == 0xFFFFFFFF
        or compressed_size == 0xFFFFFFFF
        or local_header_offset == 0xFFFFFFFF
        or disk_start == 0xFFFF
    )
    if not needs_zip64:
        return uncompressed_size, compressed_size, local_header_offset, disk_start
    if zip64_payload is None:
        raise AuditError("ZIP64 central entry lacks its ZIP64 extra field")
    cursor = 0
    if uncompressed_size == 0xFFFFFFFF:
        uncompressed_size, cursor = _zip64_value(
            zip64_payload, cursor, 8, label="uncompressed size"
        )
    if compressed_size == 0xFFFFFFFF:
        compressed_size, cursor = _zip64_value(
            zip64_payload, cursor, 8, label="compressed size"
        )
    if local_header_offset == 0xFFFFFFFF:
        local_header_offset, cursor = _zip64_value(
            zip64_payload, cursor, 8, label="local header offset"
        )
    if disk_start == 0xFFFF:
        disk_start, cursor = _zip64_value(zip64_payload, cursor, 4, label="disk start")
    return uncompressed_size, compressed_size, local_header_offset, disk_start


def _safe_member_path(path: str) -> None:
    try:
        path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AuditError("VIDIMU ZIP member paths must be ASCII") from exc
    pure = PurePosixPath(path)
    normalized = pure.as_posix()
    if (
        not path
        or path in {".", ".."}
        or "\\" in path
        or path.startswith("/")
        or "\x00" in path
        or normalized != path.rstrip("/")
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise AuditError(f"unsafe ZIP member path: {path!r}")


def _parse_central_entries(
    payload: bytes, expected_count: int
) -> tuple[CentralEntry, ...]:
    entries: list[CentralEntry] = []
    seen: set[str] = set()
    cursor = 0
    while cursor < len(payload):
        if cursor + _CENTRAL_HEADER.size > len(payload):
            raise AuditError("ZIP central-directory header is truncated")
        values = _CENTRAL_HEADER.unpack_from(payload, cursor)
        if values[0] != _CENTRAL_SIGNATURE:
            raise AuditError("ZIP central-directory entry signature is invalid")
        (
            _,
            _version_made,
            _version_needed,
            flags,
            compression,
            _modified_time,
            _modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
            comment_length,
            disk_start,
            _internal_attributes,
            external_attributes,
            local_header_offset,
        ) = values
        variable_start = cursor + _CENTRAL_HEADER.size
        variable_end = variable_start + filename_length + extra_length + comment_length
        if variable_end > len(payload):
            raise AuditError("ZIP central-directory variable fields are truncated")
        filename_bytes = payload[variable_start : variable_start + filename_length]
        extra_start = variable_start + filename_length
        extra = payload[extra_start : extra_start + extra_length]
        encoding = "utf-8" if flags & 0x0800 else "cp437"
        try:
            path = filename_bytes.decode(encoding, errors="strict")
        except UnicodeDecodeError as exc:
            raise AuditError("ZIP central member name cannot be decoded") from exc
        _safe_member_path(path)
        if path in seen:
            raise AuditError(f"duplicate ZIP central member path: {path}")
        seen.add(path)
        (
            uncompressed_size,
            compressed_size,
            local_header_offset,
            disk_start,
        ) = _resolve_zip64_fields(
            extra,
            uncompressed_size=uncompressed_size,
            compressed_size=compressed_size,
            local_header_offset=local_header_offset,
            disk_start=disk_start,
        )
        if disk_start != 0:
            raise AuditError("multi-disk ZIP archives are unsupported")
        entries.append(
            CentralEntry(
                path=path,
                crc32_hex=f"{crc32:08x}",
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                compression_method=compression,
                general_purpose_bit_flags=flags,
                external_attributes=external_attributes,
                local_header_offset=local_header_offset,
                is_directory=path.endswith("/"),
            )
        )
        cursor = variable_end
    if len(entries) != expected_count:
        raise AuditError(
            "ZIP central-directory count does not match its end record: "
            f"{len(entries)} != {expected_count}"
        )
    return tuple(entries)


def parse_central_directory_tail(
    tail: bytes,
    *,
    archive_size: int,
    tail_start: int,
) -> CentralDirectoryAudit:
    """Parse and hash a complete ZIP central directory contained in ``tail``."""

    eocd_relative = _find_eocd(tail)
    eocd_absolute = tail_start + eocd_relative
    values = _EOCD.unpack_from(tail, eocd_relative)
    (
        _,
        disk_number,
        central_disk,
        entries_on_disk,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = values
    zip64 = (
        entries_on_disk == 0xFFFF
        or total_entries == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    )
    if zip64:
        locator_absolute = eocd_absolute - _ZIP64_LOCATOR.size
        locator_relative = locator_absolute - tail_start
        if locator_relative < 0:
            raise AuditError("ZIP64 locator lies outside the audited tail")
        locator = _ZIP64_LOCATOR.unpack_from(tail, locator_relative)
        if locator[0] != _ZIP64_LOCATOR_SIGNATURE:
            raise AuditError("ZIP64 locator signature is invalid")
        _, zip64_disk, zip64_offset, total_disks = locator
        zip64_relative = zip64_offset - tail_start
        if zip64_relative < 0 or zip64_relative + _ZIP64_EOCD.size > len(tail):
            raise AuditError("ZIP64 end record lies outside the audited tail")
        zip64_values = _ZIP64_EOCD.unpack_from(tail, zip64_relative)
        if zip64_values[0] != _ZIP64_EOCD_SIGNATURE:
            raise AuditError("ZIP64 end record signature is invalid")
        (
            _,
            zip64_record_size,
            _made_by,
            _needed,
            disk_number,
            central_disk,
            entries_on_disk,
            total_entries,
            central_size,
            central_offset,
        ) = zip64_values
        if zip64_record_size < 44 or zip64_disk != 0 or total_disks != 1:
            raise AuditError("unsupported ZIP64 end-record topology")
    if disk_number != 0 or central_disk != 0 or entries_on_disk != total_entries:
        raise AuditError("multi-disk ZIP archives are unsupported")
    central_end = central_offset + central_size
    if central_offset < tail_start or central_end > eocd_absolute:
        raise AuditError(
            "complete ZIP central directory is not contained in the bounded tail"
        )
    relative_start = central_offset - tail_start
    central_payload = tail[relative_start : relative_start + central_size]
    entries = _parse_central_entries(central_payload, total_entries)
    return CentralDirectoryAudit(
        entries=entries,
        central_directory_offset=central_offset,
        central_directory_size_bytes=central_size,
        central_directory_sha256=_sha256_bytes(central_payload),
        zip64=zip64,
        archive_comment_length=comment_length,
    )


def _read_local_central(handle: BinaryIO, archive_size: int) -> CentralDirectoryAudit:
    tail_size = min(REMOTE_TAIL_BYTES, archive_size)
    handle.seek(archive_size - tail_size)
    tail = handle.read(tail_size)
    if len(tail) != tail_size:
        raise AuditError("local ZIP tail could not be read completely")
    return parse_central_directory_tail(
        tail,
        archive_size=archive_size,
        tail_start=archive_size - tail_size,
    )


def _inventory_sha256(entries: Sequence[CentralEntry]) -> str:
    normalized = [
        entry.public_dict() for entry in sorted(entries, key=lambda item: item.path)
    ]
    digest = hashlib.sha256()
    digest.update(b"TREMORA_VIDIMU_ZIP_CENTRAL_INVENTORY_V1\0")
    digest.update(_compact_json_bytes(normalized))
    return digest.hexdigest()


def _central_summary(audit: CentralDirectoryAudit) -> dict[str, object]:
    entries = audit.entries
    symlink_count = 0
    for entry in entries:
        mode = (entry.external_attributes >> 16) & 0xFFFF
        symlink_count += int(stat.S_ISLNK(mode))
    return {
        "archive_comment_length": audit.archive_comment_length,
        "central_directory_offset": audit.central_directory_offset,
        "central_directory_size_bytes": audit.central_directory_size_bytes,
        "central_directory_sha256": audit.central_directory_sha256,
        "directory_entry_count": sum(entry.is_directory for entry in entries),
        "encrypted_entry_count": sum(
            bool(entry.general_purpose_bit_flags & 1) for entry in entries
        ),
        "entry_count": len(entries),
        "file_entry_count": sum(not entry.is_directory for entry in entries),
        "inventory_sha256": _inventory_sha256(entries),
        "symlink_entry_count": symlink_count,
        "zip64": audit.zip64,
    }


def _validate_record_metadata(
    value: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    metadata = value.get("metadata")
    files = value.get("files")
    if not isinstance(metadata, Mapping) or not isinstance(files, list):
        raise AuditError("Zenodo record metadata has an unexpected schema")
    license_value = metadata.get("license")
    if not isinstance(license_value, Mapping):
        raise AuditError("Zenodo license metadata is absent")
    expected_identity = {
        "record_id": VIDIMU_ZENODO_RECORD_ID,
        "concept_record_id": "7681316",
        "doi": VIDIMU_RECORD_DOI,
        "version": f"v{VIDIMU_RELEASE_VERSION}",
        "publication_date": "2023-09-22",
        "license_id": VIDIMU_LICENSE_SPDX.lower(),
    }
    observed_identity = {
        "record_id": value.get("id"),
        "concept_record_id": value.get("conceptrecid"),
        "doi": value.get("doi"),
        "version": metadata.get("version"),
        "publication_date": metadata.get("publication_date"),
        "license_id": license_value.get("id"),
    }
    if observed_identity != expected_identity:
        raise AuditError(f"Zenodo record identity changed: {observed_identity!r}")
    by_key: dict[str, Mapping[str, object]] = {}
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("key"), str):
            raise AuditError("Zenodo file metadata is malformed")
        by_key[str(item["key"])] = item
    published: dict[str, dict[str, object]] = {}
    for key, pin in _ARCHIVE_PINS.items():
        item = by_key.get(key)
        if item is None:
            raise AuditError(f"Zenodo record omits {key}")
        links = item.get("links")
        checksum = item.get("checksum")
        if not isinstance(links, Mapping) or not isinstance(checksum, str):
            raise AuditError(f"Zenodo file metadata for {key} is incomplete")
        observed = {
            "size_bytes": item.get("size"),
            "md5": checksum.removeprefix("md5:"),
            "content_url": links.get("self"),
        }
        if observed != pin:
            raise AuditError(f"Zenodo file identity changed for {key}: {observed!r}")
        published[key] = {
            "content_url": observed["content_url"],
            "published_checksum": {
                "algorithm": "md5",
                "value": observed["md5"],
                "source": "ZENODO_RECORD_API",
            },
            "size_bytes": observed["size_bytes"],
        }
    metadata_subset = {
        **observed_identity,
        "archives": {
            key: {
                "content_url": published[key]["content_url"],
                "md5": published[key]["published_checksum"]["value"],  # type: ignore[index]
                "size_bytes": published[key]["size_bytes"],
            }
            for key in sorted(published)
        },
    }
    release = {
        **observed_identity,
        "concept_doi": VIDIMU_CONCEPT_DOI,
        "record_api_url": ZENODO_RECORD_API_URL,
        "record_metadata_subset_sha256": _sha256_bytes(
            _compact_json_bytes(metadata_subset)
        ),
        "record_url": VIDIMU_RECORD_URL,
    }
    return published, release


def _dataset_inventory(
    entries: Sequence[CentralEntry],
) -> tuple[dict[str, CentralEntry], dict[str, CentralEntry]]:
    csv_by_id: dict[str, CentralEntry] = {}
    raw_by_id: dict[str, CentralEntry] = {}
    for entry in entries:
        if entry.is_directory:
            continue
        match = _DATA_MEMBER_RE.fullmatch(entry.path)
        if match is None:
            continue
        target = csv_by_id if match.group("extension") == "csv" else raw_by_id
        recording_id = match.group("recording")
        if recording_id in target:
            raise AuditError(f"duplicate canonical source for {recording_id}")
        target[recording_id] = entry
    return csv_by_id, raw_by_id


def _copy_zip_member(
    archive: zipfile.ZipFile,
    *,
    member: CentralEntry,
    target: Path,
    max_bytes: int,
) -> None:
    info = archive.getinfo(member.path)
    if (
        info.file_size != member.uncompressed_size
        or info.compress_size != member.compressed_size
        or f"{info.CRC:08x}" != member.crc32_hex
    ):
        raise AuditError(f"ZIP reader contradicts central metadata for {member.path}")
    if info.flag_bits & 1 or info.file_size > max_bytes:
        raise AuditError(f"unsafe or oversized selected member: {member.path}")
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        raise AuditError(f"selected source member is a symlink: {member.path}")
    observed = 0
    with archive.open(info, "r") as source, target.open("xb") as output:
        while chunk := source.read(1024 * 1024):
            observed += len(chunk)
            if observed > max_bytes:
                raise AuditError(
                    f"selected member expanded beyond its limit: {member.path}"
                )
            output.write(chunk)
    if observed != info.file_size:
        raise AuditError(f"selected member length changed while reading: {member.path}")


def _ratio_dict(numerator: int, denominator: int) -> dict[str, object]:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise AuditError("held-observation ratio is outside [0,1]")
    value = Fraction(numerator, denominator)
    return {
        "decimal_6": _fraction_decimal_6(value),
        "denominator": value.denominator,
        "numerator": value.numerator,
    }


def _fraction_decimal_6(value: Fraction) -> str:
    with localcontext() as context:
        context.prec = 50
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        return format(
            decimal_value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN),
            ".6f",
        )


def _parse_source_pair(
    archive: zipfile.ZipFile,
    *,
    recording_id: str,
    csv_entry: CentralEntry,
    raw_entry: CentralEntry,
) -> dict[str, object]:
    subject_id = recording_id.split("_", maxsplit=1)[0]
    with tempfile.TemporaryDirectory(prefix="vidimu-source-audit-") as temporary:
        root = Path(temporary)
        csv_path = root / f"{recording_id}.csv"
        raw_path = root / f"{recording_id}.raw"
        _copy_zip_member(
            archive,
            member=csv_entry,
            target=csv_path,
            max_bytes=MAX_POSE_MEMBER_BYTES,
        )
        _copy_zip_member(
            archive,
            member=raw_entry,
            target=raw_path,
            max_bytes=MAX_RAW_MEMBER_BYTES,
        )
        pose = parse_vidimu_pose_csv(csv_path, source_recording_id=recording_id)
        raw = parse_vidimu_raw(raw_path, source_recording_id=recording_id)
    if pose.source_rows.num_rows != pose.row_count:
        raise AuditError(f"pose source-row count mismatch for {recording_id}")
    if raw.source_rows.num_rows != raw.source_row_count:
        raise AuditError(f"RAW source-row count mismatch for {recording_id}")
    if (
        len(raw.calibration) != 5
        or len(raw.stream_statistics) != 5
        or raw.source_row_count
        != len(raw.calibration)
        + sum(item.observation_count for item in raw.stream_statistics)
    ):
        raise AuditError(f"RAW five-stream accounting mismatch for {recording_id}")
    zero_masks = pose.source_rows["zero_triplet_mask"].to_pylist()
    fully_zero_rows = sum(bool(mask) and all(mask) for mask in zero_masks)
    calibration_tokens = {
        calibration.source_timestamp_token for calibration in raw.calibration
    }
    if len(calibration_tokens) != 1:
        raise AuditError(f"N-pose timestamp is not unique for {recording_id}")
    calibration_token = next(iter(calibration_tokens))
    whole, separator, fraction = calibration_token.partition(".")
    if not whole.isdigit() or (separator and not fraction.isdigit()):
        raise AuditError(f"invalid N-pose decimal token for {recording_id}")
    calibration_is_zero = int(whole) == 0 and int(fraction or "0") == 0
    streams = []
    for item in raw.stream_statistics:
        streams.append(
            {
                "body_location": item.body_location,
                "consecutive_distinct_payload_count": (
                    item.consecutive_distinct_payload_count
                ),
                "first_source_ordinal": item.first_source_ordinal,
                "first_source_timestamp_token": item.first_source_timestamp_token,
                "held_observation_fraction": _ratio_dict(
                    item.held_payload_observation_count,
                    item.observation_count,
                ),
                "held_payload_observation_count": (item.held_payload_observation_count),
                "last_source_ordinal": item.last_source_ordinal,
                "last_source_timestamp_token": item.last_source_timestamp_token,
                "observation_count": item.observation_count,
                "sensor_label": item.sensor_label,
                "stream_id": item.stream_id,
            }
        )
    invalid = [
        {
            "sensor_label": item.sensor_label,
            "source_line_number": item.source_line_number,
            "source_ordinal": item.source_ordinal,
            "source_timestamp_token": item.source_timestamp_token,
            "source_values": list(item.source_values),
            "stream_id": item.stream_id,
        }
        for item in raw.invalid_quaternions
    ]
    return {
        "activity_id": recording_id.split("_")[1],
        "recording_id": recording_id,
        "subject_id": subject_id,
        "trial_id": recording_id.split("_")[2],
        "source_files": {
            "bodytrack_pose_csv": {
                "central_directory_crc32_hex": csv_entry.crc32_hex,
                "fully_zero_row_count": fully_zero_rows,
                "path": csv_entry.path,
                "row_count": pose.row_count,
                "rows_with_any_zero_triplet_count": pose.rows_with_zero_triplets,
                "sha256": pose.source_sha256,
                "size_bytes": csv_entry.uncompressed_size,
            },
            "quaternion_raw": {
                "central_directory_crc32_hex": raw_entry.crc32_hex,
                "invalid_quaternion_observation_count": len(invalid),
                "invalid_quaternion_observations": invalid,
                "npose_row_count": len(raw.calibration),
                "npose_timestamp_class": (
                    "EXACT_NUMERIC_ZERO"
                    if calibration_is_zero
                    else "NONZERO_DECIMAL_TOKEN_WITH_UNRESOLVED_CLOCK"
                ),
                "npose_timestamp_token": calibration_token,
                "path": raw_entry.path,
                "row_count_including_npose": raw.source_row_count,
                "sha256": raw.source_sha256,
                "size_bytes": raw_entry.uncompressed_size,
                "stream_statistics": streams,
            },
        },
    }


def audit_dataset_archive(
    dataset_zip: Path,
    *,
    published: Mapping[str, object],
    expected_identity: tuple[int, ...] | None = None,
) -> DatasetAudit:
    """Fully verify and parse the local source archive."""

    path = Path(os.path.abspath(os.fspath(dataset_zip)))
    handle, opened = _open_pinned_dataset_archive(path)
    with handle:
        if (
            expected_identity is not None
            and _file_identity(opened) != expected_identity
        ):
            raise AuditError("local dataset ZIP identity differs from caller pin")
        observed = _hash_stream(handle)
        expected_checksum = published.get("published_checksum")
        if not isinstance(expected_checksum, Mapping):
            raise AuditError("dataset published checksum metadata is absent")
        if observed["size_bytes"] != published.get("size_bytes") or observed[
            "md5"
        ] != expected_checksum.get("value"):
            raise AuditError("local dataset ZIP does not match Zenodo size/MD5")
        central = _read_local_central(handle, int(observed["size_bytes"]))
        csv_by_id, raw_by_id = _dataset_inventory(central.entries)
        recording_ids = tuple(sorted(set(csv_by_id) & set(raw_by_id)))
        records: list[dict[str, object]] = []
        handle.seek(0)
        with zipfile.ZipFile(handle, "r") as archive:
            if len(archive.infolist()) != len(central.entries):
                raise AuditError("ZIP reader and central-directory entry counts differ")
            for recording_id in recording_ids:
                records.append(
                    _parse_source_pair(
                        archive,
                        recording_id=recording_id,
                        csv_entry=csv_by_id[recording_id],
                        raw_entry=raw_by_id[recording_id],
                    )
                )
        _revalidate_pinned_dataset_archive(
            path,
            handle,
            opened=opened,
            observed_size=int(observed["size_bytes"]),
        )
    archive_result = {
        "byte_verification": {
            "observed_md5": observed["md5"],
            "observed_sha256": observed["sha256"],
            "observed_size_bytes": observed["size_bytes"],
            "published_md5_recomputed": True,
            "status": "FULL_LOCAL_ARCHIVE_BYTES_VERIFIED",
        },
        "canonical_source_inventory": {
            "csv_only_recording_ids": sorted(set(csv_by_id) - set(raw_by_id)),
            "csv_recording_count": len(csv_by_id),
            "paired_recording_count": len(recording_ids),
            "raw_only_recording_ids": sorted(set(raw_by_id) - set(csv_by_id)),
            "raw_recording_count": len(raw_by_id),
        },
        "central_directory": _central_summary(central),
        "content_url": published["content_url"],
        "published_checksum": published["published_checksum"],
        "published_size_bytes": published["size_bytes"],
    }
    return DatasetAudit(
        archive=archive_result,
        records=tuple(records),
        recording_ids=recording_ids,
    )


def _video_patterns(wrapper: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    escaped = re.escape(wrapper)
    base = (
        rf"{escaped}/__SUBTREE__/(?P<subject>S[0-9]{{2}})/"
        rf"(?P<recording>(?P=subject)_A(?:0[1-9]|1[0-3])_T[0-9]{{2}})"
    )
    return (
        re.compile(rf"^{base.replace('__SUBTREE__', 'videosoriginal')}\.mp4$"),
        re.compile(rf"^{base.replace('__SUBTREE__', 'videosbodytrack')}_pose\.mp4$"),
    )


def audit_remote_video_archive(
    key: str,
    *,
    published: Mapping[str, object],
    paired_recording_ids: Sequence[str],
    opener: Callable[..., object] = urllib.request.urlopen,
    timeout_seconds: int = HTTP_TIMEOUT_SECONDS,
) -> VideoAudit:
    """Audit one official video central directory without member downloads."""

    if key not in {"videosmallsize.zip", "videosfullsize.zip"}:
        raise AuditError(f"unsupported video archive key: {key}")
    archive_size = published.get("size_bytes")
    content_url = published.get("content_url")
    if not isinstance(archive_size, int) or not isinstance(content_url, str):
        raise AuditError(f"published metadata for {key} is incomplete")
    tail, transport = _fetch_tail_range(
        content_url,
        archive_size=archive_size,
        opener=opener,
        timeout_seconds=timeout_seconds,
    )
    central = parse_central_directory_tail(
        tail,
        archive_size=archive_size,
        tail_start=int(transport["range_start"]),
    )
    wrapper = key.removesuffix(".zip")
    original_pattern, qa_pattern = _video_patterns(wrapper)
    original_by_id: dict[str, CentralEntry] = {}
    qa_by_id: dict[str, CentralEntry] = {}
    original_subtree_entries: list[CentralEntry] = []
    qa_pose_entries: list[CentralEntry] = []
    for entry in central.entries:
        if entry.is_directory:
            continue
        if entry.path.startswith(f"{wrapper}/videosoriginal/"):
            original_subtree_entries.append(entry)
        if entry.path.startswith(f"{wrapper}/videosbodytrack/") and entry.path.endswith(
            "_pose.mp4"
        ):
            qa_pose_entries.append(entry)
        original_match = original_pattern.fullmatch(entry.path)
        if original_match is not None:
            mode = (entry.external_attributes >> 16) & 0xFFFF
            if entry.general_purpose_bit_flags & 1 or stat.S_ISLNK(mode):
                raise AuditError(
                    f"unsafe canonical original-subtree candidate: {entry.path}"
                )
            original_by_id[original_match.group("recording")] = entry
        qa_match = qa_pattern.fullmatch(entry.path)
        if qa_match is not None:
            mode = (entry.external_attributes >> 16) & 0xFFFF
            if entry.general_purpose_bit_flags & 1 or stat.S_ISLNK(mode):
                raise AuditError(f"unsafe canonical QA-video member: {entry.path}")
            qa_by_id[qa_match.group("recording")] = entry
    paired = tuple(sorted(paired_recording_ids))
    selected: dict[str, dict[str, object]] = {}
    for recording_id in paired:
        original_candidate = original_by_id.get(recording_id)
        selected[recording_id] = {
            "bodytrack_qa": (
                None
                if recording_id not in qa_by_id
                else qa_by_id[recording_id].public_dict()
            ),
            "original_subtree_candidate": (
                None if original_candidate is None else original_candidate.public_dict()
            ),
        }
    missing_original_candidate = [item for item in paired if item not in original_by_id]
    missing_qa = [item for item in paired if item not in qa_by_id]
    archive_result = {
        "byte_verification": {
            "full_archive_bytes_downloaded": False,
            "published_md5_recomputed": False,
            "status": "REMOTE_CENTRAL_DIRECTORY_ONLY_PUBLISHED_MD5_NOT_RECOMPUTED",
        },
        "canonical_video_inventory": {
            "canonical_original_candidate_recording_count": len(original_by_id),
            "canonical_qa_recording_count": len(qa_by_id),
            "original_subtree_candidate_file_count": len(original_subtree_entries),
            "original_subtree_candidate_inventory_sha256": _inventory_sha256(
                original_subtree_entries
            ),
            "qa_pose_file_count": len(qa_pose_entries),
            "qa_pose_inventory_sha256": _inventory_sha256(qa_pose_entries),
        },
        "central_directory": _central_summary(central),
        "content_url": content_url,
        "paired_source_members": {
            "original_candidate_found_count": (
                len(paired) - len(missing_original_candidate)
            ),
            "original_candidate_missing_recording_ids": (missing_original_candidate),
            "qa_found_count": len(paired) - len(missing_qa),
            "qa_missing_recording_ids": missing_qa,
            "source_pair_count": len(paired),
        },
        "published_checksum": published["published_checksum"],
        "published_size_bytes": archive_size,
        "range_evidence": transport,
    }
    return VideoAudit(archive=archive_result, selected_members=selected)


def _ratio_from_dict(value: Mapping[str, object]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _held_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    streams: list[tuple[Fraction, str, str, int, int]] = []
    for record in records:
        sources = record["source_files"]
        assert isinstance(sources, Mapping)
        raw = sources["quaternion_raw"]
        assert isinstance(raw, Mapping)
        for stream in raw["stream_statistics"]:  # type: ignore[index]
            assert isinstance(stream, Mapping)
            ratio = stream["held_observation_fraction"]
            assert isinstance(ratio, Mapping)
            streams.append(
                (
                    _ratio_from_dict(ratio),
                    str(record["recording_id"]),
                    str(stream["sensor_label"]),
                    int(stream["held_payload_observation_count"]),
                    int(stream["observation_count"]),
                )
            )
    streams.sort(key=lambda item: (item[0], item[1], item[2]))
    if not streams:
        raise AuditError("no RAW stream statistics were emitted")

    def endpoint(item: tuple[Fraction, str, str, int, int]) -> dict[str, object]:
        value, recording_id, sensor_label, numerator, denominator = item
        return {
            **_ratio_dict(numerator, denominator),
            "recording_id": recording_id,
            "sensor_label": sensor_label,
            "unreduced_denominator": denominator,
            "unreduced_numerator": numerator,
            "decimal_6": _fraction_decimal_6(value),
        }

    middle = len(streams) // 2
    if len(streams) % 2:
        median_value = streams[middle][0]
        middle_streams = [endpoint(streams[middle])]
    else:
        median_value = (streams[middle - 1][0] + streams[middle][0]) / 2
        middle_streams = [endpoint(streams[middle - 1]), endpoint(streams[middle])]
    return {
        "definition": (
            "PER_DYNAMIC_SENSOR_STREAM;(OBSERVATION_COUNT_MINUS_"
            "CONSECUTIVE_DISTINCT_EXACT_SOURCE_PAYLOAD_COUNT)/OBSERVATION_COUNT"
        ),
        "maximum": endpoint(streams[-1]),
        "median": {
            **_ratio_dict(median_value.numerator, median_value.denominator),
            "middle_streams": middle_streams,
        },
        "minimum": endpoint(streams[0]),
        "stream_count": len(streams),
    }


def _aggregates(
    records: Sequence[Mapping[str, object]],
    *,
    video_archives: Mapping[str, VideoAudit],
) -> dict[str, object]:
    raw_rows = 0
    invalid_count = 0
    invalid_recordings: list[str] = []
    zero_npose: list[str] = []
    nonzero_npose: list[str] = []
    fully_zero_rows = 0
    fully_zero_recordings: list[str] = []
    any_zero_rows = 0
    pose_rows = 0
    for record in records:
        recording_id = str(record["recording_id"])
        sources = record["source_files"]
        assert isinstance(sources, Mapping)
        pose = sources["bodytrack_pose_csv"]
        raw = sources["quaternion_raw"]
        assert isinstance(pose, Mapping) and isinstance(raw, Mapping)
        raw_rows += int(raw["row_count_including_npose"])
        current_invalid = int(raw["invalid_quaternion_observation_count"])
        invalid_count += current_invalid
        if current_invalid:
            invalid_recordings.append(recording_id)
        if raw["npose_timestamp_class"] == "EXACT_NUMERIC_ZERO":
            zero_npose.append(recording_id)
        else:
            nonzero_npose.append(recording_id)
        current_fully_zero = int(pose["fully_zero_row_count"])
        pose_rows += int(pose["row_count"])
        fully_zero_rows += current_fully_zero
        any_zero_rows += int(pose["rows_with_any_zero_triplet_count"])
        if current_fully_zero:
            fully_zero_recordings.append(recording_id)
    return {
        "canonical_csv_raw_pair_count": len(records),
        "held_observation_fraction": _held_summary(records),
        "invalid_quaternion_observation_count": invalid_count,
        "invalid_quaternion_recording_ids": invalid_recordings,
        "npose_timestamp": {
            "nonzero_record_count": len(nonzero_npose),
            "nonzero_recording_ids": nonzero_npose,
            "zero_record_count": len(zero_npose),
        },
        "pose_zero": {
            "affected_recording_count": len(fully_zero_recordings),
            "affected_recording_ids": fully_zero_recordings,
            "fully_zero_row_count": fully_zero_rows,
            "fully_zero_row_definition": "ALL_102_COORDINATES_EXACT_NUMERIC_ZERO",
            "rows_with_any_zero_triplet_count": any_zero_rows,
        },
        "pose_source_row_count": pose_rows,
        "raw_dynamic_observation_row_count": raw_rows - len(records) * 5,
        "raw_source_row_count_including_npose": raw_rows,
        "video_pairing": {
            key: video_archives[key].archive["paired_source_members"]
            for key in sorted(video_archives)
        },
    }


def _assert_pinned_release(audit: Mapping[str, object]) -> None:
    archives = audit["archives"]
    aggregates = audit["aggregates"]
    implementation = audit["implementation"]
    assert isinstance(archives, Mapping)
    assert isinstance(aggregates, Mapping)
    assert isinstance(implementation, Mapping)
    dataset = archives["dataset.zip"]
    assert isinstance(dataset, Mapping)
    inventory = dataset["canonical_source_inventory"]
    assert isinstance(inventory, Mapping)
    held = aggregates["held_observation_fraction"]
    npose = aggregates["npose_timestamp"]
    pose_zero = aggregates["pose_zero"]
    video_pairing = aggregates["video_pairing"]
    assert all(
        isinstance(item, Mapping) for item in (held, npose, pose_zero, video_pairing)
    )
    expected_equalities = {
        "parser_version": (
            implementation["source_parser_version"],
            "vidimu-native-source-v0.1.0",
        ),
        "csv_count": (inventory["csv_recording_count"], 215),
        "raw_count": (inventory["raw_recording_count"], 208),
        "pair_count": (aggregates["canonical_csv_raw_pair_count"], 208),
        "csv_only": (tuple(inventory["csv_only_recording_ids"]), _EXPECTED_CSV_ONLY),
        "raw_only": (inventory["raw_only_recording_ids"], []),
        "raw_rows": (aggregates["raw_source_row_count_including_npose"], 10_184_045),
        "zero_npose": (npose["zero_record_count"], 197),
        "nonzero_npose": (npose["nonzero_record_count"], 11),
        "nonzero_npose_ids": (
            tuple(npose["nonzero_recording_ids"]),
            _EXPECTED_NONZERO_NPOSE,
        ),
        "invalid_quaternions": (
            aggregates["invalid_quaternion_observation_count"],
            12,
        ),
        "invalid_recording": (
            aggregates["invalid_quaternion_recording_ids"],
            ["S54_A08_T02"],
        ),
        "fully_zero_pose_rows": (pose_zero["fully_zero_row_count"], 267),
        "fully_zero_pose_files": (pose_zero["affected_recording_count"], 13),
        "any_zero_pose_rows": (
            pose_zero["rows_with_any_zero_triplet_count"],
            267,
        ),
        "pose_rows": (aggregates["pose_source_row_count"], 179_076),
        "dynamic_rows": (
            aggregates["raw_dynamic_observation_row_count"],
            10_183_005,
        ),
        "held_stream_count": (held["stream_count"], 1_040),
        "held_minimum": (held["minimum"]["decimal_6"], "0.797315"),  # type: ignore[index]
        "held_median": (held["median"]["decimal_6"], "0.855143"),  # type: ignore[index]
        "held_maximum": (held["maximum"]["decimal_6"], "0.920376"),  # type: ignore[index]
    }
    for key in ("videosmallsize.zip", "videosfullsize.zip"):
        pairing = video_pairing[key]  # type: ignore[index]
        assert isinstance(pairing, Mapping)
        expected_equalities[f"{key}_original"] = (
            pairing["original_candidate_found_count"],
            208,
        )
        expected_equalities[f"{key}_original_missing"] = (
            pairing["original_candidate_missing_recording_ids"],
            [],
        )
        expected_equalities[f"{key}_qa"] = (pairing["qa_found_count"], 206)
        expected_equalities[f"{key}_qa_missing"] = (
            tuple(pairing["qa_missing_recording_ids"]),
            _EXPECTED_QA_MISSING,
        )
    failures = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in expected_equalities.items()
        if observed != expected
    }
    if failures:
        raise AuditError(f"pinned VIDIMU release facts failed: {failures!r}")


def build_release_audit(
    dataset_zip: Path,
    *,
    record_metadata: Mapping[str, object],
    opener: Callable[..., object] = urllib.request.urlopen,
    timeout_seconds: int = HTTP_TIMEOUT_SECONDS,
    dataset_expected_identity: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Build the complete deterministic audit object and enforce release gates."""

    audit_script = Path(__file__).resolve(strict=True)
    parser_source = Path(vidimu_source.__file__).resolve(strict=True)
    initial_implementation_hashes = {
        "audit_script_sha256": _normalized_source_sha256(audit_script),
        "source_parser_sha256": _normalized_source_sha256(parser_source),
    }
    published, release = _validate_record_metadata(record_metadata)
    dataset = audit_dataset_archive(
        dataset_zip,
        published=published["dataset.zip"],
        expected_identity=dataset_expected_identity,
    )
    video_archives = {
        key: audit_remote_video_archive(
            key,
            published=published[key],
            paired_recording_ids=dataset.recording_ids,
            opener=opener,
            timeout_seconds=timeout_seconds,
        )
        for key in ("videosmallsize.zip", "videosfullsize.zip")
    }
    records: list[dict[str, object]] = []
    for source_record in dataset.records:
        recording_id = str(source_record["recording_id"])
        records.append(
            {
                **source_record,
                "video_members": {
                    key: video_archives[key].selected_members[recording_id]
                    for key in sorted(video_archives)
                },
            }
        )
    result: dict[str, object] = {
        "aggregates": _aggregates(records, video_archives=video_archives),
        "archives": {
            "dataset.zip": dataset.archive,
            **{key: video_archives[key].archive for key in sorted(video_archives)},
        },
        "artifact_kind": AUDIT_ARTIFACT_KIND,
        "audit_status": "PASS",
        "claim_boundary": {
            "filename_pairing_establishes_synchronization": False,
            "held_fraction_interpretation": (
                "EXACT_CONSECUTIVE_RAW_SOURCE_PAYLOAD_REPETITION_NOT_SENSOR_SAMPLE_RATE"
            ),
            "pose_zero_interpretation": (
                "OBSERVED_EXACT_ZERO_SENTINEL_NOT_TRACKING_FAILURE"
            ),
            "video_archive_checksum_scope": (
                "PUBLISHED_ZENODO_MD5_METADATA_ONLY_NOT_RECOMPUTED"
            ),
            "video_central_directory_scope": (
                "MEMBER_NAMES_AND_ZIP_METADATA_ONLY_NO_MEMBER_BYTES_READ"
            ),
            "video_decode_or_pts_verified": False,
            "video_member_sha256_available": False,
        },
        "implementation": {
            "audit_implementation_version": AUDIT_IMPLEMENTATION_VERSION,
            "audit_script_sha256": initial_implementation_hashes["audit_script_sha256"],
            "quaternion_norm_absolute_tolerance": format(
                VIDIMU_QUATERNION_NORM_ABS_TOL, ".3f"
            ),
            "source_hash_canonicalization": (
                "STRICT_UTF8_WITH_CRLF_AND_CR_NORMALIZED_TO_LF"
            ),
            "source_parser_module": (
                "motionbloom.tremora_store.adapters.vidimu_source"
            ),
            "source_parser_sha256": initial_implementation_hashes[
                "source_parser_sha256"
            ],
            "source_parser_version": VIDIMU_SOURCE_PARSER_VERSION,
            "tremora_store_schema_version": SCHEMA_VERSION,
        },
        "records": records,
        "release": release,
        "schema_version": AUDIT_SCHEMA_VERSION,
    }
    _assert_pinned_release(result)
    final_implementation_hashes = {
        "audit_script_sha256": _normalized_source_sha256(audit_script),
        "source_parser_sha256": _normalized_source_sha256(parser_source),
    }
    if final_implementation_hashes != initial_implementation_hashes:
        raise AuditError("audit or source-parser implementation changed during run")
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=HTTP_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def _reject_output_alias(dataset_zip: Path, output: Path) -> None:
    """Prevent the audit artifact from replacing its input archive."""

    dataset_normalized = os.path.normcase(os.path.abspath(os.fspath(dataset_zip)))
    output_normalized = os.path.normcase(os.path.abspath(os.fspath(output)))
    if dataset_normalized == output_normalized:
        raise AuditError("--output must not alias --dataset-zip")
    try:
        aliases = os.path.samefile(dataset_zip, output)
    except (FileNotFoundError, NotADirectoryError):
        aliases = False
    except OSError as exc:
        raise AuditError("cannot verify --output identity") from exc
    if aliases:
        raise AuditError("--output must not alias --dataset-zip")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_path = Path(os.path.abspath(os.fspath(args.dataset_zip)))
    dataset_guard, dataset_opened = _open_pinned_dataset_archive(dataset_path)
    with dataset_guard:
        _reject_output_alias(dataset_path, args.output)
        metadata = fetch_record_metadata(timeout_seconds=args.timeout_seconds)
        result = build_release_audit(
            dataset_path,
            record_metadata=metadata,
            timeout_seconds=args.timeout_seconds,
            dataset_expected_identity=_file_identity(dataset_opened),
        )
        payload = canonical_json_bytes(result)
        _revalidate_pinned_dataset_archive(
            dataset_path,
            dataset_guard,
            opened=dataset_opened,
            observed_size=dataset_opened.st_size,
        )
        _reject_output_alias(dataset_path, args.output)
        _atomic_write(
            args.output,
            payload,
            forbidden_identity=(dataset_opened.st_dev, dataset_opened.st_ino),
        )
        _revalidate_pinned_dataset_archive(
            dataset_path,
            dataset_guard,
            opened=dataset_opened,
            observed_size=dataset_opened.st_size,
        )
    print(
        json.dumps(
            {
                "audit_status": result["audit_status"],
                "output": str(args.output),
                "output_sha256": _sha256_bytes(payload),
                "record_count": len(result["records"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
