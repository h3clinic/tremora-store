"""Atomic Parquet snapshot writer and semantic integrity verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .integrity import (
    RESERVED_PROVENANCE_KEYS,
    StoreInvariantError,
    canonical_schema,
    schemas_equal,
    validate_provenance,
    validate_snapshot_tables,
)
from .schema import SCHEMA_VERSION, logical_schema_contract, schema_fingerprint

SNAPSHOT_MANIFEST_VERSION = 1
REQUIRED_TABLES = frozenset({
    "frame_index", "cv_estimates", "imu_samples", "clock_map",
    "frame_imu_index", "window_index", "window_rejections",
})
CANONICAL_SORT_KEYS = {
    "frame_index": ("recording_id", "video_stream_id", "canonical_ordinal"),
    "cv_estimates": ("recording_id", "video_stream_id", "canonical_ordinal"),
    "imu_samples": ("recording_id", "stream_id", "canonical_ordinal"),
    "clock_map": ("recording_id", "stream_id", "acquisition_ordinal"),
    "frame_imu_index": (
        "recording_id", "video_stream_id", "imu_stream_id",
        "frame_canonical_ordinal",
    ),
    "window_index": ("window_id",),
    "window_rejections": ("candidate_window_id",),
}
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._=-]{0,127}\Z")
_WINDOWS_RESERVED_COMPONENTS = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
})
_UTC_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")


class SnapshotError(RuntimeError):
    """Raised when a snapshot is incomplete, mixed, or has failed integrity."""


@dataclass(frozen=True, slots=True)
class _VerifiedSnapshot:
    """Manifest payload that was used for a complete snapshot verification."""

    manifest: dict[str, object]
    manifest_bytes: int
    manifest_sha256: str


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) \
            or chunk_size <= 0:
        raise SnapshotError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_descriptor_at(descriptor: int, size: int, offset: int) -> bytes:
    """Portable positional read without changing the descriptor's offset."""

    pread = getattr(os, "pread", None)
    if pread is not None:
        return pread(descriptor, size, offset)
    prior_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, offset, os.SEEK_SET)
        return os.read(descriptor, size)
    finally:
        os.lseek(descriptor, prior_offset, os.SEEK_SET)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _sha256_descriptor(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = _read_descriptor_at(
            descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise SnapshotError("descriptor ended before its declared size")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _read_only_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _verified_file_bytes(
    path: Path, *, field: str,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    """Read and bind validation to one regular-file descriptor payload."""

    flags = _read_only_flags()
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SnapshotError(f"cannot open {field}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SnapshotError(f"{field} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise SnapshotError(f"{field} changed during pinned read")
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise SnapshotError(f"missing or wrong-sized {field}")
    if expected_sha256 is not None \
            and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise SnapshotError(f"{field} hash mismatch")
    return payload


@contextmanager
def _verified_file_source(
    path: Path, *, field: str, expected_bytes: int, expected_sha256: str,
):
    """Yield a seekable pinned descriptor after a bounded-memory hash pass."""

    flags = _read_only_flags()
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SnapshotError(f"cannot open {field}") from exc
    handle = None
    source = None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SnapshotError(f"{field} is not a regular file")
        if before.st_size != expected_bytes:
            raise SnapshotError(f"missing or wrong-sized {field}")
        digest = hashlib.sha256()
        offset = 0
        while offset < expected_bytes:
            chunk = _read_descriptor_at(
                descriptor, min(1024 * 1024, expected_bytes - offset), offset)
            if not chunk:
                raise SnapshotError(f"missing or wrong-sized {field}")
            digest.update(chunk)
            offset += len(chunk)
        if digest.hexdigest() != expected_sha256:
            raise SnapshotError(f"{field} hash mismatch")
        handle = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = -1
        source = pa.PythonFile(handle, mode="r")
        try:
            yield source
        finally:
            after = os.fstat(handle.fileno())
            if _stat_identity(before) != _stat_identity(after):
                raise SnapshotError(f"{field} changed during pinned read")
    finally:
        if source is not None:
            source.close()
        elif handle is not None:
            handle.close()
        elif descriptor >= 0:
            os.close(descriptor)


def _update_length_prefixed(digest, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big", signed=False))
    digest.update(payload)


def _update_scalar(digest, scalar: pa.Scalar, data_type: pa.DataType) -> None:
    if not scalar.is_valid:
        digest.update(b"N")
        return
    value = scalar.as_py()
    if pa.types.is_boolean(data_type):
        digest.update(b"B1" if value else b"B0")
    elif pa.types.is_integer(data_type):
        digest.update(b"I")
        _update_length_prefixed(digest, str(value).encode())
    elif pa.types.is_float32(data_type):
        digest.update(b"F4")
        digest.update(struct.pack(">f", value))
    elif pa.types.is_float64(data_type):
        digest.update(b"F8")
        digest.update(struct.pack(">d", value))
    elif pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        digest.update(b"S")
        _update_length_prefixed(digest, value.encode("utf-8"))
    elif pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
        digest.update(b"Y")
        _update_length_prefixed(digest, value)
    elif pa.types.is_list(data_type) or pa.types.is_large_list(data_type) \
            or pa.types.is_fixed_size_list(data_type):
        digest.update(b"L")
        digest.update(len(value).to_bytes(8, "big", signed=False))
        for item in scalar.values:
            _update_scalar(digest, item, data_type.value_type)
    else:
        digest.update(b"J")
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False, default=str).encode()
        _update_length_prefixed(digest, payload)


def semantic_table_hash(table: pa.Table,
                        *, sort_keys: Iterable[str] = ()) -> str:
    """Hash logical Arrow values independently of Parquet byte layout.

    Finite floating values are hashed by their IEEE payload, preserving signed
    zero. Valid snapshots reject nonfinite payloads. Column order, types,
    nullability and schema metadata are included. Dictionary encoding and
    Parquet row-group layout are not.
    """

    keys = tuple(sort_keys)
    if keys:
        missing = set(keys).difference(table.column_names)
        if missing:
            raise SnapshotError(f"semantic sort keys missing: {sorted(missing)!r}")
        table = table.sort_by([(key, "ascending") for key in keys])
    table = table.combine_chunks()
    digest = hashlib.sha256()
    digest.update(b"TREMORA_LOGICAL_TABLE_V1\0")
    header = {**logical_schema_contract(table.schema), "rows": table.num_rows}
    _update_length_prefixed(
        digest, json.dumps(header, sort_keys=True,
                           separators=(",", ":")).encode())
    for row_index in range(table.num_rows):
        digest.update(b"R")
        digest.update(row_index.to_bytes(8, "big", signed=False))
        for field, column in zip(table.schema, table.columns, strict=True):
            _update_scalar(digest, column[row_index], field.type)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(
        value,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
        allow_nan=False,
    ) + "\n").encode()


def _write_json(path: Path, value: object) -> None:
    encoded = _json_bytes(value)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Windows does not expose POSIX directory descriptors. File flushes and
        # atomic os.replace still apply; directory-entry fsync is POSIX-only.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_component(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value in {".", ".."} \
            or not _SAFE_COMPONENT.fullmatch(value) or value.endswith(".") \
            or value.split(".", maxsplit=1)[0].upper() \
            in _WINDOWS_RESERVED_COMPONENTS:
        raise SnapshotError(f"{field} must be a safe single path component")
    return value


def _require_utc_timestamp(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _UTC_TIMESTAMP.fullmatch(value):
        raise SnapshotError(f"{field} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SnapshotError(f"{field} is not a valid UTC timestamp") from exc
    return value


def _contained_existing(path: Path, parent: Path, *, field: str) -> Path:
    if path.is_symlink():
        raise SnapshotError(f"{field} may not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(parent.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"{field} escapes its snapshot directory") from exc
    return resolved


def _resolved_store_root(root: str | Path, *, create: bool = False) -> Path:
    """Return a strict, canonical store root.

    Resolving the root once prevents later containment checks from being made
    against a user-controlled symlink spelling of the root path.
    """

    candidate = Path(root)
    if create:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SnapshotError("store root could not be created") from exc
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SnapshotError("store root is missing") from exc
    if not resolved.is_dir():
        raise SnapshotError("store root is not a directory")
    return resolved


def _resolved_snapshots_dir(root: Path, *, create: bool = False) -> Path:
    """Resolve the snapshots parent without ever following it as a symlink."""

    snapshots = root / "snapshots"
    if snapshots.is_symlink():
        raise SnapshotError("snapshots parent may not be a symlink")
    if create:
        try:
            snapshots.mkdir(exist_ok=True)
        except OSError as exc:
            raise SnapshotError("snapshots parent could not be created") from exc
    if snapshots.is_symlink():  # Recheck after creation for a concurrent swap.
        raise SnapshotError("snapshots parent may not be a symlink")
    if not snapshots.is_dir():
        raise SnapshotError("snapshots parent is missing or not a directory")
    resolved = _contained_existing(snapshots, root, field="snapshots parent")
    if resolved.parent != root:
        raise SnapshotError("snapshots parent escapes the store root")
    return resolved


def _row_group_sizes(parquet_file: pq.ParquetFile) -> list[int]:
    metadata = parquet_file.metadata
    return [
        metadata.row_group(index).num_rows
        for index in range(metadata.num_row_groups)
    ]


def _validate_row_group_policy(
    name: str,
    *,
    parquet_file: pq.ParquetFile,
    declared_rows: int,
    declared_row_group_size: object,
) -> int:
    if isinstance(declared_row_group_size, bool) \
            or not isinstance(declared_row_group_size, int) \
            or declared_row_group_size <= 0:
        raise SnapshotError(f"invalid row-group size for {name}")
    sizes = _row_group_sizes(parquet_file)
    if sum(sizes) != declared_rows:
        raise SnapshotError(f"row-group row count mismatch: {name}")
    if declared_rows == 0:
        if any(size != 0 for size in sizes):
            raise SnapshotError(f"invalid empty-table row groups: {name}")
        return declared_row_group_size
    if not sizes or sizes[-1] <= 0:
        raise SnapshotError(f"missing final row group: {name}")
    if any(size <= 0 or size > declared_row_group_size for size in sizes):
        raise SnapshotError(f"row group exceeds declared policy: {name}")
    if any(size != declared_row_group_size for size in sizes[:-1]):
        raise SnapshotError(f"nonfinal row group violates policy: {name}")
    return declared_row_group_size


class RecordingStoreWriter:
    """Stage and atomically publish one immutable recording snapshot."""

    def __init__(
        self, root: str | Path, *, snapshot_id: str, recording_id: str,
        created_at_utc: str, clock_map_id: str, window_policy_id: str,
        provenance: Mapping[str, object], row_group_size: int = 65_536,
    ):
        identity_fields = {
            "snapshot_id": snapshot_id,
            "recording_id": recording_id,
            "clock_map_id": clock_map_id,
            "window_policy_id": window_policy_id,
        }
        if any(not isinstance(value, str) or not value
               for value in identity_fields.values()):
            raise SnapshotError(
                "snapshot identity/version fields must be non-empty strings")
        if isinstance(row_group_size, bool) \
                or not isinstance(row_group_size, int) \
                or row_group_size <= 0:
            raise SnapshotError("row_group_size must be positive")
        _safe_component(snapshot_id, field="snapshot_id")
        _require_utc_timestamp(created_at_utc, field="created_at_utc")
        if not isinstance(provenance, Mapping):
            raise SnapshotError("provenance must be a mapping object")
        supplied_reserved = RESERVED_PROVENANCE_KEYS.intersection(provenance)
        if supplied_reserved:
            raise SnapshotError(
                "caller provenance may not supply writer-reserved fields: "
                f"{sorted(supplied_reserved)!r}")
        try:
            validate_provenance(provenance)
        except StoreInvariantError as exc:
            raise SnapshotError(str(exc)) from exc
        self.root = _resolved_store_root(root, create=True)
        self.snapshot_id = snapshot_id
        self.recording_id = recording_id
        self.created_at_utc = created_at_utc
        self.clock_map_id = clock_map_id
        self.window_policy_id = window_policy_id
        self._provenance = deepcopy(dict(provenance))
        self.row_group_size = row_group_size
        self.snapshots_dir = _resolved_snapshots_dir(self.root, create=True)
        self.staging_dir = self.root / f".staging-{snapshot_id}"
        self.final_dir = self.snapshots_dir / snapshot_id
        self._artifacts: dict[str, dict[str, object]] = {}
        if self.staging_dir.exists() or self.staging_dir.is_symlink() \
                or self.final_dir.exists() or self.final_dir.is_symlink():
            raise SnapshotError("snapshot or staging directory already exists")
        self.staging_dir.mkdir()
        self.staging_dir = _contained_existing(
            self.staging_dir, self.root, field="staging directory")
        staging_stat = self.staging_dir.stat()
        self._staging_directory_identity = (
            staging_stat.st_dev, staging_stat.st_ino)

    @property
    def provenance(self) -> dict[str, object]:
        """Return a defensive copy of the writer-owned provenance payload."""

        return deepcopy(self._provenance)

    def _assert_staging_directory(self) -> None:
        try:
            current = self.staging_dir.lstat()
        except OSError as exc:
            raise SnapshotError("staging directory changed during write") from exc
        if not stat.S_ISDIR(current.st_mode) or (
            current.st_dev, current.st_ino,
        ) != self._staging_directory_identity:
            raise SnapshotError("staging directory changed during write")
        resolved = _contained_existing(
            self.staging_dir, self.root, field="staging directory")
        if resolved != self.staging_dir:
            raise SnapshotError("staging directory changed during write")

    def _open_staged_artifact(self, filename: str) -> int:
        """Open one new artifact without following its final path component."""

        self._assert_staging_directory()
        file_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_BINARY"):
            file_flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        descriptor = -1
        directory_descriptor = -1
        success = False
        try:
            supports_dir_fd = os.open in getattr(os, "supports_dir_fd", ())
            if supports_dir_fd:
                directory_flags = os.O_RDONLY
                if hasattr(os, "O_DIRECTORY"):
                    directory_flags |= os.O_DIRECTORY
                if hasattr(os, "O_NOFOLLOW"):
                    directory_flags |= os.O_NOFOLLOW
                directory_descriptor = os.open(
                    self.staging_dir, directory_flags)
                directory_stat = os.fstat(directory_descriptor)
                if (
                    directory_stat.st_dev, directory_stat.st_ino,
                ) != self._staging_directory_identity:
                    raise SnapshotError("staging directory changed during write")
                descriptor = os.open(
                    filename,
                    file_flags,
                    0o600,
                    dir_fd=directory_descriptor,
                )
            else:
                # Windows has no openat/dir_fd support. O_EXCL and the pinned
                # post-write directory/file identity checks are the safest
                # stdlib boundary available there.
                descriptor = os.open(
                    self.staging_dir / filename, file_flags, 0o600)
            self._assert_staging_directory()
            success = True
            return descriptor
        except FileExistsError as exc:
            raise SnapshotError(
                f"staged artifact path already exists: {filename}") from exc
        except OSError as exc:
            raise SnapshotError(
                f"staged artifact could not be created safely: {filename}") from exc
        finally:
            if directory_descriptor >= 0:
                os.close(directory_descriptor)
            if descriptor >= 0 and not success:
                os.close(descriptor)

    def _finish_staged_artifact(
        self, filename: str, handle,
    ) -> tuple[Path, int, str]:
        handle.flush()
        os.fsync(handle.fileno())
        before = os.fstat(handle.fileno())
        digest = _sha256_descriptor(handle.fileno(), before.st_size)
        after = os.fstat(handle.fileno())
        if _stat_identity(before) != _stat_identity(after):
            raise SnapshotError(f"staged artifact changed during write: {filename}")
        self._assert_staging_directory()
        destination = _contained_existing(
            self.staging_dir / filename,
            self.root,
            field=f"staged artifact {filename}",
        )
        if _stat_identity(destination.stat()) != _stat_identity(after):
            raise SnapshotError(f"staged artifact changed during write: {filename}")
        return destination, after.st_size, digest

    def _write_staged_json(
        self, filename: str, value: object,
    ) -> tuple[Path, int, str]:
        descriptor = self._open_staged_artifact(filename)
        with os.fdopen(descriptor, "w+b", buffering=0) as handle:
            handle.write(_json_bytes(value))
            return self._finish_staged_artifact(filename, handle)

    def write_table(self, name: str, table: pa.Table, *, schema: pa.Schema,
                    sort_keys: Iterable[str]) -> Path:
        if name in self._artifacts:
            raise SnapshotError(f"table already written: {name}")
        if name not in REQUIRED_TABLES:
            raise SnapshotError(f"unexpected table name: {name}")
        declared_name = (schema.metadata or {}).get(b"tremora.table", b"").decode()
        if declared_name != name:
            raise SnapshotError(f"schema declares {declared_name!r}, not {name!r}")
        expected_schema = canonical_schema(name)
        if not schemas_equal(schema, expected_schema):
            raise SnapshotError(f"caller supplied a noncanonical schema for {name}")
        if not schemas_equal(table.schema, expected_schema):
            raise SnapshotError(f"{name} does not match its canonical Arrow schema")
        if table.num_rows and "recording_id" in table.column_names:
            identifiers = set(table["recording_id"].unique().to_pylist())
            if identifiers != {self.recording_id}:
                raise SnapshotError(
                    f"{name} contains recording IDs outside this snapshot")
        if table.num_rows and "window_policy_id" in table.column_names:
            policies = set(table["window_policy_id"].unique().to_pylist())
            if policies != {self.window_policy_id}:
                raise SnapshotError(f"{name} contains a different window policy")
        keys = tuple(sort_keys)
        if keys != CANONICAL_SORT_KEYS[name]:
            raise SnapshotError(f"noncanonical semantic sort keys for {name}")
        if keys:
            table = table.sort_by([(key, "ascending") for key in keys])
        table = table.replace_schema_metadata(expected_schema.metadata)
        filename = f"{name}.parquet"
        descriptor = self._open_staged_artifact(filename)
        with os.fdopen(descriptor, "w+b", buffering=0) as handle:
            pq.write_table(
                table, handle, compression="zstd", use_dictionary=False,
                write_statistics=True, data_page_version="2.0",
                row_group_size=self.row_group_size,
            )
            destination, artifact_bytes, artifact_sha256 = (
                self._finish_staged_artifact(filename, handle))
        self._artifacts[name] = {
            "path": destination.name,
            "bytes": artifact_bytes,
            "rows": table.num_rows,
            "sha256": artifact_sha256,
            "semantic_sha256": semantic_table_hash(table, sort_keys=keys),
            "schema_sha256": schema_fingerprint(table.schema),
            "sort_keys": list(keys),
            "row_group_size": self.row_group_size,
        }
        return destination

    def commit(self) -> Path:
        missing = REQUIRED_TABLES.difference(self._artifacts)
        if missing:
            raise SnapshotError(f"cannot commit; missing tables: {sorted(missing)!r}")
        provenance = {
            **self._provenance,
            "snapshot_id": self.snapshot_id,
            "recording_id": self.recording_id,
            "schema_version": SCHEMA_VERSION,
            "clock_map_id": self.clock_map_id,
            "window_policy_id": self.window_policy_id,
            "creation_timestamp_utc": self.created_at_utc,
        }
        provenance_path, provenance_bytes, provenance_sha256 = (
            self._write_staged_json("provenance.json", provenance))
        provenance_artifact = {
            "path": provenance_path.name,
            "bytes": provenance_bytes,
            "sha256": provenance_sha256,
        }
        manifest = {
            "manifest_version": SNAPSHOT_MANIFEST_VERSION,
            "snapshot_id": self.snapshot_id,
            "recording_id": self.recording_id,
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": self.created_at_utc,
            "clock_map_id": self.clock_map_id,
            "window_policy_id": self.window_policy_id,
            "row_group_size": self.row_group_size,
            "tables": self._artifacts,
            "provenance": provenance_artifact,
        }
        self._write_staged_json("snapshot_manifest.json", manifest)

        # The disk bytes, rather than cached Arrow tables, are authoritative.
        # This is the publisher-controlled content validation before the atomic
        # directory rename; independently writable files still require external
        # filesystem access control.
        staged_verification = _verify_snapshot_directory(
            self.staging_dir,
            self.snapshot_id,
            containment_parent=self.root,
        )
        staged_manifest = staged_verification.manifest
        expected_identity = {
            "recording_id": self.recording_id,
            "clock_map_id": self.clock_map_id,
            "window_policy_id": self.window_policy_id,
            "created_at_utc": self.created_at_utc,
            "row_group_size": self.row_group_size,
        }
        if any(staged_manifest.get(key) != value
               for key, value in expected_identity.items()):
            raise SnapshotError("staged snapshot identity changed before publication")

        snapshots_dir = _resolved_snapshots_dir(self.root)
        if snapshots_dir != self.snapshots_dir:
            raise SnapshotError("snapshots parent changed during publication")
        if self.final_dir.exists() or self.final_dir.is_symlink():
            raise SnapshotError("snapshot directory appeared during publication")
        _fsync_directory(self.staging_dir)
        _fsync_directory(self.root)
        os.replace(self.staging_dir, self.final_dir)
        _fsync_directory(self.snapshots_dir)
        _fsync_directory(self.root)

        # A failed final verification deliberately leaves an unreferenced
        # snapshot directory and preserves the previous CURRENT pointer.
        final_verification = _verify_snapshot_binding(
            self.root, self.snapshot_id)
        manifest_sha256 = final_verification.manifest_sha256
        if manifest_sha256 != staged_verification.manifest_sha256:
            raise SnapshotError(
                "snapshot manifest changed during final verification")
        current_temp = self.root / f".CURRENT-{self.snapshot_id}.tmp"
        if current_temp.exists() or current_temp.is_symlink():
            raise SnapshotError(f"unexpected current-pointer temp file: {current_temp}")
        _write_json(current_temp, {
            "snapshot_id": self.snapshot_id,
            "snapshot_manifest_sha256": manifest_sha256,
        })
        # Create and fsync the pointer payload before the final artifact and
        # manifest identity check. The replace is then the next operation, so an
        # in-process mutation injected while creating the pointer cannot advance
        # CURRENT. This cannot prevent a separately writable artifact from being
        # corrupted after publication.
        _assert_declared_artifact_hashes(
            self.final_dir,
            staged_manifest,
            snapshot_manifest_sha256=manifest_sha256,
        )
        os.replace(current_temp, self.root / "CURRENT.json")
        _fsync_directory(self.root)
        return self.final_dir


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _assert_declared_artifact_hashes(
    snapshot_dir: Path, manifest: dict[str, object],
    *, snapshot_manifest_sha256: str,
) -> None:
    """Recheck all declared content and its manifest before publication."""

    entries = [
        (name, artifact)
        for name, artifact in manifest["tables"].items()
    ]
    entries.append(("provenance", manifest["provenance"]))
    for name, artifact in entries:
        path = snapshot_dir / artifact["path"]
        path = _contained_existing(path, snapshot_dir, field=f"artifact {name}")
        before = path.stat()
        if before.st_size != artifact["bytes"]:
            raise SnapshotError(f"wrong-sized artifact before publication: {name}")
        observed = sha256_file(path)
        after = path.stat()
        if (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        ) != (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise SnapshotError(f"artifact changed before publication: {name}")
        if observed != artifact["sha256"]:
            raise SnapshotError(f"artifact hash mismatch before publication: {name}")
    manifest_path = _contained_existing(
        snapshot_dir / "snapshot_manifest.json",
        snapshot_dir,
        field="snapshot manifest",
    )
    before = manifest_path.stat()
    observed = sha256_file(manifest_path)
    after = manifest_path.stat()
    if (
        before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns,
    ) != (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    ):
        raise SnapshotError("snapshot manifest changed before publication")
    if observed != snapshot_manifest_sha256:
        raise SnapshotError("snapshot manifest hash mismatch before publication")


def _verify_snapshot_directory(
    snapshot_dir: Path,
    snapshot_id: str,
    *,
    containment_parent: Path,
    verify_semantic: bool = True,
) -> _VerifiedSnapshot:
    if not isinstance(verify_semantic, bool):
        raise SnapshotError("verify_semantic must be a boolean")
    snapshot_dir = _contained_existing(
        snapshot_dir, containment_parent, field="snapshot")
    if not snapshot_dir.is_dir():
        raise SnapshotError("snapshot directory is missing")
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    _contained_existing(manifest_path, snapshot_dir, field="snapshot manifest")
    try:
        manifest_payload = _verified_file_bytes(
            manifest_path, field="snapshot manifest")
        manifest = json.loads(manifest_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError("missing or invalid snapshot manifest") from exc
    if not isinstance(manifest, dict):
        raise SnapshotError("snapshot manifest must be an object")
    if manifest.get("snapshot_id") != snapshot_id:
        raise SnapshotError("snapshot directory and manifest IDs disagree")
    manifest_version = manifest.get("manifest_version")
    if isinstance(manifest_version, bool) \
            or not isinstance(manifest_version, int) \
            or manifest_version != SNAPSHOT_MANIFEST_VERSION:
        raise SnapshotError("snapshot manifest version is not supported")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SnapshotError("snapshot schema version is not supported")
    if not isinstance(manifest.get("recording_id"), str) \
            or not manifest["recording_id"]:
        raise SnapshotError("snapshot recording ID is missing")
    _require_utc_timestamp(
        manifest.get("created_at_utc"), field="snapshot created_at_utc")
    for field in ("clock_map_id", "window_policy_id"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise SnapshotError(f"snapshot manifest field {field!r} is missing")
    declared_row_group_size = manifest.get("row_group_size")
    if isinstance(declared_row_group_size, bool) \
            or not isinstance(declared_row_group_size, int) \
            or declared_row_group_size <= 0:
        raise SnapshotError("snapshot row-group size is invalid")
    table_entries = manifest.get("tables")
    if not isinstance(table_entries, dict) or set(table_entries) != REQUIRED_TABLES:
        raise SnapshotError("snapshot table inventory is incomplete or unexpected")
    provenance_entry = manifest.get("provenance")
    if not isinstance(provenance_entry, dict) \
            or set(provenance_entry) != {"path", "bytes", "sha256"} \
            or provenance_entry.get("path") != "provenance.json":
        raise SnapshotError("snapshot provenance inventory is invalid")
    listed_files = {"snapshot_manifest.json", "provenance.json"}
    artifact_paths: set[str] = set()
    loaded_tables: dict[str, pa.Table] = {}
    for name in sorted(table_entries):
        artifact = table_entries[name]
        if not isinstance(artifact, dict):
            raise SnapshotError(f"invalid artifact record for {name}")
        relative = artifact.get("path")
        expected_relative = f"{name}.parquet"
        if relative != expected_relative:
            raise SnapshotError(f"noncanonical artifact path for {name}")
        if relative in artifact_paths:
            raise SnapshotError("multiple logical tables alias one artifact")
        artifact_paths.add(relative)
        if artifact.get("sort_keys") != list(CANONICAL_SORT_KEYS[name]):
            raise SnapshotError(f"noncanonical semantic sort keys for {name}")
        if isinstance(artifact.get("rows"), bool) \
                or not isinstance(artifact.get("rows"), int) \
                or artifact["rows"] < 0:
            raise SnapshotError(f"invalid row count for {name}")
        if isinstance(artifact.get("bytes"), bool) \
                or not isinstance(artifact.get("bytes"), int) \
                or artifact["bytes"] <= 0:
            raise SnapshotError(f"invalid byte count for {name}")
        if artifact.get("row_group_size") != declared_row_group_size:
            raise SnapshotError(f"inconsistent row-group size for {name}")
        for hash_field in ("sha256", "semantic_sha256", "schema_sha256"):
            if not _is_sha256(artifact.get(hash_field)):
                raise SnapshotError(f"invalid {hash_field} for {name}")
        listed_files.add(relative)
        path = snapshot_dir / relative
        _contained_existing(path, snapshot_dir, field=f"artifact {name}")
        try:
            with _verified_file_source(
                    path,
                    field=f"artifact: {name}",
                    expected_bytes=artifact["bytes"],
                    expected_sha256=artifact["sha256"],
            ) as source:
                parquet_file = pq.ParquetFile(source)
                _validate_row_group_policy(
                    name,
                    parquet_file=parquet_file,
                    declared_rows=artifact["rows"],
                    declared_row_group_size=artifact["row_group_size"],
                )
                table = parquet_file.read()
        except SnapshotError:
            raise
        except pa.ArrowException as exc:
            raise SnapshotError(f"invalid Parquet artifact: {name}") from exc
        if table.num_rows != artifact.get("rows"):
            raise SnapshotError(f"row count mismatch: {name}")
        expected_schema = canonical_schema(name)
        if not schemas_equal(table.schema, expected_schema):
            raise SnapshotError(f"noncanonical schema: {name}")
        if schema_fingerprint(table.schema) != artifact.get("schema_sha256"):
            raise SnapshotError(f"schema fingerprint mismatch: {name}")
        if verify_semantic and semantic_table_hash(
                table, sort_keys=artifact.get("sort_keys", ())) != artifact.get(
                    "semantic_sha256"):
            raise SnapshotError(f"semantic table hash mismatch: {name}")
        loaded_tables[name] = table
    provenance = manifest.get("provenance", {})
    provenance_relative = provenance.get("path")
    if not isinstance(provenance_relative, str) \
            or Path(provenance_relative).name != provenance_relative:
        raise SnapshotError("unsafe provenance path")
    if isinstance(provenance.get("bytes"), bool) \
            or not isinstance(provenance.get("bytes"), int) \
            or provenance["bytes"] <= 0 \
            or not _is_sha256(provenance.get("sha256")):
        raise SnapshotError("snapshot provenance inventory is invalid")
    provenance_path = snapshot_dir / provenance_relative
    _contained_existing(provenance_path, snapshot_dir, field="provenance")
    try:
        provenance_value = json.loads(_verified_file_bytes(
            provenance_path,
            field="provenance",
            expected_bytes=provenance["bytes"],
            expected_sha256=provenance["sha256"],
        ))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError("provenance JSON is invalid") from exc
    if not isinstance(provenance_value, dict):
        raise SnapshotError("provenance JSON must be an object")
    expected_reserved = {
        "snapshot_id": snapshot_id,
        "recording_id": manifest["recording_id"],
        "schema_version": SCHEMA_VERSION,
        "clock_map_id": manifest["clock_map_id"],
        "window_policy_id": manifest["window_policy_id"],
        "creation_timestamp_utc": manifest["created_at_utc"],
    }
    if any(provenance_value.get(key) != value
           for key, value in expected_reserved.items()):
        raise SnapshotError("provenance and manifest identity/version fields disagree")
    try:
        validate_snapshot_tables(
            loaded_tables, recording_id=manifest["recording_id"],
            window_policy_id=manifest["window_policy_id"],
            provenance=provenance_value,
        )
    except StoreInvariantError as exc:
        raise SnapshotError(f"snapshot invariant failed: {exc}") from exc
    actual_entries = {path.name for path in snapshot_dir.iterdir()}
    if actual_entries != listed_files:
        raise SnapshotError("snapshot contains unlisted or missing entries")
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    _verified_file_bytes(
        manifest_path,
        field="snapshot manifest",
        expected_bytes=len(manifest_payload),
        expected_sha256=manifest_sha256,
    )
    return _VerifiedSnapshot(
        manifest=manifest,
        manifest_bytes=len(manifest_payload),
        manifest_sha256=manifest_sha256,
    )


def _resolve_current_snapshot(
    root: str | Path, *, verify_contents: bool,
) -> tuple[str, _VerifiedSnapshot | None]:
    if not isinstance(verify_contents, bool):
        raise SnapshotError("verify_contents must be a boolean")
    root_path = _resolved_store_root(root)
    pointer = root_path / "CURRENT.json"
    if not pointer.is_file():
        raise SnapshotError("missing or invalid CURRENT.json")
    pointer = _contained_existing(pointer, root_path, field="CURRENT pointer")
    try:
        pointer_bytes = pointer.read_bytes()
        value = json.loads(pointer_bytes)
        if not isinstance(value, dict) or set(value) != {
            "snapshot_id", "snapshot_manifest_sha256",
        }:
            raise TypeError("CURRENT pointer fields are invalid")
        snapshot_id = _safe_component(
            value["snapshot_id"], field="CURRENT snapshot_id")
        expected_manifest_hash = value["snapshot_manifest_sha256"]
    except (
        OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError,
    ) as exc:
        raise SnapshotError("missing or invalid CURRENT.json") from exc
    if not _is_sha256(expected_manifest_hash):
        raise SnapshotError("CURRENT snapshot manifest hash is invalid")
    snapshots_dir = _resolved_snapshots_dir(root_path)
    snapshot_dir = snapshots_dir / snapshot_id
    if not snapshot_dir.is_dir():
        raise SnapshotError("CURRENT snapshot directory is missing")
    snapshot_dir = _contained_existing(
        snapshot_dir, snapshots_dir, field="CURRENT snapshot")
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    manifest_path = _contained_existing(
        manifest_path, snapshot_dir, field="CURRENT snapshot manifest")
    verification = None
    if verify_contents:
        verification = _verify_snapshot_binding(root_path, snapshot_id)
        try:
            pointer_changed = pointer.read_bytes() != pointer_bytes
        except OSError as exc:
            raise SnapshotError(
                "CURRENT pointer changed during verification") from exc
        if pointer_changed:
            raise SnapshotError("CURRENT pointer changed during verification")
        if verification.manifest_sha256 != expected_manifest_hash:
            raise SnapshotError(
                "CURRENT snapshot manifest hash mismatch during verification")
    else:
        try:
            manifest_hash = sha256_file(manifest_path)
        except OSError as exc:
            raise SnapshotError(
                "CURRENT snapshot manifest changed during verification") from exc
        if manifest_hash != expected_manifest_hash:
            raise SnapshotError("CURRENT snapshot manifest hash mismatch")
    return snapshot_id, verification


def current_snapshot_id(
    root: str | Path, *, verify_contents: bool = True,
) -> str:
    return _resolve_current_snapshot(
        root, verify_contents=verify_contents)[0]


def _verify_snapshot_binding(
    root: str | Path, snapshot_id: str, *, verify_semantic: bool = True,
) -> _VerifiedSnapshot:
    if not isinstance(verify_semantic, bool):
        raise SnapshotError("verify_semantic must be a boolean")
    root_path = _resolved_store_root(root)
    snapshot_id = _safe_component(snapshot_id, field="snapshot_id")
    snapshots_dir = _resolved_snapshots_dir(root_path)
    snapshot_dir = snapshots_dir / snapshot_id
    if not snapshot_dir.is_dir():
        raise SnapshotError("snapshot directory is missing")
    return _verify_snapshot_directory(
        snapshot_dir,
        snapshot_id,
        containment_parent=snapshots_dir,
        verify_semantic=verify_semantic,
    )


def verify_snapshot(root: str | Path, snapshot_id: str,
                    *, verify_semantic: bool = True) -> dict[str, object]:
    return _verify_snapshot_binding(
        root, snapshot_id, verify_semantic=verify_semantic).manifest


def load_snapshot_manifest(root: str | Path,
                           snapshot_id: str) -> dict[str, object]:
    """Load an unchecked snapshot while retaining strict path containment.

    This is the warm-replay path. It deliberately skips hashes and relational
    verification, but it never relaxes inventory or filesystem boundaries.
    """

    root_path = _resolved_store_root(root)
    snapshot_id = _safe_component(snapshot_id, field="snapshot_id")
    snapshots_dir = _resolved_snapshots_dir(root_path)
    snapshot_dir = snapshots_dir / snapshot_id
    if not snapshot_dir.is_dir():
        raise SnapshotError("snapshot directory is missing")
    snapshot_dir = _contained_existing(
        snapshot_dir, snapshots_dir, field="snapshot")
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    _contained_existing(manifest_path, snapshot_dir, field="snapshot manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError("missing or invalid snapshot manifest") from exc
    if not isinstance(manifest, dict):
        raise SnapshotError("unchecked snapshot manifest must be an object")
    manifest_version = manifest.get("manifest_version")
    if isinstance(manifest_version, bool) \
            or not isinstance(manifest_version, int) \
            or manifest_version != SNAPSHOT_MANIFEST_VERSION \
            or manifest.get("snapshot_id") != snapshot_id \
            or manifest.get("schema_version") != SCHEMA_VERSION:
        raise SnapshotError("unchecked snapshot manifest identity/version is invalid")
    entries = manifest.get("tables")
    if not isinstance(entries, dict) or set(entries) != REQUIRED_TABLES:
        raise SnapshotError("unchecked snapshot table inventory is invalid")
    for name, artifact in entries.items():
        if not isinstance(artifact, dict) \
                or artifact.get("path") != f"{name}.parquet":
            raise SnapshotError(f"unchecked artifact path is invalid: {name}")
        path = snapshot_dir / artifact["path"]
        _contained_existing(path, snapshot_dir, field=f"artifact {name}")
        if not path.is_file():
            raise SnapshotError(f"unchecked artifact is missing: {name}")
    provenance_path = snapshot_dir / "provenance.json"
    provenance_entry = manifest.get("provenance")
    if not isinstance(provenance_entry, dict) \
            or set(provenance_entry) != {"path", "bytes", "sha256"} \
            or provenance_entry.get("path") != "provenance.json" \
            or isinstance(provenance_entry.get("bytes"), bool) \
            or not isinstance(provenance_entry.get("bytes"), int) \
            or provenance_entry["bytes"] <= 0 \
            or not _is_sha256(provenance_entry.get("sha256")):
        raise SnapshotError("unchecked snapshot provenance inventory is invalid")
    _contained_existing(provenance_path, snapshot_dir, field="provenance")
    if not provenance_path.is_file():
        raise SnapshotError("unchecked provenance is missing")
    expected_entries = {
        "snapshot_manifest.json", "provenance.json",
        *(f"{name}.parquet" for name in REQUIRED_TABLES),
    }
    if {path.name for path in snapshot_dir.iterdir()} != expected_entries:
        raise SnapshotError("unchecked snapshot contains unlisted or missing entries")
    return manifest
