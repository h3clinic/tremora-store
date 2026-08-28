"""Small deterministic and atomic I/O boundary for v0.3 frame bundles."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..parquet_writer import semantic_table_hash
from ..schema import schema_fingerprint

ROW_GROUP_SIZE = 65_536
MAX_JSON_ARTIFACT_BYTES = 4 * 1024 * 1024
PARQUET_WRITER_POLICY_ID = (
    "tremora-finalization-parquet-zstd-nodict-stats-dpv2-rg65536-1.0.0"
)
TABLE_FILES = {
    "video_frames": "video_frames.parquet",
    "cv_frame_results": "cv_frame_results.parquet",
    "cv_detections": "cv_detections.parquet",
}
FINALIZATION_FILES = frozenset({
    *TABLE_FILES.values(),
    "finalization_manifest.json",
    "finalization_audit.json",
    "_SUCCESS",
})
SOURCE_FAILURE_FILES = frozenset({
    "source_failure_manifest.json",
    "source_failure_audit.json",
    "_FAILURE",
})
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._=-]{0,127}\Z")


class FinalizationBundleError(RuntimeError):
    """Raised when a frame-finalization bundle is unsafe or inconsistent."""


def safe_component(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None \
            or value in {".", ".."}:
        raise FinalizationBundleError(
            f"{field} must be a safe single path component")
    return value


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (json.dumps(
            value,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n").encode("ascii")
    except (TypeError, ValueError) as exc:
        raise FinalizationBundleError(
            "bundle metadata is not canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_read_descriptor(
    path: str | Path,
    *,
    purpose: str,
) -> tuple[int, os.stat_result]:
    """Open and inspect one regular file without following its final component."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise FinalizationBundleError(
            "secure artifact reads require O_NOFOLLOW and O_NONBLOCK")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(Path(path), flags)
    except OSError as exc:
        raise FinalizationBundleError(
            f"could not securely open {purpose}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FinalizationBundleError(f"{purpose} must be a regular file")
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _read_at(descriptor: int, size: int, offset: int) -> bytes:
    if hasattr(os, "pread"):
        return os.pread(descriptor, size, offset)
    os.lseek(descriptor, offset, os.SEEK_SET)
    return os.read(descriptor, size)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    descriptor, before = _regular_read_descriptor(
        path, purpose="artifact being hashed")
    try:
        offset = 0
        try:
            while offset < before.st_size:
                payload = _read_at(
                    descriptor,
                    min(1024 * 1024, before.st_size - offset),
                    offset,
                )
                if not payload:
                    raise FinalizationBundleError(
                        "artifact ended before its inspected size")
                digest.update(payload)
                offset += len(payload)
        except OSError as exc:
            raise FinalizationBundleError("artifact hashing failed") from exc
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise FinalizationBundleError("artifact changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_descriptor(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(path, flags, 0o600)
    except OSError as exc:
        raise FinalizationBundleError(
            f"could not create staged artifact exclusively: {path.name}") from exc


def _write_bytes_exclusive(path: Path, payload: bytes) -> dict[str, object]:
    descriptor = _exclusive_descriptor(path)
    with os.fdopen(descriptor, "wb", buffering=0) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _rename_noreplace(
    directory_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically rename within one pinned directory without replacing a target."""

    if os.name != "posix":
        raise FinalizationBundleError(
            "atomic no-replace publication requires POSIX")
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError as exc:
            raise FinalizationBundleError(
                "atomic no-replace publication is unavailable") from exc
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            directory_descriptor,
            source,
            directory_descriptor,
            destination,
            0x00000004,  # Darwin RENAME_EXCL.
        )
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise FinalizationBundleError(
                "atomic no-replace publication is unavailable") from exc
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            directory_descriptor,
            source,
            directory_descriptor,
            destination,
            0x00000001,  # Linux RENAME_NOREPLACE.
        )
    else:
        raise FinalizationBundleError(
            "atomic no-replace publication is unsupported on this POSIX platform")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FinalizationBundleError(
            "finalization identity appeared during publication")
    raise OSError(error, os.strerror(error), destination_name)


class FinalizationBundleWriter:
    """Stage one complete identity directory, then publish it in one rename."""

    _required_files = FINALIZATION_FILES
    _json_files = frozenset({
        "finalization_manifest.json", "finalization_audit.json", "_SUCCESS",
    })
    _terminal_marker = "_SUCCESS"

    def __init__(
        self,
        root: str | Path,
        *,
        recording_id: str,
        finalization_id: str,
    ) -> None:
        recording_id = safe_component(recording_id, field="recording_id")
        finalization_id = safe_component(
            finalization_id, field="finalization_id")
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        if root_path.is_symlink() or not root_path.is_dir():
            raise FinalizationBundleError("finalization root must be a real directory")
        self.root = root_path.resolve(strict=True)
        self.recording_dir = self.root / recording_id
        self.recording_dir.mkdir(exist_ok=True)
        if self.recording_dir.is_symlink() \
                or self.recording_dir.resolve(strict=True).parent != self.root:
            raise FinalizationBundleError(
                "recording directory escaped finalization root")
        recording_state = self.recording_dir.stat()
        self._recording_identity = (
            recording_state.st_dev, recording_state.st_ino)
        self.staging_dir = self.recording_dir / f".staging-{finalization_id}"
        self.final_dir = self.recording_dir / finalization_id
        if self.staging_dir.exists() or self.staging_dir.is_symlink():
            raise FinalizationBundleError(
                "an incomplete staging directory already exists; explicit rebuild "
                "or resume is required")
        if self.final_dir.exists() or self.final_dir.is_symlink():
            raise FinalizationBundleError("finalization identity already exists")
        self.staging_dir.mkdir()
        staging_state = self.staging_dir.stat()
        self._staging_identity = (staging_state.st_dev, staging_state.st_ino)
        self._artifacts: dict[str, dict[str, object]] = {}

    def _assert_staging(self) -> None:
        try:
            state = self.staging_dir.lstat()
        except OSError as exc:
            raise FinalizationBundleError(
                "staging directory disappeared") from exc
        if not stat.S_ISDIR(state.st_mode) \
                or (state.st_dev, state.st_ino) != self._staging_identity:
            raise FinalizationBundleError("staging directory changed during write")

    @property
    def artifacts(self) -> dict[str, dict[str, object]]:
        return {name: dict(value) for name, value in self._artifacts.items()}

    def write_table(
        self,
        name: str,
        table: pa.Table,
        *,
        sort_keys: Iterable[str],
    ) -> pa.Table:
        if name not in TABLE_FILES or name in self._artifacts:
            raise FinalizationBundleError(f"invalid or duplicate table: {name}")
        keys = tuple(sort_keys)
        if keys:
            missing = set(keys).difference(table.column_names)
            if missing:
                raise FinalizationBundleError(
                    f"semantic sort keys missing for {name}: {sorted(missing)}")
            table = table.sort_by([(key, "ascending") for key in keys])
        self._assert_staging()
        destination = self.staging_dir / TABLE_FILES[name]
        descriptor = _exclusive_descriptor(destination)
        with os.fdopen(descriptor, "w+b", buffering=0) as handle:
            pq.write_table(
                table,
                handle,
                compression="zstd",
                use_dictionary=False,
                write_statistics=True,
                data_page_version="2.0",
                row_group_size=ROW_GROUP_SIZE,
            )
            handle.flush()
            os.fsync(handle.fileno())
        self._assert_staging()
        artifact_bytes = destination.stat().st_size
        self._artifacts[name] = {
            "path": destination.name,
            "bytes": artifact_bytes,
            "rows": table.num_rows,
            "sha256": sha256_file(destination),
            "semantic_sha256": semantic_table_hash(table, sort_keys=keys),
            "schema_sha256": schema_fingerprint(table.schema),
            "sort_keys": list(keys),
            "row_group_size": ROW_GROUP_SIZE,
            "writer_policy_id": PARQUET_WRITER_POLICY_ID,
        }
        return table

    def write_json(self, name: str, value: Mapping[str, object]) -> dict[str, object]:
        if name not in self._json_files:
            raise FinalizationBundleError(f"unexpected JSON artifact: {name}")
        self._assert_staging()
        return _write_bytes_exclusive(
            self.staging_dir / name, canonical_json_bytes(dict(value)))

    def publish(self) -> Path:
        self._assert_staging()
        if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
            raise FinalizationBundleError(
                "atomic publication requires O_DIRECTORY and O_NOFOLLOW")
        flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        recording_descriptor = os.open(self.recording_dir, flags)
        staging_descriptor = -1
        try:
            recording_state = os.fstat(recording_descriptor)
            if not stat.S_ISDIR(recording_state.st_mode) or (
                recording_state.st_dev, recording_state.st_ino,
            ) != self._recording_identity:
                raise FinalizationBundleError(
                    "recording directory changed during publication")
            staging_descriptor = os.open(
                self.staging_dir.name,
                flags,
                dir_fd=recording_descriptor,
            )
            staging_state = os.fstat(staging_descriptor)
            if not stat.S_ISDIR(staging_state.st_mode) or (
                staging_state.st_dev, staging_state.st_ino,
            ) != self._staging_identity:
                raise FinalizationBundleError(
                    "staging directory changed during publication")
            present = set(os.listdir(staging_descriptor))
            if present != self._required_files:
                raise FinalizationBundleError(
                    f"staged artifact inventory mismatch: {sorted(present)}")
            for name in present:
                artifact = os.stat(
                    name,
                    dir_fd=staging_descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(artifact.st_mode):
                    raise FinalizationBundleError(
                        "staged artifacts must remain regular files")
            os.fsync(staging_descriptor)
            os.fsync(recording_descriptor)
            _rename_noreplace(
                recording_descriptor,
                self.staging_dir.name,
                self.final_dir.name,
            )
            os.fsync(recording_descriptor)
        finally:
            if staging_descriptor >= 0:
                os.close(staging_descriptor)
            os.close(recording_descriptor)
        _fsync_directory(self.root)
        return self.final_dir

    def abort(self) -> None:
        """Remove only this writer's owned staging files, never published data.

        A success marker is removed first.  If an unexpected or non-regular
        entry has appeared, the remaining staging directory is left in place
        for investigation rather than recursively deleting an unknown object.
        """

        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.staging_dir, flags)
        except OSError:
            return
        removable: list[str] = []
        try:
            state = os.fstat(descriptor)
            if not stat.S_ISDIR(state.st_mode) or (
                state.st_dev, state.st_ino,
            ) != self._staging_identity:
                return
            names = set(os.listdir(descriptor))
            if self._terminal_marker in names:
                marker = os.stat(
                    self._terminal_marker,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(marker.st_mode):
                    return
                os.unlink(self._terminal_marker, dir_fd=descriptor)
                names.remove(self._terminal_marker)
            for name in names:
                if name not in self._required_files:
                    return
                entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISREG(entry.st_mode):
                    return
                removable.append(name)
            for name in removable:
                os.unlink(name, dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.rmdir(self.staging_dir)
        except OSError:
            return
        _fsync_directory(self.recording_dir)


class SourceFailureBundleWriter(FinalizationBundleWriter):
    """Atomically publish one canonical source-decode failure outcome."""

    _required_files = SOURCE_FAILURE_FILES
    _json_files = SOURCE_FAILURE_FILES
    _terminal_marker = "_FAILURE"

    def __init__(
        self,
        root: str | Path,
        *,
        recording_id: str,
        failure_id: str,
    ) -> None:
        failure_id = safe_component(failure_id, field="failure_id")
        super().__init__(
            root,
            recording_id=recording_id,
            finalization_id=failure_id,
        )


def read_json(
    path: str | Path,
    *,
    max_bytes: int = MAX_JSON_ARTIFACT_BYTES,
) -> tuple[dict[str, object], bytes]:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) \
            or max_bytes <= 0:
        raise FinalizationBundleError("JSON size limit must be positive")
    descriptor, before = _regular_read_descriptor(
        path, purpose="JSON artifact")
    try:
        if before.st_size > max_bytes:
            raise FinalizationBundleError("JSON artifact exceeds its size limit")
        encoded_buffer = bytearray()
        offset = 0
        try:
            while offset < before.st_size:
                payload = _read_at(
                    descriptor,
                    min(1024 * 1024, before.st_size - offset),
                    offset,
                )
                if not payload:
                    raise FinalizationBundleError(
                        "JSON artifact ended before its inspected size")
                encoded_buffer.extend(payload)
                offset += len(payload)
        except OSError as exc:
            raise FinalizationBundleError("JSON artifact read failed") from exc
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise FinalizationBundleError("JSON artifact changed while reading")
    finally:
        os.close(descriptor)
    encoded = bytes(encoded_buffer)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalizationBundleError("invalid JSON artifact") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != encoded:
        raise FinalizationBundleError("JSON artifact is not canonically encoded")
    return value, encoded


def read_table(path: str | Path) -> pa.Table:
    descriptor, before = _regular_read_descriptor(
        path, purpose="Parquet artifact")
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            table = pq.read_table(handle)
            after = os.fstat(descriptor)
            if _stat_identity(before) != _stat_identity(after):
                raise FinalizationBundleError(
                    "Parquet artifact changed while reading")
        return table
    finally:
        os.close(descriptor)


def read_verified_table(
    path: str | Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> tuple[pa.Table, list[int]]:
    """Hash and decode one Parquet file through the same pinned descriptor."""

    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) \
            or expected_bytes < 0:
        raise FinalizationBundleError("expected artifact byte size is invalid")
    descriptor, before = _regular_read_descriptor(
        path, purpose="Parquet artifact")
    try:
        if before.st_size != expected_bytes:
            raise FinalizationBundleError("Parquet artifact size mismatch")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            if hasattr(os, "pread"):
                payload = os.pread(
                    descriptor, min(1024 * 1024, before.st_size - offset), offset)
            else:
                os.lseek(descriptor, offset, os.SEEK_SET)
                payload = os.read(
                    descriptor, min(1024 * 1024, before.st_size - offset))
            if not payload:
                raise FinalizationBundleError(
                    "Parquet artifact ended before its declared size")
            digest.update(payload)
            offset += len(payload)
        if digest.hexdigest() != expected_sha256:
            raise FinalizationBundleError("Parquet artifact hash mismatch")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            parquet_file = pq.ParquetFile(handle)
            row_groups = [
                parquet_file.metadata.row_group(index).num_rows
                for index in range(parquet_file.metadata.num_row_groups)
            ]
            table = parquet_file.read()
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise FinalizationBundleError(
                "Parquet artifact changed during verified read")
        return table, row_groups
    finally:
        os.close(descriptor)
