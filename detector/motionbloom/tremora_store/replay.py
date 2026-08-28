"""Deterministic replay of aligned windows from one immutable snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .parquet_writer import (
    SnapshotError,
    _contained_existing,
    _read_descriptor_at,
    _read_only_flags,
    _resolve_current_snapshot,
    _resolved_snapshots_dir,
    _verify_snapshot_binding,
    current_snapshot_id,
    load_snapshot_manifest,
    semantic_table_hash,
)


@dataclass(frozen=True, slots=True)
class ReplayedWindow:
    metadata: Mapping[str, object]
    provenance: Mapping[str, object]
    frames: pa.Table
    cv_estimates: pa.Table
    imu_samples: pa.Table
    imu_nearest_context: pa.Table
    alignment: pa.Table
    clock_map: pa.Table
    provenance_sha256: str
    semantic_sha256: str


@dataclass(frozen=True, slots=True)
class _PinnedArtifact:
    path: Path
    descriptor: int
    identity: tuple[int, int, int, int, int]


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _deep_freeze(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


class RecordingStore:
    """Read-only view of one optionally verified TremoraStore snapshot."""

    def __init__(self, root: str | Path, *, snapshot_id: str | None = None,
                 verify: bool = True):
        self._closed = False
        self._pinned_artifacts: Mapping[str, _PinnedArtifact] = MappingProxyType({})
        self._pinned_controls: Mapping[str, _PinnedArtifact] = MappingProxyType({})
        if not isinstance(verify, bool):
            raise SnapshotError("verify must be a boolean")
        try:
            self.root = Path(root).resolve(strict=True)
        except OSError as exc:
            raise SnapshotError("store root is missing") from exc
        if not self.root.is_dir():
            raise SnapshotError("store root is not a directory")
        verification = None
        if snapshot_id is None:
            if verify:
                self.snapshot_id, verification = _resolve_current_snapshot(
                    self.root, verify_contents=True)
            else:
                self.snapshot_id = current_snapshot_id(
                    self.root, verify_contents=False)
        else:
            self.snapshot_id = snapshot_id
            if verify:
                verification = _verify_snapshot_binding(
                    self.root, self.snapshot_id)
        self._verified_reads = verify
        if verify:
            if verification is None:  # pragma: no cover - defensive invariant
                raise SnapshotError("verified snapshot binding is missing")
            manifest = verification.manifest
            self._manifest_sha256 = verification.manifest_sha256
            self._manifest_bytes = verification.manifest_bytes
        else:
            # Warm replay deliberately skips hashes and relational verification,
            # while retaining strict component and containment validation.
            manifest = load_snapshot_manifest(self.root, self.snapshot_id)
            self._manifest_sha256 = None
            self._manifest_bytes = None
        self.snapshot_dir = self.root / "snapshots" / self.snapshot_id
        self._manifest_value = deepcopy(manifest)
        self._artifact_paths = MappingProxyType({
            name: artifact["path"]
            for name, artifact in manifest["tables"].items()
        })
        self._verified_artifacts = MappingProxyType({
            name: (artifact.get("bytes"), artifact.get("sha256"))
            for name, artifact in manifest["tables"].items()
        })
        provenance_artifact = manifest["provenance"]
        self._provenance_artifact = (
            provenance_artifact["bytes"], provenance_artifact["sha256"])
        if self._verified_reads:
            provenance_payload = self._verified_provenance_bytes()
        else:
            try:
                provenance_payload = self._provenance_path().read_bytes()
            except OSError as exc:
                raise SnapshotError("cannot read snapshot provenance") from exc
        try:
            provenance = json.loads(provenance_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SnapshotError("snapshot provenance JSON is invalid") from exc
        if not isinstance(provenance, dict):
            raise SnapshotError("snapshot provenance must be an object")
        self._provenance_value = deepcopy(provenance)
        self._provenance_sha256 = hashlib.sha256(provenance_payload).hexdigest()
        semantic_provenance = {
            key: value for key, value in provenance.items()
            if key not in {"snapshot_id", "creation_timestamp_utc"}
        }
        self._provenance_semantic_sha256 = hashlib.sha256(json.dumps(
            semantic_provenance, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if self._verified_reads:
            pinned_artifacts: dict[str, _PinnedArtifact] = {}
            pinned_controls: dict[str, _PinnedArtifact] = {}
            try:
                for table_name, (expected_bytes, expected_sha256) in \
                        self._verified_artifacts.items():
                    pinned_artifacts[table_name] = self._pin_verified_file(
                        self._path(table_name),
                        label=f"artifact {table_name}",
                        expected_bytes=expected_bytes,
                        expected_sha256=expected_sha256,
                    )
                manifest_path = self.snapshot_dir / "snapshot_manifest.json"
                pinned_controls["snapshot manifest"] = self._pin_verified_file(
                    manifest_path,
                    label="snapshot manifest",
                    expected_bytes=self._manifest_bytes,
                    expected_sha256=self._manifest_sha256,
                )
                pinned_controls["provenance"] = self._pin_verified_file(
                    self._provenance_path(),
                    label="provenance",
                    expected_bytes=self._provenance_artifact[0],
                    expected_sha256=self._provenance_artifact[1],
                )
            except Exception:
                for pinned in (*pinned_artifacts.values(),
                               *pinned_controls.values()):
                    os.close(pinned.descriptor)
                raise
            self._pinned_artifacts = MappingProxyType(pinned_artifacts)
            self._pinned_controls = MappingProxyType(pinned_controls)

    @property
    def manifest(self) -> dict[str, object]:
        """Return a defensive copy; replay never trusts caller-mutable state."""

        return deepcopy(self._manifest_value)

    @property
    def provenance(self) -> dict[str, object]:
        """Return a defensive copy of the stream/legal/processing semantics."""

        return deepcopy(self._provenance_value)

    def _path(self, table_name: str) -> Path:
        try:
            relative = self._artifact_paths[table_name]
        except (KeyError, TypeError) as exc:
            raise SnapshotError(f"table is not in snapshot: {table_name}") from exc
        expected = f"{table_name}.parquet"
        if relative != expected:
            raise SnapshotError(f"noncanonical replay artifact path: {table_name}")
        path = self.snapshot_dir / relative
        if path.is_symlink() or not path.is_file():
            raise SnapshotError(f"replay artifact is missing or a symlink: {table_name}")
        snapshots_dir = _resolved_snapshots_dir(self.root)
        snapshot_dir = _contained_existing(
            self.snapshot_dir, snapshots_dir, field="replay snapshot")
        return _contained_existing(
            path, snapshot_dir, field=f"replay artifact {table_name}")

    def _provenance_path(self) -> Path:
        relative = self._manifest_value["provenance"]["path"]
        if relative != "provenance.json":
            raise SnapshotError("noncanonical replay provenance path")
        path = self.snapshot_dir / relative
        if path.is_symlink() or not path.is_file():
            raise SnapshotError("replay provenance is missing or a symlink")
        snapshot_dir = _contained_existing(
            self.snapshot_dir, _resolved_snapshots_dir(self.root),
            field="replay snapshot")
        return _contained_existing(
            path, snapshot_dir, field="replay provenance")

    @staticmethod
    def _open_readonly(path: Path, *, label: str) -> int:
        flags = _read_only_flags()
        try:
            return os.open(path, flags)
        except OSError as exc:
            raise SnapshotError(f"cannot open verified replay {label}") from exc

    @classmethod
    def _pin_verified_file(
        cls, path: Path, *, label: str, expected_bytes: int,
        expected_sha256: str,
    ) -> _PinnedArtifact:
        """Hash a descriptor once, then retain it as the session identity anchor."""

        descriptor = cls._open_readonly(path, label=label)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise SnapshotError(f"replay {label} is not a regular file")
            if before.st_size != expected_bytes:
                raise SnapshotError(
                    f"replay {label} changed size after verification")
            digest = hashlib.sha256()
            offset = 0
            while offset < expected_bytes:
                payload = _read_descriptor_at(
                    descriptor, min(1024 * 1024, expected_bytes - offset), offset)
                if not payload:
                    raise SnapshotError(
                        f"replay {label} changed size during pinned verification")
                digest.update(payload)
                offset += len(payload)
            after = os.fstat(descriptor)
            if _file_identity(before) != _file_identity(after):
                raise SnapshotError(
                    f"replay {label} changed during pinned verification")
            if digest.hexdigest() != expected_sha256:
                raise SnapshotError(
                    f"replay {label} hash mismatch after verification")
            return _PinnedArtifact(
                path=path, descriptor=descriptor,
                identity=_file_identity(after))
        except Exception:
            os.close(descriptor)
            raise

    def _assert_open(self) -> None:
        if self._closed:
            raise SnapshotError("recording store session is closed")

    @staticmethod
    def _assert_pinned_file(pinned: _PinnedArtifact, *, label: str) -> None:
        try:
            descriptor_stat = os.fstat(pinned.descriptor)
            path_stat = pinned.path.lstat()
        except OSError as exc:
            raise SnapshotError(
                f"replay {label} disappeared after verification") from exc
        if descriptor_stat.st_size != pinned.identity[2] \
                or path_stat.st_size != pinned.identity[2]:
            raise SnapshotError(
                f"replay {label} changed size after verification")
        if not stat.S_ISREG(path_stat.st_mode) \
                or _file_identity(descriptor_stat) != pinned.identity \
                or _file_identity(path_stat) != pinned.identity:
            raise SnapshotError(
                f"replay {label} identity or metadata changed after verification")

    def _assert_artifacts_unchanged(self, table_names: tuple[str, ...]) -> None:
        """Perform bounded identity checks around a once-verified read."""

        self._assert_open()
        if not self._verified_reads:
            return
        for label, pinned in self._pinned_controls.items():
            self._assert_pinned_file(pinned, label=label)
        for table_name in table_names:
            self._assert_pinned_file(
                self._pinned_artifacts[table_name],
                label=f"artifact {table_name}")

    @staticmethod
    def _verified_payload_bytes(
        path: Path, *, label: str, expected_bytes: int, expected_sha256: str,
    ) -> bytes:
        """Read and verify one immutable payload through a single descriptor."""

        flags = _read_only_flags()
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise SnapshotError(f"cannot open verified replay {label}") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise SnapshotError(f"replay {label} is not a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read()
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise SnapshotError(f"replay {label} changed during pinned read")
        if len(payload) != expected_bytes:
            raise SnapshotError(f"replay {label} changed size after verification")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise SnapshotError(f"replay {label} hash mismatch after verification")
        return payload

    def _verified_provenance_bytes(self) -> bytes:
        expected_bytes, expected_sha256 = self._provenance_artifact
        return self._verified_payload_bytes(
            self._provenance_path(), label="provenance",
            expected_bytes=expected_bytes, expected_sha256=expected_sha256)

    def _read_table(self, table_name: str, **kwargs) -> pa.Table:
        if self._verified_reads:
            pinned = self._pinned_artifacts[table_name]
            self._assert_pinned_file(pinned, label=f"artifact {table_name}")
            descriptor = self._open_readonly(
                pinned.path, label=f"artifact {table_name}")
            handle = None
            source = None
            try:
                before = os.fstat(descriptor)
                if _file_identity(before) != pinned.identity:
                    raise SnapshotError(
                        f"replay artifact {table_name} changed before read")
                handle = os.fdopen(descriptor, "rb", buffering=0)
                descriptor = -1
                source = pa.PythonFile(handle, mode="r")
                try:
                    result = pq.read_table(
                        source, partitioning=None, **kwargs)
                except pa.ArrowException as exc:
                    raise SnapshotError(
                        f"invalid replay Parquet artifact: {table_name}") from exc
                after = os.fstat(handle.fileno())
                if _file_identity(after) != pinned.identity:
                    raise SnapshotError(
                        f"replay artifact {table_name} changed during read")
            finally:
                if source is not None:
                    source.close()
                elif handle is not None:
                    handle.close()
                elif descriptor >= 0:
                    os.close(descriptor)
            self._assert_pinned_file(pinned, label=f"artifact {table_name}")
            return result
        try:
            return pq.read_table(
                self._path(table_name), partitioning=None, **kwargs)
        except pa.ArrowException as exc:
            raise SnapshotError(
                f"invalid replay Parquet artifact: {table_name}") from exc

    def close(self) -> None:
        """Close descriptor anchors for this verified replay session."""

        if self._closed:
            return
        self._closed = True
        for pinned in (*self._pinned_artifacts.values(),
                       *self._pinned_controls.values()):
            try:
                os.close(pinned.descriptor)
            except OSError:
                pass

    def __enter__(self) -> RecordingStore:  # noqa: PYI034
        """Return this store; ``typing.Self`` is unavailable on Python 3.10."""

        self._assert_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort descriptor cleanup
        self.close()

    def _window_row(self, window_id: str) -> dict[str, object]:
        table = self._read_table(
            "window_index", filters=[("window_id", "=", window_id)])
        if table.num_rows != 1:
            raise SnapshotError(f"expected one valid window, found {table.num_rows}")
        return table.to_pylist()[0]

    def window_ids(self) -> tuple[str, ...]:
        accessed = ("window_index",)
        self._assert_artifacts_unchanged(accessed)
        try:
            table = self._read_table(
                "window_index", columns=["window_id"])
            result = tuple(sorted(table["window_id"].to_pylist()))
        finally:
            self._assert_artifacts_unchanged(accessed)
        return result

    def replay_window(self, window_id: str) -> ReplayedWindow:
        if not isinstance(window_id, str) or not window_id:
            raise SnapshotError("window_id must be a non-empty string")
        accessed = (
            "window_index", "frame_index", "cv_estimates", "imu_samples",
            "frame_imu_index", "clock_map",
        )
        self._assert_artifacts_unchanged(accessed)
        try:
            window = self._window_row(window_id)
            recording_id = window["recording_id"]
            video_stream_id = window["video_stream_id"]
            imu_stream_id = window["imu_stream_id"]
            frame_lo = window["frame_start_ordinal"]
            frame_hi = window["frame_stop_ordinal"]
            imu_lo = window["imu_start_ordinal"]
            imu_hi = window["imu_stop_ordinal"]
            frames = self._read_table("frame_index", filters=[
                ("recording_id", "=", recording_id),
                ("video_stream_id", "=", video_stream_id),
                ("canonical_ordinal", ">=", frame_lo),
                ("canonical_ordinal", "<", frame_hi),
            ]).sort_by([("canonical_ordinal", "ascending")])
            cv_estimates = self._read_table("cv_estimates", filters=[
                ("recording_id", "=", recording_id),
                ("video_stream_id", "=", video_stream_id),
                ("canonical_ordinal", ">=", frame_lo),
                ("canonical_ordinal", "<", frame_hi),
            ]).sort_by([("canonical_ordinal", "ascending")])
            imu_samples = self._read_table("imu_samples", filters=[
                ("recording_id", "=", recording_id),
                ("stream_id", "=", imu_stream_id),
                ("canonical_ordinal", ">=", imu_lo),
                ("canonical_ordinal", "<", imu_hi),
            ]).sort_by([("canonical_ordinal", "ascending")])
            alignment = self._read_table("frame_imu_index", filters=[
                ("recording_id", "=", recording_id),
                ("video_stream_id", "=", video_stream_id),
                ("imu_stream_id", "=", imu_stream_id),
                ("frame_canonical_ordinal", ">=", frame_lo),
                ("frame_canonical_ordinal", "<", frame_hi),
            ]).sort_by(
                [("frame_canonical_ordinal", "ascending")])
            context_ordinals = sorted({
                ordinal
                for ordinal in alignment["imu_nearest_ordinal"].to_pylist()
                if ordinal is not None and not imu_lo <= ordinal < imu_hi
            })
            if context_ordinals:
                imu_nearest_context = self._read_table("imu_samples", filters=[
                    ("recording_id", "=", recording_id),
                    ("stream_id", "=", imu_stream_id),
                    ("canonical_ordinal", "in", context_ordinals),
                ]).sort_by([("canonical_ordinal", "ascending")])
            else:
                imu_nearest_context = imu_samples.slice(0, 0)
            clock_map = self._read_table("clock_map", filters=[
                ("recording_id", "=", recording_id),
            ])
            clock_map = clock_map.filter(pc.is_in(
                clock_map["stream_id"],
                value_set=pa.array([video_stream_id, imu_stream_id]),
            )).sort_by([
                ("stream_id", "ascending"),
                ("acquisition_ordinal", "ascending"),
            ])
            if not clock_map.num_rows:
                raise SnapshotError("replayed window has no clock-map provenance")
            if frames.num_rows != window["frame_count"]:
                raise SnapshotError(
                    "replayed frame count disagrees with window index")
            if imu_samples.num_rows != window["imu_sample_count"]:
                raise SnapshotError("replayed IMU count disagrees with window index")
            if cv_estimates.num_rows != frames.num_rows:
                raise SnapshotError("replayed CV row count disagrees with frames")
            if alignment.num_rows != frames.num_rows:
                raise SnapshotError(
                    "frame/IMU alignment row count disagrees with frames")
            expected_frames = list(range(frame_lo, frame_hi))
            expected_imu = list(range(imu_lo, imu_hi))
            if frames["canonical_ordinal"].to_pylist() != expected_frames \
                    or cv_estimates["canonical_ordinal"].to_pylist() \
                    != expected_frames:
                raise SnapshotError("replayed frame/CV ordinals are not complete")
            if imu_samples["canonical_ordinal"].to_pylist() != expected_imu:
                raise SnapshotError("replayed IMU ordinals are not complete")
            if alignment["frame_canonical_ordinal"].to_pylist() != expected_frames:
                raise SnapshotError("replayed alignment ordinals are not complete")
            if imu_nearest_context["canonical_ordinal"].to_pylist() \
                    != context_ordinals:
                raise SnapshotError(
                    "replayed nearest-sample context is incomplete")
            available_imu_ordinals = set(expected_imu) | set(context_ordinals)
            if any(
                ordinal is not None and ordinal not in available_imu_ordinals
                for ordinal in alignment["imu_nearest_ordinal"].to_pylist()
            ):
                raise SnapshotError(
                    "replayed alignment references an unavailable IMU sample")
            for frame, estimate, aligned in zip(
                    frames.to_pylist(), cv_estimates.to_pylist(),
                    alignment.to_pylist(), strict=True):
                if estimate["frame_index"] != frame["frame_index"] \
                        or estimate["canonical_time_ns"] \
                        != frame["canonical_time_ns"]:
                    raise SnapshotError("replayed CV row disagrees with its frame")
                if aligned["frame_index"] != frame["frame_index"] \
                        or aligned["frame_time_ns"] \
                        != frame["canonical_time_ns"]:
                    raise SnapshotError(
                        "replayed alignment row disagrees with its frame")
            start_ns = window["start_time_ns"]
            end_ns = window["end_time_ns"]
            for name, table in (("frame", frames), ("CV", cv_estimates),
                                ("IMU", imu_samples)):
                if "canonical_time_ns" not in table.column_names:
                    continue
                times = table["canonical_time_ns"].to_pylist()
                if any(time < start_ns or time >= end_ns for time in times):
                    raise SnapshotError(
                        f"replayed {name} timestamp lies outside the indexed window")
            component_hashes = {
                "window": window,
                "frames": semantic_table_hash(
                    frames, sort_keys=("canonical_ordinal",)),
                "cv_estimates": semantic_table_hash(
                    cv_estimates, sort_keys=("canonical_ordinal",)),
                "imu_samples": semantic_table_hash(
                    imu_samples, sort_keys=("canonical_ordinal",)),
                "imu_nearest_context": semantic_table_hash(
                    imu_nearest_context, sort_keys=("canonical_ordinal",)),
                "alignment": semantic_table_hash(
                    alignment, sort_keys=("frame_canonical_ordinal",)),
                "clock_map": semantic_table_hash(
                    clock_map,
                    sort_keys=("stream_id", "acquisition_ordinal")),
                "provenance_semantic_sha256": (
                    self._provenance_semantic_sha256),
            }
            encoded = json.dumps(
                component_hashes, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode()
            semantic_sha256 = hashlib.sha256(encoded).hexdigest()
            result = ReplayedWindow(
                metadata=MappingProxyType(dict(window)),
                provenance=_deep_freeze(self._provenance_value), frames=frames,
                cv_estimates=cv_estimates,
                imu_samples=imu_samples,
                imu_nearest_context=imu_nearest_context,
                alignment=alignment,
                clock_map=clock_map,
                provenance_sha256=self._provenance_sha256,
                semantic_sha256=semantic_sha256)
        finally:
            self._assert_artifacts_unchanged(accessed)
        return result
