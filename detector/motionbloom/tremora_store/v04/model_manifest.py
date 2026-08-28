"""Strict production model, preprocessing, and native-runtime binding.

The authoritative ``production_cv_model_manifest.json`` is deliberately
materialized after the code commit it names.  Keeping the writer here avoids a
false self-referential Git hash in a tracked template.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from types import MappingProxyType

from ..finalize._bundle_io import (
    FinalizationBundleError,
    canonical_json_bytes,
)

MODEL_MANIFEST_VERSION = "tremora-production-cv-model-manifest-1.0.0"
MODEL_FAMILY = "mediapipe.tasks.vision.HandLandmarker"
VENDORED_MODEL_ID = "google-mediapipe-hand-landmarker-full-v1-float16"
VENDORED_MODEL_FILENAME = "hand_landmarker.task"
VENDORED_MODEL_SIZE_BYTES = 7_819_105
VENDORED_MODEL_SHA256 = (
    "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
)
MODEL_WEIGHTS_DTYPE = "float16"
PERSISTED_DETECTION_DTYPE = "float32"
RAW_DETECTION_COORDINATE_SPACE = "NORMALIZED_CV_INPUT"
PERSISTED_DETECTION_COORDINATE_SPACE = "DISPLAY_PIXEL"
ASSOCIATION_CONTRACT_VERSION = "tremora-pose-frame-association-1.0.0"
PREPROCESSING_SCHEMA_VERSION = "tremora-production-cv-preprocessing-1.0.0"
RUNTIME_LOCK_VERSION = "tremora-production-cv-native-runtime-1.0.0"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_RUNTIME_ARTIFACT_BYTES = 1024 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_MANIFEST_FIELDS = frozenset({
    "manifest_version",
    "model_id",
    "model_family",
    "model_weights_size_bytes",
    "model_weights_sha256",
    "model_weights_dtype",
    "code_commit",
    "preprocessing_config_sha256",
    "association_contract_version",
    "runtime_lock_sha256",
    "container_digest",
    "inference_device",
    "thread_count",
    "inference_delegate_thread_count",
    "runtime_worker_concurrency_observed",
    "deterministic_mode",
    "persisted_detection_dtype",
    "raw_detection_coordinate_space",
    "persisted_detection_coordinate_space",
})
_PREPROCESSING_CONFIG = {
    "channel_conversion": "BGR_TO_RGB",
    "input_dtype": "uint8",
    "input_pixel_format": "bgr24",
    "input_scope": "FULL_DISPLAY_FRAME",
    "raw_landmark_output_space": RAW_DETECTION_COORDINATE_SPACE,
    "persisted_landmark_space": PERSISTED_DETECTION_COORDINATE_SPACE,
    "min_hand_detection_confidence": 0.5,
    "min_hand_presence_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "num_hands": 2,
    "resize_policy": "NONE",
    "rotation_policy": "DECODER_DISPLAY_TRANSFORM_ALREADY_APPLIED",
    "running_mode": "IMAGE",
    "schema_version": PREPROCESSING_SCHEMA_VERSION,
}
_RUNTIME_FIELDS = frozenset({
    "critical_artifacts",
    "inference_delegate",
    "inference_delegate_thread_count",
    "packages",
    "platform_machine",
    "platform_system",
    "python_implementation",
    "python_version",
    "runtime_lock_version",
    "runtime_content_aggregate",
    "runtime_worker_concurrency_observed",
    "whole_process_thread_count",
})
_CRITICAL_RUNTIME_ARTIFACTS = {
    "mediapipe_native_runtime": {
        "distribution": "mediapipe",
        "relative_path": "mediapipe/tasks/c/libmediapipe.dylib",
        "size_bytes": 50_690_400,
        "sha256": (
            "f183acadefa74df7d9651beb3ff8339320c544020920e8d9038637f50bfdd453"
        ),
    },
    "numpy_core_runtime": {
        "distribution": "numpy",
        "relative_path": (
            "numpy/_core/_multiarray_umath.cpython-312-darwin.so"
        ),
        "size_bytes": 3_996_152,
        "sha256": (
            "359e4f56a73e02b63b00e9d8e0b4190e1a8cf2a1dc6351c83c7eb2f76c4e16af"
        ),
    },
}
_RUNTIME_CONTENT_AGGREGATE: dict[str, object] = {
    "algorithm": "tremora-runtime-content-aggregate-1",
    "distributions": ["mediapipe"],
    "entry_count": 275,
    "python_executable_sha256": (
        "71720f1fc66989ebd691e81c96111b47ae6ff3f1a478666084d1cacbf0fccbf2"
    ),
    "python_executable_size_bytes": 18_073_984,
    "sha256": (
        "06f5b6dfd442c7a81a1c6c8ec0cdede2a339122b5fc97144a7e1d0b08c73b6e7"
    ),
}
_VERIFIED_MODEL_ATTESTATION = object()
_VENDORED_INVENTORY_ENTRY = {
    "name": "hand_landmarker",
    "path": VENDORED_MODEL_FILENAME,
    "provider": "Google MediaPipe",
    "modelFamily": "hand_landmarker",
    "variant": "full",
    "upstreamVersion": "1",
    "precision": MODEL_WEIGHTS_DTYPE,
    "sizeBytes": VENDORED_MODEL_SIZE_BYTES,
    "sha256": VENDORED_MODEL_SHA256,
}


class ProductionModelError(RuntimeError):
    """Raised when production estimator provenance is absent or has drifted."""


@dataclass(frozen=True, init=False)
class VerifiedProductionModel:
    """A model whose weights, configs, runtime, and manifest all reconciled."""

    manifest_path: Path
    weights_path: Path
    preprocessing_config_path: Path
    runtime_lock_path: Path
    manifest_sha256: str
    weights_bytes: bytes = field(repr=False)
    manifest: Mapping[str, object]
    preprocessing_config: Mapping[str, object]
    runtime_lock: Mapping[str, object]
    _loader_attestation: object = field(repr=False, compare=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ProductionModelError(
            "VerifiedProductionModel is loader-constructed only"
        )


def _loader_verified_model(
    *,
    manifest_path: Path,
    weights_path: Path,
    preprocessing_config_path: Path,
    runtime_lock_path: Path,
    manifest_sha256: str,
    weights_bytes: bytes,
    manifest: Mapping[str, object],
    preprocessing_config: Mapping[str, object],
    runtime_lock: Mapping[str, object],
) -> VerifiedProductionModel:
    model = object.__new__(VerifiedProductionModel)
    values = {
        "manifest_path": manifest_path,
        "weights_path": weights_path,
        "preprocessing_config_path": preprocessing_config_path,
        "runtime_lock_path": runtime_lock_path,
        "manifest_sha256": manifest_sha256,
        "weights_bytes": weights_bytes,
        "manifest": manifest,
        "preprocessing_config": preprocessing_config,
        "runtime_lock": runtime_lock,
        "_loader_attestation": _VERIFIED_MODEL_ATTESTATION,
    }
    for name, value in values.items():
        object.__setattr__(model, name, value)
    return model


def _is_loader_verified_production_model(value: object) -> bool:
    return type(value) is VerifiedProductionModel and getattr(
        value, "_loader_attestation", None
    ) is _VERIFIED_MODEL_ATTESTATION


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProductionModelError(f"{field} must be a lowercase SHA-256")
    return value


def _read_pinned_regular_file(
    path: Path,
    purpose: str,
    *,
    expected_size: int | None = None,
    max_size: int,
    retain_bytes: bool,
) -> tuple[int, str, bytes | None]:
    """Read/hash one non-symlink regular file through a single pinned FD."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProductionModelError(f"{purpose} must be a regular file")
        if before.st_size < 0 or before.st_size > max_size:
            raise ProductionModelError(f"{purpose} exceeds its byte limit")
        if expected_size is not None and before.st_size != expected_size:
            raise ProductionModelError(f"{purpose} size mismatch")
        digest = hashlib.sha256()
        retained = bytearray() if retain_bytes else None
        observed_size = 0
        while observed_size <= before.st_size:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES, before.st_size + 1 - observed_size),
            )
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > before.st_size:
                raise ProductionModelError(f"{purpose} changed while being read")
            digest.update(chunk)
            if retained is not None:
                retained.extend(chunk)
        after = os.fstat(descriptor)
    except ProductionModelError:
        raise
    except OSError as exc:
        raise ProductionModelError(f"{purpose} is required and unreadable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if observed_size != before.st_size or any(
        getattr(before, field_name) != getattr(after, field_name)
        for field_name in identity_fields
    ):
        raise ProductionModelError(f"{purpose} changed while being read")
    return (
        observed_size,
        digest.hexdigest(),
        None if retained is None else bytes(retained),
    )


def _read_canonical_json(path: Path, purpose: str) -> tuple[dict[str, object], bytes]:
    try:
        _, _, encoded = _read_pinned_regular_file(
            path,
            purpose,
            max_size=_MAX_JSON_BYTES,
            retain_bytes=True,
        )
        assert encoded is not None
        value = json.loads(encoded)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProductionModelError(f"{purpose} is required and unreadable") from exc
    if not isinstance(value, dict):
        raise ProductionModelError(f"{purpose} must be a JSON object")
    try:
        expected = canonical_json_bytes(value)
    except FinalizationBundleError as exc:
        raise ProductionModelError(f"{purpose} is not canonical JSON") from exc
    if encoded != expected:
        raise ProductionModelError(f"{purpose} bytes are not canonical")
    return value, encoded


def vendored_model_inventory_path() -> Path:
    """Return the tracked inventory that is authoritative for model naming."""

    return (
        Path(__file__).resolve().parents[3]
        / "electron-purecam"
        / "assets"
        / "mediapipe-models.manifest.json"
    )


def _verify_vendored_model_inventory(path: Path) -> None:
    try:
        _, _, encoded = _read_pinned_regular_file(
            path,
            "vendored model inventory",
            max_size=_MAX_JSON_BYTES,
            retain_bytes=True,
        )
        assert encoded is not None
        value = json.loads(encoded)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProductionModelError(
            "vendored model inventory is required and unreadable"
        ) from exc
    if not isinstance(value, dict) or not isinstance(value.get("models"), list):
        raise ProductionModelError("vendored model inventory is malformed")
    candidates = [
        item for item in value["models"]
        if isinstance(item, dict) and item.get("path") == VENDORED_MODEL_FILENAME
    ]
    if len(candidates) != 1:
        raise ProductionModelError(
            "vendored model inventory must name exactly one hand model"
        )
    candidate = candidates[0]
    for key, expected in _VENDORED_INVENTORY_ENTRY.items():
        if candidate.get(key) != expected:
            raise ProductionModelError(
                f"vendored model inventory mismatch for {key}"
            )


def _read_and_verify_weights(path: Path) -> bytes:
    """Read the task bundle once and pin the exact verified bytes in memory."""

    _, observed_sha256, payload = _read_pinned_regular_file(
        path,
        "model weights",
        expected_size=VENDORED_MODEL_SIZE_BYTES,
        max_size=VENDORED_MODEL_SIZE_BYTES,
        retain_bytes=True,
    )
    assert payload is not None
    if observed_sha256 != VENDORED_MODEL_SHA256:
        raise ProductionModelError("model weights hash mismatch")
    return payload


def _verify_preprocessing_config(
    path: Path,
    expected_sha256: str,
) -> dict[str, object]:
    value, encoded = _read_canonical_json(path, "preprocessing configuration")
    if hashlib.sha256(encoded).hexdigest() != _sha256(
        expected_sha256, "preprocessing_config_sha256"
    ):
        raise ProductionModelError("preprocessing configuration hash mismatch")
    if value != _PREPROCESSING_CONFIG:
        raise ProductionModelError("preprocessing configuration is not frozen v0.4")
    return value


def _validate_runtime_lock(value: dict[str, object]) -> None:
    if set(value) != _RUNTIME_FIELDS:
        raise ProductionModelError("runtime lock fields do not match the v0.4 contract")
    if value["runtime_lock_version"] != RUNTIME_LOCK_VERSION:
        raise ProductionModelError("runtime lock version is not supported")
    if value["inference_delegate"] != "CPU":
        raise ProductionModelError("production inference delegate must be CPU")
    if value["inference_delegate_thread_count"] is not None:
        raise ProductionModelError(
            "delegate thread count must remain null unless independently enforced"
        )
    if value["whole_process_thread_count"] is not None:
        raise ProductionModelError(
            "whole-process thread count must remain null unless independently enforced"
        )
    observed = value["runtime_worker_concurrency_observed"]
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 1:
        raise ProductionModelError("observed runtime worker concurrency is invalid")
    packages = value["packages"]
    if not isinstance(packages, dict) or not packages:
        raise ProductionModelError("runtime lock packages must be a non-empty object")
    for package, package_version in packages.items():
        if not isinstance(package, str) or not package or not isinstance(
            package_version, str
        ) or not package_version:
            raise ProductionModelError("runtime package locks must be strings")
    if value["critical_artifacts"] != _CRITICAL_RUNTIME_ARTIFACTS:
        raise ProductionModelError(
            "critical runtime artifact pins do not match the frozen v0.4 lock"
        )
    if value["runtime_content_aggregate"] != _RUNTIME_CONTENT_AGGREGATE:
        raise ProductionModelError(
            "runtime content aggregate does not match the frozen v0.4 lock"
        )


def _verify_runtime_artifacts(value: dict[str, object]) -> None:
    artifacts = value["critical_artifacts"]
    assert isinstance(artifacts, dict)
    for artifact_id, artifact in artifacts.items():
        assert isinstance(artifact_id, str)
        assert isinstance(artifact, dict)
        package = str(artifact["distribution"])
        relative_path = str(artifact["relative_path"])
        try:
            package_distribution = distribution(package)
        except PackageNotFoundError as exc:
            raise ProductionModelError(
                f"critical runtime distribution is missing: {package}"
            ) from exc
        matching = [
            item for item in package_distribution.files or ()
            if str(item) == relative_path
        ]
        if len(matching) != 1:
            raise ProductionModelError(
                f"critical runtime artifact is not owned by {package}: {artifact_id}"
            )
        artifact_path = Path(package_distribution.locate_file(matching[0]))
        _, observed_sha256, _ = _read_pinned_regular_file(
            artifact_path,
            f"critical runtime artifact {artifact_id}",
            expected_size=int(artifact["size_bytes"]),
            max_size=_MAX_RUNTIME_ARTIFACT_BYTES,
            retain_bytes=False,
        )
        if observed_sha256 != artifact["sha256"]:
            raise ProductionModelError(
                f"critical runtime artifact hash mismatch: {artifact_id}"
            )


def _length_prefixed(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _observed_runtime_content_aggregate(value: dict[str, object]) -> dict[str, object]:
    expected = value["runtime_content_aggregate"]
    assert isinstance(expected, dict)
    distributions = expected["distributions"]
    assert isinstance(distributions, list)
    entries: list[tuple[str, str, int, str]] = []

    python_path = Path(sys.executable).resolve(strict=True)
    python_size, python_sha256, _ = _read_pinned_regular_file(
        python_path,
        "resolved Python executable",
        max_size=_MAX_RUNTIME_ARTIFACT_BYTES,
        retain_bytes=False,
    )
    entries.append(("python", "resolved_executable", python_size, python_sha256))

    for package in distributions:
        if not isinstance(package, str):
            raise ProductionModelError("runtime aggregate distributions are invalid")
        try:
            package_distribution = distribution(package)
        except PackageNotFoundError as exc:
            raise ProductionModelError(
                f"runtime aggregate distribution is missing: {package}"
            ) from exc
        install_root = Path(package_distribution.locate_file("")).resolve(strict=True)
        for item in sorted(package_distribution.files or (), key=str):
            artifact_path = Path(os.path.abspath(package_distribution.locate_file(item)))
            try:
                inside_root = os.path.commonpath((install_root, artifact_path)) == str(
                    install_root
                )
            except ValueError:
                inside_root = False
            if not inside_root:
                continue
            size_bytes, sha256, _ = _read_pinned_regular_file(
                artifact_path,
                f"{package} distribution file {item}",
                max_size=_MAX_RUNTIME_ARTIFACT_BYTES,
                retain_bytes=False,
            )
            entries.append((package, str(item), size_bytes, sha256))

    digest = hashlib.sha256(b"tremora-runtime-content-aggregate-1\0")
    for owner, relative_path, size_bytes, sha256 in sorted(entries):
        for component in (
            owner.encode("utf-8"),
            relative_path.encode("utf-8"),
            size_bytes.to_bytes(8, "big"),
            sha256.encode("ascii"),
        ):
            digest.update(_length_prefixed(component))
    return {
        "algorithm": "tremora-runtime-content-aggregate-1",
        "distributions": distributions,
        "entry_count": len(entries),
        "python_executable_sha256": python_sha256,
        "python_executable_size_bytes": python_size,
        "sha256": digest.hexdigest(),
    }


def _verify_runtime_environment(value: dict[str, object]) -> None:
    observed = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }
    for field_name, actual in observed.items():
        if value[field_name] != actual:
            raise ProductionModelError(
                f"runtime environment mismatch for {field_name}: expected "
                f"{value[field_name]!r}, observed {actual!r}"
            )
    packages = value["packages"]
    assert isinstance(packages, dict)
    for package, expected in packages.items():
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise ProductionModelError(
                f"required runtime package is missing: {package}"
            ) from exc
        if actual != expected:
            raise ProductionModelError(
                f"runtime package mismatch for {package}: expected "
                f"{expected!r}, observed {actual!r}"
            )
    _verify_runtime_artifacts(value)
    if _observed_runtime_content_aggregate(value) != value[
        "runtime_content_aggregate"
    ]:
        raise ProductionModelError("runtime content aggregate mismatch")


def _verify_runtime_lock(
    path: Path,
    expected_sha256: str,
) -> dict[str, object]:
    value, encoded = _read_canonical_json(path, "runtime lock")
    if hashlib.sha256(encoded).hexdigest() != _sha256(
        expected_sha256, "runtime_lock_sha256"
    ):
        raise ProductionModelError("runtime lock hash mismatch")
    _validate_runtime_lock(value)
    _verify_runtime_environment(value)
    return value


def _validate_manifest(value: dict[str, object]) -> None:
    if set(value) != _MANIFEST_FIELDS:
        raise ProductionModelError("model manifest fields do not match the v0.4 contract")
    if value["manifest_version"] != MODEL_MANIFEST_VERSION:
        raise ProductionModelError("model manifest version is not supported")
    if value["model_id"] != VENDORED_MODEL_ID:
        raise ProductionModelError("model_id is not the frozen vendored hand model")
    if value["model_family"] != MODEL_FAMILY:
        raise ProductionModelError("model family is not the frozen HandLandmarker")
    if value["model_weights_size_bytes"] != VENDORED_MODEL_SIZE_BYTES:
        raise ProductionModelError("model weights size is not the vendored size")
    if value["model_weights_sha256"] != VENDORED_MODEL_SHA256:
        raise ProductionModelError("model weights hash is not the vendored hash")
    if value["model_weights_dtype"] != MODEL_WEIGHTS_DTYPE:
        raise ProductionModelError("model weights dtype must be float16")
    _sha256(value["preprocessing_config_sha256"], "preprocessing_config_sha256")
    _sha256(value["runtime_lock_sha256"], "runtime_lock_sha256")
    if not isinstance(value["code_commit"], str) or _GIT_COMMIT_RE.fullmatch(
        value["code_commit"]
    ) is None:
        raise ProductionModelError("code_commit must be a full lowercase Git SHA-1")
    if value["association_contract_version"] != ASSOCIATION_CONTRACT_VERSION:
        raise ProductionModelError("association contract version is not supported")
    if value["container_digest"] is not None:
        raise ProductionModelError("native v0.4 runtime must record container_digest null")
    if value["inference_device"] != "cpu":
        raise ProductionModelError("production inference device must be CPU")
    if value["persisted_detection_dtype"] != PERSISTED_DETECTION_DTYPE:
        raise ProductionModelError("persisted detections must be float32")
    # Native MediaPipe reported runtime workers, but no thread cap was
    # independently enforced. Observations must not be relabeled as controls.
    if value["thread_count"] is not None:
        raise ProductionModelError("thread_count is not an enforced whole-process cap")
    if value["inference_delegate_thread_count"] is not None:
        raise ProductionModelError(
            "inference delegate thread count is not independently enforced"
        )
    observed = value["runtime_worker_concurrency_observed"]
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 1:
        raise ProductionModelError("observed runtime worker concurrency is invalid")
    if value["deterministic_mode"] is not False:
        raise ProductionModelError(
            "deterministic_mode must remain false absent independent proof"
        )
    if value["raw_detection_coordinate_space"] != (
        RAW_DETECTION_COORDINATE_SPACE
    ):
        raise ProductionModelError(
            "raw estimator coordinates must be NORMALIZED_CV_INPUT"
        )
    if value["persisted_detection_coordinate_space"] != (
        PERSISTED_DETECTION_COORDINATE_SPACE
    ):
        raise ProductionModelError(
            "persisted estimator coordinates must be DISPLAY_PIXEL"
        )


def build_production_model_manifest(
    *,
    model_id: str,
    weights_path: str | Path,
    code_commit: str,
    preprocessing_config_path: str | Path,
    runtime_lock_path: str | Path,
    vendored_model_inventory: str | Path | None = None,
    deterministic_mode: bool = False,
) -> dict[str, object]:
    """Build a manifest from verified local bytes after the named code commit."""

    if model_id != VENDORED_MODEL_ID:
        raise ProductionModelError("model_id is not the frozen vendored hand model")
    _verify_vendored_model_inventory(
        vendored_model_inventory_path()
        if vendored_model_inventory is None
        else Path(vendored_model_inventory)
    )
    weights = Path(weights_path)
    weights_bytes = _read_and_verify_weights(weights)
    preprocessing = Path(preprocessing_config_path)
    runtime = Path(runtime_lock_path)
    preprocessing_value, preprocessing_bytes = _read_canonical_json(
        preprocessing, "preprocessing configuration"
    )
    if preprocessing_value != _PREPROCESSING_CONFIG:
        raise ProductionModelError("preprocessing configuration is not frozen v0.4")
    runtime_value, runtime_bytes = _read_canonical_json(runtime, "runtime lock")
    _validate_runtime_lock(runtime_value)
    _verify_runtime_environment(runtime_value)
    manifest = {
        "association_contract_version": ASSOCIATION_CONTRACT_VERSION,
        "code_commit": code_commit,
        "container_digest": None,
        "deterministic_mode": deterministic_mode,
        "inference_delegate_thread_count": runtime_value[
            "inference_delegate_thread_count"
        ],
        "inference_device": "cpu",
        "manifest_version": MODEL_MANIFEST_VERSION,
        "model_family": MODEL_FAMILY,
        "model_id": model_id,
        "model_weights_dtype": MODEL_WEIGHTS_DTYPE,
        "model_weights_sha256": hashlib.sha256(weights_bytes).hexdigest(),
        "model_weights_size_bytes": len(weights_bytes),
        "persisted_detection_coordinate_space": (
            PERSISTED_DETECTION_COORDINATE_SPACE
        ),
        "persisted_detection_dtype": PERSISTED_DETECTION_DTYPE,
        "preprocessing_config_sha256": hashlib.sha256(
            preprocessing_bytes
        ).hexdigest(),
        "raw_detection_coordinate_space": RAW_DETECTION_COORDINATE_SPACE,
        "runtime_lock_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
        "runtime_worker_concurrency_observed": runtime_value[
            "runtime_worker_concurrency_observed"
        ],
        "thread_count": None,
    }
    _validate_manifest(manifest)
    return manifest


def write_production_model_manifest(
    path: str | Path,
    manifest: Mapping[str, object],
) -> str:
    """Publish canonical manifest bytes atomically without replacing a file."""

    value = dict(manifest)
    _validate_manifest(value)
    encoded = canonical_json_bytes(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ProductionModelError(
                "production model manifest already exists and was not replaced"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def load_and_verify_production_model(
    manifest_path: str | Path,
    *,
    weights_path: str | Path,
    preprocessing_config_path: str | Path,
    runtime_lock_path: str | Path,
    expected_manifest_sha256: str,
    vendored_model_inventory: str | Path | None = None,
) -> VerifiedProductionModel:
    """Fail closed unless every production estimator provenance edge verifies."""

    anchored_manifest_sha256 = _sha256(
        expected_manifest_sha256, "expected_manifest_sha256"
    )
    _verify_vendored_model_inventory(
        vendored_model_inventory_path()
        if vendored_model_inventory is None
        else Path(vendored_model_inventory)
    )
    manifest_file = Path(manifest_path)
    manifest, manifest_bytes = _read_canonical_json(
        manifest_file, "production model manifest"
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != anchored_manifest_sha256:
        raise ProductionModelError("production model manifest hash mismatch")
    _validate_manifest(manifest)

    weights = Path(weights_path)
    weights_bytes = _read_and_verify_weights(weights)
    if hashlib.sha256(weights_bytes).hexdigest() != manifest["model_weights_sha256"]:
        raise ProductionModelError("model weights hash mismatch")
    if len(weights_bytes) != manifest["model_weights_size_bytes"]:
        raise ProductionModelError("model weights size mismatch")
    preprocessing = Path(preprocessing_config_path)
    runtime = Path(runtime_lock_path)
    preprocessing_value = _verify_preprocessing_config(
        preprocessing, str(manifest["preprocessing_config_sha256"])
    )
    runtime_value = _verify_runtime_lock(
        runtime,
        str(manifest["runtime_lock_sha256"]),
    )
    if manifest["inference_delegate_thread_count"] != runtime_value[
        "inference_delegate_thread_count"
    ] or manifest["runtime_worker_concurrency_observed"] != runtime_value[
        "runtime_worker_concurrency_observed"
    ]:
        raise ProductionModelError("model manifest and runtime lock disagree")
    return _loader_verified_model(
        manifest_path=manifest_file.resolve(),
        weights_path=weights.resolve(),
        preprocessing_config_path=preprocessing.resolve(),
        runtime_lock_path=runtime.resolve(),
        manifest_sha256=manifest_sha256,
        weights_bytes=weights_bytes,
        manifest=MappingProxyType(manifest),
        preprocessing_config=MappingProxyType(preprocessing_value),
        runtime_lock=MappingProxyType(runtime_value),
    )


def frozen_config_root() -> Path:
    return Path(__file__).with_name("config")


__all__ = [
    "ASSOCIATION_CONTRACT_VERSION",
    "MODEL_FAMILY",
    "MODEL_MANIFEST_VERSION",
    "MODEL_WEIGHTS_DTYPE",
    "PERSISTED_DETECTION_COORDINATE_SPACE",
    "PERSISTED_DETECTION_DTYPE",
    "PREPROCESSING_SCHEMA_VERSION",
    "RAW_DETECTION_COORDINATE_SPACE",
    "RUNTIME_LOCK_VERSION",
    "VENDORED_MODEL_FILENAME",
    "VENDORED_MODEL_ID",
    "VENDORED_MODEL_SHA256",
    "VENDORED_MODEL_SIZE_BYTES",
    "ProductionModelError",
    "VerifiedProductionModel",
    "build_production_model_manifest",
    "frozen_config_root",
    "load_and_verify_production_model",
    "vendored_model_inventory_path",
    "write_production_model_manifest",
]
