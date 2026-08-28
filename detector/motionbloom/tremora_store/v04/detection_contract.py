"""Payload-derived v0.4 detection identity and separate selection policy."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pyarrow as pa

from .schemas import (
    HAND_LANDMARK_COUNT,
    PRIMARY_HAND_SELECTION_CONTRACT_VERSION,
    V04_ASSOCIATION_CONTRACT_VERSION,
    cv_detections_v04_schema,
    primary_hand_selection_schema,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DETECTION_ID_DOMAIN = b"tremora-cv-detection-payload-1\0"
SELECTION_STATUSES = frozenset({"SELECTED", "ABSTAINED"})
INFERENCE_STATUSES = frozenset({
    "SUCCESS",
    "NO_DETECTION",
    "DECODE_FAILURE",
    "PREPROCESS_FAILURE",
    "INFERENCE_FAILURE",
    "REJECTED_INPUT",
})


class DetectionContractError(ValueError):
    """Raised when raw detections or primary selection are inconsistent."""


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DetectionContractError(f"{field} must be a lowercase SHA-256")
    return value


def _length_prefixed(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _canonical_float32_array(
    value: object,
    shape: tuple[int, ...],
    field: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.size != int(np.prod(shape)) or not np.isfinite(result).all():
        raise DetectionContractError(
            f"{field} must be finite float32 with shape {shape}"
        )
    result = np.ascontiguousarray(result.reshape(shape).astype("<f4", copy=False))
    # Signed zero is not a different model result for identity purposes.
    result[result == 0] = np.float32(0.0)
    return result


def _optional_probability(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DetectionContractError(f"{field} must be numeric or null")
    result = float(np.float32(value))
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise DetectionContractError(f"{field} must be in [0,1] or null")
    return result


def _optional_float32_array(
    value: object,
    shape: tuple[int, ...],
    field: str,
) -> np.ndarray | None:
    if value is None or (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == int(np.prod(shape))
        and all(item is None for item in value)
    ):
        return None
    return _canonical_float32_array(value, shape, field)


def _canonical_detection(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DetectionContractError("each detection must be a mapping")
    bbox = _canonical_float32_array(
        value.get("bbox_xyxy_display"), (4,), "bbox_xyxy_display"
    )
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise DetectionContractError("bbox_xyxy_display must be ordered")
    xy = _canonical_float32_array(
        value.get("landmarks_xy_display"),
        (HAND_LANDMARK_COUNT, 2),
        "landmarks_xy_display",
    )
    z = _optional_float32_array(
        value.get("landmarks_z_model"),
        (HAND_LANDMARK_COUNT,),
        "landmarks_z_model",
    )
    landmark_confidence = _optional_float32_array(
        value.get("landmark_confidence"),
        (HAND_LANDMARK_COUNT,),
        "landmark_confidence",
    )
    if landmark_confidence is not None and (
        np.any(landmark_confidence < 0) or np.any(landmark_confidence > 1)
    ):
        raise DetectionContractError("landmark_confidence must be in [0,1]")
    validity_value = value.get("landmark_validity_mask")
    if validity_value is None:
        validity = np.ones(HAND_LANDMARK_COUNT, dtype=np.bool_)
    else:
        validity = np.asarray(validity_value)
        if validity.shape != (HAND_LANDMARK_COUNT,) or validity.dtype.kind != "b":
            raise DetectionContractError(
                "landmark_validity_mask must contain 21 booleans"
            )
        validity = validity.astype(np.bool_, copy=False)
    if not np.any(validity):
        raise DetectionContractError("a detection needs at least one valid landmark")
    handedness_value = value.get("handedness")
    if handedness_value is None:
        handedness = None
    elif isinstance(handedness_value, str) and handedness_value:
        handedness = unicodedata.normalize("NFC", handedness_value)
    else:
        raise DetectionContractError("handedness must be non-empty text or null")
    return {
        "handedness": handedness,
        "handedness_confidence": _optional_probability(
            value.get("handedness_confidence"), "handedness_confidence"
        ),
        "detection_confidence": _optional_probability(
            value.get("detection_confidence"), "detection_confidence"
        ),
        "bbox_xyxy_display": bbox,
        "landmarks_xy_display": xy,
        "landmarks_z_model": z,
        "landmark_confidence": landmark_confidence,
        "landmark_validity_mask": validity,
    }


def _optional_float_bytes(value: float | None) -> bytes:
    if value is None:
        return b"\x00"
    return b"\x01" + np.asarray([value], dtype="<f4").tobytes()


def _optional_array_bytes(value: np.ndarray | None) -> bytes:
    if value is None:
        return b"\x00"
    return b"\x01" + value.tobytes(order="C")


def _payload_bytes(
    *,
    frame_id: str,
    model_manifest_sha256: str,
    detection: Mapping[str, object],
) -> bytes:
    frame_hash = _sha256(frame_id, "frame_id")
    model_hash = _sha256(model_manifest_sha256, "model_manifest_sha256")
    canonical = _canonical_detection(detection)
    handedness = canonical["handedness"]
    assert handedness is None or isinstance(handedness, str)
    fields = (
        frame_hash.encode("ascii"),
        model_hash.encode("ascii"),
        b"" if handedness is None else handedness.encode("utf-8"),
        canonical["bbox_xyxy_display"].tobytes(order="C"),
        canonical["landmarks_xy_display"].tobytes(order="C"),
        _optional_array_bytes(canonical["landmarks_z_model"]),
        _optional_float_bytes(canonical["handedness_confidence"]),
        _optional_float_bytes(canonical["detection_confidence"]),
        _optional_array_bytes(canonical["landmark_confidence"]),
        canonical["landmark_validity_mask"].astype(np.uint8).tobytes(),
    )
    return _DETECTION_ID_DOMAIN + b"".join(_length_prefixed(field) for field in fields)


def stable_payload_detection_id(
    *,
    frame_id: str,
    model_manifest_sha256: str,
    detection: Mapping[str, object],
    same_payload_rank: int,
) -> str:
    """Hash actual float32 result bytes plus duplicate-payload rank."""

    if isinstance(same_payload_rank, bool) or not isinstance(
        same_payload_rank, int
    ) or same_payload_rank < 0:
        raise DetectionContractError("same_payload_rank must be non-negative")
    payload = _payload_bytes(
        frame_id=frame_id,
        model_manifest_sha256=model_manifest_sha256,
        detection=detection,
    )
    digest = hashlib.sha256()
    digest.update(payload)
    digest.update(_length_prefixed(same_payload_rank.to_bytes(8, "big")))
    return digest.hexdigest()


def _persisted_row(canonical: dict[str, object]) -> dict[str, object]:
    z = canonical["landmarks_z_model"]
    confidence = canonical["landmark_confidence"]
    return {
        "handedness": canonical["handedness"],
        "handedness_confidence": canonical["handedness_confidence"],
        "detection_confidence": canonical["detection_confidence"],
        "bbox_xyxy_display": canonical["bbox_xyxy_display"].tolist(),
        "landmarks_xy_display": canonical["landmarks_xy_display"].reshape(-1).tolist(),
        "landmarks_z_model": (
            [None] * HAND_LANDMARK_COUNT if z is None else z.tolist()
        ),
        "landmark_confidence": (
            [None] * HAND_LANDMARK_COUNT if confidence is None else confidence.tolist()
        ),
        "landmark_validity_mask": canonical["landmark_validity_mask"].tolist(),
    }


def build_detection_rows(
    *,
    frame_id: str,
    model_manifest_sha256: str,
    detections: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return canonical rows whose IDs and order ignore estimator row order."""

    entries: list[tuple[bytes, dict[str, object]]] = []
    for detection in detections:
        canonical = _canonical_detection(detection)
        persisted = _persisted_row(canonical)
        payload = _payload_bytes(
            frame_id=frame_id,
            model_manifest_sha256=model_manifest_sha256,
            detection=persisted,
        )
        entries.append((payload, persisted))
    entries.sort(key=lambda item: item[0])

    seen_payloads: Counter[bytes] = Counter()
    rows: list[dict[str, object]] = []
    for detection_rank, (payload, persisted) in enumerate(entries):
        same_payload_rank = seen_payloads[payload]
        seen_payloads[payload] += 1
        detection_id = stable_payload_detection_id(
            frame_id=frame_id,
            model_manifest_sha256=model_manifest_sha256,
            detection=persisted,
            same_payload_rank=same_payload_rank,
        )
        rows.append({
            "detection_id": detection_id,
            "frame_id": frame_id,
            "model_manifest_sha256": model_manifest_sha256,
            "detection_rank": detection_rank,
            "same_payload_rank": same_payload_rank,
            **persisted,
        })
    return rows


def detection_rows_table(rows: Sequence[Mapping[str, object]]) -> pa.Table:
    return pa.Table.from_pylist([dict(row) for row in rows], cv_detections_v04_schema())


def build_primary_hand_selection(
    *,
    frame_id: str,
    detections: Sequence[Mapping[str, object]],
    inference_status: str,
) -> dict[str, object]:
    """Select only a single unambiguous detection.

    No row-owned handedness value is permitted to act as protocol authority.
    Multiple detections therefore always produce an explicit abstention until
    a separately trust-anchored protocol mapping exists.
    """

    frame_detections = [row for row in detections if row.get("frame_id") == frame_id]
    if len(frame_detections) != len(detections):
        raise DetectionContractError("selection received a detection from another frame")
    if inference_status not in INFERENCE_STATUSES:
        raise DetectionContractError("selection inference_status is invalid")
    if inference_status == "SUCCESS" and not detections:
        raise DetectionContractError("SUCCESS selection requires a detection")
    if inference_status != "SUCCESS" and detections:
        raise DetectionContractError(
            "non-SUCCESS selection cannot contain detections"
        )

    base = {
        "frame_id": frame_id,
        "primary_hand_selection_contract_version": (
            PRIMARY_HAND_SELECTION_CONTRACT_VERSION
        ),
        "inference_status": inference_status,
    }
    if inference_status not in {"SUCCESS", "NO_DETECTION"}:
        return {
            **base,
            "selection_status": "ABSTAINED",
            "selected_detection_id": None,
            "selection_reason": f"INFERENCE_STATUS_{inference_status}",
        }
    if not detections:
        return {
            **base,
            "selection_status": "ABSTAINED",
            "selected_detection_id": None,
            "selection_reason": "NO_DETECTIONS",
        }
    for row in detections:
        _sha256(row.get("detection_id"), "detection_id")
    if len(detections) != 1:
        return {
            **base,
            "selection_status": "ABSTAINED",
            "selected_detection_id": None,
            "selection_reason": "MULTIPLE_DETECTIONS",
        }
    selected = detections[0]
    return {
        **base,
        "selection_status": "SELECTED",
        "selected_detection_id": selected["detection_id"],
        "selection_reason": "SINGLE_DETECTION",
    }


def primary_selection_table(rows: Sequence[Mapping[str, object]]) -> pa.Table:
    return pa.Table.from_pylist(
        [dict(row) for row in rows], primary_hand_selection_schema()
    )


def _rows(value: pa.Table | Iterable[Mapping[str, object]]) -> list[dict[str, Any]]:
    if isinstance(value, pa.Table):
        return value.to_pylist()
    return [dict(row) for row in value]


def validate_detection_and_selection_rows(
    detections: pa.Table | Iterable[Mapping[str, object]],
    selections: pa.Table | Iterable[Mapping[str, object]],
    *,
    frame_ids: Iterable[str] | None = None,
) -> None:
    """Recompute canonical ranks, identities, and every selection outcome."""

    detection_rows = _rows(detections)
    selection_rows = _rows(selections)
    detection_fields = frozenset(cv_detections_v04_schema().names)
    selection_fields = frozenset(primary_hand_selection_schema().names)
    ids: set[str] = set()
    by_frame: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detection_rows:
        if set(row) != detection_fields:
            raise DetectionContractError("detection fields do not match v0.4 schema")
        detection_id = _sha256(row.get("detection_id"), "detection_id")
        if detection_id in ids:
            raise DetectionContractError("duplicate detection_id")
        ids.add(detection_id)
        frame_id = _sha256(row.get("frame_id"), "frame_id")
        _sha256(row.get("model_manifest_sha256"), "model_manifest_sha256")
        rank = row.get("detection_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise DetectionContractError("detection_rank must be non-negative")
        by_frame[frame_id].append(row)

    for frame_id, frame_rows in by_frame.items():
        model_hashes = {
            str(row["model_manifest_sha256"]) for row in frame_rows
        }
        if len(model_hashes) != 1:
            raise DetectionContractError(
                "one frame cannot mix model manifest identities"
            )
        model_hash = next(iter(model_hashes))
        expected_rows = build_detection_rows(
            frame_id=frame_id,
            model_manifest_sha256=model_hash,
            detections=frame_rows,
        )
        actual_rows = sorted(frame_rows, key=lambda row: int(row["detection_rank"]))
        if actual_rows != expected_rows:
            raise DetectionContractError(
                "detection ranks or identities are not canonical"
            )

    seen_frames: set[str] = set()
    for row in selection_rows:
        if set(row) != selection_fields:
            raise DetectionContractError("selection fields do not match v0.4 schema")
        frame_id = _sha256(row.get("frame_id"), "frame_id")
        if frame_id in seen_frames:
            raise DetectionContractError("duplicate primary-hand selection frame")
        seen_frames.add(frame_id)
        expected = build_primary_hand_selection(
            frame_id=frame_id,
            detections=by_frame[frame_id],
            inference_status=row.get("inference_status"),
        )
        if row != expected:
            raise DetectionContractError(
                "selection does not match the frozen v0.4 policy and reason"
            )
    detection_frames = set(by_frame)
    if frame_ids is None:
        if not detection_frames.issubset(seen_frames):
            raise DetectionContractError(
                "selection rows do not cover every detection frame"
            )
    else:
        expected_frames = {_sha256(frame_id, "frame_id") for frame_id in frame_ids}
        if seen_frames != expected_frames or not detection_frames.issubset(
            expected_frames
        ):
            raise DetectionContractError("selection rows do not cover frames exactly")


__all__ = [
    "INFERENCE_STATUSES",
    "PRIMARY_HAND_SELECTION_CONTRACT_VERSION",
    "SELECTION_STATUSES",
    "V04_ASSOCIATION_CONTRACT_VERSION",
    "DetectionContractError",
    "build_detection_rows",
    "build_primary_hand_selection",
    "detection_rows_table",
    "primary_selection_table",
    "stable_payload_detection_id",
    "validate_detection_and_selection_rows",
]
