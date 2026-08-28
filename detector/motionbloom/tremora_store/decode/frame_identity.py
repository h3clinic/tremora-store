"""Domain-separated identities for decoded frames and finalization runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

ASSOCIATION_SCHEMA_VERSION = "tremora-pose-frame-association-1.0.0"
FINALIZATION_SCHEMA_VERSION = "0.3.0"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FRAME_ID_DOMAIN = "tremora-video-frame-1"
_FINALIZATION_ID_DOMAIN = "tremora-pts-cv-finalization-1"


class IdentityError(ValueError):
    """Raised when identity inputs are incomplete or ambiguous."""


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise IdentityError(f"{field} must be a lowercase SHA-256")
    return value


def canonical_json_bytes(payload: object) -> bytes:
    """Encode JSON with the one canonical representation used for identities."""

    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise IdentityError("identity payload is not canonical JSON") from exc


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_frame_id(
    *,
    source_video_sha256: str,
    stream_index: int,
    pts: int | None,
    same_pts_rank: int,
    decode_ordinal: int | None = None,
) -> tuple[str, str]:
    """Return ``(frame_id, identity_basis)`` for one decoder-emitted frame.

    Valid PTS frames use only the source hash, stream index, raw PTS, and the
    rank among frames with that PTS. A missing-PTS frame cannot use that key, so
    it enters an explicit fallback namespace bound to the frozen decoder's
    emission ordinal. Such a frame remains ineligible for temporal analysis.
    """

    source_hash = _sha256(source_video_sha256, "source_video_sha256")
    if isinstance(stream_index, bool) or not isinstance(stream_index, int) \
            or stream_index < 0:
        raise IdentityError("stream_index must be a non-negative integer")
    if isinstance(same_pts_rank, bool) or not isinstance(same_pts_rank, int) \
            or same_pts_rank < 0:
        raise IdentityError("same_pts_rank must be a non-negative integer")

    if pts is None:
        if isinstance(decode_ordinal, bool) or not isinstance(
                decode_ordinal, int) or decode_ordinal < 0:
            raise IdentityError(
                "missing PTS identity requires a non-negative decode_ordinal")
        basis = "MISSING_PTS_DECODE_ORDINAL"
        key: dict[str, object] = {
            "decode_ordinal": decode_ordinal,
            "domain": _FRAME_ID_DOMAIN,
            "identity_basis": basis,
            "source_video_sha256": source_hash,
            "stream_index": stream_index,
        }
    else:
        if isinstance(pts, bool) or not isinstance(pts, int):
            raise IdentityError("pts must be an integer or null")
        basis = "SOURCE_PTS_SAME_PTS_RANK"
        key = {
            "domain": _FRAME_ID_DOMAIN,
            "identity_basis": basis,
            "pts": pts,
            "same_pts_rank": same_pts_rank,
            "source_video_sha256": source_hash,
            "stream_index": stream_index,
        }
    return canonical_sha256(key), basis


def finalization_identity(payload: Mapping[str, Any]) -> str:
    """Hash all inputs that can change decoded/CV artifact meaning."""

    required = {
        "source_video_sha256",
        "decoder_version",
        "decoder_config_sha256",
        "model_id",
        "model_weights_sha256",
        "preprocessing_config_sha256",
        "inference_environment_id",
        "association_schema_version",
        "finalization_schema_version",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        missing = sorted(required.difference(payload)) if isinstance(
            payload, Mapping) else sorted(required)
        extra = sorted(set(payload).difference(required)) if isinstance(
            payload, Mapping) else []
        raise IdentityError(
            f"finalization identity fields mismatch; missing={missing}, extra={extra}")
    if payload["association_schema_version"] != ASSOCIATION_SCHEMA_VERSION:
        raise IdentityError("association schema version is not supported")
    if payload["finalization_schema_version"] != FINALIZATION_SCHEMA_VERSION:
        raise IdentityError("finalization schema version is not supported")
    _sha256(payload["source_video_sha256"], "source_video_sha256")
    _sha256(payload["decoder_config_sha256"], "decoder_config_sha256")
    _sha256(payload["model_weights_sha256"], "model_weights_sha256")
    _sha256(
        payload["preprocessing_config_sha256"],
        "preprocessing_config_sha256",
    )
    for field in (
        "decoder_version", "model_id", "inference_environment_id",
    ):
        if not isinstance(payload[field], str) or not payload[field]:
            raise IdentityError(f"{field} must be a non-empty string")
    return canonical_sha256({"domain": _FINALIZATION_ID_DOMAIN, **dict(payload)})
