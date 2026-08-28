"""PTS-authoritative video decoding for offline Tremora finalization."""

from .frame_identity import (
    ASSOCIATION_SCHEMA_VERSION,
    FINALIZATION_SCHEMA_VERSION,
    canonical_sha256,
    finalization_identity,
    stable_frame_id,
)
from .pts_decoder import (
    PTS_DECODER_IMPLEMENTATION_VERSION,
    DecodeConfig,
    DecodedFrame,
    DecodeError,
    PTSDecoder,
    VerifiedSourceDecodeError,
)

__all__ = [
    "ASSOCIATION_SCHEMA_VERSION",
    "FINALIZATION_SCHEMA_VERSION",
    "PTS_DECODER_IMPLEMENTATION_VERSION",
    "DecodeConfig",
    "DecodeError",
    "DecodedFrame",
    "PTSDecoder",
    "VerifiedSourceDecodeError",
    "canonical_sha256",
    "finalization_identity",
    "stable_frame_id",
]
