"""PTS-authoritative, descriptor-pinned offline video decoding.

The decoder owns one no-follow file descriptor for the complete operation.  It
hashes that descriptor before opening FFmpeg, decodes from the same descriptor,
and then rechecks both descriptor identity and content.  Frame timing comes
only from FFmpeg's raw PTS and rational time base; nominal frame rate is never
consulted.
"""

from __future__ import annotations

import hashlib
import os
import stat
import struct
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from ..schema import QualityBits
from .display_transform import (
    DisplayTransformError,
    apply_display_transform,
    display_transform,
    rotation_from_display_matrix,
)
from .frame_identity import (
    FINALIZATION_SCHEMA_VERSION,
    canonical_sha256,
    stable_frame_id,
)

_DECODER_CONFIG_DOMAIN = "tremora-pts-decoder-config-1"
PTS_DECODER_IMPLEMENTATION_VERSION = "tremora-pts-decoder-1.0.0"
_SHA256_LENGTH = 64
_HASH_CHUNK_BYTES = 1024 * 1024
_NANOSECONDS_PER_SECOND = 1_000_000_000


class DecodeError(RuntimeError):
    """Raised when source integrity or deterministic decoding cannot be proven."""


class VerifiedSourceDecodeError(DecodeError):
    """A media-semantic decode rejection after pinned hash verification.

    This subclass is raised only after the source descriptor has matched the
    expected hash and only when the decoder's media/frame interpretation fails.
    Any final descriptor identity or hash failure supersedes it as an ordinary
    :class:`DecodeError` so acquisition/integrity faults cannot be documented as
    stable source-media failures.
    """


class _DocumentableSourceMediaError(DecodeError):
    """Private closed signal produced only by trusted media interpretation."""


@dataclass(frozen=True)
class DecodeConfig:
    """Frozen software-decoder settings that affect finalized frame meaning."""

    stream_index: int = 0
    discontinuity_threshold_ns: int = 1_000_000_000
    output_pixel_format: str = "bgr24"
    thread_count: int = 1
    thread_type: str = "NONE"
    hardware_acceleration: str = "NONE"
    apply_display_rotation: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.stream_index, bool) or not isinstance(
                self.stream_index, int) or self.stream_index < 0:
            raise ValueError("stream_index must be a non-negative integer")
        if isinstance(self.discontinuity_threshold_ns, bool) or not isinstance(
                self.discontinuity_threshold_ns, int) \
                or self.discontinuity_threshold_ns <= 0:
            raise ValueError("discontinuity_threshold_ns must be positive")
        if self.output_pixel_format != "bgr24":
            raise ValueError("output_pixel_format must be 'bgr24' for v0.3")
        if isinstance(self.thread_count, bool) or self.thread_count != 1:
            raise ValueError("thread_count must be 1 for v0.3")
        if self.thread_type != "NONE":
            raise ValueError("thread_type must be 'NONE' for v0.3")
        if self.hardware_acceleration != "NONE":
            raise ValueError("hardware_acceleration must be 'NONE' for v0.3")
        if not isinstance(self.apply_display_rotation, bool) \
                or not self.apply_display_rotation:
            raise ValueError("apply_display_rotation must be True for v0.3")

    @property
    def sha256(self) -> str:
        return canonical_sha256({
            "domain": _DECODER_CONFIG_DOMAIN,
            **asdict(self),
        })


@dataclass(frozen=True)
class RawDecodedFrame:
    """Decoder-emitted evidence before global PTS classification.

    This public intermediate also permits adversarial timing tests to inject
    PTS patterns that ordinary media muxers normalize away.
    """

    decode_ordinal: int
    pts: int | None
    time_base_num: int
    time_base_den: int
    duration_pts: int | None
    coded_width: int
    coded_height: int
    source_pixel_format: str
    key_frame: bool
    picture_type: str
    rotation_degrees: int
    source_to_display_transform: tuple[float, ...]
    display_bgr: np.ndarray = field(repr=False, compare=False)
    decode_status: str = "SUCCESS"
    quality_bits: int = 0


@dataclass(frozen=True)
class DecodedFrame:
    source_video_sha256: str
    stream_index: int
    frame_id: str
    identity_basis: str
    decode_ordinal: int
    presentation_ordinal: int | None
    pts: int | None
    time_base_num: int
    time_base_den: int
    relative_pts_ns: int | None
    same_pts_rank: int
    duration_pts: int | None
    duration_ns: int | None
    gap_before_ns: int | None
    coded_width: int
    coded_height: int
    display_width: int
    display_height: int
    rotation_degrees: int
    pixel_format: str
    key_frame: bool
    picture_type: str
    pts_status: str
    decode_status: str
    quality_bits: int
    source_to_display_transform: tuple[float, ...]
    decoder_version: str
    schema_version: str
    display_bgr: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class DecodedVideo:
    frames: tuple[DecodedFrame, ...]
    source_video_sha256: str
    source_bytes: int
    decoder_version: str
    decoder_config_sha256: str
    stream_index: int


def _required_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        # Keep all invalid timestamp parameters under the public ValueError contract.
        raise ValueError(f"{field_name} must be an integer")  # noqa: TRY004
    return value


def _round_nonnegative_fraction(value: Fraction) -> int:
    if value < 0:
        raise ValueError("frame precedes selected presentation origin")
    return (value.numerator * 2 + value.denominator) // (
        2 * value.denominator)


def relative_pts_ns(
    pts: int,
    first_pts: int,
    time_base_num: int,
    time_base_den: int,
) -> int:
    """Convert relative raw presentation time to exact integer nanoseconds."""

    pts = _required_integer(pts, "pts")
    first_pts = _required_integer(first_pts, "first_pts")
    time_base_num = _required_integer(time_base_num, "time_base_num")
    time_base_den = _required_integer(time_base_den, "time_base_den")
    if time_base_num <= 0:
        raise ValueError("time_base_num must be positive")
    if time_base_den <= 0:
        raise ValueError("time_base_den must be positive")
    exact_ns = Fraction(
        (pts - first_pts) * time_base_num * _NANOSECONDS_PER_SECOND,
        time_base_den,
    )
    return _round_nonnegative_fraction(exact_ns)


def _fraction_ns(value: Fraction) -> int:
    return _round_nonnegative_fraction(value * _NANOSECONDS_PER_SECOND)


def _timestamp(frame: RawDecodedFrame) -> Fraction | None:
    if frame.pts is None:
        return None
    return Fraction(frame.pts * frame.time_base_num, frame.time_base_den)


def _validate_raw_frame(frame: RawDecodedFrame, expected_ordinal: int) -> None:
    if not isinstance(frame, RawDecodedFrame):
        raise DecodeError("raw frame sequence contains an invalid value")
    if isinstance(frame.decode_ordinal, bool) or not isinstance(
            frame.decode_ordinal, int) or frame.decode_ordinal != expected_ordinal:
        raise DecodeError("decode ordinals must be contiguous decoder-emission order")
    if frame.pts is not None and (
        isinstance(frame.pts, bool) or not isinstance(frame.pts, int)
    ):
        raise DecodeError("frame PTS must be an integer or null")
    if isinstance(frame.time_base_num, bool) or not isinstance(
            frame.time_base_num, int) or frame.time_base_num <= 0:
        raise DecodeError("frame time-base numerator must be positive")
    if isinstance(frame.time_base_den, bool) or not isinstance(
            frame.time_base_den, int) or frame.time_base_den <= 0:
        raise DecodeError("frame time-base denominator must be positive")
    if frame.duration_pts is not None and (
        isinstance(frame.duration_pts, bool)
        or not isinstance(frame.duration_pts, int)
        or frame.duration_pts < 0
    ):
        raise DecodeError("frame duration must be null or non-negative")
    if isinstance(frame.coded_width, bool) or not isinstance(
            frame.coded_width, int) or frame.coded_width <= 0 \
            or isinstance(frame.coded_height, bool) or not isinstance(
                frame.coded_height, int) or frame.coded_height <= 0:
        raise DecodeError("decoded coded dimensions must be positive")
    if not isinstance(frame.source_pixel_format, str) \
            or not frame.source_pixel_format:
        raise DecodeError("decoded source pixel format is missing")
    if not isinstance(frame.key_frame, bool):
        raise DecodeError("decoded key-frame status must be boolean")
    if not isinstance(frame.picture_type, str) or not frame.picture_type:
        raise DecodeError("decoded picture type is missing")
    if frame.rotation_degrees not in {0, 90, 180, 270}:
        raise DecodeError("decoded display rotation must be a right angle")
    if frame.decode_status not in {"SUCCESS", "CORRUPT"}:
        raise DecodeError("decoded frame status must be SUCCESS or CORRUPT")
    if isinstance(frame.quality_bits, bool) or not isinstance(
            frame.quality_bits, int) or not 0 <= frame.quality_bits <= 0xFFFF_FFFF:
        raise DecodeError("decoded quality bits must fit uint32")
    try:
        transform_size = len(frame.source_to_display_transform)
    except TypeError as exc:
        raise DecodeError("source-to-display transform must be a sequence") from exc
    if transform_size != 9:
        raise DecodeError("source-to-display transform must be 3x3")
    try:
        finite_transform = all(
            np.isfinite(value) for value in frame.source_to_display_transform)
    except TypeError as exc:
        raise DecodeError("source-to-display transform must be numeric") from exc
    if not finite_transform:
        raise DecodeError("source-to-display transform must be finite")
    expected_transform = display_transform(
        frame.coded_width,
        frame.coded_height,
        frame.rotation_degrees,
    )
    if not np.allclose(
        frame.source_to_display_transform,
        expected_transform.source_to_display,
        rtol=0.0,
        atol=0.0,
    ):
        raise DecodeError("source-to-display transform contradicts frame geometry")
    image = frame.display_bgr
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 \
            or image.ndim != 3 or image.shape[2] != 3:
        raise DecodeError("display frame must be an HxWx3 uint8 array")
    if not image.flags.c_contiguous:
        raise DecodeError("display frame must be contiguous")
    image.setflags(write=False)
    expected_height = frame.coded_height if frame.rotation_degrees in {0, 180} \
        else frame.coded_width
    expected_width = frame.coded_width if frame.rotation_degrees in {0, 180} \
        else frame.coded_height
    if image.shape[:2] != (expected_height, expected_width):
        raise DecodeError("display frame dimensions do not match its transform")


def classify_decoded_frames(
    raw_frames: Sequence[RawDecodedFrame],
    *,
    source_video_sha256: str,
    stream_index: int,
    config: DecodeConfig,
    decoder_version: str = "synthetic-injected-decoder-1",
) -> tuple[DecodedFrame, ...]:
    """Bind frame IDs and classify PTS evidence without rewriting timestamps."""

    if not isinstance(config, DecodeConfig):
        raise DecodeError("config must be a DecodeConfig")
    _validate_sha256(source_video_sha256, field_name="source_video_sha256")
    if stream_index != config.stream_index:
        raise DecodeError("classification stream index does not match decoder config")
    if not isinstance(decoder_version, str) or not decoder_version:
        raise DecodeError("decoder_version must be a non-empty string")
    raw = tuple(raw_frames)
    for ordinal, frame in enumerate(raw):
        _validate_raw_frame(frame, ordinal)

    pts_counts = Counter(frame.pts for frame in raw if frame.pts is not None)
    ranks: dict[int, int] = defaultdict(int)
    same_pts_ranks: dict[int, int] = {}
    for frame in raw:
        if frame.pts is None:
            same_pts_ranks[frame.decode_ordinal] = 0
            continue
        same_pts_ranks[frame.decode_ordinal] = ranks[frame.pts]
        ranks[frame.pts] += 1

    timestamps = {
        frame.decode_ordinal: _timestamp(frame) for frame in raw
    }
    presentation = sorted(
        (frame for frame in raw if frame.pts is not None),
        key=lambda frame: (
            timestamps[frame.decode_ordinal],
            same_pts_ranks[frame.decode_ordinal],
            frame.decode_ordinal,
        ),
    )
    presentation_ordinals = {
        frame.decode_ordinal: ordinal for ordinal, frame in enumerate(presentation)
    }
    origin = timestamps[presentation[0].decode_ordinal] if presentation else None

    gap_before: dict[int, int | None] = {}
    discontinuities: set[int] = set()
    previous_time: Fraction | None = None
    for frame in presentation:
        current_time = timestamps[frame.decode_ordinal]
        assert current_time is not None
        if previous_time is None:
            gap_before[frame.decode_ordinal] = None
        else:
            gap_ns = _fraction_ns(current_time - previous_time)
            gap_before[frame.decode_ordinal] = gap_ns
            if gap_ns > config.discontinuity_threshold_ns:
                discontinuities.add(frame.decode_ordinal)
        previous_time = current_time

    nonmonotonic: set[int] = set()
    previous_emitted_time: Fraction | None = None
    for frame in raw:
        current_time = timestamps[frame.decode_ordinal]
        if current_time is None:
            continue
        if previous_emitted_time is not None and current_time < previous_emitted_time:
            nonmonotonic.add(frame.decode_ordinal)
        previous_emitted_time = current_time

    decoded: list[DecodedFrame] = []
    for frame in raw:
        ordinal = frame.decode_ordinal
        rank = same_pts_ranks[ordinal]
        frame_id, identity_basis = stable_frame_id(
            source_video_sha256=source_video_sha256,
            stream_index=stream_index,
            pts=frame.pts,
            same_pts_rank=rank,
            decode_ordinal=ordinal,
        )
        bits = int(frame.quality_bits)
        if frame.pts is None:
            bits |= int(QualityBits.MISSING_TIMESTAMP)
        if frame.pts is not None and pts_counts[frame.pts] > 1:
            bits |= int(QualityBits.DUPLICATE_TIMESTAMP)
        if ordinal in nonmonotonic:
            bits |= int(QualityBits.NON_MONOTONIC_TIMESTAMP)
        if ordinal in discontinuities:
            bits |= int(QualityBits.STREAM_GAP)
        if frame.decode_status != "SUCCESS":
            bits |= int(QualityBits.DECODE_FAILURE)

        if frame.pts is None:
            pts_status = "MISSING"
        elif pts_counts[frame.pts] > 1:
            pts_status = "DUPLICATE"
        elif ordinal in nonmonotonic:
            pts_status = "NON_MONOTONIC"
        elif ordinal in discontinuities:
            pts_status = "DISCONTINUITY"
        else:
            pts_status = "VALID"

        timestamp = timestamps[ordinal]
        relative_ns = None if timestamp is None or origin is None else _fraction_ns(
            timestamp - origin)
        duration_ns = None
        if frame.duration_pts is not None:
            duration_ns = _fraction_ns(Fraction(
                frame.duration_pts * frame.time_base_num,
                frame.time_base_den,
            ))
        display_height, display_width = frame.display_bgr.shape[:2]
        decoded.append(DecodedFrame(
            source_video_sha256=source_video_sha256,
            stream_index=stream_index,
            frame_id=frame_id,
            identity_basis=identity_basis,
            decode_ordinal=ordinal,
            presentation_ordinal=presentation_ordinals.get(ordinal),
            pts=frame.pts,
            time_base_num=frame.time_base_num,
            time_base_den=frame.time_base_den,
            relative_pts_ns=relative_ns,
            same_pts_rank=rank,
            duration_pts=frame.duration_pts,
            duration_ns=duration_ns,
            gap_before_ns=gap_before.get(ordinal),
            coded_width=frame.coded_width,
            coded_height=frame.coded_height,
            display_width=display_width,
            display_height=display_height,
            rotation_degrees=frame.rotation_degrees,
            pixel_format=frame.source_pixel_format,
            key_frame=frame.key_frame,
            picture_type=frame.picture_type,
            pts_status=pts_status,
            decode_status=frame.decode_status,
            quality_bits=bits,
            source_to_display_transform=frame.source_to_display_transform,
            decoder_version=decoder_version,
            schema_version=FINALIZATION_SCHEMA_VERSION,
            display_bgr=frame.display_bgr,
        ))
    return tuple(decoded)


def _read_at(descriptor: int, size: int, offset: int) -> bytes:
    pread = getattr(os, "pread", None)
    if pread is not None:
        return pread(descriptor, size, offset)
    prior_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, offset, os.SEEK_SET)
        return os.read(descriptor, size)
    finally:
        os.lseek(descriptor, prior_offset, os.SEEK_SET)


def _sha256_descriptor(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = _read_at(
            descriptor,
            min(_HASH_CHUNK_BYTES, size - offset),
            offset,
        )
        if not chunk:
            raise DecodeError("source descriptor ended before its declared size")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_sha256(
    value: object,
    *,
    field_name: str = "expected_source_sha256",
) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH \
            or any(character not in "0123456789abcdef" for character in value):
        raise DecodeError(f"{field_name} must be a lowercase SHA-256")
    return value


def _decoder_version(av_module: Any) -> str:
    try:
        major = int(str(av_module.__version__).split(".", 1)[0])
    except (AttributeError, TypeError, ValueError) as exc:
        raise DecodeError("PyAV version cannot be identified") from exc
    if major != 18:
        raise DecodeError("VIDIMU v0.3 requires PyAV major version 18")
    libraries = ",".join(
        f"{name}={'.'.join(str(part) for part in version)}"
        for name, version in sorted(av_module.library_versions.items())
    )
    return (
        f"{PTS_DECODER_IMPLEMENTATION_VERSION};"
        f"pyav-{av_module.__version__};{libraries}"
    )


def _pure_display_rotation(frame: Any) -> int:
    matrices = [
        side_data for side_data in frame.side_data
        if getattr(getattr(side_data, "type", None), "name", None)
        == "DISPLAYMATRIX"
    ]
    if not matrices:
        reported = int(getattr(frame, "rotation", 0) or 0) % 360
        if reported != 0:
            raise DecodeError(
                "frame reports display rotation without DISPLAYMATRIX evidence")
        return 0
    if len(matrices) != 1:
        raise DecodeError("frame contains multiple display matrices")
    try:
        payload = bytes(memoryview(matrices[0]))
    except (TypeError, ValueError, BufferError) as exc:
        raise DecodeError("cannot read display matrix side data") from exc
    if len(payload) != 36:
        raise DecodeError("display matrix side data has invalid length")
    values = struct.unpack("=9i", payload)
    if values[2] != 0 or values[5] != 0 or values[6] != 0 \
            or values[7] != 0 or values[8] != 1 << 30:
        raise DecodeError("display matrix is not a pure rotation")
    try:
        rotation = rotation_from_display_matrix(payload)
    except DisplayTransformError as exc:
        raise DecodeError("display matrix is not a pure right-angle rotation") from exc
    reported = int(getattr(frame, "rotation", rotation) or 0) % 360
    if reported != rotation:
        raise DecodeError("frame rotation contradicts DISPLAYMATRIX evidence")
    return rotation


def _picture_type(av_module: Any, value: object) -> str:
    try:
        return av_module.video.frame.PictureType(int(value)).name
    except (TypeError, ValueError):
        return "UNKNOWN"


def _is_recognized_pyav_media_rejection(av_module: Any, exc: BaseException) -> bool:
    """Recognize only PyAV invalid-data/end-of-input media exceptions."""

    candidates: list[type[BaseException]] = []
    error_module = getattr(av_module, "error", None)
    for owner in (av_module, error_module):
        if owner is None:
            continue
        for name in ("InvalidDataError", "EOFError"):
            candidate = getattr(owner, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                candidates.append(candidate)
    return bool(candidates) and isinstance(exc, tuple(candidates))


class PTSDecoder:
    """Decode one source path from one securely pinned descriptor."""

    def __init__(self, config: DecodeConfig | None = None) -> None:
        self.config = config if config is not None else DecodeConfig()
        if not isinstance(self.config, DecodeConfig):
            raise TypeError("config must be a DecodeConfig")

    @staticmethod
    def _load_av() -> Any:
        try:
            import av
        except ImportError as exc:
            raise DecodeError(
                "PyAV 18 is required for PTS-authoritative decoding") from exc
        _decoder_version(av)
        return av

    @property
    def decoder_version(self) -> str:
        return _decoder_version(self._load_av())

    def _decode_handle(
        self,
        handle: BinaryIO,
        *,
        av_module: Any,
        source_video_sha256: str,
    ) -> tuple[DecodedFrame, ...]:
        raw: list[RawDecodedFrame] = []
        try:
            handle.seek(0)
            with av_module.open(handle, mode="r") as container:
                video_streams = tuple(
                    candidate for candidate in container.streams
                    if candidate.type == "video"
                )
                if not video_streams:
                    raise _DocumentableSourceMediaError(
                        "source contains no video stream")
                stream = next((candidate for candidate in video_streams
                    if candidate.index == self.config.stream_index), None)
                if stream is None:
                    raise DecodeError(
                        "configured absolute stream index is not a video stream")
                stream.codec_context.thread_count = self.config.thread_count
                stream.codec_context.thread_type = self.config.thread_type
                stream.codec_context.skip_frame = "NONE"

                for decode_ordinal, frame in enumerate(container.decode(stream)):
                    time_base = frame.time_base or stream.time_base
                    if time_base is None:
                        raise _DocumentableSourceMediaError(
                            "decoded frame has no rational time base")
                    time_base_num = int(time_base.numerator)
                    time_base_den = int(time_base.denominator)
                    try:
                        rotation = _pure_display_rotation(frame)
                        transform = display_transform(
                            int(frame.width), int(frame.height), rotation)
                        source_pixels = frame.to_ndarray(
                            format=self.config.output_pixel_format)
                        display_pixels = apply_display_transform(
                            source_pixels, transform)
                    except MemoryError:
                        raise
                    except (DecodeError, DisplayTransformError) as exc:
                        raise _DocumentableSourceMediaError(
                            "decoded frame metadata violates the display contract"
                        ) from exc
                    display_pixels.setflags(write=False)
                    source_format = getattr(frame.format, "name", None)
                    if not source_format:
                        raise _DocumentableSourceMediaError(
                            "decoded frame has no source pixel format")
                    decode_status = "CORRUPT" if bool(frame.is_corrupt) else "SUCCESS"
                    raw.append(RawDecodedFrame(
                        decode_ordinal=decode_ordinal,
                        pts=None if frame.pts is None else int(frame.pts),
                        time_base_num=time_base_num,
                        time_base_den=time_base_den,
                        duration_pts=(
                            None if frame.duration is None else int(frame.duration)
                        ),
                        coded_width=int(frame.width),
                        coded_height=int(frame.height),
                        source_pixel_format=str(source_format),
                        key_frame=bool(frame.key_frame),
                        picture_type=_picture_type(av_module, frame.pict_type),
                        rotation_degrees=rotation,
                        source_to_display_transform=transform.source_to_display,
                        display_bgr=display_pixels,
                        decode_status=decode_status,
                    ))
        except _DocumentableSourceMediaError:
            raise
        except MemoryError:
            raise
        except Exception as exc:
            if _is_recognized_pyav_media_rejection(av_module, exc):
                raise _DocumentableSourceMediaError(
                    "FFmpeg rejected source media bytes") from exc
            if isinstance(exc, DecodeError):
                raise
            if isinstance(exc, OSError):
                raise DecodeError("video decode encountered an I/O failure") from exc
            raise DecodeError("unexpected video decoder failure") from exc
        if not raw:
            raise _DocumentableSourceMediaError(
                "source video emitted no decoded frames")
        try:
            return classify_decoded_frames(
                raw,
                source_video_sha256=source_video_sha256,
                stream_index=self.config.stream_index,
                config=self.config,
                decoder_version=_decoder_version(av_module),
            )
        except MemoryError:
            raise
        except DecodeError as exc:
            raise _DocumentableSourceMediaError(
                "decoded source metadata violates the frame contract") from exc

    def decode(
        self,
        source_path: str | os.PathLike[str],
        *,
        expected_source_sha256: str,
    ) -> DecodedVideo:
        """Decode ``source_path`` after binding it to one expected content hash."""

        expected_hash = _validate_sha256(expected_source_sha256)
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
            raise DecodeError(
                "secure decode requires O_NOFOLLOW and O_NONBLOCK support")
        av_module = self._load_av()
        path = Path(source_path)
        flags = os.O_RDONLY | os.O_NOFOLLOW
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= os.O_NONBLOCK
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise DecodeError("cannot securely open source video") from exc

        handle: BinaryIO | None = None
        try:
            try:
                before = os.fstat(descriptor)
            except OSError as exc:
                raise DecodeError("cannot inspect source video") from exc
            if not stat.S_ISREG(before.st_mode):
                raise DecodeError("source video must be a regular file")
            try:
                initial_hash = _sha256_descriptor(descriptor, before.st_size)
                after_initial_hash = os.fstat(descriptor)
            except OSError as exc:
                raise DecodeError("cannot hash source video") from exc
            if _stat_identity(before) != _stat_identity(after_initial_hash):
                raise DecodeError("source video changed during initial hash")
            if initial_hash != expected_hash:
                raise DecodeError("source video hash mismatch")

            handle = os.fdopen(descriptor, "rb", buffering=0)
            descriptor = -1
            frames: tuple[DecodedFrame, ...] | None = None
            verified_source_decode_error: _DocumentableSourceMediaError | None = None
            try:
                try:
                    frames = self._decode_handle(
                        handle,
                        av_module=av_module,
                        source_video_sha256=initial_hash,
                    )
                except _DocumentableSourceMediaError as exc:
                    verified_source_decode_error = exc
            finally:
                try:
                    before_final_hash = os.fstat(handle.fileno())
                    final_hash = _sha256_descriptor(
                        handle.fileno(), before_final_hash.st_size)
                    after_final_hash = os.fstat(handle.fileno())
                except OSError as exc:
                    raise DecodeError(
                        "cannot revalidate decoded source video") from exc
                identities = {
                    _stat_identity(before),
                    _stat_identity(before_final_hash),
                    _stat_identity(after_final_hash),
                }
                if len(identities) != 1 or final_hash != initial_hash:
                    raise DecodeError("source video changed during pinned decode")
            if verified_source_decode_error is not None:
                raise VerifiedSourceDecodeError(
                    "hash-verified source media decode failed"
                ) from verified_source_decode_error
            assert frames is not None
            return DecodedVideo(
                frames=frames,
                source_video_sha256=initial_hash,
                source_bytes=before.st_size,
                decoder_version=_decoder_version(av_module),
                decoder_config_sha256=self.config.sha256,
                stream_index=self.config.stream_index,
            )
        finally:
            if handle is not None:
                handle.close()
            elif descriptor >= 0:
                os.close(descriptor)
