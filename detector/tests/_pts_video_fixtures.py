"""Generated media fixtures for the PTS-preserving decoder contract.

The files are created in each test's temporary directory and are never checked
into the repository.  FFmpeg/PyAV are hard requirements for this Gate-A suite:
missing tooling is a test failure, not an optional-platform skip.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path


class MediaFixtureError(RuntimeError):
    """Raised when a required synthetic media fixture cannot be generated."""


def require_pts_media_toolchain() -> None:
    missing = [
        executable
        for executable in ("ffmpeg", "ffprobe")
        if shutil.which(executable) is None
    ]
    if importlib.util.find_spec("av") is None:
        missing.append("Python package av")
    if missing:
        raise AssertionError(
            "Gate-A PTS tests require the media toolchain; missing: "
            + ", ".join(missing)
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg(*arguments: str) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        *arguments,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise MediaFixtureError(
            f"FFmpeg fixture command failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stderr.strip()}"
        )


def _ffprobe(
    path: Path,
    *,
    section: str,
    entries: str,
) -> list[dict[str, object]]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        f"-show_{section}",
        "-show_entries",
        f"{section[:-1]}={entries}",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise MediaFixtureError(
            f"FFprobe fixture validation failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise MediaFixtureError("FFprobe returned malformed JSON") from exc
    records = payload.get(section)
    if not isinstance(records, list) or not records:
        raise MediaFixtureError(f"FFprobe returned no video {section}")
    if not all(isinstance(record, dict) for record in records):
        raise MediaFixtureError(f"FFprobe returned malformed video {section}")
    return records


def ffprobe_frames(path: Path) -> list[dict[str, object]]:
    return _ffprobe(
        path,
        section="frames",
        entries="pts,pts_time,duration,pict_type,key_frame",
    )


def ffprobe_packets(path: Path) -> list[dict[str, object]]:
    return _ffprobe(
        path,
        section="packets",
        entries="pts,pts_time,dts,dts_time,duration,flags",
    )


def _encoded_video(
    output: Path,
    *,
    rate: int,
    duration: float,
    video_filter: str | None = None,
    b_frames: int = 0,
    faststart: bool = False,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    arguments = [
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=64x48:rate={rate}:duration={duration}",
    ]
    if video_filter is not None:
        arguments.extend(("-vf", video_filter, "-fps_mode", "vfr"))
    arguments.extend(
        (
            "-an",
            "-c:v",
            "mpeg4",
            "-threads",
            "1",
            "-bf",
            str(b_frames),
            "-g",
            "30",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            "1000",
        )
    )
    if faststart:
        arguments.extend(("-movflags", "+faststart"))
    arguments.extend(("-y", str(output)))
    _ffmpeg(*arguments)
    return output


def generate_cfr_video(root: Path) -> Path:
    return _encoded_video(root / "cfr_no_b_frames.mp4", rate=5, duration=1.0)


def generate_encoded_marker_video(root: Path) -> Path:
    """Encode blank, one-marker, and two-marker frames losslessly."""

    output = root / "blank_one_two_markers.mkv"
    output.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=black:size=64x48:rate=1:duration=3",
        "-vf",
        (
            "drawbox=x=8:y=8:w=10:h=10:color=white:t=fill:"
            "enable='gte(n,1)',"
            "drawbox=x=42:y=28:w=10:h=10:color=white:t=fill:"
            "enable='eq(n,2)'"
        ),
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-threads",
        "1",
        "-pix_fmt",
        "bgr0",
        "-y",
        str(output),
    )
    frames = ffprobe_frames(output)
    if [frame.get("pts") for frame in frames] != [0, 1000, 2000]:
        raise MediaFixtureError(
            "FFmpeg changed the encoded-marker fixture presentation order"
        )
    return output


def generate_vfr_video(root: Path) -> Path:
    # Six input frames receive presentation times 0, .1, .4, .9, 1.6,
    # and 2.5 seconds.  This is deliberately incompatible with one nominal
    # frame interval.
    return _encoded_video(
        root / "vfr.mp4",
        rate=10,
        duration=0.6,
        video_filter="setpts='N*N/(10*TB)'",
    )


def generate_b_frame_video(root: Path) -> Path:
    return _encoded_video(
        root / "b_frames.mp4",
        rate=6,
        duration=1.0,
        b_frames=2,
    )


def generate_missing_pts_h264(root: Path) -> Path:
    output = root / "missing_pts_elementary.h264"
    output.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=5:duration=1",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-preset",
        "ultrafast",
        "-bf",
        "0",
        "-g",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "h264",
        "-y",
        str(output),
    )
    frames = ffprobe_frames(output)
    if any(frame.get("pts") is not None for frame in frames):
        raise MediaFixtureError(
            "FFmpeg normalized the missing-PTS elementary-stream fixture"
        )
    return output


def generate_duplicate_pts_b_frame_ts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = root / "b_frame_base.ts"
    duplicate = root / "b_frame_duplicate_pts.ts"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=6:duration=1",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-preset",
        "medium",
        "-bf",
        "2",
        "-g",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "mpegts",
        "-y",
        str(base),
    )
    # Packet two is assigned the preceding output PTS while DTS remains in
    # codec order.  With B pictures, the duplicated packet timestamps emerge
    # on two distinct decoded frames rather than adjacent decoder emissions.
    _ffmpeg(
        "-i",
        str(base),
        "-map",
        "0:v:0",
        "-an",
        "-c",
        "copy",
        "-bsf:v",
        "setts=pts='if(eq(N,2),PREV_OUTPTS,PTS)'",
        "-f",
        "mpegts",
        "-y",
        str(duplicate),
    )
    packets = ffprobe_packets(duplicate)
    frames = ffprobe_frames(duplicate)
    packet_pts = [packet.get("pts") for packet in packets]
    frame_pts = [frame.get("pts") for frame in frames]
    if len(packet_pts) == len(set(packet_pts)):
        raise MediaFixtureError(
            "FFmpeg normalized duplicate PTS out of the packet fixture"
        )
    if len(frame_pts) == len(set(frame_pts)):
        raise MediaFixtureError(
            "FFmpeg did not propagate duplicate packet PTS to decoded frames"
        )
    if "B" not in {frame.get("pict_type") for frame in frames}:
        raise MediaFixtureError("duplicate-PTS fixture contains no decoded B picture")
    return duplicate


def generate_pts_discontinuity_video(root: Path) -> Path:
    output = _encoded_video(
        root / "pts_discontinuity.mp4",
        rate=5,
        duration=1.0,
        video_filter="setpts='if(gte(N,3),PTS+5/TB,PTS)'",
    )
    pts = [frame.get("pts") for frame in ffprobe_frames(output)]
    if pts != [0, 200, 400, 5600, 5800]:
        raise MediaFixtureError(
            f"FFmpeg normalized the PTS-discontinuity fixture: {pts!r}"
        )
    return output


def generate_nonzero_pts_video(root: Path) -> Path:
    # The MP4 stream starts at raw PTS 5000 with time base 1/1000.  ffmpeg's
    # VFR output mode preserves the explicit five-second offset.
    return _encoded_video(
        root / "nonzero_start.mp4",
        rate=5,
        duration=1.0,
        video_filter="setpts=PTS+5/TB",
    )


def generate_rotated_video(root: Path) -> Path:
    base = generate_cfr_video(root)
    rotated = root / "rotated_90_ccw.mp4"
    # Applying display metadata during a stream copy avoids rotating coded
    # pixels.  The decoder must therefore observe 64x48 coded pixels and an
    # independent 90-degree display matrix.
    _ffmpeg(
        "-display_rotation",
        "90",
        "-i",
        str(base),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-y",
        str(rotated),
    )
    return rotated


def generate_damaged_video(root: Path) -> Path:
    intact = _encoded_video(
        root / "intact_faststart.mp4",
        rate=5,
        duration=1.0,
        faststart=True,
    )
    payload = intact.read_bytes()
    damaged = root / "damaged_truncated.mp4"
    # Retain an MP4-looking prefix while cutting both metadata and media data.
    # This prevents the fixture from degrading into a mere nonexistent-path
    # check while still guaranteeing that it cannot be fully decoded.
    damaged.write_bytes(payload[: max(32, min(96, len(payload) // 4))])
    return damaged
