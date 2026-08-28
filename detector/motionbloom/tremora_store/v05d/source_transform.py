"""Replay VIDIMU's published positional trims without inventing time.

The upstream estimator compares an IK ``.mot`` signal with a BodyTrack CSV
signal on a nominal 30 Hz grid.  Its modifier then removes positional rows
from CSV, MOT, and RAW files.  This module binds and reproduces that exact
source procedure.  It also audits the crucial distinction between structural
five-sensor RAW polling groups and source-authored 50 Hz STO/MOT ordinals.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path, PurePosixPath, PureWindowsPath

from .authority import (
    AMBIGUOUS_RECORDING_IDS,
    ESTIMATE_NOTEBOOK_SHA256,
    MODIFY_NOTEBOOK_SHA256,
    SOURCE_TOOLS_COMMIT,
    SYNC_UTILITY_SHA256,
    V05_AUTHORITY_REPORT_SHA256,
    V05_AUTHORITY_SCRIPT_SHA256,
)

EXPECTED_SNAPSHOT_SHA256 = (
    "a6e2194aee5478718e6f92cf9306214e361b08bb61363998f1e6e59e7378f1eb"
)
EXPECTED_SOURCE_INVENTORY_SHA256 = (
    "9f328fec92c9921733ba94fd74602bdb6e1edb99ff27ab3025191a6889123b08"
)
EXPECTED_DATASET_ARCHIVE_SHA256 = (
    "eff12be2f1c5a0cc7389726c754ea1c4ab19d8ca49c227b47344109cbf927841"
)
EXPECTED_ANALYSIS_ARCHIVE_SHA256 = (
    "3696e2b8dd211b6aedf35f4ca635190e5ee24a545da69847c7c3f6269a2bf64c"
)
EXPECTED_TOOLS_ARCHIVE_SHA256 = (
    "bc8820d5f79fcc2e58233474ed15aa2191faf2463e82b3d4d1ddacf44eaf9e27"
)
EXPECTED_INFO_TO_SYNC_SHA256 = (
    "4596803e90e2908717ab227846f7a1c10a5c6b29c9bcbddb90e940a7259c519f"
)
EXPECTED_FILE_PROCESSING_SHA256 = (
    "2b534daa9887934824e0034ce7af42414ab1884743a581a598300949693f4331"
)
EXPECTED_SIGNAL_PROCESSING_SHA256 = (
    "88e4701b71e7b7c048f64f1698220d22ee0887b97624aa2c38fae8d37861c591"
)

EXPECTED_RECORDINGS = 208
EXPECTED_INFO_ROWS = 366
EXPECTED_NON_MP4_INSTRUCTIONS = 217
EXPECTED_OVERRIDE_RECORDINGS = 181
EXPECTED_RAW_ROWS_INCLUDING_NPOSE = 10_184_045
EXPECTED_RAW_DYNAMIC_ROWS = 10_183_005
EXPECTED_RAW_POLL_GROUPS = 2_036_601
EXPECTED_STO_MOT_ROWS_INCLUDING_NPOSE = 299_919
EXPECTED_STO_MOT_DYNAMIC_ROWS = 299_711

_INFO_FIELDS = (
    "Subject",
    "Activity",
    "Trial",
    "File",
    "Type",
    "CutFrames",
    "OrigRmse",
    "TheoRmse",
)
_INFO_TYPE_COUNTS = {"csv": 149, "mot": 34, "mp4": 149, "raw": 34}
_OVERRIDE_TYPE_COUNTS = {"csv": 149, "mot": 34, "raw": 34}
_WINNER_COUNTS = {"IMU": 32, "VIDEO": 149, "ZERO": 27}
_CLASSIFICATION_COUNTS = {"AMBIGUOUS": 2, "IMU": 32, "VIDEO": 147, "ZERO": 27}
_LOWER_BODY_LAYOUT = ("qsHIPS", "qsRUL", "qsRLL", "qsLUL", "qsLLL")
_UPPER_BODY_LAYOUT = ("qsBACK", "qsRUA", "qsRLA", "qsLUA", "qsLLA")
_RECORDING_RE = re.compile(r"^S[0-9]{2}_A(?:0[1-9]|1[0-3])_T[0-9]{2}$")
_PLOT_RE = re.compile(
    r"<!-- (?P<recording>S[0-9]{2}_A(?:0[1-9]|1[0-3])_T[0-9]{2})"
    r"\.mot \(Angle: [^)]*\) -->.*?"
    r"<!-- SHIFTED, RMSE: (?P<rmse>[0-9]+\.[0-9]{2})  "
    r"\(cut imu:(?P<imu>[0-9]+), cut vid:(?P<video>[0-9]+)\) -->",
    flags=re.DOTALL,
)

_TOOLS_MEMBER_HASHES = {
    "synchronize/EstimateFileSynchronization.ipynb": ESTIMATE_NOTEBOOK_SHA256,
    "synchronize/ModifyFilesToSync.ipynb": MODIFY_NOTEBOOK_SHA256,
    "utils/fileProcessing.py": EXPECTED_FILE_PROCESSING_SHA256,
    "utils/signalProcessing.py": EXPECTED_SIGNAL_PROCESSING_SHA256,
    "utils/syncUtilities.py": SYNC_UTILITY_SHA256,
}


class SourceTransformError(ValueError):
    """Raised when pinned VIDIMU source evidence fails reconciliation."""


@dataclass(frozen=True, slots=True)
class SourceInstruction:
    """One exact source-order row from ``infoToSync.csv``."""

    row_ordinal: int
    row_sha256: str
    recording_id: str
    subject: str
    activity: str
    trial: str
    file: str
    override_type: str
    cut_frames: int
    original_rmse_token: str
    theoretical_rmse_token: str


@dataclass(frozen=True, slots=True)
class RawGroupClassification:
    """Structural classification only; this does not confer timing authority."""

    group_status: str
    sensor_row_count: int
    expected_sensor_count: int
    observed_sensor_count: int
    timing_authority: str = "NONE_RAW_POLL_GROUP_ONLY"


def canonical_json_bytes(value: object) -> bytes:
    """Return timestamp-free canonical JSON used for evidence hashes."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            separators=(",", ": "),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stat_identity(value: stat.struct_stat) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular(path: Path, *, maximum_bytes: int) -> bytes:
    if path.is_symlink():
        raise SourceTransformError(f"input must not be a symlink: {path.name}")
    try:
        before = path.stat()
    except OSError as exc:
        raise SourceTransformError(f"cannot inspect input: {path.name}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
        raise SourceTransformError(f"input is not a bounded regular file: {path.name}")
    try:
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise SourceTransformError(f"cannot read input: {path.name}") from exc
    if len(payload) != before.st_size or _stat_identity(before) != _stat_identity(after):
        raise SourceTransformError(f"input changed while read: {path.name}")
    return payload


def _sha256_regular(path: Path, *, maximum_bytes: int) -> str:
    if path.is_symlink():
        raise SourceTransformError(f"input must not be a symlink: {path.name}")
    try:
        before = path.stat()
    except OSError as exc:
        raise SourceTransformError(f"cannot inspect input: {path.name}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
        raise SourceTransformError(f"input is not a bounded regular file: {path.name}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise SourceTransformError(f"cannot hash input: {path.name}") from exc
    if _stat_identity(before) != _stat_identity(after):
        raise SourceTransformError(f"input changed while hashed: {path.name}")
    return digest.hexdigest()


@contextmanager
def _verified_regular_snapshot(
    path: Path,
    *,
    maximum_bytes: int,
    expected_sha256: str,
):
    """Copy one pinned, verified descriptor into immutable parsing bytes."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceTransformError(f"cannot open pinned input: {path.name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise SourceTransformError(
                f"pinned input is not a bounded regular file: {path.name}"
            )
        try:
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                payload = source.read(maximum_bytes + 1)
                after = os.fstat(source.fileno())
        except OSError as exc:
            raise SourceTransformError(
                f"cannot snapshot pinned input: {path.name}"
            ) from exc
        if (
            len(payload) != before.st_size
            or len(payload) > maximum_bytes
            or _stat_identity(before) != _stat_identity(after)
        ):
            raise SourceTransformError(
                f"pinned input changed while snapshotted: {path.name}"
            )
        if _sha256_bytes(payload) != expected_sha256:
            raise SourceTransformError(f"pinned input hash mismatch: {path.name}")
        with io.BytesIO(payload) as immutable:
            yield immutable
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _implementation_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve(strict=True).parent
    return {
        "v05d_authority_module_sha256": _sha256_regular(
            directory / "authority.py", maximum_bytes=1_000_000
        ),
        "v05d_schema_module_sha256": _sha256_regular(
            directory / "schemas.py", maximum_bytes=1_000_000
        ),
        "v05d_source_transform_module_sha256": _sha256_regular(
            Path(__file__).resolve(strict=True), maximum_bytes=2_000_000
        ),
    }


def _safe_zip(archive: zipfile.ZipFile, *, label: str) -> None:
    seen: set[str] = set()
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        normalized = path.as_posix()
        if (
            not normalized
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in item.filename
            or normalized in seen
        ):
            raise SourceTransformError(f"{label} has unsafe or duplicate members")
        seen.add(normalized)


def _single_tools_root(archive: zipfile.ZipFile) -> str:
    roots = {PurePosixPath(name).parts[0] for name in archive.namelist()}
    if len(roots) != 1:
        raise SourceTransformError("tools release must have one archive root")
    return next(iter(roots))


def verify_frozen_inputs(
    snapshot_root: str | Path,
    analysis_archive: str | Path,
    tools_archive: str | Path,
    v05_authority_script: str | Path,
    v05_authority_report: str | Path,
) -> dict[str, object]:
    """Verify all source/tool pins without serializing local path names."""

    snapshot = Path(snapshot_root)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise SourceTransformError("snapshot root must be a real directory")
    snapshot = snapshot.resolve(strict=True)
    if snapshot.name != EXPECTED_SNAPSHOT_SHA256:
        raise SourceTransformError("snapshot identity is not frozen v0.4")
    manifest = _read_regular(
        snapshot / "snapshot_manifest.json", maximum_bytes=2_000_000
    )
    inventory = _read_regular(
        snapshot / "source_inventory.json", maximum_bytes=2_000_000
    )
    if _sha256_bytes(manifest) != EXPECTED_SNAPSHOT_SHA256:
        raise SourceTransformError("snapshot manifest hash mismatch")
    if _sha256_bytes(inventory) != EXPECTED_SOURCE_INVENTORY_SHA256:
        raise SourceTransformError("source inventory hash mismatch")

    dataset_path = snapshot / "objects" / EXPECTED_DATASET_ARCHIVE_SHA256
    if _sha256_regular(dataset_path, maximum_bytes=400_000_000) \
            != EXPECTED_DATASET_ARCHIVE_SHA256:
        raise SourceTransformError("dataset archive hash mismatch")

    analysis_path = Path(analysis_archive)
    analysis_payload = _read_regular(analysis_path, maximum_bytes=100_000_000)
    if _sha256_bytes(analysis_payload) != EXPECTED_ANALYSIS_ARCHIVE_SHA256:
        raise SourceTransformError("analysis archive hash mismatch")

    tools_path = Path(tools_archive)
    tools_payload = _read_regular(tools_path, maximum_bytes=2_000_000)
    if _sha256_bytes(tools_payload) != EXPECTED_TOOLS_ARCHIVE_SHA256:
        raise SourceTransformError("tools archive hash mismatch")
    with zipfile.ZipFile(io.BytesIO(tools_payload)) as archive:
        _safe_zip(archive, label="tools archive")
        if archive.comment != SOURCE_TOOLS_COMMIT.encode("ascii"):
            raise SourceTransformError("tools archive commit mismatch")
        root = _single_tools_root(archive)
        observed = {
            member: _sha256_bytes(archive.read(f"{root}/{member}"))
            for member in _TOOLS_MEMBER_HASHES
        }
    if observed != _TOOLS_MEMBER_HASHES:
        raise SourceTransformError("critical source-tool member hash mismatch")

    v05_script_hash = _sha256_regular(
        Path(v05_authority_script), maximum_bytes=2_000_000
    )
    v05_report_hash = _sha256_regular(
        Path(v05_authority_report), maximum_bytes=20_000_000
    )
    if v05_script_hash != V05_AUTHORITY_SCRIPT_SHA256:
        raise SourceTransformError("frozen v0.5 authority script changed")
    if v05_report_hash != V05_AUTHORITY_REPORT_SHA256:
        raise SourceTransformError("frozen v0.5 authority report changed")

    return {
        "analysis_archive_sha256": EXPECTED_ANALYSIS_ARCHIVE_SHA256,
        "critical_source_tool_member_hashes": dict(sorted(observed.items())),
        "dataset_archive_sha256": EXPECTED_DATASET_ARCHIVE_SHA256,
        "snapshot_manifest_sha256": EXPECTED_SNAPSHOT_SHA256,
        "source_inventory_sha256": EXPECTED_SOURCE_INVENTORY_SHA256,
        "source_tools_archive_sha256": EXPECTED_TOOLS_ARCHIVE_SHA256,
        "source_tools_commit": SOURCE_TOOLS_COMMIT,
        **_implementation_hashes(),
        "v05_authority_report_sha256": V05_AUTHORITY_REPORT_SHA256,
        "v05_authority_script_sha256": V05_AUTHORITY_SCRIPT_SHA256,
    }


def parse_source_instructions(payload: bytes) -> list[SourceInstruction]:
    """Parse and exact-byte-bind all 366 source rows in source order."""

    if _sha256_bytes(payload) != EXPECTED_INFO_TO_SYNC_SHA256:
        raise SourceTransformError("infoToSync.csv hash mismatch")
    lines = payload.splitlines(keepends=True)
    if len(lines) != EXPECTED_INFO_ROWS + 1 or any(
        not line.endswith(b"\r\n") for line in lines
    ):
        raise SourceTransformError("infoToSync.csv line topology changed")
    try:
        header = next(csv.reader([lines[0].decode("utf-8", errors="strict")]))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise SourceTransformError("infoToSync.csv header is invalid") from exc
    if tuple(header) != _INFO_FIELDS:
        raise SourceTransformError("infoToSync.csv fields changed")

    expected_names = {
        "csv": lambda recording: f"{recording}.csv",
        "mot": lambda recording: f"ik_{recording}.mot",
        "mp4": lambda recording: f"{recording}.mp4",
        "raw": lambda recording: f"{recording}.raw",
    }
    instructions: list[SourceInstruction] = []
    seen: set[tuple[str, str]] = set()
    for ordinal, raw_line in enumerate(lines[1:]):
        try:
            values = next(csv.reader(
                [raw_line.decode("utf-8", errors="strict")], strict=True
            ))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise SourceTransformError("infoToSync.csv row is invalid") from exc
        if len(values) != len(_INFO_FIELDS):
            raise SourceTransformError("infoToSync.csv row width changed")
        row = dict(zip(_INFO_FIELDS, values, strict=True))
        recording = f"{row['Subject']}_{row['Activity']}_{row['Trial']}"
        kind = row["Type"]
        if _RECORDING_RE.fullmatch(recording) is None or kind not in expected_names:
            raise SourceTransformError("source instruction identity is invalid")
        if PureWindowsPath(row["File"]).name != expected_names[kind](recording):
            raise SourceTransformError("source instruction filename is inconsistent")
        try:
            cut_frames = int(row["CutFrames"])
            Decimal(row["OrigRmse"])
            Decimal(row["TheoRmse"])
        except Exception as exc:
            raise SourceTransformError("source instruction numbers are invalid") from exc
        if not 1 <= cut_frames <= 14 or (recording, kind) in seen:
            raise SourceTransformError("source instruction is duplicate or out of range")
        seen.add((recording, kind))
        instructions.append(SourceInstruction(
            row_ordinal=ordinal,
            row_sha256=_sha256_bytes(raw_line),
            recording_id=recording,
            subject=row["Subject"],
            activity=row["Activity"],
            trial=row["Trial"],
            file=row["File"],
            override_type=kind,
            cut_frames=cut_frames,
            original_rmse_token=row["OrigRmse"],
            theoretical_rmse_token=row["TheoRmse"],
        ))
    counts = dict(sorted(Counter(row.override_type for row in instructions).items()))
    if len(instructions) != EXPECTED_INFO_ROWS or counts != _INFO_TYPE_COUNTS:
        raise SourceTransformError("source instruction counts changed")
    return instructions


def parse_source_winners(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Parse the 208 frozen choices from source-produced SVG outputs."""

    plot_names = sorted(
        name
        for name in archive.namelist()
        if re.fullmatch(
            r"analysis/videoandimusync/A(?:0[1-9]|1[0-3])_.*_synchronize\.svg",
            name,
        )
    )
    if len(plot_names) != 13:
        raise SourceTransformError("analysis archive must contain 13 sync plots")
    winners: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for name in plot_names:
        payload = archive.read(name)
        hashes[name] = _sha256_bytes(payload)
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SourceTransformError("sync SVG is not strict UTF-8") from exc
        activity = PurePosixPath(name).name[:3]
        for match in _PLOT_RE.finditer(text):
            recording = match.group("recording")
            imu_cut = int(match.group("imu"))
            video_cut = int(match.group("video"))
            if (
                recording.split("_")[1] != activity
                or recording in winners
                or (imu_cut and video_cut)
                or imu_cut > 14
                or video_cut > 14
            ):
                raise SourceTransformError("sync SVG mapping is noncanonical")
            direction = "IMU" if imu_cut else "VIDEO" if video_cut else "ZERO"
            winners[recording] = {
                "selected_direction": direction,
                "selected_imu_comparison_grid_samples": imu_cut,
                "selected_video_frames": video_cut,
                "source_plot_rmse_2dp": match.group("rmse"),
            }
    counts = dict(sorted(Counter(
        str(value["selected_direction"]) for value in winners.values()
    ).items()))
    if len(winners) != EXPECTED_RECORDINGS or counts != _WINNER_COUNTS:
        raise SourceTransformError("source winner counts changed")
    return winners, hashes


def source_removed_rows(override_type: str, cut_frames: int) -> int:
    """Return the exact upstream integer-truncated positional row count."""

    if isinstance(cut_frames, bool) or not isinstance(cut_frames, int) \
            or cut_frames < 0:
        raise SourceTransformError("cut_frames must be a nonnegative integer")
    if override_type == "csv":
        return cut_frames
    if override_type == "mot":
        return (cut_frames * 5) // 3
    if override_type == "raw":
        return (cut_frames * 25) // 3
    raise SourceTransformError("only CSV, MOT, and RAW are positional transforms")


def apply_source_trim(
    source: bytes,
    *,
    retained_prefix_lines: int,
    removed_rows: int,
) -> bytes:
    """Apply ``remove_insidelines_file`` as a byte-preserving line overlay."""

    if (
        isinstance(retained_prefix_lines, bool)
        or not isinstance(retained_prefix_lines, int)
        or retained_prefix_lines < 1
        or isinstance(removed_rows, bool)
        or not isinstance(removed_rows, int)
        or removed_rows < 0
    ):
        raise SourceTransformError("trim bounds must be nonnegative integers")
    lines = source.splitlines(keepends=True)
    stop = retained_prefix_lines + removed_rows
    if stop > len(lines):
        raise SourceTransformError("source trim exceeds the source file")
    return b"".join([*lines[:retained_prefix_lines], *lines[stop:]])


def _member_paths(instruction: SourceInstruction) -> tuple[str, str, int]:
    name = (
        f"ik_{instruction.recording_id}.mot"
        if instruction.override_type == "mot"
        else f"{instruction.recording_id}.{instruction.override_type}"
    )
    base = f"dataset/videoandimus/{instruction.subject}/{name}"
    published = f"dataset/videoandimusync/{instruction.subject}/{name}"
    retained = 8 if instruction.override_type == "mot" else 1
    return base, published, retained


def reproduce_source_trims(
    archive: zipfile.ZipFile,
    instructions: Iterable[SourceInstruction],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Bind and reproduce all 217 published non-MP4 transformations."""

    selected = [row for row in instructions if row.override_type != "mp4"]
    expected_paths = {_member_paths(row)[1] for row in selected}
    expected_directories = {
        "dataset/videoandimusync/",
        *(f"dataset/videoandimusync/{row.subject}/" for row in selected),
    }
    observed_paths = {
        item.filename
        for item in archive.infolist()
        if item.filename.startswith("dataset/videoandimusync/") and not item.is_dir()
    }
    observed_directories = {
        item.filename
        for item in archive.infolist()
        if item.filename.startswith("dataset/videoandimusync/") and item.is_dir()
    }
    if observed_paths != expected_paths or observed_directories != expected_directories:
        raise SourceTransformError("published derivative subtree is not exact")

    overlays: list[dict[str, object]] = []
    discrepancies: list[dict[str, object]] = []
    for instruction in selected:
        source_path, published_path, retained = _member_paths(instruction)
        try:
            source = archive.read(source_path)
            published = archive.read(published_path)
        except KeyError as exc:
            raise SourceTransformError("source transformation asset is missing") from exc
        expected_removed = source_removed_rows(
            instruction.override_type, instruction.cut_frames
        )
        generated = apply_source_trim(
            source,
            retained_prefix_lines=retained,
            removed_rows=expected_removed,
        )
        source_lines = source.splitlines(keepends=True)
        published_lines = published.splitlines(keepends=True)
        observed_removed = len(source_lines) - len(published_lines)
        comparison = (
            "EXACT_BYTE_MATCH" if generated == published else "BYTE_MISMATCH"
        )
        row = {
            "comparison_status": comparison,
            "cut_frames": instruction.cut_frames,
            "expected_removed_rows": expected_removed,
            "generated_derivative_sha256": _sha256_bytes(generated),
            "observed_removed_rows": observed_removed,
            "override_row_ordinal": instruction.row_ordinal,
            "override_row_sha256": instruction.row_sha256,
            "override_source_asset_sha256": EXPECTED_INFO_TO_SYNC_SHA256,
            "override_type": instruction.override_type,
            "pair_id": f"VIDIMU::{instruction.recording_id}",
            "published_asset_sha256": _sha256_bytes(published),
            "published_member_path": published_path,
            "recording_id": instruction.recording_id,
            "retained_header_rows": retained,
            "source_asset_sha256": _sha256_bytes(source),
            "source_member_path": source_path,
            "trim_start_data_row_ordinal": 0,
            "trim_stop_data_row_ordinal": expected_removed,
        }
        overlays.append(row)
        if comparison != "EXACT_BYTE_MATCH" or observed_removed != expected_removed:
            discrepancies.append({
                "comparison_status": comparison,
                "expected_removed_rows": expected_removed,
                "observed_removed_rows": observed_removed,
                "override_row_ordinal": instruction.row_ordinal,
                "published_member_path": published_path,
            })
    type_counts = dict(sorted(Counter(
        str(row["override_type"]) for row in overlays
    ).items()))
    if len(overlays) != EXPECTED_NON_MP4_INSTRUCTIONS \
            or type_counts != _OVERRIDE_TYPE_COUNTS:
        raise SourceTransformError("non-MP4 source transformation count changed")
    return overlays, discrepancies


def classify_raw_sensor_group(
    labels: Iterable[str | None], expected_layout: Iterable[str]
) -> RawGroupClassification:
    """Classify one positional RAW sensor group without calling it a tick."""

    observed = tuple(labels)
    expected = tuple(expected_layout)
    nonempty = tuple(value for value in observed if isinstance(value, str) and value)
    if len(observed) != len(expected):
        status = "INCOMPLETE"
    elif len(nonempty) != len(observed):
        status = "MALFORMED"
    elif any(value not in expected for value in nonempty):
        status = "UNKNOWN_SENSOR"
    elif len(set(nonempty)) != len(nonempty):
        status = "DUPLICATE_SENSOR"
    elif nonempty != expected:
        status = "MALFORMED"
    else:
        status = "COMPLETE"
    return RawGroupClassification(
        group_status=status,
        sensor_row_count=len(observed),
        expected_sensor_count=len(expected),
        observed_sensor_count=len(set(nonempty)),
    )


def _raw_expected_layout(recording_id: str) -> tuple[str, ...]:
    activity = int(recording_id.split("_")[1][1:])
    return _LOWER_BODY_LAYOUT if activity <= 4 else _UPPER_BODY_LAYOUT


def _scan_raw_member(
    archive: zipfile.ZipFile, member: str, recording_id: str
) -> dict[str, object]:
    expected = _raw_expected_layout(recording_id)
    status_counts: Counter[str] = Counter()
    data_rows = 0
    current: list[str | None] = []
    with archive.open(member) as handle:
        header = handle.readline().rstrip(b"\r\n")
        if header != b"QUAT,w,x,y,z,timestamp":
            raise SourceTransformError("RAW header changed")
        for raw_line in handle:
            data_rows += 1
            label_bytes, separator, _ = raw_line.partition(b",")
            try:
                label = label_bytes.decode("ascii", errors="strict") if separator else None
            except UnicodeDecodeError:
                label = None
            current.append(label)
            if len(current) == 5:
                classification = classify_raw_sensor_group(current, expected)
                status_counts[classification.group_status] += 1
                current = []
    if current:
        classification = classify_raw_sensor_group(current, expected)
        status_counts[classification.group_status] += 1
    if data_rows < 5:
        raise SourceTransformError("RAW member has no calibration group")
    return {
        "all_structural_groups_complete": set(status_counts) == {"COMPLETE"},
        "data_rows_including_npose": data_rows,
        "dynamic_poll_group_count": (data_rows - 5) // 5,
        "dynamic_rows": data_rows - 5,
        "group_status_counts_including_npose": dict(sorted(status_counts.items())),
    }


def _sto_mot_counts(
    archive: zipfile.ZipFile, subject: str, recording_id: str
) -> tuple[int, int]:
    sto_path = f"dataset/videoandimus/{subject}/{recording_id}.sto"
    mot_path = f"dataset/videoandimus/{subject}/ik_{recording_id}.mot"
    try:
        sto_lines = archive.read(sto_path).splitlines()
        mot_lines = archive.read(mot_path).splitlines()
    except KeyError as exc:
        raise SourceTransformError("STO/MOT companion is missing") from exc
    if (
        len(sto_lines) < 8
        or sto_lines[0] != b"DataRate=50"
        or sto_lines[4] != b"endheader"
        or not sto_lines[5].startswith(b"time\t")
        or len(mot_lines) < 9
        or mot_lines[5] != b"endheader"
        or not mot_lines[6].startswith(b"time\t")
    ):
        raise SourceTransformError("STO/MOT source structure changed")
    return len(sto_lines) - 6, len(mot_lines) - 7


def _fraction_decimal_6(value: Fraction) -> str:
    with localcontext() as context:
        context.prec = 50
        result = Decimal(value.numerator) / Decimal(value.denominator)
        return format(result.quantize(Decimal("0.000001"), ROUND_HALF_EVEN), "f")


def reconcile_raw_poll_groups_to_sto_mot(
    archive: zipfile.ZipFile,
) -> dict[str, object]:
    """Prove that RAW groups cannot be used as nominal 50 Hz tick ordinals."""

    raw_pattern = re.compile(
        r"dataset/videoandimus/(?P<subject>S[0-9]{2})/"
        r"(?P<recording>S[0-9]{2}_A(?:0[1-9]|1[0-3])_T[0-9]{2})\.raw"
    )
    raw_members = []
    for name in archive.namelist():
        match = raw_pattern.fullmatch(name)
        if match is not None:
            raw_members.append((match.group("recording"), match.group("subject"), name))
    raw_members.sort()
    if len(raw_members) != EXPECTED_RECORDINGS:
        raise SourceTransformError("RAW source coverage changed")

    records: list[dict[str, object]] = []
    for recording, subject, member in raw_members:
        raw = _scan_raw_member(archive, member, recording)
        sto_rows, mot_rows = _sto_mot_counts(archive, subject, recording)
        if sto_rows != mot_rows:
            raise SourceTransformError("STO and MOT ordinal counts differ")
        raw_groups = int(raw["dynamic_poll_group_count"])
        dynamic_ticks = sto_rows - 1
        if dynamic_ticks < 1:
            raise SourceTransformError("STO/MOT source has no dynamic rows")
        records.append({
            "all_raw_groups_complete": raw["all_structural_groups_complete"],
            "mot_rows_including_npose": mot_rows,
            "raw_data_rows_including_npose": raw["data_rows_including_npose"],
            "raw_dynamic_poll_groups": raw_groups,
            "raw_dynamic_rows": raw["dynamic_rows"],
            "raw_group_to_sto_dynamic_ratio_den": dynamic_ticks,
            "raw_group_to_sto_dynamic_ratio_num": raw_groups,
            "recording_id": recording,
            "sto_rows_including_npose": sto_rows,
        })

    total_raw_rows = sum(int(row["raw_data_rows_including_npose"]) for row in records)
    total_raw_dynamic = sum(int(row["raw_dynamic_rows"]) for row in records)
    total_raw_groups = sum(int(row["raw_dynamic_poll_groups"]) for row in records)
    total_sto_rows = sum(int(row["sto_rows_including_npose"]) for row in records)
    total_mot_rows = sum(int(row["mot_rows_including_npose"]) for row in records)
    ratios = [
        Fraction(
            int(row["raw_group_to_sto_dynamic_ratio_num"]),
            int(row["raw_group_to_sto_dynamic_ratio_den"]),
        )
        for row in records
    ]
    sorted_ratios = sorted(ratios)
    median = (sorted_ratios[103] + sorted_ratios[104]) / 2
    if (
        total_raw_rows != EXPECTED_RAW_ROWS_INCLUDING_NPOSE
        or total_raw_dynamic != EXPECTED_RAW_DYNAMIC_ROWS
        or total_raw_groups != EXPECTED_RAW_POLL_GROUPS
        or total_sto_rows != EXPECTED_STO_MOT_ROWS_INCLUDING_NPOSE
        or total_mot_rows != EXPECTED_STO_MOT_ROWS_INCLUDING_NPOSE
        or any(not bool(row["all_raw_groups_complete"]) for row in records)
    ):
        raise SourceTransformError("RAW/STO/MOT reconciliation changed")
    aggregate_ratio = Fraction(total_raw_groups, EXPECTED_STO_MOT_DYNAMIC_ROWS)
    if aggregate_ratio == 1 or all(value == 1 for value in ratios):
        raise SourceTransformError("unexpected RAW-to-STO one-to-one relation")

    record_manifest = [
        {
            "mot_rows_including_npose": row["mot_rows_including_npose"],
            "raw_dynamic_poll_groups": row["raw_dynamic_poll_groups"],
            "recording_id": row["recording_id"],
            "sto_rows_including_npose": row["sto_rows_including_npose"],
        }
        for row in records
    ]
    return {
        "all_original_raw_groups_structurally_complete": True,
        "aggregate_raw_group_to_sto_dynamic_ratio": _fraction_decimal_6(
            aggregate_ratio
        ),
        "mot_dynamic_ordinal_rows": total_mot_rows - EXPECTED_RECORDINGS,
        "mot_rows_including_npose": total_mot_rows,
        "per_record_ratio_maximum": _fraction_decimal_6(max(ratios)),
        "per_record_ratio_median": _fraction_decimal_6(median),
        "per_record_ratio_minimum": _fraction_decimal_6(min(ratios)),
        "raw_dynamic_poll_groups": total_raw_groups,
        "raw_dynamic_rows": total_raw_dynamic,
        "raw_group_timing_authority": "NONE_RAW_POLL_GROUP_ONLY",
        "raw_groups_are_nominal_50hz_ticks": False,
        "raw_rows_including_npose": total_raw_rows,
        "record_count": len(records),
        "record_reconciliation_sha256": _sha256_bytes(
            canonical_json_bytes(record_manifest)
        ),
        "sto_dynamic_ordinal_rows": total_sto_rows - EXPECTED_RECORDINGS,
        "sto_mot_ordinal_identity_status": "ROW_COUNTS_EQUAL_ALL_RECORDINGS",
        "sto_rows_including_npose": total_sto_rows,
    }


def _instruction_summary(
    instructions: list[SourceInstruction],
) -> dict[str, object]:
    non_mp4 = [row for row in instructions if row.override_type != "mp4"]
    row_manifest = [
        {
            "cut_frames": row.cut_frames,
            "override_row_ordinal": row.row_ordinal,
            "override_row_sha256": row.row_sha256,
            "override_type": row.override_type,
            "recording_id": row.recording_id,
        }
        for row in non_mp4
    ]
    return {
        "info_row_count": len(instructions),
        "info_row_type_counts": dict(sorted(Counter(
            row.override_type for row in instructions
        ).items())),
        "info_to_sync_sha256": EXPECTED_INFO_TO_SYNC_SHA256,
        "non_mp4_instruction_count": len(non_mp4),
        "non_mp4_instruction_manifest_sha256": _sha256_bytes(
            canonical_json_bytes(row_manifest)
        ),
        "non_mp4_instruction_recording_count": len({
            row.recording_id for row in non_mp4
        }),
        "non_mp4_instruction_type_counts": dict(sorted(Counter(
            row.override_type for row in non_mp4
        ).items())),
        "override_row_ordinal_basis": "ZERO_BASED_DATA_ROW_SOURCE_ORDER",
        "override_row_sha256_basis": "EXACT_SOURCE_ROW_BYTES_INCLUDING_CRLF",
    }


def _mapping_summary(
    winners: Mapping[str, Mapping[str, object]],
    instructions: Iterable[SourceInstruction],
) -> dict[str, object]:
    by_record: dict[str, set[str]] = defaultdict(set)
    for row in instructions:
        by_record[row.recording_id].add(row.override_type)
    for recording in AMBIGUOUS_RECORDING_IDS:
        if winners[recording]["selected_direction"] != "VIDEO" or not {
            "csv", "mp4", "mot", "raw"
        }.issubset(by_record[recording]):
            raise SourceTransformError("known source ambiguity changed")
    classification_counts = Counter()
    for recording, value in winners.items():
        classification = (
            "AMBIGUOUS"
            if recording in AMBIGUOUS_RECORDING_IDS
            else str(value["selected_direction"])
        )
        classification_counts[classification] += 1
    if dict(sorted(classification_counts.items())) != _CLASSIFICATION_COUNTS:
        raise SourceTransformError("source mapping classifications changed")
    return {
        "ambiguous_pair_ids": sorted(AMBIGUOUS_RECORDING_IDS),
        "ambiguous_source_pairs": len(AMBIGUOUS_RECORDING_IDS),
        "classification_counts": dict(sorted(classification_counts.items())),
        "eligible_source_pairs_upper_bound": (
            EXPECTED_RECORDINGS - len(AMBIGUOUS_RECORDING_IDS)
        ),
        "source_winner_counts": dict(sorted(Counter(
            str(value["selected_direction"]) for value in winners.values()
        ).items())),
    }


def audit_source_transform_evidence(
    snapshot_root: str | Path,
    analysis_archive: str | Path,
    tools_archive: str | Path,
    v05_authority_script: str | Path,
    v05_authority_report: str | Path,
) -> dict[str, object]:
    """Return deterministic evidence for the v0.5D materialization gate."""

    pins = verify_frozen_inputs(
        snapshot_root,
        analysis_archive,
        tools_archive,
        v05_authority_script,
        v05_authority_report,
    )
    snapshot = Path(snapshot_root).resolve(strict=True)
    dataset_path = snapshot / "objects" / EXPECTED_DATASET_ARCHIVE_SHA256
    analysis_payload = _read_regular(Path(analysis_archive), maximum_bytes=100_000_000)
    if _sha256_bytes(analysis_payload) != EXPECTED_ANALYSIS_ARCHIVE_SHA256:
        raise SourceTransformError("analysis archive changed after input verification")
    with zipfile.ZipFile(io.BytesIO(analysis_payload)) as analysis:
        _safe_zip(analysis, label="analysis archive")
        info_payload = analysis.read("analysis/videoandimusync/infoToSync.csv")
        instructions = parse_source_instructions(info_payload)
        winners, plot_hashes = parse_source_winners(analysis)
    with _verified_regular_snapshot(
        dataset_path,
        maximum_bytes=400_000_000,
        expected_sha256=EXPECTED_DATASET_ARCHIVE_SHA256,
    ) as dataset_handle, zipfile.ZipFile(dataset_handle) as dataset:
        _safe_zip(dataset, label="dataset archive")
        overlays, discrepancies = reproduce_source_trims(dataset, instructions)
        raw_reconciliation = reconcile_raw_poll_groups_to_sto_mot(dataset)

    overlay_manifest = [
        {
            key: row[key]
            for key in (
                "comparison_status",
                "expected_removed_rows",
                "generated_derivative_sha256",
                "override_row_ordinal",
                "override_row_sha256",
                "override_type",
                "published_asset_sha256",
                "published_member_path",
                "recording_id",
                "source_asset_sha256",
                "source_member_path",
            )
        }
        for row in overlays
    ]
    raw_overlays = [row for row in overlays if row["override_type"] == "raw"]
    raw_removed_rows = sum(
        int(row["expected_removed_rows"]) for row in raw_overlays
    )
    raw_trim_defects = {
        "npose_rows_removed_total": len(raw_overlays) * 5,
        "partial_five_sensor_cycle_override_count": sum(
            int(row["expected_removed_rows"]) % 5 != 0 for row in raw_overlays
        ),
        "raw_override_count": len(raw_overlays),
        "raw_overrides_removing_all_npose_rows_count": sum(
            int(row["expected_removed_rows"]) >= 5 for row in raw_overlays
        ),
        "removed_dynamic_observation_row_count": (
            raw_removed_rows - len(raw_overlays) * 5
        ),
        "removed_source_data_row_count": raw_removed_rows,
    }
    if raw_trim_defects != {
        "npose_rows_removed_total": 170,
        "partial_five_sensor_cycle_override_count": 30,
        "raw_override_count": 34,
        "raw_overrides_removing_all_npose_rows_count": 34,
        "removed_dynamic_observation_row_count": 648,
        "removed_source_data_row_count": 818,
    }:
        raise SourceTransformError("released RAW trim defect counts changed")
    current_implementation_hashes = _implementation_hashes()
    if any(
        pins.get(key) != value
        for key, value in current_implementation_hashes.items()
    ):
        raise SourceTransformError("v0.5D implementation changed during audit")
    return {
        "frozen_inputs": pins,
        "raw_poll_to_sto_mot_reconciliation": raw_reconciliation,
        "source_alignment_decisions": {
            **_mapping_summary(winners, instructions),
            "sync_plot_hashes": plot_hashes,
        },
        "source_instructions": _instruction_summary(instructions),
        "source_trim_reproduction": {
            "all_generated_derivatives_byte_identical": not discrepancies,
            "discrepancies": discrepancies,
            "overrides_bound": len(overlays),
            "overrides_expected": EXPECTED_NON_MP4_INSTRUCTIONS,
            "overrides_reproduced": sum(
                row["comparison_status"] == "EXACT_BYTE_MATCH" for row in overlays
            ),
            "overrides_unreproduced": len(discrepancies),
            "published_derivative_directory_count": 1 + len({
                str(row["recording_id"]).split("_")[0] for row in overlays
            }),
            "raw_trim_defects": raw_trim_defects,
            "source_trim_overlay_manifest_sha256": _sha256_bytes(
                canonical_json_bytes(overlay_manifest)
            ),
            "source_transform_semantics": {
                "csv_removed_rows": "CutFrames",
                "mot_removed_rows": "floor(CutFrames*5/3)",
                "raw_removed_rows": "floor(CutFrames*25/3)",
                "replay": "EXACT_BYTE_LINE_SPLICE_PRESERVING_SOURCE_ENDINGS",
            },
            "type_counts": dict(sorted(Counter(
                str(row["override_type"]) for row in overlays
            ).items())),
        },
    }


def instruction_as_dict(value: SourceInstruction) -> dict[str, object]:
    """Expose a canonical serializable row for focused tests and tools."""

    return asdict(value)


__all__ = [
    "EXPECTED_ANALYSIS_ARCHIVE_SHA256",
    "EXPECTED_DATASET_ARCHIVE_SHA256",
    "EXPECTED_INFO_ROWS",
    "EXPECTED_INFO_TO_SYNC_SHA256",
    "EXPECTED_NON_MP4_INSTRUCTIONS",
    "EXPECTED_RAW_DYNAMIC_ROWS",
    "EXPECTED_RAW_POLL_GROUPS",
    "EXPECTED_RAW_ROWS_INCLUDING_NPOSE",
    "EXPECTED_RECORDINGS",
    "EXPECTED_SNAPSHOT_SHA256",
    "EXPECTED_SOURCE_INVENTORY_SHA256",
    "EXPECTED_STO_MOT_DYNAMIC_ROWS",
    "EXPECTED_STO_MOT_ROWS_INCLUDING_NPOSE",
    "EXPECTED_TOOLS_ARCHIVE_SHA256",
    "RawGroupClassification",
    "SourceInstruction",
    "SourceTransformError",
    "apply_source_trim",
    "audit_source_transform_evidence",
    "canonical_json_bytes",
    "classify_raw_sensor_group",
    "instruction_as_dict",
    "parse_source_instructions",
    "parse_source_winners",
    "reconcile_raw_poll_groups_to_sto_mot",
    "reproduce_source_trims",
    "source_removed_rows",
    "verify_frozen_inputs",
]
