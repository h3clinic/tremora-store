"""Audit VIDIMU's released synchronization authority without inventing clocks.

This milestone deliberately ends in ``NO_GO_RAW_NATIVE_CLOCK_AUTHORITY``.
The pinned authoritative evidence does not document the RAW timestamp unit,
clock origin, or mapping to video PTS, and two released records contain applied
cuts in both candidate directions.  This audit therefore emits evidence only:
it never materializes an IMU time or estimates an offset from correlation.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import uuid
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

ARTIFACT_KIND = "TREMORA_VIDIMU_V05_SYNC_AUTHORITY_AUDIT"
SCHEMA_VERSION = "0.5.1"
IMPLEMENTATION_VERSION = "vidimu-v05-sync-authority-audit-1.1.0"
NO_GO = "NO_GO_RAW_NATIVE_CLOCK_AUTHORITY"
AUDIT_PASS = "PASS"
AUDIT_ERROR = "ERROR"
NOT_EVALUATED = "NOT_EVALUATED"
NO_GO_EXIT_CODE = 3

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
EXPECTED_ANALYSIS_ARCHIVE_MD5 = "d68f2a16fa3fcd8cec090a3169e4764e"
EXPECTED_TOOLS_ARCHIVE_SHA256 = (
    "bc8820d5f79fcc2e58233474ed15aa2191faf2463e82b3d4d1ddacf44eaf9e27"
)
EXPECTED_TOOLS_COMMIT = "19beec4156f0109d46341a08f06b035d772afaec"
EXPECTED_TOOLS_MEMBER_MANIFEST_SHA256 = (
    "22668e7886c5545e3a54d48e34a0125dae2c652db208c1dd2adaba95144d6012"
)
EXPECTED_V04_RELEASE_AUDIT_SHA256 = (
    "d24863d20347cf2c9ab092a9f7771ada3a88ec8fbc77a7b33788df5c0637a10e"
)
EXPECTED_V2_RELEASE_AUDIT_SHA256 = (
    "41277661f9e248da2f42c0703b69beec92bcaf0037b5d46264f64852ab22ecf1"
)
EXPECTED_SOURCE_PARSER_SHA256 = (
    "244685dcb6a0de1910b23d729c644b91dc807c314690ddc7400ab7d99c3699ae"
)
EXPECTED_ARTICLE_PDF_SHA256 = (
    "c6927d0fff9f4ebbf371ddc46f5c2f67b416021b630fe2eba7770a55caf36d36"
)
EXPECTED_RECORD_METADATA_SHA256 = (
    "97fd3f755aab65631ec19bff762fe848a609cee5324644f6baedf1d1b4f3926f"
)
ARTICLE_URL = (
    "https://uvadoc.uva.es/bitstream/handle/10324/64478/"
    "s41597-023-02554-9.pdf?isAllowed=y&sequence=1"
)
RECORD_URL = "https://zenodo.org/records/15075076"
TOOLS_URL = "https://github.com/twyncoder/vidimu-tools"

EXPECTED_RECORDINGS = 208
EXPECTED_ASSET_REFERENCES = 624
EXPECTED_POSE_AND_FRAME_ROWS = 179_076
EXPECTED_INFO_ROWS = 366
EXPECTED_SYNC_OVERRIDES = 217
EXPECTED_RAW_ROWS = 10_184_045
EXPECTED_DYNAMIC_RAW_ROWS = 10_183_005
EXPECTED_RAW_STREAMS = 1_040
EXPECTED_RAW_BYTES = 545_308_276
EXPECTED_HELD_PAYLOAD_ROWS = 8_735_242
EXPECTED_DISTINCT_PAYLOAD_RUNS = 1_447_763
EXPECTED_SYNC_ALL_PATHS_SHA256 = (
    "8310351662ef1c378b484c000c0753f4eb5b99078737915562cbff629db6766b"
)
EXPECTED_SYNC_FILE_PATHS_SHA256 = (
    "56b9609d46a61b05e54ed4ad4e1129bd87d2b7a839b92862af74eb7536717496"
)
EXPECTED_SYNC_DIRECTORY_PATHS_SHA256 = (
    "4e1463ba9a1ddda0912916ee1fa201128dc60a12c910506cbeac18d19a521400"
)
EXPECTED_DIRECTION_COUNTS: Final = {"IMU": 32, "VIDEO": 149, "ZERO": 27}
EXPECTED_INFO_TYPE_COUNTS: Final = {"csv": 149, "mot": 34, "mp4": 149, "raw": 34}
EXPECTED_OVERRIDE_TYPE_COUNTS: Final = {"csv": 149, "mot": 34, "raw": 34}
EXPECTED_HELD_FRACTION_DECIMAL_6: Final = {
    "minimum": "0.797315",
    "median": "0.855143",
    "maximum": "0.920376",
}
EXPECTED_DUAL_DIRECTION_RECORDS: Final = {
    "S53_A13_T03": {
        "applied_imu_cut_frames": 1,
        "applied_mot_removed_rows": 1,
        "applied_raw_removed_rows": 8,
        "applied_video_cut_frames": 2,
    },
    "S57_A07_T01": {
        "applied_imu_cut_frames": 14,
        "applied_mot_removed_rows": 23,
        "applied_raw_removed_rows": 116,
        "applied_video_cut_frames": 11,
    },
}
_LOWER_BODY_LAYOUT: Final = ("qsHIPS", "qsRUL", "qsRLL", "qsLUL", "qsLLL")
_UPPER_BODY_LAYOUT: Final = ("qsBACK", "qsRUA", "qsRLA", "qsLUA", "qsLLA")

_TOOLS_MEMBER_HASHES: Final = {
    ".gitignore": (
        "0cd064765b740a9d791f15266ee55d644b4b0cf40294754c21d5901ed9c781c3"
    ),
    "LICENSE": (
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"
    ),
    "README.md": (
        "c8fc6a34bf55288b5c125f435029fd6de1ac9b52832b0596ed3548ac281a00ff"
    ),
    "imus/PlotImusIkJointAngles.ipynb": (
        "8c76fb8a02498cf3b72ca024c217636d4d2b2851ee120d200a82db5e59e38dc3"
    ),
    "imus/PlotImusRawQuats.ipynb": (
        "622ef9c2a0513b1a0dcbd97625c6522474a2aaef62991f882e9118d18119b6e5"
    ),
    "synchronize/EstimateFileSynchronization.ipynb": (
        "a88c3bb86a27587ca30d99f820643bb48e27500e58f62b035174143e9c1e4865"
    ),
    "synchronize/ModifyFilesToSync.ipynb": (
        "5b719a4e6b80419df18f0711dc62f44e0d7cbdb6bb4337847f3281be097d5fbf"
    ),
    "utils/fileProcessing.py": (
        "2b534daa9887934824e0034ce7af42414ab1884743a581a598300949693f4331"
    ),
    "utils/plotUtilities.py": (
        "fcea1a34b5e8709ec05ef4cda82b2a53d2e3862b846f04f2e1aa0442ba049d72"
    ),
    "utils/signalProcessing.py": (
        "88e4701b71e7b7c048f64f1698220d22ee0887b97624aa2c38fae8d37861c591"
    ),
    "utils/syncUtilities.py": (
        "f2674ded71f19a837c9e7cb5f6678ae7944ad8f09932e029845d0766b55f139d"
    ),
    "video/ConvertBodytrackToCSV.ipynb": (
        "2f644ee17b2e5fa0d2ca60b9d583097a80df68f0abb6046062b155ace2f9159b"
    ),
    "video/PlotVideoEstimatedJointAngles.ipynb": (
        "f036d2d976196c889cfaf9a400ef4af44bf9d9a1859f94dd3974c16dac60d579"
    ),
    "video/RecodeMP4toSmallsizefiles.ipynb": (
        "b15a5f06a26a701c6c066b8c23ff06e3d4654b13df4fc726ca64871f1f64e2c7"
    ),
    "video/ScriptsToBodyTrack.ipynb": (
        "fcd622e674fff2342495f24ae7730ea87beecde5a4c34e1e68eab94912fe2f58"
    ),
}

_RECORDING_RE = re.compile(r"^S[0-9]{2}_A(?:0[1-9]|1[0-3])_T[0-9]{2}$")
_PLOT_RE = re.compile(
    r"<!-- (?P<recording>S[0-9]{2}_A(?:0[1-9]|1[0-3])_T[0-9]{2})"
    r"\.mot \(Angle: [^)]*\) -->.*?"
    r"<!-- SHIFTED, RMSE: (?P<rmse>[0-9]+\.[0-9]{2})  "
    r"\(cut imu:(?P<imu>[0-9]+), cut vid:(?P<video>[0-9]+)\) -->",
    flags=re.DOTALL,
)
_SYNC_MEMBER_RE = re.compile(
    r"^dataset/videoandimusync/(?P<subject>S[0-9]{2})/"
    r"(?P<prefix>ik_)?(?P<recording>S[0-9]{2}_A(?:0[1-9]|1[0-3])_T[0-9]{2})"
    r"\.(?P<extension>csv|raw|mot)$"
)
_RAW_TIMESTAMP_RE = re.compile(rb"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,9})?\Z")


class SyncAuthorityAuditError(ValueError):
    """Raised when release evidence fails closed."""


def canonical_json_bytes(value: object) -> bytes:
    """Use the repository's canonical checked-artifact JSON convention."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            separators=(",", ": "),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path, *, maximum_bytes: int | None = None) -> bytes:
    if path.is_symlink():
        raise SyncAuthorityAuditError(f"input must not be a symlink: {path.name}")
    try:
        before = path.stat()
    except OSError as exc:
        raise SyncAuthorityAuditError(f"cannot inspect input: {path.name}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise SyncAuthorityAuditError(f"input is not a regular file: {path.name}")
    if maximum_bytes is not None and before.st_size > maximum_bytes:
        raise SyncAuthorityAuditError(f"input exceeds byte limit: {path.name}")
    try:
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise SyncAuthorityAuditError(f"cannot read input: {path.name}") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if len(payload) != before.st_size or before_identity != after_identity:
        raise SyncAuthorityAuditError(f"input changed while read: {path.name}")
    return payload


def _canonical_json_document(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyncAuthorityAuditError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise SyncAuthorityAuditError(f"{label} is not canonical JSON")
    return value


def _safe_zip(archive: zipfile.ZipFile, *, label: str) -> None:
    seen: set[str] = set()
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        normalized = path.as_posix()
        if (
            not normalized
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in info.filename
            or normalized in seen
        ):
            raise SyncAuthorityAuditError(f"{label} has unsafe or duplicate members")
        seen.add(normalized)


def _audit_tools_release(path: Path) -> dict[str, object]:
    member_payloads: dict[str, bytes] = {}
    if path.is_file():
        payload = _read_regular(path, maximum_bytes=2 * 1024 * 1024)
        if _hash_bytes(payload) != EXPECTED_TOOLS_ARCHIVE_SHA256:
            raise SyncAuthorityAuditError("VIDIMU-TOOLS archive SHA-256 mismatch")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            _safe_zip(archive, label="VIDIMU-TOOLS archive")
            if archive.comment.decode("ascii", errors="strict") != EXPECTED_TOOLS_COMMIT:
                raise SyncAuthorityAuditError("VIDIMU-TOOLS archive commit mismatch")
            roots = {
                PurePosixPath(info.filename).parts[0]
                for info in archive.infolist()
            }
            if len(roots) != 1:
                raise SyncAuthorityAuditError("VIDIMU-TOOLS archive root is ambiguous")
            root = next(iter(roots))
            files = {
                PurePosixPath(info.filename).relative_to(root).as_posix(): info
                for info in archive.infolist()
                if not info.is_dir()
            }
            if set(files) != set(_TOOLS_MEMBER_HASHES):
                raise SyncAuthorityAuditError(
                    "VIDIMU-TOOLS complete file topology mismatch"
                )
            for relative in _TOOLS_MEMBER_HASHES:
                member_payloads[relative] = archive.read(files[relative])
        release_kind = "PINNED_V1_0_ARCHIVE"
        archive_sha256: str | None = EXPECTED_TOOLS_ARCHIVE_SHA256
    else:
        raise SyncAuthorityAuditError(
            "VIDIMU-TOOLS input must be the pinned v1.0 archive"
        )

    observed = {name: _hash_bytes(value) for name, value in member_payloads.items()}
    if observed != _TOOLS_MEMBER_HASHES:
        raise SyncAuthorityAuditError("VIDIMU-TOOLS required member hashes mismatch")
    tree_sha256 = _hash_bytes(canonical_json_bytes(observed))
    if tree_sha256 != EXPECTED_TOOLS_MEMBER_MANIFEST_SHA256:
        raise SyncAuthorityAuditError("VIDIMU-TOOLS member manifest mismatch")
    return {
        "archive_sha256": archive_sha256,
        "commit": EXPECTED_TOOLS_COMMIT,
        "release_kind": release_kind,
        "required_member_hashes": observed,
        "required_member_manifest_sha256": tree_sha256,
        "authoritative_document_scope": {
            "complete_release_file_count": len(observed),
            "readme_sha256": observed["README.md"],
            "semantic_review_status": "PINNED_CONTENT_MANUALLY_REVIEWED",
        },
        "transform_contract": {
            "comparison_grid_hz": 30,
            "imu_input_hz": 50,
            "maximum_tested_shift_samples": 14,
            "selected_direction_count_per_record": 1,
            "selection_rule": "LOWER_DIRECTIONAL_RMSE",
            "log_behavior": "LOGS_EACH_POSITIVE_DIRECTIONAL_CANDIDATE",
            "mp4_transform": "SKIPPED",
            "csv_removed_rows": "cut_frames",
            "mot_removed_rows": "floor(cut_frames*50/30)",
            "raw_removed_rows": "floor(cut_frames*5*50/30)",
            "affine_drift_estimation": False,
        },
    }


def _fraction_decimal_6(value: Fraction) -> str:
    with localcontext() as context:
        context.prec = 50
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        return format(
            decimal_value.quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_EVEN
            ),
            ".6f",
        )


def _held_fraction_summary(
    streams: Sequence[tuple[Fraction, str, str, int, int]],
) -> dict[str, object]:
    if len(streams) != EXPECTED_RAW_STREAMS:
        raise SyncAuthorityAuditError("RAW stream count mismatch")
    ordered = sorted(streams, key=lambda item: (item[0], item[1], item[2]))

    def endpoint(
        item: tuple[Fraction, str, str, int, int],
    ) -> dict[str, object]:
        value, recording, sensor, numerator, denominator = item
        return {
            "decimal_6": _fraction_decimal_6(value),
            "denominator": value.denominator,
            "numerator": value.numerator,
            "recording_id": recording,
            "sensor_label": sensor,
            "unreduced_denominator": denominator,
            "unreduced_numerator": numerator,
        }

    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[middle][0]
        middle_streams = [endpoint(ordered[middle])]
    else:
        median = (ordered[middle - 1][0] + ordered[middle][0]) / 2
        middle_streams = [
            endpoint(ordered[middle - 1]),
            endpoint(ordered[middle]),
        ]
    summary = {
        "definition": (
            "PER_DYNAMIC_SENSOR_STREAM;(OBSERVATION_COUNT_MINUS_"
            "CONSECUTIVE_DISTINCT_EXACT_SOURCE_PAYLOAD_COUNT)/OBSERVATION_COUNT"
        ),
        "maximum": endpoint(ordered[-1]),
        "median": {
            "decimal_6": _fraction_decimal_6(median),
            "denominator": median.denominator,
            "middle_streams": middle_streams,
            "numerator": median.numerator,
        },
        "minimum": endpoint(ordered[0]),
        "stream_count": len(ordered),
    }
    observed = {
        key: str(summary[key]["decimal_6"])
        for key in ("minimum", "median", "maximum")
    }
    if observed != EXPECTED_HELD_FRACTION_DECIMAL_6:
        raise SyncAuthorityAuditError("RAW held-payload summary mismatch")
    return summary


def _audit_record_metadata(path: Path) -> dict[str, object]:
    payload = _read_regular(path, maximum_bytes=1_000_000)
    if _hash_bytes(payload) != EXPECTED_RECORD_METADATA_SHA256:
        raise SyncAuthorityAuditError("Zenodo record metadata SHA-256 mismatch")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyncAuthorityAuditError("Zenodo record metadata is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("id") != 15_075_076:
        raise SyncAuthorityAuditError("Zenodo record identity mismatch")
    metadata = value.get("metadata")
    files = value.get("files")
    if not isinstance(metadata, dict) or not isinstance(files, list):
        raise SyncAuthorityAuditError("Zenodo record metadata topology mismatch")
    if (
        value.get("doi") != "10.1038/s41597-023-02554-9"
        or value.get("conceptdoi") != "10.5281/zenodo.7681316"
        or metadata.get("version") != "v2.0.0"
    ):
        raise SyncAuthorityAuditError("Zenodo release contract mismatch")
    file_contract = {
        str(item.get("key")): {
            "checksum": item.get("checksum"),
            "size": item.get("size"),
        }
        for item in files
        if isinstance(item, dict)
    }
    expected = {
        "analysis.zip": {
            "checksum": f"md5:{EXPECTED_ANALYSIS_ARCHIVE_MD5}",
            "size": 71_650_185,
        },
        "dataset.zip": {
            "checksum": "md5:368d34d13651b44e6d4444c4a6c41380",
            "size": 336_819_642,
        },
    }
    if any(file_contract.get(name) != contract for name, contract in expected.items()):
        raise SyncAuthorityAuditError("Zenodo published file contract mismatch")
    return {
        "record_id": 15_075_076,
        "record_url": RECORD_URL,
        "sha256": EXPECTED_RECORD_METADATA_SHA256,
        "version": "v2.0.0",
    }


def _audit_article(path: Path) -> dict[str, object]:
    payload = _read_regular(path, maximum_bytes=10_000_000)
    if _hash_bytes(payload) != EXPECTED_ARTICLE_PDF_SHA256:
        raise SyncAuthorityAuditError("VIDIMU article PDF SHA-256 mismatch")
    if not payload.startswith(b"%PDF-"):
        raise SyncAuthorityAuditError("VIDIMU article input is not a PDF")
    return {
        "article_url": ARTICLE_URL,
        "doi": "10.1038/s41597-023-02554-9",
        "semantic_review_status": "PINNED_CONTENT_MANUALLY_REVIEWED",
        "sha256": EXPECTED_ARTICLE_PDF_SHA256,
        "size_bytes": len(payload),
    }


def _audit_v2_release(
    report_path: Path,
    parser_path: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    report_payload = _read_regular(report_path, maximum_bytes=2_000_000)
    if _hash_bytes(report_payload) != EXPECTED_V2_RELEASE_AUDIT_SHA256:
        raise SyncAuthorityAuditError("VIDIMU v2 release audit SHA-256 mismatch")
    report = _canonical_json_document(
        report_payload, label="VIDIMU v2 release audit"
    )
    parser_payload = _read_regular(parser_path, maximum_bytes=2_000_000)
    if _hash_bytes(parser_payload) != EXPECTED_SOURCE_PARSER_SHA256:
        raise SyncAuthorityAuditError("VIDIMU native parser SHA-256 mismatch")
    implementation = report.get("implementation")
    aggregates = report.get("aggregates")
    records = report.get("records")
    claim_boundary = report.get("claim_boundary")
    if (
        report.get("artifact_kind") != "VIDIMU_V2_RELEASE_AUDIT"
        or report.get("audit_status") != "PASS"
        or not isinstance(implementation, dict)
        or implementation.get("source_parser_sha256")
        != EXPECTED_SOURCE_PARSER_SHA256
        or not isinstance(aggregates, dict)
        or not isinstance(records, list)
        or not isinstance(claim_boundary, dict)
    ):
        raise SyncAuthorityAuditError("VIDIMU v2 release audit contract mismatch")
    if (
        aggregates.get("canonical_csv_raw_pair_count") != EXPECTED_RECORDINGS
        or aggregates.get("raw_source_row_count_including_npose")
        != EXPECTED_RAW_ROWS
        or aggregates.get("raw_dynamic_observation_row_count")
        != EXPECTED_DYNAMIC_RAW_ROWS
        or claim_boundary.get("held_fraction_interpretation")
        != "EXACT_CONSECUTIVE_RAW_SOURCE_PAYLOAD_REPETITION_NOT_SENSOR_SAMPLE_RATE"
    ):
        raise SyncAuthorityAuditError("VIDIMU v2 RAW aggregates mismatch")
    raw_by_record: dict[str, dict[str, object]] = {}
    stream_fractions: list[tuple[Fraction, str, str, int, int]] = []
    for record in records:
        if not isinstance(record, dict):
            raise SyncAuthorityAuditError("VIDIMU v2 record is malformed")
        recording = record.get("recording_id")
        sources = record.get("source_files")
        if not isinstance(recording, str) or not isinstance(sources, dict):
            raise SyncAuthorityAuditError("VIDIMU v2 record identity is malformed")
        raw = sources.get("quaternion_raw")
        if not isinstance(raw, dict) or recording in raw_by_record:
            raise SyncAuthorityAuditError("VIDIMU v2 RAW record is malformed")
        streams = raw.get("stream_statistics")
        if not isinstance(streams, list) or len(streams) != 5:
            raise SyncAuthorityAuditError("VIDIMU v2 RAW stream topology mismatch")
        observation_sum = 0
        for stream in streams:
            if not isinstance(stream, dict):
                raise SyncAuthorityAuditError("VIDIMU v2 RAW stream is malformed")
            numerator = stream.get("held_payload_observation_count")
            denominator = stream.get("observation_count")
            sensor = stream.get("sensor_label")
            if (
                not isinstance(numerator, int)
                or not isinstance(denominator, int)
                or denominator <= 0
                or not isinstance(sensor, str)
            ):
                raise SyncAuthorityAuditError("VIDIMU v2 RAW stream counts are invalid")
            observation_sum += denominator
            stream_fractions.append(
                (Fraction(numerator, denominator), recording, sensor, numerator, denominator)
            )
        row_count = raw.get("row_count_including_npose")
        if not isinstance(row_count, int) or row_count != observation_sum + 5:
            raise SyncAuthorityAuditError("VIDIMU v2 RAW row accounting mismatch")
        raw_by_record[recording] = {
            "member_path": raw.get("path"),
            "row_count_including_npose": row_count,
            "sha256": raw.get("sha256"),
            "size_bytes": raw.get("size_bytes"),
            "stream_statistics": streams,
        }
    if len(raw_by_record) != EXPECTED_RECORDINGS:
        raise SyncAuthorityAuditError("VIDIMU v2 RAW coverage mismatch")
    held = _held_fraction_summary(stream_fractions)
    if held != aggregates.get("held_observation_fraction"):
        raise SyncAuthorityAuditError("VIDIMU v2 held-payload evidence mismatch")
    return {
        "audit_sha256": EXPECTED_V2_RELEASE_AUDIT_SHA256,
        "raw_recording_count": len(raw_by_record),
        "raw_source_row_count_including_npose": sum(
            int(value["row_count_including_npose"])
            for value in raw_by_record.values()
        ),
        "source_parser_sha256": EXPECTED_SOURCE_PARSER_SHA256,
        "source_parser_version": implementation.get("source_parser_version"),
    }, raw_by_record


def _parse_plot_mappings(archive: zipfile.ZipFile) -> tuple[
    dict[str, dict[str, object]], dict[str, str]
]:
    plot_names = sorted(
        name
        for name in archive.namelist()
        if re.fullmatch(
            r"analysis/videoandimusync/A(?:0[1-9]|1[0-3])_.*_synchronize\.svg",
            name,
        )
    )
    if len(plot_names) != 13:
        raise SyncAuthorityAuditError("analysis archive must contain 13 sync plots")
    mappings: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for name in plot_names:
        payload = archive.read(name)
        hashes[name] = _hash_bytes(payload)
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SyncAuthorityAuditError("sync plot is not strict UTF-8") from exc
        activity = PurePosixPath(name).name[:3]
        for match in _PLOT_RE.finditer(text):
            recording = match.group("recording")
            imu_cut = int(match.group("imu"))
            video_cut = int(match.group("video"))
            if recording.split("_")[1] != activity or recording in mappings:
                raise SyncAuthorityAuditError("sync plot record identity is inconsistent")
            if imu_cut and video_cut or imu_cut > 14 or video_cut > 14:
                raise SyncAuthorityAuditError("sync plot has a noncanonical direction")
            direction = "IMU" if imu_cut else "VIDEO" if video_cut else "ZERO"
            mappings[recording] = {
                "plot_rmse_2dp": match.group("rmse"),
                "selected_direction": direction,
                "selected_imu_cut_frames": imu_cut,
                "selected_video_cut_frames": video_cut,
            }
    if len(mappings) != EXPECTED_RECORDINGS:
        raise SyncAuthorityAuditError("sync plots do not select exactly 208 records")
    return mappings, hashes


def _parse_info_rows(payload: bytes) -> dict[str, dict[str, dict[str, str]]]:
    try:
        text = payload.decode("utf-8", errors="strict")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise SyncAuthorityAuditError("infoToSync.csv is invalid") from exc
    fields = [
        "Subject",
        "Activity",
        "Trial",
        "File",
        "Type",
        "CutFrames",
        "OrigRmse",
        "TheoRmse",
    ]
    if reader.fieldnames != fields or len(rows) != EXPECTED_INFO_ROWS:
        raise SyncAuthorityAuditError("infoToSync.csv schema or row count mismatch")
    by_record: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    expected_names = {
        "csv": lambda recording: f"{recording}.csv",
        "mot": lambda recording: f"ik_{recording}.mot",
        "mp4": lambda recording: f"{recording}.mp4",
        "raw": lambda recording: f"{recording}.raw",
    }
    for row in rows:
        recording = f"{row['Subject']}_{row['Activity']}_{row['Trial']}"
        kind = row["Type"]
        if _RECORDING_RE.fullmatch(recording) is None or kind not in expected_names:
            raise SyncAuthorityAuditError("infoToSync.csv contains an invalid identity")
        if PureWindowsPath(row["File"]).name != expected_names[kind](recording):
            raise SyncAuthorityAuditError("infoToSync.csv file identity is inconsistent")
        try:
            cut = int(row["CutFrames"])
            original_rmse = float(row["OrigRmse"])
            theoretical_rmse = float(row["TheoRmse"])
        except ValueError as exc:
            raise SyncAuthorityAuditError("infoToSync.csv contains invalid numbers") from exc
        if not 1 <= cut <= 14 or not all(
            math.isfinite(value) for value in (original_rmse, theoretical_rmse)
        ):
            raise SyncAuthorityAuditError("infoToSync.csv numbers are out of range")
        if kind in by_record[recording]:
            raise SyncAuthorityAuditError("infoToSync.csv contains duplicate rows")
        by_record[recording][kind] = row
    return dict(by_record)


def _reconcile_plots_and_info(
    plots: Mapping[str, Mapping[str, object]],
    info: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> tuple[dict[str, object], dict[str, tuple[str, ...]]]:
    conflicting_types: dict[str, tuple[str, ...]] = {}
    for recording, mapping in plots.items():
        rows = info.get(recording, {})
        direction = mapping["selected_direction"]
        selected_types: set[str]
        if direction == "VIDEO":
            selected_types = {"csv", "mp4"}
            expected_cut = mapping["selected_video_cut_frames"]
        elif direction == "IMU":
            selected_types = {"mot", "raw"}
            expected_cut = mapping["selected_imu_cut_frames"]
        else:
            selected_types = set()
            expected_cut = 0
        if not selected_types.issubset(rows) or direction == "ZERO" and rows:
            raise SyncAuthorityAuditError("plot selections do not reconcile with info rows")
        for kind in selected_types:
            if int(rows[kind]["CutFrames"]) != expected_cut:
                raise SyncAuthorityAuditError("selected plot and info cut counts differ")
        extras = tuple(sorted(set(rows) - selected_types))
        if extras:
            conflicting_types[recording] = extras
    if set(info) - set(plots):
        raise SyncAuthorityAuditError("infoToSync.csv contains records absent from plots")
    if conflicting_types != {
        "S53_A13_T03": ("mot", "raw"),
        "S57_A07_T01": ("mot", "raw"),
    }:
        raise SyncAuthorityAuditError("unexpected dual-direction candidates")
    return {
        "plot_selected_direction_counts": dict(sorted(Counter(
            str(value["selected_direction"]) for value in plots.values()
        ).items())),
        "info_recording_count": len(info),
        "info_row_count": sum(len(value) for value in info.values()),
        "info_type_counts": dict(sorted(Counter(
            kind for rows in info.values() for kind in rows
        ).items())),
        "dual_direction_candidate_record_count": len(conflicting_types),
        "dual_direction_candidate_row_count": sum(
            len(value) for value in conflicting_types.values()
        ),
    }, conflicting_types


def _prefix_cut(
    base: bytes,
    synchronized: bytes,
    *,
    retained_prefix_lines: int,
) -> int:
    base_lines = base.splitlines(keepends=True)
    sync_lines = synchronized.splitlines(keepends=True)
    removed = len(base_lines) - len(sync_lines)
    if removed < 1 or (
        base_lines[:retained_prefix_lines]
        + base_lines[retained_prefix_lines + removed :]
        != sync_lines
    ):
        raise SyncAuthorityAuditError("sync override is not the exact source prefix cut")
    return removed


def _validate_sync_subtree_paths(
    sync_directories: set[str],
    sync_files: set[str],
    info: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> dict[str, object]:
    expected_subjects = {recording.split("_")[0] for recording in info}
    expected_directories = {
        "dataset/videoandimusync/",
        *(f"dataset/videoandimusync/{subject}/" for subject in expected_subjects),
    }
    expected_files = {
        (
            f"dataset/videoandimusync/{recording.split('_')[0]}/"
            f"{'ik_' if kind == 'mot' else ''}{recording}.{kind}"
        )
        for recording, rows in info.items()
        for kind in rows
        if kind != "mp4"
    }
    if sync_directories != expected_directories:
        raise SyncAuthorityAuditError("dataset sync directory topology mismatch")
    if sync_files != expected_files:
        raise SyncAuthorityAuditError(
            "dataset sync subtree differs from the exact source instruction set"
        )
    file_hash = _hash_bytes(canonical_json_bytes(sorted(sync_files)))
    directory_hash = _hash_bytes(canonical_json_bytes(sorted(sync_directories)))
    all_hash = _hash_bytes(
        canonical_json_bytes(sorted([*sync_directories, *sync_files]))
    )
    if (
        file_hash != EXPECTED_SYNC_FILE_PATHS_SHA256
        or directory_hash != EXPECTED_SYNC_DIRECTORY_PATHS_SHA256
        or all_hash != EXPECTED_SYNC_ALL_PATHS_SHA256
    ):
        raise SyncAuthorityAuditError("dataset sync path manifest mismatch")
    return {
        "sync_all_paths_sha256": all_hash,
        "sync_directory_paths_sha256": directory_hash,
        "sync_file_paths_sha256": file_hash,
    }


def _audit_dataset_sync(
    dataset_payload: bytes,
    info: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> tuple[dict[str, object], dict[str, dict[str, dict[str, object]]]]:
    overrides_by_record: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    entries: list[dict[str, object]] = []
    raw_removed: Counter[int] = Counter()
    with zipfile.ZipFile(io.BytesIO(dataset_payload)) as archive:
        _safe_zip(archive, label="dataset archive")
        sync_directories = {
            item.filename
            for item in archive.infolist()
            if item.filename.startswith("dataset/videoandimusync/")
            and item.is_dir()
        }
        sync_members = []
        sync_files = [
            item.filename
            for item in archive.infolist()
            if item.filename.startswith("dataset/videoandimusync/")
            and not item.is_dir()
        ]
        sync_path_evidence = _validate_sync_subtree_paths(
            sync_directories,
            set(sync_files),
            info,
        )
        for name in sync_files:
            match = _SYNC_MEMBER_RE.fullmatch(name)
            if match is None:
                raise SyncAuthorityAuditError(
                    "dataset sync subtree contains an unmatched file"
                )
            recording = match.group("recording")
            if match.group("subject") != recording.split("_")[0]:
                raise SyncAuthorityAuditError("sync override is in the wrong subject folder")
            kind = "mot" if match.group("prefix") else match.group("extension")
            sync_members.append((name, recording, kind))
        if len(sync_members) != EXPECTED_SYNC_OVERRIDES:
            raise SyncAuthorityAuditError("dataset sync override count mismatch")
        for name, recording, kind in sorted(sync_members):
            subject = recording.split("_")[0]
            base_name = (
                f"dataset/videoandimus/{subject}/ik_{recording}.mot"
                if kind == "mot"
                else f"dataset/videoandimus/{subject}/{recording}.{kind}"
            )
            try:
                base = archive.read(base_name)
                synchronized = archive.read(name)
            except KeyError as exc:
                raise SyncAuthorityAuditError("sync override has no exact base file") from exc
            retained = 8 if kind == "mot" else 1
            removed = _prefix_cut(
                base, synchronized, retained_prefix_lines=retained
            )
            row = info.get(recording, {}).get(kind)
            if row is None:
                raise SyncAuthorityAuditError("sync override has no infoToSync row")
            cut_frames = int(row["CutFrames"])
            expected_removed = (
                cut_frames
                if kind == "csv"
                else (cut_frames * 5) // 3
                if kind == "mot"
                else (cut_frames * 25) // 3
            )
            if removed != expected_removed:
                raise SyncAuthorityAuditError("sync override contradicts source transform")
            if kind in overrides_by_record[recording]:
                raise SyncAuthorityAuditError("duplicate sync override type")
            override_evidence = {
                "cut_frames": cut_frames,
                "path": name,
                "removed_data_lines": removed,
                "sha256": _hash_bytes(synchronized),
                "source_sha256": _hash_bytes(base),
                "type": kind,
            }
            overrides_by_record[recording][kind] = override_evidence
            if kind == "raw":
                raw_removed[removed] += 1
            entries.append(override_evidence)
        base_sto_recordings = {
            match.group("recording")
            for name in archive.namelist()
            if (match := re.fullmatch(
                r"dataset/videoandimus/S[0-9]{2}/"
                r"(?P<recording>S[0-9]{2}_A(?:0[1-9]|1[0-3])_T[0-9]{2})\.sto",
                name,
            )) is not None
        }
    type_counts = dict(sorted(Counter(
        kind for values in overrides_by_record.values() for kind in values
    ).items()))
    if (
        type_counts != EXPECTED_OVERRIDE_TYPE_COUNTS
        or len(base_sto_recordings) != EXPECTED_RECORDINGS
    ):
        raise SyncAuthorityAuditError("dataset companion topology mismatch")
    raw_override_count = sum(raw_removed.values())
    removed_source_data_rows = sum(
        removed * count for removed, count in raw_removed.items()
    )
    npose_rows_removed = raw_override_count * 5
    return {
        "base_sto_companion_count": len(base_sto_recordings),
        "complete_sync_subtree_file_count": len(sync_files),
        "complete_sync_subtree_matched_file_count": len(sync_members),
        "complete_sync_subtree_entry_count": (
            len(sync_files) + len(sync_directories)
        ),
        "sync_directory_count": len(sync_directories),
        **sync_path_evidence,
        "override_count": len(entries),
        "override_manifest_sha256": _hash_bytes(canonical_json_bytes(entries)),
        "override_recording_count": len(overrides_by_record),
        "override_type_counts": type_counts,
        "raw_override_defects": {
            "npose_rows_removed_total": npose_rows_removed,
            "partial_five_sensor_cycle_override_count": sum(
                count for removed, count in raw_removed.items() if removed % 5
            ),
            "raw_overrides_removing_all_npose_rows_count": raw_override_count,
            "raw_override_count": raw_override_count,
            "removed_line_histogram": {
                str(key): value for key, value in sorted(raw_removed.items())
            },
            "removed_dynamic_observation_row_count": (
                removed_source_data_rows - npose_rows_removed
            ),
            "removed_source_data_row_count": removed_source_data_rows,
            "strict_native_sample_status": "UNUSABLE_AS_CANONICAL_RAW_SAMPLE_TABLE",
        },
    }, dict(overrides_by_record)


def _applied_direction_ambiguities(
    plots: Mapping[str, Mapping[str, object]],
    info: Mapping[str, Mapping[str, Mapping[str, str]]],
    overrides: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> list[dict[str, object]]:
    ambiguities: list[dict[str, object]] = []
    for recording in sorted(plots):
        plot = plots[recording]
        rows = info.get(recording, {})
        applied = overrides.get(recording, {})
        if plot.get("selected_direction") != "VIDEO" or not {
            "csv",
            "mot",
            "raw",
        }.issubset(applied):
            continue
        if not {"csv", "mp4", "mot", "raw"}.issubset(rows):
            raise SyncAuthorityAuditError(
                "dual-direction override lacks both source evidence paths"
            )
        ambiguity = {
            "applied_imu_cut_frames": int(rows["raw"]["CutFrames"]),
            "applied_mot_removed_rows": int(applied["mot"]["removed_data_lines"]),
            "applied_override_types": sorted(applied),
            "applied_raw_removed_rows": int(applied["raw"]["removed_data_lines"]),
            "applied_video_cut_frames": int(rows["csv"]["CutFrames"]),
            "ambiguity_type": "DUAL_DIRECTION_APPLIED_OVERRIDE",
            "pairing_status": "AMBIGUOUS_SOURCE_MAPPING",
            "plot_selected_direction": "VIDEO",
            "recording_id": recording,
        }
        ambiguities.append(ambiguity)
    observed = {
        str(value["recording_id"]): {
            key: value[key]
            for key in (
                "applied_imu_cut_frames",
                "applied_mot_removed_rows",
                "applied_raw_removed_rows",
                "applied_video_cut_frames",
            )
        }
        for value in ambiguities
    }
    if observed != EXPECTED_DUAL_DIRECTION_RECORDS:
        raise SyncAuthorityAuditError("released dual-direction ambiguity mismatch")
    return ambiguities


def _raw_timestamp_key(token: bytes, *, recording: str) -> tuple[int, int]:
    if _RAW_TIMESTAMP_RE.fullmatch(token) is None:
        raise SyncAuthorityAuditError(
            f"RAW timestamp token is invalid for {recording}"
        )
    whole, separator, fraction = token.partition(b".")
    return int(whole), int(fraction.ljust(9, b"0") if separator else b"0")


def _audit_raw_payload(
    payload: bytes,
    *,
    recording: str,
) -> dict[str, object]:
    lines = payload.splitlines()
    if not lines or lines[0] != b"QUAT,w,x,y,z,timestamp":
        raise SyncAuthorityAuditError(f"RAW header mismatch for {recording}")
    rows = lines[1:]
    activity = int(recording.split("_")[1][1:])
    layout = _LOWER_BODY_LAYOUT if activity <= 4 else _UPPER_BODY_LAYOUT
    if len(rows) < 10 or (len(rows) - 5) % 5:
        raise SyncAuthorityAuditError(f"RAW five-sensor topology mismatch for {recording}")

    split_rows: list[tuple[bytes, ...]] = []
    for row in rows:
        fields = tuple(row.split(b","))
        if len(fields) != 6 or any(not value for value in fields):
            raise SyncAuthorityAuditError(f"RAW row schema mismatch for {recording}")
        split_rows.append(fields)
    expected_labels = tuple(value.encode("ascii") for value in layout)
    if tuple(row[0] for row in split_rows[:5]) != expected_labels:
        raise SyncAuthorityAuditError(f"RAW N-pose layout mismatch for {recording}")
    calibration_keys = {
        _raw_timestamp_key(row[5], recording=recording)
        for row in split_rows[:5]
    }
    if len(calibration_keys) != 1:
        raise SyncAuthorityAuditError(
            f"RAW N-pose timestamp is not shared for {recording}"
        )
    calibration_key = next(iter(calibration_keys))

    previous_times: list[tuple[int, int] | None] = [None] * 5
    previous_payloads: list[tuple[bytes, ...] | None] = [None] * 5
    observation_counts = [0] * 5
    held_counts = [0] * 5
    duplicate_counts = [0] * 5
    reversal_counts = [0] * 5
    first_tokens: list[str | None] = [None] * 5
    last_tokens: list[str | None] = [None] * 5
    for offset, row in enumerate(split_rows[5:]):
        sensor_index = offset % 5
        if row[0] != expected_labels[sensor_index]:
            raise SyncAuthorityAuditError(
                f"RAW dynamic sensor cycle mismatch for {recording}"
            )
        timestamp = _raw_timestamp_key(row[5], recording=recording)
        previous_time = previous_times[sensor_index]
        if previous_time is not None:
            if timestamp == previous_time:
                duplicate_counts[sensor_index] += 1
            elif timestamp < previous_time:
                reversal_counts[sensor_index] += 1
        payload_tokens = row[1:5]
        if previous_payloads[sensor_index] == payload_tokens:
            held_counts[sensor_index] += 1
        token = row[5].decode("ascii", errors="strict")
        if first_tokens[sensor_index] is None:
            first_tokens[sensor_index] = token
        last_tokens[sensor_index] = token
        observation_counts[sensor_index] += 1
        previous_times[sensor_index] = timestamp
        previous_payloads[sensor_index] = payload_tokens

    streams = []
    for index, sensor in enumerate(layout):
        count = observation_counts[index]
        if count <= 0 or first_tokens[index] is None or last_tokens[index] is None:
            raise SyncAuthorityAuditError(f"RAW stream is empty for {recording}")
        if _raw_timestamp_key(
            first_tokens[index].encode("ascii"), recording=recording
        ) <= calibration_key:
            raise SyncAuthorityAuditError(
                f"RAW dynamic observations do not begin after N-pose for {recording}"
            )
        streams.append(
            {
                "consecutive_distinct_payload_count": count - held_counts[index],
                "duplicate_timestamp_count": duplicate_counts[index],
                "first_source_timestamp_token": first_tokens[index],
                "held_payload_observation_count": held_counts[index],
                "last_source_timestamp_token": last_tokens[index],
                "observation_count": count,
                "sensor_label": sensor,
                "timestamp_reversal_count": reversal_counts[index],
            }
        )
    return {
        "npose_timestamp_class": (
            "EXACT_NUMERIC_ZERO"
            if calibration_key == (0, 0)
            else "NONZERO_DECIMAL_TOKEN_WITH_UNRESOLVED_CLOCK"
        ),
        "row_count_including_npose": len(rows),
        "sha256": _hash_bytes(payload),
        "size_bytes": len(payload),
        "stream_statistics": streams,
    }


def _partition_inventory_references(
    references: Sequence[object],
) -> dict[str, list[dict[str, object]]]:
    expected_modalities = {
        "BODYTRACK_POSE",
        "INERTIAL_QUATERNION",
        "VISUAL",
    }
    partitions: dict[str, list[dict[str, object]]] = {
        key: [] for key in expected_modalities
    }
    for value in references:
        if not isinstance(value, dict) or value.get("modality") not in partitions:
            raise SyncAuthorityAuditError("source inventory reference is malformed")
        partitions[str(value["modality"])].append(value)
    if any(len(values) != EXPECTED_RECORDINGS for values in partitions.values()):
        raise SyncAuthorityAuditError("source inventory modality topology mismatch")
    identity_sets = []
    for values in partitions.values():
        identities = [value.get("recording_id") for value in values]
        if (
            any(not isinstance(value, str) for value in identities)
            or len(set(identities)) != EXPECTED_RECORDINGS
        ):
            raise SyncAuthorityAuditError(
                "source inventory recording identity is malformed or duplicated"
            )
        identity_sets.append(set(identities))
    if any(value != identity_sets[0] for value in identity_sets[1:]):
        raise SyncAuthorityAuditError("source inventory modalities are not bijective")
    return partitions


def _audit_snapshot(snapshot_root: Path) -> tuple[
    dict[str, object], bytes, dict[str, int], dict[str, dict[str, object]]
]:
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise SyncAuthorityAuditError("snapshot root must be a real directory")
    root = snapshot_root.resolve(strict=True)
    if root.name != EXPECTED_SNAPSHOT_SHA256:
        raise SyncAuthorityAuditError("snapshot root is not the frozen v0.4 snapshot")
    manifest_payload = _read_regular(root / "snapshot_manifest.json", maximum_bytes=2_000_000)
    if _hash_bytes(manifest_payload) != EXPECTED_SNAPSHOT_SHA256:
        raise SyncAuthorityAuditError("snapshot content address mismatch")
    manifest = _canonical_json_document(manifest_payload, label="snapshot manifest")
    success = _canonical_json_document(
        _read_regular(root / "_SUCCESS", maximum_bytes=4096), label="snapshot marker"
    )
    if success.get("snapshot_manifest_sha256") != EXPECTED_SNAPSHOT_SHA256:
        raise SyncAuthorityAuditError("snapshot marker mismatch")
    inventory_payload = _read_regular(root / "source_inventory.json", maximum_bytes=1_000_000)
    if _hash_bytes(inventory_payload) != EXPECTED_SOURCE_INVENTORY_SHA256:
        raise SyncAuthorityAuditError("source inventory SHA-256 mismatch")
    inventory = _canonical_json_document(inventory_payload, label="source inventory")
    references = inventory.get("asset_references")
    if not isinstance(references, list) or len(references) != EXPECTED_ASSET_REFERENCES:
        raise SyncAuthorityAuditError("source inventory asset count mismatch")
    partitions = _partition_inventory_references(references)
    pose_refs = partitions["BODYTRACK_POSE"]
    raw_refs = partitions["INERTIAL_QUATERNION"]
    dataset_path = root / "objects" / EXPECTED_DATASET_ARCHIVE_SHA256
    dataset_payload = _read_regular(dataset_path, maximum_bytes=400_000_000)
    if _hash_bytes(dataset_payload) != EXPECTED_DATASET_ARCHIVE_SHA256:
        raise SyncAuthorityAuditError("dataset archive SHA-256 mismatch")
    pose_counts: dict[str, int] = {}
    raw_records: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(io.BytesIO(dataset_payload)) as archive:
        _safe_zip(archive, label="dataset archive")
        for reference in pose_refs:
            member = reference.get("archive_member_path")
            recording = reference.get("recording_id")
            expected_hash = reference.get("expected_sha256")
            if not all(isinstance(value, str) for value in (member, recording, expected_hash)):
                raise SyncAuthorityAuditError("pose reference is malformed")
            assert isinstance(member, str) and isinstance(recording, str)
            payload = archive.read(member)
            if _hash_bytes(payload) != expected_hash:
                raise SyncAuthorityAuditError("pose asset hash mismatch")
            lines = payload.splitlines()
            if len(lines) < 2:
                raise SyncAuthorityAuditError("pose CSV is empty")
            if recording in pose_counts:
                raise SyncAuthorityAuditError("pose recording identity is duplicated")
            pose_counts[recording] = len(lines) - 1
        for reference in raw_refs:
            member = reference.get("archive_member_path")
            recording = reference.get("recording_id")
            expected_hash = reference.get("expected_sha256")
            expected_size = reference.get("expected_size_bytes")
            if (
                not isinstance(member, str)
                or not isinstance(recording, str)
                or not isinstance(expected_hash, str)
                or not isinstance(expected_size, int)
                or recording in raw_records
            ):
                raise SyncAuthorityAuditError("RAW reference is malformed")
            try:
                raw_payload = archive.read(member)
            except KeyError as exc:
                raise SyncAuthorityAuditError("RAW referenced asset is missing") from exc
            if len(raw_payload) != expected_size or _hash_bytes(raw_payload) != expected_hash:
                raise SyncAuthorityAuditError("RAW asset hash or size mismatch")
            raw_records[recording] = _audit_raw_payload(
                raw_payload, recording=recording
            )
            raw_records[recording]["member_path"] = member
    if len(pose_counts) != EXPECTED_RECORDINGS:
        raise SyncAuthorityAuditError("pose recording identities are not unique")
    if set(raw_records) != set(pose_counts):
        raise SyncAuthorityAuditError("RAW and pose recording identities differ")

    raw_rows = sum(
        int(value["row_count_including_npose"])
        for value in raw_records.values()
    )
    raw_bytes = sum(int(value["size_bytes"]) for value in raw_records.values())
    raw_streams: list[tuple[Fraction, str, str, int, int]] = []
    duplicate_timestamps = 0
    timestamp_reversals = 0
    held_payload_rows = 0
    distinct_payload_runs = 0
    zero_npose = 0
    for recording, raw in raw_records.items():
        if raw["npose_timestamp_class"] == "EXACT_NUMERIC_ZERO":
            zero_npose += 1
        streams = raw["stream_statistics"]
        assert isinstance(streams, list)
        for stream in streams:
            assert isinstance(stream, dict)
            numerator = int(stream["held_payload_observation_count"])
            denominator = int(stream["observation_count"])
            held_payload_rows += numerator
            distinct_payload_runs += int(
                stream["consecutive_distinct_payload_count"]
            )
            raw_streams.append(
                (
                    Fraction(numerator, denominator),
                    recording,
                    str(stream["sensor_label"]),
                    numerator,
                    denominator,
                )
            )
            duplicate_timestamps += int(stream["duplicate_timestamp_count"])
            timestamp_reversals += int(stream["timestamp_reversal_count"])
    if (
        raw_rows != EXPECTED_RAW_ROWS
        or raw_bytes != EXPECTED_RAW_BYTES
        or raw_rows - EXPECTED_RECORDINGS * 5 != EXPECTED_DYNAMIC_RAW_ROWS
        or held_payload_rows != EXPECTED_HELD_PAYLOAD_ROWS
        or distinct_payload_runs != EXPECTED_DISTINCT_PAYLOAD_RUNS
        or duplicate_timestamps
        or timestamp_reversals
        or zero_npose != 197
    ):
        raise SyncAuthorityAuditError("RAW all-record timing topology mismatch")
    held_summary = _held_fraction_summary(raw_streams)
    claim = manifest.get("inventory")
    if not isinstance(claim, dict) or claim.get("recording_count") != 208 \
            or claim.get("asset_reference_count") != 624:
        raise SyncAuthorityAuditError("snapshot inventory claims mismatch")
    return {
        "asset_reference_count": len(references),
        "dataset_archive_sha256": EXPECTED_DATASET_ARCHIVE_SHA256,
        "independent_raw_scan": {
            "dynamic_observation_row_count": raw_rows - EXPECTED_RECORDINGS * 5,
            "dynamic_complete_five_sensor_cycle_count": (
                (raw_rows - EXPECTED_RECORDINGS * 5) // 5
            ),
            "exact_consecutive_held_payload_row_count": held_payload_rows,
            "exact_consecutive_distinct_payload_run_count": distinct_payload_runs,
            "aggregate_held_payload_fraction": {
                "decimal_6": _fraction_decimal_6(
                    Fraction(held_payload_rows, EXPECTED_DYNAMIC_RAW_ROWS)
                ),
                "denominator": EXPECTED_DYNAMIC_RAW_ROWS,
                "numerator": held_payload_rows,
            },
            "held_observation_fraction": held_summary,
            "source_timestamp_token_duplicate_count": duplicate_timestamps,
            "source_timestamp_token_reversal_count": timestamp_reversals,
            "npose_nonzero_timestamp_record_count": EXPECTED_RECORDINGS - zero_npose,
            "npose_zero_timestamp_record_count": zero_npose,
            "raw_asset_manifest_sha256": _hash_bytes(canonical_json_bytes([
                {
                    "member_path": raw_records[recording]["member_path"],
                    "recording_id": recording,
                    "row_count_including_npose": raw_records[recording][
                        "row_count_including_npose"
                    ],
                    "sha256": raw_records[recording]["sha256"],
                    "size_bytes": raw_records[recording]["size_bytes"],
                }
                for recording in sorted(raw_records)
            ])),
            "raw_recording_count": len(raw_records),
            "raw_source_bytes": raw_bytes,
            "raw_source_row_count_including_npose": raw_rows,
            "stream_count": len(raw_streams),
            "timestamp_unit_interpreted_by_audit": False,
        },
        "recording_count": len(pose_counts),
        "snapshot_manifest_sha256": EXPECTED_SNAPSHOT_SHA256,
        "source_inventory_sha256": EXPECTED_SOURCE_INVENTORY_SHA256,
    }, dataset_payload, pose_counts, raw_records


def _audit_v04_report(path: Path) -> tuple[dict[str, object], dict[str, int]]:
    payload = _read_regular(path, maximum_bytes=2_000_000)
    if _hash_bytes(payload) != EXPECTED_V04_RELEASE_AUDIT_SHA256:
        raise SyncAuthorityAuditError("v0.4 release audit SHA-256 mismatch")
    value = _canonical_json_document(payload, label="v0.4 release audit")
    if (
        value.get("artifact_kind") != "TREMORA_VIDIMU_V04_GATE_B_RELEASE_AUDIT"
        or value.get("overall_verdict") != "PASS"
        or value.get("recording_count") != EXPECTED_RECORDINGS
        or value.get("source_snapshot_sha256") != EXPECTED_SNAPSHOT_SHA256
        or value.get("source_inventory_sha256")
        != EXPECTED_SOURCE_INVENTORY_SHA256
    ):
        raise SyncAuthorityAuditError("v0.4 release audit is not the frozen PASS")
    per_record = value.get("per_record")
    if not isinstance(per_record, list):
        raise SyncAuthorityAuditError("v0.4 per-record evidence is missing")
    counts: dict[str, int] = {}
    for row in per_record:
        if not isinstance(row, dict):
            raise SyncAuthorityAuditError("v0.4 per-record row is invalid")
        recording = row.get("recording_id")
        count = row.get("decoded_frames")
        if not isinstance(recording, str) or not isinstance(count, int) \
                or recording in counts:
            raise SyncAuthorityAuditError("v0.4 frame count identity is invalid")
        counts[recording] = count
    if len(counts) != EXPECTED_RECORDINGS:
        raise SyncAuthorityAuditError("v0.4 frame count coverage mismatch")
    return {
        "decoded_frame_count": sum(counts.values()),
        "release_audit_sha256": EXPECTED_V04_RELEASE_AUDIT_SHA256,
        "valid_pts_frame_count": value.get("primary_run_reconciliation", {}).get(
            "valid_pts_frames"
        ) if isinstance(value.get("primary_run_reconciliation"), dict) else None,
    }, counts


def _reconcile_raw_evidence(
    current: Mapping[str, Mapping[str, object]],
    prior: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if set(current) != set(prior) or len(current) != EXPECTED_RECORDINGS:
        raise SyncAuthorityAuditError("independent RAW audit coverage differs")
    rows = []
    for recording in sorted(current):
        observed = current[recording]
        previous = prior[recording]
        for key in ("member_path", "row_count_including_npose", "sha256", "size_bytes"):
            if observed.get(key) != previous.get(key):
                raise SyncAuthorityAuditError(
                    f"independent RAW evidence differs for {recording}"
                )
        observed_streams = observed.get("stream_statistics")
        previous_streams = previous.get("stream_statistics")
        if not isinstance(observed_streams, list) or not isinstance(
            previous_streams, list
        ):
            raise SyncAuthorityAuditError("independent RAW streams are malformed")
        previous_by_sensor = {
            str(value.get("sensor_label")): value
            for value in previous_streams
            if isinstance(value, dict)
        }
        if len(previous_by_sensor) != 5 or len(observed_streams) != 5:
            raise SyncAuthorityAuditError("independent RAW stream topology differs")
        for stream in observed_streams:
            if not isinstance(stream, dict):
                raise SyncAuthorityAuditError("observed RAW stream is malformed")
            sensor = str(stream.get("sensor_label"))
            reference = previous_by_sensor.get(sensor)
            if reference is None:
                raise SyncAuthorityAuditError("independent RAW sensor labels differ")
            for key in (
                "consecutive_distinct_payload_count",
                "first_source_timestamp_token",
                "held_payload_observation_count",
                "last_source_timestamp_token",
                "observation_count",
            ):
                if stream.get(key) != reference.get(key):
                    raise SyncAuthorityAuditError(
                        f"independent RAW stream evidence differs for {recording}"
                    )
        rows.append(
            {
                "recording_id": recording,
                "row_count_including_npose": observed["row_count_including_npose"],
                "sha256": observed["sha256"],
            }
        )
    return {
        "all_record_hashes_equal": True,
        "all_record_row_counts_equal": True,
        "all_stream_statistics_equal": True,
        "raw_recording_count": len(rows),
        "reconciled_manifest_sha256": _hash_bytes(canonical_json_bytes(rows)),
    }


def audit_vidimu_v05_sync_authority(
    snapshot_root: str | Path,
    analysis_archive: str | Path,
    tools_release: str | Path,
    v04_release_audit: str | Path,
    v2_release_audit: str | Path,
    source_parser: str | Path,
    article_pdf: str | Path,
    record_metadata: str | Path,
) -> dict[str, object]:
    """Reconcile released synchronization evidence and return an honest NO-GO."""

    script_path = Path(__file__).resolve(strict=True)
    initial_script_sha256 = _hash_bytes(_read_regular(script_path))
    snapshot_evidence, dataset_payload, pose_counts, raw_records = _audit_snapshot(
        Path(snapshot_root)
    )
    v04_evidence, frame_counts = _audit_v04_report(Path(v04_release_audit))
    if pose_counts != frame_counts or sum(pose_counts.values()) != EXPECTED_POSE_AND_FRAME_ROWS:
        raise SyncAuthorityAuditError("v0.4 CSV/frame counts do not exactly reconcile")

    analysis_payload = _read_regular(Path(analysis_archive), maximum_bytes=100_000_000)
    if _hash_bytes(analysis_payload) != EXPECTED_ANALYSIS_ARCHIVE_SHA256:
        raise SyncAuthorityAuditError("analysis archive SHA-256 mismatch")
    if hashlib.md5(analysis_payload, usedforsecurity=False).hexdigest() \
            != EXPECTED_ANALYSIS_ARCHIVE_MD5:
        raise SyncAuthorityAuditError("analysis archive published MD5 mismatch")
    with zipfile.ZipFile(io.BytesIO(analysis_payload)) as archive:
        _safe_zip(archive, label="analysis archive")
        plots, plot_hashes = _parse_plot_mappings(archive)
        info_payload = archive.read("analysis/videoandimusync/infoToSync.csv")
    info = _parse_info_rows(info_payload)
    plot_info, conflicting = _reconcile_plots_and_info(plots, info)
    if plot_info["plot_selected_direction_counts"] != EXPECTED_DIRECTION_COUNTS \
            or plot_info["info_type_counts"] != EXPECTED_INFO_TYPE_COUNTS:
        raise SyncAuthorityAuditError("published synchronization counts mismatch")
    if set(plots) != set(pose_counts):
        raise SyncAuthorityAuditError("synchronization and snapshot coverage differ")
    dataset_sync, override_types = _audit_dataset_sync(dataset_payload, info)
    tools_evidence = _audit_tools_release(Path(tools_release))
    article_evidence = _audit_article(Path(article_pdf))
    record_evidence = _audit_record_metadata(Path(record_metadata))
    v2_evidence, prior_raw_records = _audit_v2_release(
        Path(v2_release_audit), Path(source_parser)
    )
    raw_reconciliation = _reconcile_raw_evidence(raw_records, prior_raw_records)
    ambiguities = _applied_direction_ambiguities(plots, info, override_types)
    ambiguous_ids = {
        str(value["recording_id"])
        for value in ambiguities
    }
    mapping_classification_counts = {
        "AMBIGUOUS": len(ambiguous_ids),
        "IMU": sum(
            value["selected_direction"] == "IMU"
            for key, value in plots.items()
            if key not in ambiguous_ids
        ),
        "VIDEO": sum(
            value["selected_direction"] == "VIDEO"
            for key, value in plots.items()
            if key not in ambiguous_ids
        ),
        "ZERO": sum(
            value["selected_direction"] == "ZERO"
            for key, value in plots.items()
            if key not in ambiguous_ids
        ),
    }
    if mapping_classification_counts != {
        "AMBIGUOUS": 2,
        "IMU": 32,
        "VIDEO": 147,
        "ZERO": 27,
    }:
        raise SyncAuthorityAuditError("source mapping classification mismatch")

    records = []
    for recording in sorted(plots):
        record = {"recording_id": recording, **plots[recording]}
        record.update({
            "conflicting_logged_types": list(conflicting.get(recording, ())),
            "decoded_frame_count": frame_counts[recording],
            "info_types": sorted(info.get(recording, {})),
            "pose_csv_row_count": pose_counts[recording],
            "raw_source_row_count": raw_records[recording][
                "row_count_including_npose"
            ],
            "raw_source_sha256": raw_records[recording]["sha256"],
            "source_mapping_status": (
                "AMBIGUOUS_SOURCE_MAPPING"
                if recording in ambiguous_ids
                else "SOURCE_CROP_DIRECTION_RECONCILED_CLOCK_UNRESOLVED"
            ),
            "sync_override_types": sorted(override_types.get(recording, {})),
        })
        records.append(record)

    result: dict[str, object] = {
        "artifact_kind": ARTIFACT_KIND,
        "audit_execution_status": AUDIT_PASS,
        "blockers": [
            {
                "blocker_id": "RAW_TIMESTAMP_UNIT_NOT_DOCUMENTED_IN_PINNED_AUTHORITY",
                "evidence": (
                    "The audited article, Zenodo metadata, complete VIDIMU-TOOLS "
                    "v1.0 tree, and native-parser evidence contain no authoritative "
                    "RAW decimal-token unit or integer-tick conversion contract."
                ),
            },
            {
                "blocker_id": "RAW_CLOCK_ORIGIN_NOT_DOCUMENTED_IN_PINNED_AUTHORITY",
                "evidence": (
                    "The audited pinned authority does not bind the RAW clock origin "
                    "to recording start, first accepted video PTS, or another video "
                    "clock event."
                ),
            },
            {
                "blocker_id": "BODYTRACK_ROW_TO_VIDEO_PTS_NOT_DOCUMENTED",
                "evidence": (
                    "Equal BodyTrack CSV-row and decoded-frame counts do not document "
                    "a row-to-video-PTS mapping."
                ),
            },
            {
                "blocker_id": "RAW_ROW_TO_NATIVE_50HZ_SAMPLE_MAPPING_UNSUPPORTED",
                "evidence": (
                    "The article declares nominal 50 Hz sensor output, while the "
                    "independent all-208 original-RAW scan observes 10,183,005 "
                    "timestamped dynamic row observations and 8,735,242 exact "
                    "consecutive held payloads. VIDIMU-TOOLS treats rows positionally "
                    "and does not derive native ticks from RAW timestamp tokens. The "
                    "pinned evidence therefore does not authorize interpreting every "
                    "RAW row as one native 50 Hz hardware sample."
                ),
            },
            {
                "blocker_id": "DUAL_DIRECTION_APPLIED_OVERRIDE",
                "evidence": (
                    "S53_A13_T03 and S57_A07_T01 have plot-selected VIDEO cuts and "
                    "released applied CSV, MOT, and RAW cuts from both candidate "
                    "directions; both are AMBIGUOUS_SOURCE_MAPPING."
                ),
            },
        ],
        "claim_boundary": {
            "canonical_clock_segments_emitted": False,
            "canonical_frame_times_emitted": False,
            "canonical_imu_times_emitted": False,
            "correlation_used_by_this_audit": False,
            "csv_frame_count_equality_proves_clock_mapping": False,
            "filename_pairing_proves_synchronization": False,
            "nominal_50hz_proves_each_raw_row_is_a_hardware_sample": False,
            "raw_timestamp_interpretation": (
                "EXACT_RELEASE_DECIMAL_TOKEN;UNIT_CLOCK_SOURCE_AND_VIDEO_RELATION_"
                "NOT_DOCUMENTED_IN_AUDITED_PINNED_AUTHORITY"
            ),
            "released_crop_transform_bytes_reconciled": True,
            "released_source_mapping_is_unambiguous": False,
            "sto_companion_promoted_to_raw_clock_authority": False,
            "v06_indexes_or_windows_emitted": False,
        },
        "gate_status": NO_GO,
        "implementation": {
            "audit_implementation_version": IMPLEMENTATION_VERSION,
            "audit_script_sha256": initial_script_sha256,
            "canonical_json": "SORTED_KEYS_UTF8_INDENT2_TRAILING_NEWLINE_NO_NAN",
        },
        "records": records,
        "reconciliation": {
            **plot_info,
            "ambiguous_source_mapping_record_count": len(ambiguities),
            "ambiguous_source_mapping_recording_ids": sorted(ambiguous_ids),
            "source_mapping_classification_counts": mapping_classification_counts,
            "csv_frame_count_equal_recordings": sum(
                pose_counts[key] == frame_counts[key] for key in pose_counts
            ),
            "decoded_frame_count": sum(frame_counts.values()),
            "plot_recording_count": len(plots),
            "pose_csv_row_count": sum(pose_counts.values()),
            "sync_override_count": dataset_sync["override_count"],
        },
        "schema_version": SCHEMA_VERSION,
        "source_evidence": {
            "analysis_archive": {
                "info_to_sync_sha256": _hash_bytes(info_payload),
                "published_md5": EXPECTED_ANALYSIS_ARCHIVE_MD5,
                "sha256": EXPECTED_ANALYSIS_ARCHIVE_SHA256,
                "sync_plot_hashes": plot_hashes,
            },
            "authoritative_article": article_evidence,
            "dataset_sync": dataset_sync,
            "official_record_metadata": record_evidence,
            "raw_independent_reconciliation": raw_reconciliation,
            "snapshot": snapshot_evidence,
            "v04_frame_evidence": v04_evidence,
            "vidimu_v2_native_source_audit": v2_evidence,
            "vidimu_tools": tools_evidence,
        },
        "source_mapping_ambiguities": ambiguities,
    }
    if _hash_bytes(_read_regular(script_path)) != initial_script_sha256:
        raise SyncAuthorityAuditError("audit implementation changed during run")
    return result


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--analysis-archive", required=True, type=Path)
    parser.add_argument("--article-pdf", required=True, type=Path)
    parser.add_argument("--record-metadata", required=True, type=Path)
    parser.add_argument("--source-parser", required=True, type=Path)
    parser.add_argument("--tools-release", required=True, type=Path)
    parser.add_argument("--v04-release-audit", required=True, type=Path)
    parser.add_argument("--v2-release-audit", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _rename_noreplace(
    directory_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    if os.name != "posix":
        raise SyncAuthorityAuditError(
            "atomic no-replace publication requires POSIX"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError as exc:
            raise SyncAuthorityAuditError(
                "atomic no-replace publication is unavailable"
            ) from exc
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
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise SyncAuthorityAuditError(
                "atomic no-replace publication is unavailable"
            ) from exc
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
            0x00000001,
        )
    else:
        raise SyncAuthorityAuditError(
            "atomic no-replace publication is unsupported"
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise SyncAuthorityAuditError("output already exists")
    raise OSError(error, os.strerror(error), destination_name)


def _write_exclusive(path: Path, payload: bytes, forbidden: Sequence[Path]) -> None:
    try:
        destination_lstat = path.lstat()
    except FileNotFoundError:
        destination_lstat = None
    except OSError as exc:
        raise SyncAuthorityAuditError("cannot inspect output destination") from exc
    if destination_lstat is not None and stat.S_ISLNK(destination_lstat.st_mode):
        raise SyncAuthorityAuditError("output destination must not be a symlink")
    destination = path.resolve(strict=False)
    for source in forbidden:
        source_resolved = source.resolve(strict=True)
        if destination == source_resolved or (
            source_resolved.is_dir() and destination.is_relative_to(source_resolved)
        ):
            raise SyncAuthorityAuditError("output must not alias or enter an input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.name or destination.name in {".", ".."}:
        raise SyncAuthorityAuditError("output filename is invalid")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        parent_descriptor = os.open(destination.parent, flags)
    except OSError as exc:
        raise SyncAuthorityAuditError("cannot pin output parent directory") from exc
    parent_stat = os.fstat(parent_descriptor)
    parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
    temporary_name = f".{destination.name}.tmp-{uuid.uuid4().hex}"
    descriptor = -1
    temporary_identity: tuple[int, int] | None = None
    published = False
    try:
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        open_flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(
                temporary_name,
                open_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise SyncAuthorityAuditError(
                "cannot create output temporary file"
            ) from exc
        temporary_stat = os.fstat(descriptor)
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(
            parent_descriptor,
            temporary_name,
            destination.name,
        )
        published = True
        os.fsync(parent_descriptor)
        final_stat = os.stat(
            destination.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            temporary_identity is None
            or not stat.S_ISREG(final_stat.st_mode)
            or (final_stat.st_dev, final_stat.st_ino) != temporary_identity
            or final_stat.st_size != len(payload)
        ):
            raise SyncAuthorityAuditError("published output identity mismatch")
        current_parent_descriptor = os.open(destination.parent, flags)
        try:
            current_parent = os.fstat(current_parent_descriptor)
        finally:
            os.close(current_parent_descriptor)
        if (current_parent.st_dev, current_parent.st_ino) != parent_identity:
            raise SyncAuthorityAuditError("output parent changed during publication")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        result = audit_vidimu_v05_sync_authority(
            args.snapshot_root,
            args.analysis_archive,
            args.tools_release,
            args.v04_release_audit,
            args.v2_release_audit,
            args.source_parser,
            args.article_pdf,
            args.record_metadata,
        )
        payload = canonical_json_bytes(result)
        if args.output is not None:
            _write_exclusive(
                args.output,
                payload,
                (
                    args.snapshot_root,
                    args.analysis_archive,
                    args.article_pdf,
                    args.record_metadata,
                    args.source_parser,
                    args.tools_release,
                    args.v04_release_audit,
                    args.v2_release_audit,
                ),
            )
        print(payload.decode("utf-8"), end="")
        return NO_GO_EXIT_CODE if result.get("gate_status") == NO_GO else 0
    except Exception as exc:  # noqa: BLE001 - CLI must emit deterministic failure
        failure = {
            "artifact_kind": ARTIFACT_KIND,
            "audit_execution_status": AUDIT_ERROR,
            "error": str(exc),
            "gate_status": NOT_EVALUATED,
            "schema_version": SCHEMA_VERSION,
        }
        print(canonical_json_bytes(failure).decode("utf-8"), end="")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
