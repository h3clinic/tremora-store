"""Trust-anchored, content-addressed VIDIMU source materialization.

This v0.4 boundary is intentionally parallel to the v0.3 PTS/CV finalizer.  It
downloads or copies immutable source objects, verifies caller-supplied length
and SHA-256 anchors, and extracts only assets explicitly named by the frozen
inventory.  It does not decode video, parse IMU data, or establish a clock.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq

from .finalize._bundle_io import (
    PARQUET_WRITER_POLICY_ID,
    ROW_GROUP_SIZE,
    FinalizationBundleError,
    _rename_noreplace,
    canonical_json_bytes,
)
from .parquet_writer import semantic_table_hash
from .schema import schema_fingerprint

VIDIMU_SOURCE_SNAPSHOT_SCHEMA_VERSION = "0.4.0"
VIDIMU_SOURCE_INVENTORY_SCHEMA_VERSION = (
    "tremora-vidimu-source-inventory-0.4.0"
)
VIDIMU_SOURCE_SNAPSHOT_ARTIFACT_KIND = "TREMORA_VIDIMU_SOURCE_SNAPSHOT"

SOURCE_OBJECTS_FILE = "vidimu_source_objects.parquet"
EXTRACTED_ASSETS_FILE = "vidimu_extracted_assets.parquet"
SOURCE_INVENTORY_FILE = "source_inventory.json"
SNAPSHOT_MANIFEST_FILE = "snapshot_manifest.json"
SUCCESS_MARKER_FILE = "_SUCCESS"

ASSET_ROLES = frozenset({
    "VIDEO",
    "IMU",
    "ANNOTATION",
    "METADATA",
    "SYNC_METADATA",
    "OTHER",
})
ARCHIVE_TYPES = frozenset({"NONE", "ZIP"})
AVAILABILITY_STATES = frozenset({"REQUIRED", "UNAVAILABLE"})
PROVIDER_CHECKSUM_ALGORITHMS = frozenset({"MD5", "SHA256"})

DEFAULT_MAX_SOURCE_OBJECT_BYTES = 64 * 1024 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_TOTAL_EXTRACTED_BYTES = 16 * 1024 * 1024 * 1024
DEFAULT_MAX_COMPRESSION_RATIO = 1000.0
DEFAULT_MAX_ARCHIVE_MEMBERS = 100_000

VIDIMU_V2_DATASET_ID = "VIDIMU"
VIDIMU_V2_DATASET_VERSION = "v2.0.0"
VIDIMU_V2_ZENODO_RECORD_ID = "15075076"
VIDIMU_V2_SOURCE_PROVIDER_ID = "ZENODO_RECORD_15075076"
VIDIMU_V2_LICENSE_ID = "CC-BY-4.0"
VIDIMU_V2_CITATION_ID = "10.1038/s41597-023-02554-9"
VIDIMU_V2_RECORD_METADATA_OBJECT_ID = "vidimu-zenodo-record-metadata"
VIDIMU_V2_RECORD_METADATA_URL = "https://zenodo.org/api/records/15075076"
VIDIMU_V2_RECORD_METADATA_BYTES = 8_040
VIDIMU_V2_RECORD_METADATA_SHA256 = (
    "97fd3f755aab65631ec19bff762fe848a609cee5324644f6baedf1d1b4f3926f"
)
VIDIMU_V2_EXPECTED_RECORDING_COUNT = 208
VIDIMU_V2_EXPECTED_ASSET_REFERENCE_COUNT = 624
VIDIMU_V2_ASSET_REFERENCE_AUTHORITY_SCHEMA_VERSION = (
    "tremora-vidimu-v2-asset-references-1.0.0"
)
VIDIMU_V2_ASSET_REFERENCE_AUTHORITY_SHA256 = (
    "52c5928b9a0ae42963f815806e6164468020d998580635ff1fd27f85784124ee"
)
VIDIMU_V2_ASSET_REFERENCE_CATALOG_TSV_SHA256 = (
    "4586440a3a3980d984f80557a278cfe1c8edc8b13b9bc37707bc6b32e9e4038b"
)
VIDIMU_V2_ASSET_REFERENCE_CATALOG_TSV_BYTES = 114_165
VIDIMU_V2_ASSET_REFERENCE_CATALOG_PATH = (
    Path(__file__).parent
    / "catalogs"
    / "vidimu_v2_asset_reference_catalog.tsv"
)
_VIDIMU_V2_ARCHIVE_PINS: Mapping[str, Mapping[str, object]] = MappingProxyType({
    "dataset.zip": MappingProxyType({
        "source_object_id": "vidimu-dataset-archive",
        "source_url": (
            "https://zenodo.org/api/records/15075076/files/dataset.zip/content"
        ),
        "expected_content_length": 336_819_642,
        "source_object_sha256": (
            "eff12be2f1c5a0cc7389726c754ea1c4ab19d8ca49c227b47344109cbf927841"
        ),
        "provider_checksum_algorithm": "MD5",
        "provider_checksum_value": "368d34d13651b44e6d4444c4a6c41380",
    }),
    "videosmallsize.zip": MappingProxyType({
        "source_object_id": "vidimu-video-archive-small",
        "source_url": (
            "https://zenodo.org/api/records/15075076/files/"
            "videosmallsize.zip/content"
        ),
        "expected_content_length": 1_095_515_926,
        "source_object_sha256": (
            "7be6d39b0c15c0e7d8d95a4814a292d3604e465abd031d109f9a55f24e110689"
        ),
        "provider_checksum_algorithm": "MD5",
        "provider_checksum_value": "b3cbfadf8cf719f47a748a6b5eeb3f2b",
    }),
})
_VIDIMU_V2_RECORD_FILE_PINS: Mapping[
    str, tuple[int, str, str]
] = MappingProxyType({
    "analysis.zip": (
        71_650_185,
        "md5:d68f2a16fa3fcd8cec090a3169e4764e",
        "https://zenodo.org/api/records/15075076/files/analysis.zip/content",
    ),
    "dataset.zip": (
        336_819_642,
        "md5:368d34d13651b44e6d4444c4a6c41380",
        "https://zenodo.org/api/records/15075076/files/dataset.zip/content",
    ),
    "videosfullsize.zip": (
        41_304_165_581,
        "md5:c6e48c41f7b6d071ceb050e9c13c3205",
        "https://zenodo.org/api/records/15075076/files/videosfullsize.zip/content",
    ),
    "videosmallsize.zip": (
        1_095_515_926,
        "md5:b3cbfadf8cf719f47a748a6b5eeb3f2b",
        "https://zenodo.org/api/records/15075076/files/videosmallsize.zip/content",
    ),
})

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MD5_RE = re.compile(r"[0-9a-f]{32}\Z")
_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._=-]{0,127}\Z")
_UTC_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z"
)
_MAX_ARCHIVE_MEMBERS = 1_000_000
_MAX_MEMBER_PATH_BYTES = 4096
_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_INVENTORY_BYTES = 64 * 1024 * 1024
_MAX_SUCCESS_BYTES = 4096
_MAX_PARQUET_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_PARQUET_DECODED_BYTES = 512 * 1024 * 1024
_MAX_ZIP_CENTRAL_DIRECTORY_BYTES = 256 * 1024 * 1024
_MAX_ZIP_EXTRA_OR_COMMENT_BYTES = 65_535
_ZIP_CENTRAL_HEADER_BYTES = 46
_WINDOWS_RESERVED = frozenset({
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
})


class SourceSnapshotError(RuntimeError):
    """Raised when a source snapshot cannot be verified or published."""


@dataclass(frozen=True, slots=True)
class SourceObjectAnchor:
    """Independent trust anchors and acquisition metadata for one object.

    ``source_url`` remains the published provenance URI.  ``local_path`` may
    point at a previously downloaded object; when absent, only HTTP(S) fetching
    is allowed.  Local filesystem paths are never persisted in the snapshot.
    """

    source_object_id: str
    source_url: str
    source_provider_id: str
    expected_content_length: int
    source_object_sha256: str
    mime_type: str
    archive_type: str
    local_path: Path | None = None
    expected_etag: str | None = None
    expected_last_modified: str | None = None
    provider_checksum_algorithm: str | None = None
    provider_checksum_value: str | None = None


@dataclass(frozen=True, slots=True)
class InventoryAssetReference:
    """One exact archive member expected by the frozen source inventory."""

    source_object_id: str
    archive_member_path: str
    recording_id: str
    asset_role: str
    modality: str
    expected_size_bytes: int | None = None
    expected_sha256: str | None = None
    availability: str = "REQUIRED"
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class VidimuSourceSnapshotRequest:
    """Complete caller-bound authority for one source materialization."""

    dataset_id: str
    dataset_version: str
    license_id: str
    citation_id: str
    terms_snapshot_id: str
    terms_source_object_id: str
    source_objects: tuple[SourceObjectAnchor, ...]
    asset_references: tuple[InventoryAssetReference, ...]
    max_source_object_bytes: int = DEFAULT_MAX_SOURCE_OBJECT_BYTES
    max_archive_member_bytes: int = DEFAULT_MAX_ARCHIVE_MEMBER_BYTES
    max_total_extracted_bytes: int = DEFAULT_MAX_TOTAL_EXTRACTED_BYTES
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO
    max_archive_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS


@dataclass(frozen=True, slots=True)
class MaterializedSourceSnapshot:
    """The immutable result of one successful materialization."""

    path: Path
    snapshot_manifest_sha256: str
    source_inventory_sha256: str
    source_object_count: int
    extracted_asset_count: int
    unavailable_asset_count: int


SOURCE_OBJECTS_SCHEMA = pa.schema([
    pa.field("source_object_id", pa.string(), nullable=False),
    pa.field("dataset_id", pa.string(), nullable=False),
    pa.field("dataset_version", pa.string(), nullable=False),
    pa.field("source_url", pa.string(), nullable=False),
    pa.field("source_provider_id", pa.string(), nullable=False),
    pa.field("download_timestamp_utc", pa.string(), nullable=False),
    pa.field("expected_content_length", pa.int64(), nullable=False),
    pa.field("observed_content_length", pa.int64(), nullable=False),
    pa.field("etag", pa.string(), nullable=True),
    pa.field("last_modified", pa.string(), nullable=True),
    pa.field("source_object_sha256", pa.string(), nullable=False),
    pa.field("provider_checksum_algorithm", pa.string(), nullable=True),
    pa.field("provider_checksum_value", pa.string(), nullable=True),
    pa.field("provider_checksum_verified", pa.bool_(), nullable=True),
    pa.field("mime_type", pa.string(), nullable=False),
    pa.field("archive_type", pa.string(), nullable=False),
    pa.field("license_id", pa.string(), nullable=False),
    pa.field("citation_id", pa.string(), nullable=False),
    pa.field("terms_snapshot_id", pa.string(), nullable=False),
    pa.field("download_status", pa.string(), nullable=False),
    pa.field("failure_reason", pa.string(), nullable=True),
], metadata={
    b"tremora.schema_version": VIDIMU_SOURCE_SNAPSHOT_SCHEMA_VERSION.encode(),
    b"tremora.table": b"vidimu_source_objects",
    b"tremora.authority": b"caller_length_and_sha256",
})

EXTRACTED_ASSETS_SCHEMA = pa.schema([
    pa.field("source_object_id", pa.string(), nullable=False),
    pa.field("archive_member_path", pa.string(), nullable=False),
    pa.field("normalized_member_path", pa.string(), nullable=False),
    pa.field("recording_id", pa.string(), nullable=False),
    pa.field("asset_role", pa.string(), nullable=False),
    pa.field("modality", pa.string(), nullable=False),
    pa.field("asset_size_bytes", pa.int64(), nullable=True),
    pa.field("asset_sha256", pa.string(), nullable=True),
    pa.field("extraction_status", pa.string(), nullable=False),
    pa.field("failure_reason", pa.string(), nullable=True),
], metadata={
    b"tremora.schema_version": VIDIMU_SOURCE_SNAPSHOT_SCHEMA_VERSION.encode(),
    b"tremora.table": b"vidimu_extracted_assets",
    b"tremora.extraction_scope": b"explicit_inventory_references_only",
})


@dataclass(frozen=True, slots=True)
class _ValidatedAsset:
    reference: InventoryAssetReference
    normalized_member_path: str
    destination_collision_key: str


@dataclass(frozen=True, slots=True)
class _DownloadResult:
    path: Path
    observed_content_length: int
    etag: str | None
    last_modified: str | None
    provider_checksum_verified: bool | None


def _stable_text(value: object, *, field: str, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum \
            or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SourceSnapshotError(f"{field} must be nonempty stable text")
    return value


def _safe_component(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT_RE.fullmatch(value) is None \
            or value in {".", ".."}:
        raise SourceSnapshotError(f"{field} must be a safe path component")
    return value


def _lower_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SourceSnapshotError(f"{field} must be a lowercase SHA-256")
    return value


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceSnapshotError(f"{field} must be a nonnegative integer")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    value = _nonnegative_integer(value, field=field)
    if value == 0:
        raise SourceSnapshotError(f"{field} must be positive")
    return value


def _provider_checksum(
    algorithm: object,
    value: object,
) -> tuple[str | None, str | None]:
    if algorithm is None and value is None:
        return None, None
    if not isinstance(algorithm, str) \
            or algorithm not in PROVIDER_CHECKSUM_ALGORITHMS \
            or not isinstance(value, str):
        raise SourceSnapshotError(
            "provider checksum algorithm and value must be supplied together")
    pattern = _MD5_RE if algorithm == "MD5" else _SHA256_RE
    if pattern.fullmatch(value) is None:
        raise SourceSnapshotError(
            f"provider {algorithm} checksum has an invalid value")
    return algorithm, value


def _validated_budgets(
    request: VidimuSourceSnapshotRequest,
) -> tuple[int, int, int, float, int]:
    max_object = _positive_integer(
        request.max_source_object_bytes, field="max_source_object_bytes")
    max_member = _positive_integer(
        request.max_archive_member_bytes, field="max_archive_member_bytes")
    max_total = _positive_integer(
        request.max_total_extracted_bytes, field="max_total_extracted_bytes")
    max_members = _positive_integer(
        request.max_archive_members, field="max_archive_members")
    ratio = request.max_compression_ratio
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) \
            or not math.isfinite(ratio) or ratio < 1:
        raise SourceSnapshotError(
            "max_compression_ratio must be a finite number of at least one")
    if max_members > _MAX_ARCHIVE_MEMBERS:
        raise SourceSnapshotError(
            "max_archive_members exceeds the implementation safety ceiling")
    return max_object, max_member, max_total, float(ratio), max_members


def _timestamp(value: str | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc).isoformat(
            timespec="seconds").replace("+00:00", "Z")
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise SourceSnapshotError(
            "download_timestamp_utc must be an RFC 3339 UTC timestamp")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SourceSnapshotError(
            "download_timestamp_utc is not a valid timestamp") from exc
    return value


def _validate_source_url(value: object) -> str:
    value = _stable_text(value, field="source_url", maximum=4096)
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc \
            or parsed.username is not None or parsed.password is not None \
            or parsed.fragment:
        raise SourceSnapshotError(
            "source_url must be an HTTP(S) URL without credentials or fragment")
    return value


def vidimu_v2_archive_anchor(
    file_key: str,
    *,
    local_path: Path | None = None,
) -> SourceObjectAnchor:
    """Build one closed VIDIMU v2 Zenodo archive anchor.

    Only the fully byte-pinned dataset and selected small-video archives are
    admitted.  Their record ID, URL, byte length, SHA-256, and published Zenodo
    MD5 are not caller-overridable.
    """

    pin = _VIDIMU_V2_ARCHIVE_PINS.get(file_key)
    if pin is None:
        raise SourceSnapshotError(
            "VIDIMU v2 archive key is not in the closed checked release set")
    return SourceObjectAnchor(
        source_object_id=str(pin["source_object_id"]),
        source_url=str(pin["source_url"]),
        source_provider_id=VIDIMU_V2_SOURCE_PROVIDER_ID,
        expected_content_length=int(pin["expected_content_length"]),
        source_object_sha256=str(pin["source_object_sha256"]),
        mime_type="application/zip",
        archive_type="ZIP",
        local_path=local_path,
        provider_checksum_algorithm=str(pin["provider_checksum_algorithm"]),
        provider_checksum_value=str(pin["provider_checksum_value"]),
    )


def vidimu_v2_record_metadata_anchor(
    *,
    local_path: Path | None = None,
) -> SourceObjectAnchor:
    """Return the exact captured Zenodo v2 record-metadata authority."""

    return SourceObjectAnchor(
        source_object_id=VIDIMU_V2_RECORD_METADATA_OBJECT_ID,
        source_url=VIDIMU_V2_RECORD_METADATA_URL,
        source_provider_id=VIDIMU_V2_SOURCE_PROVIDER_ID,
        expected_content_length=VIDIMU_V2_RECORD_METADATA_BYTES,
        source_object_sha256=VIDIMU_V2_RECORD_METADATA_SHA256,
        mime_type="application/json",
        archive_type="NONE",
        local_path=local_path,
    )


def build_vidimu_v2_source_snapshot_request(
    *,
    dataset_archive_local_path: Path | None,
    video_archive_local_path: Path | None,
    record_metadata_local_path: Path | None,
    asset_references: tuple[InventoryAssetReference, ...],
    max_source_object_bytes: int = DEFAULT_MAX_SOURCE_OBJECT_BYTES,
    max_archive_member_bytes: int = DEFAULT_MAX_ARCHIVE_MEMBER_BYTES,
    max_total_extracted_bytes: int = DEFAULT_MAX_TOTAL_EXTRACTED_BYTES,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
    max_archive_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS,
) -> VidimuSourceSnapshotRequest:
    """Build the closed VIDIMU v2 request around an independently pinned record.

    The terms/citation authority is the exact captured Zenodo record JSON.  Its
    byte length and SHA-256 are closed here and its record/version/license/DOI
    and complete file inventory are semantically checked during materialization
    and every strict verification.
    """

    terms_source_object = vidimu_v2_record_metadata_anchor(
        local_path=record_metadata_local_path)
    request = VidimuSourceSnapshotRequest(
        dataset_id=VIDIMU_V2_DATASET_ID,
        dataset_version=VIDIMU_V2_DATASET_VERSION,
        license_id=VIDIMU_V2_LICENSE_ID,
        citation_id=VIDIMU_V2_CITATION_ID,
        terms_snapshot_id=terms_source_object.source_object_sha256,
        terms_source_object_id=terms_source_object.source_object_id,
        source_objects=(
            vidimu_v2_archive_anchor(
                "dataset.zip", local_path=dataset_archive_local_path),
            vidimu_v2_archive_anchor(
                "videosmallsize.zip", local_path=video_archive_local_path),
            terms_source_object,
        ),
        asset_references=asset_references,
        max_source_object_bytes=max_source_object_bytes,
        max_archive_member_bytes=max_archive_member_bytes,
        max_total_extracted_bytes=max_total_extracted_bytes,
        max_compression_ratio=max_compression_ratio,
        max_archive_members=max_archive_members,
    )
    _validate_request(request)
    return request


def load_vidimu_v2_asset_reference_catalog(
) -> tuple[InventoryAssetReference, ...]:
    """Load and verify the repository-frozen 624-member VIDIMU v2 catalog."""

    payload = _read_regular_bytes(
        VIDIMU_V2_ASSET_REFERENCE_CATALOG_PATH,
        maximum_bytes=VIDIMU_V2_ASSET_REFERENCE_CATALOG_TSV_BYTES,
        expected_bytes=VIDIMU_V2_ASSET_REFERENCE_CATALOG_TSV_BYTES,
        expected_sha256=VIDIMU_V2_ASSET_REFERENCE_CATALOG_TSV_SHA256,
    )
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SourceSnapshotError(
            "VIDIMU v2 asset-reference catalog must be ASCII"
        ) from exc
    fieldnames = (
        "source_object_id",
        "archive_member_path",
        "recording_id",
        "asset_role",
        "modality",
        "expected_size_bytes",
        "expected_sha256",
        "availability",
        "unavailable_reason",
    )
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    if reader.fieldnames != list(fieldnames):
        raise SourceSnapshotError(
            "VIDIMU v2 asset-reference catalog header changed"
        )
    references: list[InventoryAssetReference] = []
    try:
        for row in reader:
            if set(row) != set(fieldnames) or any(
                row[field] is None for field in fieldnames[:-1]
            ):
                raise SourceSnapshotError(
                    "VIDIMU v2 asset-reference catalog row is malformed"
                )
            availability = row["availability"]
            expected_size = row["expected_size_bytes"]
            expected_sha256 = row["expected_sha256"]
            unavailable_reason = row["unavailable_reason"] or ""
            references.append(InventoryAssetReference(
                source_object_id=row["source_object_id"],
                archive_member_path=row["archive_member_path"],
                recording_id=row["recording_id"],
                asset_role=row["asset_role"],
                modality=row["modality"],
                expected_size_bytes=(
                    int(expected_size) if expected_size else None
                ),
                expected_sha256=expected_sha256 or None,
                availability=availability,
                unavailable_reason=unavailable_reason or None,
            ))
    except (TypeError, ValueError) as exc:
        raise SourceSnapshotError(
            "VIDIMU v2 asset-reference catalog has invalid numeric data"
        ) from exc
    request = build_vidimu_v2_source_snapshot_request(
        dataset_archive_local_path=None,
        video_archive_local_path=None,
        record_metadata_local_path=None,
        asset_references=tuple(references),
    )
    return request.asset_references


def _source_anchor_identity(anchor: SourceObjectAnchor) -> tuple[object, ...]:
    return (
        anchor.source_object_id,
        anchor.source_url,
        anchor.source_provider_id,
        anchor.expected_content_length,
        anchor.source_object_sha256,
        anchor.mime_type,
        anchor.archive_type,
        anchor.expected_etag,
        anchor.expected_last_modified,
        anchor.provider_checksum_algorithm,
        anchor.provider_checksum_value,
    )


def _verify_closed_vidimu_v2_authority(
    request: VidimuSourceSnapshotRequest,
    source_by_id: Mapping[str, SourceObjectAnchor],
    source_paths: Mapping[str, Path],
) -> None:
    is_vidimu_v2 = request.dataset_id == VIDIMU_V2_DATASET_ID \
        or VIDIMU_V2_RECORD_METADATA_OBJECT_ID in source_by_id
    if not is_vidimu_v2:
        return
    expected_anchors = (
        vidimu_v2_archive_anchor("dataset.zip"),
        vidimu_v2_archive_anchor("videosmallsize.zip"),
        vidimu_v2_record_metadata_anchor(),
    )
    if request.dataset_id != VIDIMU_V2_DATASET_ID \
            or request.dataset_version != VIDIMU_V2_DATASET_VERSION \
            or request.license_id != VIDIMU_V2_LICENSE_ID \
            or request.citation_id != VIDIMU_V2_CITATION_ID \
            or request.terms_snapshot_id != VIDIMU_V2_RECORD_METADATA_SHA256 \
            or request.terms_source_object_id \
            != VIDIMU_V2_RECORD_METADATA_OBJECT_ID \
            or set(source_by_id) != {
                anchor.source_object_id for anchor in expected_anchors
            }:
        raise SourceSnapshotError("VIDIMU v2 closed request identity changed")
    for expected in expected_anchors:
        observed = source_by_id[expected.source_object_id]
        if _source_anchor_identity(observed) != _source_anchor_identity(expected):
            raise SourceSnapshotError("VIDIMU v2 source authority changed")
    metadata_path = source_paths.get(VIDIMU_V2_RECORD_METADATA_OBJECT_ID)
    if metadata_path is None:
        raise SourceSnapshotError("VIDIMU v2 record metadata object is absent")
    _verify_vidimu_v2_record_metadata(metadata_path)


def _normalize_member_path(value: object) -> tuple[str, str]:
    """Return a safe NFC path and a portable case-insensitive collision key."""

    if not isinstance(value, str) or not value or "\x00" in value \
            or "\\" in value or len(value.encode("utf-8")) > _MAX_MEMBER_PATH_BYTES:
        raise SourceSnapshotError("archive member path is unsafe")
    if value.startswith("/") or PurePosixPath(value).is_absolute():
        raise SourceSnapshotError("archive member path traversal is forbidden")
    raw_parts = value.split("/")
    if any(part == ".." for part in raw_parts):
        raise SourceSnapshotError("archive member path traversal is forbidden")
    parts: list[str] = []
    collision_parts: list[str] = []
    for raw_part in raw_parts:
        if raw_part in {"", "."}:
            continue
        part = unicodedata.normalize("NFC", raw_part)
        if not part or part in {".", ".."} or part.endswith((" ", ".")) \
                or ":" in part or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in part
                ):
            raise SourceSnapshotError("archive member path is unsafe")
        if part.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED:
            raise SourceSnapshotError("archive member path is not portable")
        parts.append(part)
        collision_parts.append(part.casefold())
    if not parts:
        raise SourceSnapshotError("archive member path is empty after normalization")
    return "/".join(parts), "/".join(collision_parts)


def _vidimu_v2_asset_reference_authority_sha256(
    assets: Sequence[_ValidatedAsset],
) -> str:
    value = {
        "asset_references": [{
            "archive_member_path": asset.reference.archive_member_path,
            "asset_role": asset.reference.asset_role,
            "availability": asset.reference.availability,
            "expected_sha256": asset.reference.expected_sha256,
            "expected_size_bytes": asset.reference.expected_size_bytes,
            "modality": asset.reference.modality,
            "recording_id": asset.reference.recording_id,
            "source_object_id": asset.reference.source_object_id,
            "unavailable_reason": asset.reference.unavailable_reason,
        } for asset in assets],
        "dataset_id": VIDIMU_V2_DATASET_ID,
        "dataset_version": VIDIMU_V2_DATASET_VERSION,
        "schema_version": (
            VIDIMU_V2_ASSET_REFERENCE_AUTHORITY_SCHEMA_VERSION
        ),
    }
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SourceSnapshotError(
            "VIDIMU v2 asset-reference authority is not canonical"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _validate_request(
    request: VidimuSourceSnapshotRequest,
) -> tuple[
    tuple[SourceObjectAnchor, ...],
    tuple[_ValidatedAsset, ...],
    dict[str, SourceObjectAnchor],
]:
    if type(request) is not VidimuSourceSnapshotRequest:
        raise SourceSnapshotError("source snapshot request has an invalid type")
    max_object, max_member, max_total, _ratio, _max_members = (
        _validated_budgets(request)
    )
    for field, value in (
        ("dataset_id", request.dataset_id),
        ("dataset_version", request.dataset_version),
        ("license_id", request.license_id),
        ("citation_id", request.citation_id),
        ("terms_snapshot_id", request.terms_snapshot_id),
    ):
        _stable_text(value, field=field)
    terms_source_object_id = _safe_component(
        request.terms_source_object_id, field="terms_source_object_id")
    if not isinstance(request.source_objects, tuple) or not request.source_objects:
        raise SourceSnapshotError("source_objects must be a nonempty tuple")
    if not isinstance(request.asset_references, tuple) \
            or not request.asset_references:
        raise SourceSnapshotError("asset_references must be a nonempty tuple")

    source_by_id: dict[str, SourceObjectAnchor] = {}
    urls: set[str] = set()
    hashes: set[str] = set()
    validated_sources: list[SourceObjectAnchor] = []
    for anchor in request.source_objects:
        if type(anchor) is not SourceObjectAnchor:
            raise SourceSnapshotError("source_objects contains an invalid entry")
        source_object_id = _safe_component(
            anchor.source_object_id, field="source_object_id")
        source_url = _validate_source_url(anchor.source_url)
        _stable_text(anchor.source_provider_id, field="source_provider_id")
        _nonnegative_integer(
            anchor.expected_content_length, field="expected_content_length")
        if anchor.expected_content_length > max_object:
            raise SourceSnapshotError(
                "source object exceeds the request byte budget")
        source_hash = _lower_sha256(
            anchor.source_object_sha256, field="source_object_sha256")
        _stable_text(anchor.mime_type, field="mime_type")
        if anchor.archive_type not in ARCHIVE_TYPES:
            raise SourceSnapshotError("archive_type must be NONE or ZIP")
        if anchor.expected_etag is not None:
            _stable_text(anchor.expected_etag, field="expected_etag")
        if anchor.expected_last_modified is not None:
            _stable_text(
                anchor.expected_last_modified, field="expected_last_modified")
        _provider_checksum(
            anchor.provider_checksum_algorithm,
            anchor.provider_checksum_value,
        )
        if source_object_id in source_by_id or source_url in urls \
                or source_hash in hashes:
            raise SourceSnapshotError(
                "source objects must have distinct IDs, URLs, and hashes")
        if anchor.local_path is not None and not isinstance(anchor.local_path, Path):
            raise SourceSnapshotError("local_path must be a pathlib.Path")
        source_by_id[source_object_id] = anchor
        urls.add(source_url)
        hashes.add(source_hash)
        validated_sources.append(anchor)
    if terms_source_object_id not in source_by_id:
        raise SourceSnapshotError(
            "terms_source_object_id does not reference a source object")

    validated_assets: list[_ValidatedAsset] = []
    reference_keys: set[tuple[str, str]] = set()
    destination_keys: set[str] = set()
    expected_extracted_bytes = 0
    for reference in request.asset_references:
        if type(reference) is not InventoryAssetReference:
            raise SourceSnapshotError(
                "asset_references contains an invalid entry")
        source_object_id = _safe_component(
            reference.source_object_id, field="asset source_object_id")
        anchor = source_by_id.get(source_object_id)
        if anchor is None:
            raise SourceSnapshotError(
                "inventory asset references an unknown source object")
        if anchor.archive_type != "ZIP":
            raise SourceSnapshotError(
                "inventory assets must reference ZIP source objects")
        normalized_path, collision_key = _normalize_member_path(
            reference.archive_member_path)
        _safe_component(reference.recording_id, field="recording_id")
        if reference.asset_role not in ASSET_ROLES:
            raise SourceSnapshotError("inventory asset_role is unsupported")
        _stable_text(reference.modality, field="modality")
        if reference.availability not in AVAILABILITY_STATES:
            raise SourceSnapshotError(
                "asset availability must be REQUIRED or UNAVAILABLE")
        if reference.availability == "REQUIRED":
            expected_size = _nonnegative_integer(
                reference.expected_size_bytes, field="expected_size_bytes")
            if expected_size > max_member:
                raise SourceSnapshotError(
                    "inventory asset exceeds the archive-member byte budget")
            expected_extracted_bytes += expected_size
            if expected_extracted_bytes > max_total:
                raise SourceSnapshotError(
                    "inventory exceeds the total extraction byte budget")
            _lower_sha256(reference.expected_sha256, field="expected_sha256")
            if reference.unavailable_reason is not None:
                raise SourceSnapshotError(
                    "required asset may not have an unavailable reason")
        else:
            if reference.expected_size_bytes is not None \
                    or reference.expected_sha256 is not None:
                raise SourceSnapshotError(
                    "unavailable asset may not claim expected bytes or hash")
            _stable_text(
                reference.unavailable_reason, field="unavailable_reason")
        reference_key = (source_object_id, reference.archive_member_path)
        if reference_key in reference_keys:
            raise SourceSnapshotError(
                "inventory contains a duplicate physical asset reference")
        if collision_key in destination_keys:
            raise SourceSnapshotError(
                "inventory assets normalize to the same destination")
        reference_keys.add(reference_key)
        destination_keys.add(collision_key)
        validated_assets.append(_ValidatedAsset(
            reference=reference,
            normalized_member_path=normalized_path,
            destination_collision_key=collision_key,
        ))

    validated_sources.sort(key=lambda item: item.source_object_id)
    validated_assets.sort(key=lambda item: (
        item.reference.source_object_id,
        item.normalized_member_path,
        item.reference.recording_id,
        item.reference.asset_role,
    ))
    if request.dataset_id == VIDIMU_V2_DATASET_ID:
        by_recording: defaultdict[str, list[InventoryAssetReference]] = (
            defaultdict(list)
        )
        for asset in validated_assets:
            by_recording[asset.reference.recording_id].append(asset.reference)
        recording_count = len(by_recording)
        if recording_count != VIDIMU_V2_EXPECTED_RECORDING_COUNT \
                or len(validated_assets) \
                != VIDIMU_V2_EXPECTED_ASSET_REFERENCE_COUNT:
            raise SourceSnapshotError(
                "VIDIMU v2 frozen inventory must contain exactly 208 "
                "recordings and 624 asset references"
            )
        expected_topology = {
            (
                "vidimu-dataset-archive",
                "ANNOTATION",
                "BODYTRACK_POSE",
            ),
            (
                "vidimu-dataset-archive",
                "IMU",
                "INERTIAL_QUATERNION",
            ),
            (
                "vidimu-video-archive-small",
                "VIDEO",
                "VISUAL",
            ),
        }
        for references in by_recording.values():
            observed_topology = {
                (
                    reference.source_object_id,
                    reference.asset_role,
                    reference.modality,
                )
                for reference in references
            }
            if len(references) != len(expected_topology) \
                    or observed_topology != expected_topology:
                raise SourceSnapshotError(
                    "each VIDIMU v2 recording must bind exactly one video, "
                    "one quaternion IMU asset, and one body-track annotation"
                )
        if any(
            asset.reference.availability != "REQUIRED"
            for asset in validated_assets
        ):
            raise SourceSnapshotError(
                "the frozen VIDIMU v2 public inventory contains only "
                "required available members"
            )
        if _vidimu_v2_asset_reference_authority_sha256(validated_assets) \
                != VIDIMU_V2_ASSET_REFERENCE_AUTHORITY_SHA256:
            raise SourceSnapshotError(
                "VIDIMU v2 references do not match the frozen canonical "
                "624-member authority"
            )
    return tuple(validated_sources), tuple(validated_assets), source_by_id


def _descriptor_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise SourceSnapshotError(
            "secure source reads require O_NOFOLLOW and O_NONBLOCK")
    return (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )


@contextmanager
def _local_stream(
    anchor: SourceObjectAnchor,
) -> Iterator[tuple[BinaryIO, str | None, str | None, Callable[[], None]]]:
    if anchor.local_path is None:
        raise SourceSnapshotError("local source staging requires a local path")
    try:
        descriptor = os.open(anchor.local_path, _read_flags())
    except OSError as exc:
        raise SourceSnapshotError(
            f"could not securely open source object {anchor.source_object_id}") from exc
    handle: BinaryIO | None = None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SourceSnapshotError("local source object must be a regular file")
        if before.st_size != anchor.expected_content_length:
            raise SourceSnapshotError("source object content length mismatch")
        handle = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = -1

        def revalidate() -> None:
            if handle is None:
                raise SourceSnapshotError("local source handle is unavailable")
            if _descriptor_identity(before) != _descriptor_identity(
                os.fstat(handle.fileno())
            ):
                raise SourceSnapshotError(
                    "local source object changed during verified staging")

        yield handle, None, None, revalidate
    finally:
        if handle is not None:
            handle.close()
        elif descriptor >= 0:
            os.close(descriptor)


@contextmanager
def _http_stream(
    anchor: SourceObjectAnchor,
    *,
    timeout_seconds: float,
) -> Iterator[tuple[BinaryIO, str | None, str | None, Callable[[], None]]]:
    request = urllib.request.Request(
        anchor.source_url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "tremora-vidimu-source-snapshot/0.4.0",
        },
        method="GET",
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout_seconds)
    except (OSError, urllib.error.URLError) as exc:
        raise SourceSnapshotError(
            f"source object download failed: {anchor.source_object_id}") from exc
    try:
        status = getattr(response, "status", None)
        if status != 200:
            raise SourceSnapshotError("source object HTTP response was not 200")
        content_encoding = response.headers.get("Content-Encoding")
        if content_encoding not in {None, "identity"}:
            raise SourceSnapshotError(
                "source object response used an unsupported content encoding")
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                header_length = int(content_length)
            except ValueError as exc:
                raise SourceSnapshotError(
                    "source object Content-Length is invalid") from exc
            if header_length != anchor.expected_content_length:
                raise SourceSnapshotError(
                    "source object Content-Length does not match its trust anchor")
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
        if anchor.expected_etag is not None and etag != anchor.expected_etag:
            raise SourceSnapshotError("source object ETag changed")
        if anchor.expected_last_modified is not None \
                and last_modified != anchor.expected_last_modified:
            raise SourceSnapshotError("source object Last-Modified changed")
        yield response, etag, last_modified, lambda: None
    finally:
        response.close()


def _verify_regular_file(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> tuple[int, int]:
    try:
        descriptor = os.open(path, _read_flags())
    except OSError as exc:
        raise SourceSnapshotError("verified artifact is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise SourceSnapshotError("verified artifact size mismatch")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            payload = os.pread(
                descriptor,
                min(_COPY_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
            if not payload:
                raise SourceSnapshotError("verified artifact ended during hashing")
            digest.update(payload)
            offset += len(payload)
        after = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after) \
                or digest.hexdigest() != expected_sha256:
            raise SourceSnapshotError("verified artifact hash mismatch")
        return before.st_dev, before.st_ino
    finally:
        os.close(descriptor)


def _read_regular_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    """Read bounded bytes from one pinned, no-follow regular descriptor."""

    maximum_bytes = _positive_integer(maximum_bytes, field="read byte limit")
    try:
        descriptor = os.open(path, _read_flags())
    except OSError as exc:
        raise SourceSnapshotError("verified artifact is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SourceSnapshotError("verified artifact must be a regular file")
        if before.st_size > maximum_bytes:
            raise SourceSnapshotError("verified artifact exceeds its read limit")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise SourceSnapshotError("verified artifact size mismatch")
        payload = bytearray()
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(_COPY_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise SourceSnapshotError("verified artifact ended during read")
            payload.extend(chunk)
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise SourceSnapshotError("verified artifact changed during read")
        if expected_sha256 is not None \
                and digest.hexdigest() != expected_sha256:
            raise SourceSnapshotError("verified artifact hash mismatch")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _verify_source_object_file(
    path: Path,
    anchor: SourceObjectAnchor,
) -> bool | None:
    """Recompute SHA-256 and any provider checksum from one pinned descriptor."""

    try:
        descriptor = os.open(path, _read_flags())
    except OSError as exc:
        raise SourceSnapshotError("source object file is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) \
                or before.st_size != anchor.expected_content_length:
            raise SourceSnapshotError("source object file size mismatch")
        sha256_digest = hashlib.sha256()
        md5_digest = hashlib.md5(usedforsecurity=False)
        offset = 0
        while offset < before.st_size:
            payload = os.pread(
                descriptor,
                min(_COPY_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
            if not payload:
                raise SourceSnapshotError("source object ended during verification")
            sha256_digest.update(payload)
            md5_digest.update(payload)
            offset += len(payload)
        after = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise SourceSnapshotError("source object changed during verification")
        observed_sha256 = sha256_digest.hexdigest()
        if observed_sha256 != anchor.source_object_sha256:
            raise SourceSnapshotError("source object SHA-256 mismatch")
        algorithm, value = _provider_checksum(
            anchor.provider_checksum_algorithm,
            anchor.provider_checksum_value,
        )
        if algorithm is None:
            return None
        observed = md5_digest.hexdigest() if algorithm == "MD5" \
            else observed_sha256
        if observed != value:
            raise SourceSnapshotError("source object provider checksum mismatch")
        return True
    finally:
        os.close(descriptor)


def _verify_source_object_descriptor(
    descriptor: int,
    anchor: SourceObjectAnchor,
    *,
    expected_identity: tuple[int, int, int, int, int] | None = None,
) -> tuple[int, int, int, int, int]:
    before = os.fstat(descriptor)
    identity = _descriptor_identity(before)
    if not stat.S_ISREG(before.st_mode) \
            or before.st_size != anchor.expected_content_length:
        raise SourceSnapshotError("source object descriptor size mismatch")
    if expected_identity is not None and identity != expected_identity:
        raise SourceSnapshotError("source object descriptor identity changed")
    sha256_digest = hashlib.sha256()
    md5_digest = hashlib.md5(usedforsecurity=False)
    offset = 0
    while offset < before.st_size:
        payload = os.pread(
            descriptor,
            min(_COPY_CHUNK_BYTES, before.st_size - offset),
            offset,
        )
        if not payload:
            raise SourceSnapshotError(
                "source object descriptor ended during verification"
            )
        sha256_digest.update(payload)
        md5_digest.update(payload)
        offset += len(payload)
    after = os.fstat(descriptor)
    if identity != _descriptor_identity(after) \
            or sha256_digest.hexdigest() != anchor.source_object_sha256:
        raise SourceSnapshotError("source object descriptor hash changed")
    algorithm, expected_checksum = _provider_checksum(
        anchor.provider_checksum_algorithm,
        anchor.provider_checksum_value,
    )
    if algorithm is not None:
        observed_checksum = (
            md5_digest.hexdigest()
            if algorithm == "MD5"
            else sha256_digest.hexdigest()
        )
        if observed_checksum != expected_checksum:
            raise SourceSnapshotError(
                "source object descriptor provider checksum changed"
            )
    return identity


def _pread_exact(descriptor: int, size: int, offset: int) -> bytes:
    if size < 0 or offset < 0:
        raise SourceSnapshotError("ZIP structure uses a negative byte range")
    payload = bytearray()
    while len(payload) < size:
        chunk = os.pread(
            descriptor,
            size - len(payload),
            offset + len(payload),
        )
        if not chunk:
            raise SourceSnapshotError(
                "ZIP structure ended inside a required record"
            )
        payload.extend(chunk)
    return bytes(payload)


def _preflight_zip_central_directory(
    descriptor: int,
    *,
    archive_bytes: int,
    max_members: int,
) -> tuple[int, int]:
    """Bound central-directory allocation before ``ZipFile`` sees the object."""

    eocd_size = 22
    maximum_comment_bytes = 65_535
    tail_size = min(archive_bytes, eocd_size + maximum_comment_bytes)
    tail_offset = archive_bytes - tail_size
    tail = _pread_exact(descriptor, tail_size, tail_offset)
    signature = b"PK\x05\x06"
    eocd_index = tail.rfind(signature)
    if eocd_index < 0 or eocd_index + eocd_size > len(tail):
        raise SourceSnapshotError("ZIP end-of-central-directory record is absent")
    (
        observed_signature,
        disk_number,
        central_directory_disk,
        entries_on_disk,
        entry_count,
        central_directory_bytes,
        central_directory_offset,
        comment_bytes,
    ) = struct.unpack_from("<4s4H2LH", tail, eocd_index)
    eocd_offset = tail_offset + eocd_index
    if observed_signature != signature \
            or eocd_offset + eocd_size + comment_bytes != archive_bytes:
        raise SourceSnapshotError("ZIP end-of-central-directory record is invalid")
    if disk_number != 0 or central_directory_disk != 0 \
            or entries_on_disk != entry_count:
        raise SourceSnapshotError("multi-disk ZIP archives are unsupported")

    central_directory_limit = eocd_offset
    if entry_count == 0xFFFF \
            or central_directory_bytes == 0xFFFFFFFF \
            or central_directory_offset == 0xFFFFFFFF:
        locator_offset = eocd_offset - 20
        locator = _pread_exact(descriptor, 20, locator_offset)
        locator_signature, zip64_disk, zip64_offset, total_disks = struct.unpack(
            "<4sLQL", locator
        )
        if locator_signature != b"PK\x06\x07" \
                or zip64_disk != 0 or total_disks != 1:
            raise SourceSnapshotError("ZIP64 locator is invalid")
        zip64_prefix = _pread_exact(descriptor, 56, zip64_offset)
        (
            zip64_signature,
            zip64_record_bytes,
            _version_made_by,
            _version_needed,
            zip64_disk_number,
            zip64_central_directory_disk,
            zip64_entries_on_disk,
            zip64_entry_count,
            zip64_central_directory_bytes,
            zip64_central_directory_offset,
        ) = struct.unpack("<4sQ2H2L4Q", zip64_prefix)
        if zip64_signature != b"PK\x06\x06" \
                or zip64_record_bytes < 44 \
                or zip64_offset + 12 + zip64_record_bytes > locator_offset \
                or zip64_disk_number != 0 \
                or zip64_central_directory_disk != 0 \
                or zip64_entries_on_disk != zip64_entry_count:
            raise SourceSnapshotError("ZIP64 end record is invalid")
        entry_count = zip64_entry_count
        central_directory_bytes = zip64_central_directory_bytes
        central_directory_offset = zip64_central_directory_offset
        central_directory_limit = zip64_offset

    if entry_count > max_members or entry_count > _MAX_ARCHIVE_MEMBERS:
        raise SourceSnapshotError("ZIP exceeds the archive-member count budget")
    maximum_declared_directory_bytes = entry_count * (
        _ZIP_CENTRAL_HEADER_BYTES
        + _MAX_MEMBER_PATH_BYTES
        + 2 * _MAX_ZIP_EXTRA_OR_COMMENT_BYTES
    )
    if central_directory_bytes > _MAX_ZIP_CENTRAL_DIRECTORY_BYTES \
            or central_directory_bytes > maximum_declared_directory_bytes:
        raise SourceSnapshotError("ZIP central directory exceeds its byte budget")
    if central_directory_offset + central_directory_bytes \
            > central_directory_limit:
        raise SourceSnapshotError("ZIP central-directory byte range is invalid")
    return int(entry_count), int(central_directory_bytes)


@contextmanager
def _pinned_zip_archive(
    path: Path,
    anchor: SourceObjectAnchor,
    *,
    max_members: int,
) -> Iterator[zipfile.ZipFile]:
    try:
        descriptor = os.open(path, _read_flags())
    except OSError as exc:
        raise SourceSnapshotError(
            "source ZIP cannot be opened through a pinned descriptor"
        ) from exc
    handle: BinaryIO | None = None
    archive: zipfile.ZipFile | None = None
    try:
        identity = _verify_source_object_descriptor(descriptor, anchor)
        _preflight_zip_central_directory(
            descriptor,
            archive_bytes=anchor.expected_content_length,
            max_members=max_members,
        )
        handle = os.fdopen(descriptor, "rb", buffering=0, closefd=False)
        archive = zipfile.ZipFile(handle, "r")
        yield archive
        _verify_source_object_descriptor(
            descriptor,
            anchor,
            expected_identity=identity,
        )
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise SourceSnapshotError("source object is not a valid ZIP") from exc
    finally:
        if archive is not None:
            archive.close()
        if handle is not None:
            handle.close()
        os.close(descriptor)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise SourceSnapshotError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _canonical_json_document(payload: bytes, *, field: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceSnapshotError(f"{field} is not valid canonical JSON") from exc
    if not isinstance(value, dict):
        raise SourceSnapshotError(f"{field} must be a JSON object")
    try:
        if canonical_json_bytes(value) != payload:
            raise SourceSnapshotError(f"{field} is not canonically encoded")
    except FinalizationBundleError as exc:
        raise SourceSnapshotError(f"{field} is not canonical JSON") from exc
    return value


def _verify_vidimu_v2_record_metadata(path: Path) -> None:
    payload = _read_regular_bytes(
        path,
        maximum_bytes=VIDIMU_V2_RECORD_METADATA_BYTES,
        expected_bytes=VIDIMU_V2_RECORD_METADATA_BYTES,
        expected_sha256=VIDIMU_V2_RECORD_METADATA_SHA256,
    )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceSnapshotError("VIDIMU v2 record metadata is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SourceSnapshotError("VIDIMU v2 record metadata is not an object")
    metadata = value.get("metadata")
    files = value.get("files")
    if not isinstance(metadata, dict) or not isinstance(files, list):
        raise SourceSnapshotError("VIDIMU v2 record metadata schema changed")
    license_value = metadata.get("license")
    if value.get("id") != int(VIDIMU_V2_ZENODO_RECORD_ID) \
            or value.get("conceptrecid") != "7681316" \
            or value.get("doi") != VIDIMU_V2_CITATION_ID \
            or metadata.get("version") != VIDIMU_V2_DATASET_VERSION \
            or not isinstance(license_value, dict) \
            or license_value.get("id") != "cc-by-4.0":
        raise SourceSnapshotError("VIDIMU v2 record identity changed")
    observed_files: dict[str, tuple[object, object, object]] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str) \
                or item["key"] in observed_files:
            raise SourceSnapshotError("VIDIMU v2 record file inventory is invalid")
        links = item.get("links")
        if not isinstance(links, dict):
            raise SourceSnapshotError("VIDIMU v2 record file links are invalid")
        observed_files[item["key"]] = (
            item.get("size"), item.get("checksum"), links.get("self"),
        )
    if observed_files != dict(_VIDIMU_V2_RECORD_FILE_PINS):
        raise SourceSnapshotError("VIDIMU v2 record file inventory changed")


@dataclass(frozen=True, slots=True)
class _VerifiedParquet:
    table: pa.Table
    row_group_rows: tuple[int, ...]
    compression_codecs: frozenset[str]
    encodings: frozenset[str]
    all_columns_have_statistics: bool


def _read_verified_parquet(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> _VerifiedParquet:
    if expected_bytes > _MAX_PARQUET_MANIFEST_BYTES:
        raise SourceSnapshotError("Parquet artifact exceeds its verifier byte budget")
    try:
        descriptor = os.open(path, _read_flags())
    except OSError as exc:
        raise SourceSnapshotError("source snapshot Parquet is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise SourceSnapshotError("source snapshot Parquet size mismatch")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            payload = os.pread(
                descriptor,
                min(_COPY_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
            if not payload:
                raise SourceSnapshotError("source snapshot Parquet ended during hash")
            digest.update(payload)
            offset += len(payload)
        if digest.hexdigest() != expected_sha256:
            raise SourceSnapshotError("source snapshot Parquet hash mismatch")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            parquet_file = pq.ParquetFile(handle)
            metadata = parquet_file.metadata
            row_group_rows = tuple(
                metadata.row_group(index).num_rows
                for index in range(metadata.num_row_groups)
            )
            codecs: set[str] = set()
            encodings: set[str] = set()
            all_statistics = True
            decoded_bytes = 0
            for row_group_index in range(metadata.num_row_groups):
                row_group = metadata.row_group(row_group_index)
                for column_index in range(row_group.num_columns):
                    column = row_group.column(column_index)
                    codecs.add(str(column.compression))
                    encodings.update(str(value) for value in column.encodings)
                    all_statistics = all_statistics and column.statistics is not None
                    decoded_bytes += column.total_uncompressed_size
                    if decoded_bytes > _MAX_PARQUET_DECODED_BYTES:
                        raise SourceSnapshotError(
                            "Parquet artifact exceeds its decoded byte budget")
            table = parquet_file.read()
        after = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise SourceSnapshotError("source snapshot Parquet changed during read")
        return _VerifiedParquet(
            table=table,
            row_group_rows=row_group_rows,
            compression_codecs=frozenset(codecs),
            encodings=frozenset(encodings),
            all_columns_have_statistics=all_statistics,
        )
    except (OSError, pa.ArrowException) as exc:
        raise SourceSnapshotError("source snapshot Parquet is unreadable") from exc
    finally:
        os.close(descriptor)


def _rename_verified_noreplace(source: Path, destination: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(destination.parent, flags)
    try:
        _rename_noreplace(descriptor, source.name, destination.name)
        os.fsync(descriptor)
    except FinalizationBundleError as exc:
        raise SourceSnapshotError(
            "verified object destination already exists") from exc
    finally:
        os.close(descriptor)


def _stage_source_object(
    anchor: SourceObjectAnchor,
    objects_dir: Path,
    *,
    timeout_seconds: float,
) -> _DownloadResult:
    destination = objects_dir / anchor.source_object_sha256
    partial = objects_dir / (
        f".{anchor.source_object_sha256}.partial-{uuid.uuid4().hex}"
    )
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(partial, flags, 0o600)
        stream_context = (
            _local_stream(anchor)
            if anchor.local_path is not None
            else _http_stream(anchor, timeout_seconds=timeout_seconds)
        )
        observed = 0
        sha256_digest = hashlib.sha256()
        md5_digest = hashlib.md5(usedforsecurity=False)
        etag: str | None = None
        last_modified: str | None = None
        with os.fdopen(descriptor, "wb", buffering=0) as output:
            descriptor = -1
            with stream_context as (source, etag, last_modified, revalidate):
                while True:
                    payload = source.read(_COPY_CHUNK_BYTES)
                    if not payload:
                        break
                    observed += len(payload)
                    if observed > anchor.expected_content_length:
                        raise SourceSnapshotError(
                            "source object exceeded expected content length")
                    sha256_digest.update(payload)
                    md5_digest.update(payload)
                    output.write(payload)
                revalidate()
            output.flush()
            os.fsync(output.fileno())
        if observed != anchor.expected_content_length:
            raise SourceSnapshotError("source object content length mismatch")
        observed_sha256 = sha256_digest.hexdigest()
        observed_md5 = md5_digest.hexdigest()
        if observed_sha256 != anchor.source_object_sha256:
            raise SourceSnapshotError("source object SHA-256 mismatch")
        provider_algorithm, provider_value = _provider_checksum(
            anchor.provider_checksum_algorithm,
            anchor.provider_checksum_value,
        )
        provider_verified: bool | None = None
        if provider_algorithm is not None:
            observed_provider = (
                observed_md5 if provider_algorithm == "MD5" else observed_sha256
            )
            if observed_provider != provider_value:
                raise SourceSnapshotError("source object provider checksum mismatch")
            provider_verified = True
        staged_inode = _verify_regular_file(
            partial,
            expected_bytes=observed,
            expected_sha256=anchor.source_object_sha256,
        )
        if destination.exists() or destination.is_symlink():
            _verify_regular_file(
                destination,
                expected_bytes=observed,
                expected_sha256=anchor.source_object_sha256,
            )
            partial.unlink()
        else:
            try:
                _rename_verified_noreplace(partial, destination)
            except SourceSnapshotError:
                if destination.exists() and not destination.is_symlink():
                    _verify_regular_file(
                        destination,
                        expected_bytes=observed,
                        expected_sha256=anchor.source_object_sha256,
                    )
                    partial.unlink(missing_ok=True)
                else:
                    raise
            else:
                published_inode = _verify_regular_file(
                    destination,
                    expected_bytes=observed,
                    expected_sha256=anchor.source_object_sha256,
                )
                if published_inode != staged_inode:
                    raise SourceSnapshotError(
                        "published source object inode changed during rename")
        return _DownloadResult(
            path=destination,
            observed_content_length=observed,
            etag=etag,
            last_modified=last_modified,
            provider_checksum_verified=provider_verified,
        )
    except Exception as exc:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        if isinstance(exc, SourceSnapshotError):
            raise
        raise SourceSnapshotError(
            f"source object staging failed: {anchor.source_object_id}") from exc


def _zip_index(
    archive: zipfile.ZipFile,
    *,
    max_members: int,
    max_member_bytes: int,
    max_compression_ratio: float,
) -> tuple[dict[str, zipfile.ZipInfo], dict[str, str]]:
    infos = archive.infolist()
    if len(infos) > max_members:
        raise SourceSnapshotError("ZIP exceeds the archive-member count budget")
    by_raw_name: dict[str, zipfile.ZipInfo] = {}
    normalized_by_raw_name: dict[str, str] = {}
    collision_keys: dict[str, str] = {}
    info_by_collision_key: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        normalized, collision_key = _normalize_member_path(info.filename)
        prior = collision_keys.get(collision_key)
        if prior is not None:
            raise SourceSnapshotError(
                "ZIP contains duplicate normalized member destinations")
        if info.filename in by_raw_name:
            raise SourceSnapshotError("ZIP contains a duplicate member name")
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise SourceSnapshotError("ZIP symlink members are forbidden")
        file_type = stat.S_IFMT(mode)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise SourceSnapshotError("ZIP special-file members are forbidden")
        if info.flag_bits & 1:
            raise SourceSnapshotError("encrypted ZIP members are forbidden")
        if not info.is_dir():
            if info.file_size < 0 or info.compress_size < 0 \
                    or info.file_size > max_member_bytes:
                raise SourceSnapshotError(
                    "ZIP member exceeds the archive-member byte budget")
            if info.file_size and (
                info.compress_size == 0
                or info.file_size / info.compress_size > max_compression_ratio
            ):
                raise SourceSnapshotError(
                    "ZIP member exceeds the compression-ratio budget")
        collision_keys[collision_key] = info.filename
        info_by_collision_key[collision_key] = info
        by_raw_name[info.filename] = info
        normalized_by_raw_name[info.filename] = normalized
    for collision_key, info in info_by_collision_key.items():
        parts = collision_key.split("/")
        for index in range(1, len(parts)):
            parent = info_by_collision_key.get("/".join(parts[:index]))
            if parent is not None and not parent.is_dir():
                raise SourceSnapshotError(
                    "ZIP file and descendant members collide topologically")
    return by_raw_name, normalized_by_raw_name


def _extract_one(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    asset: _ValidatedAsset,
    assets_dir: Path,
) -> tuple[int, str, Path]:
    reference = asset.reference
    if reference.expected_size_bytes is None \
            or reference.expected_sha256 is None:
        raise SourceSnapshotError(
            "required inventory asset lacks size or hash anchors")
    if info.is_dir():
        raise SourceSnapshotError("inventory references a ZIP directory")
    if info.file_size != reference.expected_size_bytes:
        raise SourceSnapshotError("inventory asset size disagrees with ZIP metadata")
    destination = assets_dir.joinpath(*asset.normalized_member_path.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise SourceSnapshotError(
            "two inventory assets target the same extracted destination")
    partial = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(partial, flags, 0o600)
        observed = 0
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb", buffering=0) as output:
            descriptor = -1
            with archive.open(info, "r") as source:
                while True:
                    payload = source.read(_COPY_CHUNK_BYTES)
                    if not payload:
                        break
                    observed += len(payload)
                    if observed > reference.expected_size_bytes:
                        raise SourceSnapshotError(
                            "extracted asset exceeded its inventory size")
                    digest.update(payload)
                    output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if observed != reference.expected_size_bytes:
            raise SourceSnapshotError("extracted asset size mismatch")
        observed_hash = digest.hexdigest()
        if observed_hash != reference.expected_sha256:
            raise SourceSnapshotError("extracted asset SHA-256 mismatch")
        staged_inode = _verify_regular_file(
            partial,
            expected_bytes=observed,
            expected_sha256=observed_hash,
        )
        _rename_verified_noreplace(partial, destination)
        published_inode = _verify_regular_file(
            destination,
            expected_bytes=observed,
            expected_sha256=observed_hash,
        )
        if published_inode != staged_inode:
            raise SourceSnapshotError(
                "published extracted asset inode changed during rename")
        return observed, observed_hash, destination
    except Exception as exc:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        if isinstance(exc, SourceSnapshotError):
            raise
        raise SourceSnapshotError("ZIP asset extraction failed") from exc


def _extract_inventory_assets(
    assets: Sequence[_ValidatedAsset],
    source_paths: Mapping[str, Path],
    sources: Mapping[str, SourceObjectAnchor],
    assets_dir: Path,
    *,
    max_archive_members: int,
    max_archive_member_bytes: int,
    max_compression_ratio: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_source: dict[str, list[_ValidatedAsset]] = defaultdict(list)
    for asset in assets:
        by_source[asset.reference.source_object_id].append(asset)
    rows: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    for source_object_id, anchor in sorted(sources.items()):
        if anchor.archive_type != "ZIP":
            continue
        try:
            with _pinned_zip_archive(
                source_paths[source_object_id],
                anchor,
                max_members=max_archive_members,
            ) as archive:
                by_raw_name, normalized_by_raw_name = _zip_index(
                    archive,
                    max_members=max_archive_members,
                    max_member_bytes=max_archive_member_bytes,
                    max_compression_ratio=max_compression_ratio,
                )
                for asset in by_source.get(source_object_id, ()):
                    reference = asset.reference
                    info = by_raw_name.get(reference.archive_member_path)
                    if reference.availability == "UNAVAILABLE":
                        if info is not None:
                            raise SourceSnapshotError(
                                "inventory marks an available archive member unavailable")
                        rows.append({
                            "source_object_id": source_object_id,
                            "archive_member_path": reference.archive_member_path,
                            "normalized_member_path": asset.normalized_member_path,
                            "recording_id": reference.recording_id,
                            "asset_role": reference.asset_role,
                            "modality": reference.modality,
                            "asset_size_bytes": None,
                            "asset_sha256": None,
                            "extraction_status": "UNAVAILABLE",
                            "failure_reason": reference.unavailable_reason,
                        })
                        continue
                    if info is None:
                        raise SourceSnapshotError(
                            "required inventory asset is absent from its archive")
                    if normalized_by_raw_name[info.filename] \
                            != asset.normalized_member_path:
                        raise SourceSnapshotError(
                            "archive member normalization changed during reconciliation")
                    size, observed_hash, destination = _extract_one(
                        archive, info, asset, assets_dir)
                    relative_path = destination.relative_to(
                        assets_dir.parent).as_posix()
                    rows.append({
                        "source_object_id": source_object_id,
                        "archive_member_path": reference.archive_member_path,
                        "normalized_member_path": asset.normalized_member_path,
                        "recording_id": reference.recording_id,
                        "asset_role": reference.asset_role,
                        "modality": reference.modality,
                        "asset_size_bytes": size,
                        "asset_sha256": observed_hash,
                        "extraction_status": "VERIFIED",
                        "failure_reason": None,
                    })
                    artifacts.append({
                        "path": relative_path,
                        "kind": "EXTRACTED_ASSET",
                        "bytes": size,
                        "sha256": observed_hash,
                        "source_object_id": source_object_id,
                        "archive_member_path": reference.archive_member_path,
                    })
        except OSError as exc:
            raise SourceSnapshotError(
                f"source object is not a valid ZIP: {source_object_id}") from exc
    if len(rows) != len(assets):
        raise SourceSnapshotError("inventory asset reconciliation is incomplete")
    rows.sort(key=lambda row: (
        str(row["source_object_id"]),
        str(row["normalized_member_path"]),
        str(row["recording_id"]),
        str(row["asset_role"]),
    ))
    artifacts.sort(key=lambda row: str(row["path"]))
    return rows, artifacts


def _verify_inventory_against_pinned_archives(
    snapshot_path: Path,
    request: VidimuSourceSnapshotRequest,
    sources: Sequence[SourceObjectAnchor],
    assets: Sequence[_ValidatedAsset],
) -> None:
    by_source: defaultdict[str, list[_ValidatedAsset]] = defaultdict(list)
    for asset in assets:
        by_source[asset.reference.source_object_id].append(asset)
    checked_assets = 0
    for anchor in sources:
        if anchor.archive_type != "ZIP":
            continue
        archive_path = snapshot_path / "objects" / anchor.source_object_sha256
        with _pinned_zip_archive(
            archive_path,
            anchor,
            max_members=request.max_archive_members,
        ) as archive:
            by_raw_name, normalized_by_raw_name = _zip_index(
                archive,
                max_members=request.max_archive_members,
                max_member_bytes=request.max_archive_member_bytes,
                max_compression_ratio=float(request.max_compression_ratio),
            )
            for asset in by_source.get(anchor.source_object_id, ()):
                reference = asset.reference
                info = by_raw_name.get(reference.archive_member_path)
                if reference.availability == "UNAVAILABLE":
                    if info is not None:
                        raise SourceSnapshotError(
                            "unavailable inventory member exists in pinned archive"
                        )
                    checked_assets += 1
                    continue
                if info is None or info.is_dir() \
                        or normalized_by_raw_name[info.filename] \
                        != asset.normalized_member_path \
                        or info.file_size != reference.expected_size_bytes:
                    raise SourceSnapshotError(
                        "required inventory member contradicts pinned archive"
                    )
                digest = hashlib.sha256()
                observed_bytes = 0
                with archive.open(info, "r") as source:
                    while True:
                        payload = source.read(_COPY_CHUNK_BYTES)
                        if not payload:
                            break
                        observed_bytes += len(payload)
                        if observed_bytes > reference.expected_size_bytes:
                            raise SourceSnapshotError(
                                "pinned archive member exceeded inventory size"
                            )
                        digest.update(payload)
                if observed_bytes != reference.expected_size_bytes \
                        or digest.hexdigest() != reference.expected_sha256:
                    raise SourceSnapshotError(
                        "pinned archive member content contradicts inventory"
                    )
                checked_assets += 1
    if checked_assets != len(assets):
        raise SourceSnapshotError(
            "not every inventory member was checked against a pinned archive"
        )


def _exclusive_bytes(path: Path, payload: bytes) -> dict[str, object]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SourceSnapshotError(
            f"could not create snapshot artifact: {path.name}") from exc
    with os.fdopen(descriptor, "wb", buffering=0) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "kind": "CANONICAL_JSON",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_table(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    schema: pa.Schema,
    *,
    sort_keys: tuple[str, ...],
) -> dict[str, object]:
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    table = table.sort_by([(key, "ascending") for key in sort_keys])
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SourceSnapshotError(
            f"could not create Parquet artifact: {path.name}") from exc
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
    payload_hash = _sha256_path(path)
    return {
        "path": path.name,
        "kind": "PARQUET_MANIFEST",
        "bytes": path.stat().st_size,
        "sha256": payload_hash,
        "rows": table.num_rows,
        "schema_sha256": schema_fingerprint(table.schema),
        "semantic_sha256": semantic_table_hash(table, sort_keys=sort_keys),
        "sort_keys": list(sort_keys),
        "row_group_size": ROW_GROUP_SIZE,
        "writer_policy_id": PARQUET_WRITER_POLICY_ID,
    }


def _sha256_path(path: Path) -> str:
    try:
        descriptor = os.open(path, _read_flags())
    except OSError as exc:
        raise SourceSnapshotError("artifact cannot be opened for hashing") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SourceSnapshotError("artifact being hashed is not regular")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            payload = os.pread(
                descriptor,
                min(_COPY_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
            if not payload:
                raise SourceSnapshotError("artifact ended while being hashed")
            digest.update(payload)
            offset += len(payload)
        after = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise SourceSnapshotError("artifact changed while being hashed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _inventory_payload(
    request: VidimuSourceSnapshotRequest,
    sources: Sequence[SourceObjectAnchor],
    assets: Sequence[_ValidatedAsset],
) -> bytes:
    value = {
        "inventory_schema_version": VIDIMU_SOURCE_INVENTORY_SCHEMA_VERSION,
        "dataset_id": request.dataset_id,
        "dataset_version": request.dataset_version,
        "license_id": request.license_id,
        "citation_id": request.citation_id,
        "terms_snapshot_id": request.terms_snapshot_id,
        "terms_source_object_id": request.terms_source_object_id,
        "source_objects": [{
            "source_object_id": source.source_object_id,
            "source_url": source.source_url,
            "source_provider_id": source.source_provider_id,
            "expected_content_length": source.expected_content_length,
            "source_object_sha256": source.source_object_sha256,
            "mime_type": source.mime_type,
            "archive_type": source.archive_type,
            "expected_etag": source.expected_etag,
            "expected_last_modified": source.expected_last_modified,
            "provider_checksum_algorithm": source.provider_checksum_algorithm,
            "provider_checksum_value": source.provider_checksum_value,
        } for source in sources],
        "asset_references": [{
            "source_object_id": asset.reference.source_object_id,
            "archive_member_path": asset.reference.archive_member_path,
            "normalized_member_path": asset.normalized_member_path,
            "recording_id": asset.reference.recording_id,
            "asset_role": asset.reference.asset_role,
            "modality": asset.reference.modality,
            "expected_size_bytes": asset.reference.expected_size_bytes,
            "expected_sha256": asset.reference.expected_sha256,
            "availability": asset.reference.availability,
            "unavailable_reason": asset.reference.unavailable_reason,
        } for asset in assets],
        "budgets": {
            "max_source_object_bytes": request.max_source_object_bytes,
            "max_archive_member_bytes": request.max_archive_member_bytes,
            "max_total_extracted_bytes": request.max_total_extracted_bytes,
            "max_compression_ratio": float(request.max_compression_ratio),
            "max_archive_members": request.max_archive_members,
        },
    }
    try:
        return canonical_json_bytes(value)
    except FinalizationBundleError as exc:
        raise SourceSnapshotError("source inventory is not canonical JSON") from exc


def _parse_source_inventory(
    payload: bytes,
) -> tuple[
    VidimuSourceSnapshotRequest,
    tuple[SourceObjectAnchor, ...],
    tuple[_ValidatedAsset, ...],
    dict[str, SourceObjectAnchor],
]:
    value = _canonical_json_document(payload, field="source inventory")
    if set(value) != {
        "inventory_schema_version",
        "dataset_id",
        "dataset_version",
        "license_id",
        "citation_id",
        "terms_snapshot_id",
        "terms_source_object_id",
        "source_objects",
        "asset_references",
        "budgets",
    } or value["inventory_schema_version"] \
            != VIDIMU_SOURCE_INVENTORY_SCHEMA_VERSION:
        raise SourceSnapshotError("source inventory contract is invalid")
    raw_sources = value["source_objects"]
    raw_assets = value["asset_references"]
    budgets = value["budgets"]
    if not isinstance(raw_sources, list) or not isinstance(raw_assets, list) \
            or not isinstance(budgets, dict) or set(budgets) != {
                "max_source_object_bytes",
                "max_archive_member_bytes",
                "max_total_extracted_bytes",
                "max_compression_ratio",
                "max_archive_members",
            }:
        raise SourceSnapshotError("source inventory collections are invalid")
    source_fields = {
        "source_object_id",
        "source_url",
        "source_provider_id",
        "expected_content_length",
        "source_object_sha256",
        "mime_type",
        "archive_type",
        "expected_etag",
        "expected_last_modified",
        "provider_checksum_algorithm",
        "provider_checksum_value",
    }
    sources: list[SourceObjectAnchor] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict) or set(raw_source) != source_fields:
            raise SourceSnapshotError("source inventory object entry is invalid")
        sources.append(SourceObjectAnchor(
            source_object_id=raw_source["source_object_id"],
            source_url=raw_source["source_url"],
            source_provider_id=raw_source["source_provider_id"],
            expected_content_length=raw_source["expected_content_length"],
            source_object_sha256=raw_source["source_object_sha256"],
            mime_type=raw_source["mime_type"],
            archive_type=raw_source["archive_type"],
            expected_etag=raw_source["expected_etag"],
            expected_last_modified=raw_source["expected_last_modified"],
            provider_checksum_algorithm=(
                raw_source["provider_checksum_algorithm"]
            ),
            provider_checksum_value=raw_source["provider_checksum_value"],
        ))
    asset_fields = {
        "source_object_id",
        "archive_member_path",
        "normalized_member_path",
        "recording_id",
        "asset_role",
        "modality",
        "expected_size_bytes",
        "expected_sha256",
        "availability",
        "unavailable_reason",
    }
    assets: list[InventoryAssetReference] = []
    normalized_paths: list[str] = []
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, dict) or set(raw_asset) != asset_fields:
            raise SourceSnapshotError("source inventory asset entry is invalid")
        normalized_path = raw_asset["normalized_member_path"]
        if not isinstance(normalized_path, str):
            raise SourceSnapshotError(
                "source inventory normalized member path is invalid")
        normalized_paths.append(normalized_path)
        assets.append(InventoryAssetReference(
            source_object_id=raw_asset["source_object_id"],
            archive_member_path=raw_asset["archive_member_path"],
            recording_id=raw_asset["recording_id"],
            asset_role=raw_asset["asset_role"],
            modality=raw_asset["modality"],
            expected_size_bytes=raw_asset["expected_size_bytes"],
            expected_sha256=raw_asset["expected_sha256"],
            availability=raw_asset["availability"],
            unavailable_reason=raw_asset["unavailable_reason"],
        ))
    request = VidimuSourceSnapshotRequest(
        dataset_id=value["dataset_id"],
        dataset_version=value["dataset_version"],
        license_id=value["license_id"],
        citation_id=value["citation_id"],
        terms_snapshot_id=value["terms_snapshot_id"],
        terms_source_object_id=value["terms_source_object_id"],
        source_objects=tuple(sources),
        asset_references=tuple(assets),
        max_source_object_bytes=budgets["max_source_object_bytes"],
        max_archive_member_bytes=budgets["max_archive_member_bytes"],
        max_total_extracted_bytes=budgets["max_total_extracted_bytes"],
        max_compression_ratio=budgets["max_compression_ratio"],
        max_archive_members=budgets["max_archive_members"],
    )
    validated_sources, validated_assets, source_by_id = _validate_request(request)
    if normalized_paths != [
        asset.normalized_member_path for asset in validated_assets
    ] or _inventory_payload(request, validated_sources, validated_assets) != payload:
        raise SourceSnapshotError(
            "source inventory is not in canonical semantic order")
    return request, validated_sources, validated_assets, source_by_id


def _root(path: str | Path) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise SourceSnapshotError("snapshot root must be a real directory")
    return root.resolve(strict=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tree_file_hashes(path: Path) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for directory, child_directories, file_names in os.walk(
        path, topdown=True, followlinks=False,
    ):
        directory_path = Path(directory)
        for child_name in child_directories:
            child = directory_path / child_name
            metadata = child.lstat()
            if not stat.S_ISDIR(metadata.st_mode) \
                    or stat.S_ISLNK(metadata.st_mode):
                raise SourceSnapshotError(
                    "snapshot trees may contain only real directories")
        for file_name in file_names:
            candidate = directory_path / file_name
            relative = candidate.relative_to(path).as_posix()
            metadata = candidate.lstat()
            if not stat.S_ISREG(metadata.st_mode) \
                    or stat.S_ISLNK(metadata.st_mode):
                raise SourceSnapshotError(
                    "snapshot trees may contain only regular files")
            result[relative] = (metadata.st_size, _sha256_path(candidate))
    return result


def _tree_file_names(path: Path) -> set[str]:
    result: set[str] = set()
    for directory, child_directories, file_names in os.walk(
        path, topdown=True, followlinks=False,
    ):
        directory_path = Path(directory)
        for child_name in child_directories:
            child = directory_path / child_name
            flags = _read_flags() | getattr(os, "O_DIRECTORY", 0)
            try:
                descriptor = os.open(child, flags)
            except OSError as exc:
                raise SourceSnapshotError(
                    "snapshot trees may contain only real directories"
                ) from exc
            try:
                before = os.fstat(descriptor)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if not stat.S_ISDIR(before.st_mode) \
                    or _descriptor_identity(before) != _descriptor_identity(after):
                raise SourceSnapshotError(
                    "snapshot trees may contain only real directories")
        for file_name in file_names:
            candidate = directory_path / file_name
            try:
                descriptor = os.open(candidate, _read_flags())
            except OSError as exc:
                raise SourceSnapshotError(
                    "snapshot trees may contain only regular files"
                ) from exc
            try:
                before = os.fstat(descriptor)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if not stat.S_ISREG(before.st_mode) \
                    or _descriptor_identity(before) != _descriptor_identity(after):
                raise SourceSnapshotError(
                    "snapshot trees may contain only regular files")
            result.add(candidate.relative_to(path).as_posix())
    return result


def _validate_table_claims(
    artifact: Mapping[str, object],
    verified: _VerifiedParquet,
    *,
    schema: pa.Schema,
    sort_keys: tuple[str, ...],
) -> pa.Table:
    expected_fields = {
        "path",
        "kind",
        "bytes",
        "sha256",
        "rows",
        "schema_sha256",
        "semantic_sha256",
        "sort_keys",
        "row_group_size",
        "writer_policy_id",
    }
    if set(artifact) != expected_fields \
            or artifact.get("kind") != "PARQUET_MANIFEST":
        raise SourceSnapshotError("Parquet artifact claim is not closed")
    table = verified.table
    if not table.schema.equals(schema, check_metadata=True):
        raise SourceSnapshotError("source snapshot Parquet schema changed")
    rows = _nonnegative_integer(artifact.get("rows"), field="table rows")
    if rows != table.num_rows \
            or artifact.get("schema_sha256") != schema_fingerprint(table.schema) \
            or artifact.get("sort_keys") != list(sort_keys) \
            or artifact.get("row_group_size") != ROW_GROUP_SIZE \
            or artifact.get("writer_policy_id") != PARQUET_WRITER_POLICY_ID:
        raise SourceSnapshotError("Parquet artifact structural claims changed")
    semantic_hash = semantic_table_hash(table, sort_keys=sort_keys)
    if artifact.get("semantic_sha256") != semantic_hash:
        raise SourceSnapshotError("Parquet semantic hash claim changed")
    sorted_table = table.sort_by([(key, "ascending") for key in sort_keys])
    if not table.equals(sorted_table):
        raise SourceSnapshotError("Parquet rows are not in canonical sort order")
    if sum(verified.row_group_rows) != rows \
            or any(value <= 0 or value > ROW_GROUP_SIZE
                   for value in verified.row_group_rows) \
            or (rows == 0 and verified.row_group_rows):
        raise SourceSnapshotError("Parquet row-group policy changed")
    if rows > 0 and (
        verified.compression_codecs != {"ZSTD"}
        or not verified.all_columns_have_statistics
        or {"RLE_DICTIONARY", "PLAIN_DICTIONARY"}.intersection(
            verified.encodings
        )
    ):
        raise SourceSnapshotError("Parquet writer encoding policy changed")
    return table


def _directory_open_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise SourceSnapshotError(
            "snapshot publication requires O_DIRECTORY and O_NOFOLLOW"
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _neutralize_success_marker(directory_descriptor: int) -> None:
    try:
        metadata = os.stat(
            SUCCESS_MARKER_FILE,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.S_ISREG(metadata.st_mode):
        try:
            os.unlink(SUCCESS_MARKER_FILE, dir_fd=directory_descriptor)
            os.fsync(directory_descriptor)
        except OSError:
            pass


def _remove_owned_staging(
    staging: Path,
    staging_identity: tuple[int, int] | None,
) -> None:
    """Remove only the exact temporary directory created by this process."""

    try:
        current = staging.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        return
    if staging_identity is None:
        try:
            staging.rmdir()
        except OSError:
            pass
        return
    if not stat.S_ISDIR(current.st_mode) \
            or (current.st_dev, current.st_ino) != staging_identity:
        return
    shutil.rmtree(staging)


def _verified_existing_snapshot(
    root: Path,
    root_descriptor: int,
    snapshot_id: str,
) -> Path:
    try:
        destination_descriptor = os.open(
            snapshot_id,
            _directory_open_flags(),
            dir_fd=root_descriptor,
        )
    except OSError as exc:
        raise SourceSnapshotError(
            "content-addressed destination is not a real directory"
        ) from exc
    try:
        before = os.fstat(destination_descriptor)
        verified = verify_vidimu_source_snapshot(root / snapshot_id)
        after = os.fstat(destination_descriptor)
        current = os.stat(
            snapshot_id,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(before.st_mode) \
                or _descriptor_identity(before) != _descriptor_identity(after) \
                or (before.st_dev, before.st_ino) \
                != (current.st_dev, current.st_ino) \
                or verified.snapshot_manifest_sha256 != snapshot_id:
            raise SourceSnapshotError(
                "content-addressed destination changed during verification"
            )
        return verified.path
    finally:
        os.close(destination_descriptor)


def _publish_staging(
    staging: Path,
    root: Path,
    snapshot_id: str,
    *,
    staging_identity: tuple[int, int],
) -> Path:
    destination = root / snapshot_id
    root_descriptor = os.open(root, _directory_open_flags())
    staging_descriptor = -1
    destination_descriptor = -1
    try:
        root_state = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_state.st_mode):
            raise SourceSnapshotError("snapshot root descriptor is not a directory")
        try:
            staging_descriptor = os.open(
                staging.name,
                _directory_open_flags(),
                dir_fd=root_descriptor,
            )
        except OSError as exc:
            raise SourceSnapshotError(
                "snapshot staging directory disappeared before publication"
            ) from exc
        staging_state = os.fstat(staging_descriptor)
        if (staging_state.st_dev, staging_state.st_ino) != staging_identity:
            _neutralize_success_marker(staging_descriptor)
            raise SourceSnapshotError(
                "snapshot staging directory identity changed before publication"
            )
        try:
            _rename_noreplace(
                root_descriptor,
                staging.name,
                snapshot_id,
            )
        except FinalizationBundleError:
            return _verified_existing_snapshot(
                root,
                root_descriptor,
                snapshot_id,
            )
        os.fsync(root_descriptor)
        try:
            destination_descriptor = os.open(
                snapshot_id,
                _directory_open_flags(),
                dir_fd=root_descriptor,
            )
        except OSError as exc:
            raise SourceSnapshotError(
                "published snapshot cannot be pinned"
            ) from exc
        destination_state = os.fstat(destination_descriptor)
        pinned_staging_state = os.fstat(staging_descriptor)
        if (destination_state.st_dev, destination_state.st_ino) \
                != staging_identity \
                or (staging_state.st_dev, staging_state.st_ino) \
                != (pinned_staging_state.st_dev, pinned_staging_state.st_ino):
            _neutralize_success_marker(destination_descriptor)
            raise SourceSnapshotError(
                "published snapshot is not the pinned staging directory"
            )
        try:
            verified = verify_vidimu_source_snapshot(destination)
        except BaseException:
            _neutralize_success_marker(destination_descriptor)
            raise
        final_state = os.fstat(destination_descriptor)
        current = os.stat(
            snapshot_id,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if _descriptor_identity(destination_state) \
                != _descriptor_identity(final_state) \
                or (destination_state.st_dev, destination_state.st_ino) \
                != (current.st_dev, current.st_ino) \
                or verified.snapshot_manifest_sha256 != snapshot_id:
            _neutralize_success_marker(destination_descriptor)
            raise SourceSnapshotError(
                "published snapshot changed during strict verification"
            )
        return verified.path
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        os.close(root_descriptor)


def materialize_vidimu_source_snapshot(
    request: VidimuSourceSnapshotRequest,
    snapshot_root: str | Path,
    *,
    download_timestamp_utc: str | None = None,
    http_timeout_seconds: float = 120.0,
) -> MaterializedSourceSnapshot:
    """Materialize and atomically publish one immutable source snapshot.

    All trust anchors and inventory structure are validated before acquisition.
    A failed download, hash, ZIP, extraction, or reconciliation publishes no
    content-addressed directory and no ``_SUCCESS`` marker.
    """

    if isinstance(http_timeout_seconds, bool) \
            or not isinstance(http_timeout_seconds, (int, float)) \
            or http_timeout_seconds <= 0:
        raise SourceSnapshotError("http_timeout_seconds must be positive")
    timestamp = _timestamp(download_timestamp_utc)
    sources, assets, sources_by_id = _validate_request(request)
    root = _root(snapshot_root)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    published = False
    staging_identity: tuple[int, int] | None = None
    try:
        staging_state = staging.stat(follow_symlinks=False)
        if not stat.S_ISDIR(staging_state.st_mode) \
                or staging.resolve(strict=True).parent != root:
            raise SourceSnapshotError("snapshot staging directory is unsafe")
        staging_identity = (staging_state.st_dev, staging_state.st_ino)
        objects_dir = staging / "objects"
        assets_dir = staging / "assets"
        objects_dir.mkdir()
        assets_dir.mkdir()

        source_rows: list[dict[str, object]] = []
        source_paths: dict[str, Path] = {}
        source_artifacts: list[dict[str, object]] = []
        for source in sources:
            result = _stage_source_object(
                source,
                objects_dir,
                timeout_seconds=float(http_timeout_seconds),
            )
            source_paths[source.source_object_id] = result.path
            source_rows.append({
                "source_object_id": source.source_object_id,
                "dataset_id": request.dataset_id,
                "dataset_version": request.dataset_version,
                "source_url": source.source_url,
                "source_provider_id": source.source_provider_id,
                "download_timestamp_utc": timestamp,
                "expected_content_length": source.expected_content_length,
                "observed_content_length": result.observed_content_length,
                "etag": result.etag,
                "last_modified": result.last_modified,
                "source_object_sha256": source.source_object_sha256,
                "provider_checksum_algorithm": (
                    source.provider_checksum_algorithm
                ),
                "provider_checksum_value": source.provider_checksum_value,
                "provider_checksum_verified": (
                    result.provider_checksum_verified
                ),
                "mime_type": source.mime_type,
                "archive_type": source.archive_type,
                "license_id": request.license_id,
                "citation_id": request.citation_id,
                "terms_snapshot_id": request.terms_snapshot_id,
                "download_status": "VERIFIED",
                "failure_reason": None,
            })
            source_artifacts.append({
                "path": result.path.relative_to(staging).as_posix(),
                "kind": "SOURCE_OBJECT",
                "bytes": result.observed_content_length,
                "sha256": source.source_object_sha256,
                "source_object_id": source.source_object_id,
            })

        _verify_closed_vidimu_v2_authority(
            request, sources_by_id, source_paths)
        asset_rows, asset_artifacts = _extract_inventory_assets(
            assets,
            source_paths,
            sources_by_id,
            assets_dir,
            max_archive_members=request.max_archive_members,
            max_archive_member_bytes=request.max_archive_member_bytes,
            max_compression_ratio=float(request.max_compression_ratio),
        )
        inventory_bytes = _inventory_payload(request, sources, assets)
        inventory_artifact = _exclusive_bytes(
            staging / SOURCE_INVENTORY_FILE, inventory_bytes)
        inventory_artifact["kind"] = "SOURCE_INVENTORY"
        inventory_sha256 = str(inventory_artifact["sha256"])
        source_table_artifact = _write_table(
            staging / SOURCE_OBJECTS_FILE,
            source_rows,
            SOURCE_OBJECTS_SCHEMA,
            sort_keys=("source_object_id",),
        )
        extracted_table_artifact = _write_table(
            staging / EXTRACTED_ASSETS_FILE,
            asset_rows,
            EXTRACTED_ASSETS_SCHEMA,
            sort_keys=(
                "source_object_id",
                "normalized_member_path",
                "recording_id",
                "asset_role",
            ),
        )
        artifacts = [
            inventory_artifact,
            source_table_artifact,
            extracted_table_artifact,
            *source_artifacts,
            *asset_artifacts,
        ]
        artifacts.sort(key=lambda item: (str(item["path"]), str(item["kind"])))
        unavailable_count = sum(
            row["extraction_status"] == "UNAVAILABLE" for row in asset_rows)
        extracted_count = len(asset_rows) - unavailable_count
        recording_count = len({asset.reference.recording_id for asset in assets})
        terms_anchor = sources_by_id[request.terms_source_object_id]
        manifest = {
            "artifact_kind": VIDIMU_SOURCE_SNAPSHOT_ARTIFACT_KIND,
            "schema_version": VIDIMU_SOURCE_SNAPSHOT_SCHEMA_VERSION,
            "dataset": {
                "dataset_id": request.dataset_id,
                "dataset_version": request.dataset_version,
            },
            "license_and_citation": {
                "license_id": request.license_id,
                "citation_id": request.citation_id,
                "terms_snapshot_id": request.terms_snapshot_id,
                "terms_source_object_id": request.terms_source_object_id,
                "terms_source_object_sha256": terms_anchor.source_object_sha256,
            },
            "inventory": {
                "path": SOURCE_INVENTORY_FILE,
                "schema_version": VIDIMU_SOURCE_INVENTORY_SCHEMA_VERSION,
                "sha256": inventory_sha256,
                "recording_count": recording_count,
                "asset_reference_count": len(assets),
            },
            "reconciliation": {
                "inventory_records": recording_count,
                "source_objects_expected": len(sources),
                "source_objects_downloaded": len(source_rows),
                "source_objects_hash_verified": len(source_rows),
                "source_objects_failed": 0,
                "asset_references_expected": len(assets),
                "asset_references_resolved": extracted_count,
                "asset_references_unavailable": unavailable_count,
                "asset_references_ambiguous": 0,
                "unreferenced_extracted_assets": 0,
            },
            "artifacts": artifacts,
        }
        try:
            manifest_bytes = canonical_json_bytes(manifest)
        except FinalizationBundleError as exc:
            raise SourceSnapshotError(
                "snapshot manifest is not canonical JSON") from exc
        snapshot_id = hashlib.sha256(manifest_bytes).hexdigest()
        _exclusive_bytes(staging / SNAPSHOT_MANIFEST_FILE, manifest_bytes)
        success = {
            "artifact_kind": VIDIMU_SOURCE_SNAPSHOT_ARTIFACT_KIND,
            "schema_version": VIDIMU_SOURCE_SNAPSHOT_SCHEMA_VERSION,
            "snapshot_manifest_sha256": snapshot_id,
        }
        try:
            success_bytes = canonical_json_bytes(success)
        except FinalizationBundleError as exc:
            raise SourceSnapshotError("success marker is not canonical JSON") from exc
        _exclusive_bytes(staging / SUCCESS_MARKER_FILE, success_bytes)
        _fsync_directory(objects_dir)
        for directory in sorted(
            {candidate.parent for candidate in assets_dir.rglob("*")},
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if directory.is_dir():
                _fsync_directory(directory)
        _fsync_directory(assets_dir)
        _fsync_directory(staging)
        final_path = _publish_staging(
            staging,
            root,
            snapshot_id,
            staging_identity=staging_identity,
        )
        published = final_path == staging or not staging.exists()
        if staging.exists() and final_path != staging:
            _remove_owned_staging(staging, staging_identity)
        return MaterializedSourceSnapshot(
            path=final_path.resolve(strict=True),
            snapshot_manifest_sha256=snapshot_id,
            source_inventory_sha256=inventory_sha256,
            source_object_count=len(source_rows),
            extracted_asset_count=extracted_count,
            unavailable_asset_count=unavailable_count,
        )
    finally:
        if staging.exists() and not published:
            _remove_owned_staging(staging, staging_identity)


def verify_vidimu_source_snapshot(
    snapshot_path: str | Path,
) -> MaterializedSourceSnapshot:
    """Strictly verify one published source snapshot and its content address."""

    path = Path(snapshot_path)
    if path.is_symlink() or not path.is_dir():
        raise SourceSnapshotError("source snapshot must be a real directory")
    path = path.resolve(strict=True)
    if _SHA256_RE.fullmatch(path.name) is None:
        raise SourceSnapshotError("source snapshot directory is not content-addressed")
    manifest_bytes = _read_regular_bytes(
        path / SNAPSHOT_MANIFEST_FILE,
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    success_bytes = _read_regular_bytes(
        path / SUCCESS_MARKER_FILE,
        maximum_bytes=_MAX_SUCCESS_BYTES,
    )
    manifest = _canonical_json_document(
        manifest_bytes, field="snapshot manifest")
    success = _canonical_json_document(success_bytes, field="success marker")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != path.name:
        raise SourceSnapshotError("snapshot manifest does not match its content address")
    expected_success = {
        "artifact_kind": VIDIMU_SOURCE_SNAPSHOT_ARTIFACT_KIND,
        "schema_version": VIDIMU_SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_manifest_sha256": manifest_sha256,
    }
    if success != expected_success:
        raise SourceSnapshotError("source snapshot success marker is invalid")
    if set(manifest) != {
        "artifact_kind",
        "schema_version",
        "dataset",
        "license_and_citation",
        "inventory",
        "reconciliation",
        "artifacts",
    } or manifest["artifact_kind"] != VIDIMU_SOURCE_SNAPSHOT_ARTIFACT_KIND \
            or manifest["schema_version"] != VIDIMU_SOURCE_SNAPSHOT_SCHEMA_VERSION:
        raise SourceSnapshotError("source snapshot manifest contract is invalid")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise SourceSnapshotError("source snapshot artifact inventory is invalid")
    if artifacts != sorted(
        artifacts,
        key=lambda item: (
            str(item.get("path")) if isinstance(item, dict) else "",
            str(item.get("kind")) if isinstance(item, dict) else "",
        ),
    ):
        raise SourceSnapshotError("source snapshot artifacts are not canonical")
    expected_files = {SNAPSHOT_MANIFEST_FILE, SUCCESS_MARKER_FILE}
    artifact_by_path: dict[str, dict[str, object]] = {}
    artifact_field_sets = {
        "SOURCE_INVENTORY": {"path", "kind", "bytes", "sha256"},
        "PARQUET_MANIFEST": {
            "path",
            "kind",
            "bytes",
            "sha256",
            "rows",
            "schema_sha256",
            "semantic_sha256",
            "sort_keys",
            "row_group_size",
            "writer_policy_id",
        },
        "SOURCE_OBJECT": {
            "path",
            "kind",
            "bytes",
            "sha256",
            "source_object_id",
        },
        "EXTRACTED_ASSET": {
            "path",
            "kind",
            "bytes",
            "sha256",
            "source_object_id",
            "archive_member_path",
        },
    }
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise SourceSnapshotError("source snapshot artifact entry is invalid")
        kind = artifact.get("kind")
        if not isinstance(kind, str) or kind not in artifact_field_sets \
                or set(artifact) != artifact_field_sets[kind]:
            raise SourceSnapshotError("source snapshot artifact entry is not closed")
        relative = artifact["path"]
        normalized, _collision_key = _normalize_member_path(relative)
        if normalized != relative or relative in expected_files \
                or relative in artifact_by_path:
            raise SourceSnapshotError(
                "source snapshot artifact paths are not canonical and unique")
        expected_files.add(relative)
        _nonnegative_integer(artifact["bytes"], field="artifact bytes")
        _lower_sha256(artifact["sha256"], field="artifact sha256")
        artifact_by_path[relative] = artifact
    actual_files = _tree_file_names(path)
    if actual_files != expected_files:
        raise SourceSnapshotError("source snapshot file inventory mismatch")

    inventory = manifest["inventory"]
    reconciliation = manifest["reconciliation"]
    dataset = manifest["dataset"]
    license_and_citation = manifest["license_and_citation"]
    if not isinstance(inventory, dict) or set(inventory) != {
        "path",
        "schema_version",
        "sha256",
        "recording_count",
        "asset_reference_count",
    } or not isinstance(reconciliation, dict) or not isinstance(dataset, dict) \
            or not isinstance(license_and_citation, dict):
        raise SourceSnapshotError("source snapshot reconciliation is invalid")
    inventory_artifact = artifact_by_path.get(SOURCE_INVENTORY_FILE)
    if inventory_artifact is None \
            or inventory_artifact.get("kind") != "SOURCE_INVENTORY" \
            or inventory.get("path") != SOURCE_INVENTORY_FILE \
            or inventory.get("schema_version") \
            != VIDIMU_SOURCE_INVENTORY_SCHEMA_VERSION \
            or inventory.get("sha256") != inventory_artifact.get("sha256"):
        raise SourceSnapshotError("source inventory is not bound to its manifest")
    inventory_bytes = _read_regular_bytes(
        path / SOURCE_INVENTORY_FILE,
        maximum_bytes=_MAX_INVENTORY_BYTES,
        expected_bytes=_nonnegative_integer(
            inventory_artifact["bytes"], field="inventory bytes"),
        expected_sha256=_lower_sha256(
            inventory_artifact["sha256"], field="inventory sha256"),
    )
    request, sources, assets, source_by_id = _parse_source_inventory(
        inventory_bytes)
    _verify_closed_vidimu_v2_authority(
        request,
        source_by_id,
        {
            source.source_object_id: (
                path / "objects" / source.source_object_sha256
            )
            for source in sources
        },
    )
    _verify_inventory_against_pinned_archives(
        path,
        request,
        sources,
        assets,
    )
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    if inventory_sha256 != inventory.get("sha256"):
        raise SourceSnapshotError("source inventory hash claim changed")
    if dataset != {
        "dataset_id": request.dataset_id,
        "dataset_version": request.dataset_version,
    }:
        raise SourceSnapshotError("dataset identity contradicts source inventory")
    terms_anchor = source_by_id[request.terms_source_object_id]
    if license_and_citation != {
        "license_id": request.license_id,
        "citation_id": request.citation_id,
        "terms_snapshot_id": request.terms_snapshot_id,
        "terms_source_object_id": request.terms_source_object_id,
        "terms_source_object_sha256": terms_anchor.source_object_sha256,
    }:
        raise SourceSnapshotError(
            "license or citation identity contradicts source inventory")

    source_table_artifact = artifact_by_path.get(SOURCE_OBJECTS_FILE)
    asset_table_artifact = artifact_by_path.get(EXTRACTED_ASSETS_FILE)
    if source_table_artifact is None or asset_table_artifact is None:
        raise SourceSnapshotError("source snapshot omits a required Parquet manifest")
    source_verified = _read_verified_parquet(
        path / SOURCE_OBJECTS_FILE,
        expected_bytes=_nonnegative_integer(
            source_table_artifact["bytes"], field="source table bytes"),
        expected_sha256=_lower_sha256(
            source_table_artifact["sha256"], field="source table sha256"),
    )
    asset_verified = _read_verified_parquet(
        path / EXTRACTED_ASSETS_FILE,
        expected_bytes=_nonnegative_integer(
            asset_table_artifact["bytes"], field="asset table bytes"),
        expected_sha256=_lower_sha256(
            asset_table_artifact["sha256"], field="asset table sha256"),
    )
    source_table = _validate_table_claims(
        source_table_artifact,
        source_verified,
        schema=SOURCE_OBJECTS_SCHEMA,
        sort_keys=("source_object_id",),
    )
    asset_table = _validate_table_claims(
        asset_table_artifact,
        asset_verified,
        schema=EXTRACTED_ASSETS_SCHEMA,
        sort_keys=(
            "source_object_id",
            "normalized_member_path",
            "recording_id",
            "asset_role",
        ),
    )

    source_rows = source_table.to_pylist()
    if len(source_rows) != len(sources) or [
        row["source_object_id"] for row in source_rows
    ] != [source.source_object_id for source in sources]:
        raise SourceSnapshotError(
            "source-object rows do not exactly reconcile with inventory")
    source_artifacts: dict[str, dict[str, object]] = {}
    for artifact in artifacts:
        if artifact["kind"] == "SOURCE_OBJECT":
            source_object_id = artifact["source_object_id"]
            if not isinstance(source_object_id, str) \
                    or source_object_id in source_artifacts:
                raise SourceSnapshotError("source-object artifact IDs are invalid")
            source_artifacts[source_object_id] = artifact
    download_timestamps: set[str] = set()
    for row, source in zip(source_rows, sources, strict=True):
        timestamp = _timestamp(row["download_timestamp_utc"])
        download_timestamps.add(timestamp)
        for field in ("etag", "last_modified"):
            if row[field] is not None:
                _stable_text(row[field], field=f"source row {field}")
        if source.expected_etag is not None and row["etag"] != source.expected_etag \
                or source.expected_last_modified is not None \
                and row["last_modified"] != source.expected_last_modified:
            raise SourceSnapshotError("source HTTP metadata contradicts inventory")
        expected_provider_verified = (
            True if source.provider_checksum_algorithm is not None else None
        )
        expected_row = {
            "source_object_id": source.source_object_id,
            "dataset_id": request.dataset_id,
            "dataset_version": request.dataset_version,
            "source_url": source.source_url,
            "source_provider_id": source.source_provider_id,
            "expected_content_length": source.expected_content_length,
            "observed_content_length": source.expected_content_length,
            "source_object_sha256": source.source_object_sha256,
            "provider_checksum_algorithm": source.provider_checksum_algorithm,
            "provider_checksum_value": source.provider_checksum_value,
            "provider_checksum_verified": expected_provider_verified,
            "mime_type": source.mime_type,
            "archive_type": source.archive_type,
            "license_id": request.license_id,
            "citation_id": request.citation_id,
            "terms_snapshot_id": request.terms_snapshot_id,
            "download_status": "VERIFIED",
            "failure_reason": None,
        }
        for field, expected_value in expected_row.items():
            if row[field] != expected_value:
                raise SourceSnapshotError(
                    "source-object row contradicts frozen inventory")
        artifact = source_artifacts.get(source.source_object_id)
        expected_path = f"objects/{source.source_object_sha256}"
        if artifact != {
            "path": expected_path,
            "kind": "SOURCE_OBJECT",
            "bytes": source.expected_content_length,
            "sha256": source.source_object_sha256,
            "source_object_id": source.source_object_id,
        }:
            raise SourceSnapshotError(
                "source-object file does not exactly reconcile with inventory")
        provider_verified = _verify_source_object_file(
            path / "objects" / source.source_object_sha256,
            source,
        )
        if provider_verified != expected_provider_verified:
            raise SourceSnapshotError("provider checksum verification is inconsistent")
    if len(download_timestamps) != 1 or len(source_artifacts) != len(sources):
        raise SourceSnapshotError("source-object manifest cardinality is invalid")

    asset_rows = asset_table.to_pylist()
    if len(asset_rows) != len(assets):
        raise SourceSnapshotError(
            "extracted-asset rows do not reconcile with inventory")
    asset_artifacts: dict[tuple[str, str], dict[str, object]] = {}
    for artifact in artifacts:
        if artifact["kind"] == "EXTRACTED_ASSET":
            key = (artifact["source_object_id"], artifact["archive_member_path"])
            if not all(isinstance(value, str) for value in key) \
                    or key in asset_artifacts:
                raise SourceSnapshotError("extracted-asset artifact keys are invalid")
            asset_artifacts[key] = artifact
    resolved_count = 0
    unavailable_count = 0
    for row, asset in zip(asset_rows, assets, strict=True):
        reference = asset.reference
        common = {
            "source_object_id": reference.source_object_id,
            "archive_member_path": reference.archive_member_path,
            "normalized_member_path": asset.normalized_member_path,
            "recording_id": reference.recording_id,
            "asset_role": reference.asset_role,
            "modality": reference.modality,
        }
        for field, expected_value in common.items():
            if row[field] != expected_value:
                raise SourceSnapshotError(
                    "extracted-asset row contradicts frozen inventory")
        artifact_key = (
            reference.source_object_id, reference.archive_member_path)
        artifact = asset_artifacts.get(artifact_key)
        if reference.availability == "REQUIRED":
            resolved_count += 1
            expected_row_tail = {
                "asset_size_bytes": reference.expected_size_bytes,
                "asset_sha256": reference.expected_sha256,
                "extraction_status": "VERIFIED",
                "failure_reason": None,
            }
            expected_path = f"assets/{asset.normalized_member_path}"
            if artifact != {
                "path": expected_path,
                "kind": "EXTRACTED_ASSET",
                "bytes": reference.expected_size_bytes,
                "sha256": reference.expected_sha256,
                "source_object_id": reference.source_object_id,
                "archive_member_path": reference.archive_member_path,
            }:
                raise SourceSnapshotError(
                    "extracted asset file contradicts frozen inventory")
            _verify_regular_file(
                path.joinpath(*expected_path.split("/")),
                expected_bytes=_nonnegative_integer(
                    reference.expected_size_bytes, field="asset bytes"),
                expected_sha256=_lower_sha256(
                    reference.expected_sha256, field="asset sha256"),
            )
        else:
            unavailable_count += 1
            expected_row_tail = {
                "asset_size_bytes": None,
                "asset_sha256": None,
                "extraction_status": "UNAVAILABLE",
                "failure_reason": reference.unavailable_reason,
            }
            if artifact is not None:
                raise SourceSnapshotError(
                    "unavailable inventory asset has a materialized artifact")
        for field, expected_value in expected_row_tail.items():
            if row[field] != expected_value:
                raise SourceSnapshotError(
                    "extracted-asset outcome contradicts frozen inventory")
    if len(asset_artifacts) != resolved_count:
        raise SourceSnapshotError("extracted-asset artifact cardinality is invalid")

    recording_count = len({
        asset.reference.recording_id for asset in assets
    })
    expected_inventory_claim = {
        "path": SOURCE_INVENTORY_FILE,
        "schema_version": VIDIMU_SOURCE_INVENTORY_SCHEMA_VERSION,
        "sha256": inventory_sha256,
        "recording_count": recording_count,
        "asset_reference_count": len(assets),
    }
    if inventory != expected_inventory_claim:
        raise SourceSnapshotError("source inventory count claims changed")
    expected_reconciliation = {
        "inventory_records": recording_count,
        "source_objects_expected": len(sources),
        "source_objects_downloaded": len(sources),
        "source_objects_hash_verified": len(sources),
        "source_objects_failed": 0,
        "asset_references_expected": len(assets),
        "asset_references_resolved": resolved_count,
        "asset_references_unavailable": unavailable_count,
        "asset_references_ambiguous": 0,
        "unreferenced_extracted_assets": 0,
    }
    if reconciliation != expected_reconciliation:
        raise SourceSnapshotError("source reconciliation counts changed")
    expected_artifact_paths = {
        SOURCE_INVENTORY_FILE,
        SOURCE_OBJECTS_FILE,
        EXTRACTED_ASSETS_FILE,
        *(f"objects/{source.source_object_sha256}" for source in sources),
        *(
            f"assets/{asset.normalized_member_path}"
            for asset in assets
            if asset.reference.availability == "REQUIRED"
        ),
    }
    if set(artifact_by_path) != expected_artifact_paths:
        raise SourceSnapshotError("source snapshot semantic artifact inventory changed")
    return MaterializedSourceSnapshot(
        path=path,
        snapshot_manifest_sha256=manifest_sha256,
        source_inventory_sha256=inventory_sha256,
        source_object_count=len(sources),
        extracted_asset_count=resolved_count,
        unavailable_asset_count=unavailable_count,
    )


__all__ = [
    "ASSET_ROLES",
    "DEFAULT_MAX_ARCHIVE_MEMBERS",
    "DEFAULT_MAX_ARCHIVE_MEMBER_BYTES",
    "DEFAULT_MAX_COMPRESSION_RATIO",
    "DEFAULT_MAX_SOURCE_OBJECT_BYTES",
    "DEFAULT_MAX_TOTAL_EXTRACTED_BYTES",
    "EXTRACTED_ASSETS_FILE",
    "EXTRACTED_ASSETS_SCHEMA",
    "PROVIDER_CHECKSUM_ALGORITHMS",
    "SOURCE_OBJECTS_FILE",
    "SOURCE_OBJECTS_SCHEMA",
    "VIDIMU_SOURCE_INVENTORY_SCHEMA_VERSION",
    "VIDIMU_SOURCE_SNAPSHOT_SCHEMA_VERSION",
    "VIDIMU_V2_ASSET_REFERENCE_CATALOG_PATH",
    "VIDIMU_V2_EXPECTED_ASSET_REFERENCE_COUNT",
    "VIDIMU_V2_EXPECTED_RECORDING_COUNT",
    "InventoryAssetReference",
    "MaterializedSourceSnapshot",
    "SourceObjectAnchor",
    "SourceSnapshotError",
    "VidimuSourceSnapshotRequest",
    "build_vidimu_v2_source_snapshot_request",
    "load_vidimu_v2_asset_reference_catalog",
    "materialize_vidimu_source_snapshot",
    "verify_vidimu_source_snapshot",
    "vidimu_v2_archive_anchor",
    "vidimu_v2_record_metadata_anchor",
]
