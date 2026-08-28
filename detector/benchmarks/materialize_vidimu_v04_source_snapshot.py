"""Materialize the closed VIDIMU v2 source snapshot for Tremora Store v0.4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motionbloom.tremora_store.source_snapshot import (
    build_vidimu_v2_source_snapshot_request,
    load_vidimu_v2_asset_reference_catalog,
    materialize_vidimu_source_snapshot,
)


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download or copy the byte-pinned VIDIMU v2 archives, extract the "
            "frozen 624-member catalog, and publish one content-addressed snapshot."
        )
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=Path("data/snapshots/vidimu"),
    )
    parser.add_argument("--dataset-archive")
    parser.add_argument("--video-archive")
    parser.add_argument("--record-metadata")
    parser.add_argument("--download-timestamp-utc")
    parser.add_argument("--http-timeout-seconds", type=float, default=120.0)
    arguments = parser.parse_args()

    request = build_vidimu_v2_source_snapshot_request(
        dataset_archive_local_path=_optional_path(arguments.dataset_archive),
        video_archive_local_path=_optional_path(arguments.video_archive),
        record_metadata_local_path=_optional_path(arguments.record_metadata),
        asset_references=load_vidimu_v2_asset_reference_catalog(),
    )
    result = materialize_vidimu_source_snapshot(
        request,
        arguments.snapshot_root,
        download_timestamp_utc=arguments.download_timestamp_utc,
        http_timeout_seconds=arguments.http_timeout_seconds,
    )
    print(json.dumps({
        "extracted_asset_count": result.extracted_asset_count,
        "path": str(result.path),
        "snapshot_manifest_sha256": result.snapshot_manifest_sha256,
        "source_inventory_sha256": result.source_inventory_sha256,
        "source_object_count": result.source_object_count,
        "unavailable_asset_count": result.unavailable_asset_count,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
