"""What each representation costs on disk, counted the same way for all four.

Bytes are measured from the filesystem rather than reported by the writer, so
a representation cannot understate itself.  The three components -- payload,
index and metadata -- are separated because "TremoraStore is bigger" and
"TremoraStore carries indexes the baselines do not" are different statements
and the reader is entitled to both.

The duplication factor is the one number here that is about architecture
rather than encoding: stored sample instances over unique source samples.  A
representation that references its windows scores 1.0; one that copies them
scores whatever it copied.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import (
    B0,
    B1,
    B2,
    DERIVED_STORE_RATES_HZ,
    M1,
    SOURCE_SAMPLES,
    SOURCE_STREAMS,
    WINDOW_SAMPLE_INSTANCES,
    WINDOWS,
)

MOVEMENT_DIRECTORY = "movement"


class StorageAccountingError(ValueError):
    """Raised when the accounting cannot be reconciled."""


@dataclass(slots=True)
class StorageAccount:
    """One representation's thirteen published storage numbers."""

    representation: str
    source_payload_bytes: int = 0
    physical_storage_bytes: int = 0
    metadata_bytes: int = 0
    index_bytes: int = 0
    unique_samples: int = 0
    stored_sample_instances: int = 0
    file_count: int = 0
    compression: dict[str, Any] = field(default_factory=dict)

    @property
    def duplicate_sample_instances(self) -> int:
        return self.stored_sample_instances - self.unique_samples

    @property
    def duplication_factor(self) -> float:
        if not self.unique_samples:
            return 0.0
        return self.stored_sample_instances / self.unique_samples

    def as_record(self, *, original_source_bytes: int) -> dict[str, Any]:
        physical = self.physical_storage_bytes
        return {
            "representation": self.representation,
            "source_payload_bytes": self.source_payload_bytes,
            "physical_storage_bytes": physical,
            "metadata_bytes": self.metadata_bytes,
            "index_bytes": self.index_bytes,
            "unique_samples": self.unique_samples,
            "stored_sample_instances": self.stored_sample_instances,
            "duplicate_sample_instances": self.duplicate_sample_instances,
            "duplication_factor": self.duplication_factor,
            "bytes_per_unique_sample": (
                physical / self.unique_samples if self.unique_samples else 0.0
            ),
            "bytes_per_stream": (
                physical / SOURCE_STREAMS if SOURCE_STREAMS else 0.0
            ),
            "bytes_per_window": physical / WINDOWS if WINDOWS else 0.0,
            "compression_ratio_vs_original_source": (
                original_source_bytes / physical if physical else 0.0
            ),
            "file_count": self.file_count,
            "compression": dict(sorted(self.compression.items())),
        }


def _tree_bytes(root: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            total += path.stat().st_size
            count += 1
    return total, count


def _files_bytes(paths: list[Path]) -> tuple[int, int]:
    present = [path for path in paths if path.is_file()]
    return sum(path.stat().st_size for path in present), len(present)


def _stored_samples(store_root: Path) -> int:
    """How many samples the corpus actually holds, counted from the index.

    Read rather than assumed: taking the frozen P0.2.1 constant here would
    make the accounting right only for the corpus it was written against, and
    would report that corpus's number even when handed a different one.
    """

    import pyarrow.parquet as pq

    table = pq.read_table(
        store_root / "pads_stream_storage_index.parquet",
        columns=["sample_count"],
    )
    return int(sum(table.column("sample_count").to_pylist()))


def account_b0(release_root: Path, store_root: Path) -> StorageAccount:
    """The release as published: text plus its observation declarations."""

    movement = release_root / MOVEMENT_DIRECTORY
    # The device files sit under movement/timeseries/; a glob that only
    # looked beside the observations would have counted 6 MB of JSON and
    # called it the corpus.
    text_bytes, text_files = _files_bytes(sorted(movement.rglob("*.txt")))
    if not text_files:
        raise StorageAccountingError(
            f"no device text files under {movement}"
        )
    json_bytes, json_files = _files_bytes(
        sorted(movement.glob("observation_*.json"))
    )
    return StorageAccount(
        representation=B0,
        source_payload_bytes=text_bytes,
        physical_storage_bytes=text_bytes + json_bytes,
        metadata_bytes=json_bytes,
        # The observation JSONs are what resolve a stream to its file, so
        # they are this representation's index as well as its metadata.
        index_bytes=json_bytes,
        unique_samples=_stored_samples(store_root),
        stored_sample_instances=_stored_samples(store_root),
        file_count=text_files + json_files,
        compression={"codec": "none", "level": None},
    )


def account_b1(root: Path, build: Mapping[str, Any]) -> StorageAccount:
    """The duplicated-window store, measured from disk."""

    total, count = _tree_bytes(root)
    manifest = root / "b1_manifest.json"
    manifest_bytes = manifest.stat().st_size if manifest.is_file() else 0
    return StorageAccount(
        representation=B1,
        source_payload_bytes=total - manifest_bytes,
        physical_storage_bytes=total,
        metadata_bytes=manifest_bytes,
        index_bytes=manifest_bytes,
        unique_samples=int(build["unique_samples"]),
        stored_sample_instances=int(build["stored_sample_instances"]),
        file_count=count,
        compression=dict(build["compression"]),
    )


def account_b2(root: Path, build: Mapping[str, Any]) -> StorageAccount:
    """The HDF5 store, measured from disk."""

    total, count = _tree_bytes(root)
    index_bytes = int(build.get("index_bytes", 0))
    return StorageAccount(
        representation=B2,
        source_payload_bytes=total - index_bytes,
        physical_storage_bytes=total,
        metadata_bytes=int(build.get("metadata_bytes", 0)),
        index_bytes=index_bytes,
        unique_samples=int(build["unique_samples"]),
        stored_sample_instances=int(build["stored_sample_instances"]),
        file_count=count,
        compression=dict(build["compression"]),
    )


def account_m1(store_root: Path) -> StorageAccount:
    """The P0.2.1 store: samples, and the indexes that make them auditable.

    The index and evidence files are counted, not quietly dropped.  They are
    a real cost of this representation and the baselines do not carry them.
    """

    samples_root = store_root / "samples"
    payload, payload_files = _tree_bytes(samples_root)
    index_paths = sorted(store_root.glob("*.parquet"))
    index_bytes, index_files = _files_bytes(index_paths)
    metadata_paths = sorted(store_root.glob("*.json"))
    metadata_bytes, metadata_files = _files_bytes(metadata_paths)
    samples = _stored_samples(store_root)
    return StorageAccount(
        representation=M1,
        source_payload_bytes=payload,
        physical_storage_bytes=payload + index_bytes + metadata_bytes,
        metadata_bytes=metadata_bytes,
        index_bytes=index_bytes,
        unique_samples=samples,
        # The window index carries ordinal ranges; it stores no samples.
        stored_sample_instances=samples,
        file_count=payload_files + index_files + metadata_files,
        compression={"codec": "zstd", "level": 9},
    )


def derived_store_account(p04_root: Path) -> dict[str, Any]:
    """P0.4's derived stores, reported beside the comparison and not inside it.

    Folding these into the primary numbers would change the question from how
    one authoritative representation is stored to how a family of derived
    signals is stored, so their samples never enter the duplication maths.
    """

    total, count = _tree_bytes(p04_root) if p04_root.is_dir() else (0, 0)
    return {
        "included_in_primary_comparison": False,
        "rates_hz": list(DERIVED_STORE_RATES_HZ),
        "derived_store_bytes": total,
        "derived_store_file_count": count,
        "note": (
            "reported beside the comparison; these samples are not part of "
            "any representation's duplication accounting"
        ),
    }


def reconcile(
    accounts: Mapping[str, StorageAccount],
    *,
    expected_unique_samples: int = SOURCE_SAMPLES,
    expected_window_instances: int = WINDOW_SAMPLE_INSTANCES,
) -> dict[str, Any]:
    """Check the counts before publishing them.

    The expectations are parameters rather than the frozen constants read
    directly, so this reconciles a corpus of any size while the audit still
    pins the real one by passing the P0.2.1 numbers in.  Wiring the constants
    in here would have made the check runnable only against the corpus it was
    written for, which is not a property of the code.
    """

    problems: list[str] = []
    for name, account in sorted(accounts.items()):
        if account.unique_samples != expected_unique_samples:
            problems.append(
                f"{name} holds {account.unique_samples} unique samples, "
                f"not {expected_unique_samples}"
            )
        if account.physical_storage_bytes <= 0:
            problems.append(f"{name} measured no bytes on disk")
    expected_b1 = expected_unique_samples + expected_window_instances
    if B1 in accounts and accounts[B1].stored_sample_instances != expected_b1:
        problems.append(
            f"{B1} stored {accounts[B1].stored_sample_instances} instances, "
            f"not the {expected_b1} its overlapping windows require"
        )
    for name in (B0, B2, M1):
        account = accounts.get(name)
        if account is not None and account.duplication_factor != 1.0:
            problems.append(
                f"{name} duplicates samples: factor "
                f"{account.duplication_factor}"
            )
    return {
        "reconciled": not problems,
        "problems": problems,
        "expected_unique_samples": expected_unique_samples,
        "expected_b1_stored_instances": expected_b1,
    }


def storage_tables(
    *,
    accounts: Mapping[str, StorageAccount],
    original_source_bytes: int,
    p04_root: Path | None = None,
    expected_unique_samples: int = SOURCE_SAMPLES,
    expected_window_instances: int = WINDOW_SAMPLE_INSTANCES,
) -> dict[str, Any]:
    """Everything the storage half of the report publishes."""

    return {
        "original_source_bytes": original_source_bytes,
        "accounts": [
            accounts[name].as_record(
                original_source_bytes=original_source_bytes
            )
            for name in (B0, B1, B2, M1) if name in accounts
        ],
        "reconciliation": reconcile(
            accounts,
            expected_unique_samples=expected_unique_samples,
            expected_window_instances=expected_window_instances,
        ),
        "derived_stores": (
            derived_store_account(p04_root) if p04_root is not None
            else {"included_in_primary_comparison": False}
        ),
    }


def load_build_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes().decode("utf-8"))


__all__ = [
    "MOVEMENT_DIRECTORY",
    "StorageAccount",
    "StorageAccountingError",
    "_stored_samples",
    "account_b0",
    "account_b1",
    "account_b2",
    "account_m1",
    "derived_store_account",
    "load_build_report",
    "reconcile",
    "storage_tables",
]
