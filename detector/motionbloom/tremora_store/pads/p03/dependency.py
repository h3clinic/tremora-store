"""The P0.2.1 authority P0.3 is not allowed to run without.

P0.3 pins the P0.1 and P0.2.1 evidence hashes, the published P0.2.1 report
bytes, the storage-index content hash, the source manifest and the schema
fingerprints of the two P0.2 tables it reads.  It never rebuilds P0.2, and it
refuses a storage index that carries the same row counts under a different
content hash -- which is the failure a row-count check alone would miss.

Absence and disagreement are separated as elsewhere in the project: a missing
dependency, report or store blocks with exit 4, while a hash that disagrees is
evidence about a broken authority chain, so the gate is evaluated,
``P02_1_DEPENDENCY_VERIFIED`` fails, nothing is materialized, and the run
exits 3.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ...schema import schema_fingerprint
from ..p02.contract import GATE_PASS as P02_GATE_PASS
from ..p02.contract import (
    P02_CONTRACT_VERSION,
)
from ..p02.contract import (
    SUCCESS_MARKER as P02_SUCCESS_MARKER,
)
from ..p02.schemas import pads_samples_schema, pads_windows_schema
from ..p02.verification import storage_index_content_sha256
from .contract import P03_CONTRACT_VERSION, SPECTRAL_CONTRACT_VERSION

P02_DEPENDENCY_FILENAME = "pads_p02_dependency.json"
P02_REPORT_FILENAME = "pads_p02_release_audit.json"
P02_STORE_EVIDENCE_FILENAME = "pads_p02_evidence.json"
STORAGE_INDEX_FILENAME = "pads_stream_storage_index.parquet"
WINDOW_INDEX_FILENAME = "pads_windows.parquet"

DEPENDENCY_VERIFIED = "P02_1_DEPENDENCY_VERIFIED"

DEPENDENCY_FILE_ABSENT = "P02_DEPENDENCY_FILE_ABSENT"
P02_REPORT_ABSENT = "P02_REPORT_ABSENT"
STORE_ROOT_ABSENT = "P02_STORE_ROOT_ABSENT"
ABSENCE_STATUSES = frozenset({
    DEPENDENCY_FILE_ABSENT,
    P02_REPORT_ABSENT,
    STORE_ROOT_ABSENT,
})

DEPENDENCY_FILE_MALFORMED = "P02_DEPENDENCY_FILE_MALFORMED"
P02_REPORT_HASH_MISMATCH = "P02_REPORT_HASH_MISMATCH"
P02_EVIDENCE_HASH_MISMATCH = "P02_EVIDENCE_HASH_MISMATCH"
P02_GATE_NOT_PASS = "P02_GATE_NOT_PASS"
P01_EVIDENCE_HASH_MISMATCH = "P01_EVIDENCE_HASH_MISMATCH"
STORAGE_INDEX_HASH_MISMATCH = "STORAGE_INDEX_HASH_MISMATCH"
SOURCE_MANIFEST_MISMATCH = "SOURCE_MANIFEST_MISMATCH"
SCHEMA_FINGERPRINT_MISMATCH = "SCHEMA_FINGERPRINT_MISMATCH"
STORE_SUCCESS_MARKER_ABSENT = "P02_STORE_SUCCESS_MARKER_ABSENT"


class PadsP03DependencyError(ValueError):
    """Raised when the pinned dependency itself cannot be read."""


@dataclass(frozen=True, slots=True)
class PinnedDependency:
    """The exact P0.2.1 authority this milestone is built on."""

    p01_evidence_sha256: str
    p02_evidence_sha256: str
    p02_report_sha256: str
    p02_gate_status: str
    p02_contract_version: str
    storage_index_content_sha256: str
    source_manifest_sha256: str
    sample_store_schema_sha256: str
    window_index_schema_sha256: str
    spectral_contract_version: str
    spectral_implementation_version: str
    expected_streams: int
    expected_windows: int
    expected_samples: int

    def as_record(self) -> dict[str, Any]:
        return dict(sorted(asdict(self).items()))


FROZEN_DEPENDENCY = PinnedDependency(
    p01_evidence_sha256=(
        "e25ce02f7cc023061f5840e314564b153a7829ff487c996b59b25798cf4c801a"
    ),
    p02_evidence_sha256=(
        "7ca16981b3bce63c5b8262fc0efe8855a73b9a8435e2aa485dc6e79ae898a139"
    ),
    p02_report_sha256=(
        "8e5eb21cf8ecafcadc26a5a0bcdb37a4bd5bad0088a33bf42d1939b45b1f41eb"
    ),
    p02_gate_status=P02_GATE_PASS,
    p02_contract_version=P02_CONTRACT_VERSION,
    storage_index_content_sha256=(
        "22aeeb036cfe2cc6e1e0cc63d2142f75c5754b3bcf2b545aa5a77340f76420f1"
    ),
    source_manifest_sha256=(
        "514cd95405a12afcdfb126d47d1f559e2e8a744f03e586c3088d5e4fd7b02c46"
    ),
    sample_store_schema_sha256=(
        "ad2df20a0db8c053f284baff315cee98c143528d6f027bda4e363c0392423d36"
    ),
    window_index_schema_sha256=(
        "2036fe171d323171bd613ba554960e653352a2bc9486e276b797a66c6c8a1422"
    ),
    spectral_contract_version=SPECTRAL_CONTRACT_VERSION,
    spectral_implementation_version=P03_CONTRACT_VERSION,
    expected_streams=10_318,
    expected_windows=50_676,
    expected_samples=13_447_168,
)


@dataclass(frozen=True, slots=True)
class DependencyVerification:
    status: str
    detail: str
    observed_report_sha256: str | None = None
    observed_storage_index_sha256: str | None = None
    pinned: PinnedDependency | None = None

    @property
    def verified(self) -> bool:
        return self.status == DEPENDENCY_VERIFIED

    @property
    def blocks(self) -> bool:
        return self.status in ABSENCE_STATUSES

    def as_record(self) -> dict[str, Any]:
        return {
            "dependency_status": self.status,
            "detail": self.detail,
            "observed_report_sha256": self.observed_report_sha256,
            "observed_storage_index_sha256": (
                self.observed_storage_index_sha256
            ),
        }


def dependency_record(
    pinned: PinnedDependency = FROZEN_DEPENDENCY,
) -> dict[str, Any]:
    return {
        "artifact_kind": "TREMORA_PADS_P03_P02_DEPENDENCY",
        "pinned": pinned.as_record(),
    }


def load_dependency(path: Path) -> PinnedDependency:
    try:
        pinned = json.loads(path.read_bytes().decode("utf-8"))["pinned"]
        return PinnedDependency(
            p01_evidence_sha256=str(pinned["p01_evidence_sha256"]),
            p02_evidence_sha256=str(pinned["p02_evidence_sha256"]),
            p02_report_sha256=str(pinned["p02_report_sha256"]),
            p02_gate_status=str(pinned["p02_gate_status"]),
            p02_contract_version=str(pinned["p02_contract_version"]),
            storage_index_content_sha256=str(
                pinned["storage_index_content_sha256"]
            ),
            source_manifest_sha256=str(pinned["source_manifest_sha256"]),
            sample_store_schema_sha256=str(
                pinned["sample_store_schema_sha256"]
            ),
            window_index_schema_sha256=str(
                pinned["window_index_schema_sha256"]
            ),
            spectral_contract_version=str(
                pinned["spectral_contract_version"]
            ),
            spectral_implementation_version=str(
                pinned["spectral_implementation_version"]
            ),
            expected_streams=int(pinned["expected_streams"]),
            expected_windows=int(pinned["expected_windows"]),
            expected_samples=int(pinned["expected_samples"]),
        )
    except OSError as exc:
        raise PadsP03DependencyError("dependency file unreadable") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise PadsP03DependencyError("dependency file malformed") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_storage_index_hash(store_root: Path) -> str:
    """Recompute the store's content hash from the index it actually holds."""

    table = pq.read_table(store_root / STORAGE_INDEX_FILENAME)
    index = {
        str(stream_id): {"row_group_content_sha256": str(content)}
        for stream_id, content in zip(
            table.column("stream_id").to_pylist(),
            table.column("row_group_content_sha256").to_pylist(),
            strict=True,
        )
    }
    return storage_index_content_sha256(index)


def verify_dependency(
    *,
    dependency_path: Path,
    p02_report_path: Path,
    store_root: Path,
) -> DependencyVerification:
    """Check the pinned P0.2.1 authority against the report and the store."""

    if not dependency_path.is_file():
        return DependencyVerification(
            DEPENDENCY_FILE_ABSENT,
            f"{P02_DEPENDENCY_FILENAME} is not present",
        )
    try:
        pinned = load_dependency(dependency_path)
    except PadsP03DependencyError as exc:
        return DependencyVerification(DEPENDENCY_FILE_MALFORMED, str(exc))
    carry: dict[str, Any] = {"pinned": pinned}

    if not p02_report_path.is_file():
        return DependencyVerification(
            P02_REPORT_ABSENT, f"{P02_REPORT_FILENAME} is not present",
            **carry,
        )
    observed_report = _sha256_file(p02_report_path)
    if observed_report != pinned.p02_report_sha256:
        return DependencyVerification(
            P02_REPORT_HASH_MISMATCH,
            "the published P0.2 report is not the pinned one",
            observed_report_sha256=observed_report, **carry,
        )
    report = json.loads(p02_report_path.read_bytes().decode("utf-8"))
    if report.get("gate_status") != pinned.p02_gate_status:
        return DependencyVerification(
            P02_GATE_NOT_PASS, f"P0.2 gate is {report.get('gate_status')!r}",
            observed_report_sha256=observed_report, **carry,
        )
    if report.get("canonical_evidence_sha256") != pinned.p02_evidence_sha256:
        return DependencyVerification(
            P02_EVIDENCE_HASH_MISMATCH,
            "the P0.2 evidence hash is not the pinned one",
            observed_report_sha256=observed_report, **carry,
        )
    p01 = report.get("p01_dependency", {}).get("pinned", {})
    if p01.get("p01_evidence_sha256") != pinned.p01_evidence_sha256:
        return DependencyVerification(
            P01_EVIDENCE_HASH_MISMATCH,
            "the P0.1 authority beneath P0.2 is not the pinned one",
            observed_report_sha256=observed_report, **carry,
        )
    if report.get("p01_dependency", {}).get("pinned", {}).get(
        "source_manifest_sha256"
    ) != pinned.source_manifest_sha256:
        return DependencyVerification(
            SOURCE_MANIFEST_MISMATCH,
            "the release beneath P0.2 is not the pinned one",
            observed_report_sha256=observed_report, **carry,
        )

    index_path = store_root / STORAGE_INDEX_FILENAME
    if not store_root.is_dir() or not index_path.is_file():
        return DependencyVerification(
            STORE_ROOT_ABSENT,
            "the P0.2 store or its storage index is not present",
            observed_report_sha256=observed_report, **carry,
        )
    if not (store_root / P02_SUCCESS_MARKER).is_file():
        return DependencyVerification(
            STORE_SUCCESS_MARKER_ABSENT,
            f"the store carries no {P02_SUCCESS_MARKER}",
            observed_report_sha256=observed_report, **carry,
        )
    observed_index = observed_storage_index_hash(store_root)
    if observed_index != pinned.storage_index_content_sha256:
        # Row counts alone would not catch this: a different store with the
        # same shape has the same counts and a different content hash.
        return DependencyVerification(
            STORAGE_INDEX_HASH_MISMATCH,
            "the store's content hash is not the pinned one",
            observed_report_sha256=observed_report,
            observed_storage_index_sha256=observed_index, **carry,
        )

    if schema_fingerprint(
        pads_samples_schema()
    ) != pinned.sample_store_schema_sha256 or schema_fingerprint(
        pads_windows_schema()
    ) != pinned.window_index_schema_sha256:
        return DependencyVerification(
            SCHEMA_FINGERPRINT_MISMATCH,
            "a P0.2 schema has changed since the pin was frozen",
            observed_report_sha256=observed_report,
            observed_storage_index_sha256=observed_index, **carry,
        )

    return DependencyVerification(
        DEPENDENCY_VERIFIED,
        "P0.2.1 authority, published report and materialized store reconcile",
        observed_report_sha256=observed_report,
        observed_storage_index_sha256=observed_index, **carry,
    )


__all__ = [
    "ABSENCE_STATUSES",
    "DEPENDENCY_FILE_ABSENT",
    "DEPENDENCY_FILE_MALFORMED",
    "DEPENDENCY_VERIFIED",
    "FROZEN_DEPENDENCY",
    "P01_EVIDENCE_HASH_MISMATCH",
    "P02_DEPENDENCY_FILENAME",
    "P02_EVIDENCE_HASH_MISMATCH",
    "P02_GATE_NOT_PASS",
    "P02_REPORT_ABSENT",
    "P02_REPORT_FILENAME",
    "P02_REPORT_HASH_MISMATCH",
    "SCHEMA_FINGERPRINT_MISMATCH",
    "SOURCE_MANIFEST_MISMATCH",
    "STORAGE_INDEX_HASH_MISMATCH",
    "STORE_ROOT_ABSENT",
    "STORE_SUCCESS_MARKER_ABSENT",
    "WINDOW_INDEX_FILENAME",
    "DependencyVerification",
    "PadsP03DependencyError",
    "PinnedDependency",
    "dependency_record",
    "load_dependency",
    "observed_storage_index_hash",
    "verify_dependency",
]
