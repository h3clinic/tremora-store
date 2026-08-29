"""The P0.3 authority P0.4 is not allowed to run without.

P0.4 pins the P0.3 evidence and report, the spectral-table content hash, the
frozen frequency grid and the whole P0.2.1 chain beneath them, plus its own
anti-alias coefficient hash.  It never recomputes the P0.3 reference: the
native irregular source-time spectra are read as published.

Verification recomputes the spectral-table content hash from the P0.3 outputs
actually on disk, so a substituted spectra table with the same row count and a
different content is refused.

Absence blocks with exit 4; disagreement closes the gate at exit 3 with nothing
materialized.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..p02.contract import SUCCESS_MARKER as P02_SUCCESS_MARKER
from ..p03.contract import GATE_PASS as P03_GATE_PASS
from ..p03.contract import P03_CONTRACT_VERSION
from ..p03.contract import SUCCESS_MARKER as P03_SUCCESS_MARKER
from ..p03.grid import grid_hash
from ..p03.schemas import P03_TABLE_FILES
from .contract import P04_CONTRACT_VERSION, RESAMPLING_CONTRACT_VERSION
from .filters import coefficients_sha256

P03_DEPENDENCY_FILENAME = "pads_p03_dependency.json"
P03_REPORT_FILENAME = "pads_p03_release_audit.json"
SPECTRA_FILENAME = P03_TABLE_FILES["pads_p03_spectra"]

DEPENDENCY_VERIFIED = "P03_DEPENDENCY_VERIFIED"

DEPENDENCY_FILE_ABSENT = "P03_DEPENDENCY_FILE_ABSENT"
P03_REPORT_ABSENT = "P03_REPORT_ABSENT"
P03_ROOT_ABSENT = "P03_ROOT_ABSENT"
STORE_ROOT_ABSENT = "P02_STORE_ROOT_ABSENT"
ABSENCE_STATUSES = frozenset({
    DEPENDENCY_FILE_ABSENT,
    P03_REPORT_ABSENT,
    P03_ROOT_ABSENT,
    STORE_ROOT_ABSENT,
})

DEPENDENCY_FILE_MALFORMED = "P03_DEPENDENCY_FILE_MALFORMED"
P03_REPORT_HASH_MISMATCH = "P03_REPORT_HASH_MISMATCH"
P03_EVIDENCE_HASH_MISMATCH = "P03_EVIDENCE_HASH_MISMATCH"
P03_GATE_NOT_PASS = "P03_GATE_NOT_PASS"
SPECTRAL_TABLE_HASH_MISMATCH = "SPECTRAL_TABLE_HASH_MISMATCH"
FREQUENCY_GRID_MISMATCH = "FREQUENCY_GRID_MISMATCH"
ANTI_ALIAS_COEFFICIENTS_MISMATCH = "ANTI_ALIAS_COEFFICIENTS_MISMATCH"
P02_CHAIN_MISMATCH = "P02_CHAIN_MISMATCH"
P03_SUCCESS_MARKER_ABSENT = "P03_SUCCESS_MARKER_ABSENT"
P02_STORE_SUCCESS_MARKER_ABSENT = "P02_STORE_SUCCESS_MARKER_ABSENT"


class PadsP04DependencyError(ValueError):
    """Raised when the pinned dependency itself cannot be read."""


@dataclass(frozen=True, slots=True)
class PinnedDependency:
    """The exact P0.3 authority, and the P0.2.1 chain beneath it."""

    p03_evidence_sha256: str
    p03_report_sha256: str
    p03_gate_status: str
    p03_contract_version: str
    p03_spectral_table_sha256: str
    frequency_grid_sha256: str
    p02_evidence_sha256: str
    storage_index_content_sha256: str
    source_manifest_sha256: str
    anti_alias_coefficients_sha256: str
    resampling_contract_version: str
    rate_ablation_contract_version: str
    expected_workload_windows: int
    expected_audit_windows: int
    expected_spectral_rows: int

    def as_record(self) -> dict[str, Any]:
        return dict(sorted(asdict(self).items()))


FROZEN_DEPENDENCY = PinnedDependency(
    p03_evidence_sha256=(
        "a0be87d48c1146862aa83a2a1238d3df50fc344781878b6ee0a03f738548df17"
    ),
    p03_report_sha256=(
        "a2b6dfa3f598dfe7e2821285c3262dd1f817b168cb81e62683ad796445faf615"
    ),
    p03_gate_status=P03_GATE_PASS,
    p03_contract_version=P03_CONTRACT_VERSION,
    p03_spectral_table_sha256=(
        "27bb6444bdfcab77911134b1c4671f563c51084f69e6d587e339c0a00d76c97e"
    ),
    frequency_grid_sha256=(
        "b7b18ff14e6ef93a159484abd7850c732851e63c69290213706381bb32f77b25"
    ),
    p02_evidence_sha256=(
        "7ca16981b3bce63c5b8262fc0efe8855a73b9a8435e2aa485dc6e79ae898a139"
    ),
    storage_index_content_sha256=(
        "22aeeb036cfe2cc6e1e0cc63d2142f75c5754b3bcf2b545aa5a77340f76420f1"
    ),
    source_manifest_sha256=(
        "514cd95405a12afcdfb126d47d1f559e2e8a744f03e586c3088d5e4fd7b02c46"
    ),
    anti_alias_coefficients_sha256=(
        "976957f77d3ba0edbe72507bb32617751bbf1f3c1f38e299c5ce5e4120163d81"
    ),
    resampling_contract_version=RESAMPLING_CONTRACT_VERSION,
    rate_ablation_contract_version=P04_CONTRACT_VERSION,
    expected_workload_windows=9_960,
    expected_audit_windows=6_077,
    expected_spectral_rows=19_920,
)


@dataclass(frozen=True, slots=True)
class DependencyVerification:
    status: str
    detail: str
    observed_report_sha256: str | None = None
    observed_spectral_table_sha256: str | None = None
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
            "observed_spectral_table_sha256": (
                self.observed_spectral_table_sha256
            ),
        }


def dependency_record(
    pinned: PinnedDependency = FROZEN_DEPENDENCY,
) -> dict[str, Any]:
    return {
        "artifact_kind": "TREMORA_PADS_P04_P03_DEPENDENCY",
        "pinned": pinned.as_record(),
    }


def load_dependency(path: Path) -> PinnedDependency:
    try:
        pinned = json.loads(path.read_bytes().decode("utf-8"))["pinned"]
        return PinnedDependency(**{
            field: (
                int(pinned[field]) if field.startswith("expected_")
                else str(pinned[field])
            )
            for field in PinnedDependency.__dataclass_fields__
        })
    except OSError as exc:
        raise PadsP04DependencyError("dependency file unreadable") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise PadsP04DependencyError("dependency file malformed") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_spectral_table_hash(p03_root: Path) -> str:
    """Recompute the P0.3 spectral-table content hash from the outputs."""

    table = pq.read_table(p03_root / SPECTRA_FILENAME)
    digest = hashlib.sha256()
    for record_id, content in zip(
        table.column("spectral_record_id").to_pylist(),
        table.column("spectral_content_sha256").to_pylist(),
        strict=True,
    ):
        digest.update(str(record_id).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(content).encode("ascii"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def verify_dependency(
    *,
    dependency_path: Path,
    p03_report_path: Path,
    p03_root: Path,
    store_root: Path,
) -> DependencyVerification:
    """Check the pinned P0.3 authority against the report and the outputs."""

    if not dependency_path.is_file():
        return DependencyVerification(
            DEPENDENCY_FILE_ABSENT,
            f"{P03_DEPENDENCY_FILENAME} is not present",
        )
    try:
        pinned = load_dependency(dependency_path)
    except PadsP04DependencyError as exc:
        return DependencyVerification(DEPENDENCY_FILE_MALFORMED, str(exc))
    carry: dict[str, Any] = {"pinned": pinned}

    # Presence of every input is settled before any content is judged: a
    # disagreement about a chain whose parts are missing is not a meaningful
    # verdict, and absence and disagreement have different exit codes.
    if not p03_report_path.is_file():
        return DependencyVerification(
            P03_REPORT_ABSENT, f"{P03_REPORT_FILENAME} is not present",
            **carry,
        )
    if not p03_root.is_dir() or not (p03_root / SPECTRA_FILENAME).is_file():
        return DependencyVerification(
            P03_ROOT_ABSENT,
            "the P0.3 outputs or their spectra table are not present",
            **carry,
        )
    if not store_root.is_dir() or not (
        store_root / "pads_stream_storage_index.parquet"
    ).is_file():
        return DependencyVerification(
            STORE_ROOT_ABSENT, "the P0.2 store is not present", **carry,
        )

    observed_report = _sha256_file(p03_report_path)
    if observed_report != pinned.p03_report_sha256:
        return DependencyVerification(
            P03_REPORT_HASH_MISMATCH,
            "the published P0.3 report is not the pinned one",
            observed_report_sha256=observed_report, **carry,
        )
    report = json.loads(p03_report_path.read_bytes().decode("utf-8"))
    if report.get("gate_status") != pinned.p03_gate_status:
        return DependencyVerification(
            P03_GATE_NOT_PASS, f"P0.3 gate is {report.get('gate_status')!r}",
            observed_report_sha256=observed_report, **carry,
        )
    if report.get("canonical_evidence_sha256") != pinned.p03_evidence_sha256:
        return DependencyVerification(
            P03_EVIDENCE_HASH_MISMATCH,
            "the P0.3 evidence hash is not the pinned one",
            observed_report_sha256=observed_report, **carry,
        )
    p02 = report.get("p02_dependency", {}).get("pinned", {})
    if (
        p02.get("p02_evidence_sha256") != pinned.p02_evidence_sha256
        or p02.get("storage_index_content_sha256")
        != pinned.storage_index_content_sha256
        or p02.get("source_manifest_sha256") != pinned.source_manifest_sha256
    ):
        return DependencyVerification(
            P02_CHAIN_MISMATCH,
            "the P0.2.1 chain beneath P0.3 is not the pinned one",
            observed_report_sha256=observed_report, **carry,
        )
    if report.get("frequency_grid", {}).get(
        "frequency_grid_hash"
    ) != pinned.frequency_grid_sha256 or grid_hash() != (
        pinned.frequency_grid_sha256
    ):
        return DependencyVerification(
            FREQUENCY_GRID_MISMATCH,
            "the analysis grid has moved since the pin was frozen",
            observed_report_sha256=observed_report, **carry,
        )
    if coefficients_sha256() != pinned.anti_alias_coefficients_sha256:
        return DependencyVerification(
            ANTI_ALIAS_COEFFICIENTS_MISMATCH,
            "the frozen anti-alias coefficients have changed",
            observed_report_sha256=observed_report, **carry,
        )

    if not (p03_root / P03_SUCCESS_MARKER).is_file():
        return DependencyVerification(
            P03_SUCCESS_MARKER_ABSENT,
            f"the P0.3 outputs carry no {P03_SUCCESS_MARKER}",
            observed_report_sha256=observed_report, **carry,
        )
    observed_spectra = observed_spectral_table_hash(p03_root)
    if observed_spectra != pinned.p03_spectral_table_sha256:
        return DependencyVerification(
            SPECTRAL_TABLE_HASH_MISMATCH,
            "the P0.3 spectra on disk are not the published ones",
            observed_report_sha256=observed_report,
            observed_spectral_table_sha256=observed_spectra, **carry,
        )

    if not (store_root / P02_SUCCESS_MARKER).is_file():
        return DependencyVerification(
            P02_STORE_SUCCESS_MARKER_ABSENT,
            f"the store carries no {P02_SUCCESS_MARKER}",
            observed_report_sha256=observed_report,
            observed_spectral_table_sha256=observed_spectra, **carry,
        )

    return DependencyVerification(
        DEPENDENCY_VERIFIED,
        "P0.3 authority, published report, spectra and store all reconcile",
        observed_report_sha256=observed_report,
        observed_spectral_table_sha256=observed_spectra, **carry,
    )


__all__ = [
    "ABSENCE_STATUSES",
    "ANTI_ALIAS_COEFFICIENTS_MISMATCH",
    "DEPENDENCY_FILE_ABSENT",
    "DEPENDENCY_FILE_MALFORMED",
    "DEPENDENCY_VERIFIED",
    "FREQUENCY_GRID_MISMATCH",
    "FROZEN_DEPENDENCY",
    "P02_CHAIN_MISMATCH",
    "P02_STORE_SUCCESS_MARKER_ABSENT",
    "P03_DEPENDENCY_FILENAME",
    "P03_EVIDENCE_HASH_MISMATCH",
    "P03_GATE_NOT_PASS",
    "P03_REPORT_ABSENT",
    "P03_REPORT_FILENAME",
    "P03_REPORT_HASH_MISMATCH",
    "P03_ROOT_ABSENT",
    "P03_SUCCESS_MARKER_ABSENT",
    "SPECTRAL_TABLE_HASH_MISMATCH",
    "STORE_ROOT_ABSENT",
    "DependencyVerification",
    "PadsP04DependencyError",
    "PinnedDependency",
    "dependency_record",
    "load_dependency",
    "observed_spectral_table_hash",
    "verify_dependency",
]
