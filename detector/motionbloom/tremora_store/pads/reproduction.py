"""Two-layer PADS-P0.1 output: deterministic evidence and execution receipts.

The evidence record contains only source-derived, canonical content, so two
genuine executions produce byte-identical evidence.  Everything that differs
between executions -- process identity, output root, command arguments --
lives in a separate receipt.

That split is what makes the reproduction check mean something.  A report
copied to a second path satisfies every identity field, so identity alone
cannot distinguish a second execution from ``cp report.json b.json``.  The
verifier therefore requires the two receipts to disagree about *where* and *by
whom* they were produced while agreeing about *what* was audited.

This is not cryptographic remote attestation and does not defend against a
deliberately forged receipt.  It establishes two executions under the trusted
procedure, exactly as the v0.5D execution receipts did.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPRODUCTION_VERIFIED = "PADS_INDEPENDENT_REPRODUCTION_VERIFIED"
REPRODUCTION_NOT_ATTEMPTED = "REPRODUCTION_NOT_ATTEMPTED"
REPRODUCTION_SAME_RUN_ID = "REPRODUCTION_SAME_RUN_ID"
REPRODUCTION_SAME_PROCESS = "REPRODUCTION_SAME_PROCESS"
REPRODUCTION_SAME_OUTPUT_ROOT = "REPRODUCTION_SAME_OUTPUT_ROOT"
REPRODUCTION_SAME_OUTPUT_INODE = "REPRODUCTION_SAME_OUTPUT_INODE"
REPRODUCTION_SOURCE_MISMATCH = "REPRODUCTION_SOURCE_MISMATCH"
REPRODUCTION_IMPLEMENTATION_MISMATCH = (
    "REPRODUCTION_IMPLEMENTATION_MISMATCH"
)
REPRODUCTION_EVIDENCE_MISMATCH = "REPRODUCTION_EVIDENCE_MISMATCH"
REPRODUCTION_MALFORMED_RECEIPT = "REPRODUCTION_MALFORMED_RECEIPT"

#: Must agree: the two runs audited the same release with the same code.
_AGREEING_FIELDS = (
    "canonical_evidence_sha256",
    "contract_version",
    "implementation_version",
    "schema_version",
    "source_manifest_sha256",
)

#: Must differ: otherwise this is one execution, or one file copied.
_DIFFERING_FIELDS = (
    ("run_id", REPRODUCTION_SAME_RUN_ID),
    ("process_id", REPRODUCTION_SAME_PROCESS),
    ("output_root", REPRODUCTION_SAME_OUTPUT_ROOT),
    ("output_root_identity", REPRODUCTION_SAME_OUTPUT_INODE),
)

_REQUIRED_RECEIPT_FIELDS = (
    *_AGREEING_FIELDS,
    *(field for field, _ in _DIFFERING_FIELDS),
    "command_arguments",
    "dataset_root",
)


class PadsReproductionError(ValueError):
    """Raised when a receipt cannot be built or compared."""


@dataclass(frozen=True, slots=True)
class RunReceipt:
    """One execution's identity, kept apart from the evidence it produced."""

    run_id: str
    process_id: int
    dataset_root: str
    source_manifest_sha256: str
    output_root: str
    output_root_identity: str
    schema_version: str
    contract_version: str
    implementation_version: str
    canonical_evidence_sha256: str
    command_arguments: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "process_id": self.process_id,
            "dataset_root": self.dataset_root,
            "source_manifest_sha256": self.source_manifest_sha256,
            "output_root": self.output_root,
            "output_root_identity": self.output_root_identity,
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "implementation_version": self.implementation_version,
            "canonical_evidence_sha256": self.canonical_evidence_sha256,
            "command_arguments": list(self.command_arguments),
        }


def output_root_identity(path: Path) -> str:
    """Return a device/inode identity for one output root.

    Two runs that share an inode are one output root under two names, which is
    what a copied report looks like from the filesystem's side.
    """

    try:
        stat = path.stat()
    except OSError as exc:
        raise PadsReproductionError(
            f"output root {path} cannot be inspected") from exc
    return f"{stat.st_dev}:{stat.st_ino}"


def build_run_receipt(
    *,
    dataset_root: Path,
    source_manifest_sha256: str,
    output_root: Path,
    schema_version: str,
    contract_version: str,
    implementation_version: str,
    canonical_evidence_sha256: str,
    command_arguments: tuple[str, ...] = (),
    run_id: str | None = None,
    process_id: int | None = None,
) -> RunReceipt:
    """Build one execution receipt for a completed audit."""

    return RunReceipt(
        run_id=run_id or uuid.uuid4().hex,
        process_id=process_id if process_id is not None else os.getpid(),
        dataset_root=str(dataset_root),
        source_manifest_sha256=source_manifest_sha256,
        output_root=str(output_root),
        output_root_identity=output_root_identity(output_root),
        schema_version=schema_version,
        contract_version=contract_version,
        implementation_version=implementation_version,
        canonical_evidence_sha256=canonical_evidence_sha256,
        command_arguments=tuple(command_arguments),
    )


def verify_independent_reproduction(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
) -> str:
    """Decide whether two receipts describe two genuine executions."""

    if first is None or second is None:
        return REPRODUCTION_NOT_ATTEMPTED
    for receipt in (first, second):
        missing = [
            field for field in _REQUIRED_RECEIPT_FIELDS if field not in receipt
        ]
        if missing:
            return REPRODUCTION_MALFORMED_RECEIPT
    if first.get("source_manifest_sha256") != second.get(
        "source_manifest_sha256"
    ):
        return REPRODUCTION_SOURCE_MISMATCH
    for field in ("schema_version", "contract_version",
                  "implementation_version"):
        if first.get(field) != second.get(field):
            return REPRODUCTION_IMPLEMENTATION_MISMATCH
    evidence = first.get("canonical_evidence_sha256")
    if not evidence or evidence != second.get("canonical_evidence_sha256"):
        return REPRODUCTION_EVIDENCE_MISMATCH
    for field, failure in _DIFFERING_FIELDS:
        if first.get(field) == second.get(field):
            return failure
    return REPRODUCTION_VERIFIED


__all__ = [
    "REPRODUCTION_EVIDENCE_MISMATCH",
    "REPRODUCTION_IMPLEMENTATION_MISMATCH",
    "REPRODUCTION_MALFORMED_RECEIPT",
    "REPRODUCTION_NOT_ATTEMPTED",
    "REPRODUCTION_SAME_OUTPUT_INODE",
    "REPRODUCTION_SAME_OUTPUT_ROOT",
    "REPRODUCTION_SAME_PROCESS",
    "REPRODUCTION_SAME_RUN_ID",
    "REPRODUCTION_SOURCE_MISMATCH",
    "REPRODUCTION_VERIFIED",
    "PadsReproductionError",
    "RunReceipt",
    "build_run_receipt",
    "output_root_identity",
    "verify_independent_reproduction",
]
