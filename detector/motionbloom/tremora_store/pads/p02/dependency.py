"""The P0.1 authority P0.2 is not allowed to run without.

P0.2 never regenerates P0.1 inside itself.  It pins the exact P0.1 verdict, the
exact bytes of the published report, and the exact release those describe, then
checks that all three still hold before materializing anything.

Absence and disagreement are treated differently, following the rule the
project already applies elsewhere.  A missing dependency, report or release
root means there is nothing to depend on: the run reports
``BLOCKED_P01_DEPENDENCY_UNAVAILABLE`` and exits 4.  A dependency that is
present but disagrees -- a changed evidence hash, a changed report, a P0.1
verdict that is not PASS, a different source manifest -- is evidence about the
authority chain, so the gate is evaluated,
``P01_AUTHORITY_DEPENDENCY_VERIFIED`` fails, nothing is materialized, and the
run exits 3.  Reporting a disagreement as unavailability would hide a broken
authority chain behind an availability notice.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..authority import GATE_PASS as P01_GATE_PASS
from ..authority import PADS_CONTRACT_VERSION as P01_CONTRACT_VERSION
from ..release_structure import (
    PADS_EXPECTED_ASSESSMENTS,
    PADS_EXPECTED_PARTICIPANTS,
    PADS_EXPECTED_STREAMS,
    PADS_RELEASE_VERSION,
)

P01_DEPENDENCY_FILENAME = "pads_p01_dependency.json"
P01_REPORT_FILENAME = "pads_p01_release_audit.json"
CHECKSUM_FILENAME = "SHA256SUMS.txt"

DEPENDENCY_VERIFIED = "P01_AUTHORITY_DEPENDENCY_VERIFIED"

# Absence: nothing to depend on.
DEPENDENCY_FILE_ABSENT = "P01_DEPENDENCY_FILE_ABSENT"
P01_REPORT_ABSENT = "P01_REPORT_ABSENT"
RELEASE_ROOT_ABSENT = "RELEASE_ROOT_ABSENT"
ABSENCE_STATUSES = frozenset({
    DEPENDENCY_FILE_ABSENT,
    P01_REPORT_ABSENT,
    RELEASE_ROOT_ABSENT,
})

# Disagreement: evidence about a broken authority chain.
DEPENDENCY_FILE_MALFORMED = "P01_DEPENDENCY_FILE_MALFORMED"
P01_REPORT_HASH_MISMATCH = "P01_REPORT_HASH_MISMATCH"
P01_EVIDENCE_HASH_MISMATCH = "P01_EVIDENCE_HASH_MISMATCH"
P01_GATE_NOT_PASS = "P01_GATE_NOT_PASS"
SOURCE_MANIFEST_MISMATCH = "SOURCE_MANIFEST_MISMATCH"
RELEASE_COUNTS_MISMATCH = "RELEASE_COUNTS_MISMATCH"
P01_CONTRACT_MISMATCH = "P01_CONTRACT_MISMATCH"


class PadsDependencyError(ValueError):
    """Raised when the pinned dependency itself cannot be read."""


@dataclass(frozen=True, slots=True)
class PinnedDependency:
    """The exact P0.1 authority this milestone is built on."""

    p01_gate_status: str
    p01_evidence_sha256: str
    p01_report_sha256: str
    source_manifest_sha256: str
    dataset_version: str
    p01_contract_version: str
    p01_code_commit: str
    expected_participants: int
    expected_assessments: int
    expected_streams: int
    expected_samples: int

    def as_record(self) -> dict[str, Any]:
        return dict(sorted(asdict(self).items()))


#: Frozen in code, not only in the JSON file: the file is generated from this
#: and the contract tests compare the two, so editing the file alone cannot
#: move the dependency.
FROZEN_DEPENDENCY = PinnedDependency(
    p01_gate_status=P01_GATE_PASS,
    p01_evidence_sha256=(
        "e25ce02f7cc023061f5840e314564b153a7829ff487c996b59b25798cf4c801a"
    ),
    p01_report_sha256=(
        "6d2e0fab4bbcc3762e70c95b30b48293c17d785d3db9877288a4efa75f03a749"
    ),
    source_manifest_sha256=(
        "514cd95405a12afcdfb126d47d1f559e2e8a744f03e586c3088d5e4fd7b02c46"
    ),
    dataset_version=PADS_RELEASE_VERSION,
    p01_contract_version=P01_CONTRACT_VERSION,
    p01_code_commit="05fe4bfecf06a93fea2dcaf40311c4c86059993e",
    expected_participants=PADS_EXPECTED_PARTICIPANTS,
    expected_assessments=PADS_EXPECTED_ASSESSMENTS,
    expected_streams=PADS_EXPECTED_STREAMS,
    expected_samples=13_447_168,
)


@dataclass(frozen=True, slots=True)
class DependencyVerification:
    """The outcome of checking the pinned authority against reality."""

    status: str
    detail: str
    observed_report_sha256: str | None = None
    observed_evidence_sha256: str | None = None
    observed_manifest_sha256: str | None = None
    #: The pin actually read from the supplied file.  The audit uses this
    #: rather than the code constant, so the file is a real input and a
    #: substituted pin is visible in the published evidence.
    pinned: PinnedDependency | None = None

    @property
    def verified(self) -> bool:
        return self.status == DEPENDENCY_VERIFIED

    @property
    def blocks(self) -> bool:
        """Absence blocks; disagreement closes the gate."""

        return self.status in ABSENCE_STATUSES

    def as_record(self) -> dict[str, Any]:
        return {
            "dependency_status": self.status,
            "detail": self.detail,
            "observed_report_sha256": self.observed_report_sha256,
            "observed_evidence_sha256": self.observed_evidence_sha256,
            "observed_manifest_sha256": self.observed_manifest_sha256,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_record(
    pinned: PinnedDependency = FROZEN_DEPENDENCY,
) -> dict[str, Any]:
    """The canonical content of ``pads_p01_dependency.json``."""

    return {
        "artifact_kind": "TREMORA_PADS_P02_P01_DEPENDENCY",
        "pinned": pinned.as_record(),
    }


def load_dependency(path: Path) -> PinnedDependency:
    """Read a pinned dependency file."""

    try:
        document = json.loads(path.read_bytes().decode("utf-8"))
        pinned = document["pinned"]
        return PinnedDependency(
            p01_gate_status=str(pinned["p01_gate_status"]),
            p01_evidence_sha256=str(pinned["p01_evidence_sha256"]),
            p01_report_sha256=str(pinned["p01_report_sha256"]),
            source_manifest_sha256=str(pinned["source_manifest_sha256"]),
            dataset_version=str(pinned["dataset_version"]),
            p01_contract_version=str(pinned["p01_contract_version"]),
            p01_code_commit=str(pinned["p01_code_commit"]),
            expected_participants=int(pinned["expected_participants"]),
            expected_assessments=int(pinned["expected_assessments"]),
            expected_streams=int(pinned["expected_streams"]),
            expected_samples=int(pinned["expected_samples"]),
        )
    except OSError as exc:
        raise PadsDependencyError("dependency file could not be read") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise PadsDependencyError("dependency file is malformed") from exc


def verify_dependency(
    *,
    dependency_path: Path,
    report_path: Path,
    release_root: Path,
) -> DependencyVerification:
    """Check the pinned P0.1 authority against the report and the release."""

    if not dependency_path.is_file():
        return DependencyVerification(
            DEPENDENCY_FILE_ABSENT,
            f"{P01_DEPENDENCY_FILENAME} is not present",
        )
    try:
        pinned = load_dependency(dependency_path)
    except PadsDependencyError as exc:
        return DependencyVerification(DEPENDENCY_FILE_MALFORMED, str(exc))
    carry = {"pinned": pinned}

    if not report_path.is_file():
        return DependencyVerification(
            P01_REPORT_ABSENT, f"{P01_REPORT_FILENAME} is not present",
            **carry,
        )
    observed_report = _sha256_file(report_path)
    if observed_report != pinned.p01_report_sha256:
        return DependencyVerification(
            P01_REPORT_HASH_MISMATCH,
            "the published P0.1 report is not the pinned one",
            observed_report_sha256=observed_report,
            **carry,
        )

    try:
        report = json.loads(report_path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:  # pragma: no cover - hash already matched
        return DependencyVerification(
            DEPENDENCY_FILE_MALFORMED, f"P0.1 report unreadable: {exc}",
            **carry,
        )

    if report.get("gate_status") != pinned.p01_gate_status:
        return DependencyVerification(
            P01_GATE_NOT_PASS,
            f"P0.1 gate is {report.get('gate_status')!r}",
            observed_report_sha256=observed_report,
            **carry,
        )
    observed_evidence = report.get("canonical_evidence_sha256")
    if observed_evidence != pinned.p01_evidence_sha256:
        return DependencyVerification(
            P01_EVIDENCE_HASH_MISMATCH,
            "the P0.1 evidence hash is not the pinned one",
            observed_report_sha256=observed_report,
            observed_evidence_sha256=observed_evidence,
            **carry,
        )
    if report.get("contract_version") != pinned.p01_contract_version:
        return DependencyVerification(
            P01_CONTRACT_MISMATCH,
            "the P0.1 contract version is not the pinned one",
            observed_report_sha256=observed_report,
            **carry,
        )

    structure = report.get("release_structure", {})
    observed_counts = (
        structure.get("observed_participants"),
        structure.get("observed_assessments"),
        structure.get("observed_streams"),
        report.get("samples", {}).get("total"),
    )
    expected_counts = (
        pinned.expected_participants,
        pinned.expected_assessments,
        pinned.expected_streams,
        pinned.expected_samples,
    )
    if observed_counts != expected_counts:
        return DependencyVerification(
            RELEASE_COUNTS_MISMATCH,
            f"P0.1 reports {observed_counts} against pinned {expected_counts}",
            observed_report_sha256=observed_report,
            observed_evidence_sha256=observed_evidence,
            **carry,
        )

    checksum_path = release_root / CHECKSUM_FILENAME
    if not release_root.is_dir() or not checksum_path.is_file():
        return DependencyVerification(
            RELEASE_ROOT_ABSENT,
            "the pinned release root or its checksum list is not present",
            observed_report_sha256=observed_report,
            observed_evidence_sha256=observed_evidence,
            **carry,
        )
    observed_manifest = _sha256_file(checksum_path)
    if observed_manifest != pinned.source_manifest_sha256:
        return DependencyVerification(
            SOURCE_MANIFEST_MISMATCH,
            "the release root is not the audited release",
            observed_report_sha256=observed_report,
            observed_evidence_sha256=observed_evidence,
            observed_manifest_sha256=observed_manifest,
            **carry,
        )

    return DependencyVerification(
        DEPENDENCY_VERIFIED,
        "P0.1 authority, published report and release all reconcile",
        observed_report_sha256=observed_report,
        observed_evidence_sha256=observed_evidence,
        observed_manifest_sha256=observed_manifest,
        **carry,
    )


def read_checksums(path: Path) -> dict[str, str]:
    """Read the release's own checksum list as ``{relative: digest}``."""

    manifest: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition(" ")
        relative = relative.strip()
        if len(digest) != 64 or not relative:
            raise PadsDependencyError(f"malformed checksum line: {line!r}")
        manifest[relative] = digest
    return manifest


def expected_counts(
    pinned: Mapping[str, Any] | PinnedDependency = FROZEN_DEPENDENCY,
) -> dict[str, int]:
    record = (
        pinned.as_record() if isinstance(pinned, PinnedDependency)
        else dict(pinned)
    )
    return {
        "participants": int(record["expected_participants"]),
        "assessments": int(record["expected_assessments"]),
        "streams": int(record["expected_streams"]),
        "samples": int(record["expected_samples"]),
    }


__all__ = [
    "ABSENCE_STATUSES",
    "CHECKSUM_FILENAME",
    "DEPENDENCY_FILE_ABSENT",
    "DEPENDENCY_FILE_MALFORMED",
    "DEPENDENCY_VERIFIED",
    "FROZEN_DEPENDENCY",
    "P01_CONTRACT_MISMATCH",
    "P01_DEPENDENCY_FILENAME",
    "P01_EVIDENCE_HASH_MISMATCH",
    "P01_GATE_NOT_PASS",
    "P01_REPORT_ABSENT",
    "P01_REPORT_FILENAME",
    "P01_REPORT_HASH_MISMATCH",
    "RELEASE_COUNTS_MISMATCH",
    "RELEASE_ROOT_ABSENT",
    "SOURCE_MANIFEST_MISMATCH",
    "DependencyVerification",
    "PadsDependencyError",
    "PinnedDependency",
    "dependency_record",
    "expected_counts",
    "load_dependency",
    "read_checksums",
    "verify_dependency",
]
