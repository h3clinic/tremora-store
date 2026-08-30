"""Everything that must be true before a single query is timed.

Two failures made this module necessary.  A run pointed at a fresh baseline
root rebuilt 3.2 GB of baselines it already had, and nothing checked whether
the volume could hold what the run was about to write, so the disk filled
mid-benchmark and took the tooling down with it.

So: baselines are verified, never rebuilt silently, and the footprint is
projected from the actual workload and checked against actual free space
before the run starts.  Refusing to start is a third terminal state, distinct
from a NO-GO verdict -- a benchmark that could not run has not failed, it has
not happened.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import (
    B1,
    B2,
    COMPRESSION_CODEC,
    COMPRESSION_LEVEL,
    M1,
    MEASURED_ROUNDS_BY_QUERY_CLASS,
    P05_CONTRACT_VERSION,
    REPRESENTATIONS,
)

PREFLIGHT_OK = "PREFLIGHT_OK"
BASELINE_ABSENT = "BASELINE_ABSENT"
BASELINE_HASH_MISMATCH = "BASELINE_HASH_MISMATCH"
BASELINE_CONTRACT_MISMATCH = "BASELINE_CONTRACT_MISMATCH"
WORKLOAD_HASH_MISMATCH = "WORKLOAD_HASH_MISMATCH"
SOURCE_MANIFEST_MISMATCH = "SOURCE_MANIFEST_MISMATCH"
INSUFFICIENT_DISK = "INSUFFICIENT_DISK"

#: The frozen workload's content hash, fixed when the contract was frozen.
FROZEN_WORKLOAD_SHA256 = (
    "f84857e5b407af7121b1a930e503029c1d7f5eb50c426c6c2d2d9202ec8ce9da"
)

#: Bytes one measured row costs in the compressed timing table.  Measured
#: rather than guessed: the 64-character content hash dominates and barely
#: compresses, the identifiers and enums compress well.  Checked by a test
#: against a real written table so it cannot drift into optimism.
BYTES_PER_MEASURED_ROW = 96

#: An absolute floor, so a run never leaves the volume with nothing spare
#: even when its own projection is small.
MINIMUM_FREE_MARGIN_BYTES = 2 * 1024**3

#: The projection is doubled because two runs write two roots.
RUN_COUNT = 2


class PreflightError(RuntimeError):
    """Raised when the run must not start."""


@dataclass(slots=True)
class BaselineIdentity:
    """What a built baseline is, so reuse can be verified rather than hoped."""

    representation: str
    content_sha256: str
    file_count: int
    physical_storage_bytes: int
    contract_version: str = P05_CONTRACT_VERSION
    compression_codec: str = COMPRESSION_CODEC
    compression_level: int = COMPRESSION_LEVEL

    def as_record(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "content_sha256": self.content_sha256,
            "file_count": self.file_count,
            "physical_storage_bytes": self.physical_storage_bytes,
            "contract_version": self.contract_version,
            "compression_codec": self.compression_codec,
            "compression_level": self.compression_level,
        }


def _tree_sha256(root: Path) -> tuple[str, int, int]:
    """Hash a directory's contents by relative path, size and bytes."""

    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\x1f")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\x1e")
        count += 1
        total += size
    return digest.hexdigest(), count, total


def baseline_identity(representation: str, root: Path) -> BaselineIdentity:
    """Compute one baseline's identity from what is on disk."""

    if not root.is_dir():
        raise PreflightError(f"{representation}: {root} is absent")
    content, count, total = _tree_sha256(root)
    if count == 0:
        raise PreflightError(f"{representation}: {root} holds no files")
    return BaselineIdentity(
        representation=representation,
        content_sha256=content,
        file_count=count,
        physical_storage_bytes=total,
    )


def write_baseline_identities(
    path: Path, identities: Mapping[str, BaselineIdentity]
) -> None:
    payload = {
        "contract_version": P05_CONTRACT_VERSION,
        "workload_content_sha256": FROZEN_WORKLOAD_SHA256,
        "baselines": {
            name: identity.as_record()
            for name, identity in sorted(identities.items())
        },
    }
    path.write_bytes(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    )


@dataclass(slots=True)
class PreflightReport:
    """Whether the run may start, and the numbers that decided it."""

    status: str = PREFLIGHT_OK
    detail: str = ""
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    projected_run_bytes: int = 0
    projected_total_bytes: int = 0
    required_free_bytes: int = 0
    free_bytes: int = 0
    total_bytes: int = 0
    measured_rows_projected: int = 0
    workload_content_sha256: str = ""
    source_manifest_sha256: str = ""

    @property
    def ok(self) -> bool:
        return self.status == PREFLIGHT_OK

    def deterministic_record(self) -> dict[str, Any]:
        """The half of the preflight that two honest runs must agree on.

        How much room the volume happened to have is a fact about the machine
        at that moment -- run B sees less than run A because run A just wrote
        a timing table -- so it belongs with the timings, not in the evidence
        hash.  What the run verified and what it projected are deterministic
        and stay here.
        """

        return {
            "status": self.status,
            "baselines": {
                name: dict(sorted(entry.items()))
                for name, entry in sorted(self.baselines.items())
            },
            "measured_rows_projected": self.measured_rows_projected,
            "bytes_per_measured_row": BYTES_PER_MEASURED_ROW,
            "projected_run_bytes": self.projected_run_bytes,
            "projected_total_bytes": self.projected_total_bytes,
            "run_count": RUN_COUNT,
            "minimum_free_margin_bytes": MINIMUM_FREE_MARGIN_BYTES,
            "workload_content_sha256": self.workload_content_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
        }

    def volume_record(self) -> dict[str, Any]:
        """What the volume held when this run started.  Published, not hashed."""

        return {
            "required_free_bytes": self.required_free_bytes,
            "free_bytes": self.free_bytes,
            "total_bytes": self.total_bytes,
            "detail": self.detail,
        }

    def as_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "baselines": {
                name: dict(sorted(entry.items()))
                for name, entry in sorted(self.baselines.items())
            },
            "measured_rows_projected": self.measured_rows_projected,
            "bytes_per_measured_row": BYTES_PER_MEASURED_ROW,
            "projected_run_bytes": self.projected_run_bytes,
            "projected_total_bytes": self.projected_total_bytes,
            "required_free_bytes": self.required_free_bytes,
            "free_bytes": self.free_bytes,
            "total_bytes": self.total_bytes,
            "run_count": RUN_COUNT,
            "minimum_free_margin_bytes": MINIMUM_FREE_MARGIN_BYTES,
            "workload_content_sha256": self.workload_content_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
        }


def project_measured_rows(query_counts: Mapping[str, int]) -> int:
    """How many timed rows this workload will write, exactly.

    Derived from the workload's own counts and the per-class round budget, so
    it tracks the contract rather than a remembered number.
    """

    return sum(
        int(count)
        * MEASURED_ROUNDS_BY_QUERY_CLASS.get(name, 0)
        * len(REPRESENTATIONS)
        for name, count in query_counts.items()
    )


def project_run_bytes(
    query_counts: Mapping[str, int], *, table_overhead_bytes: int = 8 * 1024**2
) -> tuple[int, int]:
    """``(rows, bytes)`` one run root will occupy."""

    rows = project_measured_rows(query_counts)
    return rows, rows * BYTES_PER_MEASURED_ROW + table_overhead_bytes


def check_disk(
    target: Path, projected_run_bytes: int
) -> tuple[bool, int, int, int]:
    """``(ok, required, free, total)`` for the volume the run writes to."""

    usage = shutil.disk_usage(target)
    required = (
        RUN_COUNT * projected_run_bytes + MINIMUM_FREE_MARGIN_BYTES
    )
    return usage.free >= required, required, usage.free, usage.total


def run_preflight(
    *,
    baseline_root: Path,
    store_root: Path,
    output_root: Path,
    query_counts: Mapping[str, int],
    workload_content_sha256: str,
    source_manifest_sha256: str,
    expected_identities: Mapping[str, Mapping[str, Any]] | None = None,
    expected_workload_sha256: str = FROZEN_WORKLOAD_SHA256,
    expected_source_manifest_sha256: str | None = None,
    required_baselines: Sequence[str] = (B1, B2),
) -> PreflightReport:
    """Verify the inputs and the volume, and say whether the run may start."""

    report = PreflightReport(
        workload_content_sha256=workload_content_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )

    if workload_content_sha256 != expected_workload_sha256:
        report.status = WORKLOAD_HASH_MISMATCH
        report.detail = (
            f"workload is {workload_content_sha256[:16]}, frozen at "
            f"{expected_workload_sha256[:16]}"
        )
        return report

    if (
        expected_source_manifest_sha256 is not None
        and source_manifest_sha256 != expected_source_manifest_sha256
    ):
        report.status = SOURCE_MANIFEST_MISMATCH
        report.detail = "the representations do not share one source manifest"
        return report

    roots = {B1: baseline_root / "b1", B2: baseline_root / "b2", M1: store_root}
    for name in required_baselines:
        root = roots[name]
        if not root.is_dir():
            report.status = BASELINE_ABSENT
            report.detail = f"{name}: {root} is absent; build it first"
            return report
    for name in (*required_baselines, M1):
        try:
            identity = baseline_identity(name, roots[name])
        except PreflightError as exc:
            report.status = BASELINE_ABSENT
            report.detail = str(exc)
            return report
        report.baselines[name] = identity.as_record()
        if expected_identities and name in expected_identities:
            expected = expected_identities[name]
            if expected.get("contract_version") != P05_CONTRACT_VERSION:
                report.status = BASELINE_CONTRACT_MISMATCH
                report.detail = (
                    f"{name} was built under "
                    f"{expected.get('contract_version')!r}"
                )
                return report
            if expected.get("content_sha256") != identity.content_sha256:
                report.status = BASELINE_HASH_MISMATCH
                report.detail = (
                    f"{name} content is {identity.content_sha256[:16]}, "
                    f"recorded as {str(expected.get('content_sha256'))[:16]}"
                )
                return report

    rows, run_bytes = project_run_bytes(query_counts)
    report.measured_rows_projected = rows
    report.projected_run_bytes = run_bytes
    report.projected_total_bytes = RUN_COUNT * run_bytes
    ok, required, free, total = check_disk(output_root, run_bytes)
    report.required_free_bytes = required
    report.free_bytes = free
    report.total_bytes = total
    if not ok:
        report.status = INSUFFICIENT_DISK
        report.detail = (
            f"{free / 1024**3:.2f} GB free, {required / 1024**3:.2f} GB "
            f"required for {RUN_COUNT} runs of "
            f"{run_bytes / 1024**3:.2f} GB plus a "
            f"{MINIMUM_FREE_MARGIN_BYTES / 1024**3:.2f} GB margin"
        )
        return report

    report.detail = (
        f"{len(report.baselines)} baselines verified; {rows} rows projected "
        f"at {run_bytes / 1024**3:.2f} GB per run"
    )
    return report


__all__ = [
    "BASELINE_ABSENT",
    "BASELINE_CONTRACT_MISMATCH",
    "BASELINE_HASH_MISMATCH",
    "BYTES_PER_MEASURED_ROW",
    "FROZEN_WORKLOAD_SHA256",
    "INSUFFICIENT_DISK",
    "MINIMUM_FREE_MARGIN_BYTES",
    "PREFLIGHT_OK",
    "RUN_COUNT",
    "SOURCE_MANIFEST_MISMATCH",
    "WORKLOAD_HASH_MISMATCH",
    "BaselineIdentity",
    "PreflightError",
    "PreflightReport",
    "baseline_identity",
    "check_disk",
    "project_measured_rows",
    "project_run_bytes",
    "run_preflight",
    "write_baseline_identities",
]
