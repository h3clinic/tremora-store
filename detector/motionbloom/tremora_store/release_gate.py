"""Shared release-gate vocabulary for data-gated TremoraStore audits.

Two facts are reported separately and must never be collapsed into one:
whether the audit *ran* (``audit_execution_status``) and whether the dataset
*passed* (``gate_status``).  VIDIMU v0.5 established the pattern — a successful
audit that closes its gate — and every P0.1 audit inherits it.

A third state exists below both: an audit with nothing to audit.  It reports
:data:`BLOCKED_INPUT_DATA_UNAVAILABLE`, evaluates no gate, and publishes no
evidence hash.  Blocked means the release was absent, never that it was
malformed: an unparseable file, an empty manifest and a traversing index entry
are all evidence *about* a release, and evidence closes a gate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .timing_authority import (
    DatasetTimingBinding,
    TimingAuthorityError,
    normalize,
)

RELEASE_GATE_CONTRACT_VERSION = "tremora-p01-release-gate-0.1.0"

AUDIT_EXECUTION_PASS = "PASS"
AUDIT_EXECUTION_ERROR = "ERROR"

BLOCKED_INPUT_DATA_UNAVAILABLE = "BLOCKED_INPUT_DATA_UNAVAILABLE"

#: Exit codes are part of the contract: a caller must be able to distinguish
#: "we audited and it failed" from "we have not got the data".
EXIT_PASS = 0
EXIT_ERROR = 2
EXIT_NO_GO = 3
EXIT_BLOCKED = 4

#: Artifacts a P0.1 audit is forbidden to emit, whatever its verdict.  The
#: counts are published so a reader can see the claim boundary held rather than
#: infer it from an absence.
WITHHELD_P01_ARTIFACTS: Mapping[str, int] = {
    "contiguous_window_tables": 0,
    "frame_imu_index_tables": 0,
    "spectral_feature_tables": 0,
    "storage_benchmark_result_tables": 0,
    "success_markers": 0,
}

SUCCESS_MARKER = "_SUCCESS"


class ReleaseGateError(RuntimeError):
    """Raised when a release record would misstate its own evidence."""


def canonical_json_bytes(value: object) -> bytes:
    """Return timestamp-free canonical JSON used for evidence hashes.

    Byte-identical to the v0.5D evidence encoding; the contract tests assert
    that equality so the project keeps exactly one canonical form.
    """

    try:
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
    except (TypeError, ValueError) as exc:
        raise ReleaseGateError("release record is not canonical JSON") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: object) -> str:
    """Return the canonical-content hash of one release record."""

    return sha256_bytes(canonical_json_bytes(value))


def authority_fields(binding: DatasetTimingBinding) -> dict[str, Any]:
    """Return the authority block every record carries, blocked ones included.

    The blocked record is the one artifact most likely to escape review, so it
    asserts the same claim boundary as a verdict-bearing record.
    """

    tier = normalize(binding.timing_authority)
    return {
        "dataset_id": binding.dataset_id,
        "timing_authority": tier.value,
        "derived_under_assumption": binding.derived_under_assumption,
        "hardware_sync_claim": binding.hardware_sync_claim,
        "paired_modalities": binding.paired_modalities,
        "raw_shared_clock": tier.value == "RAW_SHARED_CLOCK",
    }


def blocked_record(
    *,
    binding: DatasetTimingBinding,
    artifact_kind: str,
    schema_version: str,
    implementation_version: str,
    reason: str,
    inspected_roots: Mapping[str, str | None],
) -> dict[str, Any]:
    """Build the canonical record for an audit that had nothing to audit.

    No gate status, no evidence hash and no materialized artifact appears:
    publishing a verdict with no data behind it is the failure mode this whole
    architecture exists to prevent.
    """

    if not reason:
        raise ReleaseGateError("a blocked record must name what was missing")
    record: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "schema_version": schema_version,
        "implementation_version": implementation_version,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "audit_execution_status": AUDIT_EXECUTION_PASS,
        "release_status": BLOCKED_INPUT_DATA_UNAVAILABLE,
        "gate_evaluated": False,
        "blocked_reason": reason,
        "inspected_roots": {
            key: value for key, value in sorted(inspected_roots.items())
        },
        "authority": authority_fields(binding),
        "withheld_artifacts": dict(WITHHELD_P01_ARTIFACTS),
        "materialized_release_artifacts": 0,
    }
    assert_blocked_record_claim_boundary(record)
    return record


def assert_blocked_record_claim_boundary(record: Mapping[str, Any]) -> None:
    """Refuse a blocked record that leaks a verdict or an evidence hash."""

    if record.get("release_status") != BLOCKED_INPUT_DATA_UNAVAILABLE:
        raise ReleaseGateError("record is not a blocked record")
    if record.get("gate_evaluated") is not False:
        raise ReleaseGateError("a blocked record cannot evaluate a gate")
    forbidden = {
        "canonical_evidence_sha256",
        "evidence_sha256",
        "gate_conditions",
        "gate_status",
    }
    present = forbidden.intersection(record)
    if present:
        raise ReleaseGateError(
            f"a blocked record cannot publish {sorted(present)!r}")
    if record.get("materialized_release_artifacts") != 0:
        raise ReleaseGateError(
            "a blocked record cannot materialize release artifacts")
    withheld = record.get("withheld_artifacts")
    if withheld != dict(WITHHELD_P01_ARTIFACTS):
        raise ReleaseGateError(
            "a blocked record must assert the full claim boundary")
    authority = record.get("authority")
    if not isinstance(authority, Mapping):
        raise ReleaseGateError("a blocked record must carry its authority")
    try:
        normalize(str(authority.get("timing_authority")))
    except TimingAuthorityError as exc:
        raise ReleaseGateError(
            "a blocked record must name a declared timing-authority tier"
        ) from exc


def exit_code_for(record: Mapping[str, Any]) -> int:
    """Map one release record onto its contractual process exit code."""

    if record.get("audit_execution_status") == AUDIT_EXECUTION_ERROR:
        return EXIT_ERROR
    if record.get("release_status") == BLOCKED_INPUT_DATA_UNAVAILABLE:
        return EXIT_BLOCKED
    if not record.get("gate_evaluated"):
        raise ReleaseGateError(
            "a non-blocked record must evaluate its gate")
    status = str(record.get("gate_status", ""))
    if status.startswith("PASS"):
        return EXIT_PASS
    if status.startswith("NO_GO"):
        return EXIT_NO_GO
    raise ReleaseGateError(f"unrecognized gate status {status!r}")


__all__ = [
    "AUDIT_EXECUTION_ERROR",
    "AUDIT_EXECUTION_PASS",
    "BLOCKED_INPUT_DATA_UNAVAILABLE",
    "EXIT_BLOCKED",
    "EXIT_ERROR",
    "EXIT_NO_GO",
    "EXIT_PASS",
    "RELEASE_GATE_CONTRACT_VERSION",
    "SUCCESS_MARKER",
    "WITHHELD_P01_ARTIFACTS",
    "ReleaseGateError",
    "assert_blocked_record_claim_boundary",
    "authority_fields",
    "blocked_record",
    "canonical_json_bytes",
    "canonical_sha256",
    "exit_code_for",
    "sha256_bytes",
]
