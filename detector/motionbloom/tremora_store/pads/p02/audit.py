"""PADS-P0.2 index materialization audit and CLI.

The audit verifies the pinned P0.1 authority, materializes the store and every
index, reads the store back to prove replay, and evaluates the sixteen gate
conditions.  Evidence and execution receipt are split exactly as in P0.1, so
two genuine executions produce byte-identical evidence while their receipts
disagree about where and by whom they ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...release_gate import (
    AUDIT_EXECUTION_ERROR,
    AUDIT_EXECUTION_PASS,
    RELEASE_GATE_CONTRACT_VERSION,
    canonical_json_bytes,
    canonical_sha256,
)
from ..release_structure import RELEASE_STRUCTURE_RECONCILED
from ..reproduction import (
    REPRODUCTION_VERIFIED as P01_RECEIPT_VERIFIED,
)
from ..reproduction import (
    build_run_receipt,
    verify_independent_reproduction,
)
from .contract import (
    BLOCKED_DEPENDENCY,
    P02_ARTIFACT_KIND,
    P02_CONTRACT_VERSION,
    P02_IMPLEMENTATION_VERSION,
    P02_SCHEMA_VERSION,
    WITHHELD_P02_ARTIFACTS,
    authority_block,
)
from .dependency import (
    FROZEN_DEPENDENCY,
    verify_dependency,
)
from .gate import (
    REPRODUCTION_NOT_ATTEMPTED,
    REPRODUCTION_VERIFIED,
    PadsP02GateFacts,
    evaluate_gate,
)
from .materialize import materialize, write_success_marker
from .verification import verify_stored_replay

EVIDENCE_FILENAME = "pads_p02_evidence.json"
RECEIPT_FILENAME = "pads_p02_run_receipt.json"
RELEASE_EVALUATED = "EVALUATED"

EXIT_PASS = 0
EXIT_ERROR = 2
EXIT_NO_GO = 3
EXIT_BLOCKED = 4


class PadsP02AuditError(RuntimeError):
    """Raised when the audit itself cannot run."""


def _blocked_record(
    *, reason: str, inspected: Mapping[str, str | None]
) -> dict[str, Any]:
    record = {
        "artifact_kind": P02_ARTIFACT_KIND,
        "schema_version": P02_SCHEMA_VERSION,
        "implementation_version": P02_IMPLEMENTATION_VERSION,
        "contract_version": P02_CONTRACT_VERSION,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "audit_execution_status": AUDIT_EXECUTION_PASS,
        "release_status": BLOCKED_DEPENDENCY,
        "gate_evaluated": False,
        "blocked_reason": reason,
        "inspected_roots": {
            key: value for key, value in sorted(inspected.items())
        },
        "authority": authority_block(),
        "withheld_artifacts": dict(WITHHELD_P02_ARTIFACTS),
        "materialized_release_artifacts": 0,
    }
    for forbidden in ("gate_status", "canonical_evidence_sha256"):
        if forbidden in record:  # pragma: no cover - constructed above
            raise PadsP02AuditError("a blocked record cannot publish a verdict")
    return record


def audit_pads_p02(
    *,
    release_root: Path,
    output_root: Path,
    dependency_path: Path,
    p01_report_path: Path,
    command_arguments: Sequence[str] = (),
    reproduction_receipt: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    process_id: int | None = None,
    progress: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run the P0.2 audit and return ``(record, receipt)``."""

    verification = verify_dependency(
        dependency_path=dependency_path,
        report_path=p01_report_path,
        release_root=release_root,
    )
    if verification.blocks:
        return _blocked_record(
            reason=f"{verification.status}: {verification.detail}",
            inspected={
                "release_root": str(release_root),
                "dependency_path": str(dependency_path),
                "p01_report_path": str(p01_report_path),
            },
        ), None

    pinned = verification.pinned or FROZEN_DEPENDENCY
    # The receipt records the output root, so it must exist even on the path
    # where a disagreeing dependency stops us materializing anything.
    output_root.mkdir(parents=True, exist_ok=True)
    materialized = None
    replayed = None
    facts = PadsP02GateFacts(
        dependency_status=verification.status,
        participants_expected=pinned.expected_participants,
        assessments_expected=pinned.expected_assessments,
        streams_expected=pinned.expected_streams,
        samples_expected=pinned.expected_samples,
        bilateral_task_pairs_expected=pinned.expected_assessments,
        emitted_forbidden_artifacts=dict(WITHHELD_P02_ARTIFACTS),
        reproduction_status=REPRODUCTION_NOT_ATTEMPTED,
    )

    if verification.verified:
        materialized = materialize(
            release_root=release_root,
            output_root=output_root,
            p01_evidence_sha256=pinned.p01_evidence_sha256,
            expected_samples=pinned.expected_samples,
            progress=progress,
        )
        replayed = verify_stored_replay(
            output_root=output_root,
            storage_index=materialized.storage_index,
            windows=materialized.window_records,
            source_sha256_by_stream=materialized.source_sha256_by_stream,
        )
        facts.source_files_expected = materialized.source_files_expected
        facts.source_files_hash_verified = (
            materialized.source_files_hash_verified
        )
        facts.source_files_failed = materialized.source_files_failed
        facts.participants_materialized = (
            materialized.participants_materialized
        )
        facts.assessments_materialized = materialized.assessments_materialized
        facts.streams_materialized = materialized.streams_materialized
        facts.streams_refused = materialized.streams_refused
        facts.samples_materialized = materialized.samples_materialized
        facts.duplicate_materialized_samples = (
            materialized.duplicate_materialized_samples
        )
        facts.row_groups = materialized.row_groups
        facts.streams_with_exactly_one_row_group = (
            materialized.streams_with_exactly_one_row_group
        )
        facts.segment_partition_failures = (
            materialized.segment_partition_failures
        )
        facts.windows = materialized.windows
        facts.windows_crossing_segments = (
            materialized.windows_crossing_segments
        )
        facts.bilateral_task_pairs = materialized.bilateral_task_pairs
        facts.sample_level_alignment_claims = (
            materialized.sample_level_alignment_claims
        )
        facts.fold_count = materialized.fold_count
        facts.participants_without_fold = (
            materialized.participants_without_fold
        )
        facts.participants_in_multiple_folds = (
            materialized.participants_in_multiple_folds
        )
        facts.windows_checked = replayed.windows_checked
        facts.window_replay_failures = replayed.window_replay_failures
        facts.replay_streams_checked = replayed.streams_checked
        facts.replay_byte_exact_streams = replayed.streams_byte_exact
        facts.samples_replayed = replayed.samples_replayed
        facts.source_time_token_failures = replayed.source_time_token_failures

    evidence: dict[str, Any] = {
        "artifact_kind": P02_ARTIFACT_KIND,
        "schema_version": P02_SCHEMA_VERSION,
        "implementation_version": P02_IMPLEMENTATION_VERSION,
        "contract_version": P02_CONTRACT_VERSION,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "authority": authority_block(),
        "p01_dependency": {
            **verification.as_record(),
            "pinned": pinned.as_record(),
        },
        "expected": {
            "participants": pinned.expected_participants,
            "assessments": pinned.expected_assessments,
            "streams": pinned.expected_streams,
            "samples": pinned.expected_samples,
        },
        "materialization": (
            materialized.as_record() if materialized is not None
            else {"materialized": False}
        ),
        "replay_verification": (
            replayed.as_record() if replayed is not None
            else {"verified": False}
        ),
        "withheld_artifacts": dict(WITHHELD_P02_ARTIFACTS),
        "release_structure_expected": RELEASE_STRUCTURE_RECONCILED,
    }
    # The storage index itself is not published row by row; its single binding
    # content hash is, and it lives inside the replay-verification block.
    evidence["materialization"].pop("storage_index", None)
    evidence_sha256 = canonical_sha256(evidence)

    receipt = build_run_receipt(
        dataset_root=release_root,
        source_manifest_sha256=pinned.source_manifest_sha256,
        output_root=output_root,
        schema_version=P02_SCHEMA_VERSION,
        contract_version=P02_CONTRACT_VERSION,
        implementation_version=P02_IMPLEMENTATION_VERSION,
        canonical_evidence_sha256=evidence_sha256,
        command_arguments=tuple(command_arguments),
        run_id=run_id,
        process_id=process_id,
    ).as_record()

    if verify_independent_reproduction(
        receipt, reproduction_receipt
    ) == P01_RECEIPT_VERIFIED:
        facts.reproduction_status = REPRODUCTION_VERIFIED

    record: dict[str, Any] = dict(evidence)
    record["audit_execution_status"] = AUDIT_EXECUTION_PASS
    record["release_status"] = RELEASE_EVALUATED
    record["gate_evaluated"] = True
    record["canonical_evidence_sha256"] = evidence_sha256
    # An execution fact, so it sits in the envelope rather than the evidence:
    # two genuine runs must agree on the evidence and disagree on this.
    record["independent_reproduction_status"] = facts.reproduction_status
    record["run_receipt"] = receipt
    gate = evaluate_gate(facts)
    record.update(gate.as_record())
    record["materialized_release_artifacts"] = (
        materialized.streams_materialized if materialized is not None else 0
    )
    if gate.satisfied:
        write_success_marker(output_root)
    return record, receipt


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dependency", required=True, type=Path)
    parser.add_argument("--p01-report", required=True, type=Path)
    parser.add_argument("--reproduction-receipt", type=Path)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    payload = b""
    try:
        args.output_root.mkdir(parents=True, exist_ok=True)
        reproduction = None
        if args.reproduction_receipt is not None:
            reproduction = json.loads(
                args.reproduction_receipt.read_bytes().decode("utf-8")
            )
        record, receipt = audit_pads_p02(
            release_root=args.release_root,
            output_root=args.output_root,
            dependency_path=args.dependency,
            p01_report_path=args.p01_report,
            command_arguments=tuple(argv or sys.argv[1:]),
            reproduction_receipt=reproduction,
            progress=args.progress,
        )
        payload = canonical_json_bytes(record)
        with (args.output_root / EVIDENCE_FILENAME).open("xb") as handle:
            handle.write(payload)
        if receipt is not None:
            with (args.output_root / RECEIPT_FILENAME).open("xb") as handle:
                handle.write(canonical_json_bytes(receipt))
    except (PadsP02AuditError, OSError, ValueError) as exc:
        print(f"{AUDIT_EXECUTION_ERROR}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    sys.stdout.buffer.write(payload)
    if not record.get("gate_evaluated"):
        return EXIT_BLOCKED
    return EXIT_PASS if record["gate_status"].startswith("PASS") else EXIT_NO_GO


if __name__ == "__main__":
    raise SystemExit(main())
