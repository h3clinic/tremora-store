"""PADS-P0.3 spectral-preservation audit and CLI.

The audit verifies the pinned P0.2.1 authority, materializes the spectral
workload and the independent source-versus-replay comparison, runs the kernel
controls in this process, and evaluates the sixteen gate conditions.  Evidence
and execution receipt are split as in P0.1 and P0.2, so two genuine executions
produce byte-identical evidence while their receipts disagree about where and
by whom they ran.
"""

from __future__ import annotations

import argparse
import json
import os
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
from ..reproduction import (
    REPRODUCTION_VERIFIED as RECEIPT_VERIFIED,
)
from ..reproduction import (
    build_run_receipt,
    verify_independent_reproduction,
)
from .contract import (
    BLOCKED_DEPENDENCY,
    FREQUENCY_BIN_COUNT,
    P03_ARTIFACT_KIND,
    P03_CONTRACT_VERSION,
    P03_IMPLEMENTATION_VERSION,
    P03_SCHEMA_VERSION,
    SENSOR_FAMILIES,
    SUCCESS_MARKER,
    WITHHELD_P03_ARTIFACTS,
    authority_block,
)
from .dependency import FROZEN_DEPENDENCY, verify_dependency
from .gate import (
    REPRODUCTION_NOT_ATTEMPTED,
    REPRODUCTION_VERIFIED,
    PadsP03GateFacts,
    evaluate_gate,
)
from .grid import grid_hash, grid_id
from .kernel_controls import run_controls
from .materialize import materialize

EVIDENCE_FILENAME = "pads_p03_evidence.json"
RECEIPT_FILENAME = "pads_p03_run_receipt.json"
RELEASE_EVALUATED = "EVALUATED"

EXIT_PASS = 0
EXIT_ERROR = 2
EXIT_NO_GO = 3
EXIT_BLOCKED = 4

#: The authoritative run is single-threaded: nothing here calls BLAS, and
#: pinning the thread count keeps that true if a dependency changes.
_SINGLE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class PadsP03AuditError(RuntimeError):
    """Raised when the audit itself cannot run."""


def pin_single_thread() -> dict[str, str]:
    """Pin numeric threading and report what was set."""

    for name in _SINGLE_THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    return {name: os.environ[name] for name in _SINGLE_THREAD_VARIABLES}


def _blocked_record(
    *, reason: str, inspected: Mapping[str, str | None]
) -> dict[str, Any]:
    return {
        "artifact_kind": P03_ARTIFACT_KIND,
        "schema_version": P03_SCHEMA_VERSION,
        "implementation_version": P03_IMPLEMENTATION_VERSION,
        "contract_version": P03_CONTRACT_VERSION,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "audit_execution_status": AUDIT_EXECUTION_PASS,
        "release_status": BLOCKED_DEPENDENCY,
        "gate_evaluated": False,
        "blocked_reason": reason,
        "inspected_roots": {
            key: value for key, value in sorted(inspected.items())
        },
        "authority": authority_block(),
        "withheld_artifacts": dict(WITHHELD_P03_ARTIFACTS),
        "materialized_release_artifacts": 0,
    }


def audit_pads_p03(
    *,
    release_root: Path,
    store_root: Path,
    output_root: Path,
    dependency_path: Path,
    p02_report_path: Path,
    command_arguments: Sequence[str] = (),
    reproduction_receipt: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    process_id: int | None = None,
    progress: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run the P0.3 audit and return ``(record, receipt)``."""

    threading = pin_single_thread()
    verification = verify_dependency(
        dependency_path=dependency_path,
        p02_report_path=p02_report_path,
        store_root=store_root,
    )
    if verification.blocks:
        return _blocked_record(
            reason=f"{verification.status}: {verification.detail}",
            inspected={
                "release_root": str(release_root),
                "store_root": str(store_root),
                "dependency_path": str(dependency_path),
                "p02_report_path": str(p02_report_path),
            },
        ), None

    pinned = verification.pinned or FROZEN_DEPENDENCY
    output_root.mkdir(parents=True, exist_ok=True)
    controls = run_controls()

    produced = None
    facts = PadsP03GateFacts(
        dependency_status=verification.status,
        grid_hash=grid_hash(),
        pinned_grid_hash=grid_hash(),
        grid_bin_count=FREQUENCY_BIN_COUNT,
        sensor_family_count=len(SENSOR_FAMILIES),
        kernel_controls_status=controls["status"],
        every_length_accepted=bool(
            controls["controls"]["every_observed_length_accepted"]
        ),
        emitted_forbidden_artifacts=dict(WITHHELD_P03_ARTIFACTS),
        reproduction_status=REPRODUCTION_NOT_ATTEMPTED,
    )

    if verification.verified:
        produced = materialize(
            release_root=release_root,
            store_root=store_root,
            output_root=output_root,
            progress=progress,
        )
        facts.workload_selection_stable = produced.workload_selection_stable
        facts.streams_with_valid_windows = produced.streams_with_valid_windows
        facts.workload_windows_selected = produced.workload_windows_selected
        facts.workload_distinct_streams = produced.workload_distinct_streams
        facts.workload_windows_eligible = produced.workload_windows_eligible
        facts.windows_differing_from_nominal_grid = (
            produced.windows_differing_from_nominal_grid
        )
        facts.nominal_grid_substitutions = produced.nominal_grid_substitutions
        facts.nyquist_derived_from_dt_ref_rows = (
            produced.nyquist_derived_from_dt_ref_rows
        )
        facts.declared_rate_nyquist_rows = produced.declared_rate_nyquist_rows
        facts.distinct_sample_counts = produced.distinct_sample_counts
        facts.windows_refused_for_length = (
            produced.windows_refused_for_length
        )
        facts.raw_axis_sum_mismatches = produced.raw_axis_sum_mismatches
        facts.vector_magnitude_uses = produced.vector_magnitude_uses
        facts.audit_windows_selected = produced.audit_windows_selected
        facts.source_replay_row_mismatches = (
            produced.source_replay_row_mismatches
        )
        facts.source_replay_input_hash_mismatches = (
            produced.source_replay_input_hash_mismatches
        )
        facts.source_replay_spectral_hash_mismatches = (
            produced.source_replay_spectral_hash_mismatches
        )
        facts.dominant_frequency_mismatches = (
            produced.dominant_frequency_mismatches
        )
        facts.maximum_observed_bin_error = (
            produced.maximum_observed_bin_error
        )
        facts.source_unreadable = produced.source_unreadable
        facts.spectral_rows_written = (
            produced.gyro_spectral_rows + produced.accel_spectral_rows
        )

    evidence: dict[str, Any] = {
        "artifact_kind": P03_ARTIFACT_KIND,
        "schema_version": P03_SCHEMA_VERSION,
        "implementation_version": P03_IMPLEMENTATION_VERSION,
        "contract_version": P03_CONTRACT_VERSION,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "authority": authority_block(),
        "p02_dependency": {
            **verification.as_record(),
            "pinned": pinned.as_record(),
        },
        "frequency_grid": {
            "frequency_grid_id": grid_id(),
            "frequency_grid_hash": grid_hash(),
            "frequency_bin_count": FREQUENCY_BIN_COUNT,
        },
        "kernel_controls": controls,
        "materialization": (
            produced.as_record() if produced is not None
            else {"materialized": False}
        ),
        "withheld_artifacts": dict(WITHHELD_P03_ARTIFACTS),
        "numeric_execution": {
            "threading": dict(sorted(threading.items())),
            "blas_used": False,
        },
    }
    evidence_sha256 = canonical_sha256(evidence)

    receipt = build_run_receipt(
        dataset_root=release_root,
        source_manifest_sha256=pinned.source_manifest_sha256,
        output_root=output_root,
        schema_version=P03_SCHEMA_VERSION,
        contract_version=P03_CONTRACT_VERSION,
        implementation_version=P03_IMPLEMENTATION_VERSION,
        canonical_evidence_sha256=evidence_sha256,
        command_arguments=tuple(command_arguments),
        run_id=run_id,
        process_id=process_id,
    ).as_record()

    if verify_independent_reproduction(
        receipt, reproduction_receipt
    ) == RECEIPT_VERIFIED:
        facts.reproduction_status = REPRODUCTION_VERIFIED

    record: dict[str, Any] = dict(evidence)
    record["audit_execution_status"] = AUDIT_EXECUTION_PASS
    record["release_status"] = RELEASE_EVALUATED
    record["gate_evaluated"] = True
    record["canonical_evidence_sha256"] = evidence_sha256
    record["independent_reproduction_status"] = facts.reproduction_status
    record["run_receipt"] = receipt
    gate = evaluate_gate(facts)
    record.update(gate.as_record())
    record["materialized_release_artifacts"] = (
        facts.spectral_rows_written if produced is not None else 0
    )
    if gate.satisfied:
        (output_root / SUCCESS_MARKER).write_bytes(b"")
    return record, receipt


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dependency", required=True, type=Path)
    parser.add_argument("--p02-report", required=True, type=Path)
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
        record, receipt = audit_pads_p03(
            release_root=args.release_root,
            store_root=args.store_root,
            output_root=args.output_root,
            dependency_path=args.dependency,
            p02_report_path=args.p02_report,
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
    except (PadsP03AuditError, OSError, ValueError) as exc:
        print(f"{AUDIT_EXECUTION_ERROR}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    sys.stdout.buffer.write(payload)
    if not record.get("gate_evaluated"):
        return EXIT_BLOCKED
    return EXIT_PASS if record["gate_status"].startswith("PASS") else EXIT_NO_GO


if __name__ == "__main__":
    raise SystemExit(main())
