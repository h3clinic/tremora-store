"""PADS-P0.4 rate-ablation and anti-aliasing audit and CLI.

The audit verifies the pinned P0.3 authority, measures the frozen filters in
this process rather than trusting the numbers written into a commit message,
materializes every derived rate over the P0.3 workload, runs the resampling
controls, and evaluates the eighteen gate conditions.  Evidence and execution
receipt are split as in P0.1, P0.2 and P0.3, so two genuine executions produce
byte-identical evidence while their receipts disagree about where and by whom
they ran.
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
    DERIVED_RATES_HZ,
    P04_ARTIFACT_KIND,
    P04_CONTRACT_VERSION,
    P04_IMPLEMENTATION_VERSION,
    P04_SCHEMA_VERSION,
    PER_OUTPUT_WEIGHT_NORMALIZATION,
    RATES_WITH_EXACT_PICOSECOND_PERIOD,
    REFERENCE_MILESTONE,
    RESAMPLING_DOMAIN,
    SENSOR_FAMILIES,
    SUCCESS_MARKER,
    WITHHELD_P04_ARTIFACTS,
    authority_block,
)
from .controls import run_controls
from .dependency import (
    FROZEN_DEPENDENCY,
    observed_spectral_table_hash,
    verify_dependency,
)
from .filters import (
    FILTER_SPECS,
    coefficients_sha256,
    dc_terminology,
    filter_manifest,
    measured_specification,
)
from .gate import (
    REPRODUCTION_NOT_ATTEMPTED,
    REPRODUCTION_VERIFIED,
    PadsP04GateFacts,
    evaluate_gate,
)
from .materialize import materialize

EVIDENCE_FILENAME = "pads_p04_evidence.json"
RECEIPT_FILENAME = "pads_p04_run_receipt.json"
RELEASE_EVALUATED = "EVALUATED"

EXIT_PASS = 0
EXIT_ERROR = 2
EXIT_NO_GO = 3
EXIT_BLOCKED = 4

#: The authoritative run is single-threaded: the resampler is written against
#: numpy's own arithmetic and never calls BLAS, and pinning the thread count
#: keeps that true if a dependency changes.
_SINGLE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class PadsP04AuditError(RuntimeError):
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
        "artifact_kind": P04_ARTIFACT_KIND,
        "schema_version": P04_SCHEMA_VERSION,
        "implementation_version": P04_IMPLEMENTATION_VERSION,
        "contract_version": P04_CONTRACT_VERSION,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "audit_execution_status": AUDIT_EXECUTION_PASS,
        "release_status": BLOCKED_DEPENDENCY,
        "gate_evaluated": False,
        "blocked_reason": reason,
        "inspected_roots": {
            key: value for key, value in sorted(inspected.items())
        },
        "authority": authority_block(),
        "withheld_artifacts": dict(WITHHELD_P04_ARTIFACTS),
        "materialized_release_artifacts": 0,
    }


def measure_filters() -> dict[str, Any]:
    """Measure every frozen filter here rather than quoting a design note."""

    measured = {
        str(rate): measured_specification(rate)
        for rate in sorted(FILTER_SPECS)
    }
    return {
        "manifest": filter_manifest(),
        "coefficients_sha256": coefficients_sha256(),
        "measured": measured,
        "dc_terminology": {
            str(rate): dc_terminology(rate) for rate in sorted(FILTER_SPECS)
        },
        "worst_passband_ripple_db": max(
            float(entry["passband_ripple_db"]) for entry in measured.values()
        ),
        "worst_stopband_attenuation_db": min(
            float(entry["stopband_attenuation_db"])
            for entry in measured.values()
        ),
    }


def audit_pads_p04(
    *,
    release_root: Path,
    store_root: Path,
    p03_root: Path,
    output_root: Path,
    dependency_path: Path,
    p03_report_path: Path,
    command_arguments: Sequence[str] = (),
    reproduction_receipt: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    process_id: int | None = None,
    progress: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run the P0.4 audit and return ``(record, receipt)``."""

    threading = pin_single_thread()
    verification = verify_dependency(
        dependency_path=dependency_path,
        p03_report_path=p03_report_path,
        p03_root=p03_root,
        store_root=store_root,
    )
    if verification.blocks:
        return _blocked_record(
            reason=f"{verification.status}: {verification.detail}",
            inspected={
                "release_root": str(release_root),
                "store_root": str(store_root),
                "p03_root": str(p03_root),
                "dependency_path": str(dependency_path),
                "p03_report_path": str(p03_report_path),
            },
        ), None

    pinned = verification.pinned or FROZEN_DEPENDENCY
    output_root.mkdir(parents=True, exist_ok=True)
    controls = run_controls()
    filters = measure_filters()
    constant = controls["measured"]["constant_input_30"]

    produced = None
    facts = PadsP04GateFacts(
        dependency_status=verification.status,
        coefficients_hash=filters["coefficients_sha256"],
        pinned_coefficients_hash=pinned.anti_alias_coefficients_sha256,
        filter_rates_measured=len(filters["measured"]),
        worst_passband_ripple_db=filters["worst_passband_ripple_db"],
        worst_stopband_attenuation_db=filters[
            "worst_stopband_attenuation_db"
        ],
        resampling_domain=RESAMPLING_DOMAIN,
        reference_spectral_table_hash=observed_spectral_table_hash(p03_root),
        pinned_spectral_table_hash=pinned.p03_spectral_table_sha256,
        reference_milestone=REFERENCE_MILESTONE,
        rates_with_exact_picosecond_period=RATES_WITH_EXACT_PICOSECOND_PERIOD,
        per_phase_normalization=PER_OUTPUT_WEIGHT_NORMALIZATION,
        branch_gain_rates_measured=len(filters["measured"]),
        published_branch_gain_spread_db=max(
            float(entry["polyphase_dc_gain_spread_db"])
            for entry in filters["dc_terminology"].values()
        ),
        observed_branch_gain_spread_db=constant["observed_ripple_db"],
        within_phase_gain_spread=constant["within_phase_spread"],
        sensor_family_count=len(SENSOR_FAMILIES),
        controls_status=controls["status"],
        intersection_control_passed=bool(
            controls["controls"]["support_intersection_precedes_the_filter_guard"]
        ),
        emitted_forbidden_artifacts=dict(WITHHELD_P04_ARTIFACTS),
        reproduction_status=REPRODUCTION_NOT_ATTEMPTED,
    )

    if verification.verified:
        produced = materialize(
            release_root=release_root,
            store_root=store_root,
            p03_root=p03_root,
            output_root=output_root,
            progress=progress,
        )
        facts.segments_derived = produced.segments_derived
        facts.windows_used_as_resampling_domain = 0
        facts.rates_materialized = tuple(produced.rates_materialized)
        facts.rational_timing_ordinals_checked = (
            produced.rational_timing_ordinals_checked
        )
        facts.rational_timing_mismatches = (
            produced.rational_timing_mismatches
        )
        facts.rounded_thirty_hz_ordinals = produced.rounded_thirty_hz_ordinals
        facts.ordinals_admitted_by_filter_guard_alone = (
            produced.ordinals_admitted_by_filter_guard_alone
        )
        facts.ordinals_removed_by_parent_stage = (
            produced.ordinals_removed_by_parent_stage
        )
        facts.ordinals_admitted_over_unbracketed_parent = (
            produced.ordinals_admitted_over_unbracketed_parent
        )
        facts.ordinals_checked_against_parent_support = (
            produced.rational_timing_ordinals_checked
        )
        facts.derived_samples_written = produced.derived_samples_written
        facts.derived_sample_count_mismatches = (
            produced.derived_sample_count_mismatches
        )
        facts.core_band_rows = produced.core_summary_rows
        facts.edge_band_rows = produced.edge_summary_rows
        facts.merged_band_rows = (
            produced.participant_summary_rows
            - produced.core_summary_rows
            - produced.edge_summary_rows
        )
        facts.audit_comparisons = produced.audit_comparisons
        facts.derived_value_mismatches = (
            produced.source_replay_derived_mismatches
            + produced.source_replay_sample_mismatches
        )
        facts.derived_spectral_mismatches = (
            produced.source_replay_spectral_mismatches
        )
        facts.maximum_observed_bin_error = produced.maximum_bin_absolute_error
        facts.source_unreadable = produced.source_unreadable
        facts.spectral_rows_written = produced.spectral_rows
        facts.eligible_rate_windows = produced.derived_rate_windows_eligible
        facts.participant_summary_rows = produced.participant_summary_rows
        facts.participants_covered = produced.participants

    evidence: dict[str, Any] = {
        "artifact_kind": P04_ARTIFACT_KIND,
        "schema_version": P04_SCHEMA_VERSION,
        "implementation_version": P04_IMPLEMENTATION_VERSION,
        "contract_version": P04_CONTRACT_VERSION,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "authority": authority_block(),
        "p03_dependency": {
            **verification.as_record(),
            "pinned": pinned.as_record(),
        },
        "derived_rates_hz": list(DERIVED_RATES_HZ),
        "anti_alias_filters": filters,
        "resampling_controls": controls,
        "materialization": (
            produced.as_record() if produced is not None
            else {"materialized": False}
        ),
        "withheld_artifacts": dict(WITHHELD_P04_ARTIFACTS),
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
        schema_version=P04_SCHEMA_VERSION,
        contract_version=P04_CONTRACT_VERSION,
        implementation_version=P04_IMPLEMENTATION_VERSION,
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
    parser.add_argument("--p03-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dependency", required=True, type=Path)
    parser.add_argument("--p03-report", required=True, type=Path)
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
        record, receipt = audit_pads_p04(
            release_root=args.release_root,
            store_root=args.store_root,
            p03_root=args.p03_root,
            output_root=args.output_root,
            dependency_path=args.dependency,
            p03_report_path=args.p03_report,
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
    except (PadsP04AuditError, OSError, ValueError) as exc:
        print(f"{AUDIT_EXECUTION_ERROR}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    sys.stdout.buffer.write(payload)
    if not record.get("gate_evaluated"):
        return EXIT_BLOCKED
    return EXIT_PASS if record["gate_status"].startswith("PASS") else EXIT_NO_GO


if __name__ == "__main__":
    raise SystemExit(main())
