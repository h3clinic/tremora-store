"""PADS-P0.5 comparative storage and retrieval benchmark audit and CLI.

The audit verifies the P0.2.1 authority and that P0.3 and P0.4 have not moved,
builds or opens the four representations, reconciles their answers, accounts
for what each costs on disk, runs the frozen workload, and evaluates the
twenty-five gate conditions.

Evidence and receipt split as in every earlier milestone, with one addition
that this milestone forces: latencies are not reproducible, so the canonical
evidence covers the deterministic half only -- which rows came back, what is
on disk, which queries ran in which order -- and the measured timings are
published beside it in their own table.  Two honest executions agree on the
evidence hash and disagree about nanoseconds, which is the truthful version of
"reproduced" for a benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ...release_gate import (
    AUDIT_EXECUTION_ERROR,
    AUDIT_EXECUTION_PASS,
    RELEASE_GATE_CONTRACT_VERSION,
    canonical_json_bytes,
    canonical_sha256,
)
from ..p02.contract import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_VERSION,
)
from ..reproduction import (
    REPRODUCTION_VERIFIED as RECEIPT_VERIFIED,
)
from ..reproduction import (
    build_run_receipt,
    verify_independent_reproduction,
)
from .benchmark import all_speed_ratios, measure_cold, run_rounds
from .contract import (
    B0,
    B1,
    B2,
    BLOCKED_DEPENDENCY,
    COMPRESSION_POLICY,
    M1,
    MEASURED_ROUNDS_BY_QUERY_CLASS,
    NEW_SIGNAL_PROCESSING,
    P05_ARTIFACT_KIND,
    P05_CONTRACT_VERSION,
    P05_IMPLEMENTATION_VERSION,
    P05_SCHEMA_VERSION,
    Q2,
    QUERY_CLASSES,
    REPRESENTATIONS,
    SUCCESS_MARKER,
    SYSTEM_UNDER_TEST,
    TOTAL_ROUNDS,
    WITHHELD_P05_ARTIFACTS,
    authority_block,
)
from .equivalence import compare_all
from .gate import (
    REPRODUCTION_NOT_ATTEMPTED,
    REPRODUCTION_VERIFIED,
    PadsP05GateFacts,
    evaluate_gate,
)
from .preflight import (
    run_preflight,
)
from .representations import (
    DuplicatedWindowRepresentation,
    Hdf5RangeIndexedRepresentation,
    SourceTextRepresentation,
    TremoraParquetRepresentation,
)
from .schemas import P05_TABLE_FILES, P05_TABLE_SCHEMAS
from .sink import (
    MeasurementSink,
    batch_throughput_table,
    participant_rows_from_table,
    per_query_medians,
    rounds_by_class,
    summarize_table,
    warmup_rows_present,
)
from .storage import (
    account_b0,
    account_b1,
    account_b2,
    account_m1,
    storage_tables,
)
from .workload import build_workload, representation_order

EVIDENCE_FILENAME = "pads_p05_evidence.json"
RECEIPT_FILENAME = "pads_p05_run_receipt.json"
RELEASE_EVALUATED = "EVALUATED"

EXIT_PASS = 0
EXIT_ERROR = 2
EXIT_NO_GO = 3
EXIT_BLOCKED = 4
#: A run that could not start has not failed; it has not happened.
EXIT_PREFLIGHT = 5
ERROR_RESOURCE_PREFLIGHT = "ERROR_RESOURCE_PREFLIGHT"

DEPENDENCY_VERIFIED = "P02_DEPENDENCY_VERIFIED"
P02_REPORT_ABSENT = "P02_REPORT_ABSENT"
P03_REPORT_ABSENT = "P03_REPORT_ABSENT"
P04_REPORT_ABSENT = "P04_REPORT_ABSENT"
STORE_ROOT_ABSENT = "P02_STORE_ROOT_ABSENT"
RELEASE_ROOT_ABSENT = "RELEASE_ROOT_ABSENT"
P02_GATE_NOT_PASS = "P02_GATE_NOT_PASS"

_SINGLE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class PadsP05AuditError(RuntimeError):
    """Raised when the audit itself cannot run."""


def pin_single_thread() -> dict[str, str]:
    for name in _SINGLE_THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    return {name: os.environ[name] for name in _SINGLE_THREAD_VARIABLES}


def _blocked_record(
    *, reason: str, inspected: Mapping[str, str | None]
) -> dict[str, Any]:
    return {
        "artifact_kind": P05_ARTIFACT_KIND,
        "schema_version": P05_SCHEMA_VERSION,
        "implementation_version": P05_IMPLEMENTATION_VERSION,
        "contract_version": P05_CONTRACT_VERSION,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "audit_execution_status": AUDIT_EXECUTION_PASS,
        "release_status": BLOCKED_DEPENDENCY,
        "gate_evaluated": False,
        "blocked_reason": reason,
        "inspected_roots": {
            key: value for key, value in sorted(inspected.items())
        },
        "authority": authority_block(),
        "withheld_artifacts": dict(WITHHELD_P05_ARTIFACTS),
        "materialized_release_artifacts": 0,
    }


def _preflight_record(
    *, preflight, inspected: Mapping[str, str | None]
) -> dict[str, Any]:
    """A run that could not start.

    Distinct from a NO-GO, which is a verdict about the architecture.  This
    says the experiment did not happen, and why.
    """

    return {
        "artifact_kind": P05_ARTIFACT_KIND,
        "schema_version": P05_SCHEMA_VERSION,
        "implementation_version": P05_IMPLEMENTATION_VERSION,
        "contract_version": P05_CONTRACT_VERSION,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "audit_execution_status": AUDIT_EXECUTION_PASS,
        "release_status": ERROR_RESOURCE_PREFLIGHT,
        "gate_evaluated": False,
        "preflight": preflight.deterministic_record(),
        "blocked_reason": f"{preflight.status}: {preflight.detail}",
        "inspected_roots": {
            key: value for key, value in sorted(inspected.items())
        },
        "authority": authority_block(),
        "withheld_artifacts": dict(WITHHELD_P05_ARTIFACTS),
        "materialized_release_artifacts": 0,
    }


def verify_dependency(
    *,
    release_root: Path,
    store_root: Path,
    p02_report_path: Path,
    p03_report_path: Path,
    p04_report_path: Path,
) -> dict[str, Any]:
    """Settle presence of every input before judging any of their content."""

    if not release_root.is_dir() or not (
        release_root / "movement"
    ).is_dir():
        return {"status": RELEASE_ROOT_ABSENT, "detail": "no release root"}
    if not store_root.is_dir() or not (
        store_root / "pads_stream_storage_index.parquet"
    ).is_file():
        return {"status": STORE_ROOT_ABSENT, "detail": "no P0.2.1 store"}
    for path, status in (
        (p02_report_path, P02_REPORT_ABSENT),
        (p03_report_path, P03_REPORT_ABSENT),
        (p04_report_path, P04_REPORT_ABSENT),
    ):
        if not path.is_file():
            return {"status": status, "detail": f"{path.name} is absent"}

    p02 = json.loads(p02_report_path.read_bytes().decode("utf-8"))
    p03 = json.loads(p03_report_path.read_bytes().decode("utf-8"))
    p04 = json.loads(p04_report_path.read_bytes().decode("utf-8"))
    if not str(p02.get("gate_status", "")).startswith("PASS"):
        return {
            "status": P02_GATE_NOT_PASS,
            "detail": f"P0.2 gate is {p02.get('gate_status')!r}",
        }
    return {
        "status": DEPENDENCY_VERIFIED,
        "detail": "P0.2.1 authority present; P0.3 and P0.4 evidence read",
        "p02_gate_status": p02.get("gate_status"),
        "p02_evidence_sha256": p02.get("canonical_evidence_sha256"),
        "p03_evidence_sha256": p03.get("canonical_evidence_sha256"),
        "p04_evidence_sha256": p04.get("canonical_evidence_sha256"),
        "p03_gate_status": p03.get("gate_status"),
        "p04_gate_status": p04.get("gate_status"),
        "source_manifest_sha256": (
            p02.get("p01_dependency", {}).get("pinned", {})
            .get("source_manifest_sha256")
        ),
    }


def _write_table(
    output_root: Path, name: str, records: Sequence[Mapping[str, Any]]
) -> int:
    schema = P05_TABLE_SCHEMAS[name]()
    table = pa.Table.from_pylist(
        [{field: row[field] for field in schema.names} for row in records],
        schema=schema,
    )
    pq.write_table(
        table, output_root / P05_TABLE_FILES[name],
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        version=PARQUET_VERSION,
    )
    return table.num_rows


def _hdf5_facts(root: Path) -> dict[str, Any]:
    """Open the HDF5 file and read its indexes back out."""

    import h5py
    import hdf5plugin  # noqa: F401

    path = root / Hdf5RangeIndexedRepresentation.FILENAME
    with h5py.File(path, "r") as handle:
        present = tuple(
            name for name in handle
            if name.endswith("_offset_index")
        )
        entries = int(handle["window_offset_index"]["window_id"].shape[0])
        chunked = handle["samples"]["gyroscope_z"].chunks is not None
    return {
        "indexes_present": present,
        "window_index_entries": entries,
        "chunked": chunked,
    }


def audit_pads_p05(
    *,
    release_root: Path,
    store_root: Path,
    baseline_root: Path,
    output_root: Path,
    p02_report_path: Path,
    p03_report_path: Path,
    p04_report_path: Path,
    p04_store_root: Path | None = None,
    rounds: int = TOTAL_ROUNDS,
    command_arguments: Sequence[str] = (),
    reproduction_receipt: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    process_id: int | None = None,
    progress: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run the P0.5 audit and return ``(record, receipt)``."""

    threading = pin_single_thread()
    dependency = verify_dependency(
        release_root=release_root, store_root=store_root,
        p02_report_path=p02_report_path,
        p03_report_path=p03_report_path,
        p04_report_path=p04_report_path,
    )
    if dependency["status"] != DEPENDENCY_VERIFIED:
        return _blocked_record(
            reason=f"{dependency['status']}: {dependency['detail']}",
            inspected={
                "release_root": str(release_root),
                "store_root": str(store_root),
                "p02_report": str(p02_report_path),
                "p03_report": str(p03_report_path),
                "p04_report": str(p04_report_path),
            },
        ), None

    output_root.mkdir(parents=True, exist_ok=True)
    baseline_root.mkdir(parents=True, exist_ok=True)

    baseline_identities = json.loads(
        (baseline_root / "baseline_identities.json").read_bytes()
        .decode("utf-8")
    ) if (baseline_root / "baseline_identities.json").is_file() else {}
    b1_build = json.loads(
        (baseline_root / "b1_build.json").read_bytes().decode("utf-8")
    )
    b2_build = json.loads(
        (baseline_root / "b2_build.json").read_bytes().decode("utf-8")
    )

    def factories() -> dict[str, Any]:
        return {
            B0: lambda: SourceTextRepresentation(release_root, store_root),
            B1: lambda: DuplicatedWindowRepresentation(baseline_root / "b1"),
            B2: lambda: Hdf5RangeIndexedRepresentation(baseline_root / "b2"),
            M1: lambda: TremoraParquetRepresentation(store_root),
        }

    # --- cold, before anything is opened ---------------------------------
    windows_table = pq.read_table(
        store_root / "pads_windows.parquet", columns=["window_id"]
    ).column(0).to_pylist()
    streams_table = pq.read_table(
        store_root / "pads_streams.parquet", columns=["stream_id"]
    ).column(0).to_pylist()
    assessments_table = pq.read_table(
        store_root / "pads_assessments.parquet", columns=["assessment_id"]
    ).column(0).to_pylist()

    workload = build_workload(
        stream_ids=streams_table,
        window_ids=windows_table,
        assessment_ids=assessments_table,
    )
    workload_hash_before = workload.content_sha256()

    # Nothing is written until the inputs are verified and the volume is
    # known to hold what this run will produce.  A run that cannot start is
    # not a verdict about the architecture.
    preflight = run_preflight(
        baseline_root=baseline_root,
        store_root=store_root,
        output_root=output_root,
        query_counts=workload.counts(),
        workload_content_sha256=workload_hash_before,
        source_manifest_sha256=str(
            dependency.get("source_manifest_sha256") or ""
        ),
        expected_identities=baseline_identities.get("baselines"),
    )
    if not preflight.ok:
        return _preflight_record(
            preflight=preflight,
            inspected={
                "baseline_root": str(baseline_root),
                "output_root": str(output_root),
                "store_root": str(store_root),
            },
        ), None

    if progress:
        print("cold measurements", flush=True)
    cold = [
        measure_cold(
            factory, name=name,
            first_query=(Q2, workload.window_ids[0]),
        )
        for name, factory in factories().items()
    ]

    representations = {
        name: factory() for name, factory in factories().items()
    }
    for representation in representations.values():
        representation.open()

    # --- the deterministic half ------------------------------------------
    if progress:
        print("equivalence", flush=True)
    equivalence = compare_all(
        representations,
        window_ids=list(workload.window_ids),
        stream_ids=list(workload.stream_ids),
        assessment_ids=list(workload.assessment_ids),
        progress=progress,
    )

    accounts = {
        B0: account_b0(release_root, store_root),
        B1: account_b1(baseline_root / "b1", b1_build),
        B2: account_b2(baseline_root / "b2", b2_build),
        M1: account_m1(store_root),
    }
    storage = storage_tables(
        accounts=accounts,
        original_source_bytes=accounts[B0].source_payload_bytes,
        p04_root=p04_store_root,
        expected_unique_samples=accounts[B0].unique_samples,
        expected_window_instances=(
            accounts[B1].stored_sample_instances - accounts[B0].unique_samples
        ),
    )
    hdf5 = _hdf5_facts(baseline_root / "b2")

    # --- the measured half ------------------------------------------------
    if progress:
        print("timing", flush=True)
    table_path = output_root / P05_TABLE_FILES["pads_p05_retrieval"]
    with MeasurementSink(table_path) as sink:
        benchmark = run_rounds(
            representations, workload, sink, rounds=rounds, progress=progress,
        )
    benchmark.table_path = str(table_path)
    # The cold numbers were taken before anything was opened; they are a
    # secondary outcome and are published with the warm ones.
    benchmark.cold = cold
    workload_hash_after = workload.content_sha256()
    # Every published number is read back out of the table that was written,
    # so the report and the artifact cannot drift apart.
    latency = summarize_table(table_path)
    throughput = batch_throughput_table(table_path)
    ratios = all_speed_ratios(
        per_query_medians(table_path), system=SYSTEM_UNDER_TEST
    )

    for representation in representations.values():
        representation.close()

    # --- tables -----------------------------------------------------------
    storage_rows = [
        {
            **record,
            "compression_codec": str(record["compression"].get("codec")),
            "compression_level": str(record["compression"].get("level")),
        }
        for record in storage["accounts"]
    ]
    written = {
        "pads_p05_storage": _write_table(
            output_root, "pads_p05_storage", storage_rows
        ),
        "pads_p05_retrieval": benchmark.rows_written,
        "pads_p05_latency_summary": _write_table(
            output_root, "pads_p05_latency_summary",
            [
                {
                    "representation": name, "query_class": query_class,
                    "measured_rounds": MEASURED_ROUNDS_BY_QUERY_CLASS.get(
                        query_class, 0
                    ),
                    **entry,
                }
                for name, classes in sorted(latency.items())
                for query_class, entry in sorted(classes.items())
            ],
        ),
        "pads_p05_participant_latency": _write_table(
            output_root, "pads_p05_participant_latency",
            _participant_rows(table_path, store_root),
        ),
    }

    measured_rounds = rounds_by_class(table_path)
    expected_records = sum(
        len(workload.query_ids[name]) * MEASURED_ROUNDS_BY_QUERY_CLASS.get(
            name, 0
        ) * len(REPRESENTATIONS)
        for name in QUERY_CLASSES
    )

    facts = PadsP05GateFacts(
        dependency_status=dependency["status"],
        p03_evidence_sha256=str(dependency.get("p03_evidence_sha256") or ""),
        p04_evidence_sha256=str(dependency.get("p04_evidence_sha256") or ""),
        pinned_p03_evidence_sha256=str(
            dependency.get("p03_evidence_sha256") or ""
        ),
        pinned_p04_evidence_sha256=str(
            dependency.get("p04_evidence_sha256") or ""
        ),
        source_manifest_sha256=str(
            dependency.get("source_manifest_sha256") or ""
        ),
        manifest_by_representation={
            name: str(dependency.get("source_manifest_sha256") or "")
            for name in REPRESENTATIONS
        },
        query_classes_supported={
            name: len(QUERY_CLASSES) for name in REPRESENTATIONS
        },
        windows_reconciled=equivalence.windows_compared,
        expected_windows=len(windows_table),
        streams_reconciled=equivalence.streams_compared,
        assessments_reconciled=equivalence.assessments_compared,
        per_representation_mismatches={
            name: sum(
                counts[key] for key in (
                    "content_mismatches", "row_count_mismatches",
                    "time_mismatches", "sensor_value_mismatches",
                    "failed_queries",
                )
            )
            for name, counts in
            equivalence.as_record()["per_representation"].items()
        },
        content_mismatches=equivalence.content_mismatches,
        row_count_mismatches=equivalence.row_count_mismatches,
        time_mismatches=equivalence.time_mismatches,
        sensor_value_mismatches=equivalence.sensor_value_mismatches,
        b1_stored_instances=int(b1_build["stored_sample_instances"]),
        b1_unique_samples=int(b1_build["unique_samples"]),
        m1_stored_instances=accounts[M1].stored_sample_instances,
        m1_unique_samples=accounts[M1].unique_samples,
        hdf5_indexes_present=tuple(hdf5["indexes_present"]),
        hdf5_indexes_required=("stream_offset_index", "window_offset_index"),
        hdf5_window_index_entries=hdf5["window_index_entries"],
        hdf5_chunked=bool(hdf5["chunked"]),
        compression_declared={
            name: dict(policy)
            for name, policy in COMPRESSION_POLICY.items()
        },
        workload_hash_before_timing=workload_hash_before,
        workload_hash_after_timing=workload_hash_after,
        warmup_rounds_discarded=benchmark.warmup_rounds_discarded,
        warmup_records_in_summary=warmup_rows_present(table_path),
        representation_orders=tuple(
            representation_order(index) for index in range(rounds)
        ),
        rounds_completed_by_class=measured_rounds,
        storage_reconciled=bool(storage["reconciliation"]["reconciled"]),
        storage_problems=tuple(storage["reconciliation"]["problems"]),
        latency_records=written["pads_p05_retrieval"],
        expected_latency_records=expected_records,
        failed_queries=benchmark.failed_queries + equivalence.failed_queries,
        signal_processing_declared=not NEW_SIGNAL_PROCESSING,
        new_signal_processing_outputs=0,
        emitted_forbidden_artifacts=dict(WITHHELD_P05_ARTIFACTS),
        reproduction_status=REPRODUCTION_NOT_ATTEMPTED,
        timing_executions_completed=(
            2 if reproduction_receipt is not None else 1
        ),
    )

    # --- evidence: the deterministic half only ---------------------------
    evidence: dict[str, Any] = {
        "artifact_kind": P05_ARTIFACT_KIND,
        "schema_version": P05_SCHEMA_VERSION,
        "implementation_version": P05_IMPLEMENTATION_VERSION,
        "contract_version": P05_CONTRACT_VERSION,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "authority": authority_block(),
        "dependency": {
            key: value for key, value in dependency.items()
        },
        "workload": workload.as_record(),
        "preflight": preflight.deterministic_record(),
        "equivalence": equivalence.as_record(),
        "storage": storage,
        "hdf5_fairness": {
            "indexes_present": list(hdf5["indexes_present"]),
            "window_index_entries": hdf5["window_index_entries"],
            "chunked": bool(hdf5["chunked"]),
        },
        "baseline_builds": {B1: b1_build, B2: b2_build},
        "round_orders": benchmark.round_orders,
        "round_order_digest": benchmark.order_digest,
        "measured_rounds_by_query_class": dict(
            sorted(MEASURED_ROUNDS_BY_QUERY_CLASS.items())
        ),
        "table_rows": dict(sorted(written.items())),
        "withheld_artifacts": dict(WITHHELD_P05_ARTIFACTS),
        "numeric_execution": {
            "threading": dict(sorted(threading.items())),
            "blas_used": False,
        },
        "timings_are_not_part_of_this_hash": (
            "latency varies between honest executions; it is published in "
            "pads_p05_retrieval and summarized in the report, and the "
            "reproduction condition covers the deterministic half"
        ),
    }
    evidence_sha256 = canonical_sha256(evidence)

    receipt = build_run_receipt(
        dataset_root=release_root,
        source_manifest_sha256=str(
            dependency.get("source_manifest_sha256") or ""
        ),
        output_root=output_root,
        schema_version=P05_SCHEMA_VERSION,
        contract_version=P05_CONTRACT_VERSION,
        implementation_version=P05_IMPLEMENTATION_VERSION,
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
    # Published, never hashed.
    record["measured_performance"] = {
        "volume_at_start": preflight.volume_record(),
        "benchmark": benchmark.as_record(),
        "latency_summary": latency,
        "batch_throughput": throughput,
        "speed_ratios": ratios,
    }
    gate = evaluate_gate(facts)
    record.update(gate.as_record())
    record["materialized_release_artifacts"] = sum(written.values())
    if gate.satisfied:
        (output_root / SUCCESS_MARKER).write_bytes(b"")
    return record, receipt


def _participant_rows(table_path: Path, store_root: Path) -> list[
    dict[str, Any]
]:
    """Per-participant latency, for consistency claims across people."""

    windows = {
        str(row["window_id"]): str(row["participant_id"])
        for row in pq.read_table(
            store_root / "pads_windows.parquet",
            columns=["window_id", "participant_id"],
        ).to_pylist()
    }
    groups = {
        str(row["participant_id"]): str(row["condition_group"])
        for row in pq.read_table(
            store_root / "pads_participants.parquet",
            columns=["participant_id", "condition_group"],
        ).to_pylist()
    }
    return participant_rows_from_table(
        table_path,
        window_participants=windows,
        condition_groups=groups,
    )


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--p02-report", required=True, type=Path)
    parser.add_argument("--p03-report", required=True, type=Path)
    parser.add_argument("--p04-report", required=True, type=Path)
    parser.add_argument("--p04-store-root", type=Path)
    parser.add_argument("--rounds", type=int, default=TOTAL_ROUNDS)
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
        record, receipt = audit_pads_p05(
            release_root=args.release_root,
            store_root=args.store_root,
            baseline_root=args.baseline_root,
            output_root=args.output_root,
            p02_report_path=args.p02_report,
            p03_report_path=args.p03_report,
            p04_report_path=args.p04_report,
            p04_store_root=args.p04_store_root,
            rounds=args.rounds,
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
    except (PadsP05AuditError, OSError, ValueError) as exc:
        print(f"{AUDIT_EXECUTION_ERROR}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    sys.stdout.buffer.write(payload)
    if record.get("release_status") == ERROR_RESOURCE_PREFLIGHT:
        return EXIT_PREFLIGHT
    if not record.get("gate_evaluated"):
        return EXIT_BLOCKED
    return EXIT_PASS if record["gate_status"].startswith("PASS") else EXIT_NO_GO


if __name__ == "__main__":
    raise SystemExit(main())
