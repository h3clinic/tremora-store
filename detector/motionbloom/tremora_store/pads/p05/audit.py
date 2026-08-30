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
import dataclasses
import gc
import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ...release_gate import (
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
from .benchmark import (
    BenchmarkReport,
    ColdMeasurement,
    all_speed_ratios,
    measure_cold,
    run_rounds,
)
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
from .memory import check_memory
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


MEASUREMENT_FILENAME = "pads_p05_measurement.json"

#: The constructor's fields; ``as_record`` adds derived keys it cannot take.
_COLD_FIELDS = tuple(
    field.name for field in dataclasses.fields(ColdMeasurement)
)


def file_sha256(path: Path) -> str:
    """Hash a finished artifact, so the summarizer can prove it read the
    table the measurement process actually wrote."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> int:
    body = canonical_json_bytes(payload)
    path.write_bytes(body)
    return len(body)


def environment_record() -> dict[str, Any]:
    """Provenance only.  Never hashed into the canonical evidence."""

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 0,
        "thread_pinning": pin_single_thread(),
    }


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
    *, preflight, memory, inspected: Mapping[str, str | None]
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
        # The full record here, volume numbers included: this record exists
        # to say why the run did not happen, and nothing about it is hashed.
        "preflight": preflight.as_record(),
        "memory": memory.as_record(),
        "blocked_reason": (
            f"{preflight.status}: {preflight.detail}"
            if not preflight.ok
            else f"{memory.status}: {memory.detail}"
        ),
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
        "release_root": str(release_root),
        "store_root": str(store_root),
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


def measure_pads_p05(
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
    run_id: str | None = None,
    process_id: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Time the four representations and write the measurement receipt.

    This process does nothing after the timing except close the table, hash
    it and exit.  The summaries are another process's job.
    """

    pin_single_thread()
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
    # Disk is not enough on its own: the previous run was killed by the
    # memory manager, which had grown swap until the disk looked like the
    # problem.  A machine already paging will page harder under a long run.
    memory = check_memory()
    if not memory.ok:
        return _preflight_record(
            preflight=preflight,
            memory=memory,
            inspected={
                "baseline_root": str(baseline_root),
                "output_root": str(output_root),
                "store_root": str(store_root),
            },
        )
    if not preflight.ok:
        return _preflight_record(
            preflight=preflight,
            memory=memory,
            inspected={
                "baseline_root": str(baseline_root),
                "output_root": str(output_root),
                "store_root": str(store_root),
            },
        )

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

    # This is where the measurement process ends.  It closes the table, hashes
    # it, releases the representations and exits; the summaries are read back
    # by a separate process that was never resident during the timing.  A
    # three-hour run should not be carrying the read-back's allocation
    # alongside its own, and the previous architecture -- which did -- was
    # killed by the memory manager after the timing had already succeeded.
    for representation in representations.values():
        representation.release()
    representations.clear()
    gc.collect()

    measurement = {
        "artifact_kind": P05_ARTIFACT_KIND,
        "schema_version": P05_SCHEMA_VERSION,
        "implementation_version": P05_IMPLEMENTATION_VERSION,
        "contract_version": P05_CONTRACT_VERSION,
        "run_id": run_id,
        "process_id": process_id,
        "timing_table": {
            "path": str(table_path),
            "content_sha256": file_sha256(table_path),
            "bytes": table_path.stat().st_size,
            "rows": benchmark.rows_written,
        },
        "benchmark": benchmark.as_record(),
        "workload": workload.as_record(),
        "workload_content_sha256_before": workload_hash_before,
        "workload_content_sha256_after": workload_hash_after,
        "equivalence": equivalence.as_record(),
        "storage": storage,
        "hdf5": hdf5,
        "baseline_builds": {B1: b1_build, B2: b2_build},
        "baseline_identities": baseline_identities,
        "dependency": dependency,
        "preflight": preflight.deterministic_record(),
        "execution_receipt": {
            "volume_at_start": preflight.volume_record(),
            "memory_at_start": memory.as_record(),
            "environment": environment_record(),
            "command_arguments": list(command_arguments),
        },
    }
    write_json(output_root / MEASUREMENT_FILENAME, measurement)
    return measurement


def summarize_pads_p05(
    *,
    output_root: Path,
    store_root: Path,
    reproduction_receipt: Mapping[str, Any] | None = None,
    progress: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a finished timing table and produce the run's evidence.

    A fresh process.  Nothing here was resident while the timing ran, so the
    read-back's few hundred megabytes never coexist with the representations
    and the measured rounds.
    """

    measurement = json.loads(
        (output_root / MEASUREMENT_FILENAME).read_bytes().decode("utf-8")
    )
    table_path = Path(measurement["timing_table"]["path"])
    recorded = str(measurement["timing_table"]["content_sha256"])
    observed = file_sha256(table_path)
    if observed != recorded:
        raise PadsP05AuditError(
            f"timing table changed after measurement: {observed[:16]} "
            f"is not the recorded {recorded[:16]}"
        )

    run_id = str(measurement["run_id"])
    process_id = int(measurement["process_id"])
    benchmark = BenchmarkReport(
        rounds_completed=int(measurement["benchmark"]["rounds_completed"]),
        warmup_rounds_discarded=int(
            measurement["benchmark"]["warmup_rounds_discarded"]
        ),
        failed_queries=int(measurement["benchmark"]["failed_queries"]),
        rows_written=int(measurement["benchmark"]["rows_written"]),
        rows_offered=int(measurement["benchmark"]["rows_offered"]),
        round_orders=list(measurement["benchmark"]["round_orders"]),
        order_digest=str(measurement["benchmark"]["round_order_digest"]),
        rounds_by_query_class=dict(
            measurement["benchmark"]["measured_rounds_by_query_class"]
        ),
        table_path=str(table_path),
        cold=[
            ColdMeasurement(**{
                name: item[name]
                for name in _COLD_FIELDS if name in item
            })
            for item in measurement["benchmark"]["cold"]
        ],
    )
    equivalence = measurement["equivalence"]
    storage = measurement["storage"]
    hdf5 = measurement["hdf5"]
    b1_build = measurement["baseline_builds"][B1]
    b2_build = measurement["baseline_builds"][B2]
    dependency = measurement["dependency"]
    workload_hash_before = str(measurement["workload_content_sha256_before"])
    workload_hash_after = str(measurement["workload_content_sha256_after"])
    workload_record = measurement["workload"]
    query_counts = dict(workload_record["query_counts"])
    rounds = int(measurement["benchmark"]["rounds_completed"]) + int(
        measurement["benchmark"]["warmup_rounds_discarded"]
    )
    accounts = {
        str(row["representation"]): row for row in storage["accounts"]
    }
    threading = dict(
        measurement["execution_receipt"]["environment"]["thread_pinning"]
    )
    equivalence_failures = int(equivalence["failed_queries"])

    # Every published number is read back out of the table that was written,
    # so the report and the artifact cannot drift apart.
    if progress:
        print("summarizing", flush=True)
    latency = summarize_table(table_path)
    throughput = batch_throughput_table(table_path)
    ratios = all_speed_ratios(
        per_query_medians(table_path), system=SYSTEM_UNDER_TEST
    )

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
        query_counts.get(name, 0) * MEASURED_ROUNDS_BY_QUERY_CLASS.get(
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
        windows_reconciled=equivalence["windows_compared"],
        expected_windows=query_counts.get(Q2, 0),
        streams_reconciled=equivalence["streams_compared"],
        assessments_reconciled=equivalence["assessments_compared"],
        per_representation_mismatches={
            name: sum(
                counts[key] for key in (
                    "content_mismatches", "row_count_mismatches",
                    "time_mismatches", "sensor_value_mismatches",
                    "failed_queries",
                )
            )
            for name, counts in
            equivalence["per_representation"].items()
        },
        content_mismatches=equivalence["content_mismatches"],
        row_count_mismatches=equivalence["row_count_mismatches"],
        time_mismatches=equivalence["time_mismatches"],
        sensor_value_mismatches=equivalence["sensor_value_mismatches"],
        b1_stored_instances=int(b1_build["stored_sample_instances"]),
        b1_unique_samples=int(b1_build["unique_samples"]),
        m1_stored_instances=int(accounts[M1]["stored_sample_instances"]),
        m1_unique_samples=int(accounts[M1]["unique_samples"]),
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
        failed_queries=benchmark.failed_queries + equivalence_failures,
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
        "workload": workload_record,
        "preflight": measurement["preflight"],
        "equivalence": equivalence,
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
        dataset_root=Path(dependency["release_root"]),
        source_manifest_sha256=str(
            dependency.get("source_manifest_sha256") or ""
        ),
        output_root=output_root,
        schema_version=P05_SCHEMA_VERSION,
        contract_version=P05_CONTRACT_VERSION,
        implementation_version=P05_IMPLEMENTATION_VERSION,
        canonical_evidence_sha256=evidence_sha256,
        command_arguments=tuple(
            measurement["execution_receipt"]["command_arguments"]
        ),
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
        # Provenance, not experimental content: how much room and memory the
        # machine had is a fact about that machine at that moment, and two
        # honest runs will disagree about it.
        "execution_receipt": measurement["execution_receipt"],
        "timing_table": measurement["timing_table"],
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


def _measure_arguments(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Time the four PADS representations and write a measurement "
            "receipt.  Summaries are a separate process."
        )
    )
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--p02-report", required=True, type=Path)
    parser.add_argument("--p03-report", required=True, type=Path)
    parser.add_argument("--p04-report", required=True, type=Path)
    parser.add_argument("--p04-store-root", type=Path)
    parser.add_argument("--rounds", type=int, default=TOTAL_ROUNDS)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--process-id", type=int, default=os.getpid())
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def measure_main(argv: Sequence[str] | None = None) -> int:
    args = _measure_arguments(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    record = measure_pads_p05(
        release_root=args.release_root,
        store_root=args.store_root,
        baseline_root=args.baseline_root,
        output_root=args.output_root,
        p02_report_path=args.p02_report,
        p03_report_path=args.p03_report,
        p04_report_path=args.p04_report,
        p04_store_root=args.p04_store_root,
        rounds=args.rounds,
        run_id=args.run_id or f"p05-{args.process_id}",
        process_id=args.process_id,
        command_arguments=tuple(argv or sys.argv[1:]),
        progress=args.progress,
    )
    if record.get("release_status") == ERROR_RESOURCE_PREFLIGHT:
        write_json(
            args.output_root / EVIDENCE_FILENAME, record
        )
        sys.stdout.buffer.write(canonical_json_bytes(record))
        return EXIT_PREFLIGHT
    if record.get("release_status") == BLOCKED_DEPENDENCY:
        write_json(args.output_root / EVIDENCE_FILENAME, record)
        sys.stdout.buffer.write(canonical_json_bytes(record))
        return EXIT_BLOCKED
    sys.stdout.buffer.write(canonical_json_bytes({
        "timing_table": record["timing_table"],
        "rows": record["benchmark"]["rows_written"],
    }))
    sys.stdout.write("\n")
    return EXIT_PASS


def _summarize_arguments(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Read a finished timing table and produce the run's evidence."
        )
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--reproduction-receipt", type=Path)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def summarize_main(argv: Sequence[str] | None = None) -> int:
    args = _summarize_arguments(argv)
    reproduction = None
    if args.reproduction_receipt and args.reproduction_receipt.is_file():
        reproduction = json.loads(
            args.reproduction_receipt.read_bytes().decode("utf-8")
        )
    record, receipt = summarize_pads_p05(
        output_root=args.output_root,
        store_root=args.store_root,
        reproduction_receipt=reproduction,
        progress=args.progress,
    )
    payload = canonical_json_bytes(record)
    (args.output_root / EVIDENCE_FILENAME).write_bytes(payload)
    if receipt is not None:
        (args.output_root / RECEIPT_FILENAME).write_bytes(
            canonical_json_bytes(receipt)
        )
    sys.stdout.buffer.write(payload)
    if not record.get("gate_evaluated"):
        return EXIT_BLOCKED
    return EXIT_PASS if record.get("gate_satisfied") else EXIT_NO_GO


