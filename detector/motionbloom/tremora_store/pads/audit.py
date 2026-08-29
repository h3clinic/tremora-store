"""PADS-P0.1 ingest audit engine and CLI.

The audit reads the PhysioNet release as published, verifies every referenced
file against the release's own ``SHA256SUMS.txt``, reconciles the whole
structure, and parses each device file against its own declaration.  It
produces no window, no spectral feature and no video association.

Output is two layers.  The evidence record contains only source-derived
canonical content, so two genuine executions produce byte-identical evidence.
The run receipt carries everything that differs between executions, which is
what makes the reproduction check mean something.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..release_gate import (
    AUDIT_EXECUTION_ERROR,
    AUDIT_EXECUTION_PASS,
    RELEASE_GATE_CONTRACT_VERSION,
    WITHHELD_P01_ARTIFACTS,
    blocked_record,
    canonical_json_bytes,
    canonical_sha256,
    exit_code_for,
)
from ..timing_authority import PADS_BINDING
from .authority import (
    PADS_ARTIFACT_KIND,
    PADS_CONTRACT_VERSION,
    PADS_IMPLEMENTATION_VERSION,
    PADS_PARSER_VERSION,
    PADS_SCHEMA_VERSION,
    RELATIVE_TIME_BASIS,
    assert_no_paired_claim,
    authority_contract,
)
from .gate import PadsGateFacts, evaluate_gate
from .movement import (
    AMBIGUITY_FAILURES,
    BLANK_SOURCE_ROW,
    CADENCE_DEVIATES_FROM_DECLARED_RATE,
    INVALID_TIME,
    NO_USABLE_VALUES,
    ROW_COLUMN_COUNT_MISMATCH,
    ROW_COUNT_MISMATCH,
    SPAN_DEVIATES_FROM_DECLARED_RATE,
    STREAM_PARSED,
    UNRECOGNIZED_DEVICE_LOCATION,
    Observation,
    PadsSourceError,
    Participant,
    expected_durations,
    parse_observation,
    parse_patient,
    parse_timeseries,
)
from .release_structure import (
    PADS_EXPECTED_PARTICIPANTS,
    PADS_RELEASE_CONTRACT_VERSION,
    PADS_RELEASE_VERSION,
    reconcile_release_structure,
)
from .reproduction import (
    build_run_receipt,
    verify_independent_reproduction,
)
from .schemas import PADS_TABLE_SCHEMAS

MOVEMENT_DIRECTORY = "movement"
PATIENTS_DIRECTORY = "patients"
CHECKSUM_FILENAME = "SHA256SUMS.txt"

RELEASE_EVALUATED = "EVALUATED"
CHECKSUMS_ABSENT = "CHECKSUMS_ABSENT"


class PadsAuditError(RuntimeError):
    """Raised when the audit itself cannot run."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    """Read the release's own ``SHA256SUMS.txt`` as the source manifest."""

    manifest: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition(" ")
        relative = relative.strip()
        if len(digest) != 64 or not relative:
            raise PadsAuditError(f"malformed checksum line: {line!r}")
        manifest[relative] = digest
    return manifest


def _video_bearing_field_count() -> int:
    """Prove the emitted schemas carry no cross-modal field."""

    count = 0
    for name, factory in PADS_TABLE_SCHEMAS.items():
        try:
            assert_no_paired_claim([name, *factory().names])
        except Exception:  # noqa: BLE001 - counted, not raised
            count += 1
    return count


def audit_pads_p01(
    *,
    dataset_root: Path,
    output_root: Path,
    command_arguments: Sequence[str] = (),
    reproduction_receipt: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    process_id: int | None = None,
    expected_participants: int = PADS_EXPECTED_PARTICIPANTS,
    progress: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run the PADS-P0.1 audit and return ``(record, receipt)``."""

    movement_root = dataset_root / MOVEMENT_DIRECTORY
    patients_root = dataset_root / PATIENTS_DIRECTORY
    if not dataset_root.is_dir() or not movement_root.is_dir():
        return blocked_record(
            binding=PADS_BINDING,
            artifact_kind=PADS_ARTIFACT_KIND,
            schema_version=PADS_SCHEMA_VERSION,
            implementation_version=PADS_IMPLEMENTATION_VERSION,
            reason="PADS movement records are not present under the root",
            inspected_roots={
                "dataset_root": str(dataset_root),
                "movement_root": str(movement_root),
                "patients_root": str(patients_root),
            },
        ), None

    failures: list[str] = []
    checksum_path = dataset_root / CHECKSUM_FILENAME
    checksums: dict[str, str] = {}
    source_manifest_sha256 = CHECKSUMS_ABSENT
    if checksum_path.is_file():
        payload = checksum_path.read_bytes()
        source_manifest_sha256 = _sha256_bytes(payload)
        try:
            checksums = read_checksums(checksum_path)
        except (PadsAuditError, OSError, UnicodeDecodeError) as exc:
            failures.append(f"checksums: {exc}")
    else:
        failures.append("release checksum list is absent")

    facts = PadsGateFacts(relative_time_basis=RELATIVE_TIME_BASIS)

    def verify_against_checksums(relative: str, payload: bytes) -> None:
        """Every referenced source file, not only the timeseries ones."""

        facts.source_files_expected += 1
        expected = checksums.get(relative)
        if expected is None or expected != _sha256_bytes(payload):
            facts.source_files_failed += 1
            failures.append(f"{relative}: not verified against the release")
        else:
            facts.source_files_hash_verified += 1

    observations: dict[str, Observation] = {}
    for path in sorted(movement_root.glob("observation_*.json")):
        relative = f"{MOVEMENT_DIRECTORY}/{path.name}"
        try:
            payload = path.read_bytes()
            verify_against_checksums(relative, payload)
            document = json.loads(payload.decode("utf-8"))
            observation = parse_observation(
                document, source_relative_path=relative
            )
        except (OSError, ValueError, PadsSourceError) as exc:
            failures.append(f"{relative}: {exc}")
            continue
        observations[observation.subject_id] = observation

    participants: dict[str, Participant] = {}
    if patients_root.is_dir():
        for path in sorted(patients_root.glob("patient_*.json")):
            relative = f"{PATIENTS_DIRECTORY}/{path.name}"
            try:
                payload = path.read_bytes()
                verify_against_checksums(relative, payload)
                document = json.loads(payload.decode("utf-8"))
                participant = parse_patient(
                    document, source_relative_path=relative
                )
            except (OSError, ValueError, PadsSourceError) as exc:
                failures.append(f"{relative}: {exc}")
                continue
            participants[participant.participant_id] = participant
    else:
        failures.append("patient records are absent")

    def file_exists(file_name: str) -> bool:
        return (movement_root / file_name).is_file()

    structure = reconcile_release_structure(
        observations,
        participants,
        file_exists=file_exists,
        expected_participants=expected_participants,
    )

    facts.release_structure_status = structure.status
    facts.video_bearing_field_count = _video_bearing_field_count()
    facts.emitted_forbidden_artifacts = dict(WITHHELD_P01_ARTIFACTS)

    per_task: dict[str, dict[str, int]] = {}
    stream_failures: list[dict[str, str]] = []
    total_samples = 0
    total_usable_values = 0
    duplicate_time_total = 0
    nonmonotonic_time_total = 0
    noncanonical_order_streams = 0

    processed = 0
    for participant_id in sorted(observations):
        observation = observations[participant_id]
        for session in observation.sessions:
            support, span = expected_durations(
                session.rows, observation.sampling_rate
            )
            task_counts = per_task.setdefault(session.record_name, {
                "streams": 0,
                "parsed": 0,
                "refused": 0,
                "samples": 0,
            })
            for declaration in session.streams:
                facts.streams_declared += 1
                task_counts["streams"] += 1
                relative = (
                    f"{MOVEMENT_DIRECTORY}/{declaration.file_name}"
                )
                path = movement_root / declaration.file_name
                subject = (
                    f"{participant_id}:{session.record_name}:"
                    f"{declaration.device_location}"
                )
                if not path.is_file():
                    facts.streams_refused += 1
                    task_counts["refused"] += 1
                    stream_failures.append({
                        "subject": subject, "status": "FILE_MISSING",
                    })
                    continue
                payload = path.read_bytes()
                observed = _sha256_bytes(payload)
                facts.source_files_expected += 1
                expected = checksums.get(relative)
                if expected is None:
                    stream_failures.append({
                        "subject": subject, "status": "NOT_IN_CHECKSUMS",
                    })
                    facts.source_files_failed += 1
                elif observed != expected:
                    stream_failures.append({
                        "subject": subject, "status": "HASH_MISMATCH",
                    })
                    facts.source_files_failed += 1
                else:
                    facts.source_files_hash_verified += 1

                stream = parse_timeseries(
                    payload,
                    declaration=declaration,
                    declared_rows=session.rows,
                    sampling_rate=observation.sampling_rate,
                )
                processed += 1
                if progress and processed % 1000 == 0:
                    print(
                        f"parsed {processed} streams", file=sys.stderr,
                        flush=True,
                    )
                if stream.stream_status == STREAM_PARSED:
                    facts.streams_parsed += 1
                    task_counts["parsed"] += 1
                    task_counts["samples"] += stream.parsed_row_count
                    total_samples += stream.parsed_row_count
                    total_usable_values += stream.usable_value_count
                    duplicate_time_total += stream.duplicate_time_count
                    nonmonotonic_time_total += stream.nonmonotonic_time_count
                    if CADENCE_DEVIATES_FROM_DECLARED_RATE in (
                        stream.issue_codes
                    ):
                        facts.cadence_deviating_streams += 1
                    if SPAN_DEVIATES_FROM_DECLARED_RATE in stream.issue_codes:
                        facts.span_deviating_streams += 1
                    if "NONCANONICAL_SOURCE_ORDER" in stream.issue_codes:
                        noncanonical_order_streams += 1
                    continue

                facts.streams_refused += 1
                task_counts["refused"] += 1
                stream_failures.append({
                    "subject": subject, "status": stream.stream_status,
                })
                status = stream.stream_status
                if status in AMBIGUITY_FAILURES:
                    facts.ambiguous_declaration_streams += 1
                if status == UNRECOGNIZED_DEVICE_LOCATION:
                    facts.unrecognized_device_location_streams += 1
                if status in {ROW_COUNT_MISMATCH, ROW_COLUMN_COUNT_MISMATCH}:
                    facts.row_count_mismatch_streams += 1
                if status == INVALID_TIME:
                    facts.invalid_time_streams += 1
                if status == NO_USABLE_VALUES:
                    facts.no_usable_value_streams += 1
                if status == BLANK_SOURCE_ROW:
                    facts.blank_row_streams += 1

            task_counts.setdefault("expected_sample_support_seconds", 0)
            per_task[session.record_name][
                "expected_sample_support_seconds"
            ] = support
            per_task[session.record_name][
                "expected_first_to_last_span_seconds"
            ] = span
            per_task[session.record_name]["declared_rows"] = session.rows

    if failures:
        facts.source_files_failed += len(failures)

    evidence: dict[str, Any] = {
        "artifact_kind": PADS_ARTIFACT_KIND,
        "schema_version": PADS_SCHEMA_VERSION,
        "implementation_version": PADS_IMPLEMENTATION_VERSION,
        "contract_version": PADS_CONTRACT_VERSION,
        "parser_version": PADS_PARSER_VERSION,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "release_version": PADS_RELEASE_VERSION,
        "release_contract_version": PADS_RELEASE_CONTRACT_VERSION,
        "authority": authority_contract(),
        "source_manifest_sha256": source_manifest_sha256,
        "source_failures": sorted(failures)[:64],
        "source_failure_count": len(failures),
        "release_structure": structure.as_record(),
        "streams": {
            "declared": facts.streams_declared,
            "parsed": facts.streams_parsed,
            "refused": facts.streams_refused,
            "hash_verified": facts.source_files_hash_verified,
            "hash_failed": facts.source_files_failed,
            "noncanonical_source_order": noncanonical_order_streams,
            "cadence_deviating": facts.cadence_deviating_streams,
            "span_deviating": facts.span_deviating_streams,
            "failures": stream_failures[:64],
            "failure_count": len(stream_failures),
        },
        "samples": {
            "total": total_samples,
            "usable_sensor_values": total_usable_values,
            "duplicate_time": duplicate_time_total,
            "nonmonotonic_time": nonmonotonic_time_total,
        },
        "per_task": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(per_task.items())
        },
        "withheld_artifacts": dict(WITHHELD_P01_ARTIFACTS),
        "materialized_release_artifacts": 0,
    }
    evidence_sha256 = canonical_sha256(evidence)

    receipt = build_run_receipt(
        dataset_root=dataset_root,
        source_manifest_sha256=source_manifest_sha256,
        output_root=output_root,
        schema_version=PADS_SCHEMA_VERSION,
        contract_version=PADS_CONTRACT_VERSION,
        implementation_version=PADS_IMPLEMENTATION_VERSION,
        canonical_evidence_sha256=evidence_sha256,
        command_arguments=tuple(command_arguments),
        run_id=run_id,
        process_id=process_id,
    ).as_record()

    facts.reproduction_status = verify_independent_reproduction(
        receipt, reproduction_receipt
    )

    record: dict[str, Any] = dict(evidence)
    record["audit_execution_status"] = AUDIT_EXECUTION_PASS
    record["release_status"] = RELEASE_EVALUATED
    record["gate_evaluated"] = True
    record["canonical_evidence_sha256"] = evidence_sha256
    record["run_receipt"] = receipt
    record.update(evaluate_gate(facts).as_record())
    return record, receipt


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
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
        record, receipt = audit_pads_p01(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            command_arguments=tuple(argv or sys.argv[1:]),
            reproduction_receipt=reproduction,
            progress=args.progress,
        )
        payload = canonical_json_bytes(record)
        with (args.output_root / "pads_p01_evidence.json").open("xb") as out:
            out.write(payload)
        if receipt is not None:
            with (
                args.output_root / "pads_p01_run_receipt.json"
            ).open("xb") as out:
                out.write(canonical_json_bytes(receipt))
    except (PadsAuditError, OSError, ValueError) as exc:
        print(f"{AUDIT_EXECUTION_ERROR}: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(payload)
    return exit_code_for(record)


if __name__ == "__main__":
    raise SystemExit(main())
