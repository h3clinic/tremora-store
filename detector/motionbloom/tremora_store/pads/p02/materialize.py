"""Materialize the PADS-P0.2 sample store and index set.

The order of work is fixed so two clean roots agree: descriptors are collected
from the release metadata, sorted by ``stream_id``, and then read, segmented,
windowed and packed in that order.  Folds are assigned before any window is
built, so a window carries its participant's fold rather than recomputing one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..movement import (
    Observation,
    PadsSourceError,
    Participant,
    StreamDeclaration,
    expected_durations,
    parse_observation,
    parse_patient,
    stream_id_for,
)
from ..release_structure import reconcile_release_structure
from .bilateral import (
    build_bilateral_tasks,
    build_bilateral_window_pairs,
)
from .contract import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_VERSION,
    SUCCESS_MARKER,
)
from .dependency import CHECKSUM_FILENAME, read_checksums
from .folds import assign_folds
from .sample_store import SampleStoreWriter, StorageIndexEntry
from .schemas import P02_INDEX_FILES, P02_TABLE_SCHEMAS
from .segments import assert_partitions_stream, build_segments
from .stream_reader import STREAM_READ_OK, read_stream
from .windows import assert_windows_inside_segments, build_windows

MOVEMENT_DIRECTORY = "movement"
PATIENTS_DIRECTORY = "patients"

PARTICIPANT_ASSIGNED = "PARTICIPANT_ASSIGNED"
PARTICIPANT_METADATA_MISSING = "PARTICIPANT_METADATA_MISSING"
STREAM_MATERIALIZED = "STREAM_MATERIALIZED"


class MaterializationError(RuntimeError):
    """Raised when materialization cannot proceed deterministically."""


@dataclass(slots=True)
class MaterializationResult:
    """Everything the gate and the release report are decided from."""

    participants_materialized: int = 0
    assessments_materialized: int = 0
    streams_materialized: int = 0
    streams_refused: int = 0
    samples_materialized: int = 0
    duplicate_materialized_samples: int = 0
    parquet_files: int = 0
    row_groups: int = 0
    streams_with_exactly_one_row_group: int = 0
    segments: int = 0
    segment_partition_failures: int = 0
    detected_time_gaps: int = 0
    windows: int = 0
    windows_crossing_segments: int = 0
    window_replay_failures: int = 0
    bilateral_task_pairs: int = 0
    bilateral_window_pairs: int = 0
    sample_level_alignment_claims: int = 0
    source_files_expected: int = 0
    source_files_hash_verified: int = 0
    source_files_failed: int = 0
    replay_streams_checked: int = 0
    replay_byte_exact_streams: int = 0
    source_time_token_failures: int = 0
    fold_count: int = 0
    participants_in_multiple_folds: int = 0
    participants_without_fold: int = 0
    release_structure_status: str = "RELEASE_STRUCTURE_NOT_EVALUATED"
    failures: list[str] = field(default_factory=list)
    fold_sizes: dict[int, int] = field(default_factory=dict)
    per_task_windows: dict[str, int] = field(default_factory=dict)
    storage_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    window_records: list[dict[str, Any]] = field(default_factory=list)
    source_sha256_by_stream: dict[str, str] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "participants_materialized": self.participants_materialized,
            "assessments_materialized": self.assessments_materialized,
            "streams_materialized": self.streams_materialized,
            "streams_refused": self.streams_refused,
            "samples_materialized": self.samples_materialized,
            "duplicate_materialized_samples": (
                self.duplicate_materialized_samples
            ),
            "parquet_files": self.parquet_files,
            "row_groups": self.row_groups,
            "streams_with_exactly_one_row_group": (
                self.streams_with_exactly_one_row_group
            ),
            "segments": self.segments,
            "segment_partition_failures": self.segment_partition_failures,
            "detected_time_gaps": self.detected_time_gaps,
            "windows": self.windows,
            "windows_crossing_segments": self.windows_crossing_segments,
            "window_replay_failures": self.window_replay_failures,
            "bilateral_task_pairs": self.bilateral_task_pairs,
            "bilateral_window_pairs": self.bilateral_window_pairs,
            "sample_level_alignment_claims": (
                self.sample_level_alignment_claims
            ),
            "source_files_expected": self.source_files_expected,
            "source_files_hash_verified": self.source_files_hash_verified,
            "source_files_failed": self.source_files_failed,
            "replay_streams_checked": self.replay_streams_checked,
            "replay_byte_exact_streams": self.replay_byte_exact_streams,
            "source_time_token_failures": self.source_time_token_failures,
            "fold_count": self.fold_count,
            "participants_in_multiple_folds": (
                self.participants_in_multiple_folds
            ),
            "participants_without_fold": self.participants_without_fold,
            "fold_sizes": {
                str(fold): size
                for fold, size in sorted(self.fold_sizes.items())
            },
            "per_task_windows": dict(sorted(self.per_task_windows.items())),
            "release_structure_status": self.release_structure_status,
            "failure_count": len(self.failures),
            "failures": sorted(self.failures)[:64],
        }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_index_table(
    output_root: Path, table_name: str, records: list[dict[str, Any]]
) -> Path:
    """Write one index table under the frozen writer configuration."""

    schema = P02_TABLE_SCHEMAS[table_name]()
    columns = {
        field.name: [record.get(field.name) for record in records]
        for field in schema
    }
    table = pa.Table.from_pydict(columns, schema=schema)
    path = output_root / P02_INDEX_FILES[table_name]
    pq.write_table(
        table,
        path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        version=PARQUET_VERSION,
        write_statistics=True,
    )
    return path


@dataclass(frozen=True, slots=True)
class _Descriptor:
    stream_id: str
    participant_id: str
    assessment_id: str
    task_name: str
    task_ordinal: int
    declared_rows: int
    sampling_rate: Fraction
    declaration: StreamDeclaration


def _descriptors(
    observations: Mapping[str, Observation],
) -> list[_Descriptor]:
    descriptors: list[_Descriptor] = []
    for participant_id in sorted(observations):
        observation = observations[participant_id]
        for session in observation.sessions:
            assessment_id = f"{participant_id}:{session.record_name}"
            for declaration in session.streams:
                descriptors.append(_Descriptor(
                    stream_id=stream_id_for(
                        participant_id,
                        session.record_name,
                        declaration.device_location,
                    ),
                    participant_id=participant_id,
                    assessment_id=assessment_id,
                    task_name=session.record_name,
                    task_ordinal=session.task_ordinal,
                    declared_rows=session.rows,
                    sampling_rate=observation.sampling_rate,
                    declaration=declaration,
                ))
    descriptors.sort(key=lambda item: item.stream_id)
    return descriptors


def materialize(
    *,
    release_root: Path,
    output_root: Path,
    p01_evidence_sha256: str,
    progress: bool = False,
) -> MaterializationResult:
    """Read the release once and write the store and every index."""

    import sys

    movement_root = release_root / MOVEMENT_DIRECTORY
    patients_root = release_root / PATIENTS_DIRECTORY
    result = MaterializationResult()
    checksums = read_checksums(release_root / CHECKSUM_FILENAME)

    def verify(relative: str, payload: bytes) -> None:
        result.source_files_expected += 1
        expected = checksums.get(relative)
        if expected is None or expected != _sha256(payload):
            result.source_files_failed += 1
            result.failures.append(f"{relative}: not verified")
        else:
            result.source_files_hash_verified += 1

    observations: dict[str, Observation] = {}
    for path in sorted(movement_root.glob("observation_*.json")):
        relative = f"{MOVEMENT_DIRECTORY}/{path.name}"
        payload = path.read_bytes()
        verify(relative, payload)
        try:
            observation = parse_observation(
                json.loads(payload.decode("utf-8")),
                source_relative_path=relative,
            )
        except (ValueError, PadsSourceError) as exc:
            result.failures.append(f"{relative}: {exc}")
            continue
        observations[observation.subject_id] = observation

    participants: dict[str, Participant] = {}
    for path in sorted(patients_root.glob("patient_*.json")):
        relative = f"{PATIENTS_DIRECTORY}/{path.name}"
        payload = path.read_bytes()
        verify(relative, payload)
        try:
            participant = parse_patient(
                json.loads(payload.decode("utf-8")),
                source_relative_path=relative,
            )
        except (ValueError, PadsSourceError) as exc:
            result.failures.append(f"{relative}: {exc}")
            continue
        participants[participant.participant_id] = participant

    structure = reconcile_release_structure(
        observations,
        participants,
        file_exists=lambda name: (movement_root / name).is_file(),
    )
    result.release_structure_status = structure.status

    conditions = {
        participant_id: (participant.condition or "UNSPECIFIED")
        for participant_id, participant in participants.items()
    }
    folds = assign_folds(conditions)
    result.fold_count = len(set(folds.values())) if folds else 0
    for fold in folds.values():
        result.fold_sizes[fold] = result.fold_sizes.get(fold, 0) + 1

    descriptors = _descriptors(observations)
    storage_entries: list[StorageIndexEntry] = []
    stream_records: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    assessment_rows: dict[str, dict[str, Any]] = {}
    streams_by_assessment: dict[str, dict[str, str]] = {}
    all_windows = []
    seen_stream_ids: set[str] = set()

    with SampleStoreWriter(
        output_root, p01_evidence_sha256=p01_evidence_sha256
    ) as writer:
        for index, descriptor in enumerate(descriptors):
            relative = (
                f"{MOVEMENT_DIRECTORY}/{descriptor.declaration.file_name}"
            )
            path = movement_root / descriptor.declaration.file_name
            if not path.is_file():
                result.streams_refused += 1
                result.failures.append(f"{relative}: absent")
                continue
            payload = path.read_bytes()
            verify(relative, payload)
            asset_sha256 = _sha256(payload)

            samples = read_stream(
                payload,
                declaration=descriptor.declaration,
                declared_rows=descriptor.declared_rows,
                sampling_rate=descriptor.sampling_rate,
                stream_id=descriptor.stream_id,
                participant_id=descriptor.participant_id,
                assessment_id=descriptor.assessment_id,
                task_name=descriptor.task_name,
                source_asset_sha256=asset_sha256,
            )
            if samples.stream_status != STREAM_READ_OK:
                result.streams_refused += 1
                result.failures.append(
                    f"{descriptor.stream_id}: {samples.stream_status}")
                continue

            # Byte-exact replay is checked for every stream, not a subset.
            result.replay_streams_checked += 1
            if _sha256(samples.replay_source_bytes()) == asset_sha256:
                result.replay_byte_exact_streams += 1
            else:
                result.failures.append(
                    f"{descriptor.stream_id}: replay is not byte exact")

            built = build_segments(samples)
            try:
                assert_partitions_stream(built, samples)
            except Exception as exc:  # noqa: BLE001 - counted as evidence
                result.segment_partition_failures += 1
                result.failures.append(f"{descriptor.stream_id}: {exc}")
            result.segments += len(built)
            result.detected_time_gaps += max(0, len(built) - 1)

            fold = folds.get(descriptor.participant_id, -1)
            windows = build_windows(
                samples,
                built,
                split_group_id=descriptor.participant_id,
                outer_fold=fold,
            )
            try:
                assert_windows_inside_segments(windows, built)
            except Exception as exc:  # noqa: BLE001 - counted as evidence
                result.windows_crossing_segments += 1
                result.failures.append(f"{descriptor.stream_id}: {exc}")
            all_windows.extend(windows)
            result.windows += len(windows)
            result.per_task_windows[descriptor.task_name] = (
                result.per_task_windows.get(descriptor.task_name, 0)
                + len(windows)
            )

            # Duplicates are counted per stream and bounded by the
            # stream's own length.  A global set keyed by every sample would
            # hold 13.4 million tuples -- gigabytes to prove a property that
            # is decidable one stream at a time, given that stream ids are
            # unique (the writer refuses a repeat) and ordinals are local.
            ordinals = samples.sample_ordinal
            result.duplicate_materialized_samples += (
                len(ordinals) - len(set(ordinals))
            )
            if descriptor.stream_id in seen_stream_ids:
                result.duplicate_materialized_samples += len(ordinals)
                result.failures.append(
                    f"{descriptor.stream_id}: materialized twice")
            seen_stream_ids.add(descriptor.stream_id)
            result.samples_materialized += samples.sample_count

            entry = writer.add(samples)
            result.source_sha256_by_stream[descriptor.stream_id] = (
                asset_sha256
            )
            storage_entries.append(entry)
            result.storage_index[entry.stream_id] = entry.as_record()
            result.streams_materialized += 1

            support, span = expected_durations(
                descriptor.declared_rows, descriptor.sampling_rate
            )
            deltas = [
                later - earlier
                for earlier, later in zip(
                    samples.source_time_ps, samples.source_time_ps[1:],
                    strict=False,
                )
            ]
            positive = sorted(delta for delta in deltas if delta > 0)
            reference = built[0].dt_ref_ps if built else None
            stream_records.append({
                "stream_id": descriptor.stream_id,
                "assessment_id": descriptor.assessment_id,
                "participant_id": descriptor.participant_id,
                "task_name": descriptor.task_name,
                "device_location": descriptor.declaration.device_location,
                "source_asset_sha256": asset_sha256,
                "source_row_count": descriptor.declared_rows,
                "stored_row_count": samples.sample_count,
                "declared_sampling_rate_hz": float(
                    descriptor.sampling_rate
                ),
                "median_interval_ps": reference,
                "minimum_interval_ps": positive[0] if positive else None,
                "maximum_interval_ps": positive[-1] if positive else None,
                "cadence_mad_ps": (
                    sorted(abs(delta - reference) for delta in positive)[
                        len(positive) // 2
                    ] if positive and reference is not None else None
                ),
                "source_time_start_ps": samples.source_time_ps[0],
                "source_time_end_ps": samples.source_time_ps[-1],
                "source_time_origin_token": samples.source_time_origin_token,
                "source_time_origin_ps": samples.source_time_origin_ps,
                "sample_support_seconds": support,
                "first_to_last_span_seconds": (
                    (samples.source_time_ps[-1] - samples.source_time_ps[0])
                    / 1e12
                ),
                "segment_count": len(built),
                "stream_status": STREAM_MATERIALIZED,
            })
            for segment in built:
                segment_records.append(segment.as_record())
            for window in windows:
                window_records.append(window.as_record())
                result.window_records.append(window_records[-1])

            wrists = streams_by_assessment.setdefault(
                descriptor.assessment_id, {}
            )
            wrists[descriptor.declaration.device_location] = (
                descriptor.stream_id
            )
            assessment_rows.setdefault(descriptor.assessment_id, {
                "assessment_id": descriptor.assessment_id,
                "participant_id": descriptor.participant_id,
                "task_name": descriptor.task_name,
                "task_ordinal": descriptor.task_ordinal,
                "declared_sampling_rate_hz": float(
                    descriptor.sampling_rate
                ),
                "declared_row_count": descriptor.declared_rows,
                "expected_sample_support_seconds": support,
            })
            del span
            if progress and (index + 1) % 1000 == 0:
                print(
                    f"materialized {index + 1} streams",
                    file=sys.stderr, flush=True,
                )
        result.parquet_files = writer.part_count

    result.row_groups = len(storage_entries)
    result.streams_with_exactly_one_row_group = len({
        entry.stream_id for entry in storage_entries
    })

    bilateral_tasks = build_bilateral_tasks(
        streams_by_assessment,
        participant_of={
            assessment_id: row["participant_id"]
            for assessment_id, row in assessment_rows.items()
        },
        task_of={
            assessment_id: row["task_name"]
            for assessment_id, row in assessment_rows.items()
        },
    )
    bilateral_pairs = build_bilateral_window_pairs(all_windows)
    result.bilateral_task_pairs = sum(
        1 for task in bilateral_tasks if task.pair_status == "PAIR_COMPLETE"
    )
    result.bilateral_window_pairs = len(bilateral_pairs)

    task_records = [task.as_record() for task in bilateral_tasks]
    pair_records = [pair.as_record() for pair in bilateral_pairs]
    result.sample_level_alignment_claims = sum(
        1 for record in (*task_records, *pair_records)
        if record.get("sample_level_fusion_allowed") is not False
        or record.get("cross_wrist_clock_alignment") != "UNRESOLVED"
    )

    assessments = []
    for assessment_id in sorted(assessment_rows):
        row = dict(assessment_rows[assessment_id])
        wrists = streams_by_assessment.get(assessment_id, {})
        row["left_stream_id"] = wrists.get("LeftWrist", "")
        row["right_stream_id"] = wrists.get("RightWrist", "")
        row["bilateral_pair_status"] = (
            "PAIR_COMPLETE"
            if row["left_stream_id"] and row["right_stream_id"]
            else "PAIR_MISSING_STREAM"
        )
        row["cross_wrist_clock_alignment"] = "UNRESOLVED"
        row["sample_level_fusion_allowed"] = False
        assessments.append(row)
    result.assessments_materialized = len(assessments)

    participant_records = []
    for participant_id in sorted(participants):
        participant = participants[participant_id]
        participant_records.append({
            "participant_id": participant_id,
            "source_patient_asset_sha256": _sha256(
                (patients_root / f"patient_{participant_id}.json").read_bytes()
            ),
            "condition_group": participant.condition or "UNSPECIFIED",
            "split_group_id": participant_id,
            "outer_fold": folds.get(participant_id, -1),
            "participant_status": (
                PARTICIPANT_ASSIGNED if participant_id in folds
                else PARTICIPANT_METADATA_MISSING
            ),
        })
    result.participants_materialized = len(participant_records)
    result.participants_without_fold = sum(
        1 for record in participant_records if record["outer_fold"] < 0
    )

    write_index_table(
        output_root, "pads_stream_storage_index",
        [entry.as_record() for entry in storage_entries],
    )
    write_index_table(output_root, "pads_participants", participant_records)
    write_index_table(output_root, "pads_assessments", assessments)
    write_index_table(output_root, "pads_streams", stream_records)
    write_index_table(output_root, "pads_segments", segment_records)
    write_index_table(output_root, "pads_windows", window_records)
    write_index_table(output_root, "pads_bilateral_tasks", task_records)
    write_index_table(
        output_root, "pads_bilateral_window_pairs", pair_records
    )
    return result


def write_success_marker(output_root: Path) -> Path:
    path = output_root / SUCCESS_MARKER
    path.write_bytes(b"")
    return path


__all__ = [
    "MOVEMENT_DIRECTORY",
    "PARTICIPANT_ASSIGNED",
    "PARTICIPANT_METADATA_MISSING",
    "PATIENTS_DIRECTORY",
    "STREAM_MATERIALIZED",
    "MaterializationError",
    "MaterializationResult",
    "materialize",
    "write_index_table",
    "write_success_marker",
]
