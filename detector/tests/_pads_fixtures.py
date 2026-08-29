"""Synthetic PADS releases for PADS-P0.1 tests.

The fixtures mirror the published structure exactly: one observation record and
one patient record per participant, eleven sessions per observation, two device
files per session, seven comma-separated columns per file with ``Time`` first.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from motionbloom.tremora_store.pads.authority import (
    CANONICAL_CHANNELS,
    CHANNEL_UNITS,
)
from motionbloom.tremora_store.pads.release_structure import (
    PADS_EXPECTED_DEVICE_LOCATIONS,
    PADS_EXPECTED_TASKS,
)

#: The row counts the published release declares, per task.
DECLARED_ROWS = {
    "Relaxed": 2048,
    "RelaxedTask": 2048,
    "Entrainment": 2048,
}
DEFAULT_ROWS = 1024
SAMPLING_RATE = 100

#: A real device clock jitters; the first published file runs from 7.13 ms to
#: 12.90 ms around a 9.99 ms median.  The fixtures use the release's own mean
#: interval rather than a perfect 1/100 grid.
OBSERVED_INTERVAL = 0.0099946


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def timeseries_bytes(
    rows: int,
    *,
    interval: float = OBSERVED_INTERVAL,
    time_override: dict[int, str] | None = None,
    value_override: str | None = None,
    columns: int = 7,
    blank_row_at: int | None = None,
) -> bytes:
    lines: list[str] = []
    for index in range(rows):
        time_token = f"{index * interval:.10f}"
        if time_override and index in time_override:
            time_token = time_override[index]
        values = [value_override or f"{0.001 * (index % 7):.10f}"] * (
            columns - 1
        )
        lines.append(",".join([time_token, *values]))
    if blank_row_at is not None:
        lines.insert(blank_row_at, "")
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_release(
    root: Path,
    *,
    participants: Sequence[str] = ("001",),
    tasks: Sequence[str] = PADS_EXPECTED_TASKS,
    devices: Sequence[str] = PADS_EXPECTED_DEVICE_LOCATIONS,
    channels: Sequence[str] | None = None,
    units: Sequence[str] | None = None,
    rows_for: dict[str, int] | None = None,
    payload_for=None,
    file_name_for=None,
    write_checksums: bool = True,
    write_patients: bool = True,
    drop_files: Sequence[str] = (),
) -> Path:
    """Write a synthetic release and return its dataset root."""

    dataset_root = root / "pads-1.0.0"
    movement = dataset_root / "movement" / "timeseries"
    movement.mkdir(parents=True, exist_ok=True)
    patients = dataset_root / "patients"
    patients.mkdir(parents=True, exist_ok=True)

    declared_channels = list(channels or CANONICAL_CHANNELS)
    if units is None:
        declared_units = [CHANNEL_UNITS[name] for name in declared_channels]
    else:
        declared_units = list(units)
    checksums: dict[str, str] = {}

    for participant in participants:
        sessions = []
        for task in tasks:
            rows = (rows_for or {}).get(
                task, DECLARED_ROWS.get(task, DEFAULT_ROWS)
            )
            records = []
            for device in devices:
                if file_name_for is not None:
                    file_name = file_name_for(participant, task, device)
                else:
                    file_name = (
                        f"timeseries/{participant}_{task}_{device}.txt"
                    )
                payload = (
                    payload_for(participant, task, device, rows)
                    if payload_for is not None
                    else timeseries_bytes(
                        rows, columns=len(declared_channels)
                    )
                )
                relative = f"movement/{file_name}"
                if file_name not in drop_files:
                    target = dataset_root / "movement" / file_name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    checksums[relative] = sha256_bytes(payload)
                records.append({
                    "device_location": device,
                    "channels": declared_channels,
                    "units": declared_units,
                    "file_name": file_name,
                })
            sessions.append({
                "record_name": task, "rows": rows, "records": records,
            })
        observation = {
            "resource_type": "observation",
            "subject_id": participant,
            "study_id": "PADS",
            "device_id": "Apple Watch Series 4",
            "id": "Neurological Assessment",
            "endianness": "little",
            "sampling_rate": SAMPLING_RATE,
            "data_type": "float",
            "bits": 32,
            "session": sessions,
        }
        observation_path = (
            dataset_root / "movement" / f"observation_{participant}.json"
        )
        payload = json.dumps(observation, indent=2).encode("utf-8")
        observation_path.write_bytes(payload)
        checksums[f"movement/observation_{participant}.json"] = sha256_bytes(
            payload
        )

        if write_patients:
            patient = {
                "resource_type": "patient",
                "id": participant,
                "study_id": "PADS",
                "condition": "Healthy",
                "disease_comment": "-",
                "handedness": "right",
            }
            patient_payload = json.dumps(patient, indent=2).encode("utf-8")
            (patients / f"patient_{participant}.json").write_bytes(
                patient_payload
            )
            checksums[f"patients/patient_{participant}.json"] = sha256_bytes(
                patient_payload
            )

    if write_checksums:
        lines = [
            f"{digest} {relative}"
            for relative, digest in sorted(checksums.items())
        ]
        (dataset_root / "SHA256SUMS.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return dataset_root
