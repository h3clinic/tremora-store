"""Materialize the P0.3 spectral workload and the source-versus-replay audit."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..p02.contract import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_VERSION,
)
from ..p02.replay import replay_stream
from .contract import (
    FAMILY_AXES,
    INELIGIBLE_ABOVE_NYQUIST,
    INELIGIBLE_COVERAGE,
    INELIGIBLE_NO_CADENCE,
    INELIGIBLE_SEGMENT_CROSSING,
    INELIGIBLE_TIME_NOT_INCREASING,
    INELIGIBLE_UNUSABLE_CHANNEL,
    INELIGIBLE_WINDOW_NOT_VALID,
    MINIMUM_COVERAGE_FRACTION,
    SENSOR_FAMILIES,
    SPECTRALLY_ELIGIBLE,
    WINDOW_DURATION_S,
)
from .grid import frequency_values, grid_id, grid_within_nyquist, nyquist_hz
from .kernel import (
    SpectralKernelError,
    family_spectrum,
    input_content_sha256,
)
from .schemas import P03_TABLE_FILES, P03_TABLE_SCHEMAS
from .selection import (
    SELECTED_CLOSEST_TO_STREAM_MIDPOINT,
    WindowFacts,
    select_audit_subset,
    select_workload,
    selection_coverage,
    stream_midpoints_ps,
    window_facts,
)
from .source_path import (
    SourcePathError,
    read_source_window,
    replay_row_identity_sha256,
)

PRESERVED = "PRESERVED"
PRESERVATION_ROW_MISMATCH = "ROW_MISMATCH"
PRESERVATION_INPUT_MISMATCH = "INPUT_MISMATCH"
PRESERVATION_SPECTRUM_MISMATCH = "SPECTRUM_MISMATCH"
PRESERVATION_SOURCE_UNREADABLE = "SOURCE_UNREADABLE"

SPECTRUM_COMPUTED = "SPECTRUM_COMPUTED"
SPECTRUM_REFUSED = "SPECTRUM_REFUSED"

_STORE_TABLES = (
    "pads_windows.parquet",
    "pads_streams.parquet",
    "pads_segments.parquet",
    "pads_stream_storage_index.parquet",
)


@dataclass(slots=True)
class P03Result:
    """Everything the gate and the release report are decided from."""

    streams_total: int = 0
    streams_with_valid_windows: int = 0
    workload_windows_selected: int = 0
    audit_windows_selected: int = 0
    workload_windows_eligible: int = 0
    workload_windows_ineligible: int = 0
    accel_spectral_rows: int = 0
    gyro_spectral_rows: int = 0
    source_replay_row_mismatches: int = 0
    source_replay_input_hash_mismatches: int = 0
    source_replay_spectral_hash_mismatches: int = 0
    dominant_frequency_mismatches: int = 0
    maximum_observed_bin_error: float = 0.0
    source_unreadable: int = 0
    nominal_grid_substitutions: int = 0
    vector_magnitude_uses: int = 0
    sample_count_histogram: dict[str, int] = field(default_factory=dict)
    dt_ref_ps_distribution: dict[str, int] = field(default_factory=dict)
    ineligibility_reasons: dict[str, int] = field(default_factory=dict)
    selection_coverage: dict[str, Any] = field(default_factory=dict)
    spectral_table_content_sha256: str = ""
    failures: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "streams_total": self.streams_total,
            "streams_with_valid_windows": self.streams_with_valid_windows,
            "workload_windows_selected": self.workload_windows_selected,
            "audit_windows_selected": self.audit_windows_selected,
            "workload_windows_eligible": self.workload_windows_eligible,
            "workload_windows_ineligible": self.workload_windows_ineligible,
            "accel_spectral_rows": self.accel_spectral_rows,
            "gyro_spectral_rows": self.gyro_spectral_rows,
            "source_replay_row_mismatches": (
                self.source_replay_row_mismatches
            ),
            "source_replay_input_hash_mismatches": (
                self.source_replay_input_hash_mismatches
            ),
            "source_replay_spectral_hash_mismatches": (
                self.source_replay_spectral_hash_mismatches
            ),
            "dominant_frequency_mismatches": (
                self.dominant_frequency_mismatches
            ),
            "maximum_observed_bin_error": self.maximum_observed_bin_error,
            "source_unreadable": self.source_unreadable,
            "nominal_grid_substitutions": self.nominal_grid_substitutions,
            "vector_magnitude_uses": self.vector_magnitude_uses,
            "sample_count_histogram": dict(
                sorted(self.sample_count_histogram.items())
            ),
            "dt_ref_ps_distribution": dict(
                sorted(self.dt_ref_ps_distribution.items())
            ),
            "ineligibility_reasons": dict(
                sorted(self.ineligibility_reasons.items())
            ),
            "selection_coverage": self.selection_coverage,
            "spectral_table_content_sha256": (
                self.spectral_table_content_sha256
            ),
            "failure_count": len(self.failures),
            "failures": sorted(self.failures)[:64],
        }


def read_store(store_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the P0.2 index tables this milestone consumes."""

    tables: dict[str, list[dict[str, Any]]] = {}
    for filename in _STORE_TABLES:
        tables[filename] = pq.read_table(store_root / filename).to_pylist()
    return tables


def evaluate_eligibility(
    window: WindowFacts,
    segment: Mapping[str, Any],
    times_ps: Sequence[int],
    axes: Mapping[str, Sequence[float]],
) -> tuple[str, str | None]:
    """Decide whether a window may carry a spectrum, and say why not."""

    if window.window_status != "WINDOW_VALID":
        return INELIGIBLE_WINDOW_NOT_VALID, window.window_status
    if window.dt_ref_ps <= 0:
        return INELIGIBLE_NO_CADENCE, "dt_ref is absent or not positive"
    if window.coverage_fraction < MINIMUM_COVERAGE_FRACTION:
        return INELIGIBLE_COVERAGE, f"{window.coverage_fraction:.4f}"
    if (
        window.first_sample_ordinal < int(segment["first_sample_ordinal"])
        or window.last_sample_ordinal > int(segment["last_sample_ordinal"])
    ):
        return INELIGIBLE_SEGMENT_CROSSING, segment["segment_id"]
    if not grid_within_nyquist(window.dt_ref_ps):
        return (
            INELIGIBLE_ABOVE_NYQUIST,
            f"{nyquist_hz(window.dt_ref_ps):.3f} Hz",
        )
    if any(later <= earlier for earlier, later in pairwise(times_ps)):
        return INELIGIBLE_TIME_NOT_INCREASING, "stored times are not ordered"
    for name, values in axes.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape[0] != len(times_ps) or not np.all(np.isfinite(array)):
            return INELIGIBLE_UNUSABLE_CHANNEL, name
    return SPECTRALLY_ELIGIBLE, None


def _write_table(output_root: Path, name: str, records: list[dict[str, Any]]) -> None:
    schema = P03_TABLE_SCHEMAS[name]()
    columns = {
        field_.name: [record.get(field_.name) for record in records]
        for field_ in schema
    }
    pq.write_table(
        pa.Table.from_pydict(columns, schema=schema),
        output_root / P03_TABLE_FILES[name],
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        version=PARQUET_VERSION,
        write_statistics=True,
    )


def _spectral_table_hash(rows: list[dict[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["spectral_record_id"].encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(row["spectral_content_sha256"].encode("ascii"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def materialize(
    *,
    release_root: Path,
    store_root: Path,
    output_root: Path,
    progress: bool = False,
) -> P03Result:
    """Compute the workload spectra and the independent source comparison."""

    result = P03Result()
    tables = read_store(store_root)
    windows = tables["pads_windows.parquet"]
    streams = tables["pads_streams.parquet"]
    segments = {
        str(row["segment_id"]): row
        for row in tables["pads_segments.parquet"]
    }
    storage_index = {
        str(row["stream_id"]): row
        for row in tables["pads_stream_storage_index.parquet"]
    }
    asset_by_stream = {
        stream_id: str(row["source_asset_sha256"])
        for stream_id, row in storage_index.items()
    }

    result.streams_total = len(streams)
    facts = window_facts(windows, segments)
    result.streams_with_valid_windows = len(
        {window.stream_id for window in facts}
    )

    workload = select_workload(facts, stream_midpoints_ps(streams))
    audit = select_audit_subset(facts)
    result.workload_windows_selected = len(workload)
    result.audit_windows_selected = len(audit)
    result.selection_coverage = selection_coverage(audit)

    grid = np.asarray(frequency_values(), dtype=np.float64)
    identifier = grid_id()

    cached_stream: str | None = None
    cached_table: pa.Table | None = None

    def stream_table(stream_id: str) -> pa.Table:
        nonlocal cached_stream, cached_table
        if cached_stream != stream_id:
            cached_table = replay_stream(store_root, storage_index, stream_id)
            cached_stream = stream_id
        assert cached_table is not None
        return cached_table

    def window_rows(window: WindowFacts) -> tuple[list[int], list[str], dict[str, list[float]]]:
        table = stream_table(window.stream_id)
        entry = storage_index[window.stream_id]
        offset = window.first_sample_ordinal - int(
            entry["first_sample_ordinal"]
        )
        length = window.last_sample_ordinal - window.first_sample_ordinal + 1
        sliced = table.slice(offset, length)
        times = sliced.column("source_time_ps").to_pylist()
        tokens = sliced.column("source_time_token").to_pylist()
        axes = {
            name: sliced.column(name).to_pylist()
            for family in SENSOR_FAMILIES for name in FAMILY_AXES[family]
        }
        return times, tokens, axes

    workload_rows: list[dict[str, Any]] = []
    spectra_rows: list[dict[str, Any]] = []

    for index, window in enumerate(sorted(workload, key=lambda w: w.window_id)):
        times, _tokens, axes = window_rows(window)
        segment = segments[window.segment_id]
        eligibility, reason = evaluate_eligibility(
            window, segment, times, axes
        )
        counts = result.sample_count_histogram
        counts[str(window.sample_count)] = counts.get(
            str(window.sample_count), 0
        ) + 1
        bucket = str(round(window.dt_ref_ps, -8))
        result.dt_ref_ps_distribution[bucket] = (
            result.dt_ref_ps_distribution.get(bucket, 0) + 1
        )
        workload_rows.append({
            "window_id": window.window_id,
            "stream_id": window.stream_id,
            "participant_id": window.participant_id,
            "assessment_id": window.assessment_id,
            "task_name": window.task_name,
            "device_location": window.device_location,
            "outer_fold": window.outer_fold,
            "sample_count": window.sample_count,
            "dt_ref_ps": window.dt_ref_ps,
            "coverage_fraction": window.coverage_fraction,
            "effective_rate_hz": window.effective_rate_hz,
            "nyquist_hz": nyquist_hz(window.dt_ref_ps),
            "gap_adjacent_status": window.gap_adjacent_status,
            "workload_selection_reason": (
                SELECTED_CLOSEST_TO_STREAM_MIDPOINT
            ),
            "spectral_eligibility": eligibility,
            "ineligibility_reason": reason,
        })
        if eligibility != SPECTRALLY_ELIGIBLE:
            result.workload_windows_ineligible += 1
            result.ineligibility_reasons[eligibility] = (
                result.ineligibility_reasons.get(eligibility, 0) + 1
            )
            continue
        result.workload_windows_eligible += 1

        for family in SENSOR_FAMILIES:
            family_axes = [axes[name] for name in FAMILY_AXES[family]]
            try:
                spectrum = family_spectrum(
                    family, times, family_axes,
                    duration_s=WINDOW_DURATION_S, frequencies=grid,
                )
            except SpectralKernelError as exc:
                result.failures.append(f"{window.window_id}:{family}: {exc}")
                continue
            record = spectrum.as_record()
            record.update({
                "spectral_record_id": f"{window.window_id}#{family}",
                "window_id": window.window_id,
                "sensor_family": family,
                "sample_count": window.sample_count,
                "dt_ref_ps": window.dt_ref_ps,
                "frequency_grid_id": identifier,
                "input_content_sha256": input_content_sha256(
                    times, family_axes
                ),
                "spectral_content_sha256": spectrum.content_sha256(),
                "spectral_status": SPECTRUM_COMPUTED,
            })
            spectra_rows.append(record)
            if family == "ACCELEROMETER":
                result.accel_spectral_rows += 1
            else:
                result.gyro_spectral_rows += 1
        if progress and (index + 1) % 2000 == 0:
            print(f"workload {index + 1}", file=sys.stderr, flush=True)

    audit_rows: list[dict[str, Any]] = []
    for index, window in enumerate(audit):
        times, tokens, axes = window_rows(window)
        ordinals = list(range(
            window.first_sample_ordinal, window.last_sample_ordinal + 1
        ))
        try:
            source = read_source_window(
                release_root=release_root,
                participant_id=window.participant_id,
                task_name=window.task_name,
                device_location=window.device_location,
                stream_id=window.stream_id,
                window_start_task_local_ps=(
                    window.window_start_task_local_ps
                ),
                window_end_task_local_ps=window.window_end_task_local_ps,
                expected_asset_sha256=asset_by_stream[window.stream_id],
            )
        except SourcePathError as exc:
            result.source_unreadable += 1
            result.failures.append(f"{window.window_id}: {exc}")
            audit_rows.append({
                "window_id": window.window_id,
                "stream_id": window.stream_id,
                "stratum_id": window.stratum_id,
                "source_asset_sha256": asset_by_stream[window.stream_id],
                "source_row_count": 0,
                "replay_row_count": len(ordinals),
                "source_row_identity_sha256": "",
                "replay_row_identity_sha256": replay_row_identity_sha256(
                    window.stream_id, ordinals, tokens
                ),
                "source_input_sha256": "",
                "replay_input_sha256": "",
                "source_accel_spectrum_sha256": "",
                "replay_accel_spectrum_sha256": "",
                "source_gyro_spectrum_sha256": "",
                "replay_gyro_spectrum_sha256": "",
                "maximum_bin_absolute_error": float("inf"),
                "dominant_frequency_match": False,
                "preservation_status": PRESERVATION_SOURCE_UNREADABLE,
            })
            continue

        replay_identity = replay_row_identity_sha256(
            window.stream_id, ordinals, tokens
        )
        rows_match = (
            list(source.ordinals) == ordinals
            and list(source.time_tokens) == tokens
            and list(source.times_ps) == times
            and source.row_identity_sha256() == replay_identity
        )
        status = PRESERVED if rows_match else PRESERVATION_ROW_MISMATCH
        if not rows_match:
            result.source_replay_row_mismatches += 1

        hashes: dict[str, str] = {}
        worst = 0.0
        dominant_match = True
        for family in SENSOR_FAMILIES:
            source_axes = source.axes(family)
            replay_axes = [axes[name] for name in FAMILY_AXES[family]]
            source_spectrum = family_spectrum(
                family, source.times_ps, source_axes,
                duration_s=WINDOW_DURATION_S, frequencies=grid,
            )
            replay_spectrum = family_spectrum(
                family, times, replay_axes,
                duration_s=WINDOW_DURATION_S, frequencies=grid,
            )
            key = "gyro" if family == "GYROSCOPE" else "accel"
            hashes[f"source_{key}"] = source_spectrum.content_sha256()
            hashes[f"replay_{key}"] = replay_spectrum.content_sha256()
            worst = max(worst, float(np.max(np.abs(
                source_spectrum.aggregate - replay_spectrum.aggregate
            ))))
            if source_spectrum.dominant_frequency_hz != (
                replay_spectrum.dominant_frequency_hz
            ):
                dominant_match = False

        source_input = input_content_sha256(
            source.times_ps, [
                source.channels[name]
                for family in SENSOR_FAMILIES for name in FAMILY_AXES[family]
            ]
        )
        replay_input = input_content_sha256(times, [
            axes[name]
            for family in SENSOR_FAMILIES for name in FAMILY_AXES[family]
        ])
        if source_input != replay_input:
            result.source_replay_input_hash_mismatches += 1
            status = PRESERVATION_INPUT_MISMATCH
        if (
            hashes["source_gyro"] != hashes["replay_gyro"]
            or hashes["source_accel"] != hashes["replay_accel"]
        ):
            result.source_replay_spectral_hash_mismatches += 1
            status = PRESERVATION_SPECTRUM_MISMATCH
        if not dominant_match:
            result.dominant_frequency_mismatches += 1
        result.maximum_observed_bin_error = max(
            result.maximum_observed_bin_error, worst
        )

        audit_rows.append({
            "window_id": window.window_id,
            "stream_id": window.stream_id,
            "stratum_id": window.stratum_id,
            "source_asset_sha256": source.source_asset_sha256,
            "source_row_count": source.row_count,
            "replay_row_count": len(ordinals),
            "source_row_identity_sha256": source.row_identity_sha256(),
            "replay_row_identity_sha256": replay_identity,
            "source_input_sha256": source_input,
            "replay_input_sha256": replay_input,
            "source_accel_spectrum_sha256": hashes["source_accel"],
            "replay_accel_spectrum_sha256": hashes["replay_accel"],
            "source_gyro_spectrum_sha256": hashes["source_gyro"],
            "replay_gyro_spectrum_sha256": hashes["replay_gyro"],
            "maximum_bin_absolute_error": worst,
            "dominant_frequency_match": dominant_match,
            "preservation_status": status,
        })
        if progress and (index + 1) % 1000 == 0:
            print(f"audit {index + 1}", file=sys.stderr, flush=True)

    workload_rows.sort(key=lambda row: row["window_id"])
    spectra_rows.sort(
        key=lambda row: (row["window_id"], row["sensor_family"])
    )
    audit_rows.sort(key=lambda row: row["window_id"])
    result.spectral_table_content_sha256 = _spectral_table_hash(spectra_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    _write_table(output_root, "pads_p03_workload_windows", workload_rows)
    _write_table(output_root, "pads_p03_spectra", spectra_rows)
    _write_table(output_root, "pads_p03_source_replay_audit", audit_rows)
    return result


__all__ = [
    "PRESERVATION_INPUT_MISMATCH",
    "PRESERVATION_ROW_MISMATCH",
    "PRESERVATION_SOURCE_UNREADABLE",
    "PRESERVATION_SPECTRUM_MISMATCH",
    "PRESERVED",
    "SPECTRUM_COMPUTED",
    "SPECTRUM_REFUSED",
    "P03Result",
    "evaluate_eligibility",
    "materialize",
    "read_store",
]
