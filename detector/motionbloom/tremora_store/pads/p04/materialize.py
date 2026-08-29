"""Materialize the derived-rate spectra and the source-versus-replay audit."""

from __future__ import annotations

import hashlib
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
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
from ..p03.contract import FAMILY_AXES, SENSOR_FAMILIES, WINDOW_DURATION_S
from ..p03.grid import frequency_values
from ..p03.kernel import family_spectrum, input_content_sha256
from ..p03.schemas import P03_TABLE_FILES
from ..p03.source_path import read_source_window
from .contract import (
    CORE_BAND,
    DERIVED_RATES_HZ,
    EDGE_BAND,
    NATIVE_RATE_LABEL,
    PARENT_RATE_HZ,
    band_of,
)
from .filters import FILTER_SPECS, design, filter_sha256, group_delay_taps
from .rational_time import (
    grid_for,
    polyphase_anchor,
    supported_output_ordinals,
)
from .resample import (
    WINDOW_ELIGIBLE,
    derive_support,
    derive_window,
    window_eligibility,
    window_times_seconds,
)
from .schemas import P04_TABLE_FILES, P04_TABLE_SCHEMAS

SPECTRUM_COMPUTED = "SPECTRUM_COMPUTED"
PRESERVED = "PRESERVED"
PRESERVATION_SAMPLE_COUNT_MISMATCH = "SAMPLE_COUNT_MISMATCH"
PRESERVATION_DERIVED_MISMATCH = "DERIVED_MISMATCH"
PRESERVATION_SPECTRUM_MISMATCH = "SPECTRUM_MISMATCH"
PRESERVATION_SOURCE_UNREADABLE = "SOURCE_UNREADABLE"
SUMMARY_COMPUTED = "SUMMARY_COMPUTED"

_GRID = np.asarray(frequency_values(), dtype=np.float64)
_CORE = np.array([band_of(value) == CORE_BAND for value in _GRID])
_EDGE = ~_CORE


@dataclass(slots=True)
class P04Result:
    """Everything the gate and the release report are decided from."""

    workload_windows: int = 0
    derived_rate_windows_attempted: int = 0
    derived_rate_windows_eligible: int = 0
    derived_rate_windows_unsupported: int = 0
    spectral_rows: int = 0
    audit_windows: int = 0
    audit_comparisons: int = 0
    source_replay_sample_mismatches: int = 0
    source_replay_derived_mismatches: int = 0
    source_replay_spectral_mismatches: int = 0
    maximum_bin_absolute_error: float = 0.0
    source_unreadable: int = 0
    participant_summary_rows: int = 0
    participants: int = 0
    eligible_by_rate: dict[str, int] = field(default_factory=dict)
    unsupported_by_rate: dict[str, int] = field(default_factory=dict)
    sample_count_by_rate: dict[str, dict[str, int]] = field(
        default_factory=dict
    )
    segments_derived: int = 0
    derived_samples_written: int = 0
    derived_sample_count_mismatches: int = 0
    rational_timing_ordinals_checked: int = 0
    rational_timing_mismatches: int = 0
    rounded_thirty_hz_ordinals: int = 0
    parent_ordinals_unbracketed: int = 0
    ordinals_removed_by_parent_stage: int = 0
    ordinals_admitted_over_unbracketed_parent: int = 0
    ordinals_admitted_by_filter_guard_alone: int = 0
    core_summary_rows: int = 0
    edge_summary_rows: int = 0
    rates_materialized: list[int] = field(default_factory=list)
    spectral_table_content_sha256: str = ""
    failures: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "workload_windows": self.workload_windows,
            "derived_rate_windows_attempted": (
                self.derived_rate_windows_attempted
            ),
            "derived_rate_windows_eligible": (
                self.derived_rate_windows_eligible
            ),
            "derived_rate_windows_unsupported": (
                self.derived_rate_windows_unsupported
            ),
            "spectral_rows": self.spectral_rows,
            "audit_windows": self.audit_windows,
            "audit_comparisons": self.audit_comparisons,
            "source_replay_sample_mismatches": (
                self.source_replay_sample_mismatches
            ),
            "source_replay_derived_mismatches": (
                self.source_replay_derived_mismatches
            ),
            "source_replay_spectral_mismatches": (
                self.source_replay_spectral_mismatches
            ),
            "maximum_bin_absolute_error": self.maximum_bin_absolute_error,
            "source_unreadable": self.source_unreadable,
            "participants": self.participants,
            "participant_summary_rows": self.participant_summary_rows,
            "eligible_by_rate": dict(sorted(self.eligible_by_rate.items())),
            "unsupported_by_rate": dict(
                sorted(self.unsupported_by_rate.items())
            ),
            "sample_count_by_rate": {
                rate: dict(sorted(counts.items()))
                for rate, counts in sorted(self.sample_count_by_rate.items())
            },
            "segments_derived": self.segments_derived,
            "derived_samples_written": self.derived_samples_written,
            "derived_sample_count_mismatches": (
                self.derived_sample_count_mismatches
            ),
            "rational_timing_ordinals_checked": (
                self.rational_timing_ordinals_checked
            ),
            "rational_timing_mismatches": self.rational_timing_mismatches,
            "rounded_thirty_hz_ordinals": self.rounded_thirty_hz_ordinals,
            "parent_ordinals_unbracketed": self.parent_ordinals_unbracketed,
            "ordinals_removed_by_parent_stage": (
                self.ordinals_removed_by_parent_stage
            ),
            "ordinals_admitted_over_unbracketed_parent": (
                self.ordinals_admitted_over_unbracketed_parent
            ),
            "ordinals_admitted_by_filter_guard_alone": (
                self.ordinals_admitted_by_filter_guard_alone
            ),
            "core_summary_rows": self.core_summary_rows,
            "edge_summary_rows": self.edge_summary_rows,
            "rates_materialized": sorted(self.rates_materialized),
            "spectral_table_content_sha256": (
                self.spectral_table_content_sha256
            ),
            "failure_count": len(self.failures),
            "failures": sorted(self.failures)[:64],
        }


def _band_power(power: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sum(power[mask]))


def _total_variation(derived: np.ndarray, native: np.ndarray,
                     mask: np.ndarray) -> float:
    """Half the L1 distance between the two band-normalized shapes."""

    left, right = derived[mask], native[mask]
    left_sum, right_sum = float(left.sum()), float(right.sum())
    if left_sum <= 0.0 or right_sum <= 0.0:
        return 1.0
    return float(0.5 * np.sum(np.abs(left / left_sum - right / right_sum)))


def _write_table(
    output_root: Path, name: str, records: list[dict[str, Any]]
) -> None:
    schema = P04_TABLE_SCHEMAS[name]()
    columns = {
        field_.name: [record.get(field_.name) for record in records]
        for field_ in schema
    }
    pq.write_table(
        pa.Table.from_pydict(columns, schema=schema),
        output_root / P04_TABLE_FILES[name],
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        version=PARQUET_VERSION,
        write_statistics=True,
    )


def _spectra_for(
    times_s: Sequence[float], channels: Mapping[str, Sequence[float]]
) -> dict[str, Any]:
    picoseconds = [round(value * 1e12) for value in times_s]
    out: dict[str, Any] = {}
    for family in SENSOR_FAMILIES:
        axes = [channels[name] for name in FAMILY_AXES[family]]
        out[family] = family_spectrum(
            family, picoseconds, axes,
            duration_s=WINDOW_DURATION_S, frequencies=_GRID,
        )
        out[f"{family}_input"] = input_content_sha256(picoseconds, axes)
    return out


def materialize(
    *,
    release_root: Path,
    store_root: Path,
    p03_root: Path,
    output_root: Path,
    progress: bool = False,
) -> P04Result:
    """Derive every rate for the frozen workload and audit the source path."""

    result = P04Result()
    workload = pq.read_table(
        p03_root / P03_TABLE_FILES["pads_p03_workload_windows"]
    ).to_pylist()
    native_rows = pq.read_table(
        p03_root / P03_TABLE_FILES["pads_p03_spectra"]
    ).to_pylist()
    audit_ids = {
        str(row["window_id"])
        for row in pq.read_table(
            p03_root / P03_TABLE_FILES["pads_p03_source_replay_audit"]
        ).to_pylist()
    }
    native = {
        (str(row["window_id"]), str(row["sensor_family"])): row
        for row in native_rows
    }

    windows = {
        str(row["window_id"]): row
        for row in pq.read_table(
            store_root / "pads_windows.parquet"
        ).to_pylist()
    }
    segments = {
        str(row["segment_id"]): row
        for row in pq.read_table(
            store_root / "pads_segments.parquet"
        ).to_pylist()
    }
    storage_index = {
        str(row["stream_id"]): row
        for row in pq.read_table(
            store_root / "pads_stream_storage_index.parquet"
        ).to_pylist()
    }
    participants = {
        str(row["participant_id"]): row
        for row in pq.read_table(
            store_root / "pads_participants.parquet"
        ).to_pylist()
    }
    result.participants = len(participants)

    cached_stream: str | None = None
    cached_table = None

    def segment_samples(stream_id: str, segment_id: str):
        nonlocal cached_stream, cached_table
        if cached_stream != stream_id:
            cached_table = replay_stream(store_root, storage_index, stream_id)
            cached_stream = stream_id
        assert cached_table is not None
        segment = segments[segment_id]
        entry = storage_index[stream_id]
        offset = int(segment["first_sample_ordinal"]) - int(
            entry["first_sample_ordinal"]
        )
        length = (
            int(segment["last_sample_ordinal"])
            - int(segment["first_sample_ordinal"]) + 1
        )
        sliced = cached_table.slice(offset, length)
        times = np.asarray(
            sliced.column("task_local_time_ps").to_pylist(), dtype=np.int64
        )
        channels = {
            name: np.asarray(
                sliced.column(name).to_pylist(), dtype=np.float64
            )
            for family in SENSOR_FAMILIES for name in FAMILY_AXES[family]
        }
        return times, channels

    grid_records: list[dict[str, Any]] = []
    spectra_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    per_participant: dict[tuple, list[dict[str, float]]] = {}
    seen_grids: set[tuple[str, int]] = set()

    for index, row in enumerate(workload):
        window_id = str(row["window_id"])
        window = windows.get(window_id)
        if window is None:
            result.failures.append(f"{window_id}: absent from the P0.2 index")
            continue
        stream_id = str(window["stream_id"])
        segment_id = str(window["segment_id"])
        times, channels = segment_samples(stream_id, segment_id)
        start_ps = int(window["window_start_task_local_ps"])
        end_ps = int(window["window_end_task_local_ps"])
        result.workload_windows += 1

        for rate in DERIVED_RATES_HZ:
            result.derived_rate_windows_attempted += 1
            key = str(rate)
            support = derive_support(times, rate)
            if (segment_id, rate) not in seen_grids:
                seen_grids.add((segment_id, rate))
                result.segments_derived += 1
                if rate not in result.rates_materialized:
                    result.rates_materialized.append(rate)
                grid = grid_for(rate)
                has_filter = rate in FILTER_SPECS
                taps = design(rate).size if has_filter else 0
                spec = FILTER_SPECS.get(rate)
                # What the FIR guard alone would have admitted, had it
                # believed the parent ran from the task-local origin rather
                # than from where this segment can actually bracket it.  The
                # difference is what the first stage removed, and it is the
                # evidence that the intersection is taken in that order.
                if support.parent.empty:
                    guard_only = range(0)
                    unbracketed = 0
                else:
                    guard_only = (
                        supported_output_ordinals(
                            rate, taps=taps, parent_first=0,
                            parent_last=support.parent.last_ordinal,
                        )
                        if has_filter
                        else range(support.parent.last_ordinal + 1)
                    )
                    unbracketed = support.parent.first_ordinal
                result.parent_ordinals_unbracketed += unbracketed
                result.ordinals_admitted_by_filter_guard_alone += len(
                    guard_only
                )
                result.ordinals_removed_by_parent_stage += max(
                    0, len(guard_only) - len(support.supported)
                )
                grid_records.append({
                    "segment_id": segment_id,
                    "stream_id": stream_id,
                    "participant_id": str(window["participant_id"]),
                    "task_name": str(window["task_name"]),
                    "device_location": str(window["device_location"]),
                    "rate_hz": rate,
                    "rate_hz_num": grid.rate_hz.numerator,
                    "rate_hz_den": grid.rate_hz.denominator,
                    "period_picoseconds_num": (
                        grid.period_picoseconds.numerator
                    ),
                    "period_picoseconds_den": (
                        grid.period_picoseconds.denominator
                    ),
                    "exact_in_picoseconds": grid.exact_in_picoseconds,
                    "grid_origin": grid.origin,
                    "first_ordinal": (
                        support.supported.start if support.supported else 0
                    ),
                    "last_ordinal": (
                        support.supported.stop - 1 if support.supported else -1
                    ),
                    "derived_sample_count": len(support.supported),
                    "segment_start_task_local_ps": int(
                        segments[segment_id]["start_task_local_time_ps"]
                    ),
                    "segment_end_task_local_ps": int(
                        segments[segment_id]["end_task_local_time_ps"]
                    ),
                    "parent_rate_hz": PARENT_RATE_HZ,
                    "upsample": spec.upsample if spec else 1,
                    "decimate": spec.decimate if spec else 1,
                    "working_rate_hz": (
                        spec.working_rate_hz if spec else PARENT_RATE_HZ
                    ),
                    "has_anti_alias_filter": has_filter,
                    "passband_hz": spec.passband_hz if spec else None,
                    "stopband_start_hz": (
                        spec.stopband_start_hz if spec else None
                    ),
                    "filter_taps": taps,
                    "filter_group_delay_taps": (
                        group_delay_taps(rate) if has_filter else 0
                    ),
                    "filter_sha256": (
                        filter_sha256(rate) if has_filter else ""
                    ),
                    "parent_samples": support.parent.count,
                    "parent_samples_interpolated": support.parent.count,
                    "parent_samples_unbracketed": unbracketed,
                    "first_supported_ordinal": (
                        support.supported.start if support.supported else 0
                    ),
                    "last_supported_ordinal": (
                        support.supported.stop - 1 if support.supported else -1
                    ),
                    "supported_sample_count": len(support.supported),
                    "unsupported_sample_count_at_edges": max(
                        0,
                        len(
                            grid.ordinals_covering(
                                int(
                                    segments[segment_id][
                                        "start_task_local_time_ps"
                                    ]
                                ),
                                int(
                                    segments[segment_id][
                                        "end_task_local_time_ps"
                                    ]
                                ),
                            )
                        ) - len(support.supported),
                    ),
                    "grid_status": support.status,
                })

            status, ordinals = window_eligibility(support, start_ps, end_ps)
            if status != WINDOW_ELIGIBLE:
                result.derived_rate_windows_unsupported += 1
                result.unsupported_by_rate[key] = (
                    result.unsupported_by_rate.get(key, 0) + 1
                )
                continue
            result.derived_rate_windows_eligible += 1
            result.eligible_by_rate[key] = (
                result.eligible_by_rate.get(key, 0) + 1
            )

            names = [
                name for family in SENSOR_FAMILIES
                for name in FAMILY_AXES[family]
            ]
            derived = derive_window(
                times, [channels[name] for name in names],
                rate_hz=rate, support=support, ordinals=ordinals,
            )
            derived_channels = {
                name: derived[position] for position, name in enumerate(names)
            }
            times_s = window_times_seconds(rate, ordinals)
            result.derived_samples_written += len(ordinals)
            if len(ordinals) != round(rate * WINDOW_DURATION_S):
                result.derived_sample_count_mismatches += 1
            grid = grid_for(rate)
            kernel_taps = design(rate).size if rate in FILTER_SPECS else 0
            for ordinal in ordinals:
                result.rational_timing_ordinals_checked += 1
                # Independently re-derive this output's own kernel support
                # and check it against the bracketable parent, rather than
                # trusting that derive_support produced the right set.
                if kernel_taps:
                    _, anchor, branch = polyphase_anchor(
                        rate, ordinal, taps=kernel_taps
                    )
                    if (
                        anchor - branch + 1 < support.parent.first_ordinal
                        or anchor > support.parent.last_ordinal
                    ):
                        result.ordinals_admitted_over_unbracketed_parent += 1
                elif not (
                    support.parent.first_ordinal
                    <= ordinal
                    <= support.parent.last_ordinal
                ):
                    result.ordinals_admitted_over_unbracketed_parent += 1
                if grid.sample_seconds(ordinal) != Fraction(ordinal, rate):
                    result.rational_timing_mismatches += 1
                if rate == 30 and grid.sample_picoseconds_exact(
                    ordinal
                ) is None and grid.sample_picoseconds(
                    ordinal
                ).denominator == 1:
                    result.rounded_thirty_hz_ordinals += 1
            counts = result.sample_count_by_rate.setdefault(key, {})
            counts[str(len(ordinals))] = counts.get(str(len(ordinals)), 0) + 1
            spectra = _spectra_for(times_s, derived_channels)

            for family in SENSOR_FAMILIES:
                reference = native.get((window_id, family))
                if reference is None:
                    result.failures.append(
                        f"{window_id}:{family}: no native reference")
                    continue
                spectrum = spectra[family]
                native_power = np.asarray(
                    reference["aggregate_power"], dtype=np.float64
                )
                spectra_records.append({
                    "spectral_record_id": f"{window_id}#{rate}#{family}",
                    "window_id": window_id,
                    "stream_id": stream_id,
                    "participant_id": str(window["participant_id"]),
                    "task_name": str(window["task_name"]),
                    "device_location": str(window["device_location"]),
                    "outer_fold": int(window["outer_fold"]),
                    "rate_hz": rate,
                    "sensor_family": family,
                    "aggregate_power": [
                        float(v) for v in spectrum.aggregate
                    ],
                    "normalized_aggregate_power": [
                        float(v) for v in spectrum.normalized_aggregate
                    ],
                    "dominant_frequency_hz": spectrum.dominant_frequency_hz,
                    "band_power": spectrum.band_power,
                    "core_band_power": _band_power(spectrum.aggregate, _CORE),
                    "edge_band_power": _band_power(spectrum.aggregate, _EDGE),
                    "native_dominant_frequency_hz": float(
                        reference["dominant_frequency_hz"]
                    ),
                    "native_core_band_power": _band_power(
                        native_power, _CORE
                    ),
                    "native_edge_band_power": _band_power(
                        native_power, _EDGE
                    ),
                    "dominant_frequency_shift_hz": (
                        spectrum.dominant_frequency_hz
                        - float(reference["dominant_frequency_hz"])
                    ),
                    "core_band_power_ratio": (
                        _band_power(spectrum.aggregate, _CORE)
                        / max(_band_power(native_power, _CORE), 1e-300)
                    ),
                    "edge_band_power_ratio": (
                        _band_power(spectrum.aggregate, _EDGE)
                        / max(_band_power(native_power, _EDGE), 1e-300)
                    ),
                    "core_normalized_spectral_distance": _total_variation(
                        spectrum.aggregate, native_power, _CORE
                    ),
                    "edge_normalized_spectral_distance": _total_variation(
                        spectrum.aggregate, native_power, _EDGE
                    ),
                    "derived_sample_count": len(ordinals),
                    "native_sample_count": int(reference["sample_count"]),
                    "input_content_sha256": spectra[f"{family}_input"],
                    "spectral_content_sha256": spectrum.content_sha256(),
                    "native_spectral_content_sha256": str(
                        reference["spectral_content_sha256"]
                    ),
                    "spectral_status": SPECTRUM_COMPUTED,
                })
                result.spectral_rows += 1
                bucket = per_participant.setdefault(
                    (str(window["participant_id"]), rate, family), []
                )
                bucket.append({
                    "core_ratio": spectra_records[-1][
                        "core_band_power_ratio"
                    ],
                    "edge_ratio": spectra_records[-1][
                        "edge_band_power_ratio"
                    ],
                    "core_distance": spectra_records[-1][
                        "core_normalized_spectral_distance"
                    ],
                    "edge_distance": spectra_records[-1][
                        "edge_normalized_spectral_distance"
                    ],
                    "shift": spectra_records[-1][
                        "dominant_frequency_shift_hz"
                    ],
                })

        if progress and (index + 1) % 1000 == 0:
            print(f"workload {index + 1}", file=sys.stderr, flush=True)

    # --- source-direct against replay-derived -----------------------------
    for index, row in enumerate(workload):
        window_id = str(row["window_id"])
        if window_id not in audit_ids:
            continue
        window = windows[window_id]
        stream_id = str(window["stream_id"])
        segment_id = str(window["segment_id"])
        result.audit_windows += 1
        replay_times, replay_channels = segment_samples(stream_id, segment_id)
        segment = segments[segment_id]
        try:
            source = read_source_window(
                release_root=release_root,
                participant_id=str(window["participant_id"]),
                task_name=str(window["task_name"]),
                device_location=str(window["device_location"]),
                stream_id=stream_id,
                window_start_task_local_ps=int(
                    segment["start_task_local_time_ps"]
                ),
                window_end_task_local_ps=int(
                    segment["end_task_local_time_ps"]
                ) + 1,
                expected_asset_sha256=str(
                    storage_index[stream_id]["source_asset_sha256"]
                ),
            )
        except Exception as exc:  # noqa: BLE001 - any failure is evidence
            result.source_unreadable += 1
            result.failures.append(f"{window_id}: {exc}")
            continue
        source_times = np.asarray(source.times_ps, dtype=np.int64) - int(
            source.times_ps[0]
        ) + int(segment["start_task_local_time_ps"])
        source_channels = {
            name: np.asarray(source.channels[name], dtype=np.float64)
            for family in SENSOR_FAMILIES for name in FAMILY_AXES[family]
        }

        start_ps = int(window["window_start_task_local_ps"])
        end_ps = int(window["window_end_task_local_ps"])
        names = [
            name for family in SENSOR_FAMILIES
            for name in FAMILY_AXES[family]
        ]
        for rate in DERIVED_RATES_HZ:
            replay_support = derive_support(replay_times, rate)
            status, ordinals = window_eligibility(
                replay_support, start_ps, end_ps
            )
            if status != WINDOW_ELIGIBLE:
                continue
            source_support = derive_support(source_times, rate)
            source_status, source_ordinals = window_eligibility(
                source_support, start_ps, end_ps
            )
            result.audit_comparisons += 1
            if source_status != status or list(source_ordinals) != list(
                ordinals
            ):
                result.source_replay_sample_mismatches += 1
                audit_records.append(_audit_row(
                    window_id, stream_id, rate, row,
                    storage_index[stream_id], len(source_ordinals),
                    len(ordinals), "", "", {},
                    float("inf"), False,
                    PRESERVATION_SAMPLE_COUNT_MISMATCH,
                ))
                continue
            replay_values = derive_window(
                replay_times, [replay_channels[n] for n in names],
                rate_hz=rate, support=replay_support, ordinals=ordinals,
            )
            source_values = derive_window(
                source_times, [source_channels[n] for n in names],
                rate_hz=rate, support=source_support, ordinals=source_ordinals,
            )
            replay_sha = hashlib.sha256(
                np.ascontiguousarray(replay_values).tobytes()
            ).hexdigest()
            source_sha = hashlib.sha256(
                np.ascontiguousarray(source_values).tobytes()
            ).hexdigest()
            times_s = window_times_seconds(rate, ordinals)
            replay_spectra = _spectra_for(
                times_s,
                {n: replay_values[i] for i, n in enumerate(names)},
            )
            source_spectra = _spectra_for(
                times_s,
                {n: source_values[i] for i, n in enumerate(names)},
            )
            worst = 0.0
            dominant_match = True
            hashes: dict[str, str] = {}
            for family in SENSOR_FAMILIES:
                key = "gyro" if family == "GYROSCOPE" else "accel"
                hashes[f"replay_{key}"] = replay_spectra[
                    family
                ].content_sha256()
                hashes[f"source_{key}"] = source_spectra[
                    family
                ].content_sha256()
                worst = max(worst, float(np.max(np.abs(
                    replay_spectra[family].aggregate
                    - source_spectra[family].aggregate
                ))))
                dominant_match &= (
                    replay_spectra[family].dominant_frequency_hz
                    == source_spectra[family].dominant_frequency_hz
                )
            preservation = PRESERVED
            if source_sha != replay_sha:
                result.source_replay_derived_mismatches += 1
                preservation = PRESERVATION_DERIVED_MISMATCH
            if any(
                hashes[f"source_{key}"] != hashes[f"replay_{key}"]
                for key in ("gyro", "accel")
            ):
                result.source_replay_spectral_mismatches += 1
                preservation = PRESERVATION_SPECTRUM_MISMATCH
            result.maximum_bin_absolute_error = max(
                result.maximum_bin_absolute_error, worst
            )
            audit_records.append(_audit_row(
                window_id, stream_id, rate, row,
                storage_index[stream_id], len(source_ordinals), len(ordinals),
                source_sha, replay_sha, hashes, worst, dominant_match,
                preservation,
            ))
        if progress and (index + 1) % 2000 == 0:
            print(f"audit {index + 1}", file=sys.stderr, flush=True)

    summary_records = _summarize(per_participant, participants)
    result.participant_summary_rows = len(summary_records)
    result.core_summary_rows = sum(
        1 for row in summary_records if row["band"] == CORE_BAND
    )
    result.edge_summary_rows = sum(
        1 for row in summary_records if row["band"] == EDGE_BAND
    )

    grid_records.sort(key=lambda item: (item["segment_id"], item["rate_hz"]))
    spectra_records.sort(
        key=lambda item: (
            item["window_id"], item["rate_hz"], item["sensor_family"]
        )
    )
    audit_records.sort(key=lambda item: (item["window_id"], item["rate_hz"]))
    digest = hashlib.sha256()
    for record in spectra_records:
        digest.update(record["spectral_record_id"].encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(record["spectral_content_sha256"].encode("ascii"))
        digest.update(b"\x1e")
    result.spectral_table_content_sha256 = digest.hexdigest()

    output_root.mkdir(parents=True, exist_ok=True)
    _write_table(output_root, "pads_p04_rate_grids", grid_records)
    _write_table(output_root, "pads_p04_rate_spectra", spectra_records)
    _write_table(
        output_root, "pads_p04_source_replay_audit", audit_records
    )
    _write_table(
        output_root, "pads_p04_participant_summary", summary_records
    )
    return result


def _audit_row(
    window_id: str, stream_id: str, rate: int, row: Mapping[str, Any],
    entry: Mapping[str, Any], source_count: int, replay_count: int,
    source_sha: str, replay_sha: str, hashes: Mapping[str, str],
    worst: float, dominant_match: bool, preservation: str,
) -> dict[str, Any]:
    return {
        "window_id": window_id,
        "stream_id": stream_id,
        "rate_hz": rate,
        "stratum_id": str(row.get("gap_adjacent_status", "")),
        "source_asset_sha256": str(entry["source_asset_sha256"]),
        "source_derived_sample_count": source_count,
        "replay_derived_sample_count": replay_count,
        "source_derived_sha256": source_sha,
        "replay_derived_sha256": replay_sha,
        "source_gyro_spectrum_sha256": hashes.get("source_gyro", ""),
        "replay_gyro_spectrum_sha256": hashes.get("replay_gyro", ""),
        "source_accel_spectrum_sha256": hashes.get("source_accel", ""),
        "replay_accel_spectrum_sha256": hashes.get("replay_accel", ""),
        "maximum_bin_absolute_error": worst,
        "dominant_frequency_match": dominant_match,
        "preservation_status": preservation,
    }


def _summarize(
    per_participant: Mapping[tuple, list[dict[str, float]]],
    participants: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (participant_id, rate, family), rows in sorted(
        per_participant.items()
    ):
        meta = participants.get(participant_id, {})
        for band, ratio_key, distance_key in (
            (CORE_BAND, "core_ratio", "core_distance"),
            (EDGE_BAND, "edge_ratio", "edge_distance"),
        ):
            shifts = [row["shift"] for row in rows]
            records.append({
                "participant_id": participant_id,
                "condition_group": str(meta.get("condition_group", "")),
                "outer_fold": int(meta.get("outer_fold", -1)),
                "rate_hz": rate,
                "sensor_family": family,
                "band": band,
                "windows": len(rows),
                "median_band_power_ratio": statistics.median(
                    row[ratio_key] for row in rows
                ),
                "median_normalized_spectral_distance": statistics.median(
                    row[distance_key] for row in rows
                ),
                "median_dominant_frequency_shift_hz": statistics.median(
                    shifts
                ),
                "windows_with_preserved_dominant_frequency": sum(
                    1 for value in shifts if value == 0.0
                ),
                "summary_status": SUMMARY_COMPUTED,
            })
    return records


NATIVE_LABEL = NATIVE_RATE_LABEL

__all__ = [
    "NATIVE_LABEL",
    "PRESERVATION_DERIVED_MISMATCH",
    "PRESERVATION_SAMPLE_COUNT_MISMATCH",
    "PRESERVATION_SOURCE_UNREADABLE",
    "PRESERVATION_SPECTRUM_MISMATCH",
    "PRESERVED",
    "SPECTRUM_COMPUTED",
    "SUMMARY_COMPUTED",
    "P04Result",
    "materialize",
]
