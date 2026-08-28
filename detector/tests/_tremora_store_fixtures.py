"""Small synthetic storage fixtures; never empirical validation evidence."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pyarrow as pa

from motionbloom.tremora_store.alignment_index import build_frame_imu_index
from motionbloom.tremora_store.clock_map import ClockSegment, PiecewiseClockMap
from motionbloom.tremora_store.parquet_writer import RecordingStoreWriter
from motionbloom.tremora_store.schema import (
    clock_map_schema,
    cv_estimates_schema,
    frame_imu_index_schema,
    frame_index_schema,
    imu_samples_schema,
    window_index_schema,
    window_rejections_schema,
)
from motionbloom.tremora_store.window_index import (
    ContinuitySegment,
    build_window_index,
)

NS = 1_000_000_000


def synthetic_frequency_signal_policies():
    return [{
        "recording_id": "rec-001",
        "video_stream_id": "video-0",
        "imu_stream_id": "imu-wrist-0",
        "cv_motion_min_peak_to_peak_stored_units": 0.01,
        "acceleration_min_peak_to_peak_stored_units": 0.01,
        "angular_velocity_min_peak_to_peak_stored_units": 0.01,
        "quaternion_min_angular_range_rad": None,
        "minimum_varying_cv_components": 1,
        "minimum_varying_imu_channels": 1,
    }]


def frame_table(*, duration_s=4, fps=30, recording_id="rec-001",
                video_stream_id="video-0"):
    count = duration_s * fps
    times = [index * NS // fps for index in range(count)]
    rows = [{
        "recording_id": recording_id,
        "video_stream_id": video_stream_id,
        "clock_epoch_id": "video-epoch-0",
        "frame_index": 1000 + index * 2,
        "source_ordinal": index,
        "canonical_ordinal": index,
        "video_pts_native_ns": time,
        "canonical_time_ns": time,
        "decode_status": "OK",
        "width": 640,
        "height": 480,
        "effective_fps": float(fps),
        "gap_before_ms": None if index == 0 else 1000.0 / fps,
        "quality_bits": 0,
    } for index, time in enumerate(times)]
    return pa.Table.from_pylist(rows, schema=frame_index_schema())


def cv_table(frames):
    rows = []
    for frame in frames.to_pylist():
        ordinal = frame["canonical_ordinal"]
        rows.append({
            "recording_id": frame["recording_id"],
            "video_stream_id": frame["video_stream_id"],
            "frame_index": frame["frame_index"],
            "canonical_ordinal": ordinal,
            "canonical_time_ns": frame["canonical_time_ns"],
            "keypoints": [float(ordinal % 7)] * 63,
            "keypoint_validity": [True] * 21,
            "motion_vector": [
                -0.0 if ordinal == 0 else ordinal / 1000.0,
                0.2 + ordinal / 2000.0,
                0.3 + ordinal / 3000.0,
            ],
            "palm_orientation": [1.0, 0.0, 0.0, 0.0],
            "hand_scale": 80.0,
            "estimated_frequency_hz": 5.0,
            "tracking_quality": 0.95,
            "estimator_version": "synthetic-fixture-v1",
        })
    return pa.Table.from_pylist(rows, schema=cv_estimates_schema())


def imu_table(*, duration_s=4, hz=100, recording_id="rec-001",
              stream_id="imu-wrist-0"):
    count = duration_s * hz
    rows = []
    for index in range(count):
        time = index * NS // hz
        rows.append({
            "recording_id": recording_id,
            "stream_id": stream_id,
            "clock_epoch_id": "imu-epoch-0",
            "sample_index": 5000 + index * 3,
            "source_ordinal": index,
            "canonical_ordinal": index,
            "sensor_time_native_ns": time,
            "canonical_time_ns": time,
            "payload_kind": "ACCEL_GYRO",
            "ax": -0.0 if index == 0 else index / 1000.0,
            "ay": 0.0,
            "az": 1.0,
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
            "qw": None,
            "qx": None,
            "qy": None,
            "qz": None,
            "validity_bits": 0,
        })
    return pa.Table.from_pylist(rows, schema=imu_samples_schema())


def clock_table(*, duration_s=4, fps=30, imu_hz=100,
                recording_id="rec-001"):
    segments = [
        ClockSegment(
            recording_id=recording_id, stream_id="video-0",
            clock_epoch_id="video-epoch-0", continuity_component_id="component-0",
            acquisition_ordinal=0, source_start_ordinal=0,
            source_stop_ordinal=duration_s * fps, native_start_ns=0,
            native_end_ns=duration_s * NS, native_anchor_ns=0,
            canonical_anchor_ns=0, scale_numerator=1, scale_denominator=1,
            residual_p50_ms=0.0, residual_p95_ms=0.0,
        ),
        ClockSegment(
            recording_id=recording_id, stream_id="imu-wrist-0",
            clock_epoch_id="imu-epoch-0", continuity_component_id="component-0",
            acquisition_ordinal=0, source_start_ordinal=0,
            source_stop_ordinal=duration_s * imu_hz, native_start_ns=0,
            native_end_ns=duration_s * NS, native_anchor_ns=0,
            canonical_anchor_ns=0, scale_numerator=1, scale_denominator=1,
            residual_p50_ms=0.0, residual_p95_ms=0.0,
        ),
    ]
    return PiecewiseClockMap(segments).to_table()


def all_tables(*, duration_s=4):
    frames = frame_table(duration_s=duration_s)
    cv = cv_table(frames)
    imu = imu_table(duration_s=duration_s)
    alignment = build_frame_imu_index(
        recording_id="rec-001", video_stream_id="video-0",
        imu_stream_id="imu-wrist-0",
        frame_indices=frames["frame_index"].to_pylist(),
        frame_canonical_ordinals=frames["canonical_ordinal"].to_pylist(),
        frame_times_ns=frames["canonical_time_ns"].to_pylist(),
        imu_canonical_ordinals=imu["canonical_ordinal"].to_pylist(),
        imu_times_ns=imu["canonical_time_ns"].to_pylist(),
        video_end_ns=duration_s * NS,
        continuity_intervals_ns=[(0, duration_s * NS)],
    )
    windows = build_window_index(
        frame_index=frames, cv_estimates=cv, imu_samples=imu,
        frame_imu_index=alignment,
        continuity_segments=[ContinuitySegment(
            segment_id="segment-0", recording_id="rec-001",
            video_stream_id="video-0", imu_stream_id="imu-wrist-0",
            start_time_ns=0, end_time_ns=duration_s * NS,
            split_group_id="public-fixture-001")],
        window_ns=4 * NS, hop_ns=NS, window_policy_id="4s-1s-v1",
        observability_policy_id="tremor-observability-v1",
        tremor_band_low_hz=3.0, tremor_band_high_hz=8.0,
        frequency_signal_policies=synthetic_frequency_signal_policies(),
    )
    return {
        "frame_index": (frames, frame_index_schema(),
                        ("recording_id", "video_stream_id", "canonical_ordinal")),
        "cv_estimates": (cv, cv_estimates_schema(),
                         ("recording_id", "video_stream_id", "canonical_ordinal")),
        "imu_samples": (imu, imu_samples_schema(),
                        ("recording_id", "stream_id", "canonical_ordinal")),
        "clock_map": (clock_table(duration_s=duration_s), clock_map_schema(),
                      ("recording_id", "stream_id", "acquisition_ordinal")),
        "frame_imu_index": (alignment, frame_imu_index_schema(),
                            ("recording_id", "video_stream_id", "imu_stream_id",
                             "frame_canonical_ordinal")),
        "window_index": (windows.valid_index, window_index_schema(),
                         ("window_id",)),
        "window_rejections": (windows.rejection_ledger,
                              window_rejections_schema(), ("candidate_window_id",)),
    }


def synthetic_provenance():
    return {
        "provenance_schema_version": "1.0",
        "source_kind": "SYNTHETIC_GENERATED",
        "source_dataset": "SYNTHETIC_TEST_FIXTURE",
        "source_dataset_version": "v1",
        "source_recording_id": "rec-001",
        "recording_identity": {
            "stored_recording_id": "rec-001",
            "source_recording_id": "rec-001",
            "mapping_method": "EXACT_SOURCE_IDENTIFIER",
        },
        "source_record_uri": "urn:tremora:synthetic:rec-001",
        "source_file_hashes": {
            "canonical_fixture": sha256(
                b"tremora-synthetic-fixture-v1").hexdigest(),
        },
        "license_id": "NOT_APPLICABLE_SYNTHETIC",
        "license_uri": "urn:tremora:license:not-applicable-synthetic",
        "license_terms_sha256": sha256(
            b"not applicable: generated synthetic test fixture").hexdigest(),
        "source_access_status": "GENERATED_LOCALLY",
        "source_redistribution_status": "GENERATED_TEST_DATA_ONLY",
        "local_analysis_allowed": True,
        "source_redistribution_allowed": True,
        "derived_artifact_release_allowed": True,
        "derived_artifact_policy": "TEST_ARTIFACTS_MUST_NOT_BE_EMPIRICAL_EVIDENCE",
        "permitted_use": "SOFTWARE_TESTING_ONLY",
        "validation_role": "SYNTHETIC_REGRESSION_ONLY",
        "use_decision": "ALLOW_ANALYSIS_AND_RELEASE",
        "artifact_release_status": "RELEASABLE",
        "ingestion_commit": "0" * 40,
        "ingestion_software_version": "tremora-store-v0.1",
        "cv_estimator_version": "synthetic-fixture-v1",
        "observability_policy_id": "tremor-observability-v1",
        "stream_semantics": {
            "schema_version": "1.0",
            "video_streams": [{
                "recording_id": "rec-001",
                "video_stream_id": "video-0",
                "source_keypoint_convention": (
                    "SYNTHETIC_21_XYZ_TRIPLES_UNITLESS"),
                "stored_keypoint_convention": (
                    "SYNTHETIC_21_XYZ_TRIPLES_UNITLESS"),
                "source_motion_vector_convention": (
                    "SYNTHETIC_DX_DY_MAGNITUDE_UNITLESS"),
                "stored_motion_vector_convention": (
                    "SYNTHETIC_DX_DY_MAGNITUDE_UNITLESS"),
                "source_palm_orientation_convention": (
                    "SYNTHETIC_FOUR_COMPONENT_WXYZ"),
                "stored_palm_orientation_convention": (
                    "SYNTHETIC_FOUR_COMPONENT_WXYZ"),
                "source_hand_scale_convention": (
                    "SYNTHETIC_POSITIVE_UNITLESS"),
                "stored_hand_scale_convention": (
                    "SYNTHETIC_POSITIVE_UNITLESS"),
                "canonicalization_transform_id": "IDENTITY_SOURCE_NATIVE",
                "canonicalization_software_version": "synthetic-fixture-v1",
            }],
            "imu_streams": [{
                "recording_id": "rec-001",
                "stream_id": "imu-wrist-0",
                "body_location": "SYNTHETIC_WRIST",
                "payload_kind": "ACCEL_GYRO",
                "source_acceleration_unit": "SYNTHETIC_UNITLESS",
                "stored_acceleration_unit": "SYNTHETIC_UNITLESS",
                "source_angular_velocity_unit": "SYNTHETIC_UNITLESS",
                "stored_angular_velocity_unit": "SYNTHETIC_UNITLESS",
                "source_quaternion_convention": "NOT_PRESENT",
                "stored_quaternion_convention": "NOT_PRESENT",
                "source_device_frame_convention": "SYNTHETIC_DEVICE_FRAME",
                "stored_device_frame_convention": "SYNTHETIC_DEVICE_FRAME",
                "canonicalization_transform_id": "IDENTITY_SOURCE_NATIVE",
                "canonicalization_software_version": "synthetic-fixture-v1",
            }],
        },
        "alignment_generation_parameters": {
            "pairs": [{
                "recording_id": "rec-001",
                "video_stream_id": "video-0",
                "imu_stream_id": "imu-wrist-0",
                "video_end_ns": 4 * NS,
                "max_imu_gap_ns": None,
                "min_coverage_fraction": 0.8,
                "max_clock_residual_p95_ms": 5.0,
            }],
        },
        "window_generation_parameters": {
            "window_ns": 4 * NS,
            "hop_ns": NS,
            "tremor_band_low_hz": 3.0,
            "tremor_band_high_hz": 8.0,
            "min_video_coverage": 0.9,
            "min_imu_coverage": 0.9,
            "max_video_gap_ns": None,
            "max_imu_gap_ns": None,
            "video_observability_factor": 0.4,
            "video_observability_cap_hz": 12.0,
            "min_frequency_cycles": 3.0,
            "max_cadence_deviation_fraction": 0.2,
            "min_tracking_quality": 0.1,
            "min_valid_keypoint_fraction": 0.5,
            "frequency_signal_policies": (
                synthetic_frequency_signal_policies()),
            "continuity_segments": [{
                "segment_id": "segment-0",
                "recording_id": "rec-001",
                "video_stream_id": "video-0",
                "imu_stream_id": "imu-wrist-0",
                "start_time_ns": 0,
                "end_time_ns": 4 * NS,
                "split_group_id": "public-fixture-001",
                "accepted": True,
                "quality_bits": 0,
            }],
        },
    }


def store_writer(root: Path, *, snapshot_id="snapshot-001", row_group_size=64,
                 provenance=None, window_policy_id="4s-1s-v1",
                 created_at_utc="2026-08-25T00:00:00Z"):
    return RecordingStoreWriter(
        root, snapshot_id=snapshot_id, recording_id="rec-001",
        created_at_utc=created_at_utc, clock_map_id="clock-map-v1",
        window_policy_id=window_policy_id,
        provenance=synthetic_provenance() if provenance is None else provenance,
        row_group_size=row_group_size,
    )


def write_store(root: Path, *, snapshot_id="snapshot-001", row_group_size=64):
    writer = store_writer(
        root, snapshot_id=snapshot_id, row_group_size=row_group_size)
    for name, (table, schema, keys) in all_tables().items():
        writer.write_table(name, table, schema=schema, sort_keys=keys)
    return writer.commit()
