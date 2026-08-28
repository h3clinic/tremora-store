"""Fail-closed legal provenance and official-layout tests for VIDIMU v2.0.0."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.compute as pc

from motionbloom.tremora_store.adapters.vidimu import (
    VIDIMU_CONCEPT_DOI,
    VIDIMU_DATASET_SUBTREE,
    VIDIMU_LICENSE,
    VIDIMU_LICENSE_SPDX,
    VIDIMU_RECORD_DOI,
    VIDIMU_RECORD_URL,
    VIDIMU_RELEASE_VERSION,
    VIDIMU_VALIDATION_ROLE,
    VIDIMU_VIDEO_SUBTREE,
    VIDIMU_ZENODO_RECORD_ID,
    VidimuAdapter,
    VidimuAdapterError,
    VidimuRecording,
)
from motionbloom.tremora_store.alignment_index import build_frame_imu_index
from motionbloom.tremora_store.clock_map import ClockSegment, PiecewiseClockMap
from motionbloom.tremora_store.integrity import (
    StoreInvariantError,
    validate_provenance,
)
from motionbloom.tremora_store.parquet_writer import SnapshotError, sha256_file
from motionbloom.tremora_store.schema import QualityBits
from motionbloom.tremora_store.window_index import (
    ContinuitySegment,
    build_window_index,
)
from tests._tremora_store_fixtures import (
    NS,
    all_tables,
    store_writer,
    synthetic_provenance,
)


def _official_layout(
    dataset_archive_root: Path,
    video_archive_root: Path,
    stem: str = "S40_A01_T01",
) -> dict[str, Path]:
    subject = stem.split("_", maxsplit=1)[0]
    data_subject = dataset_archive_root / VIDIMU_DATASET_SUBTREE / subject
    video_subject = video_archive_root / VIDIMU_VIDEO_SUBTREE / subject
    data_subject.mkdir(parents=True, exist_ok=True)
    video_subject.mkdir(parents=True, exist_ok=True)
    paths = {
        "video": video_subject / f"{stem}_pose.mp4",
        "video_pose": data_subject / f"{stem}.csv",
        "imu_quaternion": data_subject / f"{stem}.raw",
    }
    for label, path in paths.items():
        path.write_bytes(f"{label}:{stem}".encode())
    return paths


def _adapter(
    dataset_archive_root: Path,
    video_archive_root: Path,
    *,
    video_archive: str = "videosmallsize",
    terms_sha256: str = "a" * 64,
) -> VidimuAdapter:
    return VidimuAdapter(
        dataset_archive_root,
        video_archive_root=video_archive_root,
        video_archive=video_archive,
        dataset_archive_sha256="b" * 64,
        video_archive_sha256="c" * 64,
        inventory_scope="EXTRACTED_SUBSET_NO_RELEASE_COMPLETENESS_CLAIM",
        terms_sha256=terms_sha256,
    )


def _canonical_metadata_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _recompute_vidimu_identity(provenance: dict[str, object]) -> None:
    selection = provenance["source_archive_selection"]
    ledger = provenance["source_exclusion_ledger"]
    provenance["source_archive_selection_sha256"] = (
        _canonical_metadata_sha256(selection))
    provenance["source_exclusion_ledger_sha256"] = (
        _canonical_metadata_sha256(ledger))
    provenance["source_identity_sha256"] = _canonical_metadata_sha256({
        "archive_selection": selection,
        "excluded_files": ledger,
        "excluded_files_sha256": (
            provenance["source_exclusion_ledger_sha256"]),
        "files": provenance["source_files"],
    })


def _bind_internal_recording(
    provenance: dict[str, object], *, stored_recording_id: str = "rec-001",
) -> dict[str, object]:
    provenance["recording_identity"] = {
        "stored_recording_id": stored_recording_id,
        "source_recording_id": provenance["source_recording_id"],
        "mapping_method": "EXPLICIT_INTERNAL_TO_SOURCE_MAPPING",
    }
    for segment in provenance["window_generation_parameters"][
            "continuity_segments"]:
        segment["accepted"] = False
        segment["quality_bits"] |= int(QualityBits.UNRESOLVED_CLOCK_MAP)
        segment["split_group_id"] = provenance["source_recording_id"].split(
            "_", maxsplit=1)[0]
    return provenance


def _unresolved_vidimu_fixture(adapter_provenance: dict[str, object]):
    """Build a synthetic five-stream inventory state, never empirical evidence."""

    tables = all_tables()
    frames, frame_schema, frame_keys = tables["frame_index"]
    frame_rows = frames.to_pylist()
    for row in frame_rows:
        row["quality_bits"] |= int(QualityBits.UNRESOLVED_CLOCK_MAP)
    frames = pa.Table.from_pylist(frame_rows, schema=frame_schema)
    tables["frame_index"] = (frames, frame_schema, frame_keys)

    base_imu, imu_schema, imu_keys = tables["imu_samples"]
    source_rows = base_imu.to_pylist()[::2]
    imu_rows = []
    stream_ids = [f"vidimu-quaternion-{index}" for index in range(5)]
    for stream_index, stream_id in enumerate(stream_ids):
        for ordinal, source in enumerate(source_rows):
            imu_rows.append({
                **source,
                "stream_id": stream_id,
                "clock_epoch_id": f"{stream_id}-unresolved-epoch",
                "sample_index": ordinal,
                "source_ordinal": ordinal,
                "canonical_ordinal": ordinal,
                "payload_kind": "QUATERNION",
                "ax": None, "ay": None, "az": None,
                "gx": None, "gy": None, "gz": None,
                "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
                "validity_bits": int(QualityBits.UNRESOLVED_CLOCK_MAP),
            })
    imu = pa.Table.from_pylist(imu_rows, schema=imu_schema)
    tables["imu_samples"] = (imu, imu_schema, imu_keys)

    clock_segments = [ClockSegment(
        recording_id="rec-001", stream_id="video-0",
        clock_epoch_id="video-epoch-0",
        continuity_component_id="video-unresolved",
        acquisition_ordinal=0, source_start_ordinal=0,
        source_stop_ordinal=frames.num_rows, native_start_ns=0,
        native_end_ns=4 * NS, native_anchor_ns=0,
        canonical_anchor_ns=0, scale_numerator=1, scale_denominator=1,
        mapping_status="UNRESOLVED",
    )]
    for stream_id in stream_ids:
        clock_segments.append(ClockSegment(
            recording_id="rec-001", stream_id=stream_id,
            clock_epoch_id=f"{stream_id}-unresolved-epoch",
            continuity_component_id=f"{stream_id}-unresolved",
            acquisition_ordinal=0, source_start_ordinal=0,
            source_stop_ordinal=len(source_rows), native_start_ns=0,
            native_end_ns=4 * NS, native_anchor_ns=0,
            canonical_anchor_ns=0, scale_numerator=1, scale_denominator=1,
            mapping_status="UNRESOLVED",
        ))
    _, clock_schema, clock_keys = tables["clock_map"]
    tables["clock_map"] = (
        PiecewiseClockMap(clock_segments).to_table(), clock_schema, clock_keys)

    alignments = []
    segments = []
    policies = []
    pairs = []
    for stream_id in stream_ids:
        stream_imu = imu.filter(pc.equal(
            imu["stream_id"], pa.scalar(stream_id)))
        alignments.append(build_frame_imu_index(
            recording_id="rec-001", video_stream_id="video-0",
            imu_stream_id=stream_id,
            frame_indices=frames["frame_index"].to_pylist(),
            frame_canonical_ordinals=frames["canonical_ordinal"].to_pylist(),
            frame_times_ns=frames["canonical_time_ns"].to_pylist(),
            imu_canonical_ordinals=(
                stream_imu["canonical_ordinal"].to_pylist()),
            imu_times_ns=stream_imu["canonical_time_ns"].to_pylist(),
            video_end_ns=4 * NS, continuity_intervals_ns=[],
        ))
        pairs.append({
            "recording_id": "rec-001", "video_stream_id": "video-0",
            "imu_stream_id": stream_id, "video_end_ns": 4 * NS,
            "max_imu_gap_ns": None, "min_coverage_fraction": 0.8,
            "max_clock_residual_p95_ms": 5.0,
        })
        segments.append(ContinuitySegment(
            segment_id=f"{stream_id}-all-unresolved",
            recording_id="rec-001", video_stream_id="video-0",
            imu_stream_id=stream_id, start_time_ns=0, end_time_ns=4 * NS,
            split_group_id="S40", accepted=False,
            quality_bits=int(QualityBits.UNRESOLVED_CLOCK_MAP),
        ))
        policies.append({
            "recording_id": "rec-001", "video_stream_id": "video-0",
            "imu_stream_id": stream_id,
            "cv_motion_min_peak_to_peak_stored_units": 0.01,
            "acceleration_min_peak_to_peak_stored_units": None,
            "angular_velocity_min_peak_to_peak_stored_units": None,
            "quaternion_min_angular_range_rad": 0.01,
            "minimum_varying_cv_components": 1,
            "minimum_varying_imu_channels": 1,
        })
    alignment = pa.concat_tables(alignments)
    _, alignment_schema, alignment_keys = tables["frame_imu_index"]
    tables["frame_imu_index"] = (
        alignment, alignment_schema, alignment_keys)

    cv = tables["cv_estimates"][0]
    windows = build_window_index(
        frame_index=frames, cv_estimates=cv, imu_samples=imu,
        frame_imu_index=alignment, continuity_segments=segments,
        window_ns=4 * NS, hop_ns=NS, window_policy_id="4s-1s-v1",
        observability_policy_id="tremor-observability-v1",
        tremor_band_low_hz=3.0, tremor_band_high_hz=8.0,
        frequency_signal_policies=policies,
    )
    _, valid_schema, valid_keys = tables["window_index"]
    _, rejection_schema, rejection_keys = tables["window_rejections"]
    tables["window_index"] = (
        windows.valid_index, valid_schema, valid_keys)
    tables["window_rejections"] = (
        windows.rejection_ledger, rejection_schema, rejection_keys)

    provenance = _bind_internal_recording({
        **synthetic_provenance(), **adapter_provenance,
    })
    provenance["alignment_generation_parameters"] = {"pairs": pairs}
    provenance["window_generation_parameters"][
        "frequency_signal_policies"] = policies
    provenance["window_generation_parameters"]["continuity_segments"] = [
        {
            "segment_id": segment.segment_id,
            "recording_id": segment.recording_id,
            "video_stream_id": segment.video_stream_id,
            "imu_stream_id": segment.imu_stream_id,
            "start_time_ns": segment.start_time_ns,
            "end_time_ns": segment.end_time_ns,
            "split_group_id": segment.split_group_id,
            "accepted": segment.accepted,
            "quality_bits": segment.quality_bits,
        }
        for segment in segments
    ]
    provenance["stream_semantics"]["imu_streams"] = [{
        "recording_id": "rec-001", "stream_id": stream_id,
        "body_location": f"VIDIMU_RELEASE_SENSOR_SLOT_{index}",
        "payload_kind": "QUATERNION",
        "source_acceleration_unit": "NOT_PRESENT",
        "stored_acceleration_unit": "NOT_PRESENT",
        "source_angular_velocity_unit": "NOT_PRESENT",
        "stored_angular_velocity_unit": "NOT_PRESENT",
        "source_quaternion_convention": "TEST_FIXTURE_WXYZ_UNIT_QUATERNION",
        "stored_quaternion_convention": "TEST_FIXTURE_WXYZ_UNIT_QUATERNION",
        "source_device_frame_convention": f"TEST_FIXTURE_SENSOR_FRAME_{index}",
        "stored_device_frame_convention": f"TEST_FIXTURE_SENSOR_FRAME_{index}",
        "canonicalization_transform_id": "IDENTITY_SOURCE_NATIVE",
        "canonicalization_software_version": "synthetic-fixture-v1",
    } for index, stream_id in enumerate(stream_ids)]
    return tables, provenance


class TestVidimuAdapter(unittest.TestCase):
    def test_split_official_roots_emit_hashed_selection_and_legal_provenance(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            expected_paths = _official_layout(dataset_root, video_root)
            terms = dataset_root / "CC-BY-4.0.txt"
            terms.write_text("fixture license terms", encoding="utf-8")
            expected_terms_hash = hashlib.sha256(terms.read_bytes()).hexdigest()

            adapter = VidimuAdapter(
                dataset_root,
                video_archive_root=video_root,
                video_archive="videosmallsize",
                dataset_archive_sha256="b" * 64,
                video_archive_sha256="c" * 64,
                inventory_scope=(
                    "EXTRACTED_SUBSET_NO_RELEASE_COMPLETENESS_CLAIM"),
                terms_path=terms,
            )
            recording = adapter.discover()[0]
            provenance = adapter.provenance(recording)

            for label, path in expected_paths.items():
                expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(
                    provenance["source_file_hashes"][label], expected_hash)
                self.assertEqual(
                    provenance["source_files"][label]["sha256"], expected_hash)

        self.assertTrue(recording.inventory_complete)
        self.assertFalse(hasattr(recording, "complete_aligned_fixture"))
        self.assertEqual(recording.video_path.name, "S40_A01_T01_pose.mp4")
        self.assertEqual(provenance["source_dataset_version"], "2.0.0")
        self.assertEqual(provenance["source_record_id"], 15_075_076)
        self.assertEqual(provenance["source_record_url"], VIDIMU_RECORD_URL)
        self.assertEqual(provenance["source_record_doi"], VIDIMU_RECORD_DOI)
        self.assertEqual(provenance["source_concept_doi"], VIDIMU_CONCEPT_DOI)
        self.assertEqual(provenance["source_license"], "CC BY 4.0")
        self.assertEqual(provenance["source_license_spdx"], "CC-BY-4.0")
        self.assertEqual(provenance["source_terms_sha256"], expected_terms_hash)
        self.assertEqual(provenance["provenance_schema_version"], "1.0")
        self.assertEqual(provenance["source_kind"], "PUBLIC_DATASET")
        self.assertEqual(provenance["source_record_uri"], VIDIMU_RECORD_URL)
        self.assertEqual(provenance["license_id"], "CC-BY-4.0")
        self.assertEqual(provenance["license_terms_sha256"], expected_terms_hash)
        self.assertEqual(
            provenance["source_terms_hash_origin"], "COMPUTED_FROM_FILE")
        self.assertEqual(
            provenance["redistribution_status"], "PERMITTED_WITH_ATTRIBUTION")
        self.assertTrue(provenance["local_analysis_allowed"])
        self.assertTrue(provenance["source_redistribution_allowed"])
        self.assertTrue(provenance["derived_artifact_release_allowed"])
        self.assertEqual(provenance["use_decision"], "ALLOW_ANALYSIS_AND_RELEASE")
        self.assertEqual(provenance["artifact_release_status"], "RELEASABLE")
        self.assertEqual(
            provenance["allowed_validation_role"],
            "INVENTORY_ONLY_UNVERIFIED_SYNC",
        )
        self.assertEqual(
            set(provenance["source_file_hashes"]),
            {"video", "video_pose", "imu_quaternion"},
        )
        selection = provenance["source_archive_selection"]
        self.assertEqual(selection["dataset_archive"], "dataset")
        self.assertEqual(selection["dataset_archive_sha256"], "b" * 64)
        self.assertEqual(selection["dataset_subtree"], "dataset/videoandimus")
        self.assertEqual(selection["video_archive"], "videosmallsize")
        self.assertEqual(selection["video_archive_sha256"], "c" * 64)
        self.assertEqual(selection["video_subtree"], "videosbodytrack")
        self.assertEqual(
            selection["inventory_scope"],
            "EXTRACTED_SUBSET_NO_RELEASE_COMPLETENESS_CLAIM",
        )
        self.assertIn("CALLER_SUPPLIED", selection["archive_digest_binding"])
        selection_bytes = json.dumps(
            selection, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            provenance["source_archive_selection_sha256"],
            hashlib.sha256(selection_bytes).hexdigest(),
        )
        source_identity = {
            "archive_selection": selection,
            "excluded_files": provenance["source_exclusion_ledger"],
            "excluded_files_sha256": (
                provenance["source_exclusion_ledger_sha256"]),
            "files": provenance["source_files"],
        }
        source_identity_bytes = json.dumps(
            source_identity, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            provenance["source_identity_sha256"],
            hashlib.sha256(source_identity_bytes).hexdigest(),
        )
        self.assertEqual(
            provenance["source_files"]["video"]["relative_path"],
            "videosbodytrack/S40/S40_A01_T01_pose.mp4",
        )
        self.assertEqual(
            provenance["source_files"]["imu_quaternion"]["relative_path"],
            "dataset/videoandimus/S40/S40_A01_T01.raw",
        )
        self.assertFalse(provenance["raw_accelerometer_axes_available"])
        self.assertFalse(provenance["raw_gyroscope_axes_available"])
        self.assertTrue(provenance["recording_inventory_complete"])
        self.assertFalse(provenance["release_inventory_complete"])
        self.assertIn("QUATERNION", provenance["released_imu_payload"])
        self.assertIn("INDEPENDENT_SYNC", provenance["prohibited_interpretation"])
        self.assertEqual(provenance["source_exclusion_ledger"], [])
        self.assertEqual(
            provenance["source_exclusion_ledger_sha256"],
            hashlib.sha256(b"[]").hexdigest(),
        )

    def test_either_pinned_video_archive_can_be_selected_explicitly(self):
        for archive in ("videosmallsize", "videosfullsize"):
            with self.subTest(archive=archive), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                dataset_root = base / "dataset-extraction"
                video_root = base / "video-extraction"
                _official_layout(dataset_root, video_root)
                adapter = _adapter(
                    dataset_root, video_root, video_archive=archive)
                provenance = adapter.provenance(adapter.discover()[0])
                self.assertEqual(
                    provenance["source_archive_selection"]["video_archive"],
                    archive,
                )

    def test_release_constants_are_pinned_to_record_15075076(self):
        self.assertEqual(VIDIMU_RELEASE_VERSION, "2.0.0")
        self.assertEqual(VIDIMU_ZENODO_RECORD_ID, 15_075_076)
        self.assertEqual(VIDIMU_RECORD_URL, "https://zenodo.org/records/15075076")
        self.assertEqual(VIDIMU_RECORD_DOI, "10.1038/s41597-023-02554-9")
        self.assertEqual(VIDIMU_CONCEPT_DOI, "10.5281/zenodo.7681316")
        self.assertEqual(VIDIMU_LICENSE, "CC BY 4.0")
        self.assertEqual(VIDIMU_LICENSE_SPDX, "CC-BY-4.0")
        self.assertEqual(VIDIMU_VALIDATION_ROLE, "INVENTORY_ONLY_UNVERIFIED_SYNC")

    def test_terms_hash_is_mandatory_and_must_be_valid_sha256(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            _official_layout(dataset_root, video_root)
            with self.assertRaisesRegex(VidimuAdapterError, "requires terms"):
                VidimuAdapter(
                    dataset_root,
                    video_archive_root=video_root,
                    video_archive="videosmallsize",
                    dataset_archive_sha256="b" * 64,
                    video_archive_sha256="c" * 64,
                    inventory_scope=(
                        "EXTRACTED_SUBSET_NO_RELEASE_COMPLETENESS_CLAIM"),
                )
            with self.assertRaisesRegex(VidimuAdapterError, "64-character"):
                VidimuAdapter(
                    dataset_root,
                    video_archive_root=video_root,
                    video_archive="videosmallsize",
                    dataset_archive_sha256="b" * 64,
                    video_archive_sha256="c" * 64,
                    inventory_scope=(
                        "EXTRACTED_SUBSET_NO_RELEASE_COMPLETENESS_CLAIM"),
                    terms_sha256="not-a-sha256",
                )

    def test_downloaded_archive_hashes_are_mandatory_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            _official_layout(dataset_root, video_root)
            with self.assertRaisesRegex(
                    VidimuAdapterError, "dataset_archive_sha256"):
                VidimuAdapter(
                    dataset_root,
                    video_archive_root=video_root,
                    video_archive="videosmallsize",
                    dataset_archive_sha256="not-a-digest",
                    video_archive_sha256="c" * 64,
                    inventory_scope=(
                        "EXTRACTED_SUBSET_NO_RELEASE_COMPLETENESS_CLAIM"),
                    terms_sha256="a" * 64,
                )

    def test_partial_inventory_scope_acknowledgement_is_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            _official_layout(dataset_root, video_root)
            with self.assertRaises(TypeError):
                VidimuAdapter(
                    dataset_root,
                    video_archive_root=video_root,
                    video_archive="videosmallsize",
                    dataset_archive_sha256="b" * 64,
                    video_archive_sha256="c" * 64,
                    terms_sha256="a" * 64,
                )

    def test_supplied_terms_hash_can_be_verified_against_a_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            _official_layout(dataset_root, video_root)
            terms = base / "terms.txt"
            terms.write_bytes(b"terms")
            digest = hashlib.sha256(b"terms").hexdigest()
            adapter = VidimuAdapter(
                dataset_root,
                video_archive_root=video_root,
                video_archive="videosfullsize",
                dataset_archive_sha256="b" * 64,
                video_archive_sha256="c" * 64,
                inventory_scope=(
                    "EXTRACTED_SUBSET_NO_RELEASE_COMPLETENESS_CLAIM"),
                terms_sha256=digest.upper(),
                terms_path=terms,
            )
            self.assertEqual(adapter.terms_sha256, digest)
            self.assertEqual(
                adapter.terms_hash_origin, "SUPPLIED_AND_VERIFIED_FROM_FILE")
            with self.assertRaisesRegex(VidimuAdapterError, "does not match"):
                VidimuAdapter(
                    dataset_root,
                    video_archive_root=video_root,
                    video_archive="videosfullsize",
                    dataset_archive_sha256="b" * 64,
                    video_archive_sha256="c" * 64,
                    inventory_scope=(
                        "EXTRACTED_SUBSET_NO_RELEASE_COMPLETENESS_CLAIM"),
                    terms_sha256="0" * 64,
                    terms_path=terms,
                )

    def test_video_archive_selection_is_mandatory_and_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            _official_layout(dataset_root, video_root)
            with self.assertRaisesRegex(VidimuAdapterError, "exactly"):
                _adapter(dataset_root, video_root, video_archive="video-auto")

    def test_incomplete_official_inventory_is_not_ingestible(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            paths = _official_layout(dataset_root, video_root)
            paths["video_pose"].unlink()
            adapter = _adapter(dataset_root, video_root)
            recording = adapter.discover()[0]
            self.assertFalse(recording.inventory_complete)
            with self.assertRaisesRegex(VidimuAdapterError, "incomplete"):
                adapter.provenance(recording)

    def test_official_npose_companions_are_hashed_and_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            paths = _official_layout(dataset_root, video_root)
            npose_csv = paths["video_pose"].with_name(
                "S40_A01_T01_Npose.csv")
            npose_video = paths["video"].with_name(
                "S40_A01_T01_Npose_pose.mp4")
            npose_csv.write_bytes(b"official Npose coordinate companion")
            npose_video.write_bytes(b"official Npose video companion")
            npose_csv_sha256 = hashlib.sha256(npose_csv.read_bytes()).hexdigest()
            npose_video_sha256 = hashlib.sha256(
                npose_video.read_bytes()).hexdigest()

            adapter = _adapter(dataset_root, video_root)
            recordings = adapter.discover()
            self.assertEqual(
                [recording.recording_id for recording in recordings],
                ["S40_A01_T01"],
            )
            provenance = adapter.provenance(recordings[0])

        expected_ledger = sorted(
            [
                {
                    "archive": "dataset",
                    "relative_path": (
                        "dataset/videoandimus/S40/S40_A01_T01_Npose.csv"),
                    "reason": (
                        "OFFICIAL_NPOSE_COMPANION_NOT_CANONICAL_PAIR_INPUT"),
                    "sha256": npose_csv_sha256,
                },
                {
                    "archive": "videosmallsize",
                    "relative_path": (
                        "videosbodytrack/S40/S40_A01_T01_Npose_pose.mp4"),
                    "reason": (
                        "OFFICIAL_NPOSE_COMPANION_NOT_CANONICAL_PAIR_INPUT"),
                    "sha256": npose_video_sha256,
                },
            ],
            key=lambda item: (item["archive"], item["relative_path"]),
        )
        self.assertEqual(provenance["source_exclusion_ledger"], expected_ledger)
        ledger_bytes = json.dumps(
            expected_ledger, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        ledger_sha256 = hashlib.sha256(ledger_bytes).hexdigest()
        self.assertEqual(
            provenance["source_exclusion_ledger_sha256"], ledger_sha256)
        source_identity = {
            "archive_selection": provenance["source_archive_selection"],
            "excluded_files": expected_ledger,
            "excluded_files_sha256": ledger_sha256,
            "files": provenance["source_files"],
        }
        self.assertEqual(
            provenance["source_identity_sha256"],
            hashlib.sha256(json.dumps(
                source_identity, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
        )

    def test_complete_official_v2_file_forms_are_lazily_hashed_and_pinned(self):
        recording_ids = (
            "S25_A02_T01", "S41_A03_T01", "S48_A01_T01", "S49_A13_T01",
        )
        npose_reason = "OFFICIAL_NPOSE_COMPANION_NOT_CANONICAL_PAIR_INPUT"
        residue_prefix = "OFFICIAL_V2_NONCANONICAL_RESIDUE_MAPPED_TO_"
        reasons = {
            "sto": "OFFICIAL_STO_PROCESSED_COMPANION_NOT_CANONICAL_PAIR_INPUT",
            "mot": "OFFICIAL_MOT_PROCESSED_COMPANION_NOT_CANONICAL_PAIR_INPUT",
            "ik_mot": "OFFICIAL_IK_MOT_COMPANION_NOT_CANONICAL_PAIR_INPUT",
            "ik_orientation": (
                "OFFICIAL_IK_ORIENTATION_ERRORS_COMPANION_"
                "NOT_CANONICAL_PAIR_INPUT"
            ),
            "out": (
                "OFFICIAL_FULLSIZE_MP4_OUT_COMPANION_NOT_CANONICAL_PAIR_INPUT"
            ),
        }

        for archive in ("videosmallsize", "videosfullsize"):
            with self.subTest(archive=archive), \
                    tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                dataset_root = base / "dataset-extraction"
                video_root = base / "video-extraction"
                canonical = {
                    recording_id: _official_layout(
                        dataset_root, video_root, recording_id)
                    for recording_id in recording_ids
                }
                expected: dict[
                    str, list[tuple[dict[str, str], Path]]
                ] = {recording_id: [] for recording_id in recording_ids}

                def add_exclusion(
                    path: Path, *, target: str, source_archive: str, reason: str,
                    dataset_archive_root: Path = dataset_root,
                    video_archive_root: Path = video_root,
                    expected_entries: dict[
                        str, list[tuple[dict[str, str], Path]]
                    ] = expected,
                ) -> None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(
                        f"official-excluded:{path.as_posix()}".encode())
                    archive_root = (
                        dataset_archive_root
                        if source_archive == "dataset"
                        else video_archive_root
                    )
                    expected_entries[target].append(({
                        "archive": source_archive,
                        "relative_path": path.relative_to(
                            archive_root).as_posix(),
                        "reason": reason,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }, path.resolve()))

                for recording_id, paths in canonical.items():
                    data_subject = paths["video_pose"].parent
                    video_subject = paths["video"].parent
                    add_exclusion(
                        data_subject / f"{recording_id}_Npose.csv",
                        target=recording_id, source_archive="dataset",
                        reason=npose_reason,
                    )
                    add_exclusion(
                        data_subject / f"{recording_id}.sto",
                        target=recording_id, source_archive="dataset",
                        reason=reasons["sto"],
                    )
                    add_exclusion(
                        data_subject / f"ik_{recording_id}.mot",
                        target=recording_id, source_archive="dataset",
                        reason=reasons["ik_mot"],
                    )
                    add_exclusion(
                        data_subject
                        / f"ik_{recording_id}_orientationErrors.sto",
                        target=recording_id, source_archive="dataset",
                        reason=reasons["ik_orientation"],
                    )
                    add_exclusion(
                        video_subject / f"{recording_id}_Npose_pose.mp4",
                        target=recording_id, source_archive=archive,
                        reason=npose_reason,
                    )
                    if recording_id == "S48_A01_T01":
                        add_exclusion(
                            data_subject / f"{recording_id}.mot",
                            target=recording_id, source_archive="dataset",
                            reason=reasons["mot"],
                        )
                    if archive == "videosfullsize":
                        for filename in (
                            f"{recording_id}.mp4.out",
                            f"{recording_id}_Npose.mp4.out",
                        ):
                            add_exclusion(
                                video_subject / filename,
                                target=recording_id, source_archive=archive,
                                reason=reasons["out"],
                            )

                data_subtree = dataset_root / VIDIMU_DATASET_SUBTREE
                video_subtree = video_root / VIDIMU_VIDEO_SUBTREE
                for path, target in (
                    (data_subtree / "S41/S41_A03_P01.csv", "S41_A03_T01"),
                    (
                        data_subtree / "S41/S41_A03_P01_Npose.csv",
                        "S41_A03_T01",
                    ),
                    (
                        data_subtree / "S49/S49_A13_T01V2_Npose.csv",
                        "S49_A13_T01",
                    ),
                    (
                        video_subtree
                        / "S49/S49_A13_T01V2_Npose_pose.mp4",
                        "S49_A13_T01",
                    ),
                    (
                        video_subtree / "S24/S25_A02_T01_pose.mp4",
                        "S25_A02_T01",
                    ),
                ):
                    add_exclusion(
                        path, target=target,
                        source_archive=(
                            "dataset" if path.is_relative_to(data_subtree)
                            else archive
                        ),
                        reason=residue_prefix + target,
                    )
                if archive == "videosfullsize":
                    for path, target in (
                        (
                            video_subtree
                            / "S49/S49_A13_T01V2_Npose.mp4.out",
                            "S49_A13_T01",
                        ),
                        (
                            video_subtree / "S24/S25_A02_T01.mp4.out",
                            "S25_A02_T01",
                        ),
                    ):
                        add_exclusion(
                            path, target=target, source_archive=archive,
                            reason=residue_prefix + target,
                        )
                ignored_metadata = data_subtree / ".DS_Store"
                ignored_metadata.write_bytes(b"unrelated Finder metadata")

                adapter = _adapter(
                    dataset_root, video_root, video_archive=archive)
                with mock.patch(
                    "motionbloom.tremora_store.adapters.vidimu.sha256_file",
                    wraps=sha256_file,
                ) as discovery_hashing:
                    recordings = adapter.discover()
                self.assertEqual(discovery_hashing.call_count, 0)
                self.assertEqual(
                    [recording.recording_id for recording in recordings],
                    list(recording_ids),
                )

                for recording in recordings:
                    recording_id = recording.recording_id
                    with self.subTest(
                        archive=archive, recording_id=recording_id,
                    ), mock.patch(
                        "motionbloom.tremora_store.adapters.vidimu.sha256_file",
                        wraps=sha256_file,
                    ) as selected_hashing:
                        provenance = adapter.provenance(recording)
                        expected_ledger = sorted(
                            (entry for entry, _ in expected[recording_id]),
                            key=lambda item: (
                                item["archive"], item["relative_path"]),
                        )
                        self.assertEqual(
                            provenance["source_exclusion_ledger"],
                            expected_ledger,
                        )
                        self.assertEqual(
                            recording.video_path.resolve(),
                            canonical[recording_id]["video"].resolve(),
                        )
                        selected_paths = {
                            path.resolve()
                            for path in canonical[recording_id].values()
                        } | {
                            path for _, path in expected[recording_id]
                        }
                        hashed_paths = [
                            Path(call.args[0]).resolve()
                            for call in selected_hashing.call_args_list
                        ]
                        self.assertEqual(set(hashed_paths), selected_paths)
                        for path in canonical[recording_id].values():
                            self.assertEqual(
                                hashed_paths.count(path.resolve()), 1)
                        for _, path in expected[recording_id]:
                            self.assertEqual(hashed_paths.count(path), 2)

                        bound = _bind_internal_recording({
                            **synthetic_provenance(), **provenance,
                        })
                        validate_provenance(bound)
                        residue = next((
                            entry for entry in bound[
                                "source_exclusion_ledger"]
                            if entry["reason"].startswith(residue_prefix)
                        ), None)
                        if residue is not None:
                            candidate = deepcopy(bound)
                            candidate_residue = next(
                                entry for entry in candidate[
                                    "source_exclusion_ledger"]
                                if entry["relative_path"]
                                == residue["relative_path"]
                            )
                            candidate_residue["reason"] = npose_reason
                            _recompute_vidimu_identity(candidate)
                            with self.assertRaises(StoreInvariantError):
                                validate_provenance(candidate)

    def test_all_thirteen_official_s48_mot_paths_are_exactly_pinned(self):
        official_recordings = (
            "S48_A01_T01",
            "S48_A02_T02",
            "S48_A03_T02",
            "S48_A04_T01",
            "S48_A05_T02",
            "S48_A06_T02",
            "S48_A07_T02",
            "S48_A08_T01",
            "S48_A09_T01",
            "S48_A10_T02",
            "S48_A11_T02",
            "S48_A12_T02",
            "S48_A13_T02",
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            for recording_id in official_recordings:
                paths = _official_layout(
                    dataset_root, video_root, recording_id)
                paths["video_pose"].with_suffix(".mot").write_bytes(
                    f"official-mot:{recording_id}".encode())
            adapter = _adapter(dataset_root, video_root)
            recordings = adapter.discover()
            self.assertEqual(
                [recording.recording_id for recording in recordings],
                list(official_recordings),
            )
            for recording in recordings:
                with self.subTest(recording_id=recording.recording_id):
                    provenance = adapter.provenance(recording)
                    self.assertEqual(
                        provenance["source_exclusion_ledger"][0]["reason"],
                        "OFFICIAL_MOT_PROCESSED_COMPANION_"
                        "NOT_CANONICAL_PAIR_INPUT",
                    )
                    validate_provenance(_bind_internal_recording({
                        **synthetic_provenance(), **provenance,
                    }))

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            paths = _official_layout(
                dataset_root, video_root, "S48_A02_T01")
            paths["video_pose"].with_suffix(".mot").write_bytes(
                b"not an official S48 mot path")
            with self.assertRaisesRegex(
                VidimuAdapterError, "unexpected VIDIMU data",
            ):
                _adapter(dataset_root, video_root).discover()

    def test_selected_subtree_root_symlinks_fail_closed(self):
        for selected in ("dataset-parent", "dataset-leaf", "video-leaf"):
            with self.subTest(selected=selected), \
                    tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                dataset_root = base / "dataset-extraction"
                video_root = base / "video-extraction"
                dataset_root.mkdir()
                video_root.mkdir(parents=True)
                outside = base / f"outside-{selected}"
                outside.mkdir()
                data_parent = dataset_root / "dataset"
                data_subtree = data_parent / "videoandimus"
                video_subtree = video_root / "videosbodytrack"
                try:
                    if selected == "dataset-parent":
                        data_parent.symlink_to(
                            outside, target_is_directory=True)
                        video_subtree.mkdir()
                    elif selected == "dataset-leaf":
                        data_parent.mkdir()
                        data_subtree.symlink_to(
                            outside, target_is_directory=True)
                        video_subtree.mkdir()
                    else:
                        data_subtree.mkdir(parents=True)
                        video_subtree.symlink_to(
                            outside, target_is_directory=True)
                except (NotImplementedError, OSError) as exc:
                    self.skipTest(f"host cannot create test symlinks: {exc}")
                with self.assertRaisesRegex(VidimuAdapterError, "symlink"):
                    _adapter(dataset_root, video_root)

    def test_unknown_video_variants_still_fail_closed(self):
        for suffix in ("_rgb.mp4", ".mp4"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                dataset_root = base / "dataset-extraction"
                video_root = base / "video-extraction"
                paths = _official_layout(dataset_root, video_root)
                paths["video"].unlink()
                bad_video = paths["video"].with_name(f"S40_A01_T01{suffix}")
                bad_video.write_bytes(b"not the released body-track name")
                with self.assertRaisesRegex(VidimuAdapterError, "unexpected VIDIMU video"):
                    _adapter(dataset_root, video_root).discover()

    def test_provenance_rescans_for_late_additions_and_deletions(self):
        for mutation in (
            "unknown-addition", "lowercase-unknown-addition",
            "selected-deletion",
        ):
            with self.subTest(mutation=mutation), \
                    tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                dataset_root = base / "dataset-extraction"
                video_root = base / "video-extraction"
                paths = _official_layout(dataset_root, video_root)
                adapter = _adapter(dataset_root, video_root)
                recording = adapter.discover()[0]
                if mutation == "unknown-addition":
                    paths["video"].with_name(
                        "S40_A01_T01_rgb.mp4").write_bytes(b"late unknown")
                    expected = "unexpected VIDIMU video"
                elif mutation == "lowercase-unknown-addition":
                    paths["video"].with_name(
                        "s40_a01_t01_rgb.mp4").write_bytes(b"late unknown")
                    expected = "unexpected VIDIMU video"
                else:
                    paths["imu_quaternion"].unlink()
                    expected = "regular file|changed|incomplete"
                with self.assertRaisesRegex(VidimuAdapterError, expected):
                    adapter.provenance(recording)

    def test_late_target_variants_are_found_across_selected_subtrees(self):
        for mutation in (
            "dot-delimiter", "hyphen-delimiter", "nested-duplicate",
            "wrong-subject-duplicate", "unicode-digit-suffix",
        ):
            with self.subTest(mutation=mutation), \
                    tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                dataset_root = base / "dataset-extraction"
                video_root = base / "video-extraction"
                paths = _official_layout(dataset_root, video_root)
                adapter = _adapter(dataset_root, video_root)
                recording = adapter.discover()[0]
                if mutation == "dot-delimiter":
                    late = paths["video"].with_name(
                        "S40_A01_T01.pose.mp4")
                elif mutation == "hyphen-delimiter":
                    late = paths["video"].with_name(
                        "S40_A01_T01-pose.mp4")
                elif mutation == "nested-duplicate":
                    late = paths["video"].parent / "nested" / paths["video"].name
                elif mutation == "unicode-digit-suffix":
                    late = paths["video"].with_name(
                        "S40_A01_T01²_pose.mp4")
                else:
                    late = paths["video"].parent.parent / "S41" / paths["video"].name
                late.parent.mkdir(parents=True, exist_ok=True)
                late.write_bytes(b"late ambiguous target")
                with self.assertRaisesRegex(
                        VidimuAdapterError, "unexpected VIDIMU video"):
                    adapter.provenance(recording)

    def test_late_official_npose_companion_enters_provenance_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            paths = _official_layout(dataset_root, video_root)
            adapter = _adapter(dataset_root, video_root)
            recording = adapter.discover()[0]
            late = paths["video_pose"].with_name("S40_A01_T01_Npose.csv")
            late.write_bytes(b"late official companion")
            provenance = adapter.provenance(recording)
            self.assertEqual(
                provenance["source_exclusion_ledger"][0]["sha256"],
                hashlib.sha256(late.read_bytes()).hexdigest(),
            )

    def test_recording_provenance_hashes_only_selected_recording_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            _official_layout(dataset_root, video_root)
            _official_layout(dataset_root, video_root, "S41_A01_T01")
            adapter = _adapter(dataset_root, video_root)
            recording = adapter.discover()[0]
            with mock.patch(
                    "motionbloom.tremora_store.adapters.vidimu.sha256_file",
                    wraps=sha256_file) as hashing:
                provenance = adapter.provenance(recording)
            self.assertEqual(
                provenance["source_recording_id"], recording.recording_id)
            self.assertEqual(
                {Path(call.args[0]).name for call in hashing.call_args_list},
                {
                    "S40_A01_T01_pose.mp4",
                    "S40_A01_T01.csv",
                    "S40_A01_T01.raw",
                },
            )

    def test_recording_prefix_collision_does_not_cross_contaminate_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            _official_layout(dataset_root, video_root, "S40_A01_T01")
            _official_layout(dataset_root, video_root, "S40_A01_T010")
            adapter = _adapter(dataset_root, video_root)
            recordings = {
                recording.recording_id: recording
                for recording in adapter.discover()
            }

            for recording_id in ("S40_A01_T01", "S40_A01_T010"):
                with self.subTest(recording_id=recording_id):
                    provenance = adapter.provenance(recordings[recording_id])
                    self.assertEqual(
                        provenance["source_recording_id"], recording_id)
                    self.assertEqual(
                        {
                            Path(entry["relative_path"]).stem
                            for entry in provenance["source_files"].values()
                        },
                        {recording_id, f"{recording_id}_pose"},
                    )

    def test_vidimu_identity_extension_is_recomputed_by_store_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            paths = _official_layout(dataset_root, video_root)
            paths["video_pose"].with_name(
                "S40_A01_T01_Npose.csv").write_bytes(b"excluded")
            adapter = _adapter(dataset_root, video_root)
            adapter_provenance = adapter.provenance(adapter.discover()[0])
        provenance = _bind_internal_recording({
            **synthetic_provenance(), **adapter_provenance,
        })
        validate_provenance(provenance)
        mutations = (
            lambda value: value["source_files"]["video"].update(
                {"sha256": "0" * 64}),
            lambda value: value["source_archive_selection"].update(
                {"video_archive_sha256": "0" * 64}),
            lambda value: value.update(
                {"source_exclusion_ledger_sha256": "0" * 64}),
            lambda value: value.update({"source_identity_sha256": "0" * 64}),
            lambda value: value.pop("source_files"),
        )
        for mutate in mutations:
            candidate = deepcopy(provenance)
            mutate(candidate)
            with self.subTest(mutate=mutate), self.assertRaises(
                    StoreInvariantError):
                validate_provenance(candidate)

    def test_semantically_rehashed_vidimu_identity_mutations_still_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            paths = _official_layout(dataset_root, video_root)
            paths["video_pose"].with_name(
                "S40_A01_T01_Npose.csv").write_bytes(b"excluded")
            adapter = _adapter(dataset_root, video_root)
            adapter_provenance = adapter.provenance(adapter.discover()[0])
        original = _bind_internal_recording({
            **synthetic_provenance(), **adapter_provenance,
        })
        validate_provenance(original)

        candidates = []
        wrong_version = deepcopy(original)
        wrong_version["source_dataset_version"] = "9.0.0"
        wrong_version["source_archive_selection"]["release_version"] = "9.0.0"
        _recompute_vidimu_identity(wrong_version)
        candidates.append(wrong_version)

        wrong_source = deepcopy(original)
        wrong_source["source_files"]["video"]["relative_path"] = (
            "videosbodytrack/S40/S40_A01_T01_pose.mov")
        _recompute_vidimu_identity(wrong_source)
        candidates.append(wrong_source)

        foreign_exclusion = deepcopy(original)
        foreign_exclusion["source_exclusion_ledger"][0]["relative_path"] = (
            "dataset/videoandimus/S41/S41_A01_T01_Npose.csv")
        _recompute_vidimu_identity(foreign_exclusion)
        candidates.append(foreign_exclusion)

        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(
                    StoreInvariantError):
                validate_provenance(candidate)

        claim_mutations = (
            {"raw_accelerometer_axes_available": True},
            {"raw_gyroscope_axes_available": True},
            {"clock_truth_status": "INDEPENDENT_SYNC_VERIFIED"},
            {"released_imu_payload": "RAW_ACCEL_GYRO"},
            {"release_inventory_complete": True},
            {"recording_inventory_complete": False},
            {"prohibited_interpretation": ""},
            {"clinical_validation_allowed": True},
        )
        for updates in claim_mutations:
            candidate = deepcopy(original)
            candidate.update(updates)
            with self.subTest(updates=updates), self.assertRaises(
                    StoreInvariantError):
                validate_provenance(candidate)

    def test_vidimu_snapshot_rejects_synthetic_accel_gyro_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            _official_layout(dataset_root, video_root)
            adapter = _adapter(dataset_root, video_root)
            adapter_provenance = adapter.provenance(adapter.discover()[0])
            provenance = _bind_internal_recording({
                **synthetic_provenance(), **adapter_provenance,
            })
            root = base / "store"
            writer = store_writer(root, provenance=provenance)
            for name, (table, schema, keys) in all_tables().items():
                writer.write_table(name, table, schema=schema, sort_keys=keys)
            with self.assertRaisesRegex(
                    SnapshotError, "VIDIMU.*QUATERNION"):
                writer.commit()

    def test_inventory_snapshot_can_commit_without_fabricating_clock_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            _official_layout(dataset_root, video_root)
            adapter = _adapter(dataset_root, video_root)
            adapter_provenance = adapter.provenance(adapter.discover()[0])
            tables, provenance = _unresolved_vidimu_fixture(adapter_provenance)
            self.assertEqual(tables["window_index"][0].num_rows, 0)
            self.assertEqual(tables["window_rejections"][0].num_rows, 5)
            self.assertEqual(
                set(tables["clock_map"][0]["mapping_status"].to_pylist()),
                {"UNRESOLVED"},
            )
            self.assertEqual(
                set(tables["clock_map"][0]["residual_p95_ms"].to_pylist()),
                {None},
            )
            writer = store_writer(base / "store", provenance=provenance)
            for name, (table, schema, keys) in tables.items():
                writer.write_table(name, table, schema=schema, sort_keys=keys)
            snapshot = writer.commit()
            self.assertTrue(snapshot.is_dir())

    def test_inventory_snapshot_rejects_any_resolved_clock_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            _official_layout(dataset_root, video_root)
            adapter = _adapter(dataset_root, video_root)
            adapter_provenance = adapter.provenance(adapter.discover()[0])

            for mutation in (
                "valid-clock", "residuals", "frame-bit", "imu-bit",
            ):
                with self.subTest(mutation=mutation):
                    tables, provenance = _unresolved_vidimu_fixture(
                        adapter_provenance)
                    if mutation in {"valid-clock", "residuals"}:
                        table, schema, keys = tables["clock_map"]
                        rows = table.to_pylist()
                        for row in rows:
                            if mutation == "valid-clock":
                                row["mapping_status"] = "VALID"
                            row["residual_p50_ms"] = 0.0
                            row["residual_p95_ms"] = 0.0
                        tables["clock_map"] = (
                            pa.Table.from_pylist(rows, schema=schema),
                            schema,
                            keys,
                        )
                        expected = "clock rows must be UNRESOLVED"
                    else:
                        table_name = (
                            "frame_index" if mutation == "frame-bit"
                            else "imu_samples")
                        bit_field = (
                            "quality_bits" if mutation == "frame-bit"
                            else "validity_bits")
                        table, schema, keys = tables[table_name]
                        rows = table.to_pylist()
                        rows[0][bit_field] = 0
                        tables[table_name] = (
                            pa.Table.from_pylist(rows, schema=schema),
                            schema,
                            keys,
                        )
                        expected = "source rows require UNRESOLVED_CLOCK_MAP"
                    writer = store_writer(
                        base / f"store-{mutation}", provenance=provenance)
                    for name, (table, schema, keys) in tables.items():
                        writer.write_table(
                            name, table, schema=schema, sort_keys=keys)
                    with self.assertRaisesRegex(SnapshotError, expected):
                        writer.commit()

    def test_duplicate_or_ambiguous_video_layout_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            paths = _official_layout(dataset_root, video_root)
            wrong_subject = (
                video_root / VIDIMU_VIDEO_SUBTREE / "S41" / paths["video"].name)
            wrong_subject.parent.mkdir(parents=True)
            wrong_subject.write_bytes(b"duplicate logical recording")
            with self.assertRaisesRegex(VidimuAdapterError, "canonical subject subtree"):
                _adapter(dataset_root, video_root).discover()

    def test_cross_recording_handcrafted_inventory_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            first = _official_layout(dataset_root, video_root, "S40_A01_T01")
            second = _official_layout(dataset_root, video_root, "S41_A01_T01")
            adapter = _adapter(dataset_root, video_root)
            recording = VidimuRecording(
                recording_id="S40_A01_T01",
                subject_id="S40",
                activity_id="A01",
                trial_id="T01",
                video_path=second["video"],
                pose_path=first["video_pose"],
                quaternion_path=first["imu_quaternion"],
            )
            with self.assertRaisesRegex(
                    VidimuAdapterError, "another recording|exact filename"):
                adapter.provenance(recording)

    def test_handcrafted_inventory_outside_each_selected_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset_root = base / "dataset-extraction"
            video_root = base / "video-extraction"
            foreign_dataset_root = base / "foreign-dataset"
            selected = _official_layout(dataset_root, video_root)
            foreign = _official_layout(foreign_dataset_root, base / "foreign-video")
            adapter = _adapter(dataset_root, video_root)
            recording = VidimuRecording(
                recording_id="S40_A01_T01",
                subject_id="S40",
                activity_id="A01",
                trial_id="T01",
                video_path=selected["video"],
                pose_path=foreign["video_pose"],
                quaternion_path=selected["imu_quaternion"],
            )
            with self.assertRaisesRegex(VidimuAdapterError, "selected subtree"):
                adapter.provenance(recording)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
