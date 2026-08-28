"""Independent exact-inventory and byte-replay tests for v0.3 release audit."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from benchmarks import audit_vidimu_pts_cv_release as release_audit_module
from benchmarks.audit_vidimu_pts_cv_release import (
    GateBArchiveAnchor,
    GateBTrustAnchors,
    ReleaseAuditError,
    audit_vidimu_pts_cv_release,
)
from motionbloom.tremora_store.cv.offline_finalizer import EstimatorProvenance
from motionbloom.tremora_store.decode.pts_decoder import DecodeConfig, PTSDecoder
from motionbloom.tremora_store.finalize._bundle_io import (
    FINALIZATION_FILES,
    SOURCE_FAILURE_FILES,
    canonical_json_bytes,
)
from motionbloom.tremora_store.finalize.finalize_vidimu_recording import (
    RecordingProvenance,
    finalize_vidimu_recording,
)
from motionbloom.tremora_store.finalize.finalize_vidimu_snapshot import (
    VIDIMU_INVENTORY_SCHEMA_VERSION,
    FrozenSourceAsset,
    VidimuSnapshotInputs,
    VidimuSnapshotRecord,
    finalize_vidimu_snapshot,
)
from motionbloom.tremora_store.finalize.source_failure_artifact import (
    SourceFailureArtifact,
)
from tests._frame_finalization_fixtures import (
    DEFAULT_ESTIMATOR_PROVENANCE,
    DeterministicPoseEstimator,
)
from tests._pts_video_fixtures import (
    generate_cfr_video,
    require_pts_media_toolchain,
    sha256_file,
)

_RECORDING_IDS = ("fixture-release-a", "fixture-release-b")


def _provenance(recording_id: str) -> RecordingProvenance:
    return RecordingProvenance(
        dataset_id="gate-a-synthetic",
        dataset_version="1",
        recording_id=recording_id,
        source_kind="SYNTHETIC_FIXTURE",
        source_original_path=f"generated/{recording_id}.mp4",
        source_object_id=f"gate-a:{recording_id}",
        materialization_date="2026-08-27",
        license_id="generated-test-fixture",
        license_record_sha256="8" * 64,
    )


def _gate_b_inputs(
    source_root: Path,
    recording_id: str,
    *,
    inventory_name: str = "inventory.json",
    archive_variant: str = "shared",
) -> tuple[VidimuSnapshotInputs, Path]:
    source_root.mkdir(parents=True, exist_ok=True)
    video = generate_cfr_video(
        source_root / "videosoriginal" / recording_id)
    imu = source_root / "imu" / f"{recording_id}.raw"
    imu.parent.mkdir(parents=True, exist_ok=True)
    imu.write_bytes(f"paired IMU fixture for {recording_id}\n".encode("ascii"))
    dataset_archive = source_root / "archives" / f"dataset-{archive_variant}.zip"
    dataset_archive.parent.mkdir(parents=True, exist_ok=True)
    dataset_archive.write_bytes(
        f"synthetic dataset archive {archive_variant}\n".encode("ascii"))
    video_archive = (
        source_root / "archives" / f"videosmallsize-{archive_variant}.zip"
    )
    video_archive.write_bytes(
        f"synthetic video archive {archive_variant}\n".encode("ascii"))
    license_record = source_root / "license.txt"
    license_record.write_bytes(b"shared synthetic VIDIMU license evidence\n")

    video_asset = FrozenSourceAsset(
        original_path=video.relative_to(source_root).as_posix(),
        local_path=video,
        sha256=sha256_file(video),
        role="ORIGINAL_VIDEO",
    )
    imu_asset = FrozenSourceAsset(
        original_path=imu.relative_to(source_root).as_posix(),
        local_path=imu,
        sha256=sha256_file(imu),
        role="IMU",
    )
    archive_assets = (
        FrozenSourceAsset(
            original_path=dataset_archive.relative_to(source_root).as_posix(),
            local_path=dataset_archive,
            sha256=sha256_file(dataset_archive),
            role="DATASET_ARCHIVE",
        ),
        FrozenSourceAsset(
            original_path=video_archive.relative_to(source_root).as_posix(),
            local_path=video_archive,
            sha256=sha256_file(video_archive),
            role="VIDEO_ARCHIVE",
        ),
    )
    inventory = source_root / inventory_name
    inventory.write_text(json.dumps({
        "inventory_schema_version": VIDIMU_INVENTORY_SCHEMA_VERSION,
        "records": [{
            "recording_id": recording_id,
            "video": {
                "original_path": video_asset.original_path,
                "role": video_asset.role,
                "sha256": video_asset.sha256,
            },
            "imu_assets": [{
                "original_path": imu_asset.original_path,
                "role": imu_asset.role,
                "sha256": imu_asset.sha256,
            }],
        }],
    }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    provenance = RecordingProvenance(
        dataset_id="vidimu",
        dataset_version="2.0.0",
        recording_id=recording_id,
        source_kind="VIDIMU_PUBLIC",
        source_original_path=video_asset.original_path,
        source_object_id=f"vidimu:{recording_id}",
        materialization_date="2026-08-27",
        license_id="vidimu-license-fixture",
        license_record_sha256=sha256_file(license_record),
        inventory_manifest_sha256=sha256_file(inventory),
    )
    return VidimuSnapshotInputs(
        records=(VidimuSnapshotRecord(
            provenance=provenance,
            video=video_asset,
            imu_assets=(imu_asset,),
        ),),
        expected_recording_ids=(recording_id,),
        archive_assets=archive_assets,
        inventory_manifest_path=inventory,
        inventory_manifest_sha256=sha256_file(inventory),
        license_record_path=license_record,
        license_record_sha256=sha256_file(license_record),
    ), imu


def _gate_b_trust_anchors(inputs: VidimuSnapshotInputs) -> GateBTrustAnchors:
    provenance = inputs.records[0].provenance
    return GateBTrustAnchors(
        expected_dataset_id=provenance.dataset_id,
        expected_dataset_version=provenance.dataset_version,
        inventory_manifest_sha256=inputs.inventory_manifest_sha256,
        source_archives=tuple(sorted((
            GateBArchiveAnchor(
                original_path=asset.original_path,
                role=asset.role,
                sha256=asset.sha256,
            )
            for asset in inputs.archive_assets
        ), key=lambda anchor: (
            anchor.original_path, anchor.role, anchor.sha256,
        ))),
        license_record_sha256=inputs.license_record_sha256,
    )


def _mixed_gate_b_inputs(source_root: Path) -> VidimuSnapshotInputs:
    success, _ = _gate_b_inputs(
        source_root,
        "fixture-gate-b-success",
        inventory_name="scratch-success.json",
    )
    failure, _ = _gate_b_inputs(
        source_root,
        "fixture-gate-b-failure",
        inventory_name="scratch-failure.json",
    )
    failed_record = failure.records[0]
    failed_record.video.local_path.write_bytes(b"")
    failed_video = replace(
        failed_record.video,
        sha256=sha256_file(failed_record.video.local_path),
    )
    source_records = (
        success.records[0],
        replace(failed_record, video=failed_video),
    )
    inventory = source_root / "mixed-frozen-inventory.json"
    inventory.write_text(json.dumps({
        "inventory_schema_version": VIDIMU_INVENTORY_SCHEMA_VERSION,
        "records": [{
            "recording_id": record.provenance.recording_id,
            "video": {
                "original_path": record.video.original_path,
                "role": record.video.role,
                "sha256": record.video.sha256,
            },
            "imu_assets": [{
                "original_path": asset.original_path,
                "role": asset.role,
                "sha256": asset.sha256,
            } for asset in record.imu_assets],
        } for record in source_records],
    }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    inventory_sha256 = sha256_file(inventory)
    records = tuple(
        replace(
            record,
            provenance=replace(
                record.provenance,
                inventory_manifest_sha256=inventory_sha256,
            ),
        )
        for record in source_records
    )
    return VidimuSnapshotInputs(
        records=records,
        expected_recording_ids=tuple(
            record.provenance.recording_id for record in records
        ),
        archive_assets=success.archive_assets,
        inventory_manifest_path=inventory,
        inventory_manifest_sha256=inventory_sha256,
        license_record_path=success.license_record_path,
        license_record_sha256=success.license_record_sha256,
    )


class TestVidimuPtsCvReleaseAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        require_pts_media_toolchain()
        cls._temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._temporary.cleanup)
        root = Path(cls._temporary.name)
        cls.video = generate_cfr_video(root / "media")
        cls.video_sha256 = sha256_file(cls.video)
        cls.primary_root = root / "primary"
        cls.replay_root = root / "replay"
        cls._build_root(cls.primary_root, _RECORDING_IDS)
        cls._build_root(cls.replay_root, _RECORDING_IDS)
        cls.gate_b_source_root = root / "gate-b-source"
        cls.gate_b_inputs, cls.gate_b_imu = _gate_b_inputs(
            cls.gate_b_source_root,
            "fixture-gate-b",
        )
        cls.gate_b_trust_anchors = _gate_b_trust_anchors(cls.gate_b_inputs)
        cls.gate_b_primary = root / "gate-b-primary"
        cls.gate_b_replay = root / "gate-b-replay"
        for output in (cls.gate_b_primary, cls.gate_b_replay):
            finalize_vidimu_snapshot(
                cls.gate_b_inputs,
                output,
                decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                estimator_factory=lambda _record: DeterministicPoseEstimator(),
            )
        cls.mixed_gate_b_source_root = root / "mixed-gate-b-source"
        cls.mixed_gate_b_inputs = _mixed_gate_b_inputs(
            cls.mixed_gate_b_source_root)
        cls.mixed_gate_b_trust_anchors = _gate_b_trust_anchors(
            cls.mixed_gate_b_inputs)
        cls.mixed_gate_b_primary = root / "mixed-gate-b-primary"
        cls.mixed_gate_b_replay = root / "mixed-gate-b-replay"
        for output in (cls.mixed_gate_b_primary, cls.mixed_gate_b_replay):
            outcomes = finalize_vidimu_snapshot(
                cls.mixed_gate_b_inputs,
                output,
                decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                estimator_factory=lambda _record: DeterministicPoseEstimator(),
            )
            if sum(isinstance(item, SourceFailureArtifact) for item in outcomes) != 1:
                raise AssertionError("mixed Gate-B fixture outcome partition failed")

    @classmethod
    def _build_root(
        cls,
        root: Path,
        recording_ids: tuple[str, ...],
        *,
        altered_recording_id: str | None = None,
        mixed_identity_recording_id: str | None = None,
    ) -> None:
        for recording_id in recording_ids:
            provenance = DEFAULT_ESTIMATOR_PROVENANCE
            if recording_id == mixed_identity_recording_id:
                provenance = EstimatorProvenance(
                    model_id=provenance.model_id,
                    model_weights_sha256="9" * 64,
                    preprocessing_config_sha256=(
                        provenance.preprocessing_config_sha256
                    ),
                    inference_environment_id=(
                        provenance.inference_environment_id
                    ),
                )
            estimator = DeterministicPoseEstimator(
                detection_counts=(1,)
                if recording_id == altered_recording_id else (0, 1, 2, 0, 1),
                provenance=provenance,
            )
            finalize_vidimu_recording(
                cls.video,
                root,
                expected_source_video_sha256=cls.video_sha256,
                provenance=_provenance(recording_id),
                decoder=PTSDecoder(DecodeConfig()),
                estimator=estimator,
                validation_gate="GATE_A_SYNTHETIC",
            )

    def _audit_mixed_gate_b(
        self,
        *,
        primary: Path | None = None,
        replay: Path | None = None,
    ) -> dict[str, object]:
        return audit_vidimu_pts_cv_release(
            primary or self.mixed_gate_b_primary,
            expected_recording_ids=(
                self.mixed_gate_b_inputs.expected_recording_ids
            ),
            required_gate="GATE_B_VIDIMU",
            replay_root=replay or self.mixed_gate_b_replay,
            source_root=self.mixed_gate_b_source_root,
            inventory_manifest_path=(
                self.mixed_gate_b_inputs.inventory_manifest_path
            ),
            license_record_path=self.mixed_gate_b_inputs.license_record_path,
            gate_b_trust_anchors=self.mixed_gate_b_trust_anchors,
        )

    def test_two_distinct_gate_a_roots_pass_artifact_replay_audit(self):
        report = audit_vidimu_pts_cv_release(
            self.primary_root,
            expected_recording_ids=_RECORDING_IDS,
            required_gate="GATE_A_SYNTHETIC",
            replay_root=self.replay_root,
        )

        self.assertEqual(report["overall_verdict"], "PASS")
        self.assertEqual(report["inventory_record_count"], 2)
        self.assertEqual(report["decoded_frame_count"], 10)
        self.assertEqual(report["cv_frame_result_count"], 10)
        self.assertEqual(
            report["artifact_replay_status"],
            "BYTE_IDENTICAL_DISTINCT_FILES_AND_ROOTS_PASS",
        )
        self.assertEqual(
            report["independent_rerun_attestation"], "NOT_PROVIDED")
        self.assertEqual(
            report["deterministic_replay_status"],
            "STORED_ARTIFACT_BYTES_IDENTICAL_DISTINCT_FILES_AND_ROOTS_PASS",
        )
        self.assertEqual(
            report["source_assets_present"],
            "NOT_REVERIFIED_FOR_GATE_A_ARTIFACT_AUDIT",
        )
        self.assertEqual(
            report["source_hashes_verified"],
            "PINNED_DECODE_EVIDENCE_VERIFIED_IN_BUNDLES_ONLY",
        )
        self.assertEqual(report["videos_opened"], 2)
        self.assertEqual(report["videos_failed"], 0)
        self.assertEqual(set(report["artifact_hashes"]), set(_RECORDING_IDS))
        self.assertEqual(
            report["frozen_processing_identity"]["model_id"],
            DEFAULT_ESTIMATOR_PROVENANCE.model_id,
        )
        self.assertEqual(set(report["per_recording"]), set(_RECORDING_IDS))
        for recording_id in _RECORDING_IDS:
            evidence = report["per_recording"][recording_id][
                "artifact_replay_evidence"
            ]
            self.assertTrue(evidence["all_artifact_bytes_identical"])
            self.assertTrue(
                evidence["all_corresponding_artifact_files_distinct"])
            self.assertEqual(
                set(evidence["artifact_sha256"]), FINALIZATION_FILES)
            self.assertEqual(
                report["artifact_hashes"][recording_id],
                evidence["artifact_sha256"],
            )

    def test_gate_b_release_binds_inventory_and_reverifies_paired_imu(self):
        report = audit_vidimu_pts_cv_release(
            self.gate_b_primary,
            expected_recording_ids=("fixture-gate-b",),
            required_gate="GATE_B_VIDIMU",
            replay_root=self.gate_b_replay,
            source_root=self.gate_b_source_root,
            inventory_manifest_path=(
                self.gate_b_inputs.inventory_manifest_path
            ),
            license_record_path=self.gate_b_inputs.license_record_path,
            gate_b_trust_anchors=self.gate_b_trust_anchors,
        )

        self.assertEqual(report["overall_verdict"], "PASS")
        self.assertEqual(report["inventory_record_count"], 1)
        self.assertEqual(report["paired_imu_asset_count"], 1)
        self.assertEqual(report["source_archive_asset_count"], 2)
        self.assertEqual(
            report["source_assets_present"],
            "ALL_TRUST_ANCHORED_GATE_B_ASSETS_PRESENT",
        )
        self.assertEqual(
            report["source_hashes_verified"],
            "ALL_TRUST_ANCHORED_GATE_B_ASSET_HASHES_VERIFIED",
        )
        self.assertEqual(
            report["gate_b_trust_anchors"]["expected_dataset_id"],
            self.gate_b_trust_anchors.expected_dataset_id,
        )
        source_evidence = report["per_recording"]["fixture-gate-b"][
            "source_asset_evidence"
        ]
        self.assertTrue(source_evidence["all_inventory_assets_present"])
        self.assertTrue(
            source_evidence["all_inventory_asset_hashes_verified"])
        self.assertEqual(
            source_evidence["source_archives"],
            report["gate_b_trust_anchors"]["source_archives"],
        )
        self.assertTrue(source_evidence["all_source_archives_present"])
        self.assertTrue(
            source_evidence["all_source_archive_hashes_verified"])

    def test_gate_b_release_accepts_exact_success_failure_partition(self):
        report = self._audit_mixed_gate_b()

        self.assertEqual(report["overall_verdict"], "PASS")
        self.assertEqual(report["inventory_record_count"], 2)
        self.assertEqual(report["successful_recording_count"], 1)
        self.assertEqual(report["source_failure_recording_count"], 1)
        self.assertEqual(report["videos_opened"], 2)
        self.assertEqual(report["videos_failed"], 1)
        self.assertEqual(report["decoded_frame_count"], 5)
        self.assertEqual(report["cv_frame_result_count"], 5)
        failure_id = "fixture-gate-b-failure"
        success_id = "fixture-gate-b-success"
        self.assertEqual(
            report["per_recording"][failure_id]["recording_outcome"],
            "SOURCE_DECODE_FAILURE",
        )
        self.assertEqual(
            report["per_recording"][success_id]["recording_outcome"],
            "SUCCESS",
        )
        failure_replay = report["per_recording"][failure_id][
            "artifact_replay_evidence"
        ]
        self.assertEqual(
            set(failure_replay["artifact_sha256"]), SOURCE_FAILURE_FILES)
        self.assertTrue(failure_replay["all_artifact_bytes_identical"])
        self.assertTrue(
            failure_replay["all_corresponding_artifact_files_distinct"])

    def test_gate_b_release_rejects_missing_extra_or_double_failure_evidence(self):
        failure_id = "fixture-gate-b-failure"
        success_id = "fixture-gate-b-success"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            missing = root / "missing"
            shutil.copytree(self.mixed_gate_b_primary, missing)
            failure_dir = next((missing / failure_id).iterdir())
            shutil.rmtree(failure_dir)
            with self.assertRaises(ReleaseAuditError):
                self._audit_mixed_gate_b(primary=missing)

            extra = root / "extra"
            shutil.copytree(self.mixed_gate_b_primary, extra)
            failure_dir = next((extra / failure_id).iterdir())
            (failure_dir / "unexpected.txt").write_text(
                "not frozen\n", encoding="utf-8")
            with self.assertRaises(ReleaseAuditError):
                self._audit_mixed_gate_b(primary=extra)

            double = root / "double"
            shutil.copytree(self.mixed_gate_b_primary, double)
            success_bundle = next((double / success_id).iterdir())
            shutil.copytree(
                success_bundle,
                double / failure_id / "second-success-outcome",
            )
            with self.assertRaisesRegex(
                ReleaseAuditError, "exactly one selected",
            ):
                self._audit_mixed_gate_b(primary=double)

    def test_gate_b_release_rejects_failure_artifact_tampering(self):
        failure_id = "fixture-gate-b-failure"
        with tempfile.TemporaryDirectory() as temporary:
            tampered = Path(temporary) / "tampered"
            shutil.copytree(self.mixed_gate_b_primary, tampered)
            failure_dir = next((tampered / failure_id).iterdir())
            manifest_path = failure_dir / "source_failure_manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["failure"]["detail_code"] = "SUBSTITUTED_DETAIL"
            manifest_path.write_bytes(canonical_json_bytes(manifest))

            with self.assertRaises(ReleaseAuditError):
                self._audit_mixed_gate_b(primary=tampered)

    def test_gate_b_release_rejects_hardlinked_failure_artifacts(self):
        failure_id = "fixture-gate-b-failure"
        with tempfile.TemporaryDirectory() as temporary:
            replay = Path(temporary) / "hardlinked-replay"
            shutil.copytree(self.mixed_gate_b_replay, replay)
            source_bundle = next(
                (self.mixed_gate_b_primary / failure_id).iterdir())
            replay_bundle = next((replay / failure_id).iterdir())
            for filename in SOURCE_FAILURE_FILES:
                (replay_bundle / filename).unlink()
                os.link(source_bundle / filename, replay_bundle / filename)

            with self.assertRaisesRegex(ReleaseAuditError, "share a file inode"):
                self._audit_mixed_gate_b(replay=replay)

    def test_gate_b_release_rejects_changed_paired_imu_bytes(self):
        original = self.gate_b_imu.read_bytes()
        try:
            self.gate_b_imu.write_bytes(original + b"tampered\n")
            with self.assertRaises(ReleaseAuditError):
                audit_vidimu_pts_cv_release(
                    self.gate_b_primary,
                    expected_recording_ids=("fixture-gate-b",),
                    required_gate="GATE_B_VIDIMU",
                    replay_root=self.gate_b_replay,
                    source_root=self.gate_b_source_root,
                    inventory_manifest_path=(
                        self.gate_b_inputs.inventory_manifest_path
                    ),
                    license_record_path=self.gate_b_inputs.license_record_path,
                    gate_b_trust_anchors=self.gate_b_trust_anchors,
                )
        finally:
            self.gate_b_imu.write_bytes(original)

    def test_gate_b_release_rehashes_every_archive(self):
        for asset in self.gate_b_inputs.archive_assets:
            original = asset.local_path.read_bytes()
            try:
                asset.local_path.write_bytes(original + b"tampered\n")
                with self.subTest(asset.role), self.assertRaises(
                    ReleaseAuditError
                ):
                    audit_vidimu_pts_cv_release(
                        self.gate_b_primary,
                        expected_recording_ids=("fixture-gate-b",),
                        required_gate="GATE_B_VIDIMU",
                        replay_root=self.gate_b_replay,
                        source_root=self.gate_b_source_root,
                        inventory_manifest_path=(
                            self.gate_b_inputs.inventory_manifest_path
                        ),
                        license_record_path=(
                            self.gate_b_inputs.license_record_path
                        ),
                        gate_b_trust_anchors=self.gate_b_trust_anchors,
                    )
            finally:
                asset.local_path.write_bytes(original)

    def test_gate_b_archive_anchors_are_exact_and_role_bound(self):
        anchors = self.gate_b_trust_anchors.source_archives
        dataset, video = anchors
        invalid_sets = {
            "omitted": (dataset,),
            "extra": (dataset, video, GateBArchiveAnchor(
                original_path="archives/extra.zip",
                role="VIDEO_ARCHIVE",
                sha256="9" * 64,
            )),
        }
        for label, source_archives in invalid_sets.items():
            with self.subTest(label), self.assertRaises(ReleaseAuditError):
                GateBTrustAnchors(
                    expected_dataset_id=(
                        self.gate_b_trust_anchors.expected_dataset_id
                    ),
                    expected_dataset_version=(
                        self.gate_b_trust_anchors.expected_dataset_version
                    ),
                    inventory_manifest_sha256=(
                        self.gate_b_trust_anchors.inventory_manifest_sha256
                    ),
                    source_archives=source_archives,
                    license_record_sha256=(
                        self.gate_b_trust_anchors.license_record_sha256
                    ),
                )

        swapped = tuple(sorted((
            GateBArchiveAnchor(
                original_path=dataset.original_path,
                role="VIDEO_ARCHIVE",
                sha256=dataset.sha256,
            ),
            GateBArchiveAnchor(
                original_path=video.original_path,
                role="DATASET_ARCHIVE",
                sha256=video.sha256,
            ),
        ), key=lambda anchor: (
            anchor.original_path, anchor.role, anchor.sha256,
        )))
        substituted = GateBTrustAnchors(
            expected_dataset_id=self.gate_b_trust_anchors.expected_dataset_id,
            expected_dataset_version=(
                self.gate_b_trust_anchors.expected_dataset_version
            ),
            inventory_manifest_sha256=(
                self.gate_b_trust_anchors.inventory_manifest_sha256
            ),
            source_archives=swapped,
            license_record_sha256=(
                self.gate_b_trust_anchors.license_record_sha256
            ),
        )
        with self.assertRaisesRegex(ReleaseAuditError, "archives disagree"):
            audit_vidimu_pts_cv_release(
                self.gate_b_primary,
                expected_recording_ids=("fixture-gate-b",),
                required_gate="GATE_B_VIDIMU",
                replay_root=self.gate_b_replay,
                source_root=self.gate_b_source_root,
                inventory_manifest_path=(
                    self.gate_b_inputs.inventory_manifest_path
                ),
                license_record_path=self.gate_b_inputs.license_record_path,
                gate_b_trust_anchors=substituted,
            )

    def test_independent_inventory_parser_rejects_omission_extra_and_escape(self):
        baseline = json.loads(
            self.gate_b_inputs.inventory_manifest_path.read_text(encoding="utf-8")
        )
        cases: dict[str, dict[str, object]] = {}

        omitted = json.loads(json.dumps(baseline))
        omitted["records"][0]["imu_assets"] = []
        cases["omitted paired IMU"] = omitted

        duplicated = json.loads(json.dumps(baseline))
        duplicated["records"][0]["imu_assets"].append(dict(
            duplicated["records"][0]["imu_assets"][0]
        ))
        cases["extra duplicate paired IMU"] = duplicated

        escaped = json.loads(json.dumps(baseline))
        escaped["records"][0]["imu_assets"][0]["original_path"] = "../escape.raw"
        cases["path escape"] = escaped

        for label, payload in cases.items():
            with self.subTest(label), self.assertRaises(ReleaseAuditError):
                release_audit_module._parse_frozen_inventory(
                    json.dumps(payload).encode("utf-8"))

    def test_gate_b_release_rejects_mixed_inventory_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            first, _ = _gate_b_inputs(
                source,
                "mixed-a",
                inventory_name="inventory-a.json",
            )
            second, _ = _gate_b_inputs(
                source,
                "mixed-b",
                inventory_name="inventory-b.json",
            )
            primary = root / "primary"
            replay = root / "replay"
            for output in (primary, replay):
                for inputs in (first, second):
                    finalize_vidimu_snapshot(
                        inputs,
                        output,
                        decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                        estimator_factory=(
                            lambda _record: DeterministicPoseEstimator()
                        ),
                    )

            with self.assertRaises(ReleaseAuditError):
                audit_vidimu_pts_cv_release(
                    primary,
                    expected_recording_ids=("mixed-a", "mixed-b"),
                    required_gate="GATE_B_VIDIMU",
                    replay_root=replay,
                    source_root=source,
                    inventory_manifest_path=first.inventory_manifest_path,
                    license_record_path=first.license_record_path,
                    gate_b_trust_anchors=_gate_b_trust_anchors(first),
                )

    def test_gate_b_release_rejects_mixed_archive_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            first, _ = _gate_b_inputs(
                source,
                "archive-mixed-a",
                inventory_name="scratch-a.json",
                archive_variant="a",
            )
            second, _ = _gate_b_inputs(
                source,
                "archive-mixed-b",
                inventory_name="scratch-b.json",
                archive_variant="b",
            )
            combined_inventory = source / "combined-inventory.json"
            records = []
            for inputs in (first, second):
                record = inputs.records[0]
                records.append({
                    "recording_id": record.provenance.recording_id,
                    "video": {
                        "original_path": record.video.original_path,
                        "role": record.video.role,
                        "sha256": record.video.sha256,
                    },
                    "imu_assets": [{
                        "original_path": asset.original_path,
                        "role": asset.role,
                        "sha256": asset.sha256,
                    } for asset in record.imu_assets],
                })
            combined_inventory.write_text(json.dumps({
                "inventory_schema_version": VIDIMU_INVENTORY_SCHEMA_VERSION,
                "records": records,
            }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            inventory_hash = sha256_file(combined_inventory)
            combined_records = tuple(
                VidimuSnapshotRecord(
                    provenance=RecordingProvenance(
                        **{
                            **record.provenance.as_manifest(),
                            "inventory_manifest_sha256": inventory_hash,
                        }
                    ),
                    video=record.video,
                    imu_assets=record.imu_assets,
                )
                for record in (first.records[0], second.records[0])
            )

            def combined_inputs(
                archive_assets: tuple[FrozenSourceAsset, ...],
            ) -> VidimuSnapshotInputs:
                return VidimuSnapshotInputs(
                    records=combined_records,
                    expected_recording_ids=(
                        "archive-mixed-a", "archive-mixed-b",
                    ),
                    archive_assets=archive_assets,
                    inventory_manifest_path=combined_inventory,
                    inventory_manifest_sha256=inventory_hash,
                    license_record_path=first.license_record_path,
                    license_record_sha256=first.license_record_sha256,
                )

            archive_a_inputs = combined_inputs(first.archive_assets)
            archive_b_inputs = combined_inputs(second.archive_assets)
            roots: dict[str, Path] = {}
            for label, inputs in (
                ("a-primary", archive_a_inputs),
                ("a-replay", archive_a_inputs),
                ("b-primary", archive_b_inputs),
                ("b-replay", archive_b_inputs),
            ):
                roots[label] = root / label
                finalize_vidimu_snapshot(
                    inputs,
                    roots[label],
                    decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                    estimator_factory=(
                        lambda _record: DeterministicPoseEstimator()
                    ),
                )

            mixed_primary = root / "mixed-primary"
            mixed_replay = root / "mixed-replay"
            for recording_id, variant in (
                ("archive-mixed-a", "a"),
                ("archive-mixed-b", "b"),
            ):
                shutil.copytree(
                    roots[f"{variant}-primary"] / recording_id,
                    mixed_primary / recording_id,
                )
                shutil.copytree(
                    roots[f"{variant}-replay"] / recording_id,
                    mixed_replay / recording_id,
                )

            with self.assertRaisesRegex(ReleaseAuditError, "archives disagree"):
                audit_vidimu_pts_cv_release(
                    mixed_primary,
                    expected_recording_ids=(
                        "archive-mixed-a", "archive-mixed-b",
                    ),
                    required_gate="GATE_B_VIDIMU",
                    replay_root=mixed_replay,
                    source_root=source,
                    inventory_manifest_path=combined_inventory,
                    license_record_path=first.license_record_path,
                    gate_b_trust_anchors=_gate_b_trust_anchors(
                        archive_a_inputs),
                )

    def test_gate_b_release_requires_external_trust_anchors(self):
        with self.assertRaisesRegex(ReleaseAuditError, "trust anchors"):
            audit_vidimu_pts_cv_release(
                self.gate_b_primary,
                expected_recording_ids=("fixture-gate-b",),
                required_gate="GATE_B_VIDIMU",
                replay_root=self.gate_b_replay,
                source_root=self.gate_b_source_root,
                inventory_manifest_path=(
                    self.gate_b_inputs.inventory_manifest_path
                ),
                license_record_path=self.gate_b_inputs.license_record_path,
            )

    def test_gate_b_release_rejects_self_declared_substituted_identity(self):
        trusted_external_anchor = GateBTrustAnchors(
            expected_dataset_id="trusted-vidimu-dataset",
            expected_dataset_version=(
                self.gate_b_trust_anchors.expected_dataset_version
            ),
            inventory_manifest_sha256=(
                self.gate_b_trust_anchors.inventory_manifest_sha256
            ),
            source_archives=self.gate_b_trust_anchors.source_archives,
            license_record_sha256=(
                self.gate_b_trust_anchors.license_record_sha256
            ),
        )
        with self.assertRaisesRegex(ReleaseAuditError, "trust anchors"):
            audit_vidimu_pts_cv_release(
                self.gate_b_primary,
                expected_recording_ids=("fixture-gate-b",),
                required_gate="GATE_B_VIDIMU",
                replay_root=self.gate_b_replay,
                source_root=self.gate_b_source_root,
                inventory_manifest_path=(
                    self.gate_b_inputs.inventory_manifest_path
                ),
                license_record_path=self.gate_b_inputs.license_record_path,
                gate_b_trust_anchors=trusted_external_anchor,
            )

    def test_release_pass_requires_a_distinct_artifact_tree(self):
        with self.assertRaises(ReleaseAuditError):
            audit_vidimu_pts_cv_release(
                self.primary_root,
                expected_recording_ids=_RECORDING_IDS,
                required_gate="GATE_A_SYNTHETIC",
                replay_root=None,
            )

    def test_same_root_cannot_masquerade_as_distinct_artifact_tree(self):
        with self.assertRaises(ReleaseAuditError):
            audit_vidimu_pts_cv_release(
                self.primary_root,
                expected_recording_ids=_RECORDING_IDS,
                required_gate="GATE_A_SYNTHETIC",
                replay_root=self.primary_root,
            )

    def test_hardlinked_artifact_tree_is_not_distinct_replay_evidence(self):
        hardlink_root = Path(self._temporary.name) / "hardlink-replay"
        for recording_id in _RECORDING_IDS:
            source_recording = self.primary_root / recording_id
            source_bundle = next(source_recording.iterdir())
            target_bundle = (
                hardlink_root / recording_id / source_bundle.name
            )
            target_bundle.mkdir(parents=True)
            for filename in FINALIZATION_FILES:
                os.link(source_bundle / filename, target_bundle / filename)

        with self.assertRaisesRegex(ReleaseAuditError, "share a file inode"):
            audit_vidimu_pts_cv_release(
                self.primary_root,
                expected_recording_ids=_RECORDING_IDS,
                required_gate="GATE_A_SYNTHETIC",
                replay_root=hardlink_root,
            )

    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "mkfifo"),
        "FIFO semantics require POSIX",
    )
    def test_release_source_readers_reject_fifo_without_blocking(self):
        fifo = Path(self._temporary.name) / "substituted-source.fifo"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ReleaseAuditError, "not regular"):
            release_audit_module._pinned_sha256(fifo)
        with self.assertRaisesRegex(ReleaseAuditError, "bounded regular file"):
            release_audit_module._pinned_bytes(fifo, max_bytes=16)
        with self.assertRaisesRegex(ReleaseAuditError, "recording ID manifest"):
            release_audit_module._load_recording_ids(fifo)

    def test_release_rejects_mixed_processing_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "primary"
            replay = root / "replay"
            for output in (primary, replay):
                self._build_root(
                    output,
                    _RECORDING_IDS,
                    mixed_identity_recording_id=_RECORDING_IDS[1],
                )

            with self.assertRaisesRegex(ReleaseAuditError, "mixed model"):
                audit_vidimu_pts_cv_release(
                    primary,
                    expected_recording_ids=_RECORDING_IDS,
                    required_gate="GATE_A_SYNTHETIC",
                    replay_root=replay,
                )

    def test_one_valid_but_altered_replay_bundle_is_rejected(self):
        altered = Path(self._temporary.name) / "altered-replay"
        self._build_root(
            altered,
            _RECORDING_IDS,
            altered_recording_id=_RECORDING_IDS[1],
        )
        with self.assertRaises(ReleaseAuditError):
            audit_vidimu_pts_cv_release(
                self.primary_root,
                expected_recording_ids=_RECORDING_IDS,
                required_gate="GATE_A_SYNTHETIC",
                replay_root=altered,
            )

    def test_missing_recording_bundle_is_rejected(self):
        missing = Path(self._temporary.name) / "missing-replay"
        self._build_root(missing, (_RECORDING_IDS[0],))
        with self.assertRaises(ReleaseAuditError):
            audit_vidimu_pts_cv_release(
                self.primary_root,
                expected_recording_ids=_RECORDING_IDS,
                required_gate="GATE_A_SYNTHETIC",
                replay_root=missing,
            )

    def test_extra_recording_bundle_is_rejected(self):
        extra = Path(self._temporary.name) / "extra-primary"
        self._build_root(extra, (*_RECORDING_IDS, "fixture-release-extra"))
        with self.assertRaises(ReleaseAuditError):
            audit_vidimu_pts_cv_release(
                extra,
                expected_recording_ids=_RECORDING_IDS,
                required_gate="GATE_A_SYNTHETIC",
                replay_root=self.replay_root,
            )


if __name__ == "__main__":
    unittest.main()
