"""Fail-closed tests for the complete Gate-B source preflight."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

from motionbloom.tremora_store.decode.pts_decoder import (
    DecodeConfig,
    DecodeError,
    PTSDecoder,
    VerifiedSourceDecodeError,
)
from motionbloom.tremora_store.finalize._bundle_io import (
    TABLE_FILES,
    FinalizationBundleError,
    FinalizationBundleWriter,
    read_json,
    read_table,
)
from motionbloom.tremora_store.finalize.audit_finalized_recording import (
    build_finalization_audit,
)
from motionbloom.tremora_store.finalize.finalize_vidimu_recording import (
    RecordingFinalizationError,
    RecordingProvenance,
)
from motionbloom.tremora_store.finalize.finalize_vidimu_snapshot import (
    VIDIMU_INVENTORY_SCHEMA_VERSION,
    FrozenSourceAsset,
    VidimuSnapshotInputs,
    VidimuSnapshotRecord,
    finalize_vidimu_snapshot,
    preflight_vidimu_snapshot,
)
from motionbloom.tremora_store.finalize.source_failure_artifact import (
    SOURCE_FAILURE_CATEGORY,
    SOURCE_FAILURE_DETAIL_CODE,
    SOURCE_FAILURE_STAGE,
    SourceFailureArtifact,
    audit_source_failure_artifact,
    build_source_failure_audit,
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


class TestVidimuSnapshotPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        require_pts_media_toolchain()
        cls._temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._temporary.cleanup)
        root = Path(cls._temporary.name)
        source = root / "source"
        source.mkdir()
        video = generate_cfr_video(source / "video")
        imu = source / "imu.raw"
        imu.write_bytes(b"synthetic IMU presence fixture\n")
        dataset_archive = source / "dataset.zip"
        dataset_archive.write_bytes(b"synthetic dataset archive fixture\n")
        video_archive = source / "videosmallsize.zip"
        video_archive.write_bytes(b"synthetic video archive fixture\n")
        license_record = source / "license.txt"
        license_record.write_bytes(b"synthetic license evidence fixture\n")

        archive_assets = (
            FrozenSourceAsset(
                original_path="archives/dataset.zip",
                local_path=dataset_archive,
                sha256=sha256_file(dataset_archive),
                role="DATASET_ARCHIVE",
            ),
            FrozenSourceAsset(
                original_path="archives/videosmallsize.zip",
                local_path=video_archive,
                sha256=sha256_file(video_archive),
                role="VIDEO_ARCHIVE",
            ),
        )
        video_asset = FrozenSourceAsset(
            original_path="videosoriginal/S01/fixture-public.mp4",
            local_path=video,
            sha256=sha256_file(video),
            role="ORIGINAL_VIDEO",
        )
        imu_asset = FrozenSourceAsset(
            original_path="imu/S01/fixture-public.raw",
            local_path=imu,
            sha256=sha256_file(imu),
            role="IMU",
        )
        inventory = source / "inventory.json"
        inventory.write_text(json.dumps({
            "inventory_schema_version": VIDIMU_INVENTORY_SCHEMA_VERSION,
            "records": [{
                "recording_id": "fixture-public",
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
            recording_id="fixture-public",
            source_kind="VIDIMU_PUBLIC",
            source_original_path=video_asset.original_path,
            source_object_id="vidimu:fixture-public",
            materialization_date="2026-08-27",
            license_id="vidimu-license-fixture",
            license_record_sha256=sha256_file(license_record),
            inventory_manifest_sha256=sha256_file(inventory),
        )
        cls.inputs = VidimuSnapshotInputs(
            records=(VidimuSnapshotRecord(
                provenance=provenance,
                video=video_asset,
                imu_assets=(imu_asset,),
            ),),
            expected_recording_ids=(provenance.recording_id,),
            archive_assets=archive_assets,
            inventory_manifest_path=inventory,
            inventory_manifest_sha256=sha256_file(inventory),
            license_record_path=license_record,
            license_record_sha256=sha256_file(license_record),
        )

    def _with_inventory(
        self,
        payload: dict[str, object],
        *,
        name: str,
        records: tuple[VidimuSnapshotRecord, ...] | None = None,
    ) -> VidimuSnapshotInputs:
        path = Path(self._temporary.name) / name
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        inventory_hash = sha256_file(path)
        selected_records = records or self.inputs.records
        updated_records = tuple(
            replace(
                record,
                provenance=replace(
                    record.provenance,
                    inventory_manifest_sha256=inventory_hash,
                ),
            )
            for record in selected_records
        )
        return replace(
            self.inputs,
            records=updated_records,
            expected_recording_ids=tuple(
                record.provenance.recording_id for record in updated_records
            ),
            inventory_manifest_path=path,
            inventory_manifest_sha256=inventory_hash,
        )

    def _inventory_payload(self) -> dict[str, object]:
        record = self.inputs.records[0]
        return {
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
            }],
        }

    def _with_video_bytes(
        self,
        payload: bytes,
        *,
        name: str,
    ) -> VidimuSnapshotInputs:
        path = Path(self._temporary.name) / name
        path.write_bytes(payload)
        record = self.inputs.records[0]
        video = replace(
            record.video,
            local_path=path,
            sha256=sha256_file(path),
        )
        inventory = self._inventory_payload()
        inventory["records"][0]["video"]["sha256"] = video.sha256
        return self._with_inventory(
            inventory,
            name=f"{name}-inventory.json",
            records=(replace(record, video=video),),
        )

    def _assert_blocks_before_factories(self, inputs: VidimuSnapshotInputs) -> None:
        with self.assertRaises(RecordingFinalizationError):
            preflight_vidimu_snapshot(inputs)
        decoder_factory = mock.Mock(side_effect=AssertionError(
            "preflight failure reached decoder construction"
        ))
        estimator_factory = mock.Mock(side_effect=AssertionError(
            "preflight failure reached estimator construction"
        ))
        output = Path(self._temporary.name) / "must-not-publish"
        with self.assertRaises(RecordingFinalizationError):
            finalize_vidimu_snapshot(
                inputs,
                output,
                decoder_factory=decoder_factory,
                estimator_factory=estimator_factory,
            )
        decoder_factory.assert_not_called()
        estimator_factory.assert_not_called()
        self.assertEqual(list(output.rglob("_SUCCESS")), [])

    def test_valid_machine_readable_inventory_passes_preflight(self):
        self.assertIsNone(preflight_vidimu_snapshot(self.inputs))

    def test_hash_verified_zero_byte_video_publishes_atomic_failure_outcome(self):
        inputs = self._with_video_bytes(b"", name="zero-byte-source.mp4")
        estimator = DeterministicPoseEstimator()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "finalized"
            outcome = finalize_vidimu_snapshot(
                inputs,
                output,
                decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                estimator_factory=lambda _record: estimator,
            )[0]
            self.assertIsInstance(outcome, SourceFailureArtifact)
            assert isinstance(outcome, SourceFailureArtifact)
            audit = audit_source_failure_artifact(outcome.path)
            manifest, encoded = read_json(
                outcome.path / "source_failure_manifest.json")

            self.assertEqual(outcome.path.name, outcome.intended_finalization_id)
            self.assertEqual(outcome.failure_id, outcome.intended_finalization_id)
            self.assertEqual(audit["recording_outcome"], "SOURCE_DECODE_FAILURE")
            self.assertEqual(audit["videos_opened"], 1)
            self.assertEqual(audit["videos_failed"], 1)
            self.assertEqual(audit["decoded_frame_count"], 0)
            self.assertEqual(manifest["source"]["source_video_bytes"], 0)
            self.assertEqual(manifest["failure"], {
                "category": SOURCE_FAILURE_CATEGORY,
                "detail_code": SOURCE_FAILURE_DETAIL_CODE,
                "stage": SOURCE_FAILURE_STAGE,
            })
            self.assertNotIn(str(Path(self._temporary.name)), encoded.decode("ascii"))
            self.assertEqual(estimator.inference_frame_ids, [])
            self.assertEqual(list(output.rglob("_SUCCESS")), [])
            self.assertEqual(len(list(output.rglob("_FAILURE"))), 1)
            self.assertEqual(list(output.rglob(".staging-*")), [])

            changed_schema = copy.deepcopy(manifest)
            changed_schema["identity_inputs"][
                "finalization_schema_version"
            ] = "0.3.0-substituted"
            with self.assertRaises(FinalizationBundleError):
                build_source_failure_audit(
                    manifest=changed_schema,
                    manifest_sha256="0" * 64,
                )
            boolean_version = copy.deepcopy(manifest)
            boolean_version["manifest_version"] = True
            with self.assertRaises(FinalizationBundleError):
                build_source_failure_audit(
                    manifest=boolean_version,
                    manifest_sha256="0" * 64,
                )

            with self.assertRaises(FinalizationBundleError):
                FinalizationBundleWriter(
                    output,
                    recording_id=inputs.expected_recording_ids[0],
                    finalization_id=outcome.intended_finalization_id,
                )

            reused = finalize_vidimu_snapshot(
                inputs,
                output,
                decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                estimator_factory=lambda _record: DeterministicPoseEstimator(),
            )[0]
            self.assertIsInstance(reused, SourceFailureArtifact)
            self.assertTrue(reused.reused_existing)
            self.assertEqual(reused.path, outcome.path)

    def test_failure_binds_the_single_provenance_consumed_inside_finalizer(self):
        first = DEFAULT_ESTIMATOR_PROVENANCE
        second = replace(first, model_weights_sha256="9" * 64)

        class RotatingProvenanceEstimator(DeterministicPoseEstimator):
            def __init__(self) -> None:
                super().__init__(provenance=first)
                self.provenance_reads = 0

            @property
            def provenance(self):
                self.provenance_reads += 1
                return first if self.provenance_reads == 1 else second

        inputs = self._with_video_bytes(
            b"", name="rotating-provenance-source.mp4")
        estimator = RotatingProvenanceEstimator()
        with tempfile.TemporaryDirectory() as temporary:
            outcome = finalize_vidimu_snapshot(
                inputs,
                Path(temporary) / "finalized",
                decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                estimator_factory=lambda _record: estimator,
            )[0]
            self.assertIsInstance(outcome, SourceFailureArtifact)
            assert isinstance(outcome, SourceFailureArtifact)
            manifest, _ = read_json(
                outcome.path / "source_failure_manifest.json")

        self.assertEqual(estimator.provenance_reads, 1)
        self.assertEqual(manifest["estimator"], asdict(first))
        for field, value in asdict(first).items():
            self.assertEqual(manifest["identity_inputs"][field], value)

    def test_gate_b_rejects_decoder_subclass_before_it_can_forge_failure(self):
        class ForgedDecoder(PTSDecoder):
            def decode(self, *args, **kwargs):
                raise VerifiedSourceDecodeError("forged eligible failure")

        estimator_factory = mock.Mock(side_effect=AssertionError(
            "untrusted decoder reached estimator construction"
        ))
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            RecordingFinalizationError, "exact PTSDecoder",
        ):
            finalize_vidimu_snapshot(
                self.inputs,
                Path(temporary) / "finalized",
                decoder_factory=lambda: ForgedDecoder(DecodeConfig()),
                estimator_factory=estimator_factory,
            )
        estimator_factory.assert_not_called()

    def test_gate_b_discards_factory_instance_method_shadowing(self):
        candidate = PTSDecoder(DecodeConfig())
        forged_decode = mock.Mock(side_effect=VerifiedSourceDecodeError(
            "forged eligible failure"
        ))
        candidate.decode = forged_decode  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as temporary:
            outcome = finalize_vidimu_snapshot(
                self.inputs,
                Path(temporary) / "finalized",
                decoder_factory=lambda: candidate,
                estimator_factory=lambda _record: DeterministicPoseEstimator(),
            )[0]
        forged_decode.assert_not_called()
        self.assertNotIsInstance(outcome, SourceFailureArtifact)

    def test_ordinary_decode_integrity_fault_remains_snapshot_no_go(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "finalized"
            with mock.patch.object(
                PTSDecoder,
                "decode",
                side_effect=DecodeError("source video changed during pinned decode"),
            ), self.assertRaises(DecodeError):
                finalize_vidimu_snapshot(
                    self.inputs,
                    output,
                    decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                    estimator_factory=(
                        lambda _record: DeterministicPoseEstimator()
                    ),
                )
            self.assertEqual(list(output.rglob("_SUCCESS")), [])
            self.assertEqual(list(output.rglob("_FAILURE")), [])

    def test_estimator_cannot_forge_source_failure_from_provenance_property(self):
        class HostileEstimator(DeterministicPoseEstimator):
            def __init__(self) -> None:
                super().__init__()
                self.provenance_reads = 0

            @property
            def provenance(self):
                self.provenance_reads += 1
                if self.provenance_reads == 1:
                    return super().provenance
                raise VerifiedSourceDecodeError(
                    "estimator-forged eligible decode failure")

        estimator = HostileEstimator()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "finalized"
            with self.assertRaises(VerifiedSourceDecodeError):
                finalize_vidimu_snapshot(
                    self.inputs,
                    output,
                    decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                    estimator_factory=lambda _record: estimator,
                )
            self.assertEqual(estimator.provenance_reads, 2)
            self.assertEqual(estimator.inference_frame_ids, [])
            self.assertEqual(list(output.rglob("_SUCCESS")), [])
            self.assertEqual(list(output.rglob("_FAILURE")), [])

    def test_wrong_configured_stream_is_not_documented_as_source_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "finalized"
            with self.assertRaises(DecodeError):
                finalize_vidimu_snapshot(
                    self.inputs,
                    output,
                    decoder_factory=lambda: PTSDecoder(DecodeConfig(
                        stream_index=99,
                    )),
                    estimator_factory=(
                        lambda _record: DeterministicPoseEstimator()
                    ),
                )
            self.assertEqual(list(output.rglob("_FAILURE")), [])

    def test_memory_error_is_not_documented_as_source_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "finalized"
            with mock.patch.object(
                PTSDecoder,
                "_decode_handle",
                side_effect=MemoryError("simulated allocation failure"),
            ), self.assertRaises(MemoryError):
                finalize_vidimu_snapshot(
                    self.inputs,
                    output,
                    decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                    estimator_factory=(
                        lambda _record: DeterministicPoseEstimator()
                    ),
                )
            self.assertEqual(list(output.rglob("_FAILURE")), [])

    def test_mutated_frozen_decoder_config_is_revalidated_and_rejected(self):
        candidate = PTSDecoder(DecodeConfig())
        object.__setattr__(candidate.config, "thread_count", 2)
        estimator_factory = mock.Mock(side_effect=AssertionError(
            "invalid config reached estimator construction"
        ))
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            RecordingFinalizationError, "invalid frozen DecodeConfig",
        ):
            finalize_vidimu_snapshot(
                self.inputs,
                Path(temporary) / "finalized",
                decoder_factory=lambda: candidate,
                estimator_factory=estimator_factory,
            )
        estimator_factory.assert_not_called()

    def test_gate_b_manifest_contains_canonical_imu_and_archive_evidence(self):
        reversed_archives = replace(
            self.inputs,
            archive_assets=tuple(reversed(self.inputs.archive_assets)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            finalized = finalize_vidimu_snapshot(
                reversed_archives,
                Path(temporary) / "finalized",
                decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                estimator_factory=lambda _record: DeterministicPoseEstimator(),
            )
            manifest, _ = read_json(
                finalized[0].path / "finalization_manifest.json")

        source = manifest["source"]
        self.assertEqual(source["paired_imu_assets"], [{
            "original_path": self.inputs.records[0].imu_assets[0].original_path,
            "role": "IMU",
            "sha256": self.inputs.records[0].imu_assets[0].sha256,
        }])
        self.assertNotIn("local_path", source["paired_imu_assets"][0])
        self.assertEqual(source["source_archives"], [{
            "original_path": asset.original_path,
            "role": asset.role,
            "sha256": asset.sha256,
        } for asset in sorted(
            self.inputs.archive_assets,
            key=lambda item: (
                item.original_path, item.role, item.sha256,
            ),
        )])
        self.assertTrue(all(
            "local_path" not in asset for asset in source["source_archives"]
        ))
        self.assertNotIn("source_archive_sha256", source)
        self.assertNotIn("source_archive_object_id", source)

    def test_strict_audit_rejects_archive_evidence_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            finalized = finalize_vidimu_snapshot(
                self.inputs,
                Path(temporary) / "finalized",
                decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                estimator_factory=lambda _record: DeterministicPoseEstimator(),
            )[0]
            manifest, _ = read_json(
                finalized.path / "finalization_manifest.json")
            tables = {
                name: read_table(finalized.path / filename)
                for name, filename in TABLE_FILES.items()
            }

        archives = manifest["source"]["source_archives"]
        cases: dict[str, dict[str, object]] = {}

        removed = copy.deepcopy(manifest)
        removed["source"].pop("source_archives")
        cases["removed"] = removed

        reordered = copy.deepcopy(manifest)
        reordered["source"]["source_archives"].reverse()
        cases["reordered"] = reordered

        duplicated = copy.deepcopy(manifest)
        duplicated["source"]["source_archives"].append(
            copy.deepcopy(archives[0]))
        cases["duplicated"] = duplicated

        role_swapped = copy.deepcopy(manifest)
        role_swapped["source"]["source_archives"][0]["role"] = (
            role_swapped["source"]["source_archives"][1]["role"]
        )
        cases["role substituted"] = role_swapped

        extra = copy.deepcopy(manifest)
        extra["source"]["source_archives"].append({
            "original_path": "archives/extra.zip",
            "role": "VIDEO_ARCHIVE",
            "sha256": "9" * 64,
        })
        cases["extra"] = extra

        unsafe = copy.deepcopy(manifest)
        unsafe["source"]["source_archives"][0]["original_path"] = "../escape.zip"
        cases["unsafe path"] = unsafe

        aliased = copy.deepcopy(manifest)
        aliased["source"]["source_archives"][0]["original_path"] = (
            aliased["source"]["source_original_path"]
        )
        cases["video alias"] = aliased

        legacy = copy.deepcopy(manifest)
        legacy["source"]["source_archive_sha256"] = "7" * 64
        legacy["source"]["source_archive_object_id"] = "archives/legacy.zip"
        cases["legacy singular fields"] = legacy

        gate_a_injection = copy.deepcopy(manifest)
        gate_a_injection["validation_gate"] = "GATE_A_REAL_VIDEO_PILOT"
        gate_a_injection["source"].pop("paired_imu_assets")
        cases["Gate A injection"] = gate_a_injection

        for label, changed in cases.items():
            with self.subTest(label), self.assertRaises(
                FinalizationBundleError
            ):
                build_finalization_audit(
                    manifest=changed,
                    manifest_sha256="0" * 64,
                    tables=tables,
                )

    def test_missing_video_is_rejected_before_finalization(self):
        record = self.inputs.records[0]
        missing = replace(
            record.video,
            local_path=Path(self._temporary.name) / "missing-video.mp4",
        )
        self._assert_blocks_before_factories(replace(
            self.inputs,
            records=(replace(record, video=missing),),
        ))

    def test_missing_imu_is_rejected_before_finalization(self):
        record = self.inputs.records[0]
        self._assert_blocks_before_factories(replace(
            self.inputs,
            records=(replace(record, imu_assets=()),),
        ))

    def test_unavailable_imu_file_is_rejected_before_finalization(self):
        record = self.inputs.records[0]
        missing = replace(
            record.imu_assets[0],
            local_path=Path(self._temporary.name) / "missing-imu.raw",
        )
        self._assert_blocks_before_factories(replace(
            self.inputs,
            records=(replace(record, imu_assets=(missing,)),),
        ))

    def test_each_missing_archive_is_rejected_before_finalization(self):
        for index, asset in enumerate(self.inputs.archive_assets):
            missing = replace(
                asset,
                local_path=(
                    Path(self._temporary.name) / f"missing-{asset.role}.zip"
                ),
            )
            archive_assets = list(self.inputs.archive_assets)
            archive_assets[index] = missing
            with self.subTest(asset.role):
                self._assert_blocks_before_factories(replace(
                    self.inputs,
                    archive_assets=tuple(archive_assets),
                ))

    def test_each_archive_hash_mismatch_is_rejected_before_finalization(self):
        for index, asset in enumerate(self.inputs.archive_assets):
            substituted = replace(asset, sha256="0" * 64)
            archive_assets = list(self.inputs.archive_assets)
            archive_assets[index] = substituted
            with self.subTest(asset.role):
                self._assert_blocks_before_factories(replace(
                    self.inputs,
                    archive_assets=tuple(archive_assets),
                ))

    def test_archive_omission_extra_and_substitution_fail_preflight(self):
        dataset_archive, video_archive = self.inputs.archive_assets
        extra = replace(
            video_archive,
            original_path="archives/another-video.zip",
        )
        substituted = replace(dataset_archive, role="SOURCE_ARCHIVE")
        cases = {
            "omitted": (dataset_archive,),
            "extra": (dataset_archive, video_archive, extra),
            "substituted role": (substituted, video_archive),
        }
        for label, archive_assets in cases.items():
            with self.subTest(label):
                self._assert_blocks_before_factories(replace(
                    self.inputs,
                    archive_assets=archive_assets,
                ))

    def test_archives_require_distinct_paths_and_hashes(self):
        dataset_archive, video_archive = self.inputs.archive_assets
        cases = {
            "duplicate path": replace(
                video_archive,
                original_path=dataset_archive.original_path,
            ),
            "duplicate hash": replace(
                video_archive,
                sha256=dataset_archive.sha256,
                local_path=dataset_archive.local_path,
            ),
        }
        for label, replacement in cases.items():
            with self.subTest(label):
                self._assert_blocks_before_factories(replace(
                    self.inputs,
                    archive_assets=(dataset_archive, replacement),
                ))

    def test_archive_escape_alias_and_hash_substitution_fail_preflight(self):
        dataset_archive, video_archive = self.inputs.archive_assets
        record = self.inputs.records[0]
        cases = {
            "path escape": replace(
                dataset_archive,
                original_path="../dataset.zip",
            ),
            "recording asset alias": replace(
                dataset_archive,
                original_path=record.video.original_path,
            ),
            "hash substitution": replace(
                dataset_archive,
                sha256="0" * 64,
            ),
        }
        for label, replacement in cases.items():
            with self.subTest(label):
                self._assert_blocks_before_factories(replace(
                    self.inputs,
                    archive_assets=(replacement, video_archive),
                ))

    def test_missing_inventory_manifest_is_rejected_before_finalization(self):
        self._assert_blocks_before_factories(replace(
            self.inputs,
            inventory_manifest_path=(
                Path(self._temporary.name) / "missing-inventory.json"
            ),
        ))

    def test_missing_license_record_is_rejected_before_finalization(self):
        self._assert_blocks_before_factories(replace(
            self.inputs,
            license_record_path=Path(self._temporary.name) / "missing-license.txt",
        ))

    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "mkfifo"),
        "FIFO semantics require POSIX",
    )
    def test_fifo_asset_is_rejected_without_blocking_preflight(self):
        fifo = Path(self._temporary.name) / "substituted-license.fifo"
        os.mkfifo(fifo)
        self._assert_blocks_before_factories(replace(
            self.inputs,
            license_record_path=fifo,
        ))

    def test_substituted_source_kind_is_rejected_before_finalization(self):
        record = self.inputs.records[0]
        with self.assertRaises(RecordingFinalizationError):
            replace(record.provenance, source_kind="SYNTHETIC_FIXTURE")

    def test_recording_inventory_mismatch_is_rejected_before_finalization(self):
        self._assert_blocks_before_factories(replace(
            self.inputs,
            expected_recording_ids=("fixture-public", "missing-recording"),
        ))

    def test_inventory_imu_omission_is_rejected_before_finalization(self):
        payload = self._inventory_payload()
        payload["records"][0]["imu_assets"] = []
        self._assert_blocks_before_factories(self._with_inventory(
            payload,
            name="inventory-without-imu.json",
        ))

    def test_inventory_imu_extra_is_rejected_before_finalization(self):
        payload = self._inventory_payload()
        payload["records"][0]["imu_assets"].append({
            "original_path": "imu/S01/unclaimed-extra.raw",
            "role": "IMU",
            "sha256": "7" * 64,
        })
        self._assert_blocks_before_factories(self._with_inventory(
            payload,
            name="inventory-with-extra-imu.json",
        ))

    def test_inventory_imu_hash_mismatch_is_rejected_before_finalization(self):
        payload = self._inventory_payload()
        payload["records"][0]["imu_assets"][0]["sha256"] = "7" * 64
        self._assert_blocks_before_factories(self._with_inventory(
            payload,
            name="inventory-with-wrong-imu-hash.json",
        ))

    def test_inventory_path_escape_is_rejected_before_finalization(self):
        payload = self._inventory_payload()
        payload["records"][0]["imu_assets"][0]["original_path"] = "../escape.raw"
        self._assert_blocks_before_factories(self._with_inventory(
            payload,
            name="inventory-with-path-escape.json",
        ))

    def test_any_frozen_asset_hash_mismatch_blocks_all_finalization(self):
        record = self.inputs.records[0]
        bad_video = replace(record.video, sha256="0" * 64)
        self._assert_blocks_before_factories(replace(
            self.inputs,
            records=(replace(record, video=bad_video),),
        ))

    def test_factories_must_return_fresh_instances_for_each_record(self):
        first = self.inputs.records[0]
        root = Path(self._temporary.name)
        second_video_path = root / "second-video.mp4"
        second_video_path.write_bytes(first.video.local_path.read_bytes())
        second_imu_path = root / "second-imu.raw"
        second_imu_path.write_bytes(b"second paired IMU fixture\n")
        second_video = FrozenSourceAsset(
            original_path="videosoriginal/S02/fixture-public-2.mp4",
            local_path=second_video_path,
            sha256=sha256_file(second_video_path),
            role="ORIGINAL_VIDEO",
        )
        second_imu = FrozenSourceAsset(
            original_path="imu/S02/fixture-public-2.raw",
            local_path=second_imu_path,
            sha256=sha256_file(second_imu_path),
            role="IMU",
        )
        second = VidimuSnapshotRecord(
            provenance=replace(
                first.provenance,
                recording_id="fixture-public-2",
                source_original_path=second_video.original_path,
                source_object_id="vidimu:fixture-public-2",
            ),
            video=second_video,
            imu_assets=(second_imu,),
        )
        payload = self._inventory_payload()
        payload["records"].append({
            "recording_id": second.provenance.recording_id,
            "video": {
                "original_path": second.video.original_path,
                "role": second.video.role,
                "sha256": second.video.sha256,
            },
            "imu_assets": [{
                "original_path": second_imu.original_path,
                "role": second_imu.role,
                "sha256": second_imu.sha256,
            }],
        })
        inputs = self._with_inventory(
            payload,
            name="two-record-factory-inventory.json",
            records=(first, second),
        )
        shared_decoder = PTSDecoder(DecodeConfig())
        shared_estimator = DeterministicPoseEstimator()
        cases = {
            "decoder": (
                lambda: shared_decoder,
                lambda _record: DeterministicPoseEstimator(),
            ),
            "estimator": (
                lambda: PTSDecoder(DecodeConfig()),
                lambda _record: shared_estimator,
            ),
        }
        for label, factories in cases.items():
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "must-not-publish"
                with self.subTest(label), self.assertRaisesRegex(
                    RecordingFinalizationError, "fresh",
                ):
                    finalize_vidimu_snapshot(
                        inputs,
                        output,
                        decoder_factory=factories[0],
                        estimator_factory=factories[1],
                    )
                self.assertEqual(len(list(output.rglob("_SUCCESS"))), 1)

    def test_changed_archive_topology_cannot_reuse_finalization_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "finalized"
            finalize_vidimu_snapshot(
                self.inputs,
                output,
                decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                estimator_factory=lambda _record: DeterministicPoseEstimator(),
            )
            alternate_assets: list[FrozenSourceAsset] = []
            for asset in self.inputs.archive_assets:
                local_path = root / f"alternate-{asset.role}.zip"
                local_path.write_bytes(
                    f"alternate {asset.role}\n".encode("ascii"))
                alternate_assets.append(FrozenSourceAsset(
                    original_path=f"archives/alternate-{asset.role}.zip",
                    local_path=local_path,
                    sha256=sha256_file(local_path),
                    role=asset.role,
                ))
            changed = replace(
                self.inputs,
                archive_assets=tuple(alternate_assets),
            )
            estimator = DeterministicPoseEstimator()
            with self.assertRaisesRegex(
                RecordingFinalizationError, "different provenance",
            ):
                finalize_vidimu_snapshot(
                    changed,
                    output,
                    decoder_factory=lambda: PTSDecoder(DecodeConfig()),
                    estimator_factory=lambda _record: estimator,
                )
            self.assertEqual(estimator.inference_frame_ids, [])
            self.assertEqual(len(list(output.rglob("_SUCCESS"))), 1)


if __name__ == "__main__":
    unittest.main()
