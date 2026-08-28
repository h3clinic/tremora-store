from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from benchmarks import audit_vidimu_v05_sync_authority as audit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "detector/benchmarks/audit_vidimu_v05_sync_authority.py"
CHECKED_REPORT = (
    REPO_ROOT / "detector/benchmarks/vidimu_v05_sync_authority_audit.json"
)
SNAPSHOT = (
    REPO_ROOT
    / "data/snapshots/vidimu"
    / "a6e2194aee5478718e6f92cf9306214e361b08bb61363998f1e6e59e7378f1eb"
)
V04_REPORT = REPO_ROOT / "detector/benchmarks/vidimu_v04_gate_b_release_audit.json"
V2_REPORT = REPO_ROOT / "detector/benchmarks/vidimu_v2_release_audit.json"
SOURCE_PARSER = (
    REPO_ROOT
    / "detector/motionbloom/tremora_store/adapters/vidimu_source.py"
)
EXPECTED_CHECKED_REPORT_SHA256 = (
    "3d4492f984ddffaed579da2e107aaf9f7d1e9cdae1ddc83629f8708d8e75bdec"
)


def _row(kind: str, cut: int) -> dict[str, str]:
    return {
        "CutFrames": str(cut),
        "File": f"S40_A01_T01.{kind}",
        "Type": kind,
    }


def _raw_fixture() -> bytes:
    labels = ("qsHIPS", "qsRUL", "qsRLL", "qsLUL", "qsLLL")
    lines = ["QUAT,w,x,y,z,timestamp"]
    lines.extend(f"{label},1,0,0,0,0.0" for label in labels)
    lines.extend(f"{label},1,0,0,0,1.000" for label in labels)
    lines.extend(f"{label},1,0,0,0,1.020" for label in labels)
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


class CanonicalAndRawUnitTests(unittest.TestCase):
    def test_encoding_is_sorted_utf8_timestamp_free_and_rejects_nan(self) -> None:
        self.assertEqual(
            audit.canonical_json_bytes({"z": "tremor-é", "a": 1}),
            b'{\n  "a": 1,\n  "z": "tremor-\xc3\xa9"\n}\n',
        )
        with self.assertRaises(ValueError):
            audit.canonical_json_bytes({"value": float("nan")})

    def test_independent_raw_fixture_preserves_rows_and_held_payloads(self) -> None:
        result = audit._audit_raw_payload(
            _raw_fixture(), recording="S40_A01_T01"
        )

        self.assertEqual(result["row_count_including_npose"], 15)
        self.assertEqual(result["npose_timestamp_class"], "EXACT_NUMERIC_ZERO")
        streams = result["stream_statistics"]
        self.assertEqual(len(streams), 5)
        for stream in streams:
            self.assertEqual(stream["observation_count"], 2)
            self.assertEqual(stream["held_payload_observation_count"], 1)
            self.assertEqual(stream["duplicate_timestamp_count"], 0)
            self.assertEqual(stream["timestamp_reversal_count"], 0)

    def test_raw_cycle_timestamp_and_schema_defects_fail_closed(self) -> None:
        valid = _raw_fixture()
        cases = (
            valid.replace(
                b"qsRUL,1,0,0,0,1.000",
                b"qsRLL,1,0,0,0,1.000",
                1,
            ),
            valid.replace(b"1.020", b"not-a-time", 1),
            valid.replace(b",1.020", b"", 1),
        )
        for broken in cases:
            with self.subTest(broken=broken[-80:]), self.assertRaises(
                audit.SyncAuthorityAuditError
            ):
                audit._audit_raw_payload(broken, recording="S40_A01_T01")

    def test_eight_row_raw_override_only_proves_derivative_unusability(self) -> None:
        base = _raw_fixture()
        lines = base.splitlines(keepends=True)
        synchronized = b"".join([lines[0], *lines[9:]])

        removed = audit._prefix_cut(
            base,
            synchronized,
            retained_prefix_lines=1,
        )

        self.assertEqual(removed, 8)
        self.assertTrue(synchronized.splitlines()[1].startswith(b"qsLUL,"))

    def test_inventory_requires_exact_modality_bijection(self) -> None:
        references = [
            {"modality": modality, "recording_id": recording}
            for modality in ("VISUAL", "BODYTRACK_POSE", "INERTIAL_QUATERNION")
            for recording in ("record-a", "record-b")
        ]
        with mock.patch.object(audit, "EXPECTED_RECORDINGS", 2):
            result = audit._partition_inventory_references(references)
            self.assertEqual(
                {key: len(value) for key, value in result.items()},
                {
                    "BODYTRACK_POSE": 2,
                    "INERTIAL_QUATERNION": 2,
                    "VISUAL": 2,
                },
            )
            corrupted = [dict(value) for value in references]
            corrupted[-1]["recording_id"] = "record-c"
            with self.assertRaisesRegex(
                audit.SyncAuthorityAuditError, "not bijective"
            ):
                audit._partition_inventory_references(corrupted)


class SynchronizationAuthorityUnitTests(unittest.TestCase):
    def test_exact_sync_paths_reject_missing_extra_and_unmatched_entries(self) -> None:
        info = {
            "S40_A01_T01": {
                "csv": _row("csv", 2),
                "mp4": _row("mp4", 2),
            }
        }
        directories = {
            "dataset/videoandimusync/",
            "dataset/videoandimusync/S40/",
        }
        expected_file = "dataset/videoandimusync/S40/S40_A01_T01.csv"
        bad_file_sets = (
            set(),
            {"dataset/videoandimusync/S40/unexpected.txt"},
            {"dataset/videoandimusync/S40/S40_A01_T02.csv"},
            {expected_file, "dataset/videoandimusync/S40/unexpected.mp4"},
        )
        for files in bad_file_sets:
            with self.subTest(files=files), self.assertRaisesRegex(
                audit.SyncAuthorityAuditError, "exact source instruction set"
            ):
                audit._validate_sync_subtree_paths(directories, files, info)
        with self.assertRaisesRegex(
            audit.SyncAuthorityAuditError, "directory topology"
        ):
            audit._validate_sync_subtree_paths(
                {*directories, "dataset/videoandimusync/S41/"},
                {expected_file},
                info,
            )

    def test_dual_direction_applied_overrides_are_formal_ambiguities(self) -> None:
        plots = {
            "S53_A13_T03": {"selected_direction": "VIDEO"},
            "S57_A07_T01": {"selected_direction": "VIDEO"},
        }
        info = {
            "S53_A13_T03": {
                "csv": _row("csv", 2),
                "mp4": _row("mp4", 2),
                "mot": _row("mot", 1),
                "raw": _row("raw", 1),
            },
            "S57_A07_T01": {
                "csv": _row("csv", 11),
                "mp4": _row("mp4", 11),
                "mot": _row("mot", 14),
                "raw": _row("raw", 14),
            },
        }
        overrides = {
            "S53_A13_T03": {
                "csv": {"removed_data_lines": 2},
                "mot": {"removed_data_lines": 1},
                "raw": {"removed_data_lines": 8},
            },
            "S57_A07_T01": {
                "csv": {"removed_data_lines": 11},
                "mot": {"removed_data_lines": 23},
                "raw": {"removed_data_lines": 116},
            },
        }

        result = audit._applied_direction_ambiguities(plots, info, overrides)

        self.assertEqual(
            [value["recording_id"] for value in result],
            ["S53_A13_T03", "S57_A07_T01"],
        )
        self.assertTrue(
            all(
                value["pairing_status"] == "AMBIGUOUS_SOURCE_MAPPING"
                and value["ambiguity_type"] == "DUAL_DIRECTION_APPLIED_OVERRIDE"
                for value in result
            )
        )
        self.assertEqual(result[0]["applied_video_cut_frames"], 2)
        self.assertEqual(result[0]["applied_raw_removed_rows"], 8)
        self.assertEqual(result[1]["applied_video_cut_frames"], 11)
        self.assertEqual(result[1]["applied_raw_removed_rows"], 116)

    def test_dual_direction_mapping_cannot_be_silently_selected(self) -> None:
        plots = {
            "S53_A13_T03": {"selected_direction": "VIDEO"},
            "S57_A07_T01": {"selected_direction": "VIDEO"},
        }
        info = {
            recording: {
                "csv": _row("csv", video_cut),
                "mp4": _row("mp4", video_cut),
                "mot": _row("mot", imu_cut),
                "raw": _row("raw", imu_cut),
            }
            for recording, video_cut, imu_cut in (
                ("S53_A13_T03", 2, 1),
                ("S57_A07_T01", 11, 14),
            )
        }
        incomplete = {
            "S53_A13_T03": {
                "csv": {"removed_data_lines": 2},
                "mot": {"removed_data_lines": 1},
            },
            "S57_A07_T01": {
                "csv": {"removed_data_lines": 11},
                "mot": {"removed_data_lines": 23},
                "raw": {"removed_data_lines": 116},
            },
        }
        with self.assertRaisesRegex(
            audit.SyncAuthorityAuditError, "ambiguity mismatch"
        ):
            audit._applied_direction_ambiguities(plots, info, incomplete)


class PinnedPriorEvidenceTests(unittest.TestCase):
    def test_v2_raw_evidence_and_parser_are_semantically_reconciled(self) -> None:
        evidence, records = audit._audit_v2_release(V2_REPORT, SOURCE_PARSER)

        self.assertEqual(evidence["raw_recording_count"], 208)
        self.assertEqual(
            evidence["raw_source_row_count_including_npose"], 10_184_045
        )
        self.assertEqual(
            evidence["source_parser_sha256"],
            "244685dcb6a0de1910b23d729c644b91dc807c314690ddc7400ab7d99c3699ae",
        )
        self.assertEqual(len(records), 208)

    def test_v2_semantic_mutation_fails_even_with_outer_hash_rebound(self) -> None:
        value = json.loads(V2_REPORT.read_bytes())
        value["aggregates"]["raw_source_row_count_including_npose"] = 10_184_044
        payload = audit.canonical_json_bytes(value)
        with tempfile.TemporaryDirectory() as temporary:
            mutated = Path(temporary) / "v2.json"
            mutated.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            with (
                mock.patch.object(
                    audit, "EXPECTED_V2_RELEASE_AUDIT_SHA256", digest
                ),
                self.assertRaisesRegex(
                    audit.SyncAuthorityAuditError, "RAW aggregates"
                ),
            ):
                audit._audit_v2_release(mutated, SOURCE_PARSER)

    def test_v04_semantic_nonpass_fails_even_with_outer_hash_rebound(self) -> None:
        value = json.loads(V04_REPORT.read_bytes())
        value["overall_verdict"] = "FAIL"
        payload = audit.canonical_json_bytes(value)
        with tempfile.TemporaryDirectory() as temporary:
            mutated = Path(temporary) / "v04.json"
            mutated.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            with (
                mock.patch.object(
                    audit, "EXPECTED_V04_RELEASE_AUDIT_SHA256", digest
                ),
                self.assertRaisesRegex(
                    audit.SyncAuthorityAuditError, "frozen PASS"
                ),
            ):
                audit._audit_v04_report(mutated)


class PublicationAndCliTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "secure publication requires POSIX")
    def test_publication_is_atomic_durable_and_no_replace(self) -> None:
        payload = b"complete-audit\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "audit.json"
            audit._write_exclusive(destination, payload, ())
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(list(root.glob(".audit.json.tmp-*")), [])
            with self.assertRaisesRegex(
                audit.SyncAuthorityAuditError, "already exists"
            ):
                audit._write_exclusive(destination, b"replacement\n", ())
            self.assertEqual(destination.read_bytes(), payload)

    @unittest.skipUnless(os.name == "posix", "secure publication requires POSIX")
    def test_file_fsync_failure_leaves_no_final_or_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "audit.json"
            with (
                mock.patch.object(
                    audit.os, "fsync", side_effect=OSError("injected")
                ),
                self.assertRaises(OSError),
            ):
                audit._write_exclusive(destination, b"partial", ())
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".audit.json.tmp-*")), [])

    @unittest.skipUnless(os.name == "posix", "secure publication requires POSIX")
    def test_preexisting_symlinks_are_rejected_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_bytes(b"user-data")
            destination = root / "audit.json"
            destination.symlink_to(target)
            with self.assertRaisesRegex(
                audit.SyncAuthorityAuditError, "must not be a symlink"
            ):
                audit._write_exclusive(destination, b"audit", ())
            self.assertTrue(destination.is_symlink())
            self.assertEqual(target.read_bytes(), b"user-data")

            for name, link_target in (
                ("relative.json", Path("missing-relative.json")),
                ("absolute.json", root / "outside" / "missing-absolute.json"),
            ):
                link = root / name
                link.symlink_to(link_target)
                with self.subTest(name=name), self.assertRaisesRegex(
                    audit.SyncAuthorityAuditError, "must not be a symlink"
                ):
                    audit._write_exclusive(link, b"audit", ())
                self.assertTrue(link.is_symlink())
            self.assertFalse((root / "missing-relative.json").exists())
            self.assertFalse((root / "outside").exists())

    def test_cli_distinguishes_gate_nogo_execution_error_and_argparse(self) -> None:
        arguments = [
            "--snapshot-root", "snapshot",
            "--analysis-archive", "analysis",
            "--article-pdf", "article",
            "--record-metadata", "metadata",
            "--source-parser", "parser",
            "--tools-release", "tools",
            "--v04-release-audit", "v04",
            "--v2-release-audit", "v2",
        ]
        completed = {
            "artifact_kind": audit.ARTIFACT_KIND,
            "audit_execution_status": "PASS",
            "gate_status": "NO_GO_RAW_NATIVE_CLOCK_AUTHORITY",
            "schema_version": audit.SCHEMA_VERSION,
        }
        with (
            mock.patch.object(
                audit,
                "audit_vidimu_v05_sync_authority",
                return_value=completed,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(audit.main(arguments), 3)
        with (
            mock.patch.object(
                audit,
                "audit_vidimu_v05_sync_authority",
                side_effect=audit.SyncAuthorityAuditError("injected"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(audit.main(arguments), 1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
            SystemExit
        ) as raised:
            audit.main([])
        self.assertEqual(raised.exception.code, 2)


class CheckedReportTests(unittest.TestCase):
    def test_checked_report_has_scoped_nogo_and_exact_empirical_counts(self) -> None:
        value = json.loads(CHECKED_REPORT.read_bytes())

        self.assertEqual(value["audit_execution_status"], "PASS")
        self.assertEqual(value["gate_status"], "NO_GO_RAW_NATIVE_CLOCK_AUTHORITY")
        blocker_ids = {value["blocker_id"] for value in value["blockers"]}
        self.assertIn(
            "RAW_ROW_TO_NATIVE_50HZ_SAMPLE_MAPPING_UNSUPPORTED", blocker_ids
        )
        self.assertIn("DUAL_DIRECTION_APPLIED_OVERRIDE", blocker_ids)
        self.assertNotIn("RAW_ROWS_ARE_NOT_NATIVE_50HZ_SAMPLES", blocker_ids)
        raw = value["source_evidence"]["snapshot"]["independent_raw_scan"]
        self.assertEqual(raw["raw_recording_count"], 208)
        self.assertEqual(raw["raw_source_bytes"], 545_308_276)
        self.assertEqual(raw["raw_source_row_count_including_npose"], 10_184_045)
        self.assertEqual(raw["dynamic_observation_row_count"], 10_183_005)
        self.assertEqual(raw["stream_count"], 1_040)
        self.assertEqual(
            raw["exact_consecutive_held_payload_row_count"], 8_735_242
        )
        self.assertEqual(raw["source_timestamp_token_duplicate_count"], 0)
        self.assertEqual(raw["source_timestamp_token_reversal_count"], 0)
        sync = value["source_evidence"]["dataset_sync"]
        self.assertEqual(sync["complete_sync_subtree_entry_count"], 234)
        self.assertEqual(sync["sync_directory_count"], 17)
        self.assertEqual(sync["complete_sync_subtree_file_count"], 217)
        defects = sync["raw_override_defects"]
        self.assertEqual(defects["raw_override_count"], 34)
        self.assertEqual(defects["npose_rows_removed_total"], 170)
        self.assertEqual(defects["removed_source_data_row_count"], 818)
        self.assertEqual(defects["removed_dynamic_observation_row_count"], 648)
        self.assertEqual(defects["partial_five_sensor_cycle_override_count"], 30)
        self.assertNotIn("calibration_prefix_removed_count", defects)
        self.assertEqual(
            value["reconciliation"]["source_mapping_classification_counts"],
            {"AMBIGUOUS": 2, "IMU": 32, "VIDEO": 147, "ZERO": 27},
        )
        self.assertFalse(
            value["claim_boundary"]["canonical_frame_times_emitted"]
        )
        self.assertFalse(value["claim_boundary"]["canonical_imu_times_emitted"])
        self.assertFalse(value["claim_boundary"]["v06_indexes_or_windows_emitted"])


@unittest.skipUnless(
    os.environ.get("VIDIMU_V05_REAL_INPUTS") == "1",
    "set VIDIMU_V05_REAL_INPUTS=1 for pinned external-evidence release proof",
)
class RealPinnedReleaseTests(unittest.TestCase):
    def _external(self, name: str) -> Path:
        value = os.environ.get(name)
        self.assertIsNotNone(value, f"release mode requires {name}")
        path = Path(str(value))
        self.assertTrue(path.is_file(), f"release input is missing: {name}")
        return path

    def _command(self, output: Path) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT),
            "--snapshot-root", str(SNAPSHOT),
            "--analysis-archive",
            str(self._external("VIDIMU_V05_ANALYSIS_ARCHIVE")),
            "--article-pdf", str(self._external("VIDIMU_V05_ARTICLE_PDF")),
            "--record-metadata",
            str(self._external("VIDIMU_V05_RECORD_METADATA")),
            "--source-parser", str(SOURCE_PARSER),
            "--tools-release",
            str(self._external("VIDIMU_V05_TOOLS_ARCHIVE")),
            "--v04-release-audit", str(V04_REPORT),
            "--v2-release-audit", str(V2_REPORT),
            "--output", str(output),
        ]

    def test_authoritative_documents_and_complete_tools_tree_are_bound(self) -> None:
        article = self._external("VIDIMU_V05_ARTICLE_PDF")
        metadata = self._external("VIDIMU_V05_RECORD_METADATA")
        tools = self._external("VIDIMU_V05_TOOLS_ARCHIVE")

        article_evidence = audit._audit_article(article)
        record_evidence = audit._audit_record_metadata(metadata)
        tools_evidence = audit._audit_tools_release(tools)

        self.assertEqual(article_evidence["size_bytes"], 6_614_907)
        self.assertEqual(record_evidence["record_id"], 15_075_076)
        self.assertEqual(
            tools_evidence["authoritative_document_scope"][
                "complete_release_file_count"
            ],
            15,
        )
        self.assertEqual(
            tools_evidence["authoritative_document_scope"]["readme_sha256"],
            "c8fc6a34bf55288b5c125f435029fd6de1ac9b52832b0596ed3548ac281a00ff",
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mutated_article = root / "article.pdf"
            mutated_article.write_bytes(article.read_bytes() + b"changed")
            with self.assertRaisesRegex(
                audit.SyncAuthorityAuditError, "article PDF SHA-256"
            ):
                audit._audit_article(mutated_article)

            metadata_value = json.loads(metadata.read_bytes())
            metadata_value["metadata"]["version"] = "v2.0.1"
            metadata_payload = json.dumps(metadata_value).encode("utf-8")
            mutated_metadata = root / "metadata.json"
            mutated_metadata.write_bytes(metadata_payload)
            with (
                mock.patch.object(
                    audit,
                    "EXPECTED_RECORD_METADATA_SHA256",
                    hashlib.sha256(metadata_payload).hexdigest(),
                ),
                self.assertRaisesRegex(
                    audit.SyncAuthorityAuditError, "release contract"
                ),
            ):
                audit._audit_record_metadata(mutated_metadata)

            mutated_tools = root / "tools.zip"
            with zipfile.ZipFile(tools) as source, zipfile.ZipFile(
                mutated_tools, "w"
            ) as destination:
                destination.comment = source.comment
                for item in source.infolist():
                    payload = source.read(item)
                    if item.filename.endswith("README.md"):
                        payload += b"changed"
                    destination.writestr(item, payload)
            tools_payload = mutated_tools.read_bytes()
            with (
                mock.patch.object(
                    audit,
                    "EXPECTED_TOOLS_ARCHIVE_SHA256",
                    hashlib.sha256(tools_payload).hexdigest(),
                ),
                self.assertRaisesRegex(
                    audit.SyncAuthorityAuditError, "member hashes"
                ),
            ):
                audit._audit_tools_release(mutated_tools)

    def test_real_inputs_reproduce_checked_report_in_two_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_a = root / "run-a"
            run_b = root / "run-b"
            run_a.mkdir()
            run_b.mkdir()
            output_a = run_a / "report.json"
            output_b = run_b / "report.json"
            processes = [
                subprocess.Popen(
                    self._command(output),
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for output in (output_a, output_b)
            ]
            results = [process.communicate(timeout=90) for process in processes]
            self.assertNotEqual(processes[0].pid, processes[1].pid)
            for process, (stdout, stderr), output in zip(
                processes, results, (output_a, output_b)
            ):
                self.assertEqual(process.returncode, 3, stderr.decode())
                self.assertEqual(stdout, output.read_bytes())
            self.assertNotEqual(output_a.stat().st_ino, output_b.stat().st_ino)
            self.assertEqual(output_a.read_bytes(), output_b.read_bytes())
            self.assertEqual(output_a.read_bytes(), CHECKED_REPORT.read_bytes())
            self.assertEqual(
                hashlib.sha256(output_a.read_bytes()).hexdigest(),
                EXPECTED_CHECKED_REPORT_SHA256,
            )
            self.assertEqual(
                sorted(path.name for path in run_a.iterdir()), ["report.json"]
            )
            self.assertEqual(
                sorted(path.name for path in run_b.iterdir()), ["report.json"]
            )


if __name__ == "__main__":
    unittest.main()
