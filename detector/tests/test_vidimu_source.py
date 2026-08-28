"""Strict source-format and original-camera selection tests for VIDIMU v2."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import motionbloom.tremora_store.adapters.vidimu_source as vidimu_source_module
from motionbloom.tremora_store.adapters.vidimu import (
    VIDIMU_INVENTORY_SCOPE,
    VidimuAdapterError,
)
from motionbloom.tremora_store.adapters.vidimu_source import (
    VIDIMU_LOWER_BODY_LAYOUT,
    VIDIMU_POSE_SOURCE_HEADER,
    VIDIMU_RAW_HEADER,
    VIDIMU_UPPER_BODY_LAYOUT,
    VidimuCameraSourceAdapter,
    VidimuSourceError,
    parse_vidimu_pose_csv,
    parse_vidimu_raw,
)
from motionbloom.tremora_store.schema import QualityBits

_GOLDEN_RAW_HEADER = ("QUAT", "w", "x", "y", "z", "timestamp")
_GOLDEN_LOWER_BODY_LAYOUT = ("qsHIPS", "qsRUL", "qsRLL", "qsLUL", "qsLLL")
_GOLDEN_UPPER_BODY_LAYOUT = ("qsBACK", "qsRUA", "qsRLA", "qsLUA", "qsLLA")
_GOLDEN_POSE_HEADER_LINE = (
    "pelvis_x,pelvis_y,pelvis_z,left_hip_x,left_hip_y,left_hip_z,"
    "right_hip_x,right_hip_y,right_hip_z,torso_x,torso_y,torso_z,"
    "left_knee_x,left_knee_y,left_knee_z,right_knee_x,right_knee_y,"
    "right_knee_z,neck_x,neck_y,neck_z,left_ankle_x,left_ankle_y,"
    "left_ankle_z,right_ankle_x,right_ankle_y,right_ankle_z,left_big_toe_x,"
    "left_big_toe_y,left_big_toe_z,right_big_toe_x,right_big_toe_y,"
    "right_big_toe_z,left_small_toe_x,left_small_toe_y,left_small_toe_z,"
    "right_small_toe_x,right_small_toe_y,right_small_toe_z,left_heel_x,"
    "left_heel_y,left_heel_z,right_heel_x,right_heel_y,right_heel_z,nose_x,"
    "nose_y,nose_z,left_eye_x,left_eye_y,left_eye_z,right_eye_x,right_eye_y,"
    "right_eye_z,left_ear_x,left_ear_y,left_ear_z,right_ear_x,right_ear_y,"
    "right_ear_z,left_shoulder_x,left_shoulder_y,left_shoulder_z,"
    "right_shoulder_x,right_shoulder_y,right_shoulder_z,left_elbow_x,"
    "left_elbow_y,left_elbow_z,right_elbow_x,right_elbow_y,right_elbow_z,"
    "left_wrist_x,left_wrist_y,left_wrist_z,right_wrist_x,right_wrist_y,"
    "right_wrist_z,left_pinky_knuckle_x,left_pinky_knuckle_y,"
    "left_pinky_knuckle_z,right_pinky_knuckle_x,right_pinky_knuckle_y,"
    "right_pinky_knuckle_z,left_middle_tip_x,left_middle_tip_y,"
    "left_middle_tip_z,right_middle_tip_x,right_middle_tip_y,"
    "right_middle_tip_z,left_index_knuckle_x,left_index_knuckle_y,"
    "left_index_knuckle_z,right_index_knuckle_x,right_index_knuckle_y,"
    "right_index_knuckle_z,left_thumb_tip_x,left_thumb_tip_y,left_thumb_tip_z,"
    " right_thumb_tip_x, right_thumb_tip_y, right_thumb_tip_z"
)
_GOLDEN_POSE_HEADER = tuple(_GOLDEN_POSE_HEADER_LINE.split(","))


def _raw_text(
    recording_id: str,
    *,
    calibration_timestamp: str = "0.0",
    cycles: int = 3,
) -> str:
    activity = int(recording_id.split("_")[1][1:])
    layout = _GOLDEN_LOWER_BODY_LAYOUT if activity <= 4 else _GOLDEN_UPPER_BODY_LAYOUT
    lines = [",".join(_GOLDEN_RAW_HEADER)]
    for sensor in layout:
        lines.append(f"{sensor},1.0,0.0,0.0,0.0,{calibration_timestamp}")
    quaternions = (
        ("1.0", "0.0", "0.0", "0.0"),
        ("0.0", "1.0", "0.0", "0.0"),
        ("0.0", "1.0", "0.0", "0.0"),
    )
    for cycle in range(cycles):
        timestamp = f"1660000000.{100 + cycle * 20:03d}"
        values = quaternions[min(cycle, len(quaternions) - 1)]
        for sensor in layout:
            lines.append(f"{sensor},{','.join(values)},{timestamp}")
    return "\r\n".join(lines) + "\r\n"


def _pose_text(*, zero_first_joint: bool = False) -> str:
    first = [float(index + 1) for index in range(102)]
    if zero_first_joint:
        first[:3] = [0.0, 0.0, 0.0]
    second = [float(index + 201) for index in range(102)]
    rows = [
        _GOLDEN_POSE_HEADER_LINE,
        ",".join(str(value) for value in first),
        ",".join(str(value) for value in second),
    ]
    return "\r\n".join(rows) + "\r\n"


def _write_pair(
    dataset_root: Path,
    video_root: Path,
    *,
    recording_id: str = "S40_A01_T01",
    video_archive: str = "videosmallsize",
    include_camera: bool = True,
    include_qa_video: bool = True,
) -> dict[str, Path]:
    subject = recording_id.split("_", maxsplit=1)[0]
    data_subject = dataset_root / "dataset/videoandimus" / subject
    camera_subject = video_root / video_archive / "videosoriginal" / subject
    qa_subject = video_root / video_archive / "videosbodytrack" / subject
    data_subject.mkdir(parents=True, exist_ok=True)
    camera_subject.mkdir(parents=True, exist_ok=True)
    qa_subject.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": data_subject / f"{recording_id}.raw",
        "pose": data_subject / f"{recording_id}.csv",
        "camera": camera_subject / f"{recording_id}.mp4",
        "qa": qa_subject / f"{recording_id}_pose.mp4",
    }
    paths["raw"].write_bytes(_raw_text(recording_id).encode("utf-8"))
    paths["pose"].write_bytes(_pose_text().encode("utf-8"))
    if include_camera:
        paths["camera"].write_bytes(b"original-subtree-media-candidate")
    if include_qa_video:
        paths["qa"].write_bytes(b"rendered-bodytrack-fixture")
    return paths


def _source_adapter(
    dataset_root: Path,
    video_root: Path,
    *,
    video_archive: str = "videosmallsize",
) -> VidimuCameraSourceAdapter:
    return VidimuCameraSourceAdapter(
        dataset_root,
        video_archive_root=video_root,
        video_archive=video_archive,
        dataset_archive_sha256="a" * 64,
        video_archive_sha256="b" * 64,
        inventory_scope=VIDIMU_INVENTORY_SCOPE,
        terms_sha256="c" * 64,
    )


class TestVidimuRawSource(unittest.TestCase):
    def _write(self, root: Path, recording_id: str, text: str) -> Path:
        path = root / f"{recording_id}.raw"
        path.write_bytes(text.encode("utf-8"))
        return path

    def test_pinned_format_constants_match_independent_release_goldens(self):
        self.assertEqual(VIDIMU_RAW_HEADER, _GOLDEN_RAW_HEADER)
        self.assertEqual(VIDIMU_LOWER_BODY_LAYOUT, _GOLDEN_LOWER_BODY_LAYOUT)
        self.assertEqual(VIDIMU_UPPER_BODY_LAYOUT, _GOLDEN_UPPER_BODY_LAYOUT)
        self.assertEqual(VIDIMU_POSE_SOURCE_HEADER, _GOLDEN_POSE_HEADER)
        self.assertEqual(len(_GOLDEN_POSE_HEADER), 102)
        self.assertEqual(
            _GOLDEN_POSE_HEADER[-3:],
            (
                " right_thumb_tip_x",
                " right_thumb_tip_y",
                " right_thumb_tip_z",
            ),
        )

    def test_preserves_polls_as_unit_unknown_source_rows_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording_id = "S41_A02_T01"
            text = _raw_text(recording_id, calibration_timestamp="1660000000.0")
            path = self._write(root, recording_id, text)
            capture = parse_vidimu_raw(
                path,
                source_recording_id=recording_id,
                stored_recording_id="internal-001",
            )

        self.assertEqual(capture.source_row_count, 20)
        self.assertEqual(capture.source_rows.num_rows, 20)
        self.assertEqual(len(capture.calibration), 5)
        self.assertEqual(
            {row.source_timestamp_token for row in capture.calibration},
            {"1660000000.0"},
        )
        self.assertEqual(capture.nominal_sensor_hz, 50)
        self.assertIn("RAW_SOURCE_ROW", capture.source_observation_kind)
        self.assertIn("UNIT", capture.timestamp_interpretation)
        self.assertIn("UNDOCUMENTED", capture.timestamp_interpretation)
        self.assertEqual(capture.canonical_materialization_status, "DEFERRED")
        self.assertFalse(hasattr(capture, "imu_samples"))
        self.assertFalse(hasattr(capture, "clock_map"))
        rows = capture.source_rows.to_pylist()
        self.assertTrue(
            all(
                "canonical_time_ns" not in row and "sensor_time_native_ns" not in row
                for row in rows
            )
        )
        hips = [row for row in rows if row["stream_id"] == "vidimu-qsHIPS"]
        self.assertEqual([row["source_ordinal"] for row in hips], [0, 5, 10, 15])
        self.assertEqual([row["sample_index"] for row in hips], [0, 1, 2, 3])
        self.assertEqual(hips[0]["source_epoch_kind"], "NPOSE_CALIBRATION")
        self.assertEqual(hips[1]["source_epoch_kind"], "DYNAMIC_OBSERVATION")
        self.assertEqual(hips[1]["source_timestamp_token"], "1660000000.100")
        self.assertEqual(
            capture.source_rows.schema.metadata[
                b"tremora.canonical_materialization_status"
            ],
            b"DEFERRED",
        )
        stats = capture.stream_statistics[0]
        self.assertEqual(stats.observation_count, 3)
        self.assertEqual(stats.consecutive_distinct_payload_count, 2)
        self.assertEqual(stats.held_payload_observation_count, 1)
        semantics = capture.stream_semantics()
        self.assertEqual(len(semantics), 5)
        self.assertEqual(semantics[0]["body_location"], "LOWER_BACK_L3_L5")
        self.assertIn(
            "ROTATION_DIRECTION_UNSPECIFIED",
            semantics[0]["source_quaternion_convention"],
        )
        self.assertEqual(semantics[0]["source_acceleration_unit"], "NOT_PRESENT")

    def test_upper_body_layout_is_activity_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording_id = "S40_A05_T01"
            capture = parse_vidimu_raw(
                self._write(root, recording_id, _raw_text(recording_id)),
                source_recording_id=recording_id,
            )
        self.assertEqual(capture.sensor_layout, _GOLDEN_UPPER_BODY_LAYOUT)
        self.assertEqual(
            [item.sensor_label for item in capture.stream_statistics],
            list(_GOLDEN_UPPER_BODY_LAYOUT),
        )

    def test_non_unit_measurement_is_retained_as_invalid_null_not_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording_id = "S54_A08_T02"
            text = _raw_text(recording_id, cycles=1)
            text = text.replace(
                "qsRLA,1.0,0.0,0.0,0.0,1660000000.100",
                "qsRLA,1.169,-0.2629,0.1374,1.1418,1660000000.100",
            )
            capture = parse_vidimu_raw(
                self._write(root, recording_id, text),
                source_recording_id=recording_id,
            )

        self.assertEqual(len(capture.invalid_quaternions), 1)
        invalid = capture.invalid_quaternions[0]
        self.assertAlmostEqual(invalid.norm or 0.0, 1.6608020381731232)
        row = next(
            row
            for row in capture.source_rows.to_pylist()
            if row["stream_id"] == "vidimu-qsRLA"
            and row["source_quality_bits"] & int(QualityBits.INVALID_IMU_PAYLOAD)
        )
        self.assertIsNone(row["qw"])
        self.assertIsNone(row["qz"])
        self.assertTrue(
            row["source_quality_bits"] & int(QualityBits.INVALID_IMU_PAYLOAD)
        )

    def test_duplicate_timestamp_marks_both_rows_without_epsilon_jitter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording_id = "S40_A01_T01"
            text = _raw_text(recording_id)
            text = text.replace("1660000000.120", "1660000000.100")
            capture = parse_vidimu_raw(
                self._write(root, recording_id, text),
                source_recording_id=recording_id,
            )
        duplicate = int(QualityBits.DUPLICATE_TIMESTAMP)
        rows = capture.source_rows.to_pylist()
        for stream_id in {row["stream_id"] for row in rows}:
            stream_rows = [row for row in rows if row["stream_id"] == stream_id]
            self.assertFalse(stream_rows[0]["source_quality_bits"] & duplicate)
            self.assertTrue(stream_rows[1]["source_quality_bits"] & duplicate)
            self.assertTrue(stream_rows[2]["source_quality_bits"] & duplicate)
            self.assertFalse(stream_rows[3]["source_quality_bits"] & duplicate)

    def test_malformed_raw_fails_closed(self):
        recording_id = "S40_A01_T01"
        valid = _raw_text(recording_id)
        cases = {
            "header": valid.replace("QUAT,w,x,y,z,timestamp", "sensor,w,x,y,z,t"),
            "calibration-order": valid.replace("qsHIPS,1.0", "qsRUL,1.0", 1),
            "mixed-calibration-time": valid.replace(
                "qsRUL,1.0,0.0,0.0,0.0,0.0",
                "qsRUL,1.0,0.0,0.0,0.0,1.0",
                1,
            ),
            "partial-cycle": "\r\n".join(valid.splitlines()[:-1]) + "\r\n",
            "too-many-fractional-places": valid.replace(
                "1660000000.100", "1660000000.1000000001", 1
            ),
            "reversal": valid.replace("1660000000.140", "1659999999.999"),
        }
        for label, text in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                path = self._write(Path(temporary), recording_id, text)
                with self.assertRaises(VidimuSourceError):
                    parse_vidimu_raw(path, source_recording_id=recording_id)

    def test_filename_and_symlink_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording_id = "S40_A01_T01"
            wrong = root / "other.raw"
            wrong.write_text(_raw_text(recording_id), encoding="utf-8")
            with self.assertRaisesRegex(VidimuSourceError, "exactly"):
                parse_vidimu_raw(wrong, source_recording_id=recording_id)
            canonical = root / f"{recording_id}.raw"
            try:
                canonical.symlink_to(wrong)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            with self.assertRaisesRegex(VidimuSourceError, "symlink"):
                parse_vidimu_raw(canonical, source_recording_id=recording_id)

    def test_fifo_is_rejected_without_blocking(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording_id = "S40_A01_T01"
            path = root / f"{recording_id}.raw"
            os.mkfifo(path)
            started = time.monotonic()
            with self.assertRaisesRegex(VidimuSourceError, "not a regular file"):
                parse_vidimu_raw(path, source_recording_id=recording_id)
            self.assertLess(time.monotonic() - started, 1.0)

    def test_append_truncate_and_rewrite_during_read_fail_closed(self):
        module = "motionbloom.tremora_store.adapters.vidimu_source.os.read"
        for mutation in ("append", "truncate", "rewrite"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                recording_id = "S40_A01_T01"
                path = self._write(root, recording_id, _raw_text(recording_id))
                changed = False

                def mutating_read(
                    descriptor: int,
                    size: int,
                    *,
                    _read=os.read,
                    _path=path,
                    _mutation=mutation,
                ) -> bytes:
                    nonlocal changed
                    chunk = _read(descriptor, min(size, 32))
                    if chunk and not changed:
                        changed = True
                        payload = _path.read_bytes()
                        if _mutation == "append":
                            with _path.open("ab") as handle:
                                handle.write(b"X")
                        elif _mutation == "truncate":
                            _path.write_bytes(payload[: len(payload) // 2])
                        else:
                            prior = _path.stat()
                            rewritten = bytearray(payload)
                            rewritten[-1] = ord("X")
                            _path.write_bytes(rewritten)
                            os.utime(
                                _path,
                                ns=(
                                    prior.st_atime_ns,
                                    prior.st_mtime_ns + 1_000_000_000,
                                ),
                            )
                    return chunk

                with (
                    mock.patch(module, side_effect=mutating_read),
                    self.assertRaisesRegex(VidimuSourceError, "changed"),
                ):
                    parse_vidimu_raw(path, source_recording_id=recording_id)

    def test_empty_stored_identifier_fails_and_timestamp_token_stays_unitless(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording_id = "S40_A01_T01"
            valid = self._write(root, recording_id, _raw_text(recording_id))
            with self.assertRaisesRegex(VidimuSourceError, "non-empty"):
                parse_vidimu_raw(
                    valid,
                    source_recording_id=recording_id,
                    stored_recording_id="",
                )

            valid.write_bytes(
                _raw_text(
                    recording_id,
                    calibration_timestamp="1660000000.000000000",
                ).encode("utf-8"),
            )
            capture = parse_vidimu_raw(valid, source_recording_id=recording_id)
            self.assertEqual(
                {row.source_timestamp_token for row in capture.calibration},
                {"1660000000.000000000"},
            )
            self.assertFalse(hasattr(capture, "clock_map"))


class TestVidimuPoseSource(unittest.TestCase):
    def _write(self, root: Path, recording_id: str, text: str) -> Path:
        path = root / f"{recording_id}.csv"
        path.write_bytes(text.encode("utf-8"))
        return path

    def test_exact_source_header_is_normalized_only_into_unbound_mm_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording_id = "S40_A01_T01"
            text = _pose_text(zero_first_joint=True)
            capture = parse_vidimu_pose_csv(
                self._write(root, recording_id, text),
                source_recording_id=recording_id,
            )
        self.assertEqual(capture.row_count, 2)
        self.assertEqual(capture.rows_with_zero_triplets, 1)
        self.assertIn("UNBOUND", capture.timing_status)
        self.assertIn("MILLIMETRES", capture.coordinate_convention)
        self.assertEqual(
            capture.source_sha256,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        table = capture.source_rows
        self.assertEqual(table["source_row_ordinal"].to_pylist(), [0, 1])
        self.assertEqual(len(table["positions_mm"][0].as_py()), 102)
        first_mask = table["zero_triplet_mask"][0].as_py()
        self.assertTrue(first_mask[0])
        self.assertFalse(any(first_mask[1:]))
        self.assertEqual(table.schema.metadata[b"tremora.source_unit"], b"millimetres")
        self.assertIn(
            b"AXIS_DIRECTIONS_UNSPECIFIED",
            table.schema.metadata[b"tremora.coordinate_convention"],
        )
        expected_layout = ",".join(
            token.lstrip(" ").rsplit("_", 1)[0] for token in _GOLDEN_POSE_HEADER[::3]
        ).encode("ascii")
        self.assertEqual(
            table.schema.metadata[b"tremora.ordered_joint_layout"],
            expected_layout,
        )
        self.assertEqual(
            table.schema.metadata[b"tremora.ordered_joint_layout_sha256"],
            hashlib.sha256(expected_layout).hexdigest().encode("ascii"),
        )

    def test_pose_header_quirk_ragged_and_nonfinite_rows_fail_closed(self):
        recording_id = "S40_A01_T01"
        valid = _pose_text()
        cases = {
            "normalized-header-not-source": valid.replace(
                " right_thumb_tip_x", "right_thumb_tip_x"
            ),
            "ragged": valid.replace(",302.0\r\n", "\r\n", 1),
            "nonfinite": valid.replace("1.0,2.0", "nan,2.0", 1),
        }
        for label, text in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                path = self._write(Path(temporary), recording_id, text)
                with self.assertRaises(VidimuSourceError):
                    parse_vidimu_pose_csv(path, source_recording_id=recording_id)


@unittest.skipUnless(
    os.open in getattr(os, "supports_dir_fd", ())
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW"),
    "VIDIMU camera adapter intentionally requires POSIX no-follow openat",
)
class TestVidimuCameraSourceAdapter(unittest.TestCase):
    def test_original_subtree_candidate_is_explicit_and_render_is_qa_only(self):
        for archive in ("videosmallsize", "videosfullsize"):
            with (
                self.subTest(archive=archive),
                tempfile.TemporaryDirectory() as dataset_temporary,
                tempfile.TemporaryDirectory() as video_temporary,
            ):
                dataset_root = Path(dataset_temporary)
                video_root = Path(video_temporary)
                paths = _write_pair(dataset_root, video_root, video_archive=archive)
                adapter = _source_adapter(
                    dataset_root, video_root, video_archive=archive
                )
                recording = adapter.discover()[0]

                self.assertTrue(recording.inventory_complete)
                self.assertEqual(recording.camera_video_path, paths["camera"].resolve())
                self.assertEqual(
                    recording.bodytrack_qa_video_path, paths["qa"].resolve()
                )
                self.assertEqual(recording.camera_video_path.name, "S40_A01_T01.mp4")
                self.assertNotIn("_pose", recording.camera_video_path.name)
                self.assertEqual(adapter.parse_raw(recording).source_rows.num_rows, 20)
                self.assertEqual(adapter.parse_pose(recording).row_count, 2)

    def test_bodytrack_qa_absence_does_not_block_original_source(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            _write_pair(dataset_root, video_root, include_qa_video=False)
            recording = _source_adapter(dataset_root, video_root).discover()[0]
        self.assertTrue(recording.inventory_complete)
        self.assertIsNone(recording.bodytrack_qa_video_path)

    def test_missing_original_camera_blocks_parse_even_when_render_exists(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            _write_pair(dataset_root, video_root, include_camera=False)
            adapter = _source_adapter(dataset_root, video_root)
            recording = adapter.discover()[0]
            self.assertFalse(recording.inventory_complete)
            self.assertIsNotNone(recording.bodytrack_qa_video_path)
            with self.assertRaisesRegex(VidimuAdapterError, "incomplete"):
                adapter.parse_raw(recording)

    def test_pose_only_or_raw_only_inventory_is_not_a_paired_source(self):
        for missing in ("pose", "raw"):
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as dataset_temporary,
                tempfile.TemporaryDirectory() as video_temporary,
            ):
                dataset_root = Path(dataset_temporary)
                video_root = Path(video_temporary)
                paths = _write_pair(dataset_root, video_root)
                paths[missing].unlink()
                records = _source_adapter(dataset_root, video_root).discover()
                self.assertEqual(records, ())

    def test_pose_named_file_in_original_subtree_is_rejected_as_camera_input(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            paths = _write_pair(dataset_root, video_root)
            paths["camera"].unlink()
            leaked = paths["camera"].with_name("S40_A01_T01_pose.mp4")
            leaked.write_bytes(b"derived-overlay")
            with self.assertRaisesRegex(VidimuAdapterError, "original-video"):
                _source_adapter(dataset_root, video_root).discover()

    def test_camera_candidate_requires_exact_canonical_case_and_padding(self):
        for replacement_name in ("s40_a01_t01.mp4", "S40_A1_T01.mp4"):
            with (
                self.subTest(replacement_name=replacement_name),
                tempfile.TemporaryDirectory() as dataset_temporary,
                tempfile.TemporaryDirectory() as video_temporary,
            ):
                dataset_root = Path(dataset_temporary)
                video_root = Path(video_temporary)
                paths = _write_pair(dataset_root, video_root)
                paths["camera"].rename(paths["camera"].with_name(replacement_name))
                with self.assertRaisesRegex(
                    VidimuAdapterError,
                    "original-video",
                ):
                    _source_adapter(dataset_root, video_root).discover()

    def test_dataset_pair_requires_exact_canonical_case_and_padding(self):
        for replacement_id in ("s40_a01_t01", "S40_A1_T01"):
            with (
                self.subTest(replacement_id=replacement_id),
                tempfile.TemporaryDirectory() as dataset_temporary,
                tempfile.TemporaryDirectory() as video_temporary,
            ):
                dataset_root = Path(dataset_temporary)
                video_root = Path(video_temporary)
                paths = _write_pair(
                    dataset_root,
                    video_root,
                    include_camera=False,
                    include_qa_video=False,
                )
                paths["raw"].rename(paths["raw"].with_name(f"{replacement_id}.raw"))
                paths["pose"].rename(paths["pose"].with_name(f"{replacement_id}.csv"))
                with self.assertRaises(VidimuAdapterError):
                    _source_adapter(dataset_root, video_root).discover()

    def test_small_archive_rejects_fullsize_only_original_metadata(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            _write_pair(dataset_root, video_root)
            metadata = video_root / "videosmallsize/videosoriginal/toBodyTrack.txt"
            metadata.write_text("not official in this archive", encoding="utf-8")
            with self.assertRaisesRegex(
                VidimuAdapterError,
                "original-video filename or location",
            ):
                _source_adapter(dataset_root, video_root).discover()

    def test_cached_parse_does_not_repeat_release_wide_discovery(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            _write_pair(dataset_root, video_root)
            adapter = _source_adapter(dataset_root, video_root)
            recording = adapter.discover()[0]
            with (
                mock.patch.object(
                    adapter,
                    "_discover_original_videos",
                    wraps=adapter._discover_original_videos,
                ) as original_scan,
                mock.patch.object(
                    adapter._legacy_inventory,
                    "discover",
                    wraps=adapter._legacy_inventory.discover,
                ) as legacy_scan,
            ):
                adapter.parse_raw(recording)
                adapter.parse_pose(recording)
            original_scan.assert_not_called()
            legacy_scan.assert_not_called()

    def test_selected_subtree_and_source_ancestor_swaps_fail_closed(self):
        for target in ("camera-subtree", "dataset-subject"):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as dataset_temporary,
                tempfile.TemporaryDirectory() as video_temporary,
            ):
                dataset_root = Path(dataset_temporary)
                video_root = Path(video_temporary)
                paths = _write_pair(dataset_root, video_root)
                adapter = _source_adapter(dataset_root, video_root)
                recording = adapter.discover()[0]
                try:
                    if target == "camera-subtree":
                        subtree = video_root / "videosmallsize/videosoriginal"
                        backup = subtree.with_name("videosoriginal-pinned")
                        subtree.rename(backup)
                        outside = video_root / "outside-original"
                        (outside / "S40").mkdir(parents=True)
                        (outside / "S40/S40_A01_T01.mp4").write_bytes(b"outside")
                        subtree.symlink_to(outside, target_is_directory=True)
                    else:
                        subject = dataset_root / "dataset/videoandimus/S40"
                        backup = subject.with_name("S40-pinned")
                        subject.rename(backup)
                        subject.mkdir()
                        (subject / paths["raw"].name).write_bytes(
                            _raw_text("S40_A01_T01").encode("utf-8")
                        )
                        (subject / paths["pose"].name).write_bytes(
                            _pose_text().encode("utf-8")
                        )
                except (NotImplementedError, OSError) as exc:
                    self.skipTest(f"symlink creation is unavailable: {exc}")
                with self.assertRaisesRegex(VidimuAdapterError, "changed"):
                    adapter.parse_raw(recording)

    def test_subject_swap_between_validation_and_descriptor_open_fails_closed(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            _write_pair(dataset_root, video_root)
            adapter = _source_adapter(dataset_root, video_root)
            recording = adapter.discover()[0]
            subject = dataset_root / "dataset/videoandimus/S40"
            original_reader = vidimu_source_module._read_regular_source
            swapped = False

            def swap_then_read(*args, **kwargs):
                nonlocal swapped
                if not swapped:
                    swapped = True
                    subject.rename(subject.with_name("S40-pinned"))
                    subject.mkdir()
                    (subject / "S40_A01_T01.raw").write_bytes(
                        _raw_text("S40_A01_T01").encode("utf-8")
                    )
                return original_reader(*args, **kwargs)

            with (
                mock.patch.object(
                    vidimu_source_module,
                    "_read_regular_source",
                    side_effect=swap_then_read,
                ),
                self.assertRaisesRegex(VidimuSourceError, "ancestor changed"),
            ):
                adapter.parse_raw(recording)

    def test_camera_member_swap_immediately_before_nofollow_open_fails_closed(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            paths = _write_pair(dataset_root, video_root)
            adapter = _source_adapter(dataset_root, video_root)
            recording = adapter.discover()[0]
            camera = paths["camera"]
            backup = camera.with_name("camera-pinned.mp4")
            real_open = os.open
            swapped = False

            def swap_then_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if (
                    path == camera.name
                    and kwargs.get("dir_fd") is not None
                    and not swapped
                ):
                    swapped = True
                    camera.rename(backup)
                    try:
                        camera.symlink_to(backup.name)
                    except (NotImplementedError, OSError) as exc:
                        self.skipTest(f"symlink creation is unavailable: {exc}")
                return real_open(path, flags, *args, **kwargs)

            with (
                mock.patch(
                    "motionbloom.tremora_store.adapters.vidimu_source.os.open",
                    side_effect=swap_then_open,
                ),
                self.assertRaisesRegex(VidimuAdapterError, "changed"),
            ):
                adapter.parse_raw(recording)

    def test_subject_swap_during_pinned_read_fails_postvalidation(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            _write_pair(dataset_root, video_root)
            adapter = _source_adapter(dataset_root, video_root)
            recording = adapter.discover()[0]
            subject = dataset_root / "dataset/videoandimus/S40"
            real_read = os.read
            swapped = False

            def read_then_swap(descriptor: int, size: int) -> bytes:
                nonlocal swapped
                chunk = real_read(descriptor, size)
                if chunk and not swapped:
                    swapped = True
                    subject.rename(subject.with_name("S40-pinned"))
                return chunk

            with (
                mock.patch(
                    "motionbloom.tremora_store.adapters.vidimu_source.os.read",
                    side_effect=read_then_swap,
                ),
                self.assertRaisesRegex(VidimuAdapterError, "changed"),
            ):
                adapter.parse_raw(recording)

    def test_exact_original_residues_and_tool_manifest_do_not_broaden_pairing(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            _write_pair(dataset_root, video_root, video_archive="videosfullsize")
            original = video_root / "videosfullsize/videosoriginal"
            (original / "toBodyTrack.txt").write_text("official manifest")
            (original / "S24").mkdir(exist_ok=True)
            (original / "S24/S25_A02_T01.mp4").write_bytes(b"official residue")
            (original / "S49").mkdir(exist_ok=True)
            (original / "S49/S49_A13_T01V2_Npose.mp4").write_bytes(b"official residue")
            records = _source_adapter(
                dataset_root,
                video_root,
                video_archive="videosfullsize",
            ).discover()
        self.assertEqual([item.recording_id for item in records], ["S40_A01_T01"])

    def test_archive_wrapper_is_mandatory(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            (dataset_root / "dataset/videoandimus").mkdir(parents=True)
            (video_root / "videosoriginal").mkdir()
            (video_root / "videosbodytrack").mkdir()
            with self.assertRaisesRegex(VidimuAdapterError, "archive wrapper"):
                _source_adapter(dataset_root, video_root)

    def test_official_bodytrack_inventory_subtree_is_structurally_required(self):
        with (
            tempfile.TemporaryDirectory() as dataset_temporary,
            tempfile.TemporaryDirectory() as video_temporary,
        ):
            dataset_root = Path(dataset_temporary)
            video_root = Path(video_temporary)
            _write_pair(dataset_root, video_root)
            bodytrack = video_root / "videosmallsize/videosbodytrack"
            bodytrack.rename(bodytrack.with_name("videosbodytrack-absent"))
            with self.assertRaisesRegex(VidimuAdapterError, "videosbodytrack"):
                _source_adapter(dataset_root, video_root)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
