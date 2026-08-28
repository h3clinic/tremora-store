"""Offline, independent tests for the VIDIMU v2 release-audit artifact."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from benchmarks import audit_vidimu_v2_release as audit

_JOINTS = (
    "pelvis",
    "left_hip",
    "right_hip",
    "torso",
    "left_knee",
    "right_knee",
    "neck",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky_knuckle",
    "right_pinky_knuckle",
    "left_middle_tip",
    "right_middle_tip",
    "left_index_knuckle",
    "right_index_knuckle",
    "left_thumb_tip",
    "right_thumb_tip",
)
_LOWER_LAYOUT = ("qsHIPS", "qsRUL", "qsRLL", "qsLUL", "qsLLL")
_UPPER_LAYOUT = ("qsBACK", "qsRUA", "qsRLA", "qsLUA", "qsLLA")
_HAS_SECURE_DIRFD = (
    os.open in getattr(os, "supports_dir_fd", ())
    and os.stat in getattr(os, "supports_dir_fd", ())
    and os.unlink in getattr(os, "supports_dir_fd", ())
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
)


class _FakeResponse:
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]):
        self._payload = payload
        self.status = status
        self.headers = headers
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        payload = self._payload if size < 0 else self._payload[:size]
        self.bytes_read += len(payload)
        return payload

    def __enter__(self) -> "_FakeResponse":  # noqa: PYI034, UP037
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _RangeOpener:
    def __init__(
        self,
        archives: dict[str, bytes],
        *,
        status: int = 206,
        shift_content_range: int = 0,
        short_body: bool = False,
    ):
        self.archives = archives
        self.status = status
        self.shift_content_range = shift_content_range
        self.short_body = short_body
        self.requests: list[tuple[str, int, int]] = []

    def __call__(self, request: object, *, timeout: int) -> _FakeResponse:
        del timeout
        url = str(request.full_url)  # type: ignore[attr-defined]
        payload = self.archives[url]
        range_header = request.get_header("Range")  # type: ignore[attr-defined]
        match = re.fullmatch(r"bytes=([0-9]+)-([0-9]+)", str(range_header))
        if match is None:
            raise AssertionError(f"missing or malformed Range: {range_header!r}")
        start, end = (int(match.group(1)), int(match.group(2)))
        self.requests.append((url, start, end))
        body = payload[start : end + 1]
        if self.short_body:
            body = body[:-1]
        content_start = start + self.shift_content_range
        return _FakeResponse(
            body,
            status=self.status,
            headers={
                "Content-Length": str(len(body)),
                "Content-Range": (f"bytes {content_start}-{end}/{len(payload)}"),
            },
        )


def _zip_bytes(
    members: list[tuple[str, bytes, int | None]],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for path, payload, mode in members:
            info = zipfile.ZipInfo(path, date_time=(2020, 1, 2, 3, 4, 6))
            info.compress_type = compression
            info.create_system = 3
            file_mode = stat.S_IFREG | 0o644 if mode is None else mode
            info.external_attr = file_mode << 16
            archive.writestr(info, payload)
    return output.getvalue()


def _pose_header() -> tuple[str, ...]:
    normalized = tuple(
        f"{joint}_{axis}" for joint in _JOINTS for axis in ("x", "y", "z")
    )
    return (*normalized[:-3], *(f" {item}" for item in normalized[-3:]))


def _pose_text(*, fully_zero: bool, partial_zero: bool) -> str:
    if fully_zero and partial_zero:
        raise AssertionError("fixture row type must be unambiguous")
    first = [str(index + 1) for index in range(102)]
    if fully_zero:
        first = ["0"] * 102
    elif partial_zero:
        first[:3] = ["0", "0", "0"]
    second = [str(index + 201) for index in range(102)]
    return "\r\n".join(
        (",".join(_pose_header()), ",".join(first), ",".join(second), "")
    )


def _raw_text(
    recording_id: str,
    *,
    calibration_timestamp: str,
    invalid: bool,
) -> str:
    activity = int(recording_id.split("_")[1][1:])
    layout = _LOWER_LAYOUT if activity <= 4 else _UPPER_LAYOUT
    rows = ["QUAT,w,x,y,z,timestamp"]
    for sensor in layout:
        rows.append(f"{sensor},1.0,0.0,0.0,0.0,{calibration_timestamp}")
    timestamps = ("1660000000.100", "1660000000.120", "1660000000.140")
    payloads = (
        ("1.0", "0.0", "0.0", "0.0"),
        ("1.0", "0.0", "0.0", "0.0"),
        ("0.0", "1.0", "0.0", "0.0"),
    )
    for cycle, timestamp in enumerate(timestamps):
        for sensor_index, sensor in enumerate(layout):
            values = payloads[cycle]
            if invalid and cycle == 2 and sensor_index == 2:
                values = ("2.0", "0.0", "0.0", "0.0")
            rows.append(f"{sensor},{','.join(values)},{timestamp}")
    return "\r\n".join((*rows, ""))


def _published(payload: bytes, *, url: str) -> dict[str, object]:
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    return {
        "content_url": url,
        "published_checksum": {
            "algorithm": "md5",
            "source": "SYNTHETIC_GOLDEN",
            "value": md5,
        },
        "size_bytes": len(payload),
    }


class TestCentralDirectoryRangeAudit(unittest.TestCase):
    def test_single_bounded_range_selects_original_candidate_and_hashes_central(self):
        url = "https://fixtures.invalid/videosmallsize.zip"
        original_path = "videosmallsize/videosoriginal/S40/S40_A04_T01.mp4"
        qa_path = "videosmallsize/videosbodytrack/S40/S40_A04_T01_pose.mp4"
        payload = _zip_bytes(
            [
                (original_path, b"original-subtree-candidate-is-larger", None),
                (qa_path, b"qa", None),
                (
                    ("videosmallsize/videosoriginal/S40/S40_A04_T01_Npose.mp4"),
                    b"excluded-calibration-companion",
                    None,
                ),
            ]
        )
        opener = _RangeOpener({url: payload})
        result = audit.audit_remote_video_archive(
            "videosmallsize.zip",
            published=_published(payload, url=url),
            paired_recording_ids=("S40_A04_T01",),
            opener=opener,
        )

        self.assertEqual(opener.requests, [(url, 0, len(payload) - 1)])
        self.assertEqual(
            result.archive["range_evidence"]["response_byte_size"],  # type: ignore[index]
            len(payload),
        )
        self.assertEqual(
            result.archive["central_directory"]["central_directory_sha256"],  # type: ignore[index]
            "cf98ef90d0559bba13d254d96acbb2f31275a9a63c2ddad7b1b88c6992d19e45",
        )
        selected = result.selected_members["S40_A04_T01"]
        self.assertEqual(
            selected["original_subtree_candidate"]["path"],  # type: ignore[index]
            original_path,
        )
        self.assertEqual(selected["bodytrack_qa"]["path"], qa_path)  # type: ignore[index]
        self.assertFalse(
            result.archive["byte_verification"]["published_md5_recomputed"]
        )  # type: ignore[index]

    def test_large_stored_zip_uses_one_nonzero_tail_range(self):
        url = "https://fixtures.invalid/videosmallsize.zip"
        recording_id = "S40_A04_T01"
        payload = _zip_bytes(
            [
                (
                    f"videosmallsize/videosoriginal/S40/{recording_id}.mp4",
                    b"x" * (audit.REMOTE_TAIL_BYTES + 4096),
                    None,
                )
            ]
        )
        opener = _RangeOpener({url: payload})
        result = audit.audit_remote_video_archive(
            "videosmallsize.zip",
            published=_published(payload, url=url),
            paired_recording_ids=(recording_id,),
            opener=opener,
        )

        expected_start = len(payload) - audit.REMOTE_TAIL_BYTES
        self.assertGreater(expected_start, 0)
        self.assertEqual(opener.requests, [(url, expected_start, len(payload) - 1)])
        self.assertEqual(
            result.archive["range_evidence"]["request_count"],  # type: ignore[index]
            1,
        )

    def test_range_transport_fails_closed_before_accepting_full_or_shifted_body(self):
        url = "https://fixtures.invalid/videossmallsize.zip"
        payload = _zip_bytes(
            [
                (
                    "videosmallsize/videosoriginal/S40/S40_A04_T01.mp4",
                    b"rgb",
                    None,
                )
            ]
        )
        cases = (
            _RangeOpener({url: payload}, status=200),
            _RangeOpener({url: payload}, shift_content_range=1),
            _RangeOpener({url: payload}, short_body=True),
        )
        for opener in cases:
            with (
                self.subTest(opener=type(opener).__name__),
                self.assertRaises(audit.AuditError),
            ):
                audit.audit_remote_video_archive(
                    "videosmallsize.zip",
                    published=_published(payload, url=url),
                    paired_recording_ids=("S40_A04_T01",),
                    opener=opener,
                )

    def test_selected_symlink_is_rejected(self):
        url = "https://fixtures.invalid/videosfullsize.zip"
        payload = _zip_bytes(
            [
                (
                    "videosfullsize/videosoriginal/S40/S40_A04_T01.mp4",
                    b"relative-target",
                    stat.S_IFLNK | 0o777,
                )
            ]
        )
        with self.assertRaisesRegex(audit.AuditError, "unsafe canonical"):
            audit.audit_remote_video_archive(
                "videosfullsize.zip",
                published=_published(payload, url=url),
                paired_recording_ids=("S40_A04_T01",),
                opener=_RangeOpener({url: payload}),
            )

    def test_handcrafted_zip64_end_record_and_entry_extra_are_parsed(self):
        name = b"x"
        zip64_extra = struct.pack("<HHQ", 0x0001, 8, 0)
        central = (
            struct.pack(
                "<4s6H3L5H2L",
                b"PK\x01\x02",
                45,
                45,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                len(name),
                len(zip64_extra),
                0,
                0,
                0,
                (stat.S_IFREG | 0o644) << 16,
                0xFFFFFFFF,
            )
            + name
            + zip64_extra
        )
        zip64_offset = len(central)
        zip64_eocd = struct.pack(
            "<4sQ2H2L4Q",
            b"PK\x06\x06",
            44,
            45,
            45,
            0,
            0,
            1,
            1,
            len(central),
            0,
        )
        locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, zip64_offset, 1)
        eocd = struct.pack(
            "<4s4H2LH",
            b"PK\x05\x06",
            0,
            0,
            0xFFFF,
            0xFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0,
        )
        payload = central + zip64_eocd + locator + eocd
        result = audit.parse_central_directory_tail(
            payload, archive_size=len(payload), tail_start=0
        )
        self.assertTrue(result.zip64)
        self.assertEqual(result.entries[0].path, "x")
        self.assertEqual(result.entries[0].local_header_offset, 0)
        self.assertEqual(
            result.central_directory_sha256, hashlib.sha256(central).hexdigest()
        )


class TestDatasetSourceAudit(unittest.TestCase):
    def test_source_only_parser_counts_hashes_and_distinguishes_pose_zero_rows(self):
        lower_id = "S40_A04_T01"
        upper_id = "S41_A13_T02"
        lower_pose = _pose_text(fully_zero=True, partial_zero=False).encode()
        upper_pose = _pose_text(fully_zero=False, partial_zero=True).encode()
        lower_raw = _raw_text(
            lower_id, calibration_timestamp="0.0", invalid=True
        ).encode()
        upper_raw = _raw_text(
            upper_id, calibration_timestamp="1660000000.0", invalid=False
        ).encode()
        payload = _zip_bytes(
            [
                (
                    f"dataset/videoandimus/S40/{lower_id}.csv",
                    lower_pose,
                    None,
                ),
                (
                    f"dataset/videoandimus/S40/{lower_id}.raw",
                    lower_raw,
                    None,
                ),
                (
                    f"dataset/videoandimus/S41/{upper_id}.csv",
                    upper_pose,
                    None,
                ),
                (
                    f"dataset/videoandimus/S41/{upper_id}.raw",
                    upper_raw,
                    None,
                ),
                (
                    "dataset/videoandimus/S54/S54_A13_T01.csv",
                    _pose_text(fully_zero=False, partial_zero=False).encode(),
                    None,
                ),
                (
                    "dataset/videoandimus/S41/S41_A13_T02_Npose.csv",
                    b"not-a-canonical-pair-member",
                    None,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "dataset.zip"
            archive_path.write_bytes(payload)
            result = audit.audit_dataset_archive(
                archive_path,
                published=_published(
                    payload, url="https://fixtures.invalid/dataset.zip"
                ),
            )

        self.assertEqual(result.recording_ids, (lower_id, upper_id))
        inventory = result.archive["canonical_source_inventory"]
        self.assertEqual(inventory["csv_recording_count"], 3)  # type: ignore[index]
        self.assertEqual(inventory["raw_recording_count"], 2)  # type: ignore[index]
        self.assertEqual(inventory["csv_only_recording_ids"], ["S54_A13_T01"])  # type: ignore[index]
        by_id = {item["recording_id"]: item for item in result.records}
        lower_sources = by_id[lower_id]["source_files"]
        upper_sources = by_id[upper_id]["source_files"]
        self.assertEqual(
            lower_sources["bodytrack_pose_csv"]["sha256"],  # type: ignore[index]
            hashlib.sha256(lower_pose).hexdigest(),
        )
        self.assertEqual(
            upper_sources["quaternion_raw"]["sha256"],  # type: ignore[index]
            hashlib.sha256(upper_raw).hexdigest(),
        )
        raw_rows = sum(
            item["source_files"]["quaternion_raw"]["row_count_including_npose"]  # type: ignore[index]
            for item in result.records
        )
        self.assertEqual(raw_rows, 40)
        self.assertEqual(
            lower_sources["quaternion_raw"][  # type: ignore[index]
                "invalid_quaternion_observation_count"
            ],
            1,
        )
        self.assertEqual(
            lower_sources["quaternion_raw"]["npose_timestamp_class"],  # type: ignore[index]
            "EXACT_NUMERIC_ZERO",
        )
        self.assertEqual(
            upper_sources["quaternion_raw"]["npose_timestamp_class"],  # type: ignore[index]
            "NONZERO_DECIMAL_TOKEN_WITH_UNRESOLVED_CLOCK",
        )
        fully_zero = sum(
            item["source_files"]["bodytrack_pose_csv"]["fully_zero_row_count"]  # type: ignore[index]
            for item in result.records
        )
        any_zero = sum(
            item["source_files"]["bodytrack_pose_csv"][  # type: ignore[index]
                "rows_with_any_zero_triplet_count"
            ]
            for item in result.records
        )
        self.assertEqual(fully_zero, 1)
        self.assertEqual(any_zero, 2)
        held = audit._held_summary(result.records)
        self.assertEqual(held["stream_count"], 10)
        self.assertEqual(held["minimum"]["decimal_6"], "0.333333")  # type: ignore[index]
        self.assertEqual(held["median"]["decimal_6"], "0.333333")  # type: ignore[index]
        self.assertEqual(held["maximum"]["decimal_6"], "0.333333")  # type: ignore[index]

    def test_dataset_full_byte_identity_is_mandatory(self):
        payload = _zip_bytes([])
        published = _published(payload, url="https://fixtures.invalid/dataset.zip")
        published["size_bytes"] = len(payload) + 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.zip"
            path.write_bytes(payload)
            with self.assertRaisesRegex(audit.AuditError, "size/MD5"):
                audit.audit_dataset_archive(path, published=published)

    def test_dataset_symlink_is_rejected_before_open(self):
        payload = _zip_bytes([])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.zip"
            target.write_bytes(payload)
            path = root / "dataset.zip"
            try:
                path.symlink_to(target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            with self.assertRaisesRegex(audit.AuditError, "non-symlink regular"):
                audit.audit_dataset_archive(
                    path,
                    published=_published(
                        payload, url="https://fixtures.invalid/dataset.zip"
                    ),
                )

    @unittest.skipIf(os.name == "nt", "open-file replacement semantics differ")
    def test_dataset_path_replacement_during_audit_fails_closed(self):
        payload = _zip_bytes([])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "dataset.zip"
            path.write_bytes(payload)
            replacement = root / "replacement.zip"
            replacement.write_bytes(payload)
            original = audit._read_local_central

            def replace_after_hash(handle, archive_size):
                os.replace(replacement, path)
                return original(handle, archive_size)

            with (
                mock.patch.object(
                    audit,
                    "_read_local_central",
                    side_effect=replace_after_hash,
                ),
                self.assertRaisesRegex(audit.AuditError, "changed during"),
            ):
                audit.audit_dataset_archive(
                    path,
                    published=_published(
                        payload, url="https://fixtures.invalid/dataset.zip"
                    ),
                )

    def test_dataset_in_place_rewrite_during_audit_fails_closed(self):
        payload = _zip_bytes([])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "dataset.zip"
            path.write_bytes(payload)
            original = audit._read_local_central

            def rewrite_after_hash(handle, archive_size):
                prior = path.stat()
                with path.open("r+b") as writer:
                    first = writer.read(1)
                    writer.seek(0)
                    writer.write(first)
                    writer.flush()
                    os.fsync(writer.fileno())
                os.utime(
                    path,
                    ns=(prior.st_atime_ns, prior.st_mtime_ns + 1_000_000_000),
                )
                return original(handle, archive_size)

            with (
                mock.patch.object(
                    audit,
                    "_read_local_central",
                    side_effect=rewrite_after_hash,
                ),
                self.assertRaisesRegex(audit.AuditError, "changed during"),
            ):
                audit.audit_dataset_archive(
                    path,
                    published=_published(
                        payload, url="https://fixtures.invalid/dataset.zip"
                    ),
                )


class TestDeterministicArtifactWriting(unittest.TestCase):
    def test_cli_rejects_output_equal_to_dataset_before_network_or_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.zip"
            original = _zip_bytes([])
            path.write_bytes(original)
            with (
                mock.patch.object(audit, "fetch_record_metadata") as fetch,
                self.assertRaisesRegex(audit.AuditError, "must not alias"),
            ):
                audit.main(["--dataset-zip", str(path), "--output", str(path)])
            fetch.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_cli_rejects_existing_hardlink_and_symlink_output_aliases(self):
        for alias_kind in ("hardlink", "symlink"):
            with (
                self.subTest(alias_kind=alias_kind),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                dataset = root / "dataset.zip"
                original = _zip_bytes([])
                dataset.write_bytes(original)
                output = root / "audit.json"
                try:
                    if alias_kind == "hardlink":
                        os.link(dataset, output)
                    else:
                        output.symlink_to(dataset)
                except (NotImplementedError, OSError) as exc:
                    self.skipTest(f"{alias_kind} creation is unavailable: {exc}")
                with (
                    mock.patch.object(audit, "fetch_record_metadata") as fetch,
                    self.assertRaisesRegex(audit.AuditError, "must not alias"),
                ):
                    audit.main(
                        [
                            "--dataset-zip",
                            str(dataset),
                            "--output",
                            str(output),
                        ]
                    )
                fetch.assert_not_called()
                self.assertEqual(dataset.read_bytes(), original)

    def test_cli_rechecks_output_alias_after_long_audit_before_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.zip"
            original = _zip_bytes([])
            dataset.write_bytes(original)
            output = root / "audit.json"

            def alias_during_build(*_args, **_kwargs):
                try:
                    os.link(dataset, output)
                except OSError as exc:
                    self.skipTest(f"hardlink creation is unavailable: {exc}")
                return {"audit_status": "PASS", "records": []}

            with (
                mock.patch.object(audit, "fetch_record_metadata", return_value={}),
                mock.patch.object(
                    audit,
                    "build_release_audit",
                    side_effect=alias_during_build,
                ),
                self.assertRaises(audit.AuditError),
            ):
                audit.main(
                    [
                        "--dataset-zip",
                        str(dataset),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(dataset.read_bytes(), original)

    @unittest.skipUnless(
        _HAS_SECURE_DIRFD,
        "secure audit publication requires POSIX no-follow dir_fd support",
    )
    def test_cli_blocks_parent_swap_or_hardlink_after_final_alias_check(self):
        for race in ("parent-symlink", "destination-hardlink"):
            with (
                self.subTest(race=race),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                dataset_parent = root / "source"
                output_parent = root / "publish"
                dataset_parent.mkdir()
                output_parent.mkdir()
                dataset = dataset_parent / "dataset.zip"
                original = _zip_bytes([])
                dataset.write_bytes(original)
                output = output_parent / "dataset.zip"
                real_atomic_write = audit._atomic_write

                def race_after_check(
                    path,
                    payload,
                    *,
                    forbidden_identity,
                    _race=race,
                    _output_parent=output_parent,
                    _root=root,
                    _dataset_parent=dataset_parent,
                    _dataset=dataset,
                    _output=output,
                    _real_atomic_write=real_atomic_write,
                ):
                    try:
                        if _race == "parent-symlink":
                            _output_parent.rename(_root / "publish-pinned")
                            _output_parent.symlink_to(
                                _dataset_parent,
                                target_is_directory=True,
                            )
                        else:
                            os.link(_dataset, _output)
                    except (NotImplementedError, OSError) as exc:
                        self.skipTest(f"{_race} setup is unavailable: {exc}")
                    return _real_atomic_write(
                        path,
                        payload,
                        forbidden_identity=forbidden_identity,
                    )

                with (
                    mock.patch.object(
                        audit,
                        "fetch_record_metadata",
                        return_value={},
                    ),
                    mock.patch.object(
                        audit,
                        "build_release_audit",
                        return_value={"audit_status": "PASS", "records": []},
                    ),
                    mock.patch.object(
                        audit,
                        "_atomic_write",
                        side_effect=race_after_check,
                    ),
                    self.assertRaises(audit.AuditError),
                ):
                    audit.main(
                        [
                            "--dataset-zip",
                            str(dataset),
                            "--output",
                            str(output),
                        ]
                    )
                self.assertEqual(dataset.read_bytes(), original)

    def test_checked_artifact_is_canonical_and_binds_current_implementations(self):
        path = Path(audit.__file__).with_name("vidimu_v2_release_audit.json")
        payload = path.read_bytes()
        value = json.loads(payload)

        self.assertEqual(audit.canonical_json_bytes(value), payload)
        self.assertEqual(value["audit_status"], "PASS")
        self.assertEqual(value["release"]["record_id"], 15075076)
        self.assertEqual(value["release"]["version"], "v2.0.0")
        implementation = value["implementation"]
        self.assertEqual(
            implementation["audit_script_sha256"],
            audit._normalized_source_sha256(Path(audit.__file__)),
        )
        self.assertEqual(
            implementation["source_parser_sha256"],
            audit._normalized_source_sha256(Path(audit.vidimu_source.__file__)),
        )
        self.assertEqual(
            implementation["source_hash_canonicalization"],
            "STRICT_UTF8_WITH_CRLF_AND_CR_NORMALIZED_TO_LF",
        )
        aggregates = value["aggregates"]
        self.assertEqual(aggregates["canonical_csv_raw_pair_count"], 208)
        self.assertEqual(aggregates["raw_source_row_count_including_npose"], 10_184_045)
        self.assertEqual(aggregates["raw_dynamic_observation_row_count"], 10_183_005)
        self.assertEqual(aggregates["invalid_quaternion_observation_count"], 12)
        self.assertEqual(aggregates["pose_source_row_count"], 179_076)
        self.assertEqual(aggregates["npose_timestamp"]["zero_record_count"], 197)
        self.assertEqual(aggregates["npose_timestamp"]["nonzero_record_count"], 11)
        for archive in ("videosmallsize.zip", "videosfullsize.zip"):
            pairing = aggregates["video_pairing"][archive]
            self.assertEqual(pairing["source_pair_count"], 208)
            self.assertEqual(pairing["original_candidate_found_count"], 208)
            self.assertEqual(pairing["qa_found_count"], 206)

    def test_json_is_sorted_timestamp_free_and_atomically_replaceable(self):
        value = {"z": 2, "a": {"deterministic": True}}
        payload = audit.canonical_json_bytes(value)
        self.assertEqual(
            payload,
            b'{\n  "a": {\n    "deterministic": true\n  },\n  "z": 2\n}\n',
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            if not _HAS_SECURE_DIRFD:
                with self.assertRaisesRegex(audit.AuditError, "secure"):
                    audit._atomic_write(
                        path,
                        payload,
                        forbidden_identity=(-1, -1),
                    )
                return
            audit._atomic_write(path, payload, forbidden_identity=(-1, -1))
            audit._atomic_write(path, payload, forbidden_identity=(-1, -1))
            self.assertEqual(path.read_bytes(), payload)
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
            self.assertEqual(list(path.parent.glob(".tremora-audit-*.tmp")), [])

    @unittest.skipUnless(
        _HAS_SECURE_DIRFD,
        "secure audit publication requires POSIX no-follow dir_fd support",
    )
    def test_atomic_write_rejects_parent_rename_after_pinned_replace(self):
        payload = b"stable-audit\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "publish"
            parent.mkdir()
            backup = root / "publish-pinned"
            path = parent / "audit.json"
            real_replace = os.replace

            def replace_then_rename(*args, **kwargs):
                result = real_replace(*args, **kwargs)
                parent.rename(backup)
                parent.mkdir()
                return result

            with (
                mock.patch.object(
                    audit.os,
                    "replace",
                    side_effect=replace_then_rename,
                ),
                self.assertRaisesRegex(audit.AuditError, "parent changed"),
            ):
                audit._atomic_write(
                    path,
                    payload,
                    forbidden_identity=(-1, -1),
                )
            self.assertFalse(path.exists())
            self.assertEqual((backup / path.name).read_bytes(), payload)

    def test_implementation_hash_normalizes_checkout_line_endings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lf = root / "lf.py"
            crlf = root / "crlf.py"
            lf.write_bytes(b"x = 1\ny = 2\n")
            crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
            self.assertEqual(
                audit._normalized_source_sha256(lf),
                audit._normalized_source_sha256(crlf),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
