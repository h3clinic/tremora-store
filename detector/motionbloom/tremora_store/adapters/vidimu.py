"""Conservative file inventory for the pinned public VIDIMU release.

Matching recording stems establish only that expected files were discovered in
one selected dataset subtree. They do not prove synchronization. The released
sensor output is documented as nominally 50 Hz quaternion orientation, not raw
accelerometer or gyroscope axes; RAW source-row cadence is not the sensor
update rate. This adapter never invents clock truth.
Official processed companions and pinned release residues are excluded from
canonical pairing and preserved in a content-hashed provenance ledger for the
selected recording. Discovery validates names and locations without hashing
the full release.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..parquet_writer import sha256_file

VIDIMU_IMU_HZ = 50
VIDIMU_VIDEO_FPS = 30
VIDIMU_RELEASE_VERSION = "2.0.0"
VIDIMU_ZENODO_RECORD_ID = 15_075_076
VIDIMU_RECORD_URL = "https://zenodo.org/records/15075076"
VIDIMU_RECORD_DOI = "10.1038/s41597-023-02554-9"
VIDIMU_CONCEPT_DOI = "10.5281/zenodo.7681316"
VIDIMU_LICENSE = "CC BY 4.0"
VIDIMU_LICENSE_SPDX = "CC-BY-4.0"
VIDIMU_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
VIDIMU_REDISTRIBUTION_STATUS = "PERMITTED_WITH_ATTRIBUTION"
VIDIMU_VALIDATION_ROLE = "INVENTORY_ONLY_UNVERIFIED_SYNC"
VIDIMU_DATASET_ARCHIVE = "dataset"
VIDIMU_DATASET_SUBTREE = "dataset/videoandimus"
VIDIMU_VIDEO_ARCHIVES: Final = frozenset({"videosmallsize", "videosfullsize"})
VIDIMU_VIDEO_SUBTREE = "videosbodytrack"
VIDIMU_INVENTORY_SCOPE = "EXTRACTED_SUBSET_NO_RELEASE_COMPLETENESS_CLAIM"
VIDIMU_ARCHIVE_DIGEST_BINDING = (
    "CALLER_SUPPLIED_DOWNLOAD_DIGEST_NOT_RECOMPUTED_FROM_EXTRACTION")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_RECORDING_RE = re.compile(
    r"^(?P<subject>S[0-9]+)_(?P<activity>A[0-9]+)_(?P<trial>T[0-9]+)$",
    flags=re.IGNORECASE,
)
_VIDEO_RECORDING_RE = re.compile(
    r"^(?P<recording>S[0-9]+_A[0-9]+_T[0-9]+)_pose$",
    flags=re.IGNORECASE,
)
_NPOSE_DATA_RE = re.compile(
    r"^(?P<recording>S[0-9]+_A[0-9]+_T[0-9]+)_Npose$",
    flags=re.IGNORECASE,
)
_NPOSE_VIDEO_RE = re.compile(
    r"^(?P<recording>S[0-9]+_A[0-9]+_T[0-9]+)_Npose_pose$",
    flags=re.IGNORECASE,
)
_NPOSE_EXCLUSION_REASON = "OFFICIAL_NPOSE_COMPANION_NOT_CANONICAL_PAIR_INPUT"
_STO_EXCLUSION_REASON = (
    "OFFICIAL_STO_PROCESSED_COMPANION_NOT_CANONICAL_PAIR_INPUT"
)
_MOT_EXCLUSION_REASON = (
    "OFFICIAL_MOT_PROCESSED_COMPANION_NOT_CANONICAL_PAIR_INPUT"
)
_IK_MOT_EXCLUSION_REASON = (
    "OFFICIAL_IK_MOT_COMPANION_NOT_CANONICAL_PAIR_INPUT"
)
_IK_ORIENTATION_EXCLUSION_REASON = (
    "OFFICIAL_IK_ORIENTATION_ERRORS_COMPANION_NOT_CANONICAL_PAIR_INPUT"
)
_FULLSIZE_OUT_EXCLUSION_REASON = (
    "OFFICIAL_FULLSIZE_MP4_OUT_COMPANION_NOT_CANONICAL_PAIR_INPUT"
)
_RELEASE_RESIDUE_REASON_PREFIX = (
    "OFFICIAL_V2_NONCANONICAL_RESIDUE_MAPPED_TO_"
)
_STO_COMPANION_RE = re.compile(
    r"^(?P<recording>S[0-9]+_A[0-9]+_T[0-9]+)\.sto$",
)
_MOT_COMPANION_RE = re.compile(
    r"^(?P<recording>S[0-9]+_A[0-9]+_T[0-9]+)\.mot$",
)
_IK_MOT_COMPANION_RE = re.compile(
    r"^ik_(?P<recording>S[0-9]+_A[0-9]+_T[0-9]+)\.mot$",
)
_IK_ORIENTATION_COMPANION_RE = re.compile(
    r"^ik_(?P<recording>S[0-9]+_A[0-9]+_T[0-9]+)"
    r"_orientationErrors\.sto$",
)
_FULLSIZE_OUT_RE = re.compile(
    r"^(?P<recording>S[0-9]+_A[0-9]+_T[0-9]+)(?:_Npose)?\.mp4\.out$",
)
_S48_MOT_RECORDINGS: Final = frozenset({
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
})
_IGNORED_RELEASE_METADATA: Final = frozenset({
    "dataset/videoandimus/.DS_Store",
})
# Exact, path-bound residues in the pinned v2.0.0 archives. These aliases are
# exclusions only; they never broaden the canonical T-trial filename grammar.
_KNOWN_DATA_RESIDUES: Final = {
    "dataset/videoandimus/S41/S41_A03_P01.csv": (
        "S41_A03_T01", "S41_A03_P01"),
    "dataset/videoandimus/S41/S41_A03_P01_Npose.csv": (
        "S41_A03_T01", "S41_A03_P01"),
    "dataset/videoandimus/S49/S49_A13_T01V2_Npose.csv": (
        "S49_A13_T01", "S49_A13_T01V2"),
}
_KNOWN_VIDEO_RESIDUES: Final = {
    "videosbodytrack/S49/S49_A13_T01V2_Npose_pose.mp4": (
        "S49_A13_T01", "S49_A13_T01V2", VIDIMU_VIDEO_ARCHIVES),
    "videosbodytrack/S49/S49_A13_T01V2_Npose.mp4.out": (
        "S49_A13_T01", "S49_A13_T01V2", frozenset({"videosfullsize"})),
    "videosbodytrack/S24/S25_A02_T01_pose.mp4": (
        "S25_A02_T01", "S25_A02_T01", VIDIMU_VIDEO_ARCHIVES),
    "videosbodytrack/S24/S25_A02_T01.mp4.out": (
        "S25_A02_T01", "S25_A02_T01", frozenset({"videosfullsize"})),
}


class VidimuAdapterError(ValueError):
    """Raised when VIDIMU inventory, identity, or provenance is invalid."""


@dataclass(frozen=True, slots=True)
class VidimuRecording:
    recording_id: str
    subject_id: str
    activity_id: str
    trial_id: str
    video_path: Path | None
    pose_path: Path | None
    quaternion_path: Path | None

    @property
    def inventory_complete(self) -> bool:
        """Whether all expected file types are present, without a sync claim."""

        return all((self.video_path, self.pose_path, self.quaternion_path))

    def source_hashes(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for label, path in (
            ("video", self.video_path),
            ("video_pose", self.pose_path),
            ("imu_quaternion", self.quaternion_path),
        ):
            if path is not None:
                result[label] = sha256_file(path)
        return result


@dataclass(frozen=True, slots=True)
class _VidimuExcludedFile:
    recording_id: str
    path: Path
    archive: str
    relative_path: str
    reason: str
    sha256: str

    def provenance_entry(self) -> dict[str, str]:
        return {
            "archive": self.archive,
            "relative_path": self.relative_path,
            "reason": self.reason,
            "sha256": self.sha256,
        }


def parse_recording_id(stem: str) -> tuple[str, str, str, str]:
    match = _RECORDING_RE.fullmatch(stem)
    if not match:
        raise VidimuAdapterError(f"not a canonical VIDIMU recording stem: {stem}")
    subject = match.group("subject").upper()
    activity = match.group("activity").upper()
    trial = match.group("trial").upper()
    return f"{subject}_{activity}_{trial}", subject, activity, trial


def _parse_video_recording_id(stem: str) -> tuple[str, str, str, str]:
    """Normalize only the one filename suffix released for body-track video."""

    match = _VIDEO_RECORDING_RE.fullmatch(stem)
    if not match:
        raise VidimuAdapterError(
            "not a canonical VIDIMU body-track video stem; expected "
            "Sxx_Axx_Txx_pose (Npose is not a canonical pair input): "
            f"{stem}")
    return parse_recording_id(match.group("recording"))


def _parse_npose_recording_id(
    stem: str, *, video: bool,
) -> tuple[str, str, str, str] | None:
    pattern = _NPOSE_VIDEO_RE if video else _NPOSE_DATA_RE
    match = pattern.fullmatch(stem)
    if match is None:
        return None
    return parse_recording_id(match.group("recording"))


def _parse_data_companion(
    name: str,
) -> tuple[str, str, str] | None:
    """Return canonical recording, subject, and pinned exclusion reason."""

    patterns = (
        (_IK_ORIENTATION_COMPANION_RE, _IK_ORIENTATION_EXCLUSION_REASON),
        (_IK_MOT_COMPANION_RE, _IK_MOT_EXCLUSION_REASON),
        (_STO_COMPANION_RE, _STO_EXCLUSION_REASON),
        (_MOT_COMPANION_RE, _MOT_EXCLUSION_REASON),
    )
    for pattern, reason in patterns:
        match = pattern.fullmatch(name)
        if match is None:
            continue
        recording_id, subject, _, _ = parse_recording_id(
            match.group("recording"))
        if reason == _MOT_EXCLUSION_REASON \
                and recording_id not in _S48_MOT_RECORDINGS:
            return None
        return recording_id, subject, reason
    return None


def _parse_fullsize_out(name: str) -> tuple[str, str, str] | None:
    match = _FULLSIZE_OUT_RE.fullmatch(name)
    if match is None:
        return None
    recording_id, subject, _, _ = parse_recording_id(match.group("recording"))
    return recording_id, subject, _FULLSIZE_OUT_EXCLUSION_REASON


class VidimuAdapter:
    """Legacy v0.1 inventory for rendered BodyTrack-video QA.

    This API's ``video_path`` is ``*_pose.mp4`` from ``videosbodytrack`` and is
    not original camera input. New ingestion must use
    :class:`VidimuCameraSourceAdapter`, whose explicit ``camera_video_path``
    selects an original-subtree media candidate from ``videosoriginal``. The
    legacy API remains frozen so existing v0.1 inventory records do not silently
    change meaning.

    A SHA-256 of the applicable license/terms snapshot is mandatory. Callers may
    provide the digest directly, ask the adapter to hash a local terms file, or
    provide both so the supplied digest is checked against the file.

    ``dataset_archive_root`` is the extraction root containing
    ``dataset/videoandimus``. ``video_archive_root`` is an independent extraction
    root containing ``videosbodytrack``. Requiring the video archive name makes
    the small/full selection an explicit part of provenance rather than an
    inference from a local directory name.
    """

    def __init__(
        self,
        dataset_archive_root: str | Path,
        *,
        video_archive_root: str | Path,
        video_archive: str,
        dataset_archive_sha256: str,
        video_archive_sha256: str,
        inventory_scope: str,
        terms_sha256: str | None = None,
        terms_path: str | Path | None = None,
    ):
        self.dataset_archive_root = Path(dataset_archive_root).resolve()
        self.video_archive_root = Path(video_archive_root).resolve()
        if not self.dataset_archive_root.is_dir():
            raise VidimuAdapterError(
                "VIDIMU dataset archive root does not exist: "
                f"{self.dataset_archive_root}")
        if not self.video_archive_root.is_dir():
            raise VidimuAdapterError(
                "VIDIMU video archive root does not exist: "
                f"{self.video_archive_root}")
        if video_archive not in VIDIMU_VIDEO_ARCHIVES:
            raise VidimuAdapterError(
                "video_archive must be exactly videosmallsize or videosfullsize")
        self.video_archive = video_archive
        if inventory_scope != VIDIMU_INVENTORY_SCOPE:
            raise VidimuAdapterError(
                "inventory_scope must explicitly acknowledge extracted-subset "
                "inventory without release-completeness evidence")
        self.inventory_scope = inventory_scope
        self.dataset_archive_sha256 = self._required_sha256(
            dataset_archive_sha256, "dataset_archive_sha256")
        self.video_archive_sha256 = self._required_sha256(
            video_archive_sha256, "video_archive_sha256")
        self.dataset_subtree = self._required_selected_subtree(
            self.dataset_archive_root,
            VIDIMU_DATASET_SUBTREE,
            label="dataset extraction",
        )
        self.video_subtree = self._required_selected_subtree(
            self.video_archive_root,
            VIDIMU_VIDEO_SUBTREE,
            label="selected video extraction",
        )
        self.terms_sha256, self.terms_hash_origin = self._resolve_terms_hash(
            terms_sha256=terms_sha256, terms_path=terms_path)
        self._excluded_files_by_recording: dict[
            str, tuple[_VidimuExcludedFile, ...]
        ] = {}
        self._discovery_complete = False

    @staticmethod
    def _required_selected_subtree(
        archive_root: Path, relative: str, *, label: str,
    ) -> Path:
        """Resolve a selected subtree without following extraction symlinks."""

        candidate = archive_root
        for component in Path(relative).parts:
            candidate = candidate / component
            if candidate.is_symlink():
                raise VidimuAdapterError(
                    f"VIDIMU {label} path components must not be symlinks")
            if not candidate.is_dir():
                raise VidimuAdapterError(
                    f"VIDIMU {label} must contain {relative}")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise VidimuAdapterError(
                f"VIDIMU {label} must contain {relative}") from exc
        if not resolved.is_relative_to(archive_root):
            raise VidimuAdapterError(
                f"VIDIMU {label} subtree escapes its selected archive")
        return resolved

    def _selection_metadata(self) -> dict[str, str]:
        return {
            "release_version": VIDIMU_RELEASE_VERSION,
            "dataset_archive": VIDIMU_DATASET_ARCHIVE,
            "dataset_archive_sha256": self.dataset_archive_sha256,
            "dataset_subtree": VIDIMU_DATASET_SUBTREE,
            "video_archive": self.video_archive,
            "video_archive_sha256": self.video_archive_sha256,
            "video_subtree": VIDIMU_VIDEO_SUBTREE,
            "inventory_scope": self.inventory_scope,
            "archive_digest_binding": VIDIMU_ARCHIVE_DIGEST_BINDING,
        }

    def _revalidate_selected_subtrees(self) -> None:
        current_dataset = self._required_selected_subtree(
            self.dataset_archive_root,
            VIDIMU_DATASET_SUBTREE,
            label="dataset extraction",
        )
        current_video = self._required_selected_subtree(
            self.video_archive_root,
            VIDIMU_VIDEO_SUBTREE,
            label="selected video extraction",
        )
        if current_dataset != self.dataset_subtree \
                or current_video != self.video_subtree:
            raise VidimuAdapterError(
                "VIDIMU selected extraction subtrees changed after construction")

    def _selection_sha256(self) -> str:
        return self._metadata_sha256(self._selection_metadata())

    @staticmethod
    def _metadata_sha256(value: object) -> str:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _required_sha256(value: object, field: str) -> str:
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise VidimuAdapterError(
                f"{field} must be a 64-character hexadecimal SHA-256")
        return value.lower()

    @staticmethod
    def _resolve_terms_hash(
        *, terms_sha256: str | None, terms_path: str | Path | None,
    ) -> tuple[str, str]:
        supplied: str | None = None
        if terms_sha256 is not None:
            if not isinstance(terms_sha256, str) or not _SHA256_RE.fullmatch(
                    terms_sha256):
                raise VidimuAdapterError(
                    "terms_sha256 must be a 64-character hexadecimal SHA-256")
            supplied = terms_sha256.lower()

        computed: str | None = None
        if terms_path is not None:
            path = Path(terms_path)
            try:
                if not path.is_file():
                    raise OSError("not a regular file")
                computed = sha256_file(path)
            except OSError as exc:
                raise VidimuAdapterError(
                    f"cannot hash VIDIMU terms file: {path}") from exc

        if supplied is None and computed is None:
            raise VidimuAdapterError(
                "VIDIMU ingestion requires terms_sha256 or terms_path")
        if supplied is not None and computed is not None and supplied != computed:
            raise VidimuAdapterError(
                "supplied terms_sha256 does not match the terms file")
        if supplied is not None and computed is not None:
            return supplied, "SUPPLIED_AND_VERIFIED_FROM_FILE"
        if supplied is not None:
            return supplied, "SUPPLIED"
        assert computed is not None
        return computed, "COMPUTED_FROM_FILE"

    def discover(self) -> tuple[VidimuRecording, ...]:
        self._revalidate_selected_subtrees()
        grouped: dict[str, dict[str, Path]] = {}
        self._discovery_complete = False
        self._excluded_files_by_recording = {}

        for path in sorted(self.dataset_subtree.rglob("*")):
            if path.is_symlink():
                raise VidimuAdapterError(
                    "VIDIMU selected data subtree must not contain symlinks")
            if not path.is_file():
                continue
            archive_relative = path.relative_to(
                self.dataset_archive_root).as_posix()
            if archive_relative in _IGNORED_RELEASE_METADATA:
                continue
            residue = _KNOWN_DATA_RESIDUES.get(archive_relative)
            if residue is not None:
                recording_id, _ = residue
                _, subject, _, _ = parse_recording_id(recording_id)
                self._validate_excluded_location(
                    path,
                    subtree=self.dataset_subtree,
                    archive_root=self.dataset_archive_root,
                    recording_id=recording_id,
                    subject=subject,
                    pinned_archive_relative_path=archive_relative,
                )
                continue
            companion = _parse_data_companion(path.name)
            if companion is not None:
                recording_id, subject, _ = companion
                self._validate_excluded_location(
                    path,
                    subtree=self.dataset_subtree,
                    archive_root=self.dataset_archive_root,
                    recording_id=recording_id,
                    subject=subject,
                )
                continue
            if path.suffix not in {".csv", ".raw"}:
                raise VidimuAdapterError(
                    "unexpected VIDIMU data filename in "
                    f"{VIDIMU_DATASET_SUBTREE}: {path.name}")
            try:
                recording_id, subject, _, _ = parse_recording_id(path.stem)
            except VidimuAdapterError as exc:
                npose_identity = (
                    _parse_npose_recording_id(path.stem, video=False)
                    if path.suffix == ".csv"
                    else None
                )
                if npose_identity is None:
                    raise VidimuAdapterError(
                        "unexpected VIDIMU data filename in "
                        f"{VIDIMU_DATASET_SUBTREE}: {path.name}") from exc
                recording_id, subject, _, _ = npose_identity
                self._validate_excluded_location(
                    path,
                    subtree=self.dataset_subtree,
                    archive_root=self.dataset_archive_root,
                    recording_id=recording_id,
                    subject=subject,
                )
                continue
            relative = path.relative_to(self.dataset_subtree)
            if len(relative.parts) != 2 or relative.parts[0] != subject:
                raise VidimuAdapterError(
                    f"{recording_id} is outside its canonical subject subtree")
            slot = grouped.setdefault(recording_id, {})
            suffix = path.suffix
            if suffix in slot and slot[suffix] != path:
                raise VidimuAdapterError(
                    f"multiple {suffix} files for {recording_id}; inventory is ambiguous")
            slot[suffix] = path

        for path in sorted(self.video_subtree.rglob("*")):
            if path.is_symlink():
                raise VidimuAdapterError(
                    "VIDIMU selected video subtree must not contain symlinks")
            if not path.is_file():
                continue
            archive_relative = path.relative_to(
                self.video_archive_root).as_posix()
            residue = _KNOWN_VIDEO_RESIDUES.get(archive_relative)
            if residue is not None and self.video_archive in residue[2]:
                recording_id, _, _ = residue
                _, subject, _, _ = parse_recording_id(recording_id)
                self._validate_excluded_location(
                    path,
                    subtree=self.video_subtree,
                    archive_root=self.video_archive_root,
                    recording_id=recording_id,
                    subject=subject,
                    pinned_archive_relative_path=archive_relative,
                )
                continue
            fullsize_out = (
                _parse_fullsize_out(path.name)
                if self.video_archive == "videosfullsize"
                else None
            )
            if fullsize_out is not None:
                recording_id, subject, _ = fullsize_out
                self._validate_excluded_location(
                    path,
                    subtree=self.video_subtree,
                    archive_root=self.video_archive_root,
                    recording_id=recording_id,
                    subject=subject,
                )
                continue
            if path.suffix != ".mp4":
                raise VidimuAdapterError(
                    "unexpected VIDIMU video filename in "
                    f"{VIDIMU_VIDEO_SUBTREE}: {path.name}")
            try:
                recording_id, subject, _, _ = _parse_video_recording_id(path.stem)
            except VidimuAdapterError as exc:
                npose_identity = _parse_npose_recording_id(path.stem, video=True)
                if npose_identity is None:
                    raise VidimuAdapterError(
                        "unexpected VIDIMU video filename in "
                        f"{VIDIMU_VIDEO_SUBTREE}: {path.name}") from exc
                recording_id, subject, _, _ = npose_identity
                self._validate_excluded_location(
                    path,
                    subtree=self.video_subtree,
                    archive_root=self.video_archive_root,
                    recording_id=recording_id,
                    subject=subject,
                )
                continue
            relative = path.relative_to(self.video_subtree)
            if len(relative.parts) != 2 or relative.parts[0] != subject:
                raise VidimuAdapterError(
                    f"{recording_id} video is outside its canonical subject subtree")
            slot = grouped.setdefault(recording_id, {})
            if ".mp4" in slot and slot[".mp4"] != path:
                raise VidimuAdapterError(
                    f"multiple .mp4 files for {recording_id}; inventory is ambiguous")
            slot[".mp4"] = path

        recordings: list[VidimuRecording] = []
        for recording_id, files in sorted(grouped.items()):
            _, subject, activity, trial = parse_recording_id(recording_id)
            recordings.append(VidimuRecording(
                recording_id=recording_id,
                subject_id=subject,
                activity_id=activity,
                trial_id=trial,
                video_path=files.get(".mp4"),
                pose_path=files.get(".csv"),
                quaternion_path=files.get(".raw"),
            ))
        self._discovery_complete = True
        return tuple(recordings)

    def _rediscover_recording(
        self, recording_id: str,
    ) -> VidimuRecording:
        """Rescan target-prefixed metadata across both selected subtrees.

        Content hashing remains scoped to the selected recording and its
        official companions.  The metadata scan is release-subtree-wide so a
        late duplicate in a wrong subject directory, nested directory, or with
        an unknown delimiter cannot become invisible after initial discovery.
        """

        self._revalidate_selected_subtrees()
        canonical, subject, activity, trial = parse_recording_id(recording_id)
        if canonical != recording_id:
            raise VidimuAdapterError("recording identity must use canonical case")
        files: dict[str, Path] = {}
        excluded: list[_VidimuExcludedFile] = []
        known_data_paths = {
            relative: alias
            for relative, (target, alias) in _KNOWN_DATA_RESIDUES.items()
            if target == recording_id
        }
        known_video_paths = {
            relative: alias
            for relative, (target, alias, archives) in (
                _KNOWN_VIDEO_RESIDUES.items())
            if target == recording_id and self.video_archive in archives
        }
        folded_prefixes = {
            recording_id.casefold(),
            f"ik_{recording_id}".casefold(),
            *(alias.casefold() for alias in known_data_paths.values()),
            *(alias.casefold() for alias in known_video_paths.values()),
        }

        def target_paths(subtree: Path):
            for path in sorted(subtree.rglob("*")):
                if path.is_symlink():
                    raise VidimuAdapterError(
                        "VIDIMU selected subtrees must not contain symlinks")
                folded_name = path.name.casefold()
                if any(
                    folded_name.startswith(prefix)
                    and (
                        not (suffix := folded_name[len(prefix):])
                        or not "0" <= suffix[0] <= "9"
                    )
                    for prefix in folded_prefixes
                ):
                    yield path

        canonical_data_paths = {
            f"{subject}/{recording_id}.csv": ".csv",
            f"{subject}/{recording_id}.raw": ".raw",
        }
        canonical_data_exclusions = {
            f"{subject}/{recording_id}_Npose.csv": _NPOSE_EXCLUSION_REASON,
            f"{subject}/{recording_id}.sto": _STO_EXCLUSION_REASON,
            f"{subject}/ik_{recording_id}.mot": _IK_MOT_EXCLUSION_REASON,
            (
                f"{subject}/ik_{recording_id}_orientationErrors.sto"
            ): _IK_ORIENTATION_EXCLUSION_REASON,
        }
        if recording_id in _S48_MOT_RECORDINGS:
            canonical_data_exclusions[
                f"{subject}/{recording_id}.mot"
            ] = _MOT_EXCLUSION_REASON
        for path in target_paths(self.dataset_subtree):
            if not path.is_file():
                raise VidimuAdapterError(
                    f"{recording_id} data inventory contains a non-regular path")
            relative = path.relative_to(self.dataset_subtree).as_posix()
            archive_relative = path.relative_to(
                self.dataset_archive_root).as_posix()
            slot = canonical_data_paths.get(relative)
            if slot is not None:
                files[slot] = path
            elif archive_relative in known_data_paths:
                excluded.append(self._excluded_file(
                    path,
                    subtree=self.dataset_subtree,
                    archive_root=self.dataset_archive_root,
                    archive=VIDIMU_DATASET_ARCHIVE,
                    recording_id=recording_id,
                    subject=subject,
                    reason=_RELEASE_RESIDUE_REASON_PREFIX + recording_id,
                    pinned_archive_relative_path=archive_relative,
                ))
            elif relative in canonical_data_exclusions:
                excluded.append(self._excluded_file(
                    path,
                    subtree=self.dataset_subtree,
                    archive_root=self.dataset_archive_root,
                    archive=VIDIMU_DATASET_ARCHIVE,
                    recording_id=recording_id,
                    subject=subject,
                    reason=canonical_data_exclusions[relative],
                ))
            else:
                raise VidimuAdapterError(
                    "unexpected VIDIMU data filename or location for selected "
                    f"recording: {relative}")

        canonical_video = f"{subject}/{recording_id}_pose.mp4"
        canonical_video_exclusions = {
            (
                f"{subject}/{recording_id}_Npose_pose.mp4"
            ): _NPOSE_EXCLUSION_REASON,
        }
        if self.video_archive == "videosfullsize":
            canonical_video_exclusions.update({
                (
                    f"{subject}/{recording_id}.mp4.out"
                ): _FULLSIZE_OUT_EXCLUSION_REASON,
                (
                    f"{subject}/{recording_id}_Npose.mp4.out"
                ): _FULLSIZE_OUT_EXCLUSION_REASON,
            })
        for path in target_paths(self.video_subtree):
            if not path.is_file():
                raise VidimuAdapterError(
                    f"{recording_id} video inventory contains a non-regular path")
            relative = path.relative_to(self.video_subtree).as_posix()
            archive_relative = path.relative_to(
                self.video_archive_root).as_posix()
            if relative == canonical_video:
                files[".mp4"] = path
            elif archive_relative in known_video_paths:
                excluded.append(self._excluded_file(
                    path,
                    subtree=self.video_subtree,
                    archive_root=self.video_archive_root,
                    archive=self.video_archive,
                    recording_id=recording_id,
                    subject=subject,
                    reason=_RELEASE_RESIDUE_REASON_PREFIX + recording_id,
                    pinned_archive_relative_path=archive_relative,
                ))
            elif relative in canonical_video_exclusions:
                excluded.append(self._excluded_file(
                    path,
                    subtree=self.video_subtree,
                    archive_root=self.video_archive_root,
                    archive=self.video_archive,
                    recording_id=recording_id,
                    subject=subject,
                    reason=canonical_video_exclusions[relative],
                ))
            else:
                raise VidimuAdapterError(
                    "unexpected VIDIMU video filename or location for selected "
                    f"recording: {relative}")

        self._excluded_files_by_recording[recording_id] = tuple(sorted(
            excluded, key=lambda item: (item.archive, item.relative_path)))
        return VidimuRecording(
            recording_id=recording_id,
            subject_id=subject,
            activity_id=activity,
            trial_id=trial,
            video_path=files.get(".mp4"),
            pose_path=files.get(".csv"),
            quaternion_path=files.get(".raw"),
        )

    @staticmethod
    def _validate_excluded_location(
        path: Path,
        *,
        subtree: Path,
        archive_root: Path,
        recording_id: str,
        subject: str,
        pinned_archive_relative_path: str | None = None,
    ) -> Path:
        if path.is_symlink():
            raise VidimuAdapterError("excluded VIDIMU file must not be a symlink")
        resolved = path.resolve()
        resolved_subtree = subtree.resolve()
        resolved_archive_root = archive_root.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(
            resolved_archive_root
        ) or not resolved.is_relative_to(resolved_subtree):
            raise VidimuAdapterError(
                "excluded VIDIMU file is not regular inside its selected archive")
        archive_relative = resolved.relative_to(
            resolved_archive_root).as_posix()
        if pinned_archive_relative_path is not None:
            if archive_relative != pinned_archive_relative_path:
                raise VidimuAdapterError(
                    f"{recording_id} release residue moved from its pinned path")
            return resolved
        relative_to_subtree = resolved.relative_to(resolved_subtree)
        if (
            len(relative_to_subtree.parts) != 2
            or relative_to_subtree.parts[0] != subject
        ):
            raise VidimuAdapterError(
                f"{recording_id} excluded file is outside its canonical "
                "subject subtree")
        return resolved

    @staticmethod
    def _excluded_file(
        path: Path,
        *,
        subtree: Path,
        archive_root: Path,
        archive: str,
        recording_id: str,
        subject: str,
        reason: str,
        pinned_archive_relative_path: str | None = None,
    ) -> _VidimuExcludedFile:
        resolved = VidimuAdapter._validate_excluded_location(
            path,
            subtree=subtree,
            archive_root=archive_root,
            recording_id=recording_id,
            subject=subject,
            pinned_archive_relative_path=pinned_archive_relative_path,
        )
        resolved_archive_root = archive_root.resolve()
        try:
            digest = sha256_file(resolved)
        except OSError as exc:
            raise VidimuAdapterError(
                f"cannot hash excluded VIDIMU file: {path}") from exc
        return _VidimuExcludedFile(
            recording_id=recording_id,
            path=resolved,
            archive=archive,
            relative_path=resolved.relative_to(resolved_archive_root).as_posix(),
            reason=reason,
            sha256=digest,
        )

    def _exclusion_ledger(self, recording_id: str) -> list[dict[str, str]]:
        if recording_id not in self._excluded_files_by_recording:
            self._rediscover_recording(recording_id)
        ledger: list[dict[str, str]] = []
        for excluded in self._excluded_files_by_recording.get(recording_id, ()):
            try:
                current_sha256 = sha256_file(excluded.path)
            except OSError as exc:
                raise VidimuAdapterError(
                    "cannot revalidate VIDIMU exclusion ledger") from exc
            if current_sha256 != excluded.sha256:
                raise VidimuAdapterError(
                    "excluded VIDIMU file changed after discovery")
            ledger.append(excluded.provenance_entry())
        return ledger

    def _validated_source_hashes(
        self, recording: VidimuRecording, *, hash_content: bool = True,
    ) -> dict[str, str]:
        if not recording.inventory_complete:
            raise VidimuAdapterError(
                f"cannot ingest incomplete VIDIMU inventory: {recording.recording_id}")
        try:
            canonical_id, subject, activity, trial = parse_recording_id(
                recording.recording_id)
        except VidimuAdapterError as exc:
            raise VidimuAdapterError("recording inventory has an invalid identity") from exc
        if (
            recording.recording_id != canonical_id
            or recording.subject_id != subject
            or recording.activity_id != activity
            or recording.trial_id != trial
        ):
            raise VidimuAdapterError("recording inventory identity fields disagree")

        paths = {
            "video": recording.video_path,
            "video_pose": recording.pose_path,
            "imu_quaternion": recording.quaternion_path,
        }
        expected_suffixes = {
            "video": ".mp4",
            "video_pose": ".csv",
            "imu_quaternion": ".raw",
        }
        expected_roots = {
            "video": self.video_subtree,
            "video_pose": self.dataset_subtree,
            "imu_quaternion": self.dataset_subtree,
        }
        resolved_paths: dict[str, Path] = {}
        for label, optional_path in paths.items():
            assert optional_path is not None
            if optional_path.is_symlink():
                raise VidimuAdapterError(f"{label} file must not be a symlink")
            path = optional_path.resolve()
            expected_root = expected_roots[label].resolve()
            if not path.is_file() or not path.is_relative_to(expected_root):
                raise VidimuAdapterError(
                    f"{label} file is not a regular file inside its selected subtree")
            if path.suffix.lower() != expected_suffixes[label]:
                raise VidimuAdapterError(f"{label} file has the wrong extension")
            expected_name = (
                f"{recording.recording_id}_pose.mp4"
                if label == "video"
                else f"{recording.recording_id}{expected_suffixes[label]}"
            )
            if path.name != expected_name:
                raise VidimuAdapterError(
                    f"{label} file does not use the canonical exact filename")
            try:
                if label == "video":
                    path_recording_id, subject, _, _ = _parse_video_recording_id(
                        path.stem)
                else:
                    path_recording_id, subject, _, _ = parse_recording_id(path.stem)
            except VidimuAdapterError as exc:
                raise VidimuAdapterError(
                    f"{label} file does not have a canonical recording stem") from exc
            if path_recording_id != recording.recording_id:
                raise VidimuAdapterError(f"{label} file belongs to another recording")
            relative = path.relative_to(expected_root)
            if len(relative.parts) != 2 or relative.parts[0] != subject:
                raise VidimuAdapterError(
                    f"{label} file is outside its canonical subject subtree")
            resolved_paths[label] = path
        if not hash_content:
            return {}
        try:
            return {
                label: sha256_file(path)
                for label, path in resolved_paths.items()
            }
        except OSError as exc:
            raise VidimuAdapterError("cannot hash VIDIMU source inventory") from exc

    def provenance(self, recording: VidimuRecording) -> dict[str, object]:
        """Emit ingestion provenance without claiming temporal alignment."""

        # Validate the caller inventory, then perform a target-prefixed metadata
        # scan across both selected subtrees. Only selected content is hashed.
        self._validated_source_hashes(recording, hash_content=False)
        current = self._rediscover_recording(recording.recording_id)
        if current != recording:
            raise VidimuAdapterError(
                "recording inventory changed or disagrees with current discovery")
        source_hashes = self._validated_source_hashes(recording)
        selection_metadata = self._selection_metadata()
        exclusion_ledger = self._exclusion_ledger(recording.recording_id)
        exclusion_ledger_sha256 = self._metadata_sha256(exclusion_ledger)
        source_files = {
            "video": {
                "archive": self.video_archive,
                "relative_path": (
                    recording.video_path.resolve().relative_to(
                        self.video_archive_root).as_posix()),
                "sha256": source_hashes["video"],
            },
            "video_pose": {
                "archive": VIDIMU_DATASET_ARCHIVE,
                "relative_path": (
                    recording.pose_path.resolve().relative_to(
                        self.dataset_archive_root).as_posix()),
                "sha256": source_hashes["video_pose"],
            },
            "imu_quaternion": {
                "archive": VIDIMU_DATASET_ARCHIVE,
                "relative_path": (
                    recording.quaternion_path.resolve().relative_to(
                        self.dataset_archive_root).as_posix()),
                "sha256": source_hashes["imu_quaternion"],
            },
        }
        source_identity = {
            "archive_selection": selection_metadata,
            "excluded_files": exclusion_ledger,
            "excluded_files_sha256": exclusion_ledger_sha256,
            "files": source_files,
        }
        return {
            "provenance_schema_version": "1.0",
            "source_kind": "PUBLIC_DATASET",
            "source_dataset": "VIDIMU",
            "source_dataset_version": VIDIMU_RELEASE_VERSION,
            "source_record_id": VIDIMU_ZENODO_RECORD_ID,
            "source_record_url": VIDIMU_RECORD_URL,
            "source_record_uri": VIDIMU_RECORD_URL,
            "source_record_doi": VIDIMU_RECORD_DOI,
            "source_concept_doi": VIDIMU_CONCEPT_DOI,
            "source_recording_id": recording.recording_id,
            "source_file_hashes": source_hashes,
            "source_files": source_files,
            "source_exclusion_ledger": exclusion_ledger,
            "source_exclusion_ledger_sha256": exclusion_ledger_sha256,
            "source_archive_selection": selection_metadata,
            "source_archive_selection_sha256": self._selection_sha256(),
            "source_identity_sha256": self._metadata_sha256(source_identity),
            "source_license": VIDIMU_LICENSE,
            "source_license_spdx": VIDIMU_LICENSE_SPDX,
            "source_license_url": VIDIMU_LICENSE_URL,
            "source_terms_sha256": self.terms_sha256,
            "source_terms_hash_origin": self.terms_hash_origin,
            "redistribution_status": VIDIMU_REDISTRIBUTION_STATUS,
            "license_id": VIDIMU_LICENSE_SPDX,
            "license_uri": VIDIMU_LICENSE_URL,
            "license_terms_sha256": self.terms_sha256,
            "source_access_status": "PUBLIC_ZENODO_RECORD",
            "source_redistribution_status": VIDIMU_REDISTRIBUTION_STATUS,
            "local_analysis_allowed": True,
            "source_redistribution_allowed": True,
            "derived_artifact_release_allowed": True,
            "derived_artifact_policy": "CC_BY_4_0_ATTRIBUTION_REQUIRED",
            "permitted_use": "RESEARCH_AND_DERIVED_BENCHMARK_ARTIFACTS_WITH_ATTRIBUTION",
            "validation_role": VIDIMU_VALIDATION_ROLE,
            "use_decision": "ALLOW_ANALYSIS_AND_RELEASE",
            "artifact_release_status": "RELEASABLE",
            "recording_inventory_complete": True,
            "release_inventory_complete": False,
            "release_inventory_scope": self.inventory_scope,
            "released_imu_payload": "FIVE_50_HZ_QUATERNION_ORIENTATION_STREAMS",
            "video_nominal_fps": VIDIMU_VIDEO_FPS,
            "imu_nominal_hz": VIDIMU_IMU_HZ,
            "raw_accelerometer_axes_available": False,
            "raw_gyroscope_axes_available": False,
            "clock_truth_status": "UNVERIFIED_FROM_FILE_INVENTORY",
            "allowed_validation_role": VIDIMU_VALIDATION_ROLE,
            "prohibited_interpretation": (
                "RAW_ACCEL_GYRO_OR_INDEPENDENT_SYNC_OR_DRIFT_GROUND_TRUTH"),
        }
