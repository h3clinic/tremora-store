"""Audit VIDIMU v0.5D source-derived alignment and fail closed.

All 217 published non-MP4 source trims reproduce byte-for-byte.  However, the
requested materialization would falsely treat high-rate RAW polling groups as
source-authorized 50 Hz ticks, and the published tool never trims MP4 video.
The audit therefore passes while the source-derived materialization gate is an
honest ``NO_GO``.  No Parquet table or success marker is written.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import re
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from motionbloom.tremora_store.v05d.authority import (
    ALIGNMENT_CONTRACT_VERSION,
    ALIGNMENT_METHOD,
    AMBIGUOUS_RECORDING_IDS,
    RAW_NATIVE_CLOCK_GATE,
    SOURCE_DERIVED_ALIGNMENT_GATE_NO_GO,
    AlignmentAuthority,
)
from motionbloom.tremora_store.v05d.source_transform import (
    SourceTransformError,
    audit_source_transform_evidence,
    canonical_json_bytes,
)

ARTIFACT_KIND = "TREMORA_VIDIMU_V05D_DERIVED_ALIGNMENT_RELEASE_AUDIT"
SCHEMA_VERSION = "0.5d.0"
IMPLEMENTATION_VERSION = "vidimu-v05d-derived-alignment-audit-1.0.0"
AUDIT_PASS = "PASS"
AUDIT_ERROR = "ERROR"
NO_GO_EXIT_CODE = 3
GATE_REASON = "NO_GO_V05D_CONTRACT_SEMANTICS"
SUCCESS_MARKER = "_STO_DERIVED_ALIGNMENT_SUCCESS"
GENERIC_SUCCESS_MARKER = "_SUCCESS"

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._=-]{0,127}\Z")
_WITHHELD_ARTIFACTS = {
    "canonical_clock_tables": 0,
    "clock_segments": 0,
    "derived_rate_contract_parquet": 0,
    "generic_success_markers": 0,
    "imu_tick_groups_parquet": 0,
    "source_trim_overlays_parquet": 0,
    "sto_alignment_contracts_parquet": 0,
    "sto_alignment_validation_parquet": 0,
    "sto_success_markers": 0,
}


class DerivedAlignmentAuditError(ValueError):
    """Raised when the v0.5D release audit cannot fail closed safely."""


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_script_hash(path: Path) -> str:
    if path.is_symlink():
        raise DerivedAlignmentAuditError("audit implementation must not be a symlink")
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise DerivedAlignmentAuditError("cannot bind audit implementation") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or len(payload) != before.st_size
        or identity_before != identity_after
    ):
        raise DerivedAlignmentAuditError("audit implementation changed while read")
    return _hash_bytes(payload)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DerivedAlignmentAuditError(f"{field} is not a mapping")
    return value


def _assert_exact_source_evidence(evidence: Mapping[str, object]) -> None:
    """Validate the complete gate basis rather than trusting aggregate labels."""

    instructions = _mapping(evidence.get("source_instructions"), "instructions")
    trim = _mapping(evidence.get("source_trim_reproduction"), "trim")
    raw = _mapping(
        evidence.get("raw_poll_to_sto_mot_reconciliation"), "raw reconciliation"
    )
    mappings = _mapping(
        evidence.get("source_alignment_decisions"), "alignment decisions"
    )
    pins = _mapping(evidence.get("frozen_inputs"), "frozen inputs")
    exact = (
        instructions.get("info_row_count") == 366
        and instructions.get("non_mp4_instruction_count") == 217
        and instructions.get("non_mp4_instruction_recording_count") == 181
        and trim.get("overrides_expected") == 217
        and trim.get("overrides_bound") == 217
        and trim.get("overrides_reproduced") == 217
        and trim.get("overrides_unreproduced") == 0
        and trim.get("published_derivative_directory_count") == 17
        and trim.get("all_generated_derivatives_byte_identical") is True
        and trim.get("discrepancies") == []
        and raw.get("record_count") == 208
        and raw.get("raw_dynamic_poll_groups") == 2_036_601
        and raw.get("sto_dynamic_ordinal_rows") == 299_711
        and raw.get("mot_dynamic_ordinal_rows") == 299_711
        and raw.get("raw_groups_are_nominal_50hz_ticks") is False
        and raw.get("raw_group_timing_authority") == "NONE_RAW_POLL_GROUP_ONLY"
        and raw.get("aggregate_raw_group_to_sto_dynamic_ratio") == "6.795216"
        and mappings.get("ambiguous_source_pairs") == 2
        and mappings.get("ambiguous_pair_ids") == sorted(AMBIGUOUS_RECORDING_IDS)
        and mappings.get("classification_counts")
        == {"AMBIGUOUS": 2, "IMU": 32, "VIDEO": 147, "ZERO": 27}
        and mappings.get("eligible_source_pairs_upper_bound") == 206
        and pins.get("v05_authority_report_sha256")
        == "3d4492f984ddffaed579da2e107aaf9f7d1e9cdae1ddc83629f8708d8e75bdec"
        and pins.get("v05_authority_script_sha256")
        == "abbd59097a16f767729521b4968ac997a55506e467621b997b1894d45334d65a"
        and pins.get("v05d_authority_module_sha256")
        == "7b0587caad2067e4c3b1b6f42dd75f5ee8c40774c4ec12dd2d7b632e1f5981bd"
        and pins.get("v05d_schema_module_sha256")
        == "2dd1bf8f88ad2716b64f6c86d05175bfb5f9fa5bdadcf90e62bb383a28d370d3"
        and pins.get("v05d_source_transform_module_sha256")
        == "4c6a61e28f5862392870cdfe4ea681fc7c4bafe81990f00ade2e84b7f0aee38c"
    )
    defects = _mapping(trim.get("raw_trim_defects"), "RAW trim defects")
    exact = exact and defects == {
        "npose_rows_removed_total": 170,
        "partial_five_sensor_cycle_override_count": 30,
        "raw_override_count": 34,
        "raw_overrides_removing_all_npose_rows_count": 34,
        "removed_dynamic_observation_row_count": 648,
        "removed_source_data_row_count": 818,
    }
    if not exact:
        raise DerivedAlignmentAuditError(
            "source evidence does not satisfy the frozen v0.5D audit basis"
        )


def _ambiguous_pair_contracts() -> list[dict[str, object]]:
    return [
        {
            "alignment_authority": (
                AlignmentAuthority.AMBIGUOUS_SOURCE_ALIGNMENT.value
            ),
            "chosen_mapping": None,
            "eligibility_override_applied": False,
            "eligibility_status": "EXCLUDED",
            "exclusion_reason": "DUAL_DIRECTION_APPLIED_OVERRIDE",
            "range_index_created": False,
            "recording_id": recording,
            "window_created": False,
        }
        for recording in sorted(AMBIGUOUS_RECORDING_IDS)
    ]


def sto_success_marker_allowed(result: Mapping[str, object]) -> bool:
    """Withhold the marker for the frozen v0.5D contract unconditionally.

    The audited source evidence proves that this contract cannot reconcile RAW
    polling groups, STO/MOT ordinals, and decoded video frames without adding
    unsupported semantics.  A future marker requires a separately versioned
    materializer and a reviewed byte-level artifact manifest; relabeling this
    report can never authorize one.
    """

    del result
    return False


def audit_vidimu_v05d_derived_alignment(
    snapshot_root: str | Path,
    analysis_archive: str | Path,
    tools_archive: str | Path,
    v05_authority_script: str | Path,
    v05_authority_report: str | Path,
) -> dict[str, object]:
    """Return a deterministic audit PASS and materialization NO-GO."""

    script = Path(__file__).resolve(strict=True)
    script_hash = _read_script_hash(script)
    try:
        source_evidence = audit_source_transform_evidence(
            snapshot_root,
            analysis_archive,
            tools_archive,
            v05_authority_script,
            v05_authority_report,
        )
    except SourceTransformError as exc:
        raise DerivedAlignmentAuditError(str(exc)) from exc
    _assert_exact_source_evidence(source_evidence)

    evidence_hash = _hash_bytes(canonical_json_bytes(source_evidence))
    withheld_ledger = {
        "artifact_counts": _WITHHELD_ARTIFACTS,
        "gate_reason": GATE_REASON,
        "source_evidence_sha256": evidence_hash,
    }
    canonical_run_hash = _hash_bytes(canonical_json_bytes(withheld_ledger))
    trim = _mapping(source_evidence["source_trim_reproduction"], "trim")
    instructions = _mapping(source_evidence["source_instructions"], "instructions")
    mappings = _mapping(
        source_evidence["source_alignment_decisions"], "alignment decisions"
    )
    result: dict[str, object] = {
        "alignment_contract_version": ALIGNMENT_CONTRACT_VERSION,
        "alignment_method": ALIGNMENT_METHOD,
        "ambiguous_pair_contracts": _ambiguous_pair_contracts(),
        "ambiguous_pair_ids": sorted(AMBIGUOUS_RECORDING_IDS),
        "ambiguous_source_pairs": 2,
        "artifact_kind": ARTIFACT_KIND,
        "audit_execution_status": AUDIT_PASS,
        "blockers": [
            {
                "blocker_id": "RAW_POLL_GROUP_NOT_NOMINAL_50HZ_TICK",
                "evidence": (
                    "The 208 originals contain 2,036,601 structural five-sensor "
                    "RAW polling groups but only 299,711 dynamic 50 Hz STO/MOT "
                    "ordinals (aggregate ratio 6.795216)."
                ),
            },
            {
                "blocker_id": "SOURCE_RAW_TRIM_BREAKS_SENSOR_GROUP",
                "evidence": (
                    "All 34 RAW trims delete the five N-pose rows and 30 of 34 "
                    "end mid five-sensor group. Exact source replay cannot also "
                    "produce complete RAW-derived 50 Hz ticks."
                ),
            },
            {
                "blocker_id": "RAW_TO_STO_MOT_ORDINAL_BINDING_NOT_RELEASED",
                "evidence": (
                    "The complete source-tools release contains no RAW-to-STO "
                    "selection map. A positional RAW group-to-50 Hz ordinal map "
                    "would therefore be heuristic, not source-derived."
                ),
            },
            {
                "blocker_id": "SOURCE_MODIFIER_SKIPS_MP4",
                "evidence": (
                    "VIDIMU's modifier applies CSV, MOT, and RAW trims but skips "
                    "MP4. Promoting the CSV cut to decoded video-frame ordinals "
                    "would add an undocumented mapping beyond source behavior."
                ),
            },
            {
                "blocker_id": "DUAL_DIRECTION_APPLIED_OVERRIDE",
                "evidence": (
                    "S53_A13_T03 and S57_A07_T01 retain published transformations "
                    "in both directions and remain excluded without mappings."
                ),
            },
        ],
        "byte_identical_sto_materialization": False,
        "canonical_clocks_created": 0,
        "clock_segments_created": 0,
        "derived_rate_contracts_created": 0,
        "generic_success_markers_created": 0,
        "implementation": {
            "audit_implementation_version": IMPLEMENTATION_VERSION,
            "audit_script_sha256": script_hash,
            "authority_contract_sha256": source_evidence["frozen_inputs"][
                "v05d_authority_module_sha256"
            ],
            "canonical_json": "SORTED_KEYS_ASCII_INDENT2_TRAILING_NEWLINE_NO_NAN",
            "schema_contract_sha256": source_evidence["frozen_inputs"][
                "v05d_schema_module_sha256"
            ],
            "source_transform_sha256": source_evidence["frozen_inputs"][
                "v05d_source_transform_module_sha256"
            ],
        },
        "imu_tick_groups_created": 0,
        "materialization_claim_boundary": {
            "canonical_clock_claimed": False,
            "canonical_time_fields_emitted": False,
            "derived_alignment_parquet_emitted": False,
            "exact_physical_sample_times_claimed": False,
            "frame_to_imu_index_emitted": False,
            "hardware_synchronization_claimed": False,
            "measured_drift_claimed": False,
            "native_clock_mapping_claimed": False,
            "sto_success_marker_emitted": False,
            "timestamp_level_alignment_accuracy_claimed": False,
            "windows_emitted": False,
        },
        "overrides_bound": trim["overrides_bound"],
        "overrides_expected": trim["overrides_expected"],
        "overrides_parsed": instructions["non_mp4_instruction_count"],
        "overrides_reproduced": trim["overrides_reproduced"],
        "overrides_unreproduced": trim["overrides_unreproduced"],
        "raw_native_clock_gate": RAW_NATIVE_CLOCK_GATE,
        "run_a_canonical_hash": canonical_run_hash,
        "run_b_canonical_hash": canonical_run_hash,
        "run_canonical_hash_basis": (
            "DETERMINISTIC_WITHHELD_MATERIALIZATION_LEDGER_NOT_STO_BYTES"
        ),
        "schema_version": SCHEMA_VERSION,
        "source_derived_alignment_gate": SOURCE_DERIVED_ALIGNMENT_GATE_NO_GO,
        "source_derived_alignment_gate_reason": GATE_REASON,
        "source_derived_candidate_pairs_upper_bound": mappings[
            "eligible_source_pairs_upper_bound"
        ],
        "source_derived_pairs_eligible": 0,
        "source_evidence": source_evidence,
        "source_evidence_sha256": evidence_hash,
        "source_trim_overlays_created": 0,
        "sto_alignment_contracts_created": 0,
        "sto_alignment_validation_created": 0,
        "sto_success_markers_created": 0,
        "withheld_artifact_counts": _WITHHELD_ARTIFACTS,
    }
    if sto_success_marker_allowed(result):
        raise DerivedAlignmentAuditError("NO-GO audit unexpectedly permits a marker")
    if _read_script_hash(script) != script_hash:
        raise DerivedAlignmentAuditError("audit implementation changed during run")
    return result


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--analysis-archive", required=True, type=Path)
    parser.add_argument("--tools-archive", required=True, type=Path)
    parser.add_argument("--v05-authority-script", required=True, type=Path)
    parser.add_argument("--v05-authority-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _open_directory_chain(path: Path, flags: int) -> int:
    """Open an absolute directory without following any path component."""

    if not path.is_absolute() or not path.anchor:
        raise OSError(errno.EINVAL, "directory path must be absolute")
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                flags,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_exclusive(
    path: Path,
    payload: bytes,
    forbidden: Sequence[Path] = (),
) -> None:
    """Publish one complete audit with descriptor-relative no-replace semantics."""

    if _SAFE_COMPONENT.fullmatch(path.name) is None or path.name in {".", ".."}:
        raise DerivedAlignmentAuditError("output name is not a safe component")
    try:
        destination_lstat = path.lstat()
    except FileNotFoundError:
        destination_lstat = None
    except OSError as exc:
        raise DerivedAlignmentAuditError("cannot inspect output destination") from exc
    if destination_lstat is not None and stat.S_ISLNK(destination_lstat.st_mode):
        raise DerivedAlignmentAuditError("output destination must not be a symlink")
    destination = path.resolve(strict=False)
    for source in forbidden:
        source_resolved = source.resolve(strict=True)
        if destination == source_resolved or (
            source_resolved.is_dir()
            and destination.is_relative_to(source_resolved)
        ):
            raise DerivedAlignmentAuditError("output must not alias or enter an input")
    parent = destination.parent
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise DerivedAlignmentAuditError(
            "secure audit publication requires O_NOFOLLOW and O_DIRECTORY"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        directory_descriptor = _open_directory_chain(parent, flags)
    except OSError as exc:
        raise DerivedAlignmentAuditError(
            "cannot pin output parent directory"
        ) from exc
    parent_stat = os.fstat(directory_descriptor)
    parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
    temporary_name = f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary_descriptor = -1
    temporary_identity: tuple[int, int] | None = None
    linked = False
    try:
        try:
            os.stat(
                destination.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise DerivedAlignmentAuditError("audit output already exists")
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        create_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        temporary_descriptor = os.open(
            temporary_name, create_flags, 0o600, dir_fd=directory_descriptor
        )
        temporary_stat = os.fstat(temporary_descriptor)
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        offset = 0
        view = memoryview(payload)
        while offset < len(view):
            written = os.write(temporary_descriptor, view[offset:])
            if written <= 0:
                raise OSError(errno.EIO, "audit write made no progress")
            offset += written
        os.fsync(temporary_descriptor)
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise DerivedAlignmentAuditError("audit output already exists") from exc
            raise
        linked = True
        os.fsync(directory_descriptor)
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
        final_stat = os.stat(
            destination.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            temporary_identity is None
            or not stat.S_ISREG(final_stat.st_mode)
            or (final_stat.st_dev, final_stat.st_ino) != temporary_identity
            or final_stat.st_size != len(payload)
        ):
            raise DerivedAlignmentAuditError("published audit identity mismatch")
        try:
            current_parent_descriptor = _open_directory_chain(parent, flags)
        except OSError as exc:
            raise DerivedAlignmentAuditError(
                "output parent changed during publication"
            ) from exc
        try:
            current_parent = os.fstat(current_parent_descriptor)
        finally:
            os.close(current_parent_descriptor)
        if (current_parent.st_dev, current_parent.st_ino) != parent_identity:
            raise DerivedAlignmentAuditError(
                "output parent changed during publication"
            )
    except BaseException:
        if linked:
            try:
                os.unlink(destination.name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        try:
            os.fsync(directory_descriptor)
        except OSError:
            pass
        raise
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        os.close(directory_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        result = audit_vidimu_v05d_derived_alignment(
            args.snapshot_root,
            args.analysis_archive,
            args.tools_archive,
            args.v05_authority_script,
            args.v05_authority_report,
        )
        payload = canonical_json_bytes(result)
        _write_exclusive(
            args.output,
            payload,
            (
                args.snapshot_root,
                args.analysis_archive,
                args.tools_archive,
                args.v05_authority_script,
                args.v05_authority_report,
            ),
        )
    except (DerivedAlignmentAuditError, OSError) as exc:
        print(f"{AUDIT_ERROR}: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(payload)
    return NO_GO_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
