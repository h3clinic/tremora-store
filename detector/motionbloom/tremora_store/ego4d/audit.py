"""E4D-P0.1 audit engine and CLI.

The audit determines exactly which Ego4D IMU rows and video components can
support a source-authoritative canonical timeline.  It creates no frame-to-IMU
index, window, spectrum or performance benchmark; gate condition 11 fails the
audit if one is emitted.

Audit-execution status and gate status are separate, exactly as in v0.5D: a
successful audit that closes the gate reports PASS execution and a NO_GO gate,
and exits 3.  An audit whose dataset is not present reports
BLOCKED_INPUT_DATA_UNAVAILABLE and exits 4 -- distinct from 3, so a caller can
tell "we audited and it failed" from "we have not got the data".
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..release_gate import (
    AUDIT_EXECUTION_ERROR,
    AUDIT_EXECUTION_PASS,
    RELEASE_GATE_CONTRACT_VERSION,
    WITHHELD_P01_ARTIFACTS,
    blocked_record,
    canonical_json_bytes,
    canonical_sha256,
    exit_code_for,
)
from ..timing_authority import EGO4D_BINDING
from .authority import (
    EGO4D_ARTIFACT_KIND,
    EGO4D_CONTRACT_VERSION,
    EGO4D_IMPLEMENTATION_VERSION,
    EGO4D_SCHEMA_VERSION,
    MINIMUM_CAPTURE_DEVICE_GROUPS,
    MINIMUM_PAIRED_COVERAGE_HOURS,
    MINIMUM_SELECTED_VIDEOS,
    authority_contract,
)
from .coverage import (
    ComponentCoverage,
    component_coverage,
    merge_intervals,
    union_length,
)
from .gate import (
    Ego4DGateFacts,
    evaluate_gate,
    ordinal_sequence_is_intact,
    verify_reproduction,
)
from .imu_parser import (
    Ego4DParseError,
    parse_normalized_imu_csv,
)
from .metadata import (
    AssetEntry,
    Ego4DMetadataError,
    MetadataSnapshot,
    VideoTimeline,
    parse_asset_manifest,
    parse_metadata_snapshot,
)
from .pts_validation import (
    TIMELINE_RECONCILED,
    frame_times_ms,
    quantify_row_relationships,
    reconcile_pts_timeline,
)
from .row_status import IssueBit
from .selection import (
    STRATUM_CLEAN_MONOTONIC,
    STRATUM_EXTREME_TIMESTAMP,
    STRATUM_MISSING_ACCELERATION,
    STRATUM_NONMONOTONIC_SOURCE_ORDER,
    STRATUM_NULL_CANONICAL_TIMES,
    STRATUM_PARTIAL_COMPONENT_COVERAGE,
    VideoCandidate,
    select_subset,
)
from .tokens import TokenKind, classify

METADATA_SNAPSHOT_FILENAME = "ego4d_metadata_snapshot.json"
ASSET_MANIFEST_FILENAME = "ego4d_asset_manifest.json"

RELEASE_EVALUATED = "EVALUATED"

ASSET_VERIFIED = "ASSET_VERIFIED"
ASSET_MISSING = "ASSET_MISSING"
ASSET_HASH_MISMATCH = "ASSET_HASH_MISMATCH"
ASSET_UNPARSEABLE = "ASSET_UNPARSEABLE"
ASSET_NOT_IN_METADATA = "ASSET_NOT_IN_METADATA"

DecodeFrameTimes = Callable[[Path, str], tuple[float, ...]]


class Ego4DAuditError(RuntimeError):
    """Raised when the audit itself cannot run."""


@dataclass(slots=True)
class _VideoAccumulator:
    video_uid: str
    imu_rows_total: int = 0
    canonical_rows_valid: int = 0
    canonical_rows_null: int = 0
    canonical_rows_nonfinite: int = 0
    canonical_rows_nonmonotonic_source_order: int = 0
    canonical_rows_outside_video: int = 0
    canonical_rows_duplicate: int = 0
    canonical_rows_extreme: int = 0
    canonical_rows_unparseable: int = 0
    rows_missing_acceleration: int = 0
    rows_missing_gyroscope: int = 0
    components_present: set[int] | None = None
    intervals: list[tuple[float, float]] | None = None
    eligible_canonical_times: list[float] | None = None

    def __post_init__(self) -> None:
        self.components_present = set()
        self.intervals = []
        self.eligible_canonical_times = []


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _token_round_trips(token: str, stored: float | None) -> bool:
    """The stored number must be exactly what the preserved token says.

    A check that the token equals itself can only ever pass; this one
    re-derives the value from the preserved token and compares.
    """

    kind, value = classify(token)
    if kind is not TokenKind.DECIMAL:
        return stored is None
    return stored is not None and stored == value


def _default_decode_frame_times(
    path: Path, expected_sha256: str
) -> tuple[float, ...]:
    from ..decode.pts_decoder import PTSDecoder

    decoded = PTSDecoder().decode(path, expected_source_sha256=expected_sha256)
    return frame_times_ms(
        [frame.relative_pts_ns for frame in decoded.frames]
    )


def _strata_for(
    accumulator: _VideoAccumulator, timeline: VideoTimeline
) -> frozenset[str]:
    strata: set[str] = set()
    if accumulator.canonical_rows_nonmonotonic_source_order:
        strata.add(STRATUM_NONMONOTONIC_SOURCE_ORDER)
    if accumulator.canonical_rows_null:
        strata.add(STRATUM_NULL_CANONICAL_TIMES)
    if accumulator.rows_missing_acceleration:
        strata.add(STRATUM_MISSING_ACCELERATION)
    if accumulator.canonical_rows_extreme:
        strata.add(STRATUM_EXTREME_TIMESTAMP)
    expected_with_imu = timeline.components_with_imu
    present = len(accumulator.components_present or ())
    if expected_with_imu and present < expected_with_imu:
        strata.add(STRATUM_PARTIAL_COMPONENT_COVERAGE)
    if not strata and accumulator.canonical_rows_valid:
        strata.add(STRATUM_CLEAN_MONOTONIC)
    return frozenset(strata)


def _summary_record(
    accumulator: _VideoAccumulator, timeline: VideoTimeline
) -> dict[str, Any]:
    intervals = merge_intervals(accumulator.intervals or ())
    coverage_ms = union_length(intervals)
    eligible = accumulator.canonical_rows_valid > 0
    return {
        "video_uid": accumulator.video_uid,
        "imu_rows_total": accumulator.imu_rows_total,
        "canonical_rows_valid": accumulator.canonical_rows_valid,
        "canonical_rows_null": accumulator.canonical_rows_null,
        "canonical_rows_nonfinite": accumulator.canonical_rows_nonfinite,
        "canonical_rows_nonmonotonic_source_order": (
            accumulator.canonical_rows_nonmonotonic_source_order
        ),
        "canonical_rows_outside_video": (
            accumulator.canonical_rows_outside_video
        ),
        "canonical_rows_duplicate": accumulator.canonical_rows_duplicate,
        "canonical_rows_extreme": accumulator.canonical_rows_extreme,
        "canonical_rows_unparseable": accumulator.canonical_rows_unparseable,
        "rows_missing_acceleration": accumulator.rows_missing_acceleration,
        "rows_missing_gyroscope": accumulator.rows_missing_gyroscope,
        "components_expected": timeline.components_expected,
        "components_present": len(accumulator.components_present or ()),
        "components_with_imu": timeline.components_with_imu,
        "components_without_imu": (
            timeline.components_expected - timeline.components_with_imu
        ),
        "canonical_coverage_start_ms": (
            intervals[0][0] if intervals else None
        ),
        "canonical_coverage_end_ms": intervals[-1][1] if intervals else None,
        "canonical_coverage_duration_ms": coverage_ms,
        "authority_eligible": eligible,
        "ineligibility_reason": (
            None if eligible else "NO_AUTHORITY_ELIGIBLE_ROWS"
        ),
    }


def audit_ego4d_p01(
    *,
    metadata_root: Path,
    imu_root: Path,
    video_root: Path | None,
    publication_destination: str,
    reproduction_record: Mapping[str, Any] | None = None,
    decode_frame_times: DecodeFrameTimes | None = None,
    minimum_videos: int = MINIMUM_SELECTED_VIDEOS,
    minimum_coverage_hours: float = MINIMUM_PAIRED_COVERAGE_HOURS,
    minimum_capture_device_groups: int = MINIMUM_CAPTURE_DEVICE_GROUPS,
) -> dict[str, Any]:
    """Run the E4D-P0.1 timing-authority audit and return its record."""

    snapshot_path = metadata_root / METADATA_SNAPSHOT_FILENAME
    manifest_path = metadata_root / ASSET_MANIFEST_FILENAME
    if not metadata_root.is_dir() or not snapshot_path.is_file():
        # Nothing to audit.  A malformed release would be evidence and close
        # the gate; only an absent one blocks.
        return blocked_record(
            binding=EGO4D_BINDING,
            artifact_kind=EGO4D_ARTIFACT_KIND,
            schema_version=EGO4D_SCHEMA_VERSION,
            implementation_version=EGO4D_IMPLEMENTATION_VERSION,
            reason=(
                "Ego4D metadata snapshot is not present under the metadata "
                "root"
            ),
            inspected_roots={
                "metadata_root": str(metadata_root),
                "imu_root": str(imu_root),
                "video_root": str(video_root) if video_root else None,
            },
        )

    decode = decode_frame_times or _default_decode_frame_times
    failures: list[str] = []

    snapshot_bytes = snapshot_path.read_bytes()
    snapshot_sha256 = _sha256_bytes(snapshot_bytes)
    try:
        snapshot: MetadataSnapshot | None = parse_metadata_snapshot(
            snapshot_bytes, snapshot_sha256=snapshot_sha256
        )
    except Ego4DMetadataError as exc:
        snapshot = None
        failures.append(f"metadata snapshot: {exc}")

    manifest_sha256 = None
    assets: dict[str, AssetEntry] = {}
    if manifest_path.is_file():
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        try:
            assets = parse_asset_manifest(manifest_bytes)
        except Ego4DMetadataError as exc:
            failures.append(f"asset manifest: {exc}")
    else:
        failures.append("asset manifest is absent")

    accumulators: dict[str, _VideoAccumulator] = {}
    asset_records: list[dict[str, Any]] = []
    data_line_total = 0
    authority_row_total = 0
    token_failures = 0
    inferred_timestamps = 0
    unclassified_rows = 0
    files_with_ordinal_gaps = 0
    valid_outside_interval = 0
    assets_hash_verified = 0
    assets_failed = 0

    for key in sorted(assets):
        entry = assets[key]
        timeline = snapshot.video(entry.video_uid) if snapshot else None
        status = ASSET_VERIFIED
        reason: str | None = None
        row_count = 0
        payload: bytes | None = None
        imu_path = imu_root / entry.imu_relative_path
        if timeline is None:
            status = ASSET_NOT_IN_METADATA
            reason = "video_uid is not in the metadata snapshot"
        elif not imu_path.is_file():
            status = ASSET_MISSING
            reason = "IMU asset is not present under the IMU root"
        else:
            payload = imu_path.read_bytes()
            observed = _sha256_bytes(payload)
            if observed != entry.imu_asset_sha256:
                status = ASSET_HASH_MISMATCH
                reason = "IMU asset content hash does not match the manifest"
                payload = None

        if payload is not None and timeline is not None:
            try:
                parsed = parse_normalized_imu_csv(
                    payload,
                    video_uid=entry.video_uid,
                    source_asset_sha256=entry.imu_asset_sha256,
                    timeline=timeline,
                )
            except Ego4DParseError as exc:
                status = ASSET_UNPARSEABLE
                reason = str(exc)
            else:
                assets_hash_verified += 1
                row_count = len(parsed.rows)
                data_line_total += parsed.data_line_count
                authority_row_total += row_count
                if not ordinal_sequence_is_intact(
                    [row.source_row_ordinal for row in parsed.rows],
                    parsed.data_line_count,
                ):
                    files_with_ordinal_gaps += 1

                accumulator = accumulators.setdefault(
                    entry.video_uid, _VideoAccumulator(entry.video_uid)
                )
                eligible_by_component: dict[int, list[float]] = {}
                for row in parsed.rows:
                    accumulator.imu_rows_total += 1
                    assert accumulator.components_present is not None
                    accumulator.components_present.add(row.component_idx)
                    bits = IssueBit(row.issue_bits)
                    if not _token_round_trips(
                        row.canonical_timestamp_token,
                        row.canonical_timestamp_ms,
                    ) or not _token_round_trips(
                        row.component_timestamp_token,
                        row.component_timestamp_ms,
                    ):
                        token_failures += 1
                    if (
                        bits
                        & (
                            IssueBit.SOURCE_CANONICAL_NULL_AFTER_TRIM
                            | IssueBit.SOURCE_CANONICAL_NONFINITE
                            | IssueBit.SOURCE_CANONICAL_UNPARSEABLE_TOKEN
                        )
                        and row.canonical_timestamp_ms is not None
                    ):
                        inferred_timestamps += 1
                    if bits and row.canonical_authority_status == (
                        "SOURCE_CANONICAL_VALID"
                    ):
                        unclassified_rows += 1
                    if IssueBit.SOURCE_CANONICAL_NULL_AFTER_TRIM & bits:
                        accumulator.canonical_rows_null += 1
                    if IssueBit.SOURCE_CANONICAL_NONFINITE & bits:
                        accumulator.canonical_rows_nonfinite += 1
                    if IssueBit.SOURCE_CANONICAL_NONMONOTONIC & bits:
                        accumulator.canonical_rows_nonmonotonic_source_order += 1
                    if IssueBit.SOURCE_CANONICAL_OUTSIDE_VIDEO & bits:
                        accumulator.canonical_rows_outside_video += 1
                    if IssueBit.SOURCE_CANONICAL_DUPLICATE & bits:
                        accumulator.canonical_rows_duplicate += 1
                    if IssueBit.SOURCE_CANONICAL_EXTREME_MAGNITUDE & bits:
                        accumulator.canonical_rows_extreme += 1
                    if IssueBit.SOURCE_CANONICAL_UNPARSEABLE_TOKEN & bits:
                        accumulator.canonical_rows_unparseable += 1
                    if IssueBit.MISSING_ACCELERATION & bits:
                        accumulator.rows_missing_acceleration += 1
                    if IssueBit.MISSING_GYROSCOPE & bits:
                        accumulator.rows_missing_gyroscope += 1
                    if row.eligible:
                        accumulator.canonical_rows_valid += 1
                        assert row.canonical_timestamp_ms is not None
                        eligible_by_component.setdefault(
                            row.component_idx, []
                        ).append(row.canonical_timestamp_ms)
                        duration = timeline.canonical_video_duration_ms
                        if not 0.0 <= row.canonical_timestamp_ms <= duration:
                            valid_outside_interval += 1

                for component_idx, times in sorted(
                    eligible_by_component.items()
                ):
                    coverage: ComponentCoverage = component_coverage(
                        component_idx,
                        times,
                        clamp_low=0.0,
                        clamp_high=timeline.canonical_video_duration_ms,
                    )
                    assert accumulator.intervals is not None
                    accumulator.intervals.extend(coverage.intervals)
                assert accumulator.eligible_canonical_times is not None
                for times in eligible_by_component.values():
                    accumulator.eligible_canonical_times.extend(times)

        if status != ASSET_VERIFIED:
            assets_failed += 1
        component = (
            timeline.component(entry.component_idx) if timeline else None
        )
        asset_records.append({
            "video_uid": entry.video_uid,
            "component_idx": entry.component_idx,
            "imu_asset_sha256": entry.imu_asset_sha256,
            "video_component_asset_sha256": (
                entry.video_component_asset_sha256
            ),
            "canonical_video_asset_sha256": (
                entry.canonical_video_asset_sha256
            ),
            "imu_row_count": row_count,
            "source_component_start_ms": (
                component.component_start_in_canonical_ms
                if component else None
            ),
            "source_component_end_ms": (
                component.component_end_in_canonical_ms if component else None
            ),
            "asset_status": status,
            "failure_reason": reason,
        })

    timeline_records: list[dict[str, Any]] = []
    if snapshot is not None:
        for video in snapshot.videos:
            for component in video.components:
                timeline_records.append({
                    "video_uid": video.video_uid,
                    "component_idx": component.component_idx,
                    "component_start_in_canonical_ms": (
                        component.component_start_in_canonical_ms
                    ),
                    "component_end_in_canonical_ms": (
                        component.component_end_in_canonical_ms
                    ),
                    "canonical_video_duration_ms": (
                        video.canonical_video_duration_ms
                    ),
                    "video_stream_start_ms": video.video_stream_start_ms,
                    "video_stream_end_ms": video.video_stream_end_ms,
                    "metadata_source_sha256": snapshot_sha256,
                    "timeline_status": component.timeline_status,
                })

    candidates: list[VideoCandidate] = []
    summaries: list[dict[str, Any]] = []
    if snapshot is not None:
        for video_uid in sorted(accumulators):
            accumulator = accumulators[video_uid]
            timeline = snapshot.video(video_uid)
            if timeline is None:  # pragma: no cover - filtered above
                continue
            summary = _summary_record(accumulator, timeline)
            summaries.append(summary)
            candidates.append(VideoCandidate(
                video_uid=video_uid,
                strata=_strata_for(accumulator, timeline),
                paired_coverage_ms=float(
                    summary["canonical_coverage_duration_ms"]
                ),
                capture_device_group=timeline.capture_device_group,
            ))

    selection = select_subset(
        candidates,
        metadata_snapshot_sha256=snapshot_sha256,
        minimum_videos=minimum_videos,
        minimum_coverage_hours=minimum_coverage_hours,
        minimum_capture_device_groups=minimum_capture_device_groups,
    )

    reconciliations: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    videos_with_timeline = 0
    videos_disagreeing = 0
    for video_uid in selection.selected_video_uids:
        timeline = snapshot.video(video_uid) if snapshot else None
        entry = next(
            (
                assets[key]
                for key in sorted(assets)
                if assets[key].video_uid == video_uid
            ),
            None,
        )
        if timeline is None or entry is None or video_root is None:
            reconciliations.append({
                "video_uid": video_uid,
                "timeline_status": "TIMELINE_NOT_DECODED",
                "reason": (
                    "video root was not supplied"
                    if video_root is None
                    else "no asset entry or timeline for this video"
                ),
            })
            videos_disagreeing += 1
            continue
        video_path = video_root / entry.canonical_video_relative_path
        try:
            times = decode(
                video_path, entry.canonical_video_asset_sha256
            )
        except Exception as exc:  # noqa: BLE001 - any decode failure is evidence
            reconciliations.append({
                "video_uid": video_uid,
                "timeline_status": "TIMELINE_DECODE_FAILED",
                "reason": type(exc).__name__,
            })
            videos_disagreeing += 1
            continue
        videos_with_timeline += 1
        reconciliation = reconcile_pts_timeline(
            times,
            canonical_video_duration_ms=(
                timeline.canonical_video_duration_ms
            ),
        )
        if reconciliation.timeline_status != TIMELINE_RECONCILED:
            videos_disagreeing += 1
        reconciliations.append({
            "video_uid": video_uid,
            "timeline_status": reconciliation.timeline_status,
            "frame_count": reconciliation.frame_count,
            "origin_offset_ms": reconciliation.origin_offset_ms,
            "origin_tolerance_ms": reconciliation.origin_tolerance_ms,
            "span_difference_ms": reconciliation.span_difference_ms,
            "span_tolerance_ms": reconciliation.span_tolerance_ms,
            "frame_interval_ms": reconciliation.frame_interval_ms,
        })
        accumulator = accumulators.get(video_uid)
        eligible_times = sorted(
            accumulator.eligible_canonical_times or ()
        ) if accumulator else []
        quantified = quantify_row_relationships(times, eligible_times)
        relationships.append({
            "video_uid": video_uid,
            "eligible_row_count": quantified.eligible_row_count,
            "rows_inside_a_frame_interval": (
                quantified.rows_inside_a_frame_interval
            ),
            "rows_before_first_frame": quantified.rows_before_first_frame,
            "rows_after_last_frame": quantified.rows_after_last_frame,
            "max_nearest_frame_delta_ms": (
                quantified.max_nearest_frame_delta_ms
            ),
            "median_nearest_frame_delta_ms": (
                quantified.median_nearest_frame_delta_ms
            ),
        })

    evidence: dict[str, Any] = {
        "artifact_kind": EGO4D_ARTIFACT_KIND,
        "schema_version": EGO4D_SCHEMA_VERSION,
        "implementation_version": EGO4D_IMPLEMENTATION_VERSION,
        "contract_version": EGO4D_CONTRACT_VERSION,
        "release_gate_contract_version": RELEASE_GATE_CONTRACT_VERSION,
        "authority": authority_contract(),
        "metadata_snapshot_sha256": snapshot_sha256,
        "asset_manifest_sha256": manifest_sha256,
        "source_failures": sorted(failures),
        "assets": {
            "expected": len(assets),
            "hash_verified": assets_hash_verified,
            "failed": assets_failed,
            "records": asset_records,
        },
        "rows": {
            "source_data_lines": data_line_total,
            "authority_rows": authority_row_total,
            "inferred_timestamps": inferred_timestamps,
            "unclassified_issue_rows": unclassified_rows,
            "token_preservation_failures": token_failures,
            "valid_rows_outside_video_interval": valid_outside_interval,
            "files_with_ordinal_gaps": files_with_ordinal_gaps,
        },
        "video_timeline_authority": timeline_records,
        "timing_authority_summary": summaries,
        "subset_selection": selection.as_record(),
        "pts_reconciliation": reconciliations,
        "row_frame_relationships": relationships,
        "withheld_artifacts": dict(WITHHELD_P01_ARTIFACTS),
        "materialized_release_artifacts": 0,
    }
    evidence_sha256 = canonical_sha256(evidence)

    record: dict[str, Any] = dict(evidence)
    record["audit_execution_status"] = AUDIT_EXECUTION_PASS
    record["release_status"] = RELEASE_EVALUATED
    record["gate_evaluated"] = True
    record["canonical_evidence_sha256"] = evidence_sha256
    record["publication_destination"] = publication_destination

    facts = Ego4DGateFacts(
        assets_expected=len(assets),
        assets_hash_verified=assets_hash_verified,
        assets_failed=assets_failed,
        data_line_total=data_line_total,
        authority_row_total=authority_row_total,
        token_preservation_failures=token_failures,
        inferred_timestamp_count=inferred_timestamps,
        selected_video_count=selection.selected_video_count,
        videos_with_pts_timeline=videos_with_timeline,
        videos_with_timeline_disagreement=videos_disagreeing,
        valid_rows_outside_video_interval=valid_outside_interval,
        unclassified_issue_rows=unclassified_rows,
        files_with_ordinal_gaps=files_with_ordinal_gaps,
        reproduction_status=verify_reproduction(record, reproduction_record),
        subset_floors_satisfied=selection.floors_satisfied,
        subset_shortfalls=dict(selection.shortfalls),
        emitted_forbidden_artifacts=dict(WITHHELD_P01_ARTIFACTS),
    )
    if failures:
        facts.assets_failed += len(failures)
    record.update(evaluate_gate(facts).as_record())
    return record


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--imu-root", required=True, type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reproduction-record", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        reproduction = None
        if args.reproduction_record is not None:
            import json

            reproduction = json.loads(
                args.reproduction_record.read_bytes().decode("utf-8")
            )
        record = audit_ego4d_p01(
            metadata_root=args.metadata_root,
            imu_root=args.imu_root,
            video_root=args.video_root,
            publication_destination=str(args.output),
            reproduction_record=reproduction,
        )
        payload = canonical_json_bytes(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(payload)
    except (Ego4DAuditError, OSError, ValueError) as exc:
        print(f"{AUDIT_EXECUTION_ERROR}: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(payload)
    return exit_code_for(record)


if __name__ == "__main__":
    raise SystemExit(main())
