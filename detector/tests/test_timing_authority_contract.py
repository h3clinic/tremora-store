from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from motionbloom.tremora_store import release_gate, timing_authority
from motionbloom.tremora_store.timing_authority import (
    DatasetTimingBinding,
    TimingAuthority,
    TimingAuthorityError,
)
from motionbloom.tremora_store.v05d import authority as v05d_authority
from motionbloom.tremora_store.v05d.source_transform import (
    canonical_json_bytes as v05d_canonical_json_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TIER_ORDER = (
    "RAW_SHARED_CLOCK",
    "RAW_MAPPED_CLOCK",
    "SOURCE_CANONICAL_TIMESTAMP",
    "SOURCE_ALIGNED_RELATIVE_TIME",
    "SOURCE_DERIVED_ALIGNMENT",
    "SOURCE_RELATIVE_UNIMODAL_CLOCK",
    "AMBIGUOUS",
    "UNRESOLVED",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tier_order_is_frozen() -> None:
    assert tuple(
        tier.value for tier in timing_authority.TIMING_AUTHORITY_ORDER
    ) == EXPECTED_TIER_ORDER
    assert tuple(tier.value for tier in TimingAuthority) == EXPECTED_TIER_ORDER


def test_v05d_alignment_authority_enum_is_untouched() -> None:
    assert tuple(
        member.value for member in v05d_authority.AlignmentAuthority
    ) == (
        "RAW_SHARED_CLOCK",
        "RAW_MAPPED_CLOCK",
        "SOURCE_DERIVED_ALIGNMENT",
        "HEURISTIC_ALIGNMENT",
        "AMBIGUOUS_SOURCE_ALIGNMENT",
        "UNRESOLVED",
    )


def test_frozen_vidimu_evidence_remains_hash_pinned() -> None:
    assert _sha256(
        REPO_ROOT / "detector/benchmarks/vidimu_v05_sync_authority_audit.json"
    ) == "3d4492f984ddffaed579da2e107aaf9f7d1e9cdae1ddc83629f8708d8e75bdec"
    assert _sha256(
        REPO_ROOT
        / "detector/benchmarks/vidimu_v05d_derived_alignment_release_audit.json"
    ) == "131a6110d699ed8d0ebd7611c820112f1fe6af5c0e44f116181bd4a8495ac1b0"


@pytest.mark.parametrize("tier", EXPECTED_TIER_ORDER[:5])
def test_paired_index_admits_the_five_strongest_tiers(tier: str) -> None:
    assert timing_authority.assert_frame_imu_index_allowed(tier).value == tier


@pytest.mark.parametrize("tier", EXPECTED_TIER_ORDER[5:])
def test_paired_index_refuses_unimodal_and_unresolving_tiers(
    tier: str,
) -> None:
    with pytest.raises(TimingAuthorityError):
        timing_authority.assert_frame_imu_index_allowed(tier)


@pytest.mark.parametrize("tier", EXPECTED_TIER_ORDER[:6])
def test_storage_benchmark_additionally_admits_a_unimodal_clock(
    tier: str,
) -> None:
    assert timing_authority.assert_storage_benchmark_allowed(tier).value == tier


@pytest.mark.parametrize("tier", ("AMBIGUOUS", "UNRESOLVED"))
def test_storage_benchmark_refuses_unresolving_tiers(tier: str) -> None:
    with pytest.raises(TimingAuthorityError):
        timing_authority.assert_storage_benchmark_allowed(tier)


@pytest.mark.parametrize("tier", EXPECTED_TIER_ORDER[1:])
def test_only_a_shared_raw_clock_supports_a_hardware_sync_claim(
    tier: str,
) -> None:
    with pytest.raises(TimingAuthorityError):
        timing_authority.assert_hardware_sync_claim(tier)


def test_shared_raw_clock_supports_a_hardware_sync_claim() -> None:
    assert timing_authority.assert_hardware_sync_claim(
        "RAW_SHARED_CLOCK"
    ) is TimingAuthority.RAW_SHARED_CLOCK


def test_unknown_tier_spelling_is_refused() -> None:
    with pytest.raises(TimingAuthorityError):
        timing_authority.normalize("SOURCE_CANONICAL_TIMESTAMPS")


@pytest.mark.parametrize("tier", sorted(timing_authority.DERIVED_TIERS))
def test_a_derived_tier_must_state_its_assumption(
    tier: TimingAuthority,
) -> None:
    with pytest.raises(TimingAuthorityError):
        DatasetTimingBinding(dataset_id="X", timing_authority=tier)


@pytest.mark.parametrize("tier", sorted(timing_authority.UNRESOLVING_TIERS))
def test_an_unresolving_tier_cannot_state_an_assumption(
    tier: TimingAuthority,
) -> None:
    with pytest.raises(TimingAuthorityError):
        DatasetTimingBinding(
            dataset_id="X",
            timing_authority=tier,
            derived_under_assumption="wishful thinking",
        )


def test_hardware_sync_claim_is_refused_below_a_shared_raw_clock() -> None:
    with pytest.raises(TimingAuthorityError):
        DatasetTimingBinding(
            dataset_id="X",
            timing_authority=TimingAuthority.SOURCE_CANONICAL_TIMESTAMP,
            derived_under_assumption="documented origin",
            hardware_sync_claim=True,
        )


def test_vidimu_binds_unresolved_and_cannot_be_paired_indexed() -> None:
    binding = timing_authority.binding_for("VIDIMU")
    assert binding.timing_authority is TimingAuthority.UNRESOLVED
    assert binding.derived_under_assumption is None
    assert binding.hardware_sync_claim is False
    with pytest.raises(TimingAuthorityError):
        binding.assert_frame_imu_index_allowed()
    with pytest.raises(TimingAuthorityError):
        binding.assert_storage_benchmark_allowed()


def test_ego4d_binds_source_canonical_timestamp_under_a_stated_assumption(
) -> None:
    binding = timing_authority.binding_for("EGO4D")
    assert binding.timing_authority is (
        TimingAuthority.SOURCE_CANONICAL_TIMESTAMP
    )
    assert binding.hardware_sync_claim is False
    assert binding.paired_modalities is True
    assert "t=0" in (binding.derived_under_assumption or "")
    assert binding.assert_frame_imu_index_allowed() is (
        TimingAuthority.SOURCE_CANONICAL_TIMESTAMP
    )


def test_pads_binds_a_unimodal_source_clock_and_has_nothing_to_pair() -> None:
    binding = timing_authority.binding_for("PADS")
    assert binding.timing_authority is (
        TimingAuthority.SOURCE_RELATIVE_UNIMODAL_CLOCK
    )
    assert binding.paired_modalities is False
    assert "Time column" in (binding.derived_under_assumption or "")
    assert binding.assert_storage_benchmark_allowed() is (
        TimingAuthority.SOURCE_RELATIVE_UNIMODAL_CLOCK
    )
    with pytest.raises(TimingAuthorityError):
        binding.assert_frame_imu_index_allowed()


def test_a_unimodal_clock_cannot_declare_paired_modalities() -> None:
    with pytest.raises(TimingAuthorityError):
        DatasetTimingBinding(
            dataset_id="X",
            timing_authority=(
                TimingAuthority.SOURCE_RELATIVE_UNIMODAL_CLOCK
            ),
            derived_under_assumption="a declared rate",
            paired_modalities=True,
        )


def test_unknown_dataset_has_no_binding() -> None:
    with pytest.raises(TimingAuthorityError):
        timing_authority.binding_for("EGOINERTIA_MI")


def test_a_tier_table_that_omits_a_tier_is_refused() -> None:
    timing_authority.assert_tiers_exhaustive(EXPECTED_TIER_ORDER)
    with pytest.raises(TimingAuthorityError):
        timing_authority.assert_tiers_exhaustive(EXPECTED_TIER_ORDER[:-1])


def test_canonical_json_matches_the_v05d_evidence_encoding() -> None:
    payload = {"b": [1, 2, {"z": None, "a": True}], "a": "x"}
    assert release_gate.canonical_json_bytes(payload) == (
        v05d_canonical_json_bytes(payload)
    )


def test_canonical_json_refuses_non_finite_values() -> None:
    with pytest.raises(release_gate.ReleaseGateError):
        release_gate.canonical_json_bytes({"a": float("nan")})


def test_blocked_record_publishes_no_verdict_and_exits_four() -> None:
    record = release_gate.blocked_record(
        binding=timing_authority.binding_for("PADS"),
        artifact_kind="TREMORA_PADS_P01_INGEST_AUDIT",
        schema_version="0.1.0",
        implementation_version="pads-p01-audit-1.0.0",
        reason="dataset root does not exist",
        inspected_roots={"dataset_root": None},
    )
    assert record["release_status"] == "BLOCKED_INPUT_DATA_UNAVAILABLE"
    assert record["gate_evaluated"] is False
    assert "gate_status" not in record
    assert "canonical_evidence_sha256" not in record
    assert record["materialized_release_artifacts"] == 0
    assert record["authority"]["timing_authority"] == (
        "SOURCE_RELATIVE_UNIMODAL_CLOCK"
    )
    assert record["authority"]["raw_shared_clock"] is False
    assert release_gate.exit_code_for(record) == release_gate.EXIT_BLOCKED


def test_blocked_record_requires_a_stated_reason() -> None:
    with pytest.raises(release_gate.ReleaseGateError):
        release_gate.blocked_record(
            binding=timing_authority.binding_for("PADS"),
            artifact_kind="K",
            schema_version="0.1.0",
            implementation_version="i",
            reason="",
            inspected_roots={},
        )


@pytest.mark.parametrize(
    "leak",
    ("gate_status", "canonical_evidence_sha256", "evidence_sha256",
     "gate_conditions"),
)
def test_blocked_record_cannot_leak_a_verdict(leak: str) -> None:
    record = release_gate.blocked_record(
        binding=timing_authority.binding_for("EGO4D"),
        artifact_kind="K",
        schema_version="0.1.0",
        implementation_version="i",
        reason="metadata root does not exist",
        inspected_roots={"metadata_root": None},
    )
    record[leak] = "PASS"
    with pytest.raises(release_gate.ReleaseGateError):
        release_gate.assert_blocked_record_claim_boundary(record)


def test_blocked_record_cannot_drop_the_claim_boundary() -> None:
    record = release_gate.blocked_record(
        binding=timing_authority.binding_for("EGO4D"),
        artifact_kind="K",
        schema_version="0.1.0",
        implementation_version="i",
        reason="imu root does not exist",
        inspected_roots={"imu_root": None},
    )
    record["withheld_artifacts"].pop("frame_imu_index_tables")
    with pytest.raises(release_gate.ReleaseGateError):
        release_gate.assert_blocked_record_claim_boundary(record)


def test_exit_codes_separate_execution_from_verdict() -> None:
    assert release_gate.exit_code_for({
        "audit_execution_status": "PASS",
        "gate_evaluated": True,
        "gate_status": "PASS_SOURCE_CANONICAL_TIMESTAMP_AUTHORITY",
    }) == release_gate.EXIT_PASS
    assert release_gate.exit_code_for({
        "audit_execution_status": "PASS",
        "gate_evaluated": True,
        "gate_status": "NO_GO_EGO4D_CANONICAL_TIMESTAMP_AUTHORITY",
    }) == release_gate.EXIT_NO_GO
    assert release_gate.exit_code_for({
        "audit_execution_status": "ERROR",
    }) == release_gate.EXIT_ERROR


def test_a_non_blocked_record_must_evaluate_its_gate() -> None:
    with pytest.raises(release_gate.ReleaseGateError):
        release_gate.exit_code_for({
            "audit_execution_status": "PASS",
            "gate_evaluated": False,
        })


def test_unrecognized_gate_status_is_refused() -> None:
    with pytest.raises(release_gate.ReleaseGateError):
        release_gate.exit_code_for({
            "audit_execution_status": "PASS",
            "gate_evaluated": True,
            "gate_status": "PROBABLY_FINE",
        })
