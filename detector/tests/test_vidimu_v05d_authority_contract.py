from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from motionbloom.tremora_store.v05d import authority
from motionbloom.tremora_store.v05d.schemas import V05D_TABLE_SCHEMAS

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v05_no_go_artifacts_remain_unchanged() -> None:
    assert _sha256(
        REPO_ROOT / "detector/benchmarks/audit_vidimu_v05_sync_authority.py"
    ) == "abbd59097a16f767729521b4968ac997a55506e467621b997b1894d45334d65a"
    assert _sha256(
        REPO_ROOT / "detector/benchmarks/vidimu_v05_sync_authority_audit.json"
    ) == "3d4492f984ddffaed579da2e107aaf9f7d1e9cdae1ddc83629f8708d8e75bdec"


def test_sto_contract_is_separately_versioned() -> None:
    assert authority.ALIGNMENT_CONTRACT_VERSION.startswith(
        "tremora-vidimu-sto-derived-alignment-0.5d."
    )
    assert authority.RAW_NATIVE_CLOCK_GATE == "NO_GO_RAW_NATIVE_CLOCK_AUTHORITY"
    for factory in V05D_TABLE_SCHEMAS.values():
        schema = factory()
        assert schema.metadata[b"tremora.schema_version"] == b"0.5d.0"
        assert schema.metadata[b"tremora.temporal_domain"] == (
            b"ORDINAL_ONLY_NO_CANONICAL_CLOCK"
        )


def test_sto_authority_never_reports_raw_clock() -> None:
    assert tuple(value.value for value in authority.AlignmentAuthority) == (
        "RAW_SHARED_CLOCK",
        "RAW_MAPPED_CLOCK",
        "SOURCE_DERIVED_ALIGNMENT",
        "HEURISTIC_ALIGNMENT",
        "AMBIGUOUS_SOURCE_ALIGNMENT",
        "UNRESOLVED",
    )
    authority.assert_benchmark_eligible_authority("SOURCE_DERIVED_ALIGNMENT")
    with pytest.raises(authority.V05DContractError):
        authority.assert_benchmark_eligible_authority("HEURISTIC_ALIGNMENT")
    with pytest.raises(authority.V05DContractError):
        authority.assert_benchmark_eligible_authority(
            "AMBIGUOUS_SOURCE_ALIGNMENT"
        )


def test_no_canonical_time_fields_emitted() -> None:
    emitted = {
        field.name
        for factory in V05D_TABLE_SCHEMAS.values()
        for field in factory()
    }
    assert "canonical_time_ns" not in emitted
    with pytest.raises(authority.V05DContractError):
        authority.assert_no_forbidden_clock_fields(["canonical_time_ns"])


def test_no_clock_drift_fields_emitted() -> None:
    emitted = {
        field.name
        for factory in V05D_TABLE_SCHEMAS.values()
        for field in factory()
    }
    assert not emitted.intersection({
        "clock_offset_ns",
        "clock_scale",
        "drift_ppm",
        "sync_residual_ns",
    })
    with pytest.raises(authority.V05DContractError):
        authority.assert_no_forbidden_clock_fields(["drift_ppm"])


def test_known_ambiguities_are_frozen() -> None:
    assert authority.AMBIGUOUS_RECORDING_IDS == {
        "S53_A13_T03",
        "S57_A07_T01",
    }
