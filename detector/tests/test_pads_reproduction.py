from __future__ import annotations

import json
from pathlib import Path

import pytest
from motionbloom.tremora_store.pads import reproduction
from motionbloom.tremora_store.pads.reproduction import (
    PadsReproductionError,
    build_run_receipt,
    output_root_identity,
    verify_independent_reproduction,
)


def _receipt(tmp_path: Path, name: str, **overrides):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    defaults = {
        "dataset_root": tmp_path / "release",
        "source_manifest_sha256": "a" * 64,
        "output_root": root,
        "schema_version": "pads-p0.1.0",
        "contract_version": "contract-0.1.0",
        "implementation_version": "impl-1.0.0",
        "canonical_evidence_sha256": "e" * 64,
        "command_arguments": ("--dataset-root", "release"),
    }
    defaults.update(overrides)
    return build_run_receipt(**defaults).as_record()


def test_two_clean_roots_are_a_genuine_reproduction(tmp_path: Path) -> None:
    first = _receipt(tmp_path, "a", run_id="r1", process_id=101)
    second = _receipt(tmp_path, "b", run_id="r2", process_id=202)
    assert verify_independent_reproduction(first, second) == (
        reproduction.REPRODUCTION_VERIFIED
    )


def test_a_copied_report_is_not_a_second_execution(tmp_path: Path) -> None:
    first = _receipt(tmp_path, "a", run_id="r1", process_id=101)
    copied = json.loads(json.dumps(first))
    assert verify_independent_reproduction(first, copied) == (
        reproduction.REPRODUCTION_SAME_RUN_ID
    )


def test_the_same_process_is_not_a_second_execution(tmp_path: Path) -> None:
    first = _receipt(tmp_path, "a", run_id="r1", process_id=101)
    second = _receipt(tmp_path, "b", run_id="r2", process_id=101)
    assert verify_independent_reproduction(first, second) == (
        reproduction.REPRODUCTION_SAME_PROCESS
    )


def test_the_same_output_root_is_not_a_second_execution(
    tmp_path: Path,
) -> None:
    first = _receipt(tmp_path, "a", run_id="r1", process_id=101)
    second = _receipt(tmp_path, "a", run_id="r2", process_id=202)
    assert verify_independent_reproduction(first, second) == (
        reproduction.REPRODUCTION_SAME_OUTPUT_ROOT
    )


def test_a_hard_linked_output_root_shares_an_inode(tmp_path: Path) -> None:
    first = _receipt(tmp_path, "a", run_id="r1", process_id=101)
    second = _receipt(tmp_path, "b", run_id="r2", process_id=202)
    # Two different names for the same directory inode are one output root.
    second["output_root_identity"] = first["output_root_identity"]
    assert verify_independent_reproduction(first, second) == (
        reproduction.REPRODUCTION_SAME_OUTPUT_INODE
    )


def test_a_different_release_is_not_a_reproduction(tmp_path: Path) -> None:
    first = _receipt(tmp_path, "a", run_id="r1", process_id=101)
    second = _receipt(
        tmp_path, "b", run_id="r2", process_id=202,
        source_manifest_sha256="b" * 64,
    )
    assert verify_independent_reproduction(first, second) == (
        reproduction.REPRODUCTION_SOURCE_MISMATCH
    )


@pytest.mark.parametrize(
    "field", ("schema_version", "contract_version", "implementation_version")
)
def test_a_different_implementation_is_not_a_reproduction(
    tmp_path: Path, field: str
) -> None:
    first = _receipt(tmp_path, "a", run_id="r1", process_id=101)
    second = _receipt(
        tmp_path, "b", run_id="r2", process_id=202, **{field: "changed"}
    )
    assert verify_independent_reproduction(first, second) == (
        reproduction.REPRODUCTION_IMPLEMENTATION_MISMATCH
    )


def test_different_evidence_is_not_a_reproduction(tmp_path: Path) -> None:
    first = _receipt(tmp_path, "a", run_id="r1", process_id=101)
    second = _receipt(
        tmp_path, "b", run_id="r2", process_id=202,
        canonical_evidence_sha256="f" * 64,
    )
    assert verify_independent_reproduction(first, second) == (
        reproduction.REPRODUCTION_EVIDENCE_MISMATCH
    )


def test_an_absent_evidence_hash_cannot_verify(tmp_path: Path) -> None:
    first = _receipt(
        tmp_path, "a", run_id="r1", process_id=101,
        canonical_evidence_sha256="",
    )
    second = _receipt(
        tmp_path, "b", run_id="r2", process_id=202,
        canonical_evidence_sha256="",
    )
    assert verify_independent_reproduction(first, second) == (
        reproduction.REPRODUCTION_EVIDENCE_MISMATCH
    )


def test_one_execution_is_not_a_reproduction(tmp_path: Path) -> None:
    first = _receipt(tmp_path, "a", run_id="r1", process_id=101)
    assert verify_independent_reproduction(first, None) == (
        reproduction.REPRODUCTION_NOT_ATTEMPTED
    )
    assert verify_independent_reproduction(None, None) == (
        reproduction.REPRODUCTION_NOT_ATTEMPTED
    )


def test_a_receipt_missing_a_field_cannot_verify(tmp_path: Path) -> None:
    first = _receipt(tmp_path, "a", run_id="r1", process_id=101)
    second = _receipt(tmp_path, "b", run_id="r2", process_id=202)
    second.pop("process_id")
    assert verify_independent_reproduction(first, second) == (
        reproduction.REPRODUCTION_MALFORMED_RECEIPT
    )


def test_an_absent_output_root_cannot_produce_a_receipt(
    tmp_path: Path,
) -> None:
    with pytest.raises(PadsReproductionError):
        output_root_identity(tmp_path / "missing")
