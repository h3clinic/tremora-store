"""Release-input regression: re-derive the published PADS evidence.

Skipped when the release is not present.  Point ``TREMORA_PADS_DATASET_ROOT``
at an extracted PADS 1.0.0 root to run it; the audit must reproduce the exact
evidence hash the published report carries, from the real corpus, or the
implementation has drifted from the evidence it claims.

The run parses 10,318 device files and takes a couple of minutes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from motionbloom.tremora_store.pads.audit import audit_pads_p01
from motionbloom.tremora_store.pads.authority import GATE_PASS
from motionbloom.tremora_store.pads.gate import failing_conditions

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "detector/benchmarks/pads_p01_release_audit.json"
ROOT_VARIABLE = "TREMORA_PADS_DATASET_ROOT"


def _dataset_root() -> Path | None:
    value = os.environ.get(ROOT_VARIABLE)
    if not value:
        return None
    root = Path(value)
    return root if (root / "movement").is_dir() else None


pytestmark = pytest.mark.skipif(
    _dataset_root() is None,
    reason=(
        f"PADS_RELEASE_INPUT_UNAVAILABLE: set {ROOT_VARIABLE} to an extracted "
        "PADS 1.0.0 root"
    ),
)


@pytest.fixture(scope="module")
def rerun(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = _dataset_root()
    assert root is not None
    output_root = tmp_path_factory.mktemp("pads_rerun")
    record, _ = audit_pads_p01(dataset_root=root, output_root=output_root)
    return record


def test_the_real_release_reproduces_the_published_evidence_hash(
    rerun: dict,
) -> None:
    published = json.loads(REPORT.read_bytes())
    assert rerun["canonical_evidence_sha256"] == (
        published["canonical_evidence_sha256"]
    )
    assert rerun["source_manifest_sha256"] == (
        published["source_manifest_sha256"]
    )


def test_the_real_release_still_reconciles(rerun: dict) -> None:
    structure = rerun["release_structure"]
    assert structure["release_structure_status"] == (
        "PADS_RELEASE_STRUCTURE_RECONCILED"
    )
    assert structure["failure_count"] == 0
    assert rerun["streams"]["refused"] == 0


def test_only_reproduction_is_unmet_by_a_single_execution(
    rerun: dict,
) -> None:
    # One execution cannot reproduce itself, so this run closes on exactly one
    # condition and nothing else.
    assert failing_conditions(rerun) == (
        "PADS_INDEPENDENT_REPRODUCTION_VERIFIED",
    )
    assert rerun["gate_status"] != GATE_PASS
