"""Release-input regression: re-derive the published PADS-P0.2 evidence.

Skipped when the release is not present.  Point ``TREMORA_PADS_DATASET_ROOT``
at an extracted PADS 1.0.0 root to run it.

This one is expensive: it materializes the whole store again, roughly a
gigabyte of Parquet into pytest's temporary directory, and takes several
minutes.  It is the strongest regression available -- if the implementation
drifts from the evidence it publishes, the hash moves.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from motionbloom.tremora_store.pads.p02.audit import audit_pads_p02
from motionbloom.tremora_store.pads.p02.contract import GATE_PASS
from motionbloom.tremora_store.pads.p02.gate import failing_conditions

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO_ROOT / "detector/benchmarks"
REPORT = BENCHMARKS / "pads_p02_release_audit.json"
DEPENDENCY = BENCHMARKS / "pads_p01_dependency.json"
P01_REPORT = BENCHMARKS / "pads_p01_release_audit.json"
ROOT_VARIABLE = "TREMORA_PADS_DATASET_ROOT"


def _release_root() -> Path | None:
    value = os.environ.get(ROOT_VARIABLE)
    if not value:
        return None
    root = Path(value)
    return root if (root / "movement").is_dir() else None


pytestmark = pytest.mark.skipif(
    _release_root() is None or not REPORT.is_file(),
    reason=(
        f"PADS_RELEASE_INPUT_UNAVAILABLE: set {ROOT_VARIABLE} to an extracted "
        "PADS 1.0.0 root"
    ),
)


@pytest.fixture(scope="module")
def rerun(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = _release_root()
    assert root is not None
    record, _ = audit_pads_p02(
        release_root=root,
        output_root=tmp_path_factory.mktemp("pads_p02_rerun"),
        dependency_path=DEPENDENCY,
        p01_report_path=P01_REPORT,
    )
    return record


def test_the_real_release_reproduces_the_published_evidence_hash(
    rerun: dict,
) -> None:
    published = json.loads(REPORT.read_bytes())
    assert rerun["canonical_evidence_sha256"] == (
        published["canonical_evidence_sha256"]
    )
    assert rerun["materialization"] == published["materialization"]
    assert rerun["replay_verification"] == published["replay_verification"]


def test_only_reproduction_is_unmet_by_a_single_materialization(
    rerun: dict,
) -> None:
    assert failing_conditions(rerun) == (
        "INDEPENDENT_MATERIALIZATION_REPRODUCED",
    )
    assert rerun["gate_status"] != GATE_PASS
