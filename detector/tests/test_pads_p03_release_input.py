"""Release-input regression: re-derive the published PADS-P0.3 evidence.

Skipped unless both the release and the materialized P0.2.1 store are present.
Point ``TREMORA_PADS_DATASET_ROOT`` at an extracted PADS 1.0.0 root and
``TREMORA_PADS_P02_STORE`` at the store the published P0.2.1 report describes.

The run recomputes 9,960 workload spectra and 6,077 independent source-versus-
replay comparisons and takes several minutes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from motionbloom.tremora_store.pads.p03.audit import audit_pads_p03
from motionbloom.tremora_store.pads.p03.contract import GATE_PASS
from motionbloom.tremora_store.pads.p03.gate import failing_conditions

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO_ROOT / "detector/benchmarks"
REPORT = BENCHMARKS / "pads_p03_release_audit.json"
DEPENDENCY = BENCHMARKS / "pads_p02_dependency.json"
P02_REPORT = BENCHMARKS / "pads_p02_release_audit.json"
RELEASE_VARIABLE = "TREMORA_PADS_DATASET_ROOT"
STORE_VARIABLE = "TREMORA_PADS_P02_STORE"


def _roots() -> tuple[Path, Path] | None:
    release = os.environ.get(RELEASE_VARIABLE)
    store = os.environ.get(STORE_VARIABLE)
    if not release or not store:
        return None
    release_root, store_root = Path(release), Path(store)
    if not (release_root / "movement").is_dir():
        return None
    if not (store_root / "pads_stream_storage_index.parquet").is_file():
        return None
    return release_root, store_root


pytestmark = pytest.mark.skipif(
    _roots() is None or not REPORT.is_file(),
    reason=(
        f"PADS_RELEASE_INPUT_UNAVAILABLE: set {RELEASE_VARIABLE} and "
        f"{STORE_VARIABLE}"
    ),
)


@pytest.fixture(scope="module")
def rerun(tmp_path_factory: pytest.TempPathFactory) -> dict:
    roots = _roots()
    assert roots is not None
    release_root, store_root = roots
    record, _ = audit_pads_p03(
        release_root=release_root,
        store_root=store_root,
        output_root=tmp_path_factory.mktemp("pads_p03_rerun"),
        dependency_path=DEPENDENCY,
        p02_report_path=P02_REPORT,
    )
    return record


def test_the_real_inputs_reproduce_the_published_evidence_hash(
    rerun: dict,
) -> None:
    published = json.loads(REPORT.read_bytes())
    assert rerun["canonical_evidence_sha256"] == (
        published["canonical_evidence_sha256"]
    )
    assert rerun["materialization"] == published["materialization"]
    assert rerun["kernel_controls"] == published["kernel_controls"]


def test_source_and_replay_still_agree_bit_for_bit(rerun: dict) -> None:
    materialization = rerun["materialization"]
    assert materialization["maximum_observed_bin_error"] == 0.0
    assert materialization["source_replay_spectral_hash_mismatches"] == 0
    assert materialization["source_replay_row_mismatches"] == 0


def test_only_reproduction_is_unmet_by_a_single_execution(
    rerun: dict,
) -> None:
    assert failing_conditions(rerun) == (
        "INDEPENDENT_MATERIALIZATION_REPRODUCED",
    )
    assert rerun["gate_status"] != GATE_PASS
