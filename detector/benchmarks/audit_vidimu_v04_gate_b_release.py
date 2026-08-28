"""Strictly audit two independent v0.4 VIDIMU Gate-B finalization runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motionbloom.tremora_store.v04.audit import (
    FAIL,
    audit_vidimu_v04_gate_b_release,
    write_release_audit_report,
)
from motionbloom.tremora_store.v04.finalization import GateBModelInputs


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two clean-root source-to-CV Gate-B runs."
    )
    parser.add_argument("--primary-run", required=True, type=Path)
    parser.add_argument("--replay-run", required=True, type=Path)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--source-inventory-sha256", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--model-weights", required=True, type=Path)
    parser.add_argument("--preprocessing-config", required=True, type=Path)
    parser.add_argument("--runtime-lock", required=True, type=Path)
    parser.add_argument("--vendored-model-inventory", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    report = audit_vidimu_v04_gate_b_release(
        args.primary_run,
        args.replay_run,
        source_snapshot_path=args.source_snapshot,
        expected_source_snapshot_sha256=args.source_snapshot_sha256,
        expected_source_inventory_sha256=args.source_inventory_sha256,
        expected_model_manifest_sha256=args.model_manifest_sha256,
        model_inputs=GateBModelInputs(
            manifest_path=args.model_manifest,
            weights_path=args.model_weights,
            preprocessing_config_path=args.preprocessing_config,
            runtime_lock_path=args.runtime_lock,
            expected_manifest_sha256=args.model_manifest_sha256,
            vendored_model_inventory_path=args.vendored_model_inventory,
        ),
    )
    if args.report is not None:
        write_release_audit_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["status"] == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
