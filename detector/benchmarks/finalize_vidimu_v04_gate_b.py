"""Finalize one clean-root v0.4 VIDIMU Gate-B production run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motionbloom.tremora_store.v04.finalization import (
    GateBModelInputs,
    finalize_vidimu_v04_gate_b,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize all 208 recordings in a verified v0.4 source snapshot."
    )
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--model-weights", required=True, type=Path)
    parser.add_argument("--preprocessing-config", required=True, type=Path)
    parser.add_argument("--runtime-lock", required=True, type=Path)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--vendored-model-inventory", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    result = finalize_vidimu_v04_gate_b(
        args.source_snapshot,
        args.output_root,
        model_inputs=GateBModelInputs(
            manifest_path=args.model_manifest,
            weights_path=args.model_weights,
            preprocessing_config_path=args.preprocessing_config,
            runtime_lock_path=args.runtime_lock,
            expected_manifest_sha256=args.model_manifest_sha256,
            vendored_model_inventory_path=args.vendored_model_inventory,
        ),
    )
    print(json.dumps({
        "path": str(result.path),
        "run_id": result.run_id,
        "source_snapshot_sha256": result.source_snapshot_sha256,
        "model_manifest_sha256": result.model_manifest_sha256,
        "recording_count": result.recording_count,
        "run_manifest_sha256": result.run_manifest_sha256,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
