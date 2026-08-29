"""Run the PADS-P0.4 audit as two genuinely separate executions.

Two child processes derive every rate over the P0.3 workload and run the
source-versus-replay comparison into two empty output roots.  The second is handed the first's run
receipt, so the reproduction condition is decided by receipts this driver did
not author.  Run A closes on reproduction by construction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from motionbloom.tremora_store.pads.p04.audit import (
    EVIDENCE_FILENAME,
    EXIT_ERROR,
    RECEIPT_FILENAME,
)
from motionbloom.tremora_store.pads.p04.contract import SUCCESS_MARKER
from motionbloom.tremora_store.pads.p04.gate import failing_conditions
from motionbloom.tremora_store.release_gate import canonical_json_bytes

MODULE = "motionbloom.tremora_store.pads.p04.audit"


def _run(
    *,
    release_root: Path,
    store_root: Path,
    output_root: Path,
    p03_root: Path,
    dependency: Path,
    p03_report: Path,
    reproduction_receipt: Path | None,
    progress: bool,
) -> int:
    arguments = [
        sys.executable, "-m", MODULE,
        "--release-root", str(release_root),
        "--store-root", str(store_root),
        "--p03-root", str(p03_root),
        "--output-root", str(output_root),
        "--dependency", str(dependency),
        "--p03-report", str(p03_report),
    ]
    if reproduction_receipt is not None:
        arguments += ["--reproduction-receipt", str(reproduction_receipt)]
    if progress:
        arguments.append("--progress")
    return subprocess.run(
        arguments, check=False, stdout=subprocess.DEVNULL,
    ).returncode


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--p03-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dependency", required=True, type=Path)
    parser.add_argument("--p03-report", required=True, type=Path)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    run_a = args.output_root / "run_a"
    run_b = args.output_root / "run_b"
    for root in (run_a, run_b):
        if root.exists() and any(root.iterdir()):
            print(f"ERROR: {root} is not empty", file=sys.stderr)
            return EXIT_ERROR
        root.mkdir(parents=True, exist_ok=True)

    common = {
        "release_root": args.release_root,
        "store_root": args.store_root,
        "p03_root": args.p03_root,
        "dependency": args.dependency,
        "p03_report": args.p03_report,
        "progress": args.progress,
    }
    first = _run(output_root=run_a, reproduction_receipt=None, **common)
    receipt = run_a / RECEIPT_FILENAME
    if not receipt.is_file():
        print(
            f"ERROR: run A produced no receipt (exit {first})",
            file=sys.stderr,
        )
        return EXIT_ERROR
    second = _run(output_root=run_b, reproduction_receipt=receipt, **common)

    record = json.loads((run_b / EVIDENCE_FILENAME).read_bytes())
    first_record = json.loads((run_a / EVIDENCE_FILENAME).read_bytes())
    summary = {
        "run_a_exit_code": first,
        "run_b_exit_code": second,
        "run_a_canonical_hash": first_record.get("canonical_evidence_sha256"),
        "run_b_canonical_hash": record.get("canonical_evidence_sha256"),
        "independent_reproduction_status": record.get(
            "independent_reproduction_status"
        ),
        "gate_status": record.get("gate_status"),
        "failing_conditions": list(failing_conditions(record)),
        "success_marker_present": (run_b / SUCCESS_MARKER).exists(),
        "generic_success_marker_present": (run_b / "_SUCCESS").exists(),
    }
    (args.output_root / "pads_p04_release_summary.json").write_bytes(
        canonical_json_bytes(summary)
    )
    sys.stdout.buffer.write(canonical_json_bytes(summary))
    return second


if __name__ == "__main__":
    raise SystemExit(main())
