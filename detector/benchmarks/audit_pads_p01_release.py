"""Run the PADS-P0.1 release audit as two genuinely separate executions.

The driver invokes the audit twice, in two child processes, against two empty
output roots.  The second execution is handed the first's run receipt, so the
reproduction condition is decided by receipts the driver did not author rather
than by two report paths an operator supplied.

Run A closes its gate on reproduction by construction -- one execution cannot
reproduce itself.  The authoritative published record is run B's, and it can
only pass if B's evidence hash equals A's while their receipts disagree about
process, run id and output root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from motionbloom.tremora_store.pads.gate import failing_conditions
from motionbloom.tremora_store.release_gate import (
    EXIT_ERROR,
    canonical_json_bytes,
)

EVIDENCE_FILENAME = "pads_p01_evidence.json"
RECEIPT_FILENAME = "pads_p01_run_receipt.json"
MODULE = "motionbloom.tremora_store.pads.audit"


def _run(
    dataset_root: Path,
    output_root: Path,
    *,
    reproduction_receipt: Path | None,
    progress: bool,
) -> int:
    arguments = [
        sys.executable, "-m", MODULE,
        "--dataset-root", str(dataset_root),
        "--output-root", str(output_root),
    ]
    if reproduction_receipt is not None:
        arguments += ["--reproduction-receipt", str(reproduction_receipt)]
    if progress:
        arguments.append("--progress")
    completed = subprocess.run(
        arguments, check=False, stdout=subprocess.DEVNULL,
    )
    return completed.returncode


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
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

    first = _run(
        args.dataset_root, run_a,
        reproduction_receipt=None, progress=args.progress,
    )
    receipt = run_a / RECEIPT_FILENAME
    if not receipt.is_file():
        print(
            f"ERROR: run A produced no receipt (exit {first})",
            file=sys.stderr,
        )
        return EXIT_ERROR
    second = _run(
        args.dataset_root, run_b,
        reproduction_receipt=receipt, progress=args.progress,
    )

    record = json.loads((run_b / EVIDENCE_FILENAME).read_bytes())
    summary = {
        "run_a_exit_code": first,
        "run_b_exit_code": second,
        "run_a_evidence_sha256": json.loads(
            (run_a / EVIDENCE_FILENAME).read_bytes()
        ).get("canonical_evidence_sha256"),
        "run_b_evidence_sha256": record.get("canonical_evidence_sha256"),
        "gate_status": record.get("gate_status"),
        "failing_conditions": list(failing_conditions(record)),
    }
    (args.output_root / "pads_p01_release_summary.json").write_bytes(
        canonical_json_bytes(summary)
    )
    sys.stdout.buffer.write(canonical_json_bytes(summary))
    return second


if __name__ == "__main__":
    raise SystemExit(main())
