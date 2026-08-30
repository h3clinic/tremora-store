"""Run the PADS-P0.5 benchmark as two genuinely separate executions.

Baselines are built once, before either run, and both runs then read the same
verified immutable copies.  An earlier version had each run build its own,
which wrote 3.2 GB twice and filled the volume mid-benchmark.

Two child processes reconcile the four representations, account for what each
costs on disk, and run the frozen workload into two empty output roots.  The
second is handed the first's run receipt, so the reproduction condition is
decided by receipts this driver did not author.  Run A closes on reproduction
by construction.

The processes run one after another, never together: two benchmarks sharing a
machine would contend for the same disk and cores and each would be measuring
the other.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from motionbloom.tremora_store.pads.p05.audit import (
    ERROR_RESOURCE_PREFLIGHT,
    EVIDENCE_FILENAME,
    EXIT_ERROR,
    EXIT_PREFLIGHT,
    RECEIPT_FILENAME,
)
from motionbloom.tremora_store.pads.p05.build import build_all
from motionbloom.tremora_store.pads.p05.contract import SUCCESS_MARKER
from motionbloom.tremora_store.pads.p05.gate import failing_conditions
from motionbloom.tremora_store.release_gate import canonical_json_bytes

MODULE = "motionbloom.tremora_store.pads.p05.audit"


def _run(
    *,
    release_root: Path,
    store_root: Path,
    baseline_root: Path,
    output_root: Path,
    p02_report: Path,
    p03_report: Path,
    p04_report: Path,
    p04_store_root: Path | None,
    reproduction_receipt: Path | None,
    progress: bool,
) -> int:
    arguments = [
        sys.executable, "-m", MODULE,
        "--release-root", str(release_root),
        "--store-root", str(store_root),
        "--baseline-root", str(baseline_root),
        "--output-root", str(output_root),
        "--p02-report", str(p02_report),
        "--p03-report", str(p03_report),
        "--p04-report", str(p04_report),
    ]
    if p04_store_root is not None:
        arguments += ["--p04-store-root", str(p04_store_root)]
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
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--p02-report", required=True, type=Path)
    parser.add_argument("--p03-report", required=True, type=Path)
    parser.add_argument("--p04-report", required=True, type=Path)
    parser.add_argument("--p04-store-root", type=Path)
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
        "baseline_root": args.baseline_root,
        "p02_report": args.p02_report,
        "p03_report": args.p03_report,
        "p04_report": args.p04_report,
        "p04_store_root": args.p04_store_root,
        "progress": args.progress,
    }
    # A builds the baselines; B reuses them, exactly as both reuse the P0.2.1
    # store.  They are deterministic inputs, not part of what is measured.
    # Built once, here, before either run.  Both runs then read the same
    # verified immutable baselines; neither writes a second copy of them.
    if not (args.baseline_root / "baseline_identities.json").is_file():
        build_all(
            store_root=args.store_root,
            baseline_root=args.baseline_root,
            progress=args.progress,
        )

    first = _run(output_root=run_a, reproduction_receipt=None, **common)
    if first == EXIT_PREFLIGHT:
        record = json.loads((run_a / EVIDENCE_FILENAME).read_bytes())
        payload = canonical_json_bytes({
            "release_status": ERROR_RESOURCE_PREFLIGHT,
            "gate_status": None,
            "preflight": record.get("preflight"),
            "blocked_reason": record.get("blocked_reason"),
        })
        (args.output_root / "pads_p05_release_summary.json").write_bytes(
            payload
        )
        sys.stdout.buffer.write(payload)
        return EXIT_PREFLIGHT
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

    def q2_p50(payload: dict) -> dict[str, float]:
        summary = payload.get("measured_performance", {}).get(
            "latency_summary", {}
        )
        return {
            name: classes.get("Q2_SINGLE_WINDOW", {}).get(
                "p50_latency_ns", 0.0
            )
            for name, classes in sorted(summary.items())
        }

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
        # Published so the reader can see how far the two runs' timings
        # drifted; they are not part of the evidence hash and not gated.
        "run_a_q2_p50_latency_ns": q2_p50(first_record),
        "run_b_q2_p50_latency_ns": q2_p50(record),
    }
    (args.output_root / "pads_p05_release_summary.json").write_bytes(
        canonical_json_bytes(summary)
    )
    sys.stdout.buffer.write(canonical_json_bytes(summary))
    return second


if __name__ == "__main__":
    raise SystemExit(main())
