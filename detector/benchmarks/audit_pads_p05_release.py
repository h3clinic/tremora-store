"""Run the PADS-P0.5 benchmark as four genuinely separate processes.

Two measurement processes and two summarizer processes, in that alternating
order.  Splitting them is not tidiness: the first authoritative attempt
finished its three hours of timing and was then killed by the memory manager
while reading the table back, because the read-back's allocation landed on top
of four open representations in the same process.  A measurement process now
closes its table, hashes it, and exits; a fresh process does the summarizing.

Baselines are built once, before any of it, and every run verifies them by
hash rather than rebuilding them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbloom.tremora_store.pads.p05.audit import (
    ERROR_RESOURCE_PREFLIGHT,
    EVIDENCE_FILENAME,
    EXIT_ERROR,
    EXIT_NO_GO,
    EXIT_PASS,
    EXIT_PREFLIGHT,
    MEASUREMENT_FILENAME,
    RECEIPT_FILENAME,
)
from motionbloom.tremora_store.pads.p05.build import (
    build_all,
    describe_existing,
)
from motionbloom.tremora_store.release_gate import (
    canonical_json_bytes,
)

SUMMARY_FILENAME = "pads_p05_release_summary.json"


def _spawn(module: str, arguments: list[str]) -> int:
    """Run one phase in its own interpreter and wait for it to exit."""

    completed = subprocess.run(
        [sys.executable, "-u", "-m", module, *arguments],
        cwd=str(Path(__file__).resolve().parents[1]),
        check=False,
    )
    return completed.returncode


def _measure(*, output_root: Path, process_id: int, args) -> int:
    return _spawn(
        "motionbloom.tremora_store.pads.p05.measure",
        [
            "--release-root", str(args.release_root),
            "--store-root", str(args.store_root),
            "--baseline-root", str(args.baseline_root),
            "--output-root", str(output_root),
            "--p02-report", str(args.p02_report),
            "--p03-report", str(args.p03_report),
            "--p04-report", str(args.p04_report),
            *(["--p04-store-root", str(args.p04_store_root)]
              if args.p04_store_root else []),
            "--rounds", str(args.rounds),
            "--run-id", f"pads-p05-{process_id}",
            "--process-id", str(process_id),
            *(["--progress"] if args.progress else []),
        ],
    )


def _summarize(
    *, output_root: Path, args, reproduction_receipt: Path | None
) -> int:
    return _spawn(
        "motionbloom.tremora_store.pads.p05.summarize",
        [
            "--output-root", str(output_root),
            "--store-root", str(args.store_root),
            *(["--reproduction-receipt", str(reproduction_receipt)]
              if reproduction_receipt else []),
            *(["--progress"] if args.progress else []),
        ],
    )


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--p02-report", required=True, type=Path)
    parser.add_argument("--p03-report", required=True, type=Path)
    parser.add_argument("--p04-report", required=True, type=Path)
    parser.add_argument("--p04-store-root", type=Path)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def _refused(output_root: Path, run_root: Path, phase: str) -> int:
    record = json.loads((run_root / EVIDENCE_FILENAME).read_bytes())
    payload = canonical_json_bytes({
        "release_status": ERROR_RESOURCE_PREFLIGHT,
        "gate_status": None,
        "refused_phase": phase,
        "preflight": record.get("preflight"),
        "memory": record.get("memory"),
        "blocked_reason": record.get("blocked_reason"),
    })
    (output_root / SUMMARY_FILENAME).write_bytes(payload)
    sys.stdout.buffer.write(payload)
    return EXIT_PREFLIGHT


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_a = args.output_root / "run_a"
    run_b = args.output_root / "run_b"
    for root in (run_a, run_b):
        root.mkdir(parents=True, exist_ok=True)

    # Built once, before anything measures.  Both runs then read the same
    # verified immutable baselines; neither writes a second copy of them.
    identities = args.baseline_root / "baseline_identities.json"
    if not identities.is_file():
        if (args.baseline_root / "b1").is_dir():
            describe_existing(
                baseline_root=args.baseline_root, store_root=args.store_root
            )
        else:
            build_all(
                store_root=args.store_root,
                baseline_root=args.baseline_root,
                progress=args.progress,
            )

    # measurement A -> summarizer A -> measurement B -> summarizer B
    first = _measure(output_root=run_a, process_id=1, args=args)
    if first == EXIT_PREFLIGHT:
        return _refused(args.output_root, run_a, "measurement_a")
    if first != EXIT_PASS:
        print(f"ERROR: measurement A exited {first}", file=sys.stderr)
        return EXIT_ERROR

    first_summary = _summarize(
        output_root=run_a, args=args, reproduction_receipt=None
    )
    if first_summary not in (EXIT_PASS, EXIT_NO_GO):
        print(
            f"ERROR: summarizer A exited {first_summary}", file=sys.stderr
        )
        return EXIT_ERROR
    receipt = run_a / RECEIPT_FILENAME
    if not receipt.is_file():
        print("ERROR: summarizer A produced no receipt", file=sys.stderr)
        return EXIT_ERROR

    second = _measure(output_root=run_b, process_id=2, args=args)
    if second == EXIT_PREFLIGHT:
        return _refused(args.output_root, run_b, "measurement_b")
    if second != EXIT_PASS:
        print(f"ERROR: measurement B exited {second}", file=sys.stderr)
        return EXIT_ERROR

    second_summary = _summarize(
        output_root=run_b, args=args, reproduction_receipt=receipt
    )
    if second_summary not in (EXIT_PASS, EXIT_NO_GO):
        print(
            f"ERROR: summarizer B exited {second_summary}", file=sys.stderr
        )
        return EXIT_ERROR

    record_a = json.loads((run_a / EVIDENCE_FILENAME).read_bytes())
    record_b = json.loads((run_b / EVIDENCE_FILENAME).read_bytes())
    measure_a = json.loads((run_a / MEASUREMENT_FILENAME).read_bytes())
    measure_b = json.loads((run_b / MEASUREMENT_FILENAME).read_bytes())
    agree = (
        record_a["canonical_evidence_sha256"]
        == record_b["canonical_evidence_sha256"]
    )
    payload = canonical_json_bytes({
        "release_status": record_b.get("release_status"),
        "gate_status": record_b.get("gate_status"),
        "gate_satisfied": record_b.get("gate_satisfied"),
        "gate_conditions_satisfied": record_b.get(
            "gate_conditions_satisfied"
        ),
        "gate_conditions_total": record_b.get("gate_conditions_total"),
        "independent_reproduction_status": record_b.get(
            "independent_reproduction_status"
        ),
        "deterministic_evidence_identical": agree,
        "canonical_evidence_sha256": {
            "run_a": record_a["canonical_evidence_sha256"],
            "run_b": record_b["canonical_evidence_sha256"],
        },
        # Provenance, not experimental content.
        "timing_tables": {
            "run_a": measure_a["timing_table"],
            "run_b": measure_b["timing_table"],
        },
        "execution_receipts": {
            "run_a": measure_a["execution_receipt"],
            "run_b": measure_b["execution_receipt"],
        },
    })
    (args.output_root / SUMMARY_FILENAME).write_bytes(payload)
    sys.stdout.buffer.write(payload)
    sys.stdout.write("\n")
    if not agree:
        return EXIT_NO_GO
    return EXIT_PASS if record_b.get("gate_satisfied") else EXIT_NO_GO


if __name__ == "__main__":
    raise SystemExit(main())
