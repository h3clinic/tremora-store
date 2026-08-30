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
import shutil
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
from motionbloom.tremora_store.pads.p05.settle import (
    SETTLE_TIMEOUT,
    settle_between_runs,
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


#: Headline numbers reported per run and compared across runs.  The
#: bootstrap interval inside one run measures query-to-query sampling
#: uncertainty only; it is not the benchmark's total uncertainty, and the
#: replicate difference between two executions is the larger term.
WITHIN_RUN_CI_NOTE = (
    "The bootstrap interval is query-to-query sampling uncertainty within a "
    "single execution. It is not the benchmark's total uncertainty. "
    "Run-to-run variation from OS scheduling and page cache is larger, and "
    "is reported here as the replicate difference between two independent "
    "executions. Do not pool the two runs' latency samples: they were "
    "measured on the same machine in materially different states."
)


def _headline(record: dict) -> dict:
    """One run's published numbers, per representation and query class."""

    performance = record.get("measured_performance", {})
    latency = performance.get("latency_summary", {})
    return {
        "median_latency_ns": {
            name: {
                query_class: summary["p50_latency_ns"]
                for query_class, summary in sorted(classes.items())
            }
            for name, classes in sorted(latency.items())
        },
        "speed_ratios": {
            entry["baseline"]: {
                "median_ratio": entry["median_ratio"],
                "within_run_ci_low": entry["confidence_low"],
                "within_run_ci_high": entry["confidence_high"],
                "queries": entry["queries"],
            }
            for entry in performance.get("speed_ratios", [])
        },
        "batch_throughput": performance.get("batch_throughput", {}),
    }


def _replicate_comparison(record_a: dict, record_b: dict) -> dict:
    """Each headline median in both runs, and the difference between them."""

    first = _headline(record_a)
    second = _headline(record_b)
    medians: dict[str, dict] = {}
    for name, classes in first["median_latency_ns"].items():
        for query_class, value_a in classes.items():
            value_b = (
                second["median_latency_ns"].get(name, {}).get(query_class)
            )
            if value_b is None:
                continue
            medians.setdefault(name, {})[query_class] = {
                "run_a_ns": value_a,
                "run_b_ns": value_b,
                "difference_ns": value_b - value_a,
                "percent_difference": (
                    100.0 * (value_b - value_a) / value_a
                    if value_a else 0.0
                ),
            }
    ratios: dict[str, dict] = {}
    for baseline, entry_a in first["speed_ratios"].items():
        entry_b = second["speed_ratios"].get(baseline)
        if entry_b is None:
            continue
        ratios[baseline] = {
            "run_a": entry_a,
            "run_b": entry_b,
            "percent_difference": (
                100.0
                * (entry_b["median_ratio"] - entry_a["median_ratio"])
                / entry_a["median_ratio"]
                if entry_a["median_ratio"] else 0.0
            ),
        }
    return {
        "note": WITHIN_RUN_CI_NOTE,
        "run_a": first,
        "run_b": second,
        "median_latency": medians,
        "speed_ratios": ratios,
        "pooling_permitted": False,
    }


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
    parsed = parser.parse_args(argv)
    # Every phase runs in its own interpreter with its own working directory,
    # so relative paths would resolve against the wrong root and read as
    # absent.  They are resolved once, here, against the caller's cwd.
    for name in (
        "release_root", "store_root", "baseline_root", "output_root",
        "p02_report", "p03_report", "p04_report", "p04_store_root",
    ):
        value = getattr(parsed, name)
        if value is not None:
            setattr(parsed, name, Path(value).resolve())
    return parsed


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

    # SETTLE_BETWEEN_RUNS.  Run A drives swap up by several gigabytes, mostly
    # page cache from reading 3.2 GB of baselines for three hours.  Starting B
    # on that machine would not be a crash risk so much as a methodological
    # one: the two executions would not be comparable as replicates.  So B
    # waits until the machine looks like it did when A started.  No purge, no
    # cache dropping -- waiting, or refusing.
    measure_a = json.loads((run_a / MEASUREMENT_FILENAME).read_bytes())
    reference = measure_a["execution_receipt"].get("memory_at_start", {})
    settle = settle_between_runs(
        reference=reference,
        disk_free=lambda: shutil.disk_usage(args.output_root).free,
        progress=args.progress,
    )
    (args.output_root / "pads_p05_settle.json").write_bytes(
        canonical_json_bytes(settle.as_record())
    )
    if not settle.ok:
        payload = canonical_json_bytes({
            "release_status": SETTLE_TIMEOUT,
            "gate_status": None,
            "refused_phase": "settle_between_runs",
            "blocked_reason": settle.detail,
            "settle": settle.as_record(),
            "run_a_complete": True,
        })
        (args.output_root / SUMMARY_FILENAME).write_bytes(payload)
        sys.stdout.buffer.write(payload)
        return EXIT_PREFLIGHT

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
        "settle_between_runs": settle.as_record(),
        "replicate_comparison": _replicate_comparison(record_a, record_b),
    })
    (args.output_root / SUMMARY_FILENAME).write_bytes(payload)
    sys.stdout.buffer.write(payload)
    sys.stdout.write("\n")
    if not agree:
        return EXIT_NO_GO
    return EXIT_PASS if record_b.get("gate_satisfied") else EXIT_NO_GO


if __name__ == "__main__":
    raise SystemExit(main())
