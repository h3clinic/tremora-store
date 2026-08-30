"""Whether the machine has the memory to finish a three-hour run.

Free disk was checked and the run was killed anyway.  The disk had not run
out: macOS grew swap under memory pressure, which consumed the disk as a side
effect, and then the memory manager killed the largest process -- which was
the benchmark, three hours into its timing.

Disk is therefore the wrong thing to check on its own.  What matters is
whether the machine is already paging, because a machine that is already
paging will page harder under a long run and eventually kill something.

The numbers this module reads are facts about the machine at a moment.  They
belong in execution receipts as provenance, never in the canonical evidence
hash: two honest runs on the same machine will disagree about them, and
should.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

MEMORY_OK = "MEMORY_OK"
SWAP_HEAVILY_USED = "SWAP_HEAVILY_USED"
SWAP_GROWING = "SWAP_GROWING"
MEMORY_PRESSURE_ELEVATED = "MEMORY_PRESSURE_ELEVATED"
MEMORY_UNREADABLE = "MEMORY_UNREADABLE"

#: Refuse above this share of configured swap in use.  Swap that is already
#: half full means the machine is paging before the run has allocated
#: anything.
MAX_SWAP_USED_FRACTION = 0.35

#: Refuse above this much swap in use outright, however large swap has grown.
#: macOS grows the swap file on demand, so a fraction alone can be satisfied
#: by a machine that simply kept growing it.
MAX_SWAP_USED_BYTES = 3 * 1024**3

#: Refuse below this free-memory percentage as reported by memory_pressure.
MIN_FREE_PERCENTAGE = 12.0

#: Refuse if swap grows by more than this between samples.
MAX_SWAP_GROWTH_BYTES = 128 * 1024**2

#: Samples taken to see whether swap is stable or climbing.
SAMPLE_COUNT = 3
SAMPLE_INTERVAL_SECONDS = 5.0


def _run(command: list[str]) -> str:
    try:
        finished = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return finished.stdout


_SIZE = re.compile(r"([0-9.]+)([KMGT]?)")
_SCALE = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def _bytes_from(token: str) -> int:
    match = _SIZE.match(token.strip())
    if not match:
        return 0
    return int(float(match.group(1)) * _SCALE.get(match.group(2), 1))


def swap_usage() -> dict[str, int]:
    """``total``/``used``/``free`` swap in bytes, or zeros if unreadable."""

    text = _run(["sysctl", "-n", "vm.swapusage"])
    values = dict(
        re.findall(r"(total|used|free)\s*=\s*([0-9.]+[KMGT]?)", text)
    )
    return {
        name: _bytes_from(values[name]) for name in ("total", "used", "free")
    } if len(values) == 3 else {"total": 0, "used": 0, "free": 0}


def free_percentage() -> float:
    """What ``memory_pressure`` reports as system-wide free memory."""

    text = _run(["memory_pressure"])
    match = re.search(
        r"System-wide memory free percentage:\s*([0-9.]+)", text
    )
    return float(match.group(1)) if match else -1.0


@dataclass(slots=True)
class MemoryReport:
    """Whether the machine is in a state to survive a long run."""

    status: str = MEMORY_OK
    detail: str = ""
    samples: list[dict[str, Any]] = field(default_factory=list)
    swap_total_bytes: int = 0
    swap_used_bytes: int = 0
    swap_growth_bytes: int = 0
    free_percentage: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == MEMORY_OK

    def as_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "swap_used_fraction": (
                self.swap_used_bytes / self.swap_total_bytes
                if self.swap_total_bytes else 0.0
            ),
            "swap_growth_bytes": self.swap_growth_bytes,
            "free_percentage": self.free_percentage,
            "max_swap_used_fraction": MAX_SWAP_USED_FRACTION,
            "max_swap_used_bytes": MAX_SWAP_USED_BYTES,
            "min_free_percentage": MIN_FREE_PERCENTAGE,
            "max_swap_growth_bytes": MAX_SWAP_GROWTH_BYTES,
            "samples": list(self.samples),
        }


def sample_memory(
    *,
    count: int = SAMPLE_COUNT,
    interval: float = SAMPLE_INTERVAL_SECONDS,
    sleep=time.sleep,
) -> list[dict[str, Any]]:
    """Take several readings, so a climbing machine is distinguishable."""

    samples = []
    for index in range(max(1, count)):
        if index:
            sleep(interval)
        swap = swap_usage()
        samples.append({
            "index": index,
            "swap_total_bytes": swap["total"],
            "swap_used_bytes": swap["used"],
            "free_percentage": free_percentage(),
        })
    return samples


def check_memory(samples: list[dict[str, Any]] | None = None) -> MemoryReport:
    """Refuse a long run on a machine that is already paging."""

    readings = samples if samples is not None else sample_memory()
    report = MemoryReport(samples=readings)
    if not readings:
        report.status = MEMORY_UNREADABLE
        report.detail = "no memory readings were taken"
        return report

    last = readings[-1]
    report.swap_total_bytes = int(last["swap_total_bytes"])
    report.swap_used_bytes = int(last["swap_used_bytes"])
    report.free_percentage = float(last["free_percentage"])
    report.swap_growth_bytes = max(
        0,
        int(last["swap_used_bytes"])
        - int(readings[0]["swap_used_bytes"]),
    )

    if report.swap_total_bytes <= 0 and report.free_percentage < 0:
        report.status = MEMORY_UNREADABLE
        report.detail = "neither swap nor memory pressure could be read"
        return report

    used_fraction = (
        report.swap_used_bytes / report.swap_total_bytes
        if report.swap_total_bytes else 0.0
    )
    if (
        report.swap_used_bytes > MAX_SWAP_USED_BYTES
        or used_fraction > MAX_SWAP_USED_FRACTION
    ):
        report.status = SWAP_HEAVILY_USED
        report.detail = (
            f"{report.swap_used_bytes / 1024**3:.2f} GB of "
            f"{report.swap_total_bytes / 1024**3:.2f} GB swap in use "
            f"({used_fraction:.0%}); the machine is already paging"
        )
        return report

    if report.swap_growth_bytes > MAX_SWAP_GROWTH_BYTES:
        report.status = SWAP_GROWING
        report.detail = (
            f"swap grew {report.swap_growth_bytes / 1024**2:.0f} MB across "
            f"{len(readings)} samples; it is still climbing"
        )
        return report

    if 0 <= report.free_percentage < MIN_FREE_PERCENTAGE:
        report.status = MEMORY_PRESSURE_ELEVATED
        report.detail = (
            f"{report.free_percentage:.0f}% memory free, below the "
            f"{MIN_FREE_PERCENTAGE:.0f}% a long run needs"
        )
        return report

    report.detail = (
        f"{report.swap_used_bytes / 1024**3:.2f} GB swap in use, stable "
        f"across {len(readings)} samples; {report.free_percentage:.0f}% free"
    )
    return report


__all__ = [
    "MAX_SWAP_GROWTH_BYTES",
    "MAX_SWAP_USED_BYTES",
    "MAX_SWAP_USED_FRACTION",
    "MEMORY_OK",
    "MEMORY_PRESSURE_ELEVATED",
    "MEMORY_UNREADABLE",
    "MIN_FREE_PERCENTAGE",
    "SWAP_GROWING",
    "SWAP_HEAVILY_USED",
    "MemoryReport",
    "check_memory",
    "free_percentage",
    "sample_memory",
    "swap_usage",
]
