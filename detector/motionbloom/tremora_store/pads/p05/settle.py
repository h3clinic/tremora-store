"""Wait for the machine to return to the state run A started in.

Run A drove swap from nothing to 3.75 GB, mostly page cache from reading
3.2 GB of baselines over and over for three hours.  That is inherent to the
benchmark, so every back-to-back pair will face it.

Starting run B at 70% swap would not be a crash risk so much as a
methodological one: the two executions would have run on materially different
machines, and their timings would not be comparable as replicates.  So the
second run waits until the machine looks like it did when the first one
started, and the waiting is part of the protocol rather than something a
person remembers to do.

Nothing here forces the issue.  No purge, no cache dropping -- those would
substitute one artificial state for another.  It waits, and if the machine
does not recover it refuses rather than lowering the bar.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .memory import MEMORY_OK, check_memory, sample_memory

SETTLED = "SETTLED"
SETTLE_TIMEOUT = "SETTLE_TIMEOUT"

#: Absolute floors, independent of what run A happened to start with.
MAX_SWAP_USED_BYTES = 512 * 1024**2
MIN_FREE_PERCENTAGE = 60.0
MIN_DISK_FREE_BYTES = 3.5 * 1024**3

#: And relative to the first run's start, so the two executions are
#: comparable rather than merely both survivable.
MAX_FREE_PERCENTAGE_DEFICIT = 10.0
MAX_SWAP_EXCESS_BYTES = 512 * 1024**2

#: Consecutive healthy samples required, and how far apart they are taken.
#: Six samples at thirty seconds covers the five minutes of zero swap growth
#: the protocol asks for.
REQUIRED_HEALTHY_SAMPLES = 6
SAMPLE_INTERVAL_SECONDS = 30.0

#: Give up after this long rather than wait forever.
MAX_WAIT_SECONDS = 4 * 60 * 60


@dataclass(slots=True)
class SettleReport:
    """Whether the machine came back, and what it looked like while waiting."""

    status: str = SETTLED
    detail: str = ""
    waited_seconds: float = 0.0
    samples_taken: int = 0
    consecutive_healthy: int = 0
    required_healthy: int = REQUIRED_HEALTHY_SAMPLES
    reference: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    final_sample: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == SETTLED

    def as_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "waited_seconds": round(self.waited_seconds, 1),
            "samples_taken": self.samples_taken,
            "consecutive_healthy": self.consecutive_healthy,
            "required_healthy_samples": self.required_healthy,
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "reference_run_start": dict(self.reference),
            "thresholds": dict(self.thresholds),
            "final_sample": dict(self.final_sample),
            "history": list(self.history),
        }


def _thresholds(reference: dict[str, Any] | None) -> dict[str, Any]:
    """The absolute floors, tightened by what the first run started with."""

    swap_cap = MAX_SWAP_USED_BYTES
    free_floor = MIN_FREE_PERCENTAGE
    if reference:
        reference_swap = float(reference.get("swap_used_bytes") or 0.0)
        reference_free = float(reference.get("free_percentage") or 0.0)
        # Never looser than the absolute floors; tighter when run A started
        # on a cleaner machine than the floors would demand.
        swap_cap = min(swap_cap, reference_swap + MAX_SWAP_EXCESS_BYTES)
        if reference_free > 0:
            free_floor = max(
                free_floor, reference_free - MAX_FREE_PERCENTAGE_DEFICIT
            )
    return {
        "max_swap_used_bytes": int(swap_cap),
        "min_free_percentage": free_floor,
        "min_disk_free_bytes": int(MIN_DISK_FREE_BYTES),
        "max_free_percentage_deficit": MAX_FREE_PERCENTAGE_DEFICIT,
        "max_swap_excess_bytes": MAX_SWAP_EXCESS_BYTES,
    }


def _healthy(
    sample: dict[str, Any],
    thresholds: dict[str, Any],
    disk_free_bytes: int,
) -> tuple[bool, str]:
    swap_used = int(sample["swap_used_bytes"])
    free_percentage = float(sample["free_percentage"])
    if swap_used > thresholds["max_swap_used_bytes"]:
        return False, (
            f"swap {swap_used / 1024**2:.0f} MB above "
            f"{thresholds['max_swap_used_bytes'] / 1024**2:.0f} MB"
        )
    if 0 <= free_percentage < thresholds["min_free_percentage"]:
        return False, (
            f"{free_percentage:.0f}% free below "
            f"{thresholds['min_free_percentage']:.0f}%"
        )
    if disk_free_bytes < thresholds["min_disk_free_bytes"]:
        return False, (
            f"{disk_free_bytes / 1024**3:.2f} GB disk below "
            f"{thresholds['min_disk_free_bytes'] / 1024**3:.2f} GB"
        )
    return True, "healthy"


def settle_between_runs(
    *,
    reference: dict[str, Any] | None = None,
    disk_free: Any,
    required_healthy: int = REQUIRED_HEALTHY_SAMPLES,
    interval: float = SAMPLE_INTERVAL_SECONDS,
    max_wait: float = MAX_WAIT_SECONDS,
    sleep=time.sleep,
    clock=time.monotonic,
    progress: bool = False,
) -> SettleReport:
    """Poll until the machine matches run A's starting conditions.

    ``reference`` is run A's ``memory_at_start`` record; ``disk_free`` is a
    callable returning free bytes on the output volume.  Requires a run of
    consecutive healthy samples, so a machine that dips healthy for one
    reading while still draining does not qualify.
    """

    thresholds = _thresholds(reference)
    report = SettleReport(
        reference=dict(reference or {}),
        thresholds=thresholds,
        required_healthy=required_healthy,
    )
    started = clock()
    streak = 0
    previous_swap: int | None = None

    while True:
        sample = sample_memory(count=1)[0]
        free_bytes = int(disk_free())
        healthy, why = _healthy(sample, thresholds, free_bytes)
        # Still climbing counts as unhealthy even when the level is fine.
        if (
            healthy
            and previous_swap is not None
            and int(sample["swap_used_bytes"]) > previous_swap
        ):
            healthy, why = False, "swap still growing"
        previous_swap = int(sample["swap_used_bytes"])

        streak = streak + 1 if healthy else 0
        entry = {
            "elapsed_seconds": round(clock() - started, 1),
            "swap_used_bytes": int(sample["swap_used_bytes"]),
            "free_percentage": float(sample["free_percentage"]),
            "disk_free_bytes": free_bytes,
            "healthy": healthy,
            "reason": why,
            "consecutive_healthy": streak,
        }
        report.history.append(entry)
        report.samples_taken += 1
        report.consecutive_healthy = streak
        report.final_sample = entry
        if progress:
            print(
                f"  settle {report.samples_taken}: {why}"
                f" ({streak}/{required_healthy})",
                flush=True,
            )

        if streak >= required_healthy:
            report.waited_seconds = clock() - started
            report.detail = (
                f"{streak} consecutive healthy samples after "
                f"{report.waited_seconds / 60:.1f} minutes"
            )
            return report

        if clock() - started >= max_wait:
            report.status = SETTLE_TIMEOUT
            report.waited_seconds = clock() - started
            report.detail = (
                f"still {why} after {report.waited_seconds / 60:.0f} "
                f"minutes; run B was not started"
            )
            return report

        sleep(interval)


def memory_at_start() -> dict[str, Any]:
    """The reference a run records for whatever runs after it."""

    report = check_memory()
    record = report.as_record()
    record["healthy_at_start"] = report.status == MEMORY_OK
    return record


__all__ = [
    "MAX_FREE_PERCENTAGE_DEFICIT",
    "MAX_SWAP_EXCESS_BYTES",
    "MAX_SWAP_USED_BYTES",
    "MAX_WAIT_SECONDS",
    "MIN_DISK_FREE_BYTES",
    "MIN_FREE_PERCENTAGE",
    "REQUIRED_HEALTHY_SAMPLES",
    "SAMPLE_INTERVAL_SECONDS",
    "SETTLED",
    "SETTLE_TIMEOUT",
    "SettleReport",
    "memory_at_start",
    "settle_between_runs",
]
