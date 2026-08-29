"""Verify the materialized store by reading it back from disk.

The materializer checks what it holds in memory; this pass checks what it
actually wrote.  For every stream it reads the stored row group once and

* rebuilds the source text from the stored columns and compares its SHA-256
  against the source asset's own -- byte-exact replay for the whole corpus,
  not a frozen subset;
* replays every window of that stream and confirms the rows returned are
  exactly the rows the window index names.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..authority import CANONICAL_CHANNELS
from .replay import replay_stream, replay_window, source_bytes_for


@dataclass(slots=True)
class ReplayVerification:
    """What reading the store back proved."""

    streams_checked: int = 0
    streams_byte_exact: int = 0
    samples_replayed: int = 0
    streams_failed: int = 0
    windows_checked: int = 0
    window_replay_failures: int = 0
    source_time_token_failures: int = 0
    storage_index_content_sha256: str = ""
    failures: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "streams_checked": self.streams_checked,
            "streams_byte_exact": self.streams_byte_exact,
            # The third term of the headline reconciliation: the samples the
            # store actually handed back, not the samples it was told to hold.
            "samples_replayed": self.samples_replayed,
            "streams_failed": self.streams_failed,
            "windows_checked": self.windows_checked,
            "window_replay_failures": self.window_replay_failures,
            "source_time_token_failures": self.source_time_token_failures,
            "storage_index_content_sha256": (
                self.storage_index_content_sha256
            ),
            "failure_count": len(self.failures),
            "failures": sorted(self.failures)[:64],
        }


def storage_index_content_sha256(
    storage_index: Mapping[str, Mapping[str, Any]],
) -> str:
    """One hash binding every stored stream's content hash.

    Compact enough to publish, and as binding as listing all 10,318 of them.
    """

    digest = hashlib.sha256()
    for stream_id in sorted(storage_index):
        entry = storage_index[stream_id]
        digest.update(stream_id.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(entry["row_group_content_sha256"]).encode("ascii"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def verify_stored_replay(
    *,
    output_root: Path,
    storage_index: Mapping[str, Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    source_sha256_by_stream: Mapping[str, str],
    channel_order: Sequence[str] = CANONICAL_CHANNELS,
) -> ReplayVerification:
    """Read every stored stream back and prove what it replays."""

    result = ReplayVerification()
    result.storage_index_content_sha256 = storage_index_content_sha256(
        storage_index
    )
    windows_by_stream: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for window in windows:
        windows_by_stream[str(window["stream_id"])].append(window)

    for stream_id in sorted(storage_index):
        result.streams_checked += 1
        try:
            table = replay_stream(output_root, storage_index, stream_id)
        except Exception as exc:  # noqa: BLE001 - a read failure is evidence
            result.streams_failed += 1
            result.failures.append(f"{stream_id}: {exc}")
            continue

        result.samples_replayed += table.num_rows
        rebuilt = source_bytes_for(table, channel_order)
        expected = source_sha256_by_stream.get(stream_id)
        if expected is not None and hashlib.sha256(
            rebuilt
        ).hexdigest() == expected:
            result.streams_byte_exact += 1
        else:
            result.streams_failed += 1
            result.failures.append(f"{stream_id}: replay is not byte exact")

        tokens = table.column("source_time_token").to_pylist()
        picoseconds = table.column("source_time_ps").to_pylist()
        from .exact_time import exact_picoseconds

        for token, value in zip(tokens, picoseconds, strict=True):
            if exact_picoseconds(token) != value:
                result.source_time_token_failures += 1
                result.failures.append(
                    f"{stream_id}: stored time disagrees with its token")
                break

        for window in windows_by_stream.get(stream_id, ()):
            result.windows_checked += 1
            try:
                replayed = replay_window(output_root, storage_index, window)
            except Exception as exc:  # noqa: BLE001 - counted as evidence
                result.window_replay_failures += 1
                result.failures.append(f"{window['window_id']}: {exc}")
                continue
            ordinals = replayed.column("sample_ordinal").to_pylist()
            if (
                not ordinals
                or ordinals[0] != window["first_sample_ordinal"]
                or ordinals[-1] != window["last_sample_ordinal"]
                or len(ordinals) != window["sample_count"]
                or replayed.column("source_time_ps")[0].as_py()
                != window["first_source_time_ps"]
                or replayed.column("source_time_ps")[-1].as_py()
                != window["last_source_time_ps"]
            ):
                result.window_replay_failures += 1
                result.failures.append(
                    f"{window['window_id']}: replayed rows do not match the "
                    "index")
    return result


__all__ = [
    "ReplayVerification",
    "storage_index_content_sha256",
    "verify_stored_replay",
]
