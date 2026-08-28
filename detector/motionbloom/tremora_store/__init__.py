"""Timestamp-native storage and deterministic replay for video–IMU analytics."""

from .alignment_index import AlignmentError, build_frame_imu_index
from .clock_map import ClockMapError, ClockSegment, PiecewiseClockMap
from .parquet_writer import RecordingStoreWriter, SnapshotError, verify_snapshot
from .replay import RecordingStore, ReplayedWindow
from .window_index import (
    ContinuitySegment,
    WindowIndexError,
    WindowIndexResult,
    build_window_index,
)

__all__ = [
    "AlignmentError",
    "ClockMapError",
    "ClockSegment",
    "ContinuitySegment",
    "PiecewiseClockMap",
    "RecordingStore",
    "RecordingStoreWriter",
    "ReplayedWindow",
    "SnapshotError",
    "WindowIndexError",
    "WindowIndexResult",
    "build_frame_imu_index",
    "build_window_index",
    "verify_snapshot",
]

__version__ = "0.1.0"
