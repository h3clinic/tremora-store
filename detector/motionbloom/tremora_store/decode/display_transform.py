"""Explicit coded-pixel to display-pixel orientation transforms."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np


class DisplayTransformError(ValueError):
    """Raised for unsupported or invalid display transforms."""


@dataclass(frozen=True)
class DisplayTransform:
    rotation_degrees: int
    coded_width: int
    coded_height: int
    display_width: int
    display_height: int
    source_to_display: tuple[float, ...]

    @property
    def inverse(self) -> tuple[float, ...]:
        matrix = np.asarray(self.source_to_display, dtype=np.float64).reshape(3, 3)
        return tuple(float(value) for value in np.linalg.inv(matrix).reshape(-1))


def _right_angle(value: float) -> int:
    if not math.isfinite(value):
        raise DisplayTransformError("display rotation must be finite")
    normalized = value % 360.0
    nearest = int(round(normalized / 90.0) * 90) % 360
    if not math.isclose(normalized, float(nearest), abs_tol=1e-6):
        raise DisplayTransformError(
            "only lossless right-angle display rotations are supported")
    return nearest


def rotation_from_display_matrix(payload: bytes) -> int:
    """Read FFmpeg's 3x3 AVDisplayMatrix and return CCW degrees.

    The matrix contains nine native-endian signed int32 values. FFmpeg stores
    the affine terms in 16.16 fixed point. Reflection/skew matrices are rejected
    because the v0.3 contract only implements a pure right-angle rotation.
    """

    if not isinstance(payload, bytes) or len(payload) != 36:
        raise DisplayTransformError("display matrix side data must be 36 bytes")
    values = struct.unpack("=9i", payload)
    if values[2] != 0 or values[5] != 0 or values[6] != 0 \
            or values[7] != 0 or values[8] != 1 << 30:
        raise DisplayTransformError(
            "display matrix must be a pure affine rotation without translation")
    scale = 65_536.0
    affine = np.asarray(
        [
            [values[0] / scale, values[1] / scale],
            [values[3] / scale, values[4] / scale],
        ],
        dtype=np.float64,
    )
    if not np.allclose(affine.T @ affine, np.eye(2), atol=1e-6) \
            or not math.isclose(float(np.linalg.det(affine)), 1.0, abs_tol=1e-6):
        raise DisplayTransformError(
            "display matrix contains reflection, skew, or non-unit scaling")
    return _right_angle(math.degrees(math.atan2(affine[1, 0], affine[0, 0])))


def display_transform(width: int, height: int, rotation_degrees: int) -> DisplayTransform:
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise DisplayTransformError("coded width must be positive")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise DisplayTransformError("coded height must be positive")
    rotation = _right_angle(float(rotation_degrees))
    if rotation == 0:
        matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        display_width, display_height = width, height
    elif rotation == 90:  # counter-clockwise in image coordinates
        matrix = (0.0, 1.0, 0.0, -1.0, 0.0, float(width - 1), 0.0, 0.0, 1.0)
        display_width, display_height = height, width
    elif rotation == 180:
        matrix = (
            -1.0, 0.0, float(width - 1),
            0.0, -1.0, float(height - 1),
            0.0, 0.0, 1.0,
        )
        display_width, display_height = width, height
    else:  # 270 degrees counter-clockwise
        matrix = (0.0, -1.0, float(height - 1), 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        display_width, display_height = height, width
    return DisplayTransform(
        rotation_degrees=rotation,
        coded_width=width,
        coded_height=height,
        display_width=display_width,
        display_height=display_height,
        source_to_display=matrix,
    )


def apply_display_transform(image: np.ndarray, transform: DisplayTransform) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[0] != transform.coded_height \
            or frame.shape[1] != transform.coded_width:
        raise DisplayTransformError("coded image dimensions do not match transform")
    rotations = {0: 0, 90: 1, 180: 2, 270: 3}
    result = np.rot90(frame, k=rotations[transform.rotation_degrees])
    return np.ascontiguousarray(result)


def transform_points(
    points_xy: np.ndarray,
    matrix: tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise DisplayTransformError("points must be a finite Nx2 array")
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.size != 9 or not np.isfinite(transform).all():
        raise DisplayTransformError("coordinate transform must contain 9 finite values")
    transform = transform.reshape(3, 3)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    mapped = homogeneous @ transform.T
    if np.any(np.isclose(mapped[:, 2], 0.0)):
        raise DisplayTransformError("coordinate transform maps a point to infinity")
    return mapped[:, :2] / mapped[:, 2, None]
