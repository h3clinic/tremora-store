"""Coordinate-space and CV-input hashing rules for frame finalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

SOURCE_PIXEL = "SOURCE_PIXEL"
DISPLAY_PIXEL = "DISPLAY_PIXEL"
CV_INPUT_PIXEL = "CV_INPUT_PIXEL"
NORMALIZED_CV_INPUT = "NORMALIZED_CV_INPUT"
COORDINATE_SPACES = frozenset({
    SOURCE_PIXEL,
    DISPLAY_PIXEL,
    CV_INPUT_PIXEL,
    NORMALIZED_CV_INPUT,
})
PIXEL_CONVENTION = "integer_pixel_centers_origin_top_left_v1"
CV_INPUT_HASH_VERSION = "tremora-cv-input-bgr24-1"


class CoordinateMappingError(ValueError):
    """Raised when an image or coordinate mapping is ambiguous."""


@dataclass(frozen=True)
class PreparedCVInput:
    """One deterministic estimator input and its display-to-input mapping."""

    pixels: np.ndarray
    display_to_cv_transform: tuple[float, ...]
    pixel_format: str = "bgr24"

    def __post_init__(self) -> None:
        pixels = np.asarray(self.pixels)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise CoordinateMappingError(
                "CV input must be an HxWx3 uint8 array")
        if self.pixel_format != "bgr24":
            raise CoordinateMappingError("v0.3 only supports canonical bgr24 input")
        _matrix(self.display_to_cv_transform, "display_to_cv_transform")


def identity_transform() -> tuple[float, ...]:
    return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _matrix(value: tuple[float, ...] | np.ndarray, field: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != 9 or not np.isfinite(matrix).all():
        raise CoordinateMappingError(f"{field} must contain 9 finite values")
    return matrix.reshape(3, 3)


def compose_transforms(
    first: tuple[float, ...] | np.ndarray,
    second: tuple[float, ...] | np.ndarray,
) -> tuple[float, ...]:
    """Return a transform that applies ``first`` and then ``second``."""

    result = _matrix(second, "second transform") @ _matrix(
        first, "first transform")
    return tuple(float(value) for value in result.reshape(-1))


def invert_transform(
    value: tuple[float, ...] | np.ndarray,
) -> tuple[float, ...]:
    matrix = _matrix(value, "coordinate transform")
    determinant = float(np.linalg.det(matrix))
    if math_is_zero(determinant):
        raise CoordinateMappingError("coordinate transform is not invertible")
    inverse = np.linalg.inv(matrix)
    return tuple(float(item) for item in inverse.reshape(-1))


def math_is_zero(value: float) -> bool:
    return bool(np.isclose(value, 0.0, rtol=0.0, atol=1e-12))


def map_points(
    points_xy: np.ndarray,
    transform: tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise CoordinateMappingError("points must be a finite Nx2 array")
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    mapped = homogeneous @ _matrix(transform, "coordinate transform").T
    if np.any(np.isclose(mapped[:, 2], 0.0, rtol=0.0, atol=1e-12)):
        raise CoordinateMappingError("coordinate transform maps a point to infinity")
    return mapped[:, :2] / mapped[:, 2, None]


def normalized_to_pixels(
    points_xy: np.ndarray, *, width: int, height: int,
) -> np.ndarray:
    """Map normalized edge coordinates to pixel-center coordinates.

    MediaPipe-style normalized 0 and 1 map to the centers of the first and last
    pixels. Finite out-of-frame values are preserved rather than clipped.
    """

    if isinstance(width, bool) or not isinstance(width, int) or width <= 0 \
            or isinstance(height, bool) or not isinstance(height, int) \
            or height <= 0:
        raise CoordinateMappingError("image dimensions must be positive integers")
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise CoordinateMappingError("normalized points must be finite Nx2")
    scale = np.asarray((max(width - 1, 0), max(height - 1, 0)), dtype=np.float64)
    return points * scale


def map_bbox_xyxy(
    bbox: tuple[float, float, float, float] | np.ndarray,
    transform: tuple[float, ...] | np.ndarray,
) -> tuple[float, float, float, float]:
    values = np.asarray(bbox, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() \
            or values[2] < values[0] or values[3] < values[1]:
        raise CoordinateMappingError("bbox must be finite ordered xyxy")
    corners = np.asarray([
        (values[0], values[1]), (values[2], values[1]),
        (values[2], values[3]), (values[0], values[3]),
    ])
    mapped = map_points(corners, transform)
    return (
        float(np.min(mapped[:, 0])), float(np.min(mapped[:, 1])),
        float(np.max(mapped[:, 0])), float(np.max(mapped[:, 1])),
    )


def canonical_cv_input_sha256(prepared: PreparedCVInput) -> str:
    """Hash dimensions, format, dtype, and contiguous CV input pixels."""

    pixels = np.ascontiguousarray(prepared.pixels)
    header = json.dumps(
        {
            "channels": int(pixels.shape[2]),
            "dtype": str(pixels.dtype),
            "height": int(pixels.shape[0]),
            "pixel_format": prepared.pixel_format,
            "version": CV_INPUT_HASH_VERSION,
            "width": int(pixels.shape[1]),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    payload = memoryview(pixels).cast("B")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()
