"""Strict decoder-frame to CV-result association contract.

The decoder is the sole authority for frame identity.  This module validates
the three v0.3 Arrow tables without using row position, nominal FPS, or nearest
timestamp association.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np
import pyarrow as pa

from ..decode.frame_identity import canonical_sha256, stable_frame_id
from ..schema import QualityBits, logical_schema_contract
from ..schemas import (
    ASSOCIATION_SCHEMA_VERSION,
    FINALIZATION_SCHEMA_VERSION,
    HAND_LANDMARK_COUNT,
)
from ..schemas.cv_detections import (
    BBOX_COORDINATE_SPACE,
    LANDMARK_XY_COORDINATE_SPACE,
    LANDMARK_Z_COORDINATE_SPACE,
    cv_detections_schema,
)
from ..schemas.cv_frame_results import (
    INFERENCE_STATUSES,
    cv_frame_results_schema,
)
from ..schemas.video_frames import (
    DECODE_STATUSES,
    FRAME_IDENTITY_BASES,
    PTS_STATUSES,
    video_frames_schema,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DETECTION_ID_DOMAIN = "tremora-cv-detection-1"
_NORMALIZED_CV_INPUT = "NORMALIZED_CV_INPUT"
_PTS_QUALITY_MASK = int(
    QualityBits.MISSING_TIMESTAMP
    | QualityBits.NON_MONOTONIC_TIMESTAMP
    | QualityBits.DUPLICATE_TIMESTAMP
    | QualityBits.STREAM_GAP
)
_FRAME_QUALITY_MASK = _PTS_QUALITY_MASK | int(QualityBits.DECODE_FAILURE)
_RESULT_QUALITY_MASK = _FRAME_QUALITY_MASK | int(QualityBits.INVALID_CV)
_FLOAT64_ROUNDING_FACTOR = 64.0


class PoseFrameContractError(ValueError):
    """Raised when finalized CV rows are not exactly frame-associated."""


@dataclass(frozen=True)
class PoseFrameAssociationAudit:
    association_schema_version: str
    decoded_frame_count: int
    cv_frame_result_count: int
    detection_row_count: int
    frames_with_detection: int
    frames_without_detection: int
    inference_failure_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PoseFrameContractError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PoseFrameContractError(f"{field} must be a non-empty string")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PoseFrameContractError(
            f"{field} must be a non-negative integer")
    return value


def stable_detection_id(
    *,
    finalization_id: str,
    frame_id: str,
    detection_rank: int,
) -> str:
    """Return the deterministic identity for one ranked frame detection."""

    finalization_hash = _sha256(finalization_id, "finalization_id")
    frame_hash = _sha256(frame_id, "frame_id")
    rank = _nonnegative_int(detection_rank, "detection_rank")
    return canonical_sha256({
        "detection_rank": rank,
        "domain": _DETECTION_ID_DOMAIN,
        "finalization_id": finalization_hash,
        "frame_id": frame_hash,
    })


def _iter_rows(table: pa.Table) -> Iterator[dict[str, object]]:
    for batch in table.to_batches(max_chunksize=65_536):
        yield from batch.to_pylist()


def _require_schema(
    table: object,
    expected: pa.Schema,
    table_name: str,
) -> pa.Table:
    if not isinstance(table, pa.Table):
        raise PoseFrameContractError(f"{table_name} must be a pyarrow.Table")
    if logical_schema_contract(table.schema) != logical_schema_contract(expected):
        raise PoseFrameContractError(
            f"{table_name} does not match its v{FINALIZATION_SCHEMA_VERSION} "
            "logical Arrow schema"
        )
    return table


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoseFrameContractError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PoseFrameContractError(f"{field} must be finite")
    return result


def _optional_probability(value: object, field: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, field)
    if not 0.0 <= result <= 1.0:
        raise PoseFrameContractError(f"{field} must be in [0,1]")
    return result


def _matrix(value: object, field: str) -> np.ndarray:
    if value is None:
        raise PoseFrameContractError(f"{field} is required")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != 9 or not np.isfinite(matrix).all():
        raise PoseFrameContractError(
            f"{field} must contain 9 finite values")
    matrix = matrix.reshape(3, 3)
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-9):
        raise PoseFrameContractError(
            f"{field} must be an affine homogeneous transform")
    return matrix


def _float32_rounding_radius(values: np.ndarray) -> np.ndarray:
    """Return a conservative half-bin for values persisted as float32."""

    quantized = np.asarray(values, dtype=np.float32)
    lower = np.nextafter(
        quantized,
        np.full_like(quantized, -np.inf),
    ).astype(np.float64)
    upper = np.nextafter(
        quantized,
        np.full_like(quantized, np.inf),
    ).astype(np.float64)
    center = quantized.astype(np.float64)
    lower_spacing = center - lower
    upper_spacing = upper - center
    lower_spacing = np.where(
        np.isfinite(lower_spacing), lower_spacing, upper_spacing)
    upper_spacing = np.where(
        np.isfinite(upper_spacing), upper_spacing, lower_spacing)
    return np.maximum(lower_spacing, upper_spacing) * 0.5


def _reconcile_normalized_landmarks(
    *,
    detection_id: str,
    normalized_xy: np.ndarray,
    display_xy: np.ndarray,
    cv_input_width: int,
    cv_input_height: int,
    cv_to_display: np.ndarray,
) -> None:
    """Reconcile two independently float32-quantized coordinate views."""

    if cv_input_width < 2 or cv_input_height < 2:
        raise PoseFrameContractError(
            f"detection {detection_id} normalized coordinates require "
            "CV input dimensions of at least 2x2"
        )
    normalized = np.asarray(normalized_xy, dtype=np.float32).astype(np.float64)
    observed_display = np.asarray(
        display_xy, dtype=np.float32).astype(np.float64)
    pixel_scale = np.asarray(
        (cv_input_width - 1, cv_input_height - 1),
        dtype=np.float64,
    )
    cv_pixels = normalized * pixel_scale
    homogeneous = np.column_stack((
        cv_pixels,
        np.ones(len(cv_pixels), dtype=np.float64),
    ))
    expected_homogeneous = homogeneous @ cv_to_display.T
    if not np.allclose(
        expected_homogeneous[:, 2], 1.0, rtol=0.0, atol=1e-9,
    ):
        raise PoseFrameContractError(
            f"detection {detection_id} CV/display transform is not affine"
        )
    expected_display = expected_homogeneous[:, :2]

    # Normalized and display coordinates were each persisted independently as
    # float32.  Propagate half of one normalized float32 bin through the affine
    # map, then add half of one display float32 bin.  This accepts only values
    # that could share one pre-quantization coordinate pair.
    normalized_radius = _float32_rounding_radius(normalized)
    normalized_to_display = (
        cv_to_display[:2, :2] @ np.diag(pixel_scale)
    )
    propagated_radius = normalized_radius @ np.abs(
        normalized_to_display
    ).T
    display_radius = _float32_rounding_radius(observed_display)
    numerical_radius = (
        _FLOAT64_ROUNDING_FACTOR
        * np.finfo(np.float64).eps
        * np.maximum(1.0, np.abs(expected_display))
    )
    tolerance = propagated_radius + display_radius + numerical_radius
    if np.any(np.abs(observed_display - expected_display) > tolerance):
        raise PoseFrameContractError(
            f"detection {detection_id} normalized/display landmarks disagree"
        )


def _expected_source_to_display(
    width: int,
    height: int,
    rotation: int,
) -> tuple[np.ndarray, int, int]:
    if rotation == 0:
        values = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        display_width, display_height = width, height
    elif rotation == 90:
        values = (
            0.0, 1.0, 0.0,
            -1.0, 0.0, float(width - 1),
            0.0, 0.0, 1.0,
        )
        display_width, display_height = height, width
    elif rotation == 180:
        values = (
            -1.0, 0.0, float(width - 1),
            0.0, -1.0, float(height - 1),
            0.0, 0.0, 1.0,
        )
        display_width, display_height = width, height
    elif rotation == 270:
        values = (
            0.0, -1.0, float(height - 1),
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
        )
        display_width, display_height = height, width
    else:
        raise PoseFrameContractError(
            "rotation_degrees must be one of 0,90,180,270")
    return np.asarray(values, dtype=np.float64).reshape(3, 3), (
        display_width
    ), display_height


def _round_nonnegative_fraction(value: Fraction, field: str) -> int:
    if value < 0:
        raise PoseFrameContractError(f"{field} cannot be negative")
    return (value.numerator * 2 + value.denominator) // (
        2 * value.denominator
    )


def _ticks_to_ns(ticks: int, numerator: int, denominator: int, field: str) -> int:
    if denominator <= 0 or numerator <= 0:
        raise PoseFrameContractError("time base numerator/denominator must be positive")
    return _round_nonnegative_fraction(
        Fraction(ticks * numerator * 1_000_000_000, denominator),
        field,
    )


def _validate_frame_transform(row: dict[str, object]) -> None:
    frame_id = str(row["frame_id"])
    width = _nonnegative_int(row["coded_width"], f"{frame_id}.coded_width")
    height = _nonnegative_int(row["coded_height"], f"{frame_id}.coded_height")
    if width == 0 or height == 0:
        raise PoseFrameContractError("coded frame dimensions must be positive")
    rotation = row["rotation_degrees"]
    if isinstance(rotation, bool) or not isinstance(rotation, int):
        raise PoseFrameContractError("rotation_degrees must be an integer")
    expected, expected_width, expected_height = _expected_source_to_display(
        width, height, rotation)
    source_to_display = _matrix(
        row["source_to_display_transform"],
        f"{frame_id}.source_to_display_transform",
    )
    if not np.allclose(
        source_to_display, expected, rtol=0.0, atol=1e-9,
    ):
        raise PoseFrameContractError(
            f"frame {frame_id} source/display transform disagrees with rotation")
    if row["display_width"] != expected_width \
            or row["display_height"] != expected_height:
        raise PoseFrameContractError(
            f"frame {frame_id} display dimensions disagree with rotation")

    cv_fields = (
        row["cv_input_width"],
        row["cv_input_height"],
        row["cv_input_pixel_format"],
        row["cv_input_sha256"],
        row["display_to_cv_transform"],
    )
    cv_present = [value is not None for value in cv_fields]
    if any(cv_present) and not all(cv_present):
        raise PoseFrameContractError(
            f"frame {frame_id} has a partial CV-input description")
    invertible = row["preprocessing_transform_invertible"]
    if not isinstance(invertible, bool):
        raise PoseFrameContractError(
            "preprocessing_transform_invertible must be boolean")
    if not any(cv_present):
        if invertible or row["cv_to_source_transform"] is not None:
            raise PoseFrameContractError(
                f"frame {frame_id} without CV input cannot declare an inverse")
        return

    cv_width = _nonnegative_int(
        row["cv_input_width"], f"{frame_id}.cv_input_width")
    cv_height = _nonnegative_int(
        row["cv_input_height"], f"{frame_id}.cv_input_height")
    if cv_width == 0 or cv_height == 0:
        raise PoseFrameContractError("CV input dimensions must be positive")
    _nonempty_string(row["cv_input_pixel_format"], "cv_input_pixel_format")
    _sha256(row["cv_input_sha256"], "cv_input_sha256")
    display_to_cv = _matrix(
        row["display_to_cv_transform"],
        f"{frame_id}.display_to_cv_transform",
    )
    if invertible:
        cv_to_source = _matrix(
            row["cv_to_source_transform"],
            f"{frame_id}.cv_to_source_transform",
        )
        source_to_cv = display_to_cv @ source_to_display
        if abs(float(np.linalg.det(source_to_cv))) <= 1e-12:
            raise PoseFrameContractError(
                f"frame {frame_id} marks a singular transform invertible")
        if not np.allclose(
            cv_to_source @ source_to_cv,
            np.eye(3),
            rtol=0.0,
            atol=1e-8,
        ):
            raise PoseFrameContractError(
                f"frame {frame_id} CV/source transform is not the inverse")
    elif row["cv_to_source_transform"] is not None:
        raise PoseFrameContractError(
            f"frame {frame_id} marks a transform non-invertible but stores an inverse")


def _validate_detection_coordinates(
    row: dict[str, object],
    *,
    frame: dict[str, object],
) -> None:
    detection_id = str(row["detection_id"])
    display_width = int(frame["display_width"])
    display_height = int(frame["display_height"])
    if row["bbox_coordinate_space"] != BBOX_COORDINATE_SPACE:
        raise PoseFrameContractError(
            f"detection {detection_id} bbox is not in DISPLAY_PIXEL")
    if row["landmarks_xy_coordinate_space"] != LANDMARK_XY_COORDINATE_SPACE:
        raise PoseFrameContractError(
            f"detection {detection_id} XY landmarks are not in DISPLAY_PIXEL")
    bbox = np.asarray(row["bbox_xyxy_display"], dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all() \
            or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise PoseFrameContractError(
            f"detection {detection_id} has an invalid display-space bbox")
    # Finite coordinates outside the image remain evidence and are not clipped.
    if display_width <= 0 or display_height <= 0:
        raise PoseFrameContractError("display dimensions must be positive")

    xy = np.asarray(row["landmarks_xy_display"], dtype=np.float64)
    validity = np.asarray(row["landmark_validity_mask"])
    if xy.shape != (HAND_LANDMARK_COUNT * 2,) \
            or validity.shape != (HAND_LANDMARK_COUNT,) \
            or validity.dtype.kind != "b":
        raise PoseFrameContractError(
            f"detection {detection_id} has malformed landmark arrays")
    valid_xy = xy.reshape(HAND_LANDMARK_COUNT, 2)[validity]
    if not validity.any() or not np.isfinite(xy).all() \
            or not np.isfinite(valid_xy).all():
        raise PoseFrameContractError(
            f"detection {detection_id} has no finite valid landmarks")

    normalized = row["landmarks_xy_normalized_cv_input"]
    normalized_space = row["normalized_landmarks_coordinate_space"]
    if not isinstance(normalized, list) \
            or len(normalized) != HAND_LANDMARK_COUNT * 2:
        raise PoseFrameContractError(
            f"detection {detection_id} has malformed normalized coordinates")
    normalized_absent = all(value is None for value in normalized)
    if normalized_space is None:
        if not normalized_absent:
            raise PoseFrameContractError(
                f"detection {detection_id} has unlabelled normalized coordinates")
    else:
        if normalized_absent or any(value is None for value in normalized):
            raise PoseFrameContractError(
                f"detection {detection_id} has partial normalized coordinates")
        values = np.asarray(normalized, dtype=np.float64)
        if normalized_space != _NORMALIZED_CV_INPUT \
                or not np.isfinite(values).all():
            raise PoseFrameContractError(
                f"detection {detection_id} has invalid normalized coordinates")
        if not frame["preprocessing_transform_invertible"] \
                or frame["cv_input_width"] is None \
                or frame["cv_input_height"] is None \
                or frame["cv_to_source_transform"] is None:
            raise PoseFrameContractError(
                f"detection {detection_id} normalized coordinates lack an "
                "invertible CV/display mapping"
            )
        source_to_display = _matrix(
            frame["source_to_display_transform"],
            f"{frame['frame_id']}.source_to_display_transform",
        )
        cv_to_source = _matrix(
            frame["cv_to_source_transform"],
            f"{frame['frame_id']}.cv_to_source_transform",
        )
        _reconcile_normalized_landmarks(
            detection_id=detection_id,
            normalized_xy=values.reshape(HAND_LANDMARK_COUNT, 2),
            display_xy=xy.reshape(HAND_LANDMARK_COUNT, 2),
            cv_input_width=int(frame["cv_input_width"]),
            cv_input_height=int(frame["cv_input_height"]),
            cv_to_display=source_to_display @ cv_to_source,
        )

    z_values = row["landmarks_z_model"]
    z_space = row["landmarks_z_coordinate_space"]
    if not isinstance(z_values, list) or len(z_values) != HAND_LANDMARK_COUNT:
        raise PoseFrameContractError(
            f"detection {detection_id} has malformed model-relative Z data")
    z_absent = all(value is None for value in z_values)
    if z_space is None:
        if not z_absent:
            raise PoseFrameContractError(
                f"detection {detection_id} has unlabelled model-relative Z data")
    else:
        if z_absent or any(value is None for value in z_values):
            raise PoseFrameContractError(
                f"detection {detection_id} has partial model-relative Z data")
        values = np.asarray(z_values, dtype=np.float64)
        if z_space != LANDMARK_Z_COORDINATE_SPACE \
                or not np.isfinite(values).all():
            raise PoseFrameContractError(
                f"detection {detection_id} has invalid model-relative Z data")

    confidence = row["landmark_confidence"]
    if not isinstance(confidence, list) \
            or len(confidence) != HAND_LANDMARK_COUNT:
        raise PoseFrameContractError(
            f"detection {detection_id} has malformed landmark confidence")
    confidence_absent = all(value is None for value in confidence)
    if not confidence_absent:
        if any(value is None for value in confidence):
            raise PoseFrameContractError(
                f"detection {detection_id} has partial landmark confidence")
        values = np.asarray(confidence, dtype=np.float64)
        if not np.isfinite(values).all() \
                or np.any(values < 0.0) \
                or np.any(values > 1.0):
            raise PoseFrameContractError(
                f"detection {detection_id} has invalid landmark confidence")


def _detection_selection_key(row: dict[str, object]) -> tuple[float, str]:
    content_fields = (
        "handedness",
        "handedness_confidence",
        "detection_confidence",
        "bbox_xyxy_display",
        "bbox_coordinate_space",
        "landmarks_xy_display",
        "landmarks_xy_coordinate_space",
        "landmarks_xy_normalized_cv_input",
        "normalized_landmarks_coordinate_space",
        "landmarks_z_model",
        "landmarks_z_coordinate_space",
        "landmark_confidence",
        "landmark_validity_mask",
    )
    return (
        -float(row["selection_score"]),
        canonical_sha256({field: row[field] for field in content_fields}),
    )


def _uniform_value(
    rows: list[dict[str, object]],
    field: str,
) -> object:
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise PoseFrameContractError(
            f"cv_frame_results has non-uniform {field}")
    return next(iter(values))


def _require_expected(actual: object, expected: object | None, field: str) -> None:
    if expected is not None and actual != expected:
        raise PoseFrameContractError(
            f"{field} disagrees with the frozen finalization input")


def validate_pose_frame_association(
    video_frames: pa.Table,
    cv_frame_results: pa.Table,
    cv_detections: pa.Table,
    *,
    finalization_id: str,
    expected_recording_id: str | None = None,
    expected_source_video_sha256: str | None = None,
    expected_decoder_version: str | None = None,
    expected_model_id: str | None = None,
    expected_model_weights_sha256: str | None = None,
    expected_preprocessing_config_sha256: str | None = None,
    expected_inference_environment_id: str | None = None,
    expected_discontinuity_threshold_ns: int = 1_000_000_000,
) -> PoseFrameAssociationAudit:
    """Validate the complete v0.3 one-frame-to-one-result contract.

    ``finalization_id`` is required because detection identities are scoped to
    immutable decoder/model/preprocessing inputs.  Expected values, when
    supplied from the finalization manifest, close the table-to-manifest hash
    binding rather than merely checking within-table uniformity.
    """

    _sha256(finalization_id, "finalization_id")
    if isinstance(expected_discontinuity_threshold_ns, bool) or not isinstance(
            expected_discontinuity_threshold_ns, int) \
            or expected_discontinuity_threshold_ns <= 0:
        raise PoseFrameContractError(
            "expected_discontinuity_threshold_ns must be positive")
    frames_table = _require_schema(
        video_frames, video_frames_schema(), "video_frames")
    results_table = _require_schema(
        cv_frame_results, cv_frame_results_schema(), "cv_frame_results")
    detections_table = _require_schema(
        cv_detections, cv_detections_schema(), "cv_detections")
    if frames_table.num_rows == 0:
        raise PoseFrameContractError(
            "a finalized recording must contain at least one decoded frame")

    frames_by_id: dict[str, dict[str, object]] = {}
    frames_by_stream: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    dataset_ids: set[object] = set()
    recording_ids: set[object] = set()
    source_hashes: set[object] = set()
    decoder_versions: set[object] = set()

    for row in _iter_rows(frames_table):
        dataset_ids.add(row["dataset_id"])
        recording_ids.add(row["recording_id"])
        source_hashes.add(row["source_video_sha256"])
        decoder_versions.add(row["decoder_version"])
        frame_id = _sha256(row["frame_id"], "frame_id")
        if frame_id in frames_by_id:
            raise PoseFrameContractError(f"duplicate frame_id: {frame_id}")
        _nonempty_string(row["dataset_id"], "dataset_id")
        recording_id = _nonempty_string(row["recording_id"], "recording_id")
        source_hash = _sha256(
            row["source_video_sha256"], "source_video_sha256")
        stream_index = _nonnegative_int(row["stream_index"], "stream_index")
        decode_ordinal = _nonnegative_int(
            row["decode_ordinal"], "decode_ordinal")
        same_pts_rank = _nonnegative_int(
            row["same_pts_rank"], "same_pts_rank")
        identity_basis = row["identity_basis"]
        if identity_basis not in FRAME_IDENTITY_BASES:
            raise PoseFrameContractError(
                f"frame {frame_id} has unsupported identity_basis")
        pts = row["pts"]
        if pts is not None and (
            isinstance(pts, bool) or not isinstance(pts, int)
        ):
            raise PoseFrameContractError("pts must be an integer or null")
        numerator = row["time_base_num"]
        denominator = row["time_base_den"]
        if isinstance(numerator, bool) or not isinstance(numerator, int) \
                or numerator <= 0 \
                or isinstance(denominator, bool) \
                or not isinstance(denominator, int) or denominator <= 0:
            raise PoseFrameContractError(
                f"frame {frame_id} has an invalid or missing time base")
        expected_frame_id, expected_basis = stable_frame_id(
            source_video_sha256=source_hash,
            stream_index=stream_index,
            pts=pts,
            same_pts_rank=same_pts_rank,
            decode_ordinal=decode_ordinal,
        )
        if frame_id != expected_frame_id or identity_basis != expected_basis:
            raise PoseFrameContractError(
                f"frame {frame_id} identity does not recompute from source PTS")

        if row["schema_version"] != FINALIZATION_SCHEMA_VERSION:
            raise PoseFrameContractError(
                f"frame {frame_id} has unsupported schema_version")
        _nonempty_string(row["decoder_version"], "decoder_version")
        if row["pts_status"] not in PTS_STATUSES:
            raise PoseFrameContractError(f"frame {frame_id} has invalid pts_status")
        if row["decode_status"] not in DECODE_STATUSES:
            raise PoseFrameContractError(
                f"frame {frame_id} has invalid decode_status")
        quality = _nonnegative_int(row["quality_bits"], "quality_bits")
        if quality & ~_FRAME_QUALITY_MASK:
            raise PoseFrameContractError(
                f"frame {frame_id} has forbidden v0.3 quality bits")
        decode_failed = row["decode_status"] == "CORRUPT"
        has_decode_failure_bit = bool(
            quality & int(QualityBits.DECODE_FAILURE))
        if decode_failed != has_decode_failure_bit:
            raise PoseFrameContractError(
                f"frame {frame_id} decode status and quality bit disagree")
        _nonempty_string(row["pixel_format"], "pixel_format")
        if not isinstance(row["key_frame"], bool):
            raise PoseFrameContractError("key_frame must be boolean")
        _nonempty_string(row["picture_type"], "picture_type")

        if pts is None:
            if row["pts_status"] != "MISSING" \
                    or identity_basis != "MISSING_PTS_DECODE_ORDINAL" \
                    or row["presentation_ordinal"] is not None \
                    or row["relative_pts_ns"] is not None \
                    or row["gap_before_ns"] is not None \
                    or same_pts_rank != 0:
                raise PoseFrameContractError(
                    f"missing-PTS frame {frame_id} contains fabricated timing")
            if not quality & int(QualityBits.MISSING_TIMESTAMP):
                raise PoseFrameContractError(
                    f"missing-PTS frame {frame_id} lacks its quality bit")
        else:
            if identity_basis != "SOURCE_PTS_SAME_PTS_RANK":
                raise PoseFrameContractError(
                    f"PTS-bearing frame {frame_id} uses a fallback identity")
            _nonnegative_int(
                row["presentation_ordinal"], "presentation_ordinal")
            _nonnegative_int(row["relative_pts_ns"], "relative_pts_ns")
            if quality & int(QualityBits.MISSING_TIMESTAMP):
                raise PoseFrameContractError(
                    f"PTS-bearing frame {frame_id} has a missing-timestamp bit")

        duration_pts = row["duration_pts"]
        duration_ns = row["duration_ns"]
        if (duration_pts is None) != (duration_ns is None):
            raise PoseFrameContractError(
                f"frame {frame_id} has a partial duration")
        if duration_pts is not None:
            duration_ticks = _nonnegative_int(duration_pts, "duration_pts")
            if row["time_base_num"] is None or row["time_base_den"] is None:
                raise PoseFrameContractError(
                    f"frame {frame_id} duration has no rational time base")
            expected_duration_ns = _ticks_to_ns(
                duration_ticks,
                int(row["time_base_num"]),
                int(row["time_base_den"]),
                "duration_ns",
            )
            if duration_ns != expected_duration_ns:
                raise PoseFrameContractError(
                    f"frame {frame_id} duration_ns is not rationally derived")

        _validate_frame_transform(row)
        frames_by_id[frame_id] = row
        frames_by_stream[(source_hash, stream_index)].append(row)

    for field, values in (
        ("dataset_id", dataset_ids),
        ("recording_id", recording_ids),
        ("source_video_sha256", source_hashes),
        ("decoder_version", decoder_versions),
    ):
        if len(values) != 1:
            raise PoseFrameContractError(
                f"one finalized recording must have one uniform {field}")
    recording_id = next(iter(recording_ids))
    source_hash = next(iter(source_hashes))
    decoder_version = next(iter(decoder_versions))
    _require_expected(recording_id, expected_recording_id, "recording_id")
    _require_expected(
        source_hash, expected_source_video_sha256, "source_video_sha256")
    _require_expected(
        decoder_version, expected_decoder_version, "decoder_version")

    for stream_key, rows in frames_by_stream.items():
        ordered_decode = sorted(rows, key=lambda row: int(row["decode_ordinal"]))
        ordinals = [int(row["decode_ordinal"]) for row in ordered_decode]
        if ordinals != list(range(len(ordinals))):
            raise PoseFrameContractError(
                f"stream {stream_key} decode ordinals are not contiguous from zero")

        valid_rows = [row for row in rows if row["pts"] is not None]
        pts_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in valid_rows:
            pts_groups[int(row["pts"])].append(row)
        for pts, same_pts_rows in pts_groups.items():
            ranked = sorted(
                same_pts_rows, key=lambda row: int(row["decode_ordinal"]))
            if [row["same_pts_rank"] for row in ranked] != list(
                range(len(ranked))
            ):
                raise PoseFrameContractError(
                    f"PTS {pts} same_pts_rank is not stable decode order")
            for row in ranked:
                quality = int(row["quality_bits"])
                if len(ranked) > 1 and not quality & int(
                    QualityBits.DUPLICATE_TIMESTAMP
                ):
                    raise PoseFrameContractError(
                        f"duplicate PTS frame {row['frame_id']} lacks its quality bit")
                if len(ranked) == 1 and quality & int(
                    QualityBits.DUPLICATE_TIMESTAMP
                ):
                    raise PoseFrameContractError(
                        f"unique PTS frame {row['frame_id']} has a duplicate bit")

        previous_timestamp: Fraction | None = None
        for row in ordered_decode:
            if row["pts"] is None:
                continue
            current_timestamp = Fraction(
                int(row["pts"]) * int(row["time_base_num"]),
                int(row["time_base_den"]),
            )
            nonmonotonic = (
                previous_timestamp is not None
                and current_timestamp < previous_timestamp
            )
            has_bit = bool(int(row["quality_bits"]) & int(
                QualityBits.NON_MONOTONIC_TIMESTAMP
            ))
            if nonmonotonic != has_bit:
                raise PoseFrameContractError(
                    f"frame {row['frame_id']} non-monotonic bit disagrees with PTS")
            previous_timestamp = current_timestamp

        discontinuity_ids: set[str] = set()
        if valid_rows:
            presentation = sorted(
                valid_rows,
                key=lambda row: (
                    Fraction(
                        int(row["pts"]) * int(row["time_base_num"]),
                        int(row["time_base_den"]),
                    ),
                    int(row["same_pts_rank"]),
                    int(row["decode_ordinal"]),
                ),
            )
            if [row["presentation_ordinal"] for row in presentation] != list(
                range(len(presentation))
            ):
                raise PoseFrameContractError(
                    f"stream {stream_key} presentation ordinals do not follow PTS")
            origin = Fraction(
                int(presentation[0]["pts"])
                * int(presentation[0]["time_base_num"]),
                int(presentation[0]["time_base_den"]),
            )
            previous_timestamp: Fraction | None = None
            for row in presentation:
                timestamp = Fraction(
                    int(row["pts"]) * int(row["time_base_num"]),
                    int(row["time_base_den"]),
                )
                relative = _round_nonnegative_fraction(
                    (timestamp - origin) * 1_000_000_000,
                    "relative_pts_ns",
                )
                if row["relative_pts_ns"] != relative:
                    raise PoseFrameContractError(
                        f"frame {row['frame_id']} relative_pts_ns is not rationally derived")
                expected_gap = (
                    None if previous_timestamp is None else (
                        _round_nonnegative_fraction(
                            (timestamp - previous_timestamp) * 1_000_000_000,
                            "gap_before_ns",
                        )
                    )
                )
                if row["gap_before_ns"] != expected_gap:
                    raise PoseFrameContractError(
                        f"frame {row['frame_id']} gap_before_ns is inconsistent")
                if expected_gap is not None \
                        and expected_gap > expected_discontinuity_threshold_ns:
                    discontinuity_ids.add(str(row["frame_id"]))
                previous_timestamp = timestamp

        nonmonotonic_ids = {
            str(row["frame_id"])
            for row in rows
            if int(row["quality_bits"])
            & int(QualityBits.NON_MONOTONIC_TIMESTAMP)
        }
        for row in rows:
            frame_id = str(row["frame_id"])
            pts = row["pts"]
            missing = pts is None
            duplicate = pts is not None and len(pts_groups[int(pts)]) > 1
            nonmonotonic = frame_id in nonmonotonic_ids
            discontinuity = frame_id in discontinuity_ids
            expected_pts_bits = 0
            if missing:
                expected_pts_bits |= int(QualityBits.MISSING_TIMESTAMP)
            if duplicate:
                expected_pts_bits |= int(QualityBits.DUPLICATE_TIMESTAMP)
            if nonmonotonic:
                expected_pts_bits |= int(QualityBits.NON_MONOTONIC_TIMESTAMP)
            if discontinuity:
                expected_pts_bits |= int(QualityBits.STREAM_GAP)
            observed_pts_bits = int(row["quality_bits"]) & _PTS_QUALITY_MASK
            if observed_pts_bits != expected_pts_bits:
                raise PoseFrameContractError(
                    f"frame {frame_id} PTS quality bits disagree with timing")
            expected_status = (
                "MISSING" if missing else
                "DUPLICATE" if duplicate else
                "NON_MONOTONIC" if nonmonotonic else
                "DISCONTINUITY" if discontinuity else
                "VALID"
            )
            if row["pts_status"] != expected_status:
                raise PoseFrameContractError(
                    f"frame {frame_id} PTS status is not canonical")

    result_rows = list(_iter_rows(results_table))
    if len(result_rows) != len(frames_by_id):
        raise PoseFrameContractError(
            "decoded frame count does not equal CV frame-result count")
    result_counts = Counter(row["frame_id"] for row in result_rows)
    if set(result_counts) != set(frames_by_id) \
            or any(count != 1 for count in result_counts.values()):
        raise PoseFrameContractError(
            "every decoded frame must have exactly one CV frame-result row")

    for row in result_rows:
        frame_id = _sha256(row["frame_id"], "cv_frame_results.frame_id")
        frame = frames_by_id[frame_id]
        if row["recording_id"] != frame["recording_id"]:
            raise PoseFrameContractError(
                f"result {frame_id} recording_id disagrees with its frame")
        if row["relative_pts_ns"] != frame["relative_pts_ns"]:
            raise PoseFrameContractError(
                f"result {frame_id} timing disagrees with its frame")
        if row["inference_status"] not in INFERENCE_STATUSES:
            raise PoseFrameContractError(
                f"result {frame_id} has an invalid inference_status")
        count = _nonnegative_int(row["detection_count"], "detection_count")
        status = row["inference_status"]
        if (status == "SUCCESS") != (count > 0):
            raise PoseFrameContractError(
                f"result {frame_id} status/detection count is inconsistent")
        has_cv_input = frame["cv_input_sha256"] is not None
        if status in {"SUCCESS", "NO_DETECTION", "INFERENCE_FAILURE"} \
                and not has_cv_input:
            raise PoseFrameContractError(
                f"result {frame_id} status requires a materialized CV input")
        if status in {"PREPROCESS_FAILURE", "DECODE_FAILURE"} and has_cv_input:
            raise PoseFrameContractError(
                f"result {frame_id} failure status conflicts with its CV input")
        runtime = row["runtime_ms"]
        if runtime is not None and _finite_number(runtime, "runtime_ms") < 0.0:
            raise PoseFrameContractError("runtime_ms cannot be negative")
        _optional_probability(row["tracking_quality"], "tracking_quality")
        quality = _nonnegative_int(
            row["frame_quality_bits"], "frame_quality_bits")
        if quality & ~_RESULT_QUALITY_MASK:
            raise PoseFrameContractError(
                f"result {frame_id} has forbidden v0.3 quality bits")
        expected_result_quality = int(frame["quality_bits"])
        if status in {"PREPROCESS_FAILURE", "INFERENCE_FAILURE", "REJECTED_INPUT"}:
            expected_result_quality |= int(QualityBits.INVALID_CV)
        if quality != expected_result_quality:
            raise PoseFrameContractError(
                f"result {frame_id} quality bits are not canonically derived")
        if status in {"PREPROCESS_FAILURE", "INFERENCE_FAILURE", "REJECTED_INPUT"} \
                and not quality & int(QualityBits.INVALID_CV):
            raise PoseFrameContractError(
                f"result {frame_id} CV failure lacks INVALID_CV quality")
        if status == "DECODE_FAILURE" and frame["decode_status"] != "CORRUPT":
            raise PoseFrameContractError(
                f"result {frame_id} claims an unrecorded decode failure")
        if frame["decode_status"] == "CORRUPT" and status != "DECODE_FAILURE":
            raise PoseFrameContractError(
                f"result {frame_id} does not retain its decode failure")

    model_id = _uniform_value(result_rows, "model_id")
    weights_hash = _uniform_value(result_rows, "model_weights_sha256")
    preprocessing_hash = _uniform_value(
        result_rows, "preprocessing_config_sha256")
    environment_id = _uniform_value(
        result_rows, "inference_environment_id")
    _nonempty_string(model_id, "model_id")
    _sha256(weights_hash, "model_weights_sha256")
    _sha256(preprocessing_hash, "preprocessing_config_sha256")
    _nonempty_string(environment_id, "inference_environment_id")
    _require_expected(model_id, expected_model_id, "model_id")
    _require_expected(
        weights_hash, expected_model_weights_sha256, "model_weights_sha256")
    _require_expected(
        preprocessing_hash,
        expected_preprocessing_config_sha256,
        "preprocessing_config_sha256",
    )
    _require_expected(
        environment_id,
        expected_inference_environment_id,
        "inference_environment_id",
    )

    detections_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    detection_ids: set[str] = set()
    for row in _iter_rows(detections_table):
        detection_id = _sha256(row["detection_id"], "detection_id")
        if detection_id in detection_ids:
            raise PoseFrameContractError(
                f"duplicate detection_id: {detection_id}")
        detection_ids.add(detection_id)
        frame_id = _sha256(row["frame_id"], "cv_detections.frame_id")
        if frame_id not in frames_by_id:
            raise PoseFrameContractError(
                f"detection {detection_id} references an unknown frame")
        rank = _nonnegative_int(row["detection_rank"], "detection_rank")
        expected_id = stable_detection_id(
            finalization_id=finalization_id,
            frame_id=frame_id,
            detection_rank=rank,
        )
        if detection_id != expected_id:
            raise PoseFrameContractError(
                f"detection {detection_id} identity does not recompute")
        _optional_probability(
            row["handedness_confidence"], "handedness_confidence")
        _optional_probability(
            row["detection_confidence"], "detection_confidence")
        _finite_number(row["selection_score"], "selection_score")
        if not isinstance(row["selected_for_primary_track"], bool):
            raise PoseFrameContractError(
                "selected_for_primary_track must be boolean")
        frame = frames_by_id[frame_id]
        _validate_detection_coordinates(
            row,
            frame=frame,
        )
        detections_by_frame[frame_id].append(row)

    frames_with_detection = 0
    for result in result_rows:
        frame_id = str(result["frame_id"])
        rows = sorted(
            detections_by_frame.get(frame_id, ()),
            key=lambda row: int(row["detection_rank"]),
        )
        if [row["detection_rank"] for row in rows] != list(range(len(rows))):
            raise PoseFrameContractError(
                f"frame {frame_id} detection ranks are not contiguous from zero")
        if result["detection_count"] != len(rows):
            raise PoseFrameContractError(
                f"frame {frame_id} detection_count disagrees with detection rows")
        selected = [
            row for row in rows if row["selected_for_primary_track"]
        ]
        if len(selected) > 1:
            raise PoseFrameContractError(
                f"frame {frame_id} has multiple primary detections")
        selected_id = result["selected_detection_id"]
        expected_selected = rows[0]["detection_id"] if rows else None
        if selected_id != expected_selected:
            raise PoseFrameContractError(
                f"frame {frame_id} selected_detection_id is inconsistent")
        if rows and (
            len(selected) != 1 or selected[0]["detection_rank"] != 0
        ):
            raise PoseFrameContractError(
                f"frame {frame_id} detections lack a deterministic primary selection")
        if rows != sorted(rows, key=_detection_selection_key):
            raise PoseFrameContractError(
                f"frame {frame_id} detection ranks violate the frozen selection rule")
        if rows:
            frames_with_detection += 1

    failure_count = sum(
        row["inference_status"] == "INFERENCE_FAILURE"
        for row in result_rows
    )
    return PoseFrameAssociationAudit(
        association_schema_version=ASSOCIATION_SCHEMA_VERSION,
        decoded_frame_count=len(frames_by_id),
        cv_frame_result_count=len(result_rows),
        detection_row_count=len(detection_ids),
        frames_with_detection=frames_with_detection,
        frames_without_detection=len(frames_by_id) - frames_with_detection,
        inference_failure_count=failure_count,
    )


__all__ = [
    "ASSOCIATION_SCHEMA_VERSION",
    "PoseFrameAssociationAudit",
    "PoseFrameContractError",
    "stable_detection_id",
    "validate_pose_frame_association",
]
