"""Recipe Validator and Guardrail Schema Enforcement.

Strictly adheres to Appendix Item #3 and #4 in CODEX_REVIEW_03.md:
- All recipe fields required: pipeline_version, exposure, contrast, saturation, crop.
- Forbids bool as numeric, non-finite floats, wrong types, unknown keys, unsupported versions.
- Clamps finite scalars into conservative safety bounds and logs.
- Strictly validates crop bounds [ymin, xmin, ymax, xmax] in [0, 1].
- Rejects crop if goal requires preserving body/context without trusted bbox.
- Calculates pixel bounds using floor(left/top), ceil(right/bottom) and checks actual size/area.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from photo_advisor.types import (
    PIPELINE_VERSION,
    EXPOSURE_CLAMP_MIN,
    EXPOSURE_CLAMP_MAX,
    CONTRAST_CLAMP_MIN,
    CONTRAST_CLAMP_MAX,
    SATURATION_CLAMP_MIN,
    SATURATION_CLAMP_MAX,
    MIN_EXPORT_DIMENSION,
    MIN_CROP_AREA_RATIO,
    GOALS_PRESERVE_ALL,
    Recipe,
    ValidationResult,
    sanitize_value_for_json,
)


ALLOWED_KEYS = {"pipeline_version", "exposure", "contrast", "saturation", "crop"}


def _is_valid_numeric(val: Any) -> bool:
    """Check if value is a finite number (int or float) and strictly NOT a boolean."""
    if isinstance(val, bool):
        return False
    if not isinstance(val, (int, float)):
        return False
    try:
        fval = float(val)
        if not math.isfinite(fval):
            return False
        if abs(fval) > 1e12:
            return False
    except (OverflowError, ValueError):
        return False
    return True


def validate_and_clamp_recipe(
    raw_recipe: Dict[str, Any],
    image_width: int,
    image_height: int,
    goal: Optional[str] = None,
    has_trusted_bbox: bool = False,
) -> ValidationResult:
    """Validate a raw recipe dictionary and compute the effective clamped recipe.

    Args:
        raw_recipe: Dictionary received from model, rule, or imported fixture.
        image_width: Width of the upright sRGB image.
        image_height: Height of the upright sRGB image.
        goal: User style/scene goal (e.g., 'full_body', 'natural_portrait').
        has_trusted_bbox: Whether a trusted subject detection box is verified.

    Returns:
        ValidationResult containing validation status, effective recipe, logs.
    """
    if not isinstance(raw_recipe, dict):
        return ValidationResult(
            is_valid=False,
            rejection_reason=f"Recipe must be a JSON object / dict, got {type(raw_recipe).__name__}",
        )

    sanitized_requested = sanitize_value_for_json(raw_recipe)

    # 1. Unknown keys check
    extra_keys = set(raw_recipe.keys()) - ALLOWED_KEYS
    if extra_keys:
        return ValidationResult(
            is_valid=False,
            requested_recipe=sanitized_requested,
            rejection_reason=f"Unknown keys in recipe: {sorted(list(extra_keys))}. Strict schema required.",
        )

    # 2. Required keys check
    missing_keys = ALLOWED_KEYS - set(raw_recipe.keys())
    if missing_keys:
        return ValidationResult(
            is_valid=False,
            requested_recipe=sanitized_requested,
            rejection_reason=f"Missing required recipe keys: {sorted(list(missing_keys))}",
        )

    # 3. Pipeline version check
    req_version = raw_recipe.get("pipeline_version")
    if req_version != PIPELINE_VERSION:
        return ValidationResult(
            is_valid=False,
            requested_recipe=sanitized_requested,
            rejection_reason=f"Unsupported pipeline_version: '{req_version}'. Expected '{PIPELINE_VERSION}'",
        )

    warnings: List[str] = []

    # 4. Exposure validation & clamping
    raw_exp = raw_recipe.get("exposure")
    if not _is_valid_numeric(raw_exp):
        return ValidationResult(
            is_valid=False,
            requested_recipe=sanitized_requested,
            rejection_reason=f"Invalid exposure value '{raw_exp}': must be a finite number, bool forbidden.",
        )
    val_exp = float(raw_exp)
    eff_exp = max(EXPOSURE_CLAMP_MIN, min(EXPOSURE_CLAMP_MAX, val_exp))
    if eff_exp != val_exp:
        warnings.append(
            f"Clamped exposure from requested {val_exp:.4f} to effective {eff_exp:.4f} [{EXPOSURE_CLAMP_MIN}, {EXPOSURE_CLAMP_MAX}]"
        )

    # 5. Contrast validation & clamping
    raw_con = raw_recipe.get("contrast")
    if not _is_valid_numeric(raw_con):
        return ValidationResult(
            is_valid=False,
            requested_recipe=sanitized_requested,
            rejection_reason=f"Invalid contrast value '{raw_con}': must be a finite number, bool forbidden.",
        )
    val_con = float(raw_con)
    eff_con = max(CONTRAST_CLAMP_MIN, min(CONTRAST_CLAMP_MAX, val_con))
    if eff_con != val_con:
        warnings.append(
            f"Clamped contrast from requested {val_con:.4f} to effective {eff_con:.4f} [{CONTRAST_CLAMP_MIN}, {CONTRAST_CLAMP_MAX}]"
        )

    # 6. Saturation validation & clamping
    raw_sat = raw_recipe.get("saturation")
    if not _is_valid_numeric(raw_sat):
        return ValidationResult(
            is_valid=False,
            requested_recipe=sanitized_requested,
            rejection_reason=f"Invalid saturation value '{raw_sat}': must be a finite number, bool forbidden.",
        )
    val_sat = float(raw_sat)
    eff_sat = max(SATURATION_CLAMP_MIN, min(SATURATION_CLAMP_MAX, val_sat))
    if eff_sat != val_sat:
        warnings.append(
            f"Clamped saturation from requested {val_sat:.4f} to effective {eff_sat:.4f} [{SATURATION_CLAMP_MIN}, {SATURATION_CLAMP_MAX}]"
        )

    # 7. Crop validation
    raw_crop = raw_recipe.get("crop")
    eff_crop: Optional[Tuple[float, float, float, float]] = None

    if raw_crop is not None:
        if not isinstance(raw_crop, (list, tuple)) or len(raw_crop) != 4:
            return ValidationResult(
                is_valid=False,
                requested_recipe=sanitized_requested,
                rejection_reason=f"Crop must be null or a list/tuple of 4 numbers [ymin, xmin, ymax, xmax], got {raw_crop}",
            )
        for i, coord in enumerate(raw_crop):
            if not _is_valid_numeric(coord):
                return ValidationResult(
                    is_valid=False,
                    requested_recipe=sanitized_requested,
                    rejection_reason=f"Crop coordinate at index {i} ('{coord}') is not a valid finite number.",
                )

        ymin, xmin, ymax, xmax = float(raw_crop[0]), float(raw_crop[1]), float(raw_crop[2]), float(raw_crop[3])

        # Bounds check in [0.0, 1.0]
        if not (0.0 <= ymin < ymax <= 1.0 and 0.0 <= xmin < xmax <= 1.0):
            return ValidationResult(
                is_valid=False,
                requested_recipe=sanitized_requested,
                rejection_reason=f"Invalid crop bounds [{ymin}, {xmin}, {ymax}, {xmax}]. Must satisfy 0.0 <= ymin < ymax <= 1.0 and 0.0 <= xmin < xmax <= 1.0.",
            )

        # Check goal preservation: if goal is full_body or keep_context without verified bbox, reject automatic crop
        if goal and goal in GOALS_PRESERVE_ALL and not has_trusted_bbox:
            return ValidationResult(
                is_valid=False,
                requested_recipe=sanitized_requested,
                rejection_reason=f"Crop rejected: Goal '{goal}' requires preserving full body or context, but no trusted bbox is verified.",
            )

        # Pixel bounds computation: floor(left/top), ceil(right/bottom)
        left = int(math.floor(xmin * image_width))
        top = int(math.floor(ymin * image_height))
        right = int(math.ceil(xmax * image_width))
        bottom = int(math.ceil(ymax * image_height))

        # Clamp to image boundary
        left = max(0, min(image_width, left))
        top = max(0, min(image_height, top))
        right = max(0, min(image_width, right))
        bottom = max(0, min(image_height, bottom))

        crop_w = right - left
        crop_h = bottom - top

        if crop_w <= 0 or crop_h <= 0:
            return ValidationResult(
                is_valid=False,
                requested_recipe=sanitized_requested,
                rejection_reason=f"Crop results in zero or negative pixel dimensions: {crop_w}x{crop_h}",
            )

        # Check export resolution: min 400px each dimension
        if crop_w < MIN_EXPORT_DIMENSION or crop_h < MIN_EXPORT_DIMENSION:
            return ValidationResult(
                is_valid=False,
                requested_recipe=sanitized_requested,
                rejection_reason=f"Crop resolution {crop_w}x{crop_h} below minimum pilot export requirement {MIN_EXPORT_DIMENSION}x{MIN_EXPORT_DIMENSION}px.",
            )

        # Check area ratio: min 30% of original
        crop_area = crop_w * crop_h
        total_area = image_width * image_height
        area_ratio = crop_area / total_area if total_area > 0 else 0.0

        if area_ratio < MIN_CROP_AREA_RATIO:
            return ValidationResult(
                is_valid=False,
                requested_recipe=sanitized_requested,
                rejection_reason=f"Crop area ratio {area_ratio:.3f} is below minimum allowed ratio {MIN_CROP_AREA_RATIO:.2f}.",
            )

        eff_crop = (ymin, xmin, ymax, xmax)

    effective_recipe = Recipe(
        pipeline_version=PIPELINE_VERSION,
        exposure=eff_exp,
        contrast=eff_con,
        saturation=eff_sat,
        crop=eff_crop,
    )

    return ValidationResult(
        is_valid=True,
        requested_recipe=sanitized_requested,
        effective_recipe=effective_recipe,
        warnings=warnings,
    )
