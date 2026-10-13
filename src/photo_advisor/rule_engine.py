"""Rule-based Engine v0.

Strictly adheres to Appendix Item #2 in CODEX_REVIEW_03.md:
- Encoded luma Y = 0.299R + 0.587G + 0.114B, mean m.
- exposure = clip(log2(128.0 / max(m, 1.0)), -1.0, 1.0).
- contrast = 1.05 if percentile95(Y) - percentile5(Y) < 80.0 else 1.0 (numpy method='linear').
- saturation = 1.0.
- crop = None.
- Goal 'silhouette': returns No-op recipe.
- Logs rule_version ('rule_v0') and metadata.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from photo_advisor.types import (
    GOAL_SILHOUETTE,
    PIPELINE_VERSION,
    RULE_VERSION,
    Recipe,
)


def compute_rule_v0_recipe(
    norm_img: Image.Image,
    goal: Optional[str] = None,
) -> Tuple[Recipe, Dict[str, Any], List[str]]:
    """Compute Rule v0 recipe based on image luma statistics.

    Args:
        norm_img: Normalized upright sRGB 8-bit image.
        goal: Style goal (e.g. 'silhouette', 'natural_portrait').

    Returns:
        Tuple of (Recipe, stats_dict, logs).
    """
    logs: List[str] = [f"Rule Engine Version: {RULE_VERSION}"]

    # 1. Silhouette check: respect artistic intention, return No-op
    if goal and goal.lower() == GOAL_SILHOUETTE:
        logs.append("Goal is 'silhouette': Returning No-op recipe to respect dark silhouette contrast.")
        recipe = Recipe(
            pipeline_version=PIPELINE_VERSION,
            exposure=0.0,
            contrast=1.0,
            saturation=1.0,
            crop=None,
        )
        stats = {
            "mean_luma": None,
            "p5_luma": None,
            "p95_luma": None,
            "dynamic_range": None,
            "goal": goal,
            "rule_decision": "silhouette_noop",
        }
        return recipe, stats, logs

    # 2. Compute encoded luma Y = 0.299R + 0.587G + 0.114B
    arr = np.asarray(norm_img.convert("RGB"), dtype=np.float32)
    Y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mean_luma = float(np.mean(Y))

    # Exposure calculation: log2(128.0 / max(m, 1.0)) clipped to [-1.0, 1.0]
    safe_m = max(mean_luma, 1.0)
    raw_exp = math.log2(128.0 / safe_m)
    clipped_exp = max(-1.0, min(1.0, raw_exp))
    logs.append(
        f"Mean luma m={mean_luma:.2f}. Raw exposure={raw_exp:.4f}, clipped exposure={clipped_exp:.4f}."
    )

    # 3. Contrast calculation: percentile95(Y) - percentile5(Y) < 80.0
    p95 = float(np.percentile(Y, 95, method="linear"))
    p5 = float(np.percentile(Y, 5, method="linear"))
    dynamic_range = p95 - p5

    if dynamic_range < 80.0:
        contrast = 1.05
        logs.append(
            f"Percentile 95 ({p95:.2f}) - Percentile 5 ({p5:.2f}) = {dynamic_range:.2f} < 80.0. Boosted contrast to 1.05."
        )
    else:
        contrast = 1.0
        logs.append(
            f"Percentile 95 ({p95:.2f}) - Percentile 5 ({p5:.2f}) = {dynamic_range:.2f} >= 80.0. Contrast kept at 1.0."
        )

    # 4. Saturation and Crop defaults
    saturation = 1.0
    crop = None

    recipe = Recipe(
        pipeline_version=PIPELINE_VERSION,
        exposure=clipped_exp,
        contrast=contrast,
        saturation=saturation,
        crop=crop,
    )

    stats = {
        "mean_luma": round(mean_luma, 2),
        "p5_luma": round(p5, 2),
        "p95_luma": round(p95, 2),
        "dynamic_range": round(dynamic_range, 2),
        "percentile_method": "numpy_linear",
        "goal": goal,
        "rule_decision": "luma_compensated",
    }

    return recipe, stats, logs
