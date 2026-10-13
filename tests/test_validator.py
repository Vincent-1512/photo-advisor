"""Unit tests for Recipe Validator and Guardrail Schema."""

import math
import sys
import os
import unittest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from photo_advisor.types import (
    PIPELINE_VERSION,
    EXPOSURE_CLAMP_MIN,
    EXPOSURE_CLAMP_MAX,
    CONTRAST_CLAMP_MIN,
    CONTRAST_CLAMP_MAX,
    SATURATION_CLAMP_MIN,
    SATURATION_CLAMP_MAX,
)
from photo_advisor.validator import validate_and_clamp_recipe


class TestRecipeValidator(unittest.TestCase):
    def setUp(self):
        self.valid_recipe = {
            "pipeline_version": PIPELINE_VERSION,
            "exposure": 0.2,
            "contrast": 1.05,
            "saturation": 1.1,
            "crop": None,
        }
        self.img_w = 1200
        self.img_h = 800

    def test_valid_recipe_passes(self):
        res = validate_and_clamp_recipe(self.valid_recipe, self.img_w, self.img_h)
        self.assertTrue(res.is_valid)
        self.assertIsNotNone(res.effective_recipe)
        self.assertEqual(res.effective_recipe.exposure, 0.2)
        self.assertEqual(res.effective_recipe.contrast, 1.05)
        self.assertEqual(res.effective_recipe.saturation, 1.1)
        self.assertIsNone(res.effective_recipe.crop)

    def test_reject_bool_as_numeric(self):
        # Python bool is subclass of int, but bool must be rejected!
        bad_recipe = dict(self.valid_recipe)
        bad_recipe["exposure"] = True
        res = validate_and_clamp_recipe(bad_recipe, self.img_w, self.img_h)
        self.assertFalse(res.is_valid)
        self.assertIn("bool forbidden", res.rejection_reason)

        bad_recipe["exposure"] = 0.0
        bad_recipe["contrast"] = False
        res2 = validate_and_clamp_recipe(bad_recipe, self.img_w, self.img_h)
        self.assertFalse(res2.is_valid)

    def test_reject_non_finite_float(self):
        # NaN and Infinity must be rejected
        for bad_val in [float("nan"), float("inf"), float("-inf")]:
            bad_recipe = dict(self.valid_recipe)
            bad_recipe["contrast"] = bad_val
            res = validate_and_clamp_recipe(bad_recipe, self.img_w, self.img_h)
            self.assertFalse(res.is_valid)
            self.assertIn("finite number", res.rejection_reason)

    def test_reject_unknown_keys(self):
        bad_recipe = dict(self.valid_recipe)
        bad_recipe["unknown_key"] = "malicious_or_accidental"
        res = validate_and_clamp_recipe(bad_recipe, self.img_w, self.img_h)
        self.assertFalse(res.is_valid)
        self.assertIn("Unknown keys", res.rejection_reason)

    def test_reject_missing_required_keys(self):
        bad_recipe = dict(self.valid_recipe)
        del bad_recipe["saturation"]
        res = validate_and_clamp_recipe(bad_recipe, self.img_w, self.img_h)
        self.assertFalse(res.is_valid)
        self.assertIn("Missing required recipe keys", res.rejection_reason)

    def test_reject_unsupported_version(self):
        bad_recipe = dict(self.valid_recipe)
        bad_recipe["pipeline_version"] = "pipeline_v99"
        res = validate_and_clamp_recipe(bad_recipe, self.img_w, self.img_h)
        self.assertFalse(res.is_valid)
        self.assertIn("Unsupported pipeline_version", res.rejection_reason)

    def test_clamping_finite_scalars_with_warnings(self):
        # Exceeding upper clamp bounds
        over_recipe = {
            "pipeline_version": PIPELINE_VERSION,
            "exposure": 3.5,     # Max is 1.5
            "contrast": 2.0,     # Max is 1.3
            "saturation": 2.5,   # Max is 1.3
            "crop": None,
        }
        res = validate_and_clamp_recipe(over_recipe, self.img_w, self.img_h)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.effective_recipe.exposure, EXPOSURE_CLAMP_MAX)
        self.assertEqual(res.effective_recipe.contrast, CONTRAST_CLAMP_MAX)
        self.assertEqual(res.effective_recipe.saturation, SATURATION_CLAMP_MAX)
        self.assertEqual(len(res.warnings), 3)

        # Exceeding lower clamp bounds
        under_recipe = {
            "pipeline_version": PIPELINE_VERSION,
            "exposure": -5.0,    # Min is -1.5
            "contrast": 0.2,     # Min is 0.7
            "saturation": 0.1,   # Min is 0.8
            "crop": None,
        }
        res2 = validate_and_clamp_recipe(under_recipe, self.img_w, self.img_h)
        self.assertTrue(res2.is_valid)
        self.assertEqual(res2.effective_recipe.exposure, EXPOSURE_CLAMP_MIN)
        self.assertEqual(res2.effective_recipe.contrast, CONTRAST_CLAMP_MIN)
        self.assertEqual(res2.effective_recipe.saturation, SATURATION_CLAMP_MIN)

    def test_valid_crop(self):
        # Valid crop: 60% width x 60% height -> 36% area (>= 30%), dimensions 720x480 (>= 400)
        crop_recipe = dict(self.valid_recipe)
        crop_recipe["crop"] = [0.1, 0.2, 0.7, 0.8]
        res = validate_and_clamp_recipe(crop_recipe, self.img_w, self.img_h)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.effective_recipe.crop, (0.1, 0.2, 0.7, 0.8))

    def test_reject_invalid_crop_coordinates(self):
        # Inverted coords (ymin >= ymax)
        bad_recipe = dict(self.valid_recipe)
        bad_recipe["crop"] = [0.8, 0.2, 0.2, 0.8]
        res = validate_and_clamp_recipe(bad_recipe, self.img_w, self.img_h)
        self.assertFalse(res.is_valid)
        self.assertIn("Invalid crop bounds", res.rejection_reason)

        # Out of bounds (> 1.0)
        bad_recipe["crop"] = [0.0, 0.0, 1.2, 0.8]
        res2 = validate_and_clamp_recipe(bad_recipe, self.img_w, self.img_h)
        self.assertFalse(res2.is_valid)

    def test_reject_crop_below_minimum_area_ratio(self):
        # Tiny crop: 10% x 10% = 1% area (< 30%), dimensions 500x500 (>= 400px)
        tiny_crop = dict(self.valid_recipe)
        tiny_crop["crop"] = [0.1, 0.1, 0.2, 0.2]
        res = validate_and_clamp_recipe(tiny_crop, 5000, 5000)
        self.assertFalse(res.is_valid)
        self.assertIn("minimum allowed ratio", res.rejection_reason)

    def test_reject_crop_below_minimum_resolution(self):
        # Crop on small image resulting in < 400px
        crop_recipe = dict(self.valid_recipe)
        crop_recipe["crop"] = [0.0, 0.0, 0.6, 0.6]
        res = validate_and_clamp_recipe(crop_recipe, 500, 500)
        # 500 * 0.6 = 300px < 400px
        self.assertFalse(res.is_valid)
        self.assertIn("below minimum pilot export requirement", res.rejection_reason)

    def test_reject_crop_when_goal_requires_full_body_without_trusted_bbox(self):
        crop_recipe = dict(self.valid_recipe)
        crop_recipe["crop"] = [0.1, 0.2, 0.8, 0.8]
        res = validate_and_clamp_recipe(
            crop_recipe,
            self.img_w,
            self.img_h,
            goal="full_body",
            has_trusted_bbox=False,
        )
        self.assertFalse(res.is_valid)
        self.assertIn("requires preserving full body or context", res.rejection_reason)


if __name__ == "__main__":
    unittest.main()
