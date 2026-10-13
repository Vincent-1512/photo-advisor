"""Unit tests for Rule Engine v0."""

import os
import sys
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from photo_advisor.rule_engine import compute_rule_v0_recipe
from photo_advisor.types import RULE_VERSION


class TestRuleEngine(unittest.TestCase):
    def test_low_light_image_gets_positive_exposure_compensation(self):
        # Create dark image with mean luma around 64
        arr = np.full((200, 200, 3), 64, dtype=np.uint8)
        img = Image.fromarray(arr)

        recipe, stats, logs = compute_rule_v0_recipe(img, goal="natural_portrait")
        # log2(128 / 64) = 1.0
        self.assertAlmostEqual(recipe.exposure, 1.0, places=2)
        self.assertEqual(recipe.saturation, 1.0)
        self.assertIsNone(recipe.crop)
        self.assertEqual(stats["rule_decision"], "luma_compensated")

    def test_overexposed_image_gets_negative_exposure_compensation(self):
        # Create bright image with mean luma around 200
        arr = np.full((200, 200, 3), 200, dtype=np.uint8)
        img = Image.fromarray(arr)

        recipe, stats, logs = compute_rule_v0_recipe(img, goal="natural_portrait")
        # log2(128 / 200) ~= -0.6438
        self.assertLess(recipe.exposure, 0.0)
        self.assertGreaterEqual(recipe.exposure, -1.0)

    def test_low_dynamic_range_triggers_contrast_boost(self):
        # Flat image where p95 - p5 < 80
        arr = np.full((200, 200, 3), 120, dtype=np.uint8)
        # Small spread +/- 20 -> dynamic range is ~40 (< 80)
        arr[:100, :, :] = 100
        arr[100:, :, :] = 140
        img = Image.fromarray(arr)

        recipe, stats, logs = compute_rule_v0_recipe(img, goal="natural_portrait")
        self.assertEqual(recipe.contrast, 1.05)
        self.assertLess(stats["dynamic_range"], 80.0)

    def test_broad_dynamic_range_keeps_normal_contrast(self):
        # Image with wide spread: 0 to 255 -> dynamic range > 80
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        arr[:100, :, :] = 20
        arr[100:, :, :] = 220
        img = Image.fromarray(arr)

        recipe, stats, logs = compute_rule_v0_recipe(img, goal="natural_portrait")
        self.assertEqual(recipe.contrast, 1.0)
        self.assertGreaterEqual(stats["dynamic_range"], 80.0)

    def test_silhouette_goal_returns_noop_recipe(self):
        # Even on dark image, goal 'silhouette' must yield No-op (0.0 exp, 1.0 con)
        arr = np.full((200, 200, 3), 40, dtype=np.uint8)
        img = Image.fromarray(arr)

        recipe, stats, logs = compute_rule_v0_recipe(img, goal="silhouette")
        self.assertEqual(recipe.exposure, 0.0)
        self.assertEqual(recipe.contrast, 1.0)
        self.assertEqual(recipe.saturation, 1.0)
        self.assertIsNone(recipe.crop)
        self.assertEqual(stats["rule_decision"], "silhouette_noop")


if __name__ == "__main__":
    unittest.main()
