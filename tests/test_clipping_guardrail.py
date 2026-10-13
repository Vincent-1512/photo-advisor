"""Unit tests for Delta Clipping Guardrail v0 and Fallback to No-op."""

import io
import os
import sys
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from photo_advisor.processor import process_image, calculate_clipping_fractions
from photo_advisor.types import PIPELINE_VERSION, Recipe


class TestClippingGuardrail(unittest.TestCase):
    def test_clipping_calculation_fractions(self):
        # Create image with known clipped pixels: 10% black, 10% white, 80% mid-gray
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        arr[:10, :, :] = 0     # 10% under
        arr[90:, :, :] = 255   # 10% over
        img = Image.fromarray(arr)

        stats = calculate_clipping_fractions(img)
        self.assertAlmostEqual(stats["fraction_under"], 0.10, places=2)
        self.assertAlmostEqual(stats["fraction_over"], 0.10, places=2)

    def test_excessive_exposure_triggers_delta_clipping_fallback(self):
        # Create mid-gray image with light pixels near 240
        arr = np.full((500, 500, 3), 240, dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        # Aggressive brightening recipe (+1.5 EV) will push all 240 pixels to 255 (overexposed)
        # Delta over will increase by ~1.0 >> 0.01 threshold!
        recipe = Recipe(
            pipeline_version=PIPELINE_VERSION,
            exposure=1.5,
            contrast=1.0,
            saturation=1.0,
            crop=None,
        )

        res = process_image(raw_bytes, recipe)
        # Must trigger fallback_noop
        self.assertEqual(res.status, "fallback_noop")
        self.assertIsNotNone(res.fallback_reason)
        self.assertIn("Delta clipping guardrail exceeded threshold", res.fallback_reason)
        # Effective recipe should be set to No-op
        self.assertEqual(res.effective_recipe.exposure, 0.0)
        self.assertEqual(res.effective_recipe.contrast, 1.0)
        # Attempted clipping exceeded threshold, while effective returned image clipping delta is 0
        self.assertIsNotNone(res.attempted_clipping)
        self.assertGreater(res.attempted_clipping["delta_over"], 0.01)
        self.assertEqual(res.delta_clipping["delta_over"], 0.0)

    def test_moderate_adjustment_within_delta_threshold_succeeds(self):
        # Safe mid-gray image (values around 128)
        arr = np.full((500, 500, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        # Mild adjustment (+0.2 EV) leaves values around ~147, far from 0 or 255
        recipe = Recipe(
            pipeline_version=PIPELINE_VERSION,
            exposure=0.2,
            contrast=1.05,
            saturation=1.05,
            crop=None,
        )

        res = process_image(raw_bytes, recipe)
        self.assertEqual(res.status, "success")
        self.assertIsNone(res.fallback_reason)
        self.assertAlmostEqual(res.delta_clipping["delta_under"], 0.0, places=4)
        self.assertAlmostEqual(res.delta_clipping["delta_over"], 0.0, places=4)

    def test_delta_clipping_compared_on_same_cropped_region(self):
        # Image has white border (clipped) but dark center
        arr = np.zeros((1000, 1000, 3), dtype=np.uint8)
        arr[:, :100, :] = 255  # Left border white
        arr[:, 900:, :] = 255  # Right border white
        arr[200:800, 200:800, :] = 120  # Center is mid-gray

        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        # Crop only the center [0.2, 0.2, 0.8, 0.8] (no white border)
        crop_box = (0.2, 0.2, 0.8, 0.8)
        recipe = Recipe(
            pipeline_version=PIPELINE_VERSION,
            exposure=0.1,
            contrast=1.0,
            saturation=1.0,
            crop=crop_box,
        )

        # Baseline clipping must be computed on cropped center (0% white), NOT full image (20% white)
        res = process_image(raw_bytes, recipe)
        self.assertEqual(res.status, "success")
        # Delta over should be 0 because center didn't have white and mild exposure didn't create white
        self.assertLessEqual(res.delta_clipping["delta_over"], 0.01)


if __name__ == "__main__":
    unittest.main()
