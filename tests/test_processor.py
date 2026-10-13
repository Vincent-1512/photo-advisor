"""Unit tests for Processor: EXIF Orientation, ICC, Format, Reproducibility, No-Overwrite."""

import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from PIL import Image, ImageCms
from photo_advisor.processor import normalize_input_image, process_image, compute_sha256
from photo_advisor.types import PIPELINE_VERSION, Recipe


class TestProcessor(unittest.TestCase):
    def test_exif_orientation_transpose(self):
        # Create an image with EXIF orientation 6 (rotate 90 CW)
        img = Image.new("RGB", (300, 200), color=(100, 150, 200))
        exif = img.getexif()
        exif[0x0112] = 6  # Orientation tag: 6 corresponds to 90 CW transpose
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)
        raw_bytes = buf.getvalue()

        norm_img, logs = normalize_input_image(raw_bytes)
        # Dimensions should be transposed to (200, 300)
        self.assertEqual(norm_img.size, (200, 300))
        self.assertTrue(any("EXIF orientation transpose" in l for l in logs))

    def test_icc_profile_conversion(self):
        img = Image.new("RGB", (500, 500), color=(120, 140, 160))
        srgb = ImageCms.createProfile("sRGB")
        icc_bytes = ImageCms.ImageCmsProfile(srgb).tobytes()

        buf = io.BytesIO()
        img.save(buf, format="JPEG", icc_profile=icc_bytes)
        raw_bytes = buf.getvalue()

        norm_img, logs = normalize_input_image(raw_bytes)
        self.assertEqual(norm_img.mode, "RGB")
        self.assertTrue(any("ICC profile" in l for l in logs))

    def test_corrupted_icc_profile_rejected(self):
        img = Image.new("RGB", (500, 500), color=(120, 140, 160))
        # Corrupted junk bytes
        bad_icc = b"INVALID_CORRUPTED_ICC_DATA_1234567890"

        buf = io.BytesIO()
        img.save(buf, format="JPEG", icc_profile=bad_icc)
        raw_bytes = buf.getvalue()

        with self.assertRaises(ValueError) as ctx:
            normalize_input_image(raw_bytes)
        self.assertIn("Corrupted or invalid embedded ICC profile", str(ctx.exception))

    def test_heic_unsupported_rejection(self):
        # Fake HEIC bytes with HEIC marker
        buf = io.BytesIO()
        # Create valid image but simulate HEIC container signature
        img = Image.new("RGB", (500, 500))
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        # Mock image.format to HEIC
        from unittest.mock import patch
        with patch.object(Image, "open") as mock_open:
            mock_img = Image.new("RGB", (500, 500))
            mock_img.format = "HEIC"
            mock_open.return_value = mock_img
            with self.assertRaises(ValueError) as ctx:
                normalize_input_image(raw_bytes)
            self.assertIn("HEIC/HEIF format is not currently supported", str(ctx.exception))

    def test_reproducibility_identical_sha256(self):
        # Same image processed twice with same recipe must yield identical output SHA256
        img = Image.new("RGB", (600, 600), color=(150, 150, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        recipe = Recipe(
            pipeline_version=PIPELINE_VERSION,
            exposure=0.1,
            contrast=1.05,
            saturation=1.05,
            crop=None,
        )

        res1 = process_image(raw_bytes, recipe)
        res2 = process_image(raw_bytes, recipe)

        self.assertEqual(res1.status, "success")
        self.assertEqual(res2.status, "success")
        self.assertEqual(res1.output_sha256, res2.output_sha256)
        self.assertTrue(len(res1.output_sha256) == 64)

    def test_export_does_not_overwrite_original(self):
        # Ensure processing pipeline never touches or overwrites input source
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new("RGB", (500, 500), color=(100, 100, 100))
            img.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            original_bytes = f.read()
        original_hash = compute_sha256(original_bytes)

        recipe = Recipe(
            pipeline_version=PIPELINE_VERSION,
            exposure=0.2,
            contrast=1.1,
            saturation=1.1,
            crop=None,
        )
        res = process_image(original_bytes, recipe)
        self.assertEqual(res.status, "success")

        # Verify original file content is unmodified
        with open(tmp_path, "rb") as f:
            current_bytes = f.read()
        current_hash = compute_sha256(current_bytes)

        self.assertEqual(original_hash, current_hash)
        os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
