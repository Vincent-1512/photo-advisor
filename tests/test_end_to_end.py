"""End-to-End integration tests for Phase 02A Photo Advisor pipeline."""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from PIL import Image
from photo_advisor.cli import create_synthetic_test_image, main, run_pipeline
from photo_advisor.processor import compute_sha256


class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_synthetic_run_generates_all_artifacts(self):
        img_bytes = create_synthetic_test_image(800, 600, style="natural_portrait")
        output_dir = os.path.join(self.test_dir, "run_synth")

        manifest = run_pipeline(
            image_bytes=img_bytes,
            goal="natural_portrait",
            output_dir=output_dir,
            candidate_fixtures=None,
            image_id="synthetic_01",
        )

        # Check files created
        self.assertTrue(os.path.exists(os.path.join(output_dir, "reference_input.png")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "preview_noop.png")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "preview_rule.png")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "index.html")))

        # Verify manifest content
        self.assertIn("metadata", manifest)
        self.assertEqual(manifest["metadata"]["goal"], "natural_portrait")
        self.assertEqual(manifest["metadata"]["image_id"], "synthetic_01")
        self.assertIn("total_runtime_ms", manifest["metadata"])
        self.assertEqual(len(manifest["candidates"]), 2)

        # Verify candidate 0 (No-op)
        cand_0 = manifest["candidates"][0]
        self.assertEqual(cand_0["candidate_id"], "cand_0_noop")
        self.assertEqual(cand_0["status"], "success")

        # Verify file hash matches manifest output_sha256
        with open(os.path.join(output_dir, cand_0["preview_filename"]), "rb") as f:
            actual_sha = compute_sha256(f.read())
        self.assertEqual(actual_sha, cand_0["output_sha256"])

        # Verify candidate 1 (Rule v0)
        cand_1 = manifest["candidates"][1]
        self.assertEqual(cand_1["candidate_id"], "cand_1_rule_v0")
        self.assertIn(cand_1["status"], ["success", "fallback_noop"])
        with open(os.path.join(output_dir, cand_1["preview_filename"]), "rb") as f:
            cand1_sha = compute_sha256(f.read())
        self.assertEqual(cand1_sha, cand_1["output_sha256"])

        # Verify HTML content
        with open(os.path.join(output_dir, "index.html"), "r", encoding="utf-8") as f:
            html_text = f.read()
        self.assertIn("Photo Advisor Phase 02A - Offline Pipeline Verification", html_text)
        self.assertIn("reference_input.png", html_text)
        self.assertIn("cand_0_noop", html_text)
        self.assertIn("cand_1_rule_v0", html_text)
        self.assertIn("Offline / Fixture", html_text)

    def test_pipeline_with_valid_and_invalid_fixtures(self):
        img_bytes = create_synthetic_test_image(800, 600, style="natural_portrait")
        output_dir = os.path.join(self.test_dir, "run_fixtures")

        # Create valid fixture
        valid_fixture_path = os.path.join(self.test_dir, "valid_fixture.json")
        valid_fixture_data = {
            "model_id": "gemini-2.5-flash-test",
            "prompt_version": "prompt_v0",
            "recipe": {
                "pipeline_version": "pipeline_v0",
                "exposure": -0.1,
                "contrast": 1.02,
                "saturation": 1.02,
                "crop": [0.05, 0.05, 0.95, 0.95],
            },
            "advice_text": "Giảm nhẹ độ sáng để giữ chi tiết vùng sáng và crop gọn.",
        }
        with open(valid_fixture_path, "w", encoding="utf-8") as f:
            json.dump(valid_fixture_data, f)

        # Create invalid fixture (violates schema with disallowed types and extreme crop)
        invalid_fixture_path = os.path.join(self.test_dir, "invalid_fixture.json")
        invalid_fixture_data = {
            "model_id": "test-invalid-model",
            "prompt_version": "prompt_v0",
            "recipe": {
                "pipeline_version": "pipeline_v0",
                "exposure": 5.0,  # exceeds allowed [-1, 1]
                "contrast": 1.0,
                "saturation": 1.0,
                "crop": [0.0, 0.0, 0.1, 0.1],  # area = 0.01 < 0.30
            },
            "advice_text": "Invalid recipe",
        }
        with open(invalid_fixture_path, "w", encoding="utf-8") as f:
            json.dump(invalid_fixture_data, f)

        manifest = run_pipeline(
            image_bytes=img_bytes,
            goal="natural_portrait",
            output_dir=output_dir,
            candidate_fixtures=[valid_fixture_path, invalid_fixture_path],
            image_id="test_fixtures",
        )

        self.assertEqual(len(manifest["candidates"]), 4)

        cand_valid = manifest["candidates"][2]
        self.assertEqual(cand_valid["candidate_id"], "cand_2_fixture")
        self.assertEqual(cand_valid["status"], "success")
        self.assertTrue(os.path.exists(os.path.join(output_dir, "preview_candidate_1.png")))

        cand_invalid = manifest["candidates"][3]
        self.assertEqual(cand_invalid["candidate_id"], "cand_3_fixture")
        self.assertTrue("rejected" in cand_invalid["status"])
        self.assertTrue(os.path.exists(os.path.join(output_dir, "preview_candidate_2.png")))

    def test_original_image_never_overwritten(self):
        input_image_path = os.path.join(self.test_dir, "user_photo.jpg")
        img = Image.new("RGB", (600, 500), color=(140, 150, 160))
        img.save(input_image_path, format="JPEG")

        with open(input_image_path, "rb") as f:
            original_bytes = f.read()
        original_sha = compute_sha256(original_bytes)
        original_mtime = os.path.getmtime(input_image_path)

        output_dir = os.path.join(self.test_dir, "run_user")

        manifest = run_pipeline(
            image_bytes=original_bytes,
            goal="natural_portrait",
            output_dir=output_dir,
            image_id="user_photo.jpg",
        )

        # Verify original file on disk was not modified in any way
        with open(input_image_path, "rb") as f:
            after_bytes = f.read()
        self.assertEqual(compute_sha256(after_bytes), original_sha)
        self.assertEqual(os.path.getmtime(input_image_path), original_mtime)

        # Output dir has reference_input.png instead
        self.assertTrue(os.path.exists(os.path.join(output_dir, "reference_input.png")))

    def test_cli_main_synthetic_invocation(self):
        output_dir = os.path.join(self.test_dir, "cli_output")
        test_args = [
            "photo_advisor.cli",
            "--synthetic",
            "--goal",
            "natural_portrait",
            "--output-dir",
            output_dir,
        ]
        with patch.object(sys, "argv", test_args):
            main()

        self.assertTrue(os.path.exists(os.path.join(output_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "index.html")))


if __name__ == "__main__":
    unittest.main()
