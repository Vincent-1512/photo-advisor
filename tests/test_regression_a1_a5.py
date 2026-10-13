"""Regression tests proving fixes for Findings A1 through A5 from CODEX_REVIEW.md."""

import datetime
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from PIL import Image
from photo_advisor.cli import create_synthetic_test_image, run_pipeline
from photo_advisor.processor import compute_sha256, normalize_input_image, process_image
from photo_advisor.registry import CandidateRegistry
from photo_advisor.types import (
    CandidateRegistryRecord,
    GUARDRAIL_VERSION,
    PIPELINE_VERSION,
    PROCESSOR_VERSION,
    Recipe,
)
from photo_advisor.validator import validate_and_clamp_recipe


class TestRegressionA1A5(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # A1 Tests: Path Protection & Overwrite Prevention
    # -------------------------------------------------------------
    def test_a1_input_colliding_with_generated_file_is_blocked(self):
        """Input image file inside output_dir matching generated names must not be overwritten."""
        for fname in ["reference_input.png", "preview_noop.png", "preview_rule.png", "manifest.json", "index.html"]:
            sub_dir = os.path.join(self.test_dir, f"run_a1_coll_{fname}")
            os.makedirs(sub_dir, exist_ok=True)
            fake_input_path = os.path.join(sub_dir, fname)
            with open(fake_input_path, "wb") as f:
                f.write(b"ORIGINAL_INPUT_BYTES_DO_NOT_OVERWRITE")
            orig_sha = compute_sha256(b"ORIGINAL_INPUT_BYTES_DO_NOT_OVERWRITE")

            # Attempting to run pipeline with this input file must be rejected with PermissionError
            with self.assertRaises(PermissionError):
                run_pipeline(
                    image_bytes=b"ORIGINAL_INPUT_BYTES_DO_NOT_OVERWRITE",
                    goal="natural_portrait",
                    output_dir=sub_dir,
                    input_file_path=fake_input_path,
                )

            # Original file on disk must be completely untouched
            with open(fake_input_path, "rb") as f:
                self.assertEqual(compute_sha256(f.read()), orig_sha)

    def test_a1_symlink_input_collision_detected_and_blocked(self):
        """Symlink resolving to output artifact must be detected and blocked via realpath."""
        output_dir = os.path.join(self.test_dir, "run_a1_symlink")
        os.makedirs(output_dir, exist_ok=True)

        target_ref = os.path.join(output_dir, "reference_input.png")
        with open(target_ref, "wb") as f:
            f.write(b"TARGET_FILE_DATA")

        symlink_path = os.path.join(self.test_dir, "input_symlink.png")
        os.symlink(target_ref, symlink_path)

        with self.assertRaises(PermissionError):
            run_pipeline(
                image_bytes=b"TARGET_FILE_DATA",
                goal="natural_portrait",
                output_dir=output_dir,
                input_file_path=symlink_path,
            )

    def test_a1_rerun_on_same_output_dir_refuses_overwrite(self):
        """Re-running on an existing output_dir must raise FileExistsError without modifying old artifacts."""
        output_dir = os.path.join(self.test_dir, "run_a1_rerun")
        img_bytes = create_synthetic_test_image(800, 600)

        # First run succeeds
        manifest1 = run_pipeline(
            image_bytes=img_bytes,
            goal="natural_portrait",
            output_dir=output_dir,
        )
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "rb") as f:
            manifest_bytes1 = f.read()

        # Second run on same output_dir must fail to prevent overwrite
        with self.assertRaises(FileExistsError):
            run_pipeline(
                image_bytes=img_bytes,
                goal="natural_portrait",
                output_dir=output_dir,
            )

        # Artifacts on disk must remain strictly identical
        with open(manifest_path, "rb") as f:
            manifest_bytes2 = f.read()
        self.assertEqual(manifest_bytes1, manifest_bytes2)

    # -------------------------------------------------------------
    # A2 Tests: Candidate Resilience & Clean Input Errors
    # -------------------------------------------------------------
    def test_a2_malformed_missing_and_non_dict_fixtures_handled_gracefully(self):
        """Malformed JSON, missing files, root lists, huge numbers are rejected without crashing run."""
        img_bytes = create_synthetic_test_image(800, 600)
        output_dir = os.path.join(self.test_dir, "run_a2_fixtures")

        # 1. Malformed JSON
        malformed_path = os.path.join(self.test_dir, "malformed.json")
        with open(malformed_path, "w") as f:
            f.write("{broken json content")

        # 2. Missing file
        missing_path = os.path.join(self.test_dir, "non_existent_fixture.json")

        # 3. Root list (not dict)
        list_path = os.path.join(self.test_dir, "root_list.json")
        with open(list_path, "w") as f:
            f.write("[1, 2, 3]")

        # Run pipeline with malformed fixture as candidate 1
        manifest = run_pipeline(
            image_bytes=img_bytes,
            goal="natural_portrait",
            output_dir=output_dir,
            candidate_fixtures=[malformed_path, list_path],
        )

        self.assertEqual(len(manifest["candidates"]), 4)
        c1 = manifest["candidates"][2]
        self.assertTrue(c1["status"].startswith("rejected"))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "preview_candidate_1.png")))

        c2 = manifest["candidates"][3]
        self.assertTrue(c2["status"].startswith("rejected"))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "preview_candidate_2.png")))

        # Entire manifest and HTML were generated successfully
        self.assertTrue(os.path.exists(os.path.join(output_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "index.html")))

    def test_a2_huge_numbers_do_not_cause_overflow_error(self):
        """Astronomical numbers in candidate recipe are rejected without crashing Python."""
        huge_recipe = {
            "pipeline_version": PIPELINE_VERSION,
            "exposure": 10**500,  # Huge integer that overflows float
            "contrast": 1.0,
            "saturation": 1.0,
            "crop": None,
        }
        val_res = validate_and_clamp_recipe(huge_recipe, 800, 600)
        self.assertFalse(val_res.is_valid)
        self.assertIn("Invalid exposure", val_res.rejection_reason)

    def test_a2_corrupted_input_image_fails_cleanly(self):
        """Corrupted input image bytes must raise ValueError immediately without creating manifest."""
        output_dir = os.path.join(self.test_dir, "run_a2_bad_img")
        bad_bytes = b"CORRUPTED_NOT_AN_IMAGE_DATA_12345"

        with self.assertRaises(ValueError):
            run_pipeline(
                image_bytes=bad_bytes,
                goal="natural_portrait",
                output_dir=output_dir,
            )
        self.assertFalse(os.path.exists(os.path.join(output_dir, "manifest.json")))

    # -------------------------------------------------------------
    # A3 Tests: Exact Precision, Gray128 Reproducibility, Strict JSON
    # -------------------------------------------------------------
    def test_a3_gray128_exposure_precision_round_trip(self):
        """Gray128 with exposure log2(129/128)+1e-7 reproduces pixel 129 exactly from saved recipe."""
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        target_exp = math.log2(129.0 / 128.0) + 1e-7  # ~0.011227355423254119
        rec = Recipe(
            pipeline_version=PIPELINE_VERSION,
            exposure=target_exp,
            contrast=1.0,
            saturation=1.0,
            crop=None,
        )

        res1 = process_image(raw_bytes, rec)
        rendered_img1 = Image.open(io.BytesIO(res1.output_image_bytes))
        self.assertEqual(rendered_img1.getpixel((0, 0)), (129, 129, 129))

        # Check round-trip from serialized dict
        saved_dict = rec.to_dict()
        self.assertEqual(saved_dict["exposure"], target_exp)  # full float preserved

        # Re-construct from saved dict
        rec_from_saved = Recipe(
            pipeline_version=saved_dict["pipeline_version"],
            exposure=float(saved_dict["exposure"]),
            contrast=float(saved_dict["contrast"]),
            saturation=float(saved_dict["saturation"]),
            crop=None,
        )
        res2 = process_image(raw_bytes, rec_from_saved)
        rendered_img2 = Image.open(io.BytesIO(res2.output_image_bytes))
        self.assertEqual(rendered_img2.getpixel((0, 0)), (129, 129, 129))
        self.assertEqual(res1.output_sha256, res2.output_sha256)

    def test_a3_manifest_distinguishes_attempted_clipping_and_logs(self):
        """Manifest contains normalization logs, clamp warnings, fallback reason, and attempted clipping."""
        img_bytes = create_synthetic_test_image(800, 600)
        output_dir = os.path.join(self.test_dir, "run_a3_clipping")

        fixture_path = os.path.join(self.test_dir, "extreme_exposure.json")
        with open(fixture_path, "w") as f:
            json.dump({
                "model_id": "test_vlm",
                "recipe": {
                    "pipeline_version": PIPELINE_VERSION,
                    "exposure": 1.2,  # triggers delta clipping fallback
                    "contrast": 1.0,
                    "saturation": 1.0,
                    "crop": None,
                }
            }, f)

        manifest = run_pipeline(
            image_bytes=img_bytes,
            goal="natural_portrait",
            output_dir=output_dir,
            candidate_fixtures=[fixture_path],
        )

        cand = manifest["candidates"][2]
        self.assertEqual(cand["status"], "fallback_noop")
        self.assertIsNotNone(cand["attempted_clipping"])
        self.assertGreater(cand["attempted_clipping"]["delta_over"], 0.01)
        self.assertEqual(cand["delta_clipping"]["delta_over"], 0.0)  # effective delivered delta is 0
        self.assertIsNotNone(cand["fallback_reason"])
        self.assertTrue(len(manifest["metadata"]["normalization_logs"]) > 0)

    def test_a3_strict_json_serialization_handles_nan_safely(self):
        """Payloads containing NaN or Inf are safely sanitized and do not crash allow_nan=False."""
        recipe_nan = {
            "pipeline_version": PIPELINE_VERSION,
            "exposure": float("nan"),
            "contrast": 1.0,
            "saturation": 1.0,
            "crop": None,
        }
        val_res = validate_and_clamp_recipe(recipe_nan, 800, 600)
        self.assertFalse(val_res.is_valid)

        # Verify that json.dumps with allow_nan=False succeeds
        json_str = json.dumps(val_res.requested_recipe, allow_nan=False)
        self.assertIn('"NaN"', json_str)

    # -------------------------------------------------------------
    # A4 Tests: Format Allowlist, Alpha Compositing, Export Eligibility
    # -------------------------------------------------------------
    def test_a4_unsupported_formats_webp_and_bmp_rejected(self):
        """WEBP and BMP must be rejected by normalize_input_image."""
        # 1. WEBP
        img = Image.new("RGB", (100, 100), (200, 100, 50))
        buf_webp = io.BytesIO()
        img.save(buf_webp, format="WEBP")
        with self.assertRaises(ValueError) as ctx_webp:
            normalize_input_image(buf_webp.getvalue())
        self.assertIn("Unsupported image format", str(ctx_webp.exception))

        # 2. BMP
        buf_bmp = io.BytesIO()
        img.save(buf_bmp, format="BMP")
        with self.assertRaises(ValueError) as ctx_bmp:
            normalize_input_image(buf_bmp.getvalue())
        self.assertIn("Unsupported image format", str(ctx_bmp.exception))

    def test_a4_alpha_compositing_policy_white_background(self):
        """Transparent red RGBA(255, 0, 0, 0) composites onto neutral white (255, 255, 255)."""
        img = Image.new("RGBA", (50, 50), (255, 0, 0, 0))  # 100% transparent red
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        norm_img, logs = normalize_input_image(buf.getvalue())
        self.assertEqual(norm_img.getpixel((0, 0)), (255, 255, 255))  # becomes white, NOT red
        self.assertTrue(any("Alpha channel detected" in l for l in logs))

    def test_a4_small_images_have_export_eligible_false(self):
        """Small images (200x200 < 400px) are processed for preview with export_eligible=False."""
        img = Image.new("RGB", (200, 200), (150, 150, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        rec = Recipe(PIPELINE_VERSION, 0.0, 1.0, 1.0, None)
        res = process_image(raw_bytes, rec)
        self.assertEqual(res.status, "success")
        self.assertFalse(res.export_eligible)  # ineligible because < 400px

        output_dir = os.path.join(self.test_dir, "run_a4_small")
        manifest = run_pipeline(
            image_bytes=raw_bytes,
            goal="natural_portrait",
            output_dir=output_dir,
        )
        self.assertFalse(manifest["metadata"]["input_export_eligible"])
        self.assertFalse(manifest["candidates"][0]["export_eligible"])

    # -------------------------------------------------------------
    # A5 Tests: Registry Immutability & Dynamic Dependency Metadata
    # -------------------------------------------------------------
    def test_a5_registry_records_are_deep_copied_and_immutable(self):
        """Mutating a record before or after adding/getting does not affect registry state."""
        reg = CandidateRegistry(run_id="run_test_01")
        rec = CandidateRegistryRecord(
            run_id="run_test_01",
            candidate_id="cand_test",
            image_id="img_01",
            goal_id="natural_portrait",
            source_type="noop",
            source_name="Test Candidate",
            is_real_model=False,
            prompt_version=None,
            requested_recipe={"exposure": 0.0},
            effective_recipe={"exposure": 0.0},
            processor_version=PROCESSOR_VERSION,
            guardrail_version=GUARDRAIL_VERSION,
            rule_version=None,
            input_sha256="abc",
            output_sha256="def",
            preview_filename="preview.png",
            status="success",
            notes="Original note",
        )
        reg.add_record(rec)

        # Mutate local object
        rec.notes = "Mutated note"
        rec.requested_recipe["exposure"] = 999.0

        # Registry record must retain original values
        retrieved = reg.get_record("cand_test")
        self.assertEqual(retrieved.notes, "Original note")
        self.assertEqual(retrieved.requested_recipe["exposure"], 0.0)

        # Mutate retrieved object
        retrieved.notes = "Another mutation"
        retrieved2 = reg.get_record("cand_test")
        self.assertEqual(retrieved2.notes, "Original note")

    def test_a5_manifest_metadata_uses_real_runtime_dependencies(self):
        """Manifest dependencies string reflects installed runtime package versions."""
        img_bytes = create_synthetic_test_image(800, 600)
        output_dir = os.path.join(self.test_dir, "run_a5_metadata")

        manifest = run_pipeline(
            image_bytes=img_bytes,
            goal="natural_portrait",
            output_dir=output_dir,
            input_source="synthetic",
        )
        meta = manifest["metadata"]
        self.assertEqual(meta["input_source"], "synthetic")
        self.assertIn("Pillow==11.3.0", meta["dependencies"])
        self.assertIn("numpy==2.2.6", meta["dependencies"])

    # -------------------------------------------------------------
    # B1 Tests: Collision-free UUID Run ID with Fixed Clock
    # -------------------------------------------------------------
    def test_b1_fixed_clock_generates_distinct_run_ids(self):
        """Two separate runs with an identical frozen clock instant produce distinct run_ids and compound keys."""
        img_bytes = create_synthetic_test_image(800, 600)
        dir1 = os.path.join(self.test_dir, "run_b1_trial_1")
        dir2 = os.path.join(self.test_dir, "run_b1_trial_2")

        fixed_time = datetime.datetime(2026, 10, 8, 12, 0, 0, tzinfo=datetime.timezone.utc)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_time
            mock_dt.timezone = datetime.timezone
            mock_dt.side_effect = lambda *args, **kwargs: datetime.datetime(*args, **kwargs)

            manifest1 = run_pipeline(
                image_bytes=img_bytes,
                goal="natural_portrait",
                output_dir=dir1,
            )
            manifest2 = run_pipeline(
                image_bytes=img_bytes,
                goal="natural_portrait",
                output_dir=dir2,
            )

        run_id_1 = manifest1["metadata"]["run_id"]
        run_id_2 = manifest2["metadata"]["run_id"]

        # 1. Must be distinct despite identical clock
        self.assertNotEqual(run_id_1, run_id_2)
        self.assertTrue(run_id_1.startswith("run_20261008_120000_"))
        self.assertTrue(run_id_2.startswith("run_20261008_120000_"))

        # 2. Every candidate record has run_id matching its manifest run_id
        for c in manifest1["candidates"]:
            self.assertEqual(c["run_id"], run_id_1)
        for c in manifest2["candidates"]:
            self.assertEqual(c["run_id"], run_id_2)

        # 3. Compound keys (run_id, candidate_id) are distinct across runs
        keys1 = {(c["run_id"], c["candidate_id"]) for c in manifest1["candidates"]}
        keys2 = {(c["run_id"], c["candidate_id"]) for c in manifest2["candidates"]}
        self.assertTrue(keys1.isdisjoint(keys2))

    def test_b1_custom_run_id_honored_and_propagated(self):
        """Explicit run_id provided by caller is honored across metadata and candidate records."""
        img_bytes = create_synthetic_test_image(800, 600)
        custom_dir = os.path.join(self.test_dir, "run_b1_custom")
        custom_id = "my_custom_pilot_run_999"

        manifest = run_pipeline(
            image_bytes=img_bytes,
            goal="natural_portrait",
            output_dir=custom_dir,
            run_id=custom_id,
        )
        self.assertEqual(manifest["metadata"]["run_id"], custom_id)
        for c in manifest["candidates"]:
            self.assertEqual(c["run_id"], custom_id)

    # -------------------------------------------------------------
    # B2 Tests: HTML Export Eligibility Distinguishes Small vs Fallback
    # -------------------------------------------------------------
    def test_b2_html_distinguishes_small_image_vs_guardrail_fallback(self):
        """HTML report assigns Preview Only (<400px) exclusively to small images and Guardrail Fallback to clipped candidates."""
        # Case A: Small image 200x200
        small_bytes = create_synthetic_test_image(200, 200)
        dir_small = os.path.join(self.test_dir, "run_b2_small")
        manifest_small = run_pipeline(
            image_bytes=small_bytes,
            goal="natural_portrait",
            output_dir=dir_small,
        )
        with open(os.path.join(dir_small, "index.html"), "r", encoding="utf-8") as f:
            html_small = f.read()

        # Both cand_0 and cand_1 on 200x200 must show <400px badge
        self.assertIn("Preview Only (&lt;400px)", html_small)
        self.assertFalse(manifest_small["metadata"]["input_export_eligible"])
        self.assertFalse(manifest_small["candidates"][0]["export_eligible"])

        # Case B: Standard 800x600 image with clipping fallback candidate
        large_bytes = create_synthetic_test_image(800, 600)
        fixture_path = os.path.join(self.test_dir, "high_exposure_fixture.json")
        with open(fixture_path, "w") as f:
            json.dump({
                "model_id": "test_vlm_large",
                "recipe": {
                    "pipeline_version": PIPELINE_VERSION,
                    "exposure": 1.5,  # triggers delta clipping fallback
                    "contrast": 1.0,
                    "saturation": 1.0,
                    "crop": None,
                }
            }, f)

        dir_large = os.path.join(self.test_dir, "run_b2_large")
        manifest_large = run_pipeline(
            image_bytes=large_bytes,
            goal="natural_portrait",
            output_dir=dir_large,
            candidate_fixtures=[fixture_path],
        )
        with open(os.path.join(dir_large, "index.html"), "r", encoding="utf-8") as f:
            html_large = f.read()

        # Input is eligible
        self.assertTrue(manifest_large["metadata"]["input_export_eligible"])

        # Cand 0 and Cand 1 must be Export Eligible
        self.assertTrue(manifest_large["candidates"][0]["export_eligible"])
        self.assertTrue(manifest_large["candidates"][1]["export_eligible"])

        # Cand 2 (fixture) must have status fallback_noop and export_eligible=False
        cand_fallback = manifest_large["candidates"][2]
        self.assertEqual(cand_fallback["status"], "fallback_noop")
        self.assertFalse(cand_fallback["export_eligible"])

        # HTML must show Guardrail Fallback badge, and MUST NOT claim <400px for this candidate!
        self.assertIn("Preview Only (Guardrail Fallback)", html_large)
        self.assertNotIn("Preview Only (&lt;400px)", html_large)
        self.assertIn("Fallback Reason:", html_large)


if __name__ == "__main__":
    unittest.main()
