"""Unit and integration tests for Phase 02B VLM Adapter, Providers, Schema, and CLI."""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from photo_advisor.cli import create_synthetic_test_image, run_pipeline
from photo_advisor.types import PIPELINE_VERSION
from photo_advisor.vlm_adapter import (
    LocalTransformersProvider,
    MockVLMProvider,
    ReplayVLMProvider,
    VLMAdapter,
    VLMRequest,
    validate_vlm_schema,
)


class TestVLMPhase02B(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_img_bytes = create_synthetic_test_image(800, 600, style="natural_portrait")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. Payload validation
    def test_mock_provider_receives_real_image_payload(self):
        """Mock provider verifies that real image bytes, dimensions, SHA-256, and goal are transmitted."""
        provider = MockVLMProvider()
        resp = VLMAdapter.process(
            image_bytes=self.sample_img_bytes,
            goal="natural_portrait",
            provider=provider,
            image_id="test_img_01",
        )
        self.assertEqual(resp.status, "success")
        self.assertIsNotNone(provider.last_received_request)
        req = provider.last_received_request
        self.assertEqual(req.image_id, "test_img_01")
        self.assertEqual(req.goal, "natural_portrait")
        self.assertEqual(len(req.image_bytes), len(self.sample_img_bytes))
        self.assertEqual(req.image_dimensions, (800, 600))
        self.assertTrue(len(req.image_sha256) == 64)

    # 2. Schema validation
    def test_schema_validation_valid_response(self):
        """Valid response dictionary passes schema validation without errors."""
        valid_data = {
            "analysis": {
                "framing": "medium_shot",
                "subject_description": "Một người đứng giữa khung hình.",
                "background_description": "Công viên nhiều cây xanh.",
                "lighting_condition": "well_lit",
                "uncertainty": None,
            },
            "keep_original": {
                "recommended": False,
                "reason": "Ảnh hơi thiếu sáng nhẹ.",
            },
            "recommendations": [
                {
                    "option_id": "option_1",
                    "name": "Chân dung tự nhiên",
                    "rationale": "Tăng nhẹ sáng và tương phản.",
                    "recipe": {
                        "pipeline_version": PIPELINE_VERSION,
                        "exposure": 0.15,
                        "contrast": 1.05,
                        "saturation": 1.02,
                        "crop": [0.05, 0.05, 0.95, 0.95],
                    },
                }
            ],
            "retake_advice": "Đứng đón sáng nhẹ để khuôn mặt rõ hơn.",
        }
        is_valid, errors = validate_vlm_schema(valid_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_schema_validation_rejects_more_than_two_options(self):
        """Schema strictly rejects responses proposing more than 2 recommendation options."""
        too_many_options = {
            "analysis": {
                "framing": "medium_shot",
                "subject_description": "Chủ thể.",
                "background_description": "Hậu cảnh.",
                "lighting_condition": "well_lit",
                "uncertainty": None,
            },
            "keep_original": {"recommended": False, "reason": "Cần chỉnh"},
            "recommendations": [
                {"option_id": "option_1", "name": "Opt 1", "rationale": "r1", "recipe": {"pipeline_version": PIPELINE_VERSION, "exposure": 0.1, "contrast": 1.0, "saturation": 1.0, "crop": None}},
                {"option_id": "option_2", "name": "Opt 2", "rationale": "r2", "recipe": {"pipeline_version": PIPELINE_VERSION, "exposure": 0.2, "contrast": 1.0, "saturation": 1.0, "crop": None}},
                {"option_id": "option_3", "name": "Opt 3", "rationale": "r3", "recipe": {"pipeline_version": PIPELINE_VERSION, "exposure": 0.3, "contrast": 1.0, "saturation": 1.0, "crop": None}},
            ],
            "retake_advice": None,
        }
        is_valid, errors = validate_vlm_schema(too_many_options)
        self.assertFalse(is_valid)
        self.assertTrue(any("exceeds maximum allowed options" in e for e in errors))

    def test_schema_validation_rejects_invalid_framing_or_missing_keys(self):
        """Schema rejects invalid enum values for framing or missing keys."""
        bad_framing = {
            "analysis": {
                "framing": "extreme_bird_eye_drone",  # invalid
                "subject_description": "Chủ thể",
                "background_description": "Nền",
                "lighting_condition": "well_lit",
                "uncertainty": None,
            },
            "keep_original": {"recommended": True, "reason": "Tốt"},
            "recommendations": [],
            "retake_advice": None,
        }
        is_valid, errors = validate_vlm_schema(bad_framing)
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid framing" in e for e in errors))

    def test_schema_validation_rejects_unsupported_recipe_version(self):
        """Schema rejects recipes with unsupported pipeline_version or missing numeric parameters."""
        bad_recipe = {
            "analysis": {
                "framing": "close_up",
                "subject_description": "Chân dung",
                "background_description": "Nền mờ",
                "lighting_condition": "well_lit",
                "uncertainty": None,
            },
            "keep_original": {"recommended": False, "reason": "Chỉnh"},
            "recommendations": [
                {
                    "option_id": "option_1",
                    "name": "Bad version",
                    "rationale": "Sai version",
                    "recipe": {
                        "pipeline_version": "pipeline_v999",  # unsupported
                        "exposure": 0.1,
                        "contrast": 1.0,
                        "saturation": 1.0,
                    },
                }
            ],
            "retake_advice": None,
        }
        is_valid, errors = validate_vlm_schema(bad_recipe)
        self.assertFalse(is_valid)
        self.assertTrue(any("pipeline_version must be 'pipeline_v0'" in e for e in errors))

    # 3. Provider error simulation modes
    def test_mock_simulation_refusal_mode(self):
        """Mock provider in refusal mode returns status='refusal' without calling network."""
        provider = MockVLMProvider(simulation_mode="refusal")
        resp = VLMAdapter.process(self.sample_img_bytes, "natural_portrait", provider=provider)
        self.assertEqual(resp.status, "refusal")
        self.assertIsNotNone(resp.error_reason)
        self.assertFalse(resp.is_real_model)

    def test_mock_simulation_timeout_mode(self):
        """Mock provider in timeout mode returns status='timeout'."""
        provider = MockVLMProvider(simulation_mode="timeout")
        resp = VLMAdapter.process(self.sample_img_bytes, "natural_portrait", provider=provider)
        self.assertEqual(resp.status, "timeout")
        self.assertIn("timed out", resp.error_reason)

    def test_mock_simulation_truncated_json_mode(self):
        """Mock provider in truncated_json mode returns status='rejected'."""
        provider = MockVLMProvider(simulation_mode="truncated_json")
        resp = VLMAdapter.process(self.sample_img_bytes, "natural_portrait", provider=provider)
        self.assertEqual(resp.status, "rejected")
        self.assertIn("truncated", resp.error_reason)

    def test_mock_simulation_silhouette_goal_recommends_keep_original(self):
        """Silhouette goal triggers keep_original=True with 0 recommendations to preserve art contrast."""
        provider = MockVLMProvider()
        resp = VLMAdapter.process(self.sample_img_bytes, "silhouette", provider=provider)
        self.assertEqual(resp.status, "success")
        self.assertTrue(resp.keep_original["recommended"])
        self.assertEqual(len(resp.recommendations), 0)

    # 4. Replay provider
    def test_replay_provider_offline_execution(self):
        """Replay provider reads pre-saved response from disk, validates schema, and runs offline."""
        replay_file = os.path.join(self.test_dir, "saved_vlm_response.json")
        saved_payload = {
            "analysis": {
                "framing": "full_body",
                "subject_description": "Một người đứng toàn thân.",
                "background_description": "Phố đi bộ.",
                "lighting_condition": "low_light",
                "uncertainty": "Hơi tối nhẹ ở chân.",
            },
            "keep_original": {"recommended": False, "reason": "Cần tăng nhẹ sáng tổng thể."},
            "recommendations": [
                {
                    "option_id": "option_1",
                    "name": "Bù sáng tự nhiên",
                    "rationale": "Tăng nhẹ +0.2 EV và giữ nguyên tương phản.",
                    "recipe": {
                        "pipeline_version": PIPELINE_VERSION,
                        "exposure": 0.2,
                        "contrast": 1.0,
                        "saturation": 1.0,
                        "crop": None,
                    },
                }
            ],
            "retake_advice": "Nên đứng gần hơn tới nguồn đèn phố.",
        }
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(saved_payload, f)

        provider = ReplayVLMProvider(replay_file_path=replay_file)
        resp = VLMAdapter.process(self.sample_img_bytes, "natural_portrait", provider=provider)
        self.assertEqual(resp.status, "success")
        self.assertFalse(resp.is_real_model)  # Replay is offline provenance
        self.assertEqual(len(resp.recommendations), 1)
        self.assertEqual(resp.recipe["exposure"], 0.2)
        self.assertEqual(resp.fixture_source, replay_file)

    # 5. CLI end-to-end integration flows
    def test_cli_end_to_end_mock_provider_flow(self):
        """CLI with --provider mock runs full pipeline: renders previews, generates manifest and HTML."""
        out_dir = os.path.join(self.test_dir, "run_mock_cli")
        manifest = run_pipeline(
            image_bytes=self.sample_img_bytes,
            goal="natural_portrait",
            output_dir=out_dir,
            provider_type="mock",
        )

        # Artifacts check
        self.assertTrue(os.path.exists(os.path.join(out_dir, "reference_input.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "preview_noop.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "preview_rule.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "preview_candidate_1.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "preview_candidate_2.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "index.html")))

        # Manifest metadata check
        meta = manifest["metadata"]
        self.assertIn("vlm_summary", meta)
        vlm_sum = meta["vlm_summary"]
        self.assertEqual(vlm_sum["provider_type"], "mock")
        self.assertEqual(vlm_sum["status"], "success")
        self.assertFalse(vlm_sum["is_real_model"])
        self.assertIsNotNone(vlm_sum["analysis"])
        self.assertIsNotNone(vlm_sum["retake_advice"])

        # Check HTML content
        with open(os.path.join(out_dir, "index.html"), "r", encoding="utf-8") as f:
            html_text = f.read()
        self.assertIn("Photo Advisor Phase 02B - VLM Integration Report", html_text)
        self.assertIn("Đánh giá của Trợ lý VLM", html_text)
        self.assertIn("Tư vấn chụp lại cho lần sau", html_text)
        self.assertIn("không dựng preview giả lập", html_text)

    def test_cli_end_to_end_mock_refusal_fallback(self):
        """CLI with --provider mock and refusal mode safely falls back to original image without fake success."""
        out_dir = os.path.join(self.test_dir, "run_mock_refusal")
        manifest = run_pipeline(
            image_bytes=self.sample_img_bytes,
            goal="natural_portrait",
            output_dir=out_dir,
            provider_type="mock",
            mock_simulation_mode="refusal",
        )

        meta = manifest["metadata"]
        self.assertEqual(meta["vlm_summary"]["status"], "refusal")

        # Candidate 2 represents the failed VLM call
        cands = manifest["candidates"]
        vlm_cand = [c for c in cands if "vlm_failed" in c["candidate_id"]][0]
        self.assertIn("refusal", vlm_cand["status"])
        self.assertFalse(vlm_cand["export_eligible"])
        self.assertEqual(vlm_cand["effective_recipe"]["exposure"], 0.0)

    def test_provenance_is_real_model_flag(self):
        """is_real_model is True only for genuine live model inference, False for mock and replay."""
        mock_p = MockVLMProvider()
        r_mock = mock_p.generate(VLMRequest(self.sample_img_bytes, "id1", "goal1", "sha1", (800, 600)))
        self.assertFalse(r_mock.is_real_model)

        # LocalTransformersProvider without GPU raises RuntimeError gracefully
        from unittest.mock import patch
        with patch("torch.cuda.is_available", return_value=False):
            hf_p = LocalTransformersProvider(device="cuda")
            r_hf = hf_p.generate(VLMRequest(self.sample_img_bytes, "id2", "goal2", "sha2", (800, 600)))
            # When CUDA is stubbed/unavailable, initialization fails gracefully without downloading weights
            self.assertEqual(r_hf.status, "error")
            self.assertIn("CUDA GPU is not available", r_hf.error_reason)


if __name__ == "__main__":
    unittest.main()
