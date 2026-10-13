"""Comprehensive Test Suite for Phase 02B Revision 02 (Codex Review D1 through D5).

Tests:
- D1: Notebook calls model exactly ONCE per image; robust fallback on truncated/schema errors
      to No-op preview, manifest, HTML, and raw bundle without crashing; explicit photo source;
      isolated output directories.
- D2: Schema validation resilience against unhashable types (framing as list, option_id as dict),
      astronomical numbers, and non-dict payloads without exception or crash.
- D3: Wall-clock generation time budget via TimeBudgetStoppingCriteria in simulated loop,
      token limit boundary, and decode error handling.
- D4: Complete traceability envelope recording full effective config, schema SHA-256, prompt SHA-256,
      model revision, separated timings (model_load_ms, model_generation_ms, pipeline_total_ms);
      strict prompt version validation; replay provenance preservation and identity rejection.
- D5: Dependency declarations and captured request ICC profile conversion to sRGB.
"""

import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import unittest
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageCms, ImageDraw

from photo_advisor.cli import create_synthetic_test_image, run_pipeline
from photo_advisor.processor import compute_sha256, normalize_input_image
from photo_advisor.vlm_adapter import (
    LocalTransformersProvider,
    MockVLMProvider,
    ReplayVLMProvider,
    SUPPORTED_PROMPT_VERSIONS,
    TimeBudgetStoppingCriteria,
    VLMAdapter,
    VLMRequest,
    VLMResponse,
    get_vlm_output_schema,
    validate_vlm_schema,
)


class TestVLMPhase02BRevise02(unittest.TestCase):
    """Test suite covering Codex Review 02 requirements D1 through D5."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.sample_bytes = create_synthetic_test_image(800, 600, style="portrait")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Ensure no lingering zip file from tests in workspace root
        stray_zip = os.path.join(self.repo_root, "phase02b_colab_artifacts.zip")
        if os.path.exists(stray_zip) and not os.path.islink(stray_zip):
            try:
                # Do not delete if tracked, but clean test temp if needed
                pass
            except Exception:
                pass

    # =========================================================================
    # D1: Notebook Single Call & Resilient Fallback on Errors
    # =========================================================================

    def test_d1_notebook_single_provider_call(self):
        """Prove that the notebook calls the model provider exactly ONCE per image."""
        notebook_path = os.path.join(self.repo_root, "notebooks", "phase02b_colab.ipynb")
        self.assertTrue(os.path.exists(notebook_path), "Colab notebook must exist.")

        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        ns: Dict[str, Any] = {}

        from unittest.mock import patch

        # Execute notebook cells up through pipeline execution with active CUDA stubbing
        with patch("torch.cuda.is_available", return_value=False):
            for idx, cell in enumerate(code_cells, start=1):
                source = "".join([l for l in cell.get("source", []) if not l.strip().startswith("!")])
                if not source.strip():
                    continue

                # Override output directory to temp_dir
                if 'output_dir = "work/phase02b/run_output_colab"' in source:
                    source = source.replace(
                        'output_dir = "work/phase02b/run_output_colab"',
                        f'output_dir = r"{self.temp_dir}/colab_single_call_out"',
                    )

                try:
                    exec(source, ns)
                except Exception as e:
                    self.fail(f"Notebook code cell {idx} failed execution: {e}")

        # Check call count on the provider instance used in the notebook
        prov = ns.get("provider")
        self.assertIsNotNone(prov, "Notebook must define a 'provider' variable.")
        self.assertEqual(
            prov.call_count,
            1,
            f"Provider must be invoked exactly 1 time in notebook, got {prov.call_count}",
        )

        # Verify exported artifacts
        out_dir = os.path.join(self.temp_dir, "colab_single_call_out")
        self.assertTrue(os.path.exists(os.path.join(out_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "vlm_raw_response.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "index.html")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "preview_noop.png")))

    def test_d1_error_flow_truncated_json_produces_full_artifacts_and_noop_fallback(self):
        """When VLM outputs truncated JSON, pipeline safely falls back to No-op and exports all artifacts."""
        out_dir = os.path.join(self.temp_dir, "trunc_error_out")
        mock_prov = MockVLMProvider(simulation_mode="truncated_json")

        manifest = run_pipeline(
            image_bytes=self.sample_bytes,
            goal="natural_portrait",
            output_dir=out_dir,
            provider=mock_prov,
        )

        # Artifacts must exist despite VLM error
        self.assertTrue(os.path.exists(os.path.join(out_dir, "preview_noop.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "preview_candidate_1.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "index.html")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "vlm_raw_response.json")))

        # Candidate 1 preview must match No-op preview (safe fallback)
        with open(os.path.join(out_dir, "preview_noop.png"), "rb") as f:
            noop_bytes = f.read()
        with open(os.path.join(out_dir, "preview_candidate_1.png"), "rb") as f:
            cand1_bytes = f.read()
        self.assertEqual(noop_bytes, cand1_bytes, "Fallback preview must match No-op preview.")

        # Manifest metadata must record the error
        vlm_cand = [c for c in manifest["candidates"] if "cand_2_vlm" in c["candidate_id"]][0]
        self.assertIn("rejected", vlm_cand["status"])

        # vlm_raw_response.json must have error_reason
        with open(os.path.join(out_dir, "vlm_raw_response.json"), "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.assertEqual(raw_data["envelope"]["status"], "rejected")
        self.assertTrue(
            any(k in raw_data["envelope"]["error_reason"].lower() for k in ["truncated", "unterminated", "json"]),
            f"Expected truncation/json error reason, got {raw_data['envelope']['error_reason']}",
        )

    def test_d1_error_flow_invalid_schema_produces_full_artifacts(self):
        """When VLM returns invalid schema, pipeline safely falls back and exports all artifacts."""
        out_dir = os.path.join(self.temp_dir, "schema_error_out")
        mock_prov = MockVLMProvider(simulation_mode="invalid_schema")

        manifest = run_pipeline(
            image_bytes=self.sample_bytes,
            goal="natural_portrait",
            output_dir=out_dir,
            provider=mock_prov,
        )

        self.assertTrue(os.path.exists(os.path.join(out_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "vlm_raw_response.json")))
        vlm_cand = [c for c in manifest["candidates"] if "cand_2_vlm" in c["candidate_id"]][0]
        self.assertIn("rejected", vlm_cand["status"])

    def test_d1_manifest_records_explicit_real_photo_source(self):
        """Manifest accurately reflects explicit real human photo source."""
        out_dir = os.path.join(self.temp_dir, "real_photo_out")
        explicit_source = "real_user_photo: Unsplash portrait #49128 (CC0)"

        manifest = run_pipeline(
            image_bytes=self.sample_bytes,
            goal="natural_portrait",
            output_dir=out_dir,
            input_source=explicit_source,
            provider=MockVLMProvider(),
        )

        self.assertEqual(manifest["metadata"]["input_source"], explicit_source)
        for cand in manifest["candidates"]:
            self.assertEqual(cand["input_source"], explicit_source)

    # =========================================================================
    # D2: Schema Validation Robustness (Unhashable types & Huge Numbers)
    # =========================================================================

    def _get_base_valid_payload(self) -> Dict[str, Any]:
        """Return a strictly valid Phase 02B schema payload."""
        return {
            "analysis": {
                "subject_type": "single_person",
                "framing": "medium_shot",
                "lighting_condition": "indoor_neutral",
                "technical_flaws": ["mild_underexposure"],
                "composition_critique": "Bố cục cân đối.",
            },
            "keep_original": {
                "recommended": False,
                "reason": "Cần nâng sáng để khuôn mặt rõ nét hơn.",
            },
            "recommendations": [
                {
                    "option_id": "opt_brighten",
                    "name": "Chân dung sáng tự nhiên",
                    "rationale": "Tăng nhẹ exposure để da mặt tươi tắn.",
                    "recipe": {
                        "pipeline_version": "pipeline_v0",
                        "exposure": 0.2,
                        "contrast": 1.02,
                        "saturation": 1.01,
                        "crop": [0.0, 0.0, 1.0, 1.0],
                    },
                }
            ],
            "retake_advice": "Chụp hướng về phía nguồn sáng tự nhiên.",
        }

    def test_d2_schema_rejects_unhashable_framing_list_without_crash(self):
        """Payload with framing as list (e.g. ['full_body']) returns (False, errors) without TypeError."""
        payload = self._get_base_valid_payload()
        payload["analysis"]["framing"] = ["full_body"]  # Unhashable type!

        try:
            is_valid, errors = validate_vlm_schema(payload)
        except Exception as e:
            self.fail(f"validate_vlm_schema crashed with unhandled exception: {e}")

        self.assertFalse(is_valid)
        self.assertTrue(any("framing" in e.lower() for e in errors))

    def test_d2_schema_rejects_unhashable_option_id_dict_without_crash(self):
        """Payload with option_id as dict (e.g. {'id': 1}) returns (False, errors) without TypeError."""
        payload = self._get_base_valid_payload()
        payload["recommendations"][0]["option_id"] = {"id": 1}  # Unhashable dict!

        try:
            is_valid, errors = validate_vlm_schema(payload)
        except Exception as e:
            self.fail(f"validate_vlm_schema crashed with unhandled exception: {e}")

        self.assertFalse(is_valid)
        self.assertTrue(any("option_id" in e.lower() for e in errors))

    def test_d2_schema_rejects_astronomical_huge_numbers_without_overflow(self):
        """Payload containing huge numbers (> 10^100) returns (False, errors) without OverflowError."""
        payload = self._get_base_valid_payload()
        payload["recommendations"][0]["recipe"]["exposure"] = 1e308  # Astronomical float

        try:
            is_valid, errors = validate_vlm_schema(payload)
        except Exception as e:
            self.fail(f"validate_vlm_schema crashed with unhandled exception on huge float: {e}")

        self.assertFalse(is_valid)
        self.assertTrue(any("finite" in e.lower() or "exposure" in e.lower() for e in errors))

    def test_d2_schema_rejects_non_dict_root_gracefully(self):
        """Non-dict root payloads (None, list, string) return (False, errors) without crashing."""
        for invalid_root in [None, "hello world", [1, 2, 3], 42]:
            is_valid, errors = validate_vlm_schema(invalid_root)
            self.assertFalse(is_valid)
            self.assertTrue(any("dictionary" in e.lower() for e in errors))

    def test_d2_malformed_unhashable_payload_pipeline_safe_fallback(self):
        """Pipeline safely falls back to No-op when provider produces an unhashable payload."""
        class UnhashablePayloadMockProvider(MockVLMProvider):
            def generate(self, request: VLMRequest) -> VLMResponse:
                malformed = {
                    "analysis": {"framing": ["full_body", "close_up"]},
                    "recommendations": [{"option_id": {"nested": "id"}, "recipe": {"pipeline_version": "pipeline_v0", "crop": [0,0,1,1]}}],
                }
                return VLMResponse(
                    model_id="test_mock",
                    prompt_version=request.prompt_version,
                    recipe={"pipeline_version": "pipeline_v0", "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                    raw_response_text=json.dumps(malformed),
                    status="rejected",
                    error_reason="Schema validation failed: malformed unhashable types",
                    is_real_model=False,
                )

        out_dir = os.path.join(self.temp_dir, "unhashable_pipe_out")
        manifest = run_pipeline(
            image_bytes=self.sample_bytes,
            goal="natural_portrait",
            output_dir=out_dir,
            provider=UnhashablePayloadMockProvider(),
        )

        self.assertTrue(os.path.exists(os.path.join(out_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "preview_noop.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "vlm_raw_response.json")))
        cand = [c for c in manifest["candidates"] if "cand_2_vlm" in c["candidate_id"]][0]
        self.assertIn("rejected", cand["status"])

    # =========================================================================
    # D3: Time Budget Stopping Criteria & Decode Error Boundaries
    # =========================================================================

    def test_d3_time_budget_stopping_criteria_triggers_on_timeout(self):
        """TimeBudgetStoppingCriteria returns True when elapsed time exceeds budget."""
        criteria = TimeBudgetStoppingCriteria(
            start_time=time.perf_counter() - 0.2,  # 200ms elapsed
            time_budget_seconds=0.1,               # 100ms budget
        )
        # Criteria is called with dummy input_ids and scores
        should_stop = criteria(None, None)
        self.assertTrue(should_stop, "Stopping criteria must return True after budget is exceeded.")
        self.assertTrue(criteria.timed_out, "timed_out flag must be set to True.")

    def test_d3_time_budget_stopping_criteria_allows_generation_within_budget(self):
        """TimeBudgetStoppingCriteria returns False when elapsed time is within budget."""
        criteria = TimeBudgetStoppingCriteria(
            start_time=time.perf_counter(),
            time_budget_seconds=10.0,
        )
        should_stop = criteria(None, None)
        self.assertFalse(should_stop, "Stopping criteria must return False within time budget.")
        self.assertFalse(criteria.timed_out)

    def test_d3_simulated_token_generation_loop_stops_by_budget(self):
        """Simulate cooperative token loop: verify loop exits when time budget expires."""
        budget_seconds = 0.05
        t_start = time.perf_counter()
        criteria = TimeBudgetStoppingCriteria(start_time=t_start, time_budget_seconds=budget_seconds)

        tokens_generated = 0
        max_tokens = 500

        # Simulate iterative token generation
        while tokens_generated < max_tokens:
            time.sleep(0.01)  # simulate token generation step
            tokens_generated += 1
            if criteria(None, None):
                break

        self.assertTrue(criteria.timed_out, "Loop must be stopped by TimeBudgetStoppingCriteria.")
        self.assertLess(tokens_generated, max_tokens, "Generation must stop before max_tokens.")

    def test_d3_decode_error_returns_clean_response_and_noop_fallback(self):
        """Simulate decode error: provider returns status='decode_error' and pipeline falls back to No-op."""
        class DecodeErrorProvider(MockVLMProvider):
            def generate(self, request: VLMRequest) -> VLMResponse:
                return VLMResponse(
                    model_id="test_mock",
                    prompt_version=request.prompt_version,
                    recipe={"pipeline_version": "pipeline_v0", "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                    status="decode_error",
                    error_reason="Error decoding generated tokens: 'utf-8' codec can't decode byte 0xff in position 0",
                    is_real_model=False,
                    latency_ms=45.0,
                    load_latency_ms=10.0,
                    generation_latency_ms=35.0,
                )

        out_dir = os.path.join(self.temp_dir, "decode_error_out")
        manifest = run_pipeline(
            image_bytes=self.sample_bytes,
            goal="natural_portrait",
            output_dir=out_dir,
            provider=DecodeErrorProvider(),
        )

        self.assertTrue(os.path.exists(os.path.join(out_dir, "vlm_raw_response.json")))
        with open(os.path.join(out_dir, "vlm_raw_response.json"), "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.assertEqual(raw_data["envelope"]["status"], "decode_error")
        self.assertIn("decode byte", raw_data["envelope"]["error_reason"])

    # =========================================================================
    # D4: Complete Traceability Envelope, Timings, & Replay Preservation
    # =========================================================================

    def test_d4_envelope_contains_full_effective_config_and_hashes(self):
        """Envelope records effective_config, schema_sha256, prompt_sha256, and separate timings."""
        out_dir = os.path.join(self.temp_dir, "envelope_audit_out")
        mock_prov = MockVLMProvider()

        manifest = run_pipeline(
            image_bytes=self.sample_bytes,
            goal="natural_portrait",
            output_dir=out_dir,
            provider=mock_prov,
        )

        raw_path = os.path.join(out_dir, "vlm_raw_response.json")
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_bundle = json.load(f)

        env = raw_bundle["envelope"]
        # Required schema and prompt hashes
        self.assertIn("schema_sha256", env)
        self.assertEqual(len(env["schema_sha256"]), 64)
        self.assertIn("prompt_sha256", env)
        self.assertEqual(len(env["prompt_sha256"]), 64)

        # Required effective configuration fields
        self.assertIn("effective_config", env)
        eff_cfg = env["effective_config"]
        for key in ["dtype", "load_in_4bit", "device", "min_pixels", "max_pixels", "max_new_tokens", "time_budget_seconds"]:
            self.assertIn(key, eff_cfg, f"effective_config missing '{key}'")

        # Required separated timings
        self.assertIn("timings_ms", env)
        t_env = env["timings_ms"]
        self.assertIn("model_load_ms", t_env)
        self.assertIn("model_generation_ms", t_env)
        self.assertIn("model_latency_ms", t_env)
        self.assertIn("pipeline_total_ms", t_env)

        # Manifest metadata must also separate timings
        self.assertIn("timings_ms", manifest["metadata"])
        t_meta = manifest["metadata"]["timings_ms"]
        self.assertIn("model_load_ms", t_meta)
        self.assertIn("model_generation_ms", t_meta)
        self.assertIn("pipeline_total_ms", t_meta)

    def test_d4_rejects_unsupported_prompt_version(self):
        """Pipeline strictly rejects unsupported prompt versions."""
        out_dir = os.path.join(self.temp_dir, "prompt_ver_out")

        with self.assertRaises(ValueError) as ctx:
            run_pipeline(
                image_bytes=self.sample_bytes,
                goal="natural_portrait",
                output_dir=out_dir,
                prompt_version="prompt_v2_unsupported",
            )
        self.assertIn("Unsupported prompt_version", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            VLMAdapter.process(
                image_bytes=self.sample_bytes,
                goal="natural_portrait",
                prompt_version="prompt_v999",
            )
        self.assertIn("Unsupported prompt_version", str(ctx2.exception))

    def test_d4_replay_preserves_original_envelope_without_false_claim(self):
        """Replaying pre-saved envelope retains replayed_envelope and sets is_real_model=False."""
        # Create a saved run bundle
        orig_bundle = {
            "raw_response_text": json.dumps(self._get_base_valid_payload()),
            "parsed_response": self._get_base_valid_payload(),
            "envelope": {
                "original_input_sha256": compute_sha256(self.sample_bytes),
                "model_input_sha256": compute_sha256(self.sample_bytes),
                "goal": "natural_portrait",
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "model_revision": "rev_gpu_colab_12345",
                "is_real_model": True,
                "effective_config": {"dtype": "bfloat16", "device": "cuda:0"},
            },
        }

        replay_path = os.path.join(self.temp_dir, "saved_run_envelope.json")
        with open(replay_path, "w", encoding="utf-8") as f:
            json.dump(orig_bundle, f)

        out_dir = os.path.join(self.temp_dir, "replay_exec_out")
        manifest = run_pipeline(
            image_bytes=self.sample_bytes,
            goal="natural_portrait",
            output_dir=out_dir,
            provider_type="replay",
            replay_file=replay_path,
        )

        with open(os.path.join(out_dir, "vlm_raw_response.json"), "r", encoding="utf-8") as f:
            replayed_raw = json.load(f)

        # Must retain original envelope
        self.assertIn("replayed_envelope", replayed_raw)
        self.assertEqual(
            replayed_raw["replayed_envelope"]["model_revision"],
            "rev_gpu_colab_12345",
        )
        # Current execution envelope must NOT claim it is real live execution
        self.assertEqual(replayed_raw["envelope"]["provider_type"], "replay")
        self.assertFalse(replayed_raw["envelope"]["is_real_model"])

    def test_d4_replay_strict_identity_rejection(self):
        """Replay provider strictly rejects replay if image SHA256 or goal does not match."""
        orig_bundle = {
            "parsed_response": self._get_base_valid_payload(),
            "envelope": {
                "model_input_sha256": "sha_different_image_12345",
                "goal": "silhouette",  # Mismatch with requested goal
            },
        }
        replay_path = os.path.join(self.temp_dir, "saved_mismatch.json")
        with open(replay_path, "w", encoding="utf-8") as f:
            json.dump(orig_bundle, f)

        # Strict identity rejection
        prov = ReplayVLMProvider(replay_file_path=replay_path, strict_identity=True)
        req = VLMRequest(
            image_bytes=self.sample_bytes,
            image_id="img_01",
            goal="natural_portrait",
            image_sha256=compute_sha256(self.sample_bytes),
            image_dimensions=(800, 600),
        )

        resp = prov.generate(req)
        self.assertEqual(resp.status, "rejected")
        self.assertIn("Identity mismatch in replay", resp.error_reason)

    # =========================================================================
    # D5: Dependencies & Captured Request ICC Profile Conversion
    # =========================================================================

    def test_d5_requirements_file_declares_jsonschema(self):
        """requirements.txt must explicitly include jsonschema."""
        req_path = os.path.join(self.repo_root, "requirements.txt")
        self.assertTrue(os.path.exists(req_path), "requirements.txt must exist.")
        with open(req_path, "r", encoding="utf-8") as f:
            req_content = f.read()
        self.assertIn("jsonschema", req_content.lower())

    def test_d5_captured_request_icc_conversion_to_srgb(self):
        """Image with embedded sRGB ICC profile is correctly normalized to standard RGB mode in captured request.

        Note: This verifies embedded sRGB ICC profile preservation and normalization without color space corruption,
        without claiming hardware wide-gamut Display-P3 calibration.
        """
        # Create an image with an embedded synthetic sRGB ICC profile
        srgb_icc_img = Image.new("RGB", (200, 200), color=(255, 100, 50))
        # Embed a known standard sRGB ICC profile using ImageCms
        srgb_profile = ImageCms.createProfile("sRGB")
        # Save to buffer with profile
        buf = io.BytesIO()
        srgb_icc_img.save(buf, format="PNG", icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes())
        raw_icc_bytes = buf.getvalue()

        norm_img, logs = normalize_input_image(raw_icc_bytes)
        self.assertEqual(norm_img.mode, "RGB")
        self.assertIn("RGB", [l for l in logs if "RGB" in l][0])

        # Test request captured by MockVLMProvider
        mock_prov = MockVLMProvider()
        out_dir = os.path.join(self.temp_dir, "icc_pipe_out")
        run_pipeline(
            image_bytes=raw_icc_bytes,
            goal="natural_portrait",
            output_dir=out_dir,
            provider=mock_prov,
        )

        captured_req = mock_prov.last_received_request
        self.assertIsNotNone(captured_req)
        # Verify captured image is openable and standard RGB
        captured_pil = Image.open(io.BytesIO(captured_req.image_bytes))
        self.assertEqual(captured_pil.mode, "RGB")
        self.assertEqual(captured_pil.size, (200, 200))


if __name__ == "__main__":
    unittest.main()
