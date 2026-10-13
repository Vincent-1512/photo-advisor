"""Comprehensive Test Suite for Phase 02B Revision 03 (Codex Review 03: E1, E2, E3).

Tests:
- E1: Notebook execution mode separation:
      * 'mock' mode strictly uses MockVLMProvider even if CUDA is available.
      * 'real' mode requires CUDA and a real human portrait specified by user.
      * Missing real portrait in 'real' mode raises FileNotFoundError before inference
        without synthesizing images, and without downloading weights / loading model.
- E2: Run isolation & 2-run smoke test support:
      * Successive runs produce distinct output directories with run IDs/UUIDs.
      * No overwrite or deletion of previous artifacts.
      * Provider invoked exactly 1 time per image (total 2 calls for 2 runs).
      * Both runs contain complete artifacts (manifest, HTML, raw JSON, previews).
- E3: Replay envelope type safety & real pipeline integration:
      * Envelope as None (null), list, or string rejected safely without AttributeError.
      * Malformed identity metadata (non-string SHA256, non-string goal) rejected safely.
      * Real ReplayVLMProvider invoked with run_pipeline yields safe No-op fallback,
        preserves raw_response_text, and exports all 5 core artifacts.
- Item 4: Effective dtype tracking after model initialization.
"""

import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from PIL import Image

from photo_advisor.cli import create_synthetic_test_image, run_pipeline
from photo_advisor.processor import compute_sha256
from photo_advisor.vlm_adapter import (
    LocalTransformersProvider,
    MockVLMProvider,
    ReplayVLMProvider,
    VLMAdapter,
    VLMRequest,
    VLMResponse,
)


class TestVLMPhase02BRevise03(unittest.TestCase):
    """Test suite covering Codex Review 03 requirements E1, E2, E3."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.sample_bytes = create_synthetic_test_image(800, 600, style="portrait")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clean up any stray zip file
        stray_zip = os.path.join(self.repo_root, "phase02b_colab_artifacts.zip")
        if os.path.exists(stray_zip) and not os.path.islink(stray_zip):
            try:
                os.remove(stray_zip)
            except OSError:
                pass

    # =========================================================================
    # E1: Clear Mock/Real Mode & Guard Against Synthetic in Real Mode
    # =========================================================================

    def test_e1_missing_real_input_with_cuda_stub_stops_before_inference(self):
        """When EXECUTION_MODE='real' and CUDA is stubbed as available, missing real input
        MUST raise FileNotFoundError before inference, without loading model or downloading weights."""
        notebook_path = os.path.join(self.repo_root, "notebooks", "phase02b_colab.ipynb")
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        ns: Dict[str, Any] = {"__name__": "__main__"}

        # Track whether model download / initialization was ever attempted
        model_init_called = False

        def mock_init_model(provider_self):
            nonlocal model_init_called
            model_init_called = True
            raise RuntimeError("Should never attempt model initialization when input is missing!")

        non_existent_img = os.path.join(self.temp_dir, "missing_real_portrait.jpg")
        if os.path.exists(non_existent_img):
            os.remove(non_existent_img)

        # Mock CUDA as available with 16GB VRAM
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024**3

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.cuda.get_device_name", return_value="Tesla T4"), \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch.object(LocalTransformersProvider, "_initialize_model", mock_init_model):

            # Execute Cell 1 through Cell 5
            for idx in range(1, 6):
                source = "".join([l for l in code_cells[idx - 1].get("source", []) if not l.strip().startswith("!")])
                if idx == 4:
                    # Switch to real mode for this test
                    source = source.replace('EXECUTION_MODE = "mock"', 'EXECUTION_MODE = "real"')
                exec(source, ns)

            self.assertEqual(ns.get("EXECUTION_MODE"), "real")
            self.assertIsInstance(ns.get("provider"), LocalTransformersProvider)

            # Cell 6: Prepare input image with non-existent path
            cell6_source = "".join([l for l in code_cells[5].get("source", []) if not l.strip().startswith("!")])
            cell6_source = cell6_source.replace('input_image_path = "work/phase02b/sample_real_portrait.jpg"',
                                               f'input_image_path = r"{non_existent_img}"')

            # Must raise FileNotFoundError
            with self.assertRaises(FileNotFoundError) as cm:
                exec(cell6_source, ns)

            self.assertIn("DỪNG TRƯỚC INFERENCE - E1", str(cm.exception))
            self.assertIn("từ chối tự động sinh ảnh synthetic", str(cm.exception))

            # Verify no synthetic file was created at non_existent_img
            self.assertFalse(os.path.exists(non_existent_img))
            # Verify model initialization was NEVER attempted
            self.assertFalse(model_init_called, "Model initialization must not be called when input is missing!")

    def test_e1_mock_mode_always_uses_mock_provider_even_with_cuda(self):
        """When EXECUTION_MODE='mock', MockVLMProvider is ALWAYS used even if CUDA is available."""
        notebook_path = os.path.join(self.repo_root, "notebooks", "phase02b_colab.ipynb")
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        ns: Dict[str, Any] = {"__name__": "__main__"}

        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024**3

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.cuda.get_device_name", return_value="Tesla T4"), \
             patch("torch.cuda.get_device_properties", return_value=mock_props):

            # Cell 4 has EXECUTION_MODE = 'mock'
            for idx in [1, 3, 4, 5]:
                source = "".join([l for l in code_cells[idx - 1].get("source", []) if not l.strip().startswith("!")])
                exec(source, ns)

            provider = ns.get("provider")
            self.assertIsInstance(provider, MockVLMProvider, "Mock mode must instantiate MockVLMProvider even with CUDA!")

    # =========================================================================
    # E2: Output Isolation & 2-Run Smoke Test
    # =========================================================================

    def test_e2_two_pipeline_runs_create_separate_directories_single_call_each(self):
        """Two successive runs of the pipeline produce two separate output directories
        without overwriting, and the provider is called exactly 1 time per image."""
        mock_prov = MockVLMProvider(simulation_mode="normal")

        # Run 1: Portrait with natural_portrait goal
        img1_bytes = create_synthetic_test_image(800, 600, style="portrait")
        out1 = os.path.join(self.temp_dir, "run_output_1_uid_101")
        m1 = run_pipeline(
            image_bytes=img1_bytes,
            goal="natural_portrait",
            output_dir=out1,
            provider=mock_prov,
        )

        self.assertEqual(mock_prov.call_count, 1, "Provider should be called exactly once for image 1")
        self.assertTrue(os.path.exists(os.path.join(out1, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(out1, "index.html")))
        self.assertTrue(os.path.exists(os.path.join(out1, "vlm_raw_response.json")))
        self.assertTrue(os.path.exists(os.path.join(out1, "preview_noop.png")))
        self.assertTrue(os.path.exists(os.path.join(out1, "reference_input.png")))

        # Run 2: Landscape with soft_glow goal to a NEW output directory
        img2_bytes = create_synthetic_test_image(600, 800, style="landscape")
        out2 = os.path.join(self.temp_dir, "run_output_2_uid_202")
        m2 = run_pipeline(
            image_bytes=img2_bytes,
            goal="soft_glow",
            output_dir=out2,
            provider=mock_prov,
        )

        self.assertEqual(mock_prov.call_count, 2, "Provider should have exactly 2 calls total after run 2")
        self.assertTrue(os.path.exists(os.path.join(out2, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(out2, "index.html")))
        self.assertTrue(os.path.exists(os.path.join(out2, "vlm_raw_response.json")))
        self.assertTrue(os.path.exists(os.path.join(out2, "preview_noop.png")))
        self.assertTrue(os.path.exists(os.path.join(out2, "reference_input.png")))

        # Both output directories must be separate and independent
        self.assertNotEqual(out1, out2)
        # Check run IDs in manifests are distinct
        self.assertNotEqual(m1["metadata"]["run_id"], m2["metadata"]["run_id"])
        # Check goal metadata
        self.assertEqual(m1["metadata"]["goal"], "natural_portrait")
        self.assertEqual(m2["metadata"]["goal"], "soft_glow")

        # Verify old output was NOT overwritten or modified by run 2
        with open(os.path.join(out1, "manifest.json"), "r", encoding="utf-8") as f:
            m1_recheck = json.load(f)
        self.assertEqual(m1_recheck["metadata"]["goal"], "natural_portrait")

    def test_e2_notebook_cell8_generates_new_dir_per_run(self):
        """Executing Cell 8 without external override generates separate timestamped run dirs."""
        notebook_path = os.path.join(self.repo_root, "notebooks", "phase02b_colab.ipynb")
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        ns: Dict[str, Any] = {"__name__": "__main__"}

        # Run cells 1-7 in mock mode
        with patch("torch.cuda.is_available", return_value=False):
            for idx in range(1, 8):
                source = "".join([l for l in code_cells[idx - 1].get("source", []) if not l.strip().startswith("!")])
                exec(source, ns)

        # In Cell 8, change the base path to self.temp_dir/run_output_colab
        cell8_source = "".join([l for l in code_cells[7].get("source", []) if not l.strip().startswith("!")])
        temp_base = os.path.join(self.temp_dir, "test_colab_runs")
        cell8_source_custom = cell8_source.replace('"work/phase02b/run_output_colab"', f'r"{temp_base}"')
        cell8_source_custom = cell8_source_custom.replace('output_dir == "work/phase02b/run_output_colab"', f'output_dir == r"{temp_base}"')

        # Execution 1
        ns_run1 = dict(ns)
        exec(cell8_source_custom, ns_run1)
        dir1 = ns_run1.get("output_dir")
        self.assertIsNotNone(dir1)
        self.assertTrue(os.path.exists(os.path.join(dir1, "manifest.json")))

        # Execution 2 (a brief pause to ensure distinct timestamp or uuid)
        ns_run2 = dict(ns)
        exec(cell8_source_custom, ns_run2)
        dir2 = ns_run2.get("output_dir")
        self.assertIsNotNone(dir2)
        self.assertTrue(os.path.exists(os.path.join(dir2, "manifest.json")))

        # Verify separate directories were created and both exist
        self.assertNotEqual(dir1, dir2)
        self.assertTrue(os.path.isdir(dir1))
        self.assertTrue(os.path.isdir(dir2))

    # =========================================================================
    # E3: Replay Envelope/Metadata Type Checking & Real Pipeline Regression
    # =========================================================================

    def test_e3_replay_envelope_none_type_safe_rejection(self):
        """Replay JSON containing envelope=None (null) is safely rejected without AttributeError."""
        replay_file = os.path.join(self.temp_dir, "replay_envelope_none.json")
        payload = {"envelope": None, "parsed_response": {}}
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        prov = ReplayVLMProvider(replay_file_path=replay_file, strict_identity=False)
        out_dir = os.path.join(self.temp_dir, "replay_none_pipe_out")

        # Must execute cleanly via run_pipeline without crashing
        manifest = run_pipeline(
            image_bytes=self.sample_bytes,
            goal="natural_portrait",
            output_dir=out_dir,
            provider=prov,
        )

        self.assertIsNotNone(manifest)
        # Check vlm_raw_response.json
        raw_resp_path = os.path.join(out_dir, "vlm_raw_response.json")
        self.assertTrue(os.path.exists(raw_resp_path))
        with open(raw_resp_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.assertEqual(raw_data.get("envelope", {}).get("status"), "rejected")
        self.assertIn("Invalid envelope type", raw_data.get("envelope", {}).get("error_reason", ""))
        self.assertIn("envelope", raw_data.get("raw_response_text", ""))

        # Verify pipeline safely fell back to No-op preview and effective recipe
        self.assertTrue(os.path.exists(os.path.join(out_dir, "preview_noop.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "index.html")))
        vlm_cands = [c for c in manifest["candidates"] if "vlm" in c["candidate_id"]]
        for vc in vlm_cands:
            self.assertTrue(vc["status"].startswith("rejected"))
            self.assertEqual(vc["effective_recipe"]["exposure"], 0.0)
            self.assertEqual(vc["effective_recipe"]["contrast"], 1.0)
            self.assertEqual(vc["effective_recipe"]["saturation"], 1.0)

    def test_e3_replay_envelope_list_and_string_type_safe_rejection(self):
        """Replay JSON containing envelope as list or string is safely rejected."""
        for bad_env in ([1, 2, 3], "invalid_envelope_string"):
            replay_file = os.path.join(self.temp_dir, f"replay_bad_env_{type(bad_env).__name__}.json")
            with open(replay_file, "w", encoding="utf-8") as f:
                json.dump({"envelope": bad_env, "data": {}}, f)

            prov = ReplayVLMProvider(replay_file_path=replay_file, strict_identity=False)
            out_dir = os.path.join(self.temp_dir, f"out_{type(bad_env).__name__}")

            manifest = run_pipeline(
                image_bytes=self.sample_bytes,
                goal="natural_portrait",
                output_dir=out_dir,
                provider=prov,
            )

            with open(os.path.join(out_dir, "vlm_raw_response.json"), "r", encoding="utf-8") as f:
                raw_resp = json.load(f)
            self.assertEqual(raw_resp.get("envelope", {}).get("status"), "rejected")
            self.assertIn("Invalid envelope type", raw_resp.get("envelope", {}).get("error_reason", ""))

    def test_e3_replay_metadata_wrong_types_safe_rejection(self):
        """Replay envelope with non-string saved_sha or non-string goal is safely rejected."""
        # Case 1: non-string SHA (e.g. integer 123456)
        replay_file_sha = os.path.join(self.temp_dir, "replay_int_sha.json")
        with open(replay_file_sha, "w", encoding="utf-8") as f:
            json.dump({"envelope": {"model_input_sha256": 123456, "goal": "natural_portrait"}}, f)

        prov_sha = ReplayVLMProvider(replay_file_path=replay_file_sha, strict_identity=True)
        out_sha = os.path.join(self.temp_dir, "out_int_sha")
        m_sha = run_pipeline(image_bytes=self.sample_bytes, goal="natural_portrait", output_dir=out_sha, provider=prov_sha)
        with open(os.path.join(out_sha, "vlm_raw_response.json"), "r", encoding="utf-8") as f:
            raw_sha = json.load(f)
        self.assertEqual(raw_sha.get("envelope", {}).get("status"), "rejected")
        self.assertIn("Invalid image hash type", raw_sha.get("envelope", {}).get("error_reason", ""))

        # Case 2: non-string goal (e.g. list ['natural_portrait'])
        replay_file_goal = os.path.join(self.temp_dir, "replay_list_goal.json")
        with open(replay_file_goal, "w", encoding="utf-8") as f:
            json.dump({"envelope": {"model_input_sha256": compute_sha256(self.sample_bytes), "goal": ["natural_portrait"]}}, f)

        prov_goal = ReplayVLMProvider(replay_file_path=replay_file_goal, strict_identity=True)
        out_goal = os.path.join(self.temp_dir, "out_list_goal")
        m_goal = run_pipeline(image_bytes=self.sample_bytes, goal="natural_portrait", output_dir=out_goal, provider=prov_goal)
        with open(os.path.join(out_goal, "vlm_raw_response.json"), "r", encoding="utf-8") as f:
            raw_goal = json.load(f)
        self.assertEqual(raw_goal.get("envelope", {}).get("status"), "rejected")
        self.assertIn("Invalid goal type", raw_goal.get("envelope", {}).get("error_reason", ""))

    def test_e3_replay_non_dict_root_json_safe_rejection(self):
        """Replay JSON file with non-dict root (e.g. string or list) is safely rejected."""
        replay_file = os.path.join(self.temp_dir, "replay_string_root.json")
        with open(replay_file, "w", encoding="utf-8") as f:
            f.write('"simple_json_string"')

        prov = ReplayVLMProvider(replay_file_path=replay_file)
        out_dir = os.path.join(self.temp_dir, "out_str_root")
        manifest = run_pipeline(image_bytes=self.sample_bytes, goal="natural_portrait", output_dir=out_dir, provider=prov)
        with open(os.path.join(out_dir, "vlm_raw_response.json"), "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.assertEqual(raw_data.get("envelope", {}).get("status"), "rejected")
        self.assertIn("must be a dictionary", raw_data.get("envelope", {}).get("error_reason", ""))

    # =========================================================================
    # Item 4: Effective Dtype Tracking
    # =========================================================================

    def test_item4_effective_dtype_saved_after_model_auto_selection(self):
        """LocalTransformersProvider saves effective compute_dtype when auto-selected or initialized."""
        prov = LocalTransformersProvider(model_id="Qwen/Qwen2.5-VL-3B-Instruct", compute_dtype=None)
        # Before initialization, compute_dtype is None
        cfg_pre = prov.get_effective_config()
        self.assertIsNone(cfg_pre["compute_dtype"])

        # Mock model and processor to simulate _initialize_model
        mock_model = MagicMock()
        import torch
        mock_model.dtype = torch.float16
        mock_model.config._commit_hash = "abc123commit"

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_bf16_supported", return_value=False), \
             patch("transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained", return_value=mock_model), \
             patch("transformers.AutoProcessor.from_pretrained", return_value=MagicMock()):

            prov._initialize_model()

        # After initialization, effective dtype is recorded from model.dtype
        cfg_post = prov.get_effective_config()
        self.assertIsNotNone(cfg_post["compute_dtype"])
        self.assertEqual(cfg_post["compute_dtype"], str(torch.float16))
        self.assertEqual(cfg_post["model_revision"], "abc123commit")


if __name__ == "__main__":
    unittest.main()
