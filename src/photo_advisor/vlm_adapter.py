"""VLM Adapter, Providers, and Schema Validation for Photo Advisor Phase 02B.

Provides:
- VLMRequest carrying normalized upright sRGB image bytes, SHA-256 hash, dimensions, and goal,
  as well as original image hash and dimensions for coordinate/color provenance.
- Strict authoritative Draft-07 schema validation with finite numeric and type checks,
  guaranteed never to throw unhandled exceptions on malformed or invalid types.
- MockVLMProvider for network-free deterministic transport verification and error mode simulation.
- ReplayVLMProvider for offline reproducible replay of pre-saved model responses with identity verification,
  preserving original provenance without claiming new live execution.
- LocalTransformersProvider for execution on remote GPU runtime (e.g. Google Colab) with Qwen2.5-VL,
  configurable precision, quantization, image/token limits, cooperative time budget stopping criteria,
  and robust multi-stage error boundaries.
- Provenance tracking (is_real_model=True only for real model provider runs).
- Backward-compatible load_fixture_from_file for Phase 02A fixtures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
from PIL import Image, ImageOps

from photo_advisor.types import (
    CONTRAST_CLAMP_MAX,
    CONTRAST_CLAMP_MIN,
    EXPOSURE_CLAMP_MAX,
    EXPOSURE_CLAMP_MIN,
    PIPELINE_VERSION,
    SATURATION_CLAMP_MAX,
    SATURATION_CLAMP_MIN,
)

try:
    from transformers import StoppingCriteria, StoppingCriteriaList
except ImportError:
    class StoppingCriteria:  # type: ignore
        pass
    class StoppingCriteriaList(list):  # type: ignore
        pass


class TimeBudgetStoppingCriteria(StoppingCriteria):
    """Cooperative token-level generation stopping criteria based on wall-clock elapsed time."""

    def __init__(self, start_time: float, time_budget_seconds: float):
        super().__init__()
        self.start_time = start_time
        self.time_budget_seconds = time_budget_seconds
        self.timed_out = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        if (time.perf_counter() - self.start_time) >= self.time_budget_seconds:
            self.timed_out = True
            return True
        return False


@dataclass(frozen=True)
class VLMRequest:
    """Request payload sent to a VLM provider, carrying real normalized upright sRGB image bytes."""
    image_bytes: bytes
    image_id: str
    goal: str
    image_sha256: str
    image_dimensions: Tuple[int, int]  # (width, height) of normalized upright sRGB image
    prompt_version: str = "prompt_v1"
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    original_image_sha256: str = ""
    original_dimensions: Tuple[int, int] = (0, 0)


@dataclass
class VLMResponse:
    """Structured response returned by a VLM provider."""
    model_id: str
    prompt_version: str
    recipe: Dict[str, Any]              # Primary recipe (first recommendation, or No-op if empty)
    analysis: Optional[Dict[str, Any]] = None
    keep_original: Optional[Dict[str, Any]] = None
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    advice_text: Optional[str] = None
    retake_advice: Optional[str] = None
    is_real_model: bool = False
    raw_response_text: Optional[str] = None
    fixture_source: Optional[str] = None
    latency_ms: float = 0.0
    load_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    tokens_used: Optional[Dict[str, Any]] = None
    status: str = "success"             # "success", "rejected", "refusal", "timeout", "oom", "error", etc.
    error_reason: Optional[str] = None
    replayed_envelope: Optional[Dict[str, Any]] = None


_SCHEMA_CACHE: Optional[Dict[str, Any]] = None


def get_vlm_output_schema() -> Dict[str, Any]:
    """Load and cache the authoritative Phase 02B VLM output schema."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "vlm_output_schema_v1.json")
        )
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Authoritative schema file missing at: {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as f:
            _SCHEMA_CACHE = json.load(f)
    return _SCHEMA_CACHE


def _check_finite_numbers(obj: Any, path: str = "") -> List[str]:
    """Recursively verify that no numbers are boolean, NaN, infinite, or overflowing."""
    errors = []
    if isinstance(obj, bool):
        # Booleans are checked contextually (e.g. forbidden in numeric recipe parameters)
        return errors
    elif isinstance(obj, (int, float)):
        try:
            fval = float(obj)
            if math.isnan(fval):
                errors.append(f"Value at '{path}' is NaN; must be a finite number.")
            elif math.isinf(fval):
                errors.append(f"Value at '{path}' is Infinite; must be a finite number.")
            elif abs(fval) > 1e12:
                errors.append(f"Value at '{path}' exceeds finite bounds.")
        except (OverflowError, ValueError):
            errors.append(f"Value at '{path}' exceeds finite numeric range (overflow).")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            subpath = f"{path}.{k}" if path else str(k)
            errors.extend(_check_finite_numbers(v, subpath))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            subpath = f"{path}[{i}]"
            errors.extend(_check_finite_numbers(v, subpath))
    return errors


def validate_vlm_schema(data: Any) -> Tuple[bool, List[str]]:
    """Validate a parsed JSON dictionary against the authoritative Phase 02B VLM output schema.

    Guaranteed never to raise exceptions (e.g., TypeError, AttributeError, OverflowError)
    on any malformed or unexpected data types.
    """
    errors: List[str] = []
    try:
        if not isinstance(data, dict):
            return False, [f"Root of VLM response must be a dictionary, got {type(data).__name__}"]

        schema = get_vlm_output_schema()
        validator = jsonschema.Draft7Validator(schema)

        for err in sorted(validator.iter_errors(data), key=lambda e: [str(p) for p in e.path]):
            loc = ".".join(str(p) for p in err.path) if err.path else "root"
            errors.append(f"Schema violation at '{loc}': {err.message}")

        # Explicit finite numeric checks (handles NaN, Inf, and OverflowError safely)
        finite_errs = _check_finite_numbers(data)
        errors.extend(finite_errs)

        # Semantic check: analysis framing
        analysis = data.get("analysis")
        if isinstance(analysis, dict):
            valid_framings = {"close_up", "medium_shot", "full_body", "environmental", "unknown"}
            framing = analysis.get("framing")
            if not isinstance(framing, str) or framing not in valid_framings:
                errors.append(f"Invalid framing '{framing}', expected one of {sorted(valid_framings)}")

        # Semantic check: recommendations
        recs = data.get("recommendations")
        if isinstance(recs, list):
            if len(recs) > 2:
                errors.append(f"'recommendations' exceeds maximum allowed options (expected <= 2, got {len(recs)})")
            option_ids: List[str] = []
            for idx, item in enumerate(recs):
                if not isinstance(item, dict):
                    errors.append(f"recommendations[{idx}] must be an object, got {type(item).__name__}")
                    continue

                opt_id = item.get("option_id")
                if not isinstance(opt_id, str):
                    errors.append(f"recommendations[{idx}].option_id must be a string, got {type(opt_id).__name__}")
                else:
                    option_ids.append(opt_id)

                recipe = item.get("recipe")
                if not isinstance(recipe, dict):
                    errors.append(f"recommendations[{idx}].recipe must be an object, got {type(recipe).__name__}")
                    continue

                # Check pipeline_version
                if recipe.get("pipeline_version") != PIPELINE_VERSION:
                    errors.append(f"recommendations[{idx}].recipe pipeline_version must be '{PIPELINE_VERSION}', got '{recipe.get('pipeline_version')}'")

                # Check that crop field exists
                if "crop" not in recipe:
                    errors.append(f"recommendations[{idx}].recipe missing required 'crop' field")

                # Check finite numbers and forbid bool as number
                for key in ("exposure", "contrast", "saturation"):
                    val = recipe.get(key)
                    if isinstance(val, bool):
                        errors.append(f"recommendations[{idx}].recipe.{key} must not be a boolean")
                    elif isinstance(val, (int, float)):
                        try:
                            fval = float(val)
                            if math.isnan(fval) or math.isinf(fval):
                                errors.append(f"recommendations[{idx}].recipe.{key} must be a finite number")
                        except (OverflowError, ValueError):
                            errors.append(f"recommendations[{idx}].recipe.{key} exceeds finite float range")

                # Check crop ordering if crop is a list
                crop = recipe.get("crop")
                if isinstance(crop, (list, tuple)) and len(crop) == 4:
                    for c_idx, c_val in enumerate(crop):
                        if isinstance(c_val, bool):
                            errors.append(f"recommendations[{idx}].recipe.crop[{c_idx}] must not be a boolean")
                    try:
                        ymin, xmin, ymax, xmax = float(crop[0]), float(crop[1]), float(crop[2]), float(crop[3])
                        if not (ymin < ymax and xmin < xmax):
                            errors.append(
                                f"recommendations[{idx}].recipe.crop invalid ordering: ymin ({ymin}) must < ymax ({ymax}) and xmin ({xmin}) must < xmax ({xmax})"
                            )
                    except (ValueError, TypeError, OverflowError):
                        errors.append(f"recommendations[{idx}].recipe.crop coordinates must be finite numeric values")

            # Check uniqueness only among valid string option IDs
            if len(option_ids) != len(set(option_ids)):
                errors.append(f"Option IDs in recommendations must be unique: got {option_ids}")
        elif recs is not None:
            errors.append(f"'recommendations' must be an array, got {type(recs).__name__}")

    except Exception as e:
        # Ultimate fallback guarantee: never raise unhandled exceptions
        return False, [f"Schema validation internal error: {e}"]

    return len(errors) == 0, errors


class VLMProvider(ABC):
    """Abstract Base Class for VLM providers."""

    @abstractmethod
    def generate(self, request: VLMRequest) -> VLMResponse:
        """Process an image request and return a structured VLMResponse."""
        pass

    def get_effective_config(self) -> Dict[str, Any]:
        """Return provider configuration metadata for envelope tracking."""
        return {"provider_class": self.__class__.__name__}


class MockVLMProvider(VLMProvider):
    """Deterministic mock transport provider for offline testing and pipeline verification.

    Never calls the network or downloads external assets. Proves that real image bytes,
    dimensions, and hash are correctly assembled, and simulates edge cases (refusal, timeout,
    truncated output, schema violation).
    """

    def __init__(self, simulation_mode: str = "normal"):
        self.simulation_mode = simulation_mode
        self.last_received_request: Optional[VLMRequest] = None
        self.model_revision: Optional[str] = None
        self.call_count: int = 0

    def get_effective_config(self) -> Dict[str, Any]:
        return {
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct-mock",
            "model_revision": self.model_revision,
            "provider_type": "mock",
            "simulation_mode": self.simulation_mode,
            "is_simulated": True,
            "device": "cpu",
            "load_in_4bit": False,
            "dtype": "fp32",
            "compute_dtype": "fp32",
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "max_new_tokens": 512,
            "time_budget_seconds": 60.0,
        }

    def generate(self, request: VLMRequest) -> VLMResponse:
        t0 = time.perf_counter()
        self.last_received_request = request
        self.call_count += 1

        # Verify real image payload contract
        if not request.image_bytes or len(request.image_bytes) == 0:
            return VLMResponse(
                model_id=request.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="rejected",
                error_reason="Empty or missing image bytes payload in request.",
                is_real_model=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 1. Error simulation modes
        if self.simulation_mode == "refusal":
            return VLMResponse(
                model_id=request.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="refusal",
                error_reason="Model refused request: image flagged by safety policy.",
                is_real_model=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if self.simulation_mode == "timeout":
            return VLMResponse(
                model_id=request.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="timeout",
                error_reason="Request timed out after 30000ms.",
                is_real_model=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if self.simulation_mode == "truncated_json":
            raw_bad = '{"analysis": {"framing": "medium_shot", "subject_description": "Một người...'
            return VLMResponse(
                model_id=request.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                raw_response_text=raw_bad,
                status="rejected",
                error_reason="Malformed or truncated JSON response from model.",
                is_real_model=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if self.simulation_mode == "invalid_schema":
            raw_invalid = {
                "analysis": {"framing": ["invalid_list_framing"]},
                "keep_original": "not_a_dict",
            }
            return VLMResponse(
                model_id=request.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                raw_response_text=json.dumps(raw_invalid),
                status="rejected",
                error_reason="Schema validation failed: missing required keys in structured response.",
                is_real_model=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 2. Normal deterministic response tailored to goal
        if request.goal == "silhouette":
            analysis = {
                "framing": "medium_shot",
                "subject_description": "Chủ thể đứng ngược sáng tạo hình bóng nghệ thuật (silhouette).",
                "background_description": "Bầu trời hoàng hôn có màu sắc nổi bật.",
                "lighting_condition": "backlit",
                "uncertainty": None,
            }
            keep_orig = {
                "recommended": True,
                "reason": "Ảnh ngược sáng có chủ đích nghệ thuật, khuyến nghị giữ nguyên để bảo toàn độ tương phản.",
            }
            recommendations: List[Dict[str, Any]] = []
            retake_advice = "Nếu muốn chụp rõ mặt thay vì tạo bóng nghệ thuật, hãy dùng đèn trợ sáng hoặc xoay hướng đón sáng."
            primary_recipe = {"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None}
            advice_text = "Ảnh ngược sáng nghệ thuật đẹp mắt. Khuyến nghị giữ nguyên gốc."

        elif self.simulation_mode == "empty_recommendations":
            analysis = {
                "framing": "close_up",
                "subject_description": "Khuôn mặt chụp cận cảnh rõ nét, ánh sáng đều.",
                "background_description": "Nền mờ tự nhiên (bokeh).",
                "lighting_condition": "well_lit",
                "uncertainty": None,
            }
            keep_orig = {
                "recommended": True,
                "reason": "Ảnh gốc đã đạt ánh sáng chuẩn và màu da tự nhiên, không cần can thiệp.",
            }
            recommendations = []
            retake_advice = None
            primary_recipe = {"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None}
            advice_text = "Ảnh gốc đã rất đẹp, giữ nguyên gốc."

        else:
            # Default realistic 2-option recommendations for portrait (strictly adhering to 02A bounds)
            analysis = {
                "framing": "medium_shot",
                "subject_description": "Chủ thể một người ở trung tâm khung hình, tư thế đứng tự nhiên.",
                "background_description": "Bối cảnh ngoài trời với tán cây và ánh sáng ban ngày nhẹ.",
                "lighting_condition": "well_lit",
                "uncertainty": None,
            }
            keep_orig = {
                "recommended": False,
                "reason": "Ảnh gốc hơi thiếu tương phản nhẹ và có viền thừa hai bên.",
            }
            opt1 = {
                "option_id": "option_1",
                "name": "Chân dung tự nhiên tươi sáng",
                "rationale": "Tăng nhẹ độ sáng (+0.10) và bão hòa (+1.03) để da mặt rạng rỡ, cắt gọn 2% biên.",
                "recipe": {
                    "pipeline_version": PIPELINE_VERSION,
                    "exposure": 0.10,
                    "contrast": 1.04,
                    "saturation": 1.03,
                    "crop": [0.02, 0.02, 0.98, 0.98],
                },
            }
            opt2 = {
                "option_id": "option_2",
                "name": "Chân dung cổ điển có chiều sâu",
                "rationale": "Giữ nguyên độ sáng nhưng tăng tương phản (1.08) để làm nổi bật đường nét chủ thể.",
                "recipe": {
                    "pipeline_version": PIPELINE_VERSION,
                    "exposure": 0.0,
                    "contrast": 1.08,
                    "saturation": 0.98,
                    "crop": None,
                },
            }
            recommendations = [opt1, opt2]
            retake_advice = "Lần sau có thể đứng chếch 45 độ so với hướng nắng để tạo bóng đổ tự nhiên trên sống mũi."
            primary_recipe = opt1["recipe"]
            advice_text = f"Đề xuất 2 phương án: {opt1['name']} và {opt2['name']}."

        elapsed = (time.perf_counter() - t0) * 1000.0
        return VLMResponse(
            model_id=f"{request.model_id}-mock",
            prompt_version=request.prompt_version,
            recipe=primary_recipe,
            analysis=analysis,
            keep_original=keep_orig,
            recommendations=recommendations,
            advice_text=advice_text,
            retake_advice=retake_advice,
            is_real_model=False,
            raw_response_text=json.dumps({
                "analysis": analysis,
                "keep_original": keep_orig,
                "recommendations": recommendations,
                "retake_advice": retake_advice,
            }, ensure_ascii=False),
            latency_ms=elapsed,
            tokens_used={
                "prompt_tokens": 420,
                "completion_tokens": 185,
                "total_tokens": 605,
                "is_simulated": True,
            },
            status="success",
        )


class ReplayVLMProvider(VLMProvider):
    """Replays a pre-saved raw JSON model response or envelope from disk without network calls."""

    def __init__(self, replay_file_path: str, strict_identity: bool = False):
        self.replay_file_path = replay_file_path
        self.strict_identity = strict_identity
        self.replayed_envelope: Optional[Dict[str, Any]] = None
        self.model_revision: Optional[str] = None

    def get_effective_config(self) -> Dict[str, Any]:
        cfg = {
            "provider_type": "replay",
            "replay_file_path": self.replay_file_path,
            "strict_identity": self.strict_identity,
            "is_real_model": False,
        }
        if self.replayed_envelope:
            orig_cfg = self.replayed_envelope.get("effective_config") or self.replayed_envelope.get("config")
            if orig_cfg:
                cfg["replayed_original_config"] = orig_cfg
        return cfg

    def generate(self, request: VLMRequest) -> VLMResponse:
        t0 = time.perf_counter()
        if not os.path.exists(self.replay_file_path):
            return VLMResponse(
                model_id=request.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="rejected",
                error_reason=f"Replay file does not exist: {self.replay_file_path}",
                is_real_model=False,
                fixture_source=self.replay_file_path,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        try:
            with open(self.replay_file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as e:
            return VLMResponse(
                model_id=request.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="rejected",
                error_reason=f"Failed to read replay JSON file: {e}",
                is_real_model=False,
                fixture_source=self.replay_file_path,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        envelope = None
        tokens_used = None
        # Check if saved file is wrapped in an envelope
        if isinstance(content, dict) and "envelope" in content:
            raw_env = content["envelope"]
            if not isinstance(raw_env, dict):
                return VLMResponse(
                    model_id=request.model_id,
                    prompt_version=request.prompt_version,
                    recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                    status="rejected",
                    error_reason=f"Invalid envelope type in replay file: expected dictionary, got {type(raw_env).__name__}",
                    is_real_model=False,
                    fixture_source=self.replay_file_path,
                    raw_response_text=json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content),
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
            envelope = raw_env
            self.replayed_envelope = envelope
            self.model_revision = envelope.get("model_revision")
            tokens_used = envelope.get("tokens")
            if "parsed_response" in content and isinstance(content["parsed_response"], dict):
                data = content["parsed_response"]
            elif "raw_response_text" in content and isinstance(content["raw_response_text"], str):
                try:
                    data = json.loads(content["raw_response_text"])
                except Exception:
                    data = content
            else:
                data = content
        else:
            data = content

        original_model_id = envelope.get("model_id", request.model_id) if envelope else request.model_id
        if not isinstance(original_model_id, str):
            original_model_id = str(original_model_id) if original_model_id is not None else request.model_id

        # Strict identity check: verify image sha256 and goal if envelope metadata is present
        mismatch_notes = []
        if envelope:
            saved_sha = envelope.get("model_input_sha256") or envelope.get("original_input_sha256")
            if saved_sha is not None:
                if not isinstance(saved_sha, str):
                    return VLMResponse(
                        model_id=original_model_id,
                        prompt_version=request.prompt_version,
                        recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                        status="rejected",
                        error_reason=f"Invalid image hash type in replay envelope: expected string, got {type(saved_sha).__name__}",
                        is_real_model=False,
                        fixture_source=self.replay_file_path,
                        replayed_envelope=envelope,
                        raw_response_text=json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content),
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                    )
                if saved_sha != request.image_sha256 and saved_sha != request.original_image_sha256:
                    mismatch_notes.append(
                        f"Input SHA256 mismatch: replay artifact was generated for image {saved_sha[:8]}..., but request has {request.image_sha256[:8]}..."
                    )
            saved_goal = envelope.get("goal")
            if saved_goal is not None:
                if not isinstance(saved_goal, str):
                    return VLMResponse(
                        model_id=original_model_id,
                        prompt_version=request.prompt_version,
                        recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                        status="rejected",
                        error_reason=f"Invalid goal type in replay envelope: expected string, got {type(saved_goal).__name__}",
                        is_real_model=False,
                        fixture_source=self.replay_file_path,
                        replayed_envelope=envelope,
                        raw_response_text=json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content),
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                    )
                if saved_goal != request.goal:
                    mismatch_notes.append(
                        f"Goal mismatch: replay artifact was generated for goal '{saved_goal}', but request goal is '{request.goal}'"
                    )

        if mismatch_notes and self.strict_identity:
            return VLMResponse(
                model_id=original_model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="rejected",
                error_reason=f"Identity mismatch in replay: {'; '.join(mismatch_notes)}",
                is_real_model=False,
                fixture_source=self.replay_file_path,
                replayed_envelope=envelope,
                raw_response_text=json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Validate schema
        is_valid, errors = validate_vlm_schema(data)
        if not is_valid:
            return VLMResponse(
                model_id=original_model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="rejected",
                error_reason=f"Replay file failed schema validation: {'; '.join(errors)}",
                is_real_model=False,
                fixture_source=self.replay_file_path,
                replayed_envelope=envelope,
                raw_response_text=json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        recommendations = data.get("recommendations", [])
        primary_recipe = (
            recommendations[0]["recipe"]
            if recommendations
            else {"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None}
        )

        status = "warning" if mismatch_notes else "success"
        error_reason = f"Identity mismatch: {'; '.join(mismatch_notes)}" if mismatch_notes else None

        return VLMResponse(
            model_id=original_model_id,
            prompt_version=request.prompt_version,
            recipe=primary_recipe,
            analysis=data.get("analysis"),
            keep_original=data.get("keep_original"),
            recommendations=recommendations,
            advice_text=recommendations[0].get("rationale") if recommendations else data.get("keep_original", {}).get("reason"),
            retake_advice=data.get("retake_advice"),
            is_real_model=False,  # Replay is offline provenance, not live model execution
            fixture_source=self.replay_file_path,
            raw_response_text=json.dumps(data, ensure_ascii=False),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tokens_used=tokens_used,
            replayed_envelope=envelope,
            status=status,
            error_reason=error_reason,
        )


class LocalTransformersProvider(VLMProvider):
    """Runs open-weights Qwen2.5-VL model on remote GPU runtime (e.g. Google Colab).

    Configurable precision, quantization, min/max pixels, token limits, and cooperative time budget.
    All inference stages are guarded by robust error boundaries (OOM, timeout, preprocess, decode).
    Never retries automatically on OOM.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = False,
        compute_dtype: Optional[Any] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        max_new_tokens: int = 512,
        time_budget_seconds: float = 60.0,
    ):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.compute_dtype = compute_dtype
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.time_budget_seconds = time_budget_seconds
        self._model = None
        self._processor = None
        self.model_revision: Optional[str] = None
        self._load_latency_ms: float = 0.0

    def get_effective_config(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "provider_type": "local_hf",
            "device": self.device,
            "load_in_4bit": self.load_in_4bit,
            "dtype": str(self.compute_dtype) if self.compute_dtype is not None else None,
            "compute_dtype": str(self.compute_dtype) if self.compute_dtype is not None else None,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "max_new_tokens": self.max_new_tokens,
            "time_budget_seconds": self.time_budget_seconds,
        }

    def _initialize_model(self):
        if self._model is not None:
            return

        t_load_start = time.perf_counter()
        import torch
        if not torch.cuda.is_available() and self.device == "cuda":
            raise RuntimeError(
                "CUDA GPU is not available. LocalTransformersProvider requires a remote GPU runtime "
                "(e.g., Google Colab with T4/L4 GPU). For local CPU testing, use MockVLMProvider."
            )

        try:
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                f"Missing required packages for VLM inference: {e}. "
                "Install with: pip install torch transformers qwen-vl-utils accelerate bitsandbytes"
            )

        if self.compute_dtype is None:
            self.compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        dtype = self.compute_dtype

        kwargs: Dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.device if self.device != "cpu" else None

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        if hasattr(self._model, "dtype") and self._model.dtype is not None:
            self.compute_dtype = self._model.dtype
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.model_revision = getattr(getattr(self._model, "config", None), "_commit_hash", None)
        self._load_latency_ms = (time.perf_counter() - t_load_start) * 1000.0

    def generate(self, request: VLMRequest) -> VLMResponse:
        t0 = time.perf_counter()

        # STAGE 1: Model initialization
        try:
            self._initialize_model()
        except Exception as e:
            return VLMResponse(
                model_id=self.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="error",
                error_reason=f"Failed to initialize VLM model: {e}",
                is_real_model=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                load_latency_ms=self._load_latency_ms,
            )

        # STAGE 2: Image loading and prompt preparation
        try:
            raw_pil = Image.open(io.BytesIO(request.image_bytes))
            # Ensure upright coordinate system and standard RGB format
            pil_image = ImageOps.exif_transpose(raw_pil).convert("RGB")
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError:
                process_vision_info = None
        except Exception as e:
            return VLMResponse(
                model_id=self.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="preprocess_error",
                error_reason=f"Failed to preprocess image: {e}",
                is_real_model=True,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                load_latency_ms=self._load_latency_ms,
            )

        try:
            prompt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "prompts"))
            sys_prompt_file = os.path.join(prompt_dir, "vlm_system_prompt_v1.txt")
            sys_prompt_text = ""
            if os.path.exists(sys_prompt_file):
                with open(sys_prompt_file, "r", encoding="utf-8") as f:
                    sys_prompt_text = f.read()

            user_content = (
                f"Mục tiêu xử lý ảnh (goal): {request.goal}.\n"
                "Hãy phân tích ảnh chân dung này và đề xuất tối đa 2 phương án chỉnh sửa nhẹ cùng lời khuyên chụp lại.\n"
                "BẮT BUỘC trả về duy nhất 1 JSON object tuân thủ schema quy định, không kèm văn bản giải thích bên ngoài."
            )

            messages = [
                {"role": "system", "content": sys_prompt_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": user_content},
                    ],
                },
            ]

            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self._processor(text=[text], images=[pil_image], return_tensors="pt")

            if self.device != "cpu" and hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
        except Exception as e:
            return VLMResponse(
                model_id=self.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="preprocess_error",
                error_reason=f"Preprocessing error during tokenization: {e}",
                is_real_model=True,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                load_latency_ms=self._load_latency_ms,
            )

        # STAGE 3: Model generation with cooperative time budget stopping criteria and OOM boundary
        import torch
        t_gen_start = time.perf_counter()
        time_criteria = TimeBudgetStoppingCriteria(
            start_time=t_gen_start,
            time_budget_seconds=self.time_budget_seconds,
        )
        stopping_criteria = StoppingCriteriaList([time_criteria])

        try:
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    stopping_criteria=stopping_criteria,
                )
        except Exception as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or (hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError)):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return VLMResponse(
                    model_id=self.model_id,
                    prompt_version=request.prompt_version,
                    recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                    status="oom",
                    error_reason=f"CUDA Out of Memory during generation: {e}",
                    is_real_model=True,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    load_latency_ms=self._load_latency_ms,
                    generation_latency_ms=(time.perf_counter() - t_gen_start) * 1000.0,
                )
            return VLMResponse(
                model_id=self.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="generation_error",
                error_reason=f"Model generation error: {e}",
                is_real_model=True,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                load_latency_ms=self._load_latency_ms,
                generation_latency_ms=(time.perf_counter() - t_gen_start) * 1000.0,
            )

        gen_elapsed = time.perf_counter() - t_gen_start
        if time_criteria.timed_out or gen_elapsed >= self.time_budget_seconds:
            return VLMResponse(
                model_id=self.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="timeout",
                error_reason=f"Generation stopped: exceeded cooperative time budget of {self.time_budget_seconds}s.",
                is_real_model=True,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                load_latency_ms=self._load_latency_ms,
                generation_latency_ms=gen_elapsed * 1000.0,
            )

        # STAGE 4: Decoding generated tokens
        try:
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        except Exception as e:
            return VLMResponse(
                model_id=self.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                status="decode_error",
                error_reason=f"Error decoding generated tokens: {e}",
                is_real_model=True,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                load_latency_ms=self._load_latency_ms,
                generation_latency_ms=gen_elapsed * 1000.0,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0

        cleaned_json = output_text.strip()
        if cleaned_json.startswith("```json"):
            cleaned_json = cleaned_json[7:]
        if cleaned_json.startswith("```"):
            cleaned_json = cleaned_json[3:]
        if cleaned_json.endswith("```"):
            cleaned_json = cleaned_json[:-3]
        cleaned_json = cleaned_json.strip()

        # STAGE 5: JSON parsing and Schema validation
        try:
            parsed = json.loads(cleaned_json)
        except Exception as e:
            return VLMResponse(
                model_id=self.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                raw_response_text=output_text,
                status="rejected",
                error_reason=f"Failed to parse model output as JSON: {e}",
                is_real_model=True,
                latency_ms=elapsed,
                load_latency_ms=self._load_latency_ms,
                generation_latency_ms=gen_elapsed * 1000.0,
            )

        is_valid, errors = validate_vlm_schema(parsed)
        if not is_valid:
            return VLMResponse(
                model_id=self.model_id,
                prompt_version=request.prompt_version,
                recipe={"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None},
                raw_response_text=output_text,
                status="rejected",
                error_reason=f"Model output schema validation failed: {'; '.join(errors)}",
                is_real_model=True,
                latency_ms=elapsed,
                load_latency_ms=self._load_latency_ms,
                generation_latency_ms=gen_elapsed * 1000.0,
            )

        # Count tokens if measurable
        in_tokens = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else 0
        out_tokens = int(generated_ids_trimmed[0].shape[-1]) if generated_ids_trimmed else 0
        tokens_used = {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "is_simulated": False,
        }

        recommendations = parsed.get("recommendations", [])
        primary_recipe = (
            recommendations[0]["recipe"]
            if recommendations
            else {"pipeline_version": PIPELINE_VERSION, "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "crop": None}
        )

        return VLMResponse(
            model_id=self.model_id,
            prompt_version=request.prompt_version,
            recipe=primary_recipe,
            analysis=parsed.get("analysis"),
            keep_original=parsed.get("keep_original"),
            recommendations=recommendations,
            advice_text=recommendations[0].get("rationale") if recommendations else parsed.get("keep_original", {}).get("reason"),
            retake_advice=parsed.get("retake_advice"),
            is_real_model=True,
            raw_response_text=output_text,
            latency_ms=elapsed,
            load_latency_ms=self._load_latency_ms,
            generation_latency_ms=gen_elapsed * 1000.0,
            tokens_used=tokens_used,
            status="success",
        )


SUPPORTED_PROMPT_VERSIONS = {"prompt_v1"}


class VLMAdapter:
    """High-level VLM adapter managing request preparation, provider dispatch, and fixtures."""

    @staticmethod
    def process(
        image_bytes: bytes,
        goal: str,
        provider: Optional[VLMProvider] = None,
        image_id: str = "image_input_01",
        prompt_version: str = "prompt_v1",
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        original_image_sha256: Optional[str] = None,
        original_dimensions: Optional[Tuple[int, int]] = None,
    ) -> VLMResponse:
        """Construct a validated VLMRequest and dispatch to provider."""
        if prompt_version not in SUPPORTED_PROMPT_VERSIONS:
            raise ValueError(
                f"Unsupported prompt_version '{prompt_version}'. Supported versions: {sorted(SUPPORTED_PROMPT_VERSIONS)}"
            )

        if provider is None:
            provider = MockVLMProvider()

        # Compute SHA-256 and dimensions of the provided image bytes (normalized upright sRGB)
        img_sha = hashlib.sha256(image_bytes).hexdigest()
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            dimensions = pil_img.size
        except Exception:
            dimensions = (0, 0)

        req = VLMRequest(
            image_bytes=image_bytes,
            image_id=image_id,
            goal=goal,
            image_sha256=img_sha,
            image_dimensions=dimensions,
            prompt_version=prompt_version,
            model_id=model_id,
            original_image_sha256=original_image_sha256 or img_sha,
            original_dimensions=original_dimensions or dimensions,
        )

        return provider.generate(req)

    @staticmethod
    def load_fixture_from_file(filepath: str) -> VLMResponse:
        """Load a saved candidate recipe fixture from a JSON file (Phase 02A backward compatibility)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fixture file does not exist: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in fixture file '{filepath}': {e}")
        except Exception as e:
            raise ValueError(f"Error reading fixture file '{filepath}': {e}")

        if not isinstance(data, dict):
            raise ValueError(
                f"Fixture file '{filepath}' must contain a JSON object at root, got {type(data).__name__}"
            )

        recipe = data.get("recipe", data)
        if not isinstance(recipe, dict):
            raise ValueError(
                f"Recipe in fixture '{filepath}' must be a JSON object, got {type(recipe).__name__}"
            )

        model_id = str(data.get("model_id", "imported_fixture"))
        prompt_version = data.get("prompt_version", "fixture_v0")
        advice_text = data.get("advice_text")
        retake_advice = data.get("retake_advice")

        return VLMResponse(
            model_id=model_id,
            prompt_version=str(prompt_version) if prompt_version is not None else "fixture_v0",
            recipe=recipe,
            advice_text=str(advice_text) if advice_text is not None else None,
            retake_advice=str(retake_advice) if retake_advice is not None else None,
            is_real_model=False,
            fixture_source=filepath,
        )
