"""Command Line Interface for Photo Advisor Phase 02A & 02B Offline Runner.

Usage:
    python3 -m photo_advisor.cli --image <path> --goal <goal> [--candidate-1 <json>] [--candidate-2 <json>] [--output-dir <path>]
    python3 -m photo_advisor.cli --synthetic --goal natural_portrait [--output-dir <path>]
    python3 -m photo_advisor.cli --synthetic --goal natural_portrait --provider mock [--output-dir <path>]
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from photo_advisor.html_report import generate_static_html
from photo_advisor.processor import compute_sha256, normalize_input_image, process_image
from photo_advisor.registry import CandidateRegistry
from photo_advisor.rule_engine import compute_rule_v0_recipe
from photo_advisor.types import (
    CandidateRegistryRecord,
    GUARDRAIL_VERSION,
    MIN_EXPORT_DIMENSION,
    PIPELINE_VERSION,
    PROCESSOR_VERSION,
    RULE_VERSION,
    Recipe,
    sanitize_value_for_json,
)
from photo_advisor.validator import validate_and_clamp_recipe
from photo_advisor.vlm_adapter import (
    LocalTransformersProvider,
    MockVLMProvider,
    ReplayVLMProvider,
    SUPPORTED_PROMPT_VERSIONS,
    VLMAdapter,
    VLMProvider,
    VLMResponse,
)


def create_synthetic_test_image(width: int = 800, height: int = 600, style: str = "portrait") -> bytes:
    """Generate a deterministic synthetic test image in memory for offline testing."""
    img = Image.new("RGB", (width, height), color=(220, 215, 205))
    draw = ImageDraw.Draw(img)

    # Background gradient
    for y in range(height):
        r = int(180 + 40 * (y / height))
        g = int(200 + 30 * (y / height))
        b = int(220 + 20 * (y / height))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    if style == "silhouette":
        draw.ellipse([width * 0.35, height * 0.25, width * 0.65, height * 0.85], fill=(20, 15, 15))
    else:
        # Synthetic portrait subject: torso and head
        draw.rounded_rectangle([width * 0.3, height * 0.45, width * 0.7, height * 0.95], radius=20, fill=(45, 75, 115))
        draw.ellipse([width * 0.38, height * 0.18, width * 0.62, height * 0.48], fill=(225, 185, 155))
        draw.ellipse([width * 0.44, height * 0.28, width * 0.47, height * 0.32], fill=(40, 30, 25))
        draw.ellipse([width * 0.53, height * 0.28, width * 0.56, height * 0.32], fill=(40, 30, 25))
        draw.arc([width * 0.46, height * 0.36, width * 0.54, height * 0.42], start=0, end=180, fill=(180, 70, 70), width=3)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _safe_write_bytes(dest_path: str, data: bytes, input_canonical: Optional[str]) -> None:
    """Safely write bytes to destination ensuring no collision with input file canonical path."""
    dest_abs = os.path.abspath(dest_path)
    dest_real = os.path.realpath(dest_abs)
    if input_canonical:
        if dest_real == input_canonical or dest_abs == input_canonical:
            raise PermissionError(
                f"Safety conflict: Output path '{dest_path}' matches input file canonical path '{input_canonical}'. Refusing write."
            )
    with open(dest_path, "wb") as f:
        f.write(data)


def run_pipeline(
    image_bytes: bytes,
    goal: str,
    output_dir: str,
    candidate_fixtures: Optional[List[str]] = None,
    image_id: str = "image_input_01",
    input_file_path: Optional[str] = None,
    run_id: Optional[str] = None,
    input_source: str = "synthetic",
    provider_type: str = "fixtures",
    replay_file: Optional[str] = None,
    prompt_version: str = "prompt_v1",
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    mock_simulation_mode: str = "normal",
    provider: Optional[VLMProvider] = None,
) -> Dict[str, Any]:
    """Execute offline baseline comparison pipeline and export all artifacts.

    Guarantees:
    - Never mutates or overwrites the input image file.
    - Fails immediately if any output artifact name already exists in output_dir.
    - Exports manifest.json with strict validation.
    - Exports reference_input.png and candidate previews.
    - Exports vlm_raw_response.json when VLM provider is invoked.
    - Exports index.html visual inspection report.
    """
    if prompt_version not in SUPPORTED_PROMPT_VERSIONS:
        raise ValueError(
            f"Unsupported prompt_version '{prompt_version}'. Supported versions: {sorted(SUPPORTED_PROMPT_VERSIONS)}"
        )

    if run_id is None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{ts}_{uuid.uuid4().hex[:8]}"

    # Determine canonical path of input file if provided
    input_canonical = os.path.realpath(os.path.abspath(input_file_path)) if input_file_path else None

    use_vlm = (provider is not None) or (provider_type in ("mock", "replay", "local_hf"))

    # Step A1: Pre-compute all target artifact paths and verify collision / overwrite safety
    target_artifacts = [
        "reference_input.png",
        "preview_noop.png",
        "preview_rule.png",
        "manifest.json",
        "index.html",
    ]
    if candidate_fixtures:
        for idx in range(1, len(candidate_fixtures[:2]) + 1):
            target_artifacts.append(f"preview_candidate_{idx}.png")
    elif use_vlm:
        target_artifacts.append("preview_candidate_1.png")
        target_artifacts.append("preview_candidate_2.png")
        target_artifacts.append("vlm_raw_response.json")

    # Safety checks BEFORE any disk writes
    for fname in target_artifacts:
        target_path = os.path.abspath(os.path.join(output_dir, fname))
        target_real = os.path.realpath(target_path)
        # Check collision with input file
        if input_canonical:
            if target_real == input_canonical or target_path == input_canonical:
                raise PermissionError(
                    f"Safety violation: Input image path '{input_file_path}' resolves to target output artifact '{target_path}'. Refusing write to prevent source file mutation."
                )
        # Check existing artifact overwrite
        if os.path.exists(target_path):
            raise FileExistsError(
                f"Safety violation: Output directory '{output_dir}' already contains existing artifact '{fname}'. Overwrite is prohibited to guarantee immutability."
            )

    os.makedirs(output_dir, exist_ok=True)
    registry = CandidateRegistry(run_id=run_id)
    run_start_time = time.perf_counter()

    # Original input hash and dimensions
    input_sha256 = compute_sha256(image_bytes)
    try:
        with Image.open(io.BytesIO(image_bytes)) as raw_pil:
            raw_w, raw_h = raw_pil.size
    except Exception:
        raw_w, raw_h = (0, 0)

    # 1. Normalize image (EXIF orientation transpose + sRGB ICC conversion)
    try:
        norm_img, norm_logs = normalize_input_image(image_bytes)
    except Exception as e:
        raise ValueError(f"Input image normalization failed: {e}")

    norm_w, norm_h = norm_img.size
    is_input_eligible = (norm_w >= MIN_EXPORT_DIMENSION and norm_h >= MIN_EXPORT_DIMENSION)

    # Model input: normalized upright sRGB image bytes
    ref_buf = io.BytesIO()
    norm_img.save(ref_buf, format="PNG")
    normalized_image_bytes = ref_buf.getvalue()
    model_input_sha256 = compute_sha256(normalized_image_bytes)

    # Save reference input image (never overwrite original user file)
    ref_input_path = os.path.join(output_dir, "reference_input.png")
    _safe_write_bytes(ref_input_path, normalized_image_bytes, input_canonical)

    # 2. Candidate 0: No-op (Giữ gốc)
    noop_recipe = Recipe(
        pipeline_version=PIPELINE_VERSION,
        exposure=0.0,
        contrast=1.0,
        saturation=1.0,
        crop=None,
    )
    noop_res = process_image(image_bytes, noop_recipe)
    noop_preview_filename = "preview_noop.png"
    noop_preview_path = os.path.join(output_dir, noop_preview_filename)

    _safe_write_bytes(noop_preview_path, noop_res.output_image_bytes, input_canonical)

    noop_rec = CandidateRegistryRecord(
        run_id=run_id,
        candidate_id="cand_0_noop",
        image_id=image_id,
        goal_id=goal,
        source_type="noop",
        source_name="Baseline 0: Giữ nguyên gốc (No-op)",
        is_real_model=False,
        prompt_version=None,
        requested_recipe=noop_recipe.to_dict(),
        effective_recipe=noop_recipe.to_dict(),
        processor_version=PROCESSOR_VERSION,
        guardrail_version=GUARDRAIL_VERSION,
        rule_version=None,
        input_sha256=input_sha256,
        output_sha256=noop_res.output_sha256,
        preview_filename=noop_preview_filename,
        status="success",
        notes="Không can thiệp, giữ nguyên ảnh gốc đã chuẩn hóa upright sRGB.",
        model_id=None,
        model_snapshot=None,
        input_source=input_source,
        export_eligible=noop_res.export_eligible,
        export_ineligible_reason=noop_res.export_ineligible_reason,
        delta_clipping=noop_res.delta_clipping,
        attempted_clipping=noop_res.attempted_clipping,
        fallback_reason=noop_res.fallback_reason,
        clamp_warnings=[],
        logs=noop_res.logs,
    )
    registry.add_record(noop_rec)

    # 3. Candidate 1: Rule v0
    rule_recipe, rule_stats, rule_logs = compute_rule_v0_recipe(norm_img, goal=goal)
    rule_val = validate_and_clamp_recipe(
        raw_recipe=rule_recipe.to_dict(),
        image_width=norm_w,
        image_height=norm_h,
        goal=goal,
    )

    rule_preview_filename = "preview_rule.png"
    rule_preview_path = os.path.join(output_dir, rule_preview_filename)

    if rule_val.is_valid and rule_val.effective_recipe:
        rule_proc_res = process_image(image_bytes, rule_val.effective_recipe)
        rule_status = rule_proc_res.status
        rule_eff_dict = rule_proc_res.effective_recipe.to_dict()
        rule_out_sha = rule_proc_res.output_sha256
        rule_delta = rule_proc_res.delta_clipping
        rule_attempted = rule_proc_res.attempted_clipping
        rule_eligible = rule_proc_res.export_eligible
        rule_ineligible_reason = rule_proc_res.export_ineligible_reason
        rule_fallback = rule_proc_res.fallback_reason
        rule_logs_list = rule_proc_res.logs
        _safe_write_bytes(rule_preview_path, rule_proc_res.output_image_bytes, input_canonical)
    else:
        rule_status = f"rejected: {rule_val.rejection_reason}"
        rule_eff_dict = noop_recipe.to_dict()
        rule_out_sha = noop_res.output_sha256
        rule_delta = {"delta_under": 0.0, "delta_over": 0.0}
        rule_attempted = None
        rule_eligible = False
        rule_ineligible_reason = f"rule_rejected: {rule_val.rejection_reason}"
        rule_fallback = rule_val.rejection_reason
        rule_logs_list = [f"Rule rejected: {rule_val.rejection_reason}"]
        _safe_write_bytes(rule_preview_path, noop_res.output_image_bytes, input_canonical)

    rule_rec = CandidateRegistryRecord(
        run_id=run_id,
        candidate_id="cand_1_rule_v0",
        image_id=image_id,
        goal_id=goal,
        source_type="rule",
        source_name="Baseline 1: Rule Engine (rule_v0)",
        is_real_model=False,
        prompt_version=None,
        requested_recipe=rule_val.requested_recipe or rule_recipe.to_dict(),
        effective_recipe=rule_eff_dict,
        processor_version=PROCESSOR_VERSION,
        guardrail_version=GUARDRAIL_VERSION,
        rule_version=RULE_VERSION,
        input_sha256=input_sha256,
        output_sha256=rule_out_sha,
        preview_filename=rule_preview_filename,
        status=rule_status,
        notes=f"Rule v0 applied: exp={rule_eff_dict['exposure']}, con={rule_eff_dict['contrast']}, sat={rule_eff_dict['saturation']}",
        model_id=None,
        model_snapshot=None,
        input_source=input_source,
        export_eligible=rule_eligible,
        export_ineligible_reason=rule_ineligible_reason,
        delta_clipping=rule_delta,
        attempted_clipping=rule_attempted,
        fallback_reason=rule_fallback,
        clamp_warnings=rule_val.warnings,
        logs=rule_logs_list,
    )
    registry.add_record(rule_rec)

    # 4. Process Candidates from Fixtures or VLM Provider
    vlm_resp_obj: Optional[VLMResponse] = None
    actual_provider_type = provider_type
    active_provider = provider
    prov: Optional[VLMProvider] = active_provider

    if candidate_fixtures:
        for idx, fixture_path in enumerate(candidate_fixtures[:2], start=1):
            cand_id = f"cand_{idx + 1}_fixture"
            preview_name = f"preview_candidate_{idx}.png"
            preview_dest = os.path.join(output_dir, preview_name)

            try:
                vlm_resp = VLMAdapter.load_fixture_from_file(fixture_path)
            except Exception as e:
                # Load failed, safe fallback to no-op
                _safe_write_bytes(preview_dest, noop_res.output_image_bytes, input_canonical)
                cand_rec = CandidateRegistryRecord(
                    run_id=run_id,
                    candidate_id=cand_id,
                    image_id=image_id,
                    goal_id=goal,
                    source_type="fixture",
                    source_name=f"Fixture {idx}: {os.path.basename(fixture_path)}",
                    is_real_model=False,
                    prompt_version=None,
                    requested_recipe={"error": str(e)},
                    effective_recipe=noop_recipe.to_dict(),
                    processor_version=PROCESSOR_VERSION,
                    guardrail_version=GUARDRAIL_VERSION,
                    rule_version=None,
                    input_sha256=input_sha256,
                    output_sha256=noop_res.output_sha256,
                    preview_filename=preview_name,
                    status=f"rejected: fixture_load_error",
                    notes=f"Failed to load fixture: {e}",
                    model_id="fixture_loader",
                    model_snapshot=f"fixture:{os.path.basename(fixture_path)}",
                    input_source=input_source,
                    export_eligible=False,
                    export_ineligible_reason=f"fixture_load_error: {e}",
                    delta_clipping={"delta_under": 0.0, "delta_over": 0.0},
                    attempted_clipping=None,
                    fallback_reason=str(e),
                    clamp_warnings=[],
                    logs=[f"Fixture load error: {e}"],
                )
                registry.add_record(cand_rec)
                continue

            val_res = validate_and_clamp_recipe(
                raw_recipe=vlm_resp.recipe,
                image_width=norm_w,
                image_height=norm_h,
                goal=goal,
            )

            if val_res.is_valid and val_res.effective_recipe:
                p_res = process_image(image_bytes, val_res.effective_recipe)
                _safe_write_bytes(preview_dest, p_res.output_image_bytes, input_canonical)
                cand_status = p_res.status
                eff_dict = p_res.effective_recipe.to_dict()
                out_sha = p_res.output_sha256
                delta_clip = p_res.delta_clipping
                attempted_clip = p_res.attempted_clipping
                cand_eligible = p_res.export_eligible
                cand_ineligible_reason = p_res.export_ineligible_reason
                cand_fallback = p_res.fallback_reason
                cand_logs = p_res.logs
                cand_warns = val_res.warnings
                note_str = vlm_resp.advice_text or f"Fixture recipe applied: exp={eff_dict['exposure']}, con={eff_dict['contrast']}"
            else:
                _safe_write_bytes(preview_dest, noop_res.output_image_bytes, input_canonical)
                cand_status = f"rejected: {val_res.rejection_reason}"
                eff_dict = noop_recipe.to_dict()
                out_sha = noop_res.output_sha256
                delta_clip = {"delta_under": 0.0, "delta_over": 0.0}
                attempted_clip = None
                cand_eligible = False
                cand_ineligible_reason = f"validation_rejected: {val_res.rejection_reason}"
                cand_fallback = val_res.rejection_reason
                cand_logs = [f"Validation rejected: {val_res.rejection_reason}"]
                cand_warns = val_res.warnings
                note_str = f"REJECTED: {val_res.rejection_reason}"

            cand_rec = CandidateRegistryRecord(
                run_id=run_id,
                candidate_id=cand_id,
                image_id=image_id,
                goal_id=goal,
                source_type="fixture",
                source_name=f"Fixture {idx}: {os.path.basename(fixture_path)}",
                is_real_model=False,
                prompt_version=vlm_resp.prompt_version,
                requested_recipe=val_res.requested_recipe or vlm_resp.recipe,
                effective_recipe=eff_dict,
                processor_version=PROCESSOR_VERSION,
                guardrail_version=GUARDRAIL_VERSION,
                rule_version=None,
                input_sha256=input_sha256,
                output_sha256=out_sha,
                preview_filename=preview_name,
                status=cand_status,
                notes=note_str,
                model_id=vlm_resp.model_id,
                model_snapshot=f"fixture:{os.path.basename(fixture_path)}",
                input_source=input_source,
                export_eligible=cand_eligible,
                export_ineligible_reason=cand_ineligible_reason,
                delta_clipping=delta_clip,
                attempted_clipping=attempted_clip,
                fallback_reason=cand_fallback,
                clamp_warnings=cand_warns,
                logs=cand_logs,
            )
            registry.add_record(cand_rec)

    elif use_vlm:
        if active_provider is not None:
            prov = active_provider
            actual_provider_type = getattr(prov, "provider_type", None) or (
                "mock" if isinstance(prov, MockVLMProvider)
                else "replay" if isinstance(prov, ReplayVLMProvider)
                else "local_hf" if isinstance(prov, LocalTransformersProvider)
                else prov.__class__.__name__.lower().replace("vlmprovider", "")
            )
        elif provider_type == "mock":
            prov = MockVLMProvider(simulation_mode=mock_simulation_mode)
            actual_provider_type = "mock"
        elif provider_type == "replay":
            if not replay_file:
                raise ValueError("Replay provider requested but --replay-response file is missing.")
            prov = ReplayVLMProvider(replay_file_path=replay_file)
            actual_provider_type = "replay"
        elif provider_type == "local_hf":
            prov = LocalTransformersProvider(model_id=model_id)
            actual_provider_type = "local_hf"
        else:
            prov = MockVLMProvider()
            actual_provider_type = "mock"

        # Pass NORMALIZED upright sRGB image bytes to VLM provider (Fixes C2)
        vlm_resp = VLMAdapter.process(
            image_bytes=normalized_image_bytes,
            goal=goal,
            provider=prov,
            image_id=image_id,
            prompt_version=prompt_version,
            model_id=model_id,
            original_image_sha256=input_sha256,
            original_dimensions=(raw_w, raw_h),
        )
        vlm_resp_obj = vlm_resp

        model_rev = getattr(prov, "model_revision", None)
        effective_cfg = prov.get_effective_config() if hasattr(prov, "get_effective_config") else {}
        snapshot_tag = model_rev or (f"model_version:{vlm_resp.model_id}" if vlm_resp.is_real_model else f"provider:{actual_provider_type}")

        if vlm_resp.status != "success":
            preview_dest = os.path.join(output_dir, "preview_candidate_1.png")
            _safe_write_bytes(preview_dest, noop_res.output_image_bytes, input_canonical)
            cand_rec = CandidateRegistryRecord(
                run_id=run_id,
                candidate_id="cand_2_vlm_failed",
                image_id=image_id,
                goal_id=goal,
                source_type=f"vlm_{actual_provider_type}",
                source_name=f"VLM ({vlm_resp.model_id}) - {vlm_resp.status.upper()}",
                is_real_model=vlm_resp.is_real_model,
                prompt_version=vlm_resp.prompt_version,
                requested_recipe={"error": vlm_resp.error_reason},
                effective_recipe=noop_recipe.to_dict(),
                processor_version=PROCESSOR_VERSION,
                guardrail_version=GUARDRAIL_VERSION,
                rule_version=None,
                input_sha256=input_sha256,
                output_sha256=noop_res.output_sha256,
                preview_filename="preview_candidate_1.png",
                status=f"{vlm_resp.status}: {vlm_resp.error_reason}",
                notes=f"VLM failure: {vlm_resp.error_reason}",
                model_id=vlm_resp.model_id,
                model_snapshot=snapshot_tag,
                input_source=input_source,
                export_eligible=False,
                export_ineligible_reason=f"vlm_{vlm_resp.status}: {vlm_resp.error_reason}",
                delta_clipping={"delta_under": 0.0, "delta_over": 0.0},
                attempted_clipping=None,
                fallback_reason=vlm_resp.error_reason,
                clamp_warnings=[],
                logs=[f"VLM call failed with {vlm_resp.status}: {vlm_resp.error_reason}"],
                vlm_analysis=vlm_resp.analysis,
                retake_advice=vlm_resp.retake_advice,
            )
            registry.add_record(cand_rec)
        elif not vlm_resp.recommendations:
            preview_dest = os.path.join(output_dir, "preview_candidate_1.png")
            _safe_write_bytes(preview_dest, noop_res.output_image_bytes, input_canonical)
            reason_str = vlm_resp.keep_original.get("reason", "Ảnh gốc đã đạt chất lượng tốt") if vlm_resp.keep_original else "Giữ nguyên gốc"
            cand_rec = CandidateRegistryRecord(
                run_id=run_id,
                candidate_id="cand_2_vlm_keep_orig",
                image_id=image_id,
                goal_id=goal,
                source_type=f"vlm_{actual_provider_type}",
                source_name=f"VLM ({vlm_resp.model_id}) - Giữ nguyên gốc",
                is_real_model=vlm_resp.is_real_model,
                prompt_version=vlm_resp.prompt_version,
                requested_recipe=noop_recipe.to_dict(),
                effective_recipe=noop_recipe.to_dict(),
                processor_version=PROCESSOR_VERSION,
                guardrail_version=GUARDRAIL_VERSION,
                rule_version=None,
                input_sha256=input_sha256,
                output_sha256=noop_res.output_sha256,
                preview_filename="preview_candidate_1.png",
                status="success",
                notes=f"VLM khuyến nghị giữ nguyên gốc: {reason_str}",
                model_id=vlm_resp.model_id,
                model_snapshot=snapshot_tag,
                input_source=input_source,
                export_eligible=noop_res.export_eligible,
                export_ineligible_reason=noop_res.export_ineligible_reason,
                delta_clipping={"delta_under": 0.0, "delta_over": 0.0},
                attempted_clipping=None,
                fallback_reason=None,
                clamp_warnings=[],
                logs=[f"VLM recommended keep original: {reason_str}"],
                vlm_analysis=vlm_resp.analysis,
                retake_advice=vlm_resp.retake_advice,
            )
            registry.add_record(cand_rec)
        else:
            # Register up to 2 recommendation options from VLM
            for opt_idx, opt in enumerate(vlm_resp.recommendations[:2], start=1):
                cand_id = f"cand_{opt_idx + 1}_vlm"
                preview_name = f"preview_candidate_{opt_idx}.png"
                preview_dest = os.path.join(output_dir, preview_name)

                opt_recipe = opt.get("recipe", {})
                val_res = validate_and_clamp_recipe(
                    raw_recipe=opt_recipe,
                    image_width=norm_w,
                    image_height=norm_h,
                    goal=goal,
                )

                if val_res.is_valid and val_res.effective_recipe:
                    p_res = process_image(image_bytes, val_res.effective_recipe)
                    _safe_write_bytes(preview_dest, p_res.output_image_bytes, input_canonical)
                    cand_status = p_res.status
                    eff_dict = p_res.effective_recipe.to_dict()
                    out_sha = p_res.output_sha256
                    delta_clip = p_res.delta_clipping
                    attempted_clip = p_res.attempted_clipping
                    cand_eligible = p_res.export_eligible
                    cand_ineligible_reason = p_res.export_ineligible_reason
                    cand_fallback = p_res.fallback_reason
                    cand_logs = p_res.logs
                    cand_warns = val_res.warnings
                    note_str = f"VLM {opt.get('name')}: {opt.get('rationale')}"
                else:
                    _safe_write_bytes(preview_dest, noop_res.output_image_bytes, input_canonical)
                    cand_status = f"rejected: {val_res.rejection_reason}"
                    eff_dict = noop_recipe.to_dict()
                    out_sha = noop_res.output_sha256
                    delta_clip = {"delta_under": 0.0, "delta_over": 0.0}
                    attempted_clip = None
                    cand_eligible = False
                    cand_ineligible_reason = f"validation_rejected: {val_res.rejection_reason}"
                    cand_fallback = val_res.rejection_reason
                    cand_logs = [f"Validation rejected: {val_res.rejection_reason}"]
                    cand_warns = val_res.warnings
                    note_str = f"REJECTED: {val_res.rejection_reason}"

                cand_rec = CandidateRegistryRecord(
                    run_id=run_id,
                    candidate_id=cand_id,
                    image_id=image_id,
                    goal_id=goal,
                    source_type=f"vlm_{actual_provider_type}",
                    source_name=f"VLM Option {opt_idx}: {opt.get('name', 'Đề xuất')} ({vlm_resp.model_id})",
                    is_real_model=vlm_resp.is_real_model,
                    prompt_version=vlm_resp.prompt_version,
                    requested_recipe=val_res.requested_recipe or opt_recipe,
                    effective_recipe=eff_dict,
                    processor_version=PROCESSOR_VERSION,
                    guardrail_version=GUARDRAIL_VERSION,
                    rule_version=None,
                    input_sha256=input_sha256,
                    output_sha256=out_sha,
                    preview_filename=preview_name,
                    status=cand_status,
                    notes=note_str,
                    model_id=vlm_resp.model_id,
                    model_snapshot=snapshot_tag,
                    input_source=input_source,
                    export_eligible=cand_eligible,
                    export_ineligible_reason=cand_ineligible_reason,
                    delta_clipping=delta_clip,
                    attempted_clipping=attempted_clip,
                    fallback_reason=cand_fallback,
                    clamp_warnings=cand_warns,
                    logs=cand_logs,
                    vlm_analysis=vlm_resp.analysis,
                    retake_advice=vlm_resp.retake_advice,
                )
                registry.add_record(cand_rec)

    total_runtime_ms = (time.perf_counter() - run_start_time) * 1000.0

    # Retrieve real runtime package versions dynamically
    try:
        import importlib.metadata
        pillow_version = importlib.metadata.version("Pillow")
    except Exception:
        pillow_version = getattr(Image, "__version__", "unknown")
    try:
        import importlib.metadata
        numpy_version = importlib.metadata.version("numpy")
    except Exception:
        numpy_version = getattr(np, "__version__", "unknown")

    # 5. Export vlm_raw_response.json artifact with provenance envelope (Fixes C5, D4)
    prompt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "prompts"))
    schema_file = os.path.join(prompt_dir, "vlm_output_schema_v1.json")
    schema_sha256 = ""
    if os.path.exists(schema_file):
        with open(schema_file, "rb") as sf:
            schema_sha256 = hashlib.sha256(sf.read()).hexdigest()

    sys_prompt_file = os.path.join(prompt_dir, "vlm_system_prompt_v1.txt")
    prompt_sha256 = ""
    if os.path.exists(sys_prompt_file):
        with open(sys_prompt_file, "rb") as pf:
            prompt_sha256 = hashlib.sha256(pf.read()).hexdigest()

    model_load_ms = round(getattr(vlm_resp_obj, "load_latency_ms", 0.0), 2) if vlm_resp_obj else 0.0
    model_gen_ms = round(getattr(vlm_resp_obj, "generation_latency_ms", 0.0), 2) if vlm_resp_obj else 0.0
    model_lat_ms = round(getattr(vlm_resp_obj, "latency_ms", 0.0), 2) if vlm_resp_obj else 0.0
    model_rev = getattr(prov, "model_revision", None) if prov else None
    effective_cfg = prov.get_effective_config() if (prov and hasattr(prov, "get_effective_config")) else {}

    if vlm_resp_obj is not None:
        vlm_raw_path = os.path.join(output_dir, "vlm_raw_response.json")

        parsed_resp = None
        if vlm_resp_obj.raw_response_text:
            try:
                parsed_resp = json.loads(vlm_resp_obj.raw_response_text)
            except Exception:
                parsed_resp = None

        raw_bundle = {
            "raw_response_text": vlm_resp_obj.raw_response_text,
            "parsed_response": parsed_resp,
            "envelope": {
                "original_input_sha256": input_sha256,
                "model_input_sha256": model_input_sha256,
                "original_dimensions": {"width": raw_w, "height": raw_h},
                "model_input_dimensions": {"width": norm_w, "height": norm_h},
                "goal": goal,
                "prompt_version": prompt_version,
                "prompt_sha256": prompt_sha256,
                "schema_sha256": schema_sha256,
                "model_id": vlm_resp_obj.model_id,
                "model_revision": model_rev,
                "provider_type": actual_provider_type,
                "effective_config": effective_cfg,
                "is_real_model": vlm_resp_obj.is_real_model,
                "timings_ms": {
                    "model_load_ms": model_load_ms,
                    "model_generation_ms": model_gen_ms,
                    "model_latency_ms": model_lat_ms,
                    "pipeline_total_ms": round(total_runtime_ms, 2),
                },
                "tokens": vlm_resp_obj.tokens_used,
                "status": vlm_resp_obj.status,
                "error_reason": vlm_resp_obj.error_reason,
            },
        }
        if actual_provider_type == "replay" or getattr(vlm_resp_obj, "replayed_envelope", None):
            raw_bundle["replayed_envelope"] = getattr(vlm_resp_obj, "replayed_envelope", None)

        _safe_write_bytes(
            vlm_raw_path,
            json.dumps(sanitize_value_for_json(raw_bundle), ensure_ascii=False, indent=2).encode("utf-8"),
            input_canonical,
        )

    # 6. Build and export manifest.json with strict JSON dump (allow_nan=False)
    manifest = {
        "metadata": {
            "run_id": run_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "environment": f"Mac Darwin ({sys.platform}), Python {sys.version.split()[0]}",
            "dependencies": f"Pillow=={pillow_version}, numpy=={numpy_version}",
            "goal": goal,
            "image_id": image_id,
            "input_source": input_source,
            "input_sha256": input_sha256,
            "original_dimensions": {"width": raw_w, "height": raw_h},
            "model_input_sha256": model_input_sha256,
            "model_input_dimensions": {"width": norm_w, "height": norm_h},
            "input_dimensions": {"width": norm_w, "height": norm_h},  # backward compatibility
            "input_export_eligible": is_input_eligible,
            "pipeline_version": PIPELINE_VERSION,
            "processor_version": PROCESSOR_VERSION,
            "guardrail_version": GUARDRAIL_VERSION,
            "rule_version": RULE_VERSION,
            "total_runtime_ms": round(total_runtime_ms, 2),
            "timings_ms": {
                "model_load_ms": model_load_ms,
                "model_generation_ms": model_gen_ms,
                "pipeline_total_ms": round(total_runtime_ms, 2),
            },
            "normalization_logs": norm_logs,
        },
        "candidates": registry.to_dict_list(),
    }

    if vlm_resp_obj is not None:
        manifest["metadata"]["vlm_summary"] = {
            "provider_type": actual_provider_type,
            "model_id": vlm_resp_obj.model_id,
            "model_revision": model_rev,
            "effective_config": effective_cfg,
            "prompt_version": vlm_resp_obj.prompt_version,
            "prompt_sha256": prompt_sha256,
            "schema_sha256": schema_sha256,
            "status": vlm_resp_obj.status,
            "is_real_model": vlm_resp_obj.is_real_model,
            "analysis": vlm_resp_obj.analysis,
            "keep_original": vlm_resp_obj.keep_original,
            "retake_advice": vlm_resp_obj.retake_advice,
            "timings_ms": {
                "model_load_ms": model_load_ms,
                "model_generation_ms": model_gen_ms,
                "model_latency_ms": model_lat_ms,
            },
            "latency_ms": model_lat_ms,
            "tokens_used": vlm_resp_obj.tokens_used,
        }

    manifest_sanitized = sanitize_value_for_json(manifest)
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_sanitized, f, indent=2, ensure_ascii=False, allow_nan=False)

    # 7. Generate static HTML report
    html_path = os.path.join(output_dir, "index.html")
    generate_static_html(manifest_sanitized, html_path)

    return manifest_sanitized


def main() -> None:
    parser = argparse.ArgumentParser(description="Photo Advisor Phase 02B Offline & Remote VLM Runner")
    parser.add_argument("--image", "--input", dest="image", type=str, help="Path to input image (JPEG or PNG)")
    parser.add_argument("--goal", type=str, default="natural_portrait", help="Style goal (default: natural_portrait)")
    parser.add_argument("--candidate-1", type=str, help="Path to JSON file containing candidate 1 recipe fixture")
    parser.add_argument("--candidate-2", type=str, help="Path to JSON file containing candidate 2 recipe fixture")
    parser.add_argument("--output-dir", type=str, default="work/phase02b/run_output", help="Directory for output artifacts")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test image if no image path provided")
    parser.add_argument("--run-id", type=str, help="Optional run identifier")
    parser.add_argument(
        "--provider",
        type=str,
        default="fixtures",
        choices=["fixtures", "mock", "replay", "local_hf"],
        help="VLM provider type (default: fixtures)",
    )
    parser.add_argument("--replay-response", type=str, help="Path to pre-saved VLM response JSON (for --provider replay)")
    parser.add_argument("--prompt-version", type=str, default="prompt_v1", help="Prompt template version (default: prompt_v1)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="VLM Model ID (default: Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument(
        "--mock-mode",
        type=str,
        default="normal",
        choices=["normal", "refusal", "timeout", "truncated_json", "invalid_schema", "empty_recommendations"],
        help="Simulation mode for MockVLMProvider",
    )

    args = parser.parse_args()

    if not args.image and not args.synthetic:
        parser.print_help()
        sys.exit(1)

    input_file_path = None
    if args.synthetic:
        print("[INFO] Generating synthetic portrait test image (800x600)...")
        img_bytes = create_synthetic_test_image(800, 600, style=args.goal)
        img_id = "synthetic_test_01"
        input_source = "synthetic"
    else:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image file not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        input_file_path = args.image
        img_id = os.path.splitext(os.path.basename(args.image))[0]
        input_source = "user_upload"
        with open(args.image, "rb") as f:
            img_bytes = f.read()

    fixtures = []
    if args.candidate_1:
        fixtures.append(args.candidate_1)
    if args.candidate_2:
        fixtures.append(args.candidate_2)

    try:
        manifest = run_pipeline(
            image_bytes=img_bytes,
            goal=args.goal,
            output_dir=args.output_dir,
            candidate_fixtures=fixtures if fixtures else None,
            image_id=img_id,
            input_file_path=input_file_path,
            run_id=args.run_id,
            input_source=input_source,
            provider_type=args.provider,
            replay_file=args.replay_response,
            prompt_version=args.prompt_version,
            model_id=args.model_id,
            mock_simulation_mode=args.mock_mode,
        )
    except Exception as e:
        print(f"[FATAL ERROR] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

    meta = manifest["metadata"]
    print("=" * 60)
    print(f"Photo Advisor Pipeline Completed Successfully! [Phase 02B]")
    print(f"  - Run ID:          {meta['run_id']}")
    print(f"  - Goal:            {meta['goal']}")
    print(f"  - Input SHA-256:   {meta['input_sha256']}")
    print(f"  - Model SHA-256:   {meta['model_input_sha256']}")
    print(f"  - Total Runtime:   {meta['total_runtime_ms']} ms")
    print(f"  - Output Dir:      {os.path.abspath(args.output_dir)}")
    print(f"  - Manifest JSON:   {os.path.join(args.output_dir, 'manifest.json')}")
    print(f"  - Static HTML:     {os.path.join(args.output_dir, 'index.html')}")
    if "vlm_summary" in meta:
        vlm_s = meta["vlm_summary"]
        print(f"  - VLM Provider:    {vlm_s['provider_type']} ({vlm_s['model_id']})")
        print(f"  - VLM Status:      {vlm_s['status']} (is_real_model={vlm_s['is_real_model']})")
    for cand in manifest["candidates"]:
        print(f"  - [{cand['status'].upper()}] {cand['candidate_id']}: {cand['source_name']} -> {cand['preview_filename']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
