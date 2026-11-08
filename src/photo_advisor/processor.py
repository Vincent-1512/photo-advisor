"""Deterministic Image Processor using Pillow.

Strictly adheres to Appendix Items #1, #4, and #5 in CODEX_REVIEW_03.md and Review 04:
- Uses Pillow 11.3.0.
- Allowlist strictly enforces JPEG and PNG formats.
- Handles alpha transparency: composites onto solid neutral white (255, 255, 255) rather than silent channel drop.
- Pipeline order:
    1. EXIF transpose -> upright orientation.
    2. Convert ICC profile to sRGB (fail on corrupted ICC; assume sRGB if missing).
    3. Ensure 8-bit RGB mode.
    4. Crop (if requested, using floor/ceil bounds on upright image).
    5. Brightness.enhance(2**exposure) [brightness multiplier on encoded sRGB, not physical linear EV].
    6. Contrast.enhance(contrast) [Pillow contrast].
    7. Color.enhance(saturation) [Pillow color].
- Delta clipping v0:
    Measures fractions of pixels with any channel == 0 and any channel == 255.
    Compares output against the EXACT SAME CROPPED REGION of normalized input.
    Rejects and falls back to No-op if delta > 0.01 (1 percentage point).
- Distinguishes attempted clipping from final delivered delta clipping on fallback.
- Enforces export eligibility (dimension >= 400px, no upscaling permitted).
- Fallback preserves original image without overwrite.
- Reproducible deterministic output with SHA-256 calculation.
"""

import hashlib
import io
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageCms, ImageEnhance, ImageOps

from photo_advisor.types import (
    DELTA_CLIPPING_THRESHOLD,
    MIN_EXPORT_DIMENSION,
    PIPELINE_VERSION,
    ProcessingResult,
    Recipe,
)
from photo_advisor.ops.white_balance import apply_white_balance
from photo_advisor.ops.tone import apply_shadows, apply_highlights
from photo_advisor.ops.sharpen import apply_sharpen


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def calculate_clipping_fractions(img_rgb: Image.Image) -> Dict[str, float]:
    """Calculate the fraction of pixels where any RGB channel is 0 (under) or 255 (over)."""
    arr = np.asarray(img_rgb)
    if arr.size == 0:
        return {"fraction_under": 0.0, "fraction_over": 0.0}
    # Any channel is 0
    under_mask = np.any(arr == 0, axis=-1)
    # Any channel is 255
    over_mask = np.any(arr == 255, axis=-1)
    return {
        "fraction_under": float(np.mean(under_mask)),
        "fraction_over": float(np.mean(over_mask)),
    }


def normalize_input_image(image_bytes: bytes) -> Tuple[Image.Image, List[str]]:
    """Decode and normalize input image: check allowlist, EXIF orientation, alpha composite, ICC to sRGB, 8-bit RGB."""
    logs: List[str] = []
    try:
        raw_img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Failed to decode image data: {e}")

    # Enforce strict format allowlist: JPEG and PNG only
    fmt = (raw_img.format or "").upper()
    if fmt not in {"JPEG", "PNG"}:
        raise ValueError(
            f"Unsupported image format '{fmt}'. HEIC/HEIF format is not currently supported; "
            f"pipeline allowlist strictly accepts JPEG and PNG only."
        )

    # 1. EXIF orientation transpose to upright
    transposed_img = ImageOps.exif_transpose(raw_img)
    if transposed_img is not raw_img:
        logs.append("Applied EXIF orientation transpose to obtain upright image coordinate.")

    # 2. Alpha channel policy: Cleanly composite onto neutral white background (255, 255, 255)
    # Do not silently drop alpha channel.
    if transposed_img.mode in ("RGBA", "LA") or "transparency" in transposed_img.info:
        rgba = transposed_img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, rgba)
        transposed_img = composited.convert("RGB")
        logs.append(
            "Alpha channel detected: cleanly composited over neutral white background (255, 255, 255) before RGB conversion."
        )

    # 3. ICC profile conversion to sRGB
    icc_data = raw_img.info.get("icc_profile") or transposed_img.info.get("icc_profile")
    if icc_data:
        try:
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_data))
            srgb_profile = ImageCms.createProfile("sRGB")
            # Apply transformation
            normalized_img = ImageCms.profileToProfile(
                transposed_img, src_profile, srgb_profile, outputMode="RGB"
            )
            logs.append("Converted embedded ICC profile to standard sRGB.")
        except Exception as e:
            raise ValueError(f"Corrupted or invalid embedded ICC profile: {e}. Refusing silent conversion.")
    else:
        logs.append("No embedded ICC profile found; assuming standard sRGB.")
        normalized_img = transposed_img

    # 4. Ensure 8-bit RGB
    if normalized_img.mode != "RGB":
        normalized_img = normalized_img.convert("RGB")
        logs.append(f"Converted image mode from {transposed_img.mode} to RGB 8-bit.")

    return normalized_img, logs


def process_image(
    image_bytes: bytes,
    recipe: Recipe,
    min_dimension_threshold: int = MIN_EXPORT_DIMENSION,
) -> ProcessingResult:
    """Execute deterministic image processing with clipping guardrails and fallback.

    Args:
        image_bytes: Raw binary bytes of the input image.
        recipe: Effective Recipe validated by validator.
        min_dimension_threshold: Minimum allowed dimension (default 400px).

    Returns:
        ProcessingResult with output image, SHA256 hashes, delta clipping, logs, export eligibility.
    """
    start_time = time.perf_counter()
    input_sha256 = compute_sha256(image_bytes)
    logs: List[str] = []

    # Step 1: Decode & Normalize input image
    try:
        norm_img, norm_logs = normalize_input_image(image_bytes)
        logs.extend(norm_logs)
    except Exception as e:
        # Fallback / rejection if decode or ICC fails
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return ProcessingResult(
            status="rejected",
            effective_recipe=recipe,
            input_sha256=input_sha256,
            output_sha256="",
            input_clipping={"fraction_under": 0.0, "fraction_over": 0.0},
            output_clipping={"fraction_under": 0.0, "fraction_over": 0.0},
            delta_clipping={"delta_under": 0.0, "delta_over": 0.0},
            attempted_clipping=None,
            logs=[f"Image normalization failed: {e}"],
            fallback_reason=str(e),
            runtime_ms=elapsed,
            output_image_bytes=b"",
            export_eligible=False,
            export_ineligible_reason=f"normalization_failed: {e}",
        )

    orig_w, orig_h = norm_img.size

    # Check minimum input resolution for export eligibility
    is_eligible = (orig_w >= min_dimension_threshold and orig_h >= min_dimension_threshold)
    dim_ineligible_reason = None
    if not is_eligible:
        dim_ineligible_reason = f"dimension_below_{min_dimension_threshold}px ({orig_w}x{orig_h})"
        logs.append(
            f"Image dimensions ({orig_w}x{orig_h}) are below threshold {min_dimension_threshold}px: "
            f"processed for preview only; export_eligible=False (no upscaling permitted)."
        )

    if recipe.pipeline_version == "pipeline_v1":
        work = norm_img
        if recipe.straighten_deg:
            work = work.rotate(float(recipe.straighten_deg), resample=Image.Resampling.BICUBIC, expand=False)
        if recipe.crop is not None:
            ymin, xmin, ymax, xmax = recipe.crop
            work = work.crop((int(math.floor(xmin*work.width)), int(math.floor(ymin*work.height)), int(math.ceil(xmax*work.width)), int(math.ceil(ymax*work.height))))
        cropped_input = work
        input_clipping = calculate_clipping_fractions(cropped_input)
        encoded = np.asarray(work, dtype=np.float32) / np.float32(255.0)
        linear = np.where(encoded <= .04045, encoded/12.92, ((encoded+.055)/1.055)**2.4).astype(np.float32)
        linear = apply_white_balance(linear, recipe.temperature, recipe.tint)
        if recipe.exposure != 0: linear *= np.float32(2.0 ** float(recipe.exposure))
        linear = apply_shadows(linear, recipe.shadows)
        linear = apply_highlights(linear, recipe.highlights)
        y = np.sum(linear*np.asarray([.2126,.7152,.0722],dtype=np.float32),axis=-1,keepdims=True,dtype=np.float32)
        if recipe.contrast != 1:
            yn=.18+np.float32(recipe.contrast)*(y-.18); linear *= np.divide(yn,y,out=np.ones_like(y),where=y>0)
        if recipe.saturation != 1: linear = y+(linear-y)*np.float32(recipe.saturation)
        linear = apply_sharpen(linear, recipe.sharpen)
        srgb=np.where(linear<=.0031308,linear*12.92,1.055*np.maximum(linear,0)**(1/2.4)-.055)
        final_output=Image.fromarray(np.rint(np.clip(srgb,0,1)*255).astype(np.uint8),"RGB")
        is_eligible = cropped_input.width >= min_dimension_threshold and cropped_input.height >= min_dimension_threshold
        dim_ineligible_reason=None if is_eligible else f"dimension_below_{min_dimension_threshold}px ({cropped_input.width}x{cropped_input.height})"
    else:
      # Step 2: Crop selection (legacy pipeline_v0 path; preserve unchanged operations)
      if recipe.crop is not None:
        ymin, xmin, ymax, xmax = recipe.crop
        left = max(0, min(orig_w, int(math.floor(xmin * orig_w))))
        top = max(0, min(orig_h, int(math.floor(ymin * orig_h))))
        right = max(0, min(orig_w, int(math.ceil(xmax * orig_w))))
        bottom = max(0, min(orig_h, int(math.ceil(ymax * orig_h))))

        cropped_input = norm_img.crop((left, top, right, bottom))
        crop_w, crop_h = cropped_input.size
        if crop_w < min_dimension_threshold or crop_h < min_dimension_threshold:
            is_eligible = False
        logs.append(f"Applied crop box: ({left}, {top}, {right}, {bottom}) [{crop_w}x{crop_h}px].")
      else:
        cropped_input = norm_img
        logs.append("No crop requested; using full upright image.")

    # Measure clipping on normalized cropped input reference
      input_clipping = calculate_clipping_fractions(cropped_input)

    # Step 3: Brightness Enhance (recipe.exposure: multiplier = 2.0 ** exposure)
      multiplier = 2.0 ** float(recipe.exposure)
      bright_img = ImageEnhance.Brightness(cropped_input).enhance(multiplier)
      logs.append(f"Applied Brightness.enhance({multiplier:.6f}) [recipe exposure {recipe.exposure:.6f}].")

    # Step 4: Contrast Enhance
      contrast_img = ImageEnhance.Contrast(bright_img).enhance(float(recipe.contrast))
      logs.append(f"Applied Contrast.enhance({recipe.contrast:.6f}).")

    # Step 5: Color (Saturation) Enhance
      final_output = ImageEnhance.Color(contrast_img).enhance(float(recipe.saturation))
      logs.append(f"Applied Color.enhance({recipe.saturation:.6f}).")

    # Step 6: Delta clipping guardrail (compared against same cropped region of normalized input)
    output_clipping = calculate_clipping_fractions(final_output)
    delta_under = output_clipping["fraction_under"] - input_clipping["fraction_under"]
    delta_over = output_clipping["fraction_over"] - input_clipping["fraction_over"]

    attempted_dict = {
        "delta_under": float(delta_under),
        "delta_over": float(delta_over),
    }

    elapsed = (time.perf_counter() - start_time) * 1000.0

    # Guardrail check: delta > 0.01 (1 percentage point increase)
    if delta_under > DELTA_CLIPPING_THRESHOLD or delta_over > DELTA_CLIPPING_THRESHOLD:
        reason = (
            f"Delta clipping guardrail exceeded threshold {DELTA_CLIPPING_THRESHOLD:.2f}: "
            f"delta_under={delta_under:.4f}, delta_over={delta_over:.4f}. Fallback to No-op."
        )
        logs.append(f"GUARDRAIL VIOLATION: {reason}")

        # Fallback to No-op: normalized input image without enhancements
        fallback_recipe = Recipe(
            pipeline_version=PIPELINE_VERSION,
            exposure=0.0,
            contrast=1.0,
            saturation=1.0,
            crop=None,
        )
        out_bytes_io = io.BytesIO()
        norm_img.save(out_bytes_io, format="PNG")
        output_bytes = out_bytes_io.getvalue()
        output_sha256 = compute_sha256(output_bytes)

        # The effective delta clipping of the delivered fallback image relative to itself is 0
        effective_delta = {"delta_under": 0.0, "delta_over": 0.0}

        fallback_ineligible_reason = (
            dim_ineligible_reason if not is_eligible else f"guardrail_fallback: {reason}"
        )

        return ProcessingResult(
            status="fallback_noop",
            effective_recipe=fallback_recipe,
            input_sha256=input_sha256,
            output_sha256=output_sha256,
            input_clipping=input_clipping,
            output_clipping=output_clipping,
            delta_clipping=effective_delta,
            attempted_clipping=attempted_dict,
            logs=logs,
            fallback_reason=reason,
            runtime_ms=elapsed,
            output_image_bytes=output_bytes,
            export_eligible=False,
            export_ineligible_reason=fallback_ineligible_reason,
        )

    # Encode output to PNG in memory to compute deterministic SHA256
    out_bytes_io = io.BytesIO()
    final_output.save(out_bytes_io, format="PNG")
    output_bytes = out_bytes_io.getvalue()
    output_sha256 = compute_sha256(output_bytes)

    return ProcessingResult(
        status="success",
        effective_recipe=recipe,
        input_sha256=input_sha256,
        output_sha256=output_sha256,
        input_clipping=input_clipping,
        output_clipping=output_clipping,
        delta_clipping=attempted_dict,
        attempted_clipping=attempted_dict,
        logs=logs,
        fallback_reason=None,
        runtime_ms=elapsed,
        output_image_bytes=output_bytes,
        export_eligible=is_eligible,
        export_ineligible_reason=dim_ineligible_reason if not is_eligible else None,
    )
