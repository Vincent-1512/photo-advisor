"""Photo Advisor Phase 02A - Core Types and Constants.

Locked dependencies:
- Python 3.10.19
- Pillow 11.3.0
- numpy 2.2.6
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Any

# Version identifiers as specified in Codex Review 03 & 04
PIPELINE_VERSION = "pipeline_v0"
PROCESSOR_VERSION = "processor_numpy_linear_v1"
GUARDRAIL_VERSION = "guardrail_v0"
RULE_VERSION = "rule_v0"

# Guardrail and threshold limits
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG"}
MIN_EXPORT_DIMENSION = 400         # Pilot export threshold: min 400px each dimension
MIN_CROP_AREA_RATIO = 0.30          # Minimum crop area fraction relative to input
DELTA_CLIPPING_THRESHOLD = 0.01     # Max permissible clipping increase: 0.01 (1 percentage point)

# Parameter clamps (conservative bounds for pilot v0)
EXPOSURE_CLAMP_MIN = -1.5
EXPOSURE_CLAMP_MAX = 1.5
CONTRAST_CLAMP_MIN = 0.7
CONTRAST_CLAMP_MAX = 1.3
SATURATION_CLAMP_MIN = 0.8
SATURATION_CLAMP_MAX = 1.3
TEMPERATURE_CLAMP_MIN, TEMPERATURE_CLAMP_MAX = -1.0, 1.0
TINT_CLAMP_MIN, TINT_CLAMP_MAX = -1.0, 1.0
SHADOWS_CLAMP_MIN, SHADOWS_CLAMP_MAX = -0.5, 0.5
HIGHLIGHTS_CLAMP_MIN, HIGHLIGHTS_CLAMP_MAX = -0.5, 0.5
SHARPEN_CLAMP_MIN, SHARPEN_CLAMP_MAX = 0.0, 1.0

# Goals
GOAL_NATURAL = "natural_portrait"
GOAL_VIBRANT = "vibrant"
GOAL_SILHOUETTE = "silhouette"
GOAL_FULL_BODY = "full_body"
GOAL_KEEP_CONTEXT = "keep_context"
GOAL_MEMORIAL = "memorial"
GOALS_PRESERVE_ALL = {"full_body", "keep_context", "landscape"}


def sanitize_value_for_json(val: Any) -> Any:
    """Recursively convert NaN / Infinity floats to safe string representations for strict JSON."""
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        return val
    elif isinstance(val, dict):
        return {str(k): sanitize_value_for_json(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [sanitize_value_for_json(v) for v in val]
    return val


@dataclass(frozen=True)
class Recipe:
    """Immutable recipe specifying adjustments to be executed deterministically."""
    pipeline_version: str
    exposure: float          # Log2 brightness multiplier on sRGB encoded image
    contrast: float          # Pillow Contrast factor
    saturation: float        # Pillow Color factor
    crop: Optional[Tuple[float, float, float, float]] = None  # (ymin, xmin, ymax, xmax) normalized [0, 1]
    temperature: float = 0.0
    tint: float = 0.0
    shadows: float = 0.0
    highlights: float = 0.0
    sharpen: float = 0.0
    straighten_deg: float = 0.0
    local: None = None

    def to_dict(self) -> Dict[str, Any]:
        """Exact machine representation without loss of precision."""
        result = {
            "pipeline_version": self.pipeline_version,
            "exposure": float(self.exposure),
            "contrast": float(self.contrast),
            "saturation": float(self.saturation),
            "crop": list(self.crop) if self.crop is not None else None,
        }
        if self.pipeline_version == "pipeline_v1":
            result.update(temperature=self.temperature, tint=self.tint, shadows=self.shadows,
                highlights=self.highlights, sharpen=self.sharpen,
                straighten_deg=self.straighten_deg, local=self.local)
        return result

    def to_display_dict(self) -> Dict[str, Any]:
        """Formatted dictionary with rounded values for HTML presentation."""
        result = {
            "pipeline_version": self.pipeline_version,
            "exposure": round(float(self.exposure), 4),
            "contrast": round(float(self.contrast), 4),
            "saturation": round(float(self.saturation), 4),
            "crop": [round(c, 4) for c in self.crop] if self.crop is not None else None,
        }
        if self.pipeline_version == "pipeline_v1":
            result.update(temperature=round(self.temperature, 4), tint=round(self.tint, 4),
                shadows=round(self.shadows, 4), highlights=round(self.highlights, 4),
                sharpen=round(self.sharpen, 4), straighten_deg=round(self.straighten_deg, 4),
                local=self.local)
        return result


@dataclass
class ValidationResult:
    """Result of recipe validation and parameter clamping."""
    is_valid: bool
    requested_recipe: Optional[Dict[str, Any]] = None
    effective_recipe: Optional[Recipe] = None
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """Result of image processing, delta-clipping check, and rendering."""
    status: str                         # "success", "fallback_noop", "rejected"
    effective_recipe: Recipe
    input_sha256: str
    output_sha256: str
    input_clipping: Dict[str, float]    # {"fraction_under": float, "fraction_over": float}
    output_clipping: Dict[str, float]   # {"fraction_under": float, "fraction_over": float}
    delta_clipping: Dict[str, float]    # {"delta_under": float, "delta_over": float}
    logs: List[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None
    runtime_ms: float = 0.0
    output_image_bytes: bytes = b""
    attempted_clipping: Optional[Dict[str, float]] = None
    export_eligible: bool = True
    export_ineligible_reason: Optional[str] = None


@dataclass
class CandidateRegistryRecord:
    """Immutable record for candidate tracking in offline pilot runs.

    Candidate Identity Model:
    Candidates are uniquely identified across trials, datasets, and pipelines
    by the compound key (run_id, candidate_id).
    """
    candidate_id: str
    image_id: str
    goal_id: str
    source_type: str                    # "noop", "rule_v0", "imported_fixture"
    source_name: str
    is_real_model: bool                 # False for offline fixtures
    prompt_version: Optional[str]
    requested_recipe: Any               # Sanitized dict or string representation if malformed
    effective_recipe: Dict[str, Any]
    processor_version: str
    guardrail_version: str
    rule_version: Optional[str]
    input_sha256: str
    output_sha256: str
    preview_filename: str
    status: str
    notes: str
    run_id: str = "run_default"
    model_id: Optional[str] = None
    model_snapshot: Optional[str] = None
    input_source: str = "synthetic"     # "synthetic" or "user_file"
    export_eligible: bool = True
    export_ineligible_reason: Optional[str] = None
    delta_clipping: Dict[str, float] = field(default_factory=dict)
    attempted_clipping: Optional[Dict[str, float]] = None
    fallback_reason: Optional[str] = None
    clamp_warnings: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    vlm_analysis: Optional[Dict[str, Any]] = None
    retake_advice: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a clean, JSON-serializable dictionary."""
        d = asdict(self)
        return sanitize_value_for_json(d)
