"""Photo Advisor Package - Phase 02A Offline Baseline."""

from photo_advisor.types import (
    PIPELINE_VERSION,
    PROCESSOR_VERSION,
    GUARDRAIL_VERSION,
    RULE_VERSION,
    Recipe,
    ValidationResult,
    ProcessingResult,
    CandidateRegistryRecord,
)
from photo_advisor.validator import validate_and_clamp_recipe
from photo_advisor.processor import normalize_input_image, process_image, calculate_clipping_fractions
from photo_advisor.rule_engine import compute_rule_v0_recipe
from photo_advisor.registry import CandidateRegistry, TrialRecord
from photo_advisor.vlm_adapter import VLMAdapter

__all__ = [
    "PIPELINE_VERSION",
    "PROCESSOR_VERSION",
    "GUARDRAIL_VERSION",
    "RULE_VERSION",
    "Recipe",
    "ValidationResult",
    "ProcessingResult",
    "CandidateRegistryRecord",
    "validate_and_clamp_recipe",
    "normalize_input_image",
    "process_image",
    "calculate_clipping_fractions",
    "compute_rule_v0_recipe",
    "CandidateRegistry",
    "TrialRecord",
    "VLMAdapter",
]
