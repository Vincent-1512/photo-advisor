"""Pydantic response and request models aligned with OpenAPI schemas."""
from enum import Enum
from typing import Annotated, Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

class Goal(str, Enum):
    natural_portrait="natural_portrait"; vibrant="vibrant"; silhouette="silhouette"; full_body="full_body"; keep_context="keep_context"; memorial="memorial"
class JobAccepted(BaseModel): job_id: str; status: str="queued"
class Error(BaseModel): code: str; message: str
class ErrorEnvelope(BaseModel): error: Error
class Recipe(BaseModel):
    model_config=ConfigDict(extra="allow")
    pipeline_version: str
    exposure: float; contrast: float; saturation: float; crop: Optional[list[float]]
class Analysis(BaseModel):
    framing: str; subject_description: str; background_description: str; lighting_condition: str; uncertainty: Optional[str]
class Candidate(BaseModel):
    candidate_id: str; source: str; option_id: Optional[str]; name: str; rationale: str
    requested_recipe: dict[str,Any]; effective_recipe: dict[str,Any]; status: str; fallback_reason: Optional[str]
    export_eligible: bool; preview_url: str; style_id: Optional[str]=None; style_label: Optional[str]=None
class Result(BaseModel): analysis: Optional[Analysis]=None; keep_original: dict[str,Any]; candidates: list[Candidate]; retake_advice: Optional[str]
class Job(BaseModel): job_id: str; status: str; provider_type: Optional[str]; error: Optional[Error]; result: Optional[Result]
CandidateId = Annotated[str, Field(pattern=r"^candidate_[A-Za-z0-9_-]+$")]


class FeedbackInput(BaseModel):
    chosen_candidate_id: Optional[CandidateId]
    preferred_over_original: Optional[bool]
    reason: str = Field(max_length=500)
    session_id: str = Field(pattern=r"^session_[A-Za-z0-9_-]+$")
    shown_order: Optional[list[CandidateId]] = Field(default=None, min_length=1, max_length=6)

    @field_validator("shown_order")
    @classmethod
    def unique_candidates(cls, order):
        if order and len(order) != len(set(order)):
            raise ValueError("shown_order chứa candidate trùng lặp.")
        return order
