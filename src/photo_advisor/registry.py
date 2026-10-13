"""Immutable Candidate Registry and Trial Tracking.

Strictly adheres to Appendix Item #6 in CODEX_REVIEW_03.md and Review 04 (Finding A5):
- Registry stores and returns deep-copied snapshots so caller mutations never affect internal state.
- Scopes candidate records by run_id.
- Records dedicated model_id, model_snapshot, input_source, and export_eligible.
- Tracks source type, real-model vs fixture status, and input/output SHA256 hashes.
- Defines TrialRecord schema linking candidate IDs with randomized left/right placement.
"""

import copy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from photo_advisor.types import CandidateRegistryRecord


@dataclass
class TrialRecord:
    """Trial record linking comparison candidates to registry."""
    trial_id: str
    image_id: str
    goal_id: str
    reference_candidate_id: str  # Always points to No-op record
    left_candidate_id: str       # Randomized candidate placement
    right_candidate_id: str      # Randomized candidate placement
    presentation_order_seed: int
    rater_id: Optional[str] = None
    choice: Optional[str] = None # 'left', 'right', 'tie', 'both_worse_than_original', 'uncertain'
    rationale_note: Optional[str] = None
    disagreement_flag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CandidateRegistry:
    """In-memory immutable candidate registry for an execution run."""

    def __init__(self, run_id: str = "run_default") -> None:
        self.run_id = run_id
        self._records: Dict[str, CandidateRegistryRecord] = {}

    def add_record(self, record: CandidateRegistryRecord) -> None:
        """Store a deep-copied snapshot of the record in the registry."""
        if record.candidate_id in self._records:
            raise ValueError(
                f"Candidate ID '{record.candidate_id}' already exists in registry for run '{self.run_id}'."
            )
        # Deep-copy to guarantee caller modifications after add do not affect registry state
        self._records[record.candidate_id] = copy.deepcopy(record)

    def get_record(self, candidate_id: str) -> Optional[CandidateRegistryRecord]:
        """Return a deep-copied snapshot of the record."""
        rec = self._records.get(candidate_id)
        return copy.deepcopy(rec) if rec is not None else None

    def all_records(self) -> List[CandidateRegistryRecord]:
        """Return deep-copied list of all stored records."""
        return [copy.deepcopy(r) for r in self._records.values()]

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Return list of serialized dictionaries for manifest export."""
        return [r.to_dict() for r in self._records.values()]
