# src/sisf/schemas/policies.py
"""
This file defines the formal "Policy Language" for the SISF.
"""
import uuid
from abc import ABC
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union

class PolicyAction(str, Enum):
    """The primitive enforcement actions the Warden can perform."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REWRITE = "REWRITE"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"

class BasePolicy(BaseModel, ABC):
    """Abstract base class for all policy types."""
    id: str = Field(default_factory=lambda: f"pol_{uuid.uuid4().hex}")
    description: str = Field(...)
    action: PolicyAction
    created_by: str = Field("psm")
    breach_context_id: Optional[str] = Field(None)
    # --- NEW FIELD FOR STATE MANAGEMENT ---
    is_active: bool = Field(True, description="Whether the policy is currently enforced by the Warden.")

class HeuristicPolicy(BasePolicy):
    """A policy that triggers based on a regular expression."""
    type: Literal["HEURISTIC"] = "HEURISTIC"
    regex_pattern: str
    action: Literal[PolicyAction.BLOCK, PolicyAction.FLAG_FOR_REVIEW]

class EmbeddingSimilarityPolicy(BasePolicy):
    """A policy that triggers based on semantic similarity."""
    type: Literal["EMBEDDING_SIMILARITY"] = "EMBEDDING_SIMILARITY"
    reference_embedding: List[float]
    similarity_threshold: float = Field(..., gt=0, lt=1)
    action: Literal[PolicyAction.BLOCK, PolicyAction.FLAG_FOR_REVIEW]

class RewritePolicy(BasePolicy):
    """A policy that actively modifies a prompt or response."""
    type: Literal["REWRITE"] = "REWRITE"
    match_pattern: str
    rewrite_template: str
    action: Literal[PolicyAction.REWRITE] = PolicyAction.REWRITE

Policy = Union[HeuristicPolicy, EmbeddingSimilarityPolicy, RewritePolicy]