# sisf/schemas/policies.py
"""
This file defines the formal "Policy Language" for the SISF.

This is the single most critical data contract in the system.
- The Policy Synthesis Module (PSM) *generates* objects conforming to these schemas.
- The Warden's safety filter *consumes* and *enforces* these schemas.
- The Adaptive Policy Store (APS) *stores* these schemas.

Using Pydantic ensures that policies are always well-formed, validated,
and self-documenting, which is crucial for the system's stability and
for answering Critique C (Practicality/Brittleness).
"""
import uuid
from abc import ABC
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union

# --- 1. The Core Action Enum ---
class PolicyAction(str, Enum):
    """The primitive enforcement actions the Warden can perform."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REWRITE = "REWRITE"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"

# --- 2. Base Policy Schema ---
class BasePolicy(BaseModel, ABC):
    """Abstract base class for all policy types."""
    id: str = Field(
        default_factory=lambda: f"pol_{uuid.uuid4().hex}",
        description="A unique identifier for the policy."
    )
    description: str = Field(..., description="A human-readable explanation of what this policy does and why it was created.")
    action: PolicyAction = Field(..., description="The action to take if this policy is triggered.")
    
    # NEW: Metadata for traceability and analysis
    created_by: str = Field("psm", description="The component that created the policy (e.g., 'psm', 'human').")
    breach_context_id: Optional[str] = Field(None, description="Optional ID linking to the specific breach that caused this policy's creation.")

# --- 3. Specific, Enforceable Policy Types ---
class HeuristicPolicy(BasePolicy):
    """A policy that triggers based on a regular expression."""
    type: Literal["HEURISTIC"] = "HEURISTIC"
    regex_pattern: str = Field(..., description="The PCRE regex pattern to search for in the prompt or response.")
    action: Literal[PolicyAction.BLOCK, PolicyAction.FLAG_FOR_REVIEW] = Field(
        ..., description="Heuristic policies can only BLOCK or FLAG."
    )

class EmbeddingSimilarityPolicy(BasePolicy):
    """A policy that triggers based on semantic similarity."""
    type: Literal["EMBEDDING_SIMILARITY"] = "EMBEDDING_SIMILARITY"
    reference_embedding: List[float] = Field(..., description="The vector embedding of the attack to compare against.")
    similarity_threshold: float = Field(
        ..., gt=0, lt=1,
        description="The cosine similarity threshold (e.g., 0.85) to trigger the policy."
    )
    action: Literal[PolicyAction.BLOCK, PolicyAction.FLAG_FOR_REVIEW] = Field(
        ..., description="Semantic policies can only BLOCK or FLAG."
    )

class RewritePolicy(BasePolicy):
    """A policy that actively modifies a prompt or response."""
    type: Literal["REWRITE"] = "REWRITE"
    match_pattern: str = Field(..., description="The regex pattern to find in the text.")
    rewrite_template: str = Field(..., description="The template to replace the matched text with (e.g., '[REDACTED TOPIC]').")
    action: Literal[PolicyAction.REWRITE] = Field(
        PolicyAction.REWRITE, 
        description="This policy's action is always REWRITE."
    )

# --- 4. The Top-Level Policy Document ---
Policy = Union[HeuristicPolicy, EmbeddingSimilarityPolicy, RewritePolicy]

class PolicyStoreDocument(BaseModel):
    """The root object representing the entire set of active policies."""
    version: int = Field(..., description="A version number, incremented on each change.")
    policies: List[Policy] = Field(..., description="The list of all active policies.")