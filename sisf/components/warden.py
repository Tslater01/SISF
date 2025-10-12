# sisf/components/warden.py
"""
Implements the Warden, the protected LLM service.
"""
import re
from typing import Dict, Any, List, Tuple, Optional
from sisf.schemas.policies import Policy, PolicyAction, HeuristicPolicy, EmbeddingSimilarityPolicy, RewritePolicy
from sisf.utils.policy_store import PolicyStore

class MockEmbeddingModel:
    """Mock for sentence_transformers.SentenceTransformer"""
    def encode(self, text: str) -> List[float]:
        val = hash(text) % 2
        return [float(val), float(1-val)]
    
    def cos_sim(self, a: List[float], b: List[float]) -> float:
        dot_product = sum(x*y for x, y in zip(a, b))
        mag_a = sum(x*x for x in a)**0.5
        mag_b = sum(y*y for y in b)**0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot_product / (mag_a * mag_b)

class Warden:
    """The protected LLM service."""
    def __init__(self, model_name: str, policy_store: PolicyStore):
        print(f"Initializing Warden with model: {model_name}")
        print(f"NOTE: Using MOCK LLM for Phase 1. Real model '{model_name}' is commented out.")
        self.model = None
        self.tokenizer = None
        self.policy_store = policy_store
        self.embedding_model = MockEmbeddingModel()
        print("Warden initialized.")

    def _apply_policies(self, prompt: str) -> Tuple[PolicyAction, Optional[Policy]]:
        """
        The core enforcement logic. Checks a prompt against all active policies.
        Returns the highest-priority action and the policy that triggered it.
        Priority: BLOCK > REWRITE > FLAG_FOR_REVIEW > ALLOW.
        """
        active_policies = self.policy_store.get_active_policies()
        triggered_actions: Dict[PolicyAction, Policy] = {}

        for policy in active_policies:
            triggered = False
            if isinstance(policy, HeuristicPolicy):
                if re.search(policy.regex_pattern, prompt):
                    triggered = True
            elif isinstance(policy, EmbeddingSimilarityPolicy):
                prompt_embedding = self.embedding_model.encode(prompt)
                sim = self.embedding_model.cos_sim(prompt_embedding, policy.reference_embedding)
                if sim >= policy.similarity_threshold:
                    triggered = True
            elif isinstance(policy, RewritePolicy):
                # Placeholder for Phase 2
                pass
            
            if triggered:
                triggered_actions[policy.action] = policy
        
        if PolicyAction.BLOCK in triggered_actions:
            return PolicyAction.BLOCK, triggered_actions[PolicyAction.BLOCK]
        if PolicyAction.REWRITE in triggered_actions:
            return PolicyAction.REWRITE, triggered_actions[PolicyAction.REWRITE]
        if PolicyAction.FLAG_FOR_REVIEW in triggered_actions:
            return PolicyAction.FLAG_FOR_REVIEW, triggered_actions[PolicyAction.FLAG_FOR_REVIEW]
            
        return PolicyAction.ALLOW, None

    def process(self, prompt: str) -> Dict[str, Any]:
        """
        The main public-facing method.
        Applies policies, then (if allowed) queries the base LLM.
        """
        action, triggering_policy = self._apply_policies(prompt)
        
        if action == PolicyAction.BLOCK:
            print(f"Warden: Request BLOCKED by policy {triggering_policy.id}")
            return {
                "status": "BLOCKED",
                "response": "This request was blocked by a safety policy.",
                "policy_id": triggering_policy.id,
                "policy_description": triggering_policy.description
            }
        
        flagged_by = None
        if action == PolicyAction.FLAG_FOR_REVIEW:
            print(f"Warden: Request FLAGGED by policy {triggering_policy.id}")
            flagged_by = triggering_policy.id
        
        print(f"Warden: Request ALLOWED. Processing with base LLM...")
        response_text = f"This is a MOCK response from the Warden's base LLM for the prompt: '{prompt}'"
        
        return {
            "status": "ALLOWED",
            "response": response_text,
            "policy_id": None,
            "flagged_by_policy_id": flagged_by
        }