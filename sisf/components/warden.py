# sisf/components/warden.py
"""
Implements the Warden, the protected LLM service.
"""
import re
from typing import Dict, Any, List, Tuple, Optional
from sisf.schemas.policies import Policy, PolicyAction, HeuristicPolicy, EmbeddingSimilarityPolicy, RewritePolicy
from sisf.utils.policy_store import PolicyStore

class MockEmbeddingModel:
    """
    A more realistic mock for sentence_transformers.SentenceTransformer.
    It uses a predefined vocabulary to create deterministic, testable embeddings.
    """
    def __init__(self):
        self.vocab = {
            # Friendly
            "hello": [1.0, 0.0, 0.0],
            "hi": [0.9, 0.1, 0.0],
            # Unfriendly
            "goodbye": [0.0, 1.0, 0.0],
            "bye": [0.1, 0.9, 0.0],
            # Forbidden
            "secret": [0.0, 0.0, 1.0],
            "classified": [0.0, 0.1, 0.9]
        }
        self.unknown_vector = [0.33, 0.33, 0.33]

    def encode(self, text: str) -> List[float]:
        # Simple logic: find the first known word
        for word, vector in self.vocab.items():
            if word in text.lower():
                return vector
        return self.unknown_vector
    
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

    def _apply_policies(self, prompt: str) -> Tuple[PolicyAction, Optional[Policy], str]:
        """
        The core enforcement logic. Checks a prompt against all active policies.
        Returns the highest-priority action, the triggering policy, and the (potentially modified) prompt.
        Priority: BLOCK > REWRITE > FLAG_FOR_REVIEW > ALLOW.
        """
        active_policies = self.policy_store.get_active_policies()
        triggered_actions: Dict[PolicyAction, Policy] = {}
        
        modified_prompt = prompt

        # First pass for REWRITE policies
        for policy in active_policies:
            if isinstance(policy, RewritePolicy):
                if re.search(policy.match_pattern, modified_prompt):
                    modified_prompt = re.sub(policy.match_pattern, policy.rewrite_template, modified_prompt)
                    triggered_actions[PolicyAction.REWRITE] = policy

        # Second pass for BLOCK and FLAG on the *final* version of the prompt
        for policy in active_policies:
            triggered = False
            if isinstance(policy, HeuristicPolicy):
                if re.search(policy.regex_pattern, modified_prompt):
                    triggered = True
            elif isinstance(policy, EmbeddingSimilarityPolicy):
                prompt_embedding = self.embedding_model.encode(modified_prompt)
                sim = self.embedding_model.cos_sim(prompt_embedding, policy.reference_embedding)
                if sim >= policy.similarity_threshold:
                    triggered = True
            
            if triggered:
                if policy.action in [PolicyAction.BLOCK, PolicyAction.FLAG_FOR_REVIEW]:
                    triggered_actions[policy.action] = policy
        
        if PolicyAction.BLOCK in triggered_actions:
            return PolicyAction.BLOCK, triggered_actions[PolicyAction.BLOCK], modified_prompt
        if PolicyAction.REWRITE in triggered_actions:
            return PolicyAction.REWRITE, triggered_actions[PolicyAction.REWRITE], modified_prompt
        if PolicyAction.FLAG_FOR_REVIEW in triggered_actions:
            return PolicyAction.FLAG_FOR_REVIEW, triggered_actions[PolicyAction.FLAG_FOR_REVIEW], modified_prompt
            
        return PolicyAction.ALLOW, None, modified_prompt

    def process(self, prompt: str) -> Dict[str, Any]:
        """
        The main public-facing method.
        Applies policies, then (if allowed) queries the base LLM.
        """
        action, triggering_policy, final_prompt = self._apply_policies(prompt)
        
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
        
        if action == PolicyAction.REWRITE:
            print(f"Warden: Request REWRITTEN by policy {triggering_policy.id}")
        
        print(f"Warden: Request ALLOWED. Processing with base LLM...")
        response_text = f"This is a MOCK response from the Warden's base LLM for the prompt: '{final_prompt}'"
        
        return {
            "status": "ALLOWED",
            "response": response_text,
            "original_prompt": prompt,
            "final_prompt": final_prompt,
            "policy_id": triggering_policy.id if action == PolicyAction.REWRITE else None,
            "flagged_by_policy_id": flagged_by
        }