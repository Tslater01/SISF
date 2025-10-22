# src/sisf/components/warden.py
"""
Implements the Warden, the protected LLM service.
"""
import re
from typing import Dict, Any, List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

from sisf.schemas.policies import Policy, PolicyAction, HeuristicPolicy, EmbeddingSimilarityPolicy, RewritePolicy
from sisf.utils.policy_store import PolicyStore

class Warden:
    """The protected LLM service."""
    def __init__(self, model_name: str, policy_store: PolicyStore):
        print(f"Initializing Warden with REAL model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Add a padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.policy_store = policy_store
        
        print("Warden is loading real embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        print("Warden embedding model loaded.")
        
        print("Warden initialized with real LLM and real embedding model.")

    def _apply_policies(self, prompt: str) -> Tuple[PolicyAction, Optional[Policy], str]:
        # This method is already robust and requires no changes.
        active_policies = self.policy_store.get_active_policies()
        triggered_actions: Dict[PolicyAction, Policy] = {}
        modified_prompt = prompt

        for policy in active_policies:
            if isinstance(policy, RewritePolicy):
                if re.search(policy.match_pattern, modified_prompt):
                    modified_prompt = re.sub(policy.match_pattern, policy.rewrite_template, modified_prompt)
                    triggered_actions[PolicyAction.REWRITE] = policy

        prompt_embedding = None
        has_semantic_policy = any(isinstance(p, EmbeddingSimilarityPolicy) for p in active_policies)
        if has_semantic_policy:
            prompt_embedding = self.embedding_model.encode(modified_prompt, convert_to_tensor=True)

        for policy in active_policies:
            triggered = False
            if isinstance(policy, HeuristicPolicy):
                if re.search(policy.regex_pattern, modified_prompt): triggered = True
            elif isinstance(policy, EmbeddingSimilarityPolicy):
                ref_embedding = torch.tensor(policy.reference_embedding).to(self.device)
                sim_scores = util.cos_sim(prompt_embedding, ref_embedding)
                sim = sim_scores[0][0].item()
                if sim >= policy.similarity_threshold: triggered = True
            
            if triggered and policy.action in [PolicyAction.BLOCK, PolicyAction.FLAG_FOR_REVIEW]:
                triggered_actions[policy.action] = policy
        
        if PolicyAction.BLOCK in triggered_actions: return PolicyAction.BLOCK, triggered_actions[PolicyAction.BLOCK], modified_prompt
        if PolicyAction.REWRITE in triggered_actions: return PolicyAction.REWRITE, triggered_actions[PolicyAction.REWRITE], modified_prompt
        if PolicyAction.FLAG_FOR_REVIEW in triggered_actions: return PolicyAction.FLAG_FOR_REVIEW, triggered_actions[PolicyAction.FLAG_FOR_REVIEW], modified_prompt
            
        return PolicyAction.ALLOW, None, modified_prompt

    def process(self, prompt: str) -> Dict[str, Any]:
        """Applies policies, then (if allowed) queries the base LLM."""
        action, triggering_policy, final_prompt = self._apply_policies(prompt)
        
        if action == PolicyAction.BLOCK:
            return {"status": "BLOCKED", "response": "This request was blocked by a safety policy.", "policy_id": triggering_policy.id, "policy_description": triggering_policy.description}
        
        flagged_by = None
        if action == PolicyAction.FLAG_FOR_REVIEW: flagged_by = triggering_policy.id
        if action == PolicyAction.REWRITE: print(f"Warden: Request REWRITTEN by policy {triggering_policy.id}")
        
        print(f"Warden: Request ALLOWED. Processing with base LLM...")
        
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": final_prompt}]
        
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        
        # --- IMPROVEMENT: Explicitly create attention_mask to suppress warnings ---
        attention_mask = torch.ones_like(inputs)

        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask, # Pass the mask here
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id # Explicitly set pad token
        )
        response_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        
        return {
            "status": "ALLOWED", "response": response_text, "original_prompt": prompt, "final_prompt": final_prompt,
            "policy_id": triggering_policy.id if action == PolicyAction.REWRITE else None, "flagged_by_policy_id": flagged_by
        }