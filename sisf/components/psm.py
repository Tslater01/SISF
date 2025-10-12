# sisf/components/psm.py
"""
Implements the Policy Synthesis Module (PSM).
"""
from openai import OpenAI
from pydantic import ValidationError
import json
from typing import Optional

from sisf.schemas.policies import Policy, EmbeddingSimilarityPolicy, PolicyAction
from sisf.components.adjudicator import AdjudicationResult
from sisf.components.warden import MockEmbeddingModel

PSM_SYSTEM_PROMPT = """
You are an expert AI Safety Policy Synthesizer.
Your task is to analyze a "breach context" (a prompt, a failed response,
and an adjudicator's analysis) and generate a *new, general* safety policy
in valid JSON format to prevent this failure in the future.

**Your Goal:** Generalization.
- **DO NOT** create a brittle policy that *only* blocks the exact failed prompt.
- **DO** create a *general* policy that blocks the *class* of attack.

**Policy Types Available:**
1.  `"type": "HEURISTIC"`: Use for simple patterns or keywords.
    - `regex_pattern`: The general PCRE regex.
    - `action`: "BLOCK" or "FLAG_FOR_REVIEW"
2.  `"type": "EMBEDDING_SIMILARITY"`: Use for *semantic* attacks where a regex is too brittle.
    - `similarity_threshold`: Choose a float (e.g., 0.85). Higher = more specific.
    - `action`: "BLOCK" or "FLAG_FOR_REVIEW"

**Your Chain of Thought:**
1.  **Analyze Breach:** What *kind* of attack was this?
2.  **Select Policy:** Which policy type is *best* suited to generalize?
3.  **Formulate Policy:**
    - If HEURISTIC: What is the *smallest, most general* regex?
    - If EMBEDDING_SIMILARITY: What is a reasonable threshold?
4.  **Generate JSON:** Construct the final JSON object *only*. Do not add id, description, or reference_embedding fields.

**Response Format:**
You must respond *only* with a single, valid JSON object.
"""

class PolicySynthesisModule:
    """Wraps an LLM to generate Pydantic-validated policy objects."""
    def __init__(self, api_key: str, model: str = "gpt-4o", fallback_threshold: float = 0.95):
        print(f"Initializing PSM with model: {model}")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = MockEmbeddingModel()
        self.fallback_threshold = fallback_threshold

    def synthesize_policy(self, prompt: str, response: str, adjudication: AdjudicationResult) -> Optional[Policy]:
        """Analyzes a breach and attempts to generate a new policy."""
        print("PSM: Breach detected. Synthesizing new policy...")
        breach_context = {
            "failed_prompt": prompt,
            "failed_response": response,
            "adjudicator_analysis": adjudication.model_dump()
        }
        
        response_json_str = ""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": PSM_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(breach_context)}
                ]
            )
            response_json_str = completion.choices[0].message.content
            response_data = json.loads(response_json_str)
            
            # Defensively remove fields we will generate ourselves
            for key in ["id", "description", "created_by", "breach_context_id", "reference_embedding"]:
                response_data.pop(key, None)
            
            response_data["description"] = f"Auto-synthesized policy for breach: {adjudication.failure_category.value}"
            
            if response_data.get("type") == "EMBEDDING_SIMILARITY":
                response_data["reference_embedding"] = self.embedding_model.encode(prompt)
            
            new_policy = Policy.model_validate(response_data)
            
            print(f"PSM: Successfully synthesized new policy. Type: {new_policy.type}, ID: {new_policy.id}")
            return new_policy
            
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"PSM: ERROR! LLM generated invalid policy JSON. {e}")
            print(f"Invalid JSON received: {response_json_str}")
            return self._create_fallback_policy(prompt)
        except Exception as e:
            print(f"PSM: ERROR! Unknown synthesis failure. {e}")
            return None

    def _create_fallback_policy(self, prompt: str) -> EmbeddingSimilarityPolicy:
        """This is our 'Contingency' for Risk 1."""
        print("PSM: Executing Fallback Contingency. Creating simple embedding policy.")
        return EmbeddingSimilarityPolicy(
            description="Fallback: Block prompts semantically similar to this one.",
            action=PolicyAction.BLOCK,
            reference_embedding=self.embedding_model.encode(prompt),
            similarity_threshold=self.fallback_threshold
        )