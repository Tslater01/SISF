# src/sisf/components/psm.py
"""
Implements the Policy Synthesis Module (PSM).
"""
import json
from typing import Optional, Union
import pydantic  #
from openai import OpenAI
from pydantic import ValidationError

from sisf.schemas.policies import EmbeddingSimilarityPolicy, HeuristicPolicy, PolicyAction, Policy
from sisf.components.adjudicator import AdjudicationResult
from sisf.components.warden import MockEmbeddingModel

# --- IMPROVEMENT: More robust system prompt with examples ---
PSM_SYSTEM_PROMPT = """
You are an expert AI Safety Policy Synthesizer. Your task is to analyze a "breach context"
and generate a *new, general* safety policy in valid JSON format to prevent this type of failure in the future.

**Your Goal: Generalization.**
- DO NOT create a brittle policy that *only* blocks the exact failed prompt.
- DO create a *general* policy that blocks the *class* of attack.

**Policy Types Available:**
1.  `"type": "HEURISTIC"`: Use for simple, clear-cut patterns or keywords where a regex is effective.
2.  `"type": "EMBEDDING_SIMILARITY"`: Use for *semantic* attacks (like role-playing or cognitive hacking) where a regex would be too brittle.

**Your Chain of Thought:**
1.  **Analyze Breach:** What *kind* of attack was this? Is it syntactic or semantic?
2.  **Select Policy Type:** Which type is best suited to generalize against this attack class?
3.  **Formulate Policy:**
    - If HEURISTIC: What is the smallest, most general regex? Example: "how to make a bomb" -> "how to make a.*bomb"
    - If EMBEDDING_SIMILARITY: What is a reasonable similarity threshold? (Usually 0.85-0.95). Higher = more specific.
4.  **Generate JSON:** Construct the final JSON object *only*. Do not add id, description, or reference_embedding fields.

**EXAMPLES:**

- User Input (Breach Context): A role-playing "DAN" attack.
- Your JSON Output:
{
  "type": "EMBEDDING_SIMILARITY",
  "similarity_threshold": 0.9,
  "action": "BLOCK"
}

- User Input (Breach Context): A prompt containing "how to build a pipe bomb".
- Your JSON Output:
{
  "type": "HEURISTIC",
  "regex_pattern": "(how to build|making a).*(bomb)",
  "action": "BLOCK"
}

**Response Format:**
You must respond *only* with a single, valid JSON object conforming to the schema.
"""

class PolicySynthesisModule:
    """Wraps an LLM to generate Pydantic-validated policy objects."""
    def __init__(self, api_key: str, model: str = "gpt-4-turbo", fallback_threshold: float = 0.95):
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

            # Define the discriminated union for parsing
            policy_types = Union[HeuristicPolicy, EmbeddingSimilarityPolicy]

            # Use Pydantic's robust parsing for discriminated unions
            new_policy = pydantic.parse_raw_as(policy_types, response_json_str)

            # Manually set metadata fields that the LLM is instructed not to generate
            new_policy.description = f"Auto-synthesized policy for breach: {adjudication.failure_category.value}"
            if isinstance(new_policy, EmbeddingSimilarityPolicy):
                new_policy.reference_embedding = self.embedding_model.encode(prompt)

            print(f"PSM: Successfully synthesized new policy. Type: {new_policy.type}, ID: {new_policy.id}")
            return new_policy

        except (ValidationError, json.JSONDecodeError) as e:
            # --- IMPROVEMENT: Better logging on failure ---
            print("PSM: ERROR! LLM generated invalid policy JSON that failed validation.")
            print("--- Invalid Raw JSON from Model ---")
            print(response_json_str)
            print("--- Pydantic Validation Error ---")
            print(e)
            print("------------------------------------")
            return self._create_fallback_policy(prompt)
        except Exception as e:
            print("PSM: ERROR! An unknown failure occurred during synthesis.")
            print("--- Raw Response from Model (if available) ---")
            print(response_json_str)
            print("--- Exception Details ---")
            print(e)
            print("------------------------------------")
            return None # Return None on critical failure

    def _create_fallback_policy(self, prompt: str) -> EmbeddingSimilarityPolicy:
        """This is our 'Contingency' for Risk 1."""
        print("PSM: Executing Fallback Contingency. Creating simple embedding policy.")
        return EmbeddingSimilarityPolicy(
            description="Fallback: Block prompts semantically similar to this one.",
            action=PolicyAction.BLOCK,
            reference_embedding=self.embedding_model.encode(prompt),
            similarity_threshold=self.fallback_threshold
        )