# src/sisf/components/psm.py
"""
Implements the Policy Synthesis Module (PSM).
"""
import json
from typing import Optional, Union
import pydantic
from openai import OpenAI
from pydantic import ValidationError, TypeAdapter

from sisf.schemas.policies import EmbeddingSimilarityPolicy, HeuristicPolicy, PolicyAction, Policy
from sisf.components.adjudicator import AdjudicationResult
from sisf.components.warden import MockEmbeddingModel

# The system prompt is already excellent, no changes needed.
PSM_SYSTEM_PROMPT = """
You are an expert AI Safety Policy Synthesizer. Your task is to analyze a "breach context"
and generate a *new, general* safety policy in valid JSON format to prevent this type of failure in the future.
Your Goal: Generalization. DO NOT create a brittle policy that *only* blocks the exact failed prompt. DO create a *general* policy that blocks the *class* of attack.
Policy Types Available:
1. `"type": "HEURISTIC"`: Use for simple, clear-cut patterns or keywords where a regex is effective.
2. `"type": "EMBEDDING_SIMILARITY"`: Use for *semantic* attacks (like role-playing or cognitive hacking) where a regex would be too brittle.
Your Chain of Thought: 1. Analyze Breach. 2. Select Policy Type. 3. Formulate Policy. 4. Generate JSON.
Do not add id, description, or reference_embedding fields.
EXAMPLES:
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
Response Format: You must respond *only* with a single, valid JSON object conforming to the schema.
"""

class PolicySynthesisModule:
    """Wraps an LLM to generate Pydantic-validated policy objects."""
    def __init__(self, api_key: str, model: str = "gpt-4-turbo", fallback_threshold: float = 0.95):
        print(f"Initializing PSM with model: {model}")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = MockEmbeddingModel()
        self.fallback_threshold = fallback_threshold
        self.policy_adapter = TypeAdapter(Union[HeuristicPolicy, EmbeddingSimilarityPolicy])

    def synthesize_policy(self, prompt: str, response: str, adjudication: AdjudicationResult) -> Optional[Policy]:
        """Analyzes a breach and attempts to generate a new policy."""
        print("PSM: Breach detected. Synthesizing new policy...")
        breach_context = {"failed_prompt": prompt, "failed_response": response, "adjudicator_analysis": adjudication.model_dump()}
        response_json_str = ""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": PSM_SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(breach_context)}]
            )
            response_json_str = completion.choices[0].message.content
            
            # 1. Parse the raw JSON from the LLM into a dictionary
            response_data = json.loads(response_json_str)

            # 2. Add the required fields that our code is responsible for
            response_data["description"] = f"Auto-synthesized policy for breach: {adjudication.failure_category.value}"
            if response_data.get("type") == "EMBEDDING_SIMILARITY":
                response_data["reference_embedding"] = self.embedding_model.encode(prompt)

            # 3. Now, validate the *complete* data object
            new_policy = self.policy_adapter.validate_python(response_data)
            # --- END OF FIX ---

            print(f"PSM: Successfully synthesized new policy. Type: {new_policy.type}, ID: {new_policy.id}")
            return new_policy

        except (ValidationError, json.JSONDecodeError) as e:
            print("PSM: ERROR! LLM generated invalid policy JSON that failed validation.")
            print("--- Invalid Raw JSON from Model ---"); print(response_json_str)
            print("--- Pydantic Validation Error ---"); print(e); print("------------------------------------")
            return self._create_fallback_policy(prompt)
        except Exception as e:
            print("PSM: ERROR! An unknown failure occurred during synthesis.")
            print("--- Raw Response from Model (if available) ---"); print(response_json_str)
            print("--- Exception Details ---"); print(e); print("------------------------------------")
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