# sisf/api.py
"""
Main API entrypoint for the SISF.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  
import os
from typing import List, Optional
from dotenv import load_dotenv

from sisf.components.warden import Warden
from sisf.components.adjudicator import EnsembleAdjudicator, AdjudicationResult
from sisf.components.psm import PolicySynthesisModule
from sisf.components.apa import AdversarialProbingAgent
from sisf.utils.policy_store import PolicyStore
from sisf.schemas.policies import Policy

# --- Load environment variables from .env file ---
load_dotenv()

# --- Global Singletons ---
app = FastAPI(
    title="Self-Improving Safety Framework (SISF)",
    description="A reference implementation of the SISF architecture."
)

# --- Configuration Loading ---
# Load the API key from the .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Make sure it's set in your .env file.")

# IMPROVEMENT: Load model names from .env file with sensible defaults
ADJUDICATOR_MODEL = os.getenv("ADJUDICATOR_MODEL", "gpt-4o")
PSM_MODEL = os.getenv("PSM_MODEL", "gpt-4-turbo")
APA_MODEL = os.getenv("APA_MODEL", "gpt-4o") 

print("--- SISF Configuration ---")
print(f"Adjudicator Model: {ADJUDICATOR_MODEL}")
print(f"PSM Model:         {PSM_MODEL}")
print(f"APA Model:         {APA_MODEL}")
print("--------------------------")

# --- Component Initialization ---
# IMPROVEMENT: Add type hints for clarity
policy_store: PolicyStore = PolicyStore()
warden: Warden = Warden(model_name="meta-llama/Llama-3.1-8B-Instruct", policy_store=policy_store)
adjudicator: EnsembleAdjudicator = EnsembleAdjudicator(api_key=OPENAI_API_KEY, model=ADJUDICATOR_MODEL)
psm: PolicySynthesisModule = PolicySynthesisModule(api_key=OPENAI_API_KEY, model=PSM_MODEL)
apa: AdversarialProbingAgent = AdversarialProbingAgent(api_key=OPENAI_API_KEY, model=APA_MODEL)


# --- 1. Public-Facing API ---
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    status: str
    response: str
    policy_id: Optional[str] = None

@app.post("/v1/chat", response_model=ChatResponse, tags=["User API"])
async def handle_chat(request: ChatRequest):
    """The main endpoint for users to interact with the protected Warden LLM."""
    result = warden.process(request.prompt)
    return ChatResponse(
        status=result["status"],
        response=result["response"],
        policy_id=result.get("policy_id")
    )

# --- 2. Internal Adaptive Loop API ---
class LoopCycleResponse(BaseModel):
    status: str
    apa_prompt: str
    adjudication: AdjudicationResult
    new_policy: Optional[Policy] = None

@app.post("/v1/internal/run_adaptive_cycle", response_model=LoopCycleResponse, tags=["Internal Loop"])
async def run_adaptive_cycle():
    """Executes a single cycle of the self-improvement loop."""
    print("\n--- SISF: Starting new adaptive cycle ---")

    prompt = apa.generate_prompt()
    warden_output = warden.process(prompt)
    adjudication = adjudicator.analyze(prompt, warden_output["response"])

    new_policy_obj = None
    if adjudication.is_breach:
        print("SISF: Breach detected! Engaging PSM.")
        new_policy_obj = psm.synthesize_policy(prompt, warden_output["response"], adjudication)

        if new_policy_obj:
            policy_store.add_policy(new_policy_obj, activate=True)
            print(f"SISF: New policy {new_policy_obj.id} added and activated.")
        else:
            print("SISF: PSM failed to generate a new policy.")
    else:
        print("SISF: No breach detected. Cycle complete.")

    print("--- SISF: Adaptive cycle finished ---")

    return LoopCycleResponse(
        status="BREACH_MITIGATED" if new_policy_obj else "NO_BREACH",
        apa_prompt=prompt,
        adjudication=adjudication,
        new_policy=new_policy_obj
    )

# --- 3. Oversight API ---
@app.get("/v1/policies", response_model=List[Policy], tags=["Oversight API"])
async def get_all_policies():
    """Gets all policies in the store."""
    return policy_store.get_all_policies()

@app.post("/v1/policies/toggle/{policy_id}", tags=["Oversight API"])
async def toggle_policy_active_status(policy_id: str, active: bool):
    """Allows the HITL to activate or deactivate a policy."""
    if not policy_store.toggle_policy(policy_id, active):
        raise HTTPException(status_code=404, detail="Policy not found")
    return {"status": "success", "policy_id": policy_id, "active": active}