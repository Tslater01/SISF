# src/sisf/api.py
"""
Main API entrypoint for the SISF.
This version includes an expanded endpoint for detailed, forensic evaluation.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List, Optional
from dotenv import load_dotenv
from collections import deque 

from sisf.components.warden import Warden
from sisf.components.adjudicator import EnsembleAdjudicator, AdjudicationResult
from sisf.components.psm import PolicySynthesisModule
from sisf.components.apa import AdversarialProbingAgent
from sisf.utils.policy_store import PolicyStore
from sisf.schemas.policies import Policy

load_dotenv()
app = FastAPI(title="Self-Improving Safety Framework (SISF)")

# --- Configuration Loading ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY not found in .env file.")
ADJUDICATOR_MODEL = os.getenv("ADJUDICATOR_MODEL", "gpt-4o")
PSM_MODEL = os.getenv("PSM_MODEL", "gpt-4-turbo")
APA_MODEL = os.getenv("APA_MODEL", "gpt-4o")

print("--- SISF Configuration ---")
print(f"Adjudicator Model: {ADJUDICATOR_MODEL}")
print(f"PSM Model:         {PSM_MODEL}")
print(f"APA Model:         {APA_MODEL}")
print("--------------------------")

# --- Component Initialization ---
policy_store: PolicyStore = PolicyStore()
warden: Warden = Warden(model_name="mistralai/Mistral-7B-v0.1", policy_store=policy_store)
adjudicator: EnsembleAdjudicator = EnsembleAdjudicator(api_key=OPENAI_API_KEY, model=ADJUDICATOR_MODEL)
psm: PolicySynthesisModule = PolicySynthesisModule(api_key=OPENAI_API_KEY, model=PSM_MODEL)
apa: AdversarialProbingAgent = AdversarialProbingAgent(api_key=OPENAI_API_KEY, model=APA_MODEL)
attack_history = deque(maxlen=5)

# --- API Models ---
class ChatRequest(BaseModel): 
    prompt: str
    prompt_id: str = ""

class ChatResponse(BaseModel): 
    status: str
    response: str
    policy_id: Optional[str] = None

class LoopCycleResponse(BaseModel): 
    status: str
    apa_prompt: str
    adjudication: AdjudicationResult
    new_policy: Optional[Policy] = None

class EvaluationRequest(BaseModel):
    prompt: str
    prompt_id: str

# --- NEW: An expanded "Black Box" response model ---
class EvaluationResponse(BaseModel):
    was_blocked: bool
    warden_response: str                      # The full response from the LLM
    adjudication: Optional[AdjudicationResult] = None # The full verdict from the Adjudicator
    new_policy: Optional[Policy] = None       # The full policy JSON if one was created
    total_policies_now: int

# --- 1. Public-Facing API (Unchanged) ---
@app.post("/v1/chat", response_model=ChatResponse, tags=["User API"])
async def handle_chat(request: ChatRequest):
    result = warden.process(request.prompt, prompt_id=request.prompt_id)
    return ChatResponse(status=result["status"], response=result["response"], policy_id=result.get("policy_id"))

# --- 2. Internal Adaptive Loop API (Unchanged) ---
@app.post("/v1/internal/run_adaptive_cycle", response_model=LoopCycleResponse, tags=["Internal Loop"])
async def run_adaptive_cycle():
    print("\n--- SISF: Starting new adaptive cycle ---")
    prompt = apa.generate_prompt(history=list(attack_history))
    warden_output = warden.process(prompt)
    final_status = "BLOCKED" if warden_output['status'] == 'BLOCKED' else "ALLOWED"
    attack_history.append((prompt, final_status))
    
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
    return LoopCycleResponse(status="BREACH_MITIGATED" if new_policy_obj else "NO_BREACH", apa_prompt=prompt, adjudication=adjudication, new_policy=new_policy_obj)

# --- 3. NEW: Expanded Evaluation Endpoint ---
@app.post("/v1/internal/evaluate_and_learn", response_model=EvaluationResponse, tags=["Internal Evaluation"])
async def evaluate_and_learn_from_prompt(request: EvaluationRequest):
    """
    Takes a specific prompt, evaluates it, runs the full learning loop,
    and returns a detailed "black box" record of the entire event.
    """
    print(f"\n--- SISF: Evaluating prompt {request.prompt_id} ---")
    
    initial_policy_count = len(policy_store.get_all_policies())
    
    # 1. Evaluate the prompt against the current system state
    warden_output = warden.process(request.prompt, prompt_id=request.prompt_id)
    was_blocked = warden_output['status'] == 'BLOCKED'
    warden_response_text = warden_output.get('response', '')
    
    new_policy_obj = None
    adjudication_result = None
    
    if not was_blocked:
        # 2. If it wasn't blocked, check if it was a breach
        print("EVALUATE: Not blocked. Sending to Adjudicator...")
        adjudication_result = adjudicator.analyze(request.prompt, warden_response_text)
        
        if adjudication_result.is_breach:
            # 3. If it was a breach, try to create a new policy
            print("EVALUATE: Breach detected! Engaging PSM.")
            new_policy_obj = psm.synthesize_policy(request.prompt, warden_response_text, adjudication_result)
            if new_policy_obj:
                policy_store.add_policy(new_policy_obj, activate=True)
                print(f"EVALUATE: New policy {new_policy_obj.id} created and activated.")
    else:
        print("EVALUATE: Prompt was blocked by an existing policy. No learning needed.")

    return EvaluationResponse(
        was_blocked=was_blocked,
        warden_response=warden_response_text,
        adjudication=adjudication_result,
        new_policy=new_policy_obj,
        total_policies_now=len(policy_store.get_all_policies())
    )

# --- 4. Oversight API (Unchanged) ---
@app.get("/v1/policies", response_model=List[Policy], tags=["Oversight API"])
async def get_all_policies():
    return policy_store.get_all_policies()

@app.post("/v1/policies/toggle/{policy_id}", tags=["Oversight API"])
async def toggle_policy_active_status(policy_id: str, active: bool):
    if not policy_store.toggle_policy(policy_id, active):
        raise HTTPException(status_code=404, detail="Policy not found")
    return {"status": "success", "policy_id": policy_id, "active": active}