# sisf/main_loop.py
"""
The main orchestration script for running the SISF adaptive loop.

This script acts as the master controller for experiments, simulating the
continuous operation of the SISF by repeatedly calling the adaptive cycle
endpoint of the running API.
"""
import httpx
import time
import argparse
import os

# --- Configuration ---
# It's good practice to get the URL from an environment variable or default
# to a local development server.
API_BASE_URL = os.getenv("SISF_API_URL", "http://127.0.0.1:8000")
ADAPTIVE_CYCLE_ENDPOINT = f"{API_BASE_URL}/v1/internal/run_adaptive_cycle"

def run_single_cycle(client: httpx.Client) -> bool:
    """Runs one full adaptive cycle and prints the outcome."""
    try:
        print("="*80)
        print("🚀 Executing new adaptive cycle...")
        
        response = client.post(ADAPTIVE_CYCLE_ENDPOINT, timeout=120.0) # Increased timeout for LLM calls
        response.raise_for_status() # Raises an exception for 4xx/5xx responses
        
        data = response.json()
        
        print(f"   - APA Prompt: '{data['apa_prompt'][:100]}...'")
        
        adjudication = data['adjudication']
        print(f"   - Adjudicator Verdict: {'BREACH' if adjudication['is_breach'] else 'NO BREACH'}")
        print(f"     - Reasoning: {adjudication['reasoning']}")
        
        if adjudication['is_breach']:
            if data.get('new_policy'):
                new_policy = data['new_policy']
                print(f"   - PSM ACTION: SUCCESS! ✅")
                print(f"     - Synthesized New Policy (ID: {new_policy['id']})")
                print(f"     - Type: {new_policy['type']}, Action: {new_policy['action']}")
                print(f"     - Description: {new_policy['description']}")
                return True # Indicates a policy was created
            else:
                print("   - PSM ACTION: FAILED to synthesize a new policy. ❌")
        
        return False

    except httpx.RequestError as e:
        print(f"\n❌ CLIENT ERROR: Could not connect to the SISF API at {ADAPTIVE_CYCLE_ENDPOINT}")
        print("   - Is the FastAPI server running? Use: `uvicorn sisf.api:app --reload`")
        print(f"   - Details: {e}")
        return False
    except httpx.HTTPStatusError as e:
        print(f"\n❌ SERVER ERROR: The API returned an error (Status {e.response.status_code})")
        print(f"   - Details: {e.response.text}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the SISF adaptive loop.")
    parser.add_argument(
        "-n", "--num-cycles",
        type=int,
        default=5,
        help="The number of adaptive cycles to run."
    )
    args = parser.parse_args()

    print(f"SISF Main Loop Initialized. Starting {args.num_cycles} cycles.")
    
    with httpx.Client() as client:
        policies_created = 0
        for i in range(args.num_cycles):
            print(f"\n--- CYCLE {i+1} of {args.num_cycles} ---")
            if run_single_cycle(client):
                policies_created += 1
            time.sleep(1) # Small delay to prevent overwhelming the server
    
    print("="*80)
    print(f"\n🏁 SISF Main Loop Finished.")
    print(f"   - Total cycles run: {args.num_cycles}")
    print(f"   - New policies created: {policies_created}")

if __name__ == "__main__":
    main()