# experiments/run_experiment.py
import hydra
from omegaconf import DictConfig
import pandas as pd
import httpx
import time
import os
from datetime import datetime
from tqdm import tqdm # Add a progress bar for the long run

# --- Define a base class for all defense systems ---
class BaseDefenseSystem:
    """An abstract base class for any system we want to test."""
    def __init__(self, config: DictConfig):
        self.config = config

    def process_prompt(self, prompt: str, prompt_id: str) -> (dict):
        """Sends a prompt to the system and returns the result."""
        raise NotImplementedError

# --- Implement the SISF defense system ---
class SISFSystem(BaseDefenseSystem):
    """Represents our running SISF application."""
    def process_prompt(self, prompt: str, prompt_id: str) -> (dict):
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.config.api_endpoint}/v1/chat",
                    json={"prompt": prompt, "prompt_id": prompt_id},
                    timeout=600.0
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"status": "ERROR", "response": str(e)}

# --- Implement the 'No Defense' baseline system ---
class NoDefenseSystem(BaseDefenseSystem):
    """A baseline that represents the raw, unprotected Warden LLM."""
    def process_prompt(self, prompt: str, prompt_id: str) -> (dict):
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.config.api_endpoint}/v1/chat",
                    json={"prompt": prompt, "prompt_id": prompt_id},
                    timeout=600.0
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"status": "ERROR", "response": str(e)}

# --- The main experiment function, managed by Hydra ---
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """The master function for running an experimental condition."""
    
    print("--- Starting Experiment ---")
    print(f"Experiment Name: {cfg.experiment.name}")
    print(f"Defense System: {cfg.experiment.defense_system_type}")
    print(f"Number of Prompts: {cfg.num_prompts_to_test}")
    
    # --- 1. Load Data ---
    dataset_type = cfg.get("dataset", "adversarial")
    is_benign_run = (dataset_type == "benign")
    dataset_path = cfg.data.benign_path if is_benign_run else cfg.data.adversarial_path
    
    print(f"Dataset Type: {dataset_type}")
    print(f"Loading data from: {dataset_path}")
    df = pd.read_csv(dataset_path).head(cfg.num_prompts_to_test)
    
    # --- 2. Initialize Defense System ---
    system_map = {"sisf": SISFSystem, "no_defense": NoDefenseSystem}
    SystemClass = system_map.get(cfg.experiment.defense_system_type)
    if not SystemClass:
        raise ValueError(f"Unknown defense system type: {cfg.experiment.defense_system_type}")
    system = SystemClass(cfg.experiment)

    # --- 3. Run Evaluation Loop ---
    results = []
    # IMPROVEMENT: Add a tqdm progress bar for long runs
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {cfg.experiment.name}"):
        prompt = row['prompt_text']
        prompt_id = row['prompt_id']

        start_time = time.time()
        result = system.process_prompt(prompt, prompt_id)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        false_positive = (is_benign_run and result['status'] == 'BLOCKED')
        attack_successful = (not is_benign_run and result['status'] != 'BLOCKED')
        
        # --- NEW DATA COLLECTION ---
        results.append({
            "experiment_name": cfg.experiment.name,
            "dataset_type": dataset_type,
            "cycle_num": index + 1,
            "prompt_id": prompt_id,
            "prompt_text": prompt,
            "system_status": result['status'],
            "system_response": result.get('response', ''),
            "policy_id_block": result.get('policy_id', None), # Capture the blocking policy ID
            "latency_ms": latency_ms,
            "attack_successful": attack_successful,
            "false_positive": false_positive
        })

    # --- 4. Save Results ---
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{cfg.experiment.name}_{dataset_type}_{timestamp}.csv"
    output_path = os.path.join(cfg.output_dir, output_filename)
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print("\n--- Experiment Complete ---")
    print(f"Results saved to: {output_path}")

    # --- 5. Print Summary ---
    print("\n--- Summary ---")
    if is_benign_run:
        fpr = results_df['false_positive'].mean() * 100
        print(f"False Positive Rate (FPR): {fpr:.2f}%")
    else:
        asr = results_df['attack_successful'].mean() * 100
        print(f"Attack Success Rate (ASR): {asr:.2f}%")
    
    avg_latency = results_df['latency_ms'].mean()
    print(f"Average Latency: {avg_latency:.2f} ms")

if __name__ == "__main__":
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    run_experiment()