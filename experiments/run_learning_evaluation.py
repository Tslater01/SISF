# experiments/run_learning_evaluation.py
import hydra
from omegaconf import DictConfig
import pandas as pd
import httpx
import time
import os
from datetime import datetime
from tqdm import tqdm

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def run_learning_evaluation(cfg: DictConfig) -> None:
    """
    Runs a dynamic evaluation by executing the full adaptive cycle for each
    prompt, capturing a detailed forensic log of the system's learning process.
    """
    
    print("--- Starting Dynamic Learning Evaluation (Forensic Mode) ---")
    print(f"Experiment Name: {cfg.experiment.name}")
    print(f"Number of Prompts: {cfg.num_prompts_to_test}")
    
    # --- 1. Load Data ---
    # This logic correctly uses the 'dataset' override from the command line
    dataset_type = cfg.get("dataset", "adversarial") # Default to adversarial
    is_benign_run = (dataset_type == "benign")
    
    if is_benign_run:
        dataset_path = cfg.data.benign_path
    else:
        dataset_path = cfg.data.adversarial_path

    print(f"Dataset Type: {dataset_type}")
    print(f"Loading data from: {dataset_path}")
    
    df = pd.read_csv(dataset_path).head(cfg.num_prompts_to_test)
    
    # --- 2. Initialize HTTP Client ---
    client = httpx.Client(base_url=cfg.experiment.api_endpoint, timeout=600.0)

    # --- 3. Run Evaluation Loop ---
    results_list = []
    print(f"Beginning evaluation of {len(df)} prompts...")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Running Learning Cycles"):
        prompt = row['prompt_text']
        prompt_id = row['prompt_id']

        try:
            # Call the "black box" endpoint that evaluates and learns
            response = client.post(
                "/v1/internal/evaluate_and_learn", 
                json={"prompt": prompt, "prompt_id": prompt_id}
            )
            response.raise_for_status()
            data = response.json()

            adjudication_data = data.get('adjudication')
            new_policy_data = data.get('new_policy')

            # Record all detailed forensic data
            results_list.append({
                "cycle_num": index + 1,
                "prompt_id": prompt_id,
                "prompt_text": prompt,
                "was_blocked": data.get('was_blocked', False),
                "warden_response": data.get('warden_response', ''),
                "adjudicator_verdict": adjudication_data.get('is_breach', None) if adjudication_data else None,
                "adjudicator_reasoning": adjudication_data.get('reasoning', None) if adjudication_data else None,
                "adjudicator_category": adjudication_data.get('failure_category', None) if adjudication_data else None,
                "policy_created": new_policy_data is not None,
                "new_policy_id": new_policy_data.get('id', None) if new_policy_data else None,
                "new_policy_type": new_policy_data.get('type', None) if new_policy_data else None,
                "new_policy_regex": new_policy_data.get('regex_pattern', None) if new_policy_data and new_policy_data.get('type') == 'HEURISTIC' else None,
                "new_policy_threshold": new_policy_data.get('similarity_threshold', None) if new_policy_data and new_policy_data.get('type') == 'EMBEDDING_SIMILARITY' else None,
                "total_policies_now": data.get('total_policies_now', 0)
            })

        except Exception as e:
            print(f"Error on cycle {index+1} ({prompt_id}): {e}")
            results_list.append({ 
                "cycle_num": index + 1, 
                "prompt_id": prompt_id, 
                "was_blocked": False, 
                "policy_created": False, 
                "total_policies_now": -1 # Flag errors with -1
            })
        
        # Small delay to be kind to the API and let logs flush
        time.sleep(1)

    # --- 4. Save Results ---
    results_df = pd.DataFrame(results_list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"Learning_Curve_Evaluation_{dataset_type}_{timestamp}.csv"
    output_path = os.path.join(cfg.output_dir, output_filename)
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print("\n--- Dynamic Evaluation Complete ---")
    print(f"Results saved to: {output_path}")

    # --- 5. Print Summary ---
    print("\n--- Summary ---")
    if is_benign_run:
        # A false positive is when a safe prompt is blocked OR causes a policy to be made
        # (which means the Adjudicator made a mistake)
        fpr = (results_df['was_blocked'] | results_df['policy_created']).mean() * 100
        print(f"False Positive Rate (FPR): {fpr:.2f}%")
    else:
        # An attack is successful only if it was NOT blocked
        asr = (results_df['was_blocked'] == False).mean() * 100
        print(f"Attack Success Rate (ASR): {asr:.2f}%")
    
    total_policies_created = results_df['policy_created'].sum()
    final_policy_count = results_df['total_policies_now'].max()
    print(f"Total New Policies Created: {total_policies_created}")
    print(f"Final Policy Store Size: {final_policy_count}")


if __name__ == "__main__":
    # This allows Hydra to be run cleanly
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    run_learning_evaluation()