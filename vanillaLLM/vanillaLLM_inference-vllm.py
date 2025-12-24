import tqdm
import os
import json 
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import copy

# ---------------------------------------------------------
# [VLLM Import]
# ---------------------------------------------------------
from vllm import LLM, SamplingParams

# If prompt module is local, keep it; otherwise define dummy variables.
from prompt import IMPLICIT_ZS_PROMPT_TEMPLATE, IMPLICIT_FS_PROMPT_TEMPLATE, IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE

# ---------------------------------------------------------
# 1. Load Dataset & Queries
# ---------------------------------------------------------
def load_chains_dataset(fpath: str) -> pd.DataFrame:
    try:
        # JSON Lines format
        df = pd.read_json(fpath, lines=True)
        return df
    except ValueError:
        # Standard JSON List format
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

def load_query_map(fpath: str) -> Dict[str, str]:
    if not os.path.exists(fpath):
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------
# 2. Logic to Assign User Utterance (Original Logic Maintained)
# ---------------------------------------------------------
def assign_user_utterances(
    pref_list_path: str, 
    example: Dict[str, Any], 
    query_map: Dict[str, str], 
    pref_type: str, 
    pref_group_path: str = None
) -> List[Tuple[str, str]]:
    """
    Returns:
        List of (user_utterance, ground_truth_label) tuples.
    """
    results = []

    # [CASE 1] easy
    if pref_type == "easy":
        if not os.path.exists(pref_list_path):
            return []
            
        with open(pref_list_path, "r", encoding="utf-8") as f:
            pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try:
                        args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError:
                        continue 
                else:
                    domain = call_str.strip()
                    args_content = ""

                if domain not in query_map:
                    continue
                if domain not in pref_list:
                    continue

                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content)

                target_pref_slots = pref_list.get(domain, [])
                
                has_target_slot = False
                for slot, _ in matches:
                    if slot in target_pref_slots:
                        has_target_slot = True
                        break
                
                if not has_target_slot:
                    continue

                filtered_slots = []
                for slot, value in matches:
                    filtered_slots.append(f'{slot}="{value}"')

                if filtered_slots:
                    new_ground_truth = f"{domain}({', '.join(filtered_slots)})"
                    results.append((query_map[domain], new_ground_truth))
        
        return results

    # [CASE 2] medium
    elif pref_type == "medium":
        api_calls = example.get("api_calls", [])
        easy_domain_list=[]
        if isinstance(api_calls, list):
            for call_str in api_calls:
                if "(" in call_str:
                    easy_domain = call_str.split("(")[0].strip()
                else:
                    easy_domain = call_str.strip()
                easy_domain_list.append(easy_domain)

        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []
        
        if not pref_group_path or not os.path.exists(pref_group_path):
            print(f"Warning: pref_group_path not found: {pref_group_path}")
            return []
            
        with open(pref_group_path, "r", encoding="utf-8") as f:
            pref_group_data = json.load(f)

        for pref in prefs:
            if pref.get("value_group") in pref_group_data:
                evidence_list = pref.get("evidence", [])
                if not isinstance(evidence_list, list):
                    continue
                
                for evidence in evidence_list:
                    domain = evidence.get("domain")

                    if domain in easy_domain_list:
                        continue

                    elif domain and (domain in query_map):
                        slots_str_list = [f'{evidence["slot"]}="{evidence["value"]}"']
                        ground_truth_str = f"{domain}({', '.join(slots_str_list)})"
                        
                        results.append((query_map[domain], ground_truth_str))
        
        return results

    # [CASE 3] hard
    elif pref_type == "hard":
        if not pref_group_path or not os.path.exists(pref_group_path):
            print(f"Warning: pref_group_path not found: {pref_group_path}")
            return []
        
        with open(pref_group_path, "r", encoding="utf-8") as f:
            pref_group_data = json.load(f)

        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []

        for pref in prefs:
            current_group_name = pref.get("value_group")
            if not current_group_name or current_group_name not in pref_group_data:
                continue

            used_domains = set()
            for evidence in pref.get("evidence", []):
                d = evidence.get("domain")
                if d:
                    used_domains.add(d)

            group_rules = pref_group_data[current_group_name].get("rules", [])

            for rule in group_rules:
                candidate_domain = rule.get("domain")
                
                if candidate_domain and (candidate_domain in query_map) and (candidate_domain not in used_domains):
                    target_slot = rule.get("slot")
                    target_value = rule.get("value")
                    
                    if isinstance(target_value, bool):
                        val_str = "True" if target_value else "False"
                    else:
                        val_str = str(target_value)
                    
                    ground_truth_str = f'{candidate_domain}({target_slot}="{val_str}")'
                    results.append((query_map[candidate_domain], ground_truth_str))
        
        return results

    return results

# ---------------------------------------------------------
# 3. Helpers for History String Construction
# ---------------------------------------------------------
def get_api_calls_string(example: Dict[str, Any]) -> str:
    sessions = example.get("sessions", [])
    collected_apis = []

    for idx, session in enumerate(sessions, start=1):
        api_calls = session.get("api_call", [])
        if isinstance(api_calls, str) and api_calls:
            api_calls = [api_calls]
            
        if isinstance(api_calls, list):
            for call in api_calls:
                formatted_call = f"[Session {idx}] {call}"
                collected_apis.append(formatted_call)
    
    return "\n".join(collected_apis)

def get_dialogue_history_string(example: Dict[str, Any]) -> str:
    history_data = example.get("sessions", [])
    sessions_str = []
    for idx, instruction_data in enumerate(history_data, start=1):
        turns = instruction_data.get("dialogue", [])
        lines = [f"[Session {idx}]"]
        
        for turn in turns:
            role = turn.get("role", "").capitalize()
            content = turn.get("message") or turn.get("content") or ""
            if role and content:
                lines.append(f"{role}: {content}")
        
        sessions_str.append("\n".join(lines))

    return "\n\n".join(sessions_str)

# ---------------------------------------------------------
# 4. Build Input Prompt
# ---------------------------------------------------------
def build_input_prompt(example: Dict[str, Any], current_user_utterance: str, template: str, context_type: str, baseline_prompt: str) -> str:
    """
    Constructs the prompt. VLLM works best with a single string.
    We prepend the 'baseline_prompt' (System Prompt) manually.
    """
    api_str = get_api_calls_string(example)
    history_str = get_dialogue_history_string(example)
    
    final_context = ""
    if context_type == "diag-apilist":
        final_context = f"\n--- Dialogue History ---\n{history_str}\n\n--- Past API Calls ---\n{api_str}\n"
    elif context_type == "apilist-only":
        final_context = api_str
    elif context_type == "diag-only":
        final_context = history_str

    user_prompt = template.format(
        dialogue_history=final_context,
        user_utterance=current_user_utterance.strip(),
    )

    # Combine System Prompt + User Prompt
    # Note: Depending on the model, you might need specific Chat Tokens (e.g. <|start_header_id|>...)
    # Here we perform a simple concatenation for raw completion models.
    full_prompt = f"{baseline_prompt}\n\n{user_prompt}"
    return full_prompt

# ---------------------------------------------------------
# 5. Pipeline (Updated for vLLM Batch Inference)
# ---------------------------------------------------------
def process_with_vllm(
    input_path: str,
    output_path: str,
    log_path: str,
    query_map_path: str,
    pref_list_path: str, 
    pref_group_path: str,
    prompt_template: str,
    prompt_type_name: str,
    context_type: str,
    pref_type: str,
    model_name: str,
    tensor_parallel_size: int = 1
):
    # 1. Load Data
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)
    
    # 2. Load Baseline System Prompt
    baseline_prompt_path = "/data/minseo/personal-tool/conv_api/experiments3//new_baseline_prompt_update.txt"
    try:
        with open(baseline_prompt_path, "r", encoding="utf-8") as f:
            baseline_prompt = f.read()
    except FileNotFoundError:
        baseline_prompt = "You are a helpful assistant."

    print(f"Starting VLLM process... (Model: {model_name})")

    # 3. Initialize VLLM Engine (Do this once)
    # trust_remote_code=True may be needed for some models
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    batch_prompts = []
    batch_metadata = [] # Stores context to reconstruct result later
    
    skipped_count = 0

    # 4. Prepare Prompts (Batch Preparation)
    print("Preparing prompts for batch inference...")
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Preparing"):
        original_ex = row.to_dict()

        # Filtering
        if pref_type == "easy":
            if not original_ex.get("api_calls"):
                skipped_count += 1; continue
        elif pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"):
                skipped_count += 1; continue

        # Assign Utterance
        pairs_list = assign_user_utterances(
            pref_list_path, 
            original_ex, 
            query_map, 
            pref_type=pref_type, 
            pref_group_path=pref_group_path
        )
        
        if not pairs_list:
            skipped_count += 1
            continue

        for sub_idx, (utterance, ground_truth) in enumerate(pairs_list):
            current_ex = copy.deepcopy(original_ex)
            
            current_ex["user_utterance"] = utterance
            current_ex["reference_ground_truth"] = ground_truth
            current_ex["example_id_sub"] = f"{current_ex.get('example_id', 'unknown')}_{sub_idx}"
            current_ex["model_name"] = model_name
            
            # Construct Prompt
            prompt = build_input_prompt(
                current_ex, 
                current_user_utterance=utterance, 
                template=prompt_template, 
                context_type=context_type,
                baseline_prompt=baseline_prompt
            )

            batch_prompts.append(prompt)
            batch_metadata.append(current_ex)

    print(f"Skipped examples: {skipped_count}")
    print(f"Total examples to generate: {len(batch_prompts)}")

    if not batch_prompts:
        print("No prompts to generate.")
        return

    # 5. Run Batch Inference
    print("Running vLLM Inference...")
    outputs = llm.generate(batch_prompts, sampling_params)

    # 6. Process Results & Log
    processed_data = []
    
    # Sort outputs to match input order (vLLM usually preserves order but good to be safe by zipping)
    # outputs is a list of RequestOutput objects
    
    print("Saving results...")
    for i, request_output in enumerate(outputs):
        generated_text = request_output.outputs[0].text.strip()
        metadata = batch_metadata[i]
        
        # Add result to metadata object
        metadata["llm_output"] = generated_text
        
        # Create Log Record
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "example_id": metadata["example_id"],
            "example_id_sub": metadata["example_id_sub"], 
            "model_name": model_name,
            "prompt_type": prompt_type_name,
            "context_type": context_type,
            "pref_type": pref_type,
            "injected_utterance": metadata["user_utterance"],
            "reference_ground_truth": metadata["reference_ground_truth"],
            "model_input": batch_prompts[i],
            "model_output": generated_text,
        }
        
        write_log(log_path, log_record)
        processed_data.append(metadata)

    # 7. Save Final JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved -> {output_path}")


def write_log(log_path: str, record: Dict[str, Any]):
    dirpath = os.path.dirname(log_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/data/dev_4.json")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--log_path", type=str, default="process.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3//pref_group.json")

    parser.add_argument(
        "--context_type", 
        type=str, 
        choices=["diag-apilist", "apilist-only", "diag-only"], 
        default="diag-apilist"
    )
    
    parser.add_argument(
        "--pref_type", 
        type=str, 
        choices=["medium", "easy", "hard"], 
        required=True
    )
    
    parser.add_argument(
        "--prompt_type", 
        type=str, 
        choices=["imp-zs", "imp-fs", "imp-pref-group"], 
        default="imp-zs"
    )

    # Model Name (HuggingFace path)
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        choices=["deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "meta-llama/Llama-3.1-8B-Instruct"],
        help="Path to the local model or HF model ID (e.g., 'meta-llama/Llama-2-7b-chat-hf')."
    )
    
    # GPU options
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()

    if args.prompt_type == "imp-zs":
        selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-fs":
        selected_template = IMPLICIT_FS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-pref-group":
        selected_template = IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE
    else:
        selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE

    process_with_vllm(
        input_path=args.input_path,
        output_path=args.output_path,
        log_path=args.log_path,
        query_map_path=args.query_path,
        pref_list_path=args.pref_list_path,
        pref_group_path=args.pref_group_path,
        prompt_template=selected_template,
        prompt_type_name=args.prompt_type,
        context_type=args.context_type,
        pref_type=args.pref_type,
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size
    )