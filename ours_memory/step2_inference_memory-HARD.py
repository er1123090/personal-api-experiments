import tqdm
import os
import json
import argparse
import pandas as pd
from openai import OpenAI
import re
import copy

# ---------------------------------------------------------
# [Prompt Import]
# ---------------------------------------------------------
# Ensure prompt.py exists in the same directory or python path
from prompt import (
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE, 
    IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE
)

# [Prompt Map Definition]
PROMPT_MAP = {
    "prefgroup": IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE,
    "imp-zs": IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
    # Add other templates here if needed
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------
def run_evaluation(args):
    # Prompt Validation
    if args.prompt_template not in PROMPT_MAP:
        raise ValueError(f"Invalid prompt template: {args.prompt_template}. Choose from {list(PROMPT_MAP.keys())}")
    
    selected_prompt_template = PROMPT_MAP[args.prompt_template]
    print(f"Using Prompt Template: {args.prompt_template}")

    # 1. Load Resources
    if args.input_path.endswith('.jsonl'):
        df = pd.read_json(args.input_path, lines=True)
    else:
        with open(args.input_path, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
            
    with open(args.query_path, "r", encoding="utf-8") as f:
        query_map = json.load(f)

    # [System Prompt Load]
    system_prompt_content = ""
    if args.system_prompt_path:
        if os.path.exists(args.system_prompt_path):
            with open(args.system_prompt_path, "r", encoding="utf-8") as f:
                system_prompt_content = f.read().strip()
            print(f"Loaded System Prompt from: {args.system_prompt_path}")
        else:
            print(f"Warning: System prompt file not found at {args.system_prompt_path}. Using empty system prompt.")
    else:
        print("No system prompt path provided. Using empty system prompt.")

    # [Log File Init]
    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        with open(args.log_path, "w", encoding="utf-8") as f:
            pass 
        print(f"Logging inputs and outputs to: {args.log_path}")

    # [Memory File Load]
    if not os.path.exists(args.memory_path):
        raise FileNotFoundError(f"Memory file not found: {args.memory_path}. Run generate_memory.py first.")
    
    print(f"Loading memory from {args.memory_path}...")
    memory_storage = {}
    with open(args.memory_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                record = json.loads(line)
                rec_id = str(record.get("example_id"))
                memory_storage[rec_id] = record
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in memory file.")
                continue
        
    print(f"Loaded {len(memory_storage)} memory records. Evaluating with PrefType: {args.pref_type}")
    print(f"Context Type Strategy: {args.context_type}")
    
    processed_data = []
    
    # 2. Iterate Dialogues
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Inference ({args.prompt_template})"):
        original_ex = row.to_dict()
        ex_id = str(original_ex.get("example_id")) 
        
        # 3. Retrieve Pre-calculated Memory
        user_memory = memory_storage.get(ex_id)
        if not user_memory:
            continue

        # [Filter Logic based on pref_type]
        if args.pref_type == "easy":
            if not original_ex.get("api_calls"): continue
        elif args.pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"): continue

        # 4. Determine Test Cases
        eval_pairs = assign_user_utterances(
            pref_list_path=args.pref_list_path, 
            example=original_ex, 
            query_map=query_map, 
            pref_type=args.pref_type, 
            pref_group_path=args.pref_group_path
        )
        
        if not eval_pairs:
            continue

        # 5. Inference Loop
        
        # --- [Context Construction] ---
        # Fetch raw lists
        explicit_pref_data = user_memory.get('final_explicit_pref', [])
        implicit_pref_data = user_memory.get('final_implicit_pref', [])
        
        # Helper function to format list into string with Confidence
        def format_pref_list(pref_data):
            if not pref_data:
                return "None"
            
            # If it's already a string (legacy format), return as is
            if isinstance(pref_data, str):
                return pref_data
            
            lines = []
            if isinstance(pref_data, list):
                for item in pref_data:
                    if isinstance(item, dict):
                        content = item.get("content", str(item))
                        conf = item.get("confidence")
                        
                        # Formatting: Append confidence if it exists
                        if conf is not None and isinstance(conf, (int, float)):
                            line_str = f"- {content} (Confidence: {conf:.2f})"
                        else:
                            line_str = f"- {content}"
                        lines.append(line_str)
                    else:
                        # Fallback for simple string lists
                        lines.append(f"- {item}")
            
            return "\n".join(lines) if lines else "None"

        # Apply formatting
        explicit_pref = format_pref_list(explicit_pref_data)
        implicit_pref = format_pref_list(implicit_pref_data)

        accumulated_apis = user_memory.get('final_api_list', [])
        api_list_str = "\n".join(accumulated_apis) if accumulated_apis else "None"
        
        raw_turns = original_ex.get("turns", []) 
        dialogue_text = "None"
        if raw_turns and isinstance(raw_turns, list):
            dialogue_text = "\n".join([f"{t.get('speaker', 'User')}: {t.get('utterance', '')}" for t in raw_turns])
        
        retrieved_memories_block = ""
        dialogue_history_input = ""

        if args.context_type == "memory_only":
            retrieved_memories_block = f"""
            [Explicit Preferences]:
            {explicit_pref}
            
            [Implicit Preferences]:
            {implicit_pref}
            """
            dialogue_history_input = "None (Reflected in Memory)"

        elif args.context_type == "memory_api":
            retrieved_memories_block = f"""
            [Explicit Preferences]:
            {explicit_pref}
            
            [Implicit Preferences]:
            {implicit_pref}
            
            [Past API History]:
            {api_list_str}
            """
            dialogue_history_input = "None (Reflected in Memory and API History)"

        elif args.context_type == "memory_diag":
            retrieved_memories_block = f"""
            [Explicit Preferences]:
            {explicit_pref}
            
            [Implicit Preferences]:
            {implicit_pref}
            """
            dialogue_history_input = dialogue_text
        
        # --- [End Context Construction] ---

        for sub_idx, (utterance, ground_truth) in enumerate(eval_pairs):
            # Prompt Construction
            final_prompt = selected_prompt_template.format(
                retrieved_memories=retrieved_memories_block,
                dialogue_history=dialogue_history_input, 
                user_utterance=utterance
            )
            
            # API Call Preparation
            messages_payload = []
            if system_prompt_content:
                messages_payload.append({"role": "system", "content": system_prompt_content})
            messages_payload.append({"role": "user", "content": final_prompt})

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=messages_payload,
                    temperature=0.0
                )
                gpt_output = response.choices[0].message.content.strip()
            except Exception as e:
                gpt_output = f"ERROR: {e}"

            # [Log Save Logic]
            if args.log_path:
                log_entry = {
                    "example_id": ex_id,
                    "sub_idx": sub_idx,
                    "prompt_template": args.prompt_template,
                    "context_type": args.context_type,
                    "pref_type": args.pref_type,
                    "full_messages": messages_payload,
                    "final_user_prompt_text": final_prompt,
                    "gpt_output": gpt_output,
                    "ground_truth": ground_truth
                }
                with open(args.log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            # Result Record
            res_record = copy.deepcopy(original_ex)
            res_record["example_id_sub"] = f"{ex_id}_{sub_idx}"
            res_record["test_utterance"] = utterance
            res_record["reference_ground_truth"] = ground_truth
            res_record["gpt_output"] = gpt_output
            res_record["used_memory"] = user_memory
            res_record["context_type_used"] = args.context_type
            res_record["pref_type_used"] = args.pref_type
            res_record["prompt_template_used"] = args.prompt_template
            
            processed_data.append(res_record)

    # 6. Save Final Results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Evaluation Done. Saved to {args.output_path}")

# ---------------------------------------------------------
# Utility: Test Case Extraction
# ---------------------------------------------------------
def assign_user_utterances(pref_list_path, example, query_map, pref_type, pref_group_path=None):
    results = []

    # 1. easy mode
    if pref_type == "easy":
        if not os.path.exists(pref_list_path): return []
        with open(pref_list_path, "r", encoding="utf-8") as f:
            pref_list = json.load(f)
        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try: args = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except: continue
                else: domain = call_str.strip(); args = ""

                if domain not in query_map or domain not in pref_list: continue
                
                # Intersection Check
                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args)
                target_slots = pref_list.get(domain, [])
                if not any(slot in target_slots for slot, _ in matches): continue
                
                # Construct GT
                filtered = [f'{s}="{v}"' for s, v in matches]
                if filtered:
                    gt = f"{domain}({', '.join(filtered)})"
                    results.append((query_map[domain], gt))

    # 2. medium mode
    elif pref_type == "medium":
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list): return []
        for pref in prefs:
            for ev in pref.get("evidence", []):
                domain = ev.get("domain")
                if domain and domain in query_map:
                    gt = ev.get('api_call', f'{domain}({ev["slot"]}="{ev["value"]}")')
                    results.append((query_map[domain], gt))
    
    # 3. hard mode
    elif pref_type == "hard":
        if not pref_group_path or not os.path.exists(pref_group_path):
            print(f"Warning: pref_group_path not found: {pref_group_path}")
            return []
        
        with open(pref_group_path, "r", encoding="utf-8") as f:
            pref_group_data = json.load(f)

        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list): return []

        for pref in prefs:
            # (A) Get Group Name
            current_group_name = pref.get("value_group")
            if not current_group_name or current_group_name not in pref_group_data:
                continue

            # (B) Collect Used Domains
            used_domains = set()
            for evidence in pref.get("evidence", []):
                d = evidence.get("domain")
                if d: used_domains.add(d)

            # (C) Find Unseen Domain in Group
            group_rules = pref_group_data[current_group_name].get("rules", [])
            for rule in group_rules:
                candidate_domain = rule.get("domain")
                
                # Condition: Domain in QueryMap AND NOT in UsedDomains
                if candidate_domain and (candidate_domain in query_map) and (candidate_domain not in used_domains):
                    target_slot = rule.get("slot")
                    target_value = rule.get("value")
                    
                    # Formatting
                    if isinstance(target_value, bool):
                        val_str = "True" if target_value else "False"
                    else:
                        val_str = str(target_value)
                    
                    # Construct GT using Rule Definition
                    ground_truth_str = f'{candidate_domain}({target_slot}="{val_str}")'
                    results.append((query_map[candidate_domain], ground_truth_str))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/data/sgd_converted_dev_mapped_grouped_with_pref_with_constraints.json")
    parser.add_argument("--memory_path", type=str, required=True, help="Path to the generated memory jsonl file")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/pref_list.json")
    
    # Hard Mode Params
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/vanillaLLM/pref_group.json", help="Path to pref_group.json for hard")

    parser.add_argument("--system_prompt_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/new_baseline_prompt_update.txt", help="Path to the system prompt text file")
    parser.add_argument("--log_path", type=str, required=True, help="Path to save prompt input/output logs (JSONL format)")
    
    parser.add_argument("--output_path", type=str, required=True)
    
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)

    parser.add_argument("--context_type", type=str, 
                        choices=["memory_only", "memory_diag", "memory_api"], 
                        required=True,
                        help="Choose context construction strategy: 'memory_only', 'memory_diag', 'memory_api'")
    
    parser.add_argument("--prompt_template", type=str, required=True,
                        choices=list(PROMPT_MAP.keys()), 
                        help="Select inference prompt template")
    
    args = parser.parse_args()
    run_evaluation(args)