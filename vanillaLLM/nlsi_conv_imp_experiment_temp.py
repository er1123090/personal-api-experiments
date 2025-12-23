import tqdm
import os
import json 
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from openai import OpenAI
import re
import copy

# If prompt module is local, keep it; otherwise define dummy variables.
from prompt import EXPLICIT_ZS_PROMPT_TEMPLATE, EXPLICIT_FS_PROMPT_TEMPLATE, IMPLICIT_ZS_PROMPT_TEMPLATE, IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE
# Initialize OpenAI API
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
# 2. Logic to Assign User Utterance (Modified)
# ---------------------------------------------------------
def assign_user_utterances(pref_list_path: str, example: Dict[str, Any], query_map: Dict[str, str], use_rule_imp: bool = False) -> List[Tuple[str, str]]:
    """
    Returns:
        List of (user_utterance, ground_truth_label) tuples.
    """
    results = []

    # 1. --pref_type rule_pref mode (Parse api_calls + Filter based on pref_list presence)
    if use_rule_imp:
        if not os.path.exists(pref_list_path):
            return []
            
        with open(pref_list_path, "r", encoding="utf-8") as f:
            pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                # --- [Step 1] Extract Domain & Args ---
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try:
                        args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError:
                        continue 
                else:
                    domain = call_str.strip()
                    args_content = ""

                # --- [Step 2] Filter Domain ---
                if domain not in query_map:
                    continue
                if domain not in pref_list:
                    continue

                # --- [Step 3] Parse arguments ---
                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content) # List of (slot, value)

                # ------------------------------------------------------------------
                # [Logic Update] 
                # 1. Check Intersection: 해당 API Call이 pref_list의 슬롯을 하나라도 포함하는가?
                # 2. Construct GT: 포함한다면, Ground Truth는 (pref 여부와 상관없이) 모든 슬롯을 다 넣는다.
                # ------------------------------------------------------------------
                
                # A. 해당 도메인에서 우리가 관심있는(pref_list에 있는) 슬롯 목록
                target_pref_slots = pref_list.get(domain, [])
                
                # B. 현재 API Call에 있는 슬롯 중 하나라도 target_pref_slots에 포함되는지 확인
                has_target_slot = False
                for slot, _ in matches:
                    if slot in target_pref_slots:
                        has_target_slot = True
                        break
                
                # C. [Filter] 관심있는 슬롯이 하나도 없으면 이 케이스는 건너뜀 (query instance로 보지 않음)
                if not has_target_slot:
                    continue

                # D. [Construct GT] 통과했다면, 모든 슬롯을 사용하여 GT 생성
                filtered_slots = []
                for slot, value in matches:
                    # 모든 슬롯 포함 (pref_list에 없더라도)
                    filtered_slots.append(f'{slot}="{value}"')

                if filtered_slots:
                    new_ground_truth = f"{domain}({', '.join(filtered_slots)})"
                    results.append((query_map[domain], new_ground_truth))
        
        return results

    # 2. --pref_type multi_pref_medium mode (기존 로직 유지)
    else:
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []

        for pref in prefs:
            evidence_list = pref.get("evidence", [])
            if not isinstance(evidence_list, list):
                continue
            
            for evidence in evidence_list:
                domain = evidence.get("domain")
                
                if domain and domain in query_map:
                    # api_call 필드가 있다면 그것을 우선 사용, 없다면 조립
                    if 'api_call' in evidence:
                         # 괄호가 닫히지 않은 경우 처리 등은 데이터 형태에 따라 조정
                         ground_truth_str = evidence['api_call']
                    else:
                        # Fallback construction
                        slots_str_list = [f'{evidence["slot"]}="{evidence["value"]}"']
                        ground_truth_str = f"{domain}({', '.join(slots_str_list)})"
                    
                    results.append((query_map[domain], ground_truth_str))
        
        return results
# ---------------------------------------------------------
# 3. Helpers for History String Construction
# ---------------------------------------------------------
def get_api_calls_string(example: Dict[str, Any]) -> str:
    """Convert api_calls_all data to string"""
    all_api_data = example.get("api_calls_all", [])
    collected_apis = []

    for item in all_api_data:
        calls = item.get("api_call", [])
        if isinstance(calls, list):
            collected_apis.extend(calls)
        elif isinstance(calls, str) and calls:
            collected_apis.append(calls)
    
    return "\n".join(collected_apis)

def get_dialogue_history_string(example: Dict[str, Any]) -> str:
    """Convert all_standing_instructions data to dialogue history string"""
    history_data = example.get("all_standing_instructions", [])
    sessions_str = []
    for idx, instruction_data in enumerate(history_data, start=1):
        turns = instruction_data.get("generated_dialogue", [])
        lines = [f"[Session {idx}]"]
        
        for turn in turns:
            role = turn.get("role", "").capitalize()
            content = turn.get("message") or turn.get("content") or ""
            if role and content:
                lines.append(f"{role}: {content}")
        
        sessions_str.append("\n".join(lines))

    return "\n\n".join(sessions_str)

# ---------------------------------------------------------
# 4. Build Input Prompt (Updated for context_type)
# ---------------------------------------------------------
def build_input_prompt(example: Dict[str, Any], current_user_utterance: str, template: str, context_type: str) -> str:
    """
    Constructs the prompt based on context_type.
    
    context_type options:
    - 'both': Include API Calls + Dialogue History
    - 'apilist_only': Include API Calls only
    - 'dialogue_only': Include Dialogue History only
    """
    
    # Generate content strings
    api_str = get_api_calls_string(example)
    history_str = get_dialogue_history_string(example)
    
    final_context = ""

    if context_type == "both":
        final_context = f"--- Past API Calls ---\n{api_str}\n\n--- Dialogue History ---\n{history_str}"
    elif context_type == "apilist_only":
        final_context = api_str
    elif context_type == "dialogue_only":
        final_context = history_str
    else:
        # Fallback or 'none'
        final_context = ""

    # Inject into template
    # {dialogue_history} placeholder is used for generic context in these templates
    prompt = template.format(
        dialogue_history=final_context,
        user_utterance=current_user_utterance.strip(),
    )
    return prompt


# ---------------------------------------------------------
# 5. Call GPT API
# ---------------------------------------------------------
def call_gpt_api(prompt: str) -> str:
    baseline_prompt_path = "/data/minseo/personal-tool/conv_api/experiments/new_baseline_prompt_update.txt"
    
    if not os.path.exists(baseline_prompt_path):
         baseline_prompt = "You are a helpful assistant."
    else:
        with open(baseline_prompt_path, "r", encoding="utf-8") as f:
            baseline_prompt = f.read()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18", 
            messages=[
                {"role": "system", "content": baseline_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        output = response.choices[0].message.content.strip()
        return output
    except Exception as e:
        print(f"GPT Error: {e}")
        return "API_ERROR"


# ---------------------------------------------------------
# 6. Pipeline (Updated)
# ---------------------------------------------------------
def process_with_gpt(
    input_path: str,
    output_path: str,
    log_path: str,
    query_map_path: str,
    pref_list_path: str, 
    prompt_template: str,
    prompt_type_name: str,
    context_type: str,
    pref_type: str
):
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)
    
    processed_data = []
    skipped_count = 0
    total_generated_cases = 0

    print(f"Starting process... (Prompt: {prompt_type_name}, Context: {context_type}, Pref: {pref_type})")

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing Examples"):
        original_ex = row.to_dict()

        # [Logic Change] pref_type에 따른 필터링 및 rule_imp 플래그 설정
        use_rule_imp = False
        
        if pref_type == "rule_pref":
            # Rule based preference -> api_calls 데이터가 있어야 함
            if not original_ex.get("api_calls"):
                skipped_count += 1; continue
            use_rule_imp = True
            
        elif pref_type == "multi_pref_medium":
            # Multi preference -> api_calls_pref 데이터가 있어야 함
            if not original_ex.get("api_calls_pref"):
                skipped_count += 1; continue
            use_rule_imp = False

        # [Modified] Get ALL possible (Utterance, GT) pairs for this example
        pairs_list = assign_user_utterances(pref_list_path, original_ex, query_map, use_rule_imp=use_rule_imp)
        
        if not pairs_list:
            skipped_count += 1
            continue

        # Loop through each extracted case and evaluate individually
        for sub_idx, (utterance, ground_truth) in enumerate(pairs_list):
            total_generated_cases += 1
            
            # Deepcopy to ensure we don't modify the original object for next iterations
            current_ex = copy.deepcopy(original_ex)
            
            # Set current specific utterance and GT
            current_ex["user_utterance"] = utterance
            current_ex["reference_ground_truth"] = ground_truth
            # Add sub-index to ID to distinguish cases (e.g. dev_001_0, dev_001_1)
            current_ex["example_id_sub"] = f"{current_ex.get('example_id', 'unknown')}_{sub_idx}"
            
            # Generate Prompt
            prompt = build_input_prompt(
                current_ex, 
                current_user_utterance=utterance, 
                template=prompt_template, 
                context_type=context_type
            )

            # Call GPT
            gpt_output = call_gpt_api(prompt)

            # Log
            log_record = {
                "timestamp": datetime.now().isoformat(),
                "example_id": current_ex["example_id"],
                "example_id_sub": current_ex["example_id_sub"], 
                "prompt_type": prompt_type_name,
                "context_type": context_type,
                "pref_type": pref_type,
                "injected_utterance": utterance,
                "reference_ground_truth": ground_truth,
                "model_input": prompt,
                "model_output": gpt_output,
            }
            write_log(log_path, log_record)

            # Save result to current instance
            current_ex["gpt_output"] = gpt_output
            
            # Add to final list
            processed_data.append(current_ex)

    print(f"Skipped examples: {skipped_count}")
    print(f"Total test cases generated and processed: {total_generated_cases}")

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
# 7. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/data/sgd_converted_dev_mapped_grouped_with_pref_with_constraints.json")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--log_path", type=str, default="process.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/pref_list.json")
    
    # [New Arguments] Context Type Selection
    parser.add_argument(
        "--context_type", 
        type=str, 
        choices=["dialogue_only", "apilist_only", "both"], 
        required=True,
        help="Choose what context to include: 'dialogue_only', 'apilist_only', or 'both'."
    )
    
    # [New Arguments] Preference Type Selection
    parser.add_argument(
        "--pref_type", 
        type=str, 
        choices=["multi_pref_medium", "rule_pref"], 
        required=True,
        help="Select preference generation mode: 'rule_pref' (from api_calls) or 'multi_pref_medium' (from api_calls_pref)"
    )
    
    parser.add_argument("--prompt_type", type=str, choices=["exp-zs", "exp-fs", "imp-zs", "imp-fs", "imp-pref-group"], default="imp-zs")

    args = parser.parse_args()

    if args.prompt_type == "exp-zs":
        selected_template = EXPLICIT_ZS_PROMPT_TEMPLATE
    elif args.prompt_type == "exp-fs":
        selected_template = EXPLICIT_FS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-zs":
        selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-pref-group":
        selected_template = IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE
        pass 

    process_with_gpt(
        input_path=args.input_path,
        output_path=args.output_path,
        log_path=args.log_path,
        query_map_path=args.query_path,
        pref_list_path=args.pref_list_path,
        prompt_template=selected_template,
        prompt_type_name=args.prompt_type,
        context_type=args.context_type,
        pref_type=args.pref_type
    )