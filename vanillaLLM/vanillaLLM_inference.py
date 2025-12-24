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

# Gemini Library Import
import google.generativeai as genai

# If prompt module is local, keep it; otherwise define dummy variables.
from prompt import IMPLICIT_ZS_PROMPT_TEMPLATE, IMPLICIT_FS_PROMPT_TEMPLATE, IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE

# Initialize OpenAI API (Lazy initialization or global is fine, but we handle logic in the call function)
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None # Handle later if model is GPT but key is missing

# Initialize Gemini API
google_api_key = os.environ.get("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)

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

    # -------------------------------------------------------
    # [CASE 1] easy
    # -------------------------------------------------------
    if pref_type == "easy":
        if not os.path.exists(pref_list_path):
            return []
            
        with open(pref_list_path, "r", encoding="utf-8") as f:
            pref_list = json.load(f)

        #easy query extraction
        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                # --- Extract Domain & Args ---
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try:
                        args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError:
                        continue 
                else:
                    domain = call_str.strip()
                    args_content = ""

                # --- Filter Domain ---
                if domain not in query_map:
                    continue
                if domain not in pref_list:
                    continue

                # --- Parse arguments ---
                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content) # List of (slot, value)

                target_pref_slots = pref_list.get(domain, [])
                
                # Check Intersection
                has_target_slot = False
                for slot, _ in matches:
                    if slot in target_pref_slots:
                        has_target_slot = True
                        break
                
                if not has_target_slot:
                    continue

                # Construct GT
                filtered_slots = []
                for slot, value in matches:
                    filtered_slots.append(f'{slot}="{value}"')

                if filtered_slots:
                    new_ground_truth = f"{domain}({', '.join(filtered_slots)})"
                    results.append((query_map[domain], new_ground_truth))
        
        return results

# -------------------------------------------------------
    # [CASE 2] medium (Updated with Group Filtering)
    # -------------------------------------------------------
    elif pref_type == "medium":

        api_calls = example.get("api_calls", [])
        easy_domain_list=[]
        if isinstance(api_calls, list):
            for call_str in api_calls:
                # --- Extract Domain & Args ---
                if "(" in call_str:
                    easy_domain = call_str.split("(")[0].strip()
                    try:
                        args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError:
                        print(f"Warning: Malformed api_calls string: {call_str}")
                        continue 
                else:
                    easy_domain = call_str.strip()
                    args_content = ""
                easy_domain_list.append(easy_domain)


        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []
        
        # Load pref_group for filtering
        if not pref_group_path or not os.path.exists(pref_group_path):
            print(f"Warning: pref_group_path not found: {pref_group_path}")
            return []
            
        with open(pref_group_path, "r", encoding="utf-8") as f:
            pref_group_data = json.load(f)

        for pref in prefs:
            # Check if this preference's group exists in pref_group file
            if pref.get("value_group") in pref_group_data:
                
                evidence_list = pref.get("evidence", [])
                if not isinstance(evidence_list, list):
                    continue
                
                for evidence in evidence_list:
                    domain = evidence.get("domain")

                    if domain in easy_domain_list:
                        continue

                    elif domain and (domain in query_map):
                        # Use existing api_call string if available, otherwise construct it
                        #if 'api_call' in evidence:
                        #     ground_truth_str = evidence['api_call']
                        slots_str_list = [f'{evidence["slot"]}="{evidence["value"]}"']
                        ground_truth_str = f"{domain}({', '.join(slots_str_list)})"
                        
                        results.append((query_map[domain], ground_truth_str))
        
        return results
    # -------------------------------------------------------
    # [CASE 3] hard (Unseen Domain within Group)
    # -------------------------------------------------------
    elif pref_type == "hard":
        # 1. pref_group 파일 로드
        if not pref_group_path or not os.path.exists(pref_group_path):
            print(f"Warning: pref_group_path not found: {pref_group_path}")
            return []
        
        with open(pref_group_path, "r", encoding="utf-8") as f:
            pref_group_data = json.load(f)

        # 2. 예시의 api_calls_pref 확인
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []

        for pref in prefs:
            # (A) 현재 Preference의 그룹명 확인 (예: "low_friction")
            current_group_name = pref.get("value_group")
            if not current_group_name or current_group_name not in pref_group_data:
                continue

            # (B) 이미 사용된(Evidence에 있는) 도메인 수집 -> Hard 케이스는 사용되지 않은 도메인을 타겟팅
            used_domains = set()
            for evidence in pref.get("evidence", []):
                d = evidence.get("domain")
                if d:
                    used_domains.add(d)

            # (C) pref_group.json에서 해당 그룹의 전체 규칙(Rules) 가져오기
            group_rules = pref_group_data[current_group_name].get("rules", [])

            # (D) 규칙을 순회하며 "사용되지 않은 도메인" 찾기
            for rule in group_rules:
                candidate_domain = rule.get("domain")
                
                # 조건: 
                # 1. query_map에 있어서 User Utterance(질문) 생성이 가능해야 함
                # 2. used_domains에 없어야 함 (Unseen Domain)
                if candidate_domain and (candidate_domain in query_map) and (candidate_domain not in used_domains):
                    
                    # (E) Ground Truth 생성
                    # 요구사항: "Rules 안에 Query와 동일한 Domain의 Slot / Value로 구성"
                    target_slot = rule.get("slot")
                    target_value = rule.get("value")
                    
                    # 값 포맷팅 (Boolean/Int/String 처리)
                    if isinstance(target_value, bool):
                        val_str = "True" if target_value else "False"
                    else:
                        val_str = str(target_value)
                    
                    # GT 문자열 조립: Domain(slot="value")
                    # 해당 도메인의 Rule에 명시된 slot과 value만 사용하여 정답을 구성함
                    ground_truth_str = f'{candidate_domain}({target_slot}="{val_str}")'
                    
                    # 결과 추가 (User Query, Constructed GT)
                    results.append((query_map[candidate_domain], ground_truth_str))
        
        return results

    return results

# ---------------------------------------------------------
# 3. Helpers for History String Construction
# ---------------------------------------------------------
# ---------------------------------------------------------
# 3. Helpers for History String Construction
# ---------------------------------------------------------
def get_api_calls_string(example: Dict[str, Any]) -> str:
    """
    [Modified]
    Iterate through all 'sessions', collect 'api_call' entries,
    and prepend the session number (e.g., "[Session 1] GetRestaurants(...)").
    """
    sessions = example.get("sessions", [])
    collected_apis = []

    # enumerate를 사용하여 세션 인덱스(1부터 시작)를 추적
    for idx, session in enumerate(sessions, start=1):
        api_calls = session.get("api_call", [])
        
        # 문자열 하나만 있는 경우 리스트로 변환 (안전장치)
        if isinstance(api_calls, str) and api_calls:
            api_calls = [api_calls]
            
        if isinstance(api_calls, list):
            for call in api_calls:
                # [Session N] 태그를 앞에 붙임
                formatted_call = f"[Session {idx}] {call}"
                collected_apis.append(formatted_call)
    
    return "\n".join(collected_apis)

def get_dialogue_history_string(example: Dict[str, Any]) -> str:
    """Convert sessions data to dialogue history string"""
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

    return "\n\n".join(sessions_str)# ---------------------------------------------------------
# 4. Build Input Prompt (Updated for context_type)
# ---------------------------------------------------------
def build_input_prompt(example: Dict[str, Any], current_user_utterance: str, template: str, context_type: str) -> str:
    """
    Constructs the prompt based on context_type.
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
    else:
        final_context = ""

    prompt = template.format(
        dialogue_history=final_context,
        user_utterance=current_user_utterance.strip(),
    )
    return prompt


# ---------------------------------------------------------
# 5. Call LLM API (Updated for Gemini & GPT)
# ---------------------------------------------------------
def call_llm_api(prompt: str, model_name: str) -> str:
    # Baseline Prompt Loading (Common for both)
    baseline_prompt_path = "/data/minseo/personal-tool/conv_api/experiments3//new_baseline_prompt_update.txt"
    
    try:
        with open(baseline_prompt_path, "r", encoding="utf-8") as f:
            baseline_prompt = f.read()
    except FileNotFoundError:
        # Fallback if file not found during dev/test
        baseline_prompt = "You are a helpful assistant."

    try:
        # --- GEMINI LOGIC ---
        if "gemini" in model_name.lower():
            if not google_api_key:
                return "API_KEY_MISSING_GOOGLE"
            
            # Configure and call Gemini
            # system_instruction is supported in newer genai SDKs and Gemini 1.5+
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=baseline_prompt
            )
            
            # Generating content
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0
                )
            )
            return response.text.strip()

        # --- GPT LOGIC ---
        else:
            if not client:
                return "API_KEY_MISSING_OPENAI"
            
            response = client.chat.completions.create(
                model=model_name, 
                messages=[
                    {"role": "system", "content": baseline_prompt},
                    {"role": "user", "content": prompt},
                ],
                #temperature=0.0,
            )
            output = response.choices[0].message.content.strip()
            return output

    except Exception as e:
        print(f"LLM API Error ({model_name}): {e}")
        # Gemini can raise ValueError if content is blocked by safety filters
        return f"API_ERROR: {str(e)}"


# ---------------------------------------------------------
# 6. Pipeline (Updated)
# ---------------------------------------------------------
def process_with_llm(
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
    model_name: str  # Added argument
):
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)
    
    processed_data = []
    skipped_count = 0
    total_generated_cases = 0

    print(f"Starting process... (Model: {model_name}, Prompt: {prompt_type_name}, Context: {context_type}, Pref: {pref_type})")

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing Examples"):
        original_ex = row.to_dict()

        # [필터링 로직]
        if pref_type == "easy":
            if not original_ex.get("api_calls"):
                skipped_count += 1; continue
        elif pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"):
                skipped_count += 1; continue

        # [User Utterance 할당]
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

        # Loop through each extracted case
        for sub_idx, (utterance, ground_truth) in enumerate(pairs_list):
            total_generated_cases += 1
            
            current_ex = copy.deepcopy(original_ex)
            
            current_ex["user_utterance"] = utterance
            current_ex["reference_ground_truth"] = ground_truth
            current_ex["example_id_sub"] = f"{current_ex.get('example_id', 'unknown')}_{sub_idx}"
            current_ex["model_name"] = model_name
            
            # Generate Prompt
            prompt = build_input_prompt(
                current_ex, 
                current_user_utterance=utterance, 
                template=prompt_template, 
                context_type=context_type
            )

            # Call LLM (Changed from call_gpt_api)
            llm_output = call_llm_api(prompt, model_name)

            # Log
            log_record = {
                "timestamp": datetime.now().isoformat(),
                "example_id": current_ex["example_id"],
                "example_id_sub": current_ex["example_id_sub"], 
                "model_name": model_name,
                "prompt_type": prompt_type_name,
                "context_type": context_type,
                "pref_type": pref_type,
                "injected_utterance": utterance,
                "reference_ground_truth": ground_truth,
                "model_input": prompt,
                "model_output": llm_output,
            }
            write_log(log_path, log_record)

            # Save result
            current_ex["llm_output"] = llm_output # Renamed for clarity, but you can keep gpt_output if downstream depends on it
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
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/data/dev_4.json")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--log_path", type=str, default="process.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/pref_list.json")
    
    # Preference Group File Path
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3//pref_group.json", help="Path to pref_group.json for hard")

    parser.add_argument(
        "--context_type", 
        type=str, 
        choices=["diag-apilist", "apilist-only", "diag-only"], 
        default="diag-apilist",
        help="Choose what context to include: 'diag_only', 'api_only', or 'both'."
    )
    
    parser.add_argument(
        "--pref_type", 
        type=str, 
        choices=["medium", "easy", "hard"], 
        required=True,
        help="Select preference generation mode."
    )
    
    parser.add_argument(
        "--prompt_type", 
        type=str, 
        choices=["imp-zs", "imp-fs", "imp-pref-group"], 
        default="imp-zs"
    )

    # [New Argument] Model Name
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt-4o-mini-2024-07-18", 
        help="Name of the model to use (e.g., 'gpt-4o...', 'gemini-1.5-flash', 'gemini-1.5-pro')."
    )

    args = parser.parse_args()

    if args.prompt_type == "imp-zs":
        selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-fs":
        selected_template = IMPLICIT_FS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-pref-group":
        selected_template = IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE
    else:
        # Default or fail-safe
        selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE

    # Renamed function call
    process_with_llm(
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
        model_name=args.model_name # Pass model name
    )