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
from prompt import IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------
def run_evaluation(args):
    # 1. Load Resources
    if args.input_path.endswith('.jsonl'):
        df = pd.read_json(args.input_path, lines=True)
    else:
        with open(args.input_path, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
            
    with open(args.query_path, "r", encoding="utf-8") as f:
        query_map = json.load(f)

    # [시스템 프롬프트 로드]
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

    # [로그 파일 초기화]
    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        with open(args.log_path, "w", encoding="utf-8") as f:
            pass 
        print(f"Logging inputs and outputs to: {args.log_path}")

    # [메모리 파일 로드]
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
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Inference"):
        original_ex = row.to_dict()
        ex_id = str(original_ex.get("example_id")) 
        
        # 3. Retrieve Pre-calculated Memory
        user_memory = memory_storage.get(ex_id)
        if not user_memory:
            continue

        # 4. Determine Test Cases
        use_rule_imp = (args.pref_type == "rule_pref")
        eval_pairs = assign_user_utterances(args.pref_list_path, original_ex, query_map, use_rule_imp)
        
        if not eval_pairs:
            continue

        # 5. Inference Loop
        
        # --- [Context 구성 로직 수정 시작] ---
        # 기본 정보 추출
        explicit_pref = user_memory.get('final_explicit_pref', 'None')
        implicit_pref = user_memory.get('final_implicit_pref', 'None')
        
        accumulated_apis = user_memory.get('final_api_list', [])
        api_list_str = "\n".join(accumulated_apis) if accumulated_apis else "None"
        
        # 대화 이력 추출 (memory_dialogue용)
        # 데이터셋 구조에 따라 'turns' 혹은 'history' 필드를 사용한다고 가정
        raw_turns = original_ex.get("turns", []) 
        dialogue_text = "None"
        if raw_turns and isinstance(raw_turns, list):
            # speaker와 utterance가 있는 일반적인 구조라고 가정
            dialogue_text = "\n".join([f"{t.get('speaker', 'User')}: {t.get('utterance', '')}" for t in raw_turns])
        
        retrieved_memories_block = ""
        dialogue_history_input = ""

        # context_type에 따른 분기
        if args.context_type == "memory_only":
            # 1) Memory Only: 선호도만 포함, 히스토리 제외
            retrieved_memories_block = f"""
            [Explicit Preferences]: {explicit_pref}
            [Implicit Preferences]: {implicit_pref}
            """
            dialogue_history_input = "None (Reflected in Memory)"

        elif args.context_type == "memory_apilist":
            # 2) Memory + API List: 선호도 + API 호출 이력 포함
            retrieved_memories_block = f"""
            [Explicit Preferences]: {explicit_pref}
            [Implicit Preferences]: {implicit_pref}
            [Past API History]: {api_list_str}
            """
            dialogue_history_input = "None (Reflected in Memory and API History)"

        elif args.context_type == "memory_dialogue":
            # 3) Memory + Dialogue: 선호도 + 실제 대화 텍스트 포함
            retrieved_memories_block = f"""
            [Explicit Preferences]: {explicit_pref}
            [Implicit Preferences]: {implicit_pref}
            """
            dialogue_history_input = dialogue_text
        
        # --- [Context 구성 로직 수정 끝] ---

        for sub_idx, (utterance, ground_truth) in enumerate(eval_pairs):
            # Prompt Construction
            final_prompt = IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE.format(
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
                    "context_type": args.context_type, # 로그에 context type 명시
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
            
            processed_data.append(res_record)

    # 6. Save Final Results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Evaluation Done. Saved to {args.output_path}")

# ---------------------------------------------------------
# Utility: Test Case Extraction (Shared Logic)
# ---------------------------------------------------------
def assign_user_utterances(pref_list_path, example, query_map, use_rule_imp):
    results = []
    # 1. rule_pref mode
    if use_rule_imp:
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

    # 2. multi_pref_medium mode
    else:
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list): return []
        for pref in prefs:
            for ev in pref.get("evidence", []):
                domain = ev.get("domain")
                if domain and domain in query_map:
                    gt = ev.get('api_call', f'{domain}({ev["slot"]}="{ev["value"]}")')
                    results.append((query_map[domain], gt))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/data/sgd_converted_dev_mapped_grouped_with_pref_with_constraints.json")
    parser.add_argument("--memory_path", type=str, required=True, help="Path to the generated memory jsonl file")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/pref_list.json")
    
    parser.add_argument("--system_prompt_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/new_baseline_prompt_update.txt", help="Path to the system prompt text file")
    parser.add_argument("--log_path", type=str, required=True, help="Path to save prompt input/output logs (JSONL format)")
    
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--pref_type", type=str, choices=["multi_pref_medium", "rule_pref"], required=True)

    # [수정: context_type 인자 추가]
    parser.add_argument("--context_type", type=str, 
                        choices=["memory_only", "memory_dialogue", "memory_apilist"], 
                        default="memory_only",
                        help="Choose context construction strategy: 'memory_only', 'memory_dialogue', 'memory_apilist'")
    
    args = parser.parse_args()
    run_evaluation(args)