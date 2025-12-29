import tqdm
import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime
import re
import copy
import asyncio 
from openai import AsyncOpenAI 

# ---------------------------------------------------------
# [Helper] Load Tools from File
# ---------------------------------------------------------
def load_tools_from_file(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        print(f"Error: Tools schema file not found at {file_path}")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tools = json.load(f)
        print(f"[Info] Successfully loaded {len(tools)} tools from {file_path}")
        return tools
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

# ---------------------------------------------------------
# 1. Load Dataset & Queries
# ---------------------------------------------------------
def load_chains_dataset(fpath: str) -> pd.DataFrame:
    try:
        df = pd.read_json(fpath, lines=True)
        return df
    except ValueError:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

def load_query_map(fpath: str) -> Dict[str, str]:
    if not os.path.exists(fpath): return {}
    with open(fpath, "r", encoding="utf-8") as f: return json.load(f)

# ---------------------------------------------------------
# 2. Assign User Utterances (Dataset Logic)
# ---------------------------------------------------------
def assign_user_utterances(pref_list_path: str, example: Dict[str, Any], query_map: Dict[str, str], pref_type: str, pref_group_path: str = None) -> List[Tuple[str, str]]:
    results = []
    
    # [CASE 1] easy
    if pref_type == "easy":
        if not os.path.exists(pref_list_path): return []
        with open(pref_list_path, "r", encoding="utf-8") as f: pref_list = json.load(f)
        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try: args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError: continue 
                else:
                    domain = call_str.strip(); args_content = ""

                if domain not in query_map or domain not in pref_list: continue
                
                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content)
                target_pref_slots = pref_list.get(domain, [])
                
                if any(slot in target_pref_slots for slot, _ in matches):
                    filtered_slots = [f'{slot}="{value}"' for slot, value in matches]
                    if filtered_slots:
                        results.append((query_map[domain], f"{domain}({', '.join(filtered_slots)})"))
        return results

    # [CASE 2] medium
    elif pref_type == "medium":
        api_calls = example.get("api_calls", [])
        easy_domain_list=[]
        if isinstance(api_calls, list):
            for call_str in api_calls:
                easy_domain_list.append(call_str.split("(")[0].strip() if "(" in call_str else call_str.strip())

        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []
        if not pref_group_path or not os.path.exists(pref_group_path): return []
        with open(pref_group_path, "r", encoding="utf-8") as f: pref_group_data = json.load(f)

        for pref in prefs:
            if pref.get("value_group") in pref_group_data:
                for evidence in pref.get("evidence", []):
                    domain = evidence.get("domain")
                    if domain not in easy_domain_list and domain and (domain in query_map):
                        slots_str_list = [f'{evidence["slot"]}="{evidence["value"]}"']
                        results.append((query_map[domain], f"{domain}({', '.join(slots_str_list)})"))
        return results

    # [CASE 3] hard
    elif pref_type == "hard":
        if not pref_group_path or not os.path.exists(pref_group_path): return []
        with open(pref_group_path, "r", encoding="utf-8") as f: pref_group_data = json.load(f)
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []

        for pref in prefs:
            current_group_name = pref.get("value_group")
            if not current_group_name or current_group_name not in pref_group_data: continue
            used_domains = {e.get("domain") for e in pref.get("evidence", []) if e.get("domain")}
            
            for rule in pref_group_data[current_group_name].get("rules", []):
                candidate_domain = rule.get("domain")
                if candidate_domain and (candidate_domain in query_map) and (candidate_domain not in used_domains):
                    target_value = rule.get("value")
                    val_str = "True" if isinstance(target_value, bool) and target_value else "False" if isinstance(target_value, bool) else str(target_value)
                    results.append((query_map[candidate_domain], f'{candidate_domain}({rule.get("slot")}="{val_str}")'))
        return results
    return results

# ---------------------------------------------------------
# 3. Helpers (Prompt Building)
# ---------------------------------------------------------
def get_api_calls_string(example: Dict[str, Any]) -> str:
    collected_apis = []
    for idx, session in enumerate(example.get("sessions", []), start=1):
        api_calls = session.get("api_call", [])
        if isinstance(api_calls, str) and api_calls: api_calls = [api_calls]
        if isinstance(api_calls, list):
            for call in api_calls:
                collected_apis.append(f"[Session {idx}] {call}")
    return "\n".join(collected_apis)

def get_dialogue_history_string(example: Dict[str, Any]) -> str:
    sessions_str = []
    for idx, instruction_data in enumerate(example.get("sessions", []), start=1):
        lines = [f"[Session {idx}]"]
        for turn in instruction_data.get("dialogue", []):
            role = turn.get("role", "").capitalize()
            content = turn.get("message") or turn.get("content") or ""
            if role and content: lines.append(f"{role}: {content}")
        sessions_str.append("\n".join(lines))
    return "\n\n".join(sessions_str)

def build_input_prompt(example: Dict[str, Any], current_user_utterance: str, template: str, context_type: str) -> str:
    api_str = get_api_calls_string(example)
    history_str = get_dialogue_history_string(example)
    
    final_context = ""
    if context_type == "diag-apilist": final_context = f"\n--- Dialogue History ---\n{history_str}\n\n--- Past API Calls ---\n{api_str}\n"
    elif context_type == "apilist-only": final_context = api_str
    elif context_type == "diag-only": final_context = history_str
    
    return template.format(dialogue_history=final_context, user_utterance=current_user_utterance.strip())

# ---------------------------------------------------------
# [ASYNC HELPER] Single Request Processor
# ---------------------------------------------------------
async def process_single_example(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    log_path: str,
    baseline_prompt: str,
    prompt: str,
    tools_schema: List[Dict],
    model_name: str,
    current_ex: Dict[str, Any],
    prompt_type_name: str,
    context_type: str,
    pref_type: str,
    utterance: str,
    ground_truth: str,
    pbar: tqdm.tqdm
) -> Dict[str, Any]:
    
    # Semaphore를 사용하여 동시 요청 수 제한
    async with semaphore:
        llm_output = ""
        reasoning_content = "" # [init] reasoning tokens 저장 변수
        raw_content = "" # [init] raw content 저장 변수
        
        try:
            # [API CALL] Async Call
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    #{"role": "system", "content": baseline_prompt},
                    {"role": "user", "content": prompt},
                ],
                tools=tools_schema if tools_schema else None,
                # [CRITICAL] DeepSeek <think> 에러 방지용 auto
                tool_choice="auto" if tools_schema else None,
                temperature=0.0,
            )

            message = response.choices[0].message
            raw_content = message.content or "" # [SAVE] 원본 저장
            
            # ---------------------------------------------------------
            # [HYBRID PARSING LOGIC with DeepSeek Support]
            # ---------------------------------------------------------
            
            # 1. DeepSeek <think> Token Extraction (Updated: rfind logic)
            # 가장 마지막 </think> 태그를 기준으로 나눔
            end_tag = "</think>"
            end_idx = raw_content.rfind(end_tag)
            
            if end_idx != -1:
                # [Reasoning] </think> 앞에 있는 모든 내용
                # <think> 태그가 포함되어 있다면 제거하여 순수 텍스트만 추출
                reasoning_part = raw_content[:end_idx]
                reasoning_content = reasoning_part.replace("<think>", "").strip()
                
                # [Clean Content] </think> 뒤에 있는 모든 내용
                clean_content = raw_content[end_idx + len(end_tag):].strip()
            else:
                # 태그가 없으면 전체를 컨텐츠로 간주
                reasoning_content = ""
                clean_content = raw_content.strip()

            # 2. Structured Tool Call (우선순위 1)
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                    args_str_list = [f'{k}="{v}"' for k, v in func_args.items()]
                    llm_output = f"{func_name}({', '.join(args_str_list)})"
                except:
                    llm_output = f"ERROR_JSON_PARSE: {tool_call.function.arguments}"

            # 3. Text Parsing (우선순위 2 - DeepSeek R1 등 tool_call 미사용 시)
            else:
                # clean_content(사고 과정 제거됨)에서 함수 호출 패턴 추출
                match = re.search(r"([a-zA-Z0-9_]+)\((.*?)\)", clean_content, flags=re.DOTALL)
            
                if match:
                    llm_output = match.group(0).strip()
                else:
                    llm_output = f"ERROR_NO_FUNC_CALL: {clean_content}"

        except Exception as e:
            # print(f"API Error: {e}") 
            llm_output = f"ERROR_API: {str(e)}"

        # Log & Save (Thread/Async Safe)
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
            "raw_content": raw_content, # [NEW] 원본 내용 저장
            "reasoning_tokens": reasoning_content, 
            "model_output": llm_output,
        }
        
        # [Lock] 파일 쓰기 보호 (로그 파일)
        async with file_lock:
            dirpath = os.path.dirname(log_path)
            if dirpath: os.makedirs(dirpath, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        # 진행바 업데이트
        pbar.update(1)

        # 결과 반환 (여기에 추가해야 output.json에 저장됨)
        result_ex = copy.deepcopy(current_ex)
        result_ex["llm_output"] = llm_output
        result_ex["reasoning_tokens"] = reasoning_content
        result_ex["raw_content"] = raw_content # [NEW] 최종 Output JSON 포함
        return result_ex

# ---------------------------------------------------------
# 5. Pipeline (ASYNC Main)
# ---------------------------------------------------------
async def process_with_vllm_server_async(
    input_path: str, output_path: str, log_path: str, query_map_path: str,
    pref_list_path: str, pref_group_path: str, tools_schema_path: str,
    prompt_template: str, prompt_type_name: str, context_type: str, pref_type: str, 
    model_name: str, vllm_url: str, concurrency: int
):
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)
    
    # 1. Load System Prompt
    baseline_prompt_path = "/data/minseo/personal-api-experiments/new_baseline_prompt_update.txt"
    try:
        with open(baseline_prompt_path, "r", encoding="utf-8") as f: 
            baseline_prompt = f.read()
    except FileNotFoundError:
        baseline_prompt = "You are a function calling AI. Output the correct tool call."

    # 2. Load Tools Schema
    print(f"Loading tools schema from {tools_schema_path}...")
    tools_schema = load_tools_from_file(tools_schema_path)
    if not tools_schema:
        print("[Warning] Tools schema is empty! 'tool_choice' will be disabled.")

    # 3. Initialize Async Client & Sync Limits
    client = AsyncOpenAI(base_url=vllm_url, api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency) # 동시 요청 수 제한
    file_lock = asyncio.Lock() # 파일 쓰기 락
    
    print(f"Connected to vLLM Server at {vllm_url} (Max Concurrency: {concurrency})")
    print(f"Starting process... (Model: {model_name})")

    # 4. Prepare Tasks (데이터 전처리 - 동기적으로 빠르게 수행)
    tasks = []
    skipped_count = 0
    
    # (A) 먼저 처리해야 할 모든 작업을 리스트업
    print("Preparing tasks...")
    
    # 실제 API 호출용 데이터를 준비합니다.
    pending_items = []

    for _, row in df.iterrows():
        original_ex = row.to_dict()

        if pref_type == "easy":
            if not original_ex.get("api_calls"): skipped_count += 1; continue
        elif pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"): skipped_count += 1; continue

        pairs_list = assign_user_utterances(pref_list_path, original_ex, query_map, pref_type, pref_group_path)
        if not pairs_list:
            skipped_count += 1
            continue

        for sub_idx, (utterance, ground_truth) in enumerate(pairs_list):
            current_ex = copy.deepcopy(original_ex)
            current_ex.update({
                "user_utterance": utterance,
                "reference_ground_truth": ground_truth,
                "example_id_sub": f"{current_ex.get('example_id', 'unknown')}_{sub_idx}",
                "model_name": model_name
            })
            
            prompt = build_input_prompt(current_ex, utterance, prompt_template, context_type)
            
            # 작업 큐에 추가
            pending_items.append({
                "prompt": prompt,
                "current_ex": current_ex,
                "utterance": utterance,
                "ground_truth": ground_truth
            })

    total_tasks = len(pending_items)
    print(f"Total tasks prepared: {total_tasks} (Skipped source examples: {skipped_count})")

    # 5. Execute Async Tasks
    # Tqdm 진행바 생성
    pbar = tqdm.tqdm(total=total_tasks, desc="Async Inference")

    for item in pending_items:
        task = process_single_example(
            client=client,
            semaphore=semaphore,
            file_lock=file_lock,
            log_path=log_path,
            baseline_prompt=baseline_prompt,
            prompt=item["prompt"],
            tools_schema=tools_schema,
            model_name=model_name,
            current_ex=item["current_ex"],
            prompt_type_name=prompt_type_name,
            context_type=context_type,
            pref_type=pref_type,
            utterance=item["utterance"],
            ground_truth=item["ground_truth"],
            pbar=pbar
        )
        tasks.append(task)

    # 모든 작업 실행 및 대기
    # process_single_example이 반환하는 result_ex(reasoning_tokens 포함)들이 processed_data 리스트에 모임
    processed_data = await asyncio.gather(*tasks)
    
    pbar.close()
    await client.close()

    # 6. Save Final Output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # 여기에 저장될 때 reasoning_tokens도 포함됨
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved -> {output_path}")


# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-api-experiments/data/dev_4.json")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--log_path", type=str, default="process.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-api-experiments/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-api-experiments/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-api-experiments/pref_group.json")
    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/personal-api-experiments/tools_schema.json")

    parser.add_argument("--context_type", type=str, choices=["diag-apilist", "apilist-only", "diag-only"], default="diag-apilist")
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--prompt_type", type=str, choices=["imp-zs", "imp-fs", "imp-pref-group"], default="imp-zs")
    
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1")
    
    # [NEW] Concurrency Control
    parser.add_argument("--concurrency", type=int, default=50, help="Max number of concurrent requests to vLLM")

    from prompt import IMPLICIT_ZS_PROMPT_TEMPLATE, IMPLICIT_FS_PROMPT_TEMPLATE, IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE
    
    args = parser.parse_args()

    if args.prompt_type == "imp-zs": selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-fs": selected_template = IMPLICIT_FS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-pref-group": selected_template = IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE
    else: selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE

    # [NEW] Asyncio Run
    asyncio.run(process_with_vllm_server_async(
        input_path=args.input_path,
        output_path=args.output_path,
        log_path=args.log_path,
        query_map_path=args.query_path,
        pref_list_path=args.pref_list_path,
        pref_group_path=args.pref_group_path,
        tools_schema_path=args.tools_schema_path,
        prompt_template=selected_template,
        prompt_type_name=args.prompt_type,
        context_type=args.context_type,
        pref_type=args.pref_type,
        model_name=args.model_name,
        vllm_url=args.vllm_url,
        concurrency=args.concurrency
    ))