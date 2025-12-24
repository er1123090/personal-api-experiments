import tqdm
import os
import json 
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import copy
import asyncio # [Added] for async
from openai import AsyncOpenAI # [Modified] Use AsyncClient

# Gemini Library Import
import google.generativeai as genai

# If prompt module is local, keep it; otherwise define dummy variables.
from prompt import IMPLICIT_ZS_PROMPT_TEMPLATE, IMPLICIT_FS_PROMPT_TEMPLATE, IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE

# Initialize OpenAI API Key
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize Gemini API
google_api_key = os.environ.get("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)

# ---------------------------------------------------------
# 1. Load Dataset & Queries (Same as before)
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
# 2. Logic to Assign User Utterance (Same as before)
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

    # [CASE 2] medium
    elif pref_type == "medium":
        api_calls = example.get("api_calls", [])
        easy_domain_list=[]
        if isinstance(api_calls, list):
            for call_str in api_calls:
                # --- Extract Domain & Args ---
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
# 3. Helpers for History String Construction (Same as before)
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
# 4. Build Input Prompt (Same as before)
# ---------------------------------------------------------
def build_input_prompt(example: Dict[str, Any], current_user_utterance: str, template: str, context_type: str) -> str:
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
# 5. Call LLM API (Async Version)
# ---------------------------------------------------------
async def call_llm_api_async(prompt: str, model_name: str, openai_client: AsyncOpenAI = None) -> str:
    # Baseline Prompt Loading
    baseline_prompt_path = "/data/minseo/personal-tool/conv_api/experiments3//new_baseline_prompt_update.txt"
    try:
        # File I/O is blocking, but usually fast. For high performance with large files, use aiofiles.
        # Here standard open is fine for small prompts.
        with open(baseline_prompt_path, "r", encoding="utf-8") as f:
            baseline_prompt = f.read()
    except FileNotFoundError:
        baseline_prompt = "You are a helpful assistant."

    try:
        # --- GEMINI LOGIC ---
        if "gemini" in model_name.lower():
            if not google_api_key:
                return "API_KEY_MISSING_GOOGLE"
            
            # Configure Gemini
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=baseline_prompt
            )
            
            # Use async generation method if available, else wrap sync in thread
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0
                )
            )
            return response.text.strip()

        # --- GPT LOGIC ---
        else:
            if not openai_client:
                return "API_KEY_MISSING_OPENAI"
            
            response = await openai_client.chat.completions.create(
                model=model_name, 
                messages=[
                    {"role": "system", "content": baseline_prompt},
                    {"role": "user", "content": prompt},
                ],
                # temperature=0.0,
            )
            output = response.choices[0].message.content.strip()
            return output

    except Exception as e:
        print(f"LLM API Error ({model_name}): {e}")
        return f"API_ERROR: {str(e)}"


# ---------------------------------------------------------
# 6. Pipeline (Async Version)
# ---------------------------------------------------------
async def process_single_item(
    idx: int,
    original_ex: Dict[str, Any],
    utterance: str, 
    ground_truth: str,
    sub_idx: int,
    model_name: str,
    prompt_template: str,
    context_type: str,
    prompt_type_name: str,
    pref_type: str,
    openai_client: AsyncOpenAI,
    log_path: str,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    pbar: tqdm.tqdm
):
    async with semaphore:
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

        # Call LLM
        llm_output = await call_llm_api_async(prompt, model_name, openai_client)

        # Log (Use Lock for file writing)
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
        
        async with file_lock:
            dirpath = os.path.dirname(log_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        # Update Result
        current_ex["llm_output"] = llm_output
        pbar.update(1)
        return current_ex

async def process_with_llm_async(
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
    concurrency: int = 10  # Limit concurrent requests
):
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)
    
    # Initialize Async Client
    openai_client = None
    if openai_api_key:
        openai_client = AsyncOpenAI(api_key=openai_api_key)

    skipped_count = 0
    
    print(f"Starting ASYNC process... (Model: {model_name}, Prompt: {prompt_type_name}, Context: {context_type}, Pref: {pref_type})")

    tasks = []
    
    # Prepare Tasks (Pre-processing part is still synchronous because it's CPU bound and fast enough)
    # We iterate first to build the task list
    
    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    # Lock for logging
    file_lock = asyncio.Lock()
    
    # We need to know total tasks for progress bar
    # Since one row can generate multiple tasks, we first collect data
    
    print("Preparing tasks...")
    prepared_items = []
    
    for _, row in df.iterrows():
        original_ex = row.to_dict()

        # [Filtering]
        if pref_type == "easy":
            if not original_ex.get("api_calls"):
                skipped_count += 1; continue
        elif pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"):
                skipped_count += 1; continue

        # [Assignment]
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
            prepared_items.append({
                "original_ex": original_ex,
                "utterance": utterance,
                "ground_truth": ground_truth,
                "sub_idx": sub_idx
            })

    total_tasks = len(prepared_items)
    print(f"Total tasks prepared: {total_tasks}. Skipped source examples: {skipped_count}")

    # Create Progress Bar
    pbar = tqdm.tqdm(total=total_tasks, desc="Processing Async")

    for item in prepared_items:
        task = asyncio.create_task(
            process_single_item(
                idx=0, # Not strictly needed in async flow
                original_ex=item['original_ex'],
                utterance=item['utterance'],
                ground_truth=item['ground_truth'],
                sub_idx=item['sub_idx'],
                model_name=model_name,
                prompt_template=prompt_template,
                context_type=context_type,
                prompt_type_name=prompt_type_name,
                pref_type=pref_type,
                openai_client=openai_client,
                log_path=log_path,
                semaphore=semaphore,
                file_lock=file_lock,
                pbar=pbar
            )
        )
        tasks.append(task)
    
    # Run all tasks
    processed_data = await asyncio.gather(*tasks)
    
    pbar.close()

    # Save Results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved -> {output_path}")


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

    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt-4o-mini-2024-07-18"
    )
    
    # Added concurrency argument
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of concurrent API requests."
    )

    args = parser.parse_args()

    if args.prompt_type == "imp-zs":
        selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-fs":
        selected_template = IMPLICIT_FS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-pref-group":
        selected_template = IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE
    else:
        selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE

    # Run Async Loop
    asyncio.run(
        process_with_llm_async(
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
            concurrency=args.concurrency
        )
    )