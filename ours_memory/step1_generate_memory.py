import tqdm
import os
import json
import argparse
import pandas as pd
from openai import OpenAI
import re
import copy # 리스트 복사를 위해 필요

# ---------------------------------------------------------
# [Prompt Import]
# ---------------------------------------------------------
from prompt import (
    RECURSIVE_MEMORY_UPDATE_PROMPT_V1, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V2, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V3, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V4
)

# 프롬프트 매핑 딕셔너리 정의
PROMPT_MAP = {
    "v1": RECURSIVE_MEMORY_UPDATE_PROMPT_V1,
    "v2": RECURSIVE_MEMORY_UPDATE_PROMPT_V2,
    "v3": RECURSIVE_MEMORY_UPDATE_PROMPT_V3,
    "v4": RECURSIVE_MEMORY_UPDATE_PROMPT_V4
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------
# Utility: Pre-check Logic
# ---------------------------------------------------------
def check_if_testable(pref_list_path, example, query_map):
    # (기존 로직 유지)
    prefs = example.get("api_calls_pref", [])
    if isinstance(prefs, list) and prefs:
        for pref in prefs:
            for ev in pref.get("evidence", []):
                if ev.get("domain") in query_map:
                    return True 

    if os.path.exists(pref_list_path):
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
                
                if domain in query_map and domain in pref_list:
                    pattern = r'(\w+)=["\']([^"\']+)["\']'
                    matches = re.findall(pattern, args)
                    target_slots = pref_list.get(domain, [])
                    if any(slot in target_slots for slot, _ in matches):
                        return True
    return False

# ---------------------------------------------------------
# Memory System Class
# ---------------------------------------------------------
class RecursiveMemorySystem:
    def __init__(self, client, prompt_template, model="gpt-4o-mini-2024-07-18"):
        """
        prompt_template: 선택된 프롬프트 문자열 (V1 ~ V4)
        """
        self.client = client
        self.model = model
        self.prompt_template = prompt_template  # 선택된 프롬프트 저장
        self.explicit_pref = "None"
        self.implicit_pref = "None"
        self.accumulated_api_list = []
        self.t = 0

    def f_reason(self, h_t: str, s_t: str):
        self.t += 1
        # 저장된 self.prompt_template 사용
        formatted_prompt = self.prompt_template.format(
            prev_explicit=self.explicit_pref,
            prev_implicit=self.implicit_pref,
            h_t=h_t if h_t.strip() else "No dialogue",
            s_t=s_t if s_t.strip() else "No API calls"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            res_json = json.loads(response.choices[0].message.content)
            self.explicit_pref = res_json.get("explicit_pref", self.explicit_pref)
            self.implicit_pref = res_json.get("implicit_pref", self.implicit_pref)
            
            if s_t.strip():
                self.accumulated_api_list.append(f"[Session {self.t}] {s_t}")
        except Exception as e:
            print(f"[Error t={self.t}] {e}")

# ---------------------------------------------------------
# Main Generation Logic (JSONL & Per-Session Snapshot)
# ---------------------------------------------------------
def generate_memory_file(args):
    # 0. Prompt Selection Validation
    selected_version = args.prompt_version.lower()
    if selected_version not in PROMPT_MAP:
        raise ValueError(f"Invalid prompt version: {selected_version}. Choose from {list(PROMPT_MAP.keys())}")
    
    selected_prompt = PROMPT_MAP[selected_version]
    print(f"Using Prompt Version: {selected_version.upper()}")

    # Load Resources
    if args.input_path.endswith('.jsonl'):
        df = pd.read_json(args.input_path, lines=True)
    else:
        with open(args.input_path, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))

    with open(args.query_path, "r", encoding="utf-8") as f:
        query_map = json.load(f)

    print(f"Loaded {len(df)} dialogues.")
    print("Checking evaluability and generating memories...")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.memory_output_path), exist_ok=True)

    # 기존 파일이 있다면 초기화
    if os.path.exists(args.memory_output_path):
        os.remove(args.memory_output_path)
    
    generated_count = 0
    skipped_count = 0

    with open(args.memory_output_path, "w", encoding="utf-8") as f_out:
        
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Gen Memory ({selected_version.upper()})"):
            example = row.to_dict()
            ex_id = str(example.get("example_id")) 
            
            # 1. Pre-check
            is_useful = check_if_testable(args.pref_list_path, example, query_map)
            if not is_useful:
                skipped_count += 1
                continue 

            # 2. Generate Memory
            generated_count += 1
            
            # [변경점] 선택된 프롬프트를 주입하여 시스템 초기화
            memory_sys = RecursiveMemorySystem(client, prompt_template=selected_prompt)
            
            history_sessions = example.get("sessions", [])
            api_sessions = example.get("api_calls_all", [])
            
            session_history_log = []
            
            for t_idx in range(len(history_sessions)):
                # Dialogue Extract
                turns = history_sessions[t_idx].get("dialogue", [])
                h_t = "\n".join([f"{t.get('role','').capitalize()}: {t.get('message','')}" for t in turns])
                
                # API Extract
                if t_idx < len(api_sessions):
                    raw_apis = api_sessions[t_idx].get("api_call", [])
                    s_t = ", ".join(raw_apis) if isinstance(raw_apis, list) else str(raw_apis)
                else:
                    s_t = ""
                
                # Update Memory
                memory_sys.f_reason(h_t, s_t)
                
                snapshot = {
                    "session_idx": t_idx + 1,
                    "explicit_pref": memory_sys.explicit_pref,
                    "implicit_pref": memory_sys.implicit_pref,
                    "accumulated_api_list": list(memory_sys.accumulated_api_list) 
                }
                session_history_log.append(snapshot)
            
            # 3. Save to JSONL
            final_record = {
                "example_id": ex_id,
                "prompt_version": selected_version,  # 기록용으로 버전 명시
                "final_explicit_pref": memory_sys.explicit_pref,
                "final_implicit_pref": memory_sys.implicit_pref,
                "final_api_list": memory_sys.accumulated_api_list,
                "memory_evolution_history": session_history_log
            }
            
            f_out.write(json.dumps(final_record, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"Generation Summary:")
    print(f" - Generated: {generated_count}")
    print(f" - Skipped: {skipped_count}")
    print(f"Saved to {args.memory_output_path} (JSONL format)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments2/data/dev_3.json")
    parser.add_argument("--memory_output_path", type=str, required=True, help="Path to save .jsonl file")
    
    # [추가됨] 프롬프트 버전 선택 인자
    parser.add_argument("--prompt_version", type=str, default="v3", choices=["v1", "v2", "v3", "v4"], help="Select prompt version (v1, v2, v3, v4)")
    
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments2/ours_memory/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments2/ours_memory/pref_list.json")
    
    args = parser.parse_args()
    generate_memory_file(args)