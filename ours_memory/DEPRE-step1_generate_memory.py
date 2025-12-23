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
from prompt import RECURSIVE_MEMORY_UPDATE_PROMPT_V1, RECURSIVE_MEMORY_UPDATE_PROMPT_V2, RECURSIVE_MEMORY_UPDATE_PROMPT_V3

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------
# Utility: Pre-check Logic
# ---------------------------------------------------------
def check_if_testable(pref_list_path, example, query_map):
    # (기존 코드와 동일하여 생략 가능하나, 실행을 위해 유지)
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
    def __init__(self, client, model="gpt-4o-mini-2024-07-18"):
        self.client = client
        self.model = model
        self.explicit_pref = "None"
        self.implicit_pref = "None"
        self.accumulated_api_list = []
        self.t = 0

    def f_reason(self, h_t: str, s_t: str):
        self.t += 1
        formatted_prompt = RECURSIVE_MEMORY_UPDATE_PROMPT_V3.format(
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

    # 기존 파일이 있다면 초기화 (선택 사항)
    if os.path.exists(args.memory_output_path):
        os.remove(args.memory_output_path)
    
    generated_count = 0
    skipped_count = 0

    # 파일을 열어두고 루프 시작 (JSONL 작성을 위해)
    with open(args.memory_output_path, "w", encoding="utf-8") as f_out:
        
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Gen Memory"):
            example = row.to_dict()
            ex_id = str(example.get("example_id")) 
            
            # 1. Pre-check
            is_useful = check_if_testable(args.pref_list_path, example, query_map)
            if not is_useful:
                skipped_count += 1
                continue 

            # 2. Generate Memory
            generated_count += 1
            memory_sys = RecursiveMemorySystem(client)
            
            history_sessions = example.get("all_standing_instructions", [])
            api_sessions = example.get("api_calls_all", [])
            
            # [변경점] 세션별 변화를 기록할 리스트
            session_history_log = []
            
            for t_idx in range(len(history_sessions)):
                # Dialogue Extract
                turns = history_sessions[t_idx].get("generated_dialogue", [])
                h_t = "\n".join([f"{t.get('role','').capitalize()}: {t.get('message','')}" for t in turns])
                
                # API Extract
                if t_idx < len(api_sessions):
                    raw_apis = api_sessions[t_idx].get("api_call", [])
                    s_t = ", ".join(raw_apis) if isinstance(raw_apis, list) else str(raw_apis)
                else:
                    s_t = ""
                
                # Update Memory
                memory_sys.f_reason(h_t, s_t)
                
                # [핵심] 현재 세션 업데이트 직후의 메모리 상태 스냅샷 저장
                # 주의: accumulated_api_list는 mutable이므로 반드시 copy해야 함
                snapshot = {
                    "session_idx": t_idx + 1,
                    "explicit_pref": memory_sys.explicit_pref,
                    "implicit_pref": memory_sys.implicit_pref,
                    # 리스트의 현재 상태를 복사해서 저장 ([])
                    "accumulated_api_list": list(memory_sys.accumulated_api_list) 
                }
                session_history_log.append(snapshot)
            
            # 3. Save to JSONL
            # 한 Dialogue의 처리가 끝나면, 그 동안의 기록(session_history_log)을 포함해 저장
            final_record = {
                "example_id": ex_id,
                # 최종 상태 (Evaluate에서 바로 쓰기 편하게)
                "final_explicit_pref": memory_sys.explicit_pref,
                "final_implicit_pref": memory_sys.implicit_pref,
                "final_api_list": memory_sys.accumulated_api_list,
                # 세션별 변화 기록 (분석용)
                "memory_evolution_history": session_history_log
            }
            
            f_out.write(json.dumps(final_record, ensure_ascii=False) + "\n")
            f_out.flush() # 안전하게 바로 디스크에 쓰기

    print(f"Generation Summary:")
    print(f" - Generated: {generated_count}")
    print(f" - Skipped: {skipped_count}")
    print(f"Saved to {args.memory_output_path} (JSONL format)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/data/sgd_converted_dev_mapped_grouped_with_pref_with_constraints.json")
    parser.add_argument("--memory_output_path", type=str, required=True, help="Path to save .jsonl file")
    
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/pref_list.json")
    
    args = parser.parse_args()
    generate_memory_file(args)