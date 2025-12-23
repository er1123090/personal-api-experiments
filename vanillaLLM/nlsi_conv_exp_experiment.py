import tqdm
import os
import json 
import argparse
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from openai import OpenAI
from prompt import EXPLICIT_ZS_PROMPT_TEMPLATE, EXPLICIT_FS_PROMPT_TEMPLATE

# OpenAI API 초기화
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
def load_chains_dataset(fpath: str) -> pd.DataFrame:
    try:
        # JSON Lines 형태일 경우
        df = pd.read_json(fpath, lines=True)
        return df
    except ValueError:
        # 일반 JSON List 형태일 경우
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)


# ---------------------------------------------------------
# 2. Build Dialogue History (데이터 구조에 맞게 수정됨)
# ---------------------------------------------------------
def build_dialogue_history(standing_instructions: List[Dict[str, Any]], include_api_call: bool = False) -> str:
    """
    all_standing_instructions 리스트를 순회하며 Session 기록을 생성합니다.
    """
    sessions_str = []

    # standing_instructions 리스트가 곧 각 Session에 해당함
    for idx, instruction_data in enumerate(standing_instructions, start=1):
        # 1. generated_dialogue 추출
        turns = instruction_data.get("generated_dialogue", [])
        lines = [f"[Session {idx}]"]
        
        for turn in turns:
            # Role 처리 (User/Assistant 대소문자 통일)
            role = turn.get("role", "").capitalize() 
            
            # Content 처리 ('message'와 'content' 키가 혼용되어 있어 둘 다 체크)
            content = turn.get("message") or turn.get("content") or ""
            
            if role and content:
                lines.append(f"{role}: {content}")
        
        # 2. API Call 옵션 처리
        if include_api_call:
            api_call = instruction_data.get("api_call")
            if api_call:
                lines.append(f"API_CALL: {api_call}")

        sessions_str.append("\n".join(lines))

    return "\n\n".join(sessions_str)


# ---------------------------------------------------------
# 3. Build prompt (데이터 키 변경 적용)
# ---------------------------------------------------------
def build_input_prompt(example: Dict[str, Any], template: str, include_api_call: bool = False) -> str:
    """
    template: 선택된 프롬프트 템플릿 문자열 (ZS 또는 FS)
    """
    # [수정] 데이터셋의 키가 'all_standing_instructions' 임을 반영
    history_data = example.get("all_standing_instructions", [])
    
    dialogue_history = build_dialogue_history(history_data, include_api_call=include_api_call)
    user_utt = example.get("user_utterance", "").strip()

    # 템플릿 포맷팅
    prompt = template.format(
        dialogue_history=dialogue_history,
        user_utterance=user_utt,
    )
    return prompt


# ---------------------------------------------------------
# 4. Call GPT API (기존 동일)
# ---------------------------------------------------------
def call_gpt_api(prompt: str) -> str:
    # baseline_prompt 경로가 실제 환경에 맞는지 확인 필요
    baseline_prompt_path = "/data/minseo/personal-tool/conv_api/experiments/new_baseline_prompt.txt"
    
    # 파일이 없을 경우를 대비한 간단한 에러 처리 권장 (선택사항)
    if not os.path.exists(baseline_prompt_path):
         # 파일이 없으면 기본 시스템 프롬프트 사용 (혹은 에러 발생)
         baseline_prompt = "You are a helpful assistant."
    else:
        with open(baseline_prompt_path, "r", encoding="utf-8") as f:
            baseline_prompt = f.read()

    response = client.responses.create(
        model="gpt-4o-mini-2024-07-18", 
        input=[
            {"role": "system", "content": baseline_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    #print(response)
    output = response.output[0].content[0].text.strip()
    return output


# ---------------------------------------------------------
# 5. Logging (기존 동일)
# ---------------------------------------------------------
def write_log(log_path: str, record: Dict[str, Any]):
    dirpath = os.path.dirname(log_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# 6. Pipeline (기존 동일)
# ---------------------------------------------------------
def process_with_gpt(
    input_path: str,
    output_path: str,
    log_path: str,
    prompt_template: str,
    prompt_type_name: str,
    include_api_call: bool = False
):
    df = load_chains_dataset(input_path)
    processed_data = []

    print(f"Starting process... (Type: {prompt_type_name}, Include API Call: {include_api_call})")

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing examples"):
        ex = row.to_dict()

        # 수정된 build_input_prompt 호출
        prompt = build_input_prompt(ex, template=prompt_template, include_api_call=include_api_call)

        try:
            gpt_output = call_gpt_api(prompt)
        except Exception as e:
            print(f"API Error on example {ex.get('example_id')}: {e}")
            gpt_output = "ERROR"

        log_record = {
            "timestamp": datetime.now().isoformat(),
            "example_id": ex.get("example_id"),
            "prompt_type": prompt_type_name,
            "include_api_call": include_api_call,
            "model_input": prompt,
            "model_output": gpt_output,
        }
        write_log(log_path, log_record)

        ex["gpt_output"] = gpt_output
        processed_data.append(ex)

    # 출력 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print(f"Processing completed. Saved -> {output_path}")
    print(f"Log saved -> {log_path}")


# ---------------------------------------------------------
# 7. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT inference with customizable prompts.")
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/nlsi2exp2/output/data/50sampled/nlsi_test_simple_50sampled_251202-step1-2-step2-1.json")
    parser.add_argument("--output_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/exp/simple/nlsi_test_simple_50sampled_251202-step1-2-step2-1/251208-gpt4omini-noapi-zs.json")
    parser.add_argument("--log_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/exp/simple/nlsi_test_simple_50sampled_251202-step1-2-step2-1/251208-gpt4omini-noapi-zs.jsonl")
    
    parser.add_argument("--api_call", action="store_true", help="Include past API calls in history.")
    parser.add_argument("--prompt_type", type=str, choices=["zs", "fs"], default="zs", help="Select prompt type: 'zs' (Zero-Shot) or 'fs' (Few-Shot).")

    args = parser.parse_args()

    if args.prompt_type == "zs":
        selected_template = EXPLICIT_ZS_PROMPT_TEMPLATE
    else: 
        selected_template = EXPLICIT_FS_PROMPT_TEMPLATE

    process_with_gpt(
        input_path=args.input_path,
        output_path=args.output_path,
        log_path=args.log_path,
        prompt_template=selected_template,
        prompt_type_name=args.prompt_type,
        include_api_call=args.api_call
    )