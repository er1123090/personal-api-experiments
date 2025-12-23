import json
import re
from typing import Dict, Tuple, Optional

# 기존: 함수 호출 형태 정규식 (GetRestaurants(...))
API_CALL_REGEX = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)", re.DOTALL)

# 신규: Markdown JSON 코드 블록 정규식
JSON_BLOCK_REGEX = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def parse_model_output(text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    텍스트에서 API 호출 정보를 추출하여 (func_name, args_dict) 형태로 반환.
    우선순위:
    1. Markdown JSON 블록 (예: ```json { "action": ... } ```)
    2. 일반 함수 호출 텍스트 (예: GetRestaurants(...), `GetRestaurants(...)`)
    """
    if not text:
        return None
    
    cleaned_text = text.strip()

    # ---------------------------------------------------------
    # 전략 1: JSON 포맷 파싱
    # ---------------------------------------------------------
    json_match = JSON_BLOCK_REGEX.search(cleaned_text)
    if json_match:
        try:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            
            func_name = data.get("action")
            params = data.get("parameters", {})

            if func_name:
                args: Dict[str, str] = {}
                for k, v in params.items():
                    args[k] = str(v)
                return func_name, args
        except json.JSONDecodeError:
            pass

    # ---------------------------------------------------------
    # 전략 2: 함수 호출 문자열 파싱 (Text-based)
    # ---------------------------------------------------------
    
    # 1. <API_CALL> 태그 제거
    cleaned_text = re.sub(r"<API_CALL>", "", cleaned_text, flags=re.IGNORECASE).strip()

    # 2. 양끝 따옴표 제거 (문자열인 경우)
    if (cleaned_text.startswith('"') and cleaned_text.endswith('"')) or (
        cleaned_text.startswith("'") and cleaned_text.endswith("'")
    ):
        cleaned_text = cleaned_text[1:-1].strip()

    # 3. [NEW] 양끝 백틱(`) 제거 (Markdown 인라인 코드 대응)
    # 예: `GetEvents(...)` -> GetEvents(...)
    # 정규식: 시작 부분의 하나 이상의 ` 혹은 끝 부분의 하나 이상의 ` 제거
    cleaned_text = re.sub(r"^`+|`+$", "", cleaned_text).strip()

    m = API_CALL_REGEX.search(cleaned_text)
    if not m:
        return None

    func_name = m.group(1).strip()
    args_str = m.group(2).strip()

    if args_str == "":
        return func_name, {}

    args: Dict[str, str] = {}
    
    # 인자 파싱 (단순 , 분리)
    # 주의: value 내부에 쉼표(,)가 포함된 복잡한 경우는 별도 처리가 필요할 수 있으나,
    # 현재 데이터셋 포맷에 맞춰 기존 로직 유지
    for part in args_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()

        # 값 주변 따옴표 제거
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        args[name] = value

    return func_name, args


def evaluate_api_calls(json_path: str):
    """
    json_path: 리스트 JSON 파일 경로
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return

    total_examples = 0  
    func_correct = 0    

    arg_tp = 0
    arg_fp = 0
    arg_fn = 0

    def norm_val(v: str) -> str:
        return str(v).strip().lower()

    for ex in data:
        gold_list = ex.get("api_calls", [])
        if not gold_list:
            continue

        gold_raw = gold_list[0]
        pred_raw = ex.get("gpt_output", "")

        parsed_gold = parse_model_output(str(gold_raw))
        parsed_pred = parse_model_output(str(pred_raw))

        if parsed_gold is None:
            continue

        func_name_gold, args_gold = parsed_gold

        # Get* 함수만 평가
        if not func_name_gold.startswith("Get"):
             # 필요시 조건 완화
             pass 

        total_examples += 1

        if parsed_pred is None:
            arg_fn += len(args_gold)
            continue

        func_name_pred, args_pred = parsed_pred

        if func_name_gold == func_name_pred:
            func_correct += 1

        # Arguments Evaluation
        for key, gold_val in args_gold.items():
            if key in args_pred:
                pred_val = args_pred[key]
                if norm_val(gold_val) == norm_val(pred_val):
                    arg_tp += 1
                else:
                    arg_fn += 1
                    arg_fp += 1
            else:
                arg_fn += 1

        for key in args_pred.keys():
            if key not in args_gold:
                arg_fp += 1

    if total_examples == 0:
        print("No matching API calls found to evaluate.")
        return

    func_acc = func_correct / total_examples

    if arg_tp + arg_fp == 0:
        precision = 0.0
    else:
        precision = arg_tp / (arg_tp + arg_fp)

    if arg_tp + arg_fn == 0:
        recall = 0.0
    else:
        recall = arg_tp / (arg_tp + arg_fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print(f"Total evaluated examples: {total_examples}")
    print(f"Function-name accuracy  : {func_acc:.4f} ({func_correct}/{total_examples})")
    print(f"Arguments TP/FP/FN      : TP={arg_tp}, FP={arg_fp}, FN={arg_fn}")
    print(f"Arg-level Precision     : {precision:.4f}")
    print(f"Arg-level Recall        : {recall:.4f}")
    print(f"Arg-level F1            : {f1:.4f}")


if __name__ == "__main__":
    evaluate_api_calls(
        "/data/minseo/personal-tool/conv_api/experiments/exp/conflict/nlsi_base/251208-gpt4omini-fs.json"
    )