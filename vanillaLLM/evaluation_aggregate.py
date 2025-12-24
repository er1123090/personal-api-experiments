import os
import json
import argparse
import pandas as pd
import re
from collections import Counter
from tqdm import tqdm

# ==============================================================================
# 1. User Provided Evaluation Logic (Helper Functions)
# ==============================================================================
def normalize_value(value):
    if value is None: return ""
    return str(value).strip()

def parse_api_string(api_string):
    if not api_string: return []
    parsed_calls = []
    # 1. Regex Parsing
    calls = re.findall(r'(\w+)\((.*?)\)', api_string)
    if calls:
        for func_name, args_str in calls:
            slot_dict = {}
            slots = re.findall(r'(\w+)\s*=\s*(?:["\'](.*?)["\']|([^,\s)\]\'"]+))', args_str)
            for key, val_quoted, val_unquoted in slots:
                raw_val = val_quoted if val_quoted else val_unquoted
                slot_dict[key] = normalize_value(raw_val)
            parsed_calls.append((func_name, slot_dict))
        return parsed_calls
    # 2. JSON Parsing Fallback
    try:
        clean_str = api_string.strip()
        if clean_str.startswith("```"):
            lines = clean_str.splitlines()
            if len(lines) >= 3: clean_str = "\n".join(lines[1:-1])
        data = json.loads(clean_str)
        if isinstance(data, dict):
            slot_dict = {str(k): normalize_value(v) for k, v in data.items()}
            parsed_calls.append(("__JSON_PARSED__", slot_dict))
    except: pass
    return parsed_calls

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def filter_triples_by_preference(triples, preference_map):
    if not preference_map: return []
    filtered = []
    for func, slot, val in triples:
        if func in preference_map and slot in preference_map[func]:
            filtered.append((func, slot, val))
    return filtered

def evaluate_single_file(file_path, preference_map=None):
    """
    단일 JSON 파일을 로드하여 평가 메트릭을 반환합니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Counters
    d_tp, d_fp, d_fn = 0, 0, 0
    d_correct = 0
    s_tp, s_fp, s_fn = 0, 0, 0
    ps_tp, ps_fp, ps_fn = 0, 0, 0
    valid_samples = 0

    for item in data:
        gt_str = str(item.get("reference_ground_truth", "") or "").strip()
        pred_str = str(item.get("llm_output", "") or "").strip()

        if not gt_str: continue
        valid_samples += 1
        
        gt_parsed = parse_api_string(gt_str)
        pred_parsed = parse_api_string(pred_str)

        # JSON Heuristic Match
        if len(pred_parsed) == 1 and pred_parsed[0][0] == "__JSON_PARSED__":
            if len(gt_parsed) == 1:
                pred_parsed = [(gt_parsed[0][0], pred_parsed[0][1])]

        # 1. Domain
        gt_domains = Counter([x[0] for x in gt_parsed])
        pred_domains = Counter([x[0] for x in pred_parsed])
        
        d_tp += sum((gt_domains & pred_domains).values())
        d_fp += sum((pred_domains - gt_domains).values())
        d_fn += sum((gt_domains - pred_domains).values())
        if gt_domains == pred_domains: d_correct += 1

        # 2. All Slots
        gt_triples = [(f, k, v) for f, s in gt_parsed for k, v in s.items()]
        pred_triples = [(f, k, v) for f, s in pred_parsed for k, v in s.items()]
        
        gt_slots_cnt = Counter(gt_triples)
        pred_slots_cnt = Counter(pred_triples)
        
        s_tp += sum((gt_slots_cnt & pred_slots_cnt).values())
        s_fp += sum((pred_slots_cnt - gt_slots_cnt).values())
        s_fn += sum((gt_slots_cnt - pred_slots_cnt).values())

        # 3. Preferred Slots
        if preference_map:
            gt_pref = filter_triples_by_preference(gt_triples, preference_map)
            pred_pref = filter_triples_by_preference(pred_triples, preference_map)
            gt_pref_cnt = Counter(gt_pref)
            pred_pref_cnt = Counter(pred_pref)
            
            ps_tp += sum((gt_pref_cnt & pred_pref_cnt).values())
            ps_fp += sum((pred_pref_cnt - gt_pref_cnt).values())
            ps_fn += sum((gt_pref_cnt - pred_pref_cnt).values())

    # Calculate
    d_prec, d_rec, d_f1 = calculate_metrics(d_tp, d_fp, d_fn)
    d_acc = d_correct / valid_samples if valid_samples > 0 else 0
    s_prec, s_rec, s_f1 = calculate_metrics(s_tp, s_fp, s_fn)
    
    ps_prec, ps_rec, ps_f1 = 0, 0, 0
    if preference_map:
        ps_prec, ps_rec, ps_f1 = calculate_metrics(ps_tp, ps_fp, ps_fn)

    return {
        "valid_samples": valid_samples,
        "domain_acc": d_acc,
        "domain_f1": d_f1,
        "slot_prec": s_prec,
        "slot_rec": s_rec,
        "slot_f1": s_f1,
        "pref_slot_prec": ps_prec,
        "pref_slot_rec": ps_rec,
        "pref_slot_f1": ps_f1
    }

# ==============================================================================
# 2. Aggregation Logic
# ==============================================================================
def aggregate_results(root_dir, pref_file, output_csv):
    print(f"Loading Preference File: {pref_file}")
    preference_map = None
    if os.path.exists(pref_file):
        with open(pref_file, 'r', encoding='utf-8') as f:
            preference_map = json.load(f)
    else:
        print("Warning: Preference file not found!")

    results_list = []
    
    # os.walk로 모든 하위 디렉토리 탐색
    print(f"Scanning directory: {root_dir}")
    files_found = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json") or file.endswith(".jsonl"):
                files_found.append(os.path.join(root, file))

    print(f"Found {len(files_found)} JSON files. Starting evaluation...")

    for file_path in tqdm(files_found):
        # 1. Metadata Parsing from Path
        # Expected structure: .../output/<context_type>/<pref_type>/<filename>.json
        path_parts = file_path.split(os.sep)
        
        try:
            # 경로 끝에서부터 역추적 (유연성 확보)
            filename = path_parts[-1]
            pref_type = path_parts[-2]     # e.g., easy, medium, hard
            context_type = path_parts[-3]  # e.g., memory_only, diag_only
            
            # Experiment Group (ours_memory, mem0, vanillaLLM) 찾기
            exp_group = "unknown"
            if "ours_memory" in path_parts: exp_group = "ours_memory"
            elif "mem0" in path_parts: exp_group = "mem0"
            elif "vanillaLLM" in path_parts: exp_group = "vanillaLLM"
            
        except IndexError:
            pref_type = "unknown"
            context_type = "unknown"
            exp_group = "unknown"

        # 2. Run Evaluation
        metrics = evaluate_single_file(file_path, preference_map)
        
        if metrics:
            row = {
                "experiment_group": exp_group,
                "context_type": context_type,
                "pref_type": pref_type,
                "filename": filename,
                "valid_samples": metrics["valid_samples"],
                # Domain
                "domain_acc": round(metrics["domain_acc"], 4),
                # General Slots
                "slot_f1": round(metrics["slot_f1"], 4),
                "slot_prec": round(metrics["slot_prec"], 4),
                "slot_rec": round(metrics["slot_rec"], 4),
                # Preference Slots (Main Metric)
                "pref_slot_f1": round(metrics["pref_slot_f1"], 4),
                "pref_slot_prec": round(metrics["pref_slot_prec"], 4),
                "pref_slot_rec": round(metrics["pref_slot_rec"], 4),
            }
            results_list.append(row)

    # 3. Save to CSV
    if results_list:
        df = pd.DataFrame(results_list)
        # 보기 좋게 정렬 (Experiment -> Context -> Pref)
        df = df.sort_values(by=["experiment_group", "context_type", "pref_type"])
        
        output_dir = os.path.dirname(output_csv)
        print(output_dir)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)        
        
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\nSuccessfully saved aggregated results to: {output_csv}")
        print(df[["experiment_group", "context_type", "pref_type", "pref_slot_f1"]].to_string())
    else:
        print("No results found or processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing output JSON files")
    parser.add_argument("--pref_file", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/pref_list.json")
    parser.add_argument("--output_csv", type=str, default="aggregated_results.csv")
    
    args = parser.parse_args()
    aggregate_results(args.root_dir, args.pref_file, args.output_csv)