#!/bin/bash

# ==============================================================================
# 설정 (환경에 맞게 수정하세요)
# ==============================================================================

# 1. 파이썬 스크립트 경로 (위의 코드를 저장한 파일명)
PYTHON_SCRIPT="/data/minseo/personal-tool/conv_api/experiments2/vanillaLLM/nlsi_conv_imp_experiment_temp-HARD.py"

# 2. 입력 데이터 경로
INPUT_PATH="/data/minseo/personal-tool/conv_api/experiments2/data/dev_3.json"
QUERY_PATH="/data/minseo/personal-tool/conv_api/experiments2/temp_queries.json"
PREF_LIST_PATH="/data/minseo/personal-tool/conv_api/experiments2/pref_list.json"
PREF_GROUP_PATH="/data/minseo/personal-tool/conv_api/experiments2/vanillaLLM/pref_group.json"

# 3. 출력 및 로그 기본 경로
BASE_OUTPUT_DIR="/data/minseo/personal-tool/conv_api/experiments2/vanillaLLM/inference/output"
BASE_LOG_DIR="/data/minseo/personal-tool/conv_api/experiments2/vanillaLLM/inference/logs"

# 4. 파일명 태그 (날짜/모델명 등)
FILE_TAG="1223_imp-zs_gpt4omini_test1"

# ==============================================================================
# 반복 실행 로직
# ==============================================================================

# 실행할 Variation 정의
# 코드의 argparse choices에 맞춤: ["diag-only", "apilist-only", "diag-apilist"]
CONTEXT_TYPES=("diag-only" "apilist-only" "diag-apilist")

# 코드의 argparse choices에 맞춤: ["easy", "medium", "hard"]
PREF_TYPES=("easy" "medium" "hard")

echo "========================================================"
echo "Batch Inference Started at $(date)"
echo "========================================================"

for context in "${CONTEXT_TYPES[@]}"; do
    for pref in "${PREF_TYPES[@]}"; do
        
        echo ""
        echo "[Processing] Context: $context | Pref: $pref"

        # 출력 파일 경로 구성 (예: .../output/diag-only/easy/filename.json)
        OUTPUT_FILE="$BASE_OUTPUT_DIR/$context/$pref/${FILE_TAG}.json"
        
        # 로그 파일 경로 구성 (예: .../logs/diag-only/easy/filename.jsonl)
        LOG_FILE="$BASE_LOG_DIR/$context/$pref/${FILE_TAG}.jsonl"

        # 파이썬 스크립트 실행
        python "$PYTHON_SCRIPT" \
            --input_path "$INPUT_PATH" \
            --query_path "$QUERY_PATH" \
            --pref_list_path "$PREF_LIST_PATH" \
            --pref_group_path "$PREF_GROUP_PATH" \
            --context_type "$context" \
            --pref_type "$pref" \
            --output_path "$OUTPUT_FILE" \
            --log_path "$LOG_FILE" \
            --prompt_type "imp-zs"

        # 실행 결과 확인
        if [ $? -eq 0 ]; then
            echo ">> Success! Output saved to: $OUTPUT_FILE"
        else
            echo ">> Error occurred in $context / $pref"
        fi

    done
done

echo ""
echo "========================================================"
echo "All Jobs Finished at $(date)"
echo "========================================================"