#!/bin/bash

# ==============================================================================
# 1. 스크립트 및 데이터 경로 설정
# ==============================================================================
PYTHON_SCRIPT="/data/minseo/personal-tool/conv_api/experiments3/vanillaLLM/vanillaLLM_inference-api.py"

INPUT_PATH="/data/minseo/personal-tool/conv_api/experiments3/data/dev_4.json"
QUERY_PATH="/data/minseo/personal-tool/conv_api/experiments3/temp_queries.json"
PREF_LIST_PATH="/data/minseo/personal-tool/conv_api/experiments3/pref_list.json"
PREF_GROUP_PATH="/data/minseo/personal-tool/conv_api/experiments3/pref_group.json"

# 기본 출력/로그 디렉토리
BASE_OUTPUT_DIR="/data/minseo/personal-tool/conv_api/experiments3/vanillaLLM/inference/output"
BASE_LOG_DIR="/data/minseo/personal-tool/conv_api/experiments3/vanillaLLM/inference/logs"

# 날짜 태그
DATE_TAG="$(date +%m%d)"

# ==============================================================================
# 2. 실험 변수 설정
# ==============================================================================

# [Model List]
MODELS=("gemini-3-flash-preview") #gpt-5-mini

# [Prompt Types]
PROMPT_TYPES=("imp-zs" "imp-pref-group")

# [Context Types]
CONTEXT_TYPES=("diag-apilist")

# [Preference Types]
PREF_TYPES=("medium" "hard")  #easy"

# ==============================================================================
# 3. 배치 실행 로직
# ==============================================================================

echo "========================================================"
echo "Batch Inference Started at $(date)"
echo "Models: ${MODELS[*]}"
echo "Prompts: ${PROMPT_TYPES[*]}"
echo "========================================================"

for model in "${MODELS[@]}"; do
    for prompt_type in "${PROMPT_TYPES[@]}"; do
        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                echo ""
                echo "--------------------------------------------------------------------------------"
                echo "[RUNNING] Model: $model | Prompt: $prompt_type | Pref: $pref"
                echo "--------------------------------------------------------------------------------"

                # ------------------------------------------------------------------
                # [파일명 생성 로직]
                # reasoning 인자가 Python에 없으므로 파일명에서도 제외하거나 고정값 사용
                # ------------------------------------------------------------------
                FILENAME="${DATE_TAG}_${prompt_type}_${model}_test1.json"
                
                # 로그 파일명
                LOGNAME="${FILENAME}l"

                # 경로 설정
                OUTPUT_FILE="$BASE_OUTPUT_DIR/$context/$pref/$FILENAME"
                LOG_FILE="$BASE_LOG_DIR/$context/$pref/$LOGNAME"

                # 파이썬 스크립트 실행 (Python 코드에 있는 인자만 전달)
                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --query_path "$QUERY_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --prompt_type "$prompt_type" \
                    --model_name "$model" \
                    --output_path "$OUTPUT_FILE" \
                    --log_path "$LOG_FILE"

                # 실행 결과 확인
                if [ $? -eq 0 ]; then
                    echo ">> [SUCCESS] Saved to: $OUTPUT_FILE"
                else
                    echo ">> [ERROR] Failed at Model: $model"
                fi

            done
        done
    done
done

echo ""
echo "========================================================"
echo "All Jobs Finished at $(date)"
echo "========================================================"