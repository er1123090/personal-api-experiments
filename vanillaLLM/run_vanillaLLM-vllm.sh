#!/bin/bash

# ==============================================================================
# 1. 스크립트 및 데이터 경로 설정
# ==============================================================================
PYTHON_SCRIPT="/data/minseo/personal-tool/conv_api/experiments3/vanillaLLM/vanillaLLM_inference-vllm.py"

INPUT_PATH="/data/minseo/personal-tool/conv_api/experiments3/data/dev_4.json"
QUERY_PATH="/data/minseo/personal-tool/conv_api/experiments3/temp_queries.json"
PREF_LIST_PATH="/data/minseo/personal-tool/conv_api/experiments3/pref_list.json"
PREF_GROUP_PATH="/data/minseo/personal-tool/conv_api/experiments3/pref_group.json"

# 기본 출력/로그 디렉토리
BASE_OUTPUT_DIR="/data/minseo/personal-tool/conv_api/experiments3/vanillaLLM/inference/output"
BASE_LOG_DIR="/data/minseo/personal-tool/conv_api/experiments3/vanillaLLM/inference/logs"

# 날짜 태그 (원하는 형식 유지: MMDD)
DATE_TAG="$(date +%m%d)"

# 테스트 태그
TEST_TAG="test1"

# ==============================================================================
# 2. 실험 변수 설정
# ==============================================================================
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "meta-llama/Llama-3.1-8B-Instruct"
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    "Qwen/Qwen3-VL-8B-Instruct"
    "google/codegemma-7b-it"
    "google/gemma-3-12b-it"
)

PROMPT_TYPES=("imp-zs" "imp-pref-group")
CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("easy" "medium" "hard")

TP_SIZE=1

# ==============================================================================
# 3. 배치 실행 로직
# ==============================================================================
echo "========================================================"
echo "Batch Inference Started at $(date)"
echo "Models: ${MODELS[*]}"
echo "Prompts: ${PROMPT_TYPES[*]}"
echo "Contexts: ${CONTEXT_TYPES[*]}"
echo "Preferences: ${PREF_TYPES[*]}"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Test Tag: $TEST_TAG"
echo "========================================================"

for model in "${MODELS[@]}"; do
    MODEL_SAFE_NAME="${model//\//_}"

    for prompt_type in "${PROMPT_TYPES[@]}"; do
        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                echo ""
                echo "--------------------------------------------------------------------------------"
                echo "[RUNNING]"
                echo " Model: $model"
                echo " Prompt: $prompt_type | Context: $context | Pref: $pref"
                echo "--------------------------------------------------------------------------------"

                # =========================
                # 저장 구조 반영:
                # context/pref/model/날짜_prompt_test1.jsonl
                # =========================
                OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/model=$MODEL_SAFE_NAME"
                LOG_DIR="$BASE_LOG_DIR/$context/$pref/model=$MODEL_SAFE_NAME"

                mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

                FILENAME="${DATE_TAG}_${prompt_type}_${TEST_TAG}.jsonl"
                LOGNAME="${DATE_TAG}_${prompt_type}_${TEST_TAG}.log"

                OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
                LOG_FILE="$LOG_DIR/$LOGNAME"

                CUDA_VISIBLE_DEVICES=1 \
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
                    --log_path "$LOG_FILE" \
                    --tensor_parallel_size $TP_SIZE

                if [ $? -eq 0 ]; then
                    echo ">> [SUCCESS] Saved to: $OUTPUT_FILE"
                else
                    echo ">> [ERROR] Failed at Model: $model | Prompt: $prompt_type | Context: $context | Pref: $pref"
                fi

            done
        done
    done
done

echo ""
echo "========================================================"
echo "All Jobs Finished at $(date)"
echo "========================================================"
