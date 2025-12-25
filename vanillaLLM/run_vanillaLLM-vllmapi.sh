#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================
# Python 클라이언트 스크립트 (수정된 버전)
PYTHON_SCRIPT="/data/minseo/personal-api-experiments/vanillaLLM/vanillaLLM_inference-vllmapi.py"

# 데이터 및 스키마 경로
INPUT_PATH="/data/minseo/personal-api-experiments/data/dev_4.json"
QUERY_PATH="/data/minseo/personal-api-experiments/temp_queries.json"
PREF_LIST_PATH="/data/minseo/personal-api-experiments/pref_list.json"
PREF_GROUP_PATH="/data/minseo/personal-api-experiments/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/personal-api-experiments/tools_schema.json"

# 출력 디렉토리
BASE_OUTPUT_DIR="/data/minseo/personal-api-experiments/vanillaLLM/inference/output"
BASE_LOG_DIR="/data/minseo/personal-api-experiments/vanillaLLM/inference/logs"

# 태그 설정
DATE_TAG="$(date +%m%d)"
TEST_TAG="test1"

# GPU 설정 (서버가 사용할 GPU ID)
GPU_ID=1,2,3
PORT=8001
VLLM_URL="http://localhost:$PORT/v1"

# ==============================================================================
# 2. 실험 변수 (모델 목록 등)
# ==============================================================================
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "meta-llama/Llama-3.1-8B-Instruct"
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    "Qwen/Qwen3-VL-8B-Instruct"
    "google/gemma-3-12b-it"
    "google/codegemma-7b-it"

)

PROMPT_TYPES=("imp-zs" "imp-pref-group")
CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("easy" "medium" "hard")

TP_SIZE=2

# ==============================================================================
# 3. 헬퍼 함수: 서버 대기
# ==============================================================================
wait_for_server() {
    echo "Waiting for vLLM server to start at $VLLM_URL..."
    while ! curl -s "$VLLM_URL/models" > /dev/null; do
        sleep 5
        echo -n "."
    done
    echo ""
    echo ">> Server is READY!"
}

# ==============================================================================
# 4. 메인 루프
# ==============================================================================
echo "========================================================"
echo "Automated Batch Inference Started at $(date)"
echo "GPU: $GPU_ID | Port: $PORT"
echo "========================================================"

for model in "${MODELS[@]}"; do
    MODEL_SAFE_NAME="${model//\//_}"
    
    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: $model"
    echo "####################################################################"

    # 4-1. 모델별 파서 설정 (Tool Calling용)
    # 모델에 따라 파서가 다를 수 있으므로 분기 처리 (기본값: llama3_json)
    if [[ "$model" == *"Llama-3"* ]]; then
        PARSER="llama3_json"
    elif [[ "$model" == *"deepseek"* ]]; then
        PARSER="llama3_json"
    elif [[ "$model" == *"Mistral"* ]]; then
        PARSER="mistral"
    elif [[ "$model" == *"Qwen"* ]]; then
        PARSER="hermes"
    else
        # 대부분의 최신 모델이나 Llama 기반은 이걸로 시도
        PARSER="hermes"
    fi

    # 4-2. vLLM 서버 백그라운드 실행
    # --max-model-len: 컨텍스트 길이 에러 방지용 (필요시 조절)
    CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$model" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --enable-auto-tool-choice \
        --tool-call-parser $PARSER \
        --max-model-len 8192 \
        --enforce-eager \
        --trust-remote-code > vllm_server.log 2>&1 &
        #16384
    SERVER_PID=$!
    echo ">> Server PID: $SERVER_PID"

    # 4-3. 서버 준비 대기
    wait_for_server

    # 4-4. 추론 스크립트 실행 (Client)
    echo "[STEP 2] Running Python Client Scripts..."
    
    for prompt_type in "${PROMPT_TYPES[@]}"; do
        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                echo "   >> Running: Prompt=$prompt_type | Pref=$pref"

                OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/$MODEL_SAFE_NAME/$prompt_type"
                LOG_DIR="$BASE_LOG_DIR$context/$pref/$MODEL_SAFE_NAME/$prompt_type"
                mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

                FILENAME="${DATE_TAG}_${TEST_TAG}.json"
                LOGNAME="${DATE_TAG}_${TEST_TAG}.jsonl"
                OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
                LOG_FILE="$LOG_DIR/$LOGNAME"

                # Python 클라이언트는 GPU가 필요 없으므로 CUDA_VISIBLE_DEVICES 비움 (혹은 CPU만 사용)
                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --query_path "$QUERY_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --prompt_type "$prompt_type" \
                    --model_name "$model" \
                    --output_path "$OUTPUT_FILE" \
                    --log_path "$LOG_FILE" \
                    --vllm_url "$VLLM_URL" \
                    

            done
        done
    done

    echo "[STEP 3] Stopping vLLM Server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    echo ">> Server Stopped."
    echo ""
    
    # 포트 정리 대기 (잠시 휴식)
    sleep 10

done

echo "========================================================"
echo "All Jobs Finished at $(date)"
echo "========================================================"