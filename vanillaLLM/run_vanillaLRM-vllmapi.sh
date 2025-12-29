#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================
# [중요] 앞서 수정한 Python 스크립트가 저장된 경로
PYTHON_SCRIPT="/data/minseo/personal-api-experiments/vanillaLLM/vanillaLRM_inference-vllmapi.py"

# 데이터 및 스키마 경로
INPUT_PATH="/data/minseo/personal-api-experiments/data/dev_4.json"
QUERY_PATH="/data/minseo/personal-api-experiments/temp_queries.json"
PREF_LIST_PATH="/data/minseo/personal-api-experiments/pref_list.json"
PREF_GROUP_PATH="/data/minseo/personal-api-experiments/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/personal-api-experiments/tools_schema.json"

# 출력 디렉토리
BASE_OUTPUT_DIR="/data/minseo/personal-api-experiments/vanillaLLM/inference/LRM-output"
BASE_LOG_DIR="/data/minseo/personal-api-experiments/vanillaLLM/inference/LRM-logs"

# 태그 설정
DATE_TAG="$(date +%m%d)"
TEST_TAG="test3"

# GPU 및 서버 설정
GPU_ID=0,1,2,3
PORT=8001
VLLM_URL="http://localhost:$PORT/v1"

# [NEW] Async 요청 동시 처리 수 (Python 스크립트의 --concurrency와 매칭)
CONCURRENCY=64

# ==============================================================================
# 2. 실험 변수 (모델 목록 등)
# ==============================================================================
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
#    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 

# 주의: DeepSeek-R1-0528-Qwen3-8B 등 허브에 없는 모델명은 실제 경로/이름 확인 필요
# 위 리스트는 예시로 수정했습니다. 실제 사용하시는 모델명으로 변경하세요.

PROMPT_TYPES=("imp-zs" "imp-pref-group")
CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("easy" "medium" "hard")

TP_SIZE=4  # Tensor Parallel Size (GPU 개수에 맞춰 조정, 현재 3개 GPU 사용 중이면 2 or 1 권장)

# ==============================================================================
# 3. 헬퍼 함수: 서버 대기
# ==============================================================================
wait_for_server() {
    echo "Waiting for vLLM server to start at $VLLM_URL..."
    # 최대 300초 대기
    for i in {1..60}; do
        if curl -s "$VLLM_URL/models" > /dev/null; then
            echo ""
            echo ">> Server is READY!"
            return 0
        fi
        sleep 5
        echo -n "."
    done
    echo "!! Server failed to start in time."
    return 1
}

# ==============================================================================
# 4. 메인 루프
# ==============================================================================
echo "========================================================"
echo "Automated Batch Inference Started at $(date)"
echo "GPU: $GPU_ID | Port: $PORT | Concurrency: $CONCURRENCY"
echo "========================================================"

for model in "${MODELS[@]}"; do
    # 모델명에서 슬래시(/)를 언더바(_)로 치환하여 파일명 생성용 변수 만들기
    MODEL_SAFE_NAME="${model//\//_}"
    
    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: $model"
    echo "####################################################################"

    # 4-1. 모델별 파서 설정 (Tool Calling용)
    # DeepSeek R1 등은 tool calling 템플릿 지원 여부에 따라 설정이 달라질 수 있음.
    # Python 스크립트가 Regex 파싱도 지원하므로, vLLM이 툴을 지원하면 켜고 아니면 텍스트로 받음.
    if [[ "$model" == *"Llama-3"* ]] || [[ "$model" == *"Distill-Llama"* ]]; then
        PARSER="llama3_json"
    elif [[ "$model" == *"deepseek"* ]]; then
        # DeepSeek R1이 Tool Call API를 공식 지원하지 않는 버전일 경우 에러가 날 수 있음.
        # 안전하게 hermes나 llama3_json 시도, 혹은 툴 파서를 끌 수도 있음.
        PARSER="llama3_json" 
    elif [[ "$model" == *"Qwen"* ]]; then
        PARSER="hermes" 
    elif [[ "$model" == *"Mistral"* ]]; then
        PARSER="mistral"
    else
        PARSER="hermes"
    fi

    # 4-2. vLLM 서버 백그라운드 실행
    # --gpu-memory-utilization 0.9: OOM 방지
    # --disable-log-requests: 터미널 로그 과부하 방지
    CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$model" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --enable-auto-tool-choice \
        --tool-call-parser $PARSER \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.9 \
        --enforce-eager \
        --disable-log-requests \
        --trust-remote-code > vllm_server.log 2>&1 &
        #8192
        
    SERVER_PID=$!
    echo ">> Server PID: $SERVER_PID"

    # 4-3. 서버 준비 대기
    if ! wait_for_server; then
        echo "Skipping model $model due to server failure."
        kill $SERVER_PID 2>/dev/null
        continue
    fi

    # 4-4. 추론 스크립트 실행 (Client)
    echo "[STEP 2] Running Python Client Scripts..."
    
    for prompt_type in "${PROMPT_TYPES[@]}"; do
        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                echo "   >> Running: Prompt=$prompt_type | Pref=$pref | Model=$MODEL_SAFE_NAME"

                # 출력/로그 경로 생성
                OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/$MODEL_SAFE_NAME/$prompt_type"
                LOG_DIR="$BASE_LOG_DIR/$context/$pref/$MODEL_SAFE_NAME/$prompt_type" # 경로 슬래시 수정
                mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

                FILENAME="${DATE_TAG}_${TEST_TAG}.json"
                LOGNAME="${DATE_TAG}_${TEST_TAG}.jsonl"
                
                OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
                LOG_FILE="$LOG_DIR/$LOGNAME"

                # Python 스크립트 실행 (GPU 불필요, CPU만 사용해도 됨)
                # [NEW] --concurrency 인자 추가됨
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
                    --concurrency "$CONCURRENCY"

            done
        done
    done

    echo "[STEP 3] Stopping vLLM Server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    echo ">> Server Stopped."
    echo ""
    
    # 포트 정리 및 GPU 메모리 해제 대기
    sleep 15

done

echo "========================================================"
echo "All Jobs Finished at $(date)"
echo "========================================================"