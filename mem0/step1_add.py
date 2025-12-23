# step1_ingest.py
import tqdm
import os
import argparse
from mem0 import MemoryClient
# 위에서 만든 utils_mem0.py 파일에서 함수 임포트
from utils_mem0 import load_chains_dataset, prepare_messages_for_mem0

# Initialize mem0 Client
memory_client = MemoryClient(api_key=os.environ.get("MEM0_API_KEY"))

def run_ingestion(input_path: str):
    df = load_chains_dataset(input_path)
    print(f"Starting ingestion for {len(df)} users...")

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Ingesting Memories"):
        original_ex = row.to_dict()
        user_id = str(original_ex.get("example_id", "unknown_user"))

        # 1. Convert history to messages
        mem0_messages = prepare_messages_for_mem0(original_ex)
        
        # 2. Reset Memory for this user_id (Clean slate)
        try:
            memory_client.delete_all(user_id=user_id)
        except Exception:
            pass

        # 3. Add to memory
        if mem0_messages:
            memory_client.add(mem0_messages, user_id=user_id)
    
    print("Ingestion Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments2/data/dev_3.json")
    args = parser.parse_args()

    run_ingestion(args.input_path)