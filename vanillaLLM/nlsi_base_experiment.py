import tqdm
import os
import json 
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from openai import OpenAI

# OpenAI API 초기화
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
def load_chains_dataset(fpath: str) -> pd.DataFrame:
    try:
        df = pd.read_json(fpath, lines=True)
        return df
    except ValueError:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)


# ---------------------------------------------------------
# 2. Build User Profile
# ---------------------------------------------------------
def build_user_profile(instructions: List[Dict[str, Any]]) -> str:
    inst_list = [inst["nl_instruction"].strip() for inst in instructions]
    if not inst_list:
        return ""
    return "> " + "\n> ".join(inst_list)


# ---------------------------------------------------------
# 3. Prompt Template
# ---------------------------------------------------------
PROMPT_TEMPLATE = """You are an API selection assistant. 
Given the user's dialogue and the user's standing instructions (user profile),
generate the correct API call.

User Profile:
{user_profile}

User Utterance:
{user_utterance}

Output Format:
<API_CALL>
function_name(arg1="value", arg2="value", ...)

Example : 
    User Profile:
        When I request Restaurants, my go-to cuisine is Indian.
        Request Restaurants with Indian cuisine, expensive price range, and the name Amber.
        I prefer Portland, OR as my destination if I am requesting HouseStays.
        Request that your travel be suitable for children.
        I am looking for an event with the event type Music, category Pop, and event name Aly And Aj.
        Search for HouseStays in Portland, OR with a rating of 4.5 and offering laundry service.
        Request Salon services from A Businessman's Haircut as my stylist.
        Request Movies at 10:30 pm.

    User Utterance : 
        "User: I want to eat at a costly place.\nAgent: What city?\nUser: SF.\n"
    API_CALL : 
        "GetRestaurants(city=\"SF\", price_range=\"expensive\", cuisine=\"Indian\", restaurant_name=\"Amber\")"


Now produce the correct API call:
"""


# ---------------------------------------------------------
# 4. Build prompt for each example
# ---------------------------------------------------------
def build_input_prompt(example: Dict[str, Any]) -> str:
    user_profile = build_user_profile(example["all_standing_instructions"])
    user_utt = example["user_utterance"].strip()

    prompt = PROMPT_TEMPLATE.format(
        user_profile=user_profile,
        user_utterance=user_utt,
    )
    return prompt


# ---------------------------------------------------------
# 5. Call GPT API
# ---------------------------------------------------------
def call_gpt_api(prompt: str) -> str:
    """
    gpt-4o-mini 호출
    """
    # baseline prompt 파일에서 읽어오기
    baseline_prompt_path = "/data/minseo/personal-tool/conv_api/experiments/new_baseline_prompt.txt"
    with open(baseline_prompt_path, "r", encoding="utf-8") as f:
        baseline_prompt = f.read()

    response = client.responses.create(
        model="gpt-4o-mini-2024-07-18", #gpt-5.1-2025-11-13 #gpt-5-2025-08-07 #gpt-4o-2024-08-06 #gpt-4o-mini-2024-07-18
        input=[
            {
                "role": "system",
                # 여기에서 {baseline_prompt}를 실제 내용으로 넣어야 함 (f-string)
                "content": baseline_prompt,
            },
            {
                "role": "user",
                # 여기에서 "prompt" 리터럴이 아니라, 우리가 만든 prompt 변수를 넣어야 함
                "content": prompt,
            },
        ],
        #tools=tools,
        #reasoning={"effort" : "high"}
        temperature=0.0,
    )
   # print(response)
    output = response.output[0].content[0].text
    #output = response.output[1].content[0].text - gpt5.1

    #print(output)
    return output


# ---------------------------------------------------------
# 6. Logging 기능
# ---------------------------------------------------------
def write_log(log_path: str, record: Dict[str, Any]):
    dirpath = os.path.dirname(log_path)
    if dirpath:  # "./gpt_log.jsonl" 같은 경우 빈 문자열이라 mkdir 하면 안됨
        os.makedirs(dirpath, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# 7. Build Full Processing Pipeline
# ---------------------------------------------------------
def process_with_gpt(
    input_path: str,
    output_path: str,
    log_path: str,
):
    df = load_chains_dataset(input_path)
    processed_data = []

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing examples"):
        ex = row.to_dict()

        # Prompt 생성
        prompt = build_input_prompt(ex)

        # GPT 호출
        gpt_output = call_gpt_api(prompt)

        # Logging
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "example_id": ex.get("example_id"),
            "model_input": prompt,
            "model_output": gpt_output,
        }
        write_log(log_path, log_record)

        # 원본 JSON에도 gpt_output 추가
        ex["gpt_output"] = gpt_output
        processed_data.append(ex)

    # 새로운 output JSON 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print(f"Processing completed. Saved → {output_path}")
    print(f"Log saved → {log_path}")


# ---------------------------------------------------------
# 8. 실행 예시
# ---------------------------------------------------------
if __name__ == "__main__":
    process_with_gpt(
        input_path="/data/minseo/personal-tool/conv_api/nlsi2exp2/output/data/50sampled/nlsi_test_simple_50sampled_251202-step1-2-step2-1.json",
        output_path="/data/minseo/personal-tool/conv_api/experiments/exp/simple/nlsi_base/251208-gpt4omini-fs.json",
        log_path="/data/minseo/personal-tool/conv_api/experiments/exp/simple/nlsi_base/251208-gpt4omini-fs.jsonl",
    )
