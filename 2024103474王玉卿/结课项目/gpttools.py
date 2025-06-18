from typing import List, Dict, Tuple
import os
import json
import time
import yaml
import openai
from collections import Counter

from schema import GPT3_CACHE
from log import get_logger

logger = get_logger('root', filename='gpttools.log')


def setup(config_file: str) -> None:
    with open(config_file, 'r', encoding='utf-8') as f:
        yaml_configs = yaml.load(f, Loader=yaml.FullLoader)
    openai.api_key = yaml_configs['OPENAI_API_KEY']
    openai.api_base = "https://api.v36.cm/v1"  # 使用第三方 API 地址


def load_cache(gpt_cache_dir: str, seed: int = -1) -> None:
    global GPT3_CACHE
    filename = f'gpt_cache_{seed}.json' if seed != -1 else 'gpt_cache.json'
    path = os.path.join(gpt_cache_dir, filename)
    if os.path.exists(path):
        with open(path) as f:
            GPT3_CACHE = json.load(f)


def save_cache(gpt_cache_dir: str, seed: int = -1) -> None:

    filename = f'gpt_cache_{seed}.json' if seed != -1 else 'gpt_cache.json'
    path = os.path.join(gpt_cache_dir, filename)

    # 自动创建目录（如果不存在）
    os.makedirs(gpt_cache_dir, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(GPT3_CACHE, f, indent=2)
    print(f"Saved {len(GPT3_CACHE)} cache entries to {path}")



def get_gpt3_response(prompt: str, use_cache: bool = True) -> str:
    global GPT3_CACHE
    if use_cache and prompt in GPT3_CACHE:
        return GPT3_CACHE[prompt]['text']

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
            top_p=0.95
        )
        text = response['choices'][0]['message']['content'].strip()
        GPT3_CACHE[prompt] = {'text': text}
        return text
    except Exception as e:
        logger.warning(f"[ChatGPT调用失败] {e}")
        time.sleep(5)
        return ""


def get_gpt3_score(prompt: str, use_cache: bool = True, num_votes: int = 5) -> Tuple[str, List[Dict[str, float]]]:
    global GPT3_CACHE
    vocab = ['yes', 'no', 'unknown']

    if use_cache and prompt in GPT3_CACHE:
        return GPT3_CACHE[prompt]['text'], GPT3_CACHE[prompt]['logprobs']

    vote_counter = Counter()
    all_outputs = []

    for i in range(num_votes):
        try:
            response = get_gpt3_response(prompt, use_cache=False)
            label = response.strip().lower()

            # # 容错处理：宽松匹配关键字
            # if 'yes' in label:
            #     label = 'yes'
            # elif 'no' in label:
            #     label = 'no'
            # elif 'unknown' in label:
            #     label = 'unknown'
            # else:
            #     logger.warning(f"[非法label] GPT返回未知答案: '{label}'，默认设为 unknown")
            #     label = 'unknown'

            all_outputs.append(label)
            vote_counter[label] += 1
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"[GPT失败] 第{i + 1}轮失败: {e}")

    total = sum(vote_counter.values()) or 1
    logprobs = [{w: vote_counter[w] / total for w in vocab}]
    final_answer = vote_counter.most_common(1)[0][0]

    GPT3_CACHE[prompt] = {
        'text': final_answer,
        'logprobs': logprobs,
        'votes': all_outputs
    }

    return final_answer, logprobs
