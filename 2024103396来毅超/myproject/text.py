import os

import httpx
import openai
import requests
from openai import OpenAI

proxies= {
    "http": "http://172.19.80.1:7890",
    "https": "http://172.19.80.1:7890",
}

openai.api_key = "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"

proxy_client = httpx.Client(proxies={
    "http://": "http://172.19.80.1:7890",
    "https://": "http://172.19.80.1:7890",
})

# 创建 OpenAI 实例（推荐做法）
client = OpenAI(
    api_key="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",
    http_client=proxy_client
)

# 测试调用
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)

print(response.choices[0].message.content)