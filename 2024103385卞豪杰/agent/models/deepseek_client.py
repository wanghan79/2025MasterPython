# -*- coding: utf-8 -*-
import httpx
from typing import Optional
from config import DEEPSEEK_API_URL, API_KEY

class DeepSeekClient:
    def __init__(self, api_url: str = DEEPSEEK_API_URL, api_key: str = API_KEY):
        self.api_url = api_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=60.0)