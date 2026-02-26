from __future__ import annotations
import os
import requests
from typing import Protocol


class ChatClient(Protocol):
    def chat(self, prompt: str) -> str:
        ...


# --------
# 本地占位（无 API 也能跑通）
# --------
class EchoChatClient:
    def chat(self, prompt: str) -> str:
        return f"[MOCK LLM RESPONSE]\n\n{prompt[:500]}"


# --------
# DeepSeek 示例（可改为 OpenAI）
# --------
class DeepSeekChatClient:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置")

    def chat(self, prompt: str) -> str:
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个企业知识库问答助手。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }

        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def get_chat_client() -> ChatClient:
    provider = os.getenv("LLM_PROVIDER", "mock").lower()

    if provider == "deepseek":
        return DeepSeekChatClient()

    return EchoChatClient()