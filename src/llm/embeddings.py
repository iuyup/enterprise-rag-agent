from __future__ import annotations

import os
import numpy as np
import requests
from dataclasses import dataclass
from typing import List, Protocol


class EmbeddingClient(Protocol):
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return shape: (n, dim), dtype float32."""


# --------
# 1) 最稳的本地 Mock（无 API 也能跑通全流程）
# --------
@dataclass
class HashEmbeddingClient:
    dim: int = 384

    def embed(self, texts: List[str]) -> np.ndarray:
        # 简单可复现：把文本 hash 到固定维度
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            b = t.encode("utf-8", errors="ignore")
            # 简单滚动 hash
            h = 2166136261
            for x in b[:5000]:
                h ^= x
                h *= 16777619
                h &= 0xFFFFFFFF
            rng = np.random.default_rng(h)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-8)
            out[i] = v
        return out


def get_embedding_client() -> EmbeddingClient:
    """
    EMBEDDING_PROVIDER:
      - "hash" (default) : 本地可跑
      - 后续你接 OpenAI/DeepSeek 时再扩展
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "hash").lower()

    if provider == "hash":
        return HashEmbeddingClient(dim=int(os.getenv("EMBEDDING_DIM", "384")))

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")