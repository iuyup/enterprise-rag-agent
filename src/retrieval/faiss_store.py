from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np


@dataclass
class ChunkRecord:
    chunk_id: int
    text: str
    meta: Dict[str, Any]


class FaissStore:
    def __init__(self, dim: int, index_path: str, meta_path: str):
        self.dim = dim
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index = faiss.IndexFlatIP(dim)  # 用内积做相似度（配合归一化向量≈cosine）
        self.records: List[ChunkRecord] = []

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / norms

    def add(self, embeddings: np.ndarray, texts: List[str], metas: List[Dict[str, Any]]):
        assert embeddings.shape[0] == len(texts) == len(metas)
        embeddings = self._normalize(embeddings)

        start_id = len(self.records)
        for i, (t, m) in enumerate(zip(texts, metas)):
            self.records.append(ChunkRecord(chunk_id=start_id + i, text=t, meta=m))

        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, ChunkRecord]]:
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        query_embedding = self._normalize(query_embedding)

        scores, idxs = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((float(score), self.records[int(idx)]))
        return results

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        payload = {
            "dim": self.dim,
            "records": [
                {"chunk_id": r.chunk_id, "text": r.text, "meta": r.meta}
                for r in self.records
            ],
        }
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "FaissStore":
        meta_p = Path(meta_path)
        payload = json.loads(meta_p.read_text(encoding="utf-8"))
        dim = int(payload["dim"])

        store = cls(dim=dim, index_path=index_path, meta_path=meta_path)
        store.index = faiss.read_index(str(index_path))
        store.records = [
            ChunkRecord(chunk_id=r["chunk_id"], text=r["text"], meta=r["meta"])
            for r in payload["records"]
        ]
        return store