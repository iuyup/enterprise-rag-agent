from typing import List
from src.retrieval.faiss_store import ChunkRecord


def build_rag_prompt(query: str, chunks: List[ChunkRecord]) -> str:
    context = ""

    for i, chunk in enumerate(chunks, start=1):
        context += f"\n\n[文档片段 {i}]\n{chunk.text}\n"

    prompt = f"""
你是一名企业知识库问答助手。

请仅根据以下文档内容回答问题。
如果文档中没有答案，请回答“未在知识库中找到相关信息”。

{context}

问题：
{query}

请给出结构化回答。
"""

    return prompt.strip()