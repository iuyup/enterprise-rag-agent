from typing import List
from src.retrieval.faiss_store import ChunkRecord


# ==========================================
# 3. RAG 组装升级 (引入截断与引用机制)
# ==========================================
def build_commercial_rag_prompt(query: str, ranked_docs: List[Document], score_threshold: float = 0.5) -> str:
    """商业级 Prompt 组装，带有阈值过滤和溯源元数据"""
    context = ""
    valid_docs = [doc for doc in ranked_docs if doc.score >= score_threshold]
    
    if not valid_docs:
        # 尽早阻断，防止 LLM 强行编造
        return f"问题：{query}\n\n指令：知识库中未检索到置信度足够的相关信息，请直接回复'未找到相关信息'。"

    for i, doc in enumerate(valid_docs, start=1):
        # 假设 meta 中有 source 字段
        source = doc.metadata.get('source', '未知来源')
        # 将 Reranker 的打分注入 Prompt 可选，有时有助于 LLM 判断权重
        context += f"\n\n[片段 {i}] (来源: {source}, 相关度: {doc.score:.2f})\n{doc.content}\n"

    prompt = f"""
你是一名严谨的企业知识库专家。请严格遵循以下规则：
1. 仅依赖提供的[片段]内容回答问题，绝不捏造事实。
2. 在回答的结尾，请注明你参考的片段编号（例如：[片段 1]、[片段 2]）。

背景知识：
{context}

用户问题：
{query}

结构化回答：
"""
    return prompt.strip()