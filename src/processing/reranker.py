from typing import List
# 切换到更稳定的 sentence-transformers 引擎
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from src.core.interfaces import Document

class BGEReranker:
    """基于 sentence-transformers 的 BGE 重排器 (更稳定，无 Windows 导入错误)"""
    
    def __init__(self, model_name_or_path: str = "BAAI/bge-reranker-base", use_fp16: bool = False):
        if CrossEncoder is None:
            raise ImportError("请先安装依赖: pip install sentence-transformers")
        
        # CrossEncoder 会自动检测 GPU (cuda) 或 CPU
        # BGE-Reranker 是标准的 Cross-Encoder 架构，完全兼容
        print(f"正在加载精排模型: {model_name_or_path} ...")
        self.model = CrossEncoder(model_name_or_path)

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        if not documents:
            return []

        # 1. 构造输入对
        sentence_pairs = [[query, doc.content] for doc in documents]

        # 2. 推理计算得分
        scores = self.model.predict(sentence_pairs)

        # 3. 覆写分数并排序
        reranked_results = []
        for doc, score in zip(documents, scores):
            # 保持我们的 Document 契约
            new_doc = Document(doc_id=doc.doc_id, content=doc.content, metadata=doc.metadata)
            new_doc.score = float(score)
            reranked_results.append(new_doc)

        # 降序排列
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results[:top_n]