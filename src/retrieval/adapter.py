# ==========================================
# 1. 存储层适配器 (解耦 FaissStore 与 业务逻辑)
# ==========================================

from typing import List
# 1. 导入你原来的底层引擎
from src.retrieval.faiss_store import FaissStore 
# 2. 导入我们刚刚定义好的核心接口
from src.core.interfaces import BaseRetriever, Document 

class FaissRetrieverAdapter(BaseRetriever):
    """FaissStore 的适配器，使其符合标准 BaseRetriever 接口"""
    def __init__(self, faiss_store: FaissStore, embedding_model):
        self.store = faiss_store
        self.embedding_model = embedding_model # 注入你的 Embedding 模型实例

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        # 1. Query 向量化
        query_emb = self.embedding_model.encode(query)
        
        # 2. 调用原始 FaissStore 检索
        raw_results = self.store.search(query_emb, top_k)
        
        # 3. 转换为标准 Document 契约
        documents = []
        for score, record in raw_results:
            doc = Document(
                doc_id=str(record.chunk_id),
                content=record.text,
                metadata=record.meta
            )
            doc.score = score # 这里的 score 是 Faiss 的内积相似度
            documents.append(doc)
            
        return documents