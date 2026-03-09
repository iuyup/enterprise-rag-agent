from typing import List, Dict
# 导入核心契约
from src.core.interfaces import BaseRetriever, Document

class HybridRetriever(BaseRetriever):
    """混合检索调度器：RAG 2.0 的召回中枢"""
    
    def __init__(self, dense_retriever: BaseRetriever, sparse_retriever: BaseRetriever, rrf_k: int = 60):
        # 依赖注入：调度器不关心底层是 FAISS 还是 ES，只要遵守 BaseRetriever 契约即可
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        # 1. 双路并发召回 (在企业级生产环境中，这里应使用 ThreadPoolExecutor 开启多线程并发)
        # 技巧：底层召回的数量通常要比最终需要的 top_k 大一倍，给 RRF 留出融合空间
        pool_size = top_k * 2
        dense_docs = self.dense_retriever.retrieve(query, top_k=pool_size)
        sparse_docs = self.sparse_retriever.retrieve(query, top_k=pool_size)

        # 2. 执行 RRF 融合算法
        merged_docs = self._apply_rrf(dense_docs, sparse_docs)

        # 3. 截断返回最终的 top_k
        return merged_docs[:top_k]

    def _apply_rrf(self, dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
        """RRF 核心算法实现"""
        # 使用哈希表记录文档的最终 RRF 分数和对象映射
        rrf_score_map: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # 处理稠密检索 (FAISS) 排名
        for rank, doc in enumerate(dense_docs, start=1):
            if doc.doc_id not in rrf_score_map:
                rrf_score_map[doc.doc_id] = 0.0
                doc_map[doc.doc_id] = doc
            rrf_score_map[doc.doc_id] += 1.0 / (self.rrf_k + rank)

        # 处理稀疏检索 (BM25) 排名
        for rank, doc in enumerate(sparse_docs, start=1):
            if doc.doc_id not in rrf_score_map:
                rrf_score_map[doc.doc_id] = 0.0
                doc_map[doc.doc_id] = doc
            # 如果文档在双路中都命中了，这里的 += 会让它的 RRF 分数激增！
            rrf_score_map[doc.doc_id] += 1.0 / (self.rrf_k + rank)

        # 重新组装 Document 列表，并将 score 覆写为 RRF 分数
        fused_docs = []
        for doc_id, rrf_score in rrf_score_map.items():
            doc = doc_map[doc_id]
            # 创建全新的 Document 对象，防止污染底层引擎缓存的原始对象
            fused_doc = Document(doc_id=doc.doc_id, content=doc.content, metadata=doc.metadata)
            fused_doc.score = rrf_score
            fused_docs.append(fused_doc)

        # 根据 RRF 分数降序排列
        fused_docs.sort(key=lambda x: x.score, reverse=True)
        return fused_docs