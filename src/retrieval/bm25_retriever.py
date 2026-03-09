import jieba
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi

# 导入我们的核心契约
from src.core.interfaces import BaseRetriever, Document

class BM25Retriever(BaseRetriever):
    """BM25 稀疏检索器 (基于词频的精准匹配)"""
    
    def __init__(self):
        self.corpus_docs: List[Document] = []
        self.bm25_model: BM25Okapi = None

    def _tokenize(self, text: str) -> List[str]:
        """核心逻辑：中文分词。将文本切分为词组列表"""
        # 使用 jieba 进行精确模式分词
        return list(jieba.cut(text))

    def add_documents(self, documents: List[Document]):
        """构建/更新 BM25 倒排索引"""
        self.corpus_docs.extend(documents)
        
        # 将所有文档分词，喂给 BM25 模型构建索引
        tokenized_corpus = [self._tokenize(doc.content) for doc in self.corpus_docs]
        self.bm25_model = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        """执行标准召回契约"""
        if not self.bm25_model or not self.corpus_docs:
            return []

        # 1. 对 Query 进行相同的分词
        tokenized_query = self._tokenize(query)
        
        # 2. 获取文档库中所有文档的 BM25 绝对分数
        scores = self.bm25_model.get_scores(tokenized_query)
        
        # 3. 找出分数最高的 top_k 个索引
        # 使用 argsort 降序排列获取 top_k 的下标
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        # 4. 组装成标准 Document 返回，并过滤掉得分为 0 的完全不相关文档
        results = []
        for idx in top_k_indices:
            score = scores[idx]
            if score > 0:  # 只有命中关键词的文档才会被召回
                # 复制一份原文档，避免污染底层库，并注入 BM25 分数
                doc = self.corpus_docs[idx]
                result_doc = Document(
                    doc_id=doc.doc_id, 
                    content=doc.content, 
                    metadata=doc.metadata
                )
                result_doc.score = score
                results.append(result_doc)
                
        return results