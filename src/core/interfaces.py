from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Document:
    """标准内部数据契约：所有底层的召回结果，最终都要包装成这个对象"""
    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata or {}
        self.score = 0.0  # 用于记录检索或重排后的分数

    def __repr__(self):
        return f"<Document id={self.doc_id} score={self.score:.4f}>"


class BaseRetriever(ABC):
    """召回器基类：定义了所有 Retriever 必须具备的行为规范"""
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        """
        无论底层是什么引擎，入参必须是 query 和 top_k，
        出参必须是标准 Document 对象的列表。
        """
        pass