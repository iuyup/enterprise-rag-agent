import tempfile
import numpy as np
from pathlib import Path

# 导入底层组件
from src.retrieval.faiss_store import FaissStore
from src.retrieval.adapter import FaissRetrieverAdapter
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.processing.reranker import BGEReranker
from src.pipeline.rag_pipeline import RAGPipeline
from src.core.interfaces import Document

# ==========================================
# 1. 模拟环境：Mock Embedding 模型
# ==========================================
class MockEmbeddingModel:
    def __init__(self, dim: int = 128):
        self.dim = dim
    def encode(self,text:str) -> np.ndarray:
        return np.random.rand(self.dim).astype(np.float32)
    
# ==========================================
# 2. 知识库数据初始化（准备测试物料）
# ==========================================
def setup_knowledge_base():
    print("正在初始化企业知识库物料...")
    texts = [
        "2023年苹果公司发布了最新的财务报表，服务收入创下历史新高。",
        "谷歌在I/O大会上发布了新一代多模态大模型Gemini 1.5 Pro。",
        "微软宣布向OpenAI追加投资，全面将ChatGPT整合进Office全家桶。",
        "苹果公司的iPhone 15 Pro Max采用了全新的钛金属机身设计。",
        "某农业公司发布了新型土豆种植技术，亩产提升30%。" # 干扰项
    ]
    metas = [{"source": f"doc_{i}.txt"} for i in range(len(texts))]
    
    # 2.1 初始化 FAISS
    dim = 128
    temp_dir = tempfile.mkdtemp()
    store = FaissStore(dim=dim, index_path=str(Path(temp_dir)/"test.index"), meta_path=str(Path(temp_dir)/"test.json"))
    mock_model = MockEmbeddingModel(dim)
    store.add(np.array([mock_model.encode(t) for t in texts]), texts, metas)
    faiss_adapter = FaissRetrieverAdapter(store, mock_model)
    
    # 2.2 初始化 BM25
    bm25_retriever = BM25Retriever()
    # 将文本转换为标准 Document 对象喂给 BM25
    docs_for_bm25 = [Document(doc_id=str(i), content=t, metadata=m) for i, (t, m) in enumerate(zip(texts, metas))]
    bm25_retriever.add_documents(docs_for_bm25)
    
    return faiss_adapter, bm25_retriever

# ==========================================
# 3. 装配流水线并执行
# ==========================================
if __name__ == "__main__":
    print("=== [步骤 1] 组装底层双路召回引擎 ===")
    faiss_adapter, bm25_retriever = setup_knowledge_base()
    hybrid_retriever = HybridRetriever(dense_retriever=faiss_adapter, sparse_retriever=bm25_retriever, rrf_k=60)
    
    print("=== [步骤 2] 加载 BGE-Reranker 精排模型 ===")
    print("注意：首次运行会自动从 HuggingFace 下载模型文件 (约 1.1GB)，请耐心等待...")
    try:
        # 如果你没有 GPU，use_fp16=True 可能会报错，这里我们做一下兼容
        import torch
        use_fp16 = torch.cuda.is_available() 
        reranker = BGEReranker(model_name_or_path="BAAI/bge-reranker-base", use_fp16=use_fp16)
    except Exception as e:
        print(f"Reranker 加载失败: {e}\n请确保网络畅通，或检查 PyTorch/FlagEmbedding 安装。")
        exit(1)

    print("=== [步骤 3] 实例化 RAG Pipeline ===")
    pipeline = RAGPipeline(retriever=hybrid_retriever, reranker=reranker)
    
    print("\n" + "="*50)
    print("🚀 引擎点火！执行查询请求...")
    print("="*50)
    
    # 我们故意提一个既包含专有名词，又需要语义理解的问题
    query = "苹果公司最近在硬件产品和财务表现上有什么动态？"
    print(f"用户 Query: {query}\n")
    
    # 执行流水线：粗排取 Top 4，精排取 Top 2，只放行精排得分 > 0.1 的文档
    answer, referenced_docs = pipeline.run(
        query=query, 
        recall_top_k=4, 
        rerank_top_n=2, 
        score_threshold=0.1
    )
    
    print("=== 最终 LLM 输出 ===")
    print(answer)
    
    print("\n=== 最终溯源引用的优质文档 (已通过精排与阈值过滤) ===")
    for doc in referenced_docs:
        print(f"- [{doc.metadata['source']}] (精排得分: {doc.score:.4f}): {doc.content}")