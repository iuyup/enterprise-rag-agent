import numpy as np
import tempfile
from pathlib import Path

# 导入你原有的底层引擎 (确保路径正确)
from src.retrieval.faiss_store import FaissStore
# 导入我们今天写的适配器和核心契约
from src.core.interfaces import Document
from src.retrieval.adapter import FaissRetrieverAdapter

# ==========================================
# 1. 模拟环境：Mock Embedding 模型
# ==========================================
class MockEmbeddingModel:
    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        # 生成一个随机向量模拟大模型的 Embedding 输出
        vec = np.random.rand(self.dim).astype(np.float32)
        return vec

def setup_mock_data():
    """初始化一点假数据存入你写的 FaissStore"""
    dim = 128
    temp_dir = tempfile.mkdtemp()
    index_path = str(Path(temp_dir) / "test.index")
    meta_path = str(Path(temp_dir) / "test_meta.json")
    
    # 实例化你的原版 FaissStore
    store = FaissStore(dim=dim, index_path=index_path, meta_path=meta_path)
    
    # 造3条假数据
    texts = ["苹果公司财报发布", "谷歌发布新一代AI模型", "微软投资OpenAI"]
    metas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
    
    # 模拟向量化并存入
    mock_model = MockEmbeddingModel(dim)
    embeddings = np.array([mock_model.encode(t) for t in texts])
    store.add(embeddings, texts, metas)
    
    return store, mock_model

# ==========================================
# 2. 核心测试：验证 Adapter 是否生效
# ==========================================
if __name__ == "__main__":
    print("1. 正在初始化底层 FaissStore 并注入测试数据...")
    raw_store, embed_model = setup_mock_data()
    
    print("2. 正在将其包装为 FaissRetrieverAdapter...")
    # 核心架构体现：把旧引擎和 Embedding 模型一起注给适配器
    retriever = FaissRetrieverAdapter(faiss_store=raw_store, embedding_model=embed_model)
    
    print("3. 执行标准检索契约 retrieve()...")
    query = "帮我查一下科技巨头的最新动态"
    # 注意：这里调用的是标准接口 retrieve，而不是旧的 search
    results = retriever.retrieve(query=query, top_k=2)
    
    print("\n=== 检索结果 (已转换为标准 Document 对象) ===")
    for idx, doc in enumerate(results):
        # 验证返回的是否是我们定义的 Document 对象，而不是 ChunkRecord
        assert isinstance(doc, Document), "返回类型错误！必须是 Document"
        
        print(f"[{idx+1}] ID: {doc.doc_id}")
        print(f"    内容: {doc.content}")
        print(f"    元数据: {doc.metadata}")
        print(f"    相关度得分: {doc.score:.4f}\n")
        
    print("✅ 测试通过！底层引擎已成功解耦，适配器运转正常。")