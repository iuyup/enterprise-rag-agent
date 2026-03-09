import streamlit as st
import numpy as np
import tempfile
from pathlib import Path

# 导入我们重构好的核心引擎
from src.retrieval.faiss_store import FaissStore
from src.retrieval.adapter import FaissRetrieverAdapter
from src.core.interfaces import Document

# ==========================================
# 1. 模拟组件 (为了让 UI 跑起来的临时替代品)
# ==========================================
class MockEmbeddingModel:
    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        return np.random.rand(self.dim).astype(np.float32)

def build_commercial_rag_prompt(query: str, ranked_docs: list, score_threshold: float = 0.5) -> str:
    """商业级 Prompt 组装器"""
    context = ""
    valid_docs = [doc for doc in ranked_docs if doc.score >= score_threshold]
    
    if not valid_docs:
        return f"问题：{query}\n\n知识库中未检索到置信度足够的相关信息，请直接回复'未找到相关信息'。"

    for i, doc in enumerate(valid_docs, start=1):
        source = doc.metadata.get('source', '未知来源')
        context += f"\n\n[片段 {i}] (来源: {source})\n{doc.content}\n"

    prompt = f"你是一名严谨的企业知识库专家。请仅根据以下片段回答问题。\n背景知识：{context}\n\n用户问题：{query}\n回答："
    return prompt

# ==========================================
# 2. 核心引擎初始化 (带缓存机制)
# ==========================================
@st.cache_resource
def load_rag_engine():
    # 造一点假数据存入底层的 FaissStore
    dim = 128
    temp_dir = tempfile.mkdtemp()
    index_path = str(Path(temp_dir) / "test.index")
    meta_path = str(Path(temp_dir) / "test_meta.json")
    
    raw_store = FaissStore(dim=dim, index_path=index_path, meta_path=meta_path)
    texts = ["苹果公司2023年第三季度财报发布，服务业务创历史新高。", 
             "谷歌发布新一代多模态大模型 Gemini。", 
             "微软宣布进一步向 OpenAI 投资，深化合作。"]
    metas = [{"source": "苹果财报.pdf"}, {"source": "谷歌新闻.txt"}, {"source": "微软公告.md"}]
    
    mock_model = MockEmbeddingModel(dim)
    embeddings = np.array([mock_model.encode(t) for t in texts])
    raw_store.add(embeddings, texts, metas)
    
    # 包装为标准适配器并返回
    return FaissRetrieverAdapter(faiss_store=raw_store, embedding_model=mock_model)

# ==========================================
# 3. Streamlit 表现层 UI 逻辑
# ==========================================
st.set_page_config(page_title="RAG 2.0 基础版", page_icon="🤖")
st.title("🏢 企业级知识问答系统")
st.caption("当前架构：单路召回 (FAISS 适配器) + 阈值截断")

# 初始化引擎
retriever = load_rag_engine()

# 用户输入区
query = st.text_input("请输入您的问题：", placeholder="例如：苹果公司的财报表现如何？")

if st.button("检索并回答", type="primary") and query:
    with st.spinner("正在检索企业知识库..."):
        # 👉 1. 调用标准契约接口进行召回
        docs = retriever.retrieve(query, top_k=2)
        
        # 👉 2. 使用商业级组装器生成 Prompt
        # 注意：这里我们把阈值设为 0，因为 Mock 的随机向量算出来的分数不稳定
        prompt = build_commercial_rag_prompt(query, docs, score_threshold=0.0)
        
        # 👉 3. 模拟大模型返回 (因为目前还没接入真实的大模型 API)
        st.markdown("### 🤖 AI 回答")
        st.info(f"(模拟 LLM 根据 Prompt 生成的内容...)\n\n根据提供的知识库片段，为您解答：...")
        
        st.markdown("---")
        st.markdown("### 🔍 内部构造大揭秘 (Debug 面板)")
        st.text_area("发给大模型的完整 Prompt:", value=prompt, height=250)
        
        # 👉 4. 溯源展示区
        with st.expander("查看底层召回的原始文档片段"):
            for idx, doc in enumerate(docs):
                st.write(f"**[{idx+1}] 来源**: `{doc.metadata.get('source')}` | **得分**: `{doc.score:.4f}`")
                st.write(f"> {doc.content}")