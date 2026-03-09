# Enterprise RAG Engine 2.0

企业级双路召回与精排知识问答引擎
Enterprise-grade Knowledge Base QA system upgraded from standard RAG to RAG 2.0 architecture, featuring Hybrid Search and Cross-Encoder Reranking.
本系统已从标准 RAG 升级至 RAG 2.0 架构，引入了混合搜索与交叉编码器重排技术，显著提升了商业环境下的检索精度。

## sKey Evolutions 
- Semantic Chunking (语义分块): Replaced naive sliding windows with punctuation-aware semantic splitting to preserve sentence integrity for Rerankers.
    - 语义分块: 弃用暴力滑动窗口，采用基于标点的语义切分，为重排模型保留完整的句子上下文。

- Hybrid Retrieval (双路并发召回): Combines FAISS (Semantic/Dense) and BM25 (Keyword/Sparse) to capture both deep meaning and precise terminology.
    - 混合检索: 融合 FAISS（语义/稠密）与 BM25（关键词/稀疏），兼顾深层语义理解与专业术语匹配。

- RRF Fusion (倒数秩融合算法): Implements Reciprocal Rank Fusion (RRF) to normalize and merge heterogeneous scores from multiple search paths.
    - RRF 融合: 采用 RRF 算法对异构检索分值进行归一化融合，实现多路召回结果的最优排序。

- BGE Reranking (二次精排): Integrated BGE-Reranker (Cross-Encoder) to refine the Top-K candidates with fine-grained attention mechanism.
    - 二次重排: 集成 BGE-Reranker (交叉编码器)，通过细粒度注意力机制对候选文档进行精准打分。

- Decoupled Architecture (解耦架构): Fully OOP-based design with BaseRetriever interfaces, making it easy to swap Embedding models or Vector stores.
    - 解耦设计: 全面向对象架构，通过 BaseRetriever 接口实现组件解耦，支持轻松更换向量库或模型。


## Project Structure 
src/
├── core/             # Data contracts & Base classes (数据契约与基类)
├── retrieval/        # FAISS, BM25 & Hybrid logic (检索核心逻辑)
├── processing/       # Chunker & Reranker (分块与重排)
└── pipeline/         # Top-level RAG workflow (高层业务流水线)

## Quick Start
### 1)  Environment Setup
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Install dependencies
pip install torch faiss-cpu sentence-transformers rank_bm25 jieba streamlit

### 2)  Run Integration Test
Verify the full pipeline (Recall -> RRF -> Rerank -> Build) via terminal:
python test_pipeline.py

### 3) Launch UI
streamlit run app/streamlit_app.py

## Roadmap
- [ ] Data Ingestion: Support automated PDF/Docx batch processing. (自动化文档批量入库)
- [ ] Real LLM Integration: Connector for OpenAI/ZhipuAI API. (对接真实大模型接口)
- [ ] Persistent Store: Replace local files with Qdrant or Milvus. (迁移至生产级向量数据库)