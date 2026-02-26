import streamlit as st
from dotenv import load_dotenv

from src.ingestion.loaders import load_document
from src.ingestion.chunker import chunk_text

from src.llm.embeddings import get_embedding_client
from src.retrieval.faiss_store import FaissStore

from src.llm.chat import get_chat_client
from src.core.rag import build_rag_prompt

load_dotenv()

st.set_page_config(page_title="企业知识库RAG代理（MVP）", layout="wide")

st.title("企业知识库RAG代理（MVP）")
st.success("Streamlit正在运行。")

# -------------------------
# 1) 上传 & 解析
# -------------------------
st.header("1) 上传文档并解析（PDF/TXT/MD）")

uploaded = st.file_uploader(
    "选择一个文件上传",
    type=["pdf", "txt", "md", "markdown"],
    accept_multiple_files=False,
)

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    st.write("文件名：", uploaded.name)
    st.write("文件大小：", f"{len(file_bytes)/1024:.1f} KB")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("保存并解析", type="primary"):
            with st.spinner("正在保存并解析..."):
                doc = load_document(uploaded.name, file_bytes)

            # 存入 session_state，保证后续按钮可用
            st.session_state["doc"] = doc
            st.success("解析完成，已保存到 session_state。")

    with col2:
        if st.button("清空当前文档"):
            st.session_state.pop("doc", None)
            st.info("已清空 session_state['doc']。")

# 展示解析结果（只要 session_state 里有 doc 就展示）
if "doc" in st.session_state:
    doc = st.session_state["doc"]

    st.subheader("解析结果")
    st.write("保存路径：", doc.source_path)
    st.write("文件类型：", doc.file_type)
    st.write("元信息：", doc.meta)
    st.write("文本长度（chars）：", len(doc.text))

    st.subheader("文本预览（前 2000 字）")
    st.text(doc.text[:2000])

# -------------------------
# 2) 分块 Chunking
# -------------------------
st.header("2) 文本分块（Chunking）")

if "doc" not in st.session_state:
    st.info("请先上传文件并点击「保存并解析」，再进行分块测试。")
else:
    doc = st.session_state["doc"]

    # 可调参数（后面我们会讲怎么选）
    chunk_size = st.number_input("chunk_size", min_value=100, max_value=2000, value=500, step=50)
    overlap = st.number_input("overlap", min_value=0, max_value=1000, value=100, step=20)

    if st.button("进行分块测试"):
        with st.spinner("正在分块..."):
            chunks = chunk_text(doc.text, chunk_size=int(chunk_size), overlap=int(overlap))

        st.write("Chunk 数量：", len(chunks))

        # 预览前 2 个 chunk
        for i, c in enumerate(chunks[:2]):
            st.subheader(f"Chunk {i+1}")
            st.write("长度：", len(c))
            st.text(c[:800])


st.header("3) 建库（Embedding + FAISS）& 检索（TopK）")

if "doc" not in st.session_state:
    st.info("请先上传并解析文档。")
else:
    doc = st.session_state["doc"]
    embedder = get_embedding_client()

    # 用 session_state 缓存 chunks / store
    if st.button("一键建库（对 chunks 生成 embedding 并写入 FAISS）", type="primary"):
        with st.spinner("正在分块..."):
            chunks = chunk_text(doc.text, chunk_size=500, overlap=100)
        st.session_state["chunks"] = chunks

        with st.spinner("正在生成 embeddings 并建 FAISS 索引..."):
            embs = embedder.embed(chunks)  # (n, dim)
            store = FaissStore(
                dim=embedder.dim,
                index_path="data/index/faiss.index",
                meta_path="data/index/chunks.json",
            )
            metas = [{"source_path": doc.source_path, "chunk_no": i} for i in range(len(chunks))]
            store.add(embs, chunks, metas)
            store.save()

        st.session_state["store_ready"] = True
        st.success(f"建库完成：chunks={len(chunks)} dim={embedder.dim} 已保存到 data/index/")

    # 检索区
    query = st.text_input("输入你的问题（先做检索，不调用大模型）", value="这个人有哪些技能？")
    top_k = st.slider("TopK", min_value=1, max_value=10, value=5)

    if st.button("检索 TopK"):
        # 如果没建库，尝试从磁盘加载
        try:
            store = FaissStore.load("data/index/faiss.index", "data/index/chunks.json")
        except Exception as e:
            st.error(f"未找到索引，请先点“一键建库”。错误：{e}")
            st.stop()

        q_emb = embedder.embed([query])[0]
        results = store.search(q_emb, top_k=top_k)

        st.subheader("检索结果（TopK chunks）")
        for rank, (score, rec) in enumerate(results, start=1):
            st.markdown(f"### #{rank}  score={score:.4f}")
            st.write("meta:", rec.meta)
            st.text(rec.text[:800])


st.header("4) RAG 生成回答")

if st.button("生成回答（调用 LLM）"):

    try:
        store = FaissStore.load("data/index/faiss.index", "data/index/chunks.json")
    except:
        st.error("请先建库")
        st.stop()

    embedder = get_embedding_client()
    chat_client = get_chat_client()

    q_emb = embedder.embed([query])[0]
    results = store.search(q_emb, top_k=top_k)

    chunks = [r[1] for r in results]

    with st.spinner("正在生成回答..."):
        prompt = build_rag_prompt(query, chunks)
        answer = chat_client.chat(prompt)

    st.subheader("模型回答")
    st.write(answer)