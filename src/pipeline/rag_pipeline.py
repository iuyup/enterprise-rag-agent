from typing import List, Dict, Any, Tuple
# 导入核心契约和组件
from src.core.interfaces import BaseRetriever, Document
from src.processing.reranker import BGEReranker

class RAGPipeline:
    """RAG 2.0 顶层业务流水线 (外观模式)"""

    def __init__(
        self, 
        retriever: BaseRetriever, 
        reranker: BGEReranker, 
        llm_client: Any = None
    ):
        """
        依赖注入核心组件
        :param retriever: 召回器（通常是已经组装好的 HybridRetriever）
        :param reranker: 精排器（BGEReranker）
        :param llm_client: 大语言模型客户端（如 OpenAI / ZhipuAI 的 client）
        """
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client

    def run(
        self, 
        query: str, 
        recall_top_k: int = 20, 
        rerank_top_n: int = 5,
        score_threshold: float = 0.5
    ) -> Tuple[str, List[Document]]:
        """
        执行完整的 RAG 问答链路
        
        :return: (LLM的回答字符串, 最终引用的文档列表)
        """
        # ==========================================
        # 阶段 1：粗排召回 (Recall)
        # ==========================================
        # 这里的 retriever 已经是 HybridRetriever，内部自动并发双路并执行 RRF
        recalled_docs = self.retriever.retrieve(query, top_k=recall_top_k)

        # 防御性编程：如果连粗排都没搜到任何东西，直接短路返回
        if not recalled_docs:
            return "知识库中未检索到与您问题相关的信息。", []

        # ==========================================
        # 阶段 2：精排重排 (Rerank)
        # ==========================================
        reranked_docs = self.reranker.rerank(query, recalled_docs, top_n=rerank_top_n)

        # ==========================================
        # 阶段 3：阈值截断与 Prompt 组装 (Build)
        # ==========================================
        prompt, valid_docs = self._build_prompt(query, reranked_docs, score_threshold)

        if not valid_docs:
            return "检索到的信息置信度过低，为避免误导，不予回答。", []

        # ==========================================
        # 阶段 4：大模型生成 (Generate)
        # ==========================================
        answer = self._generate_answer(prompt)

        # 商业级系统的关键：必须把底层依据（valid_docs）和答案一起抛出去，供前端渲染溯源卡片
        return answer, valid_docs

    def _build_prompt(self, query: str, docs: List[Document], threshold: float) -> Tuple[str, List[Document]]:
        """内部方法：执行阈值过滤并组装提示词"""
        valid_docs = [doc for doc in docs if doc.score >= threshold]
        
        if not valid_docs:
            return "", []

        context = ""
        for i, doc in enumerate(valid_docs, start=1):
            source = doc.metadata.get('source', '未知来源')
            context += f"\n\n[片段 {i}] (来源: {source}, 置信度: {doc.score:.2f})\n{doc.content}\n"

        prompt = (
            f"你是一名严谨的企业知识库专家。请严格基于以下[片段]内容回答用户问题。\n"
            f"如果片段中没有答案，请直接回答'未找到相关信息'，绝不捏造。\n"
            f"请在回答末尾注明引用的片段编号。\n\n"
            f"背景知识：{context}\n\n"
            f"用户问题：{query}\n"
            f"结构化回答："
        )
        return prompt, valid_docs

    def _generate_answer(self, prompt: str) -> str:
        """内部方法：调用 LLM 获取回答"""
        if self.llm_client is None:
            # MVP 阶段模拟 LLM 返回
            return f"(模拟 LLM 回答) 收到 Prompt 长度为 {len(prompt)} 字符。已根据提供的知识片段为您整理答案..."
        
        # 真实接入点：
        # response = self.llm_client.chat.completions.create(
        #     model="your-model-name",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.choices[0].message.content
        return "请实现您的 LLM 客户端调用逻辑。"