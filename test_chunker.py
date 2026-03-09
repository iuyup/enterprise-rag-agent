import re
from typing import List

# ==========================================
# 1. 你的 RAG 0.1 原版：暴力滑动窗口
# ==========================================
def old_sliding_window_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ==========================================
# 2. RAG 2.0 升级版：语义分块器
# ==========================================
class SemanticChunker:
    def __init__(self, max_tokens: int, overlap_sentences: int = 1):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def chunk_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[。！？!?])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len > self.max_tokens and current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = current_chunk[-self.overlap_sentences:]
                current_length = sum(len(s) for s in current_chunk)
                
            current_chunk.append(sentence)
            current_length += sentence_len
            
        if current_chunk:
            chunks.append("".join(current_chunk))
        return chunks

# ==========================================
# 3. 测试与对比逻辑
# ==========================================
if __name__ == "__main__":
    # 准备一段典型的企业财报文本（故意制造长短句）
    sample_text = (
        "2023年第三季度，苹果公司发布了最新的财务报表。数据显示，公司在该季度的总营收达到了895亿美元，同比下降了1%。"
        "然而，服务业务的收入却创下了历史新高，达到了223亿美元。首席执行官蒂姆·库克表示，这一成绩归功于iPhone 15系列的"
        "强劲销量以及服务生态系统的持续扩展。尽管面临全球宏观经济的不确定性，公司依然保持了极高的盈利能力和充沛的现金流。"
        "未来，苹果将继续加大在人工智能和空间计算领域的研发投入！"
    )

    # 为了让截断效果明显，我们把 chunk_size 设得很小（50个字符）
    LIMIT = 50

    print("=== 测试 1：原版滑动窗口 (Character-based) ===")
    old_chunks = old_sliding_window_chunk(sample_text, chunk_size=LIMIT, overlap=10)
    for i, chunk in enumerate(old_chunks):
        print(f"Chunk {i+1} (长度 {len(chunk)}): {chunk}")

    print("\n" + "="*50 + "\n")

    print("=== 测试 2：语义分块器 (Semantic-based) ===")
    new_chunker = SemanticChunker(max_tokens=LIMIT, overlap_sentences=1)
    new_chunks = new_chunker.chunk_text(sample_text)
    for i, chunk in enumerate(new_chunks):
        print(f"Chunk {i+1} (长度 {len(chunk)}): {chunk}")