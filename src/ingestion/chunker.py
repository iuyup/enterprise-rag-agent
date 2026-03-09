# ==========================================
# 2. 语义分块器 (替换暴力的字符滑动窗口)
# ==========================================
class SemanticChunker:
    """基于标点符号的语义分块，保护 Reranker 输入的完整性"""
    def __init__(self, max_tokens: int = 400, overlap_sentences: int = 1):
        # 预留 token 给 Reranker (BGE 通常限 512 tokens，Query 占去部分，Chunk 最好 < 400)
        self.max_tokens = max_tokens 
        self.overlap_sentences = overlap_sentences

    def chunk_text(self, text: str) -> List[str]:
        # 1. 按中文/英文句号、问号、叹号进行正则切分，保留标点
        sentences = re.split(r'(?<=[。！？!?])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # 粗略估算长度 (实际生产中建议用 tiktoken 计算 token 数)
            sentence_len = len(sentence) 
            
            if current_length + sentence_len > self.max_tokens and current_chunk:
                chunks.append("".join(current_chunk))
                # 保留 overlap_sentences 数量的句子作为上下文重叠
                current_chunk = current_chunk[-self.overlap_sentences:]
                current_length = sum(len(s) for s in current_chunk)
                
            current_chunk.append(sentence)
            current_length += sentence_len
            
        if current_chunk:
            chunks.append("".join(current_chunk))
            
        return chunks