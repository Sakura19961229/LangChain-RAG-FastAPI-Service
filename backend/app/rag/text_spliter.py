import asyncio
import math
import re
from typing import Any

from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger_handler import logger
from app.utils.config import chroma_config


class AsyncTextSplitter:
    """
    异步文本分割器

    支持两种模式：
    - 基础模式（无 embedding_model）：使用 RecursiveCharacterTextSplitter 按字符切割
    - 语义模式（有 embedding_model）：先按句子切分，再根据相邻句子的 embedding 相似度
      判断语义边界，在相似度骤降处切块。保证同一话题的内容不被切断。

    语义分块算法：
    1. 将文本按句子拆分
    2. 对每个句子计算 embedding
    3. 计算相邻句子对的余弦相似度
    4. 相似度低于阈值处 = 话题转换点 = 分块边界
    5. 对超长块做二次切割兜底
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: list[str] | None = None,
                 embedding_model: Embeddings | None = None,
                 similarity_threshold: float = 0.5,
                 max_chunk_size: int = None):
        """
        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 块间重叠（仅基础模式使用）
            separators: 分割符列表
            embedding_model: 嵌入模型（传入则启用语义分块）
            similarity_threshold: 语义边界阈值，低于此值则切分（0-1）
            max_chunk_size: 单个块的最大字符数上限，超长时强制二次切割。默认为 chunk_size * 3
        """
        default_separators = chroma_config['separators']

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or default_separators
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size or chunk_size * 3

        # 基础分割器（兜底 + 无 embedding 时使用）
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators
        )

        # 句子分割正则：按中英文句号、问号、感叹号、换行切分
        self._sentence_pattern = re.compile(r'(?<=[。！？!?\n])\s*')

    # ─── 公共接口 ────────────────────────────────────────────────────

    async def split_text(self, text: str) -> list[str]:
        """分割文本为多个片段"""
        if self.embedding_model:
            return await self._semantic_split_text(text)
        chunks = await asyncio.to_thread(self.splitter.split_text, text)
        return chunks

    async def split_documents(self, documents: list[Any]) -> list[Any]:
        """分割文档列表"""
        if self.embedding_model:
            return await self._semantic_split_documents(documents)
        split_docs = await asyncio.to_thread(self.splitter.split_documents, documents)
        return split_docs

    def split_text_sync(self, text: str) -> list[str]:
        """同步分割文本"""
        if self.embedding_model:
            return self._semantic_split_text_sync(text)
        return self.splitter.split_text(text)

    def split_documents_sync(self, documents: list[Any]) -> list[Any]:
        """同步分割文档列表"""
        if self.embedding_model:
            return self._semantic_split_documents_sync(documents)
        return self.splitter.split_documents(documents)

    # ─── 语义分块核心逻辑 ─────────────────────────────────────────────

    # 标题行正则：Markdown 标题 或 全大写/短行（通常是小节标题）
    _heading_pattern = re.compile(r'^#{1,6}\s|^[A-Z一-鿿]{1,20}[：:]\s*$')

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        将文本拆分为句子列表。
        关键处理：标题行（# xxx）不独立成句，而是粘连到下一句，
        避免标题因语义差异被切开。
        """
        raw_sentences = self._sentence_pattern.split(text)
        raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

        # 标题粘连：如果一行是标题，把它和下一行合并
        merged = []
        i = 0
        while i < len(raw_sentences):
            line = raw_sentences[i]
            if self._heading_pattern.match(line) and i + 1 < len(raw_sentences):
                # 标题 + 下一句合并
                merged.append(f"{line}\n{raw_sentences[i + 1]}")
                i += 2
            else:
                merged.append(line)
                i += 1

        return merged

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(a * a for a in vec2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)

    def _find_semantic_boundaries(self, sentences: list[str], embeddings: list[list[float]]) -> list[int]:
        """
        找到语义边界位置（相似度骤降处）

        Returns:
            切割点索引列表。如 [3, 7] 表示在第 3 和第 7 个句子后切割。
        """
        if len(sentences) <= 1:
            return []

        # 计算相邻句子对的相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # 找低于阈值的位置作为切割点
        boundaries = []
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)  # 在第 i+1 个句子前切

        return boundaries

    def _group_sentences_by_boundaries(self, sentences: list[str], boundaries: list[int]) -> list[str]:
        """根据边界将句子组合为 chunks"""
        if not boundaries:
            return [" ".join(sentences)]

        chunks = []
        start = 0
        for boundary in boundaries:
            chunk_text = " ".join(sentences[start:boundary])
            if chunk_text.strip():
                chunks.append(chunk_text)
            start = boundary

        # 最后一段
        last_chunk = " ".join(sentences[start:])
        if last_chunk.strip():
            chunks.append(last_chunk)

        return chunks

    def _enforce_max_size(self, chunks: list[str]) -> list[str]:
        """对超长 chunk 做二次切割兜底"""
        result = []
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                result.append(chunk)
            else:
                # 超长块用基础分割器二次切
                sub_chunks = self.splitter.split_text(chunk)
                result.extend(sub_chunks)
        return result

    def _merge_short_chunks(self, chunks: list[str], min_size: int = 50) -> list[str]:
        """
        将过短的 chunk 合并到相邻 chunk。
        解决：标题、单行列表项被独立切成碎片的问题。
        """
        if not chunks:
            return chunks

        merged = []
        buffer = ""

        for chunk in chunks:
            if buffer:
                # 上一个太短，和当前合并
                buffer = f"{buffer}\n{chunk}"
                if len(buffer) >= min_size:
                    merged.append(buffer)
                    buffer = ""
            elif len(chunk) < min_size:
                # 当前太短，暂存
                buffer = chunk
            else:
                merged.append(chunk)

        # 处理末尾残留
        if buffer:
            if merged:
                merged[-1] = f"{merged[-1]}\n{buffer}"
            else:
                merged.append(buffer)

        return merged

    # ─── 语义分块（同步版） ───────────────────────────────────────────

    def _semantic_split_text_sync(self, text: str) -> list[str]:
        """语义分块同步实现"""
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # 批量计算 embeddings
        try:
            embeddings = self.embedding_model.embed_documents(sentences)
        except Exception as e:
            logger.warning(f"【语义分块】embedding 计算失败，降级为基础分块: {e}")
            return self.splitter.split_text(text)

        # 找语义边界

        boundaries = self._find_semantic_boundaries(sentences, embeddings)
        logger.info(f"【语义分块】{len(sentences)} 个句子，找到 {len(boundaries)} 个语义边界")

        # 按边界组合
        chunks = self._group_sentences_by_boundaries(sentences, boundaries)

        # 过短 chunk 合并（解决标题/单行列表碎片问题）
        chunks = self._merge_short_chunks(chunks)

        # 超长兜底
        chunks = self._enforce_max_size(chunks)

        return chunks

    def _semantic_split_documents_sync(self, documents: list[Any]) -> list[Any]:
        """语义分块同步版（处理 Document 对象）"""
        result = []
        for doc in documents:
            chunks = self._semantic_split_text_sync(doc.page_content)
            for chunk in chunks:
                result.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
        return result

    # ─── 语义分块（异步版） ───────────────────────────────────────────

    async def _semantic_split_text(self, text: str) -> list[str]:
        """语义分块异步实现"""
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # 批量计算 embeddings（可能是网络调用，放到线程池）
        try:
            embeddings = await asyncio.to_thread(
                self.embedding_model.embed_documents, sentences
            )
        except Exception as e:
            logger.warning(f"【语义分块】embedding 计算失败，降级为基础分块: {e}")
            return await asyncio.to_thread(self.splitter.split_text, text)

        # 找语义边界
        boundaries = self._find_semantic_boundaries(sentences, embeddings)
        logger.info(f"【语义分块】{len(sentences)} 个句子，找到 {len(boundaries)} 个语义边界")

        # 按边界组合
        chunks = self._group_sentences_by_boundaries(sentences, boundaries)

        # 过短 chunk 合并（解决标题/单行列表碎片问题）
        chunks = self._merge_short_chunks(chunks)

        # 超长兜底
        chunks = self._enforce_max_size(chunks)

        return chunks

    async def _semantic_split_documents(self, documents: list[Any]) -> list[Any]:
        """语义分块异步版（处理 Document 对象）"""
        result = []
        for doc in documents:
            chunks = await self._semantic_split_text(doc.page_content)
            for chunk in chunks:
                result.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
        return result
