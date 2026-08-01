import asyncio
import os
import re
import tempfile
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.logger_handler import logger
from app.rag.text_spliter import AsyncTextSplitter
from app.utils.config import chroma_config
from app.utils.file_handler import (
    get_file_md5_hex,
    listdir_allowed_type,
    markdown_loader,
    markdown_loader_sync,
    pdf_loader,
    pdf_loader_sync,
    ppt_loader,
    ppt_loader_sync,
    txt_loader,
    txt_loader_sync,
    word_loader,
    word_loader_sync,
)
from app.utils.pdf_multimodal_loader import pdf_multimodal_loader, pdf_multimodal_loader_sync


class DocumentProcessor:
    """文档处理器"""

    def __init__(self, vectors_store: Chroma, md5_store, embed_model=None):
        self.vectors_store = vectors_store
        self.md5_store = md5_store
        self.spliter = AsyncTextSplitter(
            chunk_size=chroma_config['chunk_size'],
            chunk_overlap=chroma_config['chunk_overlap'],
            separators=chroma_config['separators'],
            embedding_model=embed_model
        )

    async def get_file_document(self, read_path: str, md5: str = None, user_id: str = None) -> list[Document]:
        """异步加载文件"""
        if read_path.endswith('.txt'):
            return await txt_loader(read_path)
        elif read_path.endswith('.pdf'):
            # 优先使用多模态加载器（提取图片+视觉描述），仅当提供了md5和user_id时才启用；
            # 这两个参数用于定位图片的存储路径 data/extracted_images/{user_id}/{md5}/
            if md5 and user_id:
                return await pdf_multimodal_loader(read_path, md5, user_id)
            # 回退到纯文本加载器（仅提取文字，无图片）
            return await pdf_loader(read_path)
        elif read_path.endswith('.md'):
            return await markdown_loader(read_path)
        elif read_path.endswith('.pptx'):
            return await ppt_loader(read_path)
        elif read_path.endswith('.docx'):
            return await word_loader(read_path)
        else:
            return []

    def get_file_document_sync(self, read_path: str, md5: str = None, user_id: str = None) -> list[Document]:
        """同步加载文件（用于多线程场景）"""
        if read_path.endswith('.txt'):
            return txt_loader_sync(read_path)
        elif read_path.endswith('.pdf'):
            if md5 and user_id:
                return pdf_multimodal_loader_sync(read_path, md5, user_id)
            return pdf_loader_sync(read_path)
        elif read_path.endswith('.md'):
            return markdown_loader_sync(read_path)
        elif read_path.endswith('.pptx'):
            return ppt_loader_sync(read_path)
        elif read_path.endswith('.docx'):
            return word_loader_sync(read_path)
        else:
            return []

    def split_documents_sync(self, documents: list[Document]) -> list[Document]:
        """同步分割文档（用于多线程场景）"""
        return self.spliter.split_documents_sync(documents)

    # ─── Parent-Child 构建方法 ──────────────────────────────────────

    _md_heading_split = re.compile(r'(?=^## )', re.MULTILINE)

    def _build_parents(
        self, raw_documents: list[Document], filename: str, md5_hex: str, user_id: str
    ) -> tuple[list[Document], dict[str, str]]:
        """
        从原始文档构建 Parent chunk。
        返回 (parent_docs, parent_id_map)
          - parent_docs: 待存入 rag_parents 的 Document 列表
          - parent_id_map: {parent_id: parent_content} 用于匹配 Child 归属
        """
        parent_max_size = chroma_config.get('parent_max_size', 3000)

        # 合并所有原始文档内容（通常一个文件只有 1 个 Document）
        full_text = "\n\n".join(d.page_content for d in raw_documents)

        # 对 txt 文件做标准化（和 Child 分块时一致），保证 Parent/Child 内容格式匹配
        if filename.endswith('.txt'):
            full_text = self.spliter._normalize_txt_to_markdown(full_text)

        parent_docs = []
        parent_id_map = {}  # {parent_id: content}

        if len(full_text) <= parent_max_size:
            # 整篇文档作为 1 个 Parent
            parent_id = f"parent_{md5_hex}"
            parent_docs.append(Document(
                page_content=full_text,
                metadata={
                    "parent_id": parent_id,
                    "original_filename": filename,
                    "md5": md5_hex,
                    "user_id": user_id or "",
                    "type": "parent",
                }
            ))
            parent_id_map[parent_id] = full_text
        else:
            # 超长文档：按 ## 章节拆分为多个 Parent
            sections = self._md_heading_split.split(full_text)
            sections = [s.strip() for s in sections if s.strip()]

            if len(sections) <= 1:
                # 无 ## 标题，按 parent_max_size 切割
                for idx in range(0, len(full_text), parent_max_size):
                    chunk = full_text[idx:idx + parent_max_size]
                    parent_id = f"parent_{md5_hex}_{idx}"
                    parent_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "parent_id": parent_id,
                            "original_filename": filename,
                            "md5": md5_hex,
                            "user_id": user_id or "",
                            "type": "parent",
                        }
                    ))
                    parent_id_map[parent_id] = chunk
            else:
                for idx, section in enumerate(sections):
                    parent_id = f"parent_{md5_hex}_{idx}"
                    parent_docs.append(Document(
                        page_content=section,
                        metadata={
                            "parent_id": parent_id,
                            "original_filename": filename,
                            "md5": md5_hex,
                            "user_id": user_id or "",
                            "type": "parent",
                        }
                    ))
                    parent_id_map[parent_id] = section

        logger.info(f"【Parent-Child】文件 {filename}: {len(parent_docs)} 个 Parent")
        return parent_docs, parent_id_map

    @staticmethod
    def _find_parent_id(child_content: str, parent_id_map: dict[str, str]) -> str:
        """
        找到 Child 所属的 Parent ID。
        策略：
        1. 单 Parent 时直接返回（最常见情况）
        2. 多 Parent 时，用 Child 内容片段在各 Parent 中做子串匹配
        """
        # 单 Parent 时无需匹配
        if len(parent_id_map) == 1:
            return next(iter(parent_id_map))

        # 去掉 context_prefix（[文档：xxx]\n）后再匹配
        clean_content = child_content
        if clean_content.startswith("[文档："):
            newline_idx = clean_content.find("\n")
            if newline_idx > 0:
                clean_content = clean_content[newline_idx + 1:]

        # 提取前 60 个有效字符（跳过 # 和空白）用于匹配
        stripped = clean_content.lstrip("#").lstrip()
        search_text = stripped[:60]

        best_parent_id = ""
        best_len = float('inf')

        for parent_id, parent_content in parent_id_map.items():
            if search_text and search_text in parent_content and len(parent_content) < best_len:
                best_parent_id = parent_id
                best_len = len(parent_content)

        # 兜底：如果匹配失败，返回第一个 Parent（总比空好）
        if not best_parent_id:
            best_parent_id = next(iter(parent_id_map))

        return best_parent_id

    async def get_document(self, files: list = None, user_id: str = None, progress_callback=None):
        """
        处理文档并将其转为向量存入向量数据库
        :param files: 上传的文件列表，如果为None则从数据文件夹读取
        :param user_id: 用户ID，用于标记文档的所有者
        :param progress_callback: 进度回调函数，用于实时返回处理进度
        """
        file_paths = []
        file_names = {}

        if files:
            for file in files:
                temp_file_path = await asyncio.to_thread(
                    tempfile.NamedTemporaryFile,
                    delete=False,
                    suffix=os.path.splitext(file.filename)[1]
                )
                content = await file.read()
                await asyncio.to_thread(temp_file_path.write, content)
                file_paths.append(temp_file_path.name)
                file_names[temp_file_path.name] = file.filename
        else:
            allowed_file_path: tuple[str] = await listdir_allowed_type(
                chroma_config['data_path'],
                tuple(chroma_config['allow_knowledge_file_types'])
            )
            file_paths = list(allowed_file_path)

        for idx, file_path in enumerate(file_paths):
            filename = file_names.get(file_path, os.path.basename(file_path))

            md5_hex = await get_file_md5_hex(file_path)
            if await self.md5_store.check_md5_hex(md5_hex, user_id):
                if progress_callback:
                    await progress_callback({
                        'step': 'skipping',
                        'filename': filename,
                        'message': f'文件 {filename} 已存在，跳过'
                    })
                logger.info(f"【向量数据库】文件 {file_path} 的md5值 {md5_hex} 已存在，跳过")
                if files:
                    try:
                        os.unlink(file_path)
                    except OSError:
                        pass
                continue

            try:
                if progress_callback:
                    await progress_callback({
                        'step': 'loading',
                        'filename': filename,
                        'message': f'正在加载文档 {filename}...'
                    })
                logger.info(f"【向量数据库】开始加载文档: {filename}")

                # 传入 md5_hex 和 user_id 以支持多模态PDF加载（图片提取和存储路径定位）
                document: list[Document] = await self.get_file_document(file_path, md5_hex, user_id)
                if not document:
                    if progress_callback:
                        await progress_callback({
                            'step': 'error',
                            'filename': filename,
                            'message': f'文件 {filename} 加载内容为空，跳过',
                            'error_message': '文件内容为空'
                        })
                    logger.error(f"【向量数据库】文件 {file_path} 加载内容为空，跳过")
                    if files:
                        try:
                            os.unlink(file_path)
                        except Exception:
                            pass
                    continue

                if progress_callback:
                    await progress_callback({
                        'step': 'splitting',
                        'filename': filename,
                        'message': f'正在切分文档 {filename}...'
                    })
                logger.info(f"【向量数据库】开始切分文档: {filename}")

                # 保存原始文档内容（分块前），供 Parent 构建使用
                raw_documents = [Document(page_content=d.page_content, metadata=d.metadata.copy()) for d in document]

                document: list[Document] = await self.spliter.split_documents(document)
                if not document:
                    if progress_callback:
                        await progress_callback({
                            'step': 'error',
                            'filename': filename,
                            'message': f'文件 {filename} 切分内容为空，跳过',
                            'error_message': '文档切分后为空'
                        })
                    logger.error(f"【向量数据库】文件 {file_path} 切分内容为空，跳过")
                    if files:
                        try:
                            os.unlink(file_path)
                        except OSError:
                            pass
                    continue

                if progress_callback:
                    await progress_callback({
                        'step': 'storing',
                        'filename': filename,
                        'message': f'正在存储向量 {filename}...'
                    })
                logger.info(f"【向量数据库】开始存储向量: {filename}，文档数量: {len(document)}")

                # ─── Parent-Child：生成 Parent 并存储 ───────────────
                from app.rag.vector_store import VectorStoreService
                store = VectorStoreService()
                parent_docs, parent_id_map = self._build_parents(
                    raw_documents, filename, md5_hex, user_id
                )
                if parent_docs:
                    await store.store_parents(parent_docs)
                    logger.info(f"【Parent-Child】存储 {len(parent_docs)} 个 Parent chunk")

                # ─── 上下文注入 + metadata 设置 ───────────────────
                context_prefix = f"[文档：{filename}]"
                for doc in document:
                    doc.page_content = f"{context_prefix}\n{doc.page_content}"

                if user_id:
                    for doc in document:
                        doc.metadata['user_id'] = user_id

                for doc in document:
                    doc.metadata['original_filename'] = filename
                    doc.metadata['md5'] = md5_hex
                    # 给 Child 打 parent_id：匹配所属 Parent
                    doc.metadata['parent_id'] = self._find_parent_id(
                        doc.page_content, parent_id_map
                    )

                await asyncio.to_thread(self.vectors_store.add_documents, document)

                original_filename = file_names.get(file_path, filename) if files else filename
                await self.md5_store.save_md5_hex(md5_hex, filename, original_filename, user_id)

                if progress_callback:
                    await progress_callback({
                        'step': 'completed',
                        'filename': filename,
                        'message': f'文件 {filename} 处理完成'
                    })
                logger.info(f"【向量数据库】文件 {file_path} 的md5值 {md5_hex} 已保存")

                if files:
                    try:
                        os.unlink(file_path)
                    except OSError:
                        pass

            except Exception as e:
                if progress_callback:
                    await progress_callback({
                        'step': 'error',
                        'filename': filename,
                        'message': f'文件 {filename} 处理失败',
                        'error_message': str(e)
                    })
                logger.error(f"【向量数据库】文件 {file_path} 处理时出错: {e}")
                if files:
                    try:
                        os.unlink(file_path)
                    except OSError:
                        pass
                continue
