"""
调试端点 —— 以人类可读的 JSON 格式输出 Redis / MySQL / ChromaDB 全部存储内容。

仅用于开发调试，不做鉴权。所有端点支持可选 user_id 参数过滤。
"""

import asyncio
import json
import os
from collections import defaultdict
from typing import Optional

from fastapi import Depends, Query
from fastapi.routing import APIRouter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger_handler import logger
from app.core.success_response import success_response
from app.db.db_config import get_db
from app.db.redis_config import connect_redis
from app.models.chat_history import ChatMessage, ChatSession
from app.models.note import Note
from app.models.note_template import NoteTemplate
from app.models.review_record import ReviewRecord
from app.models.user_model import User

debug_router = APIRouter(prefix="/debug", tags=["debug"])


# ─────────────────────────────── 工具函数 ───────────────────────────────


def _serialize_row(row, exclude_fields: set = None) -> dict:
    """将 ORM 行转为可序列化的字典，处理 datetime 和排除字段"""
    exclude_fields = exclude_fields or set()
    result = {}
    for col in row.__table__.columns:
        # 用 col.name 作为输出 key（数据库列名），用 col.key 读取 Python 属性值。
        # 例如 ChatSession 中 metadata_ = Column(JSON, name="metadata")，
        # col.name="metadata"，col.key="metadata_"。
        # 若用 col.name 做 getattr 会命中 Base.metadata（MetaData 对象）导致递归爆栈。
        if col.name in exclude_fields:
            continue
        value = getattr(row, col.key, None)
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        result[col.name] = value
    return result


# ─────────────────────────────── Redis ───────────────────────────────


async def _dump_redis(user_id: Optional[str] = None) -> dict:
    """遍历 Redis 全部 key，返回分类汇总"""
    try:
        redis_client = await connect_redis()
        await redis_client.ping()
    except Exception as e:
        return {"error": f"Redis 连接失败: {e}"}

    keys_data = []
    summary = defaultdict(int)

    async for key in redis_client.scan_iter(match="*", count=200):
        # user_id 过滤
        if user_id and user_id not in key:
            continue

        # 分类
        if key.startswith("blacklist:"):
            category = "blacklist"
        elif key.startswith("user:"):
            category = "user_cache"
        elif key.startswith("rate_limit:"):
            category = "rate_limit"
        else:
            category = "other"
        summary[category] += 1

        # 获取值和 TTL
        ttl = await redis_client.ttl(key)
        key_type = await redis_client.type(key)

        if key_type == "string":
            raw_value = await redis_client.get(key)
            # 尝试 JSON 解析
            try:
                value = json.loads(raw_value)
            except (json.JSONDecodeError, TypeError):
                value = raw_value
        elif key_type == "hash":
            value = await redis_client.hgetall(key)
        elif key_type == "list":
            value = await redis_client.lrange(key, 0, -1)
        elif key_type == "set":
            value = list(await redis_client.smembers(key))
        elif key_type == "zset":
            value = await redis_client.zrange(key, 0, -1, withscores=True)
        else:
            value = f"<unsupported type: {key_type}>"

        keys_data.append({
            "key": key,
            "type": key_type,
            "ttl": ttl if ttl >= 0 else ("永不过期" if ttl == -1 else "已过期"),
            "category": category,
            "value": value,
        })

    return {
        "total_keys": len(keys_data),
        "summary": dict(summary),
        "keys": keys_data,
    }


@debug_router.get("/redis")
async def debug_redis(user_id: Optional[str] = Query(None, description="按 user_id 过滤 key")):
    """输出 Redis 中所有 key 及其值"""
    data = await _dump_redis(user_id)
    return success_response(data=data)


# ─────────────────────────────── MySQL ───────────────────────────────


async def _dump_mysql(db: AsyncSession, user_id: Optional[str] = None) -> dict:
    """遍历所有 ORM 表，输出全部记录"""
    tables = {}
    total_records = 0

    # 定义表及其用户字段映射
    table_config = [
        (User, "uuid", {"password"}),  # User 用 uuid 字段过滤，排除 password
        (Note, "user_id", set()),
        (NoteTemplate, "user_id", set()),
        (ReviewRecord, "user_id", set()),
        (ChatSession, "user_id", set()),
        (ChatMessage, None, set()),  # ChatMessage 没有 user_id，通过 session 关联
    ]

    for model, user_field, exclude_fields in table_config:
        table_name = model.__tablename__
        try:
            stmt = select(model)

            # 用户过滤
            if user_id and user_field:
                stmt = stmt.where(getattr(model, user_field) == user_id)
            elif user_id and model == ChatMessage:
                # ChatMessage 通过 session_id 关联到用户
                user_sessions = await db.execute(
                    select(ChatSession.id).where(ChatSession.user_id == user_id)
                )
                session_ids = [row[0] for row in user_sessions.fetchall()]
                if session_ids:
                    stmt = stmt.where(ChatMessage.session_id.in_(session_ids))
                else:
                    tables[table_name] = {"count": 0, "records": []}
                    continue

            result = await db.execute(stmt)
            rows = result.scalars().all()

            records = [_serialize_row(row, exclude_fields) for row in rows]
            tables[table_name] = {
                "count": len(records),
                "records": records,
            }
            total_records += len(records)

        except Exception as e:
            tables[table_name] = {"error": str(e)}

    return {
        "total_records": total_records,
        "tables": tables,
    }


@debug_router.get("/mysql")
async def debug_mysql(
        user_id: Optional[str] = Query(None, description="按 user_id 过滤记录"),
        db: AsyncSession = Depends(get_db),
):
    """输出 MySQL 全部表数据"""
    data = await _dump_mysql(db, user_id)
    return success_response(data=data)


# ─────────────────────────────── ChromaDB / 知识库 ───────────────────────────────


async def _dump_knowledge(user_id: Optional[str] = None) -> dict:
    """输出 ChromaDB 知识库和笔记集合的全部内容"""
    from app.core.background_init import init_manager
    from app.rag.vector_store import VectorStoreService

    result = {}

    # ── RAG Collection ──
    try:
        if not VectorStoreService._initialized:
            result["rag_collection"] = {"status": "not_initialized", "message": "VectorStoreService 尚未初始化"}
        else:
            store = VectorStoreService()
            where_clause = {"user_id": user_id} if user_id else None

            all_docs = await asyncio.to_thread(
                store.vectors_store.get,
                include=["documents", "metadatas"],
                where=where_clause,
            )

            chunks = []
            docs_summary = defaultdict(lambda: {"chunk_count": 0, "user_id": None})

            for i, doc_id in enumerate(all_docs["ids"]):
                metadata = all_docs["metadatas"][i] if i < len(all_docs["metadatas"]) else {}
                content = all_docs["documents"][i] if i < len(all_docs["documents"]) else ""
                filename = metadata.get("original_filename", metadata.get("source", "unknown"))

                chunks.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                })

                docs_summary[filename]["chunk_count"] += 1
                docs_summary[filename]["user_id"] = metadata.get("user_id")

            result["rag_collection"] = {
                "collection_name": "rag_collection",
                "total_chunks": len(chunks),
                "documents_summary": [
                    {"filename": fname, **info} for fname, info in docs_summary.items()
                ],
                "chunks": chunks,
            }
    except Exception as e:
        result["rag_collection"] = {"error": str(e)}

    # ── Notes Collection ──
    try:
        if init_manager.note_service is None:
            result["notes_collection"] = {"status": "not_initialized", "message": "NoteService 尚未初始化"}
        else:
            notes_store = init_manager.note_service._notes_store
            where_clause = {"user_id": user_id} if user_id else None

            notes_docs = await asyncio.to_thread(
                notes_store.get,
                include=["documents", "metadatas"],
                where=where_clause,
            )

            notes_chunks = []
            for i, doc_id in enumerate(notes_docs["ids"]):
                metadata = notes_docs["metadatas"][i] if i < len(notes_docs["metadatas"]) else {}
                content = notes_docs["documents"][i] if i < len(notes_docs["documents"]) else ""
                notes_chunks.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                })

            result["notes_collection"] = {
                "collection_name": "notes_collection",
                "total_chunks": len(notes_chunks),
                "chunks": notes_chunks,
            }
    except Exception as e:
        result["notes_collection"] = {"error": str(e)}

    # ── MD5 Records ──
    try:
        if not VectorStoreService._initialized:
            result["md5_records"] = {"status": "not_initialized"}
        else:
            store = VectorStoreService()
            if user_id:
                records = await store.get_all_md5_records(user_id)
                result["md5_records"] = {user_id: records}
            else:
                # 扫描所有用户的 MD5 目录
                from app.rag.md5_manager import MD5Store
                md5_store = store.md5_store
                base_dir = os.path.join(md5_store.base_dir, "user_md5")
                all_records = {}
                if os.path.isdir(base_dir):
                    for uid in os.listdir(base_dir):
                        uid_records = await store.get_all_md5_records(uid)
                        if uid_records:
                            all_records[uid] = uid_records
                result["md5_records"] = all_records
    except Exception as e:
        result["md5_records"] = {"error": str(e)}

    return result


@debug_router.get("/knowledge")
async def debug_knowledge(user_id: Optional[str] = Query(None, description="按 user_id 过滤")):
    """输出 ChromaDB 知识库和笔记集合的全部内容"""
    data = await _dump_knowledge(user_id)
    return success_response(data=data)


# ─────────────────────────────── All ───────────────────────────────


@debug_router.get("/all")
async def debug_all(
        user_id: Optional[str] = Query(None, description="按 user_id 过滤"),
        db: AsyncSession = Depends(get_db),
):
    """聚合输出 Redis + MySQL + ChromaDB 全部数据"""
    # 并行获取三个数据源
    redis_task = asyncio.create_task(_dump_redis(user_id))
    mysql_task = asyncio.create_task(_dump_mysql(db, user_id))
    knowledge_task = asyncio.create_task(_dump_knowledge(user_id))

    redis_data, mysql_data, knowledge_data = await asyncio.gather(
        redis_task, mysql_task, knowledge_task, return_exceptions=True
    )

    data = {
        "redis": redis_data if not isinstance(redis_data, Exception) else {"error": str(redis_data)},
        "mysql": mysql_data if not isinstance(mysql_data, Exception) else {"error": str(mysql_data)},
        "knowledge": knowledge_data if not isinstance(knowledge_data, Exception) else {"error": str(knowledge_data)},
    }
    return success_response(data=data)
