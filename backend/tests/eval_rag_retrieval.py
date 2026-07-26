"""
RAG 检索效果评估脚本

用法:
  1. 先通过前端或 API 上传 tmp/ 下的测试文档到知识库
  2. 运行: cd backend && uv run python tests/eval_rag_retrieval.py

输出: 每条 query 的命中情况 + 整体 Recall@K / Precision@K 报告
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.core.background_init import init_manager
from app.rag.rag_service import RagService


# ============================================================
# 测试数据集
# 每条包含: query, expected_keywords(回答中应包含的关键词), expected_source(应命中的文件名片段)
# ============================================================
TEST_CASES = [
    {
        "query": "记忆晶体的缓存命中率是多少",
        "expected_keywords": ["43.2%"],
        "expected_source": "星辰引擎",
    },
    {
        "query": "星辰引擎的P99延迟",
        "expected_keywords": ["318ms"],
        "expected_source": "星辰引擎",
    },
    {
        "query": "凤凰计划下次会议什么时候",
        "expected_keywords": ["7/22", "7-22", "7月22"],
        "expected_source": "凤凰计划",
    },
    {
        "query": "时光回溯功能是什么",
        "expected_keywords": ["时间", "历史", "版本", "快照"],
        "expected_source": "术语表",
    },
    {
        "query": "刘宏义负责什么",
        "expected_keywords": ["时光回溯", "后端", "API"],
        "expected_source": "凤凰计划",
    },
    {
        "query": "碎片化率的阈值是多少",
        "expected_keywords": ["40%"],
        "expected_source": "术语表",
    },
    {
        "query": "星辰引擎日均请求量",
        "expected_keywords": ["847"],
        "expected_source": "星辰引擎",
    },
    {
        "query": "幻觉指数是什么意思",
        "expected_keywords": ["LLM", "不存在", "引用"],
        "expected_source": "术语表",
    },
    {
        "query": "tantivy报错了怎么回事",
        "expected_keywords": ["段错误", "macOS", "ARM"],
        "expected_source": "凤凰计划",
    },
    {
        "query": "prod环境的机器配置",
        "expected_keywords": ["8C16G", "A100"],
        "expected_source": "术语表",
    },
]


async def run_evaluation(user_id: str):
    """运行评估"""
    # 等待后台初始化完成
    print("⏳ 等待后台资源初始化...")
    await init_manager.start()
    await init_manager.models_ready.wait()
    await init_manager.note_service_ready.wait()
    print("✅ 初始化完成\n")

    total_recall_hits = 0
    total_recall_expected = 0
    total_precision_hits = 0
    total_precision_returned = 0

    results = []

    for i, case in enumerate(TEST_CASES, 1):
        query = case["query"]
        expected_keywords = case["expected_keywords"]
        expected_source = case["expected_source"]

        print(f"── 测试 {i}/{len(TEST_CASES)}: {query}")

        # 调用检索
        service = RagService(user_id=user_id)
        documents = await service.retrieve_document(query)

        # 判断是否命中了预期来源
        source_hit = False
        keyword_hit = False
        matched_content = ""

        for doc in documents:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)

            # 检查来源匹配
            if expected_source.lower() in content.lower():
                source_hit = True

            # 检查关键词匹配（任一关键词出现即算命中）
            for kw in expected_keywords:
                if kw in content:
                    keyword_hit = True
                    matched_content = content[:100]
                    break

        # 统计
        is_hit = source_hit and keyword_hit
        total_recall_expected += 1
        if is_hit:
            total_recall_hits += 1

        # Precision: 返回的文档中有多少是相关的（简化为：命中=相关）
        total_precision_returned += len(documents)
        relevant_count = sum(
            1 for doc in documents
            if expected_source.lower() in (doc.page_content if hasattr(doc, 'page_content') else str(doc)).lower()
        )
        total_precision_hits += relevant_count

        status = "✅" if is_hit else "❌"
        print(f"   {status} 来源命中: {'是' if source_hit else '否'} | 关键词命中: {'是' if keyword_hit else '否'} | 返回文档数: {len(documents)}")
        if matched_content:
            print(f"   📄 匹配内容: {matched_content}...")
        print()

        results.append({
            "query": query,
            "source_hit": source_hit,
            "keyword_hit": keyword_hit,
            "is_hit": is_hit,
            "documents_returned": len(documents),
        })

    # 汇总报告
    recall = total_recall_hits / total_recall_expected if total_recall_expected > 0 else 0
    precision = total_precision_hits / total_precision_returned if total_precision_returned > 0 else 0

    print("=" * 60)
    print("📊 评估报告")
    print("=" * 60)
    print(f"   测试用例数:    {len(TEST_CASES)}")
    print(f"   完全命中:      {total_recall_hits}/{total_recall_expected}")
    print(f"   Recall@K:      {recall:.1%}")
    print(f"   Precision@K:   {precision:.1%}")
    print(f"   (Precision = 返回文档中来源匹配的比例)")
    print("=" * 60)

    # 输出到文件
    report = {
        "total_cases": len(TEST_CASES),
        "recall_hits": total_recall_hits,
        "recall_total": total_recall_expected,
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "details": results,
    }
    report_path = os.path.join(os.path.dirname(__file__), "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📁 详细报告已保存: {report_path}")


if __name__ == "__main__":
    # 默认使用测试用户 ID，可通过参数覆盖
    user_id = sys.argv[1] if len(sys.argv) > 1 else "ea38ac17cfa54cefa8264d24"
    print(f"🧪 RAG 检索评估 (user_id: {user_id})\n")
    asyncio.run(run_evaluation(user_id))
