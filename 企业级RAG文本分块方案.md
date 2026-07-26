# 企业级 RAG 文本分块方案

## 全景架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        文档分块流水线                                    │
│                                                                         │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐   ┌────────┐ │
│  │ 文档解析 │ → │ 结构识别  │ → │ 语义分块  │ → │ 上下文  │ → │ 向量化  │ │
│  │         │   │          │   │          │   │ 增强   │   │ 入库   │ │
│  └─────────┘   └──────────┘   └──────────┘   └────────┘   └────────┘ │
│                                                                         │
│  解析为纯文本    识别标题/段落     按语义边界切       补充上下文       多粒度存储  │
│  + 保留结构      /表格/代码块      不在句子中间断     防止信息孤岛     支持精确+模糊 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 核心设计：多层级 Chunk 体系

```
┌─────────────────────────────────────────┐
│           Level 0: 原始文档              │  存储完整文档（不做 embed）
├─────────────────────────────────────────┤
│           Level 1: 大块 (Parent)         │  按章节/段落切，800-1500 token
│           用于最终交给 LLM 阅读           │  保留完整语义单元
├─────────────────────────────────────────┤
│           Level 2: 小块 (Child)          │  200-400 token，精细切
│           用于向量检索匹配               │  检索精度高
└─────────────────────────────────────────┘

检索流程：
  query → embed → 匹配 Child chunk → 找到其 Parent → 把 Parent 送给 LLM

为什么？
  小块 embed 匹配更精准（噪声少）
  大块送给 LLM 上下文更完整（不会断章取义）
```

---

## 每个 Chunk 的完整结构

```json
{
  "chunk_id": "doc_001_chunk_003",
  "content": "缓存命中率 43.2%，平均缓存条目 12,847 条...",
  
  "context_prefix": "[文档：星辰引擎 Q2 指标报告 | 章节：二、记忆晶体缓存层指标]",
  "content_with_context": "[文档：星辰引擎 Q2 指标报告 | 章节：二、记忆晶体缓存层指标]\n缓存命中率 43.2%...",
  
  "metadata": {
    "source_file": "星辰引擎内部指标报告-Q2.txt",
    "parent_chunk_id": "doc_001_parent_002",
    "chunk_index": 3,
    "total_chunks": 8,
    "heading_path": ["星辰检索引擎 Q2 报告", "二、记忆晶体缓存层指标"],
    "prev_chunk_id": "doc_001_chunk_002",
    "next_chunk_id": "doc_001_chunk_004"
  },

  "embedding": [0.012, -0.034, ...]  // 对 content_with_context 做 embed
}
```

关键点：**embed 的是带上下文前缀的文本**，而非裸内容。

---

## 分阶段实施

### 阶段一：上下文注入（1-2 天）

**改动最小，效果最明显。**

```python
# 分块后，给每个 chunk 注入来源上下文
for chunk in chunks:
    prefix = f"[文档：{filename}]"
    if chunk.metadata.get("heading"):
        prefix += f" [章节：{chunk.metadata['heading']}]"
    chunk.page_content = f"{prefix}\n{chunk.page_content}"
```

**解决的问题**：
- "星辰引擎"检索不到缓存相关 chunk
- chunk 脱离上下文后语义模糊

**验证方法**：

| 步骤 | 操作 |
|------|------|
| 1 | 准备 20 条测试 query（覆盖"按文档名找""按章节找""按具体内容找"） |
| 2 | 分别用旧方案和新方案入库同一批文档 |
| 3 | 对比 Top-5 召回命中率（人工标注哪些 query 应该命中哪些 chunk） |
| 4 | 预期提升：召回率 +10~15% |

---

### 阶段二：语义分块（3-5 天）

**替换固定字符数切割，按语义边界切。**

当前项目用 `RecursiveCharacterTextSplitter`（按字符数 + 分隔符切），问题是：
- 可能把一个完整段落从中间切断
- 表格、代码块被切碎

改为：

```python
# 方案 A：基于 Markdown 结构的分块（适合本项目）
# 按 ## 标题作为切割边界，每个 section 是一个 chunk
# 超长 section 再用 RecursiveCharacterTextSplitter 二次切

# 方案 B：语义分块（通用）
# 逐句计算相邻句子的 embedding 相似度
# 相似度骤降处 = 话题转换点 = 分块边界
#
# 句子1 ─ 0.92 ─ 句子2 ─ 0.89 ─ 句子3 ─ 0.41 ─ 句子4
#                                         ↑ 这里切
```

LangChain 已有实现：`SemanticChunker`（按 embedding 相似度断句）和 `MarkdownHeaderTextSplitter`（按标题层级）。

**验证方法**：

| 指标 | 怎么测 |
|------|--------|
| chunk 完整性 | 抽样 50 个 chunk，人工判断是否"读起来是完整的一段话" |
| 碎片化率 | 同一个 query 命中的 chunks 是否来自同一段落（应该是） |
| 召回率 | 同阶段一的 20 条 query 重新测，对比提升幅度 |

---

### 阶段三：Parent-Child 双层索引（5-7 天）

**架构性改动，效果最好但工作量最大。**

```
存储结构:
┌─────────────────────────────────────────┐
│  ChromaDB collection: rag_chunks        │  ← Child 小块，用于检索
│  ChromaDB collection: rag_parents       │  ← Parent 大块，用于送给 LLM
│  MySQL table: chunk_relations           │  ← 记录 parent-child 映射关系
└─────────────────────────────────────────┘

检索流程:
  query
    │
    ▼
  在 rag_chunks 中向量匹配 → 命中 child_chunk_007
    │
    ▼
  通过 chunk_relations 找到 parent_chunk_id → parent_003
    │
    ▼
  从 rag_parents 取出 parent_003 的完整内容 → 送给 LLM 生成回答
```

**核心逻辑**：

```python
# 入库时
parent_chunks = split_by_section(document, max_tokens=1200)  # 大块
for parent in parent_chunks:
    child_chunks = split_further(parent, max_tokens=300)      # 小块
    store_parent(parent)
    for child in child_chunks:
        child.metadata["parent_id"] = parent.id
        store_child(child)  # embed 并存入向量库

# 检索时
child_results = vector_search(query, collection="rag_chunks", top_k=5)
parent_ids = set(c.metadata["parent_id"] for c in child_results)
parent_docs = fetch_parents(parent_ids)  # 拿完整大块送给 LLM
```

**验证方法**：

| 维度 | 测量方式 |
|------|----------|
| 检索精度 | 同样的 query，对比 child 检索 vs 直接用大块检索的 Top-5 命中率 |
| 回答质量 | 准备 10 个需要上下文才能回答的问题（如"记忆晶体的 TTL 和命中率分别是多少"），对比有无 Parent 时 LLM 回答的完整度 |
| 延迟 | 增加一次 Parent 查询约 +5-10ms，可接受范围内 |

---

### 阶段四：智能 Overlap + 前后文链接（2-3 天）

```
chunk 1: [文档：xxx | 章节：yyy]
          ████████████████████
                        ████████████████████  ← chunk 2（overlap 20%）
                                      ████████████████████  ← chunk 3

+ 每个 chunk 存 prev_chunk_id / next_chunk_id
+ 检索命中后可选择性扩展相邻 chunk（滑动窗口）
```

**验证方法**：
- 测试"答案横跨两个 chunk 边界"的 query（如"缓存命中率和内存占用"分别在两个 chunk 里）
- 对比有无 overlap 时能否同时检索到两块

---

### 阶段五：文档类型感知分块（持续迭代）

不同类型文档用不同分块策略：

| 文档类型 | 分块策略 |
|----------|----------|
| Markdown/文档 | 按标题层级切，保留 heading path |
| 代码文件 | 按函数/类切，保留完整代码块 |
| 会议纪要 | 按议题/行动项切 |
| 表格类 | 每行或每组行作为一个 chunk，表头作为 context |
| PDF（扫描件） | 按页切 + OCR 后再语义分块 |

---

## 各阶段投入产出对比

```
效果提升
  ↑
  │                              ★ 阶段三 (Parent-Child)
  │                        ★ 阶段五 (类型感知)
  │                  ★ 阶段二 (语义分块)
  │            ★ 阶段四 (Overlap)
  │      ★ 阶段一 (上下文注入)
  │
  └──────────────────────────────────────→ 实施复杂度
```

| 阶段 | 性价比 | 建议 |
|------|--------|------|
| 阶段一 | ⭐⭐⭐⭐⭐ | **立即做**，改动 10 行代码，效果立竿见影 |
| 阶段二 | ⭐⭐⭐⭐ | 推荐做，LangChain 有现成工具 |
| 阶段三 | ⭐⭐⭐ | 效果最好但改动大，看业务需要 |
| 阶段四 | ⭐⭐⭐ | 和阶段二配合做，边际成本低 |
| 阶段五 | ⭐⭐ | 持续迭代，按实际文档类型逐个适配 |

---

## 通用评估框架（每个阶段都用）

```python
# 评估数据集格式
test_cases = [
    {
        "query": "记忆晶体的缓存命中率是多少",
        "expected_chunks": ["doc_001_chunk_003"],   # 应该命中哪些 chunk
        "expected_answer": "43.2%",                  # 最终答案应包含
    },
    ...
]

# 核心指标
Recall@K     = 命中的相关 chunk 数 / 总相关 chunk 数   （检索全不全）
Precision@K  = 命中的相关 chunk 数 / 返回的 chunk 数    （检索准不准）
Answer Score = LLM 最终回答是否包含 expected_answer     （端到端效果）
```

建议从 20-50 条手工标注的测试 query 起步，每个阶段跑一遍对比分数变化。
