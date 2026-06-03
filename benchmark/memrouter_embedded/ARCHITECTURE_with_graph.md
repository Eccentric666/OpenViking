# MemRouter Embedded in OpenViking — Architecture (with Graph Memory)

> **Note**: The evaluation suite and OV-side implementation live in
> `D:\Code\cursorProject\OpenViking\benchmark\memrouter_embedded\`.
>
> **变更摘要**：本文档在基础版架构之上，增补了 **Graph Memory（图记忆）** 作为第三条物理接入的后端。Graph 后端基于 Neo4j 知识图谱，负责实体关系、多跳推理、共现查询等场景。

---

## 1. 端到端链路图

### Before (SDK Wrapper Pattern)

```
VikingBot AgentLoop
  ├── Layer 0: memory.py
  │     → MemRouterVikingClient.search_memory()   [SDK wrapper in VikingBot]
  │           → MemRouterPipeline.route()
  │           → HTTP POST /api/v1/search/execute_instruction   [HTTP back to OV]
  │
  └── Layer 1: tools/ov_file.py
        → MemRouterVikingClient.search()          [SDK wrapper in VikingBot]
              → MemRouterPipeline.route()
              → HTTP POST /api/v1/search/execute_instruction   [HTTP back to OV]

OpenViking Server
  └── /execute_instruction → SearchService.execute_instruction()
```

**Problem**: OV → SDK Wrapper → MemRouter → HTTP → OV (round-trip)

### After (Embedded MemRouter + Graph Backend)

```
VikingBot AgentLoop (transparent， 无 MemRouter 感知)
  ├── Layer 0: memory.py
  │     → VikingClient.search_memory(query)
  │           → HTTP POST /api/v1/search/search_memory
  │
  └── Layer 1: tools/ov_file.py
        → VikingClient.search(query)
              → HTTP POST /api/v1/search/search

OpenViking Server（MemRouter 逻辑在此）
  ├── /search/search → SearchService.search()
  │     → MemRouterService.route(query)           [进程内调用]
  │     → 若 OV backend + skip_intent_analysis:
  │           → SearchService.execute_instruction()  [fast path， 无需 VLM]
  │     → 若 graph backend + skip_intent_analysis:
  │           → SearchService.search_graph_text()    [graph fast path， 返回 NL text]
  │     → 若 streamlined backend + skip_intent_analysis:
  │           → StreamlinedMemoryService.recall_state() [streamlined fast path]
  │     → 若 llm_fallback / no template hit / route error:
  │           → SearchService.search()               [native OV， fallback]
  │
  └── /search/search_memory → 同上

Graph Memory Sidecar (Neo4j)
  └── 由 SearchService.search_graph_text() 内部调用
      → GraphManager.search() → Neo4jBackend 向量相似度 + 多跳子图检索
```

**Improvement**:
1. MemRouter 路由发生在 OV Server 内部，消除了 HTTP 回传。
2. **Graph Backend** 物理接入：高 confidence 的 entity-relation / multi-hop 查询直接命中 Neo4j，绕过原生向量检索的语义稀释问题。
3. **Streamlined Backend** 预留：模板已就绪，待 Sidecar 完整部署后接入。

---

## 2. 完整处理流程（从 VikingBot 接收到回答）

### 2.1 主流程（OV / Streamlined / Graph 统一入口）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  VikingBot Agent                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  1. Agent receives: "What is the relationship between Caroline and Melanie?" │
│  2. Calls memory.search_memory(query)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTP POST /api/v1/search/search_memory
┌─────────────────────────────────────────────────────────────────────────────┐
│  OpenViking Server                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  SearchService.search_memory(query, ctx)                                    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  MemRouterService.search(query, ctx)                                 │    │
│  │                                                                       │    │
│  │  Stage 1: route(query)                                               │    │
│  │    │                                                                  │    │
│  │    ▼                                                                  │    │
│  │  QueryNormalizer.normalize()                                         │    │
│  │    → normalized_query = "what is the relationship between PERSON     │    │
│  │      and PERSON ?"                                                    │    │
│  │                                                                       │    │
│  │  Stage 2: QueryFeatureBuilder.build()                                │    │
│  │    → entities: {PERSON_0: "Caroline", PERSON_1: "Melanie"}           │    │
│  │    → relation_hints: ["relationship"]                                 │    │
│  │                                                                       │    │
│  │  Stage 3: TemplateMatcher.match()                                     │    │
│  │    → top template: graph.entity_relation.v1 (confidence=0.89)        │    │
│  │                                                                       │    │
│  │  Stage 4: RouteDecision.decide()                                     │    │
│  │    → backend = graph_memory_backend                                   │    │
│  │    → skip_intent_analysis = true                                      │    │
│  │    → search_mode = graph_traversal                                    │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  执行决策                                                             │    │
│  │                                                                       │    │
│  │  if backend == "openviking_memory_backend" and skip_intent_analysis: │    │
│  │       → execute_instruction()  [fast path, bypass IntentAnalyzer]    │    │
│  │                                                                       │    │
│  │  elif backend == "graph_memory_backend":                             │    │
│  │       → search_graph_text(query, ctx)  [graph fast path]             │    │
│  │       (直接调用 GraphManager.search()，返回自然语言描述的实体关系文本)│    │
│  │                                                                       │    │
│  │  elif backend == "streamlined_memory_backend":                       │    │
│  │       → StreamlinedMemoryService.recall_state() [streamlined fast]   │    │
│  │                                                                       │    │
│  │  else:  # llm_fallback or route error                                │    │
│  │       → SearchService.search()  [fallback to native OV]              │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  SearchService.search_graph_text(query, ctx, top_k=10)                      │
│       │                                                                      │
│       ▼                                                                      │
│  GraphManager.search(query, account_id, user_id, top_k)                     │
│       │                                                                      │
│       ▼                                                                      │
│  Neo4jBackend: 向量相似度检索实体 → 多跳关系遍历 → 子图组装                │
│       │                                                                      │
│       ▼                                                                      │
│  GraphRetrievalFormatter.to_natural_language(results)                       │
│       │                                                                      │
│       ▼                                                                      │
│  返回自然语言文本（如："Caroline 和 Melanie 是朋友关系，一起参加了..."）   │
│       │                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTP 200 (with graph_text)
┌─────────────────────────────────────────────────────────────────────────────┐
│  VikingBot Agent                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  3. Receives response with graph_text block                                │
│  4. Agent analyzes graph_text + generates answer                           │
│  5. Returns answer to user                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Graph Backend 内部数据流

```
User Query: "What is the relationship between Caroline and Melanie?"
    │
    ▼
GraphManager.search()
    │
    ├── 1. Entity Extraction (embedding-based vector search in Neo4j)
    │      → 找到 Caroline (Node) 和 Melanie (Node)
    │
    ├── 2. Relation Traversal (Cypher 多跳查询)
    │      → MATCH (a:Node {name:'Caroline'})-[r*1..2]-(b:Node {name:'Melanie'})
    │      → 返回路径：Caroline -[:friend_of]-> Melanie
    │                Caroline -[:attended]-> Event_42 <-[:attended]- Melanie
    │
    ├── 3. Subgraph Assembly
    │      → 合并所有路径为结构化结果列表
    │
    └── 4. NL Formatting
           → GraphRetrievalFormatter 转换为人类可读文本
           → "Caroline 和 Melanie 是朋友关系。两人于 2023-05-08 一起参加了
              LGBTQ 支持小组活动..."
```

---

## 3. Fast Path vs. Fallback 对比

| 路径 | IntentAnalyzer | VLM 调用 | 物理后端 | 适用场景 |
|------|---------------|---------|---------|---------|
| **OV fast path** | ❌ 不调用 | ❌ 无 | OpenViking Native (Qdrant/AGFS) | personal fact lookup, preference profile 等高 confidence OV 模板 |
| **Graph fast path** | ❌ 不调用 | ❌ 无 | Neo4j (GraphManager) | entity relation, causal multihop, system dependency 等高 confidence graph 模板 |
| **Streamlined fast path** | ❌ 不调用 | ❌ 无 | SQLite Sidecar (:1944) | timeline fact, duration comparison, sequence reasoning (待 Sidecar 完整部署) |
| **Fallback** | ✅ 调用 | ✅ 有 | Native OV search | 无模板匹配、低 confidence、LLM fallback |

---

## 4. 组件职责（含 Graph）

| 组件 | 文件 | 职责 |
|------|------|------|
| **MemRouterService** | `openviking/service/memrouter_service.py` | 封装 MemRouter Pipeline；新增 graph backend 执行分支，调用 `search_graph_text()` |
| **MemRouterPipeline** | `openviking/memrouter/pipeline.py` | 4阶段流水线；已注册 `graph_memory_backend` |
| **GraphAdapter** | `openviking/memrouter/adapters/graph.py` | 将 MemRouter 路由结果翻译为 `POST /api/v1/graph/search/text` 参数；支持 `root_entity_hint` 注入 |
| **SearchService** | `openviking/service/search_service.py` | 集成 `search_graph_text()`；GraphManager 注入；fallback 格式化 |
| **GraphManager** | `openviking/storage/graphdb/graph_manager.py` | Neo4j 生命周期管理；检索代理（向量 + 多跳） |
| **Neo4jBackend** | `openviking/storage/graphdb/neo4j_backend.py` | Neo4j 驱动封装；Cypher 执行 |
| **GraphHandler** | `openviking/storage/graphdb/graph_handler.py` | 异步队列消费：记忆写入 → 实体抽取 → 关系抽取 → 图写入 |
| **GraphRetriever** | `openviking/storage/graphdb/retrieval/graph_retriever.py` | 社区搜索、实体搜索、子图获取 |
| **EntityExtractor / RelationExtractor** | `openviking/storage/graphdb/extractors/` | LLM 驱动的实体/关系抽取 |
| **GraphWriter / Deduplicator** | `openviking/storage/graphdb/writers/` | 图数据写入 + 实体去重合并 |
| **Graph Router** | `openviking/server/routers/graph.py` | `/search`, `/search/text`, `/neighbors`, `/status` HTTP API |
| **OpenVikingService** | `openviking/service/core.py` | 生命周期管理：初始化 GraphManager、挂载 GraphQueue |
| **VikingClient (Bot)** | `bot/vikingbot/openviking_mount/ov_server.py` | 暴露 `search_graph_text()`；`memory.py` 中集成 graph 调用 |
| **QueryInstructionBuilder** | `openviking/memrouter/query_instruction_builder.py` | 路由结果 → 查询指令转换；graph 后端生成 `graph_traversal` 指令 |

---

## 5. 配置

### ov.conf

```json
{
  "memrouter": {
    "enabled": true,
    "enabled_backends": ["openviking_memory_backend", "graph_memory_backend"],
    "route_events_path": "D:/Code/cursorProject/OpenViking/benchmark/memrouter_embedded/logs/route_events.jsonl"
  },
  "graph_db": {
    "enabled": true,
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "your_password",
    "database": "neo4j",
    "confidence_threshold": 0.8,
    "similarity_threshold": 0.7
  }
}
```

> **注意**：`streamlined_memory_backend` 目前仅在 `enabled_backends` 中预留注册位，物理 Sidecar 完整部署后方可开启。

### 自动派生配置

| MemRouter 配置 | 来源 |
|----------------|------|
| `embedding.api_key` | `embedding.dense.api_key` (OV) |
| `embedding.api_base` | `embedding.dense.api_base` (OV) |
| `embedding.model` | `embedding.dense.model` (OV) |
| `llm_fallback.api_key` | `vlm.api_key` (OV) |
| `llm_fallback.api_base` | `vlm.api_base` (OV) |
| `llm_fallback.model` | `vlm.model` (OV) |

---

## 6. 路由事件格式（Graph 扩展）

MemRouterService 将每次路由决策写入 JSONL，graph 命中时扩展如下字段：

```json
{
  "timestamp": "2026-06-02T14:30:00",
  "query": "What is the relationship between Caroline and Melanie?",
  "execution_path": "memrouter_graph_fast_path",
  "latency_ms": 38,
  "route_method": "template_embedding",
  "backend_id": "graph_memory_backend",
  "template_id": "graph.entity_relation.v1",
  "confidence": 0.89,
  "physical_backend": "neo4j_graph",
  "debug": {
    "top_templates": [...],
    "graph_params": {
      "top_k": 10,
      "root_entity_hint": "Caroline"
    }
  }
}
```

---

## 7. Fallback 策略

| MemRouter 路由结果 | 实际执行 | 说明 |
|-------------------|---------|------|
| `openviking_memory_backend` + `skip_intent_analysis=true` | `execute_instruction()` fast path | 原生 OV 绕过 IntentAnalyzer |
| `openviking_memory_backend` + `skip_intent_analysis=false` | native `search()` | 走完整 OV 检索链路 |
| `graph_memory_backend` + `skip_intent_analysis=true` | `search_graph_text()` graph fast path | 直接命中 Neo4j，返回 NL text |
| `graph_memory_backend` + GraphManager 未初始化/异常 | native `search()` | fail-open：降级到 OV 向量检索 |
| `streamlined_memory_backend` + `skip_intent_analysis=true` | `StreamlinedMemoryService.recall_state()` | Sidecar 召回（待完整部署） |
| `streamlined_memory_backend` + Sidecar 未就绪 | native `search()` | fail-open |
| LLM fallback (no confident match) | native `search()` | 兜底 |
| Route error | native `search()` | 异常兜底 |

---

## 8. 评测指标

| 指标 | 计算方式 |
|------|---------|
| `template_hit_rate` | template 命中题数 / 有路由事件的题数 |
| `backend_accuracy` | 路由正确的题数 / 有标签的题数 |
| `llm_fallback_rate` | LLM fallback 题数 / 有路由事件的题数 |
| `answer_accuracy` | Judge 判定正确数 / 总题数 |
| **graph_hit_rate** *(新增)* | graph backend 命中题数 / 总题数 |
| **graph_answer_accuracy** *(新增)* | graph fast path 回答正确数 / graph 命中题数 |

---

## 9. 评测脚本

| 文件 | 说明 |
|------|------|
| `benchmark/memrouter_embedded/scripts/eval_locomo_ov_with_memrouter_e2e.py` | 主评测脚本 |
| `benchmark/memrouter_embedded/scripts/run_e2e.ps1` | 一键启动脚本 |
| `benchmark/memrouter_embedded/config/ov.conf` | OV 服务配置（含 memrouter + graph_db 节） |
| `benchmark/memrouter_embedded/data/locomo_e2e_route_labels.v3.jsonl` | 路由标签（含 graph 标签） |

运行示例：

```powershell
# 小规模验证（5 题）
.\benchmark\memrouter_embedded\scripts\run_e2e.ps1 -LimitQuestions 5 -ForceMemorySearch

# 全量 Cat3，含 Judge
.\benchmark\memrouter_embedded\scripts\run_e2e.ps1 -Category 3 -ForceMemorySearch -Judge -JudgeToken "sk-xxxxx"
```

---

## 10. Graph 数据写入链路（独立运行，与检索解耦）

Graph Memory 的写入不依赖 MemRouter，由 OpenViking Core 在后台自动完成：

```
Session.commit()
    │
    ▼
SessionService.commit()
    │
    ├── archive_uri (原有逻辑)
    │
    └── QueueManager.enqueue(GraphMsg)
              │
              ▼
        GraphHandler.dequeue()
              │
              ├── EntityExtractor.extract()  → RawEntity[]
              ├── RelationExtractor.extract() → RawRelation[]
              │
              └── GraphWriter.write()
                        │
                        ├── NodeDeduplicator.merge()
                        └── Neo4jBackend.execute(Cypher)
```

> **设计原则**：写入（Build）与检索（Query）解耦。Graph 构建是后台异步任务，MemRouter 只负责查询时的路由决策。
