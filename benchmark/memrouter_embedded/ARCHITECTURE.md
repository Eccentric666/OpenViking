# MemRouter Embedded in OpenViking — Architecture

> **Note**: The evaluation suite and OV-side implementation live in
> `D:\Code\cursorProject\OpenViking\benchmark\memrouter_embedded\`.

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

### After (Embedded MemRouter)

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
  │     → 若 graph/streamlined/llm_fallback:
  │           → SearchService.search()               [native OV， fallback]
  │
  └── /search/search_memory → 同上
```

**Improvement**: MemRouter 路由发生在 OV Server 内部，消除了 HTTP 回传。

## 2. 完整处理流程（从 VikingBot 接收到回答）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  VikingBot Agent                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  1. Agent receives: "What does Caroline like to read?"                       │
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
│  │    - strips VikingBot prefix                                          │    │
│  │    - lowercases, collapses whitespace                                │    │
│  │    - anonymizes person names (Caroline → PERSON)                      │    │
│  │    → normalized_query = "What does PERSON like to read?"             │    │
│  │                                                                       │    │
│  │  Stage 2: QueryFeatureBuilder.build()                                │    │
│  │    - embeds normalized query via text-embedding-v3                   │    │
│  │    - extracts entities: {PERSON}                                      │    │
│  │    - extracts temporal hints, relation hints                          │    │
│  │    → QueryFeatures(embedding, entities, temporal_hints, ...)          │    │
│  │                                                                       │    │
│  │  Stage 3: TemplateMatcher.match()                                     │    │
│  │    - scores query against all template prototypes                    │    │
│  │    - multi-prototype scoring:                                        │    │
│  │        S_pos = 0.50*S_max + 0.30*S_mean@3 + 0.20*S_centroid          │    │
│  │    - hard negative penalty:                                          │    │
│  │        S_final = S_pos - λ*max(0, S_neg_best - M_neg)               │    │
│  │    - top template: personal_fact_lookup (confidence=0.92)            │    │
│  │                                                                       │    │
│  │  Stage 4: RouteDecision.decide()                                     │    │
│  │    - backend = openviking_memory_backend                             │    │
│  │    - skip_intent_analysis = true (template says so)                  │    │
│  │    - search_mode = find                                              │    │
│  │    → MemBackendRouteResult( routes=[RouteEntry(...)] )              │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  执行决策                                                             │    │
│  │                                                                       │    │
│  │  if backend == "openviking_memory_backend" and skip_intent_analysis: │    │
│  │       → execute_instruction()  [fast path, bypass IntentAnalyzer]    │    │
│  │       (直接调用 SearchService.execute_instruction()，无 VLM 调用)    │    │
│  │                                                                       │    │
│  │  elif backend in ("graph_memory_backend", "streamlined_memory_backend"):│    │
│  │       → SearchService.search()  [fallback to native OV]              │    │
│  │                                                                       │    │
│  │  else:  # llm_fallback or route error                                │    │
│  │       → SearchService.search()  [fallback to native OV]              │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  SearchService.execute_instruction(instruction, ctx)                       │
│       │                                                                      │
│       ▼                                                                      │
│  IntentAnalyzer.analyze() (only if NOT fast path)                          │
│       │                                                                      │
│       ▼                                                                      │
│  MemoryStore.search(query, search_mode, target_uri, limit)                  │
│       │                                                                      │
│       ▼                                                                      │
│  返回检索结果 (memories with scores)                                        │
│       │                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTP 200 (with memories)
┌─────────────────────────────────────────────────────────────────────────────┐
│  VikingBot Agent                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  3. Receives response with relevant_memories                                │
│  4. Agent analyzes memories + generates answer                             │
│  5. Returns answer to user                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fast Path vs. Fallback 对比

| 路径 | IntentAnalyzer | VLM 调用 | 适用场景 |
|------|---------------|---------|---------|
| **Fast path** (template hit + skip_intent_analysis=true) | ❌ 不调用 | ❌ 无 | 高confidence模板匹配（personal fact lookup等） |
| **Fallback** (graph/streamlined/llm_fallback/no template hit) | ✅ 调用 | ✅ 有 | 无模板匹配时兜底 |

## 2. 组件职责

| 组件 | 文件 | 职责 |
|------|------|------|
| **MemRouterService** | `openviking/service/memrouter_service.py` | 封装 MemRouter Pipeline，提供 `route()` 和 `search()` |
| **MemRouterPipeline** | `openviking/memrouter/pipeline.py` | 4阶段流水线：normalize → feature → match → decide → build |
| **QueryNormalizer** | `openviking/memrouter/normalizer.py` | 文本归一化、前缀剥离、人名匿名化 |
| **QueryFeatureBuilder** | `openviking/memrouter/features.py` | Embedding 生成 + 实体/时间 hints 提取 |
| **TemplateMatcher** | `openviking/memrouter/matcher.py` | 多原型向量评分 + hard negative 惩罚 |
| **RouteDecision** | `openviking/memrouter/decision.py` | 阈值判断 + LLM fallback 分发 |
| **QueryInstructionBuilder** | `openviking/memrouter/query_instruction_builder.py` | 路由结果 → 查询指令转换 |
| **BackendRouteTemplateIndex** | `openviking/memrouter/templates.py` | YAML 模板加载与索引 |
| **SearchService** | `openviking/service/search_service.py` | 集成 MemRouterService，自动路由查询 |
| **OpenVikingService** | `openviking/service/core.py` | 生命周期管理：初始化 MemRouterService |
| **VikingClient** | `bot/vikingbot/agent/...` | 普通 HTTP client，透明调用 |

## 3. 配置

### ov.conf

```json
{
  "memrouter": {
    "enabled": true,
    "enabled_backends": ["openviking_memory_backend"],
    "route_events_path": "D:/Code/cursorProject/OpenViking/benchmark/memrouter_embedded/logs/route_events.jsonl"
  }
}
```

> 注：`echomem_path` 已被移除。MemRouter 核心代码已完整复制到 OV 代码仓中。

### 自动派生配置

| MemRouter 配置 | 来源 |
|----------------|------|
| `embedding.api_key` | `embedding.dense.api_key` (OV) |
| `embedding.api_base` | `embedding.dense.api_base` (OV) |
| `embedding.model` | `embedding.dense.model` (OV) |
| `llm_fallback.api_key` | `vlm.api_key` (OV) |
| `llm_fallback.api_base` | `vlm.api_base` (OV) |
| `llm_fallback.model` | `vlm.model` (OV) |

### 环境变量（不再需要）

| 变量 | 状态 | 说明 |
|------|------|------|
| `MEMROUTER_ENABLED` | ❌ 移除 | 由 `memrouter.enabled` 替代 |
| `ECHOMEM_PATH` | ❌ 移除 | MemRouter 代码已合入 OV |
| `MEMROUTER_CONFIG` | ❌ 移除 | 配置合并到 `ov.conf` |
| `MEMROUTER_ROUTE_EVENTS` | ❌ 移除 | 由 `memrouter.route_events_path` 替代 |

## 4. 路由事件格式

MemRouterService 将每次路由决策写入 JSONL：

```json
{
  "timestamp": "2026-06-01T21:30:00",
  "query": "What does Caroline like to read?",
  "execution_path": "memrouter_fast_path",
  "latency_ms": 45,
  "route_method": "template_embedding",
  "backend_id": "openviking_memory_backend",
  "template_id": "personal_fact_lookup",
  "confidence": 0.92,
  "debug": {
    "top_templates": [...]
  }
}
```

## 5. Fallback 策略

| MemRouter 路由结果 | 实际执行 |
|-------------------|---------|
| `openviking_memory_backend` + `skip_intent_analysis=true` | `execute_instruction()` fast path |
| `openviking_memory_backend` + `skip_intent_analysis=false` | native `search()` |
| `graph_memory_backend` | native `search()` (fallback) |
| `streamlined_memory_backend` | native `search()` (fallback) |
| LLM fallback (no confident match) | native `search()` |
| Route error | native `search()` |

## 6. 评测指标

| 指标 | 计算方式 |
|------|---------|
| `template_hit_rate` | template 命中题数 / 有路由事件的题数 |
| `backend_accuracy` | 路由正确的题数 / 有标签的题数 |
| `llm_fallback_rate` | LLM fallback 题数 / 有路由事件的题数 |
| `answer_accuracy` | Judge 判定正确数 / 总题数 |

## 7. 评测脚本

| 文件 | 说明 |
|------|------|
| `benchmark/memrouter_embedded/scripts/eval_locomo_ov_with_memrouter_e2e.py` | 主评测脚本 |
| `benchmark/memrouter_embedded/scripts/run_e2e.ps1` | 一键启动脚本 |
| `benchmark/memrouter_embedded/config/ov.conf` | OV 服务配置（含 memrouter 节） |
| `benchmark/memrouter_embedded/data/locomo_e2e_route_labels.v3.jsonl` | 路由标签 |

运行示例：

```powershell
# 小规模验证（5 题）
.\benchmark\memrouter_embedded\scripts\run_e2e.ps1 -LimitQuestions 5 -ForceMemorySearch

# 全量 Cat3
.\benchmark\memrouter_embedded\scripts\run_e2e.ps1 -Category 3 -ForceMemorySearch -Judge -JudgeToken "sk-xxxxx"
```