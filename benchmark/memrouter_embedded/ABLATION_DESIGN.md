# LoCoMo conv-30 (without cat5) 消融实验设计文档

**日期**: 2026-06-03
**作者**: Claude Code
**版本**: v1.1

---

## 1. 实验目标

对比 **Native OV（Baseline）** vs **OV + MemRouter + Graph（Treatment）** 在 LoCoMo conv-30 (without cat5, 81 questions) 上的表现差异，验证 MemRouter 在以下维度的价值：

- Token 使用量节省（通过 DeepSeek 官网统一统计）
- 模板命中率
- 路由后端正确率
- 回答问题成功率

---

## 2. 数据集

| 属性 | 值 |
|------|-----|
| 数据集 | LoCoMo conv-30 (without cat5) |
| 问题总数 | **81** |
| 数据文件 | `data/conv30_exclude_cat5.json` |
| 标签文件 | `data/locomo_e2e_route_labels.v4.jsonl` |
| 标签分布 (v4) | 49 openviking + 32 graph |

### conv-30 问题分类

| Category | 说明 | 典型问题 | 预期后端 |
|----------|------|----------|----------|
| Temporal Fact (cat2) | 询问具体时间点 | "When did Jon lose his job?" | graph |
| Entity Relation (cat1/3) | 询问实体间关系 | "What do Jon and Gina have in common?" | graph |
| Personal Fact (cat4) | 询问个人事实/原因 | "Why did Jon start his dance studio?" | openviking |
| Count/List | 询问数量/列表 | "How many countries did James visit?" | openviking |

---

## 3. 实验组配置

### 3.1 Baseline 组：Native OV（MemRouter 完全禁用）

| 配置项 | 值 |
|--------|-----|
| 配置文件 | `config/ov_baseline.conf` |
| MemRouter | **禁用** (`memrouter.enabled: false`) |
| 搜索方式 | 所有问题直接走原生 OV `search()`，不经过任何 MemRouter 路由逻辑 |
| Embedding | text-embedding-v3 (DashScope) |
| VLM (主 LLM) | deepseek-v4-flash (DeepSeek API) |

### 3.2 Treatment 组：OV + MemRouter + Graph

| 配置项 | 值 |
|--------|-----|
| 配置文件 | `config/ov+graph.conf` |
| MemRouter | **启用** (`memrouter.enabled: true`) |
| Enabled Backends | `["openviking_memory_backend", "graph_memory_backend"]` |
| 路由方式 | Template embedding + lexical boost |
| LLM Fallback | deepseek-v4-flash (DeepSeek API) |
| 模板 | 14 enabled (5 ov + 6 graph + 3 streamlined_disabled) |
| Graph DB | Neo4j (bolt://127.0.0.1:7687) |

### 3.3 统一控制变量

| 变量 | 统一值 | 说明 |
|------|--------|------|
| LLM 模型 (bot + vlm) | deepseek-v4-flash | 主回答生成 + 意图分析统一模型 |
| Embedding | text-embedding-v3 | 记忆检索向量模型 |
| Temperature | 0 | 确保可复现 |
| Workspace | 相同 | 相同的记忆数据 |
| Neo4j Graph | 相同 | 相同的图数据 |
| Judge 模型 | MiniMax-M2.7 | 答案评分统一用 MiniMax |

### 3.4 模型配置总览

| 组件 | 模型 | API 端点 | 用途 | 是否 DeepSeek 可统计 |
|------|------|----------|------|---------------------|
| VLM (意图分析) | deepseek-v4-flash | `https://api.deepseek.com/v1` | 原生搜索 IntentAnalyzer | ✅ 是 |
| MemRouter LLM Fallback | deepseek-v4-flash | `https://api.deepseek.com/v1` | 路由 fallback | ✅ 是 |
| VikingBot Agent | **deepseek-v4-flash** | `https://api.deepseek.com/v1` | 对话 Agent / 回答生成 | ✅ 是 |
| Judge | **MiniMax-M2.7** | `https://api.minimaxi.com/anthropic` | 答案正确性评分 | ❌ 否 |
| Embedding | text-embedding-v3 | `https://dashscope.aliyuncs.com` | 向量检索 | ❌ 否（DashScope） |

> **关键设计**: bot (VikingBot Agent) 和 vlm (IntentAnalyzer + LLM Fallback) 统一使用 deepseek-v4-flash，确保所有 LLM 调用都走 DeepSeek API，可通过官网统一统计。
> Judge 保持 MiniMax-M2.7（确保评分一致性，不受模型切换影响）。

---

## 4. 评估指标体系

### 4.1 Token 效率指标（P0）

| 指标 | 计算方式 | 预期方向 |
|------|----------|----------|
| Avg prompt_tokens | 所有 case 的 prompt_tokens 平均值 | Treatment < Baseline |
| Avg completion_tokens | 所有 case 的 completion_tokens 平均值 | 持平 |
| Avg total_tokens | 平均值 | **Treatment < Baseline** |
| **Token 节省率** | `(Baseline_total - Treatment_total) / Baseline_total * 100%` | 目标 > 10% |

**原理**: Graph search 返回结构化关系列表（通常 10-20 条），而 OV search 返回长文本记忆片段。Graph 结果更精简，prompt 中占用更少 tokens。

### 4.2 路由质量指标（P0）

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| **后端路由正确率** | `is_backend_correct = true / 有标签数` | 与 v4 标签对比 |
| **模板命中率** | `is_template_hit = true / 总数` | 越高说明 MemRouter 向量匹配越有效 |
| **Any Backend Hit Rate** | `any_expected_backend_hit = true / 总数` | 宽松指标：预期后端被访问过即算对 |
| **LLM Fallback 率** | `route_method = "llm_backend_fallback" / 总数` | 越低说明模板覆盖越好 |
| **Graph Hit Rate** | 路由到 graph 的 case 数 / 总数 | 反映 graph 后端利用率 |

### 4.3 回答质量指标（P1）

| 指标 | 计算方式 |
|------|----------|
| **回答正确率** | `judge_correct = true / 被 judge 数` |
| **By Expected Backend 正确率** | 按预期后端分组的正确率 |
| **By Actual Backend 正确率** | 按实际后端分组的正确率 |

### 4.4 效率指标（P1）

| 指标 | 计算方式 |
|------|----------|
| 平均延迟 | `latency_ms` 平均值 |
| 路由延迟 | `route_latency_ms` 平均值 |

---

## 5. 实验脚本与参数

### 5.1 脚本清单

| 脚本 | 路径 | 用途 |
|------|------|------|
| Baseline eval | `scripts/eval_locomo_ov_with_memrouter_e2e.py` | Baseline 组评测 |
| Treatment eval | `scripts/eval_locomo_ov_with_graph_e2e.py` | Treatment 组评测 |
| 对比分析 | `scripts/compare_ablation.py` | 生成对比报告（待创建） |
| 诊断脚本 | `scripts/diagnose_template_index.py` | 检查模板加载状态 |

### 5.2 Baseline 运行命令

```powershell
# 1. 确保使用 ov_baseline.conf 启动服务
$env:OPENVIKING_CONFIG_FILE = "D:/Code/cursorProject/OpenViking/benchmark/memrouter_embedded/config/ov_baseline.conf"
# 启动 OpenViking Server (with_bot)

# 2. 运行 Baseline 评测
python scripts/eval_locomo_ov_with_memrouter_e2e.py `
  --dataset ../data/conv30_exclude_cat5.json `
  --route-labels ../data/locomo_e2e_route_labels.v4.jsonl `
  --ov-config ../config/ov_baseline.conf `
  --ov-chat-endpoint http://127.0.0.1:18790 `
  --output-base ../results/ablation `
  --force-memory-search `
  --judge `
  --judge-base-url https://api.minimaxi.com/anthropic `
  --judge-token YOUR_MINIMAX_JUDGE_TOKEN `
  --judge-model MiniMax-M2.7
```

### 5.3 Treatment 运行命令

```powershell
# 1. 确保使用 ov+graph.conf 启动服务
$env:OPENVIKING_CONFIG_FILE = "D:/Code/cursorProject/OpenViking/benchmark/memrouter_embedded/config/ov+graph.conf"
# 启动 OpenViking Server (with_bot)

# 2. 运行 Treatment 评测
python scripts/eval_locomo_ov_with_graph_e2e.py `
  --dataset ../data/conv30_exclude_cat5.json `
  --route-labels ../data/locomo_e2e_route_labels.v4.jsonl `
  --ov-config ../config/ov+graph.conf `
  --output-base ../results/ablation `
  --force-memory-search `
  --judge `
  --judge-base-url https://api.minimaxi.com/anthropic `
  --judge-token YOUR_MINIMAX_JUDGE_TOKEN `
  --judge-model MiniMax-M2.7
```

### 5.4 参数说明

| 参数 | 含义 | Baseline | Treatment |
|------|------|----------|-----------|
| `--dataset` | LoCoMo 数据文件 | `../data/conv30_exclude_cat5.json` | 同上 |
| `--route-labels` | 路由标签文件 | `../data/locomo_e2e_route_labels.v4.jsonl` | 同上 |
| `--ov-config` | OpenViking 配置文件 | `../config/ov_baseline.conf` | `../config/ov+graph.conf` |
| `--output-base` | 结果输出根目录 | `../results/ablation` | 同上 |
| `--force-memory-search` | 强制触发记忆搜索 | 启用 | 启用 |
| `--judge` | 启用 LLM Judge 评分 | 启用 | 启用 |
| `--judge-base-url` | Judge API 端点 | `https://api.deepseek.com/v1` | 同上 |
| `--judge-token` | Judge API Token | DeepSeek Token | 同上 |
| `--judge-model` | Judge 模型 | `MiniMax-M2.7` | 同上 |

---

## 6. 执行计划

```
Step 0: 预检（每次实验前必须执行）
    ├── 确认 Neo4j 运行且数据已导入
    ├── 确认 workspace index 完整
    └── 小规模验证（5 questions）确认产物正常

Step 1: Baseline 组小规模验证
    ├── 配置: ov_baseline.conf
    ├── 数据集: conv30_only.json (5 questions)
    ├── judge: true
    └── 检查: metrics_summary.json, qa_results.csv, report.md 格式正确

Step 2: Baseline 组正式运行
    ├── 配置: ov_baseline.conf
    ├── 数据集: conv30_exclude_cat5.json (81 questions)
    ├── judge: true
    └── 记录: results/ablation/baseline_<timestamp>_*/

Step 3: 重启服务，切换配置

Step 4: Treatment 组小规模验证
    ├── 配置: ov+graph.conf
    ├── 数据集: conv30_only.json (5 questions)
    ├── judge: true
    └── 检查: 同上

Step 5: Treatment 组正式运行
    ├── 配置: ov+graph.conf
    ├── 数据集: conv30_exclude_cat5.json (81 questions)
    ├── judge: true
    └── 记录: results/ablation/treatment_<timestamp>_*/

Step 6: 对比分析
    ├── 运行 compare_ablation.py
    └── 输出: results/ablation/ablation_comparison.md
```

---

## 7. 预期假设与可验证结论

| 假设 | 验证方式 | 预期结果 |
|------|----------|----------|
| **H1**: MemRouter 能正确路由 temporal 问题到 graph | 检查 temporal case 的 `actual_backend` | `graph.timeline_fact.v1` 命中，backend=graph |
| **H2**: Graph 返回更精简的结构化结果，节省 prompt tokens | 对比 graph case vs ov case 的 prompt_tokens | Graph case 平均 prompt_tokens 更低 |
| **H3**: 正确的路由提升回答准确率 | 对比 Treatment vs Baseline 的 answer accuracy | Treatment >= Baseline |
| **H4**: 多实体关系问题在 graph 上表现更好 | 检查 entity_relation case 的 judge_correct | Graph 路由的回答更准确 |
| **H5**: Template 命中率反映 normalizer 修复效果 | 对比修复前后 template hit rate | 修复后 > 修复前 |

---

## 8. 已知问题与修复记录

### 问题 1: QueryNormalizer 破坏占位符语义

**现象**: 所有模板占位符（ACTION_X, PLACE_X, ITEM_X 等）被替换为 `"person"`，导致 prototype 变成 `"when did person do person"` 这种语义垃圾。

**影响**: Embedding 相似度暴跌，temporal 模板无法匹配。

**修复**: 修改 `openviking/memrouter/normalizer.py`，按占位符类型替换为不同通用词（action, place, item, event, job 等）。

**验证**: `"When did PERSON_A do ACTION_X?"` -> `"when did person do action"`

### 问题 2: Lexical boost 未覆盖现在完成时

**现象**: Benchmark 查询使用现在完成时（"When has Jon lost..."），但 lexical boost 只匹配 "when did"。

**修复**: 扩展 matcher.py lexical boost，添加 "when has", "when was", "when were" (+0.05) 和 "has lost", "was fired" 等 (+0.03)。

### 问题 3: Template prototype 缺乏现在完成时变体

**现象**: graph.timeline_fact.v1 的 173 个 prototype 全部是一般过去时（"When did..."）。

**修复**: 添加 19 个现在完成时/被动语态 prototype（"When has PERSON_A lost...", "When was PERSON_A fired..." 等）。

---

## 9. 输出产物

| 产物 | 路径模板 | 说明 |
|------|----------|------|
| Baseline 结果 | `results/ablation/baseline_<timestamp>_conv30_exclude_cat5_ov_memrouter_e2e/` | Native OV 运行结果 |
| Treatment 结果 | `results/ablation/treatment_<timestamp>_conv30_exclude_cat5_ov_graph_e2e/` | OV+MemRouter+Graph 运行结果 |
| 对比报告 | `results/ablation/ablation_comparison.md` | 并排对比表格 |

### 每个结果目录包含

```
results/
├── metrics_summary.json    # 汇总指标（后端准确率、模板命中率、token 统计等）
├── qa_results.csv          # 每个 question 的详细结果（question, answer, response, judge_correct 等）
├── report.md               # 人类可读报告
├── route_results.jsonl     # 每个 case 的完整路由事件
└── run_config.json         # 本次运行的配置快照
```

---

## 10. DeepSeek 花费统计说明

### 10.1 为什么统一用 DeepSeek？

DeepSeek 官网 (`platform.deepseek.com`) 提供详细的 API 调用统计：
- 总 token 消耗（input + output）
- 费用明细
- 按模型分组

其他渠道（MiniMax、DashScope）没有同等透明的在线统计界面。

### 10.2 配置成 DeepSeek 的组件

| 组件 | 原配置 | 建议配置 | 理由 |
|------|--------|----------|------|
| VLM (意图分析) | deepseek-v4-flash | **保持** | 已经是 DeepSeek |
| MemRouter LLM Fallback | deepseek-v4-flash | **保持** | 已经是 DeepSeek |
| VikingBot Agent | MiniMax-M2.7 | **改为 deepseek-v4-flash** | 统一模型，便于 DeepSeek 统计 |
| Judge | MiniMax-M2.7 | **保持 MiniMax** | 确保评分一致性 |

### 10.3 统计范围

通过 DeepSeek 官网可统计到 **~95%** 的 LLM token 消耗：
- ✅ VLM IntentAnalyzer（Baseline 81 次调用）
- ✅ MemRouter LLM Fallback（Treatment 22 次调用）
- ✅ VikingBot Agent 对话（81 questions × 多轮对话）
- ❌ Judge 评分（MiniMax 渠道，81 questions × 1 call）
- ❌ Embedding 向量（DashScope 渠道，费用极低）

> Token 比较逻辑：
> - Baseline: 81 次 IntentAnalyzer + 81 次回答生成
> - Treatment: 22 次 LLM Fallback + 81 次回答生成（59 次跳过 IntentAnalyzer）
> - 节省 = Baseline 总 token - Treatment 总 token
>
> 建议：实验前后记录 DeepSeek 官网余额/消耗，相减得到精确花费。
