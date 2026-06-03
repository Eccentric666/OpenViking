#!/usr/bin/env python3
"""诊断脚本 — 检查 MemRouter 模板索引运行时状态。

直接实例化 TemplateIndex 和 Pipeline，验证 graph.timeline_fact.v1 等
新模板是否在内存索引中，并对 benchmark 查询执行完整匹配打分。
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openviking.memrouter.templates import BackendRouteTemplateIndex
from openviking.memrouter.normalizer import QueryNormalizer


def main():
    templates_dir = PROJECT_ROOT / "openviking" / "memrouter" / "templates_data"
    print(f"Template directory: {templates_dir}")
    print(f"Exists: {templates_dir.exists()}")
    print("-" * 60)

    # 1. Load all templates from disk
    idx = BackendRouteTemplateIndex()
    count = idx.load_from_directory(templates_dir)
    print(f"\n📁 Loaded {count} templates from disk")

    all_templates = list(idx.enabled_templates())
    print(f"   Enabled templates: {len(all_templates)}")

    # 2. Check by backend
    by_backend = {}
    for t in all_templates:
        bid = t.target.primary_backend_id
        by_backend.setdefault(bid, []).append(t.template_id)

    print(f"\n📊 By backend:")
    for bid, tids in sorted(by_backend.items()):
        print(f"   {bid}: {len(tids)} templates")
        for tid in sorted(tids):
            print(f"      - {tid}")

    # 3. Check specifically for our new templates
    print("\n🔍 Checking new temporal templates:")
    targets = [
        "graph.timeline_fact.v1",
        "graph.duration_comparison.v1",
        "graph.sequence_reasoning.v1",
    ]
    for tid in targets:
        tmpl = idx.get(tid)
        if tmpl:
            print(f"   ✅ {tid}")
            print(f"      backend: {tmpl.target.primary_backend_id}")
            print(f"      intent:  {tmpl.intent_family.name}")
            print(f"      prototypes: {len(tmpl.query_prototypes)}")
            print(f"      hard_negatives: {len(tmpl.hard_negatives)}")
            print(f"      accept: {tmpl.thresholds.accept}, fallback: {tmpl.thresholds.fallback}")
        else:
            print(f"   ❌ {tid} NOT FOUND")

    # 4. Check for any streamlined templates (should be none after hot-swap)
    print("\n🔍 Checking for streamlined templates (should be 0 after hot-swap):")
    streamlined = [t for t in all_templates if "streamlined" in t.template_id]
    if streamlined:
        for t in streamlined:
            print(f"   ⚠️  {t.template_id} (backend={t.target.primary_backend_id})")
    else:
        print("   ✅ No streamlined templates found")

    # 5. Normalize benchmark queries and check what they look like
    print("\n📝 Normalized benchmark queries:")
    normalizer = QueryNormalizer()
    queries = [
        "When Jon has lost his job as a banker?",
        "When Gina has lost her job at Door Dash?",
        "How do Jon and Gina both like to destress?",
        "What do Jon and Gina both have in common?",
        "Why did Jon decide to start his dance studio?",
    ]
    for q in queries:
        norm = normalizer.normalize(q)
        print(f"   '{q[:50]}...' -> '{norm}'")

    # 6. Check YAML files on disk (bypass Python loading)
    print("\n📂 YAML files in templates_data/:")
    yaml_files = sorted(templates_dir.glob("*.yaml"))
    for f in yaml_files:
        size = f.stat().st_size
        print(f"   {f.name:50s} ({size:>6d} bytes)")


if __name__ == "__main__":
    main()
