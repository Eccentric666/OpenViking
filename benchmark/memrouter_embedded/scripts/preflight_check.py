#!/usr/bin/env python
"""Quick pre-flight check for Graph Backend E2E evaluation.

Validates:
1. Workspace index integrity
2. Neo4j connectivity
3. ov.conf configuration (graph_db + memrouter.enabled_backends)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def check_workspace(workspace_path: Path) -> list[str]:
    issues = []
    if not workspace_path.exists():
        issues.append(f"Workspace missing: {workspace_path}")
        return issues

    # 1. VectorDB structure
    vectordb = workspace_path / "vectordb" / "context"
    if not vectordb.exists():
        issues.append(f"VectorDB directory missing: {vectordb}")
        return issues

    meta = vectordb / "collection_meta.json"
    if not meta.exists():
        issues.append("collection_meta.json missing")
    else:
        try:
            with meta.open("r", encoding="utf-8") as f:
                meta_data = json.load(f)
            dim = meta_data.get("Dimension")
            if dim != 1024:
                issues.append(f"Collection dimension mismatch: expected 1024, got {dim}")
        except Exception as exc:
            issues.append(f"Failed to parse collection_meta.json: {exc}")

    index_dir = vectordb / "index"
    if not any(index_dir.rglob("*")):
        issues.append("Vector index directory is empty")
    else:
        index_meta = index_dir / "default" / "index_meta.json"
        if not index_meta.exists():
            issues.append("index_meta.json missing")

    store_dir = vectordb / "store"
    if not store_dir.exists() or not any(store_dir.iterdir()):
        issues.append("Vector store directory is empty")

    # 2. Agent-level memories (actual path in workspace)
    agent_dir = workspace_path / "viking" / "default" / "agent"
    conv_count = 0
    if agent_dir.exists():
        for d in agent_dir.iterdir():
            if d.is_dir() and d.name.startswith("conv-") and (d / "memories").exists():
                conv_count += 1
    if conv_count == 0:
        issues.append("No conv-XX memory directories found in viking/default/agent/")
    else:
        print(f"  Found {conv_count} conv directories with memories.")

    # 3. System queue DB
    queue_db = workspace_path / "_system" / "queue" / "queue.db"
    if not queue_db.exists():
        issues.append("System queue DB missing")

    return issues


def check_neo4j(uri: str, password: str) -> list[str]:
    issues = []
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=("neo4j", password))
        with driver.session() as session:
            result = session.run("RETURN 1 AS ok")
            record = result.single()
            if not (record and record.get("ok") == 1):
                issues.append("Neo4j responded unexpectedly")
        driver.close()
    except ImportError:
        issues.append("neo4j driver not installed")
    except Exception as exc:
        issues.append(f"Neo4j connection failed: {exc}")
    return issues


def check_ov_config(path: Path) -> list[str]:
    issues = []
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    storage = cfg.get("storage", {})
    if not storage.get("graphdb", {}).get("enabled", False):
        issues.append("graphdb.enabled is false or missing")
    if "graph_memory_backend" not in cfg.get("memrouter", {}).get("enabled_backends", []):
        issues.append("graph_memory_backend not in memrouter.enabled_backends")
    return issues


def main() -> int:
    workspace = _repo_root() / "workspace"
    ov_config = _repo_root() / "config" / "ov+graph.conf"
    neo4j_uri = "bolt://127.0.0.1:7687"
    neo4j_password = "12345678"

    all_ok = True

    print("=" * 50)
    print("Pre-flight Check for Graph Backend E2E")
    print("=" * 50)

    print("\n[1/3] Workspace index ...")
    ws_issues = check_workspace(workspace)
    if ws_issues:
        all_ok = False
        for issue in ws_issues:
            print(f"  FAIL: {issue}")
    else:
        print("  OK")

    print("\n[2/3] Neo4j connectivity ...")
    neo4j_issues = check_neo4j(neo4j_uri, neo4j_password)
    if neo4j_issues:
        all_ok = False
        for issue in neo4j_issues:
            print(f"  FAIL: {issue}")
    else:
        print("  OK")

    print("\n[3/3] ov.conf (graph backend enabled) ...")
    if ov_config.exists():
        cfg_issues = check_ov_config(ov_config)
        if cfg_issues:
            all_ok = False
            for issue in cfg_issues:
                print(f"  FAIL: {issue}")
        else:
            print("  OK")
    else:
        print(f"  SKIP: {ov_config} not found")

    print("\n" + "=" * 50)
    if all_ok:
        print("All checks passed. Ready to run E2E evaluation.")
        return 0
    else:
        print("Some checks failed. Please fix before running evaluation.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
