#!/usr/bin/env python3
"""Export Neo4j graph data to JSON files.

Supports:
- CLI arguments for URI, auth, output directory
- Auto-read credentials from ov.conf graphdb section
- Per-conversation filtering (if nodes have conv_id or owner_agent_id)
- Summary statistics

Usage::

    python export_neo4j.py --out-dir ./data/neo4j_backup
    python export_neo4j.py --ov-config ../config/ov+graph.conf --out-dir ./data/neo4j_backup
    python export_neo4j.py --uri bolt://127.0.0.1:7687 --password 12345678 --out-dir ./data/neo4j_backup
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _serialize_value(v: Any) -> Any:
    """Serialize Neo4j values to JSON-compatible types."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool, list, dict)):
        return v
    if hasattr(v, "iso_format"):
        return v.iso_format()
    if hasattr(v, "year") and hasattr(v, "month"):
        return str(v)
    return str(v)


def _clean_record(record_dict: dict[str, Any]) -> dict[str, Any]:
    return {k: _serialize_value(v) for k, v in record_dict.items()}


def _load_config_from_ov_conf(ov_config_path: Path) -> dict[str, Any] | None:
    """Extract graphdb config from ov.conf."""
    try:
        with ov_config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("storage", {}).get("graphdb")
    except Exception:
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def export_nodes(session, conv_id: str = "") -> list[dict[str, Any]]:
    """Export all nodes, optionally filtered by conv_id."""
    if conv_id:
        # Try matching on owner_agent_id or name containing conv_id
        result = session.run(
            """
            MATCH (n:Node)
            WHERE n.owner_agent_id = $conv_id OR n.name CONTAINS $conv_id
            RETURN {
                name: n.name,
                source: n.source,
                tag: n.tag,
                properties: n.properties,
                account_id: n.account_id,
                user_id: n.user_id,
                owner_agent_id: n.owner_agent_id,
                embedding: n.embedding,
                created_at: n.created_at,
                updated_at: n.updated_at
            } AS node
            """,
            conv_id=conv_id,
        )
    else:
        result = session.run(
            """
            MATCH (n:Node)
            RETURN {
                name: n.name,
                source: n.source,
                tag: n.tag,
                properties: n.properties,
                account_id: n.account_id,
                user_id: n.user_id,
                owner_agent_id: n.owner_agent_id,
                embedding: n.embedding,
                created_at: n.created_at,
                updated_at: n.updated_at
            } AS node
            """
        )
    return [_clean_record(r["node"]) for r in result.data()]


def export_relationships(session, conv_id: str = "") -> list[dict[str, Any]]:
    """Export all relationships, optionally filtered by conv_id."""
    if conv_id:
        result = session.run(
            """
            MATCH (a:Node)-[r:RELATION]->(b:Node)
            WHERE a.owner_agent_id = $conv_id OR a.name CONTAINS $conv_id
               OR b.owner_agent_id = $conv_id OR b.name CONTAINS $conv_id
            RETURN {
                from_name: a.name,
                from_source: a.source,
                to_name: b.name,
                to_source: b.source,
                rel_desc: r.rel_desc,
                rel_from: r.rel_from,
                rel_date: r.rel_date,
                rel_content: r.rel_content,
                valid: r.valid,
                invalidated_at: r.invalidated_at,
                history: r.history,
                source: r.source
            } AS rel
            """,
            conv_id=conv_id,
        )
    else:
        result = session.run(
            """
            MATCH (a:Node)-[r:RELATION]->(b:Node)
            RETURN {
                from_name: a.name,
                from_source: a.source,
                to_name: b.name,
                to_source: b.source,
                rel_desc: r.rel_desc,
                rel_from: r.rel_from,
                rel_date: r.rel_date,
                rel_content: r.rel_content,
                valid: r.valid,
                invalidated_at: r.invalidated_at,
                history: r.history,
                source: r.source
            } AS rel
            """
        )
    return [_clean_record(r["rel"]) for r in result.data()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Neo4j graph data to JSON.")
    parser.add_argument("--out-dir", required=True, help="Output directory for JSON files.")
    parser.add_argument("--uri", default="", help="Neo4j Bolt URI.")
    parser.add_argument("--username", default="neo4j", help="Neo4j username.")
    parser.add_argument("--password", default="", help="Neo4j password.")
    parser.add_argument("--database", default="neo4j", help="Neo4j database name.")
    parser.add_argument("--ov-config", default=str(_repo_root() / "config" / "ov+graph.conf"), help="Path to ov.conf (auto-extract graphdb credentials).")
    parser.add_argument("--conv-id", default="", help="Optional: filter by conversation ID.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent level.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve credentials: CLI args > ov.conf > defaults
    uri = args.uri
    username = args.username
    password = args.password

    if not uri or not password:
        ov_config_path = Path(args.ov_config)
        if ov_config_path.exists():
            graph_cfg = _load_config_from_ov_conf(ov_config_path)
            if graph_cfg:
                uri = uri or graph_cfg.get("uri", "")
                username = username or graph_cfg.get("username", "neo4j")
                password = password or graph_cfg.get("password", "")
                print(f"[INFO] Loaded Neo4j credentials from {ov_config_path}")
            else:
                print(f"[WARN] No graphdb section in {ov_config_path}")
        else:
            print(f"[WARN] ov.conf not found: {ov_config_path}")

    if not uri:
        uri = "bolt://127.0.0.1:7687"
    if not password:
        print("[ERROR] Neo4j password is required. Provide --password or ensure ov.conf has graphdb.password.")
        return 1

    print(f"Connecting to Neo4j at {uri} ...")

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[ERROR] neo4j driver not installed. Run: pip install neo4j>=5.14.0")
        return 1

    driver = GraphDatabase.driver(uri, auth=(username, password))

    try:
        with driver.session(database=args.database) as session:
            # Quick connectivity check
            try:
                record = session.run("RETURN 1 AS ok").single()
                if not (record and record.get("ok") == 1):
                    print("[ERROR] Neo4j connectivity check failed.")
                    return 1
            except Exception as exc:
                print(f"[ERROR] Neo4j connection failed: {exc}")
                return 1

            # Export
            print("Exporting nodes ...")
            nodes = export_nodes(session, args.conv_id)
            nodes_path = out_dir / "nodes.json"
            with nodes_path.open("w", encoding="utf-8") as f:
                json.dump(nodes, f, ensure_ascii=False, indent=args.indent)
            print(f"  {len(nodes)} nodes -> {nodes_path}")

            print("Exporting relationships ...")
            rels = export_relationships(session, args.conv_id)
            rels_path = out_dir / "relationships.json"
            with rels_path.open("w", encoding="utf-8") as f:
                json.dump(rels, f, ensure_ascii=False, indent=args.indent)
            print(f"  {len(rels)} relationships -> {rels_path}")

            # Summary
            summary = {
                "export_time": datetime.now().isoformat(),
                "uri": uri,
                "database": args.database,
                "conv_filter": args.conv_id or None,
                "node_count": len(nodes),
                "relationship_count": len(rels),
                "entity_count": sum(1 for n in nodes if n.get("tag") == "entity"),
                "event_count": sum(1 for n in nodes if n.get("tag") == "event"),
                "person_count": sum(1 for n in nodes if n.get("tag") == "person"),
                "valid_rel_count": sum(1 for r in rels if r.get("valid") is True),
            }
            summary_path = out_dir / "summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=args.indent)
            print(f"  summary -> {summary_path}")
            print(json.dumps(summary, indent=2))
    finally:
        driver.close()

    print(f"\nExport complete. Files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
