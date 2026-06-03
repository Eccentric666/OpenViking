#!/usr/bin/env python3
"""Import Neo4j graph data from JSON files.

Reads nodes.json + relationships.json and recreates the graph via MERGE
(idempotent — safe to run multiple times).

Supports:
- CLI arguments for URI, auth, input directory
- Auto-read credentials from ov.conf graphdb section
- Clear-existing option (wipe before import)

Usage::

    python import_neo4j.py --in-dir ./data/neo4j_backup
    python import_neo4j.py --ov-config ../config/ov+graph.conf --in-dir ./data/neo4j_backup
    python import_neo4j.py --uri bolt://127.0.0.1:7687 --password 12345678 --in-dir ./data/neo4j_backup --clear-existing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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


def _build_node_cypher(props: dict[str, Any]) -> str:
    """Build MERGE Cypher for a node with dynamic properties."""
    name = props.get("name", "")
    if not name:
        return ""

    # Ensure we set name via MERGE key, then all other properties
    set_clauses = []
    for k, v in props.items():
        if v is not None and k != "name":
            set_clauses.append(f"n.{k} = ${k}")

    set_part = f"SET\n                {',\n                '.join(set_clauses)}" if set_clauses else ""
    return f"""
        MERGE (n:Node {{name: $name}})
        {set_part}
    """.strip()


def _build_rel_cypher(rel_props: dict[str, Any]) -> str:
    """Build MERGE Cypher for a relationship with dynamic properties."""
    from_name = rel_props.pop("from_name", "")
    to_name = rel_props.pop("to_name", "")
    if not from_name or not to_name:
        return ""

    # Re-insert names as params for the MATCH clause
    remaining = {k: v for k, v in rel_props.items() if v is not None}
    set_clauses = [f"r.{k} = ${k}" for k in remaining]
    set_part = f"SET\n                {',\n                '.join(set_clauses)}" if set_clauses else ""

    # Build param dict excluding from_name/to_name (passed separately)
    return f"""
        MATCH (a:Node {{name: $from_name}}), (b:Node {{name: $to_name}})
        MERGE (a)-[r:RELATION]->(b)
        {set_part}
    """.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Neo4j graph data from JSON.")
    parser.add_argument("--in-dir", required=True, help="Input directory containing nodes.json + relationships.json.")
    parser.add_argument("--uri", default="", help="Neo4j Bolt URI.")
    parser.add_argument("--username", default="neo4j", help="Neo4j username.")
    parser.add_argument("--password", default="", help="Neo4j password.")
    parser.add_argument("--database", default="neo4j", help="Neo4j database name.")
    parser.add_argument("--ov-config", default=str(_repo_root() / "config" / "ov+graph.conf"), help="Path to ov.conf (auto-extract graphdb credentials).")
    parser.add_argument("--clear-existing", action="store_true", help="Delete all existing nodes/relationships before import.")
    parser.add_argument("--batch-size", type=int, default=50, help="Commit every N records.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir).resolve()
    nodes_path = in_dir / "nodes.json"
    rels_path = in_dir / "relationships.json"

    if not nodes_path.exists():
        print(f"[ERROR] Nodes file not found: {nodes_path}")
        return 1
    if not rels_path.exists():
        print(f"[ERROR] Relationships file not found: {rels_path}")
        return 1

    with nodes_path.open("r", encoding="utf-8") as f:
        nodes = json.load(f)
    with rels_path.open("r", encoding="utf-8") as f:
        rels = json.load(f)

    print(f"Loaded {len(nodes)} nodes, {len(rels)} relationships from {in_dir}")

    # Resolve credentials
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
        print("[ERROR] Neo4j password is required.")
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
            # Connectivity check
            try:
                record = session.run("RETURN 1 AS ok").single()
                if not (record and record.get("ok") == 1):
                    print("[ERROR] Neo4j connectivity check failed.")
                    return 1
            except Exception as exc:
                print(f"[ERROR] Neo4j connection failed: {exc}")
                return 1

            # Optional wipe
            if args.clear_existing:
                print("[WARN] Clearing all existing nodes and relationships ...")
                session.run("MATCH (n) DETACH DELETE n")
                print("  Cleared.")

            # 1. Import nodes
            print(f"Importing {len(nodes)} nodes ...")
            imported_nodes = 0
            for i, node in enumerate(nodes, 1):
                name = node.get("name", "")
                if not name:
                    continue

                props = {k: v for k, v in node.items() if v is not None}
                cypher = _build_node_cypher(dict(props))
                if not cypher:
                    continue

                session.run(cypher, **props)
                imported_nodes += 1

                if i % args.batch_size == 0:
                    print(f"  {i}/{len(nodes)} done")
            print(f"  Imported {imported_nodes}/{len(nodes)} nodes.")

            # 2. Import relationships
            print(f"Importing {len(rels)} relationships ...")
            imported_rels = 0
            for i, rel in enumerate(rels, 1):
                from_name = rel.get("from_name", "")
                to_name = rel.get("to_name", "")
                if not from_name or not to_name:
                    continue

                # Build params: include from_name/to_name plus all other non-null props
                rel_props = {k: v for k, v in rel.items() if v is not None}
                cypher = _build_rel_cypher(dict(rel_props))
                if not cypher:
                    continue

                session.run(cypher, **rel_props)
                imported_rels += 1

                if i % args.batch_size == 0:
                    print(f"  {i}/{len(rels)} done")
            print(f"  Imported {imported_rels}/{len(rels)} relationships.")

            # 3. Post-import stats
            stats = session.run("""
                MATCH (n:Node) RETURN count(n) AS node_count
            """).single()
            rel_stats = session.run("""
                MATCH ()-[r:RELATION]->() RETURN count(r) AS rel_count
            """).single()
            print(f"\nPost-import stats: {stats['node_count']} nodes, {rel_stats['rel_count']} relations in DB.")

    finally:
        driver.close()

    print(f"\nImport complete from {in_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
