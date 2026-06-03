#!/usr/bin/env python3
"""Reassign streamlined labels to openviking or graph based on query semantics.

Heuristic:
- If query contains temporal + multi-entity signals ("X and Y", "both", "together")
  → graph_memory_backend (graph can answer via rel_date on entity relations)
- Otherwise → openviking_memory_backend (native semantic search for single-entity temporal)

Usage:
    python reassign_streamlined_labels.py \
      --input ../data/locomo_e2e_route_labels.v3.jsonl \
      --output ../data/locomo_e2e_route_labels.v3.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _has_multi_entity_signal(question: str) -> bool:
    """Check if question involves multiple entities."""
    q = question.lower()
    # "X and Y" pattern (e.g. "Jon and Gina", "Caroline and Melanie")
    # Require that both sides of "and" look like proper names or pronouns
    if re.search(r"\b[A-Z][a-z]+ and [A-Z][a-z]+\b", question):
        return True
    # Explicit multi-entity keywords
    if any(kw in q for kw in ("both ", "between ", "together", "collaborated",
                               "each other", "one another")):
        return True
    return False


def _has_temporal_signal(question: str) -> bool:
    """Check if question asks for time/date/duration."""
    q = question.lower()
    return any(kw in q for kw in (
        "when", "how long", "how many days", "how many months",
        "how many years", "how many weeks", "how many times",
        "first time", "last time", "before", "after",
        "between", "ago", "passed between",
    ))


def decide_backend(question: str, scenario: str) -> str:
    """Decide backend for a temporal question.

    Returns:
        "graph_memory_backend" if multi-entity temporal (graph has rel_date).
        "openviking_memory_backend" otherwise (single-entity temporal via semantic search).
    """
    # Multi-entity temporal → graph (e.g. "When did Jon and Gina collaborate?")
    if _has_temporal_signal(question) and _has_multi_entity_signal(question):
        return "graph_memory_backend"
    # All other temporal → ov
    return "openviking_memory_backend"


def main() -> int:
    parser = argparse.ArgumentParser(description="Reassign streamlined labels")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records: list[dict] = []
    changes: list[tuple[str, str, str, str]] = []  # (case_id, question, old, new)
    stats = {"ov": 0, "graph": 0}

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)

    for record in records:
        case_id = record.get("case_id", "")
        expected = record.get("expected_backend", "")
        question = record.get("question", "")
        scenario = record.get("scenario", "")

        if expected == "openviking_memory_backend" and scenario == "temporal_fact":
            new_backend = decide_backend(question, scenario)
            if new_backend != expected:
                changes.append((case_id, question, expected, new_backend))
                record["expected_backend"] = new_backend
                stats["graph" if new_backend == "graph_memory_backend" else "ov"] += 1

    print(f"Total records: {len(records)}")
    print(f"Changes made: {len(changes)}")
    print(f"  → graph_memory_backend: {stats['graph']}")
    print(f"  → openviking_memory_backend: {stats['ov']}")

    if changes:
        print("\nFirst 10 changes:")
        for case_id, question, old, new in changes[:10]:
            print(f"  {case_id}: {old} -> {new}")
            print(f"    Q: {question[:100]}")

    if not args.dry_run:
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nWritten to {output_path}")
    else:
        print("\nDry run — no file written")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
