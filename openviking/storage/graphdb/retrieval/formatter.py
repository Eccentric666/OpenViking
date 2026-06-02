# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Format graph retrieval results for LLM consumption."""

from typing import List

from .graph_retriever import GraphSearchResult


class GraphRetrievalFormatter:
    """Format GraphSearchResult list into natural language or structured formats."""

    @staticmethod
    def to_natural_language(results: List[GraphSearchResult]) -> str:
        """Convert results to two-line blocks: relation + metadata, then rel_content."""
        lines = []
        for r in results:
            source_tag = f" ({r.source_tag})" if r.source_tag else ""
            target_tag = f" ({r.target_tag})" if r.target_tag else ""
            lines.append(
                f"{r.source}{source_tag} -> {r.rel_desc} -> {r.target}{target_tag} "
                f"[rel_date: {r.rel_date}]"
            )
            if r.rel_content:
                lines.append(f"rel_content: {r.rel_content}")
        return "\n".join(lines)

    @staticmethod
    def to_triplets(results: List[GraphSearchResult]) -> List[List[str]]:
        """Convert to triple list for BM25 reranking."""
        return [[r.source, r.rel_desc, r.target] for r in results]

    @staticmethod
    def to_cypher_subgraph(results: List[GraphSearchResult]) -> str:
        """Generate Cypher for subgraph visualization."""
        if not results:
            return ""
        lines = ["// Subgraph query"]
        names = set()
        for r in results:
            names.add(r.source)
            names.add(r.target)
        name_list = ", ".join(f"'{n}'" for n in sorted(names))
        lines.append(f"MATCH (n:Node) WHERE n.name IN [{name_list}]")
        lines.append("RETURN n")
        return "\n".join(lines)
