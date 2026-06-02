# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Relation extraction module using LLM Tool Calling."""

import re
from typing import Any, List, Optional

from openviking_cli.utils.logger import get_logger

from ..graph_models import GraphRelation, RawRelationTriple, RelationTriple
from .prompts import (
    RELATIONS_TOOL,
    build_relation_extraction_messages,
    build_relation_extraction_messages_mode_two,
)

logger = get_logger(__name__)


def _clean_rel_content(raw: str) -> str:
    """Strip Markdown formatting symbols from rel_content, keep plain text."""
    s = raw.strip()
    # Headers: # Title -> Title
    s = re.sub(r"^#{1,6}\s+", "", s)
    # Bold/italic: **text** or *text* or _text_ or __text__ -> text
    s = re.sub(r"\*\*?(.+?)\*\*?", r"\1", s)
    s = re.sub(r"__(.+?)__", r"\1", s)
    s = re.sub(r"_(.+?)_", r"\1", s)
    # Inline code: `code` -> code
    s = re.sub(r"`(.+?)`", r"\1", s)
    # Links: [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # Bullet list markers at line start: - item or * item -> item
    s = re.sub(r"^\s*[-*+]\s+", "", s, flags=re.MULTILINE)
    # Numbered list: 1. item -> item
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)
    # Blockquote: > text -> text
    s = re.sub(r"^\s*>\s+", "", s, flags=re.MULTILINE)
    # Horizontal rules: --- or *** or ___ -> remove
    s = re.sub(r"^\s*[-*_]{3,}\s*$", "", s, flags=re.MULTILINE)
    return s.strip()


class RelationExtractor:
    """Extract relationships using LLM Tool Calling (no external entity list)."""

    def __init__(
        self,
        llm_provider: Any,
        custom_prompt: str = "",
        confidence_threshold: float = 0.8,
    ):
        self._llm = llm_provider
        self._custom_prompt = custom_prompt
        self._confidence_threshold = confidence_threshold

    async def extract(
        self,
        text: str,
        user_id: str = "",
        source: str = "",
        mode: str = "mode_one",
    ) -> List[RawRelationTriple]:
        """Extract relationships from text with full entity info.

        Args:
            text: Raw text (may contain [SOURCE: <uri>] or [YYYY-MM-DD] markers).
            user_id: User identifier for self-reference replacement.
            source: Kept for backward compatibility; not used as fallback.
            mode: "mode_one" for written_entries (URI markers) or "mode_two" for conversation transcripts.

        Returns:
            List of RawRelationTriple (confidence filtered).
        """
        if mode == "mode_two":
            messages = build_relation_extraction_messages_mode_two(text, user_id)
        else:
            messages = build_relation_extraction_messages(text, user_id)

        result = await self._llm.get_completion_async(
            messages=messages,
            tools=[RELATIONS_TOOL],
        )

        triples: List[RawRelationTriple] = []
        for tc in result.tool_calls:
            if tc.name != "establish_relationships":
                continue
            for item in tc.arguments.get("entities", []):
                confidence = float(item.get("confidence", 1.0))
                if confidence < self._confidence_threshold:
                    continue
                src = item.get("source", "").strip().lower().replace(" ", "_")
                dst = item.get("destination", "").strip().lower().replace(" ", "_")
                rel_desc = item.get("relationship", "").strip().lower().replace(" ", "_")
                rel_from = item.get("rel_from", "").strip()
                if not src or not dst or not rel_desc:
                    continue
                if mode != "mode_two" and not rel_from:
                    continue
                triples.append(RawRelationTriple(
                    from_entity=src,
                    from_entity_type=item.get("source_type", "").strip().lower().replace(" ", "_"),
                    to_entity=dst,
                    to_entity_type=item.get("destination_type", "").strip().lower().replace(" ", "_"),
                    rel_desc=rel_desc,
                    confidence=confidence,
                    rel_from=rel_from,
                    rel_date=item.get("rel_date", "").strip(),
                    rel_content=_clean_rel_content(item.get("rel_content", "")),
                ))

        return triples
