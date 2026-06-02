# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Graph extraction queue dequeue handler."""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from openviking.storage.queuefs.named_queue import DequeueHandlerBase
from openviking_cli.utils.logger import get_logger

from .extractors.entity_extractor import EntityExtractor
from .extractors.relation_extractor import RelationExtractor
from .graph_models import (
    ExtractionResult,
    GraphRelation,
    RawEntity,
    RawRelationTriple,
    RelationTriple,
)
from .graph_msg import GraphMsg
from .writers.graph_writer import GraphWriter

logger = get_logger(__name__)


_MONTH_MAP = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


def _extract_date(text: str) -> str:
    """Try to extract a date in YYYY-MM-DD format from text."""
    if not text:
        return ""
    text_lower = text.lower()

    # 1. ISO-like: 2023-01-20 or 2023/01/20
    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"

    # 2. Chinese: 2023年1月20日
    m = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", text)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"

    # 3. English: January 20, 2023 or Jan 2023
    m = re.search(
        r"\b([a-z]{3,9})[.,]?\s+(\d{1,2})?[\s,]*(\d{4})\b",
        text_lower,
    )
    if m:
        month_name = m.group(1)
        month_num = _MONTH_MAP.get(month_name, "01")
        day = m.group(2).zfill(2) if m.group(2) else "01"
        year = m.group(3)
        return f"{year}-{month_num}-{day}"

    return ""


class GraphHandler(DequeueHandlerBase):
    """Dequeue handler for graph extraction pipeline.

    Text build modes:
        mode_one: Extract text from written_entries (events/entities markdown).
        mode_two: Build text from session messages (same format as entity/event extraction).
    """

    def __init__(
        self,
        entity_extractor: EntityExtractor,
        relation_extractor: RelationExtractor,
        graph_writer: GraphWriter,
        text_build_mode: str = "mode_two",
    ):
        self._entity_extractor = entity_extractor
        self._relation_extractor = relation_extractor
        self._graph_writer = graph_writer
        self._text_build_mode = text_build_mode

    async def on_dequeue(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not data:
            return None

        try:
            import json

            # Handle AGFS wrapper format: {"id": "...", "data": "..."}
            if "data" in data and isinstance(data["data"], str):
                inner_data = json.loads(data["data"])
                if "id" in data:
                    inner_data["id"] = data["id"]
                data = inner_data

            msg = GraphMsg.from_dict(data)
            await self._process(msg)
            self.report_success()
            logger.info(
                f"[GraphHandler] Processed graph message for user={msg.user_id}, "
                f"entries={len(msg.written_entries)}"
            )
        except Exception as e:
            logger.error(f"[GraphHandler] Processing failed: {e}")
            self.report_error(str(e), data)
        return data

    @staticmethod
    def _format_message_with_parts(msg) -> str:
        """Format a single message including text and tool calls.

        Mirrors memory_extractor.py exactly.
        """
        import json

        from openviking.message.part import ToolPart

        parts = getattr(msg, "parts", [])
        has_tool_parts = any(isinstance(p, ToolPart) for p in parts)

        if not has_tool_parts:
            return getattr(msg, "content", "") or ""

        tool_lines = []
        text_lines = []
        for part in parts:
            if hasattr(part, "text") and part.text:
                text_lines.append(part.text)
            elif isinstance(part, ToolPart):
                tool_info = {
                    "type": "tool_call",
                    "tool_name": part.tool_name,
                    "tool_input": part.tool_input,
                    "tool_status": part.tool_status,
                }
                if part.skill_uri:
                    tool_info["skill_name"] = part.skill_uri.rstrip("/").split("/")[-1]
                tool_lines.append(f"[ToolCall] {json.dumps(tool_info, ensure_ascii=False)}")

        all_lines = tool_lines + text_lines
        return "\n".join(all_lines) if all_lines else ""

    def _build_text_mode_one(self, msg: GraphMsg) -> tuple[str, Dict[str, str]]:
        """Build text from written_entries (events/entities markdown).

        Returns (all_text, uri_to_content mapping).
        """
        import json, re

        text_segments: List[str] = []
        uri_to_content: Dict[str, str] = {}

        for entry in msg.written_entries:
            uri = entry.get("uri", "")
            after_md = entry.get("after", "")
            if not uri or not after_md:
                continue

            if "/events/" in uri:
                event_summary = ""
                match = re.search(
                    r"<!--\s*MEMORY_FIELDS\s*\n(.*?)\n\s*-->",
                    after_md,
                    re.DOTALL,
                )
                if match:
                    try:
                        fields = json.loads(match.group(1))
                        event_summary = fields.get("summary", "")
                    except Exception:
                        pass
                if not event_summary:
                    event_summary = after_md
                if event_summary:
                    text_segments.append(f"[SOURCE: {uri}]\n{event_summary}")
                    uri_to_content[uri] = event_summary

            elif "/entities/" in uri:
                entity_body = re.sub(
                    r"<!--\s*MEMORY_FIELDS\s*\n.*?\n\s*-->",
                    "",
                    after_md,
                    flags=re.DOTALL,
                ).strip()
                if entity_body:
                    text_segments.append(f"[SOURCE: {uri}]\n{entity_body}")
                    uri_to_content[uri] = entity_body

        all_text = "\n\n".join(text_segments) if text_segments else ""
        return all_text, uri_to_content

    def _build_text_mode_two(self, msg: GraphMsg) -> tuple[str, Dict[str, str]]:
        """Build text from session messages (same format as entity/event extraction).

        The message date is appended at the end of each line so the LLM can
        extract temporal information for relationships.

        Returns (all_text, uri_to_content mapping).
        """
        formatted_lines: List[str] = []
        uri_to_content: Dict[str, str] = {}

        for m in msg.messages:
            msg_content = self._format_message_with_parts(m)
            if not msg_content:
                continue
            created_at = getattr(m, "created_at", "") or ""
            if created_at:
                formatted_lines.append(f"[{m.role}]: {msg_content} ({created_at})")
            else:
                formatted_lines.append(f"[{m.role}]: {msg_content}")

        all_text = "\n".join(formatted_lines) if formatted_lines else ""
        return all_text, uri_to_content

    async def _process(self, msg: GraphMsg) -> None:
        """Process graph extraction message.

        Supports two text build modes:
        - mode_one: written_entries (events/entities markdown) with URI markers.
        - mode_two: session messages formatted like entity/event extraction.
        """
        import json, re

        # 1. Build text according to mode
        if self._text_build_mode == "mode_two":
            all_text, uri_to_content = self._build_text_mode_two(msg)
        else:
            all_text, uri_to_content = self._build_text_mode_one(msg)

        if not all_text:
            return

        # 2. Unified relation extraction from all text segments
        raw_triples: List[RawRelationTriple] = await self._relation_extractor.extract(
            text=all_text,
            user_id=msg.user_id,
            source=msg.source,
            mode=self._text_build_mode,
        )

        if not raw_triples:
            return

        # 2.5 mode_two: backfill rel_from with messages.jsonl URI
        if self._text_build_mode == "mode_two":
            if msg.source.startswith("viking://"):
                messages_uri = f"{msg.source}/messages.jsonl"
            elif msg.source:
                messages_uri = f"viking://session/{msg.source}/messages.jsonl"
            else:
                messages_uri = ""
            for rt in raw_triples:
                rt.rel_from = messages_uri

        # 3. Build RawEntity list from triples + user self-node
        seen: set = set()
        raw_entities: List[RawEntity] = []
        for rt in raw_triples:
            for name, tag in (
                (rt.from_entity, rt.from_entity_type),
                (rt.to_entity, rt.to_entity_type),
            ):
                if name not in seen:
                    seen.add(name)
                    raw_entities.append(RawEntity(entity=name, entity_type=tag))

        # Add user self-node
        user_id_norm = msg.user_id.lower().replace(" ", "_")
        if user_id_norm not in seen:
            raw_entities.append(RawEntity(entity=user_id_norm, entity_type="person"))

        # 4. Post-process entities (embedding + normalization)
        # Build event_content mapping from normalized event names
        entity_properties: Dict[str, Dict[str, Any]] = {}
        for entry in msg.written_entries:
            uri = entry.get("uri", "")
            if "/events/" not in uri:
                continue
            after_md = entry.get("after", "")
            if not after_md:
                continue
            event_summary = ""
            match = re.search(
                r"<!--\s*MEMORY_FIELDS\s*\n(.*?)\n\s*-->",
                after_md,
                re.DOTALL,
            )
            if match:
                try:
                    fields = json.loads(match.group(1))
                    event_summary = fields.get("summary", "")
                except Exception:
                    pass
            if not event_summary:
                event_summary = after_md
            parts = uri.split("/")
            if len(parts) >= 2:
                filename = parts[-1]
                name = os.path.splitext(filename)[0]
                normalized = name.lower().replace(" ", "_")
                entity_properties[normalized] = {"event_content": event_summary}

        entities = await self._entity_extractor.post_process(
            raw_entities=raw_entities,
            source="memory",
            account_id=msg.account_id,
            user_id=msg.user_id,
            entity_properties=entity_properties,
        )

        # 5. Convert RawRelationTriple to RelationTriple
        relations: List[RelationTriple] = []
        for rt in raw_triples:
            rel_date = rt.rel_date or _extract_date(rt.rel_content) or ""
            relations.append(RelationTriple(
                from_entity=rt.from_entity,
                to_entity=rt.to_entity,
                relation=GraphRelation(
                    rel_desc=rt.rel_desc,
                    rel_from=rt.rel_from,
                    rel_date=rel_date,
                    rel_content=rt.rel_content,
                ),
            ))

        # 6. Write to Neo4j
        await self._graph_writer.write_extraction_result(
            ExtractionResult(entities=entities, relations=relations),
            account_id=msg.account_id,
            user_id=msg.user_id,
            source_text=all_text,
        )
