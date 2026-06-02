# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Entity extraction module using LLM Tool Calling."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from openviking_cli.utils.logger import get_logger

from ..graph_models import GraphEntity, RawEntity
from .prompts import EXTRACT_ENTITIES_TOOL, build_entity_extraction_messages

logger = get_logger(__name__)


class EntityExtractor:
    """Extract entities from text using LLM Tool Calling."""

    def __init__(self, llm_provider: Any, embedder: Any):
        self._llm = llm_provider
        self._embedder = embedder

    async def raw_extract(self, text: str, user_id: str = "") -> List[RawEntity]:
        """Stage 1: LLM extraction only, return raw entities.

        Args:
            text: Raw text to extract from.
            user_id: User identifier for self-referential pronoun replacement.

        Returns:
            List of RawEntity.
        """
        messages = build_entity_extraction_messages(text, user_id)
        result = await self._llm.get_completion_async(
            messages=messages,
            tools=[EXTRACT_ENTITIES_TOOL],
        )

        raw: List[RawEntity] = []
        for tc in result.tool_calls:
            if tc.name != "extract_entities":
                continue
            for item in tc.arguments.get("entities", []):
                raw.append(RawEntity(
                    entity=item.get("entity", ""),
                    entity_type=item.get("entity_type", ""),
                ))
        return raw

    async def extract(
        self,
        text: str,
        user_id: str = "",
        account_id: str = "",
        source: str = "",
    ) -> List[GraphEntity]:
        """Full extraction: LLM extract + post-processing.

        Args:
            text: Raw text.
            user_id: User identifier.
            account_id: Tenant identifier.
            source: Data source identifier.

        Returns:
            List of GraphEntity.
        """
        raw_entities = await self.raw_extract(text, user_id)
        return await self._post_process(raw_entities, source, account_id, user_id)

    async def post_process(
        self,
        raw_entities: List[RawEntity],
        source: str,
        account_id: str,
        user_id: str,
        entity_properties: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[GraphEntity]:
        """Post-process raw entities (can be called independently).

        Args:
            raw_entities: Raw entities from LLM.
            source: Data source.
            account_id: Tenant identifier.
            user_id: User identifier.
            entity_properties: Optional pre-set extended properties keyed by normalized name.

        Returns:
            List of GraphEntity.
        """
        entity_properties = entity_properties or {}
        entities: List[GraphEntity] = []
        now = datetime.now()

        for item in raw_entities:
            name = item.entity.strip()
            if not name:
                continue

            # Normalize
            normalized_name = name.lower().replace(" ", "_")

            # Embedding
            try:
                embedding = self._embedder.embed(normalized_name).dense_vector
            except Exception as e:
                logger.warning(f"[EntityExtractor] Embedding failed for '{normalized_name}': {e}")
                embedding = None

            # Extended properties
            properties = entity_properties.get(normalized_name, {}).copy()
            tag = item.entity_type.lower().replace(" ", "_") if item.entity_type else ""

            entities.append(GraphEntity(
                name=normalized_name,
                source=source,
                tag=tag,
                properties=properties,
                account_id=account_id,
                user_id=user_id,
                embedding=embedding,
                created_at=now,
                updated_at=now,
            ))

        return entities

    async def _post_process(
        self,
        raw_entities: List[RawEntity],
        source: str,
        account_id: str,
        user_id: str,
        entity_properties: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[GraphEntity]:
        """Internal post-processing helper."""
        return await self.post_process(raw_entities, source, account_id, user_id, entity_properties)
