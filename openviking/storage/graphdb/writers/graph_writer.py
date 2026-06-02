# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Graph writer: upsert nodes and relations into Neo4j."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from openviking_cli.utils.logger import get_logger

from ..graph_models import ExtractionResult, GraphEntity, GraphRelation
from ..neo4j_backend import Neo4jBackend
from .deduplicator import NodeDeduplicator

logger = get_logger(__name__)


class GraphWriter:
    """Write extraction results to Neo4j with deduplication."""

    def __init__(self, backend: Neo4jBackend, deduplicator: NodeDeduplicator):
        self._backend = backend
        self._dedup = deduplicator

    async def write_extraction_result(
        self,
        result: ExtractionResult,
        account_id: str,
        user_id: str,
        source_text: str = "",
    ) -> Dict[str, Any]:
        """Write extraction result to graph database.

        Flow:
        1. Upsert each entity node (source + name dedup).
        2. Upsert each relation triple.
        3. Return write stats.
        """
        entity_map: Dict[str, GraphEntity] = {}

        # Step 1: Upsert nodes
        for entity in result.entities:
            db_name = await self._upsert_node(entity, account_id, user_id)
            entity_map[entity.name] = entity

        # Step 2: Upsert relations
        relation_count = 0
        for triple in result.relations:
            src_entity = entity_map.get(triple.from_entity)
            dst_entity = entity_map.get(triple.to_entity)
            if src_entity and dst_entity:
                await self._upsert_relation(
                    src_entity.name,
                    src_entity.source,
                    dst_entity.name,
                    dst_entity.source,
                    triple.relation,
                    account_id,
                    user_id,
                )
                relation_count += 1

        return {
            "entities_written": len(entity_map),
            "relations_written": relation_count,
        }

    async def _upsert_node(
        self,
        entity: GraphEntity,
        account_id: str,
        user_id: str,
    ) -> str:
        """Upsert a single node. Returns the node name used in DB."""
        # 1. Try find existing node
        existing = await self._dedup.find_existing_node(
            entity.name, entity.source, account_id, user_id
        )
        if existing:
            db_name = existing["name"]
            await self._backend.execute(
                """
                MATCH (n:Node {
                    name: $name,
                    account_id: $aid, user_id: $uid
                })
                SET n.updated_at = datetime()
                """,
                {
                    "name": db_name,
                    "aid": account_id,
                    "uid": user_id,
                },
            )
            return db_name

        # 2. Create new node
        await self._backend.execute(
            """
            CREATE (n:Node {
                name: $name, source: $source, tag: $tag,
                properties: $properties,
                account_id: $aid, user_id: $uid,
                created_at: datetime(), updated_at: datetime()
            })
            """,
            {
                "name": entity.name,
                "source": entity.source,
                "tag": entity.tag,
                "properties": json.dumps(entity.properties, ensure_ascii=False) if entity.properties else None,
                "aid": account_id,
                "uid": user_id,
            },
        )

        # 3. Write embedding
        if entity.embedding is not None:
            await self._backend.execute(
                """
                MATCH (n:Node {
                    name: $name, source: $source,
                    account_id: $aid, user_id: $uid
                })
                SET n.embedding = $embedding
                """,
                {
                    "name": entity.name,
                    "source": entity.source,
                    "aid": account_id,
                    "uid": user_id,
                    "embedding": entity.embedding,
                },
            )

        return entity.name

    async def _upsert_relation(
        self,
        from_name: str,
        from_source: str,
        to_name: str,
        to_source: str,
        relation: GraphRelation,
        account_id: str,
        user_id: str,
    ) -> None:
        """Upsert a relation edge between two nodes."""
        await self._backend.execute(
            """
            MATCH (a:Node {
                name: $from_name, source: $from_source,
                account_id: $aid, user_id: $uid
            })
            MATCH (b:Node {
                name: $to_name, source: $to_source,
                account_id: $aid, user_id: $uid
            })
            MERGE (a)-[r:RELATION {rel_desc: $rel_desc}]->(b)
            ON CREATE SET
                r.valid = true,
                r.rel_from = $rel_from,
                r.rel_date = $rel_date,
                r.rel_content = $rel_content
            ON MATCH SET
                r.valid = true,
                r.rel_from = $rel_from,
                r.rel_date = $rel_date,
                r.rel_content = $rel_content,
                r.invalidated_at = null
            """,
            {
                "from_name": from_name,
                "from_source": from_source,
                "to_name": to_name,
                "to_source": to_source,
                "rel_desc": relation.rel_desc,
                "rel_from": relation.rel_from,
                "rel_date": relation.rel_date or "",
                "rel_content": relation.rel_content,
                "aid": account_id,
                "uid": user_id,
            },
        )
