# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Graph storage manager.

Manages Neo4j lifecycle and provides query proxy methods.
Responsibilities are slimmed: CRUD and query delegation only.
Entity/relation write logic is handled by GraphWriter.
"""

from typing import Any, Dict, List, Optional

from openviking_cli.utils.logger import get_logger

from .neo4j_backend import Neo4jBackend
from .schema import GraphSchema

logger = get_logger(__name__)


class GraphManager:
    """Graph manager for Neo4j lifecycle and query operations."""

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "",
        neo4j_database: str = "neo4j",
        confidence_threshold: float = 0.8,
    ):
        self._backend = Neo4jBackend(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
        )
        self._confidence_threshold = confidence_threshold
        self._retriever: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize schema (indexes)."""
        try:
            from neo4j import AsyncGraphDatabase  # noqa: F401
        except ImportError:
            logger.warning("[GraphManager] neo4j package not installed, skipping initialization")
            return

        await GraphSchema.initialize(self._backend)
        logger.info("[GraphManager] Schema initialized")

    async def close(self) -> None:
        """Close Neo4j connection."""
        await self._backend.close()

    # ---- Query proxy methods ----

    async def get_neighbors(
        self,
        name: str,
        source: str,
        account_id: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """Get 1-hop neighbors of a node."""
        cypher = """
            MATCH (n:Node {name: $name, source: $source, account_id: $aid, user_id: $uid})
                  -[r:RELATION]-(m:Node)
            WHERE r.valid = true
              AND m.account_id = $aid
              AND m.user_id = $uid
            RETURN m.name AS neighbor, m.tag AS tag, r.rel_desc AS rel_desc
        """
        return await self._backend.execute(cypher, {
            "name": name,
            "source": source,
            "aid": account_id,
            "uid": user_id,
        })

    async def search_entities(
        self,
        name_pattern: str,
        account_id: str,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search nodes by name pattern."""
        cypher = """
            MATCH (n:Node)
            WHERE n.name CONTAINS $pattern
              AND n.account_id = $aid
              AND n.user_id = $uid
            RETURN n.name AS name, n.tag AS tag, n.source AS source
            LIMIT $limit
        """
        return await self._backend.execute(cypher, {
            "pattern": name_pattern,
            "aid": account_id,
            "uid": user_id,
            "limit": limit,
        })

    async def delete_relations_of(
        self,
        name: str,
        source: str,
        account_id: str,
        user_id: str,
    ) -> None:
        """Soft-delete all relations of a node (mark valid=false)."""
        cypher = """
            MATCH (n:Node {name: $name, source: $source, account_id: $aid, user_id: $uid})
                  -[r:RELATION]-(m)
            SET r.valid = false,
                r.invalidated_at = datetime()
        """
        await self._backend.execute(cypher, {
            "name": name,
            "source": source,
            "aid": account_id,
            "uid": user_id,
        })

    # ---- Retrieval integration ----

    async def initialize_retriever(
        self,
        embedder: Any,
        entity_extractor: Any,
    ) -> None:
        """Initialize retriever (called from ServiceCore after embedder is ready)."""
        from .retrieval.graph_retriever import GraphRetriever

        self._retriever = GraphRetriever(
            backend=self._backend,
            embedder=embedder,
            entity_extractor=entity_extractor,
        )

    async def search(
        self,
        query: str,
        account_id: str,
        user_id: str,
        top_k: int = 5,
    ) -> List[Any]:
        """Vector similarity search entry."""
        if not self._retriever:
            raise RuntimeError("Retriever not initialized. Call initialize_retriever() first.")
        return await self._retriever.search(
            query=query,
            account_id=account_id,
            user_id=user_id,
            top_k=top_k,
        )

    @property
    def backend(self) -> Neo4jBackend:
        """Expose backend for GraphWriter."""
        return self._backend
