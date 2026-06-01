# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Search Service for OpenViking.

Provides semantic search operations: search, find.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from openviking.server.identity import RequestContext
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import NotInitializedError
from openviking_cli.utils import get_logger

if TYPE_CHECKING:
    from openviking.session import Session

logger = get_logger(__name__)


class SearchService:
    """Semantic search service."""

    def __init__(self, viking_fs: Optional[VikingFS] = None):
        self._viking_fs = viking_fs

    def set_viking_fs(self, viking_fs: VikingFS) -> None:
        """Set VikingFS instance (for deferred initialization)."""
        self._viking_fs = viking_fs

    def _ensure_initialized(self) -> VikingFS:
        """Ensure VikingFS is initialized."""
        if not self._viking_fs:
            raise NotInitializedError("VikingFS")
        return self._viking_fs

    async def search(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: str = "",
        session: Optional["Session"] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict] = None,
    ) -> Any:
        """Complex search with session context.

        Args:
            query: Query string
            target_uri: Target directory URI
            session: Session object for context
            limit: Max results
            score_threshold: Score threshold
            filter: Metadata filters

        Returns:
            FindResult
        """
        viking_fs = self._ensure_initialized()

        session_info = None
        if session:
            session_info = await session.get_context_for_search(query)

        result = await viking_fs.search(
            query=query,
            ctx=ctx,
            target_uri=target_uri,
            session_info=session_info,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter,
        )
        return result

    async def find(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: str = "",
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict] = None,
    ) -> Any:
        """Semantic search without session context.

        Args:
            query: Query string
            target_uri: Target directory URI
            limit: Max results
            score_threshold: Score threshold
            filter: Metadata filters

        Returns:
            FindResult
        """
        viking_fs = self._ensure_initialized()
        result = await viking_fs.find(
            query=query,
            ctx=ctx,
            target_uri=target_uri,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter,
        )
        return result

    async def execute_instruction(
        self,
        instruction: Dict[str, Any],
        ctx: RequestContext,
    ) -> Any:
        """Execute a MemRouter-generated BackendQueryInstruction.

        Fast path: when ``skip_intent_analysis=true`` and ``typed_query`` is
        present, bypasses the native ``IntentAnalyzer`` and uses the
        MemRouter-provided ``TypedQuery`` directly.

        Args:
            instruction: Dict representation of ``BackendQueryInstruction``.
            ctx: Request context for access control.

        Returns:
            FindResult with the same shape as ``find()`` / ``search()``.
        """
        viking_fs = self._ensure_initialized()
        result = await viking_fs.execute_instruction(
            instruction=instruction,
            ctx=ctx,
        )
        return result

    def set_graph_manager(self, graph_manager: Optional[Any] = None) -> None:
        """Set GraphManager instance for graph-backed search."""
        self._graph_manager = graph_manager

    async def search_graph_text(
        self,
        query: str,
        ctx: RequestContext,
        top_k: int = 10,
    ) -> str:
        """Graph-style semantic search returning natural-language text.

        When a GraphManager is configured and initialized, delegates to Neo4j
        for vector similarity + multi-hop subgraph retrieval. Otherwise falls
        back to the lightweight bridge (Qdrant/AGFS vector search formatted as
        entity-relation text).

        Args:
            query: Natural-language query.
            ctx: Request context for ACL / telemetry.
            top_k: Max number of entities to include.

        Returns:
            Human-readable text block describing retrieved entities and
            their semantic relationships, or empty string when no results.
        """
        # Try real graph backend first
        if getattr(self, "_graph_manager", None) is not None:
            try:
                results = await self._graph_manager.search(
                    query=query,
                    account_id=ctx.account_id,
                    user_id=ctx.user.user_id,
                    top_k=top_k,
                )
                if results:
                    from openviking.storage.graphdb.retrieval.formatter import (
                        GraphRetrievalFormatter,
                    )

                    return GraphRetrievalFormatter.to_natural_language(results)
            except Exception as e:
                logger.warning(f"[search_graph_text] Graph search failed, falling back: {e}")

        # Fallback: lightweight bridge using existing vector store
        viking_fs = self._ensure_initialized()
        result = await viking_fs.find(
            query=query,
            ctx=ctx,
            limit=top_k,
        )
        return self._format_graph_text(result, query, top_k)

    @staticmethod
    def _format_graph_text(
        result: Any,
        query: str,
        top_k: int = 10,
    ) -> str:
        """Format search results as graph-style natural language (fallback)."""
        memories = []
        if hasattr(result, "memories"):
            memories = list(getattr(result, "memories", []))
        elif isinstance(result, dict):
            memories = list(result.get("memories", []))

        if not memories:
            return ""

        lines = [f"知识图谱检索结果（查询: {query}）:", ""]
        for i, mem in enumerate(memories[:top_k], 1):
            if hasattr(mem, "uri"):
                uri = getattr(mem, "uri", "")
                abstract = getattr(mem, "abstract", "") or ""
                score = getattr(mem, "score", 0.0)
            else:
                uri = mem.get("uri", "")
                abstract = mem.get("abstract", "") or ""
                score = mem.get("score", 0.0)

            lines.append(f"[{i}] 实体: {uri}")
            if abstract:
                lines.append(f"    摘要: {abstract[:300]}")
            lines.append(f"    语义相关度: {score:.3f}")
            lines.append("")

        # Add a lightweight "relation" summary
        if len(memories) >= 2:
            lines.append("关联分析:")
            lines.append(
                f"  上述 {min(len(memories), top_k)} 个实体在语义空间中与查询 '"
                f"{query[:50]}...' 存在关联。"
            )
            lines.append("")

        return "\n".join(lines)
