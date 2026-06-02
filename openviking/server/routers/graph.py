# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Graph database endpoints for OpenViking HTTP Server."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response


router = APIRouter(prefix="/api/v1/graph", tags=["graph"])


class GraphSearchRequest(BaseModel):
    """Request model for graph search."""

    query: str
    top_k: int = 10
    similarity_threshold: Optional[float] = None


class GraphNeighborRequest(BaseModel):
    """Request model for getting neighbors."""

    name: str
    source: str = ""


@router.post("/search")
async def graph_search(
    request: GraphSearchRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Vector similarity search over the knowledge graph.

    Extracts entities from the query, finds similar nodes via embedding,
    and returns their 1-hop relations.
    """
    service = get_service()
    if not service.graph_manager:
        raise HTTPException(status_code=503, detail="Graph database not initialized")

    threshold = request.similarity_threshold
    if threshold is None:
        from openviking_cli.utils.config import get_openviking_config

        threshold = get_openviking_config().storage.graphdb.similarity_threshold

    try:
        results = await service.graph_manager.search(
            query=request.query,
            account_id=_ctx.account_id,
            user_id=_ctx.user.user_id,
            top_k=request.top_k,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    # Convert dataclasses to dicts for JSON serialization
    output = []
    for r in results:
        item: Dict[str, Any] = {
            "source": r.source,
            "source_uri": r.source_uri,
            "source_tag": r.source_tag,
            "rel_desc": r.rel_desc,
            "target": r.target,
            "target_uri": r.target_uri,
            "target_tag": r.target_tag,
            "rel_from": r.rel_from,
        }
        if r.source_properties:
            item["source_properties"] = r.source_properties
        if r.target_properties:
            item["target_properties"] = r.target_properties
        if r.rel_date:
            item["rel_date"] = r.rel_date
        if r.rel_content:
            item["rel_content"] = r.rel_content
        if r.score > 0:
            item["score"] = r.score
        output.append(item)

    return Response(status="ok", result=output).model_dump(exclude_none=True)


@router.post("/search/text")
async def graph_search_text(
    request: GraphSearchRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Vector similarity search over the knowledge graph (natural-language output).

    Same logic as /search but returns a human-readable text block suitable
    for direct injection into an LLM prompt.
    """
    service = get_service()
    if not service.graph_manager:
        raise HTTPException(status_code=503, detail="Graph database not initialized")

    threshold = request.similarity_threshold
    if threshold is None:
        from openviking_cli.utils.config import get_openviking_config

        threshold = get_openviking_config().storage.graphdb.similarity_threshold

    try:
        results = await service.graph_manager.search(
            query=request.query,
            account_id=_ctx.account_id,
            user_id=_ctx.user.user_id,
            top_k=request.top_k,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    from openviking.storage.graphdb.retrieval.formatter import GraphRetrievalFormatter

    text = GraphRetrievalFormatter.to_natural_language(results)
    return Response(status="ok", result=text).model_dump(exclude_none=True)


@router.post("/neighbors")
async def graph_neighbors(
    request: GraphNeighborRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Get 1-hop neighbors of a node."""
    service = get_service()
    if not service.graph_manager:
        raise HTTPException(status_code=503, detail="Graph database not initialized")

    results = await service.graph_manager.get_neighbors(
        name=request.name,
        source=request.source,
        account_id=_ctx.account_id,
        user_id=_ctx.user.user_id,
    )
    return Response(status="ok", result=results).model_dump(exclude_none=True)


@router.get("/status")
async def graph_status(
    _ctx: RequestContext = Depends(get_request_context),
):
    """Check graph database connectivity."""
    service = get_service()
    if not service.graph_manager:
        return {"enabled": False, "initialized": False}

    return {
        "enabled": True,
        "initialized": True,
        "retriever_initialized": service.graph_manager._retriever is not None,
    }
