# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Search endpoints for OpenViking HTTP Server."""

import math
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from openviking.pyagfs.exceptions import AGFSClientError, AGFSNotFoundError
from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.server.telemetry import run_operation
from openviking.telemetry import TelemetryRequest
from openviking.utils.search_filters import merge_time_filter
from openviking_cli.exceptions import NotFoundError


def _sanitize_floats(obj: Any) -> Any:
    """Recursively replace inf/nan with 0.0 to ensure JSON compliance."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return 0.0
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


router = APIRouter(prefix="/api/v1/search", tags=["search"])
TimeField = Literal["updated_at", "created_at"]


def _resolve_search_limit(limit: int, node_limit: Optional[int]) -> int:
    return node_limit if node_limit is not None else limit


def _resolve_search_filter(
    request_filter: Optional[Dict[str, Any]],
    since: Optional[str],
    until: Optional[str],
    time_field: Optional[TimeField],
) -> Optional[Dict[str, Any]]:
    try:
        return merge_time_filter(
            request_filter,
            since=since,
            until=until,
            time_field=time_field,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class FindRequest(BaseModel):
    """Request model for find."""

    query: str
    target_uri: str = ""
    limit: int = 10
    node_limit: Optional[int] = None
    score_threshold: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None
    include_provenance: bool = False

    since: Optional[str] = None
    until: Optional[str] = None
    time_field: Optional[TimeField] = None
    telemetry: TelemetryRequest = False


class SearchRequest(BaseModel):
    """Request model for search with session."""

    query: str
    target_uri: str = ""
    session_id: Optional[str] = None
    limit: int = 10
    node_limit: Optional[int] = None
    score_threshold: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None
    include_provenance: bool = False

    since: Optional[str] = None
    until: Optional[str] = None
    time_field: Optional[TimeField] = None
    telemetry: TelemetryRequest = False


class GrepRequest(BaseModel):
    """Request model for grep."""

    uri: str
    exclude_uri: Optional[str] = None
    pattern: str
    case_insensitive: bool = False
    node_limit: Optional[int] = None
    level_limit: int = 5


class GlobRequest(BaseModel):
    """Request model for glob."""

    pattern: str
    uri: str = "viking://"
    node_limit: Optional[int] = None


class TypedQueryDict(BaseModel):
    """Sub-model for MemRouter-provided TypedQuery."""

    query: str = ""
    context_type: Optional[str] = None
    intent: str = ""
    priority: int = 1
    target_directories: Optional[List[str]] = None


class ExecuteInstructionRequest(BaseModel):
    """Request model for MemRouter fast-path execute_instruction."""

    query: str
    search_mode: str = "find"
    target_uri: str = ""
    context_type: Optional[str] = None
    limit: int = 10
    score_threshold: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None
    skip_intent_analysis: bool = False
    typed_query: Optional[TypedQueryDict] = None
    telemetry: TelemetryRequest = False


@router.post("/find")
async def find(
    request: FindRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Semantic search without session context."""
    service = get_service()
    actual_limit = _resolve_search_limit(request.limit, request.node_limit)
    effective_filter = _resolve_search_filter(
        request.filter,
        request.since,
        request.until,
        request.time_field,
    )
    execution = await run_operation(
        operation="search.find",
        telemetry=request.telemetry,
        fn=lambda: service.search.find(
            query=request.query,
            ctx=_ctx,
            target_uri=request.target_uri,
            limit=actual_limit,
            score_threshold=request.score_threshold,
            filter=effective_filter,
        ),
    )
    result = execution.result
    if hasattr(result, "to_dict"):
        result = result.to_dict(include_provenance=request.include_provenance)
    result = _sanitize_floats(result)
    return Response(
        status="ok",
        result=result,
        telemetry=execution.telemetry,
    ).model_dump(exclude_none=True)


@router.post("/search")
async def search(
    request: SearchRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Semantic search with optional session context."""
    service = get_service()
    actual_limit = _resolve_search_limit(request.limit, request.node_limit)
    effective_filter = _resolve_search_filter(
        request.filter,
        request.since,
        request.until,
        request.time_field,
    )

    async def _search():
        session = None
        if request.session_id:
            session = service.sessions.session(_ctx, request.session_id)
            await session.load()
        return await service.search.search(
            query=request.query,
            ctx=_ctx,
            target_uri=request.target_uri,
            session=session,
            limit=actual_limit,
            score_threshold=request.score_threshold,
            filter=effective_filter,
        )

    execution = await run_operation(
        operation="search.search",
        telemetry=request.telemetry,
        fn=_search,
    )
    result = execution.result
    if hasattr(result, "to_dict"):
        result = result.to_dict(include_provenance=request.include_provenance)
    result = _sanitize_floats(result)
    return Response(
        status="ok",
        result=result,
        telemetry=execution.telemetry,
    ).model_dump(exclude_none=True)


@router.post("/execute_instruction")
async def execute_instruction(
    request: ExecuteInstructionRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Execute a MemRouter-generated BackendQueryInstruction.

    Fast path: when ``skip_intent_analysis=true`` and ``typed_query`` is
    present, bypasses the native ``IntentAnalyzer`` and uses the
    MemRouter-provided ``TypedQuery`` directly.
    """
    service = get_service()

    instruction = {
        "query": request.query,
        "search_mode": request.search_mode,
        "target_uri": request.target_uri,
        "context_type": request.context_type,
        "limit": request.limit,
        "score_threshold": request.score_threshold,
        "filter": request.filter,
        "skip_intent_analysis": request.skip_intent_analysis,
    }
    if request.typed_query:
        instruction["typed_query"] = {
            "query": request.typed_query.query,
            "context_type": request.typed_query.context_type,
            "intent": request.typed_query.intent,
            "priority": request.typed_query.priority,
            "target_directories": request.typed_query.target_directories,
        }

    execution = await run_operation(
        operation="search.execute_instruction",
        telemetry=request.telemetry,
        fn=lambda: service.search.execute_instruction(
            instruction=instruction,
            ctx=_ctx,
        ),
    )
    result = execution.result
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    result = _sanitize_floats(result)
    return Response(
        status="ok",
        result=result,
        telemetry=execution.telemetry,
    ).model_dump(exclude_none=True)


@router.post("/grep")
async def grep(
    request: GrepRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Content search with pattern."""
    service = get_service()
    try:
        result = await service.fs.grep(
            request.uri,
            request.pattern,
            ctx=_ctx,
            exclude_uri=request.exclude_uri,
            case_insensitive=request.case_insensitive,
            node_limit=request.node_limit,
            level_limit=request.level_limit,
        )
    except AGFSNotFoundError:
        raise NotFoundError(request.uri, "file")
    except AGFSClientError as e:
        # Fallback for older versions without typed exceptions
        err_msg = str(e).lower()
        if "not found" in err_msg or "no such file or directory" in err_msg:
            raise NotFoundError(request.uri, "file")
        raise
    return Response(status="ok", result=result)


class GraphSearchRequest(BaseModel):
    """Request model for graph text search."""

    query: str
    top_k: int = 10


@router.post("/graph")
async def search_graph(
    request: GraphSearchRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Graph Memory semantic search returning natural-language text.

    Direct endpoint for graph-backed entity-relation retrieval.
    Bypasses MemRouter routing; callers that want automatic backend
    selection should use ``POST /search`` instead.
    """
    service = get_service()
    execution = await run_operation(
        operation="search.graph",
        telemetry=False,
        fn=lambda: service.search.search_graph_text(
            query=request.query,
            ctx=_ctx,
            top_k=request.top_k,
        ),
    )
    result = execution.result
    return Response(status="ok", result=result).model_dump(exclude_none=True)


@router.post("/glob")
async def glob(
    request: GlobRequest,
    _ctx: RequestContext = Depends(get_request_context),
):
    """File pattern matching."""
    service = get_service()
    try:
        result = await service.fs.glob(
            request.pattern, ctx=_ctx, uri=request.uri, node_limit=request.node_limit
        )
    except AGFSNotFoundError:
        raise NotFoundError(request.uri or request.pattern, "file")
    except AGFSClientError as e:
        # Fallback for older versions without typed exceptions
        err_msg = str(e).lower()
        if "not found" in err_msg or "no such file or directory" in err_msg:
            raise NotFoundError(request.uri or request.pattern, "file")
        raise
    return Response(status="ok", result=result)
