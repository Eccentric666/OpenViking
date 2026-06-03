# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""MemRouterService — embedded MemRouter routing inside OpenViking Server.

Replaces the external SDK pattern (MemRouterVikingClient wrapping VikingClient)
with direct in-process calls.  MemRouter logic (template matching + backend
routing + query instruction generation) lives inside the OpenViking Server so
that VikingBot can call standard ``/search/search`` and ``/search/search_memory``
endpoints transparently.

Graph and Streamlined memory backends are NOT physically connected yet — they
fall back to OpenViking native search automatically.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from openviking.server.identity import RequestContext
from openviking_cli.utils import get_logger

logger = get_logger(__name__)


class MemRouterService:
    """Embedded MemRouter routing service for OpenViking.

    Usage (inside OpenViking Server)::

        service = MemRouterService(search_service, config)
        result = await service.search("query", ctx)
    """

    def __init__(
        self,
        search_service: Any,
        config: Dict[str, Any],
    ) -> None:
        self._search_service = search_service
        self._config = config
        self._pipeline: Optional[Any] = None
        self._route_events_file: Optional[TextIO] = None
        self._route_events_path: Optional[Path] = None

        self._init_pipeline()
        self._init_route_events()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _init_pipeline(self) -> None:
        """Import local memrouter package and build MemRouterPipeline."""
        try:
            from openviking.memrouter.embeddings.base import OpenAIEmbeddingProvider
            from openviking.memrouter.llm_fallback import LLMRouterConfig
            from openviking.memrouter.pipeline import MemRouterPipeline

            # Build embedding provider from config
            embedder = self._build_embedder()

            # Build LLM fallback config (optional)
            llm_config = self._build_llm_router_config()

            # Load templates
            template_dir = self._config.get("template_dir")
            if template_dir:
                template_dir = Path(template_dir)

            # Enabled backends — graph/streamlined are registered but fallback to OV
            enabled_backends = self._config.get("enabled_backends")

            self._pipeline = MemRouterPipeline.with_defaults(
                embedder=embedder,
                template_dir=template_dir,
                llm_router_config=llm_config,
                enabled_backends=enabled_backends,
            )
            logger.info(
                "MemRouterService initialized (templates=%s, backends=%s)",
                template_dir or "builtin",
                enabled_backends or "all",
            )
        except Exception as exc:
            logger.error("Failed to initialize MemRouterService: %s", exc, exc_info=True)
            self._pipeline = None

    def _build_embedder(self) -> Any:
        """Build local memrouter EmbeddingProvider from OV config."""
        from openviking.memrouter.embeddings.base import OpenAIEmbeddingProvider

        embedding_cfg = self._config.get("embedding", {})
        provider = embedding_cfg.get("provider", "openai")
        model = embedding_cfg.get("model", "text-embedding-v3")
        api_key = embedding_cfg.get("api_key", "")
        api_base = embedding_cfg.get("api_base", "")

        if provider in ("openai", "dashscope"):
            return OpenAIEmbeddingProvider(
                model=model,
                api_key=api_key,
                base_url=api_base or None,
                max_batch_size=10,  # DashScope limit
            )
        else:
            raise ValueError(f"Unsupported MemRouter embedding provider: {provider}")

    def _build_llm_router_config(self) -> Optional[Any]:
        """Build LLM fallback config from ov.conf."""
        llm_cfg = self._config.get("llm_fallback")
        if not llm_cfg:
            return None

        from openviking.memrouter.llm_fallback import LLMRouterConfig

        return LLMRouterConfig(
            provider=llm_cfg.get("provider", "openai"),
            model=llm_cfg.get("model", "deepseek-v4-flash"),
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", ""),
        )

    def _init_route_events(self) -> None:
        """Open route events JSONL file for observability."""
        path_str = self._config.get("route_events_path")
        if not path_str:
            return
        self._route_events_path = Path(path_str)
        self._route_events_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._route_events_file = open(
                self._route_events_path, "a", encoding="utf-8"
            )
            logger.info("Route events logging to: %s", self._route_events_path)
        except OSError as exc:
            logger.warning("Cannot open route events file: %s", exc)

    async def close(self) -> None:
        """Close route events file."""
        if self._route_events_file:
            self._route_events_file.close()
            self._route_events_file = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def is_ready(self) -> bool:
        """Return True if MemRouter pipeline is initialized."""
        return self._pipeline is not None

    def route(self, query: str) -> Optional[Dict[str, Any]]:
        """Run template matching + backend routing + query instruction generation.

        Args:
            query: User query string.

        Returns:
            Dict representation of MemBackendRouteResult, or None if pipeline
            is not initialized.
        """
        if not self._pipeline:
            return None
        try:
            result = self._pipeline.route(query)
            return result.model_dump()
        except Exception as exc:
            logger.error("MemRouter route failed: %s", exc, exc_info=True)
            return None

    async def search(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: str = "",
        limit: int = 10,
        **kwargs: Any,
    ) -> Any:
        """Full flow: route query → execute search → log event.

        For ``openviking_memory_backend`` with ``skip_intent_analysis=True``:
            Uses ``execute_instruction()`` fast path (bypasses IntentAnalyzer).
        For other backends (graph, streamlined, llm_fallback):
            Falls back to native OV ``search()``.

        Args:
            query: User query string.
            ctx: Request context.
            target_uri: Target directory URI.
            limit: Max results.
            **kwargs: Extra arguments passed to native search.

        Returns:
            Search result (same shape as SearchService.search).
        """
        if not self._pipeline:
            # Pipeline not ready — fallback to native search
            return await self._search_service.search(
                query=query,
                ctx=ctx,
                target_uri=target_uri,
                limit=limit,
                _skip_memrouter=True,
                **kwargs,
            )

        started = time.perf_counter()
        route_result: Optional[Dict[str, Any]] = None
        execution_path = "unknown"
        error = ""
        result: Any = None

        try:
            # Stage 1: Route query through MemRouter
            route_result = self.route(query)
            if not route_result:
                execution_path = "route_error_fallback"
                result = await self._search_service.search(
                    query=query,
                    ctx=ctx,
                    target_uri=target_uri,
                    limit=limit,
                    _skip_memrouter=True,
                    **kwargs,
                )
                return result

            instructions = route_result.get("query_instructions", [])
            if not instructions:
                # No instructions produced — fallback to native
                execution_path = "no_instructions_fallback"
                result = await self._search_service.search(
                    query=query,
                    ctx=ctx,
                    target_uri=target_uri,
                    limit=limit,
                    _skip_memrouter=True,
                    **kwargs,
                )
                return result

            inst = instructions[0]
            backend_id = inst.get("backend_id", "")
            skip_ia = inst.get("skip_intent_analysis", False)

            logger.info(
                "MemRouter route: backend=%s skip_ia=%s search_mode=%s",
                backend_id,
                skip_ia,
                inst.get("search_mode", "unknown"),
            )

            # Stage 2: Execute based on routing decision
            if backend_id == "openviking_memory_backend" and skip_ia:
                # Fast path: bypass IntentAnalyzer
                execution_path = "memrouter_fast_path"
                logger.info(
                    "MemRouter fast path: template=%s confidence=%.4f",
                    route_result.get("routes", [{}])[0].get("matched_template_id", "") if route_result else "",
                    route_result.get("routes", [{}])[0].get("confidence", 0.0) if route_result else 0.0,
                )
                instruction_dict = self._build_instruction_dict(inst, query, limit)
                result = await self._search_service.execute_instruction(
                    instruction=instruction_dict, ctx=ctx
                )
            elif backend_id == "graph_memory_backend":
                # Graph fast path: direct graph retrieval via GraphManager
                execution_path = "memrouter_graph_fast_path"
                logger.info(
                    "MemRouter graph fast path: template=%s confidence=%.4f",
                    route_result.get("routes", [{}])[0].get("matched_template_id", "") if route_result else "",
                    route_result.get("routes", [{}])[0].get("confidence", 0.0) if route_result else 0.0,
                )
                try:
                    graph_text = await self._search_service.search_graph_text(
                        query=query, ctx=ctx, top_k=limit
                    )
                    if graph_text:
                        # Wrap graph NL text into a FindResult-compatible dict
                        result = {
                            "memories": [
                                {
                                    "uri": "viking://graph/result",
                                    "context_type": "memory",
                                    "level": 2,
                                    "abstract": graph_text,
                                    "score": 1.0,
                                    "category": "graph",
                                    "match_reason": "graph_backend",
                                    "relations": [],
                                }
                            ],
                            "resources": [],
                            "skills": [],
                            "total": 1,
                        }
                    else:
                        # Empty graph result — fallback to native search
                        execution_path = "memrouter_graph_empty_fallback"
                        logger.info(
                            "Graph search returned empty; falling back to native OV search"
                        )
                        result = await self._search_service.search(
                            query=query,
                            ctx=ctx,
                            target_uri=target_uri,
                            limit=limit,
                            _skip_memrouter=True,
                            **kwargs,
                        )
                except Exception as graph_exc:
                    execution_path = "memrouter_graph_error_fallback"
                    logger.warning(
                        "Graph search failed (%s); falling back to native OV search",
                        graph_exc,
                    )
                    result = await self._search_service.search(
                        query=query,
                        ctx=ctx,
                        target_uri=target_uri,
                        limit=limit,
                        _skip_memrouter=True,
                        **kwargs,
                    )
            else:
                # Fallback: llm_fallback / no template hit / route error → native OV search
                execution_path = "memrouter_fallback_to_native"
                logger.info(
                    "MemRouter fallback: backend=%s skip_ia=%s — using native OV search",
                    backend_id,
                    skip_ia,
                )
                result = await self._search_service.search(
                    query=query,
                    ctx=ctx,
                    target_uri=target_uri,
                    limit=limit,
                    _skip_memrouter=True,
                    **kwargs,
                )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("MemRouter search failed: %s", error, exc_info=True)
            # Final fallback: native search
            execution_path = "exception_fallback"
            result = await self._search_service.search(
                query=query,
                ctx=ctx,
                target_uri=target_uri,
                limit=limit,
                _skip_memrouter=True,
                **kwargs,
            )
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._log_route_event(
                query=query,
                route_result=route_result,
                execution_path=execution_path,
                latency_ms=latency_ms,
                error=error,
            )

        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_instruction_dict(
        inst: Dict[str, Any], query: str, limit: int
    ) -> Dict[str, Any]:
        """Build instruction dict for execute_instruction from BackendQueryInstruction."""
        instruction: Dict[str, Any] = {
            "query": inst.get("query", query),
            "search_mode": inst.get("search_mode", "find"),
            "target_uri": inst.get("target_uri", ""),
            "context_type": inst.get("context_type"),
            "limit": limit,
            "score_threshold": inst.get("score_threshold"),
            "filter": inst.get("filter"),
            "skip_intent_analysis": inst.get("skip_intent_analysis", False),
        }

        # Handle typed_query
        typed_query = inst.get("typed_query")
        if typed_query:
            instruction["typed_query"] = {
                "query": typed_query.get("query", query),
                "context_type": typed_query.get("context_type"),
                "intent": typed_query.get("intent", ""),
                "priority": typed_query.get("priority", 1),
                "target_directories": typed_query.get("target_directories"),
            }

        # Clean None values
        instruction = {k: v for k, v in instruction.items() if v is not None}
        return instruction

    def _log_route_event(
        self,
        query: str,
        route_result: Optional[Dict[str, Any]],
        execution_path: str,
        latency_ms: int,
        error: str = "",
    ) -> None:
        """Append a route event to the JSONL log."""
        if not self._route_events_file:
            return

        event: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query": query,
            "execution_path": execution_path,
            "latency_ms": latency_ms,
        }

        if route_result:
            event["route_method"] = route_result.get("route_method", "unknown")
            routes = route_result.get("routes", [])
            if routes:
                event["backend_id"] = routes[0].get("backend_id", "")
                event["template_id"] = routes[0].get("matched_template_id", "")
                event["confidence"] = routes[0].get("confidence", 0.0)
            event["fallback"] = route_result.get("fallback", {})
            debug = route_result.get("debug", {})
            event["debug"] = {
                "top_templates": debug.get("top_templates", []),
                "llm_fallback_meta": debug.get("llm_fallback_meta", {}),
            }

        if error:
            event["error"] = error

        try:
            self._route_events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._route_events_file.flush()
        except OSError as exc:
            logger.warning("Failed to write route event: %s", exc)
