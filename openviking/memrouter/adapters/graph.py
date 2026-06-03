"""GraphAdapter — bridges MemBackendRouteResult to OpenViking graph retrieval.

Translates MemRouter outputs into parameters for OpenViking's
``POST /api/v1/graph/search/text`` endpoint, which returns natural-language
text suitable for direct injection into an LLM prompt.
"""

import logging
from typing import Any, Dict, Optional

from openviking.memrouter.adapters.openviking import MemoryBackendAdapter
from openviking.memrouter.query_spec import GraphQuerySpec, TemplateQuerySpec
from openviking.memrouter.result import MemBackendRouteResult, QueryHints

logger = logging.getLogger(__name__)


class GraphAdapter(MemoryBackendAdapter):
    """Adapter for OpenViking graph (knowledge-graph) backend.

    In the current phase the physical graph backend is Neo4j inside OpenViking.
    MemRouter decides "route to graph" and the adapter calls
    ``/api/v1/graph/search/text`` with the original query (plus optional
    entity hints for future hop-depth control).
    """

    BACKEND_ID = "graph_memory_backend"

    # Fallback mapping: intent_family -> GraphQuerySpec defaults.
    _INTENT_QUERY_MAP: Dict[str, Dict[str, Any]] = {
        "entity_relation_query": {
            "graph_query_type": "entity_relation",
            "hop_depth": 1,
            "traversal_direction": "both",
        },
        "causal_multihop": {
            "graph_query_type": "causal_multihop",
            "hop_depth": 2,
            "traversal_direction": "out",
        },
        "system_dependency_graph": {
            "graph_query_type": "system_dependency",
            "hop_depth": 2,
            "traversal_direction": "both",
        },
        # Temporal queries routed to graph — graph relations carry rel_date attributes
        "graph_temporal_query": {
            "graph_query_type": "temporal_lookup",
            "hop_depth": 1,
            "traversal_direction": "both",
        },
        "graph_duration_query": {
            "graph_query_type": "temporal_lookup",
            "hop_depth": 2,
            "traversal_direction": "both",
        },
        "graph_sequence_query": {
            "graph_query_type": "temporal_lookup",
            "hop_depth": 2,
            "traversal_direction": "both",
        },
    }
    _DEFAULT_GRAPH_SPEC: Dict[str, Any] = {
        "graph_query_type": "entity_lookup",
        "hop_depth": 1,
        "traversal_direction": "both",
    }

    def __init__(self, endpoint: Optional[str] = None) -> None:
        """Args:
            endpoint: Base URL of OpenViking HTTP server,
                      e.g. ``http://127.0.0.1:1933``.
                      If None, translate() returns parameters only;
                      callers are responsible for HTTP transport.
        """
        self._endpoint = endpoint.rstrip("/") if endpoint else None
        logger.info("GraphAdapter initialized (endpoint=%s)", self._endpoint)

    @property
    def backend_id(self) -> str:
        return self.BACKEND_ID

    def get_default_spec(self, intent_family: str) -> Optional[TemplateQuerySpec]:
        """Return a TemplateQuerySpec with GraphQuerySpec from fallback mapping."""
        spec_dict = self._INTENT_QUERY_MAP.get(intent_family)
        if spec_dict is None and not intent_family:
            spec_dict = self._DEFAULT_GRAPH_SPEC
        if spec_dict is None:
            logger.debug(
                "No adapter fallback mapping for intent_family=%s", intent_family
            )
            return None

        graph = GraphQuerySpec(
            graph_query_type=spec_dict.get("graph_query_type", "entity_lookup"),
            hop_depth=spec_dict.get("hop_depth", 1),
            traversal_direction=spec_dict.get("traversal_direction", "both"),
        )
        return TemplateQuerySpec(
            search_mode="graph_traversal",
            graph=graph,
        )

    def enrich_spec(
        self,
        base_spec: TemplateQuerySpec,
        hints: QueryHints,
        raw_query: str,
    ) -> TemplateQuerySpec:
        """Runtime enrichment: inject detected entities as root_entity_hint.

        If exactly one entity is detected we treat it as the anchor node for
        graph traversal.  Multiple entities are left for the backend to resolve.
        """
        import copy

        spec = copy.deepcopy(base_spec)
        if spec.graph is None:
            return spec

        graph = spec.graph
        if hints.entities and graph.root_entity_hint is None:
            if len(hints.entities) == 1:
                graph.root_entity_hint = hints.entities[0]
                logger.debug(
                    "Enriched root_entity_hint with single entity: %s",
                    graph.root_entity_hint,
                )
            else:
                # Heuristic: pick the entity that appears first in the raw query
                lower_query = raw_query.lower()
                positions = {
                    ent: lower_query.find(ent.lower())
                    for ent in hints.entities
                    if lower_query.find(ent.lower()) >= 0
                }
                if positions:
                    first = min(positions, key=positions.get)  # type: ignore[arg-type]
                    graph.root_entity_hint = first
                    logger.debug(
                        "Enriched root_entity_hint with first-occurring entity: %s",
                        first,
                    )
        return spec

    def translate(self, route_result: MemBackendRouteResult) -> Dict[str, Any]:
        """Build parameters for ``POST /api/v1/graph/search/text``.

        The returned dict can be passed directly as JSON payload to the
        OpenViking graph endpoint.
        """
        if not route_result.routes:
            logger.warning("Route result has no routes; returning empty translation")
            return {}

        route = next(
            (r for r in route_result.routes if r.backend_id == self.BACKEND_ID),
            route_result.routes[0],
        )
        if route.backend_id != self.BACKEND_ID:
            raise ValueError(
                f"GraphAdapter cannot translate route for backend '{route.backend_id}'; "
                f"expected '{self.BACKEND_ID}'."
            )

        # Extract graph-specific params from the query instruction (if present)
        extra: Dict[str, Any] = {}
        instructions = route_result.query_instructions
        inst = next(
            (i for i in instructions if i.backend_id == self.BACKEND_ID),
            None,
        )
        if inst is not None:
            extra = inst.extra_params or {}

        params: Dict[str, Any] = {
            "query": route_result.raw_user_query,
            "top_k": extra.get("hop_depth", 1) * 5 + 5,  # heuristic: deeper hops → more results
        }

        # Forward root_entity_hint as a hint param (backend may use it for
        # anchor-node boosting in future versions; currently ignored by
        # /search/text but harmless).
        root_hint = extra.get("root_entity_hint")
        if root_hint:
            params["root_entity_hint"] = root_hint

        logger.debug(
            "Translated route to Graph params: backend=%s top_k=%s",
            route.backend_id,
            params["top_k"],
        )
        return params

    def execute_mock(self, route_result: MemBackendRouteResult) -> Dict[str, Any]:
        """Mock execution for testing and integration validation."""
        params = self.translate(route_result)
        logger.info(
            "[Mock Graph call] backend=%s params=%s",
            self.backend_id,
            params,
        )
        return {"status": "mock_ok", "params": params}

    async def execute(
        self,
        route_result: MemBackendRouteResult,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> str:
        """Call OpenViking ``/api/v1/graph/search/text`` and return text.

        Returns the natural-language result string, or empty string on failure.
        """
        import aiohttp

        if not self._endpoint:
            raise RuntimeError("GraphAdapter endpoint not configured")

        params = self.translate(route_result)
        url = f"{self._endpoint}/api/v1/graph/search/text"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if data.get("status") == "ok":
                    return str(data.get("result", ""))
                logger.warning("[GraphAdapter] error response: %s", data)
                return ""
