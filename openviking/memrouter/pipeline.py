"""MemRouterPipeline — assembles the full routing pipeline.

Provides a single entry point `route(query)` that runs through all stages:
    QueryNormalizer → QueryFeatureBuilder → TemplateMatcher → RouteDecision
"""

import logging
from pathlib import Path
from typing import Optional

from typing import List

from openviking.memrouter.adapters.graph import GraphAdapter
from openviking.memrouter.adapters.openviking import OpenVikingAdapter
from openviking.memrouter.adapters.streamlined import StreamlinedMemoryAdapter
from openviking.memrouter.decision import RouteDecision
from openviking.memrouter.embeddings.base import EmbeddingProvider
from openviking.memrouter.features import QueryFeatureBuilder
from openviking.memrouter.llm_fallback import LLMRouterConfig, create_llm_backend_router
from openviking.memrouter.matcher import TemplateMatcher
from openviking.memrouter.normalizer import QueryNormalizer
from openviking.memrouter.query_instruction_builder import QueryInstructionBuilder
from openviking.memrouter.registry import BackendEntry, CostProfile, MemoryBackendRegistry, QueryContract
from openviking.memrouter.request import MemoryRouteRequest
from openviking.memrouter.result import MemBackendRouteResult, QueryHints
from openviking.memrouter.templates import BackendRouteTemplateIndex

logger = logging.getLogger(__name__)

# Builtin template directory relative to this package
_BUILTIN_TEMPLATES_DIR = Path(__file__).parent / "templates_data"


class MemRouterPipeline:
    """Main routing pipeline for EchoMem MemRouter.

    Usage:
        pipeline = MemRouterPipeline.with_defaults(embedder)
        result = pipeline.route("你记得我之前说过什么吗？")
    """

    def __init__(
        self,
        registry: MemoryBackendRegistry,
        feature_builder: QueryFeatureBuilder,
        template_index: BackendRouteTemplateIndex,
        matcher: TemplateMatcher,
        decision: RouteDecision,
        query_instruction_builder: Optional[QueryInstructionBuilder] = None,
    ) -> None:
        self._registry = registry
        self._feature_builder = feature_builder
        self._template_index = template_index
        self._matcher = matcher
        self._decision = decision
        self._query_instruction_builder = query_instruction_builder
        logger.info("MemRouterPipeline initialized")

    def route_request(self, request: MemoryRouteRequest) -> MemBackendRouteResult:
        """Route a MemoryRouteRequest through the full pipeline.

        Args:
            request: Standardized memory route request.

        Returns:
            MemBackendRouteResult containing backend route and metadata.
        """
        raw_query = request.raw_user_query
        logger.info("Routing query: %s", raw_query)

        # Use caller-provided normalized query if available, else normalize
        if request.normalized_user_query:
            normalized_query = request.normalized_user_query
        else:
            normalized_query = self._feature_builder._normalizer.normalize(raw_query)

        # Stage 1: feature extraction
        # Pass caller-provided normalized query so embedding uses the same text
        # that will appear in the result; hints are still extracted from raw_query.
        features = self._feature_builder.build(
            raw_query=raw_query,
            normalized_query=normalized_query,
        )
        logger.debug(
            "Features extracted: entities=%s temporal=%s relation=%s",
            features.entities,
            features.temporal_hints,
            features.relation_hints,
        )

        # Stage 2: template matching
        template_cands, backend_cands = self._matcher.match(features)

        # Stage 3: route decision
        query_hints = QueryHints(
            entities=features.entities,
            temporal_hints=features.temporal_hints,
            relation_hints=features.relation_hints,
        )
        result = self._decision.decide(
            raw_query=raw_query,
            normalized_query=normalized_query,
            template_candidates=template_cands,
            backend_candidates=backend_cands,
            query_hints=query_hints,
        )

        # Stage 4: query instruction generation (optional, Template-First Hybrid)
        if self._query_instruction_builder is not None:
            result.query_instructions = self._query_instruction_builder.build(
                route_result=result,
                query_hints=query_hints,
            )

        logger.info(
            "Route result: method=%s backend=%s confidence=%s instructions=%d",
            result.route_method,
            result.routes[0].backend_id if result.routes else "none",
            result.routes[0].confidence if result.routes else "none",
            len(result.query_instructions),
        )
        return result

    def route(self, raw_query: str) -> MemBackendRouteResult:
        """Convenience method that wraps a raw query into MemoryRouteRequest.

        Args:
            raw_query: Original user query string.

        Returns:
            MemBackendRouteResult containing backend route and metadata.
        """
        return self.route_request(MemoryRouteRequest(raw_user_query=raw_query))

    @classmethod
    def with_defaults(
        cls,
        embedder: EmbeddingProvider,
        template_dir: Optional[Path] = None,
        llm_router_config: Optional[LLMRouterConfig] = None,
        enabled_backends: Optional[List[str]] = None,
    ) -> "MemRouterPipeline":
        """Factory that wires up the full pipeline with v1.4 defaults.

        Args:
            embedder: Embedding provider (sentence-transformers or OpenAI).
            template_dir: Optional directory of YAML templates. If None,
                uses the builtin templates shipped with the package.
            llm_router_config: Optional LLM router config. If None, uses mock
                (no real LLM calls). Pass a real config for benchmark evaluation.
            enabled_backends: Optional list of backend IDs to enable.
                If None, all backends are enabled.
                If provided, only listed backends participate in routing.
                ``openviking_memory_backend`` is always enabled regardless.
                Example: ``["openviking_memory_backend", "streamlined_memory_backend"]``
                disables graph and keeps ov+streamlined.

        Returns:
            Configured MemRouterPipeline ready for routing.
        """
        logger.info("Building MemRouterPipeline with default v1.4 configuration")

        # Normalize enabled_backends: ov is always included
        _all_backend_ids = {
            "openviking_memory_backend",
            "graph_memory_backend",
            "streamlined_memory_backend",
        }
        if enabled_backends is not None:
            enabled_set = set(enabled_backends)
            enabled_set.add("openviking_memory_backend")
        else:
            enabled_set = _all_backend_ids

        # 1. Registry — register all backends, but only enable requested ones
        registry = MemoryBackendRegistry()

        if "openviking_memory_backend" in enabled_set:
            registry.register(
                BackendEntry(
                    backend_id="openviking_memory_backend",
                    backend_kind="openviking_native",
                    status="enabled",
                    description="OpenViking native memory backend for personal semantic memory, profile, preferences, and general user context.",
                    query_contract=QueryContract(
                        input_format="natural_language_with_hints",
                        supports_entities=True,
                        supports_time_range=True,
                        supports_relation_hints=False,
                    ),
                )
            )

        if "graph_memory_backend" in enabled_set:
            registry.register(
                BackendEntry(
                    backend_id="graph_memory_backend",
                    backend_kind="knowledge_graph",
                    status="enabled",
                    description="Graph memory backend for entity relations, multi-hop queries, and co-participation. Physically connected via Neo4j.",
                    query_contract=QueryContract(
                        input_format="natural_language_with_hints",
                        supports_entities=True,
                        supports_time_range=False,
                        supports_relation_hints=True,
                    ),
                    cost_profile=CostProfile(latency_class="medium", token_cost_class="low"),
                )
            )

        if "streamlined_memory_backend" in enabled_set:
            registry.register(
                BackendEntry(
                    backend_id="streamlined_memory_backend",
                    backend_kind="streamlined_store",
                    status="enabled",
                    description="Streamlined memory backend for timeline facts, sequence reasoning, duration comparison, and task/thread/process context. Physically connected via OpenViking Streamlined Memory sidecar.",
                    query_contract=QueryContract(
                        input_format="natural_language_with_hints",
                        supports_entities=True,
                        supports_time_range=True,
                        supports_relation_hints=False,
                    ),
                    cost_profile=CostProfile(latency_class="low", token_cost_class="low"),
                )
            )
        logger.info(
            "Registered %d logical backend(s), enabled=%s",
            len(registry),
            registry.enabled_backend_ids(),
        )

        # 2. Feature builder
        normalizer = QueryNormalizer()
        feature_builder = QueryFeatureBuilder(embedder=embedder, normalizer=normalizer)

        # 3. Template index — load all templates then filter by enabled backends
        template_index = BackendRouteTemplateIndex()
        load_dir = template_dir or _BUILTIN_TEMPLATES_DIR
        loaded = template_index.load_from_directory(load_dir)
        if loaded == 0:
            logger.warning("No templates loaded from %s; routing will always default", load_dir)

        # Hot-swap: remove templates whose target backend is disabled
        _removed = []
        for tid, tmpl in list(template_index._templates.items()):
            if tmpl.target.primary_backend_id not in enabled_set:
                del template_index._templates[tid]
                _removed.append(tid)
        if _removed:
            logger.info(
                "Hot-swap: removed %d template(s) for disabled backend(s): %s",
                len(_removed),
                _removed,
            )

        # 4. Matcher
        matcher = TemplateMatcher(embedder=embedder, template_index=template_index)

        # 5. LLM router (mock by default, real for benchmark)
        llm_router = create_llm_backend_router(
            llm_router_config or LLMRouterConfig(provider="mock", model="mock")
        )

        # 6. Decision
        decision = RouteDecision(
            registry=registry,
            template_index=template_index,
            llm_router=llm_router,
        )

        # 7. Adapters & QueryInstructionBuilder (Template-First Hybrid)
        adapters: Dict[str, Any] = {
            "openviking_memory_backend": OpenVikingAdapter(),
        }
        if "streamlined_memory_backend" in enabled_set:
            adapters["streamlined_memory_backend"] = StreamlinedMemoryAdapter()
        if "graph_memory_backend" in enabled_set:
            adapters["graph_memory_backend"] = GraphAdapter()
        query_instruction_builder = QueryInstructionBuilder(
            template_index=template_index,
            adapters=adapters,
        )

        return cls(
            registry=registry,
            feature_builder=feature_builder,
            template_index=template_index,
            matcher=matcher,
            decision=decision,
            query_instruction_builder=query_instruction_builder,
        )
