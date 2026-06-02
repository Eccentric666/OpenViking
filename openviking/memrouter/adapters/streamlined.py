"""StreamlinedMemoryAdapter — bridges MemBackendRouteResult to Streamlined Memory recall.

Translates MemRouter outputs into parameters for OpenViking's
Streamlined Memory sidecar ``POST /recall_state`` endpoint.
"""

import logging
from typing import Any, Dict, Optional

from openviking.memrouter.adapters.openviking import MemoryBackendAdapter
from openviking.memrouter.query_spec import StreamlinedQuerySpec, TemplateQuerySpec
from openviking.memrouter.result import MemBackendRouteResult, QueryHints

logger = logging.getLogger(__name__)


class StreamlinedMemoryAdapter(MemoryBackendAdapter):
    """Adapter for OpenViking Streamlined Memory backend.

    Streamlined Memory handles task/thread/process state, timeline facts,
    duration comparisons, sequence reasoning, and debug traces.
    It is physically served by a sidecar process (default port 1944)
    that maintains a SQLite-backed observation store.
    """

    BACKEND_ID = "streamlined_memory_backend"

    # Fallback mapping: intent_family -> StreamlinedQuerySpec defaults.
    _INTENT_QUERY_MAP: Dict[str, Dict[str, Any]] = {
        "timeline_fact_query": {
            "recall_intent": "timeline_fact",
            "timeline_mode": "mixed",
        },
        "duration_comparison_query": {
            "recall_intent": "duration_comparison",
            "timeline_mode": "business_only",
        },
        "sequence_reasoning_query": {
            "recall_intent": "sequence_reasoning",
            "timeline_mode": "mixed",
        },
        "resume_task": {
            "recall_intent": "resume_task",
            "timeline_mode": "dialog_only",
        },
        "debug_trace": {
            "recall_intent": "debug_trace",
            "timeline_mode": "business_only",
        },
    }
    _DEFAULT_STREAMLINED_SPEC: Dict[str, Any] = {
        "scope": "auto",
        "view": "auto",
        "limit": 5,
        "timeline_mode": "auto",
        "recall_intent": "resume_task",
        "timeline_window_before": 2,
        "timeline_window_after": 1,
        "task_id_source": "caller_context",
        "session_policy": "prefer_session",
    }

    @property
    def backend_id(self) -> str:
        return self.BACKEND_ID

    def get_default_spec(self, intent_family: str) -> Optional[TemplateQuerySpec]:
        """Return a TemplateQuerySpec with StreamlinedQuerySpec from fallback mapping."""
        spec_dict = self._INTENT_QUERY_MAP.get(intent_family)
        if spec_dict is None and not intent_family:
            spec_dict = self._DEFAULT_STREAMLINED_SPEC
        if spec_dict is None:
            logger.debug(
                "No adapter fallback mapping for intent_family=%s", intent_family
            )
            return None

        streamlined = StreamlinedQuerySpec(
            scope=spec_dict.get("scope", "auto"),
            view=spec_dict.get("view", "auto"),
            limit=spec_dict.get("limit", 5),
            timeline_mode=spec_dict.get("timeline_mode", "auto"),
            recall_intent=spec_dict.get("recall_intent", "resume_task"),
            timeline_window_before=spec_dict.get("timeline_window_before", 2),
            timeline_window_after=spec_dict.get("timeline_window_after", 1),
            task_id_source=spec_dict.get("task_id_source", "caller_context"),
            session_policy=spec_dict.get("session_policy", "prefer_session"),
        )
        return TemplateQuerySpec(
            search_mode="streamlined_recall",
            streamlined=streamlined,
        )

    def enrich_spec(
        self,
        base_spec: TemplateQuerySpec,
        hints: QueryHints,
        raw_query: str,
    ) -> TemplateQuerySpec:
        """Runtime enrichment for Streamlined Memory spec.

        Currently a no-op; future versions may inject session_id or task_id
        from runtime context.
        """
        return base_spec

    def translate(self, route_result: MemBackendRouteResult) -> Dict[str, Any]:
        """Build parameters for Streamlined Memory recall.

        The returned dict can be passed as JSON payload to the sidecar.
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
                f"StreamlinedMemoryAdapter cannot translate route for backend '{route.backend_id}'; "
                f"expected '{self.BACKEND_ID}'."
            )

        # Extract streamlined-specific params from the query instruction (if present)
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
            "session_id": None,  # global recall by default
            "scope": extra.get("scope", "auto"),
            "view": extra.get("view", "auto"),
            "limit": extra.get("limit", 5),
            "timeline_mode": extra.get("timeline_mode", "auto"),
            "recall_intent": extra.get("recall_intent", "resume_task"),
        }

        logger.debug(
            "Translated route to Streamlined params: backend=%s scope=%s recall_intent=%s",
            route.backend_id,
            params["scope"],
            params["recall_intent"],
        )
        return params

    def execute_mock(self, route_result: MemBackendRouteResult) -> Dict[str, Any]:
        """Mock execution for testing and integration validation."""
        params = self.translate(route_result)
        logger.info(
            "[Mock Streamlined call] backend=%s params=%s",
            self.backend_id,
            params,
        )
        return {
            "status": "mock_ok",
            "params": params,
            "recalled": False,
            "state_block": "",
        }
