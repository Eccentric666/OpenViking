"""Quick test to verify compact LLM fallback prompt length and correctness."""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from openviking.memrouter.llm_fallback import (
    _build_system_prompt,
    _build_user_prompt,
    LLMFallbackContext,
    LLMRouterConfig,
)
from openviking.memrouter.registry import MemoryBackendRegistry, BackendEntry
from openviking.memrouter.result import QueryHints


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def main():
    # Build minimal registry with 2 backends
    registry = MemoryBackendRegistry()
    registry.register(BackendEntry(
        backend_id="openviking_memory_backend",
        backend_kind="openviking_native",
        description="Native semantic memory search",
        status="enabled",
    ))
    registry.register(BackendEntry(
        backend_id="graph_memory_backend",
        backend_kind="neo4j_graph",
        description="Graph database for relations",
        status="enabled",
    ))

    # Build context
    context = LLMFallbackContext(
        raw_user_query="What do Jon and Gina have in common?",
        normalized_user_query="what do jon and gina have in common",
        registry=registry,
        failed_template_summary=[
            {"template_id": "t_commonality", "backend_id": "graph_memory_backend", "score": 0.45}
        ],
        query_hints=QueryHints(),
        fallback_reason="no_template_above_threshold",
    )

    config = LLMRouterConfig(provider="openai_compatible", model="deepseek-v4-flash")

    system_prompt = _build_system_prompt(context, config.max_secondary_routes)
    user_prompt = _build_user_prompt(context)

    sys_tokens = estimate_tokens(system_prompt)
    user_tokens = estimate_tokens(user_prompt)
    total = sys_tokens + user_tokens

    print("=" * 60)
    print("COMPACT LLM FALLBACK PROMPT TEST")
    print("=" * 60)
    print(f"\n--- SYSTEM PROMPT ({sys_tokens} est. tokens) ---")
    print(system_prompt)
    print(f"\n--- USER PROMPT ({user_tokens} est. tokens) ---")
    print(user_prompt)
    print(f"\n--- TOTAL: {total} est. tokens ---")
    print(f"Target: < 1000 tokens")
    print(f"Status: {'✅ PASS' if total < 1000 else '❌ FAIL'}")

    # Test a few more queries
    test_queries = [
        ("What books has John read?", "openviking_memory_backend"),
        ("Who are Jon's friends?", "graph_memory_backend"),
        ("What do X and Y have in common?", "graph_memory_backend"),
    ]

    print("\n--- Routing sanity check (prompt content only) ---")
    for query, expected in test_queries:
        ctx = LLMFallbackContext(
            raw_user_query=query,
            normalized_user_query=query.lower(),
            registry=registry,
            failed_template_summary=[],
            query_hints=QueryHints(),
            fallback_reason="test",
        )
        u = _build_user_prompt(ctx)
        print(f"  {query[:50]:50} -> user={estimate_tokens(u)} tokens")


if __name__ == "__main__":
    main()
