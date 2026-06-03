"""Verify compact LLM fallback prompt against known fallback cases from treatment run.

Reads the 22 LLM fallback cases from treatment results and re-routes them
with the compact prompt, comparing routing decisions.

Usage:
    export LLM_API_KEY="sk-..."
    export LLM_BASE_URL="https://api.deepseek.com/v1"
    python verify_compact_fallback.py
"""
import csv
import json
import os
import sys
import time

# Allow running from repo root or scripts dir
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from openviking.memrouter.llm_fallback import (
    _build_system_prompt,
    _build_user_prompt,
    LLMFallbackContext,
    LLMRouterConfig,
    OpenAICompatibleLLMBackendRouter,
)
from openviking.memrouter.registry import MemoryBackendRegistry, BackendEntry
from openviking.memrouter.result import QueryHints

# The 22 fallback cases from original treatment run (verified from qa_results.csv)
FALLBACK_CASES = [
    ("conv-30_Q11", "When did Gina team up with a local artist for some cool designs?", "graph_memory_backend"),
    ("conv-30_Q25", "Which events has Jon participated in to promote his dance studio?", "openviking_memory_backend"),
    ("conv-30_Q26", "What does Jon's dance studio offer?", "openviking_memory_backend"),
    ("conv-30_Q44", "What do the dancers in the photo represent?", "openviking_memory_backend"),
    ("conv-30_Q46", "What is Jon's attitude towards being part of the dance community?", "openviking_memory_backend"),
    ("conv-30_Q48", "What did Gina find for her clothing store on 1 February 2023?", "openviking_memory_backend"),
    ("conv-30_Q49", "What did Gina design for her store?", "openviking_memory_backend"),
    ("conv-30_Q52", "What made Gina choose the furniture and decor for her store?", "openviking_memory_backend"),
    ("conv-30_Q59", "Why did Jon shut down his bank account?", "openviking_memory_backend"),
    ("conv-30_Q61", "What does Jon's dance make him?", "openviking_memory_backend"),
    ("conv-30_Q62", "What did Gina receive from a dance contest?", "openviking_memory_backend"),
    ("conv-30_Q63", "How does Gina stay confident in her business?", "openviking_memory_backend"),
    ("conv-30_Q65", "Where is Gina's fashion internship?", "openviking_memory_backend"),
    ("conv-30_Q66", "What book is Jon currently reading?", "openviking_memory_backend"),
    ("conv-30_Q7", "When is Jon's group performing at a festival?", "graph_memory_backend"),
    ("conv-30_Q70", "What did Jon take a trip to Rome for?", "openviking_memory_backend"),
    ("conv-30_Q71", "What is Jon working on opening?", "openviking_memory_backend"),
    ("conv-30_Q73", "How does Jon feel about the opening night of his dance studio?", "openviking_memory_backend"),
    ("conv-30_Q76", "What does Gina say to Jon about the grand opening?", "openviking_memory_backend"),
    ("conv-30_Q78", "What did Gina make a limited edition line of?", "openviking_memory_backend"),
    ("conv-30_Q79", "According to Gina, what makes Jon a perfect mentor?", "openviking_memory_backend"),
    ("conv-30_Q8", "When did Gina launch an ad campaign for her store?", "graph_memory_backend"),
]


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def main():
    print("=" * 70)
    print("COMPACT FALLBACK PROMPT — SMALL-SCALE VALIDATION")
    print("=" * 70)

    # Build registry
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

    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
    if not api_key:
        print("ERROR: Set LLM_API_KEY environment variable.")
        sys.exit(1)

    config = LLMRouterConfig(
        provider="openai_compatible",
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_tokens=256,
        max_secondary_routes=0,
    )

    router = OpenAICompatibleLLMBackendRouter(config)

    correct = 0
    total_tokens_prompt = 0
    total_tokens_completion = 0
    results = []

    for i, (case_id, question, expected) in enumerate(FALLBACK_CASES, 1):
        context = LLMFallbackContext(
            raw_user_query=question,
            normalized_user_query=question.lower(),
            registry=registry,
            failed_template_summary=[],
            query_hints=QueryHints(),
            fallback_reason="no_template_above_threshold",
        )

        result = router.route(context)
        actual = result.routes[0].backend_id if result.routes else "ERROR"
        is_correct = actual == expected
        if is_correct:
            correct += 1

        # Extract token usage
        meta = result.debug.llm_fallback_meta or {}
        usage = meta.get("token_usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        total_tokens_prompt += pt
        total_tokens_completion += ct

        status = "OK" if is_correct else "XX"
        print(f"{i:2d}. {case_id:<12} {status} expected={expected:<28} got={actual:<28} (prompt={pt}, completion={ct})")

        results.append({
            "case_id": case_id,
            "question": question,
            "expected": expected,
            "actual": actual,
            "correct": is_correct,
            "prompt_tokens": pt,
            "completion_tokens": ct,
        })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total cases: {len(FALLBACK_CASES)}")
    print(f"Correct routing: {correct}/{len(FALLBACK_CASES)} ({correct/len(FALLBACK_CASES)*100:.1f}%)")
    print(f"Total prompt tokens: {total_tokens_prompt}")
    print(f"Total completion tokens: {total_tokens_completion}")
    print(f"Avg prompt tokens/case: {total_tokens_prompt/len(FALLBACK_CASES):.0f}")
    print(f"Avg completion tokens/case: {total_tokens_completion/len(FALLBACK_CASES):.0f}")
    print(f"Avg total tokens/case: {(total_tokens_prompt+total_tokens_completion)/len(FALLBACK_CASES):.0f}")

    # Save results
    results_dir = os.path.join(_PROJECT_ROOT, "benchmark", "memrouter_embedded", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "compact_fallback_verify.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_cases": len(FALLBACK_CASES),
            "correct": correct,
            "accuracy": correct / len(FALLBACK_CASES),
            "total_prompt_tokens": total_tokens_prompt,
            "total_completion_tokens": total_tokens_completion,
            "avg_prompt_tokens": total_tokens_prompt / len(FALLBACK_CASES),
            "avg_completion_tokens": total_tokens_completion / len(FALLBACK_CASES),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
