"""Analyze token usage: separate bot tokens from fallback tokens.

Usage:
    python analyze_token_usage.py <master_run_dir>

Example:
    python analyze_token_usage.py ../results/ablation/treatment_20260603_150657_batched
"""
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


def extract_bot_tokens(master_dir: str) -> Dict[str, Any]:
    """Extract bot (chat completion) tokens from route_results.jsonl.

    Bot tokens = prompt_tokens + completion_tokens consumed by the bot agent
    when generating the final answer. Recorded in route_results.chat_usage field.
    """
    batch_dir = Path(master_dir) / "batch_results"
    if not batch_dir.exists():
        return {"error": f"batch_results not found: {batch_dir}"}

    total_prompt = 0
    total_completion = 0
    total = 0
    case_count = 0

    for batch_subdir in sorted(os.listdir(batch_dir)):
        route_file = batch_dir / batch_subdir / "results" / "route_results.jsonl"
        if not route_file.exists():
            continue

        with open(route_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    usage = row.get("chat_usage", {})
                    if usage:
                        pt = usage.get("prompt_tokens", 0)
                        ct = usage.get("completion_tokens", 0)
                        if pt > 0 or ct > 0:
                            total_prompt += pt
                            total_completion += ct
                            total += usage.get("total_tokens", pt + ct)
                            case_count += 1
                except:
                    pass

    return {
        "cases_with_data": case_count,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total,
        "avg_prompt_per_case": total_prompt / case_count if case_count else 0,
        "avg_completion_per_case": total_completion / case_count if case_count else 0,
        "avg_total_per_case": total / case_count if case_count else 0,
    }


def extract_fallback_tokens(master_dir: str) -> Dict[str, Any]:
    """Extract fallback routing tokens from all batch route_results.jsonl.

    Fallback tokens = prompt_tokens + completion_tokens consumed by the
    LLMBackendRouter when template matching fails and LLM fallback is triggered.
    Recorded in extra_route_events[*].debug.llm_fallback_meta.token_usage.
    """
    batch_dir = Path(master_dir) / "batch_results"
    if not batch_dir.exists():
        return {"error": f"batch_results not found: {batch_dir}"}

    # First, get QIDs where final route_method is llm_backend_fallback
    qa_path = Path(master_dir) / "results" / "qa_results.csv"
    fallback_qids = set()
    with open(qa_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("route_method") == "llm_backend_fallback":
                fallback_qids.add(row.get("case_id", ""))

    # Extract token usage from first fallback event per case
    total_prompt = 0
    total_completion = 0
    total = 0
    found_cases = []
    missing_cases = []

    for batch_subdir in sorted(os.listdir(batch_dir)):
        route_file = batch_dir / batch_subdir / "results" / "route_results.jsonl"
        if not route_file.exists():
            continue

        with open(route_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    qid = row.get("case_id", "")
                    if qid not in fallback_qids:
                        continue

                    # Find first fallback event
                    for evt in row.get("extra_route_events", []):
                        if evt.get("route_method") == "llm_backend_fallback":
                            meta = evt.get("debug", {}).get("llm_fallback_meta", {})
                            usage = meta.get("token_usage", {})
                            pt = usage.get("prompt_tokens", 0)
                            ct = usage.get("completion_tokens", 0)
                            if pt > 0 or ct > 0:
                                total_prompt += pt
                                total_completion += ct
                                total += pt + ct
                                found_cases.append({
                                    "qid": qid,
                                    "backend": evt.get("backend_id", ""),
                                    "prompt": pt,
                                    "completion": ct,
                                    "total": pt + ct,
                                })
                            else:
                                missing_cases.append(qid)
                            break
                except:
                    pass

    # Cases where fallback was triggered but no token data found
    found_qids = {c["qid"] for c in found_cases}
    for qid in fallback_qids:
        if qid not in found_qids and qid not in missing_cases:
            missing_cases.append(qid)

    result = {
        "expected_fallback_cases": len(fallback_qids),
        "found_with_token_data": len(found_cases),
        "missing_token_data": len(missing_cases),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total,
    }

    if found_cases:
        result["avg_prompt_per_fallback"] = total_prompt / len(found_cases)
        result["avg_completion_per_fallback"] = total_completion / len(found_cases)
        result["avg_total_per_fallback"] = total / len(found_cases)

    # Estimate total using average for missing cases
    if found_cases and missing_cases:
        avg_total = total / len(found_cases)
        estimated_missing = len(missing_cases) * avg_total
        result["estimated_total_with_missing"] = total + estimated_missing
        result["estimated_avg_per_fallback"] = (total + estimated_missing) / len(fallback_qids)

    result["per_case_details"] = found_cases
    result["missing_cases"] = missing_cases

    return result


def extract_all_routing_tokens(master_dir: str) -> Dict[str, Any]:
    """Extract ALL fallback calls (including verification_search phases).

    This counts every LLM fallback call made during routing, not just
    the primary routing decision.
    """
    batch_dir = Path(master_dir) / "batch_results"
    if not batch_dir.exists():
        return {"error": f"batch_results not found: {batch_dir}"}

    total_prompt = 0
    total_completion = 0
    total_calls = 0
    seen_cases = set()
    primary_fallback_cases = []

    for batch_subdir in sorted(os.listdir(batch_dir)):
        route_file = batch_dir / batch_subdir / "results" / "route_results.jsonl"
        if not route_file.exists():
            continue

        with open(route_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    qid = row.get("case_id", "")

                    for evt in row.get("extra_route_events", []):
                        if evt.get("route_method") == "llm_backend_fallback":
                            meta = evt.get("debug", {}).get("llm_fallback_meta", {})
                            usage = meta.get("token_usage", {})
                            pt = usage.get("prompt_tokens", 0)
                            ct = usage.get("completion_tokens", 0)
                            total_prompt += pt
                            total_completion += ct
                            total_calls += 1

                            if qid not in seen_cases:
                                seen_cases.add(qid)
                                primary_fallback_cases.append({
                                    "qid": qid,
                                    "backend": evt.get("backend_id", ""),
                                    "prompt": pt,
                                    "completion": ct,
                                    "total": pt + ct,
                                })
                except:
                    pass

    return {
        "unique_cases_with_fallback": len(seen_cases),
        "total_fallback_calls": total_calls,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "avg_tokens_per_call": (total_prompt + total_completion) / total_calls if total_calls else 0,
        "primary_fallback_details": primary_fallback_cases,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_token_usage.py <master_run_dir>")
        print("Example:")
        print('  python analyze_token_usage.py ../results/ablation/treatment_20260603_150657_batched')
        sys.exit(1)

    master_dir = sys.argv[1]

    print("=" * 70)
    print("TOKEN USAGE ANALYSIS")
    print("=" * 70)
    print(f"Run directory: {master_dir}")
    print()

    # 1. Bot tokens
    print("-" * 70)
    print("1. BOT TOKENS (Chat Completion - Answer Generation)")
    print("-" * 70)
    bot = extract_bot_tokens(master_dir)
    if "error" in bot:
        print(f"  ERROR: {bot['error']}")
    else:
        print(f"  Cases with bot token data: {bot['cases_with_data']} / 81")
        print(f"  Total prompt tokens:     {bot['total_prompt_tokens']:,}")
        print(f"  Total completion tokens: {bot['total_completion_tokens']:,}")
        print(f"  Total bot tokens:        {bot['total_tokens']:,}")
        print(f"  Avg prompt/case:         {bot['avg_prompt_per_case']:.0f}")
        print(f"  Avg completion/case:     {bot['avg_completion_per_case']:.0f}")
        print(f"  Avg total/case:          {bot['avg_total_per_case']:.0f}")
    print()

    # 2. Fallback tokens (primary routing only)
    print("-" * 70)
    print("2. FALLBACK TOKENS (Primary Routing Decision Only)")
    print("-" * 70)
    fallback = extract_fallback_tokens(master_dir)
    if "error" in fallback:
        print(f"  ERROR: {fallback['error']}")
    else:
        print(f"  Expected fallback cases (final route_method=llm_backend_fallback): {fallback['expected_fallback_cases']}")
        print(f"  Found with token data:  {fallback['found_with_token_data']}")
        print(f"  Missing token data:     {fallback['missing_token_data']}")
        print()
        print(f"  Total prompt tokens (known):     {fallback['total_prompt_tokens']:,}")
        print(f"  Total completion tokens (known): {fallback['total_completion_tokens']:,}")
        print(f"  Total fallback tokens (known):   {fallback['total_tokens']:,}")
        if "estimated_total_with_missing" in fallback:
            print(f"  Estimated total (with missing):  {fallback['estimated_total_with_missing']:,.0f}")
            print(f"  Estimated avg per fallback:      {fallback['estimated_avg_per_fallback']:.0f}")
        print()
        if fallback["per_case_details"]:
            print("  Per-case breakdown:")
            for c in fallback["per_case_details"]:
                print(f"    {c['qid']:<12s} backend={c['backend']:<28s} prompt={c['prompt']:>4d} comp={c['completion']:>4d} total={c['total']:>4d}")
        if fallback["missing_cases"]:
            print(f"  Missing token data for: {', '.join(fallback['missing_cases'])}")
    print()

    # 3. All routing fallback calls
    print("-" * 70)
    print("3. ALL FALLBACK CALLS (Primary + Verification Search)")
    print("-" * 70)
    all_fb = extract_all_routing_tokens(master_dir)
    if "error" in all_fb:
        print(f"  ERROR: {all_fb['error']}")
    else:
        print(f"  Unique cases with any fallback: {all_fb['unique_cases_with_fallback']}")
        print(f"  Total fallback calls (all phases): {all_fb['total_fallback_calls']}")
        print(f"  Total prompt tokens:     {all_fb['total_prompt_tokens']:,}")
        print(f"  Total completion tokens: {all_fb['total_completion_tokens']:,}")
        print(f"  Total fallback tokens:   {all_fb['total_tokens']:,}")
        print(f"  Avg tokens per call:     {all_fb['avg_tokens_per_call']:.0f}")
    print()

    # 4. Summary
    print("=" * 70)
    print("4. SUMMARY")
    print("=" * 70)
    if "error" not in bot and "error" not in fallback:
        bot_total = bot["total_tokens"]
        fb_total = fallback.get("estimated_total_with_missing", fallback.get("total_tokens", 0))
        print(f"  Bot tokens (answer generation):     {bot_total:>10,}")
        print(f"  Fallback tokens (routing):          {fb_total:>10,.0f}")
        print(f"  TOTAL LLM tokens:                   {bot_total + fb_total:>10,.0f}")
        print()
        print(f"  Routing tokens as % of total:       {fb_total / (bot_total + fb_total) * 100:.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()
