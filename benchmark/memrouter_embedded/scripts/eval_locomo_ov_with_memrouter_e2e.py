#!/usr/bin/env python
"""LoCoMo + OpenViking with Embedded MemRouter — Agent-level E2E evaluator.

This runner goes through VikingBot's ``/bot/v1/chat`` endpoint.  MemRouter
routing now lives **inside** the OpenViking Server (not as an external SDK),
so VikingBot transparently calls standard endpoints.

The evaluator reads ``route_events.jsonl`` written by the OpenViking server's
embedded ``MemRouterService`` to compute routing observability metrics.

Usage::

    python eval_locomo_ov_with_memrouter_e2e.py \
      --dataset ../data/locomo10.json \
      --route-labels ../data/locomo_e2e_route_labels.v3.jsonl \
      --route-events-path D:/Code/cursorProject/OpenViking/logs/route_events.jsonl \
      --category 3 \
      --force-memory-search

"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

BACKENDS = [
    "openviking_memory_backend",
    "graph_memory_backend",
]


def _repo_root() -> Path:
    # benchmark/memrouter_embedded/scripts → benchmark/memrouter_embedded
    return Path(__file__).resolve().parents[1]


def _default_openviking_root() -> Path:
    # benchmark/memrouter_embedded → D:/Code/cursorProject/OpenViking
    return _repo_root().parent.parent


def _load_locomo_data(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_route_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load route labels JSONL. Key is case_id."""
    labels: dict[str, dict[str, Any]] = {}
    if path is None or not path.exists():
        return labels
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                case_id = row.get("case_id") or f"{row.get('sample_id')}_Q{row.get('qi')}"
                if not case_id:
                    continue
                expected_backend = row.get("expected_backend", "")
                if not expected_backend and isinstance(row.get("expected"), dict):
                    expected_backend = row["expected"].get("primary_backend_id", "")
                if expected_backend:
                    row["expected_backend"] = expected_backend
                labels[case_id] = row
            except json.JSONDecodeError:
                continue
    return labels


def _load_case_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    case_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("{"):
                try:
                    case_id = json.loads(raw).get("case_id", "")
                except json.JSONDecodeError:
                    case_id = ""
            else:
                case_id = raw
            if case_id:
                case_ids.add(case_id)
    return case_ids


def _load_route_events(path: Path, consumed: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Read new route events from JSONL. Returns (new_events, total_count)."""
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events, 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events[consumed:], len(events)


def _is_template_route(route_method: str) -> bool:
    return route_method in {
        "template_embedding",
        "template_embedding_multi_backend",
        "template_rerank",
    }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _format_memories_xml(memories_text: str, case_id: str) -> str:
    """Format retrieved memories as XML with index, type, score, uri, content.

    Parses the XML string returned by VikingBot (from MemoryStore) and
    re-formats it with full detail including similarity scores.
    Handles truncated XML gracefully.
    """
    if not memories_text or "<memory" not in memories_text:
        return ""

    # Strip markdown prefix like "### user memories:\n" before XML parsing
    cleaned = memories_text
    if "<memory" in cleaned:
        cleaned = cleaned[cleaned.index("<memory"):]

    lines = [f'<memories case_id="{case_id}">']

    # Use regex to extract complete <memory>...</memory> blocks
    # This handles truncated XML better than ET.fromstring
    import re
    mem_pattern = re.compile(r'<memory\s+([^>]*)>(.*?)</memory>', re.DOTALL)
    matches = list(mem_pattern.finditer(cleaned))

    if matches:
        for idx, m in enumerate(matches):
            attrs_str = m.group(1)
            inner = m.group(2)
            # Parse attributes
            mtype = "memory"
            for attr_match in re.finditer(r'(\w+)="([^"]*)"', attrs_str):
                if attr_match.group(1) == "type":
                    mtype = attr_match.group(2)

            # Extract uri, score, content from inner XML
            uri = ""
            uri_m = re.search(r'<uri>(.*?)</uri>', inner, re.DOTALL)
            if uri_m:
                uri = uri_m.group(1).strip()

            score = ""
            score_m = re.search(r'<score>(.*?)</score>', inner, re.DOTALL)
            if score_m:
                score = score_m.group(1).strip()

            content = ""
            for tag in ("abstract", "content", "summary"):
                cm = re.search(rf'<{tag}>(.*?)</{tag}>', inner, re.DOTALL)
                if cm:
                    content = cm.group(1).strip()
                    break

            if not content and uri:
                content = f"Memory from {uri}"

            lines.append(f'  <memory index="{idx}" type="{mtype}">')
            if uri:
                lines.append(f"    <uri>{uri}</uri>")
            if score:
                lines.append(f"    <score>{score}</score>")
            lines.append(f"    <content>{content[:500]}</content>")
            lines.append("  </memory>")
    else:
        # No complete memory blocks found — fallback to raw
        lines.append("  <parse_error>Raw memory text:</parse_error>")
        lines.append(f"  <raw><![CDATA[{memories_text[:2000]}]]></raw>")

    lines.append("</memories>")
    return "\n".join(lines)


def run_ov_chat(
    question: str,
    endpoint: str,
    headers: dict[str, str],
    session_id: str,
    user_id: str,
    timeout: int = 300,
) -> tuple[str, dict[str, Any], str]:
    """Call OpenViking /bot/v1/chat and return (response_text, usage, relevant_memories)."""
    url = f"{endpoint.rstrip('/')}/bot/v1/chat"
    payload = {
        "message": question,
        "session_id": session_id,
        "user_id": user_id,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        message = body.get("message", "")
        usage = body.get("usage") or {}
        relevant_memories = body.get("relevant_memories", "")
        return message, usage, relevant_memories
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Connection error to {url}: {e}")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Request timeout to {url} after {timeout}s")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code} from {url}: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Error parsing response from {url}: {e}")


def fetch_ov_memories(
    query: str,
    endpoint: str,
    headers: dict[str, str],
    user_id: str,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Call OpenViking /search/find to fetch raw memory results for verification."""
    url = f"{endpoint.rstrip('/')}/api/v1/search/find"
    payload = {"query": query, "user_id": user_id, "limit": 30}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") == "ok":
            result = body.get("result", {})
            if isinstance(result, dict):
                memories = result.get("memories", [])
            else:
                memories = body.get("memories", [])
            return [
                {
                    "uri": m.get("uri", ""),
                    "score": m.get("score", 0),
                    "context_type": m.get("context_type", ""),
                    "level": m.get("level", 0),
                    "abstract": m.get("abstract", ""),
                    "content": m.get("content", ""),
                }
                for m in memories
            ]
        return []
    except Exception as exc:
        print(f"[fetch_ov_memories WARN] {type(exc).__name__}: {exc}")
        return []


def get_sample_question_time(sample: dict[str, Any]) -> str | None:
    """Extract the last session date from a LoCoMo sample."""
    from datetime import datetime

    conversation = sample.get("conversation", {})
    session_keys = [
        k for k in conversation.keys()
        if k.startswith("session_") and "date_time" not in k
    ]
    if not session_keys:
        return None

    def _num(k: str) -> int:
        try:
            return int(k.replace("session_", ""))
        except ValueError:
            return 0

    session_keys.sort(key=_num, reverse=True)
    for sk in session_keys:
        if conversation.get(sk):
            num = _num(sk)
            dt_key = f"session_{num}_date_time"
            date_str = conversation.get(dt_key)
            if date_str:
                try:
                    if " on " in date_str:
                        date_part = date_str.split(" on ")[-1]
                        dt = datetime.strptime(date_part.strip(), "%d %B, %Y")
                        return dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass
    return None


async def _grade_answer(
    llm_client: Any,
    model: str,
    question: str,
    gold_answer: str,
    response: str,
) -> tuple[bool, str]:
    """Inline LLM judge using MiniMax anthropic-compatible API."""
    system_prompt = (
        "You are an expert grader that determines if answers to questions "
        "match a gold standard answer"
    )
    prompt = f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response.

Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
"""
    try:
        base = str(llm_client.base_url) if hasattr(llm_client.base_url, '__str__') else llm_client.base_url
        url = f"{base.rstrip('/')}/v1/messages"
        api_key = llm_client.headers.get("x-api-key", "") if hasattr(llm_client, "headers") else getattr(llm_client, "api_key", "")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 1024,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text_content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text_content = block.get("text", "")
                    break

            content = text_content.strip()
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx + 1].strip()
                result = json.loads(json_str)
                is_correct = result.get("is_correct", "WRONG").strip().upper() == "CORRECT"
                reasoning = result.get("reasoning", "")
                return is_correct, reasoning
            return False, f"[PARSE ERROR] Invalid response: {content}"
    except Exception as e:
        return False, f"[API ERROR] {str(e)}"


async def _run_cases(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Path, Path, Path]:
    openviking_root = Path(args.openviking_root).resolve()
    ov_config = Path(args.ov_config).resolve()

    sys.path.insert(0, str(openviking_root))
    sys.path.insert(0, str(openviking_root / "bot"))

    os.environ["OPENVIKING_CONFIG_FILE"] = str(ov_config)

    dataset_path = Path(args.dataset).resolve()
    dataset_name = dataset_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.fixed_run_dir:
        run_dir = Path(args.fixed_run_dir).resolve()
    else:
        base_dir = Path(args.output_base).resolve()
        run_dir = base_dir / f"{timestamp}_{dataset_name}_ov_memrouter_e2e"
    logs_dir = run_dir / "logs"
    results_dir = run_dir / "results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Route events path
    # ------------------------------------------------------------------ #
    if args.route_events_path:
        route_events_path = Path(args.route_events_path).resolve()
    else:
        # Default: use MemRouterService's configured path (ov.conf)
        route_events_path = (
            Path(args.openviking_root).resolve()
            / "benchmark"
            / "memrouter_embedded"
            / "logs"
            / "route_events.jsonl"
        )

    # Clear stale events so we only read events from this run
    if route_events_path.exists():
        try:
            route_events_path.unlink()
        except PermissionError:
            with open(route_events_path, "w", encoding="utf-8") as f:
                pass
    route_events_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LoCoMo + OV with Embedded MemRouter E2E Runner")
    print("=" * 60)
    print(f"Run directory:     {run_dir}")
    print(f"Logs directory:    {logs_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Route events path: {route_events_path}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Load route labels
    # ------------------------------------------------------------------ #
    route_labels = _load_route_labels(
        Path(args.route_labels) if args.route_labels else None
    )
    selected_case_ids = _load_case_ids(
        Path(args.case_ids_file) if args.case_ids_file else None
    )
    if route_labels:
        print(f"Loaded {len(route_labels)} route labels from {args.route_labels}")
    else:
        print("WARNING: No route labels loaded. backend_accuracy will be N/A.")
    if selected_case_ids:
        print(f"Loaded {len(selected_case_ids)} selected case ids from {args.case_ids_file}")

    data = _load_locomo_data(dataset_path)
    if args.limit_samples:
        data = data[:args.limit_samples]

    # Build QA cases
    cases: list[dict[str, Any]] = []
    for item in data:
        sample_id = item["sample_id"]
        question_time = get_sample_question_time(item)
        all_qas = item.get("qa", [])
        filtered_qas = []
        for original_qi, qa in enumerate(all_qas, start=1):
            case_id = f"{sample_id}_Q{original_qi}"
            if selected_case_ids and case_id not in selected_case_ids:
                continue
            cat = str(qa.get("category", ""))
            if cat == "5":
                continue
            if args.category and cat != args.category:
                continue
            filtered_qas.append((original_qi, qa))
        if args.limit_questions:
            filtered_qas = filtered_qas[:args.limit_questions]
        for original_qi, qa in filtered_qas:
            cases.append({
                "sample_id": sample_id,
                "qi": original_qi,
                "question": qa["question"],
                "expected_answer": str(qa["answer"]),
                "category": qa.get("category", ""),
                "evidence": qa.get("evidence", []),
                "question_time": question_time,
            })

    # Apply global limit and start/end offset
    if args.start_question > 1:
        cases = cases[args.start_question - 1:]
    if args.end_question > 0:
        end_idx = args.end_question - args.start_question + 1
        cases = cases[:end_idx]
    elif args.limit_questions:
        cases = cases[:args.limit_questions]

    (results_dir / "dataset_snapshot.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases),
        encoding="utf-8",
    )
    _write_json(
        results_dir / "run_config.json",
        {
            "dataset": str(dataset_path),
            "ov_config": str(ov_config),
            "openviking_root": str(openviking_root),
            "route_events_path": str(route_events_path),
            "route_labels": args.route_labels,
            "limit_samples": args.limit_samples,
            "limit_questions": args.limit_questions,
            "judge": args.judge,
            "scope": "VikingBot /bot/v1/chat -> OpenViking Server (embedded MemRouter)",
        },
    )

    # ------------------------------------------------------------------ #
    # OpenViking chat headers
    # ------------------------------------------------------------------ #
    chat_endpoint = args.ov_chat_endpoint
    search_endpoint = args.ov_search_endpoint or chat_endpoint
    base_chat_headers = {
        "Content-Type": "application/json",
        "X-API-Key": args.ov_api_key,
        "X-OpenViking-Account": args.ov_account,
    }
    explicit_user = args.ov_user if args.ov_user else None
    explicit_agent = args.ov_agent if args.ov_agent else None

    # Optional judge client
    judge_client = None
    if args.judge and args.judge_token:
        import httpx
        judge_client = httpx.AsyncClient(
            base_url=args.judge_base_url,
            headers={"x-api-key": args.judge_token},
            timeout=60.0,
        )

    results: list[dict[str, Any]] = []
    consumed_events = 0
    session_prefix = args.session_prefix or run_dir.name

    # Real-time output files
    live_results_path = results_dir / "route_results.jsonl"
    live_log_path = results_dir / "live_progress.log"
    live_memories_path = results_dir / "retrieved_memories.xml"

    for idx, case in enumerate(cases, 1):
        case_id = f"{case['sample_id']}_Q{case['qi']}"
        question = case["question"]
        expected_answer = case["expected_answer"]
        question_time = case.get("question_time")
        sample_id = case["sample_id"]

        label_row = route_labels.get(case_id, {})
        expected_backend = label_row.get("expected_backend", "")
        scenario = label_row.get("scenario", "")

        # Data was imported under user "default" (not per-sample user_id)
        chat_headers = dict(base_chat_headers)
        chat_headers["X-OpenViking-User"] = explicit_user or "default"
        chat_headers["X-OpenViking-Agent"] = explicit_agent or sample_id
        chat_user_id = explicit_user or "default"
        chat_session_id = f"{session_prefix}_{case_id}"

        task_prefix = (
            "Before answering, search the user's OpenViking memory using the "
            "memory search tool exactly once. Then answer the question directly: "
            if args.force_memory_search
            else "Answer the question directly: "
        )
        if question_time:
            input_msg = f"Current date: {question_time}. {task_prefix}{question}"
        else:
            input_msg = f"{task_prefix}{question}"

        started = time.perf_counter()
        error = ""
        response = ""
        usage: dict[str, Any] = {}
        relevant_memories = ""
        try:
            response, usage, relevant_memories = run_ov_chat(
                input_msg,
                chat_endpoint,
                chat_headers,
                session_id=chat_session_id,
                user_id=chat_user_id,
                timeout=args.chat_timeout,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = int((time.perf_counter() - started) * 1000)

        # ------------------------------------------------------------------ #
        # Read route events and correlate
        # ------------------------------------------------------------------ #
        new_events, consumed_events = _load_route_events(
            route_events_path, consumed_events
        )

        first_event = new_events[0] if new_events else None
        all_backends = list(
            dict.fromkeys(
                ev.get("backend_id", "") for ev in new_events if ev.get("backend_id")
            )
        )
        any_expected_hit = expected_backend in all_backends if expected_backend else False

        route_method = first_event.get("route_method", "") if first_event else ""
        actual_backend = first_event.get("backend_id", "") if first_event else ""
        matched_template = first_event.get("matched_template_id", "") if first_event else ""
        confidence = first_event.get("confidence") if first_event else None
        execution_path = first_event.get("execution_path", "") if first_event else ""
        route_latency_ms = first_event.get("latency_ms", 0) if first_event else 0

        # Format memories as XML
        memories_xml = _format_memories_xml(relevant_memories, case_id)

        # Independent verification search
        verification_search = fetch_ov_memories(
            question, search_endpoint, chat_headers, chat_user_id
        )

        # Answer grading
        judge_correct: bool | None = None
        judge_reasoning = ""
        if args.judge and response and not error and judge_client:
            judge_correct, judge_reasoning = await _grade_answer(
                judge_client, args.judge_model,
                question, expected_answer, response,
            )

        result = {
            "case_id": case_id,
            "sample_id": sample_id,
            "qi": case["qi"],
            "chat_session_id": chat_session_id,
            "chat_user_id": chat_user_id,
            "question": question,
            "expected_answer": expected_answer,
            "expected_backend": expected_backend,
            "scenario": scenario,
            "response": response,
            "category": case["category"],
            "error": error,
            "latency_ms": latency_ms,
            "chat_usage": usage,
            "route_method": route_method,
            "actual_backend": actual_backend,
            "first_route_backend": actual_backend,
            "all_route_backends": all_backends,
            "any_expected_backend_hit": any_expected_hit,
            "matched_template_id": matched_template,
            "confidence": confidence,
            "execution_path": execution_path,
            "route_latency_ms": route_latency_ms,
            "route_event_count": len(new_events),
            "extra_route_events": new_events[1:] if len(new_events) > 1 else [],
            "is_template_hit": _is_template_route(route_method),
            "is_backend_correct": (
                actual_backend == expected_backend
                if expected_backend and actual_backend else False
            ),
            "retrieved_memories": relevant_memories,
            "memories_xml": memories_xml,
            "verification_search": verification_search,
            "judge_correct": judge_correct,
            "judge_reasoning": judge_reasoning,
        }
        results.append(result)

        judge_status = f" judge={judge_correct}" if judge_correct is not None else ""
        event_status = f" events={len(new_events)}" if len(new_events) != 1 else ""
        print(
            f"[{idx:03d}/{len(cases)}] {case_id} "
            f"expected={expected_backend or '?'} "
            f"actual={actual_backend or 'none'} "
            f"method={route_method or 'none'}"
            f"{judge_status}{event_status} lat={latency_ms}ms"
        )

        # Real-time flush
        with live_results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        mem_count = relevant_memories.count("<memory") if relevant_memories else 0
        verification_count = len(verification_search) if verification_search else 0
        live_line = (
            f"[{idx:03d}/{len(cases)}] {case_id} | "
            f"Q: {question[:80]}{'...' if len(question) > 80 else ''} | "
            f"Route: {actual_backend or 'none'} ({route_method or 'none'}) | "
            f"Memories(used): {mem_count} | "
            f"Memories(verify): {verification_count} | "
            f"BackendOK={result['is_backend_correct']}"
            f"{f' Judge={judge_correct}' if judge_correct is not None else ''}"
            f"{f' ERROR={error[:60]}' if error else ''}\n"
        )
        with live_log_path.open("a", encoding="utf-8") as f:
            f.write(live_line)
            if verification_search:
                for mi, mem in enumerate(verification_search, 1):
                    uri = mem.get("uri", "")
                    score = mem.get("score", 0)
                    uri_short = uri.rsplit("/", 1)[-1] if "/" in uri else uri
                    if len(uri_short) > 50:
                        uri_short = uri_short[:47] + "..."
                    f.write(f"    VerifyMem[{mi:02d}] uri={uri_short} score={score:.3f}\n")
            f.flush()

        # Write XML memories
        with live_memories_path.open("a", encoding="utf-8") as f:
            f.write(f"\n<!-- Case {case_id} -->\n")
            f.write(memories_xml)
            f.write("\n")
            f.flush()

    if judge_client:
        await judge_client.aclose()

    return results, run_dir, logs_dir, results_dir


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    with_routes = [r for r in results if r["route_method"]]

    labeled_results = [r for r in results if r.get("expected_backend")]
    backend_correct = sum(1 for r in labeled_results if r.get("is_backend_correct"))
    any_backend_hit = sum(1 for r in labeled_results if r.get("any_expected_backend_hit"))
    template_hits = sum(1 for r in results if r.get("is_template_hit"))
    fallback = sum(1 for r in results if r.get("route_method") == "llm_backend_fallback")
    invalid = sum(1 for r in results if r.get("error"))

    by_expected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_actual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_expected[row.get("expected_backend") or "unknown"].append(row)
        by_actual[row.get("actual_backend") or "none"].append(row)

    judged = [r for r in results if r.get("judge_correct") is not None]
    correct_answers = sum(1 for r in judged if r["judge_correct"] is True)

    return {
        "overall": {
            "count": total,
            "with_route_observed": len(with_routes),
            "labeled_count": len(labeled_results),
            "backend_accuracy": _rate(backend_correct, len(labeled_results)) if labeled_results else None,
            "any_backend_hit_rate": _rate(any_backend_hit, len(labeled_results)) if labeled_results else None,
            "template_hit_rate": _rate(template_hits, len(with_routes)) if with_routes else None,
            "llm_fallback_rate": _rate(fallback, len(with_routes)) if with_routes else None,
            "invalid_rate": _rate(invalid, total),
            "answer_accuracy": _rate(correct_answers, len(judged)) if judged else None,
        },
        "by_expected_backend": {
            backend: {
                "count": len(rows),
                "backend_accuracy": _rate(
                    sum(1 for r in rows if r.get("is_backend_correct")), len(rows)
                ),
            }
            for backend, rows in sorted(by_expected.items())
        },
        "by_actual_backend": {
            backend: {"count": len(rows)}
            for backend, rows in sorted(by_actual.items())
        },
        "answer": {
            "judged": len(judged),
            "correct": correct_answers,
            "accuracy": _rate(correct_answers, len(judged)) if judged else None,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(out_dir: Path, results: list[dict[str, Any]]) -> None:
    csv_path = out_dir / "qa_results.csv"
    fieldnames = [
        "case_id", "sample_id", "qi", "question", "expected_answer",
        "expected_backend", "response", "category", "error", "latency_ms",
        "route_method", "actual_backend", "matched_template_id", "confidence",
        "execution_path", "is_template_hit", "is_backend_correct",
        "judge_correct", "judge_reasoning",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)


def _write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    pct = lambda v: "N/A" if v is None else f"{v * 100:.2f}%"
    lines = [
        "# LoCoMo + OV with Embedded MemRouter — E2E Summary",
        "",
        "Scope: VikingBot `/bot/v1/chat` -> OpenViking Server (embedded MemRouter).",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total cases | {summary['overall']['count']} |",
        f"| Route observed | {summary['overall']['with_route_observed']} |",
        f"| Labeled cases | {summary['overall']['labeled_count']} |",
        f"| Backend route accuracy | {pct(summary['overall']['backend_accuracy'])} |",
        f"| Any backend hit rate | {pct(summary['overall']['any_backend_hit_rate'])} |",
        f"| Template hit rate | {pct(summary['overall']['template_hit_rate'])} |",
        f"| LLM fallback rate | {pct(summary['overall']['llm_fallback_rate'])} |",
        f"| Invalid/error rate | {pct(summary['overall']['invalid_rate'])} |",
        f"| Answer accuracy | {pct(summary['overall']['answer_accuracy'])} |",
        "",
        "## By Expected Backend",
        "",
        "| Expected backend | Cases | Accuracy |",
        "| --- | ---: | ---: |",
    ]
    for backend, vals in summary["by_expected_backend"].items():
        lines.append(f"| `{backend}` | {vals['count']} | {pct(vals['backend_accuracy'])} |")

    lines.extend([
        "",
        "## By Actual Backend",
        "",
        "| Actual backend | Cases |",
        "| --- | ---: |",
    ])
    for backend, vals in summary["by_actual_backend"].items():
        lines.append(f"| `{backend}` | {vals['count']} |")

    lines.extend([
        "",
        "## Answer Correctness",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Judged | {summary['answer']['judged']} |",
        f"| Correct | {summary['answer']['correct']} |",
        f"| Accuracy | {pct(summary['answer']['accuracy'])} |",
    ])

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to LoCoMo JSON file.")
    parser.add_argument("--route-labels", default="", help="Path to JSONL route labels.")
    parser.add_argument("--case-ids-file", default="", help="File with case_ids to evaluate.")
    parser.add_argument("--route-events-path", default="", help="Path to route_events.jsonl.")
    parser.add_argument(
        "--ov-config",
        default=str(_default_openviking_root() / "benchmark" / "memrouter_embedded" / "config" / "ov.conf"),
        help="OpenViking ov.conf path.",
    )
    parser.add_argument("--openviking-root", default=str(_default_openviking_root()), help="OpenViking repo root.")
    parser.add_argument("--output-base", default=str(_repo_root().parent / "runs"), help="Base output directory.")
    parser.add_argument("--fixed-run-dir", default="", help="Use exact run directory.")
    parser.add_argument("--ov-chat-endpoint", default="http://127.0.0.1:1937", help="VikingBot chat endpoint.")
    parser.add_argument("--ov-search-endpoint", default="", help="OV search endpoint (defaults to chat endpoint).")
    parser.add_argument("--ov-api-key", default="ov-test-key-12345", help="X-API-Key header.")
    parser.add_argument("--ov-account", default="default", help="X-OpenViking-Account header.")
    parser.add_argument("--ov-user", default="", help="X-OpenViking-User header.")
    parser.add_argument("--ov-agent", default="", help="X-OpenViking-Agent header.")
    parser.add_argument("--chat-timeout", type=int, default=120, help="Chat timeout.")
    parser.add_argument("--session-prefix", default="", help="Session ID prefix.")
    parser.add_argument("--force-memory-search", action="store_true", default=False, help="Force memory search.")
    parser.add_argument("--limit-samples", type=int, default=0, help="Limit samples.")
    parser.add_argument("--limit-questions", type=int, default=0, help="Limit questions.")
    parser.add_argument("--start-question", type=int, default=1, help="Start from question N (1-based).")
    parser.add_argument("--end-question", type=int, default=0, help="End at question N (1-based, 0=unset).")
    parser.add_argument("--category", type=str, default="", help="Filter by category.")
    parser.add_argument("--judge", action="store_true", default=False, help="Run LLM judge.")
    parser.add_argument("--judge-base-url", default="https://api.minimaxi.com/anthropic", help="Judge base URL.")
    parser.add_argument("--judge-token", default="", help="Judge API token.")
    parser.add_argument("--judge-model", default="MiniMax-M2.7", help="Judge model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.judge and not args.judge_token:
        print("Error: --judge requires --judge-token", file=sys.stderr)
        return 1

    results, run_dir, logs_dir, results_dir = asyncio.run(_run_cases(args))
    summary = _build_summary(results)

    with (results_dir / "route_results.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(results_dir / "metrics_summary.json", summary)
    _write_csv(results_dir, results)
    _write_report(results_dir, summary)

    print(f"\nRun directory:     {run_dir}")
    print(f"Logs directory:    {logs_dir}")
    print(f"Results directory: {results_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
