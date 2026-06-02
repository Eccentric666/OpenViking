#!/usr/bin/env python
"""Judge LoCoMo QA results using MiniMax (anthropic_compatible) API.

Reads qa_results.csv and judges answer correctness against expected_answer.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Judge prompt
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM_PROMPT = (
    "You are an expert grader that determines if answers to questions match a gold standard answer."
)


def build_judge_prompt(question: str, gold_answer: str, predicted: str) -> str:
    return f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {predicted}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
"""


# --------------------------------------------------------------------------- #
# MiniMax Client (Anthropic-compatible)
# --------------------------------------------------------------------------- #

class MiniMaxJudgeClient:
    """MiniMax client using Anthropic-compatible API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)

    async def judge(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = await self.client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # MiniMax returns thinking block first, text block second
        text_content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_content = block.get("text", "")
                break
        return self._parse_judge_response(text_content)

    def _parse_judge_response(self, text: str) -> dict[str, Any]:
        """Extract JSON from judge response."""
        text = text.strip()
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1].strip()
            try:
                result = json.loads(json_str)
                is_correct = result.get("is_correct", "WRONG").strip().upper() == "CORRECT"
                return {
                    "is_correct": is_correct,
                    "reasoning": result.get("reasoning", ""),
                }
            except json.JSONDecodeError:
                pass
        # Fallback: substring check
        is_correct = "CORRECT" in text.upper() and "WRONG" not in text.upper()
        return {
            "is_correct": is_correct,
            "reasoning": f"[PARSE FALLBACK] raw: {text[:200]}",
        }

    async def close(self) -> None:
        await self.client.aclose()


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #

def _get_case_id(case: dict[str, str]) -> str:
    """Extract case_id from various CSV formats (memrouter or native_ov)."""
    if "case_id" in case and case["case_id"]:
        return case["case_id"]
    if "sample_id" in case:
        q_idx = case.get("question_index", "")
        return f"{case['sample_id']}_q{q_idx}"
    return "unknown"


def _get_expected_answer(case: dict[str, str]) -> str:
    """Extract gold answer from various CSV formats."""
    for key in ("expected_answer", "answer", "gold_answer"):
        if key in case and case[key]:
            return case[key]
    return ""


async def judge_case(
    client: MiniMaxJudgeClient,
    case: dict[str, str],
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        case_id = _get_case_id(case)
        question = case.get("question", "")
        gold = _get_expected_answer(case)
        predicted = case.get("response", "")

        result = {
            "case_id": case_id,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": predicted[:500] if predicted else "",
        }

        # Skip empty responses
        if not predicted or not predicted.strip():
            result["correct"] = False
            result["judge_reason"] = "empty_response"
            return result

        # Skip error responses
        if predicted.startswith("[ERROR"):
            result["correct"] = False
            result["judge_reason"] = f"error_response: {predicted[:100]}"
            return result

        try:
            judge_prompt = build_judge_prompt(question, gold, predicted)
            judge_result = await client.judge(JUDGE_SYSTEM_PROMPT, judge_prompt)
            result["correct"] = judge_result["is_correct"]
            result["judge_reason"] = judge_result["reasoning"]
        except Exception as exc:
            logger.warning("[%s] Judge failed: %s", case_id, exc)
            # Fallback: simple substring
            result["correct"] = (
                gold.lower() in predicted.lower()
                or predicted.lower() in gold.lower()
            )
            result["judge_reason"] = f"judge_error_fallback: {exc}"

        return result


async def run_judge(
    csv_path: Path,
    output_path: Path,
    base_url: str,
    api_key: str,
    model: str,
    limit: int | None,
    concurrency: int,
) -> dict[str, Any]:
    # Read CSV
    cases: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)

    if limit:
        cases = cases[:limit]

    logger.info("Loaded %d cases from %s", len(cases), csv_path)

    client = MiniMaxJudgeClient(base_url, api_key, model)
    sem = asyncio.Semaphore(concurrency)

    results: list[dict[str, Any]] = []
    correct_count = 0
    total_judged = 0

    for i, case in enumerate(cases, 1):
        case_id = _get_case_id(case)
        logger.info("[%d/%d] Judging %s...", i, len(cases), case_id)
        result = await judge_case(client, case, sem)
        results.append(result)
        if result.get("correct") is not None:
            total_judged += 1
            if result["correct"]:
                correct_count += 1

    await client.close()

    accuracy = correct_count / total_judged * 100 if total_judged else 0

    report = {
        "judge_model": model,
        "judge_provider": "minimax",
        "total_cases": len(cases),
        "total_judged": total_judged,
        "correct": correct_count,
        "accuracy_percent": round(accuracy, 2),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Judge report saved to: %s", output_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo QA Judge with MiniMax")
    parser.add_argument("--csv", type=Path, required=True, help="Path to qa_results.csv")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--base-url", type=str, default="https://api.minimaxi.com/anthropic")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--model", type=str, default="MiniMax-M2.7")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        print("ERROR: --api-key or MINIMAX_API_KEY env var required")
        sys.exit(1)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = args.output or args.csv.parent / f"judge_minimax_{timestamp}.json"

    report = asyncio.run(
        run_judge(
            csv_path=args.csv,
            output_path=output,
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            limit=args.limit,
            concurrency=args.concurrency,
        )
    )

    s = report
    print("\n" + "=" * 60)
    print("LoCoMo QA Judge Report (MiniMax)")
    print("=" * 60)
    print(f"Judge model  : {s['judge_model']}")
    print(f"Total judged : {s['total_judged']}")
    print(f"Correct      : {s['correct']}")
    print(f"Accuracy     : {s['accuracy_percent']}%")
    print("=" * 60)
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    import os
    import sys
    main()
