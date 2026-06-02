#!/usr/bin/env python
"""Standalone judge post-processor.

Reads an existing route_results.jsonl and runs LLM judge on each
question/response pair without re-invoking VikingBot.

Usage:
    python judge_existing_results.py path/to/route_results.jsonl --token "sk-..."
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx


SYSTEM_PROMPT = (
    "You are an expert grader that determines if answers to questions "
    "match a gold standard answer"
)

ACCURACY_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
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
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
"""


async def grade_one(
    client: httpx.AsyncClient,
    judge_base_url: str,
    judge_token: str,
    judge_model: str,
    question: str,
    gold_answer: str,
    response: str,
) -> tuple[bool | None, str]:
    try:
        url = f"{judge_base_url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": judge_token,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": judge_model,
            "max_tokens": 1024,
            "temperature": 0.0,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": ACCURACY_PROMPT.format(
                        question=question,
                        gold_answer=gold_answer,
                        response=response,
                    ),
                }
            ],
        }
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
        return None, f"[PARSE ERROR] Invalid response: {content}"
    except Exception as e:
        return None, f"[API ERROR] {str(e)}"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Judge existing evaluation results")
    parser.add_argument("input", help="Path to route_results.jsonl")
    parser.add_argument("--token", required=True, help="Judge API token (MiniMax)")
    parser.add_argument("--base-url", default="https://api.minimaxi.com/anthropic")
    parser.add_argument("--model", default="MiniMax-M2.7")
    parser.add_argument("--parallel", type=int, default=5, help="Concurrent requests")
    parser.add_argument("--output", default="", help="Output path (default: overwrite input)")
    parser.add_argument("--limit", type=int, default=0, help="Limit cases to judge")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else input_path

    # Read all records
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]

    total = len(records)
    print(f"Loaded {total} records from {input_path}")
    print(f"Judge model: {args.model}, parallel: {args.parallel}")
    print("=" * 60)

    semaphore = asyncio.Semaphore(args.parallel)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async def process_one(idx: int, row: dict) -> None:
            async with semaphore:
                case_id = row.get("case_id", f"case-{idx}")
                question = row.get("question", "")
                expected = row.get("expected_answer", "")
                response = row.get("response", "")
                error = row.get("error", "")

                if not response or error:
                    row["judge_correct"] = None
                    row["judge_reasoning"] = f"[SKIP] no response or error: {error}"
                    print(f"[{idx:03d}/{total}] {case_id} SKIP")
                    return

                print(f"[{idx:03d}/{total}] {case_id} grading...")
                is_correct, reasoning = await grade_one(
                    client,
                    args.base_url,
                    args.token,
                    args.model,
                    question,
                    expected,
                    response,
                )
                row["judge_correct"] = is_correct
                row["judge_reasoning"] = reasoning
                status = "CORRECT" if is_correct is True else ("WRONG" if is_correct is False else "ERROR")
                print(f"[{idx:03d}/{total}] {case_id} {status} | {reasoning[:60]}")

        await asyncio.gather(*[
            process_one(i, r) for i, r in enumerate(records, 1)
        ])

    # Write back
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    judged = sum(1 for r in records if r.get("judge_correct") is not None)
    correct = sum(1 for r in records if r.get("judge_correct") is True)
    errors = sum(1 for r in records if r.get("judge_correct") is None and r.get("judge_reasoning", "").startswith("[API"))
    print("=" * 60)
    print(f"Judged: {judged}/{total}")
    print(f"Correct: {correct}")
    print(f"Errors: {errors}")
    if judged > 0:
        print(f"Accuracy: {correct / judged * 100:.2f}%")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
