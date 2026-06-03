#!/usr/bin/env python3
"""Merge batch evaluation results into a single unified report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def load_route_results(batch_dirs: list[Path]) -> list[dict[str, Any]]:
    """Load and merge all route_results.jsonl files."""
    all_results: list[dict[str, Any]] = []
    for batch_dir in batch_dirs:
        rr_file = batch_dir / "results" / "route_results.jsonl"
        if not rr_file.exists():
            print(f"  [WARN] Missing {rr_file}", file=sys.stderr)
            continue
        with rr_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return all_results


def load_csv_rows(batch_dirs: list[Path]) -> list[dict[str, str]]:
    """Load and merge all qa_results.csv files."""
    all_rows: list[dict[str, str]] = []
    for batch_dir in batch_dirs:
        csv_file = batch_dir / "results" / "qa_results.csv"
        if not csv_file.exists():
            print(f"  [WARN] Missing {csv_file}", file=sys.stderr)
            continue
        with csv_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
    return all_rows


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary metrics from merged results."""
    total = len(results)

    # Timeout / error stats
    errors = [r for r in results if r.get("error")]
    timeouts = [r for r in errors if "timeout" in str(r.get("error", "")).lower()]
    http_errors = [r for r in errors if "504" in str(r.get("error", "")) or "Gateway" in str(r.get("error", ""))]

    # Judge stats
    judged = [r for r in results if r.get("judge_correct") is not None]
    judge_correct = sum(1 for r in judged if r.get("judge_correct") is True)

    # Backend routing stats
    backend_labeled = [r for r in results if r.get("is_backend_correct") is not None]
    backend_correct = sum(1 for r in backend_labeled if r.get("is_backend_correct") is True)

    # Template hit stats
    template_labeled = [r for r in results if r.get("is_template_hit") is not None]
    template_hit = sum(1 for r in template_labeled if r.get("is_template_hit") is True)

    # Any backend hit stats
    any_hit_labeled = [r for r in results if r.get("any_expected_backend_hit") is not None]
    any_hit = sum(1 for r in any_hit_labeled if r.get("any_expected_backend_hit") is True)

    # Route method distribution
    route_methods: dict[str, int] = {}
    for r in results:
        rm = r.get("route_method", "unknown")
        route_methods[rm] = route_methods.get(rm, 0) + 1

    # Backend distribution
    backends: dict[str, int] = {}
    for r in results:
        b = r.get("actual_backend", "unknown")
        backends[b] = backends.get(b, 0) + 1

    # Token stats (for cases that have usage data)
    prompt_tokens = [r.get("prompt_tokens", 0) for r in results if r.get("prompt_tokens")]
    completion_tokens = [r.get("completion_tokens", 0) for r in results if r.get("completion_tokens")]
    total_tokens = [r.get("total_tokens", 0) for r in results if r.get("total_tokens")]

    def _avg(vals: list[int]) -> float | None:
        return round(sum(vals) / len(vals), 2) if vals else None

    def _sum(vals: list[int]) -> int:
        return sum(vals)

    # Latency stats
    latencies = [r.get("latency_ms", 0) for r in results if r.get("latency_ms")]
    route_latencies = [r.get("route_latency_ms", 0) for r in results if r.get("route_latency_ms")]

    summary: dict[str, Any] = {
        "total_cases": total,
        "error_count": len(errors),
        "timeout_count": len(timeouts),
        "http_504_count": len(http_errors),
        "judge": {
            "judged_count": len(judged),
            "correct_count": judge_correct,
            "accuracy": _rate(judge_correct, len(judged)),
        },
        "backend_routing": {
            "labeled_count": len(backend_labeled),
            "correct_count": backend_correct,
            "accuracy": _rate(backend_correct, len(backend_labeled)),
        },
        "template": {
            "labeled_count": len(template_labeled),
            "hit_count": template_hit,
            "hit_rate": _rate(template_hit, len(template_labeled)),
        },
        "any_backend_hit": {
            "labeled_count": len(any_hit_labeled),
            "hit_count": any_hit,
            "hit_rate": _rate(any_hit, len(any_hit_labeled)),
        },
        "route_methods": route_methods,
        "actual_backends": backends,
        "tokens": {
            "avg_prompt_tokens": _avg(prompt_tokens),
            "avg_completion_tokens": _avg(completion_tokens),
            "avg_total_tokens": _avg(total_tokens),
            "sum_prompt_tokens": _sum(prompt_tokens),
            "sum_completion_tokens": _sum(completion_tokens),
            "sum_total_tokens": _sum(total_tokens),
        },
        "latency": {
            "avg_latency_ms": _avg(latencies),
            "avg_route_latency_ms": _avg(route_latencies),
        },
    }
    return summary


def write_report(results_dir: Path, summary: dict[str, Any], batch_dirs: list[Path]) -> None:
    """Generate human-readable report.md."""
    lines: list[str] = [
        "# Baseline Batched Evaluation — Merged Results",
        "",
        f"**Total Cases**: {summary['total_cases']}",
        f"**Batches**: {len(batch_dirs)}",
        f"**Errors**: {summary['error_count']} (timeouts: {summary['timeout_count']}, 504s: {summary['http_504_count']})",
        "",
        "## Accuracy Metrics",
        "",
        "| Metric | Value | Count |",
        "|--------|-------|-------|",
    ]

    j = summary["judge"]
    lines.append(f"| Judge Accuracy | {j['accuracy'] or 'N/A'} | {j['correct_count']}/{j['judged_count']} |")

    b = summary["backend_routing"]
    lines.append(f"| Backend Accuracy | {b['accuracy'] or 'N/A'} | {b['correct_count']}/{b['labeled_count']} |")

    t = summary["template"]
    lines.append(f"| Template Hit Rate | {t['hit_rate'] or 'N/A'} | {t['hit_count']}/{t['labeled_count']} |")

    a = summary["any_backend_hit"]
    lines.append(f"| Any Backend Hit | {a['hit_rate'] or 'N/A'} | {a['hit_count']}/{a['labeled_count']} |")

    tok = summary["tokens"]
    lines.extend([
        "",
        "## Token Statistics",
        "",
        "| Metric | Avg | Sum |",
        "|--------|-----|-----|",
        f"| Prompt Tokens | {tok['avg_prompt_tokens'] or 'N/A'} | {tok['sum_prompt_tokens']} |",
        f"| Completion Tokens | {tok['avg_completion_tokens'] or 'N/A'} | {tok['sum_completion_tokens']} |",
        f"| Total Tokens | {tok['avg_total_tokens'] or 'N/A'} | {tok['sum_total_tokens']} |",
    ])

    lat = summary["latency"]
    lines.extend([
        "",
        "## Latency Statistics",
        "",
        f"- Average latency: {lat['avg_latency_ms'] or 'N/A'} ms",
        f"- Average route latency: {lat['avg_route_latency_ms'] or 'N/A'} ms",
        "",
        "## Route Methods",
        "",
        "| Method | Count |",
        "|--------|-------|",
    ])
    for method, count in sorted(summary["route_methods"].items(), key=lambda x: -x[1]):
        lines.append(f"| {method} | {count} |")

    lines.extend([
        "",
        "## Actual Backends",
        "",
        "| Backend | Count |",
        "|---------|-------|",
    ])
    for backend, count in sorted(summary["actual_backends"].items(), key=lambda x: -x[1]):
        lines.append(f"| {backend} | {count} |")

    lines.extend([
        "",
        "## Batches",
        "",
    ])
    for i, d in enumerate(batch_dirs, 1):
        lines.append(f"{i}. `{d.name}` — {d}")

    (results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge batch evaluation results")
    parser.add_argument("--master-dir", required=True, help="Master run directory")
    parser.add_argument("--batch-dirs", required=True, help="Comma-separated list of batch run directories")
    args = parser.parse_args()

    master_dir = Path(args.master_dir)
    results_dir = master_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    batch_dirs = [Path(d.strip()) for d in args.batch_dirs.split(",") if d.strip()]
    print(f"Merging {len(batch_dirs)} batches into {results_dir}")

    # Load all results
    all_results = load_route_results(batch_dirs)
    print(f"  Loaded {len(all_results)} route results")

    all_csv_rows = load_csv_rows(batch_dirs)
    print(f"  Loaded {len(all_csv_rows)} CSV rows")

    # Sort by case_id for consistent ordering
    def _sort_key(r: dict[str, Any]) -> str:
        return r.get("case_id", "")

    all_results.sort(key=_sort_key)
    all_csv_rows.sort(key=lambda r: r.get("case_id", ""))

    # Write merged route_results.jsonl
    rr_path = results_dir / "route_results.jsonl"
    with rr_path.open("w", encoding="utf-8") as f:
        for row in all_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Written: {rr_path}")

    # Write merged qa_results.csv
    if all_csv_rows:
        fieldnames = list(all_csv_rows[0].keys())
        csv_path = results_dir / "qa_results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"  Written: {csv_path}")

    # Build and write summary
    summary = build_summary(all_results)
    summary_path = results_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Written: {summary_path}")

    # Write report
    write_report(results_dir, summary, batch_dirs)
    print(f"  Written: {results_dir / 'report.md'}")

    # Print summary
    print("")
    print("=" * 50)
    print("MERGE COMPLETE")
    print("=" * 50)
    print(f"Total cases : {summary['total_cases']}")
    print(f"Errors      : {summary['error_count']} (timeouts: {summary['timeout_count']})")
    j = summary["judge"]
    print(f"Judge acc   : {j['accuracy'] or 'N/A'} ({j['correct_count']}/{j['judged_count']})")
    b = summary["backend_routing"]
    print(f"Backend acc : {b['accuracy'] or 'N/A'} ({b['correct_count']}/{b['labeled_count']})")
    t = summary["template"]
    print(f"Template hit: {t['hit_rate'] or 'N/A'} ({t['hit_count']}/{t['labeled_count']})")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
