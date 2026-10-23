"""Summarize generated evaluation results by knowledge-base snapshot."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows):
    """Use the latest result per case, keeping different snapshots separate."""
    groups = defaultdict(dict)
    for row in rows:
        groups[row["snapshot_id"]][row["id"]] = row
    summaries = []
    for snapshot, cases in groups.items():
        results = list(cases.values())
        elapsed = [r["elapsed_seconds"] for r in results]
        summaries.append(
            {
                "snapshot_id": snapshot,
                "total": len(results),
                "strict_checks_pass": sum(bool(r.get("engineering_checks_pass")) for r in results),
                "citation_integrity_pass": sum(
                    bool(r.get("checks", {}).get("citation_integrity")) for r in results
                ),
                "within_tool_budget": sum(
                    bool(r.get("checks", {}).get("within_tool_budget")) for r in results
                ),
                "within_round_budget": sum(
                    bool(r.get("checks", {}).get("within_round_budget")) for r in results
                ),
                "status_counts": dict(
                    Counter(r.get("result", {}).get("answer", {}).get("status", "error") for r in results)
                ),
                "median_seconds": round(statistics.median(elapsed), 2),
                "max_seconds": max(elapsed),
                "failed_cases": [r["id"] for r in results if not r.get("engineering_checks_pass")],
            }
        )
    return {
        "snapshots": summaries,
        "note": "统计仅覆盖结果文件中已有的题目；工程行为检查不代表法律语义准确率。",
    }


def main():
    parser = argparse.ArgumentParser(description="按知识库快照汇总评测结果")
    parser.add_argument("--input", type=Path, default=ROOT / "reports/evaluation.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/evaluation-summary.json")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("评测结果不存在，请先执行 evaluate.py 或通过 --input 指定结果文件")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
