"""Real API evaluation; opt in by executing this script. Results are resumable per snapshot."""

import argparse
import json
import time
from pathlib import Path

from patent_agent.agent import AgentService, validate_answer
from patent_agent.models import safe_error
from patent_agent.retrieval import KnowledgeBase
from patent_agent.settings import ROOT, Settings
from patent_agent.storage import active_snapshot


def missing_required_locators(case, citations):
    """Require each listed provision (or a specific descendant) in the final citations."""
    return [
        target
        for target in case.get("required_locators", [])
        if not any(
            cite["source_type"] == target["source"]
            and (
                cite["locator"] == target["locator"]
                or (
                    target["source"] == "examination_guidelines"
                    and cite["locator"].startswith(target["locator"] + ".")
                )
            )
            for cite in citations
        )
    ]


def main():
    parser = argparse.ArgumentParser(description="运行真实 API 评测，产生模型用量")
    parser.add_argument("--case", action="append", help="仅运行指定 case，可重复指定")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/evaluation.jsonl")
    parser.add_argument("--force", action="store_true", help="忽略已有完成记录并追加新结果")
    args = parser.parse_args()
    s = Settings()
    snapshot = active_snapshot(s)["id"]
    kb = KnowledgeBase(s, snapshot)
    cases = json.loads((ROOT / "eval/cases.json").read_text())
    if args.case:
        if not set(args.case) <= {case["id"] for case in cases}:
            parser.error("未知 case ID")
        selected = set(args.case)
        # Include earlier turns required by a selected follow-up.
        for i, case in enumerate(cases):
            if case["id"] in selected and case.get("thread"):
                selected.update(c["id"] for c in cases[:i] if c.get("thread") == case["thread"])
        cases = [c for c in cases if c["id"] in selected]
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    previous = (
        [json.loads(line) for line in out.read_text().splitlines()] if out.exists() and not args.force else []
    )
    completed = {r["id"] for r in previous if r["snapshot_id"] == snapshot}
    threads = {
        r["thread"]: r["result"]["session_id"]
        for r in previous
        if r["snapshot_id"] == snapshot and r.get("thread") and r.get("result")
    }
    with AgentService(s) as svc:
        for case in cases:
            if case["id"] in completed:
                continue
            sid = threads.get(case.get("thread")) or svc.create_session()["session_id"]
            if case.get("thread"):
                threads[case["thread"]] = sid
            start = time.monotonic()
            row = {
                "id": case["id"],
                "kind": case["kind"],
                "question": case["question"],
                "thread": case.get("thread"),
                "snapshot_id": snapshot,
            }
            try:
                if case.get("locator"):
                    expected = kb.get_provision(case["source"], case["locator"])
                    assert expected["evidence"], "evaluation locator is absent from corpus"
                result = svc.ask(sid, case["question"])
                cites = result["citations"]
                checks = {
                    "citation_integrity": not validate_answer(
                        result["answer"], [kb.chunk(c["id"]) for c in cites], snapshot
                    ),
                    "within_tool_budget": result["metrics"]["tools_executed"] <= 4,
                    "within_round_budget": result["metrics"]["retrieval_rounds"] <= 2,
                }
                if case.get("required_locators"):
                    checks["required_locators_cited"] = not missing_required_locators(case, cites)
                    checks["supplement_targets_complete"] = all(
                        t["status"] == "included" for t in result.get("retrieval_targets", [])
                    )
                if case.get("locator"):
                    checks["expected_locator_cited"] = any(
                        c["source_type"] == case["source"]
                        and (
                            c["locator"] == case["locator"]
                            or (
                                case["source"] == "examination_guidelines"
                                and c["locator"].startswith(case["locator"] + ".")
                            )
                        )
                        for c in cites
                    )
                if case.get("expected"):
                    checks["expected_boundary_status"] = result["answer"]["status"] == case["expected"]
                row |= {
                    "missing_required_locators": missing_required_locators(case, cites),
                    "result": result,
                    "checks": checks,
                    "engineering_checks_pass": all(checks.values()),
                    "semantic_review": "pending",
                }
            except Exception as exc:
                row |= {"error": safe_error(exc), "engineering_checks_pass": False}
            row["elapsed_seconds"] = round(time.monotonic() - start, 2)
            with out.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "id": case["id"],
                        "checks_pass": row["engineering_checks_pass"],
                        "seconds": row["elapsed_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
