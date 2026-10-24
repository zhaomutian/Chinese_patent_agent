"""Evaluate questions against textbook answers; references never enter the solver prompt."""

import argparse
import hashlib
import json
from pathlib import Path

from patent_agent.agent import AgentService, validate_answer
from patent_agent.exam.methods import load_package
from patent_agent.models import safe_error
from patent_agent.retrieval import KnowledgeBase
from patent_agent.settings import ROOT, Settings
from patent_agent.storage import dump


def render_question(row):
    text = row["stem"]
    # Only use metadata supplied with the question itself, never reference-file annotations.
    kind = {"single_choice": "单选", "multiple_choice": "多选"}.get(row.get("question_type"))
    if kind and kind not in text:
        text += f"（{kind}）"
    return text + "\n" + "\n".join(f"{label}. {value}" for label, value in row["options"].items())


def score(result, reference):
    expected = sorted({c for c in reference["reference_answer"] if c in "ABCDEFGH"})
    analysis = result.get("exam_analysis") or {}
    selected = sorted(analysis.get("selected_options", []))
    return {
        "textbook_answer_match": bool(expected) and selected == expected,
        "complete_analysis": bool(analysis.get("complete")),
        "answer_complete": bool(analysis.get("answer_complete", analysis.get("complete"))),
        "explanation_complete": bool(analysis.get("explanation_complete", analysis.get("complete"))),
        "interpretation_review_status": analysis.get("review_status", "legacy"),
        "issues_covered": bool(analysis.get("issue_coverage"))
        and all(c["status"] == "covered" for c in analysis["issue_coverage"]),
        "selected_options": selected,
        "expected_options": expected,
    }


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="按教材标准答案评分；真实模型运行产生API用量")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--dataset", choices=["development", "validation", "test"], default="development")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/exam-evaluation.jsonl")
    args = parser.parse_args()
    rows = read_rows(args.questions)
    if any(row.get("split", args.dataset) != args.dataset for row in rows):
        parser.error("题目 split 与声明的数据集不一致")
    if len({r["question_id"] for r in rows}) != len(rows):
        parser.error("题目ID重复")
    references = read_rows(args.references) if args.references else rows
    by_id = {r["question_id"]: r for r in references}
    if args.ids:
        if not set(args.ids) <= {r["question_id"] for r in rows}:
            parser.error("指定题号不存在")
        rows = [r for r in rows if r["question_id"] in args.ids]
    if args.limit <= 0:
        parser.error("limit 必须大于0")
    rows = rows[: args.limit]
    if any(not by_id.get(r["question_id"], {}).get("reference_answer") for r in rows):
        parser.error("缺少教材标准答案；请通过 --references 指定评分文件")
    settings = Settings()
    version = load_package(settings.teaching_methods_path)["version"]
    run = hashlib.sha256(dump({"questions": rows, "method_version": version}).encode()).hexdigest()[:16]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with AgentService(settings) as svc:
        session = svc.create_session()
        kb = KnowledgeBase(settings, session["snapshot_id"])
        for row in rows:
            # New session per independent question: no previous answers or reference text in context.
            sid = svc.create_session()["session_id"]
            record = {
                "question_id": row["question_id"],
                "dataset": args.dataset,
                "run": run,
                "method_version": version,
                "answer_standard": "textbook",
            }
            try:
                result = svc.ask(sid, render_question(row), mode="analyze", explanation_level="concise")
                checks = score(result, by_id[row["question_id"]])
                checks["citation_integrity"] = not validate_answer(
                    result["answer"], [kb.chunk(c["id"]) for c in result["citations"]], session["snapshot_id"]
                )
                record |= {
                    "checks": checks,
                    "result": result,
                    "interpretation_method_version": result.get("interpretation_method_version"),
                }
            except Exception as exc:
                record["error"] = safe_error(exc)
            with args.output.open("a") as f:
                f.write(dump(record) + "\n")
            print(
                dump(
                    {
                        "question_id": row["question_id"],
                        "checks": record.get("checks"),
                        "error": record.get("error"),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
