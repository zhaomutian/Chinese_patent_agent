"""Real-model checks on explicitly synthetic interpretation cases, not textbook accuracy."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_exam import render_question, score

from patent_agent.agent import AgentService, validate_answer
from patent_agent.exam.methods import load_interpretation_package
from patent_agent.retrieval import KnowledgeBase
from patent_agent.settings import ROOT, Settings
from patent_agent.storage import dump


def main():
    parser = argparse.ArgumentParser(
        description="真实模型解释正反例回归，会产生API用量；样例为自拟，不属于教材评测集"
    )
    parser.add_argument("--case", choices=["synthetic-scope", "synthetic-overreach"])
    parser.add_argument("--output", type=Path, default=ROOT / "reports/interpretation-synthetic.jsonl")
    args = parser.parse_args()
    rows = json.loads((ROOT / "eval/interpretation_cases.json").read_text())
    if args.case:
        rows = [r for r in rows if r["question_id"] == args.case]
    s = Settings()
    fingerprint = hashlib.sha256()
    for p in sorted((ROOT / "src/patent_agent").rglob("*.py")):
        fingerprint.update(str(p.relative_to(ROOT)).encode())
        fingerprint.update(p.read_bytes())
    version = load_interpretation_package(s.teaching_methods_path / "interpretation")["version"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with AgentService(s) as svc:
        for row in rows:
            session = svc.create_session()
            result = svc.ask(
                session["session_id"], render_question(row), mode="analyze", explanation_level="concise"
            )
            checks = score(result, row)
            checks["expected_answer_match"] = checks.pop("textbook_answer_match")
            checks["citation_integrity"] = not validate_answer(
                result["answer"], result["citations"], session["snapshot_id"]
            )
            kb = KnowledgeBase(s, session["snapshot_id"])
            checks["rule_quotes_exact"] = all(
                b["source_quote"] in kb.chunk(b["evidence_id"])["text"]
                for j in (result.get("exam_analysis") or {}).get("judgments", [])
                for b in j.get("rule_basis", [])
            )
            record = {
                "question_id": row["question_id"],
                "source": "synthetic",
                "answer_standard": "authored_fixture",
                "code_sha256": fingerprint.hexdigest(),
                "interpretation_method_version": version,
                "checks": checks,
                "result": result,
            }
            with args.output.open("a") as f:
                f.write(dump(record) + "\n")
            print(
                dump({"question_id": row["question_id"], "checks": checks, "metrics": result["metrics"]}),
                flush=True,
            )


if __name__ == "__main__":
    main()
