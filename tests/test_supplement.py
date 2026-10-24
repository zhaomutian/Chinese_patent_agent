import json

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from test_agent import E, answer, call, msg, service

from patent_agent.agent import BatchExactArgs, ExactArgs
from patent_agent.retrieval import KnowledgeBase
from patent_agent.settings import Settings
from patent_agent.storage import active_snapshot, connect


def target(source, locator):
    return {"source_type": source, "locator": locator}


def test_q24_structured_gaps_are_all_queried_and_retained(tmp_path):
    """Real corpus, controlled model: all identified gaps must reach generation in round two."""
    settings = Settings()
    snapshot = active_snapshot(settings)["id"]
    kb = KnowledgeBase(settings, snapshot)
    initial_targets = [target("patent_law", "article:19"), target("implementing_rules", "article:8")]
    initial_targets += [
        target("examination_guidelines", "part:5/chapter:5/section:" + x) for x in ["6.1.1", "6.2.2"]
    ]
    initial = [e for t in initial_targets for e in kb.get_provision(**t)["evidence"]]
    gaps = [target("implementing_rules", "article:9")]
    gaps += [
        target("examination_guidelines", "part:5/chapter:5/section:" + x) for x in ["6.1.2", "6.2.1", "6.3"]
    ]
    needed = [e for t in gaps for e in kb.get_provision(**t)["evidence"]]
    assert len(initial + needed) > 6
    queried = []

    class CorpusKB:
        def search_knowledge(self, **kwargs):
            return {"status": "found", "evidence": initial}

        def get_provision(self, **kwargs):
            queried.append(kwargs)
            return kb.get_provision(**kwargs)

    draft = answer()
    draft["paragraphs"][0]["evidence_ids"] = [e["id"] for e in initial + needed]
    outputs = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge", "id": "initial", "args": {"query": "保密审查"}}],
        ),
        msg({"sufficient": False, "missing_provisions": gaps + [gaps[0]]}),
        msg({"sufficient": True}),
        msg(draft),
    ]
    with service(tmp_path, outputs) as svc:
        svc.kb_factory = lambda sid: CorpusKB()
        with connect(svc.s) as db:
            db.execute("UPDATE sessions SET snapshot_id=? WHERE id='test'", (snapshot,))
        result = svc.ask("test", "向外国申请专利前的保密审查涉及哪些规定？")
        assert queried == gaps  # Duplicate gap is queried only once; no extra model planner.
        assert result["metrics"]["retrieval_rounds"] == 2
        assert result["metrics"]["tools_executed"] == 2
        assert result["answer"]["status"] == "supported"
        assert {e["id"] for e in result["citations"]} == {e["id"] for e in initial + needed}
        assert all(t["status"] == "included" for t in result["retrieval_targets"])
        payload = json.loads(svc.llm.inputs[-1][-1].content)
        assert len(payload["evidence"]) <= svc.s.context_max_chunks
        assert sum(len(e["text"]) for e in payload["evidence"]) <= svc.s.context_max_chars
        messages = svc.graph.get_state({"configurable": {"thread_id": "test"}}).values["messages"]
        calls = [c["id"] for m in messages if m.type == "ai" for c in m.tool_calls]
        assert [m.tool_call_id for m in messages if m.type == "tool"] == calls


@pytest.mark.parametrize("mode", ["not_found", "error", "context_limited"])
def test_unresolved_target_cannot_be_reported_as_fully_supported(tmp_path, mode):
    missing = target("implementing_rules", "article:9")

    class KB:
        def get_provision(self, **kwargs):
            if kwargs["source_type"] == "patent_law":
                return {"status": "found", "evidence": [E]}
            if mode == "error":
                raise RuntimeError("SECRET_VALUE")
            if mode == "not_found":
                return {"status": "not_found", "evidence": []}
            # Cannot fit the whole chunk; must not silently mark the gap as resolved.
            return {"status": "found", "evidence": [E | {"id": "large", "text": "字" * 10001}]}

    outputs = [
        call(),
        msg({"sufficient": False, "missing_provisions": [missing]}),
        msg({"sufficient": True}),
        msg(answer()),
    ]
    with service(tmp_path, outputs) as svc:
        svc.kb_factory = lambda sid: KB()
        result = svc.ask("test", "查询")
        assert result["answer"]["status"] == "limited"
        assert result["retrieval_targets"][-1]["status"] == mode
        assert "SECRET_VALUE" not in json.dumps(result)
        assert any(p["kind"] == "limitation" for p in result["answer"]["paragraphs"])
        # A follow-up starts a new target checklist, including after checkpoint restoration.
        svc.llm.outputs = [
            AIMessage(content=""),
            msg({"sufficient": False, "clarify": True}),
            msg(
                {
                    "status": "clarification",
                    "paragraphs": [{"text": "请明确问题", "kind": "clarification", "evidence_ids": []}],
                }
            ),
        ]
        assert svc.ask("test", "下一轮")["retrieval_targets"] == []


def test_batch_deadline_checked_between_targets(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("patent_agent.agent.time.monotonic", lambda: clock[0])
    queried = []

    class KB:
        def get_provision(self, **kwargs):
            queried.append(kwargs)
            clock[0] = 102.0
            return {"status": "found", "evidence": [E]}

    with service(tmp_path, []) as svc:
        targets = [ExactArgs(**target("patent_law", "article:" + str(n))) for n in [1, 2]]
        result = svc.lookup_targets(KB(), targets, deadline=101.0)
        assert len(queried) == 1
        assert result["targets"][1]["lookup_status"] == "limit"
        assert result["status"] == "partial"


def test_batch_size_and_per_round_budget_are_enforced(tmp_path):
    item = target("patent_law", "article:1")
    with pytest.raises(ValidationError):
        BatchExactArgs(targets=[item] * 9)
    with service(tmp_path, []) as svc:
        svc.s.max_tools_per_round = 1
        st = svc.prepare({"question": "问题"}) | {"snapshot_id": "snap"}
        st["pending"] = [
            {"name": "get_provisions", "args": {"targets": [item]}, "id": "one"},
            {"name": "get_provisions", "args": {"targets": [target("patent_law", "article:2")]}, "id": "two"},
        ]
        result = svc.execute(st)
        assert result["tools_executed"] == 1
        assert result["retrieval_targets"][-1]["status"] == "limit"


def test_structured_semantic_gap_is_executed_alongside_exact_gap(tmp_path):
    with service(tmp_path, []) as svc:
        st = svc.prepare({"question": "问题"}) | {"rounds": 1, "snapshot_id": "snap"}
        st["assessment"] = {
            "sufficient": False,
            "missing_provisions": [target("patent_law", "article:1")],
            "missing_search": {"query": "未知章节的概念", "source_types": ["examination_guidelines"]},
        }
        planned = svc.plan(st)
        assert [c["name"] for c in planned["pending"]] == ["get_provisions", "search_knowledge"]
        assert svc.llm.inputs == []
        result = svc.execute(st | planned)
        assert result["tools_executed"] == 2


def test_coverage_gate_rejects_previous_q24_gap_and_locator_prefix_collision():
    import runpy
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    check = runpy.run_path(str(root / "scripts/evaluate.py"))["missing_required_locators"]
    case = next(c for c in json.loads((root / "eval/cases.json").read_text()) if c["id"] == "q24")
    # The earlier answer cited these locations but omitted three necessary guideline sections.
    old_citations = [
        {"source_type": "patent_law", "locator": "article:19"},
        {"source_type": "implementing_rules", "locator": "article:8"},
        {"source_type": "implementing_rules", "locator": "article:9"},
    ] + [
        {"source_type": "examination_guidelines", "locator": "part:5/chapter:5/section:" + x}
        for x in ["6", "6.1.1", "6.2.2"]
    ]
    assert {t["locator"] for t in check(case, old_citations)} == {
        "part:5/chapter:5/section:6.1.2",
        "part:5/chapter:5/section:6.2.1",
        "part:5/chapter:5/section:6.3",
    }
    citations = [{"source_type": t["source"], "locator": t["locator"]} for t in case["required_locators"]]
    assert not check(case, citations)
    citations[-1]["locator"] = "part:5/chapter:5/section:6.30"
    assert check(case, citations)
    citations[-1]["locator"] = "part:5/chapter:5/section:6.3.1"
    assert not check(case, citations)


def test_outline_supplies_sibling_locators_without_promoting_headings_to_evidence():
    settings = Settings()
    kb = KnowledgeBase(settings, active_snapshot(settings)["id"])
    evidence = kb.get_provision("examination_guidelines", "part:5/chapter:5/section:6.1.2")["evidence"]
    outline = kb.section_outline(evidence)
    assert any(row["locator"] == "part:5/chapter:5/section:6.1.1" for row in outline)
    assert all(set(row) == {"source_type", "locator", "heading"} for row in outline)
    assert all(row["locator"].startswith("part:5/chapter:5/section:6") for row in outline)
    assert len(kb.section_outline(evidence, limit=2)) == 2
    assert kb.section_outline([E]) == []
    fake_snapshot = object.__new__(KnowledgeBase)
    fake_snapshot.s, fake_snapshot.snapshot_id = settings, "nonexistent"
    assert fake_snapshot.section_outline(evidence) == []
