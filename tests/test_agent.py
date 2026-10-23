import json

from langchain_core.messages import AIMessage

from patent_agent.agent import AgentService, validate_answer
from patent_agent.settings import Settings
from patent_agent.storage import connect

E = {
    "id": "c_one",
    "snapshot_id": "snap",
    "provision_id": "p_one",
    "document_id": "doc",
    "source_type": "patent_law",
    "title": "样本",
    "version_label": "测试版本",
    "locator": "article:1",
    "heading": "第一条",
    "page_start": 1,
    "page_end": 1,
    "text": "技术测试原文。",
    "coverage_warnings": [],
}


def msg(obj):
    return AIMessage(content=json.dumps(obj, ensure_ascii=False))


def answer(eid="c_one"):
    return {
        "status": "supported",
        "paragraphs": [{"text": "技术测试原文。", "kind": "fact", "evidence_ids": [eid]}],
        "quotes": [],
    }


def call(i="call_a"):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "get_provision", "args": {"source_type": "patent_law", "locator": "article:1"}, "id": i}
        ],
    )


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = outputs
        self.inputs = []

    def bind(self, **kwargs):
        return self

    def bind_tools(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeKB:
    def get_provision(self, **kwargs):
        return {"status": "found", "evidence": [E]}

    def search_knowledge(self, **kwargs):
        return {"status": "found", "evidence": [E]}


def service(tmp_path, outputs):
    s = Settings().model_copy(
        update={"catalog_db_path": tmp_path / "catalog.db", "checkpoint_db_path": tmp_path / "checkpoint.db"}
    )
    with connect(s) as db:
        db.execute("INSERT INTO sessions VALUES (?,?,?)", ("test", "snap", "now"))
    return AgentService(s, llm=FakeLLM(outputs), kb_factory=lambda sid: FakeKB())


def test_citations_reject_foreign_snapshot_unknown_id_and_false_quote():
    assert validate_answer(answer(), [E], "snap") == []
    assert validate_answer(answer("foreign"), [E], "snap")
    assert validate_answer(answer(), [E], "different")
    a = answer()
    a["quotes"] = [{"text": "并不存在的原话", "evidence_id": "c_one"}]
    assert validate_answer(a, [E], "snap")


def test_repair_budget_and_checkpoint_turn_reset(tmp_path):
    enough = msg({"sufficient": True})
    outputs = [
        call(),
        enough,
        msg(answer("bad")),
        msg(answer()),
        AIMessage(content="需要更多说明"),
        msg({"sufficient": False, "clarify": True}),
        msg(
            {
                "status": "clarification",
                "paragraphs": [{"text": "请明确问题。", "kind": "clarification", "evidence_ids": []}],
                "quotes": [],
            }
        ),
    ]
    with service(tmp_path, outputs) as svc:
        first = svc.ask("test", "技术问题")
        assert first["metrics"]["repairs"] == 1 and len(first["citations"]) == 1
        assert any(m.type == "tool" for m in svc.llm.inputs[2])
        assert any(m.type == "tool" for m in svc.llm.inputs[3])
        second = svc.ask("test", "下一轮")
        assert second["metrics"]["repairs"] == 0
        assert second["metrics"]["tools_executed"] == 0
        assert second["citations"] == []
        state = svc.graph.get_state({"configurable": {"thread_id": "test"}}).values
        assert state["evidence"] == [] and len(state["history"]) == 2
        assert len(svc.history("test")["history"]) == 2
    with service_reopen(tmp_path) as reopened:
        assert len(reopened.history("test")["history"]) == 2


def service_reopen(tmp_path):
    s = Settings().model_copy(
        update={"catalog_db_path": tmp_path / "catalog.db", "checkpoint_db_path": tmp_path / "checkpoint.db"}
    )
    return AgentService(s, llm=FakeLLM([]), kb_factory=lambda sid: FakeKB())


def test_two_round_limit_and_two_tool_limit(tmp_path):
    calls = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_provision",
                "args": {"source_type": "patent_law", "locator": "article:1"},
                "id": f"call_{i}",
            }
            for i in range(3)
        ],
    )
    outputs = [
        calls,
        msg({"sufficient": False, "missing_query": "补充证据"}),
        call("supplement"),
        msg({"sufficient": False}),
        msg(answer()),
    ]
    with service(tmp_path, outputs) as svc:
        result = svc.ask("test", "不充分的问题")
        assert result["metrics"]["retrieval_rounds"] == 2
        assert result["metrics"]["tools_executed"] == 3
        state = svc.graph.get_state({"configurable": {"thread_id": "test"}}).values
        tool_messages = [m for m in state["messages"] if m.type == "tool"]
        assert len(tool_messages) == 4  # Rejected excess call also receives its own output ID.
        assert "limit" in tool_messages[2].content


def test_invalid_repair_falls_back_without_fake_citations(tmp_path):
    with service(
        tmp_path, [call(), msg({"sufficient": True}), msg(answer("fake")), msg(answer("still_fake"))]
    ) as svc:
        result = svc.ask("test", "问题")
        assert result["answer"]["status"] == "limited"
        assert result["citations"] == [] and result["metrics"]["repairs"] == 1


def test_model_failure_is_sanitized(tmp_path):
    outputs = [RuntimeError("SECRET_TOKEN"), RuntimeError("SECRET_TOKEN"), RuntimeError("SECRET_TOKEN")]
    with service(tmp_path, outputs) as svc:
        result = svc.ask("test", "问题")
        assert "SECRET_TOKEN" not in json.dumps(result)
        assert result["answer"]["status"] == "limited"


def test_expired_deadline_does_not_call_model(tmp_path):
    import time

    import pytest

    with service(tmp_path, []) as svc:
        with pytest.raises(TimeoutError):
            svc.invoke({"deadline": time.monotonic() - 1}, [])
        assert svc.llm.inputs == []
