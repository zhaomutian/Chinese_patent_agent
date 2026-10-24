import json
from pathlib import Path

import pytest
from test_agent import E, call, msg, service, service_reopen

from patent_agent.exam.methods import load_package, package_digest
from patent_agent.exam.models import Analysis, Problem
from patent_agent.exam.validation import selected_options, validate_analysis, validate_problem
from patent_agent.exam.workflow import wants_exam

STEM = "示例事实。下列说法正确的是？（单选）"
QUESTION = STEM + "\nA. 技术测试原文。\nB. 未满足条件。"


def problem(action="new", **kwargs):
    return dict(
        action=action,
        stem=STEM,
        options={"A": "技术测试原文。", "B": "未满足条件。"},
        question_type="single_choice",
        polarity="select_correct",
        facts=[{"id": "f1", "text": "示例事实。", "quote": "示例事实。", "origin": "explicit"}],
        issues=[
            {
                "id": "i1",
                "question": "条件是否成立？",
                "option_labels": ["A", "B"],
                "fact_ids": ["f1"],
                "query": "条件规定",
            }
        ],
        **kwargs,
    )


def analysis():
    return {
        "judgments": [
            {
                "label": label,
                "verdict": verdict,
                "rule_basis": [{"evidence_id": E["id"], "source_quote": E["text"], "rule": E["text"]}],
                "reason": "按原文及题设判断。",
                "evidence_ids": [E["id"]],
                "issue_ids": ["i1"],
                "fact_ids": ["f1"],
            }
            for label, verdict in [("A", "correct"), ("B", "incorrect")]
        ],
        "issue_coverage": [
            {"issue_id": "i1", "status": "covered", "evidence_ids": [E["id"]], "note": "已核对"}
        ],
        "explanation": {"text": "关注原文规定与题设条件的对应关系。", "evidence_ids": [E["id"]]},
        "limitations": [],
    }


def ready():
    return {
        "sufficient": True,
        "issues": [{"issue_id": "i1", "status": "direct_supported", "evidence_ids": [E["id"]]}],
    }


def outputs(p=None):
    return [msg(p or problem()), call(), msg(ready()), msg(analysis())]


def test_exam_graph_complete_answer_and_followup_restore(tmp_path):
    with service(tmp_path, outputs()) as svc:
        svc.s.teaching_methods_path = tmp_path / "methods"
        first = svc.ask("test", QUESTION, mode="analyze")
        assert first["exam_analysis"]["selected_options"] == ["A"]
        assert first["answer"]["status"] == "supported"
        assert first["metrics"]["model_calls"] == 4
        assert first["problem_revision"] == 1
    with service_reopen(tmp_path) as svc:
        svc.s.teaching_methods_path = tmp_path / "methods"
        p = problem(action="continue", focus_option="B", student_question="为什么B错？")
        svc.llm.outputs = outputs(p)
        second = svc.ask("test", "为什么B错？")
        assert second["problem_id"] == first["problem_id"]
        assert second["problem_revision"] == first["problem_revision"]
        assert len(svc.history("test")["history"]) == 2
        intake_payload = json.loads(svc.llm.inputs[0][-1].content)
        assert intake_payload["current_problem"]["problem_id"] == first["problem_id"]


def test_hint_never_generates_or_returns_solution_and_requires_reveal(tmp_path):
    with service(tmp_path, [msg(problem()), msg(problem(action="continue"))]) as svc:
        svc.s.teaching_methods_path = tmp_path / "methods"
        first = svc.ask("test", QUESTION, mode="analyze", explanation_level="hint")
        second = svc.ask("test", "继续提示")
        for result in [first, second, svc.history("test")["last_result"]]:
            assert result["exam_analysis"] == {"mode": "hint"}
            assert result["citations"] == [] and result["retrieval_targets"] == []
            assert result["trace"] == [{"node": "exam_hint"}]
            assert result["metrics"]["tools_executed"] == 0
            assert "selected_options" not in json.dumps(result)
        svc.llm.outputs = outputs(problem(action="continue"))
        reveal = svc.ask("test", "现在告诉我答案")
        assert reveal["explanation_level"] == "detailed"
        assert reveal["exam_analysis"]["selected_options"] == ["A"]


def test_revised_problem_invalidates_analysis_and_new_problem_has_new_id(tmp_path):
    with service(tmp_path, outputs()) as svc:
        first = svc.ask("test", QUESTION, mode="analyze")
        revised = problem(action="revise")
        revised["stem"] = "修改后的事实。下列说法正确的是？（单选）"
        revised["facts"] = [
            {"id": "f1", "text": "修改后的事实。", "quote": "修改后的事实。", "origin": "explicit"}
        ]
        svc.llm.outputs = outputs(revised)
        second = svc.ask("test", "把示例事实改为：修改后的事实。")
        assert second["problem_id"] == first["problem_id"]
        assert second["problem_revision"] == 2
        assert second["metrics"]["tools_executed"] == 1
        payload = json.loads(svc.llm.inputs[-1][-1].content)
        assert payload["problem"]["facts"][0]["quote"] == "修改后的事实。"
        svc.llm.outputs = outputs()
        third = svc.ask("test", QUESTION, mode="analyze")
        assert third["problem_id"] != second["problem_id"]
        assert third["problem_revision"] == 1


def test_select_incorrect_and_multiselect_are_derived_from_judgments():
    p = Problem.model_validate(problem())
    a = Analysis.model_validate(analysis())
    p.polarity = "select_incorrect"
    assert selected_options(p, a) == ["B"]
    p.polarity, p.question_type = "select_correct", "multiple_choice"
    a.judgments[1].verdict = "correct"
    assert selected_options(p, a) == ["A", "B"]
    p.question_type = "single_choice"
    assert selected_options(p, a) == []
    a.judgments[1].verdict = "undetermined"
    assert selected_options(p, a) == []


def test_problem_validator_catches_invented_text_negative_polarity_and_missing_options():
    p = Problem.model_validate(problem())
    assert not validate_problem(p, QUESTION, None)
    p.facts[0].quote = "题干没有的日期"
    assert validate_problem(p, QUESTION, None)
    p = Problem.model_validate(problem())
    p.stem = "下列说法不正确的是？"
    assert any("否定" in error for error in validate_problem(p, p.stem + QUESTION, None))
    p = Problem.model_validate(problem())
    p.options["A"] = "偷偷替换的选项"
    assert validate_problem(p, QUESTION, None)


def test_analysis_rejects_missing_option_foreign_evidence_and_false_coverage():
    p, a = Problem.model_validate(problem()), Analysis.model_validate(analysis())
    assert not validate_analysis(a, p, [E], "snap")
    assert validate_analysis(a, p, [E], "foreign")
    a.judgments.pop()
    assert validate_analysis(a, p, [E], "snap")
    a = Analysis.model_validate(analysis())
    a.issue_coverage[0].evidence_ids = []
    assert validate_analysis(a, p, [E], "snap")


def test_method_package_integrity_and_version_binding(tmp_path):
    payload = {"cards": [], "chapters": []}
    version = package_digest(payload)
    (tmp_path / (version + ".json")).write_text(json.dumps({"payload": payload}))
    (tmp_path / "active.json").write_text(json.dumps({"version": version}))
    assert load_package(tmp_path)["version"] == version
    (tmp_path / (version + ".json")).write_text(
        json.dumps({"payload": {"cards": [], "chapters": ["tampered"]}})
    )
    with pytest.raises(ValueError):
        load_package(tmp_path, version)
    with pytest.raises(ValueError):
        load_package(tmp_path, "../../private")


def test_model_budget_and_failed_repair_fall_back_without_answer(tmp_path):
    with service(tmp_path, [msg(problem())]) as svc:
        svc.s.exam_max_model_calls = 1
        result = svc.ask("test", QUESTION, mode="analyze")
        assert result["answer"]["status"] == "limited"
        assert result["exam_analysis"] is None
        assert result["metrics"]["model_calls"] == 1


def test_import_rejects_non_development_before_sending_or_writing(tmp_path):
    import runpy

    functions = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/import_teaching.py"))
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"split": "test"}))
    with pytest.raises(ValueError, match="development"):
        functions["load_development"](path)


def test_auto_route_supports_options_and_explicit_lookup_override():
    assert wants_exam(QUESTION, None, "auto")
    assert wants_exam("为什么B错", {"id": "p"}, "auto")
    assert not wants_exam(QUESTION, {}, "lookup")
    assert not wants_exam("专利法第二十二条规定什么", None, "auto")


def test_textbook_answer_is_ground_truth_and_never_in_solver_input():
    import runpy

    functions = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evaluate_exam.py"))
    row = {
        "stem": "示例题干",
        "options": {"A": "选项甲", "B": "选项乙"},
        "reference_answer": "B",
        "explanation": "SECRET_REFERENCE_EXPLANATION",
        "question_type": "single_choice",
    }
    prompt = functions["render_question"](row)
    assert "SECRET_REFERENCE_EXPLANATION" not in prompt
    assert "reference_answer" not in prompt
    assert functions["score"]({"exam_analysis": {"selected_options": ["B"]}}, row)["textbook_answer_match"]
    assert not functions["score"]({"exam_analysis": {"selected_options": ["A"]}}, row)[
        "textbook_answer_match"
    ]


def test_api_accepts_exam_mode_and_rejects_unknown_mode():
    from patent_agent.api import ChatRequest

    request = ChatRequest(session_id="a" * 32, question=QUESTION, mode="analyze", explanation_level="hint")
    assert request.mode == "analyze"
    with pytest.raises(ValueError):
        ChatRequest(session_id="a" * 32, question=QUESTION, mode="invented")


def test_intake_retries_invalid_json_and_catches_dropped_options(tmp_path):
    with service(tmp_path, [msg({"wrong": True}), *outputs()]) as svc:
        result = svc.ask("test", QUESTION, mode="analyze")
        assert result["exam_analysis"]["complete"]
        assert result["metrics"]["model_calls"] == 5
    p = Problem.model_validate(problem())
    del p.options["B"]
    p.issues[0].option_labels = ["A"]
    assert any("完整保留" in e for e in validate_problem(p, QUESTION, None))


def test_exam_supplement_can_resolve_new_gap_without_expanding_lookup_budget(tmp_path):
    missing = {
        "sufficient": False,
        "issues": [{"issue_id": "i1", "status": "missing_rule", "evidence_ids": []}],
        "missing_provisions": [{"source_type": "patent_law", "locator": "article:1"}],
    }
    sequence = [
        msg(problem()),
        call(),
        msg(missing),
        msg(missing),
        msg(ready()),
        msg(analysis()),
    ]
    with service(tmp_path, sequence) as svc:
        result = svc.ask("test", QUESTION, mode="analyze")
        assert result["exam_analysis"]["complete"]
        assert result["metrics"]["retrieval_rounds"] == 3
        assert result["metrics"]["model_calls"] == 6
        assert svc.retrieval_round_limit({"active_mode": "lookup"}) == 2


def test_session_keeps_method_version_when_active_package_changes(tmp_path):
    directory = tmp_path / "methods"
    directory.mkdir()

    def activate(marker):
        payload = {"cards": [], "chapters": [], "marker": marker}
        version = package_digest(payload)
        (directory / (version + ".json")).write_text(json.dumps({"payload": payload}))
        (directory / "active.json").write_text(json.dumps({"version": version}))
        return version

    version = activate("first")
    with service(tmp_path, outputs()) as svc:
        svc.s.teaching_methods_path = directory
        first = svc.ask("test", QUESTION, mode="analyze")
        activate("second")
        svc.llm.outputs = outputs(problem(action="continue"))
        second = svc.ask("test", "为什么B错？")
        assert first["method_version"] == second["method_version"] == version


def test_empty_analysis_is_not_counted_as_covered():
    import runpy

    functions = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evaluate_exam.py"))
    assert not functions["score"]({"exam_analysis": None}, {"reference_answer": "A"})["issues_covered"]


def test_chat_endpoint_and_history_keep_hint_result_sanitized(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from patent_agent.api import app

    sid = "a" * 32
    with service(tmp_path, [msg(problem())]) as svc:
        from patent_agent.storage import connect

        with connect(svc.s) as db:
            db.execute("UPDATE sessions SET id=? WHERE id='test'", (sid,))
        monkeypatch.setattr(app.state, "service", svc, raising=False)
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={"session_id": sid, "question": QUESTION, "mode": "analyze", "explanation_level": "hint"},
        )
        assert response.status_code == 200
        assert response.json()["exam_analysis"] == {"mode": "hint"}
        restored = client.get("/api/v1/sessions/" + sid).json()["last_result"]
        assert restored["exam_analysis"] == {"mode": "hint"}
        assert not restored["citations"]
        assert restored["trace"] == [{"node": "exam_hint"}]


def test_intake_rejects_swapped_labels_and_truncated_option_text():
    p = Problem.model_validate(problem())
    p.options["A"], p.options["B"] = p.options["B"], p.options["A"]
    assert any("标签与原文" in e for e in validate_problem(p, QUESTION, None))
    p = Problem.model_validate(problem())
    p.options["A"] = "技术测试"
    assert any("截断" in e for e in validate_problem(p, QUESTION, None))


def test_auto_followup_respects_switch_back_to_lookup(tmp_path):
    from test_agent import answer

    with service(tmp_path, outputs()) as svc:
        svc.ask("test", QUESTION, mode="analyze")
        for mode in ["lookup", "auto"]:
            svc.llm.outputs = [call(), msg(ready()), msg(answer())]
            result = svc.ask("test", "这个条文是什么意思？", mode=mode)
            assert not any(t["node"] == "exam_intake" for t in result["trace"])
            assert "exam_analysis" not in result


def test_focused_followup_keeps_full_problem_in_evidence_assessment(tmp_path):
    with service(tmp_path, outputs()) as svc:
        svc.ask("test", QUESTION, mode="analyze")
        svc.llm.outputs = outputs(problem(action="continue", focus_option="B"))
        result = svc.ask("test", "为什么B不正确？")
        assert result["exam_analysis"]["complete"]
        assess_input = svc.llm.inputs[-2]
        payload = json.loads(assess_input[-1].content)
        assert set(payload["problem"]["options"]) == {"A", "B"}
        assert "全部答案争点" in assess_input[0].content
        assert "不缩小证据检查范围" in svc.llm.inputs[-3][-1].content


def test_semantic_supplement_preserves_existing_exact_evidence(tmp_path):
    import time

    from test_agent import FakeKB

    class MixedKB(FakeKB):
        def search_knowledge(self, **kwargs):
            return {"status": "found", "evidence": [E | {"id": "semantic", "text": "冗余语义片段" * 15}]}

    with service(tmp_path, []) as svc:
        svc.kb_factory = lambda sid: MixedKB()
        svc.s.context_max_chars = 95
        st = {
            "evidence": [E],
            "tools_executed": 1,
            "events": [],
            "snapshot_id": "snap",
            "started": time.monotonic(),
            "deadline": time.monotonic() + 30,
            "messages": [],
            "retrieval_targets": [
                {
                    "source_type": "patent_law",
                    "locator": "article:1",
                    "lookup_status": "found",
                    "evidence_ids": [E["id"]],
                }
            ],
            "pending": [{"name": "search_knowledge", "id": "search", "args": {"query": "相关规定"}}],
        }
        result = svc.execute(st)
        assert result["evidence"] == [E]
        assert result["retrieval_targets"][0]["status"] == "included"


def test_analysis_uses_only_final_evidence_payload(tmp_path):
    with service(tmp_path, outputs()) as svc:
        svc.ask("test", QUESTION, mode="analyze")
        messages = svc.llm.inputs[-1]
        assert len(messages) == 2
        assert not any(m.type == "tool" for m in messages)
        assert json.loads(messages[-1].content)["evidence"] == [E]
