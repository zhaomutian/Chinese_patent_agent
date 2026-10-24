import copy
import json
import time

import pytest
from test_agent import E, call, msg, service
from test_exam import QUESTION, analysis, problem, ready

from patent_agent.exam.interpretation import InterpretationFlow
from patent_agent.exam.methods import (
    load_interpretation_package,
    package_digest,
    select_interpretation_method,
)
from patent_agent.exam.models import Analysis, ApplicationReview, ExamAssessment, Problem
from patent_agent.exam.validation import validate_analysis, validate_normative_assessment, validate_review


def interpreted_problem():
    p = problem()
    p["issues"][0]["analysis_type"] = "subsumption"
    return p


def interpreted_analysis():
    a = analysis()
    for j in a["judgments"]:
        j.update(
            support_type="subsumption",
            interpretation="规范涵盖该类行为。",
            application="明确题设行为属于该范围。",
            boundary_check="未据此推导未规定的法律责任。",
        )
    return a


def norm_ready():
    a = ready()
    a["issues"][0]["status"] = "interpretation_ready"
    return a


def review(status="approved", **kwargs):
    return {
        "issues": [
            {
                "issue_id": "i1",
                "status": status,
                "evidence_ids": [E["id"]],
                "reason": "已核对规范范围和适用边界。",
            }
        ],
        "explanation_supported": status == "approved",
        "explanation_note": "" if status == "approved" else "适用依据尚未核实。",
        **kwargs,
    }


def initial_sequence():
    return [msg(interpreted_problem()), call(), msg(norm_ready()), msg(interpreted_analysis())]


def test_general_rule_is_reviewed_without_searching_for_identical_words(tmp_path):
    with service(tmp_path, [*initial_sequence(), msg(review())]) as svc:
        r = svc.ask("test", QUESTION, mode="analyze")
        assert r["exam_analysis"]["answer_complete"] and r["exam_analysis"]["explanation_complete"]
        assert r["exam_analysis"]["review_status"] == "passed"
        assert r["exam_analysis"]["selected_options"] == ["A"]
        assert r["metrics"]["retrieval_rounds"] == 1
        assert r["metrics"]["model_calls"] == 5
        assert "source_quote" in r["exam_analysis"]["judgments"][0]["rule_basis"][0]
        audit = json.loads(svc.llm.inputs[-1][-1].content)
        assert set(audit) == {"problem", "analysis", "evidence"}


@pytest.mark.parametrize(
    "reason",
    [
        "一般原则不能直接推出具体处罚金额。",
        "未取得权利的第三人不能自动归入专利权人。",
        "不得把未知条件当成条件不成立。",
        "不能用一般原则绕过明确的特别规定。",
        "事实不足以支持该主体归类。",
    ],
)
def test_rejected_application_never_reaches_answer_or_explanation(tmp_path, reason):
    rejected = review("invalid_application")
    rejected["issues"][0]["reason"] = reason
    a = interpreted_analysis()
    a["judgments"][0]["reason"] = "UNREVIEWED_CONCLUSION"
    a["explanation"]["text"] = "UNREVIEWED_EXPLANATION"
    with service(tmp_path, [*initial_sequence()[:-1], msg(a), msg(rejected)]) as svc:
        r = svc.ask("test", QUESTION, mode="analyze")
        assert r["answer"]["status"] == "limited"
        assert r["exam_analysis"]["selected_options"] == []
        assert "UNREVIEWED" not in json.dumps(r)
        assert r["metrics"]["interpretation_backtracks"] == 0


def test_review_missing_rule_retrieves_and_rechecks_with_shared_budget(tmp_path):
    gap = review("missing_rule", missing_provisions=[{"source_type": "patent_law", "locator": "article:2"}])
    with service(
        tmp_path,
        [*initial_sequence(), msg(gap), msg(norm_ready()), msg(interpreted_analysis()), msg(review())],
    ) as svc:
        r = svc.ask("test", QUESTION, mode="analyze")
        assert r["exam_analysis"]["complete"]
        assert r["metrics"]["interpretation_backtracks"] == 1
        assert r["metrics"]["interpretation_review_calls"] == 2
        assert r["metrics"]["model_calls"] == 8
        assert r["metrics"]["retrieval_rounds"] == 2
        assert len([e for e in r["trace"] if e["node"] == "exam_review"]) == 2


def test_second_review_gap_cannot_open_another_loop(tmp_path):
    gap = review("missing_rule", missing_search={"query": "缺少的必要规则"})
    with service(
        tmp_path, [*initial_sequence(), msg(gap), msg(norm_ready()), msg(interpreted_analysis()), msg(gap)]
    ) as svc:
        svc.s.exam_max_model_calls = 12
        r = svc.ask("test", QUESTION, mode="analyze")
        assert r["metrics"]["interpretation_backtracks"] == 1
        assert r["metrics"]["retrieval_rounds"] == 2
        assert not r["exam_analysis"]["answer_complete"]
        assert r["answer"]["status"] == "limited"


def test_no_audit_budget_fails_closed(tmp_path):
    with service(tmp_path, initial_sequence()) as svc:
        svc.s.exam_max_model_calls = 4
        r = svc.ask("test", QUESTION, mode="analyze")
        assert r["metrics"]["model_calls"] == 4
        assert r["exam_analysis"]["review_status"] == "not_completed"
        assert not r["exam_analysis"]["answer_complete"]
        assert r["exam_analysis"]["selected_options"] == []


def test_optional_explanation_gap_does_not_remove_supported_answer(tmp_path):
    p = problem()
    p["issues"].append(
        {"id": "i2", "question": "学生另外明确询问的程序问题", "query": "其他程序", "purpose": "explanation"}
    )
    a = analysis()
    a["issue_coverage"].append(
        {"issue_id": "i2", "status": "missing_rule", "note": "额外讲解所需规则未收录。"}
    )
    assessment = ready()
    assessment["sufficient"] = False
    assessment["issues"].append({"issue_id": "i2", "status": "missing_rule", "note": "缺少额外讲解依据"})
    with service(tmp_path, [msg(p), call(), msg(assessment), msg(a)]) as svc:
        r = svc.ask("test", QUESTION, mode="analyze")
        assert r["exam_analysis"]["answer_complete"]
        assert not r["exam_analysis"]["explanation_complete"]
        assert r["exam_analysis"]["selected_options"] == ["A"]
        assert r["answer"]["status"] == "limited"


@pytest.mark.parametrize(
    "mutation",
    [
        "foreign_quote",
        "invented_quote",
        "missing_basis",
        "missing_application",
        "missing_boundary",
        "unresolved_premise",
    ],
)
def test_rule_and_application_integrity(mutation):
    a = Analysis.model_validate(interpreted_analysis())
    j = a.judgments[0]
    if mutation == "foreign_quote":
        j.rule_basis[0].evidence_id = "foreign"
    elif mutation == "invented_quote":
        j.rule_basis[0].source_quote = "这不是法规原文"
    elif mutation == "missing_basis":
        j.rule_basis = []
    elif mutation == "missing_application":
        j.application = ""
    elif mutation == "missing_boundary":
        j.boundary_check = ""
    else:
        j.missing_premises = ["缺少必要条件"]
    assert validate_analysis(a, Problem.model_validate(interpreted_problem()), [E], "snap")


def test_assessment_and_audit_cannot_omit_issues_or_use_foreign_sources():
    p = Problem.model_validate(interpreted_problem())
    assessed = ExamAssessment.model_validate(norm_ready())
    assert not validate_normative_assessment(assessed, p, [E], "snap")
    assessed.issues = []
    assert validate_normative_assessment(assessed, p, [E], "snap")
    audited = ApplicationReview.model_validate(review())
    audited.issues[0].evidence_ids = ["foreign"]
    assert validate_review(audited, p, [E], "snap")


def test_malformed_audit_is_not_published_as_approved(tmp_path):
    bad = review()
    bad["issues"] = []
    with service(tmp_path, [*initial_sequence(), msg(bad)]) as svc:
        r = svc.ask("test", QUESTION, mode="analyze")
        assert r["exam_analysis"]["review_status"] == "not_completed"
        assert r["exam_analysis"]["selected_options"] == []


def test_audit_retrieval_reserves_completion_time(tmp_path):
    with service(tmp_path, []) as svc:
        svc.model_calls = 3
        st = {
            "rounds": 1,
            "assessment": {"missing_search": {"query": "必要规则"}},
            "deadline": time.monotonic() + 30,
        }
        assert not InterpretationFlow(svc).can_retrieve(st)
        st["deadline"] = time.monotonic() + 100
        assert InterpretationFlow(svc).can_retrieve(st)
        svc.model_calls = 6
        assert not InterpretationFlow(svc).can_retrieve(st)


def test_interpretation_cards_are_versioned_and_selected_within_budget(tmp_path):
    builtin = load_interpretation_package(tmp_path)
    card = select_interpretation_method(builtin, ["subsumption", "systematic"], 3000)
    assert card and card["method_id"] == "general_scope"
    assert select_interpretation_method(builtin, ["direct"], 3000) is None
    assert select_interpretation_method(builtin, ["systematic"], 1) is None
    payload = {"cards": copy.deepcopy(builtin["cards"])}
    version = package_digest(payload)
    (tmp_path / (version + ".json")).write_text(json.dumps({"payload": payload}))
    (tmp_path / "active.json").write_text(json.dumps({"version": version}))
    assert load_interpretation_package(tmp_path)["version"] == version
    assert load_interpretation_package(tmp_path, builtin["version"])["version"] == builtin["version"]
    payload["cards"][0]["title"] = "tampered"
    (tmp_path / (version + ".json")).write_text(json.dumps({"payload": payload}))
    with pytest.raises(ValueError):
        load_interpretation_package(tmp_path, version)
