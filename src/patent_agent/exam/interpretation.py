"""Issue-level normative assessment, interpretation review and bounded backtracking."""

import time

from langchain_core.messages import HumanMessage, SystemMessage

from ..agent import limited
from ..models import parse_json, safe_error
from ..storage import dump
from .models import Analysis, ApplicationReview, ExamAssessment, Problem
from .policies import ANALYZE_POLICY, ASSESS_POLICY, REVIEW_POLICY
from .validation import selected_options, validate_analysis, validate_normative_assessment, validate_review

READY = {"direct_supported", "interpretation_ready"}


def current_problem(st):
    return Problem.model_validate(
        {k: v for k, v in st["teaching_problem"].items() if k in Problem.model_fields}
    )


def needs_review(st, analysis=None):
    p = current_problem(st)
    return (
        any(i.analysis_type != "direct" for i in p.issues)
        or any(i.get("status") == "interpretation_ready" for i in st.get("assessment", {}).get("issues", []))
        or bool(analysis and any(j.support_type != "direct" for j in analysis.judgments))
    )


class InterpretationFlow:
    def __init__(self, service):
        self.service = service

    def can_retrieve(self, st, after_review=False):
        svc = self.service
        # Structured supplements need no planner request: reserve assessment, analysis and review.
        calls = 3 + int(
            not (
                st.get("assessment", {}).get("missing_provisions")
                or st.get("assessment", {}).get("missing_search")
            )
        )
        return (
            st["rounds"] < svc.s.exam_max_retrieval_rounds
            and svc.model_calls + calls <= svc.s.exam_max_model_calls
            and st["deadline"] - time.monotonic() > svc.s.exam_completion_reserve_seconds
            and (not after_review or st.get("exam_review_backtracks", 0) < 1)
        )

    def assess(self, st):
        svc = self.service
        p = current_problem(st)
        payload = {
            "problem": p.model_dump(),
            "student_message": st["question"],
            "evidence": st["evidence"],
            "retrieval_targets": st["retrieval_targets"],
            "tools": [e for e in st["events"] if e["node"] == "execute_tools"],
            "review_feedback": st.get("review_feedback"),
        }
        try:
            msg = svc.invoke(
                st,
                [
                    SystemMessage(
                        content=ASSESS_POLICY + "\nJSON schema：" + dump(ExamAssessment.model_json_schema())
                    ),
                    HumanMessage(content=dump(payload)),
                ],
            )
            assessed = ExamAssessment.model_validate(parse_json(msg))
            errors = validate_normative_assessment(assessed, p, st["evidence"], st["snapshot_id"])
            if errors:
                raise ValueError("；".join(errors))
            result = assessed.model_dump()
            result["sufficient"] = all(i.status in READY for i in assessed.issues)
            result["clarify"] = any(
                i.status == "missing_fact"
                and next(x for x in p.issues if x.id == i.issue_id).purpose == "answer"
                for i in assessed.issues
            )
        except Exception as exc:
            result = {
                "sufficient": False,
                "clarify": False,
                "issues": [],
                "reason": "规范前提检查未完成：" + safe_error(exc),
                "missing_provisions": [],
                "missing_search": None,
            }
        result["unresolved_targets"] = [
            t for t in st.get("retrieval_targets", []) if t["status"] != "included"
        ]
        return {"assessment": result, "events": svc.event(st, "assess_evidence", **result)}

    def route_assessment(self, st):
        a = st["assessment"]
        has_gap_request = bool(a.get("missing_provisions") or a.get("missing_search"))
        if not a.get("sufficient") and not a.get("clarify") and has_gap_request and self.can_retrieve(st):
            return "plan_tools"
        return "exam_analyze"

    def analyze(self, st):
        svc, p = self.service, current_problem(st)
        payload = {
            "student_message": st["question"],
            "problem": p.model_dump(),
            "methods": st["selected_methods"],
            "evidence": st["evidence"],
            "assessment": st["assessment"],
            "review_feedback": st.get("review_feedback"),
            "explanation_level": st["explanation_level"],
        }
        errors, repairs = [], st.get("exam_repairs", 0)
        try:
            for attempt in range(svc.s.exam_max_analysis_repairs + 1):
                if attempt:
                    # Repair and audit share the run budget. No unreviewed explanation is released.
                    if (
                        repairs >= svc.s.exam_max_analysis_repairs
                        or svc.model_calls + 2 > svc.s.exam_max_model_calls
                    ):
                        break
                    repairs += 1
                msg = svc.invoke(
                    st,
                    [
                        SystemMessage(
                            content=ANALYZE_POLICY + "\nJSON schema：" + dump(Analysis.model_json_schema())
                        ),
                        HumanMessage(content=dump(payload)),
                    ],
                )
                try:
                    analysis = Analysis.model_validate(parse_json(msg))
                    errors = validate_analysis(analysis, p, st["evidence"], st["snapshot_id"])
                except ValueError:
                    errors = ["分析JSON结构不合法"]
                if not errors:
                    return {
                        "analysis_draft": analysis.model_dump(),
                        "application_review": None,
                        "exam_repairs": repairs,
                        "review_required": needs_review(st, analysis),
                        "events": svc.event(
                            st, "exam_analyze", review_required=needs_review(st, analysis), repairs=repairs
                        ),
                    }
                payload["validation_errors"] = errors
            message = "分析结果未通过结构或原文校验，暂不能给出可靠结论。"
        except Exception as exc:
            errors = [safe_error(exc)]
            message = "本次分析未能完成，请稍后重试；没有生成未经核实的答案。"
        return {
            "analysis_draft": None,
            "exam_analysis": None,
            "answer": limited(message),
            "exam_repairs": repairs,
            "events": svc.event(st, "exam_analyze", errors=errors),
        }

    def route_analysis(self, st):
        return "exam_review" if st.get("analysis_draft") and st.get("review_required") else "exam_render"

    def review(self, st):
        svc, p = self.service, current_problem(st)
        payload = {"problem": p.model_dump(), "analysis": st["analysis_draft"], "evidence": st["evidence"]}
        count = st.get("exam_review_calls", 0)
        try:
            if svc.model_calls >= svc.s.exam_max_model_calls or st["deadline"] - time.monotonic() <= 1:
                raise TimeoutError("无剩余审核预算")
            count += 1
            msg = svc.invoke(
                st,
                [
                    SystemMessage(
                        content=REVIEW_POLICY
                        + "\nJSON schema："
                        + dump(ApplicationReview.model_json_schema())
                    ),
                    HumanMessage(content=dump(payload)),
                ],
            )
            review = ApplicationReview.model_validate(parse_json(msg))
            errors = validate_review(review, p, st["evidence"], st["snapshot_id"])
            if errors:
                raise ValueError("审核结构或引用无效")
            result = review.model_dump()
            missing = any(i.status == "missing_rule" for i in review.issues)
            requests = bool(review.missing_provisions or review.missing_search)
            assessment = {
                **st["assessment"],
                "sufficient": False,
                "missing_provisions": [t.model_dump() for t in review.missing_provisions] if missing else [],
                "missing_search": review.missing_search.model_dump()
                if missing and review.missing_search
                else None,
                "missing_query": "；".join(i.reason for i in review.issues if i.status == "missing_rule"),
            }
            candidate = {**st, "assessment": assessment}
            backtrack = missing and requests and self.can_retrieve(candidate, after_review=True)
            return {
                "application_review": result,
                "exam_review_calls": count,
                "review_feedback": result,
                "assessment": assessment if backtrack else st["assessment"],
                "review_retry_pending": bool(backtrack),
                "exam_review_backtracks": st.get("exam_review_backtracks", 0) + int(backtrack),
                "events": svc.event(
                    st,
                    "exam_review",
                    statuses={i.issue_id: i.status for i in review.issues},
                    backtrack=bool(backtrack),
                ),
            }
        except Exception as exc:
            return {
                "application_review": None,
                "exam_review_calls": count,
                "review_retry_pending": False,
                "events": svc.event(st, "exam_review", error=safe_error(exc)),
            }

    def route_review(self, st):
        return "plan_tools" if st.get("review_retry_pending") else "exam_render"

    def render(self, st):
        if not st.get("analysis_draft"):
            return {}
        p = current_problem(st)
        a = Analysis.model_validate(st["analysis_draft"])
        reviewed = st.get("application_review")
        required = st.get("review_required", False)
        coverage = {i.issue_id: i for i in a.issue_coverage}
        support = {i["issue_id"]: i["status"] for i in st["assessment"].get("issues", [])}
        audit = {i["issue_id"]: i for i in (reviewed or {}).get("issues", [])}
        approved = {
            i.id: coverage[i.id].status == "covered"
            and (audit.get(i.id, {}).get("status") == "approved" if required else support.get(i.id) in READY)
            for i in p.issues
        }
        for j in a.judgments:
            if not all(approved[i] for i in j.issue_ids):
                j.verdict, j.reason = "undetermined", "该选项的依据或适用判断尚未核实，暂不能确定。"
                j.evidence_ids, j.rule_basis = [], []
                j.interpretation, j.application, j.boundary_check = "", "", ""
        for c in a.issue_coverage:
            if not approved[c.issue_id]:
                decision = audit.get(c.issue_id, {})
                status = decision.get("status")
                c.status = {
                    "disputed": "interpretation_disputed",
                    "invalid_application": "interpretation_disputed",
                    "missing_fact": "missing_fact",
                }.get(status, "missing_rule")
                c.note = decision.get("reason", "该争点的依据或解释尚未通过核实。")
        answer_issues = [i for i in p.issues if i.purpose == "answer"]
        answer_complete = (
            bool(answer_issues) and all(approved[i.id] for i in answer_issues) and not p.missing_conditions
        )
        selected = selected_options(p, a) if answer_complete else []
        answer_complete = answer_complete and (
            bool(selected) if p.options else bool(a.explanation.evidence_ids)
        )
        explanation_safe = bool(a.explanation.evidence_ids) and (
            not required or bool(reviewed and reviewed["explanation_supported"])
        )
        explanation_complete = explanation_safe and all(
            approved[i.id] for i in p.issues if i.purpose == "explanation"
        )
        if not p.options:
            answer_complete = answer_complete and explanation_safe
        complete = answer_complete and explanation_complete
        paragraphs = []
        if selected:
            ids = list(dict.fromkeys(eid for j in a.judgments for eid in j.evidence_ids))
            paragraphs.append(
                {"text": "建议答案：" + "、".join(selected), "kind": "fact", "evidence_ids": ids}
            )
        labels = {"correct": "正确", "incorrect": "错误", "undetermined": "暂不能确定"}
        for j in a.judgments:
            paragraphs.append(
                {
                    "text": f"{j.label}：{labels[j.verdict]}。{j.reason}",
                    "kind": "fact" if j.evidence_ids else "limitation",
                    "evidence_ids": j.evidence_ids,
                }
            )
        if not explanation_safe:
            a.explanation.text, a.explanation.evidence_ids = (
                "部分讲解依据或适用关系尚未核实，请参照各选项的核实状态。",
                [],
            )
        paragraphs.append(
            {
                "text": a.explanation.text,
                "kind": "fact" if a.explanation.evidence_ids else "limitation",
                "evidence_ids": a.explanation.evidence_ids,
            }
        )
        limitations = list(p.missing_conditions) + [c.note for c in a.issue_coverage if c.status != "covered"]
        if not required or (reviewed and reviewed["explanation_supported"]):
            limitations += a.limitations
        if required and reviewed and not reviewed["explanation_supported"]:
            limitations.append(reviewed["explanation_note"] or "讲解存在尚未核实的内容。")
        if required and not reviewed:
            limitations.append("解释审核未完成，未输出未经审核的适用结论。")
        a.limitations = list(dict.fromkeys(limitations))[:6]
        if a.limitations:
            paragraphs.append({"text": "；".join(a.limitations), "kind": "limitation", "evidence_ids": []})
        if len(paragraphs) > 10:
            last = paragraphs.pop()
            paragraphs[-1]["text"] += "\n" + last["text"]
        review_status = (
            "not_required"
            if not required
            else (
                "not_completed"
                if not reviewed
                else ("passed" if all(approved.values()) and explanation_safe else "partial")
            )
        )
        result = a.model_dump() | {
            "selected_options": selected,
            "complete": complete,
            "answer_complete": answer_complete,
            "explanation_complete": explanation_complete,
            "answer_standard": "textbook_for_evaluation",
            "review_status": review_status,
            "normative_support": st["assessment"].get("issues", []),
            "interpretation_review": reviewed,
        }
        return {
            "answer": {
                "status": "supported" if complete else "limited",
                "paragraphs": paragraphs,
                "quotes": [],
            },
            "exam_analysis": result,
            "events": self.service.event(
                st,
                "exam_render",
                answer_complete=answer_complete,
                explanation_complete=explanation_complete,
                review_status=review_status,
            ),
        }
