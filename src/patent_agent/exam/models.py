from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..sources import SourceType


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Fact(StrictModel):
    id: str
    text: str
    quote: str = Field(min_length=1)
    origin: Literal["explicit", "hypothetical"] = "explicit"


class Issue(StrictModel):
    id: str
    question: str
    option_labels: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    query: str = Field(min_length=1, max_length=2000)
    analysis_type: Literal["direct", "subsumption", "systematic", "condition_application"] = "direct"
    purpose: Literal["answer", "explanation"] = "answer"


class Problem(StrictModel):
    action: Literal["new", "continue", "revise"]
    stem: str = Field(min_length=1, max_length=6000)
    options: dict[str, str] = Field(default_factory=dict, max_length=8)
    question_type: Literal["single_choice", "multiple_choice", "case_analysis", "unknown"]
    polarity: Literal["select_correct", "select_incorrect", "open_analysis", "unknown"]
    facts: list[Fact] = Field(default_factory=list, max_length=16)
    issues: list[Issue] = Field(default_factory=list, max_length=6)
    missing_conditions: list[str] = Field(default_factory=list, max_length=6)
    chapter_ids: list[str] = Field(default_factory=list, max_length=2)
    focus_option: str | None = None
    student_question: str = ""
    needs_clarification: bool = False


class RuleBasis(StrictModel):
    evidence_id: str
    source_quote: str = Field(min_length=1, max_length=1200)
    rule: str = Field(min_length=1, max_length=800)


class Judgment(StrictModel):
    label: str
    verdict: Literal["correct", "incorrect", "undetermined"]
    support_type: Literal["direct", "subsumption", "systematic", "condition_application"] = "direct"
    rule_basis: list[RuleBasis] = Field(default_factory=list, max_length=6)
    interpretation: str = Field(default="", max_length=800)
    application: str = Field(default="", max_length=800)
    boundary_check: str = Field(default="", max_length=800)
    missing_premises: list[str] = Field(default_factory=list, max_length=6)
    reason: str = Field(min_length=1, max_length=1800)
    evidence_ids: list[str] = Field(default_factory=list)
    issue_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)


class IssueCoverage(StrictModel):
    issue_id: str
    status: Literal["covered", "missing_rule", "missing_fact", "version_gap", "interpretation_disputed"]
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = ""


class Explanation(StrictModel):
    text: str = Field(min_length=1, max_length=2400)
    evidence_ids: list[str] = Field(default_factory=list)


class Analysis(StrictModel):
    judgments: list[Judgment] = Field(default_factory=list, max_length=8)
    issue_coverage: list[IssueCoverage] = Field(default_factory=list, max_length=6)
    explanation: Explanation
    limitations: list[str] = Field(default_factory=list, max_length=6)


class MethodCard(StrictModel):
    chapter_id: str
    title: str = Field(min_length=1, max_length=100)
    applicable_when: str = Field(min_length=1, max_length=300)
    checklist: list[str] = Field(min_length=3, max_length=7)
    pitfalls: list[str] = Field(min_length=1, max_length=4)
    source_question_ids: list[str] = Field(min_length=1)


class NormativeSupport(StrictModel):
    issue_id: str
    status: Literal[
        "direct_supported",
        "interpretation_ready",
        "missing_rule",
        "missing_fact",
        "interpretation_disputed",
        "version_gap",
    ]
    evidence_ids: list[str] = Field(default_factory=list)
    missing_premises: list[str] = Field(default_factory=list, max_length=6)
    note: str = ""


class LookupTarget(StrictModel):
    source_type: SourceType
    locator: str = Field(min_length=1, max_length=120)


class LookupSearch(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    source_types: list[SourceType] | None = None


class ExamAssessment(StrictModel):
    issues: list[NormativeSupport] = Field(default_factory=list, max_length=6)
    sufficient: bool = False  # Computed again from issue states by the application.
    clarify: bool = False
    reason: str = ""
    missing_query: str = ""
    missing_provisions: list[LookupTarget] = Field(default_factory=list, max_length=8)
    missing_search: LookupSearch | None = None


class IssueReview(StrictModel):
    issue_id: str
    status: Literal["approved", "missing_rule", "missing_fact", "disputed", "invalid_application"]
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=800)


class ApplicationReview(StrictModel):
    issues: list[IssueReview] = Field(max_length=6)
    explanation_supported: bool
    explanation_note: str = Field(default="", max_length=800)
    missing_provisions: list[LookupTarget] = Field(default_factory=list, max_length=8)
    missing_search: LookupSearch | None = None


class InterpretationCard(StrictModel):
    method_id: str = Field(pattern=r"^[a-z_]+$")
    analysis_types: list[Literal["subsumption", "systematic", "condition_application"]] = Field(min_length=1)
    title: str = Field(min_length=1, max_length=100)
    applicable_when: str = Field(min_length=1, max_length=250)
    checklist: list[str] = Field(min_length=3, max_length=6)
    pitfalls: list[str] = Field(min_length=1, max_length=4)
    lookup_triggers: list[str] = Field(min_length=1, max_length=4)
    source_question_ids: list[str] = Field(default_factory=list)
