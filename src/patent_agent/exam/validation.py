import re
import unicodedata


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def validate_problem(problem, message, previous):
    errors = []
    source = message
    if previous and problem.action != "new":
        source += "\n" + previous.get("source_text", "")
    elif problem.action != "new":
        errors.append("没有当前题目，不能作为追问或条件变更")
    original_labels = set(re.findall(r"(?:^|\n)\s*([A-H])[.．、:：)）]", source))
    if problem.action == "new" and original_labels and original_labels != set(problem.options):
        errors.append("输入中的选项未完整保留")
    if previous and problem.action == "continue" and set(problem.options) != set(previous.get("options", {})):
        errors.append("追问不能遗漏或增加原题选项")
    if problem.action == "new":
        markers = list(re.finditer(r"(?:^|\n)\s*([A-H])[.．、:：)）]", message))
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(message)
            original = normalized(message[marker.end() : end])
            extracted = normalized(problem.options.get(marker.group(1), ""))
            # The final option may be followed by the student's request for explanation.
            matches = original == extracted if index + 1 < len(markers) else original.startswith(extracted)
            if not extracted or not matches:
                errors.append("选项标签与原文内容不对应或内容被截断")
    source = normalized(source)
    if problem.action == "new" and normalized(problem.stem) not in source:
        errors.append("题干不是输入原文")
    if problem.options and problem.polarity == "open_analysis":
        errors.append("选择题需要识别设问方向")
    for label, text in problem.options.items():
        if not re.fullmatch(r"[A-H]", label) or not text.strip() or normalized(text) not in source:
            errors.append("选项标签无效或选项内容不是输入原文")
    if re.search(r"(?:不正确|错误|不符合|不属于).{0,8}(?:的是|的有|选项)", problem.stem):
        if problem.polarity != "select_incorrect":
            errors.append("否定设问被识别为选择正确项")
    if "多选" in message and problem.question_type == "single_choice":
        errors.append("明确的多选题被识别为单选")
    if "单选" in message and problem.question_type == "multiple_choice":
        errors.append("明确的单选题被识别为多选")
    if not problem.issues and not problem.needs_clarification:
        errors.append("可分析的题目必须包含争点")
    labels = set(problem.options)
    if problem.question_type in ("single_choice", "multiple_choice") and not labels:
        errors.append("选择题缺少选项")
    if problem.focus_option and problem.focus_option not in labels:
        errors.append("关注选项不在当前题目中")
    facts = {f.id for f in problem.facts}
    if len(facts) != len(problem.facts) or len({i.id for i in problem.issues}) != len(problem.issues):
        errors.append("事实或争点编号重复")
    for fact in problem.facts:
        if normalized(fact.quote) not in source:
            errors.append("事实缺少有效原文定位")
    for issue in problem.issues:
        if issue.purpose == "explanation" and issue.option_labels:
            errors.append("讲解争点不能替代选项判断争点")
        if not set(issue.fact_ids) <= facts or not set(issue.option_labels) <= labels:
            errors.append("争点引用了不存在的事实或选项")
    if labels and not problem.needs_clarification:
        covered = {
            label for issue in problem.issues if issue.purpose == "answer" for label in issue.option_labels
        }
        if not labels <= covered:
            errors.append("部分选项没有关联争点")
    return list(dict.fromkeys(errors))


def validate_analysis(analysis, problem, evidence, snapshot):
    allowed = {e["id"] for e in evidence if e.get("snapshot_id") == snapshot}
    errors = []
    labels = [j.label for j in analysis.judgments]
    if len(set(labels)) != len(labels) or set(labels) != set(problem.options):
        errors.append("逐项判断必须完整且不重复地覆盖选项")
    facts = {f.id for f in problem.facts}
    issues = {i.id for i in problem.issues}
    coverage = [c.issue_id for c in analysis.issue_coverage]
    if len(set(coverage)) != len(coverage) or set(coverage) != issues:
        errors.append("必须逐项报告争点覆盖状态")
    for entry in [*analysis.judgments, *analysis.issue_coverage, analysis.explanation]:
        if not set(entry.evidence_ids) <= allowed:
            errors.append("使用了本轮快照之外的依据")
    by_id = {e["id"]: e for e in evidence if e.get("snapshot_id") == snapshot}
    for judgment in analysis.judgments:
        if judgment.verdict != "undetermined":
            if not judgment.rule_basis or judgment.missing_premises:
                errors.append("确定判断必须有规则原文且不能仍缺必要前提")
            if judgment.support_type != "direct" and not all(
                (
                    judgment.interpretation.strip(),
                    judgment.application.strip(),
                    judgment.boundary_check.strip(),
                )
            ):
                errors.append("解释性判断必须说明规范含义、具体适用和边界检查")
        for basis in judgment.rule_basis:
            source = by_id.get(basis.evidence_id)
            if not source or basis.source_quote not in source["text"]:
                errors.append("规则引文不是对应证据的逐字原文")
            if basis.evidence_id not in judgment.evidence_ids:
                errors.append("规则依据与选项引用不一致")
        if judgment.verdict != "undetermined" and not judgment.evidence_ids:
            errors.append("确定的选项判断必须附法律依据")
        if not set(judgment.fact_ids) <= facts or not set(judgment.issue_ids) <= issues:
            errors.append("判断引用了无效事实或争点")
        if not judgment.issue_ids:
            errors.append("选项判断缺少关联争点")
        related = {i.id for i in problem.issues if judgment.label in i.option_labels}
        if not set(judgment.issue_ids) <= related:
            errors.append("选项判断与审题争点不对应")
    for item in analysis.issue_coverage:
        if item.status == "covered" and not item.evidence_ids:
            errors.append("已覆盖的争点必须附法律依据")
    if not analysis.explanation.evidence_ids and not analysis.limitations:
        errors.append("没有依据的解释必须明确资料限制")
    return list(dict.fromkeys(errors))


def selected_options(problem, analysis):
    if not problem.options or problem.polarity not in ("select_correct", "select_incorrect"):
        return []
    if any(j.verdict == "undetermined" for j in analysis.judgments):
        return []
    expected = "correct" if problem.polarity == "select_correct" else "incorrect"
    selected = sorted(j.label for j in analysis.judgments if j.verdict == expected)
    if problem.question_type == "single_choice" and len(selected) != 1:
        return []
    if problem.question_type == "unknown":
        return []
    return selected


def validate_normative_assessment(assessment, problem, evidence, snapshot):
    allowed = {e["id"] for e in evidence if e.get("snapshot_id") == snapshot}
    ids = [i.issue_id for i in assessment.issues]
    errors = []
    if len(ids) != len(set(ids)) or set(ids) != {i.id for i in problem.issues}:
        errors.append("规范检查必须逐项覆盖全部争点")
    for issue in assessment.issues:
        if not set(issue.evidence_ids) <= allowed:
            errors.append("规范检查引用不属于本轮证据")
        if issue.status in {"direct_supported", "interpretation_ready"}:
            if not issue.evidence_ids or issue.missing_premises:
                errors.append("前提就绪必须有依据且不能仍缺必要前提")
    return list(dict.fromkeys(errors))


def validate_review(review, problem, evidence, snapshot):
    allowed = {e["id"] for e in evidence if e.get("snapshot_id") == snapshot}
    ids = [i.issue_id for i in review.issues]
    errors = []
    if len(ids) != len(set(ids)) or set(ids) != {i.id for i in problem.issues}:
        errors.append("审核必须逐项覆盖全部争点")
    for issue in review.issues:
        if not set(issue.evidence_ids) <= allowed:
            errors.append("审核引用不属于本轮证据")
        if issue.status == "approved" and not issue.evidence_ids:
            errors.append("通过审核的判断必须有当前证据")
    return list(dict.fromkeys(errors))
