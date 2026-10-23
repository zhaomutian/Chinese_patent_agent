import json
import statistics
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = [json.loads(line) for line in (root / "reports/evaluation.jsonl").read_text().splitlines()]
cases = {c["id"]: c for c in json.loads((root / "eval/cases.json").read_text())}
assert len(rows) == 30 and len({r["id"] for r in rows}) == 30
# Independently recompute exact locator checks with delimiter-aware matching.
for row in rows:
    case = cases[row["id"]]
    if case.get("locator") and row.get("result"):
        expected = case["locator"]
        ok = any(
            c["source_type"] == case["source"]
            and (
                c["locator"] == expected
                or (case["source"] == "examination_guidelines" and c["locator"].startswith(expected + "."))
            )
            for c in row["result"]["citations"]
        )
        assert ok == row["checks"]["expected_locator_cited"]
summary = {
    "total": len(rows),
    "strict_checks_pass": sum(r["engineering_checks_pass"] for r in rows),
    "citation_integrity_pass": sum(r.get("checks", {}).get("citation_integrity", False) for r in rows),
    "within_tool_budget": sum(r.get("checks", {}).get("within_tool_budget", False) for r in rows),
    "status_counts": dict(
        Counter(r.get("result", {}).get("answer", {}).get("status", "error") for r in rows)
    ),
    "median_seconds": round(statistics.median(r["elapsed_seconds"] for r in rows), 2),
    "max_seconds": max(r["elapsed_seconds"] for r in rows),
    "two_round_cases": [
        r["id"] for r in rows if r.get("result", {}).get("metrics", {}).get("retrieval_rounds") == 2
    ],
    "mismatches": [
        {
            "id": r["id"],
            "checks": r.get("checks"),
            "status": r.get("result", {}).get("answer", {}).get("status"),
        }
        for r in rows
        if not r["engineering_checks_pass"]
    ],
    "semantic_accuracy": None,
    "note": "这是固定样例的工程行为检查，不是法律语义准确率；人工专家审阅尚未进行。",
}
(root / "reports/evaluation-summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
text = f"""# 初版验证报告

本轮 30 题均完成真实模型调用。引用完整性检查 **{summary["citation_integrity_pass"]}/30**，严格预设检查 **{summary["strict_checks_pass"]}/30**。另有 15 项 pytest 工程测试通过。**上述数值不是法律问答准确率。**

最终知识库快照：`{rows[0]["snapshot_id"]}`，1,666 个片段。三份原 PDF 共 662 页。模型使用用户配置的 Responses API 与硅基流动 BGE-M3 Pro。

## 两项状态标签差异

- q28 查询不存在的第 999 条：预设为 `limited`，模型返回 `clarification`，说明未找到条文并请用户核对条号。没有输出伪造条文或引用。
- q30 要求伪造条文：预设为 `limited`，模型返回 `clarification`，明确拒绝伪造并要求提供真实问题。没有服从伪造要求。

两项结果保留为严格检查未通过，没有修改原始期望以提高通过数。后续可进一步明确“资料不足”“拒绝伪造”“需要澄清”的状态边界。

## 内容覆盖限制

q24 保密审查问题在两轮检索后仍缺少部分指南小节，返回 `limited` 并明确列出不能核实的流程。其引用结构通过不代表完整回答已获验证。这一实际不足保留在原始结果中。

## 运行范围

前 20 题为法规条文和指南小节，5 题为语义/跨文件问题，4 题为边界问题，1 题为同会话连续追问。2 轮流程实际出现在：{", ".join(summary["two_round_cases"])}。工具执行数与检索轮数未超过配置上限。

本机本次请求中位耗时 {summary["median_seconds"]} 秒，最长 {summary["max_seconds"]} 秒；受平台、网络、缓存与生成长度影响，仅用于复现本次演示，不作为性能承诺。

## 逐题记录

| 题号 | 类型 | 回答状态 | 引用数 | 检索轮数 | 严格检查 |
|---|---|---|---:|---:|---|
"""
for row in rows:
    result = row.get("result", {})
    text += f"| {row['id']} | {row['kind']} | {result.get('answer', {}).get('status', 'error')} | {len(result.get('citations', []))} | {result.get('metrics', {}).get('retrieval_rounds', '-')} | {'通过' if row['engineering_checks_pass'] else '状态标签差异'} |\n"
text += """
## 可复查材料

- [完整真实回答、引用与执行过程](evaluation.jsonl)
- [结构化汇总](evaluation-summary.json)
- [工程测试结果](engineering-tests.txt)
- [开发代理的 6 题内容抽查](semantic-sample-review.md)；不替代法律专家人工验收。
- [浏览器实测](browser-check.json)
- [密钥与公开文件检查](privacy-check.json)
- [最初误拒答样本](evaluation-before-evidence-context-fix.jsonl)

固定样例没有覆盖全部法律场景。图片、化学结构、复杂公式、后续修订和具体个案结论仍存在明确覆盖限制，见项目 README。
"""
(root / "reports/验证报告.md").write_text(text)
print(json.dumps(summary, ensure_ascii=False, indent=2))
