import httpx
import streamlit as st

from patent_agent.settings import Settings

st.set_page_config(page_title="中国专利知识库助手", page_icon="⚖️", layout="wide")
s = Settings()
base = s.api_base_url.rstrip("/")


def api(method, path, **kwargs):
    try:
        r = httpx.request(method, base + path, timeout=s.run_timeout_seconds + 30, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"服务返回 {exc.response.status_code}，请稍后重试。")
    except httpx.RequestError:
        st.error("暂时无法连接服务，请确认后端已启动。")
    return None


def show_result(result):
    citations = result.get("citations", [])
    numbers = {c["id"]: str(i + 1) for i, c in enumerate(citations)}
    answer = result["answer"]
    if answer["status"] == "limited":
        st.info("本次回答存在资料或服务限制，请留意下方说明。")
    for para in answer["paragraphs"]:
        markers = " ".join("[" + numbers[eid] + "]" for eid in para["evidence_ids"] if eid in numbers)
        st.markdown(para["text"] + (" " + markers if markers else ""))
    for quote in answer.get("quotes", []):
        st.markdown("> " + quote["text"])
    analysis = result.get("exam_analysis")
    if analysis and analysis.get("issue_coverage"):
        with st.expander("查看各问题的依据覆盖"):
            labels = {
                "covered": "已取得依据",
                "missing_rule": "缺少规则原文",
                "missing_fact": "缺少题设条件",
                "version_gap": "版本依据缺口",
                "interpretation_disputed": "适用关系尚未核实",
            }
            for issue in analysis["issue_coverage"]:
                st.write(
                    labels.get(issue["status"], issue["status"]) + "：" + issue.get("note", issue["issue_id"])
                )
    if analysis and "answer_complete" in analysis:
        if analysis["answer_complete"] and not analysis["explanation_complete"]:
            st.info("选项答案已有依据，进一步讲解仍有资料缺口。")
        for judgment in analysis.get("judgments", []):
            if judgment["verdict"] == "undetermined" or not judgment.get("rule_basis"):
                continue
            with st.expander(judgment["label"] + "：原文规定与本题适用"):
                for rule in judgment["rule_basis"]:
                    marker = numbers.get(rule["evidence_id"], "")
                    st.markdown("**原文规定**" + (f" [{marker}]" if marker else ""))
                    st.text(rule["source_quote"])
                    st.write("规则含义：" + rule["rule"])
                for field, label in [
                    ("interpretation", "规范解释"),
                    ("application", "本题适用"),
                    ("boundary_check", "适用边界"),
                ]:
                    if judgment.get(field):
                        st.write(label + "：" + judgment[field])
    if citations:
        st.caption("引用依据 · 页码为 PDF 物理页序号")
    for i, c in enumerate(citations):
        with st.expander(
            f"[{i + 1}] {c['title']} · {c['version_label']} · {c['heading']} · 第 {c['page_start']}–{c['page_end']} 页"
        ):
            st.text(c["text"])
            if c.get("source_url"):
                st.link_button("查看官方来源", c["source_url"])
            for warning in c["coverage_warnings"]:
                st.warning(warning)
            st.link_button(
                "查看原 PDF 对应页",
                f"{base}/api/v1/documents/{result['snapshot_id']}/{c['document_id']}/pdf#page={c['page_start']}",
            )
            if st.button(
                "展开完整条文 / 小节",
                key=result.get("turn_id", result["session_id"] + str(result["metrics"]["elapsed_seconds"]))
                + c["id"],
            ):
                p = api("GET", f"/api/v1/provisions/{result['snapshot_id']}/{c['provision_id']}")
                if p:
                    st.text(p["text"])
    with st.expander("查看检索过程"):
        st.json(result["trace"], expanded=False)
        st.caption(
            f"检索 {result['metrics']['retrieval_rounds']} 轮 · 工具 {result['metrics']['tools_executed']} 次 · 耗时 {result['metrics']['elapsed_seconds']} 秒"
        )


st.title("中国专利知识库助手")
st.caption("从条文与审查指南中查找依据，让每次回答都能回到原文。")
knowledge = api("GET", "/api/v1/knowledge-base")
with st.sidebar:
    mode_label = st.selectbox("使用方式", ["自动识别", "法规查询", "题目分析"])
    mode = {"自动识别": "auto", "法规查询": "lookup", "题目分析": "analyze"}[mode_label]
    level_label = st.selectbox("讲解方式", ["跟随对话", "简明解释", "详细解释", "只给提示"])
    level = {"跟随对话": None, "简明解释": "concise", "详细解释": "detailed", "只给提示": "hint"}[level_label]
    st.header("知识库范围")
    if knowledge:
        for doc in knowledge.get("documents", []):
            st.markdown("**" + doc["title"] + "**")
            st.caption(doc["version_label"] + f" · {doc['page_count']} 页")
            if doc.get("source_url"):
                st.link_button("官方来源", doc["source_url"])
        st.info(knowledge.get("scope_note", "知识库正在准备。"))
    if st.button("开始新会话", use_container_width=True):
        created = api("POST", "/api/v1/sessions")
        if created:
            st.session_state.session_id = created["session_id"]
            st.query_params["session"] = created["session_id"]
            st.session_state.turns = []
            st.rerun()
    st.caption("回答用于资料检索与理解")

if not knowledge or knowledge.get("status") != "ready":
    st.warning("知识库尚未就绪，请先完成资料解析和索引。")
    st.stop()
if "session_id" not in st.session_state and st.query_params.get("session"):
    restored = api("GET", "/api/v1/sessions/" + st.query_params["session"])
    if restored:
        st.session_state.session_id = restored["id"]
        st.session_state.turns = [
            {"question": t["question"], "result": t.get("result"), "answer": t["answer"]}
            for t in restored["history"]
        ]
if "session_id" not in st.session_state:
    created = api("POST", "/api/v1/sessions")
    if not created:
        st.stop()
    st.session_state.session_id = created["session_id"]
    st.query_params["session"] = created["session_id"]
    st.session_state.turns = []
if not st.session_state.turns:
    st.markdown("可以试着提问：")
    st.markdown(
        "- 专利法第二十二条规定了什么？\n- 审查指南如何判断发明的创造性？\n- 说明书和权利要求书分别需要满足哪些要求？"
    )
for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("result"):
            show_result(turn["result"])
        else:
            st.markdown(turn.get("answer", ""))
question = st.chat_input("输入专利法相关问题", max_chars=s.question_max_chars)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("正在查找依据并核对引用…"):
            result = api(
                "POST",
                "/api/v1/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "question": question,
                    "mode": mode,
                    "explanation_level": level,
                },
            )
        if result:
            show_result(result)
            st.session_state.turns.append({"question": question, "result": result})
