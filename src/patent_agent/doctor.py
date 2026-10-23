import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .models import Embeddings, chat_client, message_text, parse_json, safe_error
from .settings import ROOT


class ProbeState(TypedDict):
    count: int


def doctor(s, live=False):
    checks = []

    def check(name, fn):
        started = time.monotonic()
        try:
            details = fn()
            checks.append(
                {
                    "name": name,
                    "ok": True,
                    "seconds": round(time.monotonic() - started, 2),
                    "details": details,
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "name": name,
                    "ok": False,
                    "seconds": round(time.monotonic() - started, 2),
                    "error": safe_error(exc),
                }
            )

    def local():
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            conn = sqlite3.connect(str(Path(tmp) / "checkpoint.db"), check_same_thread=False)
            try:
                graph = StateGraph(ProbeState)
                graph.add_node("increment", lambda st: {"count": st["count"] + 1})
                graph.add_edge(START, "increment")
                graph.add_edge("increment", END)
                compiled = graph.compile(checkpointer=SqliteSaver(conn))
                cfg = {"configurable": {"thread_id": "probe"}}
                assert compiled.invoke({"count": 0}, cfg)["count"] == 1
                assert compiled.get_state(cfg).values["count"] == 1
            finally:
                conn.close()
            client = chromadb.PersistentClient(
                path=str(Path(tmp) / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
            )
            col = client.create_collection("probe", embedding_function=None)
            col.add(ids=["a"], embeddings=[[1.0, 0.0]], documents=["技术样本"])
            assert col.query(query_embeddings=[[1.0, 0.0]], n_results=1)["ids"] == [["a"]]
        return {"checkpoint": "persisted", "chroma": "explicit vectors passed"}

    check("local_graph_sqlite_chroma", local)
    if live:
        llm = chat_client(s)

        def plain():
            msg = llm.invoke([HumanMessage(content="这是接口连通性测试。只回复：连接成功")])
            assert message_text(msg)
            return {
                "usage": msg.usage_metadata,
                "response_status": msg.response_metadata.get("status"),
                "has_text": True,
            }

        check("responses_text", plain)
        schema = {
            "name": "echo_probe",
            "description": "无副作用的接口测试；返回给定的 token。",
            "parameters": {
                "type": "object",
                "properties": {"token": {"type": "string"}},
                "required": ["token"],
                "additionalProperties": False,
            },
        }

        def tool_test(multi=False):
            prompt = (
                "仅使用 echo_probe 工具，token 为 alpha。"
                if not multi
                else "协议测试：必须在同一个响应里并行调用两次 echo_probe，分别使用 token=alpha 和 token=beta。这两个输入互不依赖，不要等待第一个结果，不要用逗号合并 token。"
            )
            msgs = [HumanMessage(content=prompt)]
            call = llm.bind_tools([schema], tool_choice="required", parallel_tool_calls=True).invoke(msgs)
            assert len(call.tool_calls) >= (2 if multi else 1), "工具调用数不符合测试要求"
            ids = [c["id"] for c in call.tool_calls]
            assert len(set(ids)) == len(ids)
            msgs.append(call)
            for c in call.tool_calls:
                msgs.append(
                    ToolMessage(content=json.dumps({"token": c["args"]["token"]}), tool_call_id=c["id"])
                )
            result = llm.bind_tools([schema], tool_choice="none").invoke(msgs)
            assert message_text(result)
            return {"call_count": len(ids), "roundtrip": True, "usage": result.usage_metadata}

        check("responses_single_tool", tool_test)
        check("responses_multiple_tools", lambda: tool_test(True))

        def structured():
            obj = parse_json(
                llm.invoke(
                    [HumanMessage(content='只输出合法 JSON：{"ok": true, "value": 2}，不要 Markdown。')]
                )
            )
            assert obj == {"ok": True, "value": 2}
            return {"json_valid": True}

        check("json_output", structured)
        emb = Embeddings(s)

        def vectors():
            a = emb.encode(["这是接口测试样本甲"])
            b = emb.encode(["这是接口测试样本甲", "这是接口测试样本乙"])
            assert len(a) == 1 and len(b) == 2
            assert sum(x * y for x, y in zip(a[0], b[0])) > sum(x * y for x, y in zip(a[0], b[1]))
            return {"dimensions": len(a[0]), "ordered": True, "requests": emb.requests, "tokens": emb.tokens}

        check("embedding_single_batch", vectors)
    report = {"live": live, "ok": all(c["ok"] for c in checks), "checks": checks}
    (ROOT / "reports" / ("doctor-live.json" if live else "doctor-static.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    return report
