import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import AgentService, BusyError
from .models import safe_error
from .retrieval import KnowledgeBase
from .settings import ROOT, Settings
from .storage import active_snapshot


@asynccontextmanager
async def lifespan(app):
    app.state.settings = Settings()
    app.state.service = AgentService(app.state.settings)
    yield
    app.state.service.close()


app = FastAPI(title="中国专利知识库助手", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    question: str = Field(min_length=1, max_length=12000)
    mode: Literal["auto", "lookup", "analyze"] = "auto"
    explanation_level: Literal["hint", "concise", "detailed"] | None = None


def kb(request, snapshot_id):
    try:
        return KnowledgeBase(request.app.state.settings, snapshot_id)
    except ValueError:
        raise HTTPException(404, "知识库快照不存在") from None


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready(request: Request):
    s = request.app.state.settings
    snap = active_snapshot(s)
    if not snap:
        raise HTTPException(503, "请先解析资料并完成索引")
    try:
        store = KnowledgeBase(s, snap["id"])
        store.assert_embedding_config()
        expected = json.loads(snap["payload"])["count"]
        if store.collection.count() != expected:
            raise ValueError("索引数量不一致")
    except Exception:
        raise HTTPException(503, "索引不可用，请运行本地诊断") from None
    return {"status": "ready", "snapshot_id": snap["id"], "chunks": expected}


@app.get("/api/v1/knowledge-base")
def knowledge_base(request: Request):
    snap = active_snapshot(request.app.state.settings)
    if not snap:
        return {"status": "not_built", "documents": []}
    payload = json.loads(snap["payload"])
    return {
        "status": "ready",
        "snapshot_id": snap["id"],
        "chunks": payload["count"],
        "documents": [
            {
                k: d[k]
                for k in (
                    "document_id",
                    "source_type",
                    "title",
                    "version_label",
                    "page_count",
                    "currency_verified",
                    "warnings",
                )
            }
            | {k: d.get(k) for k in ("source_url", "language", "effective_from", "legal_status_note")}
            for d in payload["documents"]
        ],
        "scope_note": "国内法规与指定官方条约按各自版本收录；未统一核验现行性、缔约状态或声明。图片、复杂公式和表格需结合原 PDF。",
    }


@app.post("/api/v1/sessions", status_code=201)
def create_session(request: Request):
    try:
        return request.app.state.service.create_session()
    except ValueError:
        raise HTTPException(503, "知识库未就绪") from None


@app.get("/api/v1/sessions/{session_id}")
def session(session_id: str, request: Request):
    try:
        return request.app.state.service.history(session_id)
    except KeyError:
        raise HTTPException(404, "会话不存在") from None


@app.post("/api/v1/chat")
def chat(body: ChatRequest, request: Request):
    try:
        return request.app.state.service.ask(
            body.session_id, body.question, body.mode, body.explanation_level
        )
    except KeyError:
        raise HTTPException(404, "会话不存在") from None
    except BusyError:
        raise HTTPException(429, "当前正在处理另一个问题，请稍后重试") from None
    except ValueError:
        raise HTTPException(400, "问题为空、超过长度限制或所选模式未启用") from None
    except Exception as exc:
        raise HTTPException(503, "问答暂时失败：" + safe_error(exc)) from None


@app.get("/api/v1/sources/{snapshot_id}/{chunk_id}")
def source(snapshot_id: str, chunk_id: str, request: Request):
    value = kb(request, snapshot_id).chunk(chunk_id)
    if not value:
        raise HTTPException(404, "当前快照中不存在该引用")
    return value


@app.get("/api/v1/provisions/{snapshot_id}/{provision_id}")
def provision(snapshot_id: str, provision_id: str, request: Request):
    value = kb(request, snapshot_id).provision(provision_id)
    if not value:
        raise HTTPException(404, "当前快照中不存在该条文")
    return value


@app.get("/api/v1/documents/{snapshot_id}/{document_id}/pdf")
def pdf(snapshot_id: str, document_id: str, request: Request):
    store = kb(request, snapshot_id)
    docs = json.loads(store.snapshot["payload"])["documents"]
    doc = next((d for d in docs if d["document_id"] == document_id), None)
    if not doc:
        raise HTTPException(404, "文件不存在")
    path = (ROOT / doc["raw_path"]).resolve()
    if not path.is_relative_to(request.app.state.settings.raw_data_dir.resolve()) or not path.is_file():
        raise HTTPException(404, "原始文件不可用")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=doc["original_filename"],
        content_disposition_type="inline",
    )
