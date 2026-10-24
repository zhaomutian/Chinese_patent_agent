from fastapi.testclient import TestClient

from patent_agent.api import app
from patent_agent.models import Embeddings
from patent_agent.retrieval import KnowledgeBase
from patent_agent.settings import Settings
from patent_agent.storage import active_snapshot


def test_health_metadata_sources_and_request_validation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("read-only endpoints must not call a paid model")

    monkeypatch.setattr(Embeddings, "encode", forbidden)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        knowledge = client.get("/api/v1/knowledge-base").json()
        assert {"patent_law", "implementing_rules", "examination_guidelines"} <= {
            d["source_type"] for d in knowledge["documents"]
        }
        created = client.post("/api/v1/sessions")
        assert created.status_code == 201
        sid = created.json()["session_id"]
        assert client.get("/api/v1/sessions/" + sid).json()["history"] == []
        assert client.post("/api/v1/chat", json={"session_id": sid, "question": ""}).status_code == 422
        assert client.get("/api/v1/sessions/missing").status_code == 404
        snapshot = knowledge["snapshot_id"]
        assert client.get(f"/api/v1/sources/{snapshot}/missing").status_code == 404
        kb = KnowledgeBase(Settings(), snapshot)
        c = kb.get_provision("patent_law", "article:22")["evidence"][0]
        assert client.get(f"/api/v1/sources/{snapshot}/{c['id']}").json()["page_start"] == 4
        assert client.get(f"/api/v1/provisions/{snapshot}/{c['provision_id']}").status_code == 200
        pdf = client.get(f"/api/v1/documents/{snapshot}/{c['document_id']}/pdf")
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")


def test_changed_embedding_model_is_rejected_before_network():
    import pytest

    s = Settings()
    snapshot = active_snapshot(s)["id"]
    changed = s.model_copy(update={"embedding_model": "different-model"})
    store = KnowledgeBase(changed, snapshot)
    with pytest.raises(ValueError, match="向量模型配置与快照不一致"):
        store.search_knowledge("查询")
    assert store.get_provision("patent_law", "article:22")["evidence"]
