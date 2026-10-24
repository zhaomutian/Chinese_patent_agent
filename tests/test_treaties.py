import json
from types import SimpleNamespace

import pytest

from patent_agent.agent import ExactArgs, SearchArgs
from patent_agent.exam.models import LookupSearch, LookupTarget
from patent_agent.ingest import load_pages, parse_document
from patent_agent.retrieval import KnowledgeBase, build_index
from patent_agent.settings import Settings
from patent_agent.sources import SOURCES
from patent_agent.storage import connect, dump


def documents():
    s = Settings()
    return {d["source_type"]: d for d in json.loads(s.sources_manifest_path.read_text())["documents"]}


def test_official_treaty_inventory_and_pdf_provenance():
    docs = documents()
    assert set(docs) == set(SOURCES)
    for d in docs.values():
        if d.get("parser") != "official_treaties":
            continue
        provisions, excluded = parse_document(d, load_pages(d))
        assert set(d["parser_config"]["required_locators"]) <= {p["locator"] for p in provisions}
        assert all(p["source_url"].startswith("https://") for p in provisions)
        body = {s["page"] for p in provisions for s in p["segments"]}
        assert body.isdisjoint({x["page"] for x in excluded})
        assert d["parser_config"]["body_start_page"] <= min(body) <= max(body) <= d["page_count"]


def test_paris_independence_is_body_not_contents_and_has_version_warning():
    d = documents()["paris_convention"]
    ps, _ = parse_document(d, load_pages(d))
    p = next(p for p in ps if p["locator"] == "article:4bis")
    assert p["page_start"] == 7
    assert "相互独立" in p["text"]
    assert "不论是否本联盟的成员国" in p["text"].replace("\n", "")
    assert any("日期" in w or "1979年10月2日" in w for w in p["coverage_warnings"])


def test_pct_footnote_and_deleted_range_do_not_absorb_next_article():
    docs = documents()
    d = docs["pct_regulations"]
    ps, _ = parse_document(d, load_pages(d))
    by = {p["locator"]: p for p in ps}
    assert "文档的获得" in by["rule:94"]["text"]
    assert "第95条" not in by["rule:94"]["text"]
    d = docs["pct_instructions"]
    ps, _ = parse_document(d, load_pages(d))
    by = {p["locator"]: p for p in ps}
    for n in range(426, 430):
        assert "[已删除]" in by[f"section:{n}"]["text"]
        assert "第430条" not in by[f"section:{n}"]["text"]
    assert "附件A" in by["appendix:page:70"]["text"]
    assert any("未附其全文" in w for w in by["appendix:page:70"]["coverage_warnings"])


@pytest.mark.parametrize(
    "typ,locator",
    [
        ("paris_convention", "article:4bis"),
        ("pct_regulations", "rule:13bis"),
        ("hague_instructions", "section:401"),
    ],
)
def test_planner_and_exam_schemas_accept_same_treaty_types(typ, locator):
    assert ExactArgs(source_type=typ, locator=locator).source_type == typ
    assert LookupTarget(source_type=typ, locator=locator).source_type == typ
    assert SearchArgs(query="规则", source_types=[typ]).source_types == [typ]
    assert LookupSearch(query="规则", source_types=[typ]).source_types == [typ]


def test_broad_search_does_not_truncate_later_source_by_declaration_order():
    kb = object.__new__(KnowledgeBase)
    kb.s = SimpleNamespace(question_max_chars=100, retrieval_per_source_k=4, retrieval_top_k=2)
    kb.snapshot = {"payload": dump({"documents": [{"source_type": t} for t in SOURCES]})}
    kb.assert_embedding_config = lambda: None
    kb.emb = SimpleNamespace(encode=lambda *a, **k: [[1]])
    chunks = {t: {"id": t, "source_type": t} for t in SOURCES}
    kb.chunk = lambda cid: chunks[cid]
    kb.section_outline = lambda _: []

    def query(**kw):
        t = kw["where"]["source_type"]
        return {"ids": [[t]], "distances": [[0.01 if t == "hague_instructions" else 1.0]]}

    kb.collection = SimpleNamespace(query=query)
    assert kb.search_knowledge("图片要求")["evidence"][0]["source_type"] == "hague_instructions"
    assert {
        e["source_type"] for e in kb.search_knowledge("规则", ["pct", "paris_convention"])["evidence"]
    } == {"pct", "paris_convention"}


def test_index_reuses_only_identical_embedding_configuration(tmp_path, monkeypatch):
    from patent_agent import retrieval

    monkeypatch.setattr(retrieval, "ROOT", tmp_path)
    s = Settings().model_copy(
        update={
            "catalog_db_path": tmp_path / "catalog.db",
            "chroma_persist_dir": tmp_path / "chroma",
            "processed_data_dir": tmp_path / "processed",
            "sources_manifest_path": tmp_path / "sources.json",
            "embedding_dimensions": 3,
        }
    )
    s.processed_data_dir.mkdir()
    calls = []

    def encode(self, texts):
        calls.extend(texts)
        return [[0.2, 0.3, 0.5] for _ in texts]

    monkeypatch.setattr(retrieval.Embeddings, "encode", encode)

    def inputs(names):
        cs = [
            {
                "id": n,
                "provision_id": "p_" + n,
                "source_type": "pct",
                "title": "PCT",
                "version_label": "v1",
                "chapter": "",
                "heading": "条文",
                "text": n,
            }
            for n in names
        ]
        (s.processed_data_dir / "chunks.jsonl").write_text("\n".join(map(dump, cs)))
        s.sources_manifest_path.write_text(dump({"documents": [{"sha256": "source", "source_type": "pct"}]}))
        with connect(s) as db:
            db.executemany(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?)",
                [(c["id"], c["provision_id"], c["source_type"], dump(c)) for c in cs],
            )

    inputs(["one"])
    first = build_index(s)
    inputs(["one", "two"])
    second = build_index(s)
    assert first["snapshot_id"] != second["snapshot_id"]
    assert second["copied_from_previous_snapshot"] == 1
    assert len(calls) == 2
    changed = s.model_copy(update={"embedding_model": "another-embedding-model"})
    third = build_index(changed)
    assert third["copied_from_previous_snapshot"] == 0
    assert len(calls) == 4
