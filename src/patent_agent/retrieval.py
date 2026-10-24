import json
import time
from datetime import datetime, timezone

import chromadb
from chromadb.config import Settings as ChromaSettings

from .ingest import PARSER_VERSION, digest
from .models import Embeddings, safe_error
from .settings import ROOT
from .sources import SOURCES
from .storage import connect, dump


def client(s):
    return chromadb.PersistentClient(
        path=str(s.chroma_persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )


def embedding_identity(s):
    return [s.embedding_base_url, s.embedding_model, s.embedding_dimensions, "title-heading-body-v1"]


def embedding_text(chunk):
    return f"{chunk['title']} {chunk['version_label']} {chunk['chapter']} {chunk['heading']}\n{chunk['text']}"


def build_index(s):
    chunks = [json.loads(line) for line in (s.processed_data_dir / "chunks.jsonl").read_text().splitlines()]
    manifest = json.loads(s.sources_manifest_path.read_text())
    fingerprint = digest(
        dump(
            {
                "parser": PARSER_VERSION,
                "chunks": [c["id"] for c in chunks],
                "sources": [d["sha256"] for d in manifest["documents"]],
                "chunk_config": [s.chunk_target_chars, s.chunk_max_chars, s.chunk_overlap_chars],
                "embedding": [
                    s.embedding_base_url,
                    s.embedding_model,
                    s.embedding_dimensions,
                    "title-heading-body-v1",
                ],
            }
        )
    )
    sid = "kb_" + fingerprint[:20]
    collection_name = s.chroma_collection_prefix + "_" + fingerprint[:20]
    cl = client(s)
    col = cl.get_or_create_collection(
        collection_name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "fingerprint": fingerprint},
    )
    runid = "run_" + str(time.time_ns())
    with connect(s) as db:
        old = db.execute("SELECT status FROM snapshots WHERE id=?", (sid,)).fetchone()
        if old and old["status"] == "active" and set(col.get(include=[])["ids"]) == {c["id"] for c in chunks}:
            return {"ok": True, "snapshot_id": sid, "chunks": len(chunks), "reused": True}
        db.execute(
            "INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?,?,?)",
            (
                sid,
                collection_name,
                "building",
                fingerprint,
                datetime.now(timezone.utc).isoformat(),
                dump(
                    {
                        "documents": manifest["documents"],
                        "count": len(chunks),
                        "embedding": embedding_identity(s),
                    }
                ),
            ),
        )
        db.execute("INSERT INTO ingestion_runs VALUES (?,?,?,?,?)", (runid, sid, "building", 0, None))
    existing = set(col.get(include=[])["ids"])
    emb = Embeddings(s)
    copied = 0
    try:
        # Identical chunk IDs and embedding configuration can reuse archived vectors.
        with connect(s) as db:
            previous = db.execute(
                "SELECT collection,payload FROM snapshots WHERE id<>? AND status IN ('active','archived') ORDER BY created_at DESC",
                (sid,),
            ).fetchall()
        needed = {c["id"] for c in chunks} - existing
        for prior in previous:
            if not needed or json.loads(prior["payload"]).get("embedding") != embedding_identity(s):
                continue
            source = cl.get_collection(prior["collection"], embedding_function=None)
            matching = sorted(needed & set(source.get(include=[])["ids"]))
            for offset in range(0, len(matching), 100):
                rows = source.get(
                    ids=matching[offset : offset + 100], include=["embeddings", "documents", "metadatas"]
                )
                col.upsert(
                    ids=rows["ids"],
                    embeddings=rows["embeddings"],
                    documents=rows["documents"],
                    metadatas=rows["metadatas"],
                )
                existing.update(rows["ids"])
                needed.difference_update(rows["ids"])
                copied += len(rows["ids"])
        if copied:
            print(dump({"event": "reuse_vectors", "chunks": copied}), flush=True)
        pending = [c for c in chunks if c["id"] not in existing]
        for offset in range(0, len(pending), s.embedding_batch_size):
            batch = pending[offset : offset + s.embedding_batch_size]
            vectors = emb.encode([embedding_text(c) for c in batch])
            col.upsert(
                ids=[c["id"] for c in batch],
                embeddings=vectors,
                documents=[c["text"] for c in batch],
                metadatas=[
                    {"source_type": c["source_type"], "provision_id": c["provision_id"]} for c in batch
                ],
            )
            with connect(s) as db:
                db.execute("UPDATE ingestion_runs SET completed=? WHERE id=?", (col.count(), runid))
            if offset == 0 or offset % 100 == 0:
                print(
                    dump({"event": "index_progress", "complete": col.count(), "total": len(chunks)}),
                    flush=True,
                )
        assert set(col.get(include=[])["ids"]) == {c["id"] for c in chunks}
        with connect(s) as db:
            db.executemany(
                "INSERT OR IGNORE INTO snapshot_chunks VALUES (?,?)", [(sid, c["id"]) for c in chunks]
            )
            catalog_count = db.execute(
                "SELECT COUNT(*) FROM snapshot_chunks sc JOIN chunks c ON c.id=sc.chunk_id WHERE sc.snapshot_id=?",
                (sid,),
            ).fetchone()[0]
            if catalog_count != len(chunks):
                raise ValueError("向量索引与 SQLite 原文目录不一致，快照不会激活")
            # Activation is a single transaction; historical snapshots remain readable.
            db.execute("UPDATE snapshots SET status='archived' WHERE status='active' AND id<>?", (sid,))
            db.execute("UPDATE snapshots SET status='active' WHERE id=?", (sid,))
            db.execute("UPDATE ingestion_runs SET status='complete' WHERE id=?", (runid,))
    except Exception as exc:
        with connect(s) as db:
            db.execute(
                "UPDATE ingestion_runs SET status='failed',error=? WHERE id=?", (safe_error(exc), runid)
            )
        raise
    report = {
        "ok": True,
        "snapshot_id": sid,
        "chunks": col.count(),
        "embedding_requests": emb.requests,
        "embedding_tokens": emb.tokens,
        "reused_chunks": len(existing),
        "copied_from_previous_snapshot": copied,
    }
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/index.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


class KnowledgeBase:
    def __init__(self, s, snapshot_id):
        self.s = s
        self.snapshot_id = snapshot_id
        with connect(s) as db:
            row = db.execute(
                "SELECT * FROM snapshots WHERE id=? AND status IN ('active','archived')", (snapshot_id,)
            ).fetchone()
            if not row:
                raise ValueError("知识库快照不存在或尚未完成")
            self.snapshot = dict(row)
        self.collection = client(s).get_collection(row["collection"], embedding_function=None)
        self.emb = Embeddings(s)

    def chunk(self, cid):
        with connect(self.s) as db:
            row = db.execute(
                "SELECT c.payload FROM chunks c JOIN snapshot_chunks sc ON sc.chunk_id=c.id WHERE sc.snapshot_id=? AND c.id=?",
                (self.snapshot_id, cid),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["payload"]) | {"snapshot_id": self.snapshot_id}

    def provision(self, pid):
        with connect(self.s) as db:
            row = db.execute(
                "SELECT p.payload FROM provisions p JOIN chunks c ON c.provision_id=p.id JOIN snapshot_chunks sc ON sc.chunk_id=c.id WHERE sc.snapshot_id=? AND p.id=? LIMIT 1",
                (self.snapshot_id, pid),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def get_provision(self, source_type, locator):
        if source_type not in SOURCES:
            return {"status": "invalid_arguments", "message": "source_type 无效", "evidence": []}
        import re

        if source_type == "examination_guidelines" and not re.fullmatch(
            r"part:\d+/chapter:\d+/section:\d+(?:\.\d+)*", locator
        ):
            return {
                "status": "clarify",
                "message": "审查指南请指定部分、章、节，例如 part:2/chapter:4/section:3.2.1",
                "evidence": [],
            }
        if source_type != "examination_guidelines":
            if source_type in ("patent_law", "implementing_rules"):
                pattern = r"article:\d+"
            else:
                unit = (
                    "rule"
                    if source_type.endswith("_regulations")
                    else "section"
                    if source_type.endswith("_instructions")
                    else "article"
                )
                pattern = (
                    unit + r":\d+(?:bis|ter|quater|quinquies|sexies|septies)?|preamble|appendix:page:\d+"
                )
            if not re.fullmatch(pattern, locator):
                return {
                    "status": "clarify",
                    "message": "请按资料使用 article:4bis、rule:13bis、section:101 或 appendix:page:70。",
                    "evidence": [],
                }
        with connect(self.s) as db:
            rows = db.execute(
                "SELECT DISTINCT c.payload FROM provisions p JOIN chunks c ON c.provision_id=p.id JOIN snapshot_chunks sc ON sc.chunk_id=c.id WHERE sc.snapshot_id=? AND p.source_type=? AND (p.locator=? OR p.locator LIKE ?) ORDER BY p.locator,c.id",
                (
                    self.snapshot_id,
                    source_type,
                    locator,
                    locator + ".%" if source_type == "examination_guidelines" else "__no_descendants__",
                ),
            ).fetchall()
        evidence = [json.loads(r["payload"]) | {"snapshot_id": self.snapshot_id} for r in rows]
        evidence.sort(key=lambda c: (c["page_start"], c["locator"], c["char_start"]))
        return {
            "status": "found" if evidence else "not_found",
            "evidence": evidence,
            "total_chunks": len(evidence),
        }

    def section_outline(self, evidence, limit=32):
        """Snapshot-scoped headings help locate missing sibling sections; they are not evidence."""
        scopes = list(
            dict.fromkeys(
                e["locator"].split("/section:")[0]
                + "/section:"
                + e["locator"].split("/section:")[1].split(".")[0]
                for e in evidence
                if e["source_type"] == "examination_guidelines" and "/section:" in e["locator"]
            )
        )
        outline = []
        with connect(self.s) as db:
            for scope in scopes:
                remaining = limit - len(outline)
                if remaining <= 0:
                    break
                rows = db.execute(
                    "SELECT DISTINCT p.locator, p.payload FROM provisions p "
                    "JOIN chunks c ON c.provision_id=p.id "
                    "JOIN snapshot_chunks sc ON sc.chunk_id=c.id "
                    "WHERE sc.snapshot_id=? AND p.source_type='examination_guidelines' "
                    "AND (p.locator=? OR p.locator LIKE ?) ORDER BY p.locator LIMIT ?",
                    (self.snapshot_id, scope, scope + ".%", remaining),
                ).fetchall()
                outline.extend(
                    {
                        "source_type": "examination_guidelines",
                        "locator": row["locator"],
                        "heading": json.loads(row["payload"])["heading"][:200],
                    }
                    for row in rows
                )
        return outline

    def assert_embedding_config(self):
        if json.loads(self.snapshot["payload"]).get("embedding") != embedding_identity(self.s):
            raise ValueError("向量模型配置与快照不一致，请恢复配置或重新构建索引")

    def search_knowledge(self, query, source_types=None, timeout=None):
        self.assert_embedding_config()
        types = source_types or list(SOURCES)
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > self.s.question_max_chars
            or any(t not in SOURCES for t in types)
        ):
            return {"status": "invalid_arguments", "evidence": []}
        vector = self.emb.encode([query], timeout=timeout)[0]
        available = {d["source_type"] for d in json.loads(self.snapshot["payload"]).get("documents", [])}
        types = [t for t in types if t in available]
        candidates = []
        for typ in dict.fromkeys(types):
            found = self.collection.query(
                query_embeddings=[vector],
                where={"source_type": typ},
                n_results=self.s.retrieval_per_source_k,
                include=["distances"],
            )
            for cid, dist in zip(found["ids"][0], found["distances"][0]):
                c = self.chunk(cid)
                if c:
                    candidates.append(c | {"distance": dist})
        candidates.sort(key=lambda c: c["distance"])
        selected = []
        # Reserve a slot only for a small, explicitly requested set. For a broad
        # search, source declaration order must never hide later treaties.
        if source_types and len(set(types)) <= self.s.retrieval_top_k:
            for typ in dict.fromkeys(types):
                best = next((c for c in candidates if c["source_type"] == typ), None)
                if best:
                    selected.append(best)
        for c in candidates:
            if c["id"] not in {x["id"] for x in selected}:
                selected.append(c)
        return {
            "status": "found" if selected else "not_found",
            "evidence": selected[: self.s.retrieval_top_k],
            "section_outline": self.section_outline(selected[: self.s.retrieval_top_k]),
        }
