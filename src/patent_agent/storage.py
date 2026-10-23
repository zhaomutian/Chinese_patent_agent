import json
import sqlite3
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS provisions (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, source_type TEXT NOT NULL, locator TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, provision_id TEXT NOT NULL, source_type TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, collection TEXT NOT NULL, status TEXT NOT NULL, fingerprint TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snapshot_chunks (snapshot_id TEXT NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY(snapshot_id, chunk_id));
CREATE TABLE IF NOT EXISTS ingestion_runs (id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, status TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0, error TEXT);
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS provision_lookup ON provisions(source_type,locator);
CREATE INDEX IF NOT EXISTS chunk_parent ON chunks(provision_id);
"""


@contextmanager
def connect(s):
    s.catalog_db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(s.catalog_db_path, timeout=20)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    try:
        yield db
        db.commit()
    finally:
        db.close()


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def active_snapshot(s):
    with connect(s) as db:
        row = db.execute(
            "SELECT * FROM snapshots WHERE status='active' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None
