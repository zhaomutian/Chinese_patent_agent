import json
import re

from patent_agent.ingest import chunk_provision, cn_number, load_pages, parse_document
from patent_agent.settings import Settings


def test_real_corpus_structure_and_chunk_provenance():
    s = Settings()
    manifest = json.loads(s.sources_manifest_path.read_text())
    for doc in manifest["documents"]:
        pages = load_pages(doc)
        provisions, excluded = parse_document(doc, pages)
        assert len({p["locator"] for p in provisions}) == len(provisions)
        if doc["source_type"] != "examination_guidelines":
            n = 82 if doc["source_type"] == "patent_law" else 148
            assert [p["locator"] for p in provisions] == [f"article:{i}" for i in range(1, n + 1)]
        else:
            assert len({p["locator"].split("/section:")[0].split("/preamble")[0] for p in provisions}) == 38
            assert all(p["page_start"] < 587 for p in provisions)
            assert any(p["locator"] == "part:2/chapter:4/section:3.2.1.2" for p in provisions)
            assert {x["page"] for x in excluded}.isdisjoint(
                {seg["page"] for p in provisions for seg in p["segments"]}
            )
        for p in provisions:
            if doc["source_type"] == "examination_guidelines":
                assert not any(
                    re.fullmatch(r"(?:\d+)?[（(]\d+-\d+[）)](?:\d+)?", seg["text"]) for seg in p["segments"]
                )
            chunks = list(chunk_provision(p, s))
            assert chunks[0]["char_start"] == 0 and chunks[-1]["char_end"] == len(p["text"])
            for c in chunks:
                assert len(c["text"]) <= s.chunk_max_chars
                assert c["text"] == p["text"][c["char_start"] : c["char_end"]]
                assert all(1 <= page <= doc["page_count"] for page in c["pages"])


def test_chinese_number():
    assert cn_number("一百四十八") == 148
    assert cn_number("十二") == 12
