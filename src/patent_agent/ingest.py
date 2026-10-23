"""Parser for the three supplied PDFs; unsupported layouts require new parser validation."""

import hashlib
import json
import re
import subprocess
from collections import Counter

from .settings import ROOT
from .storage import connect, dump

PARSER_VERSION = "supplied-pdfs-v2"
CN = "零一二三四五六七八九十百千"
HEADER = re.compile(r"专利审查指南第([" + CN + r"]+)部分第([" + CN + r"]+)章")
ARTICLE = re.compile(r"^第([" + CN + r"]+)条(.*)$")
SECTION = re.compile(r"^(\d+(?:\.\d+)*)(?:\.\s*|\s+)(\D.{0,65})$")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def cn_number(value):
    if value.isdigit():
        return int(value)
    total = current = 0
    for c in value:
        if c in "十百千":
            total += (current or 1) * {"十": 10, "百": 100, "千": 1000}[c]
            current = 0
        else:
            current = CN.index(c)
    return total + current


def normalize(text):
    # PDF letter spacing is removed around Chinese; preserve English word boundaries.
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[\u3000-\u9fff\uff00-\uffef]) +| +(?=[\u3000-\u9fff\uff00-\uffef])", "", text)
    return text


def load_pages(doc, reextract=False):
    raw = ROOT / doc["raw_path"]
    if hashlib.sha256(raw.read_bytes()).hexdigest() != doc["sha256"]:
        raise ValueError("原始文件哈希与 manifest 不一致: " + doc["source_type"])
    target = ROOT / doc["intake_pages_path"]
    if reextract or not target.exists():
        result = subprocess.run(["pdftotext", "-layout", str(raw), "-"], capture_output=True, check=True)
        pages = result.stdout.decode().split("\f")
        if not pages[-1].strip():
            pages.pop()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "\n".join(dump({"page_index": i + 1, "text": t}) for i, t in enumerate(pages)) + "\n"
        )
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    if len(rows) != doc["page_count"]:
        raise ValueError("页数与 manifest 不一致")
    return rows


def parse_document(doc, pages):
    provisions, excluded = [], []
    current = None
    chapter = ""
    path = ""
    section_titles = {}
    toc_titles = set()
    for pg in pages:
        for ln in pg["text"].splitlines():
            m = re.match(r"^\s*[第一二三四五六部分]?\s*(\d+(?:\.\d+)*)(?:\.)?\s+(.+?)\.{3,}", ln)
            if m:
                toc_titles.add((m[1], normalize(m[2])))

    def start(locator, heading, page):
        nonlocal current
        current = {
            "document_id": doc["document_id"],
            "source_type": doc["source_type"],
            "locator": locator,
            "heading": heading,
            "chapter": chapter,
            "title": doc["title"],
            "version_label": doc["version_label"],
            "segments": [],
            "page_start": page,
            "page_end": page,
            "coverage_warnings": [],
            "printed_page_labels": {},
        }
        provisions.append(current)

    for page in pages:
        p = page["page_index"]
        lines = page["text"].splitlines()
        guide = doc["source_type"] == "examination_guidelines"
        if guide:
            header = HEADER.search(re.sub(r"\s+", "", page["text"][:250]))
            if not header:
                excluded.append(
                    {
                        "page": p,
                        "reason": "blank" if not page["text"].strip() else "front_matter_toc_or_index",
                    }
                )
                continue
            new_path = f"part:{cn_number(header[1])}/chapter:{cn_number(header[2])}"
            if new_path != path:
                path = new_path
                current = None
                section_titles = {}
            # Running header is first nonempty line. Printed page numbers are separate from physical pages.
            first = next(i for i, line in enumerate(lines) if line.strip())
            lines = lines[first + 1 :]
        printed_label = (
            next(
                (normalize(ln) for ln in lines[-8:] if re.search(r"[（(]\d+[－-]\d+[）)]", normalize(ln))),
                None,
            )
            if guide
            else str(p)
        )
        for raw_line in lines:
            if guide:
                # Side references have their own left column separated by at least two spaces.
                raw_line = re.sub(
                    r"^\s*(?:法|细\s*则|条\s*约|PCT|规\s*则)\s*[\d.、及和之\s]+?\s{2,}(?=\S)", "", raw_line
                )
            line = normalize(raw_line)
            if not line:
                continue
            if guide and (
                re.fullmatch(r"(?:\d+)?[（(]\d+[－-]\d+[）)](?:\d+)?|\d+", line)
                or re.fullmatch(r"(?:法|细则|条约|PCT|规则)[\d.、及和之\s]+", line)
            ):
                continue
            if not guide and (
                line.startswith("中华人民共和国专利法") or re.fullmatch(r"\d+\s*/\s*\d+|第\d+页共\d+页", line)
            ):
                continue
            if re.match(r"^第[" + CN + r"]+章", line):
                chapter = line
                continue
            if doc["source_type"] == "implementing_rules" and line.startswith("法规标题"):
                break
            if guide:
                match = SECTION.match(raw_line.strip())
                if match and any(
                    n == match[1]
                    and (
                        title == normalize(match[2])
                        or (len(normalize(match[2])) >= 6 and title.startswith(normalize(match[2])))
                    )
                    for n, title in toc_titles
                ):
                    locator_num = match[1]
                    heading = normalize(match[2])
                    section_titles[locator_num] = heading
                    parents = [
                        ".".join(locator_num.split(".")[:i])
                        for i in range(1, len(locator_num.split(".")) + 1)
                    ]
                    title_path = " / ".join(
                        f"{n} {section_titles[n]}" for n in parents if n in section_titles
                    )
                    start(path + "/section:" + locator_num, title_path, p)
                if current is None:
                    start(path + "/preamble", chapter or path, p)
            else:
                match = re.match(r"^第([" + CN + r"]+)条\s+(.*)$", raw_line.strip())
                if match:
                    start("article:" + str(cn_number(match[1])), "第" + match[1] + "条", p)
            if current is not None:
                current["segments"].append({"page": p, "text": line})
                current["page_end"] = p
                current["printed_page_labels"][str(p)] = printed_label
                if guide and p in (322, 323, 324, 325, 327, 328):
                    current["coverage_warnings"] = [
                        "涉及化学结构图；文字提取不能完整表达图形，请查看原 PDF。"
                    ]
    seen = set()
    for provision in provisions:
        if provision["locator"] in seen:
            raise ValueError("重复层级定位: " + doc["source_type"] + " " + provision["locator"])
        seen.add(provision["locator"])
        provision["text"] = "\n".join(seg["text"] for seg in provision["segments"])
        provision["id"] = "p_" + digest(doc["sha256"] + PARSER_VERSION + dump(provision))[:24]
    return provisions, excluded


def chunk_provision(p, s):
    text = p["text"]
    spans, offset = [], 0
    for seg in p["segments"]:
        spans.append((offset, offset + len(seg["text"]), seg["page"]))
        offset += len(seg["text"]) + 1
    start = 0
    while start < len(text):
        end = min(start + s.chunk_target_chars, len(text))
        if end < len(text):
            # Prefer complete Chinese sentences within the maximum size.
            stops = [
                i + 1 for i in range(end, min(start + s.chunk_max_chars, len(text))) if text[i] in "。；\n"
            ]
            if stops:
                end = stops[0]
        pages = sorted({pnum for a, b, pnum in spans if b > start and a < end})
        body = text[start:end]
        yield {
            k: p[k]
            for k in (
                "document_id",
                "source_type",
                "locator",
                "heading",
                "chapter",
                "title",
                "version_label",
                "coverage_warnings",
            )
        } | {
            "id": "c_" + digest(p["id"] + str(start) + body)[:24],
            "provision_id": p["id"],
            "text": body,
            "char_start": start,
            "char_end": end,
            "pages": pages,
            "page_start": min(pages),
            "page_end": max(pages),
        }
        if end == len(text):
            break
        start = end - s.chunk_overlap_chars


def parse_all(s, reextract=False):
    manifest = json.loads(s.sources_manifest_path.read_text())
    reports, all_chunks, all_provisions = [], [], []
    for doc in manifest["documents"]:
        pages = load_pages(doc, reextract)
        provisions, excluded = parse_document(doc, pages)
        if doc["source_type"] in ("patent_law", "implementing_rules"):
            expected = 82 if doc["source_type"] == "patent_law" else 148
            if [p["locator"] for p in provisions] != [f"article:{i}" for i in range(1, expected + 1)]:
                raise ValueError("条号完整性检查失败: " + doc["source_type"])
        chunks = [c for p in provisions for c in chunk_provision(p, s)]
        all_provisions.extend(provisions)
        all_chunks.extend(chunks)
        with connect(s) as db:
            db.execute("INSERT OR REPLACE INTO documents VALUES (?,?)", (doc["document_id"], dump(doc)))
            db.executemany(
                "INSERT OR REPLACE INTO provisions VALUES (?,?,?,?,?)",
                [(p["id"], p["document_id"], p["source_type"], p["locator"], dump(p)) for p in provisions],
            )
            db.executemany(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?)",
                [(c["id"], c["provision_id"], c["source_type"], dump(c)) for c in chunks],
            )
        reports.append(
            {
                "source_type": doc["source_type"],
                "pages": len(pages),
                "provisions": len(provisions),
                "chunks": len(chunks),
                "body_pages": len({seg["page"] for p in provisions for seg in p["segments"]}),
                "excluded_pages": excluded,
                "chapters": dict(
                    Counter(p["locator"].split("/section:")[0].split("/preamble")[0] for p in provisions)
                )
                if doc["source_type"] == "examination_guidelines"
                else {},
                "text_chars": sum(len(p["text"]) for p in provisions),
            }
        )
    s.processed_data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("provisions", all_provisions), ("chunks", all_chunks)]:
        (s.processed_data_dir / f"{name}.jsonl").write_text("\n".join(dump(row) for row in rows) + "\n")
    report = {
        "parser_version": PARSER_VERSION,
        "documents": reports,
        "total_chunks": len(all_chunks),
        "coverage": "文字正文；未恢复图片、复杂表格和公式语义。原始 PDF 可按物理页查阅。未核验现行有效性。",
    }
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/parsing.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
