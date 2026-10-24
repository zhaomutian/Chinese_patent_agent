"""Parse reviewed official treaty PDFs without treating contents pages as provisions."""

import re
from copy import deepcopy

from .ingest import cn_number, digest, normalize
from .storage import dump

TREATY_PARSER_VERSION = "official-treaties-v1"
SUFFIXES = {"二": "bis", "三": "ter", "四": "quater", "五": "quinquies", "六": "sexies", "七": "septies"}
CN_HEADING = re.compile(r"^第([零一二三四五六七八九十百千\d]+)条(?:之([二三四五六七]))?(.*)$")
EN_HEADING = re.compile(r"^(Article|Rule|Section)\s+(\d+(?:bis|ter|quater|quinquies|sexies|septies)?)(.*)$")


def heading(line, config):
    match = CN_HEADING.fullmatch(line)
    if match:
        tail = match[3].strip()
        if tail and not tail.isdigit() and not config.get("inline_titles"):
            return None
        return config["unit"] + ":" + str(cn_number(match[1])) + SUFFIXES.get(match[2], ""), line
    match = EN_HEADING.fullmatch(line)
    if match:
        tail = match[3].strip()
        expected = {"article": "Article", "rule": "Rule", "section": "Section"}[config["unit"]]
        if match[1] != expected or (tail and not tail.startswith((":", "[Deleted]"))):
            return None
        return config["unit"] + ":" + match[2], line
    return None


def parse_treaty(doc, pages):
    config = doc["parser_config"]
    provisions, excluded = [], []
    current = None
    start_page, end_page = config["body_start_page"], config.get("body_end_page", doc["page_count"])
    appendices = config.get("appendix_start_page", doc["page_count"] + 1)
    started = not config.get("start_at")

    def start(locator, title, page):
        nonlocal current
        current = {
            "document_id": doc["document_id"],
            "source_type": doc["source_type"],
            "locator": locator,
            "heading": title,
            "chapter": "",
            "title": doc["title"],
            "version_label": doc["version_label"],
            "source_url": doc["source_url"],
            "language": doc["language"],
            "effective_from": doc.get("effective_from"),
            "legal_status_note": doc.get("legal_status_note", ""),
            "segments": [],
            "page_start": page,
            "page_end": page,
            "coverage_warnings": list(doc.get("warnings", [])),
            "printed_page_labels": {},
        }
        provisions.append(current)

    for pg in pages:
        page = pg["page_index"]
        if not start_page <= page <= end_page:
            excluded.append({"page": page, "reason": "reviewed_front_matter_toc_or_duplicate"})
            continue
        if page >= appendices:
            start(f"appendix:page:{page}", f"附件 / 费用表 · PDF 第{page}页", page)
            current["coverage_warnings"].append(
                "附件按页保留；表格、代码和图示请对照原始 PDF，不从文字顺序推断表格对应关系。"
            )
        for raw in pg["text"].splitlines():
            line = normalize(raw)
            if not started:
                if not re.search(config["start_at"], line):
                    continue
                started = True
            if not line or re.fullmatch(r"\d+|第\d+页|PCT/AI/\d+", line):
                continue
            if any(re.fullmatch(pattern, line) for pattern in config.get("skip_lines", [])):
                continue
            deleted_range = re.fullmatch(r"第(\d+)条至第(\d+)条", line) if page < appendices else None
            if deleted_range:
                first, last = map(int, deleted_range.groups())
                if not 0 < last - first < 20:
                    raise ValueError("未核验的条文范围")
                start(f"{config['unit']}:{first}", line, page)
                current["alias_locators"] = [f"{config['unit']}:{n}" for n in range(first + 1, last + 1)]
            found = heading(line, config) if page < appendices and config.get("headings", True) else None
            if found:
                start(*found, page)
            if current is None:
                start("preamble", "前言 / 总则", page)
            current["segments"].append({"page": page, "text": line})
            current["page_end"] = page
            # Physical PDF pages are reliable; do not guess printed numbering.
            current["printed_page_labels"][str(page)] = None
    expanded = []
    for provision in provisions:
        aliases = provision.pop("alias_locators", [])
        expanded.append(provision)
        for locator in aliases:
            duplicate = deepcopy(provision)
            duplicate["locator"] = locator
            expanded.append(duplicate)
    provisions = expanded
    seen = set()
    for p in provisions:
        if p["locator"] in seen:
            raise ValueError(f"重复条约条款: {doc['source_type']} {p['locator']}")
        seen.add(p["locator"])
        p["text"] = "\n".join(s["text"] for s in p["segments"])
        p["id"] = "p_" + digest(doc["sha256"] + TREATY_PARSER_VERSION + dump(p))[:24]
    required = set(config.get("required_locators", []))
    if not required <= seen:
        raise ValueError(f"条约条号缺失: {doc['source_type']} {sorted(required - seen)}")
    if not provisions or any(not p["text"] for p in provisions):
        raise ValueError("条约正文为空")
    return provisions, excluded
