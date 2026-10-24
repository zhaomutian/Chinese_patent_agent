"""Download pinned official PDFs from the source manifest; reject unreviewed replacements."""

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

from patent_agent.models import safe_error
from patent_agent.settings import ROOT

ALLOWED_HOSTS = {"www.wipo.int", "wipolex-res.wipo.int", "www.wto.org"}


def check_url(url):
    p = urlparse(url)
    if p.scheme != "https" or p.hostname not in ALLOWED_HOSTS or p.username or p.password:
        raise ValueError("只允许清单中WIPO/WTO官方HTTPS来源")


def fetch(client, url):
    # Validate every redirect before sending another request.
    from urllib.parse import urljoin

    for _ in range(6):
        check_url(url)
        response = client.get(url)
        if response.is_redirect:
            url = urljoin(url, response.headers["location"])
            continue
        response.raise_for_status()
        return response
    raise ValueError("重定向次数过多")


def main():
    parser = argparse.ArgumentParser(description="从官方来源恢复已登记且哈希固定的国际条约PDF")
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/sources.manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    with httpx.Client(timeout=60) as client:
        for doc in manifest["documents"]:
            if doc.get("parser") != "official_treaties":
                continue
            target = (ROOT / doc["raw_path"]).resolve()
            if not target.is_relative_to((ROOT / "data/raw").resolve()):
                raise ValueError("目标不在原始资料目录")
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != doc["sha256"]:
                    raise ValueError("已有文件哈希异常: " + doc["source_type"])
                print(doc["source_type"] + ": verified existing", flush=True)
                continue
            response = fetch(client, doc["download_url"])
            if not response.content.startswith(b"%PDF"):
                match = re.search(r"""(?:src|href)=["']([^"']+\.pdf[^"']*)""", response.text)
                if not match:
                    raise ValueError("官方详情页未找到PDF: " + doc["source_type"])
                response = fetch(client, html.unescape(match[1]))
            data = response.content
            if not data.startswith(b"%PDF") or hashlib.sha256(data).hexdigest() != doc["sha256"]:
                raise ValueError("官方文件已变化或下载异常，需核验新版后更新清单: " + doc["source_type"])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".download")
            temporary.write_bytes(data)
            temporary.replace(target)
            print(doc["source_type"] + ": downloaded and verified", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Do not print temporary signed WIPO resource URLs or provider responses.
        print(
            json.dumps(
                {"ok": False, "error": str(exc) if isinstance(exc, ValueError) else safe_error(exc)},
                ensure_ascii=False,
            )
        )
        raise SystemExit(1) from None
