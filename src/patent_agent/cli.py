import argparse
import json

from .models import safe_error
from .settings import Settings


def main():
    parser = argparse.ArgumentParser(description="中国专利法知识库智能体")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--live", action="store_true")
    p = sub.add_parser("parse")
    p.add_argument("--reextract", action="store_true")
    sub.add_parser("index")
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--session")
    p.add_argument("--mode", choices=["auto", "lookup", "analyze"], default="auto")
    p.add_argument("--explanation", choices=["hint", "concise", "detailed"])
    args = parser.parse_args()
    try:
        s = Settings()
        if args.command == "doctor":
            from .doctor import doctor

            result = doctor(s, args.live)
        elif args.command == "parse":
            from .ingest import parse_all

            full = parse_all(s, args.reextract)
            result = {k: v for k, v in full.items() if k != "documents"} | {
                "documents": [
                    {
                        k: d[k]
                        for k in ("source_type", "pages", "provisions", "chunks", "body_pages", "text_chars")
                    }
                    for d in full["documents"]
                ]
            }
        elif args.command == "index":
            from .retrieval import build_index

            result = build_index(s)
        elif args.command == "ask":
            from .agent import AgentService

            with AgentService(s) as svc:
                sid = args.session or svc.create_session()["session_id"]
                result = svc.ask(sid, args.question, args.mode, args.explanation)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("ok") is False:
            raise SystemExit(1)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": safe_error(exc)}, ensure_ascii=False))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
