"""Local metadata-only run log. No prompts, model payloads, keys or provider URLs."""

import json
from datetime import datetime, timezone


def record_run(settings, result):
    failed = result["answer"]["status"] == "limited" and any(e.get("error") for e in result["trace"])
    if settings.log_level.upper() in ("WARNING", "ERROR", "CRITICAL") and not failed:
        return
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "at": datetime.now(timezone.utc).isoformat(),
        "level": "ERROR" if failed else "INFO",
        "turn_id": result["turn_id"],
        "session_id": result["session_id"],
        "snapshot_id": result["snapshot_id"],
        "status": result["answer"]["status"],
        "metrics": result["metrics"],
        "nodes": [e["node"] for e in result["trace"]],
        "errors": [e["error"] for e in result["trace"] if e.get("error")],
    }
    with (settings.log_dir / "runs.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
