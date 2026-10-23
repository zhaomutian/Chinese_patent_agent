"""Record a completed milestone without claiming unverified acceptance."""

import json
import sys
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
phase, status, detail = sys.argv[1:4]
p = root / "reports/phases.json"
rows = (
    json.loads(p.read_text())
    if p.exists()
    else {f"P{i}": {"status": "待开发", "detail": ""} for i in range(6)}
)
now = datetime.now().astimezone().isoformat(timespec="seconds")
rows[phase] = {"status": status, "detail": detail, "updated_at": now}
p.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
with (root / "reports/progress.jsonl").open("a") as f:
    f.write(
        json.dumps({"time": now, "phase": phase, "status": status, "detail": detail}, ensure_ascii=False)
        + "\n"
    )
names = ["环境与接口", "资料解析", "检索与快照", "LangGraph", "API 与页面", "测试与交付"]
text = (
    "# 开发阶段状态\n\n更新时间："
    + now
    + "。状态以实际验证为准。\n\n| 阶段 | 状态 | 交付与验证 |\n|---|---|---|\n"
)
for i, name in enumerate(names):
    row = rows[f"P{i}"]
    text += f"| P{i} {name} | {row['status']} | {row['detail']} |\n"
text += "\n详细过程：[progress.jsonl](reports/progress.jsonl)。各阶段报告位于 reports/。\n"
(root / "STATUS.md").write_text(text)
