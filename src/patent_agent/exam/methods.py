"""Immutable method packages. Online analysis never opens the source question/answer file."""

import hashlib
import json
import re
from pathlib import Path

from .models import MethodCard


def package_digest(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def load_package(directory: Path, version=None):
    if version is None:
        pointer = directory / "active.json"
        if not pointer.exists():
            return {"version": "builtin", "cards": [], "chapters": []}
        version = json.loads(pointer.read_text())["version"]
    if version == "builtin":
        return {"version": "builtin", "cards": [], "chapters": []}
    if not re.fullmatch(r"[a-f0-9]{24}", version):
        raise ValueError("方法包版本无效")
    obj = json.loads((directory / (version + ".json")).read_text())
    if package_digest(obj["payload"]) != version:
        raise ValueError("方法包内容与版本不符")
    for card in obj["payload"]["cards"]:
        MethodCard.model_validate(card)
    return {"version": version, **obj["payload"]}


def select_methods(package, chapter_ids, max_methods, max_chars):
    selected = []
    chars = 0
    cards = {c["chapter_id"]: c for c in package["cards"]}
    for chapter in dict.fromkeys(chapter_ids):
        if chapter not in cards or len(selected) >= max_methods:
            continue
        card = cards[chapter]
        size = len(json.dumps(card, ensure_ascii=False))
        if chars + size <= max_chars:
            selected.append(card)
            chars += size
    return selected


BUILTIN_INTERPRETATION = [
    {
        "method_id": "general_scope",
        "analysis_types": ["subsumption"],
        "title": "概括规范的适用范围",
        "applicable_when": "判断某主体或具体行为是否属于概括性规定的范围。",
        "checklist": [
            "提取规范的主体、行为、条件和效果。",
            "说明具体行为与概括用语的对应关系。",
            "检查是否存在特别规定或例外。",
        ],
        "pitfalls": ["不能仅因未逐字列举阶段就排除适用。", "不得由原则直接推出具体处罚或扩大责任主体。"],
        "lookup_triggers": ["归类需要额外定义或法律拟制。", "选项涉及具体法律后果但原文未规定。"],
        "source_question_ids": [],
    },
    {
        "method_id": "systematic_rules",
        "analysis_types": ["systematic"],
        "title": "一般规定与特别规定的关系",
        "applicable_when": "多条规范共同决定概念范围、条件或后果。",
        "checklist": [
            "分别列明相关规则与适用对象。",
            "核对定义、一般规则和特别规定之间的关系。",
            "逐项核对版本、条件及例外，说明共同适用的理由。",
        ],
        "pitfalls": ["引用多个条文不等于已经建立适用关系。", "不得用抽象原则绕过明确特别规定。"],
        "lookup_triggers": ["欠缺连接两条规范的必要定义。", "条文存在可能改变结论的冲突。"],
        "source_question_ids": [],
    },
    {
        "method_id": "conditions_and_effects",
        "analysis_types": ["condition_application"],
        "title": "条件、例外与法律后果",
        "applicable_when": "依据题设事实判断规范条件与例外是否成立。",
        "checklist": [
            "列出规范必要条件及明确后果。",
            "把题设事实与每项条件对应，不补写未知事实。",
            "核查例外是否成立及是否需要额外规范。",
        ],
        "pitfalls": ["未知条件不等于条件不成立。", "行为要求不能直接替代法律责任条款。"],
        "lookup_triggers": ["关键条件的法律含义不明。", "例外或后果缺少原文依据。"],
        "source_question_ids": [],
    },
]


def load_interpretation_package(directory, version=None):
    from .models import InterpretationCard

    if version is None:
        pointer = directory / "active.json"
        version = (
            json.loads(pointer.read_text())["version"] if pointer.exists() else "builtin-interpretation-v1"
        )
    if version == "builtin-interpretation-v1":
        return {"version": version, "cards": BUILTIN_INTERPRETATION}
    if not re.fullmatch(r"[a-f0-9]{24}", version):
        raise ValueError("解释方法包版本无效")
    payload = json.loads((directory / (version + ".json")).read_text())["payload"]
    if package_digest(payload) != version:
        raise ValueError("解释方法包内容与版本不符")
    ids = []
    for c in payload["cards"]:
        card = InterpretationCard.model_validate(c)
        ids.append(card.method_id)
    if len(ids) != len(set(ids)):
        raise ValueError("解释方法卡ID重复")
    return {"version": version, **payload}


def select_interpretation_method(package, analysis_types, max_chars):
    for kind in dict.fromkeys(analysis_types):
        if kind == "direct":
            continue
        for card in package["cards"]:
            if kind in card["analysis_types"] and len(json.dumps(card, ensure_ascii=False)) <= max_chars:
                return card
    return None
