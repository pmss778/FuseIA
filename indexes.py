"""Index-name notation helpers for FuseIA (mirrors the legacy conventions).

The canonical index name is ``table(column)`` — single-column in this workflow.
Composite names like ``table(a,b)`` are tolerated by the parsers so a later
Consolidator stage can be added without changing the plumbing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_NAME_RE = re.compile(r"^[A-Za-z_][\w]*\((?:[A-Za-z_][\w]*,?)+\)$")


def normalize_indexes(indexes: Sequence[str]) -> List[str]:
    """Ordered, duplicate-free, lowercased index names."""

    seen: set = set()
    out: List[str] = []
    for item in indexes or []:
        text = str(item).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def index_components(index: str) -> Tuple[str, List[str]]:
    table, _, rest = index.partition("(")
    columns = [c.strip() for c in rest.rstrip(")").split(",") if c.strip()]
    return table.strip(), columns


def indexes_to_output(indexes: Sequence[str]) -> str:
    return json.dumps(normalize_indexes(indexes))


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the first balanced JSON object from a completion."""

    text = str(raw or "").strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def parse_index_list(raw: str, key: str) -> List[str]:
    """Parse a JSON object holding a list of index names under ``key``.

    Malformed or off-notation entries are dropped; grounding against the legal
    list is the caller's job.
    """

    obj = parse_json_object(raw)
    if obj is None:
        return []
    value = obj.get(key)
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item).strip().lower()
        if _NAME_RE.match(text) and text not in out:
            out.append(text)
    return out


def parse_single_index(raw: str, key: str) -> Optional[str]:
    obj = parse_json_object(raw)
    if obj is None:
        return None
    text = str(obj.get(key) or "").strip().lower()
    return text if _NAME_RE.match(text) else None
