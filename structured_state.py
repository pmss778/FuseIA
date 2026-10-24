"""Structured state records used by local JOB manifest generation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from relational_graph import anonymize_index, anonymize_structure, build_relation_graph, make_rename_map
from structured_sql import parse_sql_structure, stable_id


def index_size(store, indexes: Sequence[str]) -> float:
    return round(float(store.estimate_storage_mib(list(indexes))) if indexes else 0.0, 3)


def make_state(queries: list[dict], current: list[str], legal: list[str], budget: float,
               store, seed: int, view: str = "original",
               extra_symbols: Sequence[str] | None = None) -> dict[str, Any]:
    structures = [parse_sql_structure(q["sql"]) for q in queries]
    mapping = (make_rename_map(structures, seed, extra_symbols)
               if view == "anonymous" else {})
    if view == "anonymous":
        structures = [anonymize_structure(s, mapping) for s in structures]
        current_view = [anonymize_index(x, mapping) for x in current]
        legal_view = [anonymize_index(x, mapping) for x in legal]
    else:
        current_view, legal_view = list(current), list(legal)
    used = index_size(store, current)
    graphs = [build_relation_graph(s) for s in structures]
    state_id = stable_id(str(seed), str(sorted(q["query_id"] for q in queries)), str(sorted(current)), str(budget))
    return {
        "state_id": state_id,
        "seed": seed,
        "view": view,
        "rename_map": mapping,
        "queries": queries,
        "structures": [s.to_dict() for s in structures],
        "graphs": graphs,
        "current_indexes": current_view,
        "legal_candidates": legal_view,
        "budget_mib": round(float(budget), 3),
        "storage_used_mib": used,
        "storage_left_mib": round(max(0.0, budget - used), 3),
        "candidate_count": len(legal_view),
    }
