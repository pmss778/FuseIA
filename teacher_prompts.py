"""Teacher prompt rendering for Proposer and Arbiter recommendation decisions.

The renderer provides workload, state, budget, and candidate evidence to the
role-specific policies while preserving the structured teacher output contract.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

SPECIALIST_SYSTEM_SUFFIX = "Obey the output contract exactly."
GATE_PCT = "1"


def _state_block(title: str, payload: Any) -> List[str]:
    return [title, json.dumps(payload, ensure_ascii=False, sort_keys=True)]


def _workload_block(queries: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = ["WORKLOAD SQL CONTEXT"]
    for query in queries or []:
        lines.append(f"-- query {query.get('query_id', '')}")
        lines.append(str(query.get("sql", "")))
    return lines


def _display_evidence(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Evidence projection shared by all roles: no utility, no cost."""

    keep = ("columns", "plan_node", "plan_rows", "rows_kept_fraction",
            "referenced_by_queries", "storage_mib", "n_distinct", "ndv_ratio",
            "null_frac", "table_rows", "source")
    return {k: item[k] for k in keep if k in item}


def _evidence_block(title: str, evidence: Optional[Mapping[str, Any]]) -> List[str]:
    if not evidence:
        return []
    payload = {name: _display_evidence(item) for name, item in evidence.items()
               if isinstance(item, Mapping)}
    return _state_block(title, payload) + [
        "table_rows = the table's row estimate; indexing a very small table rarely pays off. "
        "n_distinct / ndv_ratio = distinct values, absolute and as a fraction of the table; a "
        "low ndv_ratio means many repeated values. plan_rows / rows_kept_fraction = the planner's "
        "row estimate where the column is used; rows_kept_fraction = plan_rows / table_rows, "
        "so a HIGH value means weak filtering and a LOW value means strong filtering. null_frac "
        "= NULL fraction. clauses = the workload clauses the column appears in. "
        "No benefit or cost estimate is provided."
    ]


def render_selection(
    *,
    queries: Sequence[Mapping[str, Any]],
    current_indexes: Sequence[str],
    legal_candidates: Sequence[str],
    storage_left_mib: float,
    candidate_count: int,
    index_evidence: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Propose a ranked shortlist of at most ``candidate_count`` indexes.

    The legal list is a hard constraint, stated three ways; Selection sees
    cardinality and scale facts only.
    """

    forbidden_lines = (
        _state_block(
            "FORBIDDEN ANSWERS (already selected -- adding one again changes nothing)",
            list(current_indexes),
        )
        + [
            "Every name above is ALREADY in the selection: it is absent from LEGAL CANDIDATE "
            "INDEXES, and returning it is a contract violation that fails the run. Answer only "
            "with names from LEGAL CANDIDATE INDEXES; if a candidate you would like is already "
            "selected, return the best names that are still legal instead. An empty list is a "
            "valid answer when nothing legal is worth adding -- a name from this block never is.",
        ]
        if current_indexes
        else []
    )
    contract = (
        "ROLE OUTPUT CONTRACT\n"
        'Return one JSON object with exactly one key, "candidates", whose value is a list '
        "of index names ordered best first.\n"
        "Hard constraints:\n"
        f"1. Return AT MOST {int(candidate_count)} names. Return fewer -- including zero -- "
        "when fewer are genuinely worth adding; do NOT pad the list up to the cap.\n"
        "2. Every name MUST be copied verbatim from LEGAL CANDIDATE INDEXES. Do not "
        "reformat, reorder columns, invent, or transliterate a name. A name from CURRENT "
        "SELECTED INDEXES is NOT in the legal list, so returning one also breaks this rule.\n"
        "3. NEVER return a name that appears in CURRENT SELECTED INDEXES or FORBIDDEN ANSWERS. "
        "Those are already present and adding them changes nothing. If the candidates you "
        "would like are already selected, return the best names that are still legal instead. "
        'If NO legal candidate is worth adding, return an empty list: {"candidates": []}.\n'
        "4. Gate every candidate on estimated rows read per use, computed only from the "
        "evidence fields. Equality or join column: rows_per_use = table_rows / n_distinct. "
        "Range column: rows_per_use = table_rows * rows_kept_fraction. If rows_per_use "
        f"exceeds roughly {GATE_PCT} percent of table_rows, the planner will prefer a sequential "
        "scan and the index is worthless -- no matter how many queries reference the column. "
        "referenced_by_queries never rescues a candidate that fails this gate. Rank the "
        "survivors by smallest rows_per_use, then by referenced_by_queries.\n"
        "5. Join columns are a separate value class from filter columns. A column with "
        "join_query_count > 0 is a JOIN KEY: its index turns nested-loop joins into index "
        "lookups for every query that joins through it, and its worth is NOT measured by "
        "filter selectivity at all -- rows_kept_fraction and plan_rows are single-query "
        "plan artifacts and underestimate join keys by design. When a join key passes the "
        "gate (rows_per_use = table_rows / n_distinct is small), rank it among the top "
        "candidates: prefer the key with the highest join_query_count, then the largest "
        "table_rows. Do not drop a join key merely because another candidate has a smaller "
        "rows_per_use.\n"
        "6. Conviction over coverage: return the fewest candidates that cover the workload's "
        "dominant selective access paths. Every extra name must be justified by clearly "
        "better lookup selectivity or clearly wider selective reach; do NOT spend storage on "
        "weak indexes -- an index that barely helps still consumes budget and can leave the "
        "whole workload no better off.\n"
        "7. Prefer indexes that sit on large tables (see table_rows). Avoid indexing very small "
        "tables and columns with a very low ndv_ratio.\n"
        "Final check before answering: for each name you are about to return, confirm it is "
        "copied verbatim from LEGAL CANDIDATE INDEXES and is absent from CURRENT SELECTED "
        "INDEXES and FORBIDDEN ANSWERS. Drop any name that fails this check.\n"
        'Example shape: {"candidates": ["orders(o_custkey)", "lineitem(l_partkey)"]}'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the Selection Agent for a PostgreSQL index advisor. You choose which "
                "candidate indexes are worth adding to the current selection. "
                + SPECIALIST_SYSTEM_SUFFIX
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                _workload_block(queries)
                + _state_block("CURRENT SELECTED INDEXES", list(current_indexes))
                + _state_block("LEGAL CANDIDATE INDEXES (the only allowed answers)",
                               list(legal_candidates))
                + _state_block("STORAGE LEFT (MiB)", round(float(storage_left_mib), 3))
                + _evidence_block("CANDIDATE EVIDENCE (per candidate; cardinality and scale "
                                  "facts only)", index_evidence)
                + forbidden_lines
                + [contract]
            ),
        },
    ]





def render_arbiter_stop(*, queries, current_indexes, candidate_indexes,
                        storage_left_mib, index_evidence=None):
    """Render the Arbiter's add-one-index-or-STOP decision prompt."""
    evidence_block = []
    if index_evidence:
        payload = {name: _display_evidence(item)
                   for name, item in index_evidence.items()
                   if isinstance(item, Mapping)}
        evidence_block = _state_block(
            "CANDIDATE EVIDENCE (per candidate; cardinality and scale facts only)",
            payload,
        ) + [
            "table_rows is the table row estimate; n_distinct and ndv_ratio describe "
            "distinct values. For equality or join columns, rows_per_use is approximately "
            "table_rows / n_distinct; for range columns, use table_rows * "
            "rows_kept_fraction. Lower rows_per_use indicates stronger filtering."
        ]
    contract = (
        "ROLE OUTPUT CONTRACT\n"
        'Return exactly one JSON object and nothing else, with the single key "winner". '
        "Do not write a reason.\n"
        '"winner" must be copied verbatim from CANDIDATE INDEXES or be the literal "STOP".\n'
        'Choose "STOP" when no candidate is clearly worth its storage, or when the '
        "remaining budget is better kept for later candidates. An empty decision is valid.\n"
        "Prefer candidates with lower estimated rows_per_use, then higher workload "
        "coverage among those that pass the selectivity gate.\n"
        'Example: {"winner": "orders(o_orderkey)"} or {"winner": "STOP"}'
    )
    content = (
        _workload_block(queries)
        + _state_block("CURRENT SELECTED INDEXES", list(current_indexes))
        + _state_block("CANDIDATE INDEXES", list(candidate_indexes))
        + _state_block("STORAGE LEFT (MiB)", round(float(storage_left_mib), 3))
        + evidence_block
        + [contract]
    )
    return [
        {"role": "system", "content": (
            "You are the Index Arbiter for a PostgreSQL index advisor. Pick the "
            "single best candidate to add now, or STOP when adding nothing is best. "
            "Obey the output contract exactly.")},
        {"role": "user", "content": "\n".join(content)},
    ]

