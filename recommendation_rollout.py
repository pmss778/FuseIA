"""Shared text-policy rollout for teacher/base/SFT comparisons, without DDL."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from candidate_pool import OfflineStore
from teacher_prompts import render_arbiter_stop, render_selection


def load_queries(path: Path) -> list[dict]:
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        return [{"query_id": row["Query_id"].strip(),
                 "sql": row["Query_text"].strip()}
                for row in csv.DictReader(handle)]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(store: OfflineStore, names: list[str], queries=None) -> dict:
    if queries is not None:
        # preferred path: join counts aggregated from the real workload SQL
        return store.evidence(names, queries=queries)
    result = {}
    for name in names:
        row = store.rows[name]
        result[name] = {
            "columns": [row["column"]],
            "plan_node": row.get("plan_node", ""),
            "plan_rows": row.get("plan_rows"),
            "rows_kept_fraction": row.get("rows_kept_fraction"),
            "selectivity": row.get("selectivity"),
            "clauses": row.get("clauses"),
            "referenced_by_queries": row.get("referenced_by_queries", 0),
            "storage_mib": row["storage_mib"],
            **row.get("pg_stats", {}),
        }
    return result


def parse_response(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def match_name(name: str, candidates: list[str]) -> str | None:
    if not isinstance(name, str):
        return None
    return next((c for c in candidates if c.lower() == name.strip().lower()), None)


def run_rollout(proposer, arbiter, label: str, *, queries: list[dict],
                store: OfflineStore, shortlist_size: int, max_steps: int,
                on_progress: Callable[[dict], None] | None = None) -> dict:
    current: list[str] = []
    steps: list[dict] = []
    result = {"policy": label, "protocol": "proposer_arbiter_stop",
              "steps": steps, "final_indexes": current,
              "final_storage_mib": 0.0, "stop_reason": "running"}

    def publish(stop_reason: str = "running") -> None:
        result["final_indexes"] = list(current)
        result["final_storage_mib"] = store.estimate_storage_mib(current)
        result["stop_reason"] = stop_reason
        if on_progress is not None:
            on_progress(result)

    started = time.monotonic()
    for step_i in range(1, max_steps + 1):
        used = store.estimate_storage_mib(current)
        remaining = max(0.0, store.budget_mib - used)
        legal = store.legal_candidates(current)
        rec = {"step": step_i, "current": list(current), "legal": legal,
               "storage_used_mib": used, "storage_left_mib": round(remaining, 3)}
        steps.append(rec)
        if not legal:
            rec["verdict"] = "budget_wall"
            publish("budget_wall")
            break
        p_msgs = render_selection(
            queries=queries, current_indexes=current, legal_candidates=legal,
            storage_left_mib=remaining, candidate_count=shortlist_size,
            index_evidence=evidence(store, legal, queries=queries))
        rec["proposer_messages"] = p_msgs
        p_start = time.monotonic()
        try:
            p_raw, n_tokens = proposer.chat(p_msgs)
        except Exception as exc:
            rec.update(verdict="proposer_call_failed", error=str(exc)[:300])
            publish("proposer_call_failed")
            break
        rec.update(proposer_raw=p_raw, proposer_seconds=round(time.monotonic()-p_start, 2),
                   prompt_tokens=n_tokens)
        obj = parse_response(p_raw)
        raw_list = obj.get("candidates") if obj is not None else None
        shortlist = ([match_name(c, legal) for c in raw_list]
                     if isinstance(raw_list, list) else None)
        if (shortlist is None or any(c is None for c in shortlist)
                or len(shortlist) > shortlist_size
                or len(set(shortlist)) != len(shortlist)):
            rec["verdict"] = "proposer_contract_violation"
            publish("proposer_contract_violation")
            break
        rec["shortlist"] = shortlist
        if not shortlist:
            rec["verdict"] = "proposer_empty"
            publish("proposer_empty")
            break
        a_msgs = render_arbiter_stop(
            queries=queries, current_indexes=current, candidate_indexes=shortlist,
            storage_left_mib=remaining,
                           index_evidence=evidence(store, shortlist, queries=queries))
        rec["arbiter_messages"] = a_msgs
        a_start = time.monotonic()
        try:
            a_raw, _ = arbiter.chat(a_msgs)
        except Exception as exc:
            rec.update(verdict="arbiter_call_failed", error=str(exc)[:300])
            publish("arbiter_call_failed")
            break
        rec.update(arbiter_raw=a_raw, arbiter_seconds=round(time.monotonic()-a_start, 2))
        obj = parse_response(a_raw)
        winner_raw = obj.get("winner") if obj is not None else None
        if isinstance(winner_raw, str) and winner_raw.strip().upper() == "STOP":
            rec.update(winner="STOP", verdict="arbiter_stop")
            publish("arbiter_stop")
            break
        winner = match_name(winner_raw, shortlist)
        if winner is None:
            rec["verdict"] = "arbiter_contract_violation"
            publish("arbiter_contract_violation")
            break
        rec.update(winner=winner, verdict="applied")
        current.append(winner)
        print(f"[{label} step {step_i}] shortlist={shortlist} winner={winner}", flush=True)
        publish()
    else:
        publish("max_steps")
    result["elapsed_seconds"] = round(time.monotonic()-started, 2)
    publish(result["stop_reason"])
    return result


def save_result(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)



