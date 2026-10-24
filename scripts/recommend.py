#!/usr/bin/env python3
"""Greedy FuseIA recommendation from workload, frozen data, and role checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True,
                        help="CSV with Query_id and Query_text columns")
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--proposer-checkpoint", type=Path, required=True)
    parser.add_argument("--arbiter-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True,
                        help="Local Qwen2.5-3B-Instruct directory")
    parser.add_argument("--budget-mib", type=float, required=True)
    parser.add_argument("--view", choices=("original", "anonymous"), default="original")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument("--feasibility-mask", action="store_true",
                        help="Mask over-budget actions; default follows the trained protocol")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    for path in (args.workload, args.candidate_pool, args.schema, args.stats,
                 args.proposer_checkpoint, args.arbiter_checkpoint, args.backbone):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.budget_mib <= 0 or args.max_steps < 1 or args.shortlist_size < 1:
        raise ValueError("budget, max-steps, and shortlist-size must be positive")

    import torch
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for CPU inference")
    from candidate_pool import OfflineStore, load_pool
    from grpo_rollout import DecisionEncoder, share_frozen_base, to_device
    from model_checkpoint import load_checkpoint
    from policy_actions import select_actions
    from policy_inputs import build_context
    from predicate_stats import read_stats
    from recommendation_rollout import load_queries

    queries = load_queries(args.workload)
    pool, _ = load_pool(args.candidate_pool)
    if not queries:
        raise ValueError("workload contains no queries")
    if not pool:
        raise ValueError("candidate pool contains no candidates")
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    stats = read_stats(args.stats)
    if isinstance(schema, dict) and "tables" in schema:
        schema = schema["tables"]
    for table in schema:
        table["rows"] = stats.get("rows", {}).get(table["table"], table.get("rows", 0))
    context = build_context(queries, pool, schema, stats, args.seed)
    store = OfflineStore("offline", pool, args.budget_mib)
    names = sorted(store.rows)
    if not names:
        raise ValueError("candidate pool rows must include name and storage_mib")

    proposer, proposer_meta = load_checkpoint(
        args.proposer_checkpoint, device="cpu", backbone_name=str(args.backbone))
    arbiter, arbiter_meta = load_checkpoint(
        args.arbiter_checkpoint, device="cpu", backbone_name=str(args.backbone))
    if proposer.config.role != "proposer" or arbiter.config.role != "arbiter":
        raise ValueError("The supplied checkpoints must be a Proposer/Arbiter pair")
    shared = share_frozen_base(arbiter, proposer)
    proposer.config.max_shortlist = min(proposer.config.max_shortlist, args.shortlist_size)
    encoder = DecisionEncoder(context, pool, stats, proposer.config.chunk_tokens)
    selected = []
    steps = []
    stop_reason = "max_steps"
    started = time.perf_counter()

    def infer_one(model, current, candidates, step, role):
        model.to(args.device)
        try:
            encoded = encoder.encode(
                model.backbone.tokenizer, current, args.budget_mib, candidates,
                args.view, args.seed + step * 2 + int(role == "arbiter"))
            encoded = to_device(encoded, args.device)
            model.eval()
            with torch.inference_mode():
                return select_actions(
                    model, model(encoded), greedy=True, temperature=1.0,
                    feasibility_mask=args.feasibility_mask)
        finally:
            model.to("cpu")
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    try:
        for step in range(args.max_steps):
            legal = store.legal_candidates(selected)
            if not legal:
                stop_reason = "budget_wall"
                break
            proposed = infer_one(proposer, selected, names, step, "proposer")
            shortlist = [names[i] for i in proposed["candidate_indices"]
                         if names[i] not in selected][:args.shortlist_size]
            entry = {"step": step + 1, "current_indexes": list(selected),
                     "proposer_shortlist": shortlist,
                     "proposer_ended": proposed["terminated"]}
            if not shortlist:
                entry["arbiter_decision"] = "SKIPPED_EMPTY_SHORTLIST"
                steps.append(entry)
                stop_reason = "proposer_empty"
                break
            decision = infer_one(arbiter, selected, shortlist, step, "arbiter")
            choice = (decision["candidate_indices"][0]
                      if decision["candidate_indices"] else len(shortlist))
            if choice == len(shortlist):
                entry["arbiter_decision"] = "STOP"
                steps.append(entry)
                stop_reason = "arbiter_stop"
                break
            winner = shortlist[choice]
            if winner not in legal:
                entry.update(arbiter_decision=winner,
                             status="infeasible_or_already_selected")
                steps.append(entry)
                stop_reason = "arbiter_selected_illegal_candidate"
                break
            selected.append(winner)
            entry.update(arbiter_decision=winner,
                         storage_used_mib=store.estimate_storage_mib(selected))
            steps.append(entry)
        result = {
            "status": "completed", "method": "FuseIA greedy Proposer-Arbiter",
            "workload": str(args.workload), "query_count": len(queries),
            "candidate_pool": str(args.candidate_pool),
            "database_statistics": str(args.stats), "view": args.view,
            "budget_mib": args.budget_mib,
            "estimated_storage_mib": store.estimate_storage_mib(selected),
            "proposer_checkpoint": str(args.proposer_checkpoint),
            "proposer_update": proposer_meta.get("progress", {}).get("update"),
            "arbiter_checkpoint": str(args.arbiter_checkpoint),
            "arbiter_update": arbiter_meta.get("progress", {}).get("update"),
            "shared_frozen_tensors": shared,
            "device_policy": "one role on accelerator at a time",
            "indexes": selected, "stop_reason": stop_reason, "steps": steps,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    finally:
        proposer.to("cpu")
        arbiter.to("cpu")
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": result["status"], "indexes": selected,
                      "estimated_storage_mib": store.estimate_storage_mib(selected),
                      "stop_reason": stop_reason, "output": str(args.output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
