"""Local GRPO rollouts for the final dual-path policies.

One process plays episodes against the local JOB database. The trained role
samples actions (temperature 0.8, no feasibility mask so violations are
possible and penalized); the frozen role acts as part of the environment
(verified decision: greedy when the arbiter is frozen, temperature sampling
when the proposer is frozen). Rewards come from grpo_reward.

Advantages are per-step (confirmed): the group's first decision shares one
state and gets group mean/std normalization; after divergence the baseline
is a running per-state value (option A).

Proposer credit (confirmed): the arbiter's actually-picked r_add for that
step; on any terminal stop the credit is the negative opportunity cost
-probe(best remaining r_add) for both roles' final decision.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Sequence

import torch

from candidate_pool import OfflineStore
from grpo_reward import GRPOReward
from policy_actions import select_actions
from policy_data import _index_ids, graph_input
from policy_inputs import build_state
from semantic_structure_model import SemanticStructureModel


# ---------------------------------------------------------------------------
# Weight sharing: two roles, one copy of frozen base weights on the GPU
# ---------------------------------------------------------------------------

def to_device(obj, device):
    import torch as _torch
    if _torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj


def share_frozen_base(dst: SemanticStructureModel,
                      src: SemanticStructureModel) -> int:
    """Alias dst's frozen backbone tensors to src's storage.

    LoRA and side-network parameters stay independent, so the two roles'
    trainable weights never mix; only the frozen Qwen tensors are shared,
    which is explicitly allowed (physical storage of frozen base weights).
    Returns the number of shared tensors; names must match exactly.
    """
    src_params = dict(src.backbone.hf.named_parameters())
    shared = 0
    for name, parameter in dst.backbone.hf.named_parameters():
        if ".lora_" in name:            # role-specific adapters never shared
            continue
        source = src_params.get(name)
        if source is None or parameter.requires_grad or source.requires_grad:
            raise ValueError(f"Unexpected trainable/unmatched tensor: {name}")
        parameter.data = source.data
        shared += 1
    return shared


# ---------------------------------------------------------------------------
# Runtime encoding (mirrors policy_data.prepare_pair for live states)
# ---------------------------------------------------------------------------

class DecisionEncoder:
    """Builds dual-path inputs from live episode state, not SFT records."""

    def __init__(self, context: dict, pool_rows: list, stats: dict,
                 chunk_tokens: int):
        self.context = context
        self.pool_rows = pool_rows
        self.stats = stats
        self.chunk_tokens = chunk_tokens
        self.graphs = None      # graphs are view-independent: build once

    def encode(self, tokenizer, current: list, budget: float,
               candidates: list, view: str, seed: int) -> dict:
        if self.graphs is None:
            rng = random.Random(seed)
            self.graphs = [graph_input(g, rng, hide_relations=True)
                           for g in self.context["graphs"]]
        state = build_state(self.context, self.pool_rows, self.stats,
                            current, budget, candidates)
        k = len(state["candidate_physical"])
        queries = self.context["queries_" + view]
        chunks = []
        for qi, q in enumerate(queries):
            ids = tokenizer.encode("SCHEMA\n" + self.context["schema_" + view] +
                                   "\nSQL\n" + q["sql"], add_special_tokens=False)
            ids = ids or [tokenizer.eos_token_id]
            for start in range(0, len(ids), self.chunk_tokens):
                chunks.append({"ids": ids[start:start + self.chunk_tokens],
                               "query_index": qi, "offset": start})
        names = state["candidate_" + view]
        return {
            "semantic_chunks": chunks, "graphs": self.graphs,
            "candidate_ids": _index_ids(tokenizer, names),
            "current_ids": _index_ids(tokenizer, state["current_" + view]),
            "candidate_features": torch.tensor(state["candidate_features"],
                                               dtype=torch.float32).reshape(k, -1),
            "candidate_node_links": state["candidate_node_links"],
            "state_features": torch.tensor(state["state_features"],
                                           dtype=torch.float32),
            "candidate_mask": torch.tensor(state["candidate_available"],
                                           dtype=torch.bool),
            "feasible_mask": torch.tensor(state["candidate_feasible"],
                                          dtype=torch.bool),
            "actions": [], "sample_id": "grpo",
            "candidate_physical": state["candidate_physical"],
        }


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

@dataclass
class ActionRecord:
    role: str
    step: int
    state_key: str
    current_indexes: list
    candidates: list            # physical action space of this decision
    actions: list               # sampled ids (proposer: full sequence)
    logprob: float              # sequence logprob at sampling time
    reward: float = 0.0
    reward_detail: dict = field(default_factory=dict)
    view: str = "original"
    reward_future: object = None
    reward_slot: str | None = None


@dataclass
class Trajectory:
    records: list = field(default_factory=list)
    final_indexes: list = field(default_factory=list)
    stop_reason: str = ""
    measurements: dict = field(default_factory=dict)
    final_future: object = None


def resolve_group_rewards(groups: list[list[Trajectory]]) -> None:
    """Resolve queued DB rewards after GPU rollout has continued meanwhile."""
    for group in groups:
        for trajectory in group:
            for record in trajectory.records:
                if record.reward_future is None:
                    continue
                detail = record.reward_future.result()
                record.reward += float(detail.get("reward", 0.0))
                slot = record.reward_slot or "reward"
                record.reward_detail[slot] = detail
                if "time_gain" in detail:
                    record.reward_detail["time_gain"] = detail["time_gain"]
                if "band" in detail:
                    record.reward_detail["band"] = detail["band"]
                record.reward_future = None
            if trajectory.final_future is not None:
                final_detail = trajectory.final_future.result()
                trajectory.measurements["final"] = final_detail
                trajectory.final_future = None


def play_episode(proposer: SemanticStructureModel, arbiter: SemanticStructureModel,
                 context: dict, pool_rows: list, stats: dict,
                 initial: list, budget: float, reward_machine: GRPOReward,
                 max_steps: int, temperature: float, chunk_tokens: int,
                 seed: int, device, trained_role: str,
                 proposer_temperature: float = 0.8,
                 rollout_view: str = "original",
                 reward_executor=None,
                 unbuildable_snapshot: set[str] | None = None) -> Trajectory:
    """Play one episode; returns per-decision records for the trained role.

    Sampling can use either semantic view. Rewards always map selected
    candidates back to physical IDs, so anonymous inputs still execute
    against the same real database state.
    """
    encoder = DecisionEncoder(context, pool_rows, stats, chunk_tokens)
    tokenizer = proposer.backbone.tokenizer
    store = OfflineStore("grpo", pool_rows, budget)
    names = sorted(store.rows)
    current = list(initial)
    trajectory = Trajectory()
    rng = random.Random(seed)

    def record_for(role, step, candidates, actions, logprob):
        key = json.dumps([sorted(current), [candidates[a] for a in actions
                                            if a < len(candidates)]])
        return ActionRecord(role=role, step=step, state_key=key,
                            current_indexes=list(current),
                            candidates=list(candidates), actions=list(actions),
                            logprob=float(logprob))

    def record_reward(record, slot, current_state, chosen, legal_state):
        """Queue DB-bound reward work, or execute directly for legacy callers."""
        frozen_current = tuple(current_state)
        frozen_legal = tuple(legal_state)
        if reward_executor is not None:
            record.reward_slot = slot
            record.reward_future = reward_executor.submit(
                reward_machine.step_reward, frozen_current, chosen, frozen_legal)
            return None
        detail = reward_machine.step_reward(frozen_current, chosen, frozen_legal)
        record.reward += float(detail.get("reward", 0.0))
        record.reward_detail[slot] = detail
        if "time_gain" in detail:
            record.reward_detail["time_gain"] = detail["time_gain"]
        if "band" in detail:
            record.reward_detail["band"] = detail["band"]
        return detail

    for step in range(max_steps):
        unbuildable = (unbuildable_snapshot if unbuildable_snapshot is not None
                       else reward_machine.runner.unbuildable)
        legal = [n for n in store.legal_candidates(current)
                 if n not in unbuildable]
        remaining = max(0.0, budget - store.estimate_storage_mib(current))
        if not legal:
            trajectory.stop_reason = "empty_legal"
            break

        # --- proposer: sample a shortlist over the full pool ----------------
        encoded_p = encoder.encode(tokenizer, current, budget, names,
                                   rollout_view, seed + 1000 * step + 1)
        encoded_p = to_device(encoded_p, device)
        proposer.eval()
        with torch.no_grad():
            picked = select_actions(proposer, proposer(encoded_p), greedy=False,
                                    temperature=proposer_temperature)
        shortlist_idx = picked["candidate_indices"]
        # protocol/business violations: current-index repeats in the shortlist
        violation_penalty = 0.0
        clean_idx = []
        for index in shortlist_idx:
            if names[index] in current:
                violation_penalty += reward_machine.config.repeated_penalty
            else:
                clean_idx.append(index)
        shortlist = [names[i] for i in clean_idx]

        if trained_role == "proposer":
            record = record_for("proposer", step, names, picked["actions"],
                                picked["logprob"])
            record.reward += violation_penalty
            record.reward_detail["violations"] = -violation_penalty / 0.1 if violation_penalty else 0

        if not shortlist:
            # legal normal stop: empty shortlist ends the episode
            if trained_role == "proposer":
                record_reward(record, "stop", current, None, legal)
                trajectory.records.append(record)
            trajectory.stop_reason = "proposer_empty"
            break

        # --- arbiter: pick from shortlist or STOP ---------------------------
        encoded_a = encoder.encode(tokenizer, current, budget, shortlist,
                                   rollout_view, seed + 1000 * step + 2)
        encoded_a = to_device(encoded_a, device)
        arbiter.eval()
        frozen_greedy = trained_role == "proposer"     # confirmed decision
        with torch.no_grad():
            chosen = select_actions(arbiter, arbiter(encoded_a),
                                    greedy=frozen_greedy,
                                    temperature=0.8 if not frozen_greedy else 1.0)
        action = chosen["actions"][0]
        k = len(shortlist)

        if trained_role == "arbiter":
            record = record_for("arbiter", step, shortlist, chosen["actions"],
                                chosen["logprob"])

        if action == k:      # STOP
            if trained_role == "proposer":
                record_reward(record, "stop", current, None, legal)
                trajectory.records.append(record)
            elif trained_role == "arbiter":
                record_reward(record, "stop", current, None, legal)
                trajectory.records.append(record)
            trajectory.stop_reason = "arbiter_stop"
            break

        winner = shortlist[action]
        detail = None
        # Detect business-rule violations synchronously for either role;
        # legal candidates can be rewarded asynchronously without changing
        # the rollout.
        if winner not in legal:
            detail = reward_machine.step_reward(current, winner, legal)
        if detail is not None and detail["kind"] == "illegal":
            record.reward += detail["reward"]
            record.reward_detail["illegal"] = detail
            trajectory.records.append(record)
            trajectory.stop_reason = "rejected_action"
            trajectory.measurements[f"step{step}"] = detail
            break
        current.append(winner)
        if trained_role == "proposer":
            record_reward(record, "add", current[:-1], winner, legal)
            trajectory.records.append(record)
        elif trained_role == "arbiter":
            record_reward(record, "add", current[:-1], winner, legal)
            trajectory.records.append(record)
    else:
        trajectory.stop_reason = "max_steps_truncated"

    trajectory.final_indexes = list(current)
    m = reward_machine.storage_of(current)
    if reward_executor is not None:
        def final_measurement(index_set, storage_mib):
            total, info = reward_machine.measure_total(index_set)
            return {"storage_mib": storage_mib,
                    "total_seconds": total,
                    "utility": reward_machine.utility(index_set, total),
                    "failures": info.get("failures", {})}
        trajectory.final_future = reward_executor.submit(
            final_measurement, tuple(current), m)
    else:
        final_total, info = reward_machine.measure_total(current)
        trajectory.measurements["final"] = {
            "storage_mib": m, "total_seconds": final_total,
            "utility": reward_machine.utility(current, final_total),
            "failures": info.get("failures", {})}
    return trajectory


# ---------------------------------------------------------------------------
# Per-step advantages (confirmed: group baseline pre-divergence, option A after)
# ---------------------------------------------------------------------------

def assign_advantages(groups: list[list[Trajectory]], state_values: dict,
                       alpha: float = 0.1) -> list[dict]:
    """Fill ActionRecord.reward with advantages, in place.

    Step-0 decisions share one state per group: advantage = (r - mean)/std
    over the group. Post-divergence decisions use the running per-state mean
    (option A). Raw rewards are preserved in reward_detail['raw_reward'].
    """
    flat = []
    for group in groups:
        firsts = [t.records[0].reward for t in group if t.records]
        if len(firsts) > 1:
            mean = sum(firsts) / len(firsts)
            var = sum((x - mean) ** 2 for x in firsts) / len(firsts)
            std = var ** 0.5
        else:
            mean, std = 0.0, 0.0
        for trajectory in group:
            for position, record in enumerate(trajectory.records):
                record.reward_detail["raw_reward"] = record.reward
                if position == 0 and std > 1e-8 and len(firsts) > 1:
                    record.reward = (record.reward - mean) / (std + 1e-8)
                    record.reward_detail["baseline"] = "group"
                else:
                    baseline = state_values.get(record.state_key)
                    if baseline is None:
                        state_values[record.state_key] = record.reward
                        baseline = record.reward
                    else:
                        state_values[record.state_key] = \
                            (1 - alpha) * baseline + alpha * record.reward
                    record.reward = record.reward - state_values[record.state_key]
                    record.reward_detail["baseline"] = "state_ema"
                flat.append({"role": record.role, "step": record.step,
                             "advantage": record.reward,
                             "raw": record.reward_detail["raw_reward"]})
    return flat


def normalize_advantages(groups: list[list[Trajectory]], clip: float = 5.0) -> dict:
    """Normalize and bound all post-baseline advantages in one update."""
    records = [r for group in groups for trajectory in group
               for r in trajectory.records]
    values = [float(r.reward) for r in records]
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "clipped": 0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var ** 0.5
    if std <= 1e-8:
        for r in records:
            r.reward = 0.0
        return {"count": len(values), "mean": mean, "std": 0.0, "clipped": 0}
    clipped = 0
    for r in records:
        normalized = (float(r.reward) - mean) / (std + 1e-8)
        bounded = max(-clip, min(clip, normalized))
        clipped += int(bounded != normalized)
        r.reward = bounded
    return {"count": len(values), "mean": mean, "std": std, "clipped": clipped}
