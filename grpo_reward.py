"""Ground-truth execution rewards for JOB GRPO.

Local PG14 database provides every number: T0/T(S) are wall-clock medians of
full workload runs, M(S) is physical storage of the recommended set. Nothing
here reads teacher labels or planner costs.

Conventions (FINAL_PLAN section 8 plus 2026-10-13 confirmed decisions):
- Training-time reward uses the configured run count (GRPO uses 1; final
  evaluation keeps 5 runs).
- T(S) is incremental: only queries whose EXPLAIN plan changed under the new
  index set are re-executed; unchanged queries reuse cached timings.
- Cache key = (stats version, workload id, index set, measurement protocol),
  so identical sets are never measured twice.
- B in F(S) is the episode budget (confirmed reading 1), matching the
  over-budget penalty denominator.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import psycopg2

import db as db_mod


def _cv(samples: Sequence[float]) -> float:
    """Coefficient of variation; 1.0 (worst) for degenerate batches."""
    if len(samples) < 2:
        return 1.0
    mean = sum(samples) / len(samples)
    if mean <= 0:
        return 1.0
    var = sum((s - mean) ** 2 for s in samples) / len(samples)
    return (var ** 0.5) / mean


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------

@dataclass
class WorkloadRunner:
    """Serial workload execution on one database with per-query timing cache.

    Indexes are materialized for real (CREATE INDEX) rather than HypoPG:
    GRPO needs execution time, and hypothetical indexes cannot be executed.
    The namespace and cleanup discipline follow FINAL_PLAN section 9.
    """

    db_id: str
    queries: list[dict]                     # [{'query_id', 'sql'}, ...]
    namespace: str = "FuseIA_grpo"
    statement_timeout_s: int = 600
    # ---- caches -----------------------------------------------------------
    plan_cache: dict = field(default_factory=dict)    # (set, qid) -> plan json str
    time_cache: dict = field(default_factory=dict)    # (set, qid) -> [seconds]
    unbuildable: set = field(default_factory=set)     # indexes PG cannot build
    # --- intra-batch self-normalization (2026-10-15, user decision) ---
    cv_gate: float = 0.15            # batch CV above this = environment noise
    retry_backoff_s: float = 10.0    # pause before a noise-retry
    quiesce_timeout_s: float = 600.0 # max wait for autovacuum to finish
    low_confidence: dict = field(default_factory=dict)  # qid -> last batch CV

    def index_name(self, index_text: str) -> str:
        digest = hashlib.sha1(index_text.encode()).hexdigest()[:12]
        return f"ix_{self.namespace}_{digest}"

    # -- plan comparison ----------------------------------------------------
    def _plan_key(self, cursor, sql: str, index_set: tuple) -> str:
        key = (index_set, sql)
        if key in self.plan_cache:
            return self.plan_cache[key]
        db_mod.hypopg_load(cursor, self._conn, index_set)
        plan = db_mod.explain_json(cursor, sql)
        text = json.dumps(plan, sort_keys=True)
        self.plan_cache[key] = text
        return text

    def changed_queries(self, index_set: Sequence[str],
                        base_set: Sequence[str]) -> list[str]:
        """Query IDs whose plan under index_set differs from base_set (Q4-A).

        HypoPG comparison keeps this metadata-only: no execution, no writes.
        """
        a, b = tuple(sorted(index_set)), tuple(sorted(base_set))
        if a == b:
            return []
        changed = []
        self._conn, cursor = self._connect()
        try:
            for query in self.queries:
                try:
                    before = self._plan_key(cursor, query["sql"], b)
                    after = self._plan_key(cursor, query["sql"], a)
                except Exception:
                    changed.append(query["query_id"])   # plan failure: re-measure
                    continue
                if before != after:
                    changed.append(query["query_id"])
        finally:
            cursor.close()
            self._conn.close()
            del self._conn
        return changed

    # -- real execution -----------------------------------------------------
    def _connect(self):
        conn = psycopg2.connect(host=os.getenv("PGHOST", "127.0.0.1"),
                                port=int(os.getenv("PGPORT", "55432")),
                                user=os.getenv("PGUSER", "postgres"),
                                password=os.getenv("PGPASSWORD", "postgres"),
                                dbname=self.db_id, connect_timeout=10)
        conn.autocommit = False
        cursor = conn.cursor()
        cursor.execute(f"SET statement_timeout = {self.statement_timeout_s * 1000}")
        return conn, cursor

    def _physical_indexes(self, cursor) -> dict[str, str]:
        cursor.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND indexname LIKE %s",
            (f"ix_{self.namespace}_%",))
        return {name: definition for name, definition in cursor.fetchall()}

    def _reconcile(self, cursor, wanted: Sequence[str]) -> None:
        """Bring physical namespace to exactly `wanted` (base set untouched)."""
        have = self._physical_indexes(cursor)
        wanted_names = {self.index_name(i): i for i in wanted}
        for name, definition in have.items():
            if name not in wanted_names:
                cursor.execute(f'DROP INDEX public."{name}"')
        for name, index_text in wanted_names.items():
            if name not in have:
                if index_text in self.unbuildable:
                    continue
                table, columns = db_mod.parse_index(index_text)
                column_list = ",".join(f'"{c}"' for c in columns)
                try:
                    cursor.execute(
                        f'CREATE INDEX "{name}" ON public."{table}" ({column_list})')
                except psycopg2.errors.ProgramLimitExceeded:
                    # b-tree entry size limit: this candidate cannot be
                    # materialized at all; record and exclude everywhere.
                    self.unbuildable.add(index_text)
                    self._conn.rollback()
                    if index_text not in self.unbuildable:
                        raise
        self._conn.commit()

    def _timed_runs(self, cursor, sql: str, runs: int) -> tuple[list, list]:
        """Run `sql` `runs` times; return (kept_samples, discarded_batches)."""
        samples, discarded = [], []
        self._last_timing_error = None
        for _ in range(runs):
            try:
                cursor.execute("DISCARD PLANS")
                start = time.perf_counter()
                cursor.execute(sql)
                cursor.fetchall()
                samples.append(time.perf_counter() - start)
            except Exception as error:
                self._last_timing_error = f"{type(error).__name__}: {error}"
                self._conn.rollback()
                samples = []
                break
        if len(samples) >= 2:
            med = sorted(samples)[len(samples) // 2]
            for s in samples:
                if s > med * 1.5:               # single-run outlier (host jitter)
                    discarded.append({"sample": s, "median": med})
            samples = [s for s in samples if s <= med * 1.5] or samples
        return samples, discarded

    def _pre_measure_quiesce(self, cursor) -> None:
        """Force a checkpoint and wait out autovacuum so timing starts clean.

        Waits up to quiesce_timeout_s for autovacuum workers to finish; if
        still busy, proceeds anyway (caller-side outlier gate handles noise).
        """
        try:
            cursor.execute("CHECKPOINT")
            self._conn.commit()
        except Exception:
            self._conn.rollback()
        deadline = time.time() + self.quiesce_timeout_s
        while time.time() < deadline:
            cursor.execute(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() AND backend_type = 'autovacuum worker'")
            if cursor.fetchone()[0] == 0:
                return
            time.sleep(2.0)

    def measure(self, index_set: Sequence[str], runs: int = 3,
                force_queries: Sequence[str] | None = None) -> dict:
        """Median wall-clock seconds per query under (base + index_set).

        Queries whose cached timing already exists under this exact set are
        reused; `force_queries` re-executes specific ids (plan changed).
        Returns {'seconds': {qid: median}, 'total': sum-of-medians,
                 'failures': {qid: error}}.
        """
        wanted = tuple(sorted(set(index_set)))
        cached_ids = {qid for qid in self._query_ids
                      if self.time_cache.get((wanted, qid)) is not None}
        # First arrival at this set executes the full workload even when no
        # plan changed; force_queries only adds re-measurement on top.
        todo = (set(self._query_ids) - cached_ids) | set(force_queries or ())
        failures: dict[str, str] = {}
        if todo:
            self._conn, cursor = self._connect()
            retries = 0
            discarded_batches = []
            try:
                self._pre_measure_quiesce(cursor)
                self._reconcile(cursor, wanted)
                for qid in sorted(todo):
                    sql = self._sql[qid]
                    samples, discarded = self._timed_runs(cursor, sql, runs)
                    # intra-batch self-normalization gate: discard a whole batch
                    # when the environment (autovacuum, host IO, OS jitter) made
                    # it unrepresentative; retry up to twice, then keep the best
                    # CV batch and flag low confidence.
                    if not samples:
                        failures[qid] = (getattr(self, "_last_timing_error", None)
                                         or "no successful timing sample")
                        continue
                    # A single sample has no measurable variance. Skip CV
                    # retries so --runs=1 remains one execution, and flag the
                    # result as lower-confidence in the audit metadata.
                    cv = _cv(samples) if runs >= 2 else None
                    attempts = 0
                    while cv is not None and cv > self.cv_gate and attempts < 2:
                        attempts += 1
                        retries += 1
                        discarded_batches.append(
                            {"query_id": qid, "samples": samples, "cv": round(cv, 4)})
                        time.sleep(self.retry_backoff_s)
                        self._pre_measure_quiesce(cursor)
                        samples, _ = self._timed_runs(cursor, sql, runs)
                        if not samples:
                            failures[qid] = (getattr(self, "_last_timing_error", None)
                                             or "no successful retry timing sample")
                            break
                        cv = _cv(samples) if runs >= 2 else None
                    if qid in failures:
                        continue
                    if cv is None:
                        self.low_confidence[qid] = "single_run_cv_unavailable"
                    elif cv > self.cv_gate:
                        self.low_confidence[qid] = round(cv, 4)
                    elif qid in self.low_confidence:
                        del self.low_confidence[qid]
                    if discarded:
                        discarded_batches.extend(discarded)
                    self.time_cache[(wanted, qid)] = samples
                self._conn.commit()
            finally:
                self._last_retries = retries
                self._last_discarded = discarded_batches
                self._conn.close()
                del self._conn
        seconds = {}
        for qid in self._query_ids:
            samples = self.time_cache.get((wanted, qid))
            if samples:
                seconds[qid] = sorted(samples)[len(samples) // 2]
        return {"seconds": seconds, "total": sum(seconds.values()),
                "failures": failures,
                "cached_queries": len(cached_ids),
                "executed_queries": len(todo),
                "retries": getattr(self, "_last_retries", 0),
                "discarded_batches": getattr(self, "_last_discarded", []),
                "low_confidence": dict(getattr(self, "low_confidence", {}))}

    @property
    def _query_ids(self) -> list[str]:
        return [q["query_id"] for q in self.queries]

    @property
    def _sql(self) -> dict[str, str]:
        return {q["query_id"]: q["sql"] for q in self.queries}

    def base_index_snapshot(self, cursor) -> list[str]:
        """Record base indexes once per run so pollution is detectable."""
        cursor.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname='public' "
            f"AND indexname NOT LIKE 'ix_{self.namespace}_%'")
        return [row[0] for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    runs: int = 1                     # training single sample; eval uses 5
    # Combined global-time and storage-efficiency shaping (user-confirmed).
    # storage_share = M(index) / episode budget.
    storage_weight: float = 0.05      # retained for config compatibility
    deadband: float = 0.01            # noise floor on relative time gain
    deadband_penalty: float = -0.02   # deterministic "no benefit" penalty
    slow_multiplier: float = 2.0      # slowdowns punished harder than gains rewarded
    speed_bonus_multiplier: float = 2.0  # storage-share bonus for big speedups
    repeated_penalty: float = -0.1
    invalid_penalty: float = -1.0
    over_budget_base: float = -1.0
    stop_probe_cap: int = 3
    warmup_runs: int = 1              # DISCARD PLANS then one throwaway pass


@dataclass
class GRPOReward:
    """F(S) bookkeeping for one episode (one workload, one budget)."""

    runner: WorkloadRunner
    budget_mib: float
    storage_mib: Mapping[str, float]  # physical name -> estimated MiB
    config: RewardConfig = field(default_factory=RewardConfig)
    t0: float | None = None
    # running baseline for state-anchored advantages (confirmed option A)
    state_values: dict = field(default_factory=dict)

    def storage_of(self, index_set: Sequence[str]) -> float:
        return sum(self.storage_mib[name] for name in index_set)

    def measure_total(self, index_set: Sequence[str],
                      force_queries: Sequence[str] | None = None) -> tuple[float, dict]:
        result = self.runner.measure(index_set, runs=self.config.runs,
                                     force_queries=force_queries)
        return result["total"], result

    def utility(self, index_set: Sequence[str], total_seconds: float) -> float:
        """Relative workload improvement over the base T0."""
        return (self.t0 - total_seconds) / self.t0

    def _shaped_add_reward(self, time_gain: float, index_mib: float) -> tuple[float, str, dict]:
        """Combine global time improvement and storage efficiency.

        The first term keeps the policy from selecting tiny, high-ratio indexes
        while the second rewards improvement per unit of index storage. Storage
        also scales no-benefit and slowdown penalties, so a large harmful index
        is strictly worse than an equally slow small index.
        """
        cfg = self.config
        storage_share = max(0.0, float(index_mib) / max(self.budget_mib, 1e-9))
        if time_gain > cfg.deadband:
            ratio_gain = time_gain / max(storage_share, 0.01)
            reward = 0.5 * time_gain + 0.5 * ratio_gain
            band = "speedup"
        elif time_gain < -cfg.deadband:
            reward = time_gain * (1.0 + storage_share)
            band = "slowdown"
        else:
            reward = cfg.deadband_penalty * (1.0 + storage_share)
            band = "no_benefit"
        return reward, band, {
            "storage_share": storage_share,
            "ratio_gain": time_gain / max(storage_share, 0.01),
        }

    def step_reward(self, current: Sequence[str], chosen: str | None,
                    legal: Sequence[str]) -> dict:
        """r_add for one arbiter decision (candidate add or STOP).

        STOP gets a counterfactual probe value over up to
        `stop_probe_cap` remaining feasible candidates: the best r_add the
        policy left on the table. Illegal actions return their contractual
        penalty and never touch the database.
        """
        detail = {"current_size": len(current)}
        if chosen is not None and chosen not in legal:
            # feasible list recomputed from the store; treat as over-budget /
            # illegal structural action with the contractual penalty
            over = self.storage_of(list(current) + [chosen]) / self.budget_mib
            detail["reward"] = self.config.over_budget_base - max(0.0, over - 1.0)
            detail["kind"] = "illegal"
            return detail
        if chosen is None:                              # STOP probe
            current_set = tuple(sorted(current))
            t_current, current_info = self.measure_total(current_set)
            if current_info["failures"]:
                detail.update({"reward": 0.0, "kind": "stop_probe",
                               "failures": current_info["failures"],
                               "note": "measurement_failure_kept_zero"})
                return detail
            probes = []
            for name in sorted(set(legal) - set(current))[:self.config.stop_probe_cap]:
                candidate_set = tuple(sorted(current_set + (name,)))
                t_probe, probe_info = self.measure_total(
                    candidate_set,
                    force_queries=self.runner.changed_queries(candidate_set, current_set))
                if probe_info["failures"]:
                    detail.update({"reward": 0.0, "kind": "stop_probe",
                                   "failures": probe_info["failures"],
                                   "note": "measurement_failure_kept_zero"})
                    return detail
                gain = (t_current - t_probe) / self.t0
                candidate_mib = self.storage_of([name])
                probe_reward, band, extra = self._shaped_add_reward(
                    gain, candidate_mib)
                probes.append(probe_reward)
                detail.setdefault("probe_details", []).append({
                    "candidate": name, "time_gain": gain,
                    "band": band, **extra})
            detail["reward"] = max(probes) if probes else 0.0
            detail["kind"] = "stop_probe"
            detail["probe_count"] = len(probes)
            return detail
        next_set = list(current) + [chosen]
        t_before, _ = self.measure_total(tuple(sorted(current)))
        t_after, info = self.measure_total(
            next_set, force_queries=self.runner.changed_queries(next_set, current))
        detail.update({"kind": "add", "executed_queries": info["executed_queries"],
                       "cached_queries": info["cached_queries"],
                       "failures": info["failures"],
                       "low_confidence_queries": sorted(info["low_confidence"]),
                       "timing_retries": info["retries"]})
        if info["failures"]:
            # SQL failure is recorded, never silently treated as zero gain.
            detail["reward"] = 0.0
            detail["note"] = "measurement_failure_kept_zero"
            return detail
        time_gain = (t_before - t_after) / self.t0
        detail["time_gain"] = time_gain
        detail["t_before"], detail["t_after"] = t_before, t_after
        m_idx = self.storage_mib[chosen]
        reward, band, extra = self._shaped_add_reward(time_gain, m_idx)
        detail["band"] = band
        detail.update(extra)
        detail["reward"] = reward
        return detail


    def initialize_t0(self, warmup: bool = False) -> float:
        """Measure T0 once; training must not add a throwaway workload pass."""
        if warmup:
            # Kept as an explicit opt-in for diagnostics/evaluation callers.
            self.runner.measure([], runs=1)
            self.runner.time_cache.clear()
        total, _ = self.measure_total([])
        self.t0 = total
        return total
