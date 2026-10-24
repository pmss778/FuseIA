"""Frozen candidate facts for structured data preparation and teacher prompts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_pool(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {}
    return payload.get("candidates", []), {
        key: value for key, value in payload.items() if key != "candidates"
    }


class OfflineStore:
    """Estimate storage from a frozen pool without database or cost calls."""

    def __init__(self, db_id: str, rows: Sequence[Mapping[str, Any]],
                 budget_mib: float, query_roles=None, clause_evidence=None):
        self.db_id = db_id
        self.rows = {str(row["name"]): dict(row) for row in rows}
        self.budget_mib = float(budget_mib)
        self.query_roles = list(query_roles or [])
        self.clause_evidence = dict(clause_evidence or {})
        self._sizes = {
            name: float(row["storage_mib"]) for name, row in self.rows.items()
        }

    def estimate_storage_mib(self, indexes: Sequence[str]) -> float:
        return round(sum(self._sizes[name] for name in indexes), 3)

    def legal_candidates(self, current: Sequence[str]) -> list[str]:
        selected = set(current)
        used = sum(self._sizes[name] for name in current)
        return [name for name in sorted(self.rows)
                if name not in selected
                and used + self._sizes[name] <= self.budget_mib]

    def evidence(self, names: Sequence[str], queries: Sequence[Mapping[str, Any]] | None = None) -> dict[str, dict]:
        # Keep the frozen teacher projection compatible with existing prompts.
        # Join identity and aggregated selectivity are computed from the real
        # workload SQL at evidence time; the frozen per-query plan values are
        # single-query artifacts and mislabel join keys (observed: a query
        # joining on l_orderkey recorded it as a selection predicate).
        ref_counts, total_queries = self._join_profile(queries)
        result = {}
        for name in names:
            row = self.rows.get(name)
            if row is None:
                result[name] = {"source": "not_a_candidate"}
                continue
            join_n = 0
            table = row["table"].lower()
            column = row["column"].lower()
            join_n = ref_counts.get(f"{table}.{column}", 0)
            if not join_n:
                # unresolved-alias fallback: prefix-match candidate table
                for ref, count in ref_counts.items():
                    alias, _, col = ref.partition(".")
                    if col != column:
                        continue
                    base = alias.rstrip("0123456789")
                    if base and table.startswith(base):
                        join_n += count
            result[name] = {
                "columns": [row["column"]],
                "plan_node": row["plan_node"],
                "plan_rows": row["plan_rows"],
                "rows_kept_fraction": row["rows_kept_fraction"],
                "selectivity": row["selectivity"],
                "clauses": row["clauses"],
                "referenced_by_queries": row["referenced_by_queries"],
                "join_query_count": join_n,
                "storage_mib": row["storage_mib"],
                **row["pg_stats"],
            }
            if total_queries:
                result[name]["workload_queries"] = total_queries
        return result

    def _join_profile(self, queries):
        """Count, per physical column, in how many queries it appears in a
        cross-table comparison (binary predicate with columns from two
        different tables). Aliases resolve through sqlglot scopes; columns
        without a table qualifier resolve through this pool's column index
        (a name owned by exactly one pool table is unambiguous). Returns
        ({"table.column": count}, parsed_count)."""
        if not queries:
            return {}, 0
        # column-name -> owning tables, from the frozen pool (universe)
        column_owners: dict[str, set] = {}
        for row in self.rows.values():
            column_owners.setdefault(str(row["column"]).lower(), set()).add(
                str(row["table"]).lower())
        try:
            import sqlglot
            from sqlglot import exp
            from sqlglot.optimizer.scope import traverse_scope
        except ImportError:
            return {}, 0
        counts: dict[str, int] = {}
        parsed_any = 0
        for query in queries:
            sql = query.get("sql") if isinstance(query, dict) else None
            if not sql:
                continue
            try:
                tree = sqlglot.parse_one(sql, read="postgres")
                alias_map: dict[str, set] = {}
                for scope in traverse_scope(tree):
                    for name, source in scope.sources.items():
                        if isinstance(source, exp.Table):
                            alias_map.setdefault(name.lower(), set()).add(
                                source.name.lower())
            except Exception:
                continue
            parsed_any += 1
            keys = set()
            for predicate in tree.find_all(exp.Binary):
                columns = list(predicate.find_all(exp.Column))
                if len(columns) != 2:
                    continue
                resolved = []
                for c in columns:
                    ref = (c.table or "").lower()
                    tables = alias_map.get(ref, {ref} if ref else set())
                    exact = [t for t in tables
                             if not t.startswith("__subquery__")]
                    if not exact and not ref:
                        owners = column_owners.get(c.name.lower(), set())
                        exact = sorted(owners) if len(owners) == 1 else []
                    resolved.append(sorted(exact))
                if (len(resolved) == 2 and resolved[0] and resolved[1]
                        and resolved[0] != resolved[1]):
                    for c, tables in zip(columns, resolved):
                        # one physical table per side; ties count as unresolved
                        if len(tables) == 1:
                            keys.add(f"{tables[0]}.{c.name.lower()}")
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
        return counts, parsed_any
