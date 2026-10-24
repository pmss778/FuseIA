"""Structured policy data, graph batching, and paired-view collation."""
from __future__ import annotations

import math
import re
from typing import Any

import torch
from torch.utils.data import Dataset

from relational_graph import anonymize_index
from semantic_structure_model import graph_to_tensors


def merge_graphs(graphs: list[dict]) -> dict:
    nodes = [{"node_id": "workload", "kind": "query_block", "value": "workload", "attrs": {}}]
    edges = []
    for qi, graph in enumerate(graphs):
        prefix = f"q{qi}:"
        ids = {node["node_id"] for node in graph.get("nodes", [])}
        for node in graph.get("nodes", []):
            item = dict(node)
            item["node_id"] = prefix + item["node_id"]
            nodes.append(item)
        for edge in graph.get("edges", []):
            item = dict(edge)
            if item["source"] in ids:
                item["source"] = prefix + item["source"]
            if item["target"] in ids:
                item["target"] = prefix + item["target"]
            edges.append(item)
        root = next((node for node in graph.get("nodes", [])
                     if node.get("kind") == "query_block"), None)
        if root:
            edges.append({"source": "workload",
                          "target": prefix + root["node_id"],
                          "relation": "contains_block"})
    return {"nodes": nodes, "edges": edges}


def _rename_sql(sql: str, mapping: dict[str, str]) -> str:
    text = str(sql)
    for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
        if "." in old:
            table, column = old.split(".", 1)
            new_table, new_column = new.split(".", 1)
            text = re.sub(rf"\b{re.escape(table)}\.{re.escape(column)}\b",
                          f"{new_table}.{new_column}", text, flags=re.I)
        else:
            text = re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.I)
    return text


def _rename_structure(structure: dict, mapping: dict[str, str]) -> dict:
    result = dict(structure)
    result["parameterized_sql"] = _rename_sql(result.get("parameterized_sql", ""), mapping)
    for key in ("tables", "columns", "group_by", "order_by", "ctes"):
        result[key] = [mapping.get(value, value) for value in result.get(key, [])]
    result["aliases"] = {key: mapping.get(value, value)
                          for key, value in result.get("aliases", {}).items()}
    predicates = []
    for predicate in result.get("predicates", []):
        item = dict(predicate)
        ref = mapping.get(f"{item.get('table', '')}.{item.get('column', '')}",
                          f"{item.get('table', '')}.{item.get('column', '')}")
        item["table"], _, item["column"] = ref.partition(".")
        item["raw"] = _rename_sql(item.get("raw", ""), mapping)
        predicates.append(item)
    result["predicates"] = predicates
    result["joins"] = [
        {key: mapping.get(value, value) if key in ("left", "right") else value
         for key, value in join.items()}
        for join in result.get("joins", [])]
    result["aggregates"] = [_rename_sql(value, mapping)
                             for value in result.get("aggregates", [])]
    return result


def _rename_graph(graph: dict, mapping: dict[str, str]) -> dict:
    out = {"nodes": [], "edges": [dict(edge) for edge in graph.get("edges", [])]}
    for node in graph.get("nodes", []):
        item = dict(node)
        value = str(item.get("value", ""))
        kind = item.get("kind")
        if kind in {"table", "table_instance"}:
            item["value"] = mapping.get(value.lower(), value)
            attrs = dict(item.get("attrs") or {})
            if "table" in attrs:
                attrs["table"] = mapping.get(str(attrs["table"]).lower(), attrs["table"])
            item["attrs"] = attrs
        elif kind in {"column", "predicate", "join", "aggregate", "group_by", "order_by"}:
            item["value"] = _rename_sql(value, mapping)
        out["nodes"].append(item)
    return out


def _feature_row(row: dict, budget: float, current: set[str]) -> list[float]:
    stats = row.get("pg_stats") or {}
    size = float(row.get("storage_mib", 0.0))
    return [math.log1p(size), size / max(budget, 1e-6),
            float(row.get("selectivity") or row.get("rows_kept_fraction") or 0.0),
            math.log1p(float(row.get("referenced_by_queries", 0))),
            float(row["name"] in current), float(size <= budget),
            float(stats.get("ndv_ratio") or 0.0),
            float(len(str(row.get("column", "")).split(",")))]


def materialize_view(state: dict, view: str, role: str,
                     current_physical: list[str], candidate_physical: list[str],
                     target: int | list[int], current_used: float,
                     pool: dict[str, dict], pair_key: str, source_trace: str,
                     source_step: int, manifest: str) -> dict[str, Any]:
    mapping = state.get("rename_map", {}) if view == "anonymous" else {}
    structures = state["structures"]
    graphs = state["graphs"]
    if view == "anonymous":
        display_current = [anonymize_index(name, mapping) for name in current_physical]
        display_candidates = [anonymize_index(name, mapping) for name in candidate_physical]
        # Manifest anonymous states are already anonymized. Do not rename them twice.
        display_structures = structures
        display_graphs = graphs
    else:
        display_current = list(current_physical)
        display_candidates = list(candidate_physical)
        display_structures = structures
        display_graphs = graphs
    return {
        "pair_key": pair_key, "role": role, "view": view,
        "state_id": state["state_id"], "seed": state["seed"],
        "split": state["split"], "state_index": state.get("state_index"),
        "workload_id": state["workload_id"],
        "queries": [{"query_id": query["query_id"],
                    "sql": structure.get("parameterized_sql", query["sql"])}
                   for query, structure in zip(state["queries"], display_structures)],
        "structures": display_structures, "graphs": display_graphs,
        "current_physical": list(current_physical),
        "current_display": display_current,
        "candidate_physical": list(candidate_physical),
        "candidate_display": display_candidates,
        "candidate_features": [_feature_row(pool[name], float(state["budget_mib"]),
                                             set(current_physical))
                               for name in candidate_physical],
        "budget_mib": float(state["budget_mib"]),
        "storage_used_mib": float(current_used),
        "storage_left_mib": max(0.0, float(state["budget_mib"]) - current_used),
        "target": target, "manifest": manifest,
        "source_trace": source_trace, "source_step": source_step,
        "stats_version": state.get("stats_version"), "view_mapping": mapping,
    }


class StructuredDataset(Dataset):
    """Paired original/anonymous records grouped by pair_key."""
    def __init__(self, path):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()
                if line.strip()]
        grouped = {}
        for row in rows:
            grouped.setdefault(row["pair_key"], {})[row["view"]] = row
        self.rows = [pair for pair in grouped.values()
                     if "original" in pair and "anonymous" in pair]
        if not self.rows:
            raise ValueError(f"no paired records in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _pad_graph(graphs: list[dict]) -> dict:
    tensors = [graph_to_tensors(graph, max_nodes=min(2048, max(1, len(graph.get("nodes", [])))))
               for graph in graphs]
    n = max(item["node_kind"].shape[0] for item in tensors)
    batch = len(tensors)
    rel = torch.zeros(batch, n, n, dtype=torch.long)
    kind = torch.zeros(batch, n, dtype=torch.long)
    nums = torch.zeros(batch, n, 3)
    mask = torch.zeros(batch, n)
    texts = []
    for i, item in enumerate(tensors):
        size = item["node_kind"].shape[0]
        rel[i, :size, :size] = item["rel"]
        kind[i, :size] = item["node_kind"]
        nums[i, :size] = item["node_num"]
        mask[i, :size] = item["node_mask"]
        texts.append(item["node_text"] + [""] * (n - size))
    return {"rel": rel, "node_kind": kind, "node_num": nums,
            "node_mask": mask, "node_text": texts}


def collate_structured(rows: list[dict]) -> dict:
    """Collate a paired batch; both views share physical candidate positions."""
    output = {}
    for view in ("original", "anonymous"):
        records = [row[view] for row in rows]
        max_k = max(len(record["candidate_physical"]) for record in records)
        names, features, masks, states, graphs, sqls = [], [], [], [], [], []
        actions, action_masks, arbiter_targets = [], [], []
        for record in records:
            count = len(record["candidate_physical"])
            names.append(record["candidate_display"] + [""] * (max_k - count))
            features.append(record["candidate_features"] + [[0.0] * 8] * (max_k - count))
            masks.append([1.0] * count + [0.0] * (max_k - count))
            states.append([math.log1p(record["budget_mib"]),
                           record["storage_used_mib"] / max(record["budget_mib"], 1e-6),
                           math.log1p(len(record["current_physical"])),
                           math.log1p(count)])
            graphs.append(merge_graphs(record["graphs"]))
            sqls.append("\n\n".join(item["sql"] for item in record["queries"]))
            if record["role"] == "proposer":
                action = list(record["target"] or [])
                actions.append(action + [count] * (4 - len(action)))
                action_masks.append([1.0] * len(action) + [0.0] * (4 - len(action)))
                arbiter_targets.append(count)
            else:
                actions.append([count] * 4)
                action_masks.append([0.0] * 4)
                arbiter_targets.append(int(record["target"]))
        output[view] = {
            "sql_texts": sqls, "graph": _pad_graph(graphs),
            "cand_names": names,
            "cand_feats": torch.tensor(features, dtype=torch.float32),
            "cand_mask": torch.tensor(masks, dtype=torch.float32),
            "state_feats": torch.tensor(states, dtype=torch.float32),
            "proposer_actions": torch.tensor(actions, dtype=torch.long),
            "proposer_action_mask": torch.tensor(action_masks, dtype=torch.float32),
            "arbiter_targets": torch.tensor(arbiter_targets, dtype=torch.long),
        }
    return output
