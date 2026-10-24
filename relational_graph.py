"""Relation graph and consistent anonymization for SQL training views."""
from __future__ import annotations

import random
from pathlib import Path
import re
from dataclasses import dataclass, asdict
from typing import Any, Iterable

from structured_sql import SqlStructure, stable_id


@dataclass
class GraphNode:
    node_id: str
    kind: str
    value: str
    attrs: dict[str, Any]


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str


def build_relation_graph(structure: SqlStructure) -> dict[str, Any]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # Query-block nodes: one per scope (select / subquery / cte / union),
    # with parent edges expressing nesting.
    blocks = structure.blocks or [{"id": "qb0", "kind": "select",
                                   "parent": None, "name": "", "depth": 0}]
    block_ids = [b["id"] for b in blocks]
    known = set(block_ids)

    def bid_of(block: str | None) -> str:
        return block if block in known else block_ids[0]

    for b in blocks:
        attrs = {"block_kind": b.get("kind", "select"), "depth": b.get("depth", 0)}
        if b.get("name"):
            attrs["name"] = b["name"]
        nodes.append(GraphNode(b["id"], "query_block", b["id"], attrs))
        if b.get("parent"):
            edges.append(GraphEdge(bid_of(b["parent"]), b["id"], "contains_block"))

    root = block_ids[0]
    cte_names = set(structure.ctes or [])

    table_nodes: dict[str, str] = {}
    col_nodes: dict[str, str] = {}
    for table in structure.tables:
        nid = f"table:{table}"
        table_nodes[table] = nid
        nodes.append(GraphNode(nid, "table", table, {}))
        edges.append(GraphEdge(root, nid, "contains_table"))
    # One table_instance node per alias: self-joins keep distinct instances.
    for alias, table in sorted(structure.aliases.items()):
        nid = f"table_instance:{alias}"
        nodes.append(GraphNode(nid, "table_instance", alias, {"table": table}))
        if table in table_nodes:
            edges.append(GraphEdge(nid, table_nodes[table], "instance_of"))
        elif table in cte_names:
            # Alias of a derived table / CTE: point at its producing block.
            target = next((b["id"] for b in blocks
                           if b.get("name") == table or b["id"] == table), None)
            if target:
                edges.append(GraphEdge(nid, target, "instance_of_block"))
        edges.append(GraphEdge(root, nid, "contains_table_instance"))
    for col in structure.columns:
        nid = f"column:{col}"
        col_nodes[col] = nid
        table = col.split(".", 1)[0] if "." in col else ""
        nodes.append(GraphNode(nid, "column", col, {}))
        if table in table_nodes:
            edges.append(GraphEdge(table_nodes[table], nid, "owns_column"))
    for i, pred in enumerate(structure.predicates):
        nid = f"predicate:{i}"
        nodes.append(GraphNode(nid, "predicate", pred.raw,
                               {"operator": pred.operator,
                                "value_kind": pred.value_kind,
                                "block": bid_of(pred.block)}))
        col = f"{pred.table}.{pred.column}"
        if col in col_nodes:
            edges.append(GraphEdge(nid, col_nodes[col], "applies_to"))
        edges.append(GraphEdge(bid_of(pred.block), nid, "contains_predicate"))
    for i, join in enumerate(structure.joins):
        nid = f"join:{i}"
        nodes.append(GraphNode(nid, "join", "=", {}))
        for side in ("left", "right"):
            if join[side] in col_nodes:
                edges.append(GraphEdge(nid, col_nodes[join[side]], side))
        edges.append(GraphEdge(bid_of(join.get("block")), nid, "contains_join"))
    for i, agg in enumerate(structure.aggregates):
        func, _, arg = agg.partition("(")
        arg = arg.rstrip(")")
        nid = f"aggregate:{i}"
        nodes.append(GraphNode(nid, "aggregate", agg, {"func": func}))
        if arg in col_nodes:
            edges.append(GraphEdge(nid, col_nodes[arg], "uses"))
        edges.append(GraphEdge(root, nid, "contains_aggregate"))
    for kind, values in (("group_by", structure.group_by), ("order_by", structure.order_by)):
        for i, col in enumerate(values):
            nid = f"{kind}:{i}"
            nodes.append(GraphNode(nid, kind, col, {}))
            if col in col_nodes:
                edges.append(GraphEdge(nid, col_nodes[col], "uses"))
    return {"nodes": [asdict(n) for n in nodes], "edges": [asdict(e) for e in edges]}


def make_rename_map(structures: Iterable[SqlStructure], seed: int,
                    extra_symbols: Iterable[str] | None = None,
                    mode: str = "alphabet") -> dict[str, str]:
    """Build a consistent rename map for one anonymous view.

    ``extra_symbols`` closes the candidate-pool leak: a state also displays
    candidate/current index names, which may reference tables and columns
    that never appear in the workload SQL. Symbols are "table" or
    "table.col"; anything not covered here would fall back to the ORIGINAL
    name in the anonymous view (observed: person_info(note) leaked).
    """
    tables: set[str] = set()
    cols: set[str] = set()
    for s in structures:
        tables.update(s.tables)
        tables.update(s.ctes or [])  # CTE / derived-table symbols anonymize too
        cols.update(s.columns)
    for sym in extra_symbols or ():
        if "." in sym:
            t, _, _ = sym.partition(".")
            cols.add(sym)
            tables.add(t)
        else:
            tables.add(sym)
    if mode == "token":
        # vocab-token anonymization (2026-10-16 user decision): names are real
        # tokenizer words, semantically unrelated to the schema and pattern-
        # unguessable (no t1/c1 alphabet ordering to leak). Deterministic per
        # seed; consistent within the workload.
        return _token_rename_map(seed, sorted(tables), sorted(cols),
                                 tok=_ANON_TOKENIZER["tok"])
    rng = random.Random(seed)
    table_names = [f"t{i+1}" for i in range(len(tables))]
    col_names = [f"c{i+1}" for i in range(len(cols))]
    rng.shuffle(table_names)
    rng.shuffle(col_names)
    mapping: dict[str, str] = {}
    for old, new in zip(sorted(tables), table_names):
        mapping[old] = new
    for old, new in zip(sorted(cols), col_names):
        table, _, col = old.partition(".")
        mapping[old] = f"{mapping.get(table, table)}.{new}"
    return mapping


_TOKEN_POOL_CACHE: dict[str, list[str]] = {}
_ANON_TOKENIZER: dict = {}


def set_anon_tokenizer(tok) -> None:
    """Register a loaded tokenizer for token-mode anonymization (cloud-safe)."""
    _ANON_TOKENIZER["tok"] = tok


def _token_pool(seed: str, n: int, tok=None) -> list[str]:
    """Deterministically sample n neutral alpha words from the Qwen vocab.

    Rejections: non-alpha, too short/long, schema-domain stoplist words, and
    any token whose lower form collides with an already-picked word.
    ``tok``: an already-loaded tokenizer (preferred -- avoids env-dependent
    model path lookups that break on cloud hosts)."""
    from transformers import AutoTokenizer
    cache_key = f"{seed}:{n}"
    if cache_key in _TOKEN_POOL_CACHE:
        return _TOKEN_POOL_CACHE[cache_key]
    if tok is None:
        raise ValueError(
            "no tokenizer registered for token-mode anonymization; "
            "call set_anon_tokenizer() first (cloud-safe: the trainer "
            "registers its loaded policy tokenizer)")
    stop = {"title", "name", "keyword", "info", "note", "movie", "person",
            "company", "cast", "link", "aka", "production", "year",
            "lineitem", "order", "orders", "part", "parts", "supp", "supplier",
            "nation", "customer", "region", "id", "user", "date", "type",
            "kind", "role", "complete", "char", "index"}
    rng = random.Random(seed)
    pool: list[str] = []
    seen: set[str] = set()
    tries = 0
    while len(pool) < n and tries < 500000:
        tries += 1
        token_id = rng.randrange(0, len(tok))
        word = tok.decode([token_id]).strip()
        if not (word.isalpha() and word.isascii() and 5 <= len(word) <= 10):
            continue
        lower = word.lower()
        if lower in stop or lower in seen:
            continue
        seen.add(lower)
        pool.append(word)
    _TOKEN_POOL_CACHE[cache_key] = pool
    return pool


def _token_rename_map(seed: int, tables: list[str], cols: list[str],
                      tok=None) -> dict:
    """Assign vocab-token names: tables get whole words, columns get
    f'{table_word}.{column_word}' pairs derived from independent draws."""
    rng = random.Random(seed)
    table_words = _token_pool(f"{seed}:tables", len(tables), tok)
    col_words = _token_pool(f"{seed}:cols", len(cols), tok)
    rng.shuffle(table_words)
    rng.shuffle(col_words)
    mapping: dict[str, str] = {}
    for table, word in zip(sorted(tables), table_words):
        mapping[table] = word
    for old, word in zip(sorted(cols), col_words):
        table, _, col = old.partition(".")
        t_word = mapping.get(table, table)
        mapping[old] = f"{t_word}.{word}"
    return mapping


def anonymize_sql_text(sql: str, mapping: dict[str, str], aliases: dict[str, str]) -> str:
    """Rewrite qualified refs, table names and aliases in SQL text.

    Anonymization only changes symbols: ``alias.col`` / ``table.col`` become
    the mapped ``tX.cY`` refs, and bare table/alias/CTE names become ``tX``.
    """
    alias_to_table = {a.lower(): t for a, t in aliases.items()}

    def repl_qualified(m: re.Match) -> str:
        lhs, col = m.group(1), m.group(2)
        table = alias_to_table.get(lhs.lower(), lhs.lower())
        mapped = mapping.get(f"{table}.{col.lower()}")
        return mapped if mapped else m.group(0)

    text = re.sub(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", repl_qualified, sql)
    bare_names = sorted(set(list(alias_to_table) + [k for k in mapping if "." not in k]),
                        key=len, reverse=True)
    for name in bare_names:
        table = alias_to_table.get(name, name)
        new = mapping.get(table, name)
        if new != name:
            text = re.sub(rf"\b{re.escape(name)}\b", new, text)
    return text


def anonymize_structure(structure: SqlStructure, mapping: dict[str, str]) -> SqlStructure:
    def rename(ref: str) -> str:
        return mapping.get(ref, ref)
    aliases = {a: mapping.get(t, t) for a, t in structure.aliases.items()}
    sql_text = anonymize_sql_text(structure.parameterized_sql, mapping, structure.aliases)
    predicates = []
    for p in structure.predicates:
        ref = rename(f"{p.table}.{p.column}")
        table, _, col = ref.partition(".")
        raw = anonymize_sql_text(p.raw, mapping, structure.aliases)
        predicates.append(type(p)(table, col, p.operator, p.value_kind, raw, p.block))
    joins = [{k: rename(v) for k, v in j.items()} for j in structure.joins]
    blocks = []
    for b in structure.blocks or []:
        b2 = dict(b)
        if b2.get("name"):
            b2["name"] = rename(b2["name"])
        blocks.append(b2)
    return SqlStructure(
        sql_text,
        [mapping.get(t, t) for t in structure.tables], aliases,
        [rename(c) for c in structure.columns], predicates, joins,
        [rename(c) for c in structure.group_by], [rename(c) for c in structure.order_by],
        [anonymize_sql_text(x, mapping, structure.aliases) for x in structure.aggregates],
        structure.query_blocks, blocks,
        [rename(c) for c in (structure.ctes or [])],
        structure.parser)


def anonymize_index(name: str, mapping: dict[str, str]) -> str:
    table, _, cols = name.partition("(")
    columns = [c.strip() for c in cols.rstrip(")").split(",") if c.strip()]
    table_new = mapping.get(table.lower(), table.lower())
    col_new = []
    for col in columns:
        col_new.append(mapping.get(f"{table.lower()}.{col.lower()}", f"{table_new}.{col.lower()}").split(".")[-1])
    return f"{table_new}({', '.join(col_new)})"
