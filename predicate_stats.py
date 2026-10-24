"""Predicate selectivity from a read-only PostgreSQL statistics snapshot."""
from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
import os
from datetime import date, timedelta
from typing import Any

import sqlglot
from sqlglot import exp

from structured_sql import Predicate


def connect(db: str):
    import psycopg2
    return psycopg2.connect(host=os.getenv("PGHOST", "127.0.0.1"),
                            port=int(os.getenv("PGPORT", "55432")),
                            user=os.getenv("PGUSER", "postgres"),
                            password=os.getenv("PGPASSWORD", "postgres"),
                            dbname=db, connect_timeout=10)


def _parse_pg_array(text, cast=str):
    if not text:
        return []
    values = next(csv.reader(io.StringIO(text[1:-1]), escapechar="\\"))
    result = []
    for value in values:
        try:
            result.append(cast(value))
        except (ValueError, TypeError):
            result.append(value)
    return result


def load_stats(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT c.relname, c.reltuples::bigint FROM pg_class c "
                    "JOIN pg_namespace n ON c.relnamespace=n.oid "
                    "WHERE n.nspname='public' AND c.relkind='r'")
        rows = {table: max(int(count), 1) for table, count in cur.fetchall()}
        cur.execute("SELECT s.tablename,s.attname,s.n_distinct,s.null_frac, "
                    "s.most_common_vals::text,s.most_common_freqs::text, "
                    "s.histogram_bounds::text,format_type(a.atttypid,a.atttypmod) "
                    "FROM pg_stats s JOIN pg_namespace n ON n.nspname=s.schemaname "
                    "JOIN pg_class c ON c.relname=s.tablename AND c.relnamespace=n.oid "
                    "JOIN pg_attribute a ON a.attrelid=c.oid AND a.attname=s.attname "
                    "WHERE s.schemaname='public'")
        columns = {}
        for table, column, ndv, null, mcv, freq, hist, dtype in cur.fetchall():
            columns[(table, column)] = {
                "n_distinct": float(ndv) if ndv is not None else None,
                "null_frac": float(null or 0), "mcv": _parse_pg_array(mcv),
                "mcv_freq": _parse_pg_array(freq, float),
                "histogram": _parse_pg_array(hist), "data_type": dtype}
    serial = {"rows": rows, "columns": {f"{t}.{c}": v for (t, c), v in sorted(columns.items())}}
    version = hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()[:16]
    return {"rows": rows, "columns": columns, "stats_version": version}


def save_stats(stats: dict, path):
    serial = dict(stats)
    serial["columns"] = {f"{t}.{c}": value for (t, c), value in stats["columns"].items()}
    path.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")


def read_stats(path):
    stats = json.loads(path.read_text(encoding="utf-8"))
    stats["columns"] = {tuple(name.split(".", 1)): value for name, value in stats["columns"].items()}
    return stats


def _scalar(value):
    if isinstance(value, (float, int, date)):
        return value.toordinal() if isinstance(value, date) else float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        try:
            return float(date.fromisoformat(str(value).split(" ")[0]).toordinal())
        except ValueError:
            return None


def _constant(node):
    if isinstance(node, exp.Literal):
        if node.is_string:
            try:
                return date.fromisoformat(node.this)
            except ValueError:
                return node.this
        return float(node.this)
    if isinstance(node, exp.Neg):
        value = _constant(node.this)
        return -value if isinstance(value, (float, int)) else None
    if isinstance(node, exp.Paren):
        return _constant(node.this)
    if isinstance(node, exp.Cast):
        value = _constant(node.this)
        target = node.args["to"].sql().lower()
        if "date" in target and value is not None:
            return value if isinstance(value, date) else date.fromisoformat(str(value))
        return value
    if isinstance(node, exp.Interval):
        value = _constant(node.this)
        unit = str(node.args.get("unit", "DAY")).lower()
        # PostgreSQL's parser sometimes stores the unit inside the string.
        parts = str(value).split()
        amount = float(parts[0])
        if len(parts) > 1:
            unit = parts[1].lower()
        return (amount, unit)
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
        left, right = _constant(node.left), _constant(node.right)
        if isinstance(left, date) and isinstance(right, tuple):
            amount, unit = right
            sign = -1 if isinstance(node, exp.Sub) else 1
            if unit.startswith("day"):
                return left + timedelta(days=sign * amount)
            if unit.startswith(("month", "year")):
                months = int(amount * (12 if unit.startswith("year") else 1)) * sign
                month = left.year * 12 + left.month - 1 + months
                year, rem = divmod(month, 12)
                import calendar
                return date(year, rem + 1, min(left.day, calendar.monthrange(year, rem + 1)[1]))
        if isinstance(left, (float, int)) and isinstance(right, (float, int)):
            if isinstance(node, exp.Add): return left + right
            if isinstance(node, exp.Sub): return left - right
            if isinstance(node, exp.Mul): return left * right
            if right: return left / right
    return None


def _rhs_values(pred: Predicate):
    try:
        expression = sqlglot.parse_one(pred.raw, read="postgres")
        if isinstance(expression, exp.Not): expression = expression.this
        if isinstance(expression, exp.Between):
            return [_constant(expression.args["low"]), _constant(expression.args["high"])]
        if isinstance(expression, exp.In):
            return [_constant(x) for x in expression.expressions]
        if isinstance(expression, exp.Binary):
            return [_constant(expression.right if isinstance(expression.left, exp.Column) else expression.left)]
    except (sqlglot.errors.ParseError, ValueError, TypeError):
        pass
    value = pred.literal
    return value if isinstance(value, list) else [value]


def _equality(value, col, ndv, null):
    mcv = [str(x) for x in col.get("mcv", [])]
    frequencies = col.get("mcv_freq", [])
    string = value.isoformat() if isinstance(value, date) else str(value)
    if isinstance(value, float) and value.is_integer(): string = str(int(value))
    if string in mcv:
        return float(frequencies[mcv.index(string)]), "mcv"
    mass = max(0.0, 1.0 - null - sum(frequencies))
    return mass / max((ndv or 200) - len(mcv), 1), "n_distinct" if ndv else "default"


def _range(value, op, col, null):
    numeric = _scalar(value)
    histogram = [_scalar(x) for x in col.get("histogram", [])]
    if numeric is None or len(histogram) < 2 or any(x is None for x in histogram):
        return 0.25 * (1 - null), "default_range"
    histogram.sort()
    position = bisect.bisect_left(histogram, numeric)
    if position == 0: cdf = 0.0
    elif position >= len(histogram): cdf = 1.0
    else:
        lo, hi = histogram[position - 1], histogram[position]
        local = (numeric - lo) / (hi - lo) if hi > lo else 0.0
        cdf = (position - 1 + local) / (len(histogram) - 1)
    if op in (">", ">="): cdf = 1 - cdf
    frequencies = col.get("mcv_freq", [])
    mass = cdf * max(0.0, 1 - null - sum(frequencies))
    for val, freq in zip(col.get("mcv", []), frequencies):
        other = _scalar(val)
        if other is None: continue
        keep = {"<": other < numeric, "<=": other <= numeric,
                ">": other > numeric, ">=": other >= numeric}[op]
        if keep: mass += float(freq)
    return mass, "histogram"


def estimate_predicate(pred: Predicate, stats: dict) -> dict:
    col = stats["columns"].get((pred.table, pred.column), {})
    table_rows = stats["rows"].get(pred.table)
    raw_ndv = col.get("n_distinct")
    ndv = (-raw_ndv * table_rows if raw_ndv is not None and raw_ndv < 0 and table_rows
           else raw_ndv)
    null = float(col.get("null_frac", 0))
    values = _rhs_values(pred)
    op = pred.operator.upper()
    source = "missing_stats"
    fraction = None
    if col:
        if op == "IS NULL": fraction, source = null, "null_frac"
        elif op == "IS NOT NULL": fraction, source = 1 - null, "null_frac"
        elif op in ("=", "<>") and values and values[0] is not None:
            fraction, source = _equality(values[0], col, ndv, null)
            if op == "<>": fraction = max(0.0, 1 - null - fraction)
        elif op in ("<", "<=", ">", ">=") and values and values[0] is not None:
            fraction, source = _range(values[0], op, col, null)
        elif op == "BETWEEN" and len(values) == 2 and all(x is not None for x in values):
            upper, source = _range(values[1], "<=", col, null)
            lower, _ = _range(values[0], "<", col, null)
            fraction = max(0.0, upper - lower)
        elif op in ("IN", "NOT IN") and values and all(x is not None for x in values):
            estimates = [_equality(x, col, ndv, null) for x in dict.fromkeys(values)]
            fraction = min(1 - null, sum(x[0] for x in estimates))
            source = "mcv_in" if all(x[1] == "mcv" for x in estimates) else "ndv_in"
            if op == "NOT IN": fraction = max(0.0, 1 - null - fraction)
        elif op in ("LIKE", "ILIKE", "NOT LIKE", "NOT ILIKE"):
            fraction, source = 0.005 * (1 - null), "default_like"
            if op.startswith("NOT "): fraction = 1 - null - fraction
        if fraction is None:
            fraction, source = 0.005 * (1 - null), "default"
    valid = fraction is not None
    if valid: fraction = min(1.0, max(0.0, fraction))
    return {"predicate_id": pred.predicate_id, "table": pred.table, "column": pred.column,
            "operator": op, "estimated_match_fraction": fraction,
            "estimated_matched_rows": fraction * table_rows if valid and table_rows else None,
            "table_rows": table_rows, "n_distinct": ndv, "null_frac": null if col else None,
            "data_type": col.get("data_type", "unknown"), "source": source,
            "valid": valid, "literal": [x.isoformat() if isinstance(x, date) else x for x in values]}


def annotate_structures(structures_by_qid: dict, stats: dict) -> dict:
    return {qid: [estimate_predicate(p, stats) for p in structure.predicates]
            for qid, structure in structures_by_qid.items()}
