"""PostgreSQL access for FuseIA: connections, EXPLAIN, HypoPG index sets.

Self-contained on purpose: FuseIA does not import the legacy advisor codebase.
Index-set notation follows the legacy convention ``table(col1,col2)`` so that
results collected under either project stay comparable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence, Tuple

import psycopg2

DEFAULT_DB = "indexselection_tpch___10"


def connect(db_id: str = DEFAULT_DB):
    """Connect using the shared environment contract (PGHOST/PGPORT/PGUSER/PGPASSWORD)."""

    parameters = {
        "host": os.getenv("PGHOST", "127.0.0.1"),
        "port": int(os.getenv("PGPORT", "55432")),
        "database": db_id,
        "user": os.getenv("PGUSER", "postgres"),
    }
    password = os.getenv("PGPASSWORD")
    if password:
        parameters["password"] = password
    return psycopg2.connect(**parameters)


def explain_json(cursor, sql: str) -> Dict[str, Any]:
    """Run EXPLAIN (FORMAT JSON) and return the top-level plan node dict.

    No ANALYZE: the query is planned, never executed, so this is safe to call
    against large workload tables at data-collection time.
    """

    cursor.execute("EXPLAIN (FORMAT JSON) " + sql)
    result = cursor.fetchall()
    return result[0][0][0]["Plan"]


def hypopg_reset(cursor, conn) -> None:
    cursor.execute("SELECT hypopg_reset();")
    conn.commit()


def hypopg_load(cursor, conn, index_set: Sequence[str]) -> None:
    """Reset HypoPG and create one hypothetical index set (plan-only, no writes)."""

    hypopg_reset(cursor, conn)
    for index_text in index_set:
        cursor.execute(
            "SELECT hypopg_create_index('CREATE INDEX ON %s');" % index_text
        )
    conn.commit()


def format_index(table: str, columns: Sequence[str]) -> str:
    """Render an index in the shared ``table(col1,col2)`` notation."""

    return f"{table}({','.join(columns)})"


def parse_index(index_text: str) -> Tuple[str, List[str]]:
    """Parse ``table(col1,col2)`` into ``(table, [columns])``."""

    text = index_text.strip()
    if "(" not in text or not text.endswith(")"):
        raise ValueError(f"bad index notation: {index_text!r}")
    table, _, rest = text.partition("(")
    columns = [c.strip() for c in rest[:-1].split(",") if c.strip()]
    return table.strip(), columns
