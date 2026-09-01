"""Postgres connection for the credit ledger.

Reachable from BOTH runtimes on purpose. The Modal containers that spend the
money connect to Neon directly rather than calling back through a Vercel
function: a spawned container that cannot reach the ledger means work done for
free with no record, and adding a cold-start-prone hop to the one path where
failure is unrecoverable is the wrong trade.

Unconfigured is a first-class state. With no ``DATABASE_URL`` every accessor
returns ``None``/raises ``LedgerUnavailable`` rather than crashing at import,
mirroring how ``oreos/jobs.py`` degrades to a 503 instead of taking the process
down. That is what lets the whole billing layer ship dark.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

# Neon's pooled endpoint runs PgBouncer in transaction mode, which is
# incompatible with psycopg's automatic prepared statements. Setting the
# threshold to None disables them. Using the direct (non-pooler) host instead
# would work but burns a Neon connection per container, and Modal fans out.
_CONNECT_KWARGS: dict[str, Any] = {"prepare_threshold": None, "autocommit": True}

_pool = None
_pool_lock = threading.Lock()
_unavailable_logged = False


class LedgerUnavailable(RuntimeError):
    """No ledger configured, or it cannot be reached.

    Callers decide the posture: the advisory access gate swallows this and lets
    the request through (it is defence in depth, not the real gate), while
    anything that would spend money must refuse."""


def database_url() -> Optional[str]:
    url = os.environ.get("DATABASE_URL", "").strip()
    return url or None


def is_configured() -> bool:
    return database_url() is not None


def get_pool():
    """Lazy connection pool, or None when unconfigured. Safe to call per request."""
    global _pool, _unavailable_logged
    url = database_url()
    if url is None:
        if not _unavailable_logged:
            _unavailable_logged = True
            print("[billing] DATABASE_URL not set — credit ledger disabled")
        return None

    if _pool is None:
        with _pool_lock:
            if _pool is None:
                try:
                    from psycopg_pool import ConnectionPool

                    _pool = ConnectionPool(
                        url,
                        min_size=int(os.environ.get("BILLING_DB_MIN_CONNS", "1")),
                        max_size=int(os.environ.get("BILLING_DB_MAX_CONNS", "4")),
                        kwargs=_CONNECT_KWARGS,
                        open=True,
                        # A worker that cannot reach the ledger should fail fast
                        # and let the caller decide, not hang the frame loop.
                        timeout=float(os.environ.get("BILLING_DB_TIMEOUT_S", "5")),
                    )
                except Exception as exc:
                    print(f"[billing] connection pool init failed: {exc}")
                    return None
    return _pool


def call(sql: str, params: tuple = (), *, fetch: str = "one"):
    """Run one SQL statement against the ledger.

    ``fetch`` is ``one`` (first row), ``all`` (every row), or ``none``. Raises
    ``LedgerUnavailable`` when there is no pool; every other database error
    propagates so the caller can distinguish "no ledger" from "the ledger said
    no", which are opposite situations.
    """
    pool = get_pool()
    if pool is None:
        raise LedgerUnavailable("credit ledger is not configured")

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if fetch == "none":
                return None
            if fetch == "all":
                return cur.fetchall()
            return cur.fetchone()


def reset_pool_for_tests() -> None:
    """Drop the cached pool so a test can repoint DATABASE_URL."""
    global _pool, _unavailable_logged
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.close()
            except Exception:
                pass
        _pool = None
        _unavailable_logged = False


def apply_migrations(conn=None) -> None:
    """Apply the SQL in ``migrations/`` in filename order.

    Idempotent — every statement is CREATE ... IF NOT EXISTS or CREATE OR
    REPLACE — so this doubles as the test-fixture setup and as the one-time
    operator step against Neon.
    """
    import pathlib

    here = pathlib.Path(__file__).parent / "migrations"
    files = sorted(here.glob("*.sql"))
    if not files:
        raise RuntimeError(f"no migrations found in {here}")

    sql = "\n".join(f.read_text() for f in files)

    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(sql)
        return

    pool = get_pool()
    if pool is None:
        raise LedgerUnavailable("credit ledger is not configured")
    with pool.connection() as c:
        with c.cursor() as cur:
            cur.execute(sql)
