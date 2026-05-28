"""
utils/db_helper.py
Provides a ready-to-use SQLite connection that works on virtiofs-mounted
workspaces (OneDrive, etc.) by copying the DB to /tmp before opening it.

Usage:
    from utils.db_helper import get_db_connection
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM renewal_risk_suppliers").fetchall()
    conn.close()
"""

import shutil
import sqlite3
import tempfile
from pathlib import Path

from utils.config import SQLITE_PATH
from utils.logger import logger

# Works on Windows (%TEMP%), macOS (/var/folders/…), and Linux (/tmp)
TMP_DB = Path(tempfile.gettempdir()) / "procurement_runtime.db"


def get_db_connection(refresh: bool = False) -> sqlite3.Connection:
    """
    Return a sqlite3.Connection pointed at a /tmp copy of the workspace DB.

    Args:
        refresh: If True, re-copy the workspace DB to /tmp even if one exists.
                 Use this when data has been updated.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    if refresh or not TMP_DB.exists() or TMP_DB.stat().st_size == 0:
        if not SQLITE_PATH.exists() or SQLITE_PATH.stat().st_size == 0:
            raise FileNotFoundError(
                f"SQLite DB not found or empty at {SQLITE_PATH}. "
                "Run `python -m ingestion.structured_ingester` first."
            )
        data = SQLITE_PATH.read_bytes()
        TMP_DB.write_bytes(data)
        logger.debug(f"DB copied to /tmp ({len(data):,} bytes)")

    conn = sqlite3.connect(str(TMP_DB))
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params: tuple = (), refresh: bool = False) -> list[dict]:
    """
    Convenience one-shot query. Returns list of dicts.

    Example:
        rows = query("SELECT * FROM renewal_risk_suppliers")
    """
    conn = get_db_connection(refresh=refresh)
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
