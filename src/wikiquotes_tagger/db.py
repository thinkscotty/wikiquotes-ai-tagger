"""SQLite database schema and query helpers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS quotes (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    author TEXT NOT NULL,
    source_work TEXT,
    source_confidence TEXT,
    keywords TEXT,
    category TEXT,
    status TEXT NOT NULL DEFAULT 'parsed',
    batch_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_quotes_text_author ON quotes(text, author);
CREATE INDEX IF NOT EXISTS idx_quotes_status ON quotes(status);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and row factory."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    """Create tables and indexes if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.close()


def insert_quote(
    conn: sqlite3.Connection,
    *,
    text: str,
    author: str,
    source_work: str | None = None,
    source_confidence: str | None = None,
) -> bool:
    """Insert a quote, returning True on success, False on duplicate."""
    cursor = conn.execute(
        "INSERT OR IGNORE INTO quotes (text, author, source_work, source_confidence) "
        "VALUES (?, ?, ?, ?)",
        (text, author, source_work, source_confidence),
    )
    return cursor.rowcount > 0


def get_untagged_batch(conn: sqlite3.Connection, batch_size: int) -> list[dict]:
    """Fetch up to batch_size quotes WHERE status='parsed', ordered by id."""
    cursor = conn.execute(
        "SELECT id, text, author, source_work FROM quotes "
        "WHERE status = 'parsed' ORDER BY id LIMIT ?",
        (batch_size,),
    )
    return [dict(row) for row in cursor.fetchall()]


def update_tagged(
    conn: sqlite3.Connection,
    quote_id: int,
    *,
    keywords: list[str],
    category: str,
    batch_id: int,
) -> bool:
    """Update a single quote with AI-generated tags. Returns True if a row was updated."""
    cursor = conn.execute(
        "UPDATE quotes SET keywords = ?, category = ?, status = 'tagged', batch_id = ? "
        "WHERE id = ?",
        (json.dumps(keywords), category, batch_id, quote_id),
    )
    return cursor.rowcount > 0


def next_batch_id(conn: sqlite3.Connection) -> int:
    """Get the next batch_id (max existing + 1, or 1 if none)."""
    cursor = conn.execute("SELECT COALESCE(MAX(batch_id), 0) + 1 FROM quotes")
    return cursor.fetchone()[0]


def get_stats(conn: sqlite3.Connection) -> dict:
    """Return counts: total, parsed, tagged, errored, plus top 10 categories."""
    total = conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
    parsed = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'parsed'").fetchone()[0]
    tagged = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'tagged'").fetchone()[0]
    errored = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'error'").fetchone()[0]

    top_categories: list[tuple[str, int]] = []
    if tagged > 0:
        cursor = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM quotes "
            "WHERE status = 'tagged' AND category IS NOT NULL "
            "GROUP BY category ORDER BY cnt DESC LIMIT 10"
        )
        top_categories = [(row[0], row[1]) for row in cursor.fetchall()]

    return {
        "total": total,
        "parsed": parsed,
        "tagged": tagged,
        "errored": errored,
        "top_categories": top_categories,
    }
