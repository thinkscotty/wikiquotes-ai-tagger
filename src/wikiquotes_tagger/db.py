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
    author_type TEXT,
    religious_sentiment TEXT,
    quality INTEGER,
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
    """Create tables and indexes if they don't exist, then upgrade schema."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    upgrade_schema(conn)
    conn.close()


def upgrade_schema(conn: sqlite3.Connection) -> None:
    """Add new columns if they don't exist (idempotent)."""
    cursor = conn.execute("PRAGMA table_info(quotes)")
    existing = {row[1] for row in cursor.fetchall()}
    if "author_type" not in existing:
        conn.execute("ALTER TABLE quotes ADD COLUMN author_type TEXT")
    if "religious_sentiment" not in existing:
        conn.execute("ALTER TABLE quotes ADD COLUMN religious_sentiment TEXT")
    if "quality" not in existing:
        conn.execute("ALTER TABLE quotes ADD COLUMN quality INTEGER")
    conn.commit()


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


def get_random_untagged_ids(conn: sqlite3.Connection, count: int) -> list[int]:
    """Fetch `count` random quote IDs from the untagged pool.

    Uses a single ORDER BY RANDOM() to pick all IDs at once, avoiding
    duplicates across batches. Returns a list of integer IDs.
    """
    cursor = conn.execute(
        "SELECT id FROM quotes WHERE status = 'parsed' ORDER BY RANDOM() LIMIT ?",
        (count,),
    )
    return [row[0] for row in cursor.fetchall()]


def get_quotes_by_ids(conn: sqlite3.Connection, ids: list[int]) -> list[dict]:
    """Fetch quotes by a list of IDs, preserving the input order."""
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    cursor = conn.execute(
        f"SELECT id, text, author, source_work FROM quotes WHERE id IN ({placeholders})",
        ids,
    )
    rows_by_id = {row["id"]: dict(row) for row in cursor.fetchall()}
    return [rows_by_id[qid] for qid in ids if qid in rows_by_id]


def update_tagged(
    conn: sqlite3.Connection,
    quote_id: int,
    *,
    keywords: list[str],
    categories: list[str],
    batch_id: int,
    author_type: str | None = None,
    religious_sentiment: str | None = None,
) -> bool:
    """Update a single quote with AI-generated tags. Returns True if a row was updated.

    Categories are stored as a JSON array string (e.g., '["Courage", "Philosophy"]').
    """
    cursor = conn.execute(
        "UPDATE quotes SET keywords = ?, category = ?, status = 'tagged', batch_id = ?, "
        "author_type = ?, religious_sentiment = ? "
        "WHERE id = ?",
        (json.dumps(keywords), json.dumps(categories), batch_id,
         author_type, religious_sentiment, quote_id),
    )
    return cursor.rowcount > 0


def get_unscored_batch(conn: sqlite3.Connection, batch_size: int) -> list[dict]:
    """Fetch up to batch_size tagged quotes that have no quality score."""
    cursor = conn.execute(
        "SELECT id, text, author, source_work FROM quotes "
        "WHERE status = 'tagged' AND quality IS NULL ORDER BY id LIMIT ?",
        (batch_size,),
    )
    return [dict(row) for row in cursor.fetchall()]


def get_random_unscored_ids(conn: sqlite3.Connection, count: int) -> list[int]:
    """Fetch `count` random quote IDs from the unscored tagged pool."""
    cursor = conn.execute(
        "SELECT id FROM quotes WHERE status = 'tagged' AND quality IS NULL "
        "ORDER BY RANDOM() LIMIT ?",
        (count,),
    )
    return [row[0] for row in cursor.fetchall()]


def update_quality(conn: sqlite3.Connection, quote_id: int, quality: int) -> bool:
    """Set the quality score for a single quote. Returns True if a row was updated."""
    cursor = conn.execute(
        "UPDATE quotes SET quality = ? WHERE id = ?",
        (quality, quote_id),
    )
    return cursor.rowcount > 0


def reset_scores(conn: sqlite3.Connection) -> int:
    """Reset all quality scores to NULL. Returns count of rows reset."""
    count = conn.execute("SELECT COUNT(*) FROM quotes WHERE quality IS NOT NULL").fetchone()[0]
    if count > 0:
        conn.execute("UPDATE quotes SET quality = NULL WHERE quality IS NOT NULL")
        conn.commit()
    return count


def get_scoring_stats(conn: sqlite3.Connection) -> dict:
    """Return scoring statistics: scored/unscored counts, average, distribution."""
    tagged = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'tagged'").fetchone()[0]
    scored = conn.execute(
        "SELECT COUNT(*) FROM quotes WHERE status = 'tagged' AND quality IS NOT NULL"
    ).fetchone()[0]
    unscored = tagged - scored

    avg_quality = None
    distribution: dict[int, int] = {}
    if scored > 0:
        avg_quality = conn.execute(
            "SELECT AVG(quality) FROM quotes WHERE quality IS NOT NULL"
        ).fetchone()[0]
        cursor = conn.execute(
            "SELECT quality, COUNT(*) as cnt FROM quotes "
            "WHERE quality IS NOT NULL GROUP BY quality ORDER BY quality"
        )
        distribution = {row[0]: row[1] for row in cursor.fetchall()}

    return {
        "tagged": tagged,
        "scored": scored,
        "unscored": unscored,
        "avg_quality": round(avg_quality, 1) if avg_quality is not None else None,
        "distribution": distribution,
    }


def reset_tagged(conn: sqlite3.Connection) -> tuple[int, int]:
    """Reset all tagged and errored quotes to 'parsed' for re-tagging.

    Returns (tagged_count, errored_count) of rows reset.
    """
    tagged = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'tagged'").fetchone()[0]
    errored = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'error'").fetchone()[0]

    if tagged > 0:
        conn.execute(
            "UPDATE quotes SET status = 'parsed', keywords = NULL, category = NULL, "
            "batch_id = NULL, author_type = NULL, religious_sentiment = NULL "
            "WHERE status = 'tagged'"
        )
    if errored > 0:
        conn.execute(
            "UPDATE quotes SET status = 'parsed', keywords = NULL, category = NULL, "
            "batch_id = NULL, author_type = NULL, religious_sentiment = NULL "
            "WHERE status = 'error'"
        )

    conn.commit()
    return tagged, errored


def next_batch_id(conn: sqlite3.Connection) -> int:
    """Get the next batch_id (max existing + 1, or 1 if none)."""
    cursor = conn.execute("SELECT COALESCE(MAX(batch_id), 0) + 1 FROM quotes")
    return cursor.fetchone()[0]


def get_stats(conn: sqlite3.Connection) -> dict:
    """Return counts: total, parsed, tagged, errored, non_english, plus top 10 categories."""
    total = conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
    parsed = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'parsed'").fetchone()[0]
    tagged = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'tagged'").fetchone()[0]
    errored = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'error'").fetchone()[0]
    non_english = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'non_english'").fetchone()[0]

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
        "non_english": non_english,
        "top_categories": top_categories,
    }
