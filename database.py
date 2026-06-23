"""SQLite persistence layer with FTS5 full-text search for developer tools."""

import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from logging_config import get_logger

logger = get_logger("devtools.db")

# FTS5 special-character and keyword patterns for query sanitization
_FTS5_OPERATORS = re.compile(r'["\*\(\)\+\-\^:/\{\}]')
_FTS5_KEYWORDS = re.compile(r'\b(AND|OR|NOT|NEAR)\b', re.IGNORECASE)


def _sanitize_fts_query(query: str) -> str:
    """Strip FTS5 operators from a query string to prevent OperationalError."""
    cleaned = _FTS5_OPERATORS.sub(' ', query)
    cleaned = _FTS5_KEYWORDS.sub(' ', cleaned)
    return ' '.join(cleaned.split())


SOURCE_REGISTRY: dict[str, dict] = {
    "github": {
        "display": "GitHub Trending",
        "where": "source = ?",
        "params": ["GitHub Trending"],
        "match": lambda s: s == "GitHub Trending",
    },
    "hackernews": {
        "display": "Hacker News",
        "where": "source LIKE ? OR source LIKE ?",
        "params": ["Hacker News%", "Show HN%"],
        "match": lambda s: s.startswith("Hacker News") or s.startswith("Show HN"),
    },
    "producthunt": {
        "display": "Product Hunt",
        "where": "source = ?",
        "params": ["Product Hunt"],
        "match": lambda s: s == "Product Hunt",
    },
}


def classify_source(source: str) -> str:
    """Classify a source string into a registry key, or 'other'."""
    if not source:
        return "other"
    for key, info in SOURCE_REGISTRY.items():
        if info["match"](source):
            return key
    return "other"

DEFAULT_DATA_DIR = Path(os.getcwd()) / "data"
DATA_DIR = Path(os.getenv("DEVTOOLS_DATA_DIR", DEFAULT_DATA_DIR))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.getenv("DEVTOOLS_DB_PATH", DATA_DIR / "startups.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DB_NAME = str(DB_PATH)


def _connect() -> sqlite3.Connection:
    """Create a new SQLite connection with Row factory enabled."""
    start = time.perf_counter()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    duration = round((time.perf_counter() - start) * 1000, 2)
    logger.debug(
        "db.connect",
        extra={"event": "db.connect", "duration_ms": duration, "db_path": str(DB_PATH)},
    )
    return conn


@contextmanager
def _db_connection() -> Iterator[sqlite3.Connection]:
    """Open a database connection and ensure it is closed after use."""
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def _append_pagination(query: str, params: list, limit: Optional[int], offset: Optional[int]) -> tuple[str, list]:
    """Append LIMIT/OFFSET clauses to a SQL query string, returning the updated query and params."""
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    if offset is not None:
        if limit is None:
            query += " LIMIT -1"
        query += " OFFSET ?"
        params.append(offset)
    return query, params


def init_db() -> None:
    """Initialize the database schema, creating tables and indices if needed.

    Retries once on OperationalError to handle transient filesystem issues.
    """
    logger.info("db.init.start", extra={"event": "db.init.start", "db_path": str(DB_PATH)})
    conn = None
    for attempt in range(2):
        try:
            conn = _connect()
            break
        except sqlite3.OperationalError:
            if attempt == 1:
                raise
            continue

    try:
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS startups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT UNIQUE,
                description TEXT,
                source TEXT,
                date_found TIMESTAMP
            )
        ''')

        # Add index on name for faster duplicate checking
        c.execute('CREATE INDEX IF NOT EXISTS idx_startups_name ON startups(name)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_startups_source ON startups(source)')

        # Create table for tracking last scrape time
        c.execute('''
            CREATE TABLE IF NOT EXISTS scrape_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_scrape TIMESTAMP NOT NULL,
                scrapers_run TEXT
            )
        ''')

        # Create FTS index for fast search
        c.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS startups_fts
            USING fts5(name, description, content='startups', content_rowid='id')
        ''')
        c.executescript('''
            CREATE TRIGGER IF NOT EXISTS startups_ai AFTER INSERT ON startups BEGIN
                INSERT INTO startups_fts(rowid, name, description) VALUES (new.id, new.name, new.description);
            END;
            CREATE TRIGGER IF NOT EXISTS startups_ad AFTER DELETE ON startups BEGIN
                INSERT INTO startups_fts(startups_fts, rowid, name, description) VALUES('delete', old.id, old.name, old.description);
            END;
            CREATE TRIGGER IF NOT EXISTS startups_au AFTER UPDATE ON startups BEGIN
                INSERT INTO startups_fts(startups_fts, rowid, name, description) VALUES('delete', old.id, old.name, old.description);
                INSERT INTO startups_fts(rowid, name, description) VALUES (new.id, new.name, new.description);
            END;
        ''')
        try:
            c.execute("INSERT INTO startups_fts(startups_fts) VALUES('rebuild')")
        except sqlite3.OperationalError:
            # Rebuild may fail if table is empty or FTS not initialised yet; safe to ignore
            pass

        conn.commit()
    finally:
        conn.close()
    logger.info("db.init.complete", extra={"event": "db.init.complete"})


def is_duplicate(name: str, url: str) -> bool:
    """Check if a startup already exists by name or URL."""
    with _db_connection() as conn:
        cursor = conn.cursor()
        if url is None:
            cursor.execute(
                "SELECT COUNT(*) FROM startups WHERE name = ? OR url IS NULL",
                (name,),
            )
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM startups WHERE name = ? OR url = ?",
                (name, url),
            )
        count = cursor.fetchone()[0]
    logger.debug(
        "db.is_duplicate",
        extra={
            "event": "db.is_duplicate",
            "startup_name": name,
            "url": url,
            "is_duplicate": count > 0,
        },
    )
    return count > 0


def save_startup(startup: Dict[str, Any]) -> None:
    """Persist a startup record, skipping duplicates by name or URL."""
    if is_duplicate(startup['name'], startup['url']):
        logger.warning(
            "db.startup_duplicate",
            extra={
                "event": "db.startup_duplicate",
                "startup_name": startup.get('name'),
                "url": startup.get('url'),
                "reason": "name_or_url_match",
            },
        )
        return

    with _db_connection() as conn:
        c = conn.cursor()
        try:
            c.execute('''
                INSERT INTO startups (name, url, description, source, date_found)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                startup['name'],
                startup['url'],
                startup['description'],
                startup['source'],
                startup['date_found']
            ))
            conn.commit()
            logger.info(
                "db.startup_saved",
                extra={
                    "event": "db.startup_saved",
                    "startup_name": startup.get('name'),
                    "url": startup.get('url'),
                    "source": startup.get('source'),
                },
            )
        except sqlite3.IntegrityError:
            # URL already exists (race condition fallback)
            logger.warning(
                "db.startup_duplicate",
                extra={
                    "event": "db.startup_duplicate",
                    "startup_name": startup.get('name'),
                    "url": startup.get('url'),
                },
            )


def get_startup_by_id(startup_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single startup by its primary key."""
    with _db_connection() as conn:
        row = conn.execute(
            '''
            SELECT id, name, url, description, source, date_found
            FROM startups WHERE id = ?
            ''',
            (startup_id,),
        ).fetchone()
    result = dict(row) if row else None
    logger.debug(
        "db.get_startup_by_id",
        extra={
            "event": "db.get_startup_by_id",
            "startup_id": startup_id,
            "found": bool(result),
        },
    )
    return result


def get_related_startups(source: str, exclude_id: int, limit: int = 4) -> list[Dict[str, Any]]:
    """Fetch startups from the same source, excluding a given ID."""
    source_key = classify_source(source)
    entry = SOURCE_REGISTRY.get(source_key)
    if entry:
        where_clause = entry["where"]
        params = list(entry["params"])
    else:
        where_clause = "source = ?"
        params = [source]

    query = f'''
        SELECT id, name, url, description, source, date_found
        FROM startups
        WHERE ({where_clause}) AND id != ?
        ORDER BY date_found DESC
        LIMIT ?
    '''

    query_params = [*params, exclude_id, limit]

    with _db_connection() as conn:
        rows = conn.execute(query, query_params).fetchall()
    results = [dict(row) for row in rows]
    logger.debug(
        "db.get_related_startups",
        extra={
            "event": "db.get_related_startups",
            "source": source,
            "source_key": source_key,
            "exclude_id": exclude_id,
            "limit": limit,
            "returned": len(results),
        },
    )
    return results


_ALLOWED_WHERE_CLAUSES = frozenset(entry["where"] for entry in SOURCE_REGISTRY.values())


def get_startups_by_sources(where_clause: str, params: Iterable, limit: Optional[int] = None, offset: Optional[int] = None) -> list[Dict[str, Any]]:
    """Query startups matching a dynamic WHERE clause with optional pagination."""
    if where_clause not in _ALLOWED_WHERE_CLAUSES:
        raise ValueError(f"Disallowed where_clause: {where_clause!r}")
    query = '''
        SELECT id, name, url, description, source, date_found
        FROM startups
        WHERE {} ORDER BY date_found DESC
    '''.format(where_clause)

    args = list(params)
    query, args = _append_pagination(query, args, limit, offset)

    with _db_connection() as conn:
        rows = conn.execute(query, args).fetchall()
    results = [dict(row) for row in rows]
    logger.debug(
        "db.get_startups_by_sources",
        extra={
            "event": "db.get_startups_by_sources",
            "where": where_clause,
            "limit": limit,
            "offset": offset,
            "returned": len(results),
        },
    )
    return results


def count_startups_by_sources(where_clause: str, params: Iterable) -> int:
    """Count startups matching a dynamic WHERE clause."""
    if where_clause not in _ALLOWED_WHERE_CLAUSES:
        raise ValueError(f"Disallowed where_clause: {where_clause!r}")
    query = 'SELECT COUNT(*) FROM startups WHERE {}'.format(where_clause)
    with _db_connection() as conn:
        (count,) = conn.execute(query, params).fetchone()
    logger.debug(
        "db.count_startups_by_sources",
        extra={
            "event": "db.count_startups_by_sources",
            "where": where_clause,
            "count": count,
        },
    )
    return count


def get_startups_by_source_key(source_key: str, limit: Optional[int] = None, offset: Optional[int] = None) -> list[Dict[str, Any]]:
    """Fetch startups for a named source key (github, hackernews, producthunt) with pagination."""
    entry = SOURCE_REGISTRY.get(source_key)
    if entry:
        results = get_startups_by_sources(entry["where"], entry["params"], limit, offset)
    else:
        results = get_all_startups(limit, offset)
    logger.debug(
        "db.get_startups_by_source_key",
        extra={
            "event": "db.get_startups_by_source_key",
            "source_key": source_key,
            "limit": limit,
            "offset": offset,
            "returned": len(results),
        },
    )
    return results


def count_startups_by_source_key(source_key: str) -> int:
    """Count startups for a named source key."""
    entry = SOURCE_REGISTRY.get(source_key)
    if entry:
        count = count_startups_by_sources(entry["where"], entry["params"])
    else:
        count = count_all_startups()
    logger.debug(
        "db.count_startups_by_source_key",
        extra={
            "event": "db.count_startups_by_source_key",
            "source_key": source_key,
            "count": count,
        },
    )
    return count


def get_source_counts() -> Dict[str, int]:
    """Aggregate startup counts grouped by source category."""
    with _db_connection() as conn:
        rows = conn.execute("SELECT source, COUNT(*) as count FROM startups GROUP BY source").fetchall()

    summary: Dict[str, int] = {"total": 0, "other": 0}
    for key in SOURCE_REGISTRY:
        summary[key] = 0

    for row in rows:
        source = row["source"]
        count = row["count"]
        summary["total"] += count
        bucket = classify_source(source)
        summary[bucket] += count
    logger.debug(
        "db.get_source_counts",
        extra={"event": "db.get_source_counts", "summary": summary},
    )
    return summary


def get_all_startups(limit: Optional[int] = None, offset: Optional[int] = None) -> list[Dict[str, Any]]:
    """Fetch all startups ordered by date, with optional pagination."""
    query = '''
        SELECT id, name, url, description, source, date_found
        FROM startups ORDER BY date_found DESC
    '''
    params: list = []
    query, params = _append_pagination(query, params, limit, offset)

    with _db_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    results = [dict(row) for row in rows]
    logger.debug(
        "db.get_all_startups",
        extra={
            "event": "db.get_all_startups",
            "limit": limit,
            "offset": offset,
            "returned": len(results),
        },
    )
    return results


def get_existing_startup_keys() -> list[Dict[str, str]]:
    """Return existing startup name/url pairs for fast duplicate pre-filtering."""
    with _db_connection() as conn:
        rows = conn.execute('SELECT name, url FROM startups').fetchall()

    results = [{"name": row["name"], "url": row["url"]} for row in rows]
    logger.debug(
        "db.get_existing_startup_keys",
        extra={"event": "db.get_existing_startup_keys", "returned": len(results)},
    )
    return results


def count_all_startups() -> int:
    """Return the total number of startups in the database."""
    with _db_connection() as conn:
        (count,) = conn.execute('SELECT COUNT(*) FROM startups').fetchone()
    logger.debug(
        "db.count_all_startups",
        extra={"event": "db.count_all_startups", "count": count},
    )
    return count


def search_startups(query: str, limit: int = 20, offset: int = 0) -> list[Dict[str, Any]]:
    """Search startups using FTS5 full-text search."""
    if not query:
        return []

    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return []

    with _db_connection() as conn:
        rows = conn.execute(
            '''
            SELECT s.id, s.name, s.url, s.description, s.source, s.date_found
            FROM startups s
            JOIN startups_fts fts ON s.id = fts.rowid
            WHERE startups_fts MATCH ?
            ORDER BY rank
            LIMIT ? OFFSET ?
            ''',
            (sanitized, limit, offset),
        ).fetchall()

    results = [dict(row) for row in rows]
    logger.debug(
        "db.search_startups",
        extra={
            "event": "db.search_startups",
            "query": query,
            "limit": limit,
            "offset": offset,
            "returned": len(results),
        },
    )
    return results


def count_search_results(query: str) -> int:
    """Count FTS matches for the given query string."""
    if not query:
        return 0
    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return 0
    with _db_connection() as conn:
        (count,) = conn.execute(
            '''
            SELECT COUNT(*) FROM startups_fts
            WHERE startups_fts MATCH ?
            ''',
            (sanitized,),
        ).fetchone()
    logger.debug(
        "db.count_search_results",
        extra={"event": "db.count_search_results", "query": query, "count": count},
    )
    return count


def get_startup_by_url(url: str) -> Optional[Dict[str, Any]]:
    """Fetch a single startup by its URL."""
    with _db_connection() as conn:
        row = conn.execute('''
            SELECT id, name, url, description, source, date_found
            FROM startups WHERE url = ?
        ''', (url,)).fetchone()

    result = dict(row) if row else None
    logger.debug(
        "db.get_startup_by_url",
        extra={"event": "db.get_startup_by_url", "url": url, "found": bool(result)},
    )
    return result


def record_scrape_completion(scrapers_run: Optional[str] = None) -> None:
    """Record that a scrape has completed, replacing any previous entry."""
    with _db_connection() as conn:
        c = conn.cursor()
        # Clear old entries and insert new one
        c.execute('DELETE FROM scrape_log')
        c.execute('''
            INSERT INTO scrape_log (last_scrape, scrapers_run)
            VALUES (?, ?)
        ''', (datetime.now().isoformat(), scrapers_run))
        conn.commit()
    logger.info(
        "db.scrape_log_updated",
        extra={
            "event": "db.scrape_log_updated",
            "scrapers_run": scrapers_run,
        },
    )


def get_last_scrape_time() -> Optional[str]:
    """Return the ISO timestamp of the most recent completed scrape."""
    with _db_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT last_scrape FROM scrape_log ORDER BY id DESC LIMIT 1')
        row = c.fetchone()

    result = row[0] if row else None
    logger.debug(
        "db.get_last_scrape_time",
        extra={"event": "db.get_last_scrape_time", "last_scrape": result},
    )
    return result
