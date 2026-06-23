from datetime import datetime, timedelta

import os
import sqlite3
import pytest


def _fetch_all(connection, query):
    cur = connection.cursor()
    cur.execute(query)
    results = cur.fetchall()
    cur.close()
    return results


def test_init_db_creates_tables(fresh_db):
    fresh_db.init_db()
    # Run again to exercise the early exit path when schema already exists
    fresh_db.init_db()
    conn = sqlite3.connect(fresh_db.DB_NAME)
    tables = _fetch_all(conn, "SELECT name FROM sqlite_master WHERE type='table'")
    assert {"startups", "scrape_log"}.issubset({name for (name,) in tables})
    conn.close()


def test_save_and_retrieve_startups(fresh_db):
    now = datetime.now()
    startup = {
        "name": "AwesomeTool",
        "url": "https://example.com",
        "description": "A great developer helper",
        "source": "GitHub Trending",
        "date_found": now,
    }
    fresh_db.save_startup(startup)

    all_startups = fresh_db.get_all_startups()
    assert len(all_startups) == 1
    stored = all_startups[0]
    assert stored["name"] == "AwesomeTool"
    assert stored["url"] == "https://example.com"
    assert stored["source"] == "GitHub Trending"

    fetched = fresh_db.get_startup_by_id(stored["id"])
    assert fetched["id"] == stored["id"]

    assert fresh_db.get_startups_by_source_key("github")
    assert fresh_db.get_startups_by_source_key("hackernews") == []

    with_offset = fresh_db.get_all_startups(offset=0)
    assert len(with_offset) == len(all_startups)

    limited = fresh_db.get_all_startups(limit=1, offset=0)
    assert len(limited) == 1

    assert fresh_db.count_all_startups() == len(all_startups)

    by_url = fresh_db.get_startup_by_url("https://example.com")
    assert by_url["id"] == stored["id"]


def test_is_duplicate_and_integrity_handling(fresh_db):
    now = datetime.now()
    base = {
        "name": "DupTool",
        "url": "https://duplicated.com",
        "description": "First",
        "source": "HN",
        "date_found": now,
    }
    fresh_db.save_startup(base)
    assert fresh_db.is_duplicate("DupTool", "https://duplicated.com")

    duplicate = dict(base, description="Second insertion")
    fresh_db.save_startup(duplicate)  # Should not raise despite unique constraint

    all_rows = fresh_db.get_all_startups()
    assert len(all_rows) == 1


def test_search_returns_matching_rows(fresh_db):
    now = datetime.now()
    second = now + timedelta(days=1)
    fresh_db.save_startup(
        {
            "name": "Searchable Tool",
            "url": "https://searchable.io",
            "description": "CLI for developers",
            "source": "Product Hunt",
            "date_found": second,
        }
    )
    fresh_db.save_startup(
        {
            "name": "Another App",
            "url": "https://another.app",
            "description": "Not devtools",
            "source": "Other",
            "date_found": now,
        }
    )

    results = fresh_db.search_startups("CLI")
    assert len(results) == 1
    assert results[0]["name"] == "Searchable Tool"

    assert fresh_db.count_search_results("CLI") >= 1
    producthunt = fresh_db.get_startups_by_source_key("producthunt")
    assert producthunt and producthunt[0]["source"] == "Product Hunt"
    paged = fresh_db.search_startups("CLI", limit=1, offset=0)
    assert len(paged) == 1


def test_record_scrape_completion_overwrites_history(fresh_db):
    # No rows yet -> ensure None path covered
    assert fresh_db.get_last_scrape_time() is None

    fresh_db.record_scrape_completion("GitHub Trending")
    first = fresh_db.get_last_scrape_time()
    assert first is not None

    fresh_db.record_scrape_completion("Product Hunt")
    second = fresh_db.get_last_scrape_time()
    assert second != first

    conn = sqlite3.connect(fresh_db.DB_NAME)
    rows = _fetch_all(conn, "SELECT scrapers_run FROM scrape_log")
    conn.close()
    # Only the most recent entry should remain
    assert len(rows) == 1
    assert rows[0][0] == "Product Hunt"


def test_init_db_recovers_from_partial_database(tmp_path, monkeypatch):
    import importlib
    import database

    db_file = tmp_path / "partial.db"
    monkeypatch.setenv("DEVTOOLS_DB_PATH", str(db_file))
    monkeypatch.setenv("DEVTOOLS_DATA_DIR", str(tmp_path))

    importlib.reload(database)

    # Create an existing SQLite file without the expected tables
    conn = sqlite3.connect(database.DB_NAME)
    conn.execute("CREATE TABLE temp_table(id INTEGER)")
    conn.commit()
    conn.close()

    database.init_db()
    conn = sqlite3.connect(database.DB_NAME)
    tables = _fetch_all(conn, "SELECT name FROM sqlite_master WHERE type='table'")
    conn.close()
    assert "startups" in {name for (name,) in tables}


def test_get_startup_by_url_missing(fresh_db):
    assert fresh_db.get_startup_by_url("https://missing.example.com") is None


def test_init_db_handles_initial_connect_failure(tmp_path, monkeypatch):
    import importlib
    import database

    db_file = tmp_path / "retry.db"
    monkeypatch.setenv("DEVTOOLS_DB_PATH", str(db_file))
    monkeypatch.setenv("DEVTOOLS_DATA_DIR", str(tmp_path))

    importlib.reload(database)

    # Pre-create file with content so init_db attempts the validation branch
    db_file.write_text("placeholder")

    original_connect = database.sqlite3.connect
    call_tracker = {"count": 0}

    def flaky_connect(path, *args, **kwargs):
        call_tracker["count"] += 1
        if call_tracker["count"] == 1:
            # Remove the invalid placeholder so the next attempt can create a real DB
            if os.path.exists(path):
                os.remove(path)
            raise database.sqlite3.OperationalError("first attempt fails")
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(database.sqlite3, "connect", flaky_connect)

    database.init_db()
    assert call_tracker["count"] >= 2


def test_get_source_counts(fresh_db):
    fresh_db.save_startup(
        {
            "name": "HN Dev Tool",
            "url": "https://hn.dev",
            "description": "HN source",
            "source": "Hacker News (score: 20)",
            "date_found": datetime.now(),
        }
    )
    fresh_db.save_startup(
        {
            "name": "GitHub Dev Tool",
            "url": "https://github.dev",
            "description": "GH source",
            "source": "GitHub Trending",
            "date_found": datetime.now(),
        }
    )
    fresh_db.save_startup(
        {
            "name": "Product Hunt Dev Tool",
            "url": "https://ph.dev",
            "description": "PH source",
            "source": "Product Hunt",
            "date_found": datetime.now(),
        }
    )
    fresh_db.save_startup(
        {
            "name": "Other Dev Tool",
            "url": "https://other.dev",
            "description": "Misc source",
            "source": "Indie Hackers",
            "date_found": datetime.now(),
        }
    )
    counts = fresh_db.get_source_counts()
    assert counts["total"] >= 1
    assert counts["github"] >= 0
    assert counts["hackernews"] >= 1
    assert counts["producthunt"] >= 1
    assert counts["other"] >= 1

    ph_list = fresh_db.get_startups_by_source_key("producthunt")
    assert any(item["source"] == "Product Hunt" for item in ph_list)
    assert len(fresh_db.get_startups_by_source_key("producthunt", limit=1, offset=0)) == 1
    assert fresh_db.get_startups_by_source_key("producthunt", offset=1) == []

    assert fresh_db.get_startups_by_source_key("unknown") == fresh_db.get_all_startups()
    assert fresh_db.count_startups_by_source_key("producthunt") >= 1
    assert fresh_db.count_startups_by_source_key("github") >= 0
    assert fresh_db.count_startups_by_source_key("hackernews") >= 1
    assert fresh_db.count_startups_by_source_key("unknown") == fresh_db.count_all_startups()


def test_save_startup_rejects_duplicate_name_different_url(fresh_db):
    """save_startup should call is_duplicate and skip INSERT when name matches."""
    now = datetime.now()
    first = {
        "name": "SameName",
        "url": "https://first-url.com",
        "description": "Original",
        "source": "GitHub Trending",
        "date_found": now,
    }
    fresh_db.save_startup(first)
    assert fresh_db.count_all_startups() == 1

    # Same name, different URL -- should be rejected as duplicate
    second = {
        "name": "SameName",
        "url": "https://second-url.com",
        "description": "Duplicate name",
        "source": "GitHub Trending",
        "date_found": now,
    }
    fresh_db.save_startup(second)
    assert fresh_db.count_all_startups() == 1  # Should still be 1


def test_search_startups_empty_query_returns_empty(fresh_db):
    assert fresh_db.search_startups("") == []
    assert fresh_db.count_search_results("") == 0


def test_init_db_rebuild_failure_is_ignored(monkeypatch):
    import database

    class _FakeCursor:
        def execute(self, sql, params=None):
            if sql.strip().startswith("INSERT INTO startups_fts") and "'rebuild'" in sql:
                raise database.sqlite3.OperationalError("rebuild failed")
            return self

        def executescript(self, script):
            return None

    class _FakeConn:
        def __init__(self):
            self.cursor_obj = _FakeCursor()

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(database, "_connect", lambda: _FakeConn())
    database.init_db()  # Should not raise despite rebuild failure


def test_init_db_retry_exhaustion(monkeypatch):
    import database

    def failing_connect():
        raise database.sqlite3.OperationalError("permanent failure")

    monkeypatch.setattr(database, "_connect", failing_connect)

    with pytest.raises(database.sqlite3.OperationalError):
        database.init_db()


# --- SOURCE_REGISTRY and classify_source tests ---

def test_source_registry_keys():
    import database

    assert "github" in database.SOURCE_REGISTRY
    assert "hackernews" in database.SOURCE_REGISTRY
    assert "producthunt" in database.SOURCE_REGISTRY


def test_classify_source_known_sources():
    import database

    assert database.classify_source("GitHub Trending") == "github"
    assert database.classify_source("Hacker News (score: 42)") == "hackernews"
    assert database.classify_source("Show HN (score: 10)") == "hackernews"
    assert database.classify_source("Product Hunt") == "producthunt"


def test_classify_source_unknown_returns_other():
    import database

    assert database.classify_source("Indie Hackers") == "other"
    assert database.classify_source("") == "other"


def test_get_related_startups_matches_hackernews_by_prefix(fresh_db):
    now = datetime.now()
    fresh_db.save_startup(
        {
            "name": "HN Alpha",
            "url": "https://hn-alpha.dev",
            "description": "HN A",
            "source": "Hacker News (score: 11)",
            "date_found": now,
        }
    )
    fresh_db.save_startup(
        {
            "name": "HN Beta",
            "url": "https://hn-beta.dev",
            "description": "HN B",
            "source": "Hacker News (score: 99)",
            "date_found": now,
        }
    )
    fresh_db.save_startup(
        {
            "name": "Show HN Gamma",
            "url": "https://show-hn-gamma.dev",
            "description": "Show HN",
            "source": "Show HN (score: 42)",
            "date_found": now,
        }
    )
    fresh_db.save_startup(
        {
            "name": "PH Delta",
            "url": "https://ph-delta.dev",
            "description": "PH",
            "source": "Product Hunt",
            "date_found": now,
        }
    )

    rows_by_name = {row["name"]: row for row in fresh_db.get_all_startups()}
    target = rows_by_name["HN Alpha"]

    related = fresh_db.get_related_startups(target["source"], target["id"], limit=10)
    related_names = {row["name"] for row in related}

    assert "HN Beta" in related_names
    assert "Show HN Gamma" in related_names
    assert "PH Delta" not in related_names


def test_source_registry_consistency_with_get_source_counts(fresh_db):
    """get_source_counts should use the same classification as classify_source."""
    fresh_db.save_startup({
        "name": "GH Tool", "url": "https://gh.test",
        "description": "d", "source": "GitHub Trending",
        "date_found": datetime.now(),
    })
    fresh_db.save_startup({
        "name": "HN Tool", "url": "https://hn.test",
        "description": "d", "source": "Hacker News (score: 5)",
        "date_found": datetime.now(),
    })
    fresh_db.save_startup({
        "name": "SHN Tool", "url": "https://shn.test",
        "description": "d", "source": "Show HN (score: 3)",
        "date_found": datetime.now(),
    })
    fresh_db.save_startup({
        "name": "PH Tool", "url": "https://ph.test",
        "description": "d", "source": "Product Hunt",
        "date_found": datetime.now(),
    })
    fresh_db.save_startup({
        "name": "Other Tool", "url": "https://other.test",
        "description": "d", "source": "Unknown Source",
        "date_found": datetime.now(),
    })

    counts = fresh_db.get_source_counts()
    assert counts["github"] == 1
    assert counts["hackernews"] == 2  # HN + Show HN
    assert counts["producthunt"] == 1
    assert counts["other"] == 1
    assert counts["total"] == 5


# --- _db_connection context manager tests ---

def test_db_connection_closes_on_success(fresh_db):
    """_db_connection should close the connection after the block."""
    import database

    with database._db_connection() as conn:
        conn.execute("SELECT 1")
    # After exiting, the connection should be closed.
    with pytest.raises(database.sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_db_connection_closes_on_exception(fresh_db):
    """_db_connection should close the connection even if an exception occurs."""
    import database

    with pytest.raises(ValueError):
        with database._db_connection() as conn:
            raise ValueError("boom")
    with pytest.raises(database.sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
