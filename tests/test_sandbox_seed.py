"""Tests for sandbox seed validation and restore behavior.

These tests build temporary SQLite databases, package them as gzip fixtures,
and verify restore refuses accidental overwrites while still supporting forced resets.
"""

from __future__ import annotations

import gzip
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import create_sandbox_seed  # noqa: E402
import restore_sandbox_db  # noqa: E402


def _create_valid_database(database_path: Path) -> None:
    assert database_path.parent.exists(), "database parent must exist"
    with sqlite3.connect(str(database_path)) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE startups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT UNIQUE,
                description TEXT,
                source TEXT,
                date_found TIMESTAMP
            )
            """
        )
        cursor.execute("CREATE INDEX idx_startups_name ON startups(name)")
        cursor.execute("CREATE INDEX idx_startups_source ON startups(source)")
        cursor.execute(
            """
            CREATE TABLE scrape_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_scrape TIMESTAMP NOT NULL,
                scrapers_run TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE VIRTUAL TABLE startups_fts
            USING fts5(name, description, content='startups', content_rowid='id')
            """
        )
        cursor.executescript(
            """
            CREATE TRIGGER startups_ai AFTER INSERT ON startups BEGIN
                INSERT INTO startups_fts(rowid, name, description)
                VALUES (new.id, new.name, new.description);
            END;
            CREATE TRIGGER startups_ad AFTER DELETE ON startups BEGIN
                INSERT INTO startups_fts(startups_fts, rowid, name, description)
                VALUES('delete', old.id, old.name, old.description);
            END;
            CREATE TRIGGER startups_au AFTER UPDATE ON startups BEGIN
                INSERT INTO startups_fts(startups_fts, rowid, name, description)
                VALUES('delete', old.id, old.name, old.description);
                INSERT INTO startups_fts(rowid, name, description)
                VALUES (new.id, new.name, new.description);
            END;
            """
        )
        cursor.execute(
            """
            INSERT INTO startups (name, url, description, source, date_found)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "Example Dev Tool",
                "https://example.com/tool",
                "A public developer tool listing.",
                "GitHub Trending",
                "2026-01-01T00:00:00",
            ),
        )
        cursor.execute(
            """
            INSERT INTO scrape_log (last_scrape, scrapers_run)
            VALUES (?, ?)
            """,
            ("2026-01-01T00:00:00", "GitHub"),
        )
        connection.commit()


def _create_fixture(tmp_path: Path) -> Path:
    database_path = tmp_path / "source.db"
    fixture_path = tmp_path / "startups.db.gz"
    _create_valid_database(database_path)
    row_count = create_sandbox_seed.create_seed(database_path, fixture_path)
    assert row_count == 1
    assert fixture_path.stat().st_size > 0
    return fixture_path


def test_validate_database_accepts_expected_schema(tmp_path: Path) -> None:
    """A seed database with expected public schema should validate."""
    database_path = tmp_path / "source.db"
    _create_valid_database(database_path)

    assert create_sandbox_seed.validate_database(database_path) == 1


def test_validate_database_rejects_private_table(tmp_path: Path) -> None:
    """Unexpected or private-looking tables should block fixture creation."""
    database_path = tmp_path / "source.db"
    _create_valid_database(database_path)
    with sqlite3.connect(str(database_path)) as connection:
        connection.execute("CREATE TABLE api_tokens (id INTEGER PRIMARY KEY, token TEXT)")

    with pytest.raises(create_sandbox_seed.SeedValidationError, match="unexpected tables"):
        create_sandbox_seed.validate_database(database_path)


def test_validate_database_rejects_email_like_text(tmp_path: Path) -> None:
    """Public seed fixtures should not contain personal contact strings."""
    database_path = tmp_path / "source.db"
    _create_valid_database(database_path)
    with sqlite3.connect(str(database_path)) as connection:
        connection.execute("PRAGMA trusted_schema=ON")
        connection.execute(
            "UPDATE startups SET description = ? WHERE id = 1",
            ("Contact person@example.com for details.",),
        )

    with pytest.raises(create_sandbox_seed.SeedValidationError, match="email-like text"):
        create_sandbox_seed.validate_database(database_path)


def test_validate_database_rejects_product_hunt_tracking_source(tmp_path: Path) -> None:
    """Public seed fixtures should not expose source app tracking identifiers."""
    database_path = tmp_path / "source.db"
    _create_valid_database(database_path)
    with sqlite3.connect(str(database_path)) as connection:
        connection.execute("PRAGMA trusted_schema=ON")
        connection.execute(
            "UPDATE startups SET url = ? WHERE id = 1",
            ("https://www.producthunt.com/products/example?utm_source=Application%3A+devtoolscrape",),
        )

    with pytest.raises(create_sandbox_seed.SeedValidationError, match="sensitive text marker"):
        create_sandbox_seed.validate_database(database_path)


def test_restore_creates_missing_target(tmp_path: Path) -> None:
    """Restore should create the target database when it does not exist."""
    fixture_path = _create_fixture(tmp_path)
    target_path = tmp_path / "data" / "startups.db"

    restored = restore_sandbox_db.restore_database(fixture_path, target_path)

    assert restored is True
    assert target_path.exists()
    assert create_sandbox_seed.validate_database(target_path) == 1


def test_restore_preserves_nonempty_target_without_force(tmp_path: Path) -> None:
    """Restore should not overwrite local sandbox data by default."""
    fixture_path = _create_fixture(tmp_path)
    target_path = tmp_path / "data" / "startups.db"
    target_path.parent.mkdir(parents=True)
    target_path.write_bytes(b"local data")

    restored = restore_sandbox_db.restore_database(fixture_path, target_path)

    assert restored is False
    assert target_path.read_bytes() == b"local data"


def test_restore_replaces_nonempty_target_with_force(tmp_path: Path) -> None:
    """Forced restore should replace an existing local database."""
    fixture_path = _create_fixture(tmp_path)
    target_path = tmp_path / "data" / "startups.db"
    target_path.parent.mkdir(parents=True)
    target_path.write_bytes(b"local data")

    restored = restore_sandbox_db.restore_database(fixture_path, target_path, force=True)

    assert restored is True
    assert create_sandbox_seed.validate_database(target_path) == 1


def test_restore_rejects_corrupt_fixture(tmp_path: Path) -> None:
    """Restore should validate decompressed fixtures before replacing the target."""
    fixture_path = tmp_path / "broken.db.gz"
    target_path = tmp_path / "data" / "startups.db"
    target_path.parent.mkdir(parents=True)
    with gzip.open(fixture_path, "wb") as fixture_file:
        fixture_file.write(b"not a sqlite database")

    with pytest.raises(sqlite3.DatabaseError):
        restore_sandbox_db.restore_database(fixture_path, target_path, force=True)

    assert not target_path.exists()
