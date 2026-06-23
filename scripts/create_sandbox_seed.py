#!/usr/bin/env python3
"""Create a validated compressed SQLite seed for the sandbox repository."""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

EXPECTED_TABLES = {
    "scrape_log",
    "startups",
    "startups_fts",
    "startups_fts_config",
    "startups_fts_data",
    "startups_fts_docsize",
    "startups_fts_idx",
}
EXPECTED_INDEXES = {"idx_startups_name", "idx_startups_source"}
EXPECTED_TRIGGERS = {"startups_ad", "startups_ai", "startups_au"}
EXPECTED_STARTUP_COLUMNS = ["id", "name", "url", "description", "source", "date_found"]
EXPECTED_SCRAPE_LOG_COLUMNS = ["id", "last_scrape", "scrapers_run"]
PRIVATE_NAME_MARKERS = (
    "api_key",
    "apikey",
    "credential",
    "secret",
    "session",
    "token",
    "user",
)
SENSITIVE_TEXT_MARKERS = (
    "OPENAI_API_KEY",
    "DATADOG_API_KEY",
    "DD_API_KEY",
    "Application%3A+devtoolscrape",
    "sk-",
)
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


class SeedValidationError(ValueError):
    """Raised when a SQLite database is not safe to package as a seed."""


def _fetch_names(connection: sqlite3.Connection, object_type: str) -> set[str]:
    assert object_type in {"table", "index", "trigger"}, "object_type must be a SQLite object type"
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = ? AND name NOT LIKE 'sqlite_%'",
        (object_type,),
    ).fetchall()
    return {str(row[0]) for row in rows}


def _validate_exact_names(actual_names: set[str], expected_names: set[str], object_type: str) -> None:
    assert object_type, "object_type is required"
    if actual_names == expected_names:
        return

    unexpected = sorted(actual_names - expected_names)
    missing = sorted(expected_names - actual_names)
    details = []
    if unexpected:
        details.append(f"unexpected {object_type}: {', '.join(unexpected)}")
    if missing:
        details.append(f"missing {object_type}: {', '.join(missing)}")
    raise SeedValidationError("; ".join(details))


def _validate_columns(
    connection: sqlite3.Connection,
    table_name: str,
    expected_columns: Sequence[str],
) -> None:
    assert table_name, "table_name is required"
    assert expected_columns, "expected_columns is required"
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    actual_columns = [str(row[1]) for row in rows]
    if actual_columns != list(expected_columns):
        raise SeedValidationError(
            f"{table_name} columns differ: expected {list(expected_columns)}, got {actual_columns}"
        )


def _validate_no_private_object_names(names: Iterable[str]) -> None:
    checked_names = list(names)
    assert checked_names, "names must not be empty"
    for name in checked_names:
        lowered_name = name.lower()
        for marker in PRIVATE_NAME_MARKERS:
            if marker in lowered_name:
                raise SeedValidationError(f"private-looking SQLite object name blocked: {name}")


def _validate_no_sensitive_text(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT id, name, url, description, source FROM startups ORDER BY id"
    ).fetchall()
    assert rows, "startup rows must exist before sensitive-text scan"
    for row in rows:
        row_id = row[0]
        values = [str(value) for value in row[1:] if value is not None]
        startup_text = "\n".join(values)
        for marker in SENSITIVE_TEXT_MARKERS:
            if marker in startup_text:
                raise SeedValidationError(f"sensitive text marker found in startup row {row_id}")
        if EMAIL_PATTERN.search(startup_text):
            raise SeedValidationError(f"email-like text found in startup row {row_id}")


def validate_database(database_path: Path) -> int:
    """Validate a SQLite database and return the startup row count.

    Args:
        database_path: Path to the SQLite database to validate.

    Returns:
        Number of rows in the startups table.

    Raises:
        FileNotFoundError: If the database path does not exist.
        SeedValidationError: If the database fails sandbox seed checks.
    """
    if not database_path.exists():
        raise FileNotFoundError(f"database not found: {database_path}")
    if not database_path.is_file():
        raise SeedValidationError(f"database path is not a file: {database_path}")
    if database_path.stat().st_size <= 0:
        raise SeedValidationError(f"database is empty: {database_path}")

    with sqlite3.connect(str(database_path)) as connection:
        integrity_result = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity_result is None or integrity_result[0] != "ok":
            raise SeedValidationError(f"SQLite integrity check failed: {integrity_result}")

        table_names = _fetch_names(connection, "table")
        index_names = _fetch_names(connection, "index")
        trigger_names = _fetch_names(connection, "trigger")
        _validate_exact_names(table_names, EXPECTED_TABLES, "tables")
        _validate_exact_names(index_names, EXPECTED_INDEXES, "indexes")
        _validate_exact_names(trigger_names, EXPECTED_TRIGGERS, "triggers")
        _validate_no_private_object_names(table_names | index_names | trigger_names)
        _validate_columns(connection, "startups", EXPECTED_STARTUP_COLUMNS)
        _validate_columns(connection, "scrape_log", EXPECTED_SCRAPE_LOG_COLUMNS)

        row_count = int(connection.execute("SELECT COUNT(*) FROM startups").fetchone()[0])
        if row_count <= 0:
            raise SeedValidationError("startups table must contain at least one row")
        _validate_no_sensitive_text(connection)
        return row_count


def create_seed(source_path: Path, output_path: Path) -> int:
    """Validate and gzip a SQLite database fixture.

    Args:
        source_path: SQLite database to package.
        output_path: Compressed fixture path to write.

    Returns:
        Number of startup rows packaged in the fixture.
    """
    assert source_path != output_path, "source_path and output_path must differ"
    row_count = validate_database(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with source_path.open("rb") as source_file:
            with gzip.open(temp_path, "wb", compresslevel=9) as gzip_file:
                shutil.copyfileobj(source_file, gzip_file)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return row_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the sandbox SQLite seed fixture.")
    parser.add_argument("source", type=Path, help="Source SQLite database path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("seed/startups.db.gz"),
        help="Compressed seed fixture output path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    row_count = create_seed(args.source, args.output)
    print(f"Created {args.output} with {row_count} startup rows.")


if __name__ == "__main__":
    main()
