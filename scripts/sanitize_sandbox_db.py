#!/usr/bin/env python3
"""Create a public-safe copy of a sandbox SQLite database."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from create_sandbox_seed import EMAIL_PATTERN, SECRET_TOKEN_PATTERN, validate_database

SENSITIVE_ENV_MARKERS = (
    "OPENAI_API_KEY",
    "DATADOG_API_KEY",
    "DD_API_KEY",
)
PRODUCTHUNT_HOSTS = {"www.producthunt.com", "producthunt.com"}


def _redact_sensitive_text(value: str | None) -> str | None:
    """Redact personal or secret-looking text while preserving record shape."""
    if value is None:
        return None

    redacted = value
    for marker in SENSITIVE_ENV_MARKERS:
        redacted = redacted.replace(marker, "[REDACTED_ENV_VAR]")
    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
    return SECRET_TOKEN_PATTERN.sub("[REDACTED_TOKEN]", redacted)


def _strip_producthunt_tracking(url: str | None) -> str | None:
    """Remove Product Hunt API tracking query strings from public fixture URLs."""
    if url is None:
        return None
    parts = urlsplit(url)
    if parts.netloc.lower() not in PRODUCTHUNT_HOSTS:
        return url
    if not parts.query and not parts.fragment:
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _sanitize_row_values(row: sqlite3.Row) -> tuple[str | None, str | None, str | None, str | None]:
    """Return sanitized startup fields for a SQLite row."""
    name = _redact_sensitive_text(row["name"])
    url = _strip_producthunt_tracking(_redact_sensitive_text(row["url"]))
    description = _redact_sensitive_text(row["description"])
    source = _redact_sensitive_text(row["source"])
    return name, url, description, source


def sanitize_database(source_path: Path, output_path: Path) -> int:
    """Copy and sanitize a SQLite database for public sandbox distribution.

    Args:
        source_path: SQLite database to copy from.
        output_path: Destination path for the sanitized database.

    Returns:
        Number of startup rows changed.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"source database not found: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("source_path and output_path must differ")

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
        shutil.copyfile(source_path, temp_path)
        changed_rows = _sanitize_database_in_place(temp_path)
        validate_database(temp_path)
        temp_path.replace(output_path)
        return changed_rows
    finally:
        temp_path.unlink(missing_ok=True)


def _sanitize_database_in_place(database_path: Path) -> int:
    """Sanitize startup rows inside an existing SQLite database file."""
    assert database_path.exists(), "database_path must exist"
    changed_rows = 0
    with sqlite3.connect(str(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, name, url, description, source FROM startups ORDER BY id"
        ).fetchall()
        for row in rows:
            sanitized_values = _sanitize_row_values(row)
            current_values = (row["name"], row["url"], row["description"], row["source"])
            if sanitized_values == current_values:
                continue
            connection.execute(
                """
                UPDATE startups
                SET name = ?, url = ?, description = ?, source = ?
                WHERE id = ?
                """,
                (*sanitized_values, row["id"]),
            )
            changed_rows += 1
        connection.execute("INSERT INTO startups_fts(startups_fts) VALUES('rebuild')")
        connection.commit()
    return changed_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize a sandbox SQLite database copy.")
    parser.add_argument("source", type=Path, help="Source SQLite database path")
    parser.add_argument("output", type=Path, help="Sanitized SQLite database output path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    changed_rows = sanitize_database(args.source, args.output)
    print(f"Sanitized {changed_rows} startup rows into {args.output}.")


if __name__ == "__main__":
    main()
