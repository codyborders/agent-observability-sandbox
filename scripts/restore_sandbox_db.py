#!/usr/bin/env python3
"""Restore the sandbox SQLite database from the compressed seed fixture."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import tempfile
from pathlib import Path

from create_sandbox_seed import validate_database


class RestoreRefusedError(RuntimeError):
    """Raised when restore would overwrite local data without explicit force."""


def project_root() -> Path:
    """Return the repository root based on this script location."""
    root = Path(__file__).resolve().parents[1]
    assert root.exists(), "project root must exist"
    assert (root / "scripts").is_dir(), "project root must contain scripts directory"
    return root


def default_database_path(root: Path) -> Path:
    """Return the database path requested by environment or sandbox defaults."""
    assert root.exists(), "root must exist"
    database_path = os.getenv("DEVTOOLS_DB_PATH")
    if database_path:
        return Path(database_path)

    data_dir = Path(os.getenv("DEVTOOLS_DATA_DIR", str(root / "data")))
    return data_dir / "startups.db"


def _target_has_data(target_path: Path) -> bool:
    if not target_path.exists():
        return False
    if not target_path.is_file():
        raise RestoreRefusedError(f"target database path is not a file: {target_path}")
    return target_path.stat().st_size > 0


def restore_database(fixture_path: Path, target_path: Path, *, force: bool = False) -> bool:
    """Restore a SQLite database from a gzip fixture.

    Args:
        fixture_path: Compressed seed fixture path.
        target_path: SQLite database path to create or replace.
        force: Whether to replace an existing nonempty database.

    Returns:
        True when a database was restored, False when an existing database was preserved.

    Raises:
        FileNotFoundError: If the fixture does not exist.
        RestoreRefusedError: If a fixture or target path has the wrong file type.
        SeedValidationError: If the decompressed fixture is invalid.
    """
    if not fixture_path.exists():
        raise FileNotFoundError(f"seed fixture not found: {fixture_path}")
    if not fixture_path.is_file():
        raise RestoreRefusedError(f"seed fixture path is not a file: {fixture_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if _target_has_data(target_path) and not force:
        print(f"Existing database preserved at {target_path}.")
        return False

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f"{target_path.name}.",
        suffix=".tmp",
        dir=str(target_path.parent),
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with gzip.open(fixture_path, "rb") as fixture_file:
            with temp_path.open("wb") as database_file:
                shutil.copyfileobj(fixture_file, database_file)
        row_count = validate_database(temp_path)
        temp_path.replace(target_path)
        print(f"Restored sandbox database to {target_path} with {row_count} startup rows.")
        return True
    finally:
        temp_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Restore the sandbox SQLite database.")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=root / "seed" / "startups.db.gz",
        help="Compressed SQLite seed fixture path",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=default_database_path(root),
        help="SQLite database path to restore",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing nonempty target database",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    restore_database(args.fixture, args.target, force=args.force)


if __name__ == "__main__":
    main()
