#!/usr/bin/env python3
"""Fail when repository text files contain emoji characters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
)
EMOJI_CODEPOINTS = {0x200D, 0xFE0F}
SKIPPED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "coverage",
    "data",
    "htmlcov",
    "logs",
    "node_modules",
    "venv",
}
SKIPPED_SUFFIXES = {
    ".db",
    ".gz",
    ".ico",
    ".jpg",
    ".jpeg",
    ".png",
    ".pyc",
    ".sqlite",
    ".sqlite3",
    ".webp",
}
SKIPPED_FILENAMES = {".coverage", ".env"}


def _is_emoji(character: str) -> bool:
    codepoint = ord(character)
    if codepoint in EMOJI_CODEPOINTS:
        return True
    return any(start <= codepoint <= end for start, end in EMOJI_RANGES)


def _should_skip(path: Path, root: Path) -> bool:
    assert path.is_absolute(), "path must be absolute"
    assert root.is_absolute(), "root must be absolute"
    relative_parts = path.relative_to(root).parts
    if any(part in SKIPPED_DIRECTORIES for part in relative_parts):
        return True
    if path.name in SKIPPED_FILENAMES:
        return True
    return path.suffix.lower() in SKIPPED_SUFFIXES


def _read_text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def find_emoji(root: Path) -> list[tuple[Path, int, int, str]]:
    """Return all emoji occurrences in decodable project text files."""
    root = root.resolve()
    assert root.exists(), "root must exist"
    findings: list[tuple[Path, int, int, str]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _should_skip(path, root):
            continue
        text = _read_text(path)
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for column_number, character in enumerate(line, start=1):
                if _is_emoji(character):
                    findings.append((path.relative_to(root), line_number, column_number, character))
    return findings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check repository text files for emoji.")
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    findings = find_emoji(args.root)
    if not findings:
        print("No emoji found.")
        return 0

    for relative_path, line_number, column_number, character in findings:
        codepoint = f"U+{ord(character):04X}"
        print(f"{relative_path}:{line_number}:{column_number}: emoji {codepoint} is not allowed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
