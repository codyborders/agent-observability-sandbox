"""Tests enforcing the sandbox repository no-emoji rule."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_contains_no_emoji() -> None:
    """Project text files should not contain emoji characters."""
    result = subprocess.run(
        [sys.executable, "scripts/check-no-emoji.py", str(ROOT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
