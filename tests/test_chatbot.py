"""Tests for chatbot query sanitization."""

import pytest

from chatbot import _sanitize_fts_query


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("CI/CD tools", "CI CD tools"),
        ('python "web framework"', "python web framework"),
        ("node*", "node"),
        ("(react OR vue)", "react vue"),
        ("test AND deploy", "test deploy"),
        ("NEAR/3 rust", "3 rust"),
        ("key:value", "key value"),
        ("boost^2", "boost 2"),
        ("{prefix}", "prefix"),
        ("term1 + term2", "term1 term2"),
        ("front-end tools", "front end tools"),
        ("plain query", "plain query"),
        ("", ""),
        ("***", ""),
    ],
)
def test_sanitize_fts_query_strips_operators(raw: str, expected: str) -> None:
    """Verify FTS5 special characters and keywords are stripped from queries."""
    assert _sanitize_fts_query(raw) == expected
