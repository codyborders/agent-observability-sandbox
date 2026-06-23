"""Tests for bug-bash fixes: classify_source None guard, FTS5 sanitization, tracing resilience."""

from unittest.mock import MagicMock, patch


# --- Bug 4: classify_source(None) should not crash ---


def test_classify_source_none_returns_other():
    """classify_source must not crash when source is None (nullable column)."""
    import database

    assert database.classify_source(None) == "other"


# --- Bug 25: FTS5 special characters should not crash search ---


def test_search_startups_fts_special_chars_do_not_crash(fresh_db):
    """FTS5 special characters in search input should not raise OperationalError."""
    for query in ["c++", 'react*', '"unmatched', "(parens)", "test AND deploy", "***"]:
        results = fresh_db.search_startups(query)
        assert isinstance(results, list)


def test_count_search_results_fts_special_chars_do_not_crash(fresh_db):
    """FTS5 special characters in count query should not raise OperationalError."""
    for query in ["c++", 'react*', '"unmatched', "(parens)", "test AND deploy", "***"]:
        count = fresh_db.count_search_results(query)
        assert isinstance(count, int)


# --- Bug 24: tracer.trace() exception should not block operations ---


def test_trace_http_call_yields_none_when_tracer_raises():
    """If tracer.trace() raises, the context manager should yield None, not propagate."""
    mock_tracer = MagicMock()
    mock_tracer.trace.side_effect = RuntimeError("tracer broken")

    with patch("observability.tracer", mock_tracer):
        from observability import trace_http_call

        with trace_http_call("test.resource", "GET", "https://example.com") as span:
            assert span is None


def test_trace_external_call_yields_none_when_tracer_raises():
    """If tracer.trace() raises, the context manager should yield None, not propagate."""
    mock_tracer = MagicMock()
    mock_tracer.trace.side_effect = RuntimeError("tracer broken")

    with patch("observability.tracer", mock_tracer):
        from observability import trace_external_call

        with trace_external_call("test.span", "test.resource") as span:
            assert span is None
