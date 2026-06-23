"""Tests for custom W3C trace ID generation, span tagging, and TraceFilter.

Verifies that:
- generate_trace_id_w3c() produces valid 128-bit hex trace IDs
- tag_root_span_with_custom_trace_id() sets tags without modifying trace_id
- CustomTraceIdFilter rewrites trace_ids on all spans in a trace
- install_custom_trace_id_filter() registers the filter on the tracer
- The Flask before_request hook respects the CUSTOM_TRACE_ID_ENABLED flag
"""

import importlib
import re
import sys
from unittest.mock import ANY, MagicMock, patch

import pytest

from observability import (
    CustomTraceIdFilter,
    generate_trace_id_w3c,
    install_custom_trace_id_filter,
    tag_root_span_with_custom_trace_id,
)


# ---------------------------------------------------------------------------
# Unit tests: generate_trace_id_w3c
# ---------------------------------------------------------------------------

_W3C_HEX_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def test_generate_trace_id_w3c_format():
    """Generated trace IDs must be exactly 32 lowercase hex characters, and
    each call must produce a unique value (statistical sanity check)."""
    sample_count = 10
    ids = set()

    for _ in range(sample_count):
        trace_id = generate_trace_id_w3c()
        assert _W3C_HEX_PATTERN.match(trace_id), (
            f"Trace ID '{trace_id}' is not 32 lowercase hex chars"
        )
        ids.add(trace_id)

    assert len(ids) == sample_count, (
        f"Expected {sample_count} unique IDs, got {len(ids)} -- possible RNG issue"
    )


def test_generate_trace_id_w3c_not_all_zeros():
    """W3C spec forbids all-zero trace IDs. Verify over many samples."""
    for _ in range(100):
        trace_id = generate_trace_id_w3c()
        assert trace_id != "0" * 32, "Generated all-zero trace ID"


# ---------------------------------------------------------------------------
# Unit tests: tag_root_span_with_custom_trace_id
# ---------------------------------------------------------------------------

def _make_mock_span(trace_id: int = 99999):
    """Build a mock span with a mutable trace_id and recording set_tag."""
    span = MagicMock()
    span.trace_id = trace_id
    span.parent_id = 0
    return span


def _make_mock_tracer(root_span=None):
    """Build a mock tracer whose current_root_span() returns the given span."""
    mock_tracer = MagicMock()
    mock_tracer.current_root_span.return_value = root_span
    return mock_tracer


def test_tag_root_span_succeeds():
    """When a root span exists, tags should be set but trace_id should NOT be
    changed (the TraceFilter handles that at send time)."""
    original_id = 12345
    custom_hex = "aabbccdd" * 4  # 32 hex chars

    span = _make_mock_span(trace_id=original_id)
    mock_tracer = _make_mock_tracer(root_span=span)

    with patch("observability.tracer", mock_tracer):
        result = tag_root_span_with_custom_trace_id(custom_hex)

    assert result == original_id
    original_id_hex = format(original_id, "032x")
    span.set_tag.assert_any_call("custom.trace_id", custom_hex)
    span.set_tag.assert_any_call("original.trace_id", original_id_hex)
    assert span.set_tag.call_count == 2
    # trace_id must NOT be modified by the tagging function.
    assert span.trace_id == original_id


def test_tag_root_span_no_tracer():
    """When tracer is None (ddtrace not installed), return None gracefully."""
    with patch("observability.tracer", None):
        result = tag_root_span_with_custom_trace_id("aa" * 16)

    assert result is None


def test_tag_root_span_no_root_span():
    """When current_root_span() returns None, return None gracefully."""
    mock_tracer = _make_mock_tracer(root_span=None)

    with patch("observability.tracer", mock_tracer):
        result = tag_root_span_with_custom_trace_id("bb" * 16)

    assert result is None
    mock_tracer.current_root_span.assert_called_once()


def test_tag_root_span_validates_input():
    """Bad inputs must raise AssertionError, not silently corrupt spans."""
    # Wrong length (16 chars instead of 32).
    with pytest.raises(AssertionError, match="32 hex chars"):
        tag_root_span_with_custom_trace_id("aabb" * 4)

    # Non-string input.
    with pytest.raises(AssertionError, match="must be str"):
        tag_root_span_with_custom_trace_id(12345)


# ---------------------------------------------------------------------------
# Unit tests: CustomTraceIdFilter
# ---------------------------------------------------------------------------

def test_trace_filter_rewrites_all_span_trace_ids():
    """When the root span has custom.trace_id, every span in the trace
    should get the new trace_id."""
    custom_hex = "aabbccdd" * 4
    new_trace_id = int(custom_hex, 16)
    original_id = 99999

    root = _make_mock_span(trace_id=original_id)
    root.parent_id = 0
    root.get_tag.return_value = custom_hex

    child_one = _make_mock_span(trace_id=original_id)
    child_one.parent_id = root.span_id
    child_one.get_tag.return_value = None

    child_two = _make_mock_span(trace_id=original_id)
    child_two.parent_id = root.span_id
    child_two.get_tag.return_value = None

    trace = [root, child_one, child_two]
    trace_filter = CustomTraceIdFilter()
    result = trace_filter.process_trace(trace)

    assert result is trace
    for span in result:
        assert span.trace_id == new_trace_id


def test_trace_filter_skips_when_no_custom_tag():
    """When the root span has no custom.trace_id tag, trace_ids must not
    be modified."""
    original_id = 55555
    root = _make_mock_span(trace_id=original_id)
    root.parent_id = 0
    root.get_tag.return_value = None

    child = _make_mock_span(trace_id=original_id)
    child.parent_id = 1

    trace = [root, child]
    trace_filter = CustomTraceIdFilter()
    result = trace_filter.process_trace(trace)

    assert result is trace
    for span in result:
        assert span.trace_id == original_id


def test_trace_filter_handles_empty_trace():
    """Empty trace list must pass through without error."""
    trace_filter = CustomTraceIdFilter()
    assert trace_filter.process_trace([]) == []


# ---------------------------------------------------------------------------
# Unit tests: install_custom_trace_id_filter
# ---------------------------------------------------------------------------

def test_install_filter_calls_configure():
    """The filter should be installed via tracer.configure(trace_processors=...)."""
    mock_tracer = MagicMock()

    with patch("observability.tracer", mock_tracer):
        result = install_custom_trace_id_filter()

    assert result is True
    mock_tracer.configure.assert_called_once()
    call_kwargs = mock_tracer.configure.call_args[1]
    assert "trace_processors" in call_kwargs
    assert len(call_kwargs["trace_processors"]) == 1
    assert isinstance(call_kwargs["trace_processors"][0], CustomTraceIdFilter)


def test_install_filter_falls_back_to_filters():
    """When configure raises TypeError, fall back to tracer._filters."""
    mock_tracer = MagicMock()
    mock_tracer.configure.side_effect = TypeError("unsupported kwarg")
    mock_tracer._filters = []

    with patch("observability.tracer", mock_tracer):
        result = install_custom_trace_id_filter()

    assert result is True
    assert len(mock_tracer._filters) == 1
    assert isinstance(mock_tracer._filters[0], CustomTraceIdFilter)


def test_install_filter_no_tracer():
    """When tracer is None, installation should return False."""
    with patch("observability.tracer", None):
        result = install_custom_trace_id_filter()

    assert result is False


# ---------------------------------------------------------------------------
# Integration tests: Flask before_request hook
# ---------------------------------------------------------------------------

@pytest.fixture
def app_module(monkeypatch):
    """Import app_production with database init stubbed out."""
    import database

    monkeypatch.setattr(database, "init_db", lambda: None)
    sys.modules.pop("app_production", None)
    module = importlib.import_module("app_production")
    return module


def _stub_db(module, monkeypatch):
    """Wire minimal stubs so routes render without a real database."""
    sample = [
        {
            "id": 1,
            "name": "Test Tool",
            "url": "https://example.com",
            "description": "Test",
            "source": "GitHub Trending",
            "date_found": "2024-01-01T00:00:00",
        }
    ]
    monkeypatch.setattr(module, "get_all_startups", lambda limit=None, offset=None: sample)
    monkeypatch.setattr(module, "count_all_startups", lambda: 1)
    monkeypatch.setattr(
        module,
        "get_source_counts",
        lambda: {"total": 1, "github": 1, "hackernews": 0, "producthunt": 0, "other": 0},
    )
    monkeypatch.setattr(module, "get_last_scrape_time", lambda: "2024-01-01T00:00:00")


def test_before_request_applies_custom_trace_tags_when_enabled(app_module, monkeypatch):
    """With the flag on and a root span available, the span should have
    custom.trace_id and original.trace_id tags but trace_id left unchanged."""
    module = app_module
    _stub_db(module, monkeypatch)

    monkeypatch.setattr(module, "_CUSTOM_TRACE_ID_ENABLED", True)

    original_id = 77777
    span = _make_mock_span(trace_id=original_id)
    mock_tracer = _make_mock_tracer(root_span=span)
    monkeypatch.setattr("observability.tracer", mock_tracer)

    client = module.app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    # Tags should be set.
    original_id_hex = format(original_id, "032x")
    span.set_tag.assert_any_call("original.trace_id", original_id_hex)
    span.set_tag.assert_any_call("custom.trace_id", ANY)
    assert span.set_tag.call_count == 2
    # trace_id should NOT be modified by the before_request hook.
    assert span.trace_id == original_id


def test_before_request_skips_when_disabled(app_module, monkeypatch):
    """With the flag off, tag_root_span_with_custom_trace_id should never be called."""
    module = app_module
    _stub_db(module, monkeypatch)

    monkeypatch.setattr(module, "_CUSTOM_TRACE_ID_ENABLED", False)

    mock_tracer = _make_mock_tracer()
    monkeypatch.setattr("observability.tracer", mock_tracer)

    client = module.app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    # current_root_span should never be called when the flag is off.
    mock_tracer.current_root_span.assert_not_called()


def test_before_request_handles_missing_root_span(app_module, monkeypatch):
    """With the flag on but no root span, the request must still complete
    without errors."""
    module = app_module
    _stub_db(module, monkeypatch)

    monkeypatch.setattr(module, "_CUSTOM_TRACE_ID_ENABLED", True)

    mock_tracer = _make_mock_tracer(root_span=None)
    monkeypatch.setattr("observability.tracer", mock_tracer)

    client = module.app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    mock_tracer.current_root_span.assert_called_once()
