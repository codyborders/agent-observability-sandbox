from unittest.mock import MagicMock, patch, call
from contextlib import contextmanager

import pytest


def _make_mock_tracer():
    """Build a mock tracer whose .trace() returns a context-manager span."""
    mock_tracer = MagicMock()
    mock_span = MagicMock()

    @contextmanager
    def fake_trace(*args, **kwargs):
        # Store the call args on the span so tests can inspect them
        mock_span._trace_args = args
        mock_span._trace_kwargs = kwargs
        yield mock_span

    mock_tracer.trace = MagicMock(side_effect=fake_trace)
    return mock_tracer, mock_span


def test_trace_http_call_uses_default_span_name_and_resource():
    """trace_http_call should use span_name for the Datadog span name and
    resource for Datadog resource grouping."""
    mock_tracer, mock_span = _make_mock_tracer()

    with patch("observability.tracer", mock_tracer):
        from observability import trace_http_call

        with trace_http_call("hackernews.topstories", "GET", "https://hn.algolia.com/api") as span:
            pass

    # The first positional arg to tracer.trace() is the span name
    trace_call = mock_tracer.trace.call_args
    span_name = trace_call[0][0]  # first positional arg
    resource_kwarg = trace_call[1].get("resource", "")

    assert span_name == "external.http.request", (
        f"Expected default span name 'external.http.request' but got '{span_name}'"
    )
    assert resource_kwarg == "hackernews.topstories", (
        f"Expected resource 'hackernews.topstories' but got '{resource_kwarg}'"
    )


def test_trace_http_call_honors_custom_span_name():
    """Callers can override the default HTTP span name when needed."""
    mock_tracer, _ = _make_mock_tracer()

    with patch("observability.tracer", mock_tracer):
        from observability import trace_http_call

        with trace_http_call(
            "hackernews.topstories",
            "GET",
            "https://hn.algolia.com/api",
            span_name="hackernews.fetch",
        ):
            pass

    trace_call = mock_tracer.trace.call_args
    span_name = trace_call[0][0]
    resource_kwarg = trace_call[1].get("resource", "")
    assert span_name == "hackernews.fetch"
    assert resource_kwarg == "hackernews.topstories"


def test_trace_external_call_uses_descriptive_span_name():
    """Bug #7: trace_external_call should also pass the descriptive name as
    the span name to tracer.trace().
    """
    mock_tracer, mock_span = _make_mock_tracer()

    with patch("observability.tracer", mock_tracer):
        from observability import trace_external_call

        with trace_external_call("openai.chat.completion", "classify_batch") as span:
            pass

    trace_call = mock_tracer.trace.call_args
    span_name = trace_call[0][0]
    resource_kwarg = trace_call[1].get("resource", "")

    assert span_name == "openai.chat.completion", (
        f"Expected span name 'openai.chat.completion' but got '{span_name}'"
    )
    assert resource_kwarg == "classify_batch", (
        f"Expected resource 'classify_batch' but got '{resource_kwarg}'"
    )
