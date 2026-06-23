"""
Lightweight helpers for Datadog tracing so we can consistently instrument
outbound calls even when ddtrace isn't available (e.g., in tests).
"""
from __future__ import annotations

import logging
import os
import secrets
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from ddtrace.span import Span  # type: ignore

try:  # pragma: no cover - ddtrace may not be installed in all environments
    from ddtrace import tracer  # type: ignore
except Exception:  # pragma: no cover - fallback to no-op tracing
    tracer = None  # type: ignore

logger = logging.getLogger("devtools.observability")

DEFAULT_SERVICE = os.getenv("DD_SERVICE", "devtoolscrape")


@contextmanager
def trace_http_call(
    resource: str,
    method: str,
    url: str,
    *,
    service: Optional[str] = None,
    span_name: str = "external.http.request",
) -> Iterator[Optional["Span"]]:
    """
    Wrap an outbound HTTP call in a Datadog span if ddtrace is available.

    Returns the span so callers can attach additional tags (status code, etc.).
    When ddtrace isn't installed the context manager yields ``None``.
    """
    if tracer is None:
        yield None
        return

    effective_service = service or DEFAULT_SERVICE

    try:
        ctx = tracer.trace(span_name, service=effective_service, resource=resource, span_type="http")
    except Exception:
        yield None
        return

    with ctx as span:
        try:
            span.set_tag("http.method", method.upper())
            span.set_tag("http.url", url)
        except Exception:
            span = None
        yield span


@contextmanager
def trace_external_call(
    span_name: str,
    resource: str,
    *,
    service: Optional[str] = None,
    span_type: str = "custom",
    tags: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional["Span"]]:
    """
    Generic tracing helper for non-HTTP calls (e.g., OpenAI SDK).
    """
    if tracer is None:
        yield None
        return

    effective_service = service or DEFAULT_SERVICE

    try:
        ctx = tracer.trace(span_name, service=effective_service, resource=resource, span_type=span_type)
    except Exception:
        yield None
        return

    with ctx as span:
        try:
            if tags:
                for key, value in tags.items():
                    span.set_tag(key, value)
        except Exception:
            span = None
        yield span


_TRACE_ID_HEX_LENGTH = 32  # 128 bits = 16 bytes = 32 hex characters


def generate_trace_id_w3c() -> str:
    """Generate a 128-bit W3C-compatible trace ID as a 32-character hex string.

    Uses secrets.token_hex for cryptographic randomness, which is the stdlib
    recommendation for security-sensitive random values.
    """
    trace_id_hex = secrets.token_hex(16)
    assert len(trace_id_hex) == _TRACE_ID_HEX_LENGTH, (
        f"Expected {_TRACE_ID_HEX_LENGTH} hex chars, got {len(trace_id_hex)}"
    )
    # W3C Trace Context spec forbids all-zero trace IDs.
    assert trace_id_hex != "0" * _TRACE_ID_HEX_LENGTH, (
        "Generated all-zero trace ID, which is invalid per W3C spec"
    )
    return trace_id_hex


def tag_root_span_with_custom_trace_id(trace_id_hex: str) -> Optional[int]:
    """Tag the current root span with a custom W3C trace ID.

    Sets custom.trace_id and original.trace_id as span tags on the root span.
    The actual trace_id rewrite happens later in CustomTraceIdFilter, which
    rewrites ALL spans in the trace consistently before they are sent to the
    dd-agent. This avoids fragmenting the trace by changing only the root span's
    trace_id while child spans retain the original.

    Returns the original trace ID (as int) if tagging succeeded, None otherwise.
    """
    assert isinstance(trace_id_hex, str), (
        f"trace_id_hex must be str, got {type(trace_id_hex)}"
    )
    assert len(trace_id_hex) == _TRACE_ID_HEX_LENGTH, (
        f"W3C trace ID must be {_TRACE_ID_HEX_LENGTH} hex chars, got {len(trace_id_hex)}"
    )

    if tracer is None:
        return None

    try:
        root_span = tracer.current_root_span()
    except AttributeError:
        # Tracer stub or incompatible ddtrace version lacks current_root_span.
        return None

    if root_span is None:
        return None

    original_trace_id = root_span.trace_id
    original_trace_id_hex = format(original_trace_id, "032x")

    # Only set tags here. The TraceFilter rewrites trace_ids on all spans at
    # send time so every span in the trace shares the same custom trace_id.
    root_span.set_tag("custom.trace_id", trace_id_hex)
    root_span.set_tag("original.trace_id", original_trace_id_hex)

    return original_trace_id


class CustomTraceIdFilter:
    """ddtrace TraceFilter that rewrites trace IDs on all spans in a trace.

    When a root span carries the custom.trace_id tag, this filter replaces
    the trace_id on every span in the trace with the custom value. This
    ensures the dd-agent sees a coherent trace under the new ID, not a
    fragmented one where root and children have different IDs.
    """

    def process_trace(self, trace: List["Span"]) -> Optional[List["Span"]]:
        """Rewrite trace IDs if the root span has a custom.trace_id tag."""
        if not trace:
            return trace

        # Find the root span (parent_id == 0 or None).
        root_span = None
        for span in trace:
            if span.parent_id is None or span.parent_id == 0:
                root_span = span
                break

        if root_span is None:
            return trace

        custom_trace_id_hex = root_span.get_tag("custom.trace_id")
        if custom_trace_id_hex is None:
            return trace

        new_trace_id = int(custom_trace_id_hex, 16)

        # Rewrite every span's trace_id so the entire trace is coherent.
        for span in trace:
            span.trace_id = new_trace_id

        logger.info(
            "trace.filter_rewrite",
            extra={
                "event": "trace.filter_rewrite",
                "span_count": len(trace),
                "custom_trace_id": custom_trace_id_hex,
                "original_trace_id": root_span.get_tag("original.trace_id"),
            },
        )

        return trace


def install_custom_trace_id_filter() -> bool:
    """Install the CustomTraceIdFilter on the global tracer.

    Uses tracer.configure(trace_processors=...) which is the supported API
    in ddtrace 4.x. Falls back to tracer._filters for older versions.

    Safe to call multiple times; checks if a filter is already installed.
    Returns True if the filter was installed, False otherwise.
    """
    if tracer is None:
        return False

    # ddtrace 4.x uses tracer.configure(trace_processors=[...]) to register
    # processors that implement process_trace(). The configure call replaces
    # the processor list, so we must read the existing ones first.
    try:
        # In ddtrace 4.x, trace processors live on the internal deferred init
        # or are passed via configure(). We use configure() which merges.
        tracer.configure(trace_processors=[CustomTraceIdFilter()])
    except TypeError:
        # Older ddtrace versions may not support trace_processors kwarg.
        # Fall back to _filters if available.
        try:
            existing_filters = tracer._filters
        except AttributeError:
            logger.warning(
                "trace.filter_install_failed",
                extra={"reason": "tracer supports neither configure(trace_processors=) nor _filters"},
            )
            return False

        for existing_filter in existing_filters:
            if isinstance(existing_filter, CustomTraceIdFilter):
                return True

        existing_filters.append(CustomTraceIdFilter())
        tracer._filters = existing_filters

    logger.info(
        "trace.filter_installed",
        extra={"event": "trace.filter_installed"},
    )
    return True
