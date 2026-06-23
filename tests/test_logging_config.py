import logging
from unittest.mock import patch, MagicMock

import pytest

import logging_config


def test_context_filter_unpacks_correlation_ids_correctly():
    """Bug #6: ContextFilter.filter() unpacks get_correlation_ids() expecting
    5 values, but ddtrace returns a 2-tuple (trace_id, span_id). This causes
    a ValueError at runtime.
    """
    ctx_filter = logging_config.ContextFilter(service="test", env="test", hostname="localhost")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None,
    )

    # Inject a fake get_correlation_ids that returns a 2-tuple (like real ddtrace)
    fake_get_ids = MagicMock(return_value=(12345, 67890))
    original_available = logging_config._DDTRACE_AVAILABLE
    logging_config._DDTRACE_AVAILABLE = True
    logging_config.get_correlation_ids = fake_get_ids
    try:
        # This must not raise ValueError from too many values to unpack
        result = ctx_filter.filter(record)
    finally:
        logging_config._DDTRACE_AVAILABLE = original_available
        if hasattr(logging_config, "get_correlation_ids"):
            delattr(logging_config, "get_correlation_ids")

    assert result is True
    assert record.dd_trace_id == "12345"
    assert record.dd_span_id == "67890"
