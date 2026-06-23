"""Structured JSON logging with Datadog trace correlation."""

import logging
import logging.config
import logging.handlers
import os
import socket
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

try:  # pragma: no cover - optional dependency
    from pythonjsonlogger import jsonlogger
except Exception as exc:  # pragma: no cover - fail fast if missing
    raise RuntimeError("python-json-logger is required for structured logging") from exc

try:  # pragma: no cover - optional ddtrace correlation
    from ddtrace.helpers import get_correlation_ids  # type: ignore

    _DDTRACE_AVAILABLE = True
except Exception:  # pragma: no cover
    _DDTRACE_AVAILABLE = False


_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("devtools_log_context", default={})
_CONFIGURED = False


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _resolve_log_dir() -> str:
    requested = os.getenv("LOG_DIR", "/var/log/devtoolscrape")
    fallback = os.path.join(os.getcwd(), "logs")
    for path in (requested, fallback):
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError:
            continue
    # As a last resort, do not configure file logging
    print("Warning: unable to create log directory; defaulting to stdout only", file=sys.stderr)
    return ""


class DatadogJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that injects ISO timestamps and standard keys."""

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)

        if "timestamp" not in log_record:
            log_record["timestamp"] = getattr(record, "timestamp", _iso_timestamp())
        if "level" not in log_record:
            log_record["level"] = record.levelname
        if "logger" not in log_record:
            log_record["logger"] = record.name
        if "message" not in log_record:
            log_record["message"] = record.getMessage()

        # Ensure consistent casing for Datadog attributes
        if hasattr(record, "dd_trace_id") and record.dd_trace_id:
            log_record["dd.trace_id"] = record.dd_trace_id
        if hasattr(record, "dd_span_id") and record.dd_span_id:
            log_record["dd.span_id"] = record.dd_span_id

        # Preserve numeric types when possible
        for key, value in list(log_record.items()):
            if isinstance(value, datetime):
                log_record[key] = value.isoformat()


class ContextFilter(logging.Filter):
    """Populate log records with shared fields from context vars and defaults."""

    def __init__(self, service: str, env: str, hostname: str) -> None:
        super().__init__()
        self.service = service
        self.env = env
        self.hostname = hostname

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple data attachment
        record.timestamp = _iso_timestamp()
        record.service = getattr(record, "service", self.service)
        record.env = getattr(record, "env", self.env)
        record.hostname = getattr(record, "hostname", self.hostname)

        context = _CONTEXT.get()
        for key, value in context.items():
            setattr(record, key, value)

        if _DDTRACE_AVAILABLE:
            try:
                trace_id, span_id = get_correlation_ids()
            except Exception:
                trace_id = span_id = None
            if trace_id:
                record.dd_trace_id = str(trace_id)
            if span_id:
                record.dd_span_id = str(span_id)
        return True


def setup_logging() -> None:
    """Configure base logging for the application."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    service = os.getenv("DD_SERVICE", "devtoolscrape")
    env = os.getenv("DD_ENV", os.getenv("LOG_ENV", "local"))
    hostname = socket.gethostname()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    log_dir = _resolve_log_dir()
    log_filename = os.getenv("LOG_FILENAME", "app.log")
    log_path = os.path.join(log_dir, log_filename) if log_dir else None
    log_to_stdout = os.getenv("LOG_STDOUT", "true").lower() in {"1", "true", "yes"}

    formatters = {
        "json": {
            "()": "logging_config.DatadogJSONFormatter",
            "fmt": "%(timestamp)s %(level)s %(message)s",
        },
        "console": {
            "format": "%(timestamp)s | %(levelname)s | %(name)s | %(message)s",
        },
    }

    filters = {
        "context": {
            "()": "logging_config.ContextFilter",
            "service": service,
            "env": env,
            "hostname": hostname,
        }
    }

    handlers: Dict[str, Dict[str, Any]] = {}

    if log_path:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json" if log_format == "json" else "console",
            "filters": ["context"],
            "filename": log_path,
            "maxBytes": int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024)),
            "backupCount": int(os.getenv("LOG_BACKUP_COUNT", "5")),
            "encoding": "utf-8",
        }

    if log_to_stdout or not handlers:
        handlers["stdout"] = {
            "class": "logging.StreamHandler",
            "formatter": "json" if log_format == "json" else "console",
            "filters": ["context"],
            "stream": "ext://sys.stdout",
        }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "filters": filters,
        "handlers": handlers,
        "root": {
            "level": log_level,
            "handlers": list(handlers.keys()),
        },
        "loggers": {
            "devtools": {
                "level": log_level,
                "propagate": False,
                "handlers": list(handlers.keys()),
            },
            "openai": {
                "level": os.getenv("OPENAI_LOG_LEVEL", "WARNING"),
            },
            "urllib3": {
                "level": os.getenv("HTTP_LOG_LEVEL", "WARNING"),
            },
        },
    }

    logging.config.dictConfig(config)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def bind_context(**kwargs: Any) -> None:
    current = dict(_CONTEXT.get())
    for key, value in kwargs.items():
        if value is None:
            continue
        current[key] = value
    _CONTEXT.set(current)


def unbind_context(*keys: str) -> None:
    if not keys:
        _CONTEXT.set({})
        return
    current = dict(_CONTEXT.get())
    for key in keys:
        current.pop(key, None)
    _CONTEXT.set(current)


@contextmanager
def logging_context(**kwargs: Any) -> Iterator[None]:
    previous: Dict[str, Any] = dict(_CONTEXT.get())
    bind_context(**kwargs)
    try:
        yield
    finally:
        _CONTEXT.set(previous)
