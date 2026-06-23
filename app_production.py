"""Flask web application serving the DevTools Scrape frontend and JSON API."""

import os
import secrets
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, g

from database import (
    SOURCE_REGISTRY,
    classify_source,
    count_all_startups,
    count_search_results,
    count_startups_by_source_key,
    get_all_startups,
    get_last_scrape_time,
    get_related_startups,
    get_source_counts,
    get_startup_by_id,
    get_startups_by_source_key,
    init_db,
    search_startups,
)
from chatbot import generate_chat_response
from logging_config import bind_context, get_logger, unbind_context
from observability import (
    generate_trace_id_w3c,
    install_custom_trace_id_filter,
    tag_root_span_with_custom_trace_id,
)

# Load environment variables
load_dotenv()

app = Flask(__name__)
logger = get_logger("devtools.app")

# Initialize database
init_db()

# Production configuration
_secret_key = os.getenv('SECRET_KEY')
if not _secret_key:
    _secret_key = secrets.token_hex(32)
    logger.warning("config.generated_secret_key", extra={"event": "config.generated_secret_key"})
app.config['SECRET_KEY'] = _secret_key
app.config['DEBUG'] = False

def _truthy_env(var_name: str, default: bool = False) -> bool:
    """Parse an environment variable as a boolean flag."""
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _safe_float_env(var_name: str, default: float) -> float:
    """Read an environment variable as a float, returning default on failure."""
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


_CUSTOM_TRACE_ID_ENABLED = _truthy_env("CUSTOM_TRACE_ID_ENABLED", default=False)


def _parse_csv_env(var_name: str) -> list[str]:
    """Split a comma-separated environment variable into a list of strings."""
    raw_value = os.getenv(var_name, "")
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _safe_int(value, default: int) -> int:
    """Convert value to int, returning default on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rum_script_source(site: str) -> str:
    """Return the Datadog RUM browser agent script URL for the given site."""
    region_map = {
        "datadoghq.com": "us1",
        "datadoghq.eu": "eu1",
        "us3.datadoghq.com": "us3",
        "us5.datadoghq.com": "us5",
        "ap1.datadoghq.com": "ap1",
        "ddog-gov.com": "us1-gov",
    }
    region = region_map.get(site.lower(), "us1")
    version = os.getenv("DATADOG_RUM_BROWSER_VERSION", "v6")
    version_segment = version if version.startswith("v") else f"v{version}"
    return f"https://www.datadoghq-browser-agent.com/{region}/{version_segment}/datadog-rum.js"


def _build_rum_context() -> Optional[Dict[str, Any]]:
    """Build the Datadog RUM configuration dict, or None when credentials are missing."""
    application_id = os.getenv("DATADOG_RUM_APPLICATION_ID")
    client_token = os.getenv("DATADOG_RUM_CLIENT_TOKEN")
    if not application_id or not client_token:
        return None

    site = os.getenv("DATADOG_RUM_SITE", os.getenv("DD_SITE", "datadoghq.com"))
    service_name = os.getenv("DATADOG_RUM_SERVICE", os.getenv("DD_SERVICE", "devtoolscrape"))
    environment = os.getenv("DATADOG_RUM_ENV", os.getenv("DD_ENV", "prod"))
    version = os.getenv("DATADOG_RUM_VERSION", os.getenv("DD_VERSION", "1.1"))
    allowed_tracing_urls = _parse_csv_env("DATADOG_RUM_ALLOWED_TRACING_URLS")
    if not allowed_tracing_urls:
        allowed_tracing_urls = [
            "https://devtoolscrape.com",
            "https://*.devtoolscrape.com",
        ]
        if request:
            allowed_tracing_urls.append(request.host_url.rstrip("/"))

    rum_config = {
        "applicationId": application_id,
        "clientToken": client_token,
        "site": site,
        "service": service_name,
        "env": environment,
        "version": version,
        "sampleRate": _safe_float_env("DATADOG_RUM_SAMPLE_RATE", 100.0),
        "traceSampleRate": _safe_float_env("DATADOG_RUM_TRACE_SAMPLE_RATE", 100.0),
        "profilingSampleRate": _safe_float_env("DATADOG_RUM_PROFILING_SAMPLE_RATE", 100.0),
        "tracePropagationMode": os.getenv("DATADOG_RUM_TRACE_PROPAGATION_MODE", "datadog"),
        "trackUserInteractions": _truthy_env("DATADOG_RUM_TRACK_USER_INTERACTIONS", True),
        "trackResources": _truthy_env("DATADOG_RUM_TRACK_RESOURCES", True),
        "trackLongTasks": _truthy_env("DATADOG_RUM_TRACK_LONG_TASKS", True),
        "defaultPrivacyLevel": os.getenv("DATADOG_RUM_DEFAULT_PRIVACY_LEVEL", "allow"),
        "allowedTracingUrls": allowed_tracing_urls,
    }

    session_replay_enabled = _truthy_env("DATADOG_RUM_SESSION_REPLAY", False)
    if session_replay_enabled:
        rum_config["sessionReplaySampleRate"] = _safe_float_env(
            "DATADOG_RUM_SESSION_REPLAY_SAMPLE_RATE",
            _safe_float_env("DATADOG_RUM_SAMPLE_RATE", 100.0),
        )

    return {
        "config": rum_config,
        "script_src": os.getenv("DATADOG_RUM_SCRIPT_SRC", _rum_script_source(site)),
        "enable_session_replay": session_replay_enabled,
    }


@app.context_processor
def inject_datadog_rum() -> Dict[str, Any]:
    """Inject Datadog RUM configuration into every template context."""
    return {"datadog_rum": _build_rum_context()}

if not app.debug:
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

def _apply_custom_trace_id():
    """Generate and apply a W3C trace ID to the current root span.

    Stores the custom trace ID on ``flask.g`` and in the logging context so
    that downstream log lines and the Datadog trace share the same identifier.
    """
    custom_trace_id_hex = generate_trace_id_w3c()
    original_trace_id = tag_root_span_with_custom_trace_id(custom_trace_id_hex)

    if original_trace_id is None:
        logger.debug(
            "trace.custom_id_skipped",
            extra={
                "event": "trace.custom_id_skipped",
                "reason": "no_root_span",
            },
        )
        return

    g.custom_trace_id = custom_trace_id_hex
    bind_context(custom_trace_id=custom_trace_id_hex)
    logger.debug(
        "trace.custom_id_applied",
        extra={
            "event": "trace.custom_id_applied",
            "custom_trace_id": custom_trace_id_hex,
            "original_trace_id": str(original_trace_id),
        },
    )


@app.before_request
def _start_request_logging():
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    g.request_id = request_id
    g.request_start_time = time.perf_counter()
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    bind_context(
        request_id=request_id,
        http_method=request.method,
        http_path=request.path,
        client_ip=client_ip,
    )

    if _CUSTOM_TRACE_ID_ENABLED:
        _apply_custom_trace_id()

    logger.debug(
        "request.start",
        extra={
            "event": "request.start",
            "http_method": request.method,
            "http_path": request.path,
            "client_ip": client_ip,
        },
    )


@app.after_request
def _complete_request_logging(response):
    duration_ms = None
    if hasattr(g, "request_start_time"):
        duration_ms = round((time.perf_counter() - g.request_start_time) * 1000, 2)
    logger.info(
        "request.complete",
        extra={
            "event": "request.complete",
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "content_length": response.calculate_content_length(),
        },
    )
    return response


@app.teardown_request
def _teardown_request_logging(exc):
    if exc is not None:
        logger.exception(
            "request.error",
            extra={
                "event": "request.error",
                "error_type": exc.__class__.__name__,
            },
        )
    keys_to_unbind = ["request_id", "http_method", "http_path", "client_ip"]
    if hasattr(g, "custom_trace_id"):
        keys_to_unbind.append("custom_trace_id")
    unbind_context(*keys_to_unbind)

def _parse_pagination(default_per_page: int = 20, max_per_page: int = 100):
    """Parse page and per_page from request args, returning (page, per_page, offset)."""
    raw_per_page = _safe_int(request.args.get('per_page', default_per_page), default_per_page)
    per_page = min(max(raw_per_page, 1), max_per_page)
    page = max(_safe_int(request.args.get('page', 1), 1), 1)
    offset = (page - 1) * per_page
    return page, per_page, offset


def _total_pages(total_results: int, per_page: int) -> int:
    """Compute total number of pages, minimum 1."""
    return max((total_results + per_page - 1) // per_page, 1)


def _pagination_vars(items: list, total_results: int, page: int, per_page: int, offset: int) -> dict:
    """Compute the common pagination template variables."""
    return {
        "page": page,
        "per_page": per_page,
        "total_pages": _total_pages(total_results, per_page) if total_results else 1,
        "total_results": total_results,
        "first_item": offset + 1 if items else 0,
        "last_item": offset + len(items),
    }


def summarize_sources(startups: list[Dict[str, Any]]) -> Dict[str, int]:
    """Aggregate startup counts by source category."""
    logger.debug(
        "summarize.sources",
        extra={"event": "summarize.sources", "startup_count": len(startups)},
    )
    counts: dict[str, int] = {"total": len(startups), "other": 0}
    for key in SOURCE_REGISTRY:
        counts[key] = 0

    for startup in startups:
        bucket = classify_source(startup["source"])
        counts[bucket] += 1

    return counts

@app.route('/')
def index():
    """Main page showing all devtools"""
    source_filter = request.args.get('source', '')
    page, per_page, offset = _parse_pagination()

    if source_filter:
        total_results = count_startups_by_source_key(source_filter)
        startups = get_startups_by_source_key(source_filter, limit=per_page, offset=offset)
    else:
        total_results = count_all_startups()
        startups = get_all_startups(limit=per_page, offset=offset)

    paging = _pagination_vars(startups, total_results, page, per_page, offset)
    response = render_template(
        'index.html',
        startups=startups,
        source_counts=get_source_counts(),
        current_filter=source_filter,
        last_scrape_time=get_last_scrape_time(),
        **paging,
    )
    logger.info(
        "render.index",
        extra={
            "event": "render.index",
            "source_filter": source_filter or "all",
            "page": page,
            "per_page": per_page,
            "returned": len(startups),
            "total_results": total_results,
        },
    )
    return response

@app.route('/source/<source_name>')
def filter_by_source(source_name):
    """Filter tools by source"""
    page, per_page, offset = _parse_pagination()

    filtered_startups = get_startups_by_source_key(source_name, limit=per_page, offset=offset)
    entry = SOURCE_REGISTRY.get(source_name)
    source_display = entry["display"] if entry else "All Sources"

    total_results = count_startups_by_source_key(source_name)
    paging = _pagination_vars(filtered_startups, total_results, page, per_page, offset)
    response = render_template(
        'index.html',
        startups=filtered_startups,
        source_counts=get_source_counts(),
        current_filter=source_name,
        source_display=source_display,
        last_scrape_time=get_last_scrape_time(),
        **paging,
    )
    logger.info(
        "render.source",
        extra={
            "event": "render.source",
            "source": source_name,
            "page": page,
            "per_page": per_page,
            "returned": len(filtered_startups),
            "total_results": total_results,
        },
    )
    return response

@app.route('/search')
def search():
    """Search page"""
    query = request.args.get('q', '')
    page, per_page, offset = _parse_pagination()

    if query:
        total_results = count_search_results(query)
        startups = search_startups(query, limit=per_page, offset=offset)
    else:
        total_results = 0
        startups = []
    paging = _pagination_vars(startups, total_results, page, per_page, offset)
    response = render_template(
        'search.html',
        startups=startups,
        query=query,
        source_counts=summarize_sources(startups),
        last_scrape_time=get_last_scrape_time(),
        **paging,
    )
    logger.info(
        "render.search",
        extra={
            "event": "render.search",
            "query": query,
            "page": page,
            "per_page": per_page,
            "returned": len(startups),
            "total_results": total_results,
        },
    )
    return response

@app.route('/tool/<int:tool_id>')
def tool_detail(tool_id):
    """Show detailed view of a specific tool"""
    tool = get_startup_by_id(tool_id)
    if not tool:
        logger.warning(
            "render.tool_missing",
            extra={"event": "render.tool_missing", "tool_id": tool_id},
        )
        return "Tool not found", 404

    related = get_related_startups(tool['source'], tool['id'], limit=4)

    last_scrape_time = get_last_scrape_time()
    logger.info(
        "render.tool_detail",
        extra={
            "event": "render.tool_detail",
            "tool_id": tool_id,
            "source": tool.get("source"),
            "related_count": len(related),
        },
    )
    return render_template('tool_detail.html', tool=tool, startups=related, last_scrape_time=last_scrape_time)

@app.route('/api/startups')
def api_startups():
    """API endpoint for getting all startups"""
    page, per_page, offset = _parse_pagination(default_per_page=50, max_per_page=200)

    startups = get_all_startups(limit=per_page, offset=offset)
    total = count_all_startups()
    payload = {
        'items': startups,
        'total': total,
        **_pagination_vars(startups, total, page, per_page, offset),
    }
    logger.info(
        "api.startups",
        extra={
            "event": "api.startups",
            "page": page,
            "per_page": per_page,
            "returned": len(startups),
            "total": total,
        },
    )
    return jsonify(payload)

@app.route('/api/search')
def api_search():
    """API endpoint for searching startups"""
    query = request.args.get('q', '')
    page, per_page, offset = _parse_pagination(max_per_page=200)

    if query:
        total = count_search_results(query)
        startups = search_startups(query, limit=per_page, offset=offset)
    else:
        total = 0
        startups = []

    payload = {
        'items': startups,
        'total': total,
        **_pagination_vars(startups, total, page, per_page, offset),
    }
    logger.info(
        "api.search",
        extra={
            "event": "api.search",
            "query": query,
            "page": page,
            "per_page": per_page,
            "returned": len(startups),
            "total": total,
        },
    )
    return jsonify(payload)


# Chat endpoint rate limiting
_chat_rate_limits: dict[str, list[float]] = defaultdict(list)
_CHAT_RATE_LIMIT = int(os.getenv("CHATBOT_RATE_LIMIT", "10"))
_CHAT_RATE_WINDOW = 60


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chatbot tool recommendations."""
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    now = time.time()
    _chat_rate_limits[client_ip] = [
        t for t in _chat_rate_limits[client_ip] if now - t < _CHAT_RATE_WINDOW
    ]
    if len(_chat_rate_limits[client_ip]) >= _CHAT_RATE_LIMIT:
        logger.warning(
            "api.chat.rate_limited",
            extra={"event": "api.chat.rate_limited", "client_ip": client_ip},
        )
        return jsonify({
            'error': 'Rate limit exceeded',
            'response': 'You are sending messages too quickly. Please wait a moment and try again.',
            'tools': [],
        }), 429
    data = request.get_json(silent=True)
    user_message = (data.get("message", "") if data else "").strip()
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    if len(user_message) > 500:
        return jsonify({"error": "Message too long (max 500 characters)"}), 400

    _chat_rate_limits[client_ip].append(now)
    session_id = (data.get("session_id") if data else None)
    result = generate_chat_response(user_message, session_id=session_id)
    logger.info(
        "api.chat",
        extra={
            "event": "api.chat",
            "message_length": len(user_message),
            "tools_found": len(result.get("tools", [])),
        },
    )
    return jsonify(result)


@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    logger.debug("health_check", extra={"event": "health_check"})

    # Temporary diagnostic: check if trace filter is installed in this worker.
    filter_status = "unknown"
    if _CUSTOM_TRACE_ID_ENABLED:
        try:
            from ddtrace import tracer as _dt
            agg = getattr(_dt, "_span_aggregator", None)
            if agg is not None:
                user_procs = getattr(agg, "user_processors", [])
                filter_names = [type(p).__name__ for p in user_procs]
                filter_status = f"installed({','.join(filter_names)})" if filter_names else "empty"
            else:
                filter_status = "no_aggregator"
        except Exception as exc:
            filter_status = f"error:{exc}"

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': 'connected',
        'trace_filter': filter_status,
    })


def _parse_iso_date(date_str):
    """Parse an ISO date string, returning None on failure or non-string input."""
    if not isinstance(date_str, str):
        return None
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None


@app.template_filter('format_date')
def format_date(date_str):
    """Format date for display"""
    dt = _parse_iso_date(date_str)
    return dt.strftime('%B %d, %Y') if dt else date_str


@app.template_filter('format_datetime')
def format_datetime(date_str):
    """Format datetime for display"""
    dt = _parse_iso_date(date_str)
    return dt.strftime('%B %d, %Y at %I:%M %p') if dt else date_str


# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    logger.warning("http.404", extra={"event": "http.404"})
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    logger.exception("http.500", extra={"event": "http.500"})
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Production settings
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    
    app.run(host=host, port=port, debug=False)
