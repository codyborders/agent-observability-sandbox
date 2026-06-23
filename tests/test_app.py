import importlib
import sys
from datetime import datetime

import pytest


@pytest.fixture
def app_module(monkeypatch):
    import database

    monkeypatch.setattr(database, "init_db", lambda: None)
    sys.modules.pop("app_production", None)
    module = importlib.import_module("app_production")
    return module


def _sample_startups():
    return [
        {"id": 1, "name": "GitHub Tool", "url": "https://github.com/tool", "description": "For devs", "source": "GitHub Trending", "date_found": "2024-01-01T00:00:00"},
        {"id": 2, "name": "HN Tool", "url": "https://hn.tool", "description": "For devs", "source": "Hacker News (score: 10)", "date_found": "2024-01-02T00:00:00"},
        {"id": 3, "name": "Product Hunt Tool", "url": "https://ph.tool", "description": "For devs", "source": "Product Hunt", "date_found": "2024-01-03T00:00:00"},
        {"id": 4, "name": "Other Tool", "url": "https://other.tool", "description": "Misc", "source": "Indie Hackers", "date_found": "2024-01-04T00:00:00"},
    ]


def test_index_route_filters_sources(app_module, monkeypatch):
    module = app_module
    def _paginate(data, limit, offset):
        start = offset or 0
        end = start + limit if limit is not None else None
        return data[start:end]

    monkeypatch.setattr(
        module,
        "get_all_startups",
        lambda limit=None, offset=None: _paginate(_sample_startups(), limit, offset),
    )

    def fake_get_startups_by_source_key(key, limit=None, offset=None):
        data = _sample_startups()
        if key == "github":
            data = [s for s in data if s["source"] == "GitHub Trending"]
        elif key == "hackernews":
            data = [s for s in data if "Hacker News" in s["source"]]
        elif key == "producthunt":
            data = [s for s in data if s["source"] == "Product Hunt"]
        return _paginate(data, limit, offset)

    monkeypatch.setattr(module, "get_startups_by_source_key", fake_get_startups_by_source_key)
    monkeypatch.setattr(module, "count_all_startups", lambda: len(_sample_startups()))
    monkeypatch.setattr(module, "count_startups_by_source_key", lambda key: len(fake_get_startups_by_source_key(key)))
    monkeypatch.setattr(
        module,
        "get_source_counts",
        lambda: {
            "total": len(_sample_startups()),
            "github": 1,
            "hackernews": 1,
            "producthunt": 1,
            "other": 1,
        },
    )
    monkeypatch.setattr(module, "get_last_scrape_time", lambda: "2024-01-04T00:00:00")

    client = module.app.test_client()

    assert client.get("/").status_code == 200
    assert client.get("/?source=github").status_code == 200
    assert client.get("/?source=hackernews").status_code == 200
    assert client.get("/?source=producthunt").status_code == 200

    assert module.get_startups_by_source_key("other") == _sample_startups()


def test_filter_by_source_route_variants(app_module, monkeypatch):
    module = app_module
    monkeypatch.setattr(module, "get_all_startups", lambda limit=None, offset=None: _sample_startups())
    monkeypatch.setattr(module, "get_startups_by_source_key", lambda key, limit=None, offset=None: _sample_startups())
    monkeypatch.setattr(module, "count_all_startups", lambda: len(_sample_startups()))
    monkeypatch.setattr(module, "count_startups_by_source_key", lambda key: len(_sample_startups()))
    monkeypatch.setattr(
        module,
        "get_source_counts",
        lambda: {
            "total": len(_sample_startups()),
            "github": 1,
            "hackernews": 1,
            "producthunt": 1,
            "other": 1,
        },
    )
    monkeypatch.setattr(module, "get_last_scrape_time", lambda: "2024-01-04T00:00:00")

    client = module.app.test_client()
    assert client.get("/source/github").status_code == 200
    assert client.get("/source/hackernews").status_code == 200
    assert client.get("/source/producthunt").status_code == 200
    assert client.get("/source/other").status_code == 200


def test_search_route_with_and_without_query(app_module, monkeypatch):
    module = app_module
    monkeypatch.setattr(module, "search_startups", lambda q, limit=20, offset=0: _sample_startups()[offset:offset + limit] if q else [])
    monkeypatch.setattr(module, "count_search_results", lambda q: len(_sample_startups()) if q else 0)
    monkeypatch.setattr(module, "count_startups_by_source_key", lambda key: len(_sample_startups()))
    monkeypatch.setattr(module, "count_all_startups", lambda: len(_sample_startups()))
    monkeypatch.setattr(module, "get_last_scrape_time", lambda: None)

    client = module.app.test_client()
    assert client.get("/search?q=dev").status_code == 200
    assert client.get("/search?q=dev&page=2").status_code == 200
    assert client.get("/search").status_code == 200


def test_tool_detail_routes(app_module, monkeypatch):
    module = app_module
    sample = _sample_startups()

    def fake_get_startup_by_id(tool_id):
        return next((s for s in sample if s["id"] == tool_id), None)

    def fake_get_related_startups(source, exclude_id, limit=4):
        data = [s for s in sample if s["source"] == source and s["id"] != exclude_id]
        return data[:limit]

    monkeypatch.setattr(module, "get_startup_by_id", fake_get_startup_by_id)
    monkeypatch.setattr(module, "get_related_startups", fake_get_related_startups)
    monkeypatch.setattr(module, "get_last_scrape_time", lambda: None)

    client = module.app.test_client()
    assert client.get("/tool/1").status_code == 200
    assert client.get("/tool/3").status_code == 200
    assert client.get("/tool/2").status_code == 200
    assert client.get("/tool/4").status_code == 200
    assert client.get("/tool/999").status_code == 404
    related = module.get_related_startups("GitHub Trending", 1, limit=2)
    assert len(related) <= 2


def test_api_endpoints(app_module, monkeypatch):
    module = app_module
    monkeypatch.setattr(module, "get_all_startups", lambda limit=None, offset=None: _sample_startups()[offset or 0:(offset or 0) + limit] if limit is not None else _sample_startups())
    monkeypatch.setattr(module, "count_all_startups", lambda: len(_sample_startups()))
    monkeypatch.setattr(module, "search_startups", lambda q, limit=20, offset=0: _sample_startups()[offset:offset + limit] if q else [])
    monkeypatch.setattr(module, "count_search_results", lambda q: len(_sample_startups()) if q else 0)

    client = module.app.test_client()
    payload = client.get("/api/startups").get_json()
    assert payload["total"] == len(_sample_startups())
    assert len(payload["items"]) == len(_sample_startups())
    page2 = client.get("/api/startups?page=2&per_page=2").get_json()
    assert page2["page"] == 2
    assert len(page2["items"]) == 2

    search_payload = client.get("/api/search?q=dev").get_json()
    assert search_payload["total"] == len(_sample_startups())
    assert len(search_payload["items"]) <= search_payload["per_page"]
    empty_payload = client.get("/api/search").get_json()
    assert empty_payload["total"] == 0
    assert empty_payload["items"] == []


def test_health_and_error_handlers(app_module, monkeypatch):
    module = app_module
    with module.app.app_context():
        response = module.health_check()
        assert response.get_json()["status"] == "healthy"

    client = module.app.test_client()
    assert client.get("/nonexistent").status_code == 404

    with module.app.app_context():
        rendered, status = module.internal_error(Exception("boom"))
        assert status == 500


def test_template_filters(app_module):
    module = app_module
    formatted = module.format_date("2024-01-05T00:00:00")
    assert "January" in formatted
    assert module.format_date("invalid-date") == "invalid-date"
    assert module.format_date(123) == 123

    dt_formatted = module.format_datetime("2024-01-05T12:30:00")
    assert "12:30 PM" in dt_formatted
    assert module.format_datetime("invalid-date") == "invalid-date"
    assert module.format_datetime(123) == 123


def test_parse_pagination_defaults(app_module):
    module = app_module
    with module.app.test_request_context("/?"):
        page, per_page, offset = module._parse_pagination()
        assert page == 1
        assert per_page == 20
        assert offset == 0


def test_parse_pagination_custom_values(app_module):
    module = app_module
    with module.app.test_request_context("/?page=3&per_page=10"):
        page, per_page, offset = module._parse_pagination()
        assert page == 3
        assert per_page == 10
        assert offset == 20


def test_parse_pagination_clamps_values(app_module):
    module = app_module
    # per_page clamped to max
    with module.app.test_request_context("/?per_page=500"):
        page, per_page, offset = module._parse_pagination(max_per_page=100)
        assert per_page == 100
    # per_page clamped to min
    with module.app.test_request_context("/?per_page=0"):
        page, per_page, offset = module._parse_pagination()
        assert per_page == 1
    # page clamped to min
    with module.app.test_request_context("/?page=-5"):
        page, per_page, offset = module._parse_pagination()
        assert page == 1


def test_total_pages_helper(app_module):
    module = app_module
    assert module._total_pages(0, 20) == 1
    assert module._total_pages(1, 20) == 1
    assert module._total_pages(20, 20) == 1
    assert module._total_pages(21, 20) == 2
    assert module._total_pages(100, 10) == 10


def _stub_all_db(module, monkeypatch):
    """Wire up minimal stubs so every route can render without touching a real DB."""
    monkeypatch.setattr(module, "get_all_startups", lambda limit=None, offset=None: _sample_startups())
    monkeypatch.setattr(module, "get_startups_by_source_key", lambda key, limit=None, offset=None: _sample_startups())
    monkeypatch.setattr(module, "count_all_startups", lambda: len(_sample_startups()))
    monkeypatch.setattr(module, "count_startups_by_source_key", lambda key: len(_sample_startups()))
    monkeypatch.setattr(
        module,
        "get_source_counts",
        lambda: {"total": 4, "github": 1, "hackernews": 1, "producthunt": 1, "other": 1},
    )
    monkeypatch.setattr(module, "get_last_scrape_time", lambda: "2024-01-04T00:00:00")
    monkeypatch.setattr(module, "search_startups", lambda q, limit=20, offset=0: _sample_startups() if q else [])
    monkeypatch.setattr(module, "count_search_results", lambda q: len(_sample_startups()) if q else 0)


def test_safe_int_helper(app_module):
    """_safe_int returns default on non-numeric or None input."""
    module = app_module
    assert module._safe_int("abc", 5) == 5
    assert module._safe_int(None, 10) == 10
    assert module._safe_int("", 3) == 3
    assert module._safe_int("7", 1) == 7
    assert module._safe_int(42, 1) == 42


def test_non_numeric_page_returns_200_not_500(app_module, monkeypatch):
    """Non-numeric page/per_page params should fall back to defaults, not crash."""
    module = app_module
    _stub_all_db(module, monkeypatch)
    client = module.app.test_client()

    # index route
    assert client.get("/?page=abc").status_code == 200
    assert client.get("/?per_page=xyz").status_code == 200
    assert client.get("/?page=abc&per_page=xyz").status_code == 200

    # filter_by_source route
    assert client.get("/source/github?page=abc").status_code == 200
    assert client.get("/source/github?per_page=xyz").status_code == 200

    # search route
    assert client.get("/search?q=dev&page=abc").status_code == 200
    assert client.get("/search?q=dev&per_page=xyz").status_code == 200

    # api_startups route
    resp = client.get("/api/startups?page=abc&per_page=xyz")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["page"] >= 1
    assert payload["per_page"] >= 1

    # api_search route
    resp = client.get("/api/search?q=dev&page=abc&per_page=xyz")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["page"] >= 1
    assert payload["per_page"] >= 1


def test_pagination_vars_with_results(app_module):
    module = app_module
    items = [1, 2, 3]
    result = module._pagination_vars(items, total_results=50, page=2, per_page=10, offset=10)
    assert result["page"] == 2
    assert result["per_page"] == 10
    assert result["total_pages"] == 5
    assert result["total_results"] == 50
    assert result["first_item"] == 11
    assert result["last_item"] == 13


def test_pagination_vars_empty_results(app_module):
    module = app_module
    result = module._pagination_vars([], total_results=0, page=1, per_page=20, offset=0)
    assert result["total_pages"] == 1
    assert result["first_item"] == 0
    assert result["last_item"] == 0


def test_summarize_sources_uses_classify_source(app_module, monkeypatch):
    """summarize_sources should classify sources via the shared registry."""
    module = app_module
    startups = [
        {"source": "GitHub Trending"},
        {"source": "Hacker News (score: 50)"},
        {"source": "Show HN (score: 10)"},
        {"source": "Product Hunt"},
        {"source": "Indie Hackers"},
    ]
    counts = module.summarize_sources(startups)
    assert counts["total"] == 5
    assert counts["github"] == 1
    assert counts["hackernews"] == 2
    assert counts["producthunt"] == 1
    assert counts["other"] == 1


def test_app_main_guard(monkeypatch):
    import runpy

    monkeypatch.setattr("database.init_db", lambda: None)
    monkeypatch.setattr("flask.app.Flask.run", lambda self, *args, **kwargs: None, raising=False)

    runpy.run_module("app_production", run_name="__main__")
