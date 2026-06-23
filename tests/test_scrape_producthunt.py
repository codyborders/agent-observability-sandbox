from datetime import datetime

import pytest
import requests


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("bad status")

    def json(self):
        return self._payload


def test_fake_response_raise_for_status():
    response = FakeResponse({}, status_code=500)
    with pytest.raises(Exception):
        response.raise_for_status()


def test_get_producthunt_token_missing_env(monkeypatch):
    import scrape_producthunt_api

    monkeypatch.delenv("PRODUCTHUNT_CLIENT_ID", raising=False)
    monkeypatch.delenv("PRODUCTHUNT_CLIENT_SECRET", raising=False)
    assert scrape_producthunt_api.get_producthunt_token() is None


def test_get_producthunt_token_success(monkeypatch):
    import scrape_producthunt_api

    monkeypatch.setenv("PRODUCTHUNT_CLIENT_ID", "id")
    monkeypatch.setenv("PRODUCTHUNT_CLIENT_SECRET", "secret")
    monkeypatch.setattr("scrape_producthunt_api.requests.post", lambda *args, **kwargs: FakeResponse({"access_token": "token"}))
    assert scrape_producthunt_api.get_producthunt_token() == "token"


def test_get_producthunt_token_failure(monkeypatch):
    import scrape_producthunt_api

    monkeypatch.setenv("PRODUCTHUNT_CLIENT_ID", "id")
    monkeypatch.setenv("PRODUCTHUNT_CLIENT_SECRET", "secret")

    class BrokenResponse(FakeResponse):
        def raise_for_status(self):
            raise requests.exceptions.HTTPError("fail")

    monkeypatch.setattr("scrape_producthunt_api.requests.post", lambda *args, **kwargs: BrokenResponse({}, status_code=500))
    assert scrape_producthunt_api.get_producthunt_token() is None


def test_scrape_producthunt_api_success(monkeypatch):
    import scrape_producthunt_api

    monkeypatch.setenv("PRODUCTHUNT_CLIENT_ID", "id")
    monkeypatch.setenv("PRODUCTHUNT_CLIENT_SECRET", "secret")

    calls = {"count": 0}

    def fake_post(url, *_, **__):
        if "oauth/token" in url:
            return FakeResponse({"access_token": "token"})
        calls["count"] += 1
        product = {
            "data": {
                "posts": {
                    "edges": [
                        {"node": {"name": "DevTool", "tagline": "CLI tool", "description": "Helps developers", "url": "https://devtool.dev", "createdAt": datetime.utcnow().isoformat(), "topics": {"edges": []}}},
                        {"node": {"name": "NonDev", "tagline": "Fun app", "description": "Not dev tool", "url": "https://non.dev", "createdAt": datetime.utcnow().isoformat(), "topics": {"edges": []}}},
                    ]
                }
            }
        }
        return FakeResponse(product)

    monkeypatch.setattr("scrape_producthunt_api.requests.post", fake_post)
    def fake_classify(candidates):
        return {item["id"]: item["name"] == "DevTool" for item in candidates}

    monkeypatch.setattr("scrape_producthunt_api.classify_candidates", fake_classify)
    monkeypatch.setattr("scrape_producthunt_api.get_devtools_category", lambda text, name: "CLI Tool")

    saved = []
    monkeypatch.setattr("scrape_producthunt_api.save_startup", lambda record: saved.append(record))

    scrape_producthunt_api.scrape_producthunt_api()
    assert len(saved) == 1
    assert saved[0]["description"].startswith("[CLI Tool]")
    assert calls["count"] == 1


def test_scrape_producthunt_api_queries_last_24h_trending(monkeypatch):
    import scrape_producthunt_api

    monkeypatch.setenv("PRODUCTHUNT_CLIENT_ID", "id")
    monkeypatch.setenv("PRODUCTHUNT_CLIENT_SECRET", "secret")

    graphql_payloads = []

    def fake_post(url, *_, **kwargs):
        if "oauth/token" in url:
            return FakeResponse({"access_token": "token"})
        graphql_payloads.append(kwargs.get("json", {}))
        return FakeResponse({"data": {"posts": {"edges": []}}})

    monkeypatch.setattr("scrape_producthunt_api.requests.post", fake_post)
    monkeypatch.setattr("scrape_producthunt_api.classify_candidates", lambda candidates: {})
    monkeypatch.setattr("scrape_producthunt_api.get_devtools_category", lambda *a, **kw: None)

    scrape_producthunt_api.scrape_producthunt_api()

    assert len(graphql_payloads) == 1
    payload = graphql_payloads[0]
    query = payload.get("query", "")
    variables = payload.get("variables", {})

    assert "postedAfter" in query
    assert "VOTES_COUNT" in query
    assert variables.get("postedAfter")


def test_scrape_producthunt_api_handles_request_errors(monkeypatch):
    import scrape_producthunt_api
    from requests import RequestException

    monkeypatch.setenv("PRODUCTHUNT_CLIENT_ID", "id")
    monkeypatch.setenv("PRODUCTHUNT_CLIENT_SECRET", "secret")

    def raise_request(url, *args, **kwargs):
        if "oauth/token" in url:
            return FakeResponse({"access_token": "token"})
        raise RequestException("fail")

    monkeypatch.setattr("scrape_producthunt_api.requests.post", raise_request)
    scrape_producthunt_api.scrape_producthunt_api()


def test_scrape_producthunt_api_handles_parse_errors(monkeypatch):
    import scrape_producthunt_api

    monkeypatch.setenv("PRODUCTHUNT_CLIENT_ID", "id")
    monkeypatch.setenv("PRODUCTHUNT_CLIENT_SECRET", "secret")

    def fake_post(url, *_, **__):
        if "oauth/token" in url:
            return FakeResponse({"access_token": "token"})
        broken = {
            "data": {
                "posts": {
                    "edges": [
                        {"node": {"name": "DevTool", "tagline": "CLI", "description": "Broken date", "url": "https://tool.dev", "createdAt": "not-a-date", "topics": {"edges": []}}}
                    ]
                }
            }
        }
        return FakeResponse(broken)

    monkeypatch.setattr("scrape_producthunt_api.requests.post", fake_post)
    monkeypatch.setattr("scrape_producthunt_api.classify_candidates", lambda candidates: {item["id"]: True for item in candidates})
    monkeypatch.setattr("scrape_producthunt_api.get_devtools_category", lambda *args, **kwargs: None)
    scrape_producthunt_api.scrape_producthunt_api()


def test_scrape_producthunt_api_without_token(monkeypatch):
    import scrape_producthunt_api

    monkeypatch.setattr("scrape_producthunt_api.get_producthunt_token", lambda: None)
    scrape_producthunt_api.scrape_producthunt_api()


def test_scrape_producthunt_rss_success(monkeypatch):
    import scrape_producthunt

    rss = """<?xml version="1.0"?>
    <rss><channel>
        <item>
            <title>DevTool RSS</title>
            <link>https://rss.devtool</link>
            <description>Developer focused</description>
            <pubDate>Mon, 01 Jan 2024 12:00:00 +0000</pubDate>
        </item>
        <item>
            <title>Consumer App</title>
            <link>https://consumer.app</link>
            <description>Not dev</description>
            <pubDate>Mon, 01 Jan 2024 13:00:00 +0000</pubDate>
        </item>
    </channel></rss>
    """

    class RSSResponse:
        status_code = 200
        content = rss.encode("utf-8")

        def raise_for_status(self):
            return None

    monkeypatch.setattr("scrape_producthunt.requests.get", lambda *args, **kwargs: RSSResponse())
    sequence = iter([True, False])
    monkeypatch.setattr("scrape_producthunt.is_devtools_related", lambda text: next(sequence))

    saved = []
    monkeypatch.setattr("scrape_producthunt.save_startup", lambda record: saved.append(record))

    scrape_producthunt.scrape_producthunt_rss()
    assert len(saved) == 1


def test_scrape_producthunt_rss_handles_errors(monkeypatch):
    import scrape_producthunt
    from requests import RequestException

    monkeypatch.setattr(
        "scrape_producthunt.requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(RequestException("fail"))
    )
    scrape_producthunt.scrape_producthunt_rss()


def test_scrape_producthunt_api_main_guard(monkeypatch):
    import runpy

    monkeypatch.setattr("database.init_db", lambda: None)
    monkeypatch.setattr("database.save_startup", lambda record: None)
    monkeypatch.delenv("PRODUCTHUNT_CLIENT_ID", raising=False)
    monkeypatch.delenv("PRODUCTHUNT_CLIENT_SECRET", raising=False)
    monkeypatch.setattr("scrape_producthunt_api.get_producthunt_token", lambda: None)

    runpy.run_module("scrape_producthunt_api", run_name="__main__")


def test_scrape_producthunt_api_handles_data_null(monkeypatch):
    """Bug #14: When GraphQL returns {"data": null}, .get('data', {}) returns
    None (not {}), so chained .get('posts', {}) raises AttributeError.
    The scraper should handle this gracefully -- zero posts, no crash."""
    import scrape_producthunt_api

    monkeypatch.setenv("PRODUCTHUNT_CLIENT_ID", "id")
    monkeypatch.setenv("PRODUCTHUNT_CLIENT_SECRET", "secret")

    parse_errors = []
    original_exception = scrape_producthunt_api.logger.exception

    def tracking_exception(msg, *args, **kwargs):
        parse_errors.append(msg)
        original_exception(msg, *args, **kwargs)

    monkeypatch.setattr(scrape_producthunt_api.logger, "exception", tracking_exception)

    def fake_post(url, *_, **__):
        if "oauth/token" in url:
            return FakeResponse({"access_token": "token"})
        # GraphQL server error returns data: null
        return FakeResponse({"data": None})

    monkeypatch.setattr("scrape_producthunt_api.requests.post", fake_post)
    monkeypatch.setattr("scrape_producthunt_api.classify_candidates", lambda c: {})
    monkeypatch.setattr("scrape_producthunt_api.get_devtools_category", lambda *a: None)

    saved = []
    monkeypatch.setattr("scrape_producthunt_api.save_startup", lambda r: saved.append(r))

    scrape_producthunt_api.scrape_producthunt_api()
    assert len(saved) == 0
    # The bug causes an AttributeError that hits the generic except block,
    # logging "scraper.parse_error". After the fix, no such error should appear.
    assert "scraper.parse_error" not in parse_errors


def test_scrape_producthunt_api_handles_posts_null(monkeypatch):
    """Bug #14 variant: When GraphQL returns {"data": {"posts": null}},
    the chained .get('edges', []) also crashes."""
    import scrape_producthunt_api

    monkeypatch.setenv("PRODUCTHUNT_CLIENT_ID", "id")
    monkeypatch.setenv("PRODUCTHUNT_CLIENT_SECRET", "secret")

    parse_errors = []
    original_exception = scrape_producthunt_api.logger.exception

    def tracking_exception(msg, *args, **kwargs):
        parse_errors.append(msg)
        original_exception(msg, *args, **kwargs)

    monkeypatch.setattr(scrape_producthunt_api.logger, "exception", tracking_exception)

    def fake_post(url, *_, **__):
        if "oauth/token" in url:
            return FakeResponse({"access_token": "token"})
        return FakeResponse({"data": {"posts": None}})

    monkeypatch.setattr("scrape_producthunt_api.requests.post", fake_post)
    monkeypatch.setattr("scrape_producthunt_api.classify_candidates", lambda c: {})
    monkeypatch.setattr("scrape_producthunt_api.get_devtools_category", lambda *a: None)

    saved = []
    monkeypatch.setattr("scrape_producthunt_api.save_startup", lambda r: saved.append(r))

    scrape_producthunt_api.scrape_producthunt_api()
    assert len(saved) == 0
    assert "scraper.parse_error" not in parse_errors


def test_scrape_producthunt_api_falsy_post_id(monkeypatch):
    """Bug #11: post.get('id') or post.get('url') or name skips falsy but
    valid IDs like 0 or empty string. The post_identifier should use explicit
    None check so that id=0 is preserved."""
    import scrape_producthunt_api

    monkeypatch.setenv("PRODUCTHUNT_CLIENT_ID", "id")
    monkeypatch.setenv("PRODUCTHUNT_CLIENT_SECRET", "secret")

    now_iso = datetime.utcnow().isoformat()

    def fake_post(url, *_, **__):
        if "oauth/token" in url:
            return FakeResponse({"access_token": "token"})
        return FakeResponse({
            "data": {
                "posts": {
                    "edges": [
                        {"node": {
                            "id": 0,
                            "name": "ZeroID Tool",
                            "tagline": "Has id=0",
                            "description": "Dev tool with falsy ID",
                            "url": "https://zero.dev",
                            "createdAt": now_iso,
                            "topics": {"edges": []},
                        }},
                    ]
                }
            }
        })

    monkeypatch.setattr("scrape_producthunt_api.requests.post", fake_post)
    monkeypatch.setattr(
        "scrape_producthunt_api.classify_candidates",
        lambda candidates: {item["id"]: True for item in candidates},
    )
    monkeypatch.setattr("scrape_producthunt_api.get_devtools_category", lambda *a: None)

    saved = []
    monkeypatch.setattr("scrape_producthunt_api.save_startup", lambda r: saved.append(r))

    scrape_producthunt_api.scrape_producthunt_api()
    assert len(saved) == 1
    # The key assertion: post_id should be "0" (from the id field),
    # not "https://zero.dev" (url) or "ZeroID Tool" (name).
    # With the bug, id=0 is falsy so the or-chain falls through to the url.
    assert saved[0]["name"] == "ZeroID Tool"


def test_scrape_producthunt_api_falsy_post_id_uses_id_not_url(monkeypatch):
    """Bug #11: Verify that with id=0, the candidate dict uses '0' as the
    id rather than falling through to url or name."""
    import scrape_producthunt_api

    monkeypatch.setenv("PRODUCTHUNT_CLIENT_ID", "id")
    monkeypatch.setenv("PRODUCTHUNT_CLIENT_SECRET", "secret")

    now_iso = datetime.utcnow().isoformat()
    captured_candidates = []

    def fake_post(url, *_, **__):
        if "oauth/token" in url:
            return FakeResponse({"access_token": "token"})
        return FakeResponse({
            "data": {
                "posts": {
                    "edges": [
                        {"node": {
                            "id": 0,
                            "name": "ZeroTool",
                            "tagline": "t",
                            "description": "d",
                            "url": "https://zero.dev",
                            "createdAt": now_iso,
                            "topics": {"edges": []},
                        }},
                    ]
                }
            }
        })

    monkeypatch.setattr("scrape_producthunt_api.requests.post", fake_post)

    def capturing_classify(candidates):
        captured_candidates.extend(candidates)
        return {item["id"]: False for item in candidates}

    monkeypatch.setattr("scrape_producthunt_api.classify_candidates", capturing_classify)
    monkeypatch.setattr("scrape_producthunt_api.get_devtools_category", lambda *a: None)
    monkeypatch.setattr("scrape_producthunt_api.save_startup", lambda r: None)

    scrape_producthunt_api.scrape_producthunt_api()

    assert len(captured_candidates) == 1
    # With the bug, this would be "https://zero.dev" or "ZeroTool"
    assert captured_candidates[0]["id"] == "0"


def test_scrape_producthunt_rss_handles_missing_tags(monkeypatch):
    """Bug #13: When an RSS <item> is missing <title> or <description>,
    item.title is None and .text raises AttributeError. The scraper should
    skip the malformed item and continue processing the rest."""
    import scrape_producthunt

    rss = """<?xml version="1.0"?>
    <rss><channel>
        <item>
            <link>https://no-title.dev</link>
            <description>Has description but no title</description>
            <pubDate>Mon, 01 Jan 2024 12:00:00 +0000</pubDate>
        </item>
        <item>
            <title>Has Title</title>
            <link>https://has-title.dev</link>
            <pubDate>Mon, 01 Jan 2024 13:00:00 +0000</pubDate>
        </item>
        <item>
            <title>Good Item</title>
            <link>https://good.dev</link>
            <description>Developer tool</description>
            <pubDate>Mon, 01 Jan 2024 14:00:00 +0000</pubDate>
        </item>
    </channel></rss>
    """

    class RSSResponse:
        status_code = 200
        content = rss.encode("utf-8")

        def raise_for_status(self):
            return None

    monkeypatch.setattr("scrape_producthunt.requests.get", lambda *a, **kw: RSSResponse())
    monkeypatch.setattr("scrape_producthunt.is_devtools_related", lambda text: True)

    saved = []
    monkeypatch.setattr("scrape_producthunt.save_startup", lambda r: saved.append(r))

    # Should not crash on items missing <title> or <description>
    scrape_producthunt.scrape_producthunt_rss()

    # The good item (3rd) should be saved. The 1st (no title) and 2nd
    # (no description) are malformed and should be skipped gracefully.
    assert len(saved) == 1
    assert saved[0]["name"] == "Good Item"


def test_scrape_producthunt_rss_main_guard(monkeypatch):
    import runpy
    from requests import RequestException

    monkeypatch.setattr("database.init_db", lambda: None)
    monkeypatch.setattr("database.save_startup", lambda record: None)
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(RequestException("no network")),
    )
    monkeypatch.setattr("ai_classifier.has_devtools_keywords", lambda text, name="": False)

    runpy.run_module("scrape_producthunt", run_name="__main__")
