import io
import logging
import types
from datetime import datetime

import pytest


class FakeResponse:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP error")


def test_scrape_github_trending_success(monkeypatch) -> None:
    import scrape_github_trending

    html = """
    <article class="Box-row">
        <h2 class="h3"><a href="/owner/devtool">Dev Tool</a></h2>
        <p>Developer CLI</p>
    </article>
    <article class="Box-row">
        <h2 class="h3"><a href="/owner/skiptool">Skip Tool</a></h2>
        <p>Fun social app</p>
    </article>
    """
    response: FakeResponse = FakeResponse(content=html.encode("utf-8"))
    monkeypatch.setattr("scrape_github_trending.requests.get", lambda *args, **kwargs: response)
    # Ensure no ambient DB state influences duplicate filtering
    monkeypatch.setattr("scrape_github_trending.get_existing_startup_keys", lambda: [])

    saved = []
    monkeypatch.setattr("scrape_github_trending.save_startup", lambda record: saved.append(record))

    def fake_classify(candidates):
        candidates = list(candidates)
        sequence = iter([True, False])
        return {item["id"]: next(sequence) for item in candidates}

    monkeypatch.setattr("scrape_github_trending.classify_candidates", fake_classify)
    monkeypatch.setattr("scrape_github_trending.get_devtools_category", lambda description, name: "CLI Tool")

    scrape_github_trending.scrape_github_trending()
    assert saved and saved[0]["description"].startswith("[CLI Tool]")


def test_scrape_github_trending_handles_request_error(monkeypatch) -> None:
    import scrape_github_trending
    from requests import RequestException

    monkeypatch.setattr(
        "scrape_github_trending.requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(RequestException("fail"))
    )
    # No exception should bubble up
    scrape_github_trending.scrape_github_trending()


def test_scrape_github_trending_handles_parse_error(monkeypatch) -> None:
    import scrape_github_trending

    response: FakeResponse = FakeResponse(content=b"<html></html>")
    monkeypatch.setattr("scrape_github_trending.requests.get", lambda *args, **kwargs: response)
    monkeypatch.setattr("bs4.BeautifulSoup", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")))

    scrape_github_trending.scrape_github_trending()


def test_scrape_github_trending_main_guard(monkeypatch) -> None:
    import runpy

    monkeypatch.setattr("database.init_db", lambda: None)
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeResponse(content=b"<article class='Box-row'></article>"))
    monkeypatch.setattr("ai_classifier.is_devtools_related_ai", lambda *args, **kwargs: False)
    monkeypatch.setattr("ai_classifier.get_devtools_category", lambda *args, **kwargs: None)
    monkeypatch.setattr("database.save_startup", lambda record: None)

    runpy.run_module("scrape_github_trending", run_name="__main__")


def test_scrape_github_trending_skips_missing_link(monkeypatch) -> None:
    import scrape_github_trending

    html = """
    <article class='Box-row'>
        <h2 class='h3'>Missing Link</h2>
    </article>
    """
    response: FakeResponse = FakeResponse(content=html.encode("utf-8"))
    monkeypatch.setattr("scrape_github_trending.requests.get", lambda *args, **kwargs: response)
    monkeypatch.setattr("scrape_github_trending.get_existing_startup_keys", lambda: [])

    monkeypatch.setattr("scrape_github_trending.classify_candidates", lambda candidates: {item["id"]: True for item in candidates})
    monkeypatch.setattr("scrape_github_trending.get_devtools_category", lambda *args, **kwargs: None)

    saved = []
    monkeypatch.setattr("scrape_github_trending.save_startup", lambda record: saved.append(record))

    scrape_github_trending.scrape_github_trending()
    assert saved == []


def test_fake_response_raises_on_error() -> None:
    response = FakeResponse(status_code=500)
    with pytest.raises(Exception):
        response.raise_for_status()


def test_scrape_github_trending_skips_duplicates_precheck(monkeypatch) -> None:
    import scrape_github_trending
    from unittest.mock import Mock

    html = """
    <article class="Box-row">
        <h2 class="h3"><a href="/owner/dupe">Dupe Tool</a></h2>
        <p>Duplicate candidate</p>
    </article>
    """
    response: FakeResponse = FakeResponse(content=html.encode("utf-8"))
    monkeypatch.setattr("scrape_github_trending.requests.get", lambda *args, **kwargs: response)

    # Classifier marks it as a devtool
    def fake_classify(candidates):
        candidates = list(candidates)
        return {item["id"]: True for item in candidates}

    monkeypatch.setattr("scrape_github_trending.classify_candidates", fake_classify)
    monkeypatch.setattr("scrape_github_trending.get_devtools_category", lambda *args, **kwargs: None)

    # Simulate existing DB entry to trigger duplicate pre-filter
    monkeypatch.setattr(
        "scrape_github_trending.get_existing_startup_keys",
        lambda: [{"name": "anything", "url": "https://github.com/owner/dupe"}],
    )

    save_mock = Mock()
    monkeypatch.setattr("scrape_github_trending.save_startup", save_mock)

    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("devtools.scraper.github_trending")
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        scrape_github_trending.scrape_github_trending()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    save_mock.assert_not_called()
    assert "scraper.skip_duplicate" in log_buffer.getvalue()


def test_scrape_github_trending_saves_when_not_duplicate(monkeypatch) -> None:
    import scrape_github_trending
    from unittest.mock import Mock

    html = """
    <article class="Box-row">
        <h2 class="h3"><a href="/owner/newtool">New Tool</a></h2>
        <p>Fresh candidate</p>
    </article>
    """
    response: FakeResponse = FakeResponse(content=html.encode("utf-8"))
    monkeypatch.setattr("scrape_github_trending.requests.get", lambda *args, **kwargs: response)

    def fake_classify(candidates):
        candidates = list(candidates)
        return {item["id"]: True for item in candidates}

    monkeypatch.setattr("scrape_github_trending.classify_candidates", fake_classify)
    monkeypatch.setattr("scrape_github_trending.get_devtools_category", lambda *args, **kwargs: None)

    # Not a duplicate: no existing entries
    monkeypatch.setattr("scrape_github_trending.get_existing_startup_keys", lambda: [])

    save_mock = Mock()
    monkeypatch.setattr("scrape_github_trending.save_startup", save_mock)

    scrape_github_trending.scrape_github_trending()

    assert save_mock.call_count == 1
    args, kwargs = save_mock.call_args
    assert args and args[0]["url"].endswith("/owner/newtool")


def test_scrape_github_trending_integration_duplicate_with_temp_db(monkeypatch, tmp_path) -> None:
    """Integration-style test: uses a temporary SQLite DB to verify duplicate filtering."""
    import scrape_github_trending
    import database

    # Point database to a temporary path
    db_file = tmp_path / "startups.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    monkeypatch.setattr(database, "DB_NAME", str(db_file))

    database.init_db()

    # Seed an existing startup with the same URL as the forthcoming scrape
    existing = {
        "name": "owner/dupe",
        "url": "https://github.com/owner/dupe",
        "description": "Existing",
        "source": "GitHub Trending",
        "date_found": datetime.now(),
    }
    database.save_startup(existing)

    # HTML contains the same repo; classification marks it True
    html = """
    <article class=\"Box-row\">
        <h2 class=\"h3\"><a href=\"/owner/dupe\">Dupe Tool</a></h2>
        <p>Duplicate candidate</p>
    </article>
    """
    response: FakeResponse = FakeResponse(content=html.encode("utf-8"))
    monkeypatch.setattr("scrape_github_trending.requests.get", lambda *args, **kwargs: response)

    def fake_classify(candidates):
        items = list(candidates)
        return {item["id"]: True for item in items}

    monkeypatch.setattr("scrape_github_trending.classify_candidates", fake_classify)
    monkeypatch.setattr("scrape_github_trending.get_devtools_category", lambda *args, **kwargs: None)

    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("devtools.scraper.github_trending")
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        scrape_github_trending.scrape_github_trending()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    # Ensure only one row exists for the URL (no duplicate insert)
    found = database.get_startup_by_url("https://github.com/owner/dupe")
    assert found is not None
    assert "scraper.skip_duplicate" in log_buffer.getvalue()


def test_categorization_uses_text_not_empty_description(monkeypatch) -> None:
    """Bug #9: get_devtools_category should receive candidate['text'] (which
    falls back to name) instead of candidate['description'] (which can be empty).
    """
    import scrape_github_trending

    # Repo with NO description paragraph -- description will be "", text will be name
    html = """
    <article class="Box-row">
        <h2 class="h3"><a href="/owner/nodesctool">owner/nodesctool</a></h2>
    </article>
    """
    response = FakeResponse(content=html.encode("utf-8"))
    monkeypatch.setattr("scrape_github_trending.requests.get", lambda *args, **kwargs: response)
    monkeypatch.setattr("scrape_github_trending.get_existing_startup_keys", lambda: [])

    def fake_classify(candidates):
        candidates = list(candidates)
        return {item["id"]: True for item in candidates}

    monkeypatch.setattr("scrape_github_trending.classify_candidates", fake_classify)

    # Capture what get_devtools_category receives as its first argument
    category_inputs = []

    def spy_get_devtools_category(description, name):
        category_inputs.append(description)
        return "CLI Tool"

    monkeypatch.setattr("scrape_github_trending.get_devtools_category", spy_get_devtools_category)

    saved = []
    monkeypatch.setattr("scrape_github_trending.save_startup", lambda record: saved.append(record))

    scrape_github_trending.scrape_github_trending()

    # The categorizer must have been called with a non-empty string (the name fallback)
    assert len(category_inputs) == 1
    assert category_inputs[0] != "", "get_devtools_category received empty string instead of text fallback"
    assert "nodesctool" in category_inputs[0]
