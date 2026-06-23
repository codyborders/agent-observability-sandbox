from datetime import datetime
from typing import Any, Dict, List, Tuple, Union
from unittest.mock import MagicMock, call

import pytest


# Expected timeout tuple: (connection_timeout, read_timeout)
EXPECTED_TIMEOUT = (5, 10)


class TestBuildDescription:
    """Tests for the _build_description helper function."""

    def test_with_category_and_text(self):
        from scrape_hackernews import _build_description

        result = _build_description("My Tool", "details here", "CLI Tool")
        assert result == "[CLI Tool] My Tool\n\ndetails here"

    def test_with_category_no_text(self):
        from scrape_hackernews import _build_description

        result = _build_description("My Tool", "", "CLI Tool")
        assert result == "[CLI Tool] My Tool"

    def test_without_category_with_text(self):
        from scrape_hackernews import _build_description

        result = _build_description("My Tool", "details here", None)
        assert result == "My Tool\n\ndetails here"

    def test_without_category_no_text(self):
        from scrape_hackernews import _build_description

        result = _build_description("My Tool", "", None)
        assert result == "My Tool"


class FakeJSONResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("bad status")

    def json(self):
        return self._payload


def test_fake_json_response_raise_for_status():
    response = FakeJSONResponse({}, status_code=500)
    with pytest.raises(Exception):
        response.raise_for_status()


def test_scrape_hackernews_success(monkeypatch):
    import scrape_hackernews

    top_story_ids = [1, 2, 3, 4, 5]
    stories = {
        1: {"type": "story", "title": "Devtool Launch", "url": "https://devtool.com", "text": "Launch details", "score": 50, "time": datetime.now().timestamp()},
        2: {"type": "story", "title": "Skip due to score", "url": "https://skip.com", "text": "", "score": 5, "time": datetime.now().timestamp()},
        3: {"type": "job"},  # ignored non-story
        4: {"type": "story", "title": "Non Devtool", "url": "https://not.dev", "text": "", "score": 60, "time": datetime.now().timestamp()},
        5: {"type": "story", "title": "Categoryless Tool", "url": "https://categoryless.dev", "text": "More details", "score": 70, "time": datetime.now().timestamp()},
    }

    def fake_get(url, timeout):
        if url.endswith("topstories.json"):
            return FakeJSONResponse(top_story_ids)
        story_id = int(url.split("/")[-1].split(".")[0])
        return FakeJSONResponse(stories[story_id])

    monkeypatch.setattr("scrape_hackernews.requests.get", fake_get)

    def fake_classify(candidates):
        mapping = {}
        for item in candidates:
            mapping[item["id"]] = item["name"] in {"Devtool Launch", "Categoryless Tool"}
        return mapping

    monkeypatch.setattr("scrape_hackernews.classify_candidates", fake_classify)
    def categorizer(text, name):
        if name == "Devtool Launch":
            return "Build/Deploy"
        return None

    monkeypatch.setattr("scrape_hackernews.get_devtools_category", categorizer)

    saved = []
    monkeypatch.setattr("scrape_hackernews.save_startup", lambda record: saved.append(record))

    scrape_hackernews.scrape_hackernews()
    assert len(saved) == 2
    assert saved[0]["description"].startswith("[Build/Deploy]")
    assert "More details" in saved[1]["description"]


def test_scrape_hackernews_show_success(monkeypatch):
    import scrape_hackernews

    show_ids = [10, 11, 12, 13, 14]
    show_story = {
        10: {"type": "story", "title": "Show HN: Dev CLI", "url": "https://devcli.dev", "text": "Details", "score": 40, "time": datetime.now().timestamp()},
        11: {"type": "story", "title": "Show HN: Dev Notes", "url": "https://devnotes.dev", "text": "Extras", "score": 60, "time": datetime.now().timestamp()},
        12: {"type": "comment"},
        13: {"type": "story", "title": "Show HN: Low Score", "url": "https://lowscore.dev", "text": "", "score": 1, "time": datetime.now().timestamp()},
        14: {"type": "story", "title": "Show HN: Not Dev", "url": "https://nodev.dev", "text": "", "score": 70, "time": datetime.now().timestamp()},
    }

    def fake_get(url, timeout):
        if url.endswith("showstories.json"):
            return FakeJSONResponse(show_ids)
        story_id = int(url.split("/")[-1].split(".")[0])
        return FakeJSONResponse(show_story[story_id])

    monkeypatch.setattr("scrape_hackernews.requests.get", fake_get)

    def fake_classify(candidates):
        mapping = {}
        for item in candidates:
            mapping[item["id"]] = item["name"] in {"Show HN: Dev CLI", "Show HN: Dev Notes"}
        return mapping

    monkeypatch.setattr("scrape_hackernews.classify_candidates", fake_classify)

    def categorizer(text, name):
        return "CLI Tool" if name == "Show HN: Dev CLI" else None

    monkeypatch.setattr("scrape_hackernews.get_devtools_category", categorizer)

    saved = []
    monkeypatch.setattr("scrape_hackernews.save_startup", lambda record: saved.append(record))

    scrape_hackernews.scrape_hackernews_show()
    assert len(saved) == 2
    assert saved[0]["description"].startswith("[CLI Tool]")
    assert "Extras" in saved[1]["description"]


def test_scrape_hackernews_handles_errors(monkeypatch):
    import scrape_hackernews
    from requests import RequestException

    # Network level failure
    monkeypatch.setattr(
        "scrape_hackernews.requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(RequestException("network down"))
    )
    scrape_hackernews.scrape_hackernews()

    # Request exception for Show HN
    monkeypatch.setattr(
        "scrape_hackernews.requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(RequestException("show down")),
    )
    scrape_hackernews.scrape_hackernews_show()

    # General parsing failure for top stories
    class BadTopStoriesResponse:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("invalid json")

    monkeypatch.setattr("scrape_hackernews.requests.get", lambda *args, **kwargs: BadTopStoriesResponse())
    scrape_hackernews.scrape_hackernews()

    # General parsing failure for Show HN
    class BadResponse:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("invalid json")

    monkeypatch.setattr("scrape_hackernews.requests.get", lambda *args, **kwargs: BadResponse())
    scrape_hackernews.scrape_hackernews_show()


def test_scrape_hackernews_handles_classifier_exceptions_per_feed(monkeypatch):
    """Bug #35: classifier failures should be isolated to the current feed."""
    import scrape_hackernews

    top_story = {
        "type": "story",
        "title": "Top Story",
        "url": "https://top.dev",
        "text": "Developer tool",
        "score": 50,
        "time": datetime.now().timestamp(),
    }
    show_story = {
        "type": "story",
        "title": "Show Story",
        "url": "https://show.dev",
        "text": "Developer tool",
        "score": 50,
        "time": datetime.now().timestamp(),
    }

    def fake_get(url, timeout):
        if url.endswith("topstories.json"):
            return FakeJSONResponse([1])
        if url.endswith("showstories.json"):
            return FakeJSONResponse([2])
        if "/item/1.json" in url:
            return FakeJSONResponse(top_story)
        if "/item/2.json" in url:
            return FakeJSONResponse(show_story)
        return FakeJSONResponse({})

    monkeypatch.setattr("scrape_hackernews.requests.get", fake_get)

    classify_calls = {"count": 0}

    def flaky_classify(candidates):
        classify_calls["count"] += 1
        if classify_calls["count"] == 1:
            raise RuntimeError("OpenAI rate limited")
        return {item["id"]: True for item in candidates}

    saved = []
    monkeypatch.setattr("scrape_hackernews.classify_candidates", flaky_classify)
    monkeypatch.setattr("scrape_hackernews.get_devtools_category", lambda *_: None)
    monkeypatch.setattr("scrape_hackernews.save_startup", lambda record: saved.append(record))

    # First feed should log-and-continue despite classifier exception.
    scrape_hackernews.scrape_hackernews()
    # Second feed should still run.
    scrape_hackernews.scrape_hackernews_show()

    assert classify_calls["count"] == 2
    assert len(saved) == 1
    assert saved[0]["name"] == "Show Story"


def test_scrape_hackernews_main_guard(monkeypatch):
    import runpy

    def fake_get(url, timeout):
        if url.endswith("topstories.json") or url.endswith("showstories.json"):
            return FakeJSONResponse([])
        return FakeJSONResponse({})

    # Exercise both branches explicitly for coverage
    assert isinstance(fake_get("https://hacker-news.firebaseio.com/v0/topstories.json", 5), FakeJSONResponse)
    assert isinstance(fake_get("https://example.com/other", 5), FakeJSONResponse)

    monkeypatch.setattr("database.init_db", lambda: None)
    monkeypatch.setattr("database.save_startup", lambda record: None)
    monkeypatch.setattr("ai_classifier.is_devtools_related_ai", lambda *args, **kwargs: False)
    monkeypatch.setattr("ai_classifier.get_devtools_category", lambda *args, **kwargs: None)
    monkeypatch.setattr("requests.get", fake_get)

    runpy.run_module("scrape_hackernews", run_name="__main__")


class TestRequestTimeouts:
    """Test suite for verifying correct timeout configuration on HTTP requests.

    The scraper should use a tuple timeout (connection_timeout, read_timeout)
    to prevent SSL handshake hangs while allowing adequate time for data transfer.
    """

    @pytest.fixture
    def mock_requests_get(self, monkeypatch) -> MagicMock:
        """Create a mock for requests.get that tracks all calls."""
        mock = MagicMock()
        mock.return_value = FakeJSONResponse([])
        mock.return_value.raise_for_status = MagicMock()
        monkeypatch.setattr("scrape_hackernews.requests.get", mock)
        return mock

    @pytest.fixture
    def stub_dependencies(self, monkeypatch) -> None:
        """Stub out classifier and database dependencies."""
        monkeypatch.setattr("scrape_hackernews.classify_candidates", lambda c: {})
        monkeypatch.setattr("scrape_hackernews.get_devtools_category", lambda t, n: None)
        monkeypatch.setattr("scrape_hackernews.save_startup", lambda r: None)

    def test_topstories_request_uses_tuple_timeout(
        self, monkeypatch, mock_requests_get, stub_dependencies
    ) -> None:
        """Verify topstories endpoint uses (connection, read) timeout tuple."""
        import scrape_hackernews

        scrape_hackernews.scrape_hackernews()

        # Find the call to topstories.json
        topstories_calls = [
            c for c in mock_requests_get.call_args_list
            if "topstories.json" in str(c)
        ]
        assert len(topstories_calls) == 1, "Expected exactly one topstories request"

        _, kwargs = topstories_calls[0]
        assert kwargs.get("timeout") == EXPECTED_TIMEOUT, (
            f"topstories request should use timeout={EXPECTED_TIMEOUT}, "
            f"got timeout={kwargs.get('timeout')}"
        )

    def test_story_requests_use_tuple_timeout(
        self, monkeypatch, mock_requests_get, stub_dependencies
    ) -> None:
        """Verify individual story requests use (connection, read) timeout tuple."""
        import scrape_hackernews

        # Return some story IDs so we make story requests
        story_ids = [101, 102, 103]
        story_data = {
            "type": "story",
            "title": "Test Story",
            "url": "https://example.com",
            "text": "",
            "score": 100,
            "time": datetime.now().timestamp(),
        }

        def mock_get(url: str, timeout: Tuple[int, int]) -> FakeJSONResponse:
            if "topstories.json" in url:
                return FakeJSONResponse(story_ids)
            return FakeJSONResponse(story_data)

        mock_requests_get.side_effect = mock_get

        scrape_hackernews.scrape_hackernews()

        # Check all story item requests
        story_calls = [
            c for c in mock_requests_get.call_args_list
            if "/item/" in str(c)
        ]
        assert len(story_calls) == len(story_ids), (
            f"Expected {len(story_ids)} story requests, got {len(story_calls)}"
        )

        for story_call in story_calls:
            _, kwargs = story_call
            assert kwargs.get("timeout") == EXPECTED_TIMEOUT, (
                f"Story request should use timeout={EXPECTED_TIMEOUT}, "
                f"got timeout={kwargs.get('timeout')}"
            )

    def test_showstories_request_uses_tuple_timeout(
        self, monkeypatch, mock_requests_get, stub_dependencies
    ) -> None:
        """Verify showstories endpoint uses (connection, read) timeout tuple."""
        import scrape_hackernews

        scrape_hackernews.scrape_hackernews_show()

        # Find the call to showstories.json
        showstories_calls = [
            c for c in mock_requests_get.call_args_list
            if "showstories.json" in str(c)
        ]
        assert len(showstories_calls) == 1, "Expected exactly one showstories request"

        _, kwargs = showstories_calls[0]
        assert kwargs.get("timeout") == EXPECTED_TIMEOUT, (
            f"showstories request should use timeout={EXPECTED_TIMEOUT}, "
            f"got timeout={kwargs.get('timeout')}"
        )

    def test_show_hn_story_requests_use_tuple_timeout(
        self, monkeypatch, mock_requests_get, stub_dependencies
    ) -> None:
        """Verify Show HN story requests use (connection, read) timeout tuple."""
        import scrape_hackernews

        # Return some story IDs so we make story requests
        story_ids = [201, 202]
        story_data = {
            "type": "story",
            "title": "Show HN: Test",
            "url": "https://show.example.com",
            "text": "Description",
            "score": 50,
            "time": datetime.now().timestamp(),
        }

        def mock_get(url: str, timeout: Tuple[int, int]) -> FakeJSONResponse:
            if "showstories.json" in url:
                return FakeJSONResponse(story_ids)
            return FakeJSONResponse(story_data)

        mock_requests_get.side_effect = mock_get

        scrape_hackernews.scrape_hackernews_show()

        # Check all story item requests
        story_calls = [
            c for c in mock_requests_get.call_args_list
            if "/item/" in str(c)
        ]
        assert len(story_calls) == len(story_ids), (
            f"Expected {len(story_ids)} story requests, got {len(story_calls)}"
        )

        for story_call in story_calls:
            _, kwargs = story_call
            assert kwargs.get("timeout") == EXPECTED_TIMEOUT, (
                f"Show HN story request should use timeout={EXPECTED_TIMEOUT}, "
                f"got timeout={kwargs.get('timeout')}"
            )

    def test_timeout_is_tuple_not_single_value(
        self, monkeypatch, mock_requests_get, stub_dependencies
    ) -> None:
        """Verify timeout is a tuple, not a single integer value.

        A tuple timeout separates connection timeout from read timeout,
        which helps prevent SSL handshake hangs while allowing adequate
        time for data transfer.
        """
        import scrape_hackernews

        scrape_hackernews.scrape_hackernews()

        for call_args in mock_requests_get.call_args_list:
            _, kwargs = call_args
            timeout = kwargs.get("timeout")
            assert isinstance(timeout, tuple), (
                f"Timeout should be a tuple (connection, read), got {type(timeout).__name__}: {timeout}"
            )
            assert len(timeout) == 2, (
                f"Timeout tuple should have 2 elements (connection, read), got {len(timeout)}"
            )

    def test_connection_timeout_less_than_read_timeout(
        self, monkeypatch, mock_requests_get, stub_dependencies
    ) -> None:
        """Verify connection timeout is shorter than read timeout.

        Connection timeout should be shorter since establishing a connection
        should be fast. Read timeout can be longer to allow for slow responses.
        """
        import scrape_hackernews

        scrape_hackernews.scrape_hackernews()

        for call_args in mock_requests_get.call_args_list:
            _, kwargs = call_args
            timeout = kwargs.get("timeout")
            connection_timeout, read_timeout = timeout
            assert connection_timeout < read_timeout, (
                f"Connection timeout ({connection_timeout}s) should be less than "
                f"read timeout ({read_timeout}s)"
            )


class TestRetryLogic:
    """Test suite for HTTP request retry logic on transient failures.

    The scraper should automatically retry requests that fail due to transient
    network errors (SSL errors, connection resets, timeouts, 5xx errors) while
    NOT retrying on permanent failures (4xx errors, invalid JSON).
    """

    @pytest.fixture
    def stub_dependencies(self, monkeypatch) -> None:
        """Stub out classifier and database dependencies."""
        monkeypatch.setattr("scrape_hackernews.classify_candidates", lambda c: {})
        monkeypatch.setattr("scrape_hackernews.get_devtools_category", lambda t, n: None)
        monkeypatch.setattr("scrape_hackernews.save_startup", lambda r: None)

    @pytest.fixture
    def mock_sleep(self, monkeypatch) -> MagicMock:
        """Mock time.sleep to avoid actual delays in tests."""
        mock = MagicMock()
        monkeypatch.setattr("scrape_hackernews.time.sleep", mock)
        return mock

    # --- SSL Error Retry Tests ---

    def test_retries_on_ssl_eof_error(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper retries when SSL EOF error occurs (like the 4pm failure)."""
        import ssl
        import scrape_hackernews
        from requests.exceptions import SSLError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Simulate SSL EOF error like the production failure
                ssl_error = ssl.SSLEOFError(
                    8, "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"
                )
                raise SSLError(ssl_error)
            # Succeed on third attempt
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        assert call_count == 3, f"Expected 3 attempts (2 retries), got {call_count}"

    def test_retries_on_ssl_error(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper retries on general SSL errors."""
        import scrape_hackernews
        from requests.exceptions import SSLError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise SSLError("SSL handshake failed")
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        assert call_count == 2, f"Expected 2 attempts (1 retry), got {call_count}"

    # --- Connection Error Retry Tests ---

    def test_retries_on_connection_error(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper retries on connection errors."""
        import scrape_hackernews
        from requests.exceptions import ConnectionError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection refused")
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        assert call_count == 2, f"Expected 2 attempts (1 retry), got {call_count}"

    def test_retries_on_connection_reset(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper retries when connection is reset by peer."""
        import scrape_hackernews
        from requests.exceptions import ConnectionError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection reset by peer")
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        assert call_count == 2

    # --- Timeout Retry Tests ---

    def test_retries_on_read_timeout(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper retries on read timeout."""
        import scrape_hackernews
        from requests.exceptions import ReadTimeout

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ReadTimeout("Read timed out")
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        assert call_count == 2

    def test_retries_on_connect_timeout(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper retries on connection timeout."""
        import scrape_hackernews
        from requests.exceptions import ConnectTimeout

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectTimeout("Connection timed out")
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        assert call_count == 2

    # --- Server Error Retry Tests ---

    def test_retries_on_502_bad_gateway(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper retries on 502 Bad Gateway."""
        import scrape_hackernews
        from requests.exceptions import HTTPError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                response = FakeJSONResponse([], status_code=502)
                response.raise_for_status = lambda: (_ for _ in ()).throw(
                    HTTPError("502 Bad Gateway")
                )
                return response
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        assert call_count == 2

    def test_retries_on_503_service_unavailable(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper retries on 503 Service Unavailable."""
        import scrape_hackernews
        from requests.exceptions import HTTPError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                response = FakeJSONResponse([], status_code=503)
                response.raise_for_status = lambda: (_ for _ in ()).throw(
                    HTTPError("503 Service Unavailable")
                )
                return response
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        assert call_count == 2

    def test_retries_on_504_gateway_timeout(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper retries on 504 Gateway Timeout."""
        import scrape_hackernews
        from requests.exceptions import HTTPError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                response = FakeJSONResponse([], status_code=504)
                response.raise_for_status = lambda: (_ for _ in ()).throw(
                    HTTPError("504 Gateway Timeout")
                )
                return response
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        assert call_count == 2

    # --- Max Retry Limit Tests ---

    def test_stops_after_max_retries(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper stops retrying after max attempts (default 3)."""
        import scrape_hackernews
        from requests.exceptions import SSLError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            # Always fail with SSL error
            raise SSLError("Persistent SSL failure")

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)

        # Should not raise - the scraper catches and logs the error
        scrape_hackernews.scrape_hackernews()

        # Default max_retries=3 means 4 total attempts (1 initial + 3 retries)
        assert call_count == 4, f"Expected 4 attempts (1 + 3 retries), got {call_count}"

    def test_max_retries_per_request_not_cumulative(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify each URL gets its own retry budget, not shared."""
        import scrape_hackernews
        from requests.exceptions import ConnectionError

        urls_called = []

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            urls_called.append(url)
            if "topstories.json" in url:
                # First request succeeds after 1 retry
                if urls_called.count(url) < 2:
                    raise ConnectionError("Temp failure")
                return FakeJSONResponse([101])
            elif "/item/101.json" in url:
                # Second request also gets its own retry budget
                if urls_called.count(url) < 2:
                    raise ConnectionError("Temp failure")
                return FakeJSONResponse({
                    "type": "story",
                    "title": "Test",
                    "url": "https://test.com",
                    "score": 100,
                    "time": datetime.now().timestamp()
                })
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        # Each URL should have been called twice (1 fail + 1 success)
        topstories_calls = [u for u in urls_called if "topstories.json" in u]
        item_calls = [u for u in urls_called if "/item/101.json" in u]

        assert len(topstories_calls) == 2, "topstories should retry independently"
        assert len(item_calls) == 2, "item request should retry independently"

    # --- Exponential Backoff Tests ---

    def test_uses_exponential_backoff(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify retry delays increase exponentially."""
        import scrape_hackernews
        from requests.exceptions import SSLError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise SSLError("SSL error")
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        # Check sleep was called with increasing delays
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert len(sleep_calls) == 3, f"Expected 3 sleep calls, got {len(sleep_calls)}"

        # Verify exponential backoff pattern (e.g., 1, 2, 4 or similar)
        for i in range(1, len(sleep_calls)):
            assert sleep_calls[i] > sleep_calls[i-1], (
                f"Backoff should increase: {sleep_calls[i-1]} -> {sleep_calls[i]}"
            )

    def test_backoff_has_reasonable_max_delay(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify backoff delay is capped at a reasonable maximum."""
        import scrape_hackernews
        from requests.exceptions import SSLError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise SSLError("SSL error")
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        # No sleep should exceed 30 seconds
        for call in mock_sleep.call_args_list:
            delay = call[0][0]
            assert delay <= 30, f"Backoff delay {delay}s exceeds max 30s"

    # --- Non-Retryable Error Tests ---

    def test_does_not_retry_on_404(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper does NOT retry on 404 Not Found."""
        import scrape_hackernews
        from requests.exceptions import HTTPError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            response = FakeJSONResponse([], status_code=404)
            response.raise_for_status = lambda: (_ for _ in ()).throw(
                HTTPError("404 Not Found")
            )
            return response

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        # Should only be called once - no retry on 404
        assert call_count == 1, f"Should not retry on 404, but got {call_count} calls"

    def test_does_not_retry_on_400(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper does NOT retry on 400 Bad Request."""
        import scrape_hackernews
        from requests.exceptions import HTTPError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            response = FakeJSONResponse([], status_code=400)
            response.raise_for_status = lambda: (_ for _ in ()).throw(
                HTTPError("400 Bad Request")
            )
            return response

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        assert call_count == 1, f"Should not retry on 400, but got {call_count} calls"

    def test_does_not_retry_on_json_decode_error(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify scraper does NOT retry on JSON decode errors."""
        import scrape_hackernews

        call_count = 0

        class BadJSONResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                raise ValueError("Invalid JSON")

        def mock_get(url: str, timeout):
            nonlocal call_count
            call_count += 1
            return BadJSONResponse()

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        # JSON errors are not retryable
        assert call_count == 1, f"Should not retry on JSON error, but got {call_count} calls"

    # --- Show HN Retry Tests ---

    def test_show_hn_retries_on_ssl_error(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify Show HN scraper also retries on SSL errors."""
        import scrape_hackernews
        from requests.exceptions import SSLError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise SSLError("SSL error")
            if "showstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews_show()

        assert call_count == 2

    def test_show_hn_retries_on_connection_error(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify Show HN scraper retries on connection errors."""
        import scrape_hackernews
        from requests.exceptions import ConnectionError

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection failed")
            if "showstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews_show()

        assert call_count == 2

    # --- Logging Tests ---

    def test_logs_retry_attempts(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify retry attempts are logged for observability."""
        import scrape_hackernews
        from requests.exceptions import SSLError

        log_messages = []

        class MockLogger:
            def info(self, msg, **kwargs):
                log_messages.append(("info", msg, kwargs))
            def warning(self, msg, **kwargs):
                log_messages.append(("warning", msg, kwargs))
            def debug(self, msg, **kwargs):
                log_messages.append(("debug", msg, kwargs))
            def exception(self, msg, **kwargs):
                log_messages.append(("exception", msg, kwargs))

        monkeypatch.setattr("scrape_hackernews.logger", MockLogger())

        call_count = 0

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise SSLError("SSL error")
            if "topstories.json" in url:
                return FakeJSONResponse([])
            return FakeJSONResponse({})

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        # Check that retry was logged
        retry_logs = [
            (level, msg) for level, msg, _ in log_messages
            if "retry" in msg.lower() or "retrying" in msg.lower()
        ]
        assert len(retry_logs) > 0, "Expected retry attempt to be logged"

    def test_logs_final_failure_after_retries_exhausted(
        self, monkeypatch, stub_dependencies, mock_sleep
    ) -> None:
        """Verify final failure is logged when all retries are exhausted."""
        import scrape_hackernews
        from requests.exceptions import SSLError

        log_messages = []

        class MockLogger:
            def info(self, msg, **kwargs):
                log_messages.append(("info", msg, kwargs))
            def warning(self, msg, **kwargs):
                log_messages.append(("warning", msg, kwargs))
            def debug(self, msg, **kwargs):
                log_messages.append(("debug", msg, kwargs))
            def exception(self, msg, **kwargs):
                log_messages.append(("exception", msg, kwargs))

        monkeypatch.setattr("scrape_hackernews.logger", MockLogger())

        def mock_get(url: str, timeout) -> FakeJSONResponse:
            raise SSLError("Persistent SSL failure")

        monkeypatch.setattr("scrape_hackernews.requests.get", mock_get)
        scrape_hackernews.scrape_hackernews()

        # Check that final failure was logged
        failure_logs = [
            (level, msg) for level, msg, _ in log_messages
            if level == "exception" or "failed" in msg.lower()
        ]
        assert len(failure_logs) > 0, "Expected final failure to be logged"


class TestSingleStoryFailureDoesNotAbortLoop:
    """Bug #15: A single story returning 404 (or any HTTP error) should not
    kill the entire scrape loop. Valid stories fetched before and after the
    bad one must still be processed."""

    @pytest.fixture
    def stub_dependencies(self, monkeypatch) -> None:
        """Stub out classifier and database dependencies."""
        monkeypatch.setattr("scrape_hackernews.classify_candidates", lambda c: {
            item["id"]: True for item in c
        })
        monkeypatch.setattr("scrape_hackernews.get_devtools_category", lambda t, n: None)

    def test_scrape_hackernews_continues_after_story_404(
        self, monkeypatch, stub_dependencies
    ) -> None:
        """scrape_hackernews should skip a story that returns 404 and keep
        processing the remaining stories."""
        import scrape_hackernews

        top_story_ids = [1, 2, 3]
        stories = {
            1: {"type": "story", "title": "Good Story 1", "url": "https://good1.dev", "text": "", "score": 50, "time": datetime.now().timestamp()},
            # story 2 will return 404
            3: {"type": "story", "title": "Good Story 3", "url": "https://good3.dev", "text": "", "score": 60, "time": datetime.now().timestamp()},
        }

        def fake_get(url, timeout):
            if url.endswith("topstories.json"):
                return FakeJSONResponse(top_story_ids)
            story_id = int(url.split("/")[-1].split(".")[0])
            if story_id == 2:
                return FakeJSONResponse(None, status_code=404)
            return FakeJSONResponse(stories[story_id])

        monkeypatch.setattr("scrape_hackernews.requests.get", fake_get)

        saved = []
        monkeypatch.setattr("scrape_hackernews.save_startup", lambda record: saved.append(record))

        scrape_hackernews.scrape_hackernews()

        assert len(saved) == 2, (
            f"Expected 2 saved stories (skipping the 404), got {len(saved)}"
        )
        saved_names = {s["name"] for s in saved}
        assert "Good Story 1" in saved_names
        assert "Good Story 3" in saved_names

    def test_scrape_hackernews_show_continues_after_story_404(
        self, monkeypatch, stub_dependencies
    ) -> None:
        """scrape_hackernews_show should skip a story that returns 404 and keep
        processing the remaining stories."""
        import scrape_hackernews

        show_story_ids = [10, 11, 12]
        stories = {
            10: {"type": "story", "title": "Show HN: Good 1", "url": "https://good1.dev", "text": "", "score": 50, "time": datetime.now().timestamp()},
            # story 11 will return 404
            12: {"type": "story", "title": "Show HN: Good 3", "url": "https://good3.dev", "text": "", "score": 60, "time": datetime.now().timestamp()},
        }

        def fake_get(url, timeout):
            if url.endswith("showstories.json"):
                return FakeJSONResponse(show_story_ids)
            story_id = int(url.split("/")[-1].split(".")[0])
            if story_id == 11:
                return FakeJSONResponse(None, status_code=404)
            return FakeJSONResponse(stories[story_id])

        monkeypatch.setattr("scrape_hackernews.requests.get", fake_get)

        saved = []
        monkeypatch.setattr("scrape_hackernews.save_startup", lambda record: saved.append(record))

        scrape_hackernews.scrape_hackernews_show()

        assert len(saved) == 2, (
            f"Expected 2 saved stories (skipping the 404), got {len(saved)}"
        )
        saved_names = {s["name"] for s in saved}
        assert "Show HN: Good 1" in saved_names
        assert "Show HN: Good 3" in saved_names

    def test_scrape_hackernews_continues_after_json_decode_error_on_story(
        self, monkeypatch, stub_dependencies
    ) -> None:
        """If a single story response has invalid JSON, the loop should
        continue processing the remaining stories."""
        import scrape_hackernews

        top_story_ids = [1, 2, 3]
        stories = {
            1: {"type": "story", "title": "Good Story A", "url": "https://gooda.dev", "text": "", "score": 50, "time": datetime.now().timestamp()},
            3: {"type": "story", "title": "Good Story B", "url": "https://goodb.dev", "text": "", "score": 60, "time": datetime.now().timestamp()},
        }

        class BadJSONStoryResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                raise ValueError("Invalid JSON")

        def fake_get(url, timeout):
            if url.endswith("topstories.json"):
                return FakeJSONResponse(top_story_ids)
            story_id = int(url.split("/")[-1].split(".")[0])
            if story_id == 2:
                return BadJSONStoryResponse()
            return FakeJSONResponse(stories[story_id])

        monkeypatch.setattr("scrape_hackernews.requests.get", fake_get)

        saved = []
        monkeypatch.setattr("scrape_hackernews.save_startup", lambda record: saved.append(record))

        scrape_hackernews.scrape_hackernews()

        assert len(saved) == 2, (
            f"Expected 2 saved stories (skipping bad JSON), got {len(saved)}"
        )


class TestNoneTimestampDoesNotCrash:
    """Bug #10: When a story's 'time' key exists but its value is None,
    dict.get('time', default) returns None (not the default). Passing
    None to datetime.fromtimestamp() raises TypeError, crashing the
    save path for that story."""

    @pytest.fixture
    def stub_dependencies(self, monkeypatch) -> None:
        """Stub out classifier and database dependencies."""
        monkeypatch.setattr("scrape_hackernews.classify_candidates", lambda c: {
            item["id"]: True for item in c
        })
        monkeypatch.setattr("scrape_hackernews.get_devtools_category", lambda t, n: None)

    def test_scrape_hackernews_handles_none_timestamp(
        self, monkeypatch, stub_dependencies
    ) -> None:
        """scrape_hackernews should not crash when story['time'] is None."""
        import scrape_hackernews

        top_story_ids = [1]
        stories = {
            1: {
                "type": "story",
                "title": "None Time Story",
                "url": "https://nonetime.dev",
                "text": "",
                "score": 50,
                "time": None,  # key exists but value is None
            },
        }

        def fake_get(url, timeout):
            if url.endswith("topstories.json"):
                return FakeJSONResponse(top_story_ids)
            story_id = int(url.split("/")[-1].split(".")[0])
            return FakeJSONResponse(stories[story_id])

        monkeypatch.setattr("scrape_hackernews.requests.get", fake_get)

        saved = []
        monkeypatch.setattr("scrape_hackernews.save_startup", lambda record: saved.append(record))

        scrape_hackernews.scrape_hackernews()

        assert len(saved) == 1, (
            f"Expected 1 saved story even with None timestamp, got {len(saved)}"
        )
        assert isinstance(saved[0]["date_found"], datetime)

    def test_scrape_hackernews_handles_missing_timestamp(
        self, monkeypatch, stub_dependencies
    ) -> None:
        """scrape_hackernews should not crash when story has no 'time' key."""
        import scrape_hackernews

        top_story_ids = [1]
        stories = {
            1: {
                "type": "story",
                "title": "No Time Story",
                "url": "https://notime.dev",
                "text": "",
                "score": 50,
                # 'time' key is missing entirely
            },
        }

        def fake_get(url, timeout):
            if url.endswith("topstories.json"):
                return FakeJSONResponse(top_story_ids)
            story_id = int(url.split("/")[-1].split(".")[0])
            return FakeJSONResponse(stories[story_id])

        monkeypatch.setattr("scrape_hackernews.requests.get", fake_get)

        saved = []
        monkeypatch.setattr("scrape_hackernews.save_startup", lambda record: saved.append(record))

        scrape_hackernews.scrape_hackernews()

        assert len(saved) == 1, (
            f"Expected 1 saved story even with missing timestamp, got {len(saved)}"
        )
        assert isinstance(saved[0]["date_found"], datetime)

    def test_scrape_hackernews_show_handles_none_timestamp(
        self, monkeypatch, stub_dependencies
    ) -> None:
        """scrape_hackernews_show should not crash when story['time'] is None."""
        import scrape_hackernews

        show_story_ids = [10]
        stories = {
            10: {
                "type": "story",
                "title": "Show HN: None Time",
                "url": "https://nonetime.dev",
                "text": "",
                "score": 50,
                "time": None,
            },
        }

        def fake_get(url, timeout):
            if url.endswith("showstories.json"):
                return FakeJSONResponse(show_story_ids)
            story_id = int(url.split("/")[-1].split(".")[0])
            return FakeJSONResponse(stories[story_id])

        monkeypatch.setattr("scrape_hackernews.requests.get", fake_get)

        saved = []
        monkeypatch.setattr("scrape_hackernews.save_startup", lambda record: saved.append(record))

        scrape_hackernews.scrape_hackernews_show()

        assert len(saved) == 1, (
            f"Expected 1 saved story even with None timestamp, got {len(saved)}"
        )
        assert isinstance(saved[0]["date_found"], datetime)
