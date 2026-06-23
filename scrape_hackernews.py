"""Hacker News scraper for top stories and Show HN posts."""

import time
import uuid
from datetime import datetime

import requests
from requests.exceptions import (
    ConnectionError,
    ConnectTimeout,
    ReadTimeout,
    SSLError,
    Timeout,
)

from ai_classifier import classify_candidates, get_devtools_category
from database import init_db, save_startup
from logging_config import get_logger, logging_context
from observability import trace_http_call

logger = get_logger("devtools.scraper.hackernews")


def _build_description(title: str, text: str, category: str | None) -> str:
    """Build a description string from title, text, and optional category prefix."""
    if category:
        description = f"[{category}] {title}"
    else:
        description = title
    if text:
        description += f"\n\n{text}"
    return description


# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 30  # seconds
BACKOFF_MULTIPLIER = 2

# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    SSLError,
    ConnectionError,
    ConnectTimeout,
    ReadTimeout,
    Timeout,
)


def _is_retryable_status_code(status_code: int) -> bool:
    """Check if HTTP status code is retryable (5xx server errors)."""
    return status_code in (502, 503, 504)


def _backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff delay for a given attempt number."""
    return min(INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** attempt), MAX_BACKOFF)


def _request_with_retry(url: str, timeout: tuple, max_retries: int = MAX_RETRIES) -> requests.Response:
    """Make HTTP GET request with retry logic for transient failures.

    Retries on:
    - SSL errors (including SSL EOF)
    - Connection errors
    - Timeouts (connect and read)
    - 5xx server errors (502, 503, 504)

    Does NOT retry on:
    - 4xx client errors (400, 404, etc.)
    - JSON decode errors
    - Other non-transient failures

    Uses exponential backoff between retries.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)

            if _is_retryable_status_code(response.status_code):
                if attempt < max_retries:
                    backoff = _backoff_delay(attempt)
                    logger.warning(
                        "scraper.retrying",
                        extra={
                            "event": "scraper.retrying",
                            "url": url,
                            "status_code": response.status_code,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "backoff_seconds": backoff,
                        },
                    )
                    time.sleep(backoff)
                    continue
                else:
                    # Exhausted retries, raise the error
                    response.raise_for_status()

            return response

        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < max_retries:
                backoff = _backoff_delay(attempt)
                logger.warning(
                    "scraper.retrying",
                    extra={
                        "event": "scraper.retrying",
                        "url": url,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "backoff_seconds": backoff,
                    },
                )
                time.sleep(backoff)
            else:
                # Exhausted retries
                raise

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry logic")

def _scrape_hn_feed(
    list_url: str,
    trace_name: str,
    scraper_name: str,
    source_label: str,
    key_prefix: str = "",
    max_stories: int = 50,
    min_score: int = 10,
) -> None:
    """Shared scrape loop for Hacker News feeds (top stories and Show HN)."""
    run_id = str(uuid.uuid4())
    with logging_context(scraper=scraper_name, scrape_run_id=run_id):
        try:
            with trace_http_call(trace_name, "GET", list_url) as span:
                list_resp = _request_with_retry(list_url, timeout=(5, 10))
                if span:
                    span.set_tag("http.status_code", list_resp.status_code)
            list_resp.raise_for_status()
            story_ids = list_resp.json()[:max_stories]

            logger.info(
                "scraper.stories_fetched",
                extra={"event": "scraper.stories_fetched", "count": len(story_ids)},
            )

            story_cache = {}
            candidates = []
            for story_id in story_ids:
                try:
                    story_url = f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json'
                    with trace_http_call("hackernews.story", "GET", story_url) as span:
                        story_resp = _request_with_retry(story_url, timeout=(5, 10))
                        if span:
                            span.set_tag("http.status_code", story_resp.status_code)
                            span.set_tag("hackernews.story_id", story_id)
                    story_resp.raise_for_status()
                    story = story_resp.json()

                    if not story or story.get('type') != 'story':
                        continue

                    title = story.get('title', '')
                    url = story.get('url', '')
                    text = story.get('text', '')
                    score = story.get('score', 0)

                    if not url or score < min_score:
                        continue

                    key = f"{key_prefix}{story_id}"
                    full_text = f"{title} {text}"
                    story_cache[key] = (story, title, url, text, score, full_text)
                    candidates.append({"id": key, "name": title, "text": full_text})
                except Exception:
                    # Intentionally broad: one bad story must not kill the scrape loop
                    logger.warning(
                        "scraper.story_fetch_failed",
                        extra={"event": "scraper.story_fetch_failed", "story_id": story_id},
                    )
                    continue

            results = classify_candidates(candidates)

            devtools_count = 0
            for key, (story, title, url, text, score, full_text) in story_cache.items():
                if not results.get(key):
                    logger.debug(
                        "scraper.skip_non_devtool",
                        extra={"event": "scraper.skip_non_devtool", "story_id": key},
                    )
                    continue

                devtools_count += 1

                category = get_devtools_category(full_text, title)
                description = _build_description(title, text, category)

                story_time = story.get('time')
                timestamp = story_time if story_time is not None else datetime.now().timestamp()
                startup = {
                    "name": title,
                    "url": url,
                    "description": description,
                    "date_found": datetime.fromtimestamp(timestamp),
                    "source": f"{source_label} (score: {score})"
                }

                save_startup(startup)

            logger.info(
                "scraper.complete",
                extra={"event": "scraper.complete", "devtools_count": devtools_count, "candidates": len(candidates)},
            )

        except requests.RequestException:
            logger.exception("scraper.request_failed", extra={"event": "scraper.request_failed"})
        except Exception:
            # Intentionally broad: isolate this feed from classifier/runtime failures
            logger.exception("scraper.error", extra={"event": "scraper.error"})


def scrape_hackernews():
    """Scrape Hacker News for devtools using their API"""
    _scrape_hn_feed(
        list_url="https://hacker-news.firebaseio.com/v0/topstories.json",
        trace_name="hackernews.topstories",
        scraper_name="hackernews",
        source_label="Hacker News",
        key_prefix="",
        max_stories=50,
        min_score=10,
    )


def scrape_hackernews_show():
    """Scrape Hacker News Show HN posts (often devtools)"""
    _scrape_hn_feed(
        list_url="https://hacker-news.firebaseio.com/v0/showstories.json",
        trace_name="hackernews.showstories",
        scraper_name="hackernews_show",
        source_label="Show HN",
        key_prefix="show-",
        max_stories=30,
        min_score=5,
    )

if __name__ == "__main__":
    init_db()
    scrape_hackernews()
    scrape_hackernews_show()
    logger.info("scraper.script_complete", extra={"event": "scraper.script_complete"})
