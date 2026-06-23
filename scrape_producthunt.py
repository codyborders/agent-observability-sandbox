"""Product Hunt RSS feed scraper for developer tool discovery."""

import requests
import uuid
from datetime import datetime

from bs4 import BeautifulSoup

from database import init_db, save_startup
from ai_classifier import has_devtools_keywords as is_devtools_related
from logging_config import get_logger, logging_context
from observability import trace_http_call

logger = get_logger("devtools.scraper.producthunt_rss")

def scrape_producthunt_rss() -> None:
    """Fetch the Product Hunt RSS feed and persist devtools-related entries."""
    url = "https://www.producthunt.com/feed"
    run_id = str(uuid.uuid4())
    with logging_context(scraper="producthunt_rss", scrape_run_id=run_id):
        try:
            with trace_http_call("producthunt.rss", "GET", url) as span:
                resp = requests.get(url, timeout=10)
                if span:
                    span.set_tag("http.status_code", resp.status_code)
            resp.raise_for_status()
            logger.info(
                "scraper.response",
                extra={
                    "event": "scraper.response",
                    "status_code": resp.status_code,
                    "content_length": len(resp.content),
                },
            )
        except requests.RequestException:
            logger.exception("scraper.request_failed", extra={"event": "scraper.request_failed"})
            return
        
        soup = BeautifulSoup(resp.content, features="xml")
        items = soup.find_all("item")
        logger.info(
            "scraper.items_found",
            extra={"event": "scraper.items_found", "count": len(items)},
        )

        devtools_count = 0
        for item in items:
            try:
                title = item.title.text
                description = item.description.text
            except AttributeError:
                logger.warning(
                    "scraper.skip_malformed_item",
                    extra={
                        "event": "scraper.skip_malformed_item",
                        "has_title": item.title is not None,
                        "has_description": item.description is not None,
                    },
                )
                continue

            if not is_devtools_related(title + " " + description):
                logger.debug(
                    "scraper.skip_non_devtool",
                    extra={"event": "scraper.skip_non_devtool", "title": title},
                )
                continue

            devtools_count += 1

            startup = {
                "name": title,
                "url": item.link.text,
                "description": description,
                "date_found": datetime.strptime(item.pubDate.text, "%a, %d %b %Y %H:%M:%S %z"),
                "source": "Product Hunt"
            }

            save_startup(startup)
        
        logger.info(
            "scraper.complete",
            extra={"event": "scraper.complete", "devtools_count": devtools_count, "total_items": len(items)},
        )

if __name__ == "__main__":
    init_db()
    scrape_producthunt_rss()
    logger.info("scraper.script_complete", extra={"event": "scraper.script_complete"})
