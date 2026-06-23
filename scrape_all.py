#!/usr/bin/env python3
"""Master scraper that runs all data sources."""

import importlib.util
import uuid
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from database import init_db, record_scrape_completion
from logging_config import get_logger, logging_context

logger = get_logger("devtools.scraper.runner")

SCRAPER_ENTRYPOINTS: dict[str, list[str]] = {
    "scrape_github_trending": ["scrape_github_trending"],
    "scrape_hackernews": ["scrape_hackernews", "scrape_hackernews_show"],
    "scrape_producthunt_api": ["scrape_producthunt_api"],
}


def run_scraper(module_name: str, description: str) -> bool:
    """Run a scraper module and handle any errors."""
    run_id = str(uuid.uuid4())
    with logging_context(scraper_runner_module=module_name, scrape_run_id=run_id):
        logger.info(
            "runner.start_scraper",
            extra={"event": "runner.start_scraper", "description": description},
        )
        try:
            spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            fn_names = SCRAPER_ENTRYPOINTS.get(module_name)
            if fn_names:
                for fn_name in fn_names:
                    getattr(module, fn_name)()
            else:
                logger.warning(
                    "runner.missing_entrypoint",
                    extra={"event": "runner.missing_entrypoint"},
                )
                return False
        except Exception:
            # Intentionally broad: scraper modules may raise arbitrary exceptions
            logger.exception(
                "runner.scraper_failed",
                extra={"event": "runner.scraper_failed"},
            )
            return False
        logger.info(
            "runner.scraper_complete",
            extra={"event": "runner.scraper_complete"},
        )
        return True

def main() -> None:
    """Run all scrapers."""
    logger.info("runner.start", extra={"event": "runner.start"})

    # Initialize database
    init_db()

    # Define scrapers to run
    scrapers = [
        ("scrape_github_trending", "GitHub Trending Repositories"),
        ("scrape_hackernews", "Hacker News & Show HN"),
        ("scrape_producthunt_api", "Product Hunt API"),
    ]

    successful_names = []
    for module_name, description in scrapers:
        if run_scraper(module_name, description):
            successful_names.append(description)

    logger.info(
        "runner.summary",
        extra={
            "event": "runner.summary",
            "successful_scrapers": len(successful_names),
            "total_scrapers": len(scrapers),
        },
    )

    if successful_names:
        record_scrape_completion(', '.join(successful_names))
    else:
        logger.warning(
            "runner.no_successful_scrapers",
            extra={"event": "runner.no_successful_scrapers"},
        )

if __name__ == "__main__":
    main()
