"""GitHub Trending page scraper for discovering new developer tools."""

import sqlite3
import uuid
from datetime import datetime

import requests

from ai_classifier import classify_candidates, get_devtools_category
from database import get_existing_startup_keys, init_db, save_startup
from logging_config import get_logger, logging_context
from observability import trace_http_call

logger = get_logger("devtools.scraper.github_trending")


def scrape_github_trending() -> None:
    """Scrape GitHub Trending and persist new devtools.

    Skips repositories that already exist in the database (by name or URL) and
    logs a debug event "scraper.skip_duplicate" when a duplicate is detected.
    """
    url = "https://github.com/trending"
    run_id = str(uuid.uuid4())
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    with logging_context(scraper="github_trending", scrape_run_id=run_id):
        try:
            with trace_http_call("github.trending", "GET", url) as span:
                resp = requests.get(url, headers=headers, timeout=10)
                if span:
                    span.set_tag("http.status_code", resp.status_code)
            resp.raise_for_status()
            logger.info(
                "scraper.response",
                extra={"event": "scraper.response", "status_code": resp.status_code},
            )
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            repos = soup.find_all('article', class_='Box-row')
            logger.info(
                "scraper.repos_found",
                extra={"event": "scraper.repos_found", "count": len(repos)},
            )
            
            candidates = []
            for repo in repos:
                name_elem = repo.find('h2', class_='h3')
                if not name_elem:
                    continue

                name = name_elem.get_text(strip=True).replace('\n', '').replace(' ', '')
                desc_elem = repo.find('p')
                description = desc_elem.get_text(strip=True) if desc_elem else ""
                link_elem = name_elem.find('a')
                if not link_elem:
                    continue
                repo_url = f"https://github.com{link_elem['href']}"

                candidates.append({
                    "id": repo_url,
                    "name": name,
                    "text": description or name,
                    "description": description,
                    "url": repo_url,
                })

            # Prefetch existing records to avoid per-candidate DB checks and
            # filter duplicates before classification/LLM work.
            existing_names = set()
            existing_urls = set()
            try:
                existing = get_existing_startup_keys()
                for row in existing:
                    nm = row.get("name")
                    u = row.get("url")
                    if nm:
                        existing_names.add(nm)
                    if u:
                        existing_urls.add(u)
            except sqlite3.Error:
                # Log DB issues explicitly; continue without pre-filtering
                logger.exception("scraper.db_error", extra={"event": "scraper.db_error"})

            filtered_candidates = []
            for candidate in candidates:
                if candidate["name"] in existing_names or candidate["url"] in existing_urls:
                    logger.debug(
                        "scraper.skip_duplicate",
                        extra={"event": "scraper.skip_duplicate", "url": candidate["url"]},
                    )
                    continue
                filtered_candidates.append(candidate)

            try:
                results = classify_candidates(
                    {
                        "id": candidate["id"],
                        "name": candidate["name"],
                        "text": candidate["text"],
                    }
                    for candidate in filtered_candidates
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                logger.exception(
                    "scraper.classify_error", extra={"event": "scraper.classify_error"}
                )
                results = {}

            devtools_count = 0
            for candidate in filtered_candidates:
                if not results.get(candidate["id"]):
                    logger.debug(
                        "scraper.skip_non_devtool",
                        extra={
                            "event": "scraper.skip_non_devtool",
                            "repo_url": candidate["id"],
                        },
                    )
                    continue

                devtools_count += 1

                description = candidate["text"]
                # get_devtools_category handles its own exceptions and returns None on failure
                category = get_devtools_category(description, candidate["name"])
                if category:
                    description = f"[{category}] {description}" if description else f"[{category}]"

                startup = {
                    "name": candidate["name"],
                    "url": candidate["url"],
                    "description": description,
                    "date_found": datetime.now(),
                    "source": "GitHub Trending",
                }

                save_startup(startup)
            
            logger.info(
                "scraper.complete",
                extra={
                    "event": "scraper.complete",
                    "devtools_count": devtools_count,
                    "total_repos": len(repos),
                },
            )
            
        except requests.RequestException:
            logger.exception("scraper.request_failed", extra={"event": "scraper.request_failed"})
        except (KeyError, TypeError, ValueError, AttributeError, sqlite3.Error):
            logger.exception("scraper.parse_error", extra={"event": "scraper.parse_error"})

if __name__ == "__main__":
    init_db()
    scrape_github_trending()
    logger.info("scraper.script_complete", extra={"event": "scraper.script_complete"})
