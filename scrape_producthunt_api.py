"""Product Hunt GraphQL API scraper for developer tool discovery."""

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

from ai_classifier import classify_candidates, get_devtools_category
from database import init_db, save_startup
from logging_config import get_logger, logging_context
from observability import trace_http_call

# Load environment variables from .env file
load_dotenv()

logger = get_logger("devtools.scraper.producthunt_api")

def get_producthunt_token():
    """Get Product Hunt access token using API key and secret"""
    client_id = os.getenv('PRODUCTHUNT_CLIENT_ID')
    client_secret = os.getenv('PRODUCTHUNT_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        logger.error(
            "producthunt.credentials_missing",
            extra={"event": "producthunt.credentials_missing"},
        )
        return None
    
    # Get access token
    token_url = "https://api.producthunt.com/v2/oauth/token"
    token_data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'client_credentials'
    }
    
    try:
        with trace_http_call("producthunt.token", "POST", token_url) as span:
            token_resp = requests.post(token_url, data=token_data, timeout=10)
            if span:
                span.set_tag("http.status_code", token_resp.status_code)
        token_resp.raise_for_status()
        token_info = token_resp.json()
        return token_info.get('access_token')
    except (requests.RequestException, KeyError, json.JSONDecodeError):
        logger.exception(
            "producthunt.token_error",
            extra={"event": "producthunt.token_error"},
        )
        return None

def scrape_producthunt_api():
    """Scrape Product Hunt using their official API"""
    run_id = str(uuid.uuid4())
    with logging_context(scraper="producthunt_api", scrape_run_id=run_id):
        logger.info(
            "scraper.start",
            extra={"event": "scraper.start"},
        )
        access_token = get_producthunt_token()
        if not access_token:
            logger.error(
                "scraper.credentials_missing",
                extra={"event": "scraper.credentials_missing"},
            )
            return
        
        url = "https://api.producthunt.com/v2/api/graphql"
        
        posted_after = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).isoformat().replace("+00:00", "Z")

        query = """
    query($postedAfter: DateTime!) {
        posts(first: 20, order: VOTES_COUNT, postedAfter: $postedAfter) {
            edges {
                node {
                    id
                    name
                    tagline
                    description
                    url
                    createdAt
                    topics {
                        edges {
                            node {
                                name
                            }
                        }
                    }
                }
            }
        }
    }
    """
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}',
            'User-Agent': 'DevTools Scraper/1.0'
        }

        payload = {
            "query": query,
            "variables": {"postedAfter": posted_after},
        }
        
        try:
            with trace_http_call("producthunt.graphql", "POST", url) as span:
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if span:
                    span.set_tag("http.status_code", resp.status_code)
            resp.raise_for_status()
            logger.info(
                "scraper.response",
                extra={
                    "event": "scraper.response",
                    "status_code": resp.status_code,
                },
            )
            
            data = resp.json()
            posts_node = (data.get('data') or {}).get('posts') or {}
            posts = posts_node.get('edges', [])
            logger.info(
                "scraper.posts_found",
                extra={"event": "scraper.posts_found", "count": len(posts)},
            )
            
            candidates = []
            post_map = {}
            seen_ids = set()
            for post_edge in posts:
                post = post_edge['node']
                name = post['name']
                tagline = post.get('tagline', '')
                description = post.get('description', '')
                full_text = f"{name} {tagline} {description}"
                # Prefer API id, fall back to URL, then name for deduplication.
                # Use explicit None check so a zero-valued id is still used.
                raw_id = post.get('id')
                post_id = str(raw_id if raw_id is not None else (post.get('url') or name))
                if post_id in seen_ids:
                    logger.warning(
                        "scraper.duplicate_post_id",
                        extra={"event": "scraper.duplicate_post_id", "post_id": post_id},
                    )
                    continue
                seen_ids.add(post_id)
                post_map[post_id] = (post, name, tagline, description, full_text)
                candidates.append({"id": post_id, "name": name, "text": full_text})

            results = classify_candidates(candidates)

            devtools_count = 0
            for post_id, (post, name, tagline, description, full_text) in post_map.items():
                if not results.get(post_id):
                    logger.debug(
                        "scraper.skip_non_devtool",
                        extra={"event": "scraper.skip_non_devtool", "post_id": post_id},
                    )
                    continue

                devtools_count += 1
                category = get_devtools_category(full_text, name)
                if category:
                    description_text = f"[{category}] {tagline}\n\n{description}"
                else:
                    description_text = f"{tagline}\n\n{description}"

                startup = {
                    "name": name,
                    "url": post.get('url', ''),
                    "description": description_text,
                    "date_found": datetime.fromisoformat(post['createdAt'].replace('Z', '+00:00')),
                    "source": "Product Hunt"
                }

                save_startup(startup)
            
            logger.info(
                "scraper.complete",
                extra={
                    "event": "scraper.complete",
                    "devtools_count": devtools_count,
                    "total_posts": len(posts),
                },
            )
            
        except requests.RequestException:
            logger.exception(
                "scraper.request_failed",
                extra={"event": "scraper.request_failed"},
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            logger.exception(
                "scraper.parse_error",
                extra={"event": "scraper.parse_error"},
            )

if __name__ == "__main__":
    init_db()
    scrape_producthunt_api()
    logger.info("scraper.script_complete", extra={"event": "scraper.script_complete"})
