"""OpenAI-powered classifier for identifying developer tools from scraped content."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from cachetools import TTLCache
from dotenv import load_dotenv
import openai
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

# Load environment variables from .env file before importing modules that depend on them
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from logging_config import get_logger, logging_context
from observability import trace_external_call

# Ensure DD_API_KEY is set for ddtrace prompt management
if not os.getenv("DD_API_KEY") and os.getenv("DATADOG_API_KEY"):
    os.environ["DD_API_KEY"] = os.getenv("DATADOG_API_KEY")

try:
    from ddtrace.llmobs import LLMObs as _LLMObs
except (ImportError, ModuleNotFoundError):
    _LLMObs = None

# Set up OpenAI client — LLM Observability is handled by ddtrace-run at process
# startup (DD_LLMOBS_ENABLED=1); calling LLMObs.enable() again here would
# conflict with the auto-instrumented session and suppress OpenAI LLM Obs spans.
logger = get_logger("devtools.ai")
client: Any | None = None
_client_lock = threading.Lock()
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Feature flags and tuning knobs
_CACHE_ENABLED = os.getenv("AI_CLASSIFIER_DISABLE_CACHE", "0") != "1"
_CACHE_TTL = float(os.getenv("AI_CLASSIFIER_CACHE_TTL", "3600"))
_CACHE_SIZE = int(os.getenv("AI_CLASSIFIER_CACHE_SIZE", "2048"))
_USE_BATCH = os.getenv("AI_CLASSIFIER_DISABLE_BATCH", "0") != "1"
_BATCH_SIZE = max(1, int(os.getenv("AI_CLASSIFIER_BATCH_SIZE", "8")))
_MAX_CONCURRENCY = max(1, int(os.getenv("AI_CLASSIFIER_MAX_CONCURRENCY", "4")))
_MAX_RETRIES = max(1, int(os.getenv("AI_CLASSIFIER_MAX_RETRIES", "3")))

# Input sanitization limits
_MAX_NAME_LENGTH = 200
_MAX_TEXT_LENGTH = 2000


def _has_openai_key() -> bool:
    """Return True when an OpenAI API key is configured."""
    return bool(os.getenv('OPENAI_API_KEY'))


def _get_openai_client() -> Any | None:
    """Lazily create the OpenAI client when an API key is available."""
    global client
    with _client_lock:
        if client is not None:
            return client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            client = openai.OpenAI(api_key=api_key)
        except Exception:
            logger.exception(
                "classifier.client_init_failed",
                extra={"event": "classifier.client_init_failed"},
            )
            return None
        return client


if _has_openai_key():
    _get_openai_client()


_classification_cache = TTLCache(_CACHE_SIZE, _CACHE_TTL)
_category_cache = TTLCache(_CACHE_SIZE, _CACHE_TTL)
_cache_lock = threading.RLock()


def _cache_get(cache: TTLCache, key: str) -> Any | None:
    """Look up a cached value, returning None on miss or when caching is disabled."""
    if not _CACHE_ENABLED:
        return None
    with _cache_lock:
        return cache.get(key)


def _cache_set(cache: TTLCache, key: str, value: Any) -> None:
    """Store a value in the cache unless caching is disabled."""
    if not _CACHE_ENABLED:
        return
    with _cache_lock:
        cache[key] = value

DEVTOOLS_KEYWORDS = (
    "developer", "devtool", "CLI", "SDK", "API", "code", "coding", "debug", "git",
    "CI", "CD", "DevOps", "terminal", "IDE", "framework", "testing", "monitoring",
    "observability", "build", "deploy", "infra", "cloud-native", "backend", "log",
    "linter", "formatter", "package manager", "dependency", "compiler", "interpreter",
    "container", "kubernetes", "docker", "microservice", "serverless", "database",
    "query", "schema", "migration", "deployment", "orchestration", "automation",
)

_DEVTOOLS_KEYWORDS_LOWER = tuple(kw.lower() for kw in DEVTOOLS_KEYWORDS)
_DEVTOOLS_PATTERN = re.compile("|".join(re.escape(kw) for kw in _DEVTOOLS_KEYWORDS_LOWER))

# ---------------------------------------------------------------------------
# Prompt Management: fallback chat templates used when Datadog prompts are
# unavailable.  Variable placeholders use {{var}} syntax matching the Datadog
# Prompt Management template format.
# ---------------------------------------------------------------------------

_BINARY_CLASSIFIER_FALLBACK = [
    {
        "role": "system",
        "content": (
            "You are a binary classifier for developer tools. "
            "Respond with EXACTLY 'yes' or 'no'. If uncertain, respond 'no'."
        ),
    },
    {
        "role": "user",
        "content": (
            "You are a classifier that determines if software/tools are "
            "developer tools (devtools).\n\n"
            "Devtools include:\n"
            "- Development tools (IDEs, text editors, debuggers)\n"
            "- Build tools, package managers, CI/CD tools\n"
            "- Testing frameworks, monitoring tools\n"
            "- API tools, SDKs, libraries\n"
            "- DevOps tools, infrastructure tools\n"
            "- Code analysis, linting, formatting tools\n"
            "- Database tools, deployment tools\n"
            "- Terminal tools, CLI applications\n"
            "- Developer productivity tools\n\n"
            "NOT devtools:\n"
            "- End-user applications (games, social media, productivity apps)\n"
            "- Business software, marketing tools\n"
            "- Consumer apps, entertainment apps\n"
            "- E-commerce, finance apps (unless specifically for developers)\n\n"
            "Content to classify:\n"
            "Name: {{name}}\n"
            "Description: {{description}}\n\n"
            "Answer with ONLY \"yes\" or \"no\"."
        ),
    },
]

_BATCH_CLASSIFIER_FALLBACK = [
    {
        "role": "system",
        "content": (
            "Classify each item as devtools-related. "
            "Respond with JSON object {\"results\": {\"<item_id>\": \"yes\"|\"no\", ...}}. "
            "If unsure, respond with \"no\"."
        ),
    },
    {
        "role": "user",
        "content": "{{items_json}}",
    },
]

_CATEGORY_CLASSIFIER_FALLBACK = [
    {
        "role": "system",
        "content": (
            "You are a devtool categorizer. Respond with EXACTLY one of the "
            "specified category names. If the tool doesn't fit, respond with 'Other'."
        ),
    },
    {
        "role": "user",
        "content": (
            "Classify this devtool into one of these categories:\n"
            "- IDE/Editor: Integrated development environments, code editors\n"
            "- CLI Tool: Command line tools, terminal applications\n"
            "- Testing: Testing frameworks, test runners, mocking tools\n"
            "- Build/Deploy: Build tools, deployment tools, CI/CD\n"
            "- Monitoring/Observability: Logging, metrics, tracing, alerting\n"
            "- Database: Database tools, ORMs, query builders\n"
            "- API/SDK: API tools, SDKs, client libraries\n"
            "- DevOps: Infrastructure, containerization, orchestration\n"
            "- Code Quality: Linters, formatters, static analysis\n"
            "- Package Manager: Dependency management, package managers\n"
            "- Other: Anything else\n\n"
            "Examples:\n"
            "Name: VSCode\n"
            "Description: A code editor for developers.\n"
            "Category: IDE/Editor\n\n"
            "Name: GitHub Actions\n"
            "Description: A CI/CD automation tool for code repositories.\n"
            "Category: Build/Deploy\n\n"
            "Name: Postman\n"
            "Description: API development and testing tool.\n"
            "Category: API/SDK\n\n"
            "Name: Datadog\n"
            "Description: Cloud monitoring and observability platform.\n"
            "Category: Monitoring/Observability\n\n"
            "Name: Slack\n"
            "Description: A team chat and collaboration app.\n"
            "Category: Other\n\n"
            "Name: QuickBooks\n"
            "Description: Accounting software for small businesses.\n"
            "Category: Other\n\n"
            "Name: {{name}}\n"
            "Description: {{description}}\n\n"
            "Respond with ONLY the category name."
        ),
    },
]


class _LocalPrompt:
    """Minimal ManagedPrompt stand-in when ddtrace is unavailable."""

    def __init__(self, prompt_id: str, template: list[dict[str, str]]) -> None:
        self.id = prompt_id
        self.version = "fallback"
        self.label = "local"
        self._template = template

    def format(self, **variables: str) -> list[dict[str, str]]:
        result = []
        for msg in self._template:
            content = msg["content"]
            for key, value in variables.items():
                content = content.replace("{{" + key + "}}", str(value))
            result.append({"role": msg["role"], "content": content})
        return result

    def to_annotation_dict(self, **variables: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "template": self._template,
            "variables": variables,
        }


def _get_prompt(prompt_id: str, fallback: list[dict[str, str]]) -> Any:
    """Fetch a managed prompt from Datadog, falling back to local template."""
    if _LLMObs is not None:
        try:
            return _LLMObs.get_prompt(prompt_id, label="production", fallback=fallback)
        except Exception:
            logger.warning(
                "prompt.fetch_failed",
                extra={"event": "prompt.fetch_failed", "prompt_id": prompt_id},
            )
    return _LocalPrompt(prompt_id, fallback)


@contextmanager
def _prompt_context(prompt: Any, variables: dict[str, str]) -> Iterator[None]:
    """Wrap a block in LLMObs.annotation_context when available."""
    if _LLMObs is not None:
        with _LLMObs.annotation_context(prompt=prompt.to_annotation_dict(**variables)):
            yield
    else:
        yield


def has_devtools_keywords(text: str, name: str = "") -> bool:
    """Quick keyword pre-filter to avoid unnecessary API calls."""
    combined_text = f"{name} {text}".lower()
    return bool(_DEVTOOLS_PATTERN.search(combined_text))


def _cache_key(name: str, text: str) -> str:
    """Build a normalised, collision-resistant cache key."""
    raw = f"{name.strip().lower()}\x00{text.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True for transient OpenAI errors worth retrying."""
    message = str(exc).lower()
    return any(keyword in message for keyword in ("rate limit", "timeout", "429"))


def _build_openai_retry() -> Retrying:
    """Create a tenacity Retrying instance for OpenAI API calls."""
    return Retrying(
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1),
        retry=retry_if_exception(_is_retryable_error),
        reraise=True,
    )


def _call_openai(
    input_messages: list[dict[str, str]],
    max_output_tokens: int,
    temperature: float = 0.0,
    text_format: dict[str, str] | None = None,
) -> Any:
    """Send a Responses API request to OpenAI with automatic retries."""
    openai_client = _get_openai_client()
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is missing or OpenAI client failed to initialize")

    # Wrap caller-provided format spec for the Responses API text parameter
    text_param = {"format": text_format} if text_format else None

    retrying = _build_openai_retry()
    for attempt in retrying:
        with attempt:
            attempt_number = attempt.retry_state.attempt_number
            logger.debug(
                "openai.request",
                extra={
                    "event": "openai.request",
                    "attempt": attempt_number,
                    "max_output_tokens": max_output_tokens,
                    "temperature": temperature,
                    "text_format": bool(text_format),
                },
            )
            try:
                with trace_external_call(
                    "openai.responses.create",
                    resource="classifier.batch",
                    span_type="llm",
                    tags={
                        "openai.model": _OPENAI_MODEL,
                        "openai.attempt": attempt_number,
                        "openai.max_output_tokens": max_output_tokens,
                        "openai.temperature": temperature,
                        "openai.text_format": bool(text_param),
                        "openai.input_count": len(input_messages),
                    },
                ) as span:
                    if span:
                        span.set_tag("span.kind", "client")
                        span.set_tag("component", "openai")
                    kwargs: dict[str, Any] = {
                        "model": _OPENAI_MODEL,
                        "input": input_messages,
                        "max_output_tokens": max_output_tokens,
                        "temperature": temperature,
                    }
                    if text_param:
                        kwargs["text"] = text_param
                    response = openai_client.responses.create(**kwargs)
                    if span and getattr(response, "usage", None):
                        usage = response.usage
                        input_t = getattr(usage, "input_tokens", 0)
                        output_t = getattr(usage, "output_tokens", 0)
                        span.set_tag("openai.input_tokens", input_t)
                        span.set_tag("openai.output_tokens", output_t)
                        span.set_tag("openai.total_tokens", input_t + output_t)
                    return response
            except Exception as exc:  # pragma: no cover - relies on API behaviour
                # Intentionally broad: log any API error and let tenacity decide whether to retry
                logger.warning(
                    "openai.error",
                    extra={
                        "event": "openai.error",
                        "attempt": attempt_number,
                        "error": type(exc).__name__,
                    },
                )
                raise


def classify_candidates(candidates: Iterable[dict[str, str]]) -> dict[str, bool]:
    """
    Batch classify multiple candidates. Each candidate must contain keys:
    - id: unique identifier
    - name: tool name
    - text: description
    """
    candidates = list(candidates)
    results: dict[str, bool] = {}
    pending: list[dict[str, str]] = []

    for candidate in candidates:
        cid = candidate["id"]
        name = candidate.get("name", "")
        text = candidate.get("text", "")
        with logging_context(candidate_id=cid, candidate_name=name):
            if not has_devtools_keywords(text, name):
                logger.debug(
                    "classifier.keyword_filter",
                    extra={
                        "event": "classifier.keyword_filter",
                    },
                )
                results[cid] = False
                continue
            key = _cache_key(name, text)
            cached = _cache_get(_classification_cache, key)
            if cached is not None:
                logger.debug(
                    "classifier.cache_hit",
                    extra={
                        "event": "classifier.cache_hit",
                        "outcome": cached,
                    },
                )
                results[cid] = cached
            else:
                candidate["_cache_key"] = key
                pending.append(candidate)

    if not pending:
        return results

    # Fetch prompts once per invocation instead of per-chunk
    single_prompt = _get_prompt("devtools-binary-classifier", _BINARY_CLASSIFIER_FALLBACK)

    if not _has_openai_key() or not _USE_BATCH or len(pending) == 1:
        for candidate in pending:
            outcome = _classify_single(
                candidate.get("name", ""),
                candidate.get("text", ""),
                prompt=single_prompt,
            )
            _cache_set(_classification_cache, candidate["_cache_key"], outcome)
            results[candidate["id"]] = outcome
        return results

    batch_prompt = _get_prompt("devtools-batch-classifier", _BATCH_CLASSIFIER_FALLBACK)
    chunks = [pending[i:i + _BATCH_SIZE] for i in range(0, len(pending), _BATCH_SIZE)]

    def worker(chunk: list[dict[str, str]]) -> dict[str, bool]:
        local_results: dict[str, bool] = {}
        payload = [
            {
                "item_id": candidate["id"],
                "name": candidate.get("name", "")[:_MAX_NAME_LENGTH],
                "description": candidate.get("text", "")[:_MAX_TEXT_LENGTH],
            }
            for candidate in chunk
        ]
        batch_variables = {"items_json": json.dumps(payload)}
        batch_messages = batch_prompt.format(**batch_variables)
        try:
            with _prompt_context(batch_prompt, batch_variables):
                response = _call_openai(
                    batch_messages,
                    max_output_tokens=len(payload) * 20 + 50,
                    temperature=0.0,
                    text_format={"type": "json_object"},
                )
            content = response.output_text.strip()
            data = json.loads(content)
            result_map = data.get("results", {})
        except Exception as exc:
            # Intentionally broad: any batch failure falls back to per-item classification
            logger.exception(
                "classifier.batch_failure",
                extra={"event": "classifier.batch_failure", "error": type(exc).__name__},
            )
            result_map = {}

        if not result_map:
            for candidate in chunk:
                with logging_context(candidate_id=candidate["id"], candidate_name=candidate.get("name")):
                    outcome = _classify_single(
                        candidate.get("name", ""),
                        candidate.get("text", ""),
                        prompt=single_prompt,
                    )
                    _cache_set(_classification_cache, candidate["_cache_key"], outcome)
                    local_results[candidate["id"]] = outcome
        else:
            for candidate in chunk:
                with logging_context(candidate_id=candidate["id"], candidate_name=candidate.get("name")):
                    answer = result_map.get(candidate["id"])
                    if answer is None:
                        logger.warning(
                            "classifier.batch_missing_id",
                            extra={
                                "event": "classifier.batch_missing_id",
                                "candidate_id": candidate["id"],
                            },
                        )
                        outcome = _classify_single(
                            candidate.get("name", ""),
                            candidate.get("text", ""),
                            prompt=single_prompt,
                        )
                    else:
                        outcome = str(answer).strip().lower() == "yes"
                    logger.debug(
                        "classifier.batch_result",
                        extra={
                            "event": "classifier.batch_result",
                            "outcome": outcome,
                        },
                    )
                    _cache_set(_classification_cache, candidate["_cache_key"], outcome)
                    local_results[candidate["id"]] = outcome

        return local_results

    with ThreadPoolExecutor(max_workers=min(_MAX_CONCURRENCY, len(chunks))) as executor:
        futures = [executor.submit(worker, chunk) for chunk in chunks]
        for future in as_completed(futures):
            results.update(future.result())

    return results


def _classify_single(
    name: str,
    text: str,
    *,
    prompt: Any | None = None,
) -> bool:
    """Classify a single candidate via the OpenAI API, falling back to keywords."""
    if not _has_openai_key():
        logger.warning(
            "classifier.no_api_key",
            extra={"event": "classifier.no_api_key"},
        )
        return is_devtools_related_fallback(text, name)

    if prompt is None:
        prompt = _get_prompt("devtools-binary-classifier", _BINARY_CLASSIFIER_FALLBACK)
    variables = {
        "name": name[:_MAX_NAME_LENGTH],
        "description": text[:_MAX_TEXT_LENGTH],
    }
    messages = prompt.format(**variables)
    try:
        with _prompt_context(prompt, variables):
            response = _call_openai(
                messages,
                max_output_tokens=16,
                temperature=0.0,
            )
        answer = response.output_text.strip().lower()
        if answer not in {"yes", "no"}:
            logger.warning(
                "classifier.unexpected_answer",
                extra={
                    "event": "classifier.unexpected_answer",
                    "answer": answer,
                    "tool_name": name,
                },
            )
            return is_devtools_related_fallback(text, name)
        return answer == "yes"
    except Exception:
        # Intentionally broad: any classification failure degrades to keyword matching
        logger.exception(
            "classifier.single_error",
            extra={"event": "classifier.single_error", "tool_name": name},
        )
        return is_devtools_related_fallback(text, name)


def is_devtools_related_ai(text: str, name: str = "") -> bool:
    """
    Use OpenAI to classify if content is devtools-related.
    Returns True if it's devtools, False otherwise.
    """
    # classify_candidates handles caching internally; no outer cache layer needed
    return classify_candidates([{"id": "_single", "name": name, "text": text}]).get("_single", False)


def is_devtools_related_fallback(text: str, name: str = "") -> bool:
    """Fallback keyword-based classifier when AI is unavailable."""
    return has_devtools_keywords(text, name)


def get_devtools_category(text: str, name: str = "") -> str | None:
    """
    Get a more specific category for the devtool.
    Returns category like 'IDE', 'CLI Tool', 'Testing', etc.
    """
    if not _has_openai_key():
        return None

    key = _cache_key(name, text)
    cached = _cache_get(_category_cache, key)
    if cached is not None:
        return cached

    prompt = _get_prompt("devtools-category-classifier", _CATEGORY_CLASSIFIER_FALLBACK)
    variables = {
        "name": name[:_MAX_NAME_LENGTH],
        "description": text[:_MAX_TEXT_LENGTH],
    }
    messages = prompt.format(**variables)

    try:
        with _prompt_context(prompt, variables):
            response = _call_openai(
                messages,
                max_output_tokens=16,
                temperature=0.0,
            )

        category = response.output_text.strip()
        _cache_set(_category_cache, key, category)
        logger.debug(
            "classifier.category",
            extra={
                "event": "classifier.category",
                "tool_name": name,
                "category": category,
            },
        )
        return category

    except Exception:
        # Intentionally broad: category failures return None rather than crashing
        logger.exception(
            "classifier.category_error",
            extra={"event": "classifier.category_error", "tool_name": name},
        )
        return None
