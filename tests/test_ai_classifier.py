"""Tests for the ai_classifier module."""

import importlib
import json
import time
import types

import cachetools
import pytest
import tenacity


def test_import_without_api_key_skips_client_initialization(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    import openai
    import ai_classifier

    def fail_if_called(*args, **kwargs):
        raise RuntimeError("OpenAI client should not initialize without API key")

    monkeypatch.setattr(openai, "OpenAI", fail_if_called)
    importlib.reload(ai_classifier)

    assert ai_classifier.client is None


def test_devtools_keywords_module_level_constant(reset_ai_classifier):
    classifier = reset_ai_classifier
    # Verify the module-level constant exists and is non-empty
    assert len(classifier.DEVTOOLS_KEYWORDS) > 0
    assert len(classifier._DEVTOOLS_KEYWORDS_LOWER) == len(classifier.DEVTOOLS_KEYWORDS)
    # Verify all lowercase entries match the original list
    for kw, kw_lower in zip(classifier.DEVTOOLS_KEYWORDS, classifier._DEVTOOLS_KEYWORDS_LOWER):
        assert kw_lower == kw.lower()


def test_has_devtools_keywords(reset_ai_classifier):
    classifier = reset_ai_classifier
    assert classifier.has_devtools_keywords("Great CLI for developers", "Tool")
    assert not classifier.has_devtools_keywords("A gardening app", "Gardenify")


def test_is_devtools_related_ai_pre_filter(reset_ai_classifier, monkeypatch):
    classifier = reset_ai_classifier
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    # Patch client to record calls; keyword filter should skip API call
    calls = {"count": 0}

    def fake_create(*args, **kwargs):
        calls["count"] += 1
        return types.SimpleNamespace(output_text="no")

    classifier.client.responses.create = fake_create
    assert not classifier.is_devtools_related_ai("A travel planner")
    assert calls["count"] == 0


def test_is_devtools_related_ai_without_api_key_uses_fallback(reset_ai_classifier, monkeypatch):
    classifier = reset_ai_classifier
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert classifier.is_devtools_related_ai("Developer platform with SDKs")
    assert not classifier.is_devtools_related_ai("Photo album for families")


def test_is_devtools_related_ai_with_invalid_response_falls_back(reset_ai_classifier, monkeypatch):
    classifier = reset_ai_classifier
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    def fake_create(*args, **kwargs):
        return types.SimpleNamespace(output_text="maybe")

    classifier.client.responses.create = fake_create
    assert classifier.is_devtools_related_ai("Developer utilities for CI pipelines")


def test_is_devtools_related_ai_handles_api_errors(reset_ai_classifier, monkeypatch):
    classifier = reset_ai_classifier
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    def failing_create(*args, **kwargs):
        raise RuntimeError("down")

    classifier.client.responses.create = failing_create
    assert classifier.is_devtools_related_ai("Developer monitoring with logs")


def test_is_devtools_related_ai_with_valid_response(reset_ai_classifier, monkeypatch):
    classifier = reset_ai_classifier
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    def fake_create(*args, **kwargs):
        return types.SimpleNamespace(output_text="yes")

    classifier.client.responses.create = fake_create
    assert classifier.is_devtools_related_ai("CLI deployment helper", "DeployTool")


def test_get_devtools_category_without_api_key(reset_ai_classifier, monkeypatch):
    classifier = reset_ai_classifier
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert classifier.get_devtools_category("Any text") is None


def test_get_devtools_category_success(reset_ai_classifier, monkeypatch):
    classifier = reset_ai_classifier
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    def fake_create(*args, **kwargs):
        return types.SimpleNamespace(output_text="CLI Tool")

    classifier.client.responses.create = fake_create
    assert classifier.get_devtools_category("CLI utilities", "Tooler") == "CLI Tool"


def test_is_devtools_related_fallback(reset_ai_classifier):
    classifier = reset_ai_classifier
    assert classifier.is_devtools_related_fallback("Manage CI pipelines")
    assert not classifier.is_devtools_related_fallback("Handmade pottery marketplace")


def test_get_devtools_category_handles_exception(reset_ai_classifier, monkeypatch):
    classifier = reset_ai_classifier
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    def failing_create(*args, **kwargs):
        raise RuntimeError("bad request")

    classifier.client.responses.create = failing_create
    assert classifier.get_devtools_category("API helper", "API Helper") is None


def test_openai_stub_queue_and_raise(reset_ai_classifier):
    classifier = reset_ai_classifier
    stub = classifier.client.responses
    stub.queue_response("no")
    stub.queue_response("yes")
    first = stub.create(None)
    assert first.output_text == "no"
    second = stub.create(None)
    assert second.output_text == "yes"

    stub.raise_on_create()
    with pytest.raises(Exception):
        stub.create(None)


def test_classify_candidates_caches_results(monkeypatch):
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_CACHE", "0")
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_BATCH", "0")
    monkeypatch.setenv("AI_CLASSIFIER_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    import ai_classifier

    importlib.reload(ai_classifier)

    stub = ai_classifier.client.responses
    stub.queue_response({"results": {"a": "yes", "b": "no"}})
    stub.queue_response({"results": {"a": "no", "b": "yes"}})

    candidates = [
        {"id": "a", "name": "Tool A", "text": "developer CLI"},
        {"id": "b", "name": "Tool B", "text": "developer API"},
    ]

    first = ai_classifier.classify_candidates(candidates)
    assert first["a"] is True
    assert first["b"] is False
    assert stub.calls == 1

    second = ai_classifier.classify_candidates(candidates)
    assert second["a"] is True
    assert second["b"] is False
    assert stub.calls == 1  # cache hit prevents second API call


def test_classify_candidates_cache_uses_cachetools_ttl(monkeypatch):
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_CACHE", "0")
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_BATCH", "1")
    monkeypatch.setenv("AI_CLASSIFIER_CACHE_TTL", "0.01")
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    import ai_classifier

    importlib.reload(ai_classifier)

    assert isinstance(ai_classifier._classification_cache, cachetools.TTLCache)

    stub = ai_classifier.client.responses
    stub.queue_response("yes")
    stub.queue_response("no")

    candidate = [{"id": "a", "name": "Tool A", "text": "developer CLI"}]
    first = ai_classifier.classify_candidates(candidate)
    assert first["a"] is True
    assert stub.calls == 1

    time.sleep(0.05)

    second = ai_classifier.classify_candidates(candidate)
    assert second["a"] is False
    assert stub.calls == 2


def test_call_openai_retries_with_tenacity(monkeypatch):
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_CACHE", "1")
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_BATCH", "1")
    monkeypatch.setenv("AI_CLASSIFIER_MAX_RETRIES", "3")
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    import ai_classifier

    importlib.reload(ai_classifier)

    retry_instance = ai_classifier._build_openai_retry()
    assert isinstance(retry_instance, tenacity.Retrying)

    stub = ai_classifier.client.responses
    calls = {"count": 0}

    def fake_create(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("rate limit exceeded")
        return types.SimpleNamespace(output_text="yes")

    stub.create = types.MethodType(fake_create, stub)
    monkeypatch.setattr(tenacity.nap, "sleep", lambda *_, **__: None)

    response = ai_classifier._call_openai(
        [{"role": "user", "content": "hello"}],
        max_output_tokens=5,
    )

    assert calls["count"] == 3
    assert response.output_text == "yes"


def test_batch_max_tokens_sufficient_for_payload_size(monkeypatch):
    """Bug #16: max_tokens=payload.__len__() * 4 gives 32 tokens for a batch
    of 8, but a JSON response like {"results": {"id": "yes", ...}} for 8 items
    needs ~60+ tokens. Verify max_tokens scales adequately with batch size."""
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_CACHE", "1")
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_BATCH", "0")
    monkeypatch.setenv("AI_CLASSIFIER_BATCH_SIZE", "8")
    monkeypatch.setenv("AI_CLASSIFIER_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    import ai_classifier

    importlib.reload(ai_classifier)

    # Track the max_tokens value passed to the OpenAI API
    captured_kwargs = {}
    original_create = ai_classifier.client.responses.create

    def capturing_create(*args, **kwargs):
        captured_kwargs.update(kwargs)
        # Return a valid batch response for all 8 items
        result = {
            "results": {
                f"item-{i}": "yes" for i in range(8)
            }
        }
        return types.SimpleNamespace(output_text=json.dumps(result))

    ai_classifier.client.responses.create = capturing_create

    candidates = [
        {"id": f"item-{i}", "name": f"Tool {i}", "text": "developer CLI tool"}
        for i in range(8)
    ]

    ai_classifier.classify_candidates(candidates)

    # With 8 items, the old formula gave 8*4=32 tokens. A minimal JSON response
    # for 8 items like {"results":{"item-0":"yes",...}} needs ~60+ tokens.
    # The fix should give at least len(payload)*20+50 = 210 tokens.
    assert "max_output_tokens" in captured_kwargs, "max_output_tokens was not passed to the API"
    assert captured_kwargs["max_output_tokens"] >= 60, (
        f"max_output_tokens={captured_kwargs['max_output_tokens']} is too low for a batch of 8 items; "
        f"expected at least 60 tokens to avoid JSON truncation"
    )


def test_partial_batch_response_falls_back_to_single_for_missing_ids(monkeypatch):
    """Bug #2: When result_map is missing some candidate IDs (e.g. due to
    truncation or LLM quirks), result_map.get(id) returns None and
    str(None).strip().lower() == 'yes' silently evaluates to False. Instead,
    missing IDs should fall back to _classify_single."""
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_CACHE", "1")
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_BATCH", "0")
    monkeypatch.setenv("AI_CLASSIFIER_BATCH_SIZE", "4")
    monkeypatch.setenv("AI_CLASSIFIER_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    import ai_classifier

    importlib.reload(ai_classifier)

    call_log = []

    # The batch response only includes 2 of 4 items -- items "c" and "d" are missing
    def fake_create(*args, **kwargs):
        call_log.append(kwargs.get("input", args[0] if args else None))
        messages = kwargs.get("input", args[0] if args else [])

        # First call is the batch call: return partial results (only a and b)
        if len(call_log) == 1:
            result = {"results": {"a": "yes", "b": "no"}}
            content = json.dumps(result)
        else:
            # Subsequent calls are single-item fallbacks -- classify as "yes"
            content = "yes"

        return types.SimpleNamespace(output_text=content)

    ai_classifier.client.responses.create = fake_create

    candidates = [
        {"id": "a", "name": "Tool A", "text": "developer CLI tool"},
        {"id": "b", "name": "Tool B", "text": "developer API framework"},
        {"id": "c", "name": "Tool C", "text": "developer testing library"},
        {"id": "d", "name": "Tool D", "text": "developer monitoring SDK"},
    ]

    results = ai_classifier.classify_candidates(candidates)

    # Items present in batch response should use the batch answer
    assert results["a"] is True, "Item 'a' was in batch response as 'yes'"
    assert results["b"] is False, "Item 'b' was in batch response as 'no'"

    # Items missing from batch response should NOT silently be False.
    # They should fall back to _classify_single which returns "yes" in our stub.
    assert results["c"] is True, (
        "Item 'c' was missing from batch response and should have fallen back "
        "to _classify_single (which returns 'yes'), but was silently set to False"
    )
    assert results["d"] is True, (
        "Item 'd' was missing from batch response and should have fallen back "
        "to _classify_single (which returns 'yes'), but was silently set to False"
    )

    # Verify that fallback calls were made (batch + 2 single calls = 3 total)
    assert len(call_log) >= 3, (
        f"Expected at least 3 API calls (1 batch + 2 single fallbacks), got {len(call_log)}"
    )
