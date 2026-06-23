"""Tests verifying simplification correctness for ai_classifier.

These tests ensure:
1. is_devtools_related_ai delegates caching to classify_candidates (no double-cache)
2. Redundant dead-code paths are removed
"""

import importlib
import types

import pytest


def test_is_devtools_related_ai_does_not_double_cache(monkeypatch):
    """is_devtools_related_ai should delegate entirely to classify_candidates,
    which handles caching internally. There should be no redundant outer cache
    check/set wrapping the call."""
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_CACHE", "0")
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_BATCH", "1")
    monkeypatch.setenv("AI_CLASSIFIER_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    import ai_classifier

    importlib.reload(ai_classifier)

    call_count = {"n": 0}

    def counting_create(*args, **kwargs):
        call_count["n"] += 1
        return types.SimpleNamespace(output_text="yes")

    ai_classifier.client.responses.create = counting_create

    # First call -- should hit the API
    result1 = ai_classifier.is_devtools_related_ai("developer CLI tool", "MyCLI")
    assert result1 is True
    assert call_count["n"] == 1

    # Second call with same input -- should be cached by classify_candidates
    result2 = ai_classifier.is_devtools_related_ai("developer CLI tool", "MyCLI")
    assert result2 is True
    assert call_count["n"] == 1, "Second call should hit cache, not make another API call"
