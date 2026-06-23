"""Tests for Datadog Prompt Management integration.

These tests verify:
1. The LLMObs stub supports get_prompt/clear_prompt_cache/refresh_prompt
2. ai_classifier uses managed prompts instead of hardcoded f-strings
3. chatbot uses managed prompts for annotation tracking
"""

import types

import pytest


def _fake_openai_response(content):
    """Build a minimal OpenAI Responses API-shaped response."""
    return types.SimpleNamespace(output_text=content)


def _spy_get_prompt(classifier, monkeypatch):
    """Patch _get_prompt on classifier and return the list of captured prompt IDs."""
    prompt_ids = []
    orig = classifier._get_prompt

    def spy(pid, fb):
        prompt_ids.append(pid)
        return orig(pid, fb)

    monkeypatch.setattr(classifier, "_get_prompt", spy)
    return prompt_ids


class TestLLMObsPromptStubs:
    """Tests that the LLMObs test stub supports prompt management methods."""

    def test_get_prompt_returns_managed_prompt(self):
        from ddtrace.llmobs import LLMObs

        fallback = [{"role": "user", "content": "Hello {{name}}"}]
        prompt = LLMObs.get_prompt("test-id", label="production", fallback=fallback)

        assert prompt.id == "test-id"
        assert prompt.label == "production"
        assert hasattr(prompt, "format")
        assert hasattr(prompt, "to_annotation_dict")

    def test_managed_prompt_format_replaces_variables(self):
        from ddtrace.llmobs import LLMObs

        prompt = LLMObs.get_prompt(
            "test-id",
            fallback=[
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Name: {{name}}, Desc: {{description}}"},
            ],
        )
        messages = prompt.format(name="MyTool", description="A dev tool")
        assert messages[0]["content"] == "Be helpful."
        assert "MyTool" in messages[1]["content"]
        assert "A dev tool" in messages[1]["content"]

    def test_managed_prompt_format_string_fallback(self):
        from ddtrace.llmobs import LLMObs

        prompt = LLMObs.get_prompt("t", fallback="Hello {{name}}")
        assert prompt.format(name="World") == "Hello World"

    def test_to_annotation_dict_structure(self):
        from ddtrace.llmobs import LLMObs

        fallback = [{"role": "user", "content": "{{q}}"}]
        prompt = LLMObs.get_prompt("p", label="production", fallback=fallback)
        ann = prompt.to_annotation_dict(q="test")
        assert ann["id"] == "p"
        assert "version" in ann
        assert ann["variables"] == {"q": "test"}

    def test_clear_prompt_cache_noop(self):
        from ddtrace.llmobs import LLMObs

        LLMObs.clear_prompt_cache()

    def test_refresh_prompt_returns_prompt(self):
        from ddtrace.llmobs import LLMObs

        prompt = LLMObs.refresh_prompt("pid", label="production")
        assert prompt.id == "pid"


class TestClassifierPromptManagement:
    """Tests that ai_classifier uses _get_prompt for managed prompts."""

    def test_classify_single_calls_get_prompt(self, reset_ai_classifier, monkeypatch):
        classifier = reset_ai_classifier
        monkeypatch.setenv("OPENAI_API_KEY", "present")
        classifier.client.responses.create = lambda *a, **kw: _fake_openai_response("yes")

        prompt_ids = _spy_get_prompt(classifier, monkeypatch)
        assert classifier._classify_single("DevCLI", "developer CLI tool") is True
        assert "devtools-binary-classifier" in prompt_ids

    def test_get_devtools_category_calls_get_prompt(self, reset_ai_classifier, monkeypatch):
        classifier = reset_ai_classifier
        monkeypatch.setenv("OPENAI_API_KEY", "present")
        classifier.client.responses.create = lambda *a, **kw: _fake_openai_response("CLI Tool")

        prompt_ids = _spy_get_prompt(classifier, monkeypatch)
        assert classifier.get_devtools_category("CLI utils", "MyCLI") == "CLI Tool"
        assert "devtools-category-classifier" in prompt_ids

    def test_get_prompt_falls_back_when_llmobs_none(self, reset_ai_classifier, monkeypatch):
        classifier = reset_ai_classifier
        monkeypatch.setattr(classifier, "_LLMObs", None)

        prompt = classifier._get_prompt("test", [
            {"role": "user", "content": "Hello {{name}}"},
        ])
        assert prompt.id == "test"
        assert prompt.version == "fallback"
        messages = prompt.format(name="World")
        assert messages[0]["content"] == "Hello World"

    def test_fallback_templates_exist(self, reset_ai_classifier):
        classifier = reset_ai_classifier
        assert isinstance(classifier._BINARY_CLASSIFIER_FALLBACK, list)
        assert isinstance(classifier._BATCH_CLASSIFIER_FALLBACK, list)
        assert isinstance(classifier._CATEGORY_CLASSIFIER_FALLBACK, list)
        assert len(classifier._BINARY_CLASSIFIER_FALLBACK) >= 2
