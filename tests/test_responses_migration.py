"""Tests that ai_classifier uses the OpenAI Responses API.

These tests verify:
1. _call_openai uses client.responses.create (not chat.completions)
2. Parameters are translated: input, max_output_tokens, text.format
3. Response parsing uses output_text (not choices[0].message.content)
"""

import importlib
import json
import types

import pytest


def test_call_openai_uses_responses_create(reset_ai_classifier):
    """Verify _call_openai calls client.responses.create."""
    classifier = reset_ai_classifier
    captured = {}

    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(output_text="yes")

    classifier.client.responses.create = fake_create

    response = classifier._call_openai(
        [{"role": "user", "content": "test"}],
        max_output_tokens=5,
    )

    assert response.output_text == "yes"
    assert "input" in captured, "Should pass 'input' param, not 'messages'"
    assert "max_output_tokens" in captured, "Should pass 'max_output_tokens', not 'max_tokens'"


def test_call_openai_translates_response_format_to_text(reset_ai_classifier):
    """Verify response_format is translated to text.format for Responses API."""
    classifier = reset_ai_classifier
    captured = {}

    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(output_text='{"results": {}}')

    classifier.client.responses.create = fake_create

    classifier._call_openai(
        [{"role": "user", "content": "test"}],
        max_output_tokens=50,
        text_format={"type": "json_object"},
    )

    assert "text" in captured, "Should pass 'text' param for JSON format"
    assert captured["text"] == {"format": {"type": "json_object"}}
    assert "response_format" not in captured, "Should not pass legacy 'response_format'"


def test_call_openai_omits_text_when_no_format(reset_ai_classifier):
    """Verify text param is omitted when no response_format is requested."""
    classifier = reset_ai_classifier
    captured = {}

    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(output_text="yes")

    classifier.client.responses.create = fake_create

    classifier._call_openai(
        [{"role": "user", "content": "test"}],
        max_output_tokens=5,
    )

    assert "text" not in captured, "Should not pass 'text' when no format requested"

