"""Tests for chatbot query sanitization."""

from types import SimpleNamespace

import pytest
from agents.items import ToolCallOutputItem

from chatbot import (
    _collect_tools,
    _manual_search_candidates,
    _normalize_workflow,
    _queries_from_intent,
    _run_agent_with_annotation,
    _sanitize_fts_query,
    _search_agent,
    _select_response_tools,
    generate_chat_response,
    generate_recommendation_council_response,
)


class _RecordingSpan:
    """Context manager that records span kind, name, and session id."""

    def __init__(
        self,
        spans: list[tuple[str, str, str | None]],
        kind: str,
        name: str,
        session_id: str | None = None,
    ) -> None:
        self._spans = spans
        self.kind = kind
        self.name = name
        self.session_id = session_id

    def __enter__(self) -> "_RecordingSpan":
        self._spans.append((self.kind, self.name, self.session_id))
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        return False


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("CI/CD tools", "CI CD tools"),
        ('python "web framework"', "python web framework"),
        ("node*", "node"),
        ("(react OR vue)", "react vue"),
        ("test AND deploy", "test deploy"),
        ("NEAR/3 rust", "3 rust"),
        ("key:value", "key value"),
        ("boost^2", "boost 2"),
        ("{prefix}", "prefix"),
        ("term1 + term2", "term1 term2"),
        ("front-end tools", "front end tools"),
        ("metrics, logs, traces.", "metrics logs traces"),
        ("feature 'code search'", "feature code search"),
        ("plain query", "plain query"),
        ("", ""),
        ("***", ""),
    ],
)
def test_sanitize_fts_query_strips_operators(raw: str, expected: str) -> None:
    """Verify FTS5 special characters and keywords are stripped from queries."""
    assert _sanitize_fts_query(raw) == expected


def test_queries_from_intent_sanitizes_and_keeps_user_fallback() -> None:
    """Council search queries should be safe and include the original ask as fallback."""
    intent_text = '{"search_queries": ["python AND monitoring", "agent* tracing"]}'

    queries = _queries_from_intent(intent_text, "OpenAI agent observability")

    assert queries == [
        "python monitoring",
        "agent tracing",
        "OpenAI agent observability",
    ]


def test_manual_search_candidates_falls_back_to_singular_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Long generated queries should still find seed data through bounded term search."""
    queries_seen: list[str] = []

    def fake_search_startups(query: str, limit: int) -> list[dict[str, object]]:
        queries_seen.append(query)
        assert limit >= 1
        if query == "agent":
            return [{"id": 8, "name": "AgentKit", "description": "AI agent toolkit."}]
        return []

    monkeypatch.setattr("chatbot.search_startups", fake_search_startups)

    candidates = _manual_search_candidates(["Recommend tools for running AI agents"])

    assert [candidate["id"] for candidate in candidates] == [8]
    assert "agents" in queries_seen
    assert "agent" in queries_seen


def test_collect_tools_deduplicates_tool_call_output() -> None:
    """SearchAgent tool outputs should become unique tool cards for the response."""
    class DummyAgent:
        pass

    agent = DummyAgent()
    first_item = ToolCallOutputItem(
        agent=agent,
        raw_item={"type": "function_call_output", "call_id": "1", "output": ""},
        output='[{"id": 1, "name": "TraceKit"}, {"id": 1, "name": "TraceKit"}]',
    )
    second_item = ToolCallOutputItem(
        agent=agent,
        raw_item={"type": "function_call_output", "call_id": "2", "output": ""},
        output='[{"id": 2, "name": "LogLens"}]',
    )
    result = SimpleNamespace(new_items=[first_item, second_item])

    tools = _collect_tools(result)

    assert [tool["id"] for tool in tools] == [1, 2]


def test_select_response_tools_prefers_skeptic_order() -> None:
    """SkepticAgent approval should decide which cards are returned first."""
    candidates = [
        {"id": 1, "name": "TraceKit"},
        {"id": 2, "name": "LogLens"},
        {"id": 3, "name": "MetricBox"},
    ]
    evaluation = '{"ranked_tools": [{"id": 1}, {"id": 2}]}'
    skeptic = '{"approved_ids": [2, 1]}'

    selected = _select_response_tools(candidates, evaluation, skeptic)

    assert [tool["id"] for tool in selected] == [2, 1]


def test_select_response_tools_respects_empty_skeptic_approval() -> None:
    """No tool cards should be returned when SkepticAgent rejects every candidate."""
    candidates = [
        {"id": 1, "name": "TraceKit"},
        {"id": 2, "name": "LogLens"},
    ]
    evaluation = '{"ranked_tools": [{"id": 1}, {"id": 2}]}'
    skeptic = '{"approved_ids": [], "concerns": ["weak matches"]}'

    selected = _select_response_tools(candidates, evaluation, skeptic)

    assert selected == []


def test_run_agent_with_annotation_creates_agent_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each council agent run should be wrapped in an Agent Observability span."""
    spans: list[tuple[str, str, str | None]] = []
    annotations: list[dict[str, object]] = []

    class RecordingPrompt:
        def to_annotation_dict(self):
            return {"id": "devtools-council-intent", "version": "test"}

    class RecordingLLMObs:
        @classmethod
        def agent(cls, name: str, session_id: str | None = None):
            return _RecordingSpan(spans, "agent", name, session_id)

        @classmethod
        def annotation_context(cls, **kwargs):
            return _RecordingSpan(spans, "annotation_context", "prompt")

        @classmethod
        def annotate(cls, **kwargs):
            annotations.append(kwargs)

        @classmethod
        def get_prompt(cls, prompt_id: str, label: str | None = None, fallback: str | None = None):
            assert prompt_id == "devtools-council-intent"
            assert label == "production"
            assert fallback
            return RecordingPrompt()

    def fake_run_sync(agent, input: str, max_turns: int):
        assert input == "intent input"
        assert max_turns == 2
        return SimpleNamespace(final_output="intent output", new_items=[])

    monkeypatch.setattr("chatbot.LLMObs", RecordingLLMObs)
    monkeypatch.setattr("chatbot.Runner.run_sync", fake_run_sync)

    result = _run_agent_with_annotation(
        SimpleNamespace(name="IntentAgent"),
        "devtools-council-intent",
        "Extract intent.",
        "intent input",
        "intent",
        "task-1",
        "rum-1",
        2,
    )

    agent_annotations = [annotation for annotation in annotations if annotation.get("span")]
    assert result.final_output == "intent output"
    assert ("agent", "IntentAgent", "rum-1") in spans
    assert agent_annotations[0]["input_data"] == "intent input"
    assert agent_annotations[0]["output_data"] == "intent output"
    assert agent_annotations[0]["tags"]["agent.role"] == "intent"
    assert agent_annotations[0]["tags"]["task.id"] == "task-1"


def test_recommendation_council_uses_deterministic_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Council should retrieve candidates without a SearchAgent tool-call loop."""
    calls: list[str] = []
    search_inputs: list[str] = []
    span_names: list[tuple[str, str, str | None]] = []

    def fake_search_startups(query: str, limit: int) -> list[dict[str, object]]:
        assert limit >= 1
        if "monitoring" not in query:
            return []
        return [
            {
                "id": 7,
                "name": "TraceKit",
                "description": "Production monitoring for Python services.",
                "source": "seed",
            }
        ]

    def fake_run_agent(
        agent,
        prompt_id: str,
        prompt_template: str,
        agent_input: str,
        role: str,
        task_id: str,
        session_id: str | None,
        max_turns: int,
    ) -> SimpleNamespace:
        calls.append(role)
        assert prompt_id.startswith("devtools-council-")
        assert prompt_template.strip()
        assert task_id
        assert max_turns >= 1
        if role == "intent":
            return SimpleNamespace(final_output='{"search_queries": ["monitoring"]}', new_items=[])
        if role == "search":
            search_inputs.append(agent_input)
            return SimpleNamespace(final_output="Found one monitoring candidate.", new_items=[])
        if role == "evaluator":
            return SimpleNamespace(final_output='{"ranked_tools": [{"id": 7, "score": 5}]}', new_items=[])
        if role == "skeptic":
            return SimpleNamespace(final_output='{"approved_ids": [7], "concerns": []}', new_items=[])
        if role == "writer":
            return SimpleNamespace(final_output="Use **TraceKit** for production monitoring.", new_items=[])
        raise AssertionError(f"unexpected role: {role}")

    class RecordingLLMObs:
        @classmethod
        def workflow(cls, name: str, session_id: str | None = None):
            return _RecordingSpan(span_names, "workflow", name, session_id)

        @classmethod
        def retrieval(cls, name: str, session_id: str | None = None):
            return _RecordingSpan(span_names, "retrieval", name, session_id)

        @classmethod
        def annotate(cls, **kwargs):
            return None

    monkeypatch.setattr("chatbot.LLMObs", RecordingLLMObs)
    monkeypatch.setattr("chatbot.search_startups", fake_search_startups)
    monkeypatch.setattr("chatbot._run_agent_with_annotation", fake_run_agent)

    result = generate_recommendation_council_response("monitoring tools", session_id="rum-1")

    assert _search_agent.tools == []
    assert calls == ["intent", "search", "evaluator", "skeptic", "writer"]
    assert result["workflow"] == "recommendation_council"
    assert result["response"] == "Use **TraceKit** for production monitoring."
    assert [tool["id"] for tool in result["tools"]] == [7]
    assert ("workflow", "recommendation_council", "rum-1") in span_names
    assert ("retrieval", "candidate_database_search", "rum-1") in span_names
    assert "Search queries used" in search_inputs[0]
    assert "Candidate tools JSON" in search_inputs[0]


def test_generate_chat_response_routes_workflows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public chatbot entrypoint should support council and simple workflows."""
    calls = []

    def fake_council(message: str, session_id: str | None = None):
        calls.append(("council", message, session_id))
        return {"workflow": "recommendation_council", "response": "council", "tools": []}

    def fake_simple(message: str, session_id: str | None = None):
        calls.append(("simple", message, session_id))
        return {"workflow": "single_agent", "response": "simple", "tools": []}

    monkeypatch.setattr("chatbot.generate_recommendation_council_response", fake_council)
    monkeypatch.setattr("chatbot._generate_single_agent_response", fake_simple)

    assert generate_chat_response("monitoring tools", session_id="rum-1")["workflow"] == "recommendation_council"
    assert generate_chat_response("monitoring tools", workflow="simple")["workflow"] == "single_agent"
    assert calls == [
        ("council", "monitoring tools", "rum-1"),
        ("simple", "monitoring tools", None),
    ]


def test_normalize_workflow_defaults_to_council() -> None:
    """Unknown workflow values should fall back to the council demo."""
    assert _normalize_workflow(None) == "council"
    assert _normalize_workflow("agent_council") == "council"
    assert _normalize_workflow("single") == "simple"
    assert _normalize_workflow("unknown") == "council"
