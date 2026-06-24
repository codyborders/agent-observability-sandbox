"""Tests for chatbot query sanitization."""

from types import SimpleNamespace

import pytest
from agents.items import ToolCallOutputItem

from chatbot import (
    _collect_tools,
    _evaluator_agent,
    _intent_agent,
    _normalize_workflow,
    _sanitize_fts_query,
    _search_agent,
    _skeptic_agent,
    _writer_agent,
    generate_chat_response,
    generate_recommendation_council_response,
)


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


def test_recommendation_council_uses_native_handoff_graph() -> None:
    """Council should expose one OpenAI Agents SDK handoff chain for Datadog."""
    search_tool_names = [tool.name for tool in _search_agent.tools]

    assert [_intent_agent.handoffs[0].agent_name, _intent_agent.handoffs[0].tool_name] == [
        "SearchAgent",
        "transfer_to_search_agent",
    ]
    assert [_search_agent.handoffs[0].agent_name, _search_agent.handoffs[0].tool_name] == [
        "EvaluatorAgent",
        "transfer_to_evaluator_agent",
    ]
    assert [_evaluator_agent.handoffs[0].agent_name, _evaluator_agent.handoffs[0].tool_name] == [
        "SkepticAgent",
        "transfer_to_skeptic_agent",
    ]
    assert [_skeptic_agent.handoffs[0].agent_name, _skeptic_agent.handoffs[0].tool_name] == [
        "WriterAgent",
        "transfer_to_writer_agent",
    ]
    assert _writer_agent.handoffs == []
    assert _intent_agent.model_settings.tool_choice == "transfer_to_search_agent"
    assert _search_agent.model_settings.tool_choice == "search_tools"
    assert _evaluator_agent.model_settings.tool_choice == "transfer_to_skeptic_agent"
    assert _skeptic_agent.model_settings.tool_choice == "transfer_to_writer_agent"
    assert search_tool_names == ["search_tools"]


def test_recommendation_council_runs_one_sdk_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Council should let the SDK produce graph spans from one run."""
    calls: list[tuple[object, str, int]] = []

    class DummyAgent:
        pass

    agent = DummyAgent()
    tool_item = ToolCallOutputItem(
        agent=agent,
        raw_item={"type": "function_call_output", "call_id": "1", "output": ""},
        output='[{"id": 7, "name": "TraceKit", "description": "Production monitoring for Python services."}]',
    )

    def fake_run_sync(agent, input: str, max_turns: int):
        calls.append((agent, input, max_turns))
        return SimpleNamespace(
            final_output="Use **TraceKit** for production monitoring.",
            new_items=[tool_item],
        )

    monkeypatch.setattr("chatbot.Runner.run_sync", fake_run_sync)

    result = generate_recommendation_council_response("monitoring tools", session_id="rum-1")

    assert calls == [(_intent_agent, "monitoring tools", 10)]
    assert result["workflow"] == "recommendation_council"
    assert result["response"] == "Use **TraceKit** for production monitoring."
    assert [tool["id"] for tool in result["tools"]] == [7]


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
