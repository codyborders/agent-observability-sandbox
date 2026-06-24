"""OpenAI Agents SDK chatbot for natural language developer tool recommendations."""

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from agents import Agent, ModelSettings, Runner, function_tool, handoff
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.items import ToolCallOutputItem
from ddtrace.llmobs import LLMObs

from database import count_all_startups, search_startups
from logging_config import get_logger

logger = get_logger("devtools.chatbot")


def _positive_int_env(var_name: str, default: int) -> int:
    """Read a positive integer environment variable."""
    assert default >= 1, "default must be positive"
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default
    try:
        parsed_value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed_value)


_CHATBOT_MODEL = os.getenv("CHATBOT_MODEL", "gpt-5-nano")
_CHATBOT_WORKFLOW = os.getenv("CHATBOT_WORKFLOW", "council")
_CHATBOT_MAX_TURNS = _positive_int_env("CHATBOT_MAX_TURNS", 3)
_COUNCIL_MODEL = os.getenv("CHATBOT_COUNCIL_MODEL", _CHATBOT_MODEL)
_COUNCIL_MAX_TURNS = _positive_int_env("CHATBOT_COUNCIL_MAX_TURNS", 10)
_MAX_TOOLS_IN_CONTEXT = _positive_int_env("CHATBOT_MAX_TOOLS", 10)
_WORKFLOW_NAME = "recommendation_council"
_COUNCIL_AGENT_NAMES = ["evaluator", "intent", "search", "skeptic", "writer"]

# FTS5 operator pattern to sanitize user-supplied queries
_FTS5_OPERATORS = re.compile(r"[^\w\s]")
_FTS5_KEYWORDS = re.compile(r"\b(AND|OR|NOT|NEAR)\b", re.IGNORECASE)

_SYSTEM_PROMPT = """\
You are a helpful assistant for DevTools Scraper, a developer tools discovery platform.
Your job is to recommend developer tools from our database based on what the user asks.

Guidelines:
- Be conversational and concise. Give a 2-4 sentence intro, then list relevant tools.
- Recommend at most 5 tools per response.
- For each tool, include its name in bold with **Name** and a brief explanation of why it matches.
- If a tool description starts with a [Category] tag, mention the category.
- If no tools match, say so honestly and suggest the user try different terms or browse /search.
- Never invent tools that are not in the search results.
- Keep the total response under 300 words.
"""

_INTENT_PROMPT = """\
You are IntentAgent in a developer-tools recommendation council.
Extract the user's goal, constraints, and search strategy.
Your only successful action is calling transfer_to_search_agent. Do not answer the user directly.
"""

_SEARCH_PROMPT = """\
You are SearchAgent in a developer-tools recommendation council.
First call search_tools with a concise database query that matches the user's request.
Never invent tools that are not returned by search_tools.
After at least one search_tools result is in the conversation, call transfer_to_evaluator_agent.
Do not answer the user directly.
"""

_EVALUATOR_PROMPT = """\
You are EvaluatorAgent in a developer-tools recommendation council.
Review the user request and SearchAgent tool results in the conversation history.
Rank candidate tools against the user's goal using only returned database evidence.
Your only successful action is calling transfer_to_skeptic_agent. Do not answer the user directly.
"""

_SKEPTIC_PROMPT = """\
You are SkepticAgent in a developer-tools recommendation council.
Review the candidate tools for weak evidence, hallucinated claims, or poor fit.
Prefer fewer strong recommendations over many weak ones.
Your only successful action is calling transfer_to_writer_agent. Do not answer the user directly.
"""

_WRITER_PROMPT = """\
You are WriterAgent in a developer-tools recommendation council.
Write the final user-facing answer from the searched and approved candidate tools.
Rules:
- Recommend at most 5 tools.
- Use **Tool Name** for each recommendation.
- Explain why each tool matches the user's goal using only supplied database evidence.
- If no strong tools were found, say that no strong matches were found and suggest better search terms.
- Keep the answer under 300 words.
"""


def _prompt_tracking(prompt_id: str, template: str) -> dict[str, Any]:
    """Build fallback prompt metadata for Datadog annotation."""
    assert prompt_id.strip(), "prompt_id must not be blank"
    assert template.strip(), "template must not be blank"
    return {"id": prompt_id, "template": template, "version": "1.0"}


def _get_prompt_annotation(prompt_id: str, template: str) -> dict[str, Any]:
    """Fetch managed prompt metadata for annotation tracking."""
    if not prompt_id.strip():
        raise ValueError("prompt_id must not be blank")
    if not template.strip():
        raise ValueError("template must not be blank")
    try:
        prompt = LLMObs.get_prompt(prompt_id, label="production", fallback=template)
        return prompt.to_annotation_dict()
    except Exception:
        return _prompt_tracking(prompt_id, template)


def _sanitize_fts_query(query: str) -> str:
    """Strip FTS5 operators from a query string to prevent injection."""
    cleaned = _FTS5_OPERATORS.sub(" ", query)
    cleaned = _FTS5_KEYWORDS.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _search_database(query: str) -> list[dict[str, Any]]:
    """Search the database after sanitizing the query and logging the request."""
    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return []

    logger.info(
        "chatbot.search",
        extra={"event": "chatbot.search", "query": sanitized},
    )
    return search_startups(sanitized, limit=_MAX_TOOLS_IN_CONTEXT)


@function_tool
def search_tools(query: str) -> str:
    """Search the developer tools database for tools matching a query.

    Args:
        query: A search term or phrase to find matching developer tools.
    """
    return json.dumps(_search_database(query), default=str)


@function_tool
def count_tools() -> str:
    """Return the total number of developer tools in the database."""
    total = count_all_startups()
    logger.debug(
        "chatbot.count",
        extra={"event": "chatbot.count", "total": total},
    )
    return str(total)


def _deduplicate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tools with duplicate IDs removed while preserving order."""
    if not isinstance(tools, list):
        raise TypeError("tools must be a list")
    seen_ids: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_id = tool.get("id")
        if tool_id is None:
            continue
        tool_key = str(tool_id)
        if tool_key in seen_ids:
            continue
        seen_ids.add(tool_key)
        deduplicated.append(tool)
    return deduplicated


def _collect_tools(result: Any) -> list[dict[str, Any]]:
    """Extract deduplicated tool dicts from an agent run's tool call outputs."""
    collected_tools: list[dict[str, Any]] = []
    for item in getattr(result, "new_items", []):
        if not isinstance(item, ToolCallOutputItem):
            continue
        try:
            parsed = json.loads(item.output)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, list):
            continue
        collected_tools.extend(tool for tool in parsed if isinstance(tool, dict))
    return _deduplicate_tools(collected_tools)


def _result_text(result: Any) -> str:
    """Return an agent result's final output as stripped text."""
    output = getattr(result, "final_output", "")
    if output is None:
        return ""
    if not isinstance(output, str):
        return str(output).strip()
    return output.strip()


def _fallback_council_response(user_message: str, tools: list[dict[str, Any]]) -> str:
    """Build a deterministic final answer if WriterAgent returns empty output."""
    if not user_message.strip():
        raise ValueError("user_message must not be blank")
    if not tools:
        return "I could not find strong database matches. Try a narrower tool category or technology name."

    lines = ["Here are the strongest matches I found:"]
    for tool in tools[:5]:
        name = str(tool.get("name", "Unknown tool"))
        description = str(tool.get("description", "")).strip()
        if description.startswith("[") and "]" in description:
            description = description.split("]", 1)[1].strip()
        lines.append(f"- **{name}**: {description[:180]}")
    return "\n".join(lines)


def _normalize_workflow(workflow: str | None) -> str:
    """Map workflow aliases to supported chatbot execution modes."""
    raw_workflow = workflow if workflow is not None else _CHATBOT_WORKFLOW
    if not isinstance(raw_workflow, str):
        return "council"
    normalized = raw_workflow.strip().lower()
    if normalized in {"simple", "single", "assistant"}:
        return "simple"
    if normalized in {"council", "recommendation_council", "agent_council"}:
        return "council"
    logger.warning(
        "chatbot.workflow_unknown",
        extra={"event": "chatbot.workflow_unknown", "workflow": normalized},
    )
    return "council"


_agent = Agent(
    name="DevToolsAssistant",
    instructions=_SYSTEM_PROMPT,
    model=_CHATBOT_MODEL,
    tools=[search_tools, count_tools],
)

_writer_agent = Agent(
    name="WriterAgent",
    handoff_description="Writes the final user-facing recommendation answer.",
    instructions=_WRITER_PROMPT,
    model=_COUNCIL_MODEL,
)

_skeptic_agent = Agent(
    name="SkepticAgent",
    handoff_description="Checks candidates for weak evidence before final writing.",
    instructions=prompt_with_handoff_instructions(_SKEPTIC_PROMPT),
    handoffs=[
        handoff(
            _writer_agent,
            tool_name_override="transfer_to_writer_agent",
            tool_description_override="Required next step after checking candidate evidence.",
        )
    ],
    model=_COUNCIL_MODEL,
    model_settings=ModelSettings(tool_choice="transfer_to_writer_agent"),
)

_evaluator_agent = Agent(
    name="EvaluatorAgent",
    handoff_description="Ranks searched tools against the user's stated goal.",
    instructions=prompt_with_handoff_instructions(_EVALUATOR_PROMPT),
    handoffs=[
        handoff(
            _skeptic_agent,
            tool_name_override="transfer_to_skeptic_agent",
            tool_description_override="Required next step after ranking searched candidates.",
        )
    ],
    model=_COUNCIL_MODEL,
    model_settings=ModelSettings(tool_choice="transfer_to_skeptic_agent"),
)

_search_agent = Agent(
    name="SearchAgent",
    handoff_description="Searches the local developer-tools database with search_tools.",
    instructions=prompt_with_handoff_instructions(_SEARCH_PROMPT),
    tools=[search_tools],
    handoffs=[
        handoff(
            _evaluator_agent,
            tool_name_override="transfer_to_evaluator_agent",
            tool_description_override="Required next step after search_tools returns candidate tools.",
        )
    ],
    model=_COUNCIL_MODEL,
    model_settings=ModelSettings(tool_choice="search_tools"),
)

_intent_agent = Agent(
    name="IntentAgent",
    instructions=prompt_with_handoff_instructions(_INTENT_PROMPT),
    handoffs=[
        handoff(
            _search_agent,
            tool_name_override="transfer_to_search_agent",
            tool_description_override="Required next step for every developer-tool recommendation request.",
        )
    ],
    model=_COUNCIL_MODEL,
    model_settings=ModelSettings(tool_choice="transfer_to_search_agent"),
)


@dataclass(frozen=True)
class _CouncilRun:
    """Result of one recommendation council execution."""

    response_text: str
    selected_tools: list[dict[str, Any]]
    candidate_count: int


def _run_council_agents(user_message: str) -> _CouncilRun:
    """Run the native OpenAI Agents SDK handoff chain once."""
    if not user_message.strip():
        raise ValueError("user_message must not be blank")

    result = Runner.run_sync(
        _intent_agent,
        input=user_message,
        max_turns=_COUNCIL_MAX_TURNS,
    )
    selected_tools = _collect_tools(result)[:5]
    response_text = _result_text(result) or _fallback_council_response(
        user_message,
        selected_tools,
    )

    return _CouncilRun(
        response_text=response_text,
        selected_tools=selected_tools,
        candidate_count=len(selected_tools),
    )


def generate_recommendation_council_response(
    user_message: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run a multi-agent recommendation council for a user question."""
    if not user_message.strip():
        raise ValueError("user_message must not be blank")

    task_id = str(uuid.uuid4())
    try:
        council_run = _run_council_agents(user_message)
        logger.info(
            "chatbot.council.response",
            extra={
                "event": "chatbot.council.response",
                "task_id": task_id,
                "message_length": len(user_message),
                "candidate_count": council_run.candidate_count,
                "tools_found": len(council_run.selected_tools),
                "response_length": len(council_run.response_text),
                "model": _COUNCIL_MODEL,
                "session_id": session_id,
            },
        )
        return {
            "response": council_run.response_text,
            "tools": council_run.selected_tools,
            "workflow": _WORKFLOW_NAME,
            "agents": _COUNCIL_AGENT_NAMES,
            "model": _COUNCIL_MODEL,
            "task_id": task_id,
        }

    except Exception:
        logger.exception(
            "chatbot.council.error",
            extra={"event": "chatbot.council.error", "message_length": len(user_message)},
        )
        return {
            "response": (
                "I'm having trouble running the recommendation council right now. "
                "Please try the search page at /search for finding tools."
            ),
            "tools": [],
            "workflow": _WORKFLOW_NAME,
            "agents": _COUNCIL_AGENT_NAMES,
            "model": _COUNCIL_MODEL,
            "task_id": task_id,
        }


def _generate_single_agent_response(
    user_message: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Generate a single-agent chatbot response for a user question."""
    if not user_message.strip():
        raise ValueError("user_message must not be blank")

    try:
        with LLMObs.annotation_context(
            prompt=_get_prompt_annotation("devtools-assistant", _SYSTEM_PROMPT)
        ):
            result = Runner.run_sync(
                _agent,
                input=user_message,
                max_turns=_CHATBOT_MAX_TURNS,
            )

        response_text = _result_text(result)
        tools = _collect_tools(result)

        logger.info(
            "chatbot.response",
            extra={
                "event": "chatbot.response",
                "message_length": len(user_message),
                "tools_found": len(tools),
                "response_length": len(response_text),
                "model": _CHATBOT_MODEL,
                "session_id": session_id,
            },
        )
        return {
            "response": response_text,
            "tools": tools,
            "workflow": "single_agent",
            "agents": ["assistant"],
            "model": _CHATBOT_MODEL,
        }

    except Exception:
        logger.exception(
            "chatbot.error",
            extra={"event": "chatbot.error", "message_length": len(user_message)},
        )
        return {
            "response": (
                "I'm having trouble connecting to my AI service right now. "
                "Please try the search page at /search for finding tools."
            ),
            "tools": [],
            "workflow": "single_agent",
            "agents": ["assistant"],
            "model": _CHATBOT_MODEL,
        }


def generate_chat_response(
    user_message: str,
    session_id: str | None = None,
    workflow: str | None = None,
) -> dict[str, Any]:
    """Generate a chatbot response using the selected agent workflow.

    Args:
        user_message: The user's question, already validated by the caller.
        session_id: Optional Datadog RUM session ID for browser correlation.
        workflow: Optional execution mode. Use ``council`` for the multi-agent
            recommendation council or ``simple`` for the original single agent.

    Returns:
        Dict with response text, matched tools, workflow metadata, and model name.
    """
    if not user_message.strip():
        raise ValueError("user_message must not be blank")

    selected_workflow = _normalize_workflow(workflow)
    if selected_workflow == "simple":
        return _generate_single_agent_response(user_message, session_id=session_id)
    return generate_recommendation_council_response(user_message, session_id=session_id)
