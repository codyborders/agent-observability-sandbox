"""OpenAI Agents SDK chatbot for natural language developer tool recommendations."""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from agents import Agent, Runner, function_tool
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
_COUNCIL_MAX_TURNS = _positive_int_env("CHATBOT_COUNCIL_MAX_TURNS", 5)
_MAX_TOOLS_IN_CONTEXT = _positive_int_env("CHATBOT_MAX_TOOLS", 10)
_WORKFLOW_NAME = "recommendation_council"
_COUNCIL_ROLES = {"intent", "search", "evaluator", "skeptic", "writer"}
_COUNCIL_AGENT_NAMES = sorted(_COUNCIL_ROLES)

# FTS5 operator pattern to sanitize user-supplied queries
_FTS5_OPERATORS = re.compile(r'["\*\(\)\+\-\^:/\{\}]')
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
Extract the user's goal, constraints, and database search strategy.
Return only JSON with these keys:
- summary: one sentence describing the user's goal
- criteria: 3 to 6 short strings that define a good match
- search_queries: 3 to 5 concise search queries for the local developer-tools database
Do not recommend tools. Do not add markdown.
"""

_SEARCH_PROMPT = """\
You are SearchAgent in a developer-tools recommendation council.
Use the search_tools function to search the local database for the provided queries.
Call search_tools for multiple distinct queries when possible.
You may call count_tools if the user asks about inventory size.
After searching, summarize what kinds of candidates were found.
Never invent tools that did not come from search_tools.
"""

_EVALUATOR_PROMPT = """\
You are EvaluatorAgent in a developer-tools recommendation council.
Score candidate tools against the user's stated goal and criteria.
Return only JSON with this shape:
{
  "ranked_tools": [
    {"id": 123, "score": 5, "reason": "short evidence-based reason"}
  ],
  "missing_needs": ["short gap if any"]
}
Use only candidate IDs from the provided candidate list. Do not add markdown.
"""

_SKEPTIC_PROMPT = """\
You are SkepticAgent in a developer-tools recommendation council.
Review the evaluator's ranking for weak evidence, hallucinated claims, or tools that do not match the user goal.
Return only JSON with this shape:
{
  "approved_ids": [123, 456],
  "concerns": ["short concern if any"]
}
Use only candidate IDs from the provided candidate list. Prefer fewer strong recommendations over many weak ones.
"""

_WRITER_PROMPT = """\
You are WriterAgent in a developer-tools recommendation council.
Write the final user-facing answer from the approved candidate tools.
Rules:
- Recommend at most 5 tools.
- Use **Tool Name** for each recommendation.
- Explain why each tool matches the user's goal using only supplied candidate evidence.
- If no tools are approved, say that no strong matches were found and suggest better search terms.
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


def _annotate_current_span(tags: dict[str, str]) -> None:
    """Attach tags to the active LLM span when LLMObs supports annotation."""
    if not isinstance(tags, dict):
        raise TypeError("tags must be a dictionary")
    if not tags:
        return
    annotate = getattr(LLMObs, "annotate", None)
    if annotate is None:
        return
    try:
        annotate(span=None, tags=tags)
    except Exception:
        logger.debug("chatbot.annotation_skipped", extra={"event": "chatbot.annotation_skipped"})


def _sanitize_fts_query(query: str) -> str:
    """Strip FTS5 operators from a query string to prevent injection."""
    cleaned = _FTS5_OPERATORS.sub(" ", query)
    cleaned = _FTS5_KEYWORDS.sub(" ", cleaned)
    return " ".join(cleaned.split())


@function_tool
def search_tools(query: str) -> str:
    """Search the developer tools database for tools matching a query.

    Args:
        query: A search term or phrase to find matching developer tools.
    """
    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return json.dumps([])

    logger.info(
        "chatbot.search",
        extra={"event": "chatbot.search", "query": sanitized},
    )
    return json.dumps(
        search_startups(sanitized, limit=_MAX_TOOLS_IN_CONTEXT),
        default=str,
    )


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


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from text, including fenced or prefixed output."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start_index = stripped.find("{")
        end_index = stripped.rfind("}")
        if start_index < 0 or end_index <= start_index:
            return {}
        try:
            parsed = json.loads(stripped[start_index : end_index + 1])
        except json.JSONDecodeError:
            return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _queries_from_intent(intent_text: str, user_message: str) -> list[str]:
    """Extract safe search queries from intent output with user text as fallback."""
    if not user_message.strip():
        raise ValueError("user_message must not be blank")
    parsed = _parse_json_object(intent_text)
    raw_queries = parsed.get("search_queries", [])
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    if not isinstance(raw_queries, list):
        raw_queries = []

    queries: list[str] = []
    for value in [*raw_queries, user_message]:
        candidate = _sanitize_fts_query(str(value))
        if candidate and candidate not in queries:
            queries.append(candidate)
    return queries[:5]


def _manual_search_candidates(queries: list[str]) -> list[dict[str, Any]]:
    """Search directly when SearchAgent does not produce tool-call output."""
    if not isinstance(queries, list):
        raise TypeError("queries must be a list")
    if not queries:
        return []

    candidates: list[dict[str, Any]] = []
    for query in queries:
        sanitized = _sanitize_fts_query(str(query))
        if not sanitized:
            continue
        candidates.extend(search_startups(sanitized, limit=_MAX_TOOLS_IN_CONTEXT))
    return _deduplicate_tools(candidates)[:_MAX_TOOLS_IN_CONTEXT]


def _candidate_payload(candidates: list[dict[str, Any]]) -> str:
    """Serialize a compact candidate list for council agents."""
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list")
    compact_candidates: list[dict[str, Any]] = []
    for tool in candidates[:_MAX_TOOLS_IN_CONTEXT]:
        if not isinstance(tool, dict):
            continue
        compact_candidates.append(
            {
                "id": tool.get("id"),
                "name": str(tool.get("name", ""))[:120],
                "description": str(tool.get("description", ""))[:500],
                "source": str(tool.get("source", ""))[:80],
            }
        )
    return json.dumps(compact_candidates, ensure_ascii=False)


def _ids_from_review(text: str, key: str) -> list[str]:
    """Extract ordered tool IDs from a review JSON object."""
    if not key.strip():
        raise ValueError("key must not be blank")
    parsed = _parse_json_object(text)
    raw_ids = parsed.get(key, [])
    if key == "ranked_tools" and isinstance(raw_ids, list):
        return [
            str(item.get("id"))
            for item in raw_ids
            if isinstance(item, dict) and item.get("id")
        ]
    if not isinstance(raw_ids, list):
        return []
    return [str(item) for item in raw_ids if item is not None]


def _select_response_tools(
    candidates: list[dict[str, Any]],
    evaluation_text: str,
    skeptic_text: str,
) -> list[dict[str, Any]]:
    """Select the tool cards that should accompany the council answer."""
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list")
    candidate_by_id = {
        str(tool.get("id")): tool
        for tool in candidates
        if tool.get("id") is not None
    }
    if not candidate_by_id:
        return []

    ordered_ids = _ids_from_review(skeptic_text, "approved_ids")
    if not ordered_ids:
        ordered_ids = _ids_from_review(evaluation_text, "ranked_tools")

    selected_tools = [
        candidate_by_id[tool_id]
        for tool_id in ordered_ids
        if tool_id in candidate_by_id
    ]
    if selected_tools:
        return selected_tools[:5]
    return candidates[:5]


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


def _run_agent_with_annotation(
    agent: Agent,
    prompt_id: str,
    prompt_template: str,
    agent_input: str,
    role: str,
    task_id: str,
    session_id: str | None,
    max_turns: int,
) -> Any:
    """Run one council agent with Datadog prompt and workflow metadata."""
    if role not in _COUNCIL_ROLES:
        raise ValueError(f"unsupported council role: {role}")
    if not agent_input.strip():
        raise ValueError("agent_input must not be blank")

    with LLMObs.annotation_context(prompt=_get_prompt_annotation(prompt_id, prompt_template)):
        result = Runner.run_sync(agent, input=agent_input, max_turns=max_turns)
        tags = {
            "workflow.name": _WORKFLOW_NAME,
            "agent.role": role,
            "task.id": task_id,
            "model.name": _COUNCIL_MODEL,
        }
        if session_id:
            tags["session_id"] = session_id
        _annotate_current_span(tags)
        return result


def _build_search_input(user_message: str, intent_text: str, queries: list[str]) -> str:
    """Build SearchAgent input from the intent stage."""
    if not user_message.strip():
        raise ValueError("user_message must not be blank")
    if not queries:
        raise ValueError("queries must not be empty")
    return (
        f"User request:\n{user_message}\n\n"
        f"IntentAgent output:\n{intent_text}\n\n"
        f"Search queries to run:\n{json.dumps(queries, ensure_ascii=False)}"
    )


def _build_review_input(
    user_message: str,
    intent_text: str,
    candidate_json: str,
    evaluation_text: str = "",
) -> str:
    """Build EvaluatorAgent or SkepticAgent input from candidate evidence."""
    if not candidate_json.strip():
        raise ValueError("candidate_json must not be blank")
    if not user_message.strip():
        raise ValueError("user_message must not be blank")
    parts = [
        f"User request:\n{user_message}",
        f"IntentAgent output:\n{intent_text}",
        f"Candidate tools JSON:\n{candidate_json}",
    ]
    if evaluation_text:
        parts.append(f"EvaluatorAgent output:\n{evaluation_text}")
    return "\n\n".join(parts)


def _build_writer_input(
    user_message: str,
    selected_json: str,
    evaluation_text: str,
    skeptic_text: str,
) -> str:
    """Build WriterAgent input from approved candidates and council notes."""
    if not user_message.strip():
        raise ValueError("user_message must not be blank")
    if not selected_json.strip():
        raise ValueError("selected_json must not be blank")
    return (
        f"User request:\n{user_message}\n\n"
        f"Approved candidate tools JSON:\n{selected_json}\n\n"
        f"EvaluatorAgent output:\n{evaluation_text}\n\n"
        f"SkepticAgent output:\n{skeptic_text}"
    )


_agent = Agent(
    name="DevToolsAssistant",
    instructions=_SYSTEM_PROMPT,
    model=_CHATBOT_MODEL,
    tools=[search_tools, count_tools],
)

_intent_agent = Agent(
    name="IntentAgent",
    instructions=_INTENT_PROMPT,
    model=_COUNCIL_MODEL,
)

_search_agent = Agent(
    name="SearchAgent",
    instructions=_SEARCH_PROMPT,
    model=_COUNCIL_MODEL,
    tools=[search_tools, count_tools],
)

_evaluator_agent = Agent(
    name="EvaluatorAgent",
    instructions=_EVALUATOR_PROMPT,
    model=_COUNCIL_MODEL,
)

_skeptic_agent = Agent(
    name="SkepticAgent",
    instructions=_SKEPTIC_PROMPT,
    model=_COUNCIL_MODEL,
)

_writer_agent = Agent(
    name="WriterAgent",
    instructions=_WRITER_PROMPT,
    model=_COUNCIL_MODEL,
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
        intent_result = _run_agent_with_annotation(
            _intent_agent,
            "devtools-council-intent",
            _INTENT_PROMPT,
            user_message,
            "intent",
            task_id,
            session_id,
            _COUNCIL_MAX_TURNS,
        )
        intent_text = _result_text(intent_result)
        queries = _queries_from_intent(intent_text, user_message)

        search_result = _run_agent_with_annotation(
            _search_agent,
            "devtools-council-search",
            _SEARCH_PROMPT,
            _build_search_input(user_message, intent_text, queries),
            "search",
            task_id,
            session_id,
            _COUNCIL_MAX_TURNS,
        )
        candidates = _collect_tools(search_result)
        if not candidates:
            candidates = _manual_search_candidates(queries)
        candidate_json = _candidate_payload(candidates)

        evaluation_result = _run_agent_with_annotation(
            _evaluator_agent,
            "devtools-council-evaluator",
            _EVALUATOR_PROMPT,
            _build_review_input(user_message, intent_text, candidate_json),
            "evaluator",
            task_id,
            session_id,
            _COUNCIL_MAX_TURNS,
        )
        evaluation_text = _result_text(evaluation_result)

        skeptic_result = _run_agent_with_annotation(
            _skeptic_agent,
            "devtools-council-skeptic",
            _SKEPTIC_PROMPT,
            _build_review_input(user_message, intent_text, candidate_json, evaluation_text),
            "skeptic",
            task_id,
            session_id,
            _COUNCIL_MAX_TURNS,
        )
        skeptic_text = _result_text(skeptic_result)
        selected_tools = _select_response_tools(candidates, evaluation_text, skeptic_text)
        selected_json = _candidate_payload(selected_tools)

        writer_result = _run_agent_with_annotation(
            _writer_agent,
            "devtools-council-writer",
            _WRITER_PROMPT,
            _build_writer_input(user_message, selected_json, evaluation_text, skeptic_text),
            "writer",
            task_id,
            session_id,
            _COUNCIL_MAX_TURNS,
        )
        response_text = _result_text(writer_result) or _fallback_council_response(
            user_message,
            selected_tools,
        )

        logger.info(
            "chatbot.council.response",
            extra={
                "event": "chatbot.council.response",
                "task_id": task_id,
                "message_length": len(user_message),
                "candidate_count": len(candidates),
                "tools_found": len(selected_tools),
                "response_length": len(response_text),
                "model": _COUNCIL_MODEL,
            },
        )
        return {
            "response": response_text,
            "tools": selected_tools,
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
            tags = {"workflow.name": "single_agent", "agent.role": "assistant"}
            if session_id:
                tags["session_id"] = session_id
            _annotate_current_span(tags)

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
