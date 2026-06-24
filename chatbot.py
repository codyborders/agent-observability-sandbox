"""OpenAI Agents SDK chatbot for natural language developer tool recommendations."""

import json
import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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
_SEARCH_STOPWORDS = {
    "and",
    "for",
    "from",
    "recommend",
    "run",
    "running",
    "the",
    "tool",
    "tools",
    "with",
}

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
Extract the user's goal, constraints, and database search strategy.
Return only JSON with these keys:
- summary: one sentence describing the user's goal
- criteria: 3 to 6 short strings that define a good match
- search_queries: 3 to 5 concise search queries for the local developer-tools database
Do not recommend tools. Do not add markdown.
"""

_SEARCH_PROMPT = """\
You are SearchAgent in a developer-tools recommendation council.
The application has already searched the local developer-tools database for you.
Review the provided search queries and candidate tools, then summarize what kinds of candidates were found.
Never invent tools that are not in the candidate JSON.
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


@contextmanager
def _llmobs_span(span_kind: str, name: str, session_id: str | None) -> Iterator[Any | None]:
    """Open an LLMObs span while allowing older tracers to skip custom spans."""
    if not span_kind.strip():
        raise ValueError("span_kind must not be blank")
    if not name.strip():
        raise ValueError("name must not be blank")

    span_factory = getattr(LLMObs, span_kind, None)
    if span_factory is None:
        yield None
        return
    try:
        span_context = span_factory(name=name, session_id=session_id)
    except Exception:
        logger.debug(
            "chatbot.llmobs_span_skipped",
            extra={"event": "chatbot.llmobs_span_skipped", "span_kind": span_kind, "span_name": name},
        )
        yield None
        return

    with span_context as span:
        yield span


def _annotate_llmobs_span(
    span: Any | None,
    *,
    input_data: Any | None = None,
    output_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Attach input, output, metadata, and tags to an LLMObs span."""
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    if tags is not None and not isinstance(tags, dict):
        raise TypeError("tags must be a dictionary")
    if span is None:
        return

    payload: dict[str, Any] = {}
    if input_data is not None:
        payload["input_data"] = input_data
    if output_data is not None:
        payload["output_data"] = output_data
    if metadata is not None:
        payload["metadata"] = metadata
    if tags is not None:
        payload["tags"] = tags
    if not payload:
        return

    annotate = getattr(LLMObs, "annotate", None)
    if annotate is None:
        return
    try:
        annotate(span=span, **payload)
    except Exception:
        logger.debug("chatbot.annotation_skipped", extra={"event": "chatbot.annotation_skipped"})


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


def _council_tags(task_id: str, role: str | None = None) -> dict[str, str]:
    """Build common Datadog tags for one recommendation council trace."""
    if not task_id.strip():
        raise ValueError("task_id must not be blank")
    if role is not None and not role.strip():
        raise ValueError("role must not be blank")

    tags = {
        "workflow.name": _WORKFLOW_NAME,
        "task.id": task_id,
        "model.name": _COUNCIL_MODEL,
    }
    if role is not None:
        tags["agent.role"] = role
    return tags


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


def _search_terms(query: str) -> list[str]:
    """Extract bounded fallback search terms from a sanitized query."""
    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return []

    terms: list[str] = []
    for raw_term in sanitized.split():
        term = raw_term.lower()
        if len(term) < 3 or term in _SEARCH_STOPWORDS:
            continue
        for candidate in (term, term[:-1] if term.endswith("s") else ""):
            if candidate and candidate not in terms:
                terms.append(candidate)
    return terms[:6]


def _manual_search_candidates(queries: list[str]) -> list[dict[str, Any]]:
    """Search directly with full queries, then bounded term fallbacks for recall."""
    if not isinstance(queries, list):
        raise TypeError("queries must be a list")
    if not queries:
        return []

    candidates: list[dict[str, Any]] = []
    for query in queries:
        query_text = str(query)
        results = _search_database(query_text)
        candidates.extend(results)
        if results:
            continue
        for term in _search_terms(query_text):
            candidates.extend(_search_database(term))
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


def _retrieval_documents(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format local search results for Datadog retrieval span output."""
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list")

    documents: list[dict[str, Any]] = []
    for candidate in candidates[:_MAX_TOOLS_IN_CONTEXT]:
        if not isinstance(candidate, dict):
            continue
        candidate_id = candidate.get("id")
        documents.append(
            {
                "id": str(candidate_id) if candidate_id is not None else "unknown",
                "name": str(candidate.get("name") or "Unknown tool")[:120],
                "text": str(candidate.get("description") or "")[:500],
                "score": 1.0,
            }
        )
    return documents


def _retrieve_candidate_tools(
    queries: list[str],
    session_id: str | None,
    task_id: str,
) -> list[dict[str, Any]]:
    """Search local tools inside a Datadog retrieval span."""
    if not queries:
        raise ValueError("queries must not be empty")
    if not task_id.strip():
        raise ValueError("task_id must not be blank")

    with _llmobs_span("retrieval", "candidate_database_search", session_id) as retrieval_span:
        candidates = _manual_search_candidates(queries)
        _annotate_llmobs_span(
            retrieval_span,
            input_data={"queries": queries},
            output_data=_retrieval_documents(candidates),
            metadata={"candidate_count": len(candidates), "task_id": task_id},
            tags=_council_tags(task_id),
        )
        return candidates


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

    skeptic_review = _parse_json_object(skeptic_text)
    if "approved_ids" in skeptic_review:
        ordered_ids = _ids_from_review(skeptic_text, "approved_ids")
        return [
            candidate_by_id[tool_id]
            for tool_id in ordered_ids
            if tool_id in candidate_by_id
        ][:5]

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


def _agent_span_name(agent: Agent, role: str) -> str:
    """Return a stable display name for a council agent span."""
    if role not in _COUNCIL_ROLES:
        raise ValueError(f"unsupported council role: {role}")
    agent_name = getattr(agent, "name", None)
    if isinstance(agent_name, str) and agent_name.strip():
        return agent_name.strip()
    return f"{role.title()}Agent"


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
    """Run one council agent inside a Datadog agent span."""
    if role not in _COUNCIL_ROLES:
        raise ValueError(f"unsupported council role: {role}")
    if not agent_input.strip():
        raise ValueError("agent_input must not be blank")

    tags = _council_tags(task_id, role)
    if session_id:
        tags["session_id"] = session_id

    with _llmobs_span("agent", _agent_span_name(agent, role), session_id) as agent_span:
        with LLMObs.annotation_context(prompt=_get_prompt_annotation(prompt_id, prompt_template)):
            result = Runner.run_sync(agent, input=agent_input, max_turns=max_turns)
            _annotate_current_span(tags)

        result_text = _result_text(result)
        _annotate_llmobs_span(
            agent_span,
            input_data=agent_input,
            output_data=result_text,
            metadata={"prompt_id": prompt_id, "max_turns": max_turns, "model": _COUNCIL_MODEL},
            tags=tags,
        )
        return result


def _build_search_input(
    user_message: str,
    intent_text: str,
    queries: list[str],
    candidate_json: str,
) -> str:
    """Build SearchAgent input from deterministic database search results."""
    if not user_message.strip():
        raise ValueError("user_message must not be blank")
    if not queries:
        raise ValueError("queries must not be empty")
    if not candidate_json.strip():
        raise ValueError("candidate_json must not be blank")
    return (
        f"User request:\n{user_message}\n\n"
        f"IntentAgent output:\n{intent_text}\n\n"
        f"Search queries used:\n{json.dumps(queries, ensure_ascii=False)}\n\n"
        f"Candidate tools JSON:\n{candidate_json}"
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


@dataclass(frozen=True)
class _CouncilRun:
    """Result of one recommendation council execution."""

    response_text: str
    selected_tools: list[dict[str, Any]]
    candidate_count: int


def _run_council_agents(
    user_message: str,
    session_id: str | None,
    task_id: str,
) -> _CouncilRun:
    """Execute council agents in order and return final response data."""
    if not user_message.strip():
        raise ValueError("user_message must not be blank")
    if not task_id.strip():
        raise ValueError("task_id must not be blank")

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

    candidates = _retrieve_candidate_tools(queries, session_id, task_id)
    candidate_json = _candidate_payload(candidates)
    _run_agent_with_annotation(
        _search_agent,
        "devtools-council-search",
        _SEARCH_PROMPT,
        _build_search_input(user_message, intent_text, queries, candidate_json),
        "search",
        task_id,
        session_id,
        _COUNCIL_MAX_TURNS,
    )

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

    return _CouncilRun(
        response_text=response_text,
        selected_tools=selected_tools,
        candidate_count=len(candidates),
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
        with _llmobs_span("workflow", _WORKFLOW_NAME, session_id) as workflow_span:
            workflow_tags = _council_tags(task_id)
            if session_id:
                workflow_tags["session_id"] = session_id
            _annotate_llmobs_span(
                workflow_span,
                input_data=user_message,
                metadata={
                    "task_id": task_id,
                    "model": _COUNCIL_MODEL,
                    "agents": _COUNCIL_AGENT_NAMES,
                },
                tags=workflow_tags,
            )

            council_run = _run_council_agents(user_message, session_id, task_id)
            _annotate_llmobs_span(
                workflow_span,
                output_data=council_run.response_text,
                metadata={
                    "task_id": task_id,
                    "candidate_count": council_run.candidate_count,
                    "tools_found": len(council_run.selected_tools),
                    "response_length": len(council_run.response_text),
                    "model": _COUNCIL_MODEL,
                },
                tags=workflow_tags,
            )
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
