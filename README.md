# Agent Observability Sandbox

Agent Observability Sandbox is a local teaching repo for the DevTools Scrape app and Datadog telemetry. Use it to change features, trigger LLM calls, inspect traces, compare logs, and test optional RUM without connecting to the production deployment.

Your clone reports to your Datadog account. Put credentials in `.env`, change the service and environment tags as needed, and reset the local database whenever an experiment gets messy.

## Requirements

Install Podman Desktop with Compose support. You also need an OpenAI API key for chatbot and classifier calls, plus a Datadog API key from your own account. Set `DD_SITE` to your Datadog region, such as `datadoghq.com`, `datadoghq.eu`, `us3.datadoghq.com`, `us5.datadoghq.com`, or `ap1.datadoghq.com`.

## Quick start

```bash
cp .env.sandbox.example .env
# Edit .env and add your OpenAI key, Datadog API key, and Datadog site.
podman compose up --build
```

Open `http://localhost:8000`.

Health check:

```bash
curl http://localhost:8000/health
```

On first boot, the entrypoint restores `data/startups.db` from `seed/startups.db.gz` when the local database is missing or empty. Nonempty local databases are left alone.

## Agents SDK workflow

The chat widget runs a multi-agent Recommendation Council by default. It uses the OpenAI Agents SDK to run five role-specific agents against the local developer-tools database:

- `IntentAgent` turns the user's request into match criteria plus database queries.
- `SearchAgent` calls `search_tools` against the local SQLite data.
- `EvaluatorAgent` ranks candidates against the user's criteria.
- `SkepticAgent` removes weak or unsupported matches.
- `WriterAgent` turns approved candidates into the final response.

The workflow creates a Datadog Agent Observability workflow span with nested agent spans for each council role and a retrieval span for the local database search. Spans are tagged with `workflow.name`, `agent.role`, `task.id`, and `model.name` so the Datadog trace graph can show the council flow. Set `CHATBOT_WORKFLOW=simple` to use the original single-agent assistant.

The default model is `gpt-5-nano`, selected because it is newer than the previous `gpt-4o-mini` default and priced for low-cost agent routing plus compact classification and summary work. OpenAI's model docs list it for Agents SDK use, and this council workflow mainly needs tool calls plus short synthesis. Override `CHATBOT_MODEL` or `CHATBOT_COUNCIL_MODEL` in `.env` to compare cost and latency against response quality.

## Datadog versions and preview features

The Python app pins `ddtrace==4.5.0rc1` in `requirements.txt`. The Dockerfile and local test command install that preview tracer from Datadog's build index at `https://dd-trace-py-builds.s3.amazonaws.com/96035140/index.html`.

This tracer version is used for Prompt Management preview paths in `ai_classifier.py` and `chatbot.py`. Both call `LLMObs.get_prompt(..., label="production", fallback=...)` and keep bundled prompt templates as a local fallback.

Classifier prompt IDs:

- `devtools-binary-classifier`
- `devtools-batch-classifier`
- `devtools-category-classifier`

Recommendation Council prompt IDs:

- `devtools-council-intent`
- `devtools-council-search`
- `devtools-council-evaluator`
- `devtools-council-skeptic`
- `devtools-council-writer`

When Prompt Management works, classifier and council calls are wrapped in `LLMObs.annotation_context(...)` so prompt metadata is attached to LLM Observability spans.

The Compose stack uses Datadog Agent 7 through `gcr.io/datadoghq/agent:7`. Browser RUM is optional; when RUM credentials are set, the app loads Datadog Browser RUM `v6` unless `DATADOG_RUM_BROWSER_VERSION` overrides it.

## Datadog signals to look for

Use your Datadog account to check the `agent-observability-sandbox` service in the `sandbox` environment unless you changed those tags. Chatbot and classifier calls should appear under the LLM Observability or Agent Observability app named by `DD_LLMOBS_ML_APP`.

The stack can emit Flask request traces, container logs, runtime metrics, profiler data, LLM spans, prompt annotations, and optional RUM sessions.

## Manual traced scrape

Scheduled scrapes are disabled by default to avoid surprise API usage and data changes. To run one scrape with tracing:

```bash
podman compose exec app ddtrace-run python scrape_all.py
```

## Reset the database

```bash
podman compose down
python scripts/restore_sandbox_db.py --force
podman compose up --build
```

## Tests

Create a virtual environment before running Python tooling:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --find-links=https://dd-trace-py-builds.s3.amazonaws.com/96035140/index.html -r requirements.txt
.venv/bin/python -m pytest tests/
```

Convenience wrapper:

```bash
bash scripts/run-tests.sh
```

More details are in `docs/sandbox.md`.
