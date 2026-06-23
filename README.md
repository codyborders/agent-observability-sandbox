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

## Datadog signals to look for

Use your Datadog account to check the `agent-observability-sandbox` service in the `sandbox` environment unless you changed those tags. Chatbot and classifier calls should appear under the LLM Observability or Agent Observability app named by `DD_LLMOBS_ML_APP`.

The stack can emit Flask request traces, container logs, runtime metrics, profiler data, LLM spans, and optional RUM sessions.

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
