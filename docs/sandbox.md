# Sandbox Guide

This repository is a self-contained sandbox for DevTools Scrape. Use it to explore the app, change features, and learn Datadog telemetry without touching the production repository.

## Setup

Copy the environment template and fill in personal credentials:

```bash
cp .env.sandbox.example .env
```

Required values:

- `OPENAI_API_KEY` from your OpenAI account.
- `DATADOG_API_KEY` from your Datadog account.
- `DD_SITE` for your Datadog account region.

Start the stack:

```bash
docker compose up --build
```

Open `http://localhost:8000` and check `http://localhost:8000/health`.

## Database seed

The sandbox ships with `seed/startups.db.gz`. On first boot, the entrypoint restores it to `data/startups.db` when the target file is absent or empty.

Existing nonempty databases are preserved. This lets you experiment with data changes and restart containers without losing work.

To reset the database:

```bash
docker compose down
python scripts/restore_sandbox_db.py --force
docker compose up --build
```

To regenerate the fixture from another SQLite database:

```bash
python scripts/create_sandbox_seed.py path/to/startups.db --output seed/startups.db.gz
```

The seed tool validates SQLite integrity, schema shape, public columns, row count, and secret-looking text before writing the compressed fixture.

## Datadog account isolation

Each teammate should use their own Datadog account or sandbox organization. The `.env` file controls where telemetry goes.

Important variables:

- `DATADOG_API_KEY` authenticates the Agent.
- `DD_SITE` selects the Datadog region.
- `DD_ENV` defaults to `sandbox`.
- `DD_SERVICE` defaults to `agent-observability-sandbox`.
- `DD_LLMOBS_ML_APP` defaults to `agent-observability-sandbox`.

Change these tags when you want to separate experiments. For example, set `DD_SERVICE=agent-observability-sandbox-yourname` before comparing traces with a teammate.

## Expected telemetry

After `docker compose up --build`, Datadog should receive:

- Flask request traces for page loads and API calls.
- Container logs from the app service.
- Runtime metrics from the Python tracer.
- Profiling data when profiling is enabled.
- LLM Observability or Agent Observability spans when the chatbot or classifier calls OpenAI.

Run a chat request from the browser to create an LLM span. You can also call the API directly:

```bash
curl -s http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Find developer tools for testing APIs"}'
```

## Optional RUM setup

Create a Datadog RUM Browser Application in your own Datadog account. Copy its application ID and client token into `.env`:

```bash
DATADOG_RUM_APPLICATION_ID=your-application-id
DATADOG_RUM_CLIENT_TOKEN=your-client-token
DATADOG_RUM_SITE=datadoghq.com
```

Restart the stack. Browser sessions should appear in RUM for the service and environment configured in `.env`.

The app automatically adds the current request host to RUM allowed tracing URLs unless `DATADOG_RUM_ALLOWED_TRACING_URLS` is set. This helps local browser requests correlate with backend traces.

## Manual traced scrapes

Scheduled scraping is disabled by default with `SCRAPER_CRON_ENABLED=false`. This prevents unexpected API usage and keeps the seeded database stable.

Run a traced scrape manually when you want scraper spans:

```bash
docker compose exec app ddtrace-run python scrape_all.py
```

If you want Product Hunt API coverage, add `PRODUCTHUNT_CLIENT_ID` and `PRODUCTHUNT_CLIENT_SECRET` to `.env` first.

## Troubleshooting

### No traces in Datadog

Check that `DATADOG_API_KEY` is set, `DD_SITE` matches your Datadog account, and the app uses `DD_AGENT_HOST=dd-agent`.

Inspect containers:

```bash
docker compose ps
docker compose logs app
docker compose logs dd-agent
```

### Logs missing

Confirm these values in `.env`:

```bash
DD_LOGS_ENABLED=true
DD_LOGS_CONFIG_CONTAINER_COLLECT_ALL=true
```

Then restart the stack.

### LLM spans missing

Check these values:

```bash
OPENAI_API_KEY=...
DD_LLMOBS_ENABLED=1
DD_LLMOBS_ML_APP=agent-observability-sandbox
```

Generate a chat request after the app starts. LLM spans are only created when code calls an LLM provider.

### RUM sessions missing

Check that `DATADOG_RUM_APPLICATION_ID` and `DATADOG_RUM_CLIENT_TOKEN` are set. Reload the browser page after restarting the app.

View page source and confirm a Datadog browser agent script appears in the `<head>` section.

### Database empty

Reset from the seed:

```bash
python scripts/restore_sandbox_db.py --force
```

If this fails, validate the fixture:

```bash
python scripts/create_sandbox_seed.py data/startups.db --output /tmp/startups.db.gz
```

### Wrong Datadog region

A valid API key for one site will not send data to another site. Match `DD_SITE` to the site where you created the API key. Common values include `datadoghq.com`, `datadoghq.eu`, `us3.datadoghq.com`, `us5.datadoghq.com`, and `ap1.datadoghq.com`.

## No-emoji rule

Do not add emoji to docs, code comments, scripts, log messages, examples, commit messages, or generated project output.

Run the check:

```bash
python scripts/check-no-emoji.py
```

The pytest suite also runs this check.
