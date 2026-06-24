# Sandbox Guide

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

The sandbox ships with `seed/startups.db.gz`, a sanitized copy of the full DevTools Scrape corpus. On first boot, the entrypoint restores it to `data/startups.db` when the target file is absent or empty.

Existing nonempty databases are preserved. This lets you experiment with data changes and restart containers without losing work.

To reset the database:

```bash
docker compose down
python scripts/restore_sandbox_db.py --force
docker compose up --build
```

To regenerate the fixture from another SQLite database, sanitize the source copy first, then package the sanitized database:

```bash
python scripts/sanitize_sandbox_db.py path/to/startups.db /tmp/startups-sanitized.db
python scripts/create_sandbox_seed.py /tmp/startups-sanitized.db --output seed/startups.db.gz
```

The sanitizer strips Product Hunt API tracking query strings and redacts email addresses plus secret-looking tokens. The seed tool then validates SQLite integrity, schema shape, public columns, row count, and secret-looking text before writing the compressed fixture.

## Datadog account isolation

Each teammate should use their own Datadog account or sandbox organization. The `.env` file controls where telemetry goes.

Important variables:

- `DATADOG_API_KEY` authenticates the Agent.
- `DD_SITE` selects the Datadog region.
- `DD_ENV` defaults to `sandbox`.
- `DD_SERVICE` defaults to `agent-observability-sandbox`.
- `DD_LLMOBS_ML_APP` defaults to `agent-observability-sandbox`.

Change these tags when you want to separate experiments. For example, set `DD_SERVICE=agent-observability-sandbox-yourname` before comparing traces with a teammate.

## Datadog features enabled by default

After `docker compose up --build`, the app runs with Datadog Agent 7 and Python `ddtrace` enabled. These features are on by default:

| Feature | Default setting | What to expect |
| --- | --- | --- |
| APM tracing | `DD_APM_ENABLED=true`, `DD_TRACE_ENABLED=true` | Flask request traces for page loads and API calls. |
| Container logs | `DD_LOGS_ENABLED=true`, `DD_LOGS_CONFIG_CONTAINER_COLLECT_ALL=true` | App container logs collected by the Datadog Agent. |
| Runtime metrics | `DD_RUNTIME_METRICS_ENABLED=true` | Python runtime metrics attached to the configured service and environment. |
| Profiling | `DD_PROFILING_ENABLED=true` | CPU, timeline, and memory profiling data from the app process. |
| DogStatsD | `DD_DOGSTATSD_URL=udp://dd-agent:8125` | Custom metric traffic can flow from the app container to the Agent. |
| Exception Replay | `DD_EXCEPTION_REPLAY_ENABLED=true` | Captured exception context for supported traced errors. |
| Code origin metadata | `DD_CODE_ORIGIN_FOR_SPANS_ENABLED=true` | Span metadata can point back to source locations. |
| LLM Observability | `DD_LLMOBS_ENABLED=1` | Chatbot and classifier OpenAI calls create LLM spans under `DD_LLMOBS_ML_APP`. |
| Agent Observability graph | `CHATBOT_WORKFLOW=council` | Browser chat runs the OpenAI Agents SDK Recommendation Council with handoff spans and a `search_tools` tool call. |
| Prompt Management fallback path | Built into `ai_classifier.py` and `chatbot.py` | Managed prompts are used when Datadog returns them; local fallback prompts keep the app running otherwise. |

RUM is configured but inactive until `DATADOG_RUM_APPLICATION_ID` and `DATADOG_RUM_CLIENT_TOKEN` are set.

These Datadog features are disabled by default for a lower-risk sandbox start: Dynamic Instrumentation, Live Debugging, AppSec, IAST, and the Process Agent.

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

## Project ideas

See [`project_ideas`](../project_ideas/) for build ideas that extend this sandbox. The directory covers evaluations, Datadog Experiments, datasets, Prompt Management, Prompt Tracking, and Prompt Optimization in the context of the Recommendation Council.

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
