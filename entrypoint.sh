#!/bin/bash
set -euo pipefail

truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

DATA_DIR="${DEVTOOLS_DATA_DIR:-/app/data}"
DB_FILE="${DEVTOOLS_DB_PATH:-${DATA_DIR%/}/startups.db}"
LEGACY_DB="/app/startups.db"
SEED_FIXTURE="${SANDBOX_SEED_FIXTURE:-/app/seed/startups.db.gz}"

mkdir -p "$(dirname "$DB_FILE")"

if truthy "${SANDBOX_RESTORE_SEED:-true}"; then
    if [ -f "$SEED_FIXTURE" ]; then
        python3 /app/scripts/restore_sandbox_db.py --fixture "$SEED_FIXTURE" --target "$DB_FILE"
    else
        echo "Seed fixture not found at $SEED_FIXTURE; continuing without restore."
    fi
else
    echo "Sandbox seed restore disabled."
fi

# If an old root-level database exists and the data volume is empty, migrate it.
if [ -f "$LEGACY_DB" ] && [ ! -L "$LEGACY_DB" ] && [ ! -s "$DB_FILE" ]; then
    cp "$LEGACY_DB" "$DB_FILE"
    echo "Migrated legacy database to $DB_FILE."
fi

# Ensure the primary database file exists so SQLite commands do not fail on start.
if [ ! -f "$DB_FILE" ]; then
    touch "$DB_FILE"
    echo "Created empty database file at $DB_FILE."
fi

# Keep legacy path in place for older code paths that reference startups.db directly.
ln -sf "$DB_FILE" "$LEGACY_DB"

PYTHON_BIN="$(command -v python3 || true)"
if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi
DDTRACE_BIN="$(command -v ddtrace-run || true)"
if [ -z "$DDTRACE_BIN" ]; then
    DDTRACE_BIN="ddtrace-run"
fi

APP_ENV="OPENAI_API_KEY=${OPENAI_API_KEY:-} PRODUCTHUNT_CLIENT_ID=${PRODUCTHUNT_CLIENT_ID:-} PRODUCTHUNT_CLIENT_SECRET=${PRODUCTHUNT_CLIENT_SECRET:-} DATADOG_API_KEY=${DATADOG_API_KEY:-} DD_API_KEY=${DATADOG_API_KEY:-} DD_SITE=${DD_SITE:-datadoghq.com}"
DD_ENV_VARS="DD_ENV=${DD_ENV:-sandbox} DD_SERVICE=${DD_SERVICE:-agent-observability-sandbox} DD_VERSION=${DD_VERSION:-sandbox} DD_AGENT_HOST=${DD_AGENT_HOST:-dd-agent} DD_TRACE_ENABLED=${DD_TRACE_ENABLED:-true} DD_APM_ENABLED=${DD_APM_ENABLED:-true} DD_RUNTIME_METRICS_ENABLED=${DD_RUNTIME_METRICS_ENABLED:-true} DD_DOGSTATSD_URL=${DD_DOGSTATSD_URL:-udp://dd-agent:8125} DD_APPSEC_ENABLED=${DD_APPSEC_ENABLED:-false} DD_IAST_ENABLED=${DD_IAST_ENABLED:-false} DD_IAST_REQUEST_SAMPLING=${DD_IAST_REQUEST_SAMPLING:-100} DD_LLMOBS_ENABLED=${DD_LLMOBS_ENABLED:-1} DD_LLMOBS_ML_APP=${DD_LLMOBS_ML_APP:-agent-observability-sandbox} DD_CODE_ORIGIN_FOR_SPANS_ENABLED=${DD_CODE_ORIGIN_FOR_SPANS_ENABLED:-true} DD_EXCEPTION_REPLAY_ENABLED=${DD_EXCEPTION_REPLAY_ENABLED:-true} DD_PROFILING_ENABLED=${DD_PROFILING_ENABLED:-true}"

if truthy "${SCRAPER_CRON_ENABLED:-false}"; then
    CRON_ENV="${APP_ENV} ${DD_ENV_VARS}"
    echo "0 */4 * * * cd /app && env ${CRON_ENV} ${DDTRACE_BIN} ${PYTHON_BIN} scrape_all.py >> /var/log/cron.log 2>&1" > /etc/cron.d/scrape_all
    chmod 0644 /etc/cron.d/scrape_all
    crontab /etc/cron.d/scrape_all
    cron
    echo "Scraper cron enabled."
else
    echo "Scraper cron disabled."
fi

exec "$DDTRACE_BIN" gunicorn -c gunicorn.conf.py app_production:app
