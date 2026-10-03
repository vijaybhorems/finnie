#!/bin/sh
# Container entrypoint: bring the database up to date, then serve.
#
# With DATABASE_URL set, migrations and the knowledge-base sync run before
# Streamlit listens, so a new Cloud Run revision only takes traffic once its
# database is ready. If they fail, the container exits non-zero, the revision
# never becomes ready, and Cloud Run keeps serving the previous one.
set -eu

if [ -n "${DATABASE_URL:-}" ]; then
  python -m src.persistence.migrate
fi

# Cloud Run sets PORT (8501 with --port=8501); default for plain `docker run`.
# src.web_app.serve loads LangGraph + the embedding model before Streamlit opens
# the port, so the startup probe only passes once the instance is warm and no
# visitor is routed to a half-loaded one. Set FINNIE_WARM_BEFORE_SERVE=0 to open
# the port at once and warm up in the background instead.
export FINNIE_WARM_BEFORE_SERVE="${FINNIE_WARM_BEFORE_SERVE:-1}"
exec python -m src.web_app.serve \
  --server.port="${PORT:-8501}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.fileWatcherType=none \
  --browser.gatherUsageStats=false
