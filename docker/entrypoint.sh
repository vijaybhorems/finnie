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
# src.web_app.serve starts loading LangGraph + the embedding model in the
# background, then runs Streamlit in the same process, so the sign-in page is
# served at once and the heavy imports overlap boot and the user's sign-in.
exec python -m src.web_app.serve \
  --server.port="${PORT:-8501}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.fileWatcherType=none \
  --browser.gatherUsageStats=false
