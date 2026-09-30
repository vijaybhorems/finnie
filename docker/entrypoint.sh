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
exec python -m streamlit run src/web_app/app.py \
  --server.port="${PORT:-8501}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.fileWatcherType=none \
  --browser.gatherUsageStats=false
