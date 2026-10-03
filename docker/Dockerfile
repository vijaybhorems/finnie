# --- Build stage: compile/install dependencies into a venv -------------------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch first. From PyPI, sentence-transformers pulls the CUDA build and
# ~2.5 GB of nvidia-* libraries that Cloud Run (no GPU) never loads, but every
# new instance still pulls them and pays for the bigger import. With torch
# already satisfied, the requirements install below keeps this one.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && if pip list 2>/dev/null | grep -qi '^nvidia-'; then \
      echo "CUDA libraries were installed; a dependency replaced the CPU-only torch" >&2; exit 1; \
    fi

# --- Runtime stage: no compilers --------------------------------------------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Streamlit's index.html says "Streamlit" with Streamlit's favicon until app.py's
# set_page_config runs — seconds on a cold start, and always for link previews.
# Brand the static page with the same title and icon. The grep fails the build
# if a Streamlit upgrade changes the <title> line, rather than silently reverting.
COPY src/web_app/assets/favicon.png /tmp/finnie-favicon.png
RUN STATIC="$(python -c 'import os, streamlit; print(os.path.join(os.path.dirname(streamlit.__file__), "static"))')" \
 && grep -q '<title>Streamlit</title>' "$STATIC/index.html" \
 && sed -i 's|<title>Streamlit</title>|<title>Finnie — AI Finance Assistant</title>|' "$STATIC/index.html" \
 && cp /tmp/finnie-favicon.png "$STATIC/favicon.png" \
 && rm /tmp/finnie-favicon.png

# Pre-download the sentence-transformers embedding model so it is baked into the
# image — the runtime model load then costs no network round-trip to the HF Hub.
# Keep EMBEDDING_MODEL in sync with `embeddings.model` in config.yaml.
ENV HF_HOME=/app/.cache/huggingface
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# Load the embedding model fully offline at runtime (it is already in the image).
# Placed AFTER the download above so the build itself can still reach the Hub.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy application code
COPY . .

# Bake the FAISS index into the image so the first chat query doesn't rebuild it
# on a cold-started container (e.g. Cloud Run). Runs offline — the embedding model
# is already in the image (above). Docker Compose mounts a volume over
# data/faiss_index and its rag_indexer service rebuilds there, so this is harmless
# there and load-bearing for single-container deploys.
RUN python -c "from src.rag.indexer import RAGIndexer; RAGIndexer().build_index()"

# Suppress noisy transformers lazy-import warnings (torchvision not needed)
ENV TRANSFORMERS_VERBOSITY=error
ENV TOKENIZERS_PARALLELISM=false
# Env var form — most reliable way to disable Streamlit file watcher
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Copy Streamlit config (belt-and-suspenders with env var + config file)
COPY .streamlit/config.toml /app/.streamlit/config.toml

EXPOSE 8501

# start-period covers the warm-up, which runs before the port opens (entrypoint.sh).
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8501}/_stcore/health" || exit 1

# Migrate (when DATABASE_URL is set), then start Streamlit on $PORT.
RUN chmod +x docker/entrypoint.sh
ENTRYPOINT ["docker/entrypoint.sh"]
