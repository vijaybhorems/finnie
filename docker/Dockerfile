FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["python", "-m", "streamlit", "run", "src/web_app/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.fileWatcherType=none", \
    "--browser.gatherUsageStats=false"]
