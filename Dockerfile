FROM python:3.12-slim

# Build tools needed for fasttext (C++17) and other compiled wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Copy project files
COPY --chown=user pyproject.toml .
COPY --chown=user README.md .
COPY --chown=user steuer_rag/ steuer_rag/
COPY --chown=user app.py .
# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Pre-download model weights into the image so the first /ask request doesn't time out.
# bge-m3 ~570 MB + bge-reranker-v2-m3 ~570 MB = ~1.1 GB added to image.
RUN python -c "from steuer_rag.pipeline.embed import get_embeddings; get_embeddings(); print('embed model cached')"
RUN python -c "from steuer_rag.retrieval.reranker import _load_cross_encoder; _load_cross_encoder(); print('reranker cached')"

# Streamlit UI calls the local FastAPI backend
ENV STEUER_RAG_API_URL=http://localhost:8000
ENV PYTHONUNBUFFERED=1

# HF Spaces exposes port 7860
EXPOSE 7860

# startup restores the Chroma snapshot from HF Dataset (fast) or runs full ingest + uploads snapshot.
# uvicorn and streamlit start immediately in parallel so the UI is live while data loads.
CMD bash -c "python -m steuer_rag.cli.main startup & uvicorn steuer_rag.api:app --host 0.0.0.0 --port 8000 & exec streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true"
