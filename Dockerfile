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

# Streamlit UI calls the local FastAPI backend
ENV STEUER_RAG_API_URL=http://localhost:8000

# HF Spaces exposes port 7860
EXPOSE 7860

# Ingest runs at startup (not build time — crawl is too slow for build timeout).
# Streamlit starts immediately so port 7860 is live; the API becomes available
# once ingest + uvicorn finish (~30-50 min after first start).
CMD bash -c "(python -m steuer_rag.cli.main ingest all && uvicorn steuer_rag.api:app --host 0.0.0.0 --port 8000) & streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true"
