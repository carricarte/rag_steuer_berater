FROM python:3.12-slim

# HF Spaces requires a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Copy project files
COPY --chown=user pyproject.toml .
COPY --chown=user steuer_rag/ steuer_rag/
COPY --chown=user app.py .
# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Pre-populate the vector index during build (web crawl → embed → index)
RUN python -m steuer_rag.cli.main ingest all

# Streamlit UI calls the local FastAPI backend
ENV STEUER_RAG_API_URL=http://localhost:8000

# HF Spaces exposes port 7860
EXPOSE 7860

# Start FastAPI backend in background, then Streamlit in foreground
CMD uvicorn steuer_rag.api:app --host 0.0.0.0 --port 8000 & \
    streamlit run app.py \
      --server.port 7860 \
      --server.address 0.0.0.0 \
      --server.headless true
