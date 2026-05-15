#!/usr/bin/env bash
# Render build script: install dependencies then populate the vector index.
# Safe to re-run: ingest is idempotent (content-addressable chunk IDs).
set -euo pipefail

echo "==> Installing package..."
pip install -e .

echo "==> Running ingest (crawl → embed → index)..."
python -m steuer_rag.cli.main ingest all

echo "==> Build complete. Indexed chunks:"
python -m steuer_rag.cli.main info
