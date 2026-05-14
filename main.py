"""Top-level entrypoint. Delegates to the Typer CLI in `steuer_rag.cli.main`."""

from steuer_rag.cli.main import app

if __name__ == "__main__":
    app()
