"""CLI: `steuer-rsb <command>`.

Commands:
- `ingest <source>` — crawl + chunk + embed + index for one source (or `all`)
- `search <query>`  — print top-k retrieval hits
- `ask <question>`  — full RAG: retrieve + LLM-answer with citations
- `serve`           — launch FastAPI server
- `info`            — show index size + settings
"""

from __future__ import annotations

import json
import logging

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from steuer_rag.config import get_settings
from steuer_rag.pipeline.ingest import ingest_all_sync, ingest_source_sync
from steuer_rag.pipeline.index import get_index
from steuer_rag.retrieval.search import search as run_search
from steuer_rag.schema.models import SourceName

app = typer.Typer(
    name="steuer-rsb",
    help="Bilingual (DE/EN) RAG for the German Steuererklärung.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _configure_logging(level: str | None = None) -> None:
    s = get_settings()
    logging.basicConfig(
        level=(level or s.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    _configure_logging("DEBUG" if verbose else None)


@app.command()
def ingest(
    source: str = typer.Argument(
        "all",
        help="Source name or 'all'. Options: bundesfinanzministerium | elster | bzst | all",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Max documents to fetch per source."),
) -> None:
    """Crawl source(s), chunk, embed, and write to the vector index."""
    if source == "all":
        results = ingest_all_sync(limit=limit)
    else:
        results = [ingest_source_sync(source, limit=limit)]
    table = Table(title="Ingest results")
    table.add_column("Source")
    table.add_column("Docs", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Status")
    for r in results:
        if "error" in r:
            table.add_row(r["source"], "-", "-", f"[red]{r['error']}[/red]")
        else:
            table.add_row(r["source"], str(r["docs"]), str(r["chunks"]), "[green]ok[/green]")
    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(...),
    k: int = typer.Option(5, "--k", "-k"),
    strategy: str = typer.Option("hybrid_rerank", "--strategy"),
    source: str | None = typer.Option(None, "--source"),
    language: str | None = typer.Option(None, "--lang"),
) -> None:
    """Retrieve top-k chunks for `query` and pretty-print them."""
    result = run_search(query, k=k, strategy=strategy, source=source, language=language)
    table = Table(title=f"Top {len(result.documents)} ({result.strategy}, lang={result.query_language.value})")
    table.add_column("#", justify="right")
    table.add_column("Source")
    table.add_column("Lang")
    table.add_column("Title")
    table.add_column("Snippet", max_width=80)
    for i, d in enumerate(result.documents, 1):
        m = d.metadata or {}
        snippet = (d.page_content or "").strip().replace("\n", " ")[:160]
        table.add_row(
            str(i),
            m.get("source", "-"),
            m.get("language", "-"),
            (m.get("doc_title") or "")[:50],
            snippet + "…" if len(snippet) == 160 else snippet,
        )
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(...),
    k: int = typer.Option(8, "--k", "-k"),
    source: str | None = typer.Option(None, "--source"),
    language: str | None = typer.Option(None, "--lang"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Ask a question and stream the cited answer."""
    from steuer_rag.generation.chain import ask as run_ask

    result = run_ask(question, k=k, source=source, language=language)
    if json_output:
        payload = {
            "question": result.question,
            "answer": result.answer,
            "language": result.language.value,
            "citations": [c.__dict__ for c in result.citations],
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    console.rule(f"[bold]Answer ({result.language.value})[/bold]")
    console.print(Markdown(result.answer))
    console.rule("[bold]Citations[/bold]")
    for c in result.citations:
        score = f" (rerank={c.rerank_score:.2f})" if c.rerank_score is not None else ""
        console.print(f"[{c.n}] [cyan]{c.source}[/cyan] — {c.title}{score}\n    {c.url}")


@app.command()
def info() -> None:
    """Print index stats + active settings."""
    s = get_settings()
    idx = get_index()
    table = Table(title="steuer-rsb")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("vector_backend", s.vector_backend)
    table.add_row("collection", s.collection)
    table.add_row("chroma_dir", str(s.chroma_dir))
    table.add_row("embed_model", s.embed_model)
    table.add_row("rerank_model", s.rerank_model)
    table.add_row("llm_provider", s.llm_provider)
    table.add_row("llm_model", s.llm_model)
    table.add_row("indexed_chunks", str(idx.count()))
    table.add_row("sources", ", ".join(s.value for s in SourceName))
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("steuer_rag.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
