"""RAG generation chain built with LCEL.

Flow:
    1. Detect query language → DE or EN
    2. Hybrid retrieve (dense + sparse) → cross-encoder rerank → top-k passages
    3. Build numbered context string (each passage tagged [n] with title + URL)
    4. Render bilingual prompt, call LLM
    5. Return answer + structured citations

Re-use: `build_rag_chain()` returns an LCEL `Runnable` so it composes with LangChain agents,
streaming, callbacks, and tracing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from steuer_rag.generation.llm import get_llm
from steuer_rag.generation.prompts import get_prompt
from steuer_rag.retrieval.search import HybridRetriever, build_retriever
from steuer_rag.schema.models import Language, SourceName, detect_language


@dataclass
class Citation:
    n: int
    source: str
    url: str
    title: str
    section: str | None = None
    rerank_score: float | None = None


@dataclass
class RAGAnswer:
    question: str
    answer: str
    language: Language
    citations: list[Citation] = field(default_factory=list)
    passages: list[Document] = field(default_factory=list)


def _format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        title = meta.get("doc_title") or meta.get("url") or "Quelle"
        url = meta.get("url", "")
        src = meta.get("source", "")
        snippet = d.page_content.strip()
        parts.append(f"[{i}] {title} ({src})\n{url}\n{snippet}")
    return "\n\n---\n\n".join(parts)


def _citations(docs: list[Document]) -> list[Citation]:
    out: list[Citation] = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata or {}
        out.append(
            Citation(
                n=i,
                source=m.get("source", ""),
                url=m.get("url", ""),
                title=m.get("doc_title", ""),
                section=m.get("section") or None,
                rerank_score=m.get("rerank_score"),
            )
        )
    return out


def build_rag_chain(
    *,
    k: int | None = None,
    source: str | SourceName | None = None,
    language: str | Language | None = None,
    strategy: str = "hybrid_rerank",
) -> Runnable:
    """Return an LCEL `Runnable` mapping `{"question": str}` → `RAGAnswer`."""

    def _retrieve(inputs: dict[str, Any]) -> dict[str, Any]:
        q: str = inputs["question"]
        lang = language or detect_language(q)
        retriever: HybridRetriever = build_retriever(
            k=k, strategy=strategy, source=source, language=lang  # type: ignore[arg-type]
        )
        result = retriever.invoke(q)
        return {
            "question": q,
            "docs": result.documents,
            "language": result.query_language.value,
        }

    def _prompt_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "question": inputs["question"],
            "context": _format_context(inputs["docs"]),
            "_language": inputs["language"],
            "_docs": inputs["docs"],
        }

    def _call_llm(inputs: dict[str, Any]) -> dict[str, Any]:
        prompt = get_prompt(inputs["_language"])
        chain = prompt | get_llm() | StrOutputParser()
        answer = chain.invoke(
            {"question": inputs["question"], "context": inputs["context"]}
        )
        return {
            "answer": answer,
            "question": inputs["question"],
            "language": inputs["_language"],
            "docs": inputs["_docs"],
        }

    def _wrap(inputs: dict[str, Any]) -> RAGAnswer:
        return RAGAnswer(
            question=inputs["question"],
            answer=inputs["answer"],
            language=Language(inputs["language"]),
            citations=_citations(inputs["docs"]),
            passages=inputs["docs"],
        )

    return (
        RunnablePassthrough()
        | RunnableLambda(_retrieve)
        | RunnableLambda(_prompt_inputs)
        | RunnableLambda(_call_llm)
        | RunnableLambda(_wrap)
    )


def ask(
    question: str,
    *,
    k: int | None = None,
    source: str | SourceName | None = None,
    language: str | Language | None = None,
    strategy: str = "hybrid_rerank",
) -> RAGAnswer:
    chain = build_rag_chain(k=k, source=source, language=language, strategy=strategy)
    return chain.invoke({"question": question})
