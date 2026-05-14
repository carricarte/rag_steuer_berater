"""Base scraper: polite, async, with robots.txt respect, retry/backoff, and on-disk caching."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
import urllib.robotparser
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Retry on transient errors only.

    - All transport-level errors (timeouts, connection resets) are retryable.
    - HTTP status errors: only 5xx and 429 (rate-limit) are retryable. 4xx (other than 429) is
      permanent — retrying just wastes time and risks getting IP-banned by polite portals.
    """
    if isinstance(exc, (httpx.TransportError, httpx.ReadTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    return False

from steuer_rag.config import get_settings
from steuer_rag.schema.models import DocumentCore, SourceName

log = logging.getLogger(__name__)


# ---------- robots ----------

_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def _robots_for(url: str, user_agent: str) -> urllib.robotparser.RobotFileParser:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base in _robots_cache:
        return _robots_cache[base]
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{base}/robots.txt")
    try:
        rp.read()
    except Exception as e:
        log.warning("robots.txt fetch failed for %s: %s — assuming allow", base, e)
    _robots_cache[base] = rp
    return rp


def robots_allows(url: str, user_agent: str) -> bool:
    try:
        return _robots_for(url, user_agent).can_fetch(user_agent, url)
    except Exception:
        return True


# ---------- HTML cleanup ----------

_BOILER_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")
_BOILER_CLASS_RE = re.compile(
    r"(cookie|consent|breadcrumb|menu|navigation|sidebar|footer|header)", re.I
)


def html_to_text(html: str) -> tuple[str, str]:
    """Strip boilerplate from HTML. Returns (title, body_text). Uses trafilatura when helpful."""
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
            no_fallback=False,
        )
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string or "").strip() if soup.title else ""
        if extracted and len(extracted) > 200:
            return title, extracted.strip()
    except Exception:
        pass

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title else ""
    for tag in soup(_BOILER_TAGS):
        tag.decompose()
    for el in soup.find_all(attrs={"class": _BOILER_CLASS_RE}):
        el.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def _pick_title_from_text(text: str) -> str:
    """Fallback title heuristic: first non-empty line of reasonable length."""
    for raw in text.splitlines():
        line = raw.strip()
        # Skip pure page numbers, very short fragments, and overly long paragraphs.
        if 5 <= len(line) <= 140 and not line.isdigit():
            return line
    return ""


def pdf_to_text(pdf_bytes: bytes) -> tuple[str, str]:
    """Extract (title, plain text) from a PDF byte stream. pdfplumber > pypdf fallback.

    Title resolution order: PDF `/Title` metadata → first heading-like line of extracted text →
    empty (caller falls back to filename).
    """
    title = ""
    text = ""

    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            meta = pdf.metadata or {}
            raw_title = meta.get("Title") or meta.get("title") or ""
            title = str(raw_title).strip()
            out: list[str] = []
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    out.append(t)
            text = "\n\n".join(out).strip()
    except Exception as e:
        log.warning("pdfplumber failed (%s); falling back to pypdf", e)

    if not text:
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(io.BytesIO(pdf_bytes))
            if not title and reader.metadata and reader.metadata.title:
                title = str(reader.metadata.title).strip()
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages).strip()
        except Exception as e:
            log.error("pypdf also failed: %s", e)

    if not title and text:
        title = _pick_title_from_text(text)
    return title, text


# ---------- base scraper ----------


class BaseScraper(ABC):
    source: SourceName
    base_url: str
    # paths under base_url that are explicitly in-scope for Steuererklärung content
    seed_paths: tuple[str, ...] = ()
    # regex of URLs to keep when crawling links
    allow_pattern: re.Pattern[str] | None = None
    # max pages to retrieve in one ingest run (safety guardrail)
    max_pages: int = 200

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.settings = get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": self.settings.user_agent, "Accept-Language": "de,en;q=0.8"},
            timeout=self.settings.scraper_timeout_s,
            follow_redirects=True,
        )
        self._sem = asyncio.Semaphore(self.settings.max_concurrency)
        self._seen: set[str] = set()

    async def __aenter__(self) -> "BaseScraper":
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ----- fetching -----

    @retry(
        retry=retry_if_exception(_is_retryable_http_error),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _fetch_raw(self, url: str) -> httpx.Response:
        async with self._sem:
            await asyncio.sleep(self.settings.request_delay_ms / 1000.0)
            resp = await self._client.get(url)
            resp.raise_for_status()
            return resp

    async def fetch(self, url: str) -> httpx.Response | None:
        """Fetch with robots-respect + retry. Returns None if disallowed/error."""
        if not robots_allows(url, self.settings.user_agent):
            log.info("[robots] skipping %s", url)
            return None
        cache_path = self._cache_path(url)
        if cache_path.exists():
            content = cache_path.read_bytes()
            return _CachedResponse(url=url, content=content, headers={"x-cache": "hit"})
        try:
            resp = await self._fetch_raw(url)
        except Exception as e:
            log.warning("fetch failed for %s: %s", url, e)
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        return resp

    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return self.settings.raw_dir / self.source.value / f"{h}.bin"

    # ----- link extraction -----

    def discover_links(self, html: str, base: str) -> Iterable[str]:
        soup = BeautifulSoup(html, "lxml")
        # Respect <base href> — BMF, BZSt et al. set it to the site root, which means
        # relative hrefs like "Web/DE/..." resolve to "https://host/Web/DE/...", not to
        # "<page_dir>/Web/DE/...". Falling back to the request URL breaks those sites.
        base_tag = soup.find("base", href=True)
        link_base = base_tag["href"] if base_tag else base
        link_base = urljoin(base, link_base)  # ensure absolute

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("mailto:", "javascript:", "#")):
                continue
            full = urljoin(link_base, href)
            full = full.split("#")[0]
            if not full.startswith(self.base_url):
                continue
            if self.allow_pattern and not self.allow_pattern.search(full):
                continue
            yield full

    # ----- entry point -----

    @abstractmethod
    async def crawl(self) -> AsyncIterator[DocumentCore]:
        """Yield `DocumentCore` records. Must be implemented per-source."""
        if False:  # pragma: no cover  — make it an async generator
            yield  # type: ignore[unreachable]

    # ----- helpers concrete subclasses use -----

    async def fetch_and_parse(
        self,
        url: str,
        *,
        section: str | None = None,
    ) -> DocumentCore | None:
        """Fetch a URL, detect type, return a normalized `DocumentCore`."""
        if url in self._seen:
            return None
        self._seen.add(url)
        resp = await self.fetch(url)
        if resp is None:
            return None
        ctype = (resp.headers.get("content-type") or "").lower()
        # Strip query/fragment before extension matching — BMF serves PDFs at URLs like
        # `...foo.pdf?__blob=publicationFile&v=4`, where `url.endswith('.pdf')` is False.
        parsed = urlparse(url)
        path_lc = parsed.path.lower()
        filename = parsed.path.rsplit("/", 1)[-1]

        if "pdf" in ctype or path_lc.endswith(".pdf"):
            pdf_title, text = pdf_to_text(resp.content)
            # PDF /Title metadata → first heading → filename without .pdf extension
            title = pdf_title or filename.removesuffix(".pdf").replace("-", " ").replace("_", " ").strip()
            doc_type = "pdf"
        elif (
            "html" in ctype
            or path_lc.endswith((".html", ".htm", "/"))
            or "." not in filename
        ):
            try:
                html = resp.content.decode(resp.encoding or "utf-8", errors="ignore")
            except Exception:
                html = resp.content.decode("utf-8", errors="ignore")
            title, text = html_to_text(html)
            doc_type = "html"
        else:
            return None
        if not text or len(text) < 200:
            return None
        return DocumentCore.build(
            source=self.source,
            url=url,
            title=title or url,
            content=text,
            doc_type=doc_type,  # type: ignore[arg-type]
            section=section,
        )


# ---------- a tiny shim so the cache path returns the same interface ----------


class _CachedResponse:
    def __init__(self, *, url: str, content: bytes, headers: dict) -> None:
        self.url = url
        self.content = content
        self.headers = headers
        # very loose encoding guess
        self.encoding = "utf-8"

    def raise_for_status(self) -> None:  # pragma: no cover
        return
