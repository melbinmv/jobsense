"""
src/rag/retriever.py
~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Chunk         = dict[str, Any]
RetrievedChunk = dict[str, Any]   # Chunk + "score" key


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Thin orchestration layer between the user query and the vector store.

    Parameters
    ----------
    embedder:
        An initialised Embedder instance (model already loaded).
    store:
        A VectorStore instance with load() already called.
    top_k:
        Default number of results.  Overridable per-call.
    dedup:
        If True (default), return at most one chunk per job_id.
        Increases result diversity significantly.
    score_threshold:
        Discard chunks with cosine similarity below this value.
        0.0 = return everything; 0.3 = loose filter; 0.5 = strict.
    candidate_multiplier:
        Search for top_k * multiplier candidates before dedup so
        deduplication doesn't reduce results below top_k.
    """

    def __init__(
        self,
        embedder,
        store,
        top_k:                int   = 5,
        dedup:                bool  = True,
        score_threshold:      float = 0.0,
        candidate_multiplier: int   = 3,
    ):
        self.embedder             = embedder
        self.store                = store
        self.top_k                = top_k
        self.dedup                = dedup
        self.score_threshold      = score_threshold
        self.candidate_multiplier = candidate_multiplier

        logger.info(
            "Retriever ready  top_k=%d  dedup=%s  threshold=%.2f",
            self.top_k, self.dedup, self.score_threshold,
        )

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query:           str,
        top_k:           int   | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Run the full retrieve pipeline for *query*.

        Parameters
        ----------
        query:
            Natural-language job search query.
            e.g. "senior data engineer Python Spark London"
        top_k:
            Override the instance default.
        score_threshold:
            Override the instance default.

        Returns
        -------
        list[RetrievedChunk]
            Sorted best-first.  Each item is a Chunk dict with
            "score" (float, 0-1) added.
        """
        k         = top_k           or self.top_k
        threshold = score_threshold if score_threshold is not None \
                    else self.score_threshold

        # ── 1. Embed the query ────────────────────────────────────────────
        query_vec = self.embedder.embed_query(query)

        # ── 2. Search (over-fetch so dedup doesn't starve results) ───────
        candidates_needed = k * self.candidate_multiplier if self.dedup else k
        candidates = self.store.search(query_vec, top_k=candidates_needed)

        logger.debug(
            "retrieve  query=%r  candidates=%d  threshold=%.2f  dedup=%s",
            query[:60], len(candidates), threshold, self.dedup,
        )

        # ── 3. Score threshold filter ─────────────────────────────────────
        if threshold > 0.0:
            before = len(candidates)
            candidates = [c for c in candidates if c["score"] >= threshold]
            logger.debug(
                "Threshold filter  %.2f  %d → %d chunks",
                threshold, before, len(candidates),
            )

        # ── 4. Deduplicate by job_id ──────────────────────────────────────
        if self.dedup:
            candidates = _deduplicate_by_job(candidates)

        # ── 5. Trim to top_k ──────────────────────────────────────────────
        results = candidates[:k]

        logger.info(
            "retrieve  query=%r  → %d result(s)  "
            "best=%.4f  worst=%.4f",
            query[:60],
            len(results),
            results[0]["score"]  if results else 0.0,
            results[-1]["score"] if results else 0.0,
        )
        return results

    # ------------------------------------------------------------------
    # Formatted output (used by synthesiser + API)
    # ------------------------------------------------------------------

    def retrieve_formatted(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Like retrieve(), but returns a cleaner dict without internal
        fields (_raw, metadata, embedding, etc.) — safe to serialise
        as JSON for the API response.

        Each item has:
            rank, score, job_id, title, company, location,
            salary, job_type, url, source, text_snippet
        """
        results = self.retrieve(query, top_k=top_k)
        formatted = []

        for rank, chunk in enumerate(results, start=1):
            formatted.append({
                "rank":         rank,
                "score":        round(chunk["score"], 4),
                "job_id":       chunk.get("job_id",   ""),
                "title":        chunk.get("title",    ""),
                "company":      chunk.get("company",  ""),
                "location":     chunk.get("location", ""),
                "salary":       chunk.get("salary",   ""),
                "job_type":     chunk.get("job_type", ""),
                "url":          chunk.get("url",      ""),
                "source":       chunk.get("source",   ""),
                "text_snippet": chunk.get("text", "")[:300],
            })

        return formatted

    # ------------------------------------------------------------------
    # Context string for the synthesiser
    # ------------------------------------------------------------------

    def build_context(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[str, list[RetrievedChunk]]:
        """
        Retrieve chunks and format them as a numbered context block
        ready to be injected into the synthesiser's prompt.

        Returns
        -------
        (context_str, raw_results)
            context_str  — the formatted string for the LLM prompt
            raw_results  — the list[RetrievedChunk] for structured output
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return "No relevant jobs found in the index.", []

        lines = ["Relevant job listings retrieved from the index:\n"]
        for i, chunk in enumerate(results, start=1):
            lines.append(
                f"[{i}] {chunk.get('title', 'Unknown Role')} "
                f"at {chunk.get('company', 'Unknown')} "
                f"({chunk.get('location', '')})"
            )
            if chunk.get("salary"):
                lines.append(f"    Salary: {chunk['salary']}")
            if chunk.get("url"):
                lines.append(f"    URL: {chunk['url']}")
            lines.append(f"    Relevance score: {chunk['score']:.4f}")
            lines.append(f"    Context: {chunk.get('text', '')[:400]}")
            lines.append("")

        return "\n".join(lines), results


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

def _deduplicate_by_job(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Keep only the highest-scoring chunk per job_id.
    Input must already be sorted best-first (VectorStore.search() guarantees this).
    """
    seen:   set[str]            = set()
    unique: list[RetrievedChunk] = []

    for chunk in chunks:
        job_id = chunk.get("job_id") or chunk.get("chunk_id", "")
        if job_id not in seen:
            seen.add(job_id)
            unique.append(chunk)

    logger.debug(
        "Dedup  %d → %d chunks  (%d duplicates removed)",
        len(chunks), len(unique), len(chunks) - len(unique),
    )
    return unique


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
#   python src/rag/retriever.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow imports from project root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import logging
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )

    from src.pipeline.embedder     import Embedder
    from src.pipeline.vector_store import VectorStore

    # ── Load components ───────────────────────────────────────────────────
    embedder = Embedder()

    store = VectorStore()
    try:
        store.load()
    except FileNotFoundError:
        print(
            "\n  ⚠  No index found. Building a small synthetic one first …\n"
        )
        rng = np.random.default_rng(42)
        dim = 384

        sample_jobs = [
            ("Data Engineer",       "Acme Analytics",    "Apache Spark dbt Airflow Python pipelines"),
            ("ML Engineer",         "DeepMind",          "PyTorch model training inference GPU Python"),
            ("MLOps Engineer",      "Monzo",             "Kubernetes MLflow model deployment CI CD"),
            ("Backend Engineer",    "Revolut",           "Python FastAPI PostgreSQL REST APIs"),
            ("Analytics Engineer",  "Deliveroo",         "SQL dbt Looker data modelling warehouse"),
            ("LLM Engineer",        "Anthropic",         "LLMs fine-tuning RAG embeddings prompt engineering"),
            ("Data Scientist",      "Spotify",           "statistics A/B testing Python R experimentation"),
            ("Platform Engineer",   "Wise",              "Terraform AWS infrastructure reliability SRE"),
            ("AI Engineer",         "Google DeepMind",   "machine learning Python research production"),
            ("Software Engineer",   "Meta",              "distributed systems Python Go microservices"),
        ]

        chunks = []
        for i, (title, company, desc) in enumerate(sample_jobs):
            vec = embedder.embed_query(f"{title} {desc}")
            chunks.append({
                "chunk_id":    f"job_{i:03d}_c0",
                "job_id":      f"job_{i:03d}",
                "title":       title,
                "company":     company,
                "location":    "London, UK",
                "salary":      "£70,000 – £100,000",
                "url":         f"https://example.com/job/{i}",
                "source":      "adzuna",
                "chunk_index": 0,
                "total_chunks": 1,
                "text":        f"Job: {title} at {company} | London, UK\n{desc}",
                "metadata":    {},
                "embedding":   vec,
            })

        store.build(chunks, run_name="smoke_test_index")
        store.load()

    retriever = Retriever(embedder, store)

    # ── Run test queries ──────────────────────────────────────────────────
    queries = [
        "data engineer Python Spark",
        "machine learning model deployment",
        "LLM RAG embeddings",
    ]

    divider = "─" * 64
    for query in queries:
        print(f"\n{divider}")
        print(f"  Query: {query!r}")
        print(divider)

        results = retriever.retrieve_formatted(query, top_k=3)
        for r in results:
            print(f"  [{r['rank']}] score={r['score']:.4f}  {r['title']} @ {r['company']}")

    # ── Show context string ───────────────────────────────────────────────
    print(f"\n{divider}")
    print("  build_context() output (fed to synthesiser):")
    print(divider)
    ctx, _ = retriever.build_context("data engineer Python", top_k=2)
    print(ctx)