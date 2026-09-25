"""
src/rag/synthesiser.py
~~~~~~~~~~~~~~~~~~~~~~~

"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any

import anthropic
import mlflow
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL         = "claude-sonnet-4-6"
_MAX_TOKENS    = 1024
_SYSTEM_PROMPT = """\
You are JobSense, an intelligent job market assistant.

Your job is to help users find relevant roles based on their skills, \
experience, and career goals. You answer ONLY using the job listings \
provided in the context — do not invent roles, companies, or salaries \
that are not in the context.

When answering:
1. Briefly explain why the top matches suit the user's query.
2. Highlight key skills or requirements the user should be aware of.
3. Mention salary ranges if available in the context.
4. If no listings match well (all scores low), say so honestly and \
   suggest refining the query.

Be concise, specific, and practical. Use plain English — no bullet-point \
overload. Aim for 150-250 words unless the query demands more detail.\
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SynthesisResult:
    """Structured output from Synthesiser.synthesise()."""
    answer:        str
    query:         str
    chunks_used:   list[dict[str, Any]]
    model:         str
    input_tokens:  int
    output_tokens: int
    latency_secs:  float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        """JSON-serialisable dict for the API response."""
        return {
            "answer":        self.answer,
            "query":         self.query,
            "model":         self.model,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_secs":  round(self.latency_secs, 3),
            "sources": [
                {
                    "rank":    i + 1,
                    "job_id":  c.get("job_id",  ""),
                    "title":   c.get("title",   ""),
                    "company": c.get("company", ""),
                    "location":c.get("location",""),
                    "salary":  c.get("salary",  ""),
                    "url":     c.get("url",     ""),
                    "score":   round(c.get("score", 0.0), 4),
                }
                for i, c in enumerate(self.chunks_used)
            ],
        }


# ---------------------------------------------------------------------------
# Synthesiser
# ---------------------------------------------------------------------------

class Synthesiser:
    """
    Wraps the Anthropic Messages API and formats RAG prompts.

    Parameters
    ----------
    model:
        Anthropic model ID.  Defaults to claude-sonnet-4-6.
    max_tokens:
        Max tokens in Claude's response.
    mlflow_experiment:
        MLflow experiment to log synthesis runs under.
    """

    def __init__(
        self,
        model:             str = _MODEL,
        max_tokens:        int = _MAX_TOKENS,
        mlflow_experiment: str = "jobsense-synthesis",
    ):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not found. Add it to your .env file."
            )

        self.model      = model
        self.max_tokens = max_tokens
        self._client    = anthropic.Anthropic(api_key=api_key)

        mlflow.set_experiment(mlflow_experiment)
        self._mlflow_experiment = mlflow_experiment

        logger.info("Synthesiser ready  model=%s  max_tokens=%d", model, max_tokens)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def synthesise(
        self,
        query:           str,
        retriever,
        top_k:           int  | None = None,
        log_to_mlflow:   bool        = True,
    ) -> SynthesisResult:
        """
        Full RAG pipeline: retrieve → build context → call Claude.

        Parameters
        ----------
        query:
            User's natural-language question or job search query.
        retriever:
            An initialised Retriever instance (store already loaded).
        top_k:
            Number of chunks to retrieve.  Defaults to retriever.top_k.
        log_to_mlflow:
            Log this run to MLflow.

        Returns
        -------
        SynthesisResult
        """
        # ── 1. Retrieve context ───────────────────────────────────────────
        context_str, chunks = retriever.build_context(query, top_k=top_k)

        # ── 2. Build prompt ───────────────────────────────────────────────
        user_prompt = _build_user_prompt(query, context_str)

        # ── 3. Call Claude ────────────────────────────────────────────────
        logger.info(
            "Calling Claude  model=%s  chunks=%d  query=%r",
            self.model, len(chunks), query[:60],
        )
        t0       = time.perf_counter()
        response = self._client.messages.create(
            model      = self.model,
            max_tokens = self.max_tokens,
            system     = _SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_prompt}],
        )
        latency = time.perf_counter() - t0

        # ── 4. Parse response ─────────────────────────────────────────────
        answer        = response.content[0].text
        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        logger.info(
            "Claude response  %.2fs  in=%d  out=%d  tokens",
            latency, input_tokens, output_tokens,
        )

        # ── 5. MLflow ─────────────────────────────────────────────────────
        if log_to_mlflow:
            with mlflow.start_run(run_name="synthesise"):
                mlflow.log_params({
                    "model":      self.model,
                    "max_tokens": self.max_tokens,
                    "top_k":      len(chunks),
                })
                mlflow.log_metrics({
                    "input_tokens":  input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens":  input_tokens + output_tokens,
                    "latency_secs":  round(latency, 3),
                    "chunks_used":   len(chunks),
                })

        return SynthesisResult(
            answer        = answer,
            query         = query,
            chunks_used   = chunks,
            model         = self.model,
            input_tokens  = input_tokens,
            output_tokens = output_tokens,
            latency_secs  = latency,
        )

    # ------------------------------------------------------------------
    # Convenience: synthesise from pre-built context string
    # ------------------------------------------------------------------

    def synthesise_from_context(
        self,
        query:       str,
        context_str: str,
        chunks:      list[dict[str, Any]],
    ) -> SynthesisResult:
        """
        Call Claude directly with a pre-built context string.
        Useful when you want to control retrieval separately
        (e.g. in the FastAPI endpoint).
        """
        user_prompt = _build_user_prompt(query, context_str)

        t0       = time.perf_counter()
        response = self._client.messages.create(
            model      = self.model,
            max_tokens = self.max_tokens,
            system     = _SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_prompt}],
        )
        latency = time.perf_counter() - t0

        return SynthesisResult(
            answer        = response.content[0].text,
            query         = query,
            chunks_used   = chunks,
            model         = self.model,
            input_tokens  = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            latency_secs  = latency,
        )


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(query: str, context_str: str) -> str:
    """
    Assemble the user-turn prompt from the query and retrieved context.
    Keeping prompt construction here (not inside the class) makes it
    easy to iterate without touching the API logic.
    """
    return (
        f"User query: {query}\n\n"
        f"{context_str}\n\n"
        "Based only on the job listings above, give the user a helpful, "
        "grounded answer to their query."
    )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
#   python src/rag/synthesiser.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )

    from src.pipeline.embedder     import Embedder
    from src.pipeline.vector_store import VectorStore
    from src.rag.retriever         import Retriever

    # ── Boot the stack ────────────────────────────────────────────────────
    embedder  = Embedder()
    store     = VectorStore()

    try:
        store.load()
        logger.info("Loaded existing index  size=%d", store.size)
    except FileNotFoundError:
        logger.info("No index found — building synthetic one …")
        import numpy as np
        rng = np.random.default_rng(0)
        sample = [
            ("Data Engineer",      "Monzo",       "Spark dbt Airflow Python pipelines AWS"),
            ("ML Engineer",        "DeepMind",    "PyTorch model training Python research GPU"),
            ("MLOps Engineer",     "Revolut",     "MLflow Kubernetes Docker CI CD Python"),
            ("LLM Engineer",       "Anthropic",   "LLMs RAG embeddings fine-tuning prompt eng"),
            ("Analytics Engineer", "Deliveroo",   "SQL dbt Looker BigQuery data modelling"),
            ("Data Scientist",     "Spotify",     "Python R statistics A/B testing experiments"),
        ]
        chunks = []
        for i, (title, company, desc) in enumerate(sample):
            vec = embedder.embed_query(f"{title} {desc}")
            chunks.append({
                "chunk_id":    f"smoke_{i}_c0",
                "job_id":      f"smoke_{i}",
                "title":       title,
                "company":     company,
                "location":    "London, UK",
                "salary":      "£70,000 – £100,000",
                "url":         f"https://example.com/{i}",
                "source":      "synthetic",
                "chunk_index": 0,
                "total_chunks": 1,
                "text":        f"Job: {title} at {company} | London, UK\n{desc}",
                "metadata":    {},
                "embedding":   vec,
            })
        store.build(chunks)
        store.load()

    retriever = Retriever(embedder, store, top_k=3)
    synth     = Synthesiser()

    # ── Run a query ───────────────────────────────────────────────────────
    query  = "I have strong Python skills and experience with Spark and dbt. What roles suit me?"
    result = synth.synthesise(query, retriever, log_to_mlflow=False)

    divider = "─" * 64
    print(f"\n{divider}")
    print(f"  Query  : {query}")
    print(f"  Model  : {result.model}")
    print(f"  Tokens : {result.input_tokens} in / {result.output_tokens} out")
    print(f"  Latency: {result.latency_secs:.2f}s")
    print(divider)
    print(f"\n{result.answer}\n")
    print(divider)
    print("  Sources:")
    for s in result.to_dict()["sources"]:
        print(f"  [{s['rank']}] {s['score']:.4f}  {s['title']} @ {s['company']}")
    print(f"{divider}\n")