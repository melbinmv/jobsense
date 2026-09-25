"""
src/pipeline/chunker.py
~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
JobRecord = dict[str, Any]
Chunk     = dict[str, Any]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(config_path: str | Path = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning("config.yaml not found — using defaults")
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Light normalisation:
    - Collapse runs of whitespace / newlines to a single space
    - Strip HTML-ish tags that sometimes sneak through job APIs
    - Strip leading / trailing whitespace
    """
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML tags
    text = re.sub(r"[^\S\n]+", " ", text)          # collapse horizontal WS
    text = re.sub(r"\n{2,}", "\n", text)            # collapse blank lines
    text = text.replace("\n", " ")                  # flatten to single line
    return text.strip()


def _build_header(job: JobRecord) -> str:
    """
    Short context string prepended to every chunk from this job.
    Keeps RAG chunks self-contained — even a mid-description chunk
    carries the role / company / location signal.
    """
    parts = [f"Job: {job['title']} at {job['company']}"]
    if job.get("location"):
        parts.append(job["location"])
    if job.get("salary"):
        parts.append(job["salary"])
    if job.get("job_type"):
        parts.append(job["job_type"])
    return " | ".join(parts)


def _sliding_window(
    words:      list[str],
    chunk_size: int,
    overlap:    int,
) -> list[str]:
    """
    Yield text chunks as strings using a sliding window over *words*.

    Parameters
    ----------
    words:      Whitespace-tokenised word list.
    chunk_size: Max words per chunk.
    overlap:    Words shared between consecutive chunks.
    """
    if not words:
        return []

    chunks: list[str] = []
    step  = max(1, chunk_size - overlap)
    start = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks


def _chunk_id(job_id: str, index: int) -> str:
    """Stable, unique chunk identifier."""
    return f"{job_id}_c{index}"


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

class Chunker:
    """
    Converts a list of JobRecords into a flat list of Chunks.

    Usage
    -----
    ::
        chunker = Chunker()
        chunks  = chunker.chunk_jobs(jobs)
        # chunks → list[Chunk] → pass to Embedder
    """

    def __init__(self, config_path: str | Path = "config.yaml"):
        cfg = _load_config(config_path)
        self.chunk_size: int = int(cfg.get("chunk_size",    256))
        self.overlap:    int = int(cfg.get("chunk_overlap",  32))
        logger.info(
            "Chunker ready  chunk_size=%d  overlap=%d",
            self.chunk_size, self.overlap,
        )

    # ------------------------------------------------------------------

    def chunk_job(self, job: JobRecord) -> list[Chunk]:
        """
        Produce Chunks for a single JobRecord.

        If the description is empty the job still produces one chunk
        containing only the header — so the job title / company is
        always searchable even if the description was missing.
        """
        job_id      = job.get("job_id") or _make_fallback_id(job)
        header      = _build_header(job)
        description = _clean_text(job.get("description", ""))

        # Combine header + description for splitting
        full_text = f"{header}\n{description}" if description else header
        words     = full_text.split()

        raw_chunks = _sliding_window(words, self.chunk_size, self.overlap)

        # Guard: always at least one chunk
        if not raw_chunks:
            raw_chunks = [header]

        total = len(raw_chunks)
        chunks: list[Chunk] = []

        for i, text in enumerate(raw_chunks):
            chunks.append({
                "chunk_id":    _chunk_id(job_id, i),
                "job_id":      job_id,
                "title":       job.get("title",    ""),
                "company":     job.get("company",  ""),
                "location":    job.get("location", ""),
                "salary":      job.get("salary",   ""),
                "url":         job.get("url",      ""),
                "source":      job.get("source",   ""),
                "chunk_index": i,
                "total_chunks": total,
                "text":        text,
                "metadata":    job,     # full record for result rendering
            })

        logger.debug(
            "chunk_job  job_id=%r  → %d chunk(s)  (%d words total)",
            job_id, total, len(words),
        )
        return chunks

    def chunk_jobs(self, jobs: list[JobRecord]) -> list[Chunk]:
        """
        Chunk an entire list of JobRecords.

        Parameters
        ----------
        jobs:  Output of JobFetcher.fetch() or JobFetcher.fetch_multi().

        Returns
        -------
        list[Chunk] — flat list, all jobs interleaved.
        """
        all_chunks: list[Chunk] = []

        for job in jobs:
            all_chunks.extend(self.chunk_job(job))

        logger.info(
            "chunk_jobs  %d jobs → %d chunks  (avg %.1f chunks/job)",
            len(jobs),
            len(all_chunks),
            len(all_chunks) / max(len(jobs), 1),
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Stats helper (useful during development)
    # ------------------------------------------------------------------

    def stats(self, chunks: list[Chunk]) -> dict:
        """
        Return a summary dict for a list of chunks.
        Useful for quick sanity-checks in the CLI.
        """
        if not chunks:
            return {"total_chunks": 0}

        word_counts = [len(c["text"].split()) for c in chunks]
        jobs_seen   = {c["job_id"] for c in chunks}

        return {
            "total_chunks":  len(chunks),
            "unique_jobs":   len(jobs_seen),
            "avg_words":     round(sum(word_counts) / len(word_counts), 1),
            "min_words":     min(word_counts),
            "max_words":     max(word_counts),
        }


# ---------------------------------------------------------------------------
# Fallback ID helper
# ---------------------------------------------------------------------------

def _make_fallback_id(job: JobRecord) -> str:
    """
    Generate a deterministic ID from title + company when job_id is absent.
    Prevents duplicate chunk_ids on re-runs.
    """
    seed = f"{job.get('title', '')}{job.get('company', '')}{job.get('url', '')}"
    return hashlib.md5(seed.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
#   python src/pipeline/chunker.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )

    # Minimal synthetic job so we can test without hitting Adzuna
    sample_jobs = [
        {
            "job_id":      "test_001",
            "title":       "Senior Data Engineer",
            "company":     "Acme Analytics",
            "location":    "London, UK",
            "salary":      "£70,000 – £90,000",
            "job_type":    "full_time / permanent",
            "url":         "https://example.com/job/test_001",
            "description": (
                "We are looking for a Senior Data Engineer to join our growing "
                "data platform team. You will design, build, and maintain scalable "
                "data pipelines using Apache Spark, dbt, and Airflow. "
                "Strong Python skills are essential along with experience in cloud "
                "environments (AWS or GCP). You will collaborate closely with "
                "data scientists and ML engineers to deliver high-quality datasets "
                "for model training and business intelligence. "
                "Experience with Kafka, Flink, or real-time streaming is a strong "
                "plus. We offer a hybrid working arrangement with 2 days per week "
                "in our London office. Competitive salary, equity, and benefits package."
            ),
            "source": "adzuna",
        }
    ]

    chunker = Chunker()
    chunks  = chunker.chunk_jobs(sample_jobs)
    s       = chunker.stats(chunks)

    divider = "─" * 64
    print(f"\n{divider}")
    print(f"  Stats: {json.dumps(s, indent=2)}")
    print(divider)

    for c in chunks:
        print(f"\n  Chunk [{c['chunk_index']+1}/{c['total_chunks']}]  id={c['chunk_id']}")
        print(f"  Words: {len(c['text'].split())}")
        print(f"  Text preview:")
        print(f"    {c['text'][:180]}…" if len(c["text"]) > 180 else f"    {c['text']}")

    print(f"\n{divider}\n")
