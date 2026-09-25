"""
src/pipeline/embedder.py
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Chunk = dict[str, Any]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(config_path: str | Path = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning("config.yaml not found — using defaults")
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """
    Thin wrapper around SentenceTransformer that adds:
    - Batch encoding of Chunk lists
    - L2 normalisation (cosine → dot product in vector store)
    - MLflow run logging per encode_chunks() call

    Parameters
    ----------
    config_path:
        Path to config.yaml.  Reads ``model`` key.
    batch_size:
        Sentences per encoding batch.  32 works well on CPU;
        raise to 128 if you have a GPU.
    normalise:
        L2-normalise every embedding.  Keep True unless you switch
        the vector store to Euclidean distance.
    mlflow_experiment:
        MLflow experiment name.  All runs are grouped here.
    """

    def __init__(
        self,
        config_path:        str | Path = "config.yaml",
        batch_size:         int        = 32,
        normalise:          bool       = True,
        mlflow_experiment:  str        = "jobsense-embeddings",
    ):
        cfg              = _load_config(config_path)
        self.model_name  = cfg.get("model", "all-MiniLM-L6-v2")
        self.batch_size  = batch_size
        self.normalise   = normalise

        # ── Load sentence-transformer (downloads on first use, cached after) ──
        logger.info("Loading model: %s …", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Model ready  dim=%d  device=%s",
            self.embedding_dim, self._model.device,
        )

        # ── MLflow setup ──────────────────────────────────────────────────────
        mlflow.set_experiment(mlflow_experiment)
        self._mlflow_experiment = mlflow_experiment

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings → float32 matrix  (N, embedding_dim).
        Applies L2 normalisation if self.normalise is True.
        """
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,   # progress bar for big batches
            convert_to_numpy=True,
            normalize_embeddings=self.normalise,
        )
        return vectors.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(
        self,
        chunks:       list[Chunk],
        run_name:     str  = "embed_chunks",
        log_to_mlflow: bool = True,
    ) -> list[Chunk]:
        """
        Embed every Chunk in *chunks* and attach the vector as
        chunk["embedding"].

        Mutates the input list in-place AND returns it so you can
        chain calls::

            chunks = embedder.embed_chunks(chunker.chunk_jobs(jobs))

        Parameters
        ----------
        chunks:
            list[Chunk] from Chunker.chunk_jobs().
        run_name:
            Display name for the MLflow run.
        log_to_mlflow:
            Set False to skip MLflow (e.g. unit tests).

        Returns
        -------
        The same list[Chunk], each chunk now has an "embedding" key.
        """
        if not chunks:
            logger.warning("embed_chunks called with empty list — nothing to do")
            return chunks

        texts = [c["text"] for c in chunks]

        logger.info(
            "Embedding %d chunks with %s …", len(chunks), self.model_name
        )
        t0      = time.perf_counter()
        vectors = self._encode(texts)
        elapsed = time.perf_counter() - t0

        # Attach embeddings back to chunks
        for chunk, vec in zip(chunks, vectors):
            chunk["embedding"] = vec

        cps = len(chunks) / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Embedding done  %d chunks  %.2fs  (%.0f chunks/sec)",
            len(chunks), elapsed, cps,
        )

        # ── MLflow logging ────────────────────────────────────────────────
        if log_to_mlflow:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "model_name":    self.model_name,
                    "embedding_dim": self.embedding_dim,
                    "normalise":     self.normalise,
                    "batch_size":    self.batch_size,
                })
                mlflow.log_metrics({
                    "num_chunks":       len(chunks),
                    "encode_seconds":   round(elapsed, 3),
                    "chunks_per_second": round(cps, 1),
                })
            logger.info(
                "MLflow run logged to experiment: %s", self._mlflow_experiment
            )

        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string for retrieval.

        Returns
        -------
        np.ndarray  shape (embedding_dim,)  dtype float32
        (L2-normalised if self.normalise is True)
        """
        vec = self._encode([query])[0]
        logger.debug("embed_query  query=%r  vec[:4]=%s", query[:60], vec[:4])
        return vec


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
#   python src/pipeline/embedder.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )

    # Minimal synthetic chunks — no need for fetcher / chunker
    sample_chunks = [
        {
            "chunk_id":     "test_001_c0",
            "job_id":       "test_001",
            "title":        "Senior Data Engineer",
            "company":      "Acme Analytics",
            "location":     "London, UK",
            "salary":       "£70,000 – £90,000",
            "url":          "https://example.com/job/1",
            "source":       "adzuna",
            "chunk_index":  0,
            "total_chunks": 2,
            "text": (
                "Job: Senior Data Engineer at Acme Analytics | London, UK | "
                "£70,000 – £90,000 We are looking for a Senior Data Engineer "
                "to design and build scalable data pipelines using Apache Spark, "
                "dbt, and Airflow. Strong Python skills essential."
            ),
            "metadata": {},
        },
        {
            "chunk_id":     "test_001_c1",
            "job_id":       "test_001",
            "title":        "Senior Data Engineer",
            "company":      "Acme Analytics",
            "location":     "London, UK",
            "salary":       "£70,000 – £90,000",
            "url":          "https://example.com/job/1",
            "source":       "adzuna",
            "chunk_index":  1,
            "total_chunks": 2,
            "text": (
                "Job: Senior Data Engineer at Acme Analytics | London, UK | "
                "£70,000 – £90,000 Experience with Kafka and real-time streaming "
                "is a strong plus. Hybrid working, 2 days London office. "
                "Competitive salary, equity, and benefits."
            ),
            "metadata": {},
        },
    ]

    embedder = Embedder()
    chunks   = embedder.embed_chunks(sample_chunks, log_to_mlflow=False)

    divider  = "─" * 64
    print(f"\n{divider}")
    print(f"  Model       : {embedder.model_name}")
    print(f"  Embedding dim: {embedder.embedding_dim}")
    print(f"  Chunks      : {len(chunks)}")
    print(divider)

    for c in chunks:
        vec = c["embedding"]
        print(f"\n  chunk_id : {c['chunk_id']}")
        print(f"  shape    : {vec.shape}   dtype: {vec.dtype}")
        print(f"  norm     : {float(np.linalg.norm(vec)):.6f}  (should be ≈1.0 if normalised)")
        print(f"  vec[:5]  : {vec[:5]}")

    # Sanity-check: cosine similarity between the two chunks
    # (dot product works because vectors are L2-normalised)
    v0, v1 = chunks[0]["embedding"], chunks[1]["embedding"]
    sim = float(np.dot(v0, v1))
    print(f"\n  Cosine sim between chunk 0 and chunk 1: {sim:.4f}")
    print(f"  (Both from same job — expect high similarity ~0.8+)\n")

    # Also test embed_query
    q_vec = embedder.embed_query("data engineer Spark Python")
    print(f"  Query embedding shape : {q_vec.shape}")
    print(f"  Query vs chunk 0 sim  : {float(np.dot(q_vec, v0)):.4f}")
    print(f"{divider}\n")