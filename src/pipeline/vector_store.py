"""
src/pipeline/vector_store.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml

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
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Flat numpy vector index with JSON metadata sidecar.

    Parameters
    ----------
    config_path:
        Path to config.yaml.  Reads ``top_k`` default.
    store_dir:
        Directory for embeddings.npy + chunks.json.
        Created automatically if it doesn't exist.
    mlflow_experiment:
        MLflow experiment to log build runs under.
    """

    _EMBEDDINGS_FILE = "embeddings.npy"
    _CHUNKS_FILE     = "chunks.json"

    def __init__(
        self,
        config_path:       str | Path = "config.yaml",
        store_dir:         str | Path = "data/processed",
        mlflow_experiment: str        = "jobsense-index",
    ):
        cfg           = _load_config(config_path)
        self.top_k    = int(cfg.get("top_k", 5))
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._embeddings_path = self.store_dir / self._EMBEDDINGS_FILE
        self._chunks_path     = self.store_dir / self._CHUNKS_FILE

        # In-memory state (populated by build() or load())
        self._matrix:  np.ndarray | None = None   # shape (N, dim)
        self._chunks:  list[Chunk]        = []

        mlflow.set_experiment(mlflow_experiment)
        self._mlflow_experiment = mlflow_experiment

        logger.info(
            "VectorStore ready  store_dir=%s  top_k=%d",
            self.store_dir, self.top_k,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of chunks currently in the index."""
        return len(self._chunks)

    @property
    def is_loaded(self) -> bool:
        return self._matrix is not None and len(self._chunks) > 0

    # ------------------------------------------------------------------
    # Build (index time)
    # ------------------------------------------------------------------

    def build(
        self,
        chunks:     list[Chunk],
        dvc_track:  bool = False,
        run_name:   str  = "build_index",
    ) -> None:
        """
        Build the index from a list of embedded Chunks and persist to disk.

        Parameters
        ----------
        chunks:
            Output of Embedder.embed_chunks() — each chunk must have
            an "embedding" key (np.ndarray, shape (dim,)).
        dvc_track:
            If True, call `dvc add data/processed/` after writing so
            the index is versioned.  Requires DVC to be installed and
            `dvc init` to have been run.
        run_name:
            Display name for the MLflow run.
        """
        if not chunks:
            raise ValueError("build() called with empty chunk list — nothing to index.")

        missing = [i for i, c in enumerate(chunks) if "embedding" not in c]
        if missing:
            raise ValueError(
                f"{len(missing)} chunk(s) have no 'embedding' key. "
                "Run Embedder.embed_chunks() before building the store."
            )

        logger.info("Building index from %d chunks …", len(chunks))
        t0 = time.perf_counter()

        # ── 1. Stack embeddings into a matrix ────────────────────────────
        self._matrix = np.vstack(
            [c["embedding"] for c in chunks]
        ).astype(np.float32)                        # shape (N, dim)
        embedding_dim = self._matrix.shape[1]

        # ── 2. Strip embeddings from metadata (they live in the .npy) ────
        self._chunks = []
        for c in chunks:
            meta = {k: v for k, v in c.items() if k != "embedding"}
            # Make metadata JSON-serialisable (drop numpy arrays, etc.)
            meta = _serialisable(meta)
            self._chunks.append(meta)

        elapsed = time.perf_counter() - t0

        # ── 3. Persist ────────────────────────────────────────────────────
        np.save(self._embeddings_path, self._matrix)
        with open(self._chunks_path, "w") as fh:
            json.dump(self._chunks, fh, indent=2)

        logger.info(
            "Index built  %d chunks  dim=%d  %.2fs  → %s",
            self.size, embedding_dim, elapsed, self.store_dir,
        )

        # ── 4. MLflow ─────────────────────────────────────────────────────
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "embedding_dim": embedding_dim,
                "store_dir":     str(self.store_dir),
            })
            mlflow.log_metrics({
                "num_chunks":   self.size,
                "build_seconds": round(elapsed, 3),
            })
        logger.info("MLflow run logged to experiment: %s", self._mlflow_experiment)

        # ── 5. DVC (optional) ─────────────────────────────────────────────
        if dvc_track:
            _dvc_add(self.store_dir)

    # ------------------------------------------------------------------
    # Load (query time)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load a previously built index from disk into memory.

        Raises FileNotFoundError if build() has never been run.
        """
        if not self._embeddings_path.exists():
            raise FileNotFoundError(
                f"No index found at {self._embeddings_path}.\n"
                "Run `python pipeline.py index` to build the index first."
            )

        logger.info("Loading index from %s …", self.store_dir)
        self._matrix = np.load(self._embeddings_path)           # (N, dim)
        with open(self._chunks_path) as fh:
            self._chunks = json.load(fh)

        logger.info(
            "Index loaded  %d chunks  dim=%d",
            self.size, self._matrix.shape[1],
        )

    # ------------------------------------------------------------------
    # Search (query time)
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        top_k:     int | None = None,
    ) -> list[Chunk]:
        """
        Return the top-k most similar chunks to *query_vec*.

        Parameters
        ----------
        query_vec:
            L2-normalised query embedding from Embedder.embed_query().
            Shape: (dim,)
        top_k:
            Number of results.  Defaults to config.yaml ``top_k`` (5).

        Returns
        -------
        list[Chunk]
            Sorted best-first.  Each chunk has a "score" key (float,
            0–1) added — this is the cosine similarity.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Index not loaded. Call load() before search(), or build() "
                "to create a new index."
            )

        k = top_k or self.top_k

        # Dot product = cosine similarity (vectors are L2-normalised)
        # matrix @ query_vec → shape (N,)  — one score per chunk
        scores: np.ndarray = self._matrix @ query_vec.astype(np.float32)

        # Top-k indices, best first
        if k >= self.size:
            top_indices = np.argsort(scores)[::-1]
        else:
            # argpartition is O(N) vs O(N log N) for full sort
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[Chunk] = []
        for idx in top_indices:
            chunk = dict(self._chunks[idx])       # shallow copy
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        logger.debug(
            "search  top_k=%d  best=%.4f  worst=%.4f",
            k,
            results[0]["score"]  if results else 0,
            results[-1]["score"] if results else 0,
        )
        return results

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], dvc_track: bool = False) -> None:
        """
        Append new embedded chunks to an existing loaded index.

        Useful for daily top-ups without rebuilding from scratch.
        Calls build() internally after merging, so MLflow is updated.
        """
        if not self.is_loaded:
            logger.info("No existing index — building fresh.")
            self.build(chunks, dvc_track=dvc_track)
            return

        # Re-attach placeholder embeddings from the matrix so build() is happy
        combined: list[Chunk] = []
        for i, meta in enumerate(self._chunks):
            c = dict(meta)
            c["embedding"] = self._matrix[i]
            combined.append(c)
        combined.extend(chunks)

        logger.info(
            "add()  existing=%d  new=%d  → total=%d",
            len(self._chunks), len(chunks), len(combined),
        )
        self.build(combined, dvc_track=dvc_track, run_name="add_chunks")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialisable(obj: Any) -> Any:
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _dvc_add(path: Path) -> None:
    """Call `dvc add <path>` and log the result."""
    try:
        result = subprocess.run(
            ["dvc", "add", str(path)],
            capture_output=True, text=True, check=True,
        )
        logger.info("DVC: %s", result.stdout.strip())
    except FileNotFoundError:
        logger.warning("DVC not found — skipping. Install with: pip install dvc")
    except subprocess.CalledProcessError as exc:
        logger.warning("DVC add failed: %s", exc.stderr.strip())


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
#   python src/pipeline/vector_store.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )

    # ── Build a tiny synthetic index ──────────────────────────────────────
    rng = np.random.default_rng(42)
    dim = 384

    def _make_chunk(i: int, title: str) -> Chunk:
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)          # L2-normalise
        return {
            "chunk_id":    f"job_{i:03d}_c0",
            "job_id":      f"job_{i:03d}",
            "title":       title,
            "company":     f"Company {i}",
            "location":    "London, UK",
            "salary":      "£60,000 – £80,000",
            "url":         f"https://example.com/job/{i}",
            "source":      "adzuna",
            "chunk_index": 0,
            "total_chunks": 1,
            "text":        f"Job: {title} at Company {i} | London, UK\nSample description {i}.",
            "metadata":    {},
            "embedding":   vec,
        }

    titles = [
        "Data Engineer", "ML Engineer", "Backend Engineer",
        "Data Scientist", "MLOps Engineer", "Platform Engineer",
        "Analytics Engineer", "AI Engineer", "Software Engineer",
        "LLM Engineer",
    ]
    chunks = [_make_chunk(i, t) for i, t in enumerate(titles)]

    store = VectorStore()
    store.build(chunks)

    # ── Reload from disk (simulates a fresh process) ──────────────────────
    store2 = VectorStore()
    store2.load()

    # ── Query ─────────────────────────────────────────────────────────────
    # Build a query vec similar to "Data Engineer" chunk (index 0)
    query_vec = chunks[0]["embedding"] + rng.standard_normal(dim).astype(np.float32) * 0.3
    query_vec /= np.linalg.norm(query_vec)

    results = store2.search(query_vec, top_k=3)

    divider = "─" * 64
    print(f"\n{divider}")
    print(f"  Index size : {store2.size} chunks")
    print(f"  Query      : synthetic vector close to 'Data Engineer'")
    print(f"  Top-3 results:")
    print(divider)
    for r in results:
        print(f"  [{r['score']:.4f}]  {r['title']}  @  {r['company']}")

    print(f"\n  Files written:")
    for f in sorted(store.store_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<25}  {size_kb:.1f} KB")
    print(f"{divider}\n")