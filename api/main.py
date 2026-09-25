"""
api/main.py
~~~~~~~~~~~~
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow imports from project root when running as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.job_fetcher      import JobFetcher
from src.pipeline.chunker           import Chunker
from src.pipeline.embedder          import Embedder
from src.pipeline.vector_store      import VectorStore
from src.rag.retriever              import Retriever
from src.rag.synthesiser            import Synthesiser

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App state (populated at startup)
# ---------------------------------------------------------------------------
_state: dict = {}


# ---------------------------------------------------------------------------
# Lifespan — load heavy components once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + index on startup; clean up on shutdown."""
    logger.info("JobSense API starting up …")

    # Embedder loads the sentence-transformer model (~90 MB, cached after first run)
    _state["embedder"]  = Embedder()
    _state["store"]     = VectorStore()
    _state["chunker"]   = Chunker()
    _state["synth"]     = Synthesiser()

    # Load index if it exists; otherwise the /index endpoint must be called first
    try:
        _state["store"].load()
        _state["retriever"] = Retriever(_state["embedder"], _state["store"])
        logger.info("Index loaded  size=%d chunks", _state["store"].size)
    except FileNotFoundError:
        logger.warning(
            "No index found. Call POST /index to build one before querying."
        )
        _state["retriever"] = None

    logger.info("JobSense API ready.")
    yield

    # Shutdown
    logger.info("JobSense API shutting down.")
    _state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "JobSense API",
    description = "Real-time job market intelligence powered by RAG.",
    version     = "0.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length = 3,
        max_length = 500,
        example    = "Python data engineer with Spark and dbt experience",
    )
    top_k: Optional[int] = Field(
        default = None,
        ge      = 1,
        le      = 20,
        description = "Number of job chunks to retrieve (default: config.yaml top_k)",
    )


class IndexRequest(BaseModel):
    queries: list[str] = Field(
        default  = ["data engineer", "ML engineer", "MLOps engineer"],
        min_length = 1,
        example  = ["data engineer", "machine learning engineer"],
        description = "Job search queries to fetch from Adzuna",
    )
    location: Optional[str] = Field(
        default     = None,
        example     = "London, UK",
        description = "Override the location in config.yaml",
    )
    max_per_query: int = Field(
        default = 20,
        ge      = 1,
        le      = 50,
        description = "Max jobs to fetch per query",
    )


class HealthResponse(BaseModel):
    status:      str
    index_size:  int
    model:       str
    index_ready: bool


class IndexStatsResponse(BaseModel):
    index_ready:  bool
    index_size:   int
    store_dir:    str
    model:        str
    top_k:        int


class IndexResponse(BaseModel):
    status:         str
    jobs_fetched:   int
    chunks_indexed: int
    queries_used:   list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check. Returns index status and model name."""
    store: VectorStore = _state["store"]
    embedder: Embedder = _state["embedder"]
    return {
        "status":      "ok",
        "index_size":  store.size,
        "model":       embedder.model_name,
        "index_ready": store.is_loaded,
    }


@app.get("/index/stats", response_model=IndexStatsResponse, tags=["Index"])
async def index_stats():
    """Return detailed statistics about the current vector index."""
    store:    VectorStore = _state["store"]
    embedder: Embedder    = _state["embedder"]
    retriever             = _state.get("retriever")

    return {
        "index_ready": store.is_loaded,
        "index_size":  store.size,
        "store_dir":   str(store.store_dir),
        "model":       embedder.model_name,
        "top_k":       retriever.top_k if retriever else 0,
    }


@app.post("/query", tags=["RAG"])
async def query(request: QueryRequest):
    """
    Run a RAG query against the job index.

    Returns Claude's grounded answer plus the source job listings
    that were used as context.
    """
    retriever = _state.get("retriever")
    if retriever is None or not _state["store"].is_loaded:
        raise HTTPException(
            status_code = 503,
            detail      = (
                "Index not ready. Call POST /index to build the job index first."
            ),
        )

    synth: Synthesiser = _state["synth"]

    try:
        result = synth.synthesise(
            query     = request.query,
            retriever = retriever,
            top_k     = request.top_k,
        )
    except Exception as exc:
        logger.exception("Synthesis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result.to_dict()


@app.post("/index", response_model=IndexResponse, tags=["Index"])
async def build_index(request: IndexRequest):
    """
    Fetch fresh jobs from Adzuna, chunk, embed, and rebuild the vector index.

    This replaces the existing index. Typical run time: 2-5 minutes
    depending on max_per_query and network speed.
    """
    logger.info(
        "Index build triggered  queries=%s  location=%s  max_per_query=%d",
        request.queries, request.location, request.max_per_query,
    )

    try:
        # 1. Fetch
        fetcher = JobFetcher()
        jobs    = fetcher.fetch_multi(
            queries       = request.queries,
            location      = request.location,
            max_per_query = request.max_per_query,
        )
        logger.info("Fetched %d jobs", len(jobs))

        # 2. Chunk
        chunker: Chunker = _state["chunker"]
        chunks           = chunker.chunk_jobs(jobs)
        logger.info("Produced %d chunks", len(chunks))

        # 3. Embed
        embedder: Embedder = _state["embedder"]
        chunks             = embedder.embed_chunks(chunks)

        # 4. Build index
        store: VectorStore = _state["store"]
        store.build(chunks)

        # 5. Reload retriever with fresh index
        _state["retriever"] = Retriever(embedder, store)
        logger.info("Index rebuilt  size=%d", store.size)

    except Exception as exc:
        logger.exception("Index build failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status":         "ok",
        "jobs_fetched":   len(jobs),
        "chunks_indexed": len(chunks),
        "queries_used":   request.queries,
    }


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
#   python api/main.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        log_level = "info",
    )