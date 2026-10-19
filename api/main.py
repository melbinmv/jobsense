"""
api/main.py
~~~~~~~~~~~~
"""

from __future__ import annotations

import logging
import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Allow imports from project root when running as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.job_fetcher import JobFetcher
from src.pipeline.chunker import Chunker
from src.pipeline.embedder import Embedder
from src.pipeline.vector_store import VectorStore
from src.rag.live_search import LiveSearch
from src.rag.retriever import Retriever
from src.rag.search_planner import SearchPlanner
from src.rag.streaming import stream_query
from src.rag.synthesiser import Synthesiser

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
    _state["live"]      = LiveSearch(
        store    = _state["store"],
        embedder = _state["embedder"],
        chunker  = _state["chunker"],
        planner  = SearchPlanner(),
    )

    # Load index (from Azure Blob if configured, else local disk).
    # If nothing exists, the /index endpoint must be called first.
    try:
        _state["store"].load()
        _state["retriever"] = Retriever(_state["embedder"], _state["store"])
        logger.info(
            "Index loaded  size=%d chunks  source=%s",
            _state["store"].size, _state["store"].index_source,
        )
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
    top_k: int | None = Field(
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
    location: str | None = Field(
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
    model_config = {"protected_namespaces": ()}

    status:       str
    index_size:   int
    model:        str
    index_ready:  bool
    index_source: str      # none | built | blob | disk
    blob_enabled: bool


class IndexStatsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    index_ready:  bool
    index_size:   int
    store_dir:    str
    model:        str
    top_k:        int


class IndexResponse(BaseModel):
    status:            str
    jobs_fetched:      int
    chunks_indexed:    int
    queries_used:      list[str]
    persisted_to_blob: bool


# ---------------------------------------------------------------------------
# Admin auth (protects /index)
# ---------------------------------------------------------------------------

def require_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Require an X-API-Key header matching the INDEX_ADMIN_KEY env var.

    Fails closed: if INDEX_ADMIN_KEY is not set on the server, /index is
    disabled rather than left open.
    """
    expected = os.environ.get("INDEX_ADMIN_KEY")
    if not expected:
        raise HTTPException(
            status_code = 503,
            detail      = "Indexing is disabled: INDEX_ADMIN_KEY is not configured on the server.",
        )
    if not x_api_key or not secrets.compare_digest(
        x_api_key.encode(), expected.encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


def _get_retriever():
    """Return a ready Retriever, creating it once an index exists; else None."""
    store: VectorStore = _state["store"]
    if store.is_loaded and _state.get("retriever") is None:
        _state["retriever"] = Retriever(_state["embedder"], store)
    retriever = _state.get("retriever")
    return retriever if (retriever is not None and store.is_loaded) else None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check. Returns index status and model name."""
    store: VectorStore = _state["store"]
    embedder: Embedder = _state["embedder"]
    return {
        "status":       "ok",
        "index_size":   store.size,
        "model":        embedder.model_name,
        "index_ready":  store.is_loaded,
        "index_source": store.index_source,
        "blob_enabled": store.blob_enabled,
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
def query(request: QueryRequest):
    """
    Run a RAG query against the job index.

    Returns Claude's grounded answer plus the source job listings
    that were used as context.
    """
    # 1. Fetch live listings for new topics (falls back to the existing index)
    live: LiveSearch = _state["live"]
    try:
        live_info = live.ensure_coverage(request.query)
    except Exception:
        logger.exception("Live search step failed — continuing with existing index")
        live_info = {"searched": False, "reason": "error"}

    store: VectorStore = _state["store"]
    if store.is_loaded and _state.get("retriever") is None:
        _state["retriever"] = Retriever(_state["embedder"], store)

    retriever = _state.get("retriever")
    if retriever is None or not store.is_loaded:
        raise HTTPException(
            status_code = 503,
            detail      = (
                "No job listings are indexed yet. Ask about a role, for example "
                "'data engineer jobs in London'."
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
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc))

    payload = result.to_dict()
    payload["live_search"] = live_info
    return payload


@app.post("/query/stream", tags=["RAG"])
def query_stream(request: QueryRequest):
    """
    Same pipeline as /query, but streamed as Server-Sent Events so the
    answer appears word by word.

    Events: status (only when live listings are being fetched), sources,
    token (many), done, error.
    """
    return StreamingResponse(
        stream_query(
            query         = request.query,
            top_k         = request.top_k,
            live          = _state["live"],
            get_retriever = _get_retriever,
            synth         = _state["synth"],
        ),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":     "no-cache, no-transform",
            "X-Accel-Buffering": "no",     # ask proxies not to buffer the stream
            "Connection":        "keep-alive",
        },
    )


@app.post(
    "/index",
    response_model = IndexResponse,
    tags           = ["Index"],
    dependencies   = [Depends(require_admin_key)],
)
def build_index(request: IndexRequest):
    """
    Fetch fresh jobs from Adzuna, chunk, embed, and rebuild the vector index.

    Admin-only: requires the X-API-Key header.

    This replaces the existing index (and the copy in Azure Blob, if
    configured). Typical run time: 2-5 minutes depending on
    max_per_query and network speed.
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

        # 4. Build index (also uploads to Blob if configured)
        # Hold the live-search lock so a live add can't interleave with the rebuild
        store: VectorStore = _state["store"]
        with _state["live"].lock:
            store.build(chunks)
            _state["live"].reset()   # topics searched earlier are no longer indexed

        # 5. Reload retriever with fresh index
        _state["retriever"] = Retriever(embedder, store)
        logger.info("Index rebuilt  size=%d", store.size)

    except Exception as exc:
        logger.exception("Index build failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status":            "ok",
        "jobs_fetched":      len(jobs),
        "chunks_indexed":    len(chunks),
        "queries_used":      request.queries,
        "persisted_to_blob": store.blob_synced,
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