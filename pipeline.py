"""
pipeline.py
~~~~~~~~~~~~
CLI entry point for the JobSense pipeline.

Commands
--------
    python pipeline.py index               — fetch jobs + build vector index
    python pipeline.py index --no-fetch    — re-index from saved raw jobs
    python pipeline.py query "your query"  — RAG query against the index

Examples
--------
    # Fetch fresh jobs and build the index
    python pipeline.py index

    # Use custom queries and location
    python pipeline.py index \
        --queries "PySpark engineer" "MLOps engineer" "LLM engineer" \
        --location "London, UK" \
        --max-per-query 20

    # Re-chunk and re-embed without hitting Adzuna again
    python pipeline.py index --no-fetch

    # Query the index
    python pipeline.py query "Python data engineer with Spark experience"
    python pipeline.py query "entry level ML engineer" --top-k 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR      = Path(__file__).resolve().parent
RAW_JOBS_PATH = ROOT_DIR / "data" / "raw" / "jobs.json"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_index(args: argparse.Namespace) -> None:
    """
    Fetch jobs from Adzuna -> chunk -> embed -> build vector index.

    If --no-fetch is passed, skips the Adzuna call and re-uses the
    raw jobs saved from the previous run (data/raw/jobs.json).
    """
    from src.ingestion.job_fetcher  import JobFetcher
    from src.pipeline.chunker       import Chunker
    from src.pipeline.embedder      import Embedder
    from src.pipeline.vector_store  import VectorStore

    divider = "─" * 64

    # -- 1. Fetch or load raw jobs -----------------------------------------
    if args.no_fetch:
        if not RAW_JOBS_PATH.exists():
            logger.error(
                "--no-fetch specified but %s does not exist.\n"
                "Run without --no-fetch first to fetch and save jobs.",
                RAW_JOBS_PATH,
            )
            sys.exit(1)

        logger.info("Loading raw jobs from %s ...", RAW_JOBS_PATH)
        with open(RAW_JOBS_PATH) as fh:
            jobs = json.load(fh)
        logger.info("Loaded %d raw jobs", len(jobs))

    else:
        queries  = args.queries or ["data engineer", "ML engineer", "MLOps engineer"]
        location = args.location

        logger.info("Fetching jobs from Adzuna ...")
        logger.info("  Queries       : %s", queries)
        logger.info("  Location      : %s", location or "config.yaml default")
        logger.info("  Max per query : %d", args.max_per_query)

        fetcher = JobFetcher()
        jobs    = fetcher.fetch_multi(
            queries       = queries,
            location      = location,
            max_per_query = args.max_per_query,
        )
        logger.info("Fetched %d unique jobs", len(jobs))

        # Save raw jobs so --no-fetch works next time
        RAW_JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RAW_JOBS_PATH, "w") as fh:
            json.dump(jobs, fh, indent=2)
        logger.info("Raw jobs saved -> %s", RAW_JOBS_PATH)

    if not jobs:
        logger.error("No jobs to index. Check your Adzuna credentials and queries.")
        sys.exit(1)

    # -- 2. Chunk ----------------------------------------------------------
    logger.info("Chunking %d jobs ...", len(jobs))
    chunker = Chunker()
    chunks  = chunker.chunk_jobs(jobs)
    s       = chunker.stats(chunks)
    logger.info(
        "Chunking done  %d chunks  avg %.1f words/chunk",
        s["total_chunks"], s["avg_words"],
    )

    # -- 3. Embed ----------------------------------------------------------
    logger.info("Embedding %d chunks ...", len(chunks))
    embedder = Embedder()
    chunks   = embedder.embed_chunks(chunks)

    # -- 4. Build index ----------------------------------------------------
    logger.info("Building vector index ...")
    store = VectorStore()
    store.build(chunks, dvc_track=args.dvc)

    # -- Summary -----------------------------------------------------------
    print(f"\n{divider}")
    print(f"  Index built successfully")
    print(f"  Jobs fetched   : {len(jobs)}")
    print(f"  Chunks indexed : {store.size}")
    print(f"  Model          : {embedder.model_name}")
    print(f"  Index location : {store.store_dir}")
    print(f"  Raw jobs saved : {RAW_JOBS_PATH}")
    print(f"\n  Run a query:")
    print(f'  python pipeline.py query "your job search query"')
    print(f"{divider}\n")


def cmd_query(args: argparse.Namespace) -> None:
    """
    Load the vector index and run a RAG query.
    Prints Claude's answer + source job listings to the terminal.
    """
    from src.pipeline.embedder      import Embedder
    from src.pipeline.vector_store  import VectorStore
    from src.rag.retriever          import Retriever
    from src.rag.synthesiser        import Synthesiser

    divider = "─" * 64

    # -- Load stack --------------------------------------------------------
    embedder = Embedder()

    store = VectorStore()
    try:
        store.load()
    except FileNotFoundError:
        logger.error("No index found. Run `python pipeline.py index` first.")
        sys.exit(1)

    retriever = Retriever(embedder, store, top_k=args.top_k)
    synth     = Synthesiser()

    # -- Run query ---------------------------------------------------------
    print(f"\n{divider}")
    print(f"  Query : {args.query}")
    print(f"  Top-K : {args.top_k}")
    print(divider)
    print("  Thinking ...\n")

    result = synth.synthesise(args.query, retriever, top_k=args.top_k)

    # -- Print answer ------------------------------------------------------
    print(result.answer)

    # -- Print sources -----------------------------------------------------
    print(f"\n{divider}")
    print(f"  Sources  ({len(result.chunks_used)} jobs retrieved)")
    print(divider)
    for s in result.to_dict()["sources"]:
        print(f"  [{s['rank']}] {s['score']:.4f}  {s['title']} @ {s['company']}")
        if s.get("salary"):
            print(f"         {s['salary']}")
        if s.get("url"):
            print(f"         {s['url']}")

    print(f"\n{divider}")
    print(
        f"  Tokens : {result.input_tokens} in / {result.output_tokens} out  |  "
        f"Latency : {result.latency_secs:.2f}s"
    )
    print(f"{divider}\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "pipeline",
        description = "JobSense -- real-time job market intelligence CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- index -------------------------------------------------------------
    p_index = sub.add_parser(
        "index",
        help = "Fetch jobs from Adzuna and build the vector index",
    )
    p_index.add_argument(
        "--queries", nargs="+",
        default = None,
        metavar = "QUERY",
        help    = "Job search queries (default: data engineer, ML engineer, MLOps engineer)",
    )
    p_index.add_argument(
        "--location",
        default = None,
        help    = "Location override (default: config.yaml location)",
    )
    p_index.add_argument(
        "--max-per-query", type=int, default=20,
        dest    = "max_per_query",
        help    = "Max jobs per query (default: 20)",
    )
    p_index.add_argument(
        "--no-fetch", action="store_true",
        help = "Skip Adzuna fetch -- re-use data/raw/jobs.json from last run",
    )
    p_index.add_argument(
        "--dvc", action="store_true",
        help = "Track index files with DVC after building",
    )

    # -- query -------------------------------------------------------------
    p_query = sub.add_parser(
        "query",
        help = "Run a RAG query against the job index",
    )
    p_query.add_argument(
        "query",
        help = "Natural-language query e.g. 'Python data engineer with Spark'",
    )
    p_query.add_argument(
        "--top-k", type=int, default=5,
        dest = "top_k",
        help = "Number of job chunks to retrieve (default: 5)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)