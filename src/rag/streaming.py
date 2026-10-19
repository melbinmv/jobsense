"""
src/rag/streaming.py
~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

from src.rag.synthesiser import sources_payload

logger = logging.getLogger(__name__)

NO_INDEX_MESSAGE = (
    "No job listings are indexed yet. Ask about a role, for example "
    "'data engineer jobs in London'."
)
SEARCHING_MESSAGE = "Searching live listings…"


def sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_query(
    *,
    query:         str,
    top_k:         int | None,
    live,
    get_retriever: Callable[[], Any],
    synth,
) -> Iterator[str]:
    """Run the whole RAG pipeline for *query*, yielding SSE strings."""
    try:
        # 1. Decide whether live listings are needed (cheap)
        try:
            info, plan = live.plan_coverage(query)
        except Exception:
            logger.exception("Live search planning failed — using existing index")
            info, plan = {"searched": False, "reason": "error"}, None

        # 2. Slow part: tell the user, then fetch
        if plan is not None:
            yield sse("status", {"stage": "searching", "message": SEARCHING_MESSAGE})
            try:
                info = live.run_fetch(plan, info)
            except Exception:
                logger.exception("Live search failed — using existing index")
                info = {**info, "reason": "error"}

        # 3. Retrieve
        retriever = get_retriever()
        if retriever is None:
            yield sse("error", {"status": 503, "detail": NO_INDEX_MESSAGE})
            return
        context_str, chunks = retriever.build_context(query, top_k=top_k)
        yield sse("sources", {"sources": sources_payload(chunks)})

        # 4. Stream the answer
        for event in synth.stream_from_context(query, context_str, chunks):
            if event["type"] == "token":
                yield sse("token", {"text": event["text"]})
            elif event["type"] == "done":
                payload = {k: v for k, v in event.items() if k != "type"}
                payload["live_search"] = info
                yield sse("done", payload)

    except Exception:
        logger.exception("Streaming query failed")
        yield sse("error", {
            "status": 500,
            "detail": "Something went wrong while generating the answer.",
        })