"""
src/rag/live_search.py
~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from src.rag.search_planner import SearchPlan, SearchPlanner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MAX_JOBS_PER_SEARCH = 10          # jobs requested from Adzuna per new topic
FRESHNESS_HOURS     = 24          # don't re-fetch a topic within this window
MAX_SEARCHES_HOUR   = 10          # Adzuna calls per rolling hour
MAX_SEARCHES_DAY    = 40          # Adzuna calls per rolling day

LiveInfo = dict[str, Any]


def _fallback_keywords(query: str) -> str:
    """Used only when the planner failed AND nothing is indexed yet."""
    words = re.findall(r"[A-Za-z0-9+#.]+", query)[:5]
    return " ".join(words).lower()


class LiveSearch:
    def __init__(
        self,
        store,
        embedder,
        chunker,
        planner:        SearchPlanner,
        *,
        max_jobs:        int   = MAX_JOBS_PER_SEARCH,
        freshness_hours: float = FRESHNESS_HOURS,
        max_per_hour:    int   = MAX_SEARCHES_HOUR,
        max_per_day:     int   = MAX_SEARCHES_DAY,
        fetcher_factory: Callable[[], Any] | None = None,
    ):
        self.store    = store
        self.embedder = embedder
        self.chunker  = chunker
        self.planner  = planner

        self.max_jobs        = max_jobs
        self.freshness_secs  = freshness_hours * 3600
        self.max_per_hour    = max_per_hour
        self.max_per_day     = max_per_day

        self._fetcher_factory = fetcher_factory
        self._fetcher         = None

        self._searched: dict[str, float] = {}       # key → time last fetched
        self._attempts: deque[float]     = deque()  # times of Adzuna attempts

        # Serialises live searches (and /index) so the index has one writer
        self.lock = threading.Lock()

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget which topics were searched (call after the index is replaced)."""
        self._searched.clear()

    def _get_fetcher(self):
        if self._fetcher is None:
            if self._fetcher_factory is not None:
                self._fetcher = self._fetcher_factory()
            else:
                from src.ingestion.job_fetcher import JobFetcher  # lazy import
                self._fetcher = JobFetcher()
        return self._fetcher

    def _is_fresh(self, key: str) -> bool:
        ts = self._searched.get(key)
        return ts is not None and (time.time() - ts) < self.freshness_secs

    def _cap_available(self) -> bool:
        """True if the rolling hourly/daily caps allow another attempt."""
        now = time.time()
        while self._attempts and now - self._attempts[0] > 86400:
            self._attempts.popleft()
        last_hour = sum(1 for t in self._attempts if now - t <= 3600)
        return last_hour < self.max_per_hour and len(self._attempts) < self.max_per_day

    def _allow_attempt(self) -> bool:
        """Like _cap_available(), but records the attempt when allowed."""
        if not self._cap_available():
            return False
        self._attempts.append(time.time())
        return True

    # ------------------------------------------------------------------
    # Step 1: cheap decision
    # ------------------------------------------------------------------

    def plan_coverage(self, query: str) -> tuple[LiveInfo, SearchPlan | None]:
        """
        Plan the search and decide whether a live fetch is needed.

        Returns (info, plan).  *plan* is None when no fetch will happen
        (not a job question, searched recently, capped, planner failed with a
        usable index); otherwise pass both values to run_fetch().
        """
        plan = self.planner.plan(query)
        info: LiveInfo = {
            "searched": False,
            "keywords": plan.keywords,
            "location": plan.location,
            "new_jobs": 0,
            "reason":   "",
        }

        if plan.failed:
            if self.store.is_loaded:
                info["reason"] = "planner_failed_used_existing_index"
                return info, None
            plan = SearchPlan(keywords=_fallback_keywords(query), location=None)
            info["keywords"] = plan.keywords
        elif plan.keywords is None:
            info["reason"] = "not_a_job_search"
            return info, None

        if self._is_fresh(plan.key):
            info["reason"] = "recently_searched"
            return info, None

        if not self._cap_available():
            logger.warning("Live search cap reached — using existing index")
            info["reason"] = "rate_limited"
            return info, None

        return info, plan

    # ------------------------------------------------------------------
    # Step 2: slow fetch
    # ------------------------------------------------------------------

    def run_fetch(self, plan: SearchPlan, info: LiveInfo) -> LiveInfo:
        """Fetch, embed and index listings for *plan*. Never raises."""
        key = plan.key
        with self.lock:
            # Another request may have fetched this topic while we waited
            if self._is_fresh(key):
                info["reason"] = "recently_searched"
                return info

            if not self._allow_attempt():
                logger.warning("Live search cap reached — using existing index")
                info["reason"] = "rate_limited"
                return info

            try:
                self._fetch_and_index(plan, key, info)
            except Exception:
                logger.exception("Live search failed — using existing index")
                info["reason"] = "error"
        return info

    def ensure_coverage(self, query: str) -> LiveInfo:
        """
        Make sure the index has listings relevant to *query*.

        Returns a small dict for the API response / logs:
        {searched, keywords, location, new_jobs, reason}
        """
        info, plan = self.plan_coverage(query)
        if plan is None:
            return info
        return self.run_fetch(plan, info)

    # ------------------------------------------------------------------

    def _fetch_and_index(self, plan: SearchPlan, key: str, info: LiveInfo) -> None:
        fetcher = self._get_fetcher()
        jobs = fetcher.fetch_multi(
            [plan.keywords],
            location      = plan.location,
            max_per_query = self.max_jobs,
        )

        # JobFetcher swallows HTTP errors (e.g. 429) and returns [], so an empty
        # result is not cached: it is retried later, bounded by the caps above.
        if not jobs:
            info["reason"] = "no_listings"
            return

        chunks = self.chunker.chunk_jobs(jobs)
        known  = self.store.job_ids()
        new_chunks = [c for c in chunks if c.get("job_id") not in known]

        # Raw Adzuna payloads bloat chunks.json and every Blob upload
        for c in new_chunks:
            meta = c.get("metadata")
            if isinstance(meta, dict):
                meta.pop("_raw", None)

        if new_chunks:
            self.embedder.embed_chunks(
                new_chunks, run_name="live_embed", log_to_mlflow=False,
            )
            self.store.add(new_chunks)

        self._searched[key] = time.time()
        info.update(
            searched = True,
            new_jobs = len({c["job_id"] for c in new_chunks}),
            reason   = "fetched",
        )
        logger.info(
            "Live search  keywords=%r  location=%r  fetched=%d  new=%d  index=%d",
            plan.keywords, plan.location, len(jobs), info["new_jobs"], self.store.size,
        )