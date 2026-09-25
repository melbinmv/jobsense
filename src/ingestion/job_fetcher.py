"""
src/ingestion/job_fetcher.py
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_URL       = "https://api.adzuna.com/v1/api/jobs"
_COUNTRY        = "gb"          # Great Britain — change to "us", "de", etc. if needed
_REQUEST_DELAY  = 0.3           # seconds between paginated requests
_MAX_PER_PAGE   = 50            # Adzuna max results per page
_TIMEOUT        = 15            # HTTP timeout in seconds

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
JobRecord = dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str | Path = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning("config.yaml not found at %s — using defaults", path)
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _salary_string(raw: dict) -> str:
    """Build a human-readable salary string from Adzuna's min/max fields."""
    lo = raw.get("salary_min")
    hi = raw.get("salary_max")
    if lo and hi:
        return f"£{int(lo):,} – £{int(hi):,}"
    if lo:
        return f"from £{int(lo):,}"
    if hi:
        return f"up to £{int(hi):,}"
    return ""


def _normalise_job(raw: dict) -> JobRecord:
    """
    Map Adzuna's JSON fields to the canonical JobRecord schema used
    by the rest of the JobSense pipeline.
    """
    return {
        "job_id":      str(raw.get("id", "")),
        "title":       raw.get("title", ""),
        "company":     raw.get("company", {}).get("display_name", ""),
        "location":    raw.get("location", {}).get("display_name", ""),
        "url":         raw.get("redirect_url", ""),
        "description": raw.get("description", ""),
        "salary":      _salary_string(raw),
        "posted_date": raw.get("created", ""),
        "job_type":    " / ".join(filter(None, [
                           raw.get("contract_time", ""),   # full_time / part_time
                           raw.get("contract_type", ""),   # permanent / contract
                       ])),
        "remote":      False,   # Adzuna doesn't expose this reliably in v1
        "skills":      [],      # extracted later by the chunker/NLP layer
        "category":    raw.get("category", {}).get("label", ""),
        "source":      "adzuna",
        "_raw":        raw,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class JobFetcher:
    """
    High-level wrapper around the Adzuna Jobs Search API.

    Same public interface as the original IndeedFetcher so nothing
    downstream needs to change.

    Parameters
    ----------
    config_path:
        Path to config.yaml.  Reads ``location`` key for default search area.
    country:
        Adzuna country code.  Defaults to ``"gb"`` (Great Britain).
        Other options: ``"us"``, ``"de"``, ``"au"``, ``"ca"``, ``"fr"``, etc.
    """

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        country: str = _COUNTRY,
    ):
        self.config           = _load_config(config_path)
        self.default_location: str = self.config.get("location", "London")
        self.country          = country

        self._app_id  = os.getenv("ADZUNA_APP_ID")
        self._api_key = os.getenv("ADZUNA_API_KEY")

        if not self._app_id or not self._api_key:
            raise EnvironmentError(
                "Adzuna credentials missing.\n"
                "Add these to your .env file:\n"
                "  ADZUNA_APP_ID=your_app_id\n"
                "  ADZUNA_API_KEY=your_api_key\n"
                "Sign up free at https://developer.adzuna.com"
            )

        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

        logger.info(
            "JobFetcher ready  country=%s  location=%r",
            self.country, self.default_location,
        )

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(self, page: int, params: dict) -> dict:
        """
        GET /v1/api/jobs/{country}/search/{page} with auth injected.
        Raises requests.HTTPError on non-2xx responses.
        """
        url = f"{_BASE_URL}/{self.country}/search/{page}"
        params = {
            "app_id":    self._app_id,
            "app_key":   self._api_key,
            "content-type": "application/json",
            **params,
        }
        resp = self._session.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search_jobs(
        self,
        query:    str,
        location: Optional[str] = None,
        limit:    int = 20,
    ) -> list[JobRecord]:
        """
        Search Adzuna for jobs matching *query*.

        Parameters
        ----------
        query:    Role / keyword string  (e.g. "senior data scientist").
        location: City / region override.  Defaults to config.yaml ``location``.
        limit:    Maximum number of results (across pages if needed).

        Returns
        -------
        list[JobRecord]
        """
        loc = location or self.default_location
        jobs: list[JobRecord] = []
        page = 1

        logger.info("search_jobs  query=%r  location=%r  limit=%d", query, loc, limit)

        while len(jobs) < limit:
            per_page = min(_MAX_PER_PAGE, limit - len(jobs))
            try:
                data = self._get(page, {
                    "what":             query,
                    "where":            loc,
                    "results_per_page": per_page,
                    "sort_by":          "date",         # freshest first
                })
            except requests.HTTPError as exc:
                logger.error("Adzuna HTTP error on page %d: %s", page, exc)
                break
            except requests.RequestException as exc:
                logger.error("Network error on page %d: %s", page, exc)
                break

            results = data.get("results", [])
            if not results:
                break   # no more pages

            jobs.extend(_normalise_job(r) for r in results)
            logger.debug("Page %d — fetched %d jobs (total so far: %d)", page, len(results), len(jobs))

            # Stop if Adzuna has no more results
            if len(results) < per_page:
                break

            page += 1
            time.sleep(_REQUEST_DELAY)

        logger.info("search_jobs  → %d jobs returned", len(jobs))
        return jobs[:limit]

    # ------------------------------------------------------------------
    # Convenience entry-points (matching IndeedFetcher interface)
    # ------------------------------------------------------------------

    def fetch(
        self,
        query:         str,
        location:      Optional[str] = None,
        max_results:   int  = 20,
        fetch_details: bool = True,   # kept for API compatibility; Adzuna returns
    ) -> list[JobRecord]:             # full descriptions in search results already
        """
        Search for jobs and return enriched JobRecords.

        Unlike the Indeed fetcher, Adzuna includes full job descriptions in
        the search response itself, so ``fetch_details`` is a no-op — you
        always get complete data in one round-trip.

        Parameters
        ----------
        query:         Role / keyword search string.
        location:      Override config location.
        max_results:   Maximum jobs to retrieve.
        fetch_details: Ignored (descriptions already included). Kept so
                       callers don't need to change their code.

        Returns
        -------
        list[JobRecord]
        """
        if not fetch_details:
            logger.debug(
                "fetch_details=False ignored — Adzuna returns full descriptions "
                "in the search response."
            )
        return self.search_jobs(query, location=location, limit=max_results)

    def fetch_multi(
        self,
        queries:       list[str],
        location:      Optional[str] = None,
        max_per_query: int  = 20,
        fetch_details: bool = True,
    ) -> list[JobRecord]:
        """
        Run ``fetch()`` across multiple role queries and deduplicate by job_id.

        Example
        -------
        ::
            jobs = fetcher.fetch_multi(
                ["data engineer", "MLOps engineer", "LLM engineer"],
                max_per_query=15,
            )
        """
        seen:     set[str]       = set()
        all_jobs: list[JobRecord] = []

        for query in queries:
            results = self.fetch(
                query,
                location      = location,
                max_results   = max_per_query,
                fetch_details = fetch_details,
            )
            for job in results:
                dedup_key = job.get("job_id") or job.get("url") or f"_idx_{len(all_jobs)}"
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    all_jobs.append(job)

        logger.info(
            "fetch_multi done  %d unique jobs  %d queries",
            len(all_jobs), len(queries),
        )
        return all_jobs


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
#   python src/ingestion/job_fetcher.py "data engineer" --limit 3
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description = "Smoke-test JobFetcher against the live Adzuna API."
    )
    parser.add_argument(
        "query", nargs="?", default="machine learning engineer",
        help="Job search query  (default: 'machine learning engineer')"
    )
    parser.add_argument("--location", default=None,  help="Override config location")
    parser.add_argument("--limit",    type=int, default=3, help="Number of jobs (default: 3)")
    args = parser.parse_args()

    fetcher = JobFetcher()
    jobs    = fetcher.fetch(
        query       = args.query,
        location    = args.location,
        max_results = args.limit,
    )

    divider = "─" * 64
    print(f"\n{divider}")
    print(f"  {len(jobs)} job(s) returned for: {args.query!r}")
    print(divider)

    for j in jobs:
        print(f"\n  🏷  [{j['job_id']}]  {j['title']} @ {j['company']}")
        print(f"     📍 {j['location']}")
        if j.get("salary"):
            print(f"     💷 {j['salary']}")
        if j.get("job_type"):
            print(f"     ⏱  {j['job_type']}")
        if j.get("category"):
            print(f"     🏷  {j['category']}")
        if j.get("url"):
            print(f"     🔗 {j['url']}")
        if j.get("description"):
            snippet = j["description"][:200].replace("\n", " ").strip()
            print(f"     📄 {snippet}…")

    print(f"\n{divider}\n")