"""
src/rag/search_planner.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Turns a user's free-text question into a job-board search
(keywords + optional UK location) using a small, fast Claude model.

Example
-------
"I know Python and Spark - what jobs suit me?"  → keywords="data engineer"
"Any nursing roles in Leeds?"                   → keywords="registered nurse", location="Leeds"
"hello!"                                        → keywords=None  (not a job question)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_MODEL      = os.getenv("PLANNER_MODEL", "claude-haiku-4-5-20251001")
_MAX_TOKENS = 100
_MAX_LEN    = 60

# Locations that mean "no specific place" for a UK-wide Adzuna search
_GENERIC_LOCATIONS = {
    "uk", "united kingdom", "great britain", "gb", "britain", "england",
    "anywhere", "remote", "any", "none", "null", "n/a", "",
}

_SYSTEM_PROMPT = """\
You convert a user's message into a UK job-board search.

Reply with ONLY a JSON object, no other text:
{"keywords": string or null, "location": string or null}

Rules:
- keywords: 1-4 words a job board understands - a job title or key skills,
  e.g. "data engineer", "registered nurse", "react developer".
  If the user describes their skills, infer the best-matching job title.
- keywords must be null if the message is NOT about jobs, roles, salaries,
  hiring or careers (greetings, general chat, unrelated questions).
- location: a UK city or region only if the user names one, otherwise null.
"""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchPlan:
    keywords: str | None
    location: str | None = None
    failed:   bool       = False      # True if the planner call itself failed

    @property
    def key(self) -> str:
        """Normalised cache key for 'have we searched this recently?'."""
        return f"{self.keywords or ''}|{(self.location or '').lower()}"


# ---------------------------------------------------------------------------
# Parsing (pure function, easy to test)
# ---------------------------------------------------------------------------

def _clean_keywords(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"[^a-z0-9+#./\- ]", "", value.lower())
    text = re.sub(r"\s+", " ", text).strip()[:_MAX_LEN]
    return text or None


def _clean_location(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"[^A-Za-z0-9 ,.'\-]", "", value)
    text = re.sub(r"\s+", " ", text).strip()[:_MAX_LEN]
    if text.lower() in _GENERIC_LOCATIONS:
        return None
    return text or None


def parse_plan(raw_text: str) -> SearchPlan:
    """Parse the model's reply. Raises ValueError if it isn't usable JSON."""
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in planner reply: {raw_text[:80]!r}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Planner reply was not a JSON object")
    return SearchPlan(
        keywords = _clean_keywords(data.get("keywords")),
        location = _clean_location(data.get("location")),
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class SearchPlanner:
    """One small Claude call per query. Never raises: failures return failed=True."""

    def __init__(self, model: str = _MODEL):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise OSError("ANTHROPIC_API_KEY not found. Add it to your .env file.")
        self.model   = model
        self._client = anthropic.Anthropic(api_key=api_key, timeout=10.0, max_retries=1)
        logger.info("SearchPlanner ready  model=%s", model)

    def plan(self, query: str) -> SearchPlan:
        try:
            response = self._client.messages.create(
                model      = self.model,
                max_tokens = _MAX_TOKENS,
                system     = _SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": query[:500]}],
            )
            plan = parse_plan(response.content[0].text)
            logger.info(
                "Planned search  query=%r  → keywords=%r  location=%r",
                query[:60], plan.keywords, plan.location,
            )
            return plan
        except Exception:
            logger.exception("Search planner failed — falling back")
            return SearchPlan(keywords=None, location=None, failed=True)