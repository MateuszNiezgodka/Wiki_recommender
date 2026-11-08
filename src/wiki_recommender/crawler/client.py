"""Polite MediaWiki Action API client: identification, throttling, retries and backoff."""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

import httpx

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "WikiRecommenderCourseProject/0.1 "
    "(https://github.com/MateuszNiezgodka/Wiki_recommender; educational IR course crawler) "
    f"httpx/{httpx.__version__}"
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRYABLE_API_ERRORS = {"maxlag", "ratelimited", "readonly", "internal_api_error_DBQueryTimeoutError"}

log = logging.getLogger(__name__)


class ApiRequestError(RuntimeError):
    """The request could not be completed, even after retries."""


class ApiResponseError(RuntimeError):
    """The API answered with a permanent error, such as a missing page."""

    def __init__(self, code: str, info: str = "") -> None:
        super().__init__(f"{code}: {info}")
        self.code = code


class RateLimiter:
    """Spaces requests across threads, and lets any thread pause all of them."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._min_interval
        if slot > now:
            time.sleep(slot - now)

    def pause(self, seconds: float) -> None:
        with self._lock:
            self._next_slot = max(self._next_slot, time.monotonic() + seconds)


class WikiClient:
    """Thread-safe client. ``maxlag`` makes the crawl yield when Wikipedia's servers are busy."""

    def __init__(
        self,
        *,
        min_interval: float = 0.25,
        max_retries: int = 6,
        backoff_base: float = 2.0,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._limiter = RateLimiter(min_interval)
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._http = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )
        self.retry_count = 0

    def __enter__(self) -> WikiClient:
        return self

    def __exit__(self, *_: object) -> None:
        self._http.close()

    def get(self, **params: Any) -> dict[str, Any]:
        """Call the API and return its JSON body, retrying transient failures."""
        query = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        last_problem = "no attempt made"
        for attempt in range(self._max_retries + 1):
            if attempt:
                self.retry_count += 1
            self._limiter.wait()
            try:
                response = self._http.get(API_URL, params=query)
            except httpx.TransportError as exc:
                last_problem = repr(exc)
                self._back_off(attempt, last_problem)
                continue

            if response.status_code in _RETRYABLE_STATUS:
                last_problem = f"HTTP {response.status_code}"
                self._back_off(attempt, last_problem, response.headers.get("Retry-After"))
                continue
            if response.status_code != 200:
                raise ApiRequestError(f"HTTP {response.status_code} for {query}")

            body = response.json()
            error = body.get("error")
            if error is None:
                return body
            code = error.get("code", "unknown")
            if code not in _RETRYABLE_API_ERRORS:
                raise ApiResponseError(code, error.get("info", ""))
            last_problem = f"API error {code}"
            self._back_off(attempt, last_problem, response.headers.get("Retry-After"))
        raise ApiRequestError(f"Gave up after {self._max_retries} retries: {last_problem}")

    def _back_off(self, attempt: int, problem: str, retry_after: str | None = None) -> None:
        delay = self._backoff_base * 2**attempt + random.uniform(0, 1)
        if retry_after is not None and retry_after.isdigit():
            delay = max(delay, float(retry_after))
        log.warning("%s; pausing all requests for %.1fs", problem, delay)
        self._limiter.pause(delay)
