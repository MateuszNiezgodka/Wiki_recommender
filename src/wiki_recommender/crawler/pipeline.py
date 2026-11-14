"""Crawl orchestration: per-topic quotas, resumable checkpointing and Parquet output."""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from wiki_recommender.crawler.client import ApiRequestError, ApiResponseError
from wiki_recommender.crawler.extract import extract_article
from wiki_recommender.crawler.seeds import (
    ApiGetter,
    Candidate,
    discover_candidates,
    plan_crawl_order,
)
from wiki_recommender.crawler.titles import is_prose_article_title, title_to_url

log = logging.getLogger(__name__)

ARTICLE_SCHEMA = pa.schema(
    [
        ("article_id", pa.int64()),
        ("title", pa.string()),
        ("url", pa.string()),
        ("raw_text", pa.string()),
        ("categories", pa.list_(pa.string())),
        ("outlinks", pa.list_(pa.string())),
        ("summary", pa.string()),
        ("sections", pa.list_(pa.string())),
        ("topic", pa.string()),
        ("seed_section", pa.string()),
        ("vital_level", pa.int8()),
        ("word_count", pa.int32()),
        ("revision_id", pa.int64()),
        ("retrieved_at", pa.string()),  # ISO 8601 UTC. A Parquet timezone would need tzdata on Windows.
    ]
)

_ACCEPTED = "ok"
_MAX_CONSECUTIVE_FAILURES = 25
_PROGRESS_EVERY = 100


class CrawlAborted(RuntimeError):
    """Too many requests failed in a row, which points to a network or server problem."""


@dataclass(frozen=True)
class CrawlConfig:
    output: Path
    per_topic: int = 200
    level4_share: float = 0.6
    min_words: int = 100
    workers: int = 3
    seed: int = 42
    resume: bool = True

    @property
    def checkpoint(self) -> Path:
        return self.output.with_suffix(".checkpoint.jsonl")

    @property
    def stats_path(self) -> Path:
        return self.output.with_suffix(".stats.json")


def fetch_article(
    client: ApiGetter, candidate: Candidate, min_words: int
) -> tuple[str, dict[str, Any] | None]:
    """Download and filter one article, returning ``(status, record)``.

    The status is ``"ok"`` or the reason for rejection. Redirects are followed, so the record
    always carries the target page's canonical title and id.
    """
    try:
        parsed = client.get(
            action="parse",
            page=candidate.title,
            prop="text|categories|properties|revid",
            redirects=1,
        )["parse"]
    except ApiResponseError as exc:
        return f"api_{exc.code}", None

    title = parsed["title"]
    if "disambiguation" in parsed.get("properties", {}):
        return "disambiguation", None
    if not is_prose_article_title(title):
        return "non_article_title", None

    article = extract_article(parsed["text"])
    word_count = len(article.text.split())
    if word_count < min_words:
        return "too_short", None

    return _ACCEPTED, {
        "article_id": parsed["pageid"],
        "title": title,
        "url": title_to_url(title),
        "raw_text": article.text,
        "categories": [
            c["category"].replace("_", " ") for c in parsed.get("categories", []) if not c.get("hidden")
        ],
        "outlinks": article.outlinks,
        "summary": article.summary,
        "sections": article.sections,
        "topic": candidate.topic,
        "seed_section": candidate.section,
        "vital_level": candidate.vital_level,
        "word_count": word_count,
        "revision_id": parsed.get("revid"),
        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }


class _CrawlState:
    """Everything already decided, rebuilt from the checkpoint when a crawl resumes."""

    def __init__(self) -> None:
        self.records: dict[int, dict[str, Any]] = {}
        self.attempted: set[str] = set()
        self.statuses: Counter[str] = Counter()
        self.per_topic: Counter[str] = Counter()

    def apply(self, candidate_title: str, status: str, record: dict[str, Any] | None) -> None:
        self.attempted.add(candidate_title)
        if status == _ACCEPTED and record is not None:
            if record["article_id"] in self.records:
                status = "duplicate"
            else:
                self.records[record["article_id"]] = record
                self.per_topic[record["topic"]] += 1
        self.statuses[status] += 1


def _load_checkpoint(path: Path) -> _CrawlState:
    state = _CrawlState()
    if not path.exists():
        return state
    valid_lines: list[bytes] = []
    dirty = False
    with path.open("rb") as lines:
        for raw_line in lines:
            try:
                if b"\x00" in raw_line:
                    raise ValueError("null bytes in checkpoint line")
                entry = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                log.warning("Ignoring a truncated checkpoint line, likely from an interrupted run")
                dirty = True
                continue
            if not raw_line.endswith(b"\n"):
                raw_line += b"\n"
                dirty = True
            valid_lines.append(raw_line)
            state.apply(entry["candidate"], entry["status"], entry.get("record"))
    if dirty:
        temporary = path.with_suffix(".checkpoint.tmp")
        temporary.write_bytes(b"".join(valid_lines))
        temporary.replace(path)
    return state


class _Scheduler:
    """Hands out the next candidate of whichever topic is furthest from its quota."""

    def __init__(self, queues: Mapping[str, list[Candidate]], state: _CrawlState, per_topic: int) -> None:
        self._queues = {t: deque(c for c in q if c.title not in state.attempted) for t, q in queues.items()}
        self._state = state
        self._per_topic = per_topic
        self._in_flight: Counter[str] = Counter()

    def next(self) -> Candidate | None:
        deficits = {
            topic: self._per_topic - self._state.per_topic[topic] - self._in_flight[topic]
            for topic, queue in self._queues.items()
            if queue
        }
        topic = max(sorted(deficits), key=deficits.__getitem__, default=None)
        if topic is None or deficits[topic] <= 0:
            return None
        self._in_flight[topic] += 1
        return self._queues[topic].popleft()

    def finished(self, candidate: Candidate) -> None:
        self._in_flight[candidate.topic] -= 1


def crawl(client: ApiGetter, config: CrawlConfig) -> list[dict[str, Any]]:
    """Run the crawl to completion and return the accepted article records."""
    config.output.parent.mkdir(parents=True, exist_ok=True)
    if not config.resume:
        config.checkpoint.unlink(missing_ok=True)

    candidates = discover_candidates(client)
    log.info("Discovered %d candidate articles", len(candidates))
    queues = plan_crawl_order(candidates, level4_share=config.level4_share, seed=config.seed)
    state = _load_checkpoint(config.checkpoint)
    if state.records:
        log.info("Resuming with %d articles from %s", len(state.records), config.checkpoint)

    scheduler = _Scheduler(queues, state, config.per_topic)
    pending: dict[Future[tuple[str, dict[str, Any] | None]], Candidate] = {}
    failures_in_a_row = 0

    with (
        ThreadPoolExecutor(config.workers, thread_name_prefix="crawl") as pool,
        config.checkpoint.open("a", encoding="utf-8") as checkpoint,
    ):
        try:
            while True:
                while len(pending) < config.workers * 2 and (candidate := scheduler.next()):
                    pending[pool.submit(fetch_article, client, candidate, config.min_words)] = candidate
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    candidate = pending.pop(future)
                    scheduler.finished(candidate)
                    try:
                        status, record = future.result()
                    except ApiRequestError as exc:
                        # Not checkpointed, so the article is tried again on the next run.
                        state.statuses["request_failed"] += 1
                        failures_in_a_row += 1
                        log.error("Could not fetch %r: %s", candidate.title, exc)
                        if failures_in_a_row >= _MAX_CONSECUTIVE_FAILURES:
                            raise CrawlAborted(
                                f"{failures_in_a_row} requests failed in a row; rerun to resume"
                            ) from exc
                        continue
                    failures_in_a_row = 0
                    entry = {"candidate": candidate.title, "status": status, "record": record}
                    checkpoint.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                    before = len(state.records)
                    state.apply(candidate.title, status, record)
                    if len(state.records) != before and len(state.records) % _PROGRESS_EVERY == 0:
                        log.info("Collected %d articles", len(state.records))
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise

    records = sorted(state.records.values(), key=lambda r: r["article_id"])
    for topic in sorted(queues):
        if state.per_topic[topic] < config.per_topic:
            log.warning("Topic %r ran out of candidates at %d articles", topic, state.per_topic[topic])
    write_parquet(records, config.output)
    write_stats(records, state.statuses, len(candidates), config.stats_path, config)
    return records


def write_parquet(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Write records atomically, so that a failure never leaves a half-written corpus."""
    temporary = path.with_suffix(".tmp")
    table = pa.Table.from_pylist(list(records), schema=ARTICLE_SCHEMA)
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def summarize(records: list[dict[str, Any]], statuses: Mapping[str, int], candidates: int) -> dict[str, Any]:
    """Describe the corpus: its size, composition and what the filters removed."""
    words = sorted(record["word_count"] for record in records)
    deciles = statistics.quantiles(words, n=10) if len(words) > 1 else words
    return {
        "articles": len(records),
        "candidates_discovered": candidates,
        "statuses": dict(sorted(statuses.items())),
        "per_topic": dict(sorted(Counter(r["topic"] for r in records).items())),
        "per_vital_level": dict(sorted(Counter(str(r["vital_level"]) for r in records).items())),
        "distinct_seed_sections": len({(r["topic"], r["seed_section"]) for r in records}),
        "distinct_categories": len({c for r in records for c in r["categories"]}),
        "words": {
            "total": sum(words),
            "min": words[0] if words else 0,
            "p10": round(deciles[0]) if deciles else 0,
            "median": round(statistics.median(words)) if words else 0,
            "p90": round(deciles[-1]) if deciles else 0,
            "max": words[-1] if words else 0,
        },
    }


def write_stats(
    records: list[dict[str, Any]],
    statuses: Mapping[str, int],
    candidates: int,
    path: Path,
    config: CrawlConfig,
) -> None:
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    stats = {"config": settings, **summarize(records, statuses, candidates)}
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
