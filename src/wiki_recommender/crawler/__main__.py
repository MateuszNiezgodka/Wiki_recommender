"""Command-line entry point: ``python -m wiki_recommender.crawler``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from wiki_recommender.crawler.client import WikiClient
from wiki_recommender.crawler.pipeline import CrawlAborted, CrawlConfig, crawl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m wiki_recommender.crawler",
        description="Build the raw article corpus from English Wikipedia's Vital articles lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/articles_raw.parquet"), help="output file"
    )
    parser.add_argument(
        "--per-topic", type=int, default=200, help="articles to collect from each of the 11 topics"
    )
    parser.add_argument(
        "--level4-share",
        type=float,
        default=0.6,
        help="share of level 4 versus level 5 articles once the core (levels 1-3) is taken",
    )
    parser.add_argument("--min-words", type=int, default=100, help="reject articles shorter than this")
    parser.add_argument("--workers", type=int, default=3, help="concurrent requests")
    parser.add_argument("--min-interval", type=float, default=0.25, help="minimum seconds between requests")
    parser.add_argument("--seed", type=int, default=42, help="seed for the crawl order")
    parser.add_argument("--no-resume", action="store_true", help="discard any checkpoint and start over")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = CrawlConfig(
        output=args.output,
        per_topic=args.per_topic,
        level4_share=args.level4_share,
        min_words=args.min_words,
        workers=args.workers,
        seed=args.seed,
        resume=not args.no_resume,
    )
    with WikiClient(min_interval=args.min_interval) as client:
        try:
            records = crawl(client, config)
        except CrawlAborted as exc:
            logging.error("%s", exc)
            return 1
        logging.info("Retried %d requests in total", client.retry_count)

    logging.info("Saved %d articles to %s", len(records), config.output)
    print(json.dumps(json.loads(config.stats_path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
