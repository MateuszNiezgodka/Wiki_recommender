"""Command-line entry point: ``python -m wiki_recommender.preprocessing``."""

import argparse
import json
import logging
import sys
from pathlib import Path

from wiki_recommender.preprocessing.pipeline import PreprocessConfig, process_corpus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = PreprocessConfig()
    parser = argparse.ArgumentParser(
        prog="python -m wiki_recommender.preprocessing",
        description="Clean, lemmatize and stem the raw corpus into articles_processed.parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=defaults.input, help="raw corpus from Stage 1")
    parser.add_argument("--output", type=Path, default=defaults.output, help="output file")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes (default: up to 4, leaving 2 CPUs free); 4 workers peak near 5 GB of RAM",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = process_corpus(PreprocessConfig(input=args.input, output=args.output, workers=args.workers))
    summary = {key: value for key, value in stats.items() if not key.startswith("top_")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
