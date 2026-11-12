"""Command-line entry point.

python -m wiki_recommender.recommender "Dog" "https://en.wikipedia.org/wiki/Cat" -k 10
python -m wiki_recommender.recommender --build          # (re)build and save the model
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from wiki_recommender.recommender.engine import Recommender
from wiki_recommender.recommender.model import MODEL_PATH, build_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m wiki_recommender.recommender",
        description="Recommend Wikipedia articles similar to the visited ones (titles or URLs).",
    )
    parser.add_argument("visited", nargs="*", help="visited article titles or URLs")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="number of recommendations")
    parser.add_argument("--build", action="store_true", help="rebuild the model before recommending")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="saved model file")
    parser.add_argument("--offline", action="store_true", help="never fetch pages missing from the corpus")
    parser.add_argument(
        "--aggregation", choices=["power", "interests"], help="override how the history is aggregated"
    )
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Titles such as "Schrödinger equation" must survive consoles and pipes with a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.build:
        build_model().save(args.model)
        logging.info("Saved the model to %s", args.model)
    if not args.visited:
        if not args.build:
            print("Give at least one visited article title or URL. See --help.", file=sys.stderr)
            return 2
        return 0

    recommender = Recommender.load(args.model, fetch_unseen=not args.offline)
    overrides = {"aggregation": args.aggregation} if args.aggregation else {}
    result = recommender.recommend(args.visited, args.top_k, **overrides)
    if args.json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    else:
        print(result)
    return 0 if result.recommendations else 1


if __name__ == "__main__":
    sys.exit(main())
