"""Share the built data, so the recommender and the report run without crawling and preprocessing again.

    python -m wiki_recommender.download            # fetch the data archive and unpack it into data/
    python -m wiki_recommender.download --pack     # build that archive from a local data/ (to publish it)

The archive holds what Stages 3-5 read: the processed corpus, the fitted model, the cached embeddings and
the statistics and measurements behind the report. Rebuilding all of it instead takes about 2.5 hours
(see README). Only download an archive you trust: the model is a pickle, which runs code when loaded.
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

ARCHIVE_NAME = "wiki_recommender_data.zip"
DEFAULT_URL = f"https://github.com/MateuszNiezgodka/Wiki_recommender/releases/download/data-v1/{ARCHIVE_NAME}"

FILES = [
    "data/raw/articles_raw.stats.json",
    "data/processed/articles_processed.parquet",
    "data/processed/articles_processed.repairs.json",
    "data/processed/articles_processed.stats.json",
    "data/processed/recommender.pkl",
    "data/processed/recommender_eval.json",
    "data/processed/representation_eval.json",
    "data/processed/embeddings/BAAI__bge-small-en-v1.5.passages1.npz",
]


def pack(root: Path = Path("."), archive: Path = Path(ARCHIVE_NAME)) -> Path:
    """Zip the shared files of `root`. Parquet and pickle barely compress, so they are stored as is."""
    missing = [name for name in FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"build these first: {', '.join(missing)}")
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in FILES:
            method = zipfile.ZIP_DEFLATED if name.endswith(".json") else zipfile.ZIP_STORED
            bundle.write(root / name, name, compress_type=method)
    return archive


def unpack(archive: Path, root: Path = Path(".")) -> list[Path]:
    """Extract the shared files into `root`, refusing any entry that is not one of them."""
    written = []
    with zipfile.ZipFile(archive) as bundle:
        unexpected = sorted(set(bundle.namelist()) - set(FILES))
        if unexpected:
            raise ValueError(f"unexpected files in the archive: {', '.join(unexpected)}")
        for name in bundle.namelist():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, target.open("wb") as destination:
                while chunk := source.read(1 << 20):
                    destination.write(chunk)
            written.append(target)
    return written


def download(url: str, archive: Path) -> Path:
    """Stream `url` to `archive`, through a temporary file so an interrupted download leaves nothing."""
    partial = archive.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        done = 0
        with partial.open("wb") as file:
            for chunk in response.iter_bytes(1 << 20):
                file.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done / 2**20:,.0f} / {total / 2**20:,.0f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    partial.replace(archive)
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wiki_recommender.download", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--pack", action="store_true", help="build the archive from the local data/ instead")
    parser.add_argument("--url", default=DEFAULT_URL, help="where to download the archive from")
    parser.add_argument("--archive", type=Path, default=Path(ARCHIVE_NAME), help="local archive path")
    parser.add_argument("--keep", action="store_true", help="keep the downloaded archive after unpacking")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.pack:
        archive = pack(archive=args.archive)
        log.info(
            "Wrote %s (%.0f MB); attach it to the GitHub release", archive, archive.stat().st_size / 2**20
        )
        return 0
    if not args.archive.exists():
        log.info("Downloading %s", args.url)
        try:
            download(args.url, args.archive)
        except httpx.HTTPError as error:
            log.error("Download failed: %s", error)
            return 1
    for path in unpack(args.archive):
        log.info("Unpacked %s", path)
    if not args.keep:
        args.archive.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
