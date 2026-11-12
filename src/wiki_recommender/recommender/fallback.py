"""Fetch a Wikipedia article that is not in the corpus, so it can still serve as visited history.

The page goes through the same steps as the corpus: the crawler's extraction, Stage 2 preprocessing with
the corpus-wide lemma repairs, and the model's vectorizers. A title that redirects to a corpus article
("USA") resolves to that article instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from wiki_recommender.crawler.client import ApiRequestError, WikiClient
from wiki_recommender.crawler.pipeline import fetch_article
from wiki_recommender.crawler.seeds import ApiGetter, Candidate
from wiki_recommender.preprocessing.pipeline import load_repairs, process_article, repairs_path
from wiki_recommender.preprocessing.tokens import LemmaRepairs, load_nlp
from wiki_recommender.recommender.model import PROCESSED_PATH, ArticleVectors, RecommenderModel

log = logging.getLogger(__name__)

_REASONS = {
    "api_missingtitle": "no English Wikipedia article has this title",
    "api_invalidtitle": "this is not a valid Wikipedia title",
    "disambiguation": "this is a disambiguation page; pick one of the meanings it lists",
    "non_article_title": "this page is not an article (a list, outline or other namespace)",
    "too_short": "the article has no prose text",
}


class FetchFailed(RuntimeError):
    """The page could not be used. The message says why, in words fit for the user."""


@dataclass
class FetchedArticle:
    title: str
    url: str
    vectors: ArticleVectors


@lru_cache(maxsize=1)
def _nlp():
    return load_nlp()


@lru_cache(maxsize=1)
def _repairs(path: Path) -> LemmaRepairs:
    return load_repairs(path)


def query_client() -> WikiClient:
    """A client tuned for a person waiting on the answer: short timeout, few retries."""
    return WikiClient(min_interval=0.1, max_retries=2, timeout=20.0)


def fetch_unseen(
    title: str,
    model: RecommenderModel,
    find_in_corpus,
    client: ApiGetter | None = None,
    processed_path: Path = PROCESSED_PATH,
) -> FetchedArticle | int:
    """Fetch and vectorize a page. Returns a corpus index instead if the title redirects into the corpus.

    `find_in_corpus` maps a canonical title to a corpus index or None.
    """
    owned = client is None
    client = client or query_client()
    try:
        status, record = fetch_article(client, Candidate(title, topic="", section="", vital_level=0), 1)
    except ApiRequestError as error:
        raise FetchFailed(f"Wikipedia could not be reached ({error})") from error
    finally:
        if owned:
            client.__exit__()

    if record is None:
        raise FetchFailed(_REASONS.get(status, f"Wikipedia refused the page ({status})"))
    corpus_index = find_in_corpus(record["title"])
    if corpus_index is not None:
        return corpus_index

    log.info("Preprocessing %s, which is not in the corpus", record["title"])
    processed = process_article(_nlp(), record, _repairs(repairs_path(processed_path)))
    return FetchedArticle(record["title"], record["url"], model.vectorize(processed, record["raw_text"]))
