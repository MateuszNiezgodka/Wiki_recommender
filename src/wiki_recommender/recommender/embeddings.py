"""Dense sentence-embedding vectors for articles, the semantic signal next to TF-IDF and links.

An article is longer than an embedding model reads at once, so it is split into passages of about
`PASSAGE_WORDS` words, each prefixed with the title to keep its context. The article vector is the
normalized mean of its first passage vectors. Passage vectors are kept, so the number of passages per
article can be compared without embedding the corpus again.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from wiki_recommender.preprocessing.clean import clean_wiki_text

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_PASSAGES = 1  # more passages did not help once combined with TF-IDF
PASSAGE_WORDS = 150
MAX_PASSAGES = 6
_BATCH_SIZE = 64


def article_passages(
    title: str, raw_text: str, max_passages: int = MAX_PASSAGES, words: int = PASSAGE_WORDS
) -> list[str]:
    """Split an article into title-prefixed passages, in reading order, so the lead comes first."""
    tokens = clean_wiki_text(raw_text).split()
    passages = [
        f"{title}. {' '.join(tokens[start : start + words])}"
        for start in range(0, min(len(tokens), max_passages * words), words)
    ]
    return passages or [title]


@lru_cache(maxsize=2)
def load_model(name: str = DEFAULT_MODEL):
    """Load an embedding model once per process. The first call downloads it from Hugging Face."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


@dataclass
class PassageVectors:
    """Normalized passage vectors with the article each belongs to and its position in that article."""

    vectors: np.ndarray
    owners: np.ndarray
    positions: np.ndarray
    articles: int

    def pool(self, max_passages: int) -> np.ndarray:
        """One L2-normalized vector per article: the mean of its first `max_passages` passages."""
        keep = self.positions < max_passages
        sums = np.zeros((self.articles, self.vectors.shape[1]), dtype=np.float32)
        np.add.at(sums, self.owners[keep], self.vectors[keep])
        return sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)


def embed_passages(
    passages: Sequence[Sequence[str]], model_name: str = DEFAULT_MODEL, show_progress: bool = False
) -> PassageVectors:
    model = load_model(model_name)
    counts = [len(article) for article in passages]
    vectors = model.encode(
        [passage for article in passages for passage in article],
        batch_size=_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    ).astype(np.float32)
    return PassageVectors(
        vectors=vectors,
        owners=np.repeat(np.arange(len(passages)), counts),
        positions=np.concatenate([np.arange(count) for count in counts]) if counts else np.array([], int),
        articles=len(passages),
    )


def embed_articles(passages: Sequence[Sequence[str]], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Article vectors from all the given passages."""
    return embed_passages(passages, model_name).pool(max(map(len, passages), default=1))
