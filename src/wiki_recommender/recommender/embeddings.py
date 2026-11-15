"""Dense sentence-embedding vectors for articles, the semantic signal next to TF-IDF.

Each article is represented by its title and the first `PASSAGE_WORDS` words of its cleaned text, and
encoded into one L2-normalized vector.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from wiki_recommender.preprocessing.clean import clean_wiki_text

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
PASSAGE_WORDS = 150
_BATCH_SIZE = 64


def article_passage(title: str, raw_text: str, words: int = PASSAGE_WORDS) -> str:
    """Title-prefixed lead passage of an article."""
    tokens = clean_wiki_text(raw_text).split()[:words]
    return f"{title}. {' '.join(tokens)}" if tokens else title


@lru_cache(maxsize=2)
def load_model(name: str = DEFAULT_MODEL):
    """Load an embedding model once per process. The first call downloads it from Hugging Face."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


def embed_articles(
    passages: Sequence[str], model_name: str = DEFAULT_MODEL, show_progress: bool = False
) -> np.ndarray:
    """Encode one passage per article into L2-normalized vectors."""
    if not passages:
        return np.zeros((0, 0), dtype=np.float32)
    model = load_model(model_name)
    return model.encode(
        list(passages),
        batch_size=_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    ).astype(np.float32)
