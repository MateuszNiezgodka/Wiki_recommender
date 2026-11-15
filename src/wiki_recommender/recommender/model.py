"""The recommender model: article vectors for two signals, and how they combine into one similarity.

    similarity(a, b) = w_text * cos_text(a, b) + w_emb * max(0, cos_emb(a, b))

Every cosine lies in [0, 1] and the weights sum to 1, so the similarity does too. `text` is the TF-IDF
representation chosen in Stage 2 and `embedding` is a sentence-embedding vector (`embeddings.py`).
The weights come from `recommender.evaluate`.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

from wiki_recommender.preprocessing.vectors import ArticleParts, DocumentVectorizer
from wiki_recommender.recommender.embeddings import (
    DEFAULT_MODEL,
    article_passage,
    embed_articles,
)

log = logging.getLogger(__name__)

SIGNALS = ("text", "embedding")
PROCESSED_PATH = Path("data/processed/articles_processed.parquet")
RAW_PATH = Path("data/raw/articles_raw.parquet")
MODEL_PATH = Path("data/processed/recommender.pkl")
EMBEDDINGS_DIR = Path("data/processed/embeddings")


@dataclass(frozen=True)
class Settings:
    """Everything that shapes a ranking.

    Defaults are the values `python -m wiki_recommender.recommender.evaluate` selected on the tuning half
    (see `data/processed/recommender_eval.json`).
    """

    weights: tuple[float, float] = (0.55, 0.45)  # text, embedding
    aggregation: str = "interests"  # "power" or "interests"
    power: float = 1.0
    interest_threshold: float = 0.35
    embedding_model: str = DEFAULT_MODEL

    def weight(self, signal: str) -> float:
        return self.weights[SIGNALS.index(signal)] / sum(self.weights)


@dataclass
class ArticleVectors:
    """Vectors of some articles for every signal. Rows align across the two."""

    text: sp.csr_matrix
    embedding: np.ndarray

    def __len__(self) -> int:
        return self.text.shape[0]

    def rows(self, indices: Sequence[int]) -> ArticleVectors:
        indices = list(indices)
        return ArticleVectors(self.text[indices], self.embedding[indices])

    @staticmethod
    def stack(parts: Sequence[ArticleVectors]) -> ArticleVectors:
        return ArticleVectors(
            sp.vstack([part.text for part in parts]).tocsr(),
            np.vstack([part.embedding for part in parts]),
        )


def signal_similarities(queries: ArticleVectors, corpus: ArticleVectors) -> dict[str, np.ndarray]:
    """Dense (queries x corpus) cosine similarity for each signal, every value in [0, 1]."""
    return {
        "text": np.asarray((queries.text @ corpus.text.T).todense(), dtype=np.float32),
        "embedding": np.clip(queries.embedding @ corpus.embedding.T, 0.0, 1.0).astype(np.float32),
    }


def combine(similarities: dict[str, np.ndarray], settings: Settings) -> np.ndarray:
    return sum(settings.weight(signal) * similarities[signal] for signal in SIGNALS)


@dataclass
class ArticleIndex:
    """What is shown about each corpus article. Row i describes row i of the vectors."""

    article_ids: list[int]
    titles: list[str]
    urls: list[str]
    topics: list[str]
    sections: list[str]
    summaries: list[str]

    def __len__(self) -> int:
        return len(self.titles)


@dataclass
class RecommenderModel:
    index: ArticleIndex
    vectors: ArticleVectors
    text_vectorizer: DocumentVectorizer
    settings: Settings = field(default_factory=Settings)

    def vectorize(self, processed: dict[str, Any], raw_text: str) -> ArticleVectors:
        """Vectors for an article outside the corpus, from its preprocessed record and its raw text."""
        parts = ArticleParts(
            tokens=[processed["tokens_lemmatized"]],
            definitions=[processed["tokens_definition"]],
            categories=[[token for c in processed["categories_lemmatized"] for token in c.split()]],
        )
        passage = article_passage(processed["title"], raw_text)
        return ArticleVectors(
            text=self.text_vectorizer.transform(parts),
            embedding=embed_articles([passage], self.settings.embedding_model),
        )

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> RecommenderModel:
        with path.open("rb") as file:
            return pickle.load(file)


def _embeddings_path(model_name: str, directory: Path) -> Path:
    return directory / f"{model_name.replace('/', '__')}.npz"


def corpus_embeddings(
    article_ids: list[int],
    model_name: str,
    raw_path: Path = RAW_PATH,
    directory: Path = EMBEDDINGS_DIR,
    compute: bool = True,
) -> np.ndarray:
    """One embedding per corpus article, in corpus order, cached on disk."""
    path = _embeddings_path(model_name, directory)
    if path.exists():
        with np.load(path) as data:
            if np.array_equal(data["article_ids"], article_ids):
                return data["vectors"]
    if not compute:
        raise FileNotFoundError(f"no cached {model_name} embeddings at {path}")

    raw = pq.read_table(raw_path, columns=["article_id", "title", "raw_text"]).to_pylist()
    by_id = {row["article_id"]: row for row in raw}
    passages = [article_passage(by_id[i]["title"], by_id[i]["raw_text"]) for i in article_ids]
    log.info("Embedding %d articles with %s", len(passages), model_name)
    vectors = embed_articles(passages, model_name, show_progress=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vectors=vectors, article_ids=np.asarray(article_ids))
    return vectors


def load_index(processed_path: Path = PROCESSED_PATH) -> ArticleIndex:
    table = pq.read_table(
        processed_path, columns=["article_id", "title", "url", "topic", "seed_section", "summary"]
    )
    return ArticleIndex(
        article_ids=table["article_id"].to_pylist(),
        titles=table["title"].to_pylist(),
        urls=table["url"].to_pylist(),
        topics=table["topic"].to_pylist(),
        sections=table["seed_section"].to_pylist(),
        summaries=table["summary"].to_pylist(),
    )


def build_model(
    settings: Settings | None = None,
    processed_path: Path = PROCESSED_PATH,
    raw_path: Path = RAW_PATH,
    embeddings_dir: Path = EMBEDDINGS_DIR,
) -> RecommenderModel:
    """Fit every signal on the processed corpus. Embeddings are reused from cache when present."""
    settings = settings or Settings()
    from wiki_recommender.preprocessing.vectors import load_parts

    index = load_index(processed_path)
    log.info("Fitting TF-IDF on %d articles", len(index))
    text_vectorizer = DocumentVectorizer()
    text = text_vectorizer.fit_transform(load_parts(processed_path))
    embedding = corpus_embeddings(index.article_ids, settings.embedding_model, raw_path, embeddings_dir)
    return RecommenderModel(
        index=index,
        vectors=ArticleVectors(text, embedding),
        text_vectorizer=text_vectorizer,
        settings=settings,
    )
