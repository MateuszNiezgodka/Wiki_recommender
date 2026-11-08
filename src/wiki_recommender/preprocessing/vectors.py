"""Build the document vectors that represent articles best, from the processed corpus.

An article's vector combines three separately normalized parts:

    text = normalize(whole_article + 0.5 * definition)
    vector = normalize(concatenate(text, 0.25 * categories))

The whole article supplies breadth, the definition states what the article is about, and the categories
add an editor's judgement that the text alone does not contain. `representation_eval.json`, written by
`python -m wiki_recommender.preprocessing.evaluate`, holds the measurements behind this choice.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from wiki_recommender.preprocessing.clean import CATEGORY_TOKEN_PREFIX

DEFINITION_WEIGHT = 0.5
CATEGORY_WEIGHT = 0.25


def _identity(tokens: Sequence[str]) -> Sequence[str]:
    return tokens


def tfidf_vectorizer(min_df: int = 2) -> TfidfVectorizer:
    """TF-IDF over ready token lists. Sublinear TF stops one repeated word dominating a long article."""
    return TfidfVectorizer(analyzer=_identity, sublinear_tf=True, min_df=min_df)


@dataclass
class ArticleParts:
    """Token lists per article for the parts that form a vector. All three lists align."""

    tokens: list[list[str]]
    definitions: list[list[str]]
    categories: list[list[str]]


def load_parts(path: Path, stemmed: bool = False) -> ArticleParts:
    """Read the parts from `articles_processed.parquet`. Category tokens stay lemmatized either way."""
    tokens = "tokens_stemmed" if stemmed else "tokens_lemmatized"
    definition = "tokens_definition_stemmed" if stemmed else "tokens_definition"
    table = pq.read_table(path, columns=[tokens, definition, "categories_lemmatized"])
    return ArticleParts(
        tokens=table[tokens].to_pylist(),
        definitions=table[definition].to_pylist(),
        categories=[
            [token for category in categories for token in category.split()]
            for categories in table["categories_lemmatized"].to_pylist()
        ],
    )


class DocumentVectorizer:
    """Fit on a corpus, then turn any article, including one fetched later, into a comparable vector."""

    def __init__(
        self, definition_weight: float = DEFINITION_WEIGHT, category_weight: float = CATEGORY_WEIGHT
    ):
        self.definition_weight = definition_weight
        self.category_weight = category_weight
        self.text = tfidf_vectorizer()
        self.category = tfidf_vectorizer(min_df=1)

    def fit(self, parts: ArticleParts) -> DocumentVectorizer:
        self.text.fit(parts.tokens)
        self.category.fit(parts.categories)
        return self

    def transform(self, parts: ArticleParts) -> sp.csr_matrix:
        """Return one L2-normalized row per article. Cosine similarity is then a plain dot product."""
        whole = self.text.transform(parts.tokens)
        definition = self.text.transform(parts.definitions)
        text = normalize(whole + self.definition_weight * definition)
        categories = self.category.transform(parts.categories)
        return normalize(sp.hstack([text, self.category_weight * categories]).tocsr())

    def fit_transform(self, parts: ArticleParts) -> sp.csr_matrix:
        return self.fit(parts).transform(parts)

    @property
    def feature_names(self) -> np.ndarray:
        """Names of the matrix columns, for explaining scores. Category features carry a prefix."""
        categories = [CATEGORY_TOKEN_PREFIX + name for name in self.category.get_feature_names_out()]
        return np.concatenate([self.text.get_feature_names_out(), categories])
