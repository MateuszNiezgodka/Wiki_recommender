"""Link-graph vectors: two articles are related when they link to the same pages or to each other.

Each article becomes the set of pages it links to, plus its own title. Shared outlinks capture
bibliographic coupling; the title feature makes a direct link count too, since "A links to B" puts
B's title in A's set. IDF weighting stops hub pages such as "United States" from relating everything.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from wiki_recommender.crawler.titles import normalize_title


def link_features(title: str, outlinks: Iterable[str]) -> list[str]:
    """The article's own title and every distinct page it links to, in MediaWiki form."""
    features = {normalize_title(link) for link in outlinks}
    features.add(normalize_title(title))
    features.discard("")
    return sorted(features)


def _identity(features: Sequence[str]) -> Sequence[str]:
    return features


class LinkVectorizer:
    """TF-IDF over link features. A link no other article shares cannot relate two articles, so it is
    dropped (`min_df=2`)."""

    def __init__(self) -> None:
        self.tfidf = TfidfVectorizer(analyzer=_identity, binary=True, min_df=2, norm="l2")

    def fit(self, features: Sequence[Sequence[str]]) -> LinkVectorizer:
        self.tfidf.fit(features)
        return self

    def transform(self, features: Sequence[Sequence[str]]) -> sp.csr_matrix:
        return self.tfidf.transform(features).tocsr()

    def fit_transform(self, features: Sequence[Sequence[str]]) -> sp.csr_matrix:
        return self.fit(features).transform(features)
