"""Recommend articles for a history of visited articles given as titles or URLs.

    from wiki_recommender.recommender import recommend
    result = recommend(["Dog", "https://en.wikipedia.org/wiki/Cat"], top_k=10)
    print(result)

Scoring, for a history V of m articles and a candidate c outside V:

    s(v, c) = w_text * cos_text(v, c) + w_emb * max(0, cos_emb(v, c))
    score(c) = ( (1/m) * sum over v in V of s(v, c)^p )^(1/p)

Both lie in [0, 1]. With the "interests" aggregation, V is first split into groups of related articles,
score(c) is the formula above within the group c is closest to, and the list shares its slots between
groups in proportion to their size.

Every recommendation explains its score. With the default p = 1 the score is a mean over the interest I
the candidate is closest to, and it splits exactly into

    score(c) = sum over terms t of w_text * u_t * c_t + w_emb * mean cos_emb

where u is the mean text vector of I and the mean runs over v in I. `term_contributions` holds the
largest terms of the first sum and `signal_contributions` the two signal totals.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from wiki_recommender.preprocessing.clean import CATEGORY_TOKEN_PREFIX
from wiki_recommender.recommender.fallback import FetchedArticle, FetchFailed, fetch_unseen
from wiki_recommender.recommender.model import (
    MODEL_PATH,
    SIGNALS,
    ArticleVectors,
    RecommenderModel,
    Settings,
    build_model,
    combine,
    signal_similarities,
)
from wiki_recommender.recommender.resolve import TitleResolver, parse_query
from wiki_recommender.recommender.scoring import interest_ranking, power_mean, top_indices

log = logging.getLogger(__name__)

_TOP_TERMS = 6
_TOP_CONTRIBUTIONS = 12


@dataclass
class VisitedArticle:
    """An input that was understood. `index` is its corpus row, or None for a page fetched live."""

    query: str
    title: str
    url: str
    index: int | None


@dataclass
class UnresolvedInput:
    query: str
    reason: str
    suggestions: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    rank: int
    title: str
    url: str
    score: float
    by_visited: dict[str, float]  # combined similarity s(v, c) to each visited article
    by_signal: dict[str, float]  # mean similarity over visited articles, per signal (unweighted)
    shared_terms: list[str]  # terms that contribute most to the text similarity
    topic: str
    interest: list[str] = field(default_factory=list)  # visited articles the score is computed from
    signal_contributions: dict[str, float] = field(default_factory=dict)  # weighted, sum to the score
    term_contributions: list[tuple[str, float]] = field(default_factory=list)  # largest parts of the text one


@dataclass
class RecommendationResult:
    visited: list[VisitedArticle]
    unresolved: list[UnresolvedInput]
    recommendations: list[Recommendation]
    interests: list[list[str]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = []
        for item in self.unresolved:
            hint = f" Did you mean: {', '.join(item.suggestions)}?" if item.suggestions else ""
            lines.append(f"! Skipped '{item.query}': {item.reason}.{hint}")
        if self.visited:
            sources = [v.title + ("" if v.index is not None else " (fetched)") for v in self.visited]
            lines.append("Visited: " + "; ".join(sources))
        if len(self.interests) > 1:
            lines.append("Interests: " + " | ".join(", ".join(group) for group in self.interests))
        if not self.recommendations:
            lines.append("No recommendations: none of the inputs matched an article.")
            return "\n".join(lines)
        width = max(len(r.title) for r in self.recommendations)
        lines.append(f"\n{'#':>3}  {'Score':>6}  {'Title':<{width}}  Shared terms")
        for r in self.recommendations:
            lines.append(f"{r.rank:>3}  {r.score:6.3f}  {r.title:<{width}}  {', '.join(r.shared_terms[:4])}")
        return "\n".join(lines)


def _display_term(feature: str) -> str:
    if feature.startswith(CATEGORY_TOKEN_PREFIX):
        return "category: " + feature.removeprefix(CATEGORY_TOKEN_PREFIX)
    return feature


class Recommender:
    """Holds a fitted model and answers queries. `fetch_unseen=False` keeps it fully offline."""

    def __init__(self, model: RecommenderModel, fetch_unseen: bool = True) -> None:
        self.model = model
        self.fetch_unseen = fetch_unseen
        self.resolver = TitleResolver(model.index.titles)
        self._feature_names = model.text_vectorizer.feature_names

    @classmethod
    def load(cls, path: Path = MODEL_PATH, fetch_unseen: bool = True) -> Recommender:
        """Load the saved model, building and saving it first if there is none."""
        if not path.exists():
            log.info("No model at %s; building it (a few minutes, longer if embeddings are not cached)", path)
            build_model().save(path)
        return cls(RecommenderModel.load(path), fetch_unseen)

    def resolve(
        self, inputs: Sequence[str]
    ) -> tuple[list[VisitedArticle], list[ArticleVectors], list[UnresolvedInput]]:
        visited: list[VisitedArticle] = []
        vectors: list[ArticleVectors] = []
        unresolved: list[UnresolvedInput] = []
        seen: set[str] = set()
        index = self.model.index
        for text in inputs:
            if not text or not text.strip():
                continue
            query = parse_query(text)
            if query.title is None:
                unresolved.append(UnresolvedInput(text, "not an English Wikipedia article URL"))
                continue
            found: int | FetchedArticle | None = self.resolver.find(query.title)
            if found is None and self.fetch_unseen:
                try:
                    found = fetch_unseen(query.title, self.model, self.resolver.find)
                except FetchFailed as error:
                    unresolved.append(UnresolvedInput(text, str(error), self.resolver.suggest(text)))
                    continue
            if found is None:
                unresolved.append(UnresolvedInput(text, "not in the corpus", self.resolver.suggest(text)))
                continue
            if isinstance(found, FetchedArticle):
                article = VisitedArticle(text, found.title, found.url, None)
                article_vectors = found.vectors
            else:
                article = VisitedArticle(text, index.titles[found], index.urls[found], found)
                article_vectors = self.model.vectors.rows([found])
            if article.title in seen:
                continue
            seen.add(article.title)
            visited.append(article)
            vectors.append(article_vectors)
        return visited, vectors, unresolved

    def recommend(
        self, inputs: Sequence[str] | str, top_k: int = 10, settings: Settings | None = None, **overrides
    ) -> RecommendationResult:
        """Rank corpus articles for the visited `inputs` (titles or URLs).

        `settings` or keyword overrides (e.g. `aggregation="interests"`, `power=1.0`) change the scoring.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        settings = replace(settings or self.model.settings, **overrides)
        visited, vector_parts, unresolved = self.resolve(inputs)
        if not visited:
            return RecommendationResult(visited, unresolved, [])

        queries = ArticleVectors.stack(vector_parts)
        per_signal = signal_similarities(queries, self.model.vectors)
        similarity = combine(per_signal, settings)
        excluded = [v.index for v in visited if v.index is not None]

        interests: list[list[str]] = []
        if settings.aggregation == "interests" and len(visited) > 1:
            pairwise = combine(signal_similarities(queries, queries), settings)
            chosen, scores, groups = interest_ranking(
                similarity, pairwise, top_k, excluded, settings.interest_threshold, settings.power
            )
            interests = [[visited[row].title for row in group] for group in groups]
        else:
            scores = power_mean(similarity, settings.power)
            chosen = top_indices(scores, top_k, excluded)
            groups = [list(range(len(visited)))]

        # The interest each candidate is scored by, and the mean text vector of every interest.
        per_group = np.stack([power_mean(similarity[group], settings.power) for group in groups])
        closest = per_group.argmax(axis=0)
        centroids = [np.asarray(queries.text[group].mean(axis=0)).ravel() for group in groups]
        recommendations = []
        for rank, c in enumerate(chosen, start=1):
            group = int(closest[c])
            explanation = _Explanation(groups[group], centroids[group], similarity, per_signal, settings)
            recommendations.append(self._explain(rank, int(c), float(scores[c]), visited, explanation))
        return RecommendationResult(visited, unresolved, recommendations, interests)

    def _explain(
        self,
        rank: int,
        candidate: int,
        score: float,
        visited: list[VisitedArticle],
        explanation: _Explanation,
    ) -> Recommendation:
        similarity, per_signal, group = explanation.similarity, explanation.per_signal, explanation.group
        row = self.model.vectors.text[candidate]
        contributions = row.data * explanation.centroid[row.indices]
        order = [i for i in np.argsort(-contributions)[:_TOP_CONTRIBUTIONS] if contributions[i] > 0]
        terms = [_display_term(self._feature_names[row.indices[i]]) for i in order]

        signals = {
            s: explanation.settings.weight(s) * float(per_signal[s][group, candidate].mean()) for s in SIGNALS
        }
        # The split is exact for the mean (p = 1). Other powers are shown in the same proportions.
        total = sum(signals.values())
        scale = score / total if total > 0 else 0.0
        text_weight = explanation.settings.weight("text") * scale
        index = self.model.index
        return Recommendation(
            rank=rank,
            title=index.titles[candidate],
            url=index.urls[candidate],
            score=round(score, 4),
            by_visited={v.title: round(float(similarity[i, candidate]), 4) for i, v in enumerate(visited)},
            by_signal={s: round(float(per_signal[s][:, candidate].mean()), 4) for s in SIGNALS},
            shared_terms=terms[:_TOP_TERMS],
            topic=index.topics[candidate],
            interest=[visited[i].title for i in group],
            signal_contributions={s: round(value * scale, 5) for s, value in signals.items()},
            term_contributions=[
                (term, round(float(text_weight * contributions[i]), 5))
                for term, i in zip(terms, order, strict=True)
            ],
        )


@dataclass
class _Explanation:
    """What `_explain` needs about the interest a candidate is scored by."""

    group: list[int]  # history rows of the interest
    centroid: np.ndarray  # their mean text vector
    similarity: np.ndarray
    per_signal: dict[str, np.ndarray]
    settings: Settings


@lru_cache(maxsize=1)
def default_recommender() -> Recommender:
    return Recommender.load()


def recommend(visited: Sequence[str] | str, top_k: int = 10, **overrides) -> RecommendationResult:
    """Recommend `top_k` articles for visited titles or URLs, with the default saved model."""
    return default_recommender().recommend(visited, top_k, **overrides)
