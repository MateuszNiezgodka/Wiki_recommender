"""Compare document representations against relevance labels that do not come from the article text.

For every article, the 10 most similar articles are scored against three signals:

- `section`: same subsection of the Vital articles lists, an editor's grouping of related subjects.
  Normalized by how many other articles the subsection holds. Subsections below 5 articles are skipped.
- `category`: shares a held-out category. Half the categories are hidden from every representation and
  used only here, so a representation that uses categories cannot score by reading the answer.
- `mutual_link`: both articles link to each other. Article text contains link anchors, which favours
  text-heavy representations, so this is a supporting signal rather than part of the primary score.

Primary score = mean(section, category). Reports the test half of the articles; the weights were chosen
on the other half, so these numbers are not fitted to the articles they describe.

Run with ``python -m wiki_recommender.preprocessing.evaluate``.
"""

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

from wiki_recommender.crawler.titles import normalize_title
from wiki_recommender.preprocessing.vectors import ArticleParts, DocumentVectorizer, tfidf_vectorizer

log = logging.getLogger(__name__)

K = 10
_MIN_SECTION_SIZE = 5
_MAX_CATEGORY_ARTICLES = 200
_SIMILARITY_BATCH = 400
_BOOTSTRAP_ROUNDS = 2000
_PRIMARY_METRICS = ("section", "category")
_COMPARISONS = (
    ("final", "full text"),
    ("final", "definition only"),
    ("final", "lead only"),
    ("full text", "definition only"),
    ("final", "final, stemmed"),
    ("full text", "full text, stemmed"),
)


def _hash_is_odd(text: str) -> bool:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % 2 == 1


def is_hidden_category(category: str) -> bool:
    """Decide by hash, so the split is stable across runs and independent of article order."""
    return _hash_is_odd(category)


def is_test_article(title: str) -> bool:
    return _hash_is_odd("split:" + title)


@dataclass
class Labels:
    section: np.ndarray
    section_size: np.ndarray
    category: sp.csr_matrix
    mutual_link: sp.csr_matrix


def build_labels(
    titles: list[str],
    topics: list[str],
    sections: list[str],
    categories: list[list[str]],
    outlinks: list[list[str]],
) -> Labels:
    """Encode the three relevance signals."""
    keys = list(zip(topics, sections, strict=True))
    sizes = Counter(keys)
    codes = {key: code for code, key in enumerate(sizes)}

    hidden = [[c for c in entry if is_hidden_category(c)] for entry in categories]
    frequency = Counter(c for entry in hidden for c in set(entry))
    usable = [c for c, count in frequency.items() if 2 <= count <= _MAX_CATEGORY_ARTICLES]
    columns = {c: column for column, c in enumerate(usable)}
    pairs = [(row, columns[c]) for row, entry in enumerate(hidden) for c in set(entry) if c in columns]

    index = {title: row for row, title in enumerate(titles)}
    links = [
        (row, index[target])
        for row, targets in enumerate(outlinks)
        for target in {normalize_title(link) for link in targets}
        if target in index and index[target] != row
    ]
    return Labels(
        section=np.array([codes[key] for key in keys]),
        section_size=np.array([sizes[key] for key in keys]),
        category=_binary_matrix(pairs, (len(titles), len(usable))),
        mutual_link=(lambda m: m.multiply(m.T).tocsr())(_binary_matrix(links, (len(titles), len(titles)))),
    )


def _binary_matrix(pairs: list[tuple[int, int]], shape: tuple[int, int]) -> sp.csr_matrix:
    rows = [row for row, _ in pairs]
    columns = [column for _, column in pairs]
    return sp.csr_matrix((np.ones(len(pairs)), (rows, columns)), shape=shape)


def nearest_neighbours(vectors: sp.csr_matrix, k: int = K) -> np.ndarray:
    """Return the k most similar articles for each article, most similar first, excluding itself."""
    vectors = vectors.tocsr()
    transposed = vectors.T.tocsc()
    result = np.zeros((vectors.shape[0], k), dtype=np.int32)
    for start in range(0, vectors.shape[0], _SIMILARITY_BATCH):
        similarities = (vectors[start : start + _SIMILARITY_BATCH] @ transposed).toarray()
        rows = np.arange(similarities.shape[0])
        similarities[rows, start + rows] = -1.0
        top = np.argpartition(-similarities, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(similarities, top, axis=1), axis=1)
        result[start : start + _SIMILARITY_BATCH] = np.take_along_axis(top, order, axis=1)
    return result


def score_queries(labels: Labels, neighbours: np.ndarray) -> dict[str, np.ndarray]:
    """Score each query article per metric. NaN marks a query the metric has no label for."""
    n, k = neighbours.shape
    same_section = (labels.section[neighbours] == labels.section[:, None]).sum(axis=1)
    possible = np.minimum(k, np.maximum(labels.section_size - 1, 1))
    section = np.where(labels.section_size >= _MIN_SECTION_SIZE, same_section / possible, np.nan)

    has_category = np.asarray(labels.category.sum(axis=1)).ravel() > 0
    has_link = np.asarray(labels.mutual_link.sum(axis=1)).ravel() > 0
    category = np.full(n, np.nan)
    mutual_link = np.full(n, np.nan)
    for row in range(n):
        if has_category[row]:
            shared = labels.category[neighbours[row]] @ labels.category[row].T
            category[row] = np.mean(shared.toarray().ravel() > 0)
        if has_link[row]:
            linked = labels.mutual_link[row, neighbours[row]]
            mutual_link[row] = np.mean(linked.toarray().ravel() > 0)
    return {"section": section, "category": category, "mutual_link": mutual_link}


def summarize_scores(scores: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    summary = {metric: float(np.nanmean(values[mask])) for metric, values in scores.items()}
    summary["primary"] = float(np.mean([summary[metric] for metric in _PRIMARY_METRICS]))
    return summary


def bootstrap_difference(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], mask: np.ndarray, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap of the primary score's difference over query articles, with a 95% interval."""
    rng = np.random.default_rng(seed)
    indices = np.flatnonzero(mask)
    differences = []
    for _ in range(_BOOTSTRAP_ROUNDS):
        sample = rng.choice(indices, len(indices))
        differences.append(
            np.mean([np.nanmean(a[m][sample]) - np.nanmean(b[m][sample]) for m in _PRIMARY_METRICS])
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": float(np.mean(differences)), "low": float(low), "high": float(high)}


def build_representations(table: Any, visible_categories: list[list[str]]) -> dict[str, sp.csr_matrix]:
    """Build the compared representations. IDF comes from whole articles in every case."""
    tokens = table["tokens_lemmatized"].to_pylist()
    stems = table["tokens_stemmed"].to_pylist()
    definitions = table["tokens_definition"].to_pylist()
    stem_definitions = table["tokens_definition_stemmed"].to_pylist()
    lead_counts = table["lead_token_count"].to_pylist()
    leads = [article[:count] for article, count in zip(tokens, lead_counts, strict=True)]

    text = tfidf_vectorizer().fit(tokens)
    stem_text = tfidf_vectorizer().fit(stems)
    lemma_parts = ArticleParts(tokens, definitions, visible_categories)
    stem_parts = ArticleParts(stems, stem_definitions, visible_categories)
    return {
        "definition only": text.transform(definitions),
        "lead only": text.transform(leads),
        "full text": text.transform(tokens),
        "full text + definition": DocumentVectorizer(category_weight=0.0).fit_transform(lemma_parts),
        "final": DocumentVectorizer().fit_transform(lemma_parts),
        "full text, stemmed": stem_text.transform(stems),
        "final, stemmed": DocumentVectorizer().fit_transform(stem_parts),
    }


def evaluate(path: Path) -> dict[str, Any]:
    columns = [
        "title",
        "topic",
        "seed_section",
        "outlinks",
        "lead_token_count",
        "tokens_lemmatized",
        "tokens_stemmed",
        "tokens_definition",
        "tokens_definition_stemmed",
        "categories_lemmatized",
    ]
    table = pq.read_table(path, columns=columns)
    titles = table["title"].to_pylist()
    categories = table["categories_lemmatized"].to_pylist()
    labels = build_labels(
        titles,
        table["topic"].to_pylist(),
        table["seed_section"].to_pylist(),
        categories,
        table["outlinks"].to_pylist(),
    )
    visible = [
        [token for category in entry if not is_hidden_category(category) for token in category.split()]
        for entry in categories
    ]
    test = np.array([is_test_article(title) for title in titles])

    results: dict[str, dict[str, np.ndarray]] = {}
    for name, vectors in build_representations(table, visible).items():
        results[name] = score_queries(labels, nearest_neighbours(vectors))
        log.info("Scored %s", name)
    return {
        "k": K,
        "test_articles": int(test.sum()),
        "test": {name: summarize_scores(scores, test) for name, scores in results.items()},
        "tuning": {name: summarize_scores(scores, ~test) for name, scores in results.items()},
        "comparisons_on_test": {
            f"{a} vs {b}": bootstrap_difference(results[a], results[b], test) for a, b in _COMPARISONS
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m wiki_recommender.preprocessing.evaluate")
    parser.add_argument("--input", type=Path, default=Path("data/processed/articles_processed.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/representation_eval.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    report = evaluate(args.input)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, summary in report["test"].items():
        print(
            f"{name:26} primary={summary['primary']:.3f} section={summary['section']:.3f} "
            f"category={summary['category']:.3f} mutual_link={summary['mutual_link']:.3f}"
        )
    for name, comparison in report["comparisons_on_test"].items():
        print(
            f"{name}: {comparison['difference']:+.3f} [{comparison['low']:+.3f}, {comparison['high']:+.3f}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
