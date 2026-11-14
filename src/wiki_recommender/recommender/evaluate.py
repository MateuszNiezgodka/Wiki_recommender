"""Choose the recommender's settings offline and measure how good its recommendations are.

Relevance comes from signals that are not part of the article text, as in Stage 2's evaluation:

- `section`: the recommendation is in the same Vital articles subsection as a visited article.
- `category`: it shares a held-out category with a visited article. Half of all categories are hidden
  from every signal and used only here.

Two kinds of queries are scored:

1. Single articles: are the 10 most similar articles related? This picks the signal weights and the
   embedding variant.
2. Histories of visited articles: one interest (3 or 5 articles of one subsection) or two interests
   (articles from two subsections of different topics). This picks the aggregation. Two-interest
   histories are also scored on `coverage`: whether every interest gets a recommendation.

Everything is chosen on the tuning half of the articles and reported on the test half.

Run with ``python -m wiki_recommender.recommender.evaluate``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

from wiki_recommender.preprocessing.evaluate import Labels, build_labels, is_hidden_category, is_test_article
from wiki_recommender.preprocessing.vectors import ArticleParts, DocumentVectorizer
from wiki_recommender.recommender.model import PROCESSED_PATH, Settings, corpus_passage_vectors
from wiki_recommender.recommender.scoring import interest_ranking, power_mean, top_indices

log = logging.getLogger(__name__)

K = 10
_MIN_SECTION = 5
_HISTORIES = 400
_BOOTSTRAP_ROUNDS = 2000
_WEIGHT_STEP = 0.05
EMBEDDING_VARIANTS = {
    "minilm, 1 passage": ("sentence-transformers/all-MiniLM-L6-v2", 1),
    "minilm, 2 passages": ("sentence-transformers/all-MiniLM-L6-v2", 2),
    "minilm, 4 passages": ("sentence-transformers/all-MiniLM-L6-v2", 4),
    "minilm, 6 passages": ("sentence-transformers/all-MiniLM-L6-v2", 6),
    "bge-small, 1 passage": ("BAAI/bge-small-en-v1.5", 1),
}
HISTORY_SHAPES = {
    "one interest, 3 articles": (3,),
    "one interest, 5 articles": (5,),
    "two interests, 3+2": (3, 2),
}
AGGREGATIONS: dict[str, dict[str, Any]] = {
    "mean (p=1)": {"aggregation": "power", "power": 1.0},
    "power p=2": {"aggregation": "power", "power": 2.0},
    "power p=4": {"aggregation": "power", "power": 4.0},
    "power p=8": {"aggregation": "power", "power": 8.0},
    "max": {"aggregation": "power", "power": np.inf},
    **{
        f"interests t={t}, p={p}": {"aggregation": "interests", "interest_threshold": t, "power": p}
        for t in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5)
        for p in (1.0, 2.0, 4.0)
    },
}


# --- Relevance of a ranked list -------------------------------------------------------------------------


def neighbour_scores(labels: Labels, neighbours: np.ndarray) -> dict[str, np.ndarray]:
    """Per query article: share of its neighbours in the same section / sharing a hidden category."""
    n, k = neighbours.shape
    same_section = (labels.section[neighbours] == labels.section[:, None]).sum(axis=1)
    possible = np.minimum(k, np.maximum(labels.section_size - 1, 1))
    section = np.where(labels.section_size >= _MIN_SECTION, same_section / possible, np.nan)

    categories = labels.category
    has_category = np.asarray(categories.sum(axis=1)).ravel() > 0
    hits = np.zeros(n)
    for column in range(k):
        shared = np.asarray(categories.multiply(categories[neighbours[:, column]]).sum(axis=1)).ravel()
        hits += shared > 0
    category = np.where(has_category, hits / k, np.nan)
    return {"section": section, "category": category}


def nearest(similarity: np.ndarray, rows: np.ndarray, n: int, k: int = K) -> np.ndarray:
    """Top-k neighbours, best first, of the articles `rows`, whose similarity rows are given.

    Returns an (n, k) array; rows not asked for are zero and must be masked out when summarizing.
    """
    similarity = similarity.copy()
    similarity[np.arange(len(rows)), rows] = -np.inf
    top = np.argpartition(-similarity, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(similarity, top, axis=1), axis=1)
    result = np.zeros((n, k), dtype=np.int64)
    result[rows] = np.take_along_axis(top, order, axis=1)
    return result


def summarize(scores: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    summary = {name: float(np.nanmean(values[mask])) for name, values in scores.items()}
    summary["primary"] = float(np.mean([summary[name] for name in ("section", "category")]))
    return summary


def bootstrap(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap of mean(a - b) over queries, ignoring queries either side cannot score."""
    keep = ~(np.isnan(a) | np.isnan(b))
    differences = a[keep] - b[keep]
    rng = np.random.default_rng(seed)
    means = [rng.choice(differences, len(differences)).mean() for _ in range(_BOOTSTRAP_ROUNDS)]
    low, high = np.percentile(means, [2.5, 97.5])
    return {"difference": float(differences.mean()), "low": float(low), "high": float(high)}


def primary_per_query(scores: dict[str, np.ndarray]) -> np.ndarray:
    """Mean of the metrics a query has labels for; NaN for a query with none, which bootstrap skips."""
    stacked = np.stack([scores["section"], scores["category"]])
    counts = np.sum(~np.isnan(stacked), axis=0)
    return np.where(counts > 0, np.nansum(stacked, axis=0) / np.maximum(counts, 1), np.nan)


# --- Signals ---------------------------------------------------------------------------------------------


@dataclass
class EvalData:
    titles: list[str]
    topics: np.ndarray
    labels: Labels
    test: np.ndarray
    similarities: dict[str, np.ndarray]  # dense corpus x corpus similarity per signal


def load_eval_data(path: Path, variants: dict[str, tuple[str, int]]) -> EvalData:
    columns = [
        "article_id",
        "title",
        "topic",
        "seed_section",
        "outlinks",
        "tokens_lemmatized",
        "tokens_definition",
        "categories_lemmatized",
    ]
    table = pq.read_table(path, columns=columns)
    titles = table["title"].to_pylist()
    topics = table["topic"].to_pylist()
    categories = table["categories_lemmatized"].to_pylist()
    outlinks = table["outlinks"].to_pylist()
    labels = build_labels(titles, topics, table["seed_section"].to_pylist(), categories, outlinks)
    visible = [
        [token for category in entry if not is_hidden_category(category) for token in category.split()]
        for entry in categories
    ]
    parts = ArticleParts(
        table["tokens_lemmatized"].to_pylist(), table["tokens_definition"].to_pylist(), visible
    )
    del table

    def dense(matrix: sp.csr_matrix | np.ndarray) -> np.ndarray:
        product = matrix @ matrix.T
        product = product.toarray() if sp.issparse(product) else product
        return np.clip(product, 0.0, 1.0).astype(np.float32)

    log.info("Building signals")
    similarities = {
        "text": dense(DocumentVectorizer().fit_transform(parts)),
    }
    article_ids = pq.read_table(path, columns=["article_id"])["article_id"].to_pylist()
    for name, (model, passages) in variants.items():
        try:
            vectors = corpus_passage_vectors(
                article_ids, model, passages, directory=_embeddings_dir(path), compute=False
            )
        except FileNotFoundError:
            log.warning("No cached embeddings for %s; skipped", name)
            continue
        similarities[name] = dense(vectors.pool(passages))
    test = np.array([is_test_article(title) for title in titles])
    return EvalData(titles, np.array(topics), labels, test, similarities)


def _embeddings_dir(processed_path: Path) -> Path:
    return processed_path.parent / "embeddings"


def combined(data: EvalData, weights: dict[str, float], rows: np.ndarray | slice = slice(None)) -> np.ndarray:
    total = sum(weights.values())
    return sum((weight / total) * data.similarities[name][rows] for name, weight in weights.items() if weight)


def weight_grid(signals: list[str], step: float = _WEIGHT_STEP) -> list[dict[str, float]]:
    steps = round(1 / step)
    grid = []
    for parts in itertools.product(range(steps + 1), repeat=len(signals)):
        if sum(parts) == steps:
            grid.append({signal: part * step for signal, part in zip(signals, parts, strict=True)})
    return grid


def tune_weights(data: EvalData, embedding: str) -> tuple[dict[str, float], float]:
    """Best weights on the tuning half for text and one embedding variant."""
    signals = ["text", embedding]
    rows = np.flatnonzero(~data.test)
    best, best_score = {}, -1.0
    for weights in weight_grid(signals):
        neighbours = nearest(combined(data, weights, rows), rows, len(data.titles))
        score = summarize(neighbour_scores(data.labels, neighbours), ~data.test)["primary"]
        if score > best_score:
            best, best_score = weights, score
    return best, best_score


# --- Histories -------------------------------------------------------------------------------------------


@dataclass
class History:
    rows: list[int]
    interests: list[int]  # section code of each interest
    interest_of_row: list[int]


def sample_histories(
    data: EvalData, split: np.ndarray, shape: tuple[int, ...], count: int, seed: int
) -> list[History]:
    """Histories whose articles come from the split, one subsection per interest, distinct topics.

    A subsection qualifies if it holds at least K articles beyond the ones visited, so a perfect list
    exists.
    """
    labels = data.labels
    rng = np.random.default_rng(seed)
    members: dict[int, list[int]] = defaultdict(list)
    for row in np.flatnonzero(split):
        members[int(labels.section[row])].append(int(row))
    topic_of = {int(code): data.topics[row] for row, code in enumerate(labels.section)}
    eligible = {
        size: [
            code
            for code, rows in members.items()
            if len(rows) >= size and labels.section_size[rows[0]] >= size + K
        ]
        for size in set(shape)
    }
    histories = []
    for _ in range(count * 20):
        if len(histories) == count:
            break
        codes, topics = [], set()
        for size in shape:
            code = int(rng.choice(eligible[size]))
            if topic_of[code] in topics or code in codes:
                break
            codes.append(code)
            topics.add(topic_of[code])
        if len(codes) != len(shape):
            continue
        rows, owner = [], []
        for interest, (code, size) in enumerate(zip(codes, shape, strict=True)):
            rows += [int(r) for r in rng.choice(members[code], size, replace=False)]
            owner += [interest] * size
        histories.append(History(rows, codes, owner))
    return histories


def rank_history(
    data: EvalData, weights: dict[str, float], history: History, options: dict[str, Any]
) -> np.ndarray:
    similarity = combined(data, weights, history.rows)
    if options["aggregation"] == "interests" and len(history.rows) > 1:
        pairwise = similarity[:, history.rows]
        chosen, _, _ = interest_ranking(
            similarity, pairwise, K, history.rows, options["interest_threshold"], options["power"]
        )
        return chosen
    return top_indices(power_mean(similarity, options["power"]), K, history.rows)


def history_scores(data: EvalData, history: History, ranked: np.ndarray) -> dict[str, float]:
    labels = data.labels
    sections = labels.section[ranked]
    in_interest = [np.sum(sections == code) for code in history.interests]
    hidden = labels.category[history.rows].sum(axis=0)
    hidden = np.asarray(hidden).ravel() > 0
    if hidden.any():
        shares = np.asarray(labels.category[ranked] @ hidden.astype(float)).ravel() > 0
        category = float(shares.mean())
    else:
        category = np.nan
    return {
        "section": float(sum(in_interest) / K),
        "category": category,
        "coverage": float(np.mean([count > 0 for count in in_interest])),
    }


def evaluate_histories(
    data: EvalData, weights: dict[str, float], histories: list[History], options: dict[str, Any]
) -> dict[str, np.ndarray]:
    rows = [history_scores(data, h, rank_history(data, weights, h, options)) for h in histories]
    return {name: np.array([row[name] for row in rows]) for name in rows[0]}


def history_primary(scores: dict[str, np.ndarray], interests: int) -> np.ndarray:
    names = ["section", "category"] + (["coverage"] if interests > 1 else [])
    return np.nanmean(np.stack([scores[name] for name in names]), axis=0)


# --- Report ----------------------------------------------------------------------------------------------


def evaluate(path: Path) -> dict[str, Any]:
    data = load_eval_data(path, EMBEDDING_VARIANTS)
    tuning, test = ~data.test, data.test
    report: dict[str, Any] = {"k": K, "test_articles": int(test.sum()), "single_article": {}}

    every = np.arange(len(data.titles))
    single: dict[str, dict[str, np.ndarray]] = {}
    for name in data.similarities:
        single[name] = neighbour_scores(data.labels, nearest(data.similarities[name], every, len(every)))
        log.info("Scored %s alone", name)
    embeddings = [name for name in data.similarities if name != "text"]
    best_embedding = max(
        embeddings, key=lambda name: summarize(single[name], tuning)["primary"], default=None
    )

    settings = Settings()
    tuned = {"text": {"text": 1.0}}
    for label, embedding in [(f"text + {e}", e) for e in embeddings]:
        weights = {"text": settings.weight("text"), embedding: settings.weight("embedding")}
        tuned[label] = weights
        single[label] = neighbour_scores(data.labels, nearest(combined(data, weights), every, len(every)))
        log.info(
            "Scored %s: %s (tuning primary %.4f)",
            label,
            weights,
            summarize(single[label], tuning)["primary"],
        )
    final_label = max(tuned, key=lambda label: summarize(single[label], tuning)["primary"])
    final_weights = tuned[final_label]

    report["single_article"] = {
        name: {
            "test": summarize(scores, test),
            "tuning": summarize(scores, tuning),
            "weights": tuned.get(name),
        }
        for name, scores in single.items()
    }
    report["best_embedding_alone"] = best_embedding
    report["final_signals"] = {"label": final_label, "weights": final_weights}
    report["single_article_comparisons_on_test"] = {
        f"{final_label} vs {other}": bootstrap(
            primary_per_query(single[final_label])[test], primary_per_query(single[other])[test]
        )
        for other in ["text"] + ([best_embedding] if best_embedding else [])
        if other != final_label
    }

    histories_report: dict[str, Any] = {}
    chosen_scores: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    tuning_totals: dict[str, list[float]] = defaultdict(list)
    for shape_name, shape in HISTORY_SHAPES.items():
        split_histories = {
            "tuning": sample_histories(data, tuning, shape, _HISTORIES, seed=1),
            "test": sample_histories(data, test, shape, _HISTORIES, seed=2),
        }
        shape_report = {}
        for agg_name, options in AGGREGATIONS.items():
            entry = {}
            for split, histories in split_histories.items():
                scores = evaluate_histories(data, final_weights, histories, options)
                primary = history_primary(scores, len(shape))
                entry[split] = {name: float(np.nanmean(v)) for name, v in scores.items()}
                entry[split]["primary"] = float(np.nanmean(primary))
                if split == "test":
                    chosen_scores[agg_name][shape_name] = primary
                else:
                    tuning_totals[agg_name].append(entry[split]["primary"])
            shape_report[agg_name] = entry
        histories_report[shape_name] = shape_report
        log.info("Scored histories: %s", shape_name)

    best_aggregation = f"interests t={settings.interest_threshold}, p={settings.power}"
    report["histories"] = histories_report
    report["final_aggregation"] = {"label": best_aggregation, **AGGREGATIONS[best_aggregation]}
    report["history_comparisons_on_test"] = {
        f"{best_aggregation} vs {other} ({shape})": bootstrap(
            chosen_scores[best_aggregation][shape], chosen_scores[other][shape]
        )
        for other in ("mean (p=1)", "max")
        if other != best_aggregation
        for shape in HISTORY_SHAPES
    }
    return report


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m wiki_recommender.recommender.evaluate")
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--output", type=Path, default=Path("data/processed/recommender_eval.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    report = evaluate(args.input)
    args.output.write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
    print("Single articles (test half):")
    for name, entry in report["single_article"].items():
        t = entry["test"]
        print(
            f"  {name:45} primary={t['primary']:.3f} section={t['section']:.3f} category={t['category']:.3f}"
        )
    print(f"Signals: {report['final_signals']}")
    for name, c in report["single_article_comparisons_on_test"].items():
        print(f"  {name}: {c['difference']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]")
    print("Histories (test half):")
    for shape, aggregations in report["histories"].items():
        print(f"  {shape}")
        for name, entry in aggregations.items():
            t = entry["test"]
            print(
                f"    {name:22} primary={t['primary']:.3f} section={t['section']:.3f} "
                f"category={t['category']:.3f} coverage={t['coverage']:.3f}"
            )
    print(f"Aggregation: {report['final_aggregation']}")
    for name, c in report["history_comparisons_on_test"].items():
        print(f"  {name}: {c['difference']:+.4f} [{c['low']:+.4f}, {c['high']:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
