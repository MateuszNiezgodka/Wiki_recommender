"""Statistics of the corpus for the report: its composition, lengths, vocabulary and similarities.

Loaders return plain data (a DataFrame, dicts, arrays) and every `plot_*` function draws one chart from it,
so the numbers behind each chart can also be shown as a table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from wiki_recommender.preprocessing.clean import CATEGORY_TOKEN_PREFIX
from wiki_recommender.recommender.model import (
    PROCESSED_PATH,
    SIGNALS,
    RecommenderModel,
    combine,
    signal_similarities,
)
from wiki_recommender.visualization.style import (
    BAR_HEIGHT,
    BLUES,
    CONTEXT,
    INK,
    INK_SECONDARY,
    SERIES,
    SIGNAL_COLORS,
    SIGNAL_LABELS,
    SURFACE,
    apply_style,
    heading,
    value_grid,
)

RAW_STATS_PATH = Path("data/raw/articles_raw.stats.json")
PROCESSED_STATS_PATH = Path("data/processed/articles_processed.stats.json")


# --- Loading ---------------------------------------------------------------------------------------------


def load_corpus_frame(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """One row per article with its topic, level and sizes. The token lists themselves are not read."""
    columns = [
        "title",
        "topic",
        "seed_section",
        "vital_level",
        "word_count",
        "token_count",
        "lead_token_count",
    ]
    table = pq.read_table(path, columns=[*columns, "categories", "outlinks", "sections"])
    frame = table.select(columns).to_pandas()
    for name in ("categories", "outlinks", "sections"):
        frame[name] = [len(values) for values in table[name].to_pylist()]
    return frame


def load_stats(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _median_label(ax: plt.Axes, x: float, text: str) -> None:
    """Label a median line just above the plot, where it cannot cover the bars."""
    ax.annotate(
        text,
        (x, 1),
        xycoords=("data", "axes fraction"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=INK,
    )


def _row_axes(count: int, size: tuple[float, float], **kwargs) -> tuple[plt.Figure, np.ndarray]:
    apply_style()
    fig, axes = plt.subplots(1, count, figsize=size, **kwargs)
    return fig, np.atleast_1d(axes)


# --- Composition and lengths -----------------------------------------------------------------------------


def plot_topic_composition(frame: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Articles per topic, split by Vital articles level (1 = most vital). Levels are ordered, so one ramp."""
    if ax is None:
        apply_style()
        ax = plt.subplots(figsize=(8.5, 4.6))[1]
    counts = frame.groupby(["topic", "vital_level"]).size().unstack(fill_value=0).sort_index(ascending=False)
    levels = sorted(counts.columns)
    shades = [BLUES(x) for x in np.linspace(0.95, 0.35, len(levels))]
    left = np.zeros(len(counts))
    for level, color in zip(levels, shades, strict=True):
        values = counts[level].to_numpy()
        ax.barh(
            counts.index,
            values,
            left=left,
            height=BAR_HEIGHT,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            label=f"level {level}",
        )
        left += values
    for row, total in enumerate(left):
        ax.text(total + 8, row, f"{int(total):,}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, left.max() * 1.1)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("articles")
    value_grid(ax)
    heading(
        ax,
        f"{len(frame):,} articles in {counts.shape[0]} topics, by Vital articles level",
        handles=ax.get_legend_handles_labels()[0],
    )
    return ax


def plot_length_histograms(frame: pd.DataFrame) -> plt.Figure:
    """Distribution of article lengths before (words) and after preprocessing (tokens), log scale."""
    fig, axes = _row_axes(2, (12, 3.8), sharey=True)
    for ax, column, label, color in (
        (axes[0], "word_count", "words in raw text", SERIES[0]),
        (axes[1], "token_count", "tokens after preprocessing", SERIES[1]),
    ):
        values = frame[column].to_numpy()
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 45)
        ax.hist(values, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.8)
        median = float(np.median(values))
        ax.axvline(median, color=INK, lw=1)
        _median_label(ax, median, f"median {median:,.0f}")
        ax.set_xscale("log")
        ax.set_xlabel(label)
        value_grid(ax, "y")
    axes[0].set_ylabel("articles")
    axes[0].set_title("Article length")
    kept = frame["token_count"].sum() / frame["word_count"].sum()
    axes[1].set_title(f"Preprocessing keeps {kept:.0%} of the words")
    return fig


def plot_length_by_topic(frame: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Token counts per topic as box plots (box = middle half, line = median, whiskers = 5-95 %)."""
    if ax is None:
        apply_style()
        ax = plt.subplots(figsize=(8.5, 4.8))[1]
    order = frame.groupby("topic")["token_count"].median().sort_values().index
    data = [frame.loc[frame["topic"] == topic, "token_count"].to_numpy() for topic in order]
    ax.boxplot(
        data,
        orientation="horizontal",
        whis=(5, 95),
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": BLUES(0.3), "edgecolor": SERIES[0], "linewidth": 1},
        medianprops={"color": INK, "linewidth": 1.6},
        whiskerprops={"color": SERIES[0], "linewidth": 1},
        capprops={"color": SERIES[0], "linewidth": 1},
    )
    ax.set_yticks(range(1, len(order) + 1), order)
    ax.set_xscale("log")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("tokens per article")
    value_grid(ax)
    ax.set_title("Article length by topic")
    return ax


# --- Vocabulary and frequent words -----------------------------------------------------------------------


def plot_vocabulary(stats: dict, ax: plt.Axes | None = None) -> plt.Axes:
    """Distinct terms after each preprocessing step."""
    if ax is None:
        apply_style()
        ax = plt.subplots(figsize=(7.5, 2.8))[1]
    vocabulary = stats["vocabulary"]
    steps = [
        ("raw, case-folded", vocabulary["raw_casefolded"]),
        ("without stopwords", vocabulary["raw_without_stopwords"]),
        ("lemmas", vocabulary["lemmas"]),
        ("stems", vocabulary["stems"]),
    ]
    ax.barh([label for label, _ in steps], [value for _, value in steps], height=BAR_HEIGHT, color=SERIES[0])
    first = steps[0][1]
    for row, (_, value) in enumerate(steps):
        change = "" if row == 0 else f"  ({value / first - 1:+.1%})"
        ax.text(value + first * 0.01, row, f"{value:,}{change}", va="center", fontsize=9, color=INK)
    ax.set_ylim(len(steps) - 0.5, -0.5)
    ax.set_xlim(0, first * 1.3)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("distinct terms")
    value_grid(ax)
    ax.set_title("Vocabulary size after each step")
    return ax


def plot_top_words(stats: dict, count: int = 25) -> plt.Figure:
    """The most frequent lemmas and stems side by side."""
    fig, axes = _row_axes(2, (11, 0.24 * count + 1.4), gridspec_kw={"wspace": 0.35})
    for ax, key, label, color in (
        (axes[0], "top_lemmas", "lemmas", SERIES[0]),
        (axes[1], "top_stems", "stems", SERIES[1]),
    ):
        words, counts = zip(*stats[key][:count], strict=True)
        ax.barh(words, counts, height=BAR_HEIGHT, color=color)
        ax.set_ylim(count - 0.5, -0.5)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("occurrences in the corpus")
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}K" if x else "0")
        value_grid(ax)
        ax.set_title(f"Most frequent {label}")
    return fig


def mean_tfidf(model: RecommenderModel, rows: np.ndarray | None = None) -> np.ndarray:
    """Mean TF-IDF weight of every feature over the given articles (all by default)."""
    matrix = model.vectors.text if rows is None else model.vectors.text[rows]
    return np.asarray(matrix.mean(axis=0)).ravel()


def top_features(
    model: RecommenderModel, weights: np.ndarray, count: int, categories: bool = False
) -> np.ndarray:
    """Columns of the `count` highest weights, among words or among category tokens."""
    names = model.text_vectorizer.feature_names.astype(str)
    is_category = np.char.startswith(names, CATEGORY_TOKEN_PREFIX)
    keep = np.flatnonzero(is_category if categories else ~is_category)
    return keep[np.argsort(-weights[keep])[:count]]


def top_terms(model: RecommenderModel, weights: np.ndarray, count: int, categories: bool = False) -> list:
    """The `count` features with the highest weight, as (name, weight)."""
    names = model.text_vectorizer.feature_names
    return [
        (str(names[i]).removeprefix(CATEGORY_TOKEN_PREFIX), float(weights[i]))
        for i in top_features(model, weights, count, categories)
    ]


def plot_top_tfidf(model: RecommenderModel, count: int = 25) -> plt.Figure:
    """Terms with the highest mean TF-IDF weight: frequent, but also concentrated in some articles."""
    weights = mean_tfidf(model)
    fig, axes = _row_axes(2, (11, 0.24 * count + 1.4), gridspec_kw={"wspace": 0.45})
    for ax, categories, title in ((axes[0], False, "words"), (axes[1], True, "category tokens")):
        terms = top_terms(model, weights, count, categories)
        ax.barh([t for t, _ in terms], [w for _, w in terms], height=BAR_HEIGHT, color=SERIES[0])
        ax.set_ylim(count - 0.5, -0.5)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("mean TF-IDF weight")
        value_grid(ax)
        ax.set_title(f"Highest mean TF-IDF: {title}")
    return fig


def plot_topic_terms(model: RecommenderModel, count: int = 8) -> plt.Figure:
    """For every topic, the words whose mean TF-IDF weight is highest there relative to the whole corpus."""
    topics = np.array(model.index.topics)
    names = sorted(set(topics))
    overall = mean_tfidf(model) + 1e-4
    apply_style()
    columns = 4
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, rows * (0.24 * count + 1.1)))
    for ax, topic in zip(axes.flat, names, strict=False):
        inside = mean_tfidf(model, np.flatnonzero(topics == topic))
        # Weight in the topic, favouring terms that are rare elsewhere: a log-odds-like lift.
        lift = inside * np.log((inside + 1e-4) / overall)
        best = top_features(model, lift, count)
        best = best[np.argsort(-inside[best])]  # longest bar on top
        labels = model.text_vectorizer.feature_names[best]
        ax.barh([str(n) for n in labels], inside[best], height=BAR_HEIGHT, color=SERIES[0])
        ax.set_ylim(count - 0.5, -0.5)
        ax.set_xticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(topic, fontsize=10)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Most characteristic words of each topic (bar = mean TF-IDF weight in the topic)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# --- Similarities between documents ----------------------------------------------------------------------


@dataclass
class SimilaritySample:
    """Similarities among a random sample of articles, per signal and combined."""

    rows: np.ndarray  # corpus rows of the sampled articles
    topics: np.ndarray
    per_signal: dict[str, np.ndarray]  # (n x n) per signal
    combined: np.ndarray  # (n x n) weighted similarity s(a, b)

    def pairs(self, matrix: np.ndarray) -> np.ndarray:
        """Values of the distinct pairs (upper triangle)."""
        return matrix[np.triu_indices(len(self.rows), k=1)]

    @property
    def same_topic(self) -> np.ndarray:
        return self.pairs(self.topics[:, None] == self.topics[None, :])


def similarity_sample(model: RecommenderModel, size: int = 2000, seed: int = 0) -> SimilaritySample:
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(model.index), size=min(size, len(model.index)), replace=False))
    vectors = model.vectors.rows(rows)
    per_signal = signal_similarities(vectors, vectors)
    return SimilaritySample(
        rows=rows,
        topics=np.array(model.index.topics)[rows],
        per_signal=per_signal,
        combined=combine(per_signal, model.settings),
    )


def plot_similarity_distributions(sample: SimilaritySample) -> plt.Figure:
    """Histogram of pairwise similarity for each signal and for the combined score."""
    fig, axes = _row_axes(len(SIGNALS) + 1, (12, 3.2), sharey=False)
    panels = [(SIGNAL_LABELS[s], sample.per_signal[s], SIGNAL_COLORS[s]) for s in SIGNALS]
    panels.append(("Combined s(a, b)", sample.combined, INK_SECONDARY))
    for ax, (label, matrix, color) in zip(axes, panels, strict=True):
        values = sample.pairs(matrix)
        ax.hist(values, bins=60, range=(0, max(values.max(), 0.05)), color=color, edgecolor=SURFACE, lw=0.5)
        ax.set_yscale("log")
        ax.set_xlabel("cosine similarity")
        heading(ax, label, f"mean {values.mean():.3f}, median {np.median(values):.3f}")
        value_grid(ax, "y")
    axes[0].set_ylabel("article pairs (log)")
    fig.suptitle(
        f"Similarity between {len(sample.rows):,} random articles",
        x=0.01,
        ha="left",
        fontweight="bold",
        y=1.08,
    )
    return fig


def plot_same_vs_other_topic(sample: SimilaritySample, ax: plt.Axes | None = None) -> plt.Axes:
    """Combined similarity of article pairs from the same topic versus different topics."""
    if ax is None:
        apply_style()
        ax = plt.subplots(figsize=(8, 3.6))[1]
    values = sample.pairs(sample.combined)
    same = sample.same_topic
    bins = np.linspace(0, values.max(), 60)
    for mask, label, color in ((~same, "different topics", SERIES[1]), (same, "same topic", SERIES[0])):
        ax.hist(
            values[mask],
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            color=color,
            label=f"{label} (mean {values[mask].mean():.3f})",
        )
    ax.set_xlabel("combined similarity s(a, b)")
    ax.set_ylabel("density")
    value_grid(ax, "y")
    ax.legend(loc="upper right")
    ax.set_title("Articles of the same topic are more similar")
    return ax


def plot_nearest_neighbour(
    model: RecommenderModel, sample: SimilaritySample, ax: plt.Axes | None = None
) -> plt.Axes:
    """Similarity of each sampled article to its closest other article in the whole corpus."""
    if ax is None:
        apply_style()
        ax = plt.subplots(figsize=(8, 3.4))[1]
    queries = model.vectors.rows(sample.rows)
    combined = combine(signal_similarities(queries, model.vectors), model.settings)
    combined[np.arange(len(sample.rows)), sample.rows] = -1  # an article is not its own neighbour
    best = combined.max(axis=1)
    ax.hist(best, bins=50, color=SERIES[0], edgecolor=SURFACE, lw=0.6)
    median = float(np.median(best))
    ax.axvline(median, color=INK, lw=1)
    _median_label(ax, median, f"median {median:.3f}")
    ax.set_xlabel("similarity to the nearest other article")
    ax.set_ylabel("articles")
    value_grid(ax, "y")
    ax.set_title("How close is the best possible recommendation?")
    return ax


def topic_similarity_matrix(sample: SimilaritySample) -> pd.DataFrame:
    """Mean combined similarity between articles of two topics (diagonal: distinct pairs of one topic)."""
    names = sorted(set(sample.topics))
    matrix = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        rows_a = np.flatnonzero(sample.topics == a)
        for j, b in enumerate(names):
            rows_b = np.flatnonzero(sample.topics == b)
            block = sample.combined[np.ix_(rows_a, rows_b)]
            if i == j:
                block = block[~np.eye(len(rows_a), dtype=bool)]
            matrix[i, j] = block.mean()
    return pd.DataFrame(matrix, index=names, columns=names)


def plot_topic_heatmap(matrix: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        apply_style()
        ax = plt.subplots(figsize=(8.5, 7))[1]
    values = matrix.to_numpy()
    low, high = values.min(), values.max()  # the range of the data, so small differences show
    image = ax.imshow(values, cmap=BLUES, vmin=low, vmax=high)

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value = matrix.iat[i, j]
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if value > low + 0.6 * (high - low) else INK,
            )
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.set_label("mean similarity s(a, b)", color=INK_SECONDARY)
    ax.set_title("Mean similarity between topics")
    return ax


# --- Map of the corpus -----------------------------------------------------------------------------------


def embedding_map(model: RecommenderModel, size: int = 3000, seed: int = 0) -> pd.DataFrame:
    """2D t-SNE map of a sample of article embeddings, with titles and topics."""
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(model.index), size=min(size, len(model.index)), replace=False))
    points = TSNE(
        n_components=2, metric="cosine", init="pca", perplexity=min(30, len(rows) - 1), random_state=seed
    ).fit_transform(model.vectors.embedding[rows])
    return pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "title": [model.index.titles[r] for r in rows],
            "topic": [model.index.topics[r] for r in rows],
        }
    )


def plot_embedding_map(points: pd.DataFrame, labels_per_topic: int = 1) -> plt.Figure:
    """One small map per topic: its articles in blue over the rest of the corpus in gray.

    Eleven topics are too many to tell apart by color in one scatter, so each gets its own panel."""
    apply_style()
    names = sorted(points["topic"].unique())
    columns = 4
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, rows * 3.3))
    rng = np.random.default_rng(0)
    for ax, topic in zip(axes.flat, names, strict=False):
        inside = points["topic"] == topic
        ax.scatter(points.loc[~inside, "x"], points.loc[~inside, "y"], s=3, color=CONTEXT, linewidths=0)
        ax.scatter(points.loc[inside, "x"], points.loc[inside, "y"], s=6, color=SERIES[0], linewidths=0)
        chosen = points[inside].iloc[rng.choice(int(inside.sum()), size=labels_per_topic, replace=False)]
        for _, point in chosen.iterrows():
            ax.annotate(
                point["title"][:22],
                (point["x"], point["y"]),
                fontsize=7,
                color=INK,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(topic, fontsize=10)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    fig.suptitle("t-SNE map of article embeddings, one panel per topic", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig
