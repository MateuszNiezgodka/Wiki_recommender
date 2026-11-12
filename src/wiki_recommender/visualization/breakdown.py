"""Graphical explanations of recommendations: where each score comes from.

With the default settings a score splits exactly into parts (see `recommender.engine`):

    score(c) = sum over terms t of w_text * u_t * c_t  +  w_links * mean cos_links  +  w_emb * mean cos_emb
             = (1 / |I|) * sum over v in I of s(v, c)

The first line splits it by signal and term (`plot_score_waterfall`, `plot_signal_breakdown`), the second
by visited article (`plot_visited_breakdown`, `plot_visited_heatmap`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from wiki_recommender.recommender.engine import Recommendation, RecommendationResult
from wiki_recommender.recommender.model import SIGNALS
from wiki_recommender.visualization.style import (
    BAR_HEIGHT,
    BLUES,
    INK,
    INK_SECONDARY,
    MUTED,
    SIGNAL_COLORS,
    SIGNAL_LABELS,
    SURFACE,
    apply_style,
    heading,
    series_color,
    shorten,
    value_grid,
)

_GAP = 1.5  # surface-colored gap between touching segments, in points


def _axes(ax: plt.Axes | None, size: tuple[float, float]) -> plt.Axes:
    if ax is not None:
        return ax
    apply_style()
    return plt.subplots(figsize=size)[1]


def _label_rows(ax: plt.Axes, labels: list[str]) -> None:
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)  # first row on top
    ax.spines["left"].set_visible(False)


def _recommendation_labels(result: RecommendationResult) -> list[str]:
    return [f"{r.rank}. {shorten(r.title, 30)}" for r in result.recommendations]


def _end_label(ax: plt.Axes, y: float, x: float, text: str, offset: float) -> None:
    ax.text(x + offset, y, text, va="center", ha="left", fontsize=9, color=INK)


def plot_score_waterfall(item: Recommendation, top_terms: int = 8, ax: plt.Axes | None = None) -> plt.Axes:
    """How one recommendation's score adds up: its strongest shared terms, the rest of the text signal,
    then the link and embedding signals. Each bar starts where the previous one ended."""
    terms = item.term_contributions[:top_terms]
    text_total = item.signal_contributions.get("text", 0.0)
    rest = max(text_total - sum(value for _, value in terms), 0.0)
    steps = [(term, value, "text") for term, value in terms]
    steps.append(("other shared terms", rest, "text"))
    steps += [(SIGNAL_LABELS[s], item.signal_contributions.get(s, 0.0), s) for s in ("links", "embedding")]

    ax = _axes(ax, (7.5, 0.34 * (len(steps) + 1) + 1.2))
    start = 0.0
    for row, (_, value, signal) in enumerate(steps):
        ax.barh(row, value, left=start, height=BAR_HEIGHT, color=SIGNAL_COLORS[signal])
        _end_label(ax, row, start + value, f"+{value:.3f}", item.score * 0.01)
        start += value
    total_row = len(steps)
    ax.barh(total_row, item.score, height=BAR_HEIGHT, color=INK_SECONDARY)
    _end_label(ax, total_row, item.score, f"{item.score:.3f}", item.score * 0.01)
    # Faint guides show where each step continues from.
    for row in range(1, total_row + 1):
        x = sum(value for _, value, _ in steps[:row])
        ax.plot([x, x], [row - 1 + BAR_HEIGHT / 2, row - BAR_HEIGHT / 2], color=MUTED, lw=0.8)

    _label_rows(ax, [label for label, _, _ in steps] + ["score"])
    ax.set_xlim(0, item.score * 1.15)
    ax.set_xlabel("contribution to the score")
    value_grid(ax)
    interest = ", ".join(shorten(title, 24) for title in item.interest)
    heading(
        ax,
        f"Why “{item.title}”?  Score {item.score:.3f}",
        f"compared with: {interest}",
        [Patch(color=SIGNAL_COLORS[s], label=SIGNAL_LABELS[s]) for s in SIGNALS],
    )
    return ax


def plot_signal_breakdown(result: RecommendationResult, ax: plt.Axes | None = None) -> plt.Axes:
    """Every recommendation's score as a stack of its weighted text, link and embedding parts."""
    items = result.recommendations
    ax = _axes(ax, (8, 0.36 * len(items) + 1.4))
    left = np.zeros(len(items))
    for signal in SIGNALS:
        values = np.array([r.signal_contributions.get(signal, 0.0) for r in items])
        ax.barh(
            range(len(items)),
            values,
            left=left,
            height=BAR_HEIGHT,
            color=SIGNAL_COLORS[signal],
            edgecolor=SURFACE,
            linewidth=_GAP,
            label=SIGNAL_LABELS[signal],
        )
        left += values
    top = max((r.score for r in items), default=1.0)
    for row, r in enumerate(items):
        _end_label(ax, row, r.score, f"{r.score:.3f}", top * 0.01)
    _label_rows(ax, _recommendation_labels(result))
    ax.set_xlim(0, top * 1.12)
    ax.set_xlabel("score, split by signal")
    value_grid(ax)
    handles = [Patch(color=SIGNAL_COLORS[s], label=SIGNAL_LABELS[s]) for s in SIGNALS]
    heading(ax, "What each signal adds", handles=handles)
    return ax


def _visited_colors(result: RecommendationResult) -> dict[str, str]:
    return {v.title: series_color(i) for i, v in enumerate(result.visited)}


def plot_visited_breakdown(result: RecommendationResult, ax: plt.Axes | None = None) -> plt.Axes:
    """Every recommendation's score split by visited article: each one of its interest adds s(v, c) / |I|."""
    items = result.recommendations
    colors = _visited_colors(result)
    ax = _axes(ax, (8, 0.36 * len(items) + 1.6))
    for row, r in enumerate(items):
        left = 0.0
        share = r.score / sum(r.by_visited[t] for t in r.interest) if r.interest else 0.0
        for title in r.interest:
            value = r.by_visited[title] * share
            ax.barh(
                row,
                value,
                left=left,
                height=BAR_HEIGHT,
                color=colors[title],
                edgecolor=SURFACE,
                linewidth=_GAP,
            )
            left += value
    top = max((r.score for r in items), default=1.0)
    for row, r in enumerate(items):
        _end_label(ax, row, r.score, f"{r.score:.3f}", top * 0.01)
    _label_rows(ax, _recommendation_labels(result))
    ax.set_xlim(0, top * 1.12)
    ax.set_xlabel("score, split by the visited article it comes from")
    value_grid(ax)
    handles = [Patch(color=colors[v.title], label=shorten(v.title, 24)) for v in result.visited]
    heading(ax, "Which visited article it comes from", handles=handles, ncols=min(len(handles), 4))
    return ax


def plot_visited_heatmap(result: RecommendationResult, ax: plt.Axes | None = None) -> plt.Axes:
    """Similarity s(v, c) of every visited article to every recommendation.

    Outlined cells are the articles the score is averaged over (the candidate's interest)."""
    items = result.recommendations
    visited = [v.title for v in result.visited]
    matrix = np.array([[r.by_visited[title] for r in items] for title in visited])
    ax = _axes(ax, (0.62 * len(items) + 2.8, 0.45 * len(visited) + 2.4))
    image = ax.imshow(matrix, cmap=BLUES, vmin=0, vmax=max(matrix.max(), 1e-6), aspect="auto")
    threshold = 0.6 * matrix.max()
    for i, title in enumerate(visited):
        for j, r in enumerate(items):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if matrix[i, j] > threshold else INK,
            )
            if title in r.interest:
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6, zorder=3)
                )
    ax.set_xticks(
        range(len(items)), [f"{r.rank}. {shorten(r.title, 22)}" for r in items], rotation=40, ha="right"
    )
    ax.set_yticks(range(len(visited)), [shorten(t, 28) for t in visited])
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.set_label("similarity s(v, c)", color=INK_SECONDARY)
    heading(
        ax,
        "Similarity of each visited article to each recommendation",
        "outlined: the visited articles a recommendation's score is averaged over",
    )
    return ax


def explain(result: RecommendationResult, rank: int = 1, top_terms: int = 8) -> plt.Figure:
    """One recommendation: its score waterfall beside its similarity to each visited article."""
    apply_style()
    item = result.recommendations[rank - 1]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 0.34 * (top_terms + 5) + 1.4), gridspec_kw={"width_ratios": [3, 2], "wspace": 0.55}
    )
    plot_score_waterfall(item, top_terms, ax=left)
    colors = _visited_colors(result)
    titles = [v.title for v in result.visited]
    values = [item.by_visited[t] for t in titles]
    right.barh(range(len(titles)), values, height=BAR_HEIGHT, color=[colors[t] for t in titles])
    for row, (title, value) in enumerate(zip(titles, values, strict=True)):
        weight = "bold" if title in item.interest else "normal"
        right.text(value + 0.01, row, f"{value:.3f}", va="center", fontsize=9, color=INK, fontweight=weight)
    _label_rows(right, [shorten(t, 26) for t in titles])
    right.set_xlim(0, max(max(values, default=0), 0.01) * 1.25)
    right.set_xlabel("similarity s(v, c)")
    value_grid(right)
    right.set_title("Similarity to each visited article", loc="left")
    return fig


def explain_result(result: RecommendationResult) -> plt.Figure:
    """The whole list at once: scores split by signal (left) and by visited article (right)."""
    apply_style()
    height = 0.36 * len(result.recommendations) + 1.8
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, height), sharey=True, gridspec_kw={"wspace": 0.08})
    plot_signal_breakdown(result, ax=left)
    plot_visited_breakdown(result, ax=right)
    right.tick_params(labelleft=False)
    return fig
