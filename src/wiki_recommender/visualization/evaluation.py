"""Charts of the offline measurements behind every setting (the `*_eval.json` files).

Each score is the share of an article's 10 nearest neighbours that are related to it by a signal the
representation never sees: the same Vital articles subsection, or a shared category from a held-out half.
The chosen option is drawn in blue, the alternatives in gray.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wiki_recommender.visualization.style import (
    BAR_HEIGHT,
    CONTEXT,
    INK,
    INK_SECONDARY,
    SERIES,
    apply_style,
    value_grid,
)

REPRESENTATION_EVAL_PATH = Path("data/processed/representation_eval.json")
RECOMMENDER_EVAL_PATH = Path("data/processed/recommender_eval.json")

_HISTORIES = ["one interest, 3 articles", "one interest, 5 articles", "two interests, 3+2"]
_AGGREGATIONS = {
    "mean (p=1)": "mean (centroid)",
    "power p=4": "power mean, p = 4",
    "max": "maximum",
    "interests t=0.3, p=1.0": "interests, t = 0.3",
}


def load_eval(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ranked_bars(ax: plt.Axes, labels: list[str], values: list[float], chosen: str, xlabel: str) -> None:
    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = [SERIES[0] if label == chosen else CONTEXT for label in labels]
    ax.barh(labels, values, height=BAR_HEIGHT, color=colors)
    for row, value in enumerate(values):
        weight = "bold" if labels[row] == chosen else "normal"
        ax.text(value + 0.003, row, f"{value:.3f}", va="center", fontsize=9, color=INK, fontweight=weight)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(0, max(values) * 1.15)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlabel)
    value_grid(ax)


def plot_representation_eval(results: dict, ax: plt.Axes | None = None) -> plt.Axes:
    """Stage 2: which text representation finds related articles best (test half)."""
    if ax is None:
        apply_style()
        ax = plt.subplots(figsize=(8, 3.6))[1]
    names = list(results["test"])
    values = [results["test"][name]["primary"] for name in names]
    _ranked_bars(ax, names, values, "final", "share of the 10 nearest neighbours that are related")
    ax.set_title(f"Text representations, {results['test_articles']:,} test articles")
    return ax


def plot_signal_eval(results: dict, ax: plt.Axes | None = None) -> plt.Axes:
    """Stage 3: single signals and their combinations, for one visited article (test half)."""
    if ax is None:
        apply_style()
        ax = plt.subplots(figsize=(8, 4.6))[1]
    table = results["single_article"]
    names = list(table)
    values = [table[name]["test"]["primary"] for name in names]
    _ranked_bars(ax, names, values, results["final_signals"]["label"], "share of related nearest neighbours")
    ax.set_title("Signals and their combinations")
    return ax


def plot_aggregation_eval(results: dict) -> plt.Figure:
    """Stage 3: aggregations of a history, for three kinds of history, and the interests they serve."""
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(15, 2.9), sharey=True, gridspec_kw={"wspace": 0.12})
    chosen = results["final_aggregation"]["label"]
    labels = list(_AGGREGATIONS.values())
    panels = [(h, "primary", h) for h in _HISTORIES] + [
        ("two interests, 3+2", "coverage", "coverage, 2 interests")
    ]
    for ax, (history, metric, title) in zip(axes, panels, strict=True):
        values = [results["histories"][history][key]["test"][metric] for key in _AGGREGATIONS]
        colors = [SERIES[0] if key == chosen else CONTEXT for key in _AGGREGATIONS]
        ax.barh(labels, values, height=BAR_HEIGHT, color=colors)
        for row, value in enumerate(values):
            ax.text(value + 0.01, row, f"{value:.3f}", va="center", fontsize=9, color=INK)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlim(0, max(values) * 1.3)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        "Aggregating a history: share of related recommendations (test half)",
        x=0.01,
        ha="left",
        fontweight="bold",
        y=1.06,
    )
    return fig


def plot_interest_threshold(results: dict, ax: plt.Axes | None = None) -> plt.Axes:
    """Interest aggregation (p = 1) as the clustering threshold t varies, against the plain mean."""
    if ax is None:
        apply_style()
        ax = plt.subplots(figsize=(8, 3.6))[1]
    for position, history in enumerate(_HISTORIES):
        table = results["histories"][history]
        keys = [k for k in table if k.startswith("interests") and k.endswith("p=1.0")]
        thresholds = [float(k.split("t=")[1].split(",")[0]) for k in keys]
        values = [table[k]["test"]["primary"] for k in keys]
        color = SERIES[position]
        ax.plot(thresholds, values, color=color, marker="o", ms=5, mec="#fcfcfb", mew=1.5)
        ax.text(thresholds[-1] + 0.006, values[-1], history, fontsize=9, color=INK_SECONDARY, va="center")
        mean = table["mean (p=1)"]["test"]["primary"]
        ax.axhline(mean, color=color, lw=1, alpha=0.45)
    chosen = results["final_aggregation"]["interest_threshold"]
    ax.axvline(chosen, color=INK, lw=1)
    ax.text(chosen + 0.004, ax.get_ylim()[0] + 0.005, f"chosen t = {chosen}", fontsize=9, color=INK)
    ax.set_xlim(min(thresholds) - 0.02, max(thresholds) + 0.16)
    ax.set_xlabel("interest threshold t (articles stay together while their mean similarity is at least t)")
    ax.set_ylabel("share of related recommendations")
    value_grid(ax, "y")
    ax.set_title("Interest threshold (thin lines: plain mean)")
    return ax
