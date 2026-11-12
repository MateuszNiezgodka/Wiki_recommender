"""One look for every chart of the report: colors, thin marks, quiet axes.

Colors follow the job they do. Identity (signals, visited articles, lemma vs stem) takes the categorical
slots in their fixed order, magnitude (heatmaps) takes one blue ramp from light to dark, and anything
that is context rather than data is gray. Text always stays in ink colors, never in a series color.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIGURES_DIR = Path("reports/figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#d6d5ce"  # gray for marks that are background, not the subject

# Categorical slots, always used in this order. Their order keeps neighbours apart for color-blind readers.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#a8a7a0"  # everything past the eighth slot folds into "other"

SIGNAL_COLORS = {"text": SERIES[0], "links": SERIES[1], "embedding": SERIES[2]}
SIGNAL_LABELS = {"text": "Text (TF-IDF)", "links": "Links", "embedding": "Embedding"}

BLUES = LinearSegmentedColormap.from_list(
    "blues",
    ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

BAR_HEIGHT = 0.62  # share of a category band a bar fills; the rest is air


def apply_style() -> None:
    """Set matplotlib defaults for the report. Safe to call more than once."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 5,
            "ytick.major.pad": 5,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0,
        }
    )


def value_grid(ax: plt.Axes, axis: str = "x") -> None:
    """Hairline gridlines along the value axis only, behind the marks."""
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)


def heading(
    ax: plt.Axes,
    title: str,
    subtitle: str | None = None,
    handles: list | None = None,
    ncols: int | None = None,
) -> None:
    """Title, optional subtitle and optional legend, stacked above the plot and aligned to its left edge."""
    offset = 4.0
    if handles:
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncols=ncols or len(handles),
            borderaxespad=0.2,
            handlelength=1.2,
            columnspacing=1.4,
        )
        offset += 20
    if subtitle:
        ax.annotate(
            subtitle,
            (0, 1),
            xycoords="axes fraction",
            xytext=(0, offset),
            textcoords="offset points",
            fontsize=9,
            color=INK_SECONDARY,
        )
        offset += 15
    ax.set_title(title, loc="left", pad=offset + 2)


def series_color(position: int) -> str:
    """Color of the n-th identity series. Past the eighth it is the shared "other" gray."""
    return SERIES[position] if position < len(SERIES) else OTHER


def save(fig: plt.Figure, name: str, directory: Path | None = FIGURES_DIR) -> Path | None:
    """Write the figure as PNG into `directory` (the report's figures). `directory=None` skips saving."""
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    return path


def shorten(text: str, limit: int = 32) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
