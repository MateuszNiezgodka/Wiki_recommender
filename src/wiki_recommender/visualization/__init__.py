"""Stage 5: corpus statistics and graphical explanations of recommendations.

    from wiki_recommender.recommender import recommend
    from wiki_recommender.visualization import explain, explain_result

    result = recommend(["Dog", "Cat"])
    explain_result(result)   # every score split by signal and by visited article
    explain(result, rank=1)  # one score as a waterfall of terms and signals

`corpus` holds the database statistics and `evaluation` the charts of the offline measurements.
"""

from wiki_recommender.visualization.breakdown import (
    explain,
    explain_result,
    plot_score_waterfall,
    plot_signal_breakdown,
    plot_visited_breakdown,
    plot_visited_heatmap,
)
from wiki_recommender.visualization.style import apply_style, save

__all__ = [
    "apply_style",
    "explain",
    "explain_result",
    "plot_score_waterfall",
    "plot_signal_breakdown",
    "plot_visited_breakdown",
    "plot_visited_heatmap",
    "save",
]
