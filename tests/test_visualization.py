import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from test_recommender import model, offline  # noqa: E402, F401  (shared toy-corpus fixtures)

from wiki_recommender.visualization import breakdown, corpus, evaluation  # noqa: E402
from wiki_recommender.visualization.style import save  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def mixed(offline):  # noqa: F811
    """A history with two interests."""
    return offline.recommend(["Dog", "Wolf", "Electron"], top_k=4)


def test_waterfall_ends_at_the_score(offline):  # noqa: F811
    item = offline.recommend(["Electron", "Photon"], top_k=1).recommendations[0]
    ax = breakdown.plot_score_waterfall(item)
    bars = ax.patches
    steps, total = bars[:-1], bars[-1]
    assert steps[-1].get_x() + steps[-1].get_width() == pytest.approx(item.score, abs=1e-3)
    assert total.get_width() == pytest.approx(item.score)


def test_visited_breakdown_stacks_add_up_to_each_score(mixed):
    ax = breakdown.plot_visited_breakdown(mixed)
    ends: dict[float, float] = {}
    for bar in ax.patches:
        row = round(bar.get_y() + bar.get_height() / 2)
        ends[row] = max(ends.get(row, 0.0), bar.get_x() + bar.get_width())
    assert [ends[row] for row in sorted(ends)] == pytest.approx(
        [r.score for r in mixed.recommendations], abs=1e-3
    )


def test_signal_breakdown_has_one_stack_per_recommendation(mixed):
    ax = breakdown.plot_signal_breakdown(mixed)
    assert len(ax.patches) == 3 * len(mixed.recommendations)


def test_heatmap_outlines_the_interest_of_each_recommendation(mixed):
    ax = breakdown.plot_visited_heatmap(mixed)
    outlined = sum(len(r.interest) for r in mixed.recommendations)
    assert len(ax.patches) == outlined


def test_combined_figures_render_and_save(mixed, tmp_path):
    for name, figure in (("result", breakdown.explain_result(mixed)), ("one", breakdown.explain(mixed, 2))):
        assert save(figure, name, tmp_path).exists()
    assert save(plt.figure(), "skipped", None) is None


def test_similarity_sample_and_topic_matrix(model):  # noqa: F811
    sample = corpus.similarity_sample(model, size=8, seed=1)
    assert sample.combined.shape == (8, 8)
    assert len(sample.pairs(sample.combined)) == 8 * 7 // 2
    matrix = corpus.topic_similarity_matrix(sample)
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)
    for figure in (
        corpus.plot_similarity_distributions(sample),
        corpus.plot_same_vs_other_topic(sample).figure,
        corpus.plot_nearest_neighbour(model, sample).figure,
        corpus.plot_topic_heatmap(matrix).figure,
    ):
        assert figure.axes


def test_term_charts_use_words_or_categories(model):  # noqa: F811
    weights = corpus.mean_tfidf(model)
    words = corpus.top_terms(model, weights, 5)
    categories = corpus.top_terms(model, weights, 5, categories=True)
    assert len(words) == 5 and all(not term.startswith("cat:") for term, _ in words)
    assert {term for term, _ in categories} <= {"canine", "pet", "feline", "physics", "particle", "city"}
    corpus.plot_top_tfidf(model, count=4)
    corpus.plot_topic_terms(model, count=3)


def test_embedding_map_has_a_point_per_sampled_article(model):  # noqa: F811
    points = corpus.embedding_map(model, size=10)
    assert len(points) == 10 and set(points.columns) == {"x", "y", "title", "topic"}
    corpus.plot_embedding_map(points)


def test_corpus_frame_charts():
    frame = pd.DataFrame(
        {
            "topic": ["A", "A", "B", "B"],
            "vital_level": [3, 4, 4, 5],
            "word_count": [200, 1500, 900, 4000],
            "token_count": [100, 800, 450, 2100],
        }
    )
    stats = {
        "vocabulary": {"raw_casefolded": 100, "raw_without_stopwords": 90, "lemmas": 70, "stems": 60},
        "top_lemmas": [["dog", 5], ["cat", 3]],
        "top_stems": [["dog", 5], ["cat", 3]],
    }
    corpus.plot_topic_composition(frame)
    corpus.plot_length_histograms(frame)
    corpus.plot_length_by_topic(frame)
    corpus.plot_vocabulary(stats)
    corpus.plot_top_words(stats, count=2)


def _score(primary: float) -> dict:
    return {"test": {"primary": primary, "coverage": primary}}


def test_evaluation_charts():
    representation = {"test_articles": 10, "test": {"final": {"primary": 0.3}, "full text": {"primary": 0.2}}}
    recommender = {
        "final_signals": {"label": "text"},
        "single_article": {"text": _score(0.3), "links": _score(0.2)},
        "final_aggregation": {"label": "interests t=0.3, p=1.0", "interest_threshold": 0.3},
        "histories": {
            history: {
                "mean (p=1)": _score(0.4),
                "power p=4": _score(0.4),
                "max": _score(0.3),
                "interests t=0.2, p=1.0": _score(0.4),
                "interests t=0.3, p=1.0": _score(0.5),
            }
            for history in ("one interest, 3 articles", "one interest, 5 articles", "two interests, 3+2")
        },
    }
    evaluation.plot_representation_eval(representation)
    evaluation.plot_signal_eval(recommender)
    evaluation.plot_aggregation_eval(recommender)
    evaluation.plot_interest_threshold(recommender)
