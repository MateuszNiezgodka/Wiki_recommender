"""Stage 3: recommend articles similar to a history of visited ones.

    from wiki_recommender.recommender import recommend
    print(recommend(["Dog", "https://en.wikipedia.org/wiki/Cat"], top_k=10))

Run ``python -m wiki_recommender.recommender --help`` for the command line, and
``python -m wiki_recommender.recommender.evaluate`` for the measurements behind the settings.
"""

from wiki_recommender.recommender.engine import (
    Recommendation,
    RecommendationResult,
    Recommender,
    recommend,
)

__all__ = ["Recommendation", "RecommendationResult", "Recommender", "recommend"]
