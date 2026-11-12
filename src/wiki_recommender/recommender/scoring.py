"""Turn the similarities between visited articles and candidates into one ranked list.

`similarity[v, c]` is the weighted similarity of visited article v to candidate c, in [0, 1]. A history
holds several articles, so the per-article similarities must be aggregated:

- `power_mean` with exponent p: (mean over v of similarity^p)^(1/p). p = 1 is the plain mean (the
  centroid ranking), p -> infinity is the maximum. Values in between reward a candidate that is very
  close to one visited article without ignoring the others. The result stays in [0, 1].
- `interest_ranking` first groups the history into interests, then fills the list from each interest
  in proportion to its size, so a history about both dogs and quantum physics gets both.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def power_mean(similarity: np.ndarray, p: float) -> np.ndarray:
    """Aggregate rows (visited articles) into one score per column (candidate)."""
    if np.isinf(p):
        return similarity.max(axis=0)
    return np.power(np.mean(np.power(similarity, p), axis=0), 1.0 / p)


def group_interests(pairwise: np.ndarray, threshold: float) -> list[list[int]]:
    """Cluster visited articles whose average-linkage similarity reaches the threshold.

    `pairwise` is the visited-to-visited similarity matrix. Returns groups of row indices, largest first.
    """
    count = pairwise.shape[0]
    if count == 1:
        return [[0]]
    distance = 1.0 - np.clip(pairwise, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(linkage(squareform(distance, checks=False), "average"), 1.0 - threshold, "distance")
    groups: dict[int, list[int]] = {}
    for row, label in enumerate(labels):
        groups.setdefault(label, []).append(row)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))


def top_indices(scores: np.ndarray, k: int, excluded: Sequence[int] = ()) -> np.ndarray:
    """Indices of the k highest scores, best first, skipping excluded ones."""
    scores = scores.astype(np.float64, copy=True)
    scores[list(excluded)] = -np.inf
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return np.array([], dtype=np.int64)
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top], kind="stable")]


def interest_ranking(
    similarity: np.ndarray,
    pairwise: np.ndarray,
    k: int,
    excluded: Sequence[int],
    threshold: float,
    p: float,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Pick k candidates, giving each interest a share of the slots proportional to its size.

    Returns the chosen indices, ordered by score, the score of every candidate (its power-mean
    similarity to the interest it is closest to) and the interests as groups of history rows.
    """
    groups = group_interests(pairwise, threshold)
    per_group = np.stack([power_mean(similarity[group], p) for group in groups])
    scores = per_group.max(axis=0)
    if len(groups) == 1:
        return top_indices(scores, k, excluded), scores, groups

    # Each interest ranks enough candidates to fill the list even if others took its best ones.
    rankings = [list(top_indices(row, k * len(groups), excluded)) for row in per_group]
    sizes = np.array([len(group) for group in groups], dtype=float)
    taken: list[int] = []
    chosen: set[int] = set()
    filled = np.zeros(len(groups))
    while len(taken) < k and any(rankings):
        # The interest furthest below its fair share of the list picks next.
        order = np.argsort(filled / sizes, kind="stable")
        for group in order:
            while rankings[group] and rankings[group][0] in chosen:
                rankings[group].pop(0)
            if rankings[group]:
                index = rankings[group].pop(0)
                taken.append(index)
                chosen.add(index)
                filled[group] += 1
                break
    taken_array = np.array(taken, dtype=np.int64)
    return taken_array[np.argsort(-scores[taken_array], kind="stable")], scores, groups
