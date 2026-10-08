"""Stage B -- discretization: continuous distortion tokens -> discrete states.

  1. For every map and grid node, take the (normalized) distortion token and its
     distance from the undeformed reference,
         delta = ratio_distance(normalized_token, undeformed_vec)
     where ``undeformed_vec`` is the *raw* 3x3 all-to-all distance vector.
  2. At each locus, maps with delta <= the distortion threshold are labelled 0
     (undistorted). The remaining (distorted) maps are clustered by complete-
     linkage hierarchical clustering under the chosen metric, cut at
     ``cluster_distance``. Default metric is the **distortion-aware ratio
     distance** (D = ratio / (1 + alpha * mean_delta)), which relaxes the
     distance between strongly-distorted tokens so they group more readily.
  3. Singleton clusters are relabelled -1 (noise); real clusters get 1, 2, ...

Paper parameters: distortion threshold 0.01; alpha 20; cluster distance 0.40
(150 m grid) / 0.32 (50 m grid).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .metrics import (
    cosine_distance,
    normalize_vector,
    pairwise_distortion_aware_ratio,
    ratio_distance,
)


def gather_tokens(
    token_fields: list[np.ndarray],
    grid_shape: tuple[int, int],
    undeformed_vec: np.ndarray,
) -> dict:
    """Collect, per locus, ``(map_idx, normalized_token, delta)`` for every map
    covering that locus. ``delta = ratio_distance(normalized_token,
    undeformed_vec)`` with the *raw* undeformed reference.

    ``token_fields[m]`` is map ``m``'s token field of shape ``(rows, cols, dim)``
    (NaN where uncovered), as produced by ``georef.padded_token_fields``.
    """
    rows, cols = grid_shape
    out: dict = defaultdict(list)
    for map_idx, tv in enumerate(token_fields):
        for i in range(rows):
            for j in range(cols):
                vec = tv[i, j]
                if np.isnan(vec).any():
                    continue
                nvec = normalize_vector(vec)
                delta = ratio_distance(nvec, undeformed_vec)
                out[(i, j)].append((map_idx, nvec, delta))
    return dict(out)


def _pairwise(X: np.ndarray, delta: np.ndarray, method: str, alpha: float):
    """Condensed pairwise distances between distorted tokens under ``method``."""
    m = method.lower()
    if m == "distortion":  # distortion-aware ratio distance (reference default)
        return pairwise_distortion_aware_ratio(X, delta, alpha)
    if m == "ratio":
        return pdist(X, metric=ratio_distance)
    if m in ("correlation", "spearman"):
        if m == "spearman":  # correlation on the coordinate ranks
            from scipy.stats import rankdata

            X = np.apply_along_axis(rankdata, 1, X)
        return pdist(X, metric="correlation")
    if m == "cosine":
        return pdist(X, metric=cosine_distance)
    raise ValueError(f"unknown clustering metric: {method}")


def cluster_tokens(
    token_data: dict,
    cluster_distance: float = 0.4,
    distortion_threshold: float = 0.01,
    alpha: float = 20.0,
    method: str = "distortion",
) -> dict:
    """Cluster the tokens at each locus into discrete states.

    Returns ``{locus: {"clusters": labels, "vectors": array,
    "map_indices": tuple, "distance_from_undeformed": array}}`` where ``labels``
    are the final states: ``0`` undistorted, ``-1`` noise (singleton), positive
    integers for shared distortion clusters.
    """
    results: dict = {}
    for locus, entries in token_data.items():
        map_indices = tuple(e[0] for e in entries)
        vectors = np.asarray([e[1] for e in entries])
        deltas = np.asarray([e[2] for e in entries])

        labels = np.full(len(vectors), -1, dtype=int)
        undistorted = deltas <= distortion_threshold
        labels[undistorted] = 0

        to_cluster = np.where(~undistorted)[0]
        if len(to_cluster) >= 2:
            X = vectors[to_cluster]
            pw = np.asarray(_pairwise(X, deltas[to_cluster], method, alpha), dtype=float)
            finite = pw[np.isfinite(pw)]
            pw[~np.isfinite(pw)] = finite.max() if finite.size else 1.0
            clusters = fcluster(linkage(pw, method="complete"), cluster_distance, criterion="distance")
            # singletons -> noise (-1); real clusters -> 1, 2, ...
            uniq, counts = np.unique(clusters, return_counts=True)
            remap, nxt = {}, 1
            for cl, cnt in zip(uniq, counts):
                if cnt == 1:
                    remap[cl] = -1
                else:
                    remap[cl] = nxt
                    nxt += 1
            for idx, cl in zip(to_cluster, clusters):
                labels[idx] = remap[cl]

        results[locus] = {
            "clusters": labels,
            "vectors": vectors,
            "map_indices": map_indices,
            "distance_from_undeformed": deltas,
        }
    return results
