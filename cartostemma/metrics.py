"""Distance metrics on distortion tokens.

A distortion token is the vector of pairwise (all-to-all) distances among a grid
node's 3x3 neighbourhood (36 values for 9 points). Comparing two tokens tells us
how differently two maps deform the same local patch of territory.
"""
from __future__ import annotations

import numpy as np


def normalize_vector(vector):
    """Unit-normalise a token vector (a zero vector is returned unchanged)."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def ratio_distance(vec1, vec2) -> float:
    """Spread (max - min) of the entrywise ratio between two token vectors.

    The smaller-max vector is divided by the larger-max one entrywise so ratios
    stay <= 1; a small epsilon avoids division by zero. Identical shapes give 0;
    the more the two local deformations differ in *proportion*, the larger the
    value. This is the metric used for both the undeformed-threshold test and the
    hierarchical clustering of tokens.
    """
    eps = 1e-12
    v1 = np.asarray(vec1, dtype=float) + eps
    v2 = np.asarray(vec2, dtype=float) + eps
    ratio_vec = v1 / v2 if v1.max() < v2.max() else v2 / v1
    return float(ratio_vec.max() - ratio_vec.min())


def cosine_distance(vec1, vec2) -> float:
    """1 - cosine similarity between two vectors (alternative token metric)."""
    v1 = np.asarray(vec1, dtype=float)
    v2 = np.asarray(vec2, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 1.0 if denom == 0 else float(1.0 - np.dot(v1, v2) / denom)


def distortion_aware_ratio_distance(vec1, vec2, ref, alpha: float = 20.0) -> float:
    """Distortion-aware ratio distance: the base ``ratio_distance`` shrunk when
    both vectors are far from the undeformed reference ``ref``.

        d(v1, v2) = ratio_distance(v1, v2) / (1 + alpha * mean_dist_from_ref)

    where ``mean_dist_from_ref`` averages each vector's ratio_distance from
    ``ref``. Rationale: two strongly-distorted tokens should cluster together
    more readily than two near-reference ones, so the same absolute difference
    counts for less when the distortion is large. This is the clustering metric
    used in the paper (α=20 for both grid sizes; thresholds 0.32 at 50 m and
    0.40 at 150 m).
    """
    d1 = ratio_distance(vec1, ref)
    d2 = ratio_distance(vec2, ref)
    factor = 1.0 + alpha * 0.5 * (d1 + d2)
    return float(ratio_distance(vec1, vec2) / factor) if factor > 1e-12 else float("inf")


def pairwise_distortion_aware_ratio(X, dist_from_ref, alpha: float = 20.0) -> np.ndarray:
    """Condensed pairwise distance matrix (for ``scipy`` linkage) under the
    distortion-aware ratio distance.

    ``X`` is an (N, dim) array of tokens; ``dist_from_ref`` is the pre-computed
    per-token distance from the undeformed reference (δ), i.e.
    ``ratio_distance(token, raw_undeformed_vec)`` for each token, computed once
    and passed in rather than recomputed per pair."""
    X = np.asarray(X, dtype=float)
    dist_from_ref = np.asarray(dist_from_ref, dtype=float)
    n = len(X)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            factor = 1.0 + alpha * 0.5 * (dist_from_ref[i] + dist_from_ref[j])
            base = ratio_distance(X[i], X[j])
            out.append(base / factor if factor > 1e-12 else np.inf)
    return np.array(out)
