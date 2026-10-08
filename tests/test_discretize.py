"""Smoke test for Stage B (discretization).

Two maps share a distinctive distortion at one locus; a third is undistorted
there. The undistorted map must get state 0; the two distorted maps must share a
positive cluster label.
"""
import numpy as np
from scipy.spatial.distance import pdist

from cartostemma.discretize import cluster_tokens, gather_tokens

# a plausible raw undeformed reference (36 all-to-all distances of a 3x3 grid)
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
UNDEF = pdist(np.asarray(_OFFSETS), metric="euclidean")
DIM = len(UNDEF)


def _nan_grid(rows, cols):
    return np.full((rows, cols, DIM), np.nan)


def test_gather_and_cluster():
    grid_shape = (1, 1)
    # a distinctive distortion: perturb the undeformed vector strongly
    distorted = UNDEF.copy()
    distorted[:5] *= 3.0

    m0, m1, m2 = _nan_grid(1, 1), _nan_grid(1, 1), _nan_grid(1, 1)
    m0[0, 0] = distorted
    m1[0, 0] = distorted * 1.02   # essentially the same distortion pattern
    m2[0, 0] = UNDEF              # undistorted here

    token_data = gather_tokens([m0, m1, m2], grid_shape, UNDEF)
    assert (0, 0) in token_data

    res = cluster_tokens(token_data, cluster_distance=0.4, distortion_threshold=0.01,
                         alpha=20.0, method="distortion")
    labels = dict(zip(res[(0, 0)]["map_indices"], res[(0, 0)]["clusters"]))
    # map 2 is undistorted -> state 0
    assert labels[2] == 0
    # maps 0 and 1 share the same positive cluster
    assert labels[0] > 0 and labels[0] == labels[1]


if __name__ == "__main__":
    test_gather_and_cluster()
    print("ok")
