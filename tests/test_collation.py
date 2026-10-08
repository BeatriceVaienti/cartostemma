"""Test that final cluster labels map straight into the collation matrix."""
import numpy as np

from cartostemma.collation import assign_states_from_clusters


def test_assign_states_from_clusters():
    names = ["A", "B", "C"]
    # cluster labels are already final states: 0 undistorted, -1 noise, >0 cluster.
    # locus (0,0): A,B share cluster 1; C undistorted (0).
    # locus (0,1): only A and C cover it (B absent -> stays NaN).
    clustering = {
        (0, 0): {"map_indices": (0, 1, 2), "clusters": np.array([1, 1, 0])},
        (0, 1): {"map_indices": (0, 2), "clusters": np.array([2, -1])},
    }
    col_mapping = {(0, 0): "0_0", (0, 1): "0_1"}
    df = assign_states_from_clusters(clustering, col_mapping, names)

    assert df.at["A", "0_0"] == 1 and df.at["B", "0_0"] == 1 and df.at["C", "0_0"] == 0
    assert df.at["A", "0_1"] == 2 and df.at["C", "0_1"] == -1
    assert np.isnan(df.at["B", "0_1"])  # B does not cover locus (0,1)


if __name__ == "__main__":
    test_assign_states_from_clusters()
    print("ok")
