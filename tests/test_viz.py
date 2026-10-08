"""Graph construction, family assignment, and summary (networkx core; no plots)."""
import pandas as pd

from cartostemma import viz


def _edges_and_fp():
    edges = pd.DataFrame(
        [
            {"source": "A", "target": "B", "n_shared": 5, "inheritance_ratio": 0.8, "tier": "primary"},
            {"source": "A", "target": "C", "n_shared": 2, "inheritance_ratio": 0.2, "tier": "secondary"},
            {"source": "B", "target": "C", "n_shared": 4, "inheritance_ratio": 0.7, "tier": "primary"},
        ]
    )
    fingerprints = pd.DataFrame(
        {"innovation": {"A": 5, "B": 1, "C": 0}, "derivation": {"A": 0, "B": 4, "C": 6}, "hapax": {"A": 0, "B": 0, "C": 0}}
    )
    return edges, fingerprints


def test_build_graph_roles_and_families():
    edges, fp = _edges_and_fp()
    G = viz.build_graph(edges, fp, years={"A": 1810, "B": 1820, "C": 1830})
    assert G.number_of_edges() == 3
    assert G.nodes["A"]["role"] == "innovation"
    assert G.nodes["C"]["role"] == "derivation"
    # A-B-C connected through primary edges -> one family
    fam = viz.assign_families(G)
    assert len(set(fam.values())) == 1


def test_summary():
    edges, fp = _edges_and_fp()
    s = viz.stemma_summary(edges, fp).set_index("map_name")
    # C's strongest predecessor is B (4 shared) ahead of A (2)
    assert s.at["C", "predecessors"][0] == ["B", 4]
    assert s.at["A", "distorted_tokens"] == 5


if __name__ == "__main__":
    test_build_graph_roles_and_families()
    test_summary()
    print("ok")
