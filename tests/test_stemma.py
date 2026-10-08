"""End-to-end smoke test of the stemma stage on a tiny synthetic corpus.

Three maps A(1810) < B(1820) < C(1830), three loci. Designed so that:
  * cluster 5 at L1 originates in A and is inherited by B and C;
  * cluster 5 at L2 originates in A and is inherited by C;
  * cluster 7 at L3 originates in B and is inherited by C.
Hence A is a pure innovation source and C a pure deriver.
"""
import numpy as np
import pandas as pd

from cartostemma.collation import classify_loci, merge_clone_groups
from cartostemma.stemma import (
    build_edges,
    build_map_token_cluster,
    chronological_order,
    classify_maps,
    inheritance_ratios,
    prune_single_ancestor,
    prune_triplets,
    u_sizes,
)


def _corpus():
    states = pd.DataFrame(
        {
            "L1": {"A": 5, "B": 5, "C": 5},
            "L2": {"A": 5, "B": 0, "C": 5},
            "L3": {"A": 0, "B": 7, "C": 7},
        },
        dtype=object,
    )
    years = {"A": 1810, "B": 1820, "C": 1830}
    return states, years


def test_stemma_pipeline():
    states, years = _corpus()
    groups = merge_clone_groups(states, clone_groups={}, years=years)
    df_stemma, df_labels = classify_loci(groups)

    # A introduces cluster 5 at L1 -> innovation; C reproduces it -> derivation
    assert df_labels.at["A", "L1"] == "innovation"
    assert df_labels.at["C", "L1"] == "derivation"
    assert df_labels.at["B", "L3"] == "innovation"

    loci = [c for c in groups.columns if c not in ("year", "maps")]
    mtc = build_map_token_cluster(df_stemma, loci)
    order = chronological_order(groups, loci)
    assert order == ["A", "B", "C"]

    edges = build_edges(prune_triplets(mtc, order), order, mtc, min_shared_loci=1)
    pairs = {(e["source"], e["target"]) for e in edges}
    assert ("A", "C") in pairs and ("B", "C") in pairs

    ratios = inheritance_ratios(edges, u_sizes(df_stemma, loci))
    assert (ratios["inheritance_ratio"] > 0).all()

    fp = classify_maps(df_stemma)
    assert fp.at["A", "innovation"] == 2 and fp.at["A", "derivation"] == 0
    assert fp.at["C", "derivation"] == 3 and fp.at["C", "innovation"] == 0


def test_single_ancestor_pruning():
    """The paper's main pruning method. At L1, A, B and C all share cluster 5:
    A and B are tied on total agreement with C (2 loci each), so the tie is
    broken toward the more recent predecessor (B) -- unlike triplet pruning
    (test_stemma_pipeline), which keeps both A->C and B->C at L1."""
    states, years = _corpus()
    groups = merge_clone_groups(states, clone_groups={}, years=years)
    df_stemma, _ = classify_loci(groups)
    loci = [c for c in groups.columns if c not in ("year", "maps")]
    mtc = build_map_token_cluster(df_stemma, loci)
    order = chronological_order(groups, loci)

    edges = build_edges(prune_single_ancestor(mtc, order), order, mtc, min_shared_loci=1)
    shared = {(e["source"], e["target"]): set(e["shared_token_clusters"]) for e in edges}

    assert shared[("A", "C")] == {"L2"}
    assert shared[("B", "C")] == {"L1", "L3"}

    ratios = inheritance_ratios(edges, u_sizes(df_stemma, loci))
    assert (ratios["inheritance_ratio"] > 0).all()


if __name__ == "__main__":
    test_stemma_pipeline()
    test_single_ancestor_pruning()
    print("ok")
