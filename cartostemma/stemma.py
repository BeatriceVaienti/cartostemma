"""Stemmatic inference from the classified collation matrix.

This is the core, CBGM-inspired method of the paper. It works locus by locus,
orients edges by chronology (a substitute for CBGM's editorial reading priority,
which distortion clusters cannot supply), and resolves entangled sources with a
per-locus single-ancestor majority rule, analogous to CBGM's coherence-based
ancestor selection (:func:`prune_single_ancestor`, the paper's main method).
:func:`prune_triplets`, a more conservative triplet-based alternative, was also
tested and is kept here for comparison; see the paper §5.4 for the tradeoffs.

Pipeline:
    build_map_token_cluster  ->  per map, the set of positive clusters per locus
    prune_single_ancestor    ->  per-locus single-ancestor majority-rule pruning
    build_edges              ->  directed edges with >= N shared loci
    inheritance_ratios       ->  the Inheritance Ratio, and primary/secondary stratification
    classify_maps            ->  innovation / derivation / hapax fingerprint

Uses ``innovation`` (rather than ``archetype``) for the label, matching the
paper's terminology: the corpus has no single archetype; family roots are
innovation sources.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import pandas as pd


# --------------------------------------------------------------------------- #
# Prepare the per-map, per-locus cluster sets                                  #
# --------------------------------------------------------------------------- #
def build_map_token_cluster(
    df_stemma: pd.DataFrame,
    locus_cols: list,
    include_hapax: bool = False,
) -> dict:
    """From the classified matrix, collect for each map the set of positive
    cluster ids present at each locus (hapax excluded by default).

    Returns a nested ``defaultdict``: ``mtc[map][locus] -> set[int]``. The
    defaultdict behaviour is relied upon by :func:`prune_triplets` (missing
    entries read as the empty set).
    """
    mtc: dict = defaultdict(lambda: defaultdict(set))
    for g in df_stemma.index:
        for locus in locus_cols:
            cell = df_stemma.at[g, locus]
            if not isinstance(cell, dict):
                continue
            for cluster_id, label in cell.items():
                if label == "hapax" and not include_hapax:
                    continue
                if cluster_id <= 0:
                    continue
                mtc[g][locus].add(cluster_id)
    return mtc


def u_sizes(df_stemma: pd.DataFrame, locus_cols: list) -> dict:
    """|U_B| for each map: number of loci at which it exhibits a valid cluster
    (the denominator of the Inheritance Ratio)."""
    return {
        g: sum(
            1
            for locus in locus_cols
            if isinstance(df_stemma.at[g, locus], dict) and df_stemma.at[g, locus]
        )
        for g in df_stemma.index
    }


def chronological_order(df_groups: pd.DataFrame, locus_cols: list) -> list:
    """Maps sorted by (year, then descending coverage) — the processing order
    for triplet pruning. Coverage breaks ties toward better-covered maps."""
    coverage = df_groups[locus_cols].notna().sum(axis=1).to_dict()
    return sorted(df_groups.index, key=lambda g: (df_groups.at[g, "year"], -coverage[g]))


# --------------------------------------------------------------------------- #
# Triplet-based pruning: branching vs. flow (alternative; see module docstring) #
# --------------------------------------------------------------------------- #
def prune_triplets(map_token_cluster: dict, sorted_maps: list) -> dict:
    """Alternative to :func:`prune_single_ancestor`: locus-level matching
    followed by triplet pruning, tested in the thesis but not used as the
    paper's main method -- it is more conservative (retains more multi-source
    structure) but, being resolved greedily over ordered triplets, its outcome
    depends on processing order.

    For each map C (in chronological order) we first seed, for every earlier map
    A, the set of loci where A and C share a positive cluster. Then, for every
    ordered pair of predecessors (A, B) we compare their *exclusive* agreements
    with C:

        L_AC = loci where C matches A but not B        (Delta_{AC\\B})
        L_BC = loci where C matches B but not A        (Delta_{BC\\A})

    At loci where A, B and C all share the cluster (a triple match):
        |L_AC| > |L_BC|  -> BRANCHING: C derives from A independently of B;
                            drop the triple loci from edge B->C.
        |L_BC| > |L_AC|  -> FLOW: B mediates A->C;
                            drop the triple loci from edge A->C.
        equal            -> evidence balanced; keep both.

    Returns ``tokens_for_edge[A][C] -> set(loci)`` of surviving matches.
    """
    tokens_for_edge: dict = defaultdict(lambda: defaultdict(set))
    processed: list = []

    for C in sorted_maps:
        if not processed:
            processed.append(C)
            continue

        # seed A -> C matches
        for A in processed:
            tokens_for_edge[A][C] = {
                locus
                for locus, cC in map_token_cluster[C].items()
                if cC and (map_token_cluster[A].get(locus, set()) & cC)
            }

        # resolve every predecessor pair against C
        for A, B in combinations(processed, 2):
            cand_AC = tokens_for_edge[A][C]
            cand_BC = tokens_for_edge[B][C]
            L_AC = cand_AC - cand_BC
            L_BC = cand_BC - cand_AC
            triple = {
                locus
                for locus in (cand_AC & cand_BC)
                if (
                    map_token_cluster[A][locus]
                    & map_token_cluster[B][locus]
                    & map_token_cluster[C][locus]
                )
            }
            if len(L_AC) > len(L_BC):
                tokens_for_edge[B][C] -= triple
            elif len(L_BC) > len(L_AC):
                tokens_for_edge[A][C] -= triple

        processed.append(C)

    return tokens_for_edge


def prune_single_ancestor(map_token_cluster: dict, sorted_maps: list) -> dict:
    """The paper's main pruning method: a CBGM-like *single ancestor per
    locus* rule.

    For each map C (chronological order) and each locus where C shows a positive
    cluster, among all earlier maps sharing that cluster we keep the edge only to
    the *single highest-ranked predecessor*. Ranking approximates CBGM's "closest
    potential ancestor": the predecessor with the greatest total number of shared
    loci with C (an agreement-based, pre-genealogical proxy), ties broken toward
    the more recent predecessor. Every other candidate edge at that locus is
    discarded.

    Returns ``tokens_for_edge[A][C] -> set(loci)`` in the same format as
    :func:`prune_triplets`, so it plugs into :func:`build_edges` unchanged.
    """
    tokens_for_edge: dict = defaultdict(lambda: defaultdict(set))
    processed: list = []
    order_index = {m: i for i, m in enumerate(sorted_maps)}  # later map -> higher

    for C in sorted_maps:
        if not processed:
            processed.append(C)
            continue

        shared_per_pred: dict = {}
        for A in processed:
            s = {
                locus
                for locus, cC in map_token_cluster[C].items()
                if cC and (map_token_cluster[A].get(locus, set()) & cC)
            }
            if s:
                shared_per_pred[A] = s

        # rank each predecessor by total agreement with C; recency breaks ties
        rank = {A: (len(s), order_index[A]) for A, s in shared_per_pred.items()}

        all_loci: set = set().union(*shared_per_pred.values()) if shared_per_pred else set()
        for x in all_loci:
            cands = [A for A in shared_per_pred if x in shared_per_pred[A]]
            best = max(cands, key=lambda A: rank[A])
            tokens_for_edge[best][C].add(x)

        processed.append(C)

    return tokens_for_edge


def build_edges(
    tokens_for_edge: dict,
    sorted_maps: list,
    map_token_cluster: dict,
    min_shared_loci: int = 3,
) -> list[dict]:
    """Materialise surviving matches into directed edges, dropping any edge
    supported by fewer than ``min_shared_loci`` loci (global pruning)."""
    edges: list[dict] = []
    for C in sorted_maps:
        for A, targets in tokens_for_edge.items():
            kept = targets.get(C, set())
            if len(kept) < min_shared_loci:
                continue
            shared = {
                locus: (map_token_cluster[A][locus] & map_token_cluster[C][locus])
                for locus in kept
                if map_token_cluster[A][locus] & map_token_cluster[C][locus]
            }
            if shared:
                edges.append(
                    {"source": A, "target": C, "shared_token_clusters": shared}
                )
    return edges


# --------------------------------------------------------------------------- #
# Inheritance Ratio and stemma stratification                                  #
# --------------------------------------------------------------------------- #
def inheritance_ratios(
    edges: list[dict],
    u_size: dict,
    primary: float = 0.5,
    secondary: float = 0.15,
    min_shared: int = 0,
) -> pd.DataFrame:
    """Compute the Inheritance Ratio for each edge and tag its tier.

    InheritanceRatio(A, B) = #shared_loci(A, B) / |U_B|

    Tier: ``primary`` (ratio >= primary), ``secondary`` (secondary <= ratio <
    primary), or ``hidden`` (below secondary; kept in the data, not displayed).
    Note the ratios of B's several sources may sum to > 1 by design, since a map
    inherits different loci from different sources (horizontal transmission).

    ``min_shared`` is the paper's absolute-count floor ``T_abs`` (15 sub-edges,
    in the Jerusalem demonstration): an edge with fewer than ``min_shared``
    shared loci is forced to ``hidden`` regardless of its ratio, so *both* the
    primary and secondary tiers require ``shared_token_count >= T_abs``. This is
    decoupled from :func:`build_edges`' existence filter, which only prunes edges
    with almost no support; ``T_abs`` governs which edges are actually displayed.
    """
    rows = []
    for e in edges:
        B = e["target"]
        n = len(e["shared_token_clusters"])
        denom = u_size.get(B, 0)
        ratio = n / denom if denom else 0.0
        if n < min_shared:
            tier = "hidden"
        elif ratio >= primary:
            tier = "primary"
        elif ratio >= secondary:
            tier = "secondary"
        else:
            tier = "hidden"
        rows.append(
            {
                "source": e["source"], "target": B,
                "n_shared": n,               # inherited loci
                "n_target_loci": int(denom), # distorted loci in the receiving map (|U_B|)
                "inheritance_ratio": ratio,
                "tier": tier,
            }
        )
    columns = ["source", "target", "n_shared", "n_target_loci", "inheritance_ratio", "tier"]
    return pd.DataFrame(rows, columns=columns)


def classify_maps(df_stemma: pd.DataFrame) -> pd.DataFrame:
    """Innovation / derivation / hapax fingerprint of each map: every locus
    labelled by :func:`collation.classify_loci`, independent of whether its
    sub-edges survive the later pruning -- a locus's role is a
    property of (map, locus) alone, fixed before any edge is drawn.

    A predominantly-innovation map acts as an innovation source (family root);
    a predominantly-derivation map largely copies its predecessors.
    """
    counts = {m: {"innovation": 0, "derivation": 0, "hapax": 0} for m in df_stemma.index}
    for m in df_stemma.index:
        for locus in df_stemma.columns:
            cell = df_stemma.at[m, locus]
            if isinstance(cell, dict):
                for label in cell.values():
                    if label in counts[m]:
                        counts[m][label] += 1
    return pd.DataFrame.from_dict(counts, orient="index")
