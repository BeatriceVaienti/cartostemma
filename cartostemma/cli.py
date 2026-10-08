"""Command-line interface.

    cartostemma stemma   config.yaml   # stemma stage from an existing collation matrix
    cartostemma run      config.yaml   # full pipeline: georef -> discretize -> collation -> stemma

Both accept ``--pruning {single,triplet}`` (default ``single``, the paper's
main method; see :mod:`cartostemma.stemma`).

``stemma`` runs on a collation matrix + a map metadata table (map_id, year),
which is the stable interchange documented in the README -- so you can bring
your own collation matrix and skip Stage A/B entirely.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import re

from .collation import (
    assign_states_from_clusters,
    classify_loci,
    merge_clone_groups,
    select_loci as collation_select_loci,
)
from .config import Config
from .stemma import (
    build_edges,
    build_map_token_cluster,
    chronological_order,
    classify_maps,
    inheritance_ratios,
    prune_triplets,
    u_sizes,
)


def parse_year(name: str, lo: int = 1810, hi: int = 1925) -> int:
    """Year from a map/folder name prefix (e.g. '1841_kiepert' -> 1841),
    clamped to the study period. Returns -1 if unparseable."""
    m = re.match(r"(\d{3,4})", name.replace("-", "5"))
    if not m:
        return -1
    return max(lo, min(hi, int(m.group(1))))


def _resolve_area(cfg: Config):
    """Turn the ``stemma.area`` config into an area argument for select_loci:
    None (full extent); a bbox tuple ``(xmin, ymin, xmax, ymax)`` (4 scalars); a
    shapely ``Polygon`` from a vertex list ``[[x, y], ...]`` (e.g. a drawn
    polygon); or a shapely geometry loaded from a shapefile path."""
    area = cfg.stemma.get("area")
    if area is None:
        return None
    if isinstance(area, (list, tuple)) and area:
        if isinstance(area[0], (list, tuple)):        # polygon vertices [[x, y], ...]
            from shapely.geometry import Polygon
            return Polygon(area)
        if len(area) == 4:                            # bbox of 4 scalars
            return tuple(area)
    # a path to a polygon file -> needs the geo stack
    from .georef import load_area

    return load_area(area, cfg.crs["projected_epsg"], cfg.stemma.get("area_buffer", 100.0))


def run_stemma(cfg: Config, pruning: str = "single") -> None:
    """Stemma stage from a pre-built collation matrix.

    Expects, under ``paths.output``:
        collation_matrix.parquet  (rows=maps, cols=loci, int/NaN states)
        maps.csv                  (columns: map_id, year)
        loci.csv                  (columns: locus, x, y)  -- only if an area is set

    Loci are classified corpus-wide, then (optionally) restricted to a region of
    interest for a local stemma; with no area, the full extent is used (global
    stemma).

    ``pruning`` selects the edge-pruning rule: ``"single"`` (default, the paper's
    main method -- CBGM-like single ancestor per locus) or ``"triplet"`` (the
    more conservative branching/flow alternative tested in the thesis).
    """
    out = Path(cfg.paths["output"])
    states = pd.read_parquet(out / "collation_matrix.parquet")
    meta = pd.read_csv(out / "maps.csv").set_index("map_id")
    years = meta["year"].to_dict()

    groups = merge_clone_groups(states, cfg.clone_groups, years)
    df_stemma, df_labels = classify_loci(groups)   # corpus-wide classification

    all_loci = [c for c in groups.columns if c not in ("year", "maps")]
    area = _resolve_area(cfg)
    if area is None:
        loci = all_loci
    else:
        loci_xy = pd.read_csv(out / "loci.csv").set_index("locus")
        selected = set(collation_select_loci(loci_xy, area, cfg.stemma.get("area_buffer", 0.0)))
        loci = [c for c in all_loci if c in selected]
        print(f"Area selection: {len(loci)} of {len(all_loci)} loci in region")

    df_stemma = df_stemma[loci]
    # Keep every witness that COVERS the region (has any non-NaN state in the raw
    # matrix), even if all its loci are undistorted -- e.g. the near-perfect 1925
    # Survey of Palestine has zero distorted loci but must still appear (as a grey,
    # familyless node). Only witnesses with no coverage at all (e.g. entirely
    # outside a selected area) are dropped.
    covered = groups.index[groups[loci].notna().any(axis=1)]
    df_stemma = df_stemma.loc[covered]
    groups = groups.loc[covered]

    mtc = build_map_token_cluster(df_stemma, loci)
    order = chronological_order(groups, loci)

    from .stemma import prune_single_ancestor
    prune_fn = prune_single_ancestor if pruning == "single" else prune_triplets
    tokens_for_edge = prune_fn(mtc, order)
    edges = build_edges(tokens_for_edge, order, mtc, cfg.stemma["min_shared_loci"])

    ratios = inheritance_ratios(
        edges,
        u_sizes(df_stemma, loci),
        cfg.stemma["inheritance_primary"],
        cfg.stemma["inheritance_secondary"],
        cfg.stemma.get("inheritance_min_shared", 0),
    )
    fingerprints = classify_maps(df_stemma)

    out.mkdir(parents=True, exist_ok=True)
    ratios.to_csv(out / "edges.csv", index=False)
    fingerprints.to_csv(out / "map_fingerprints.csv")
    df_labels.to_parquet(out / "locus_labels.parquet")

    # graph export + summary (networkx, always available)
    from . import viz

    coverage = groups[loci].notna().sum(axis=1).to_dict()
    # use the group-level (effective) years: graph nodes are clone-group ids,
    # not raw map names, so the raw years dict would miss the merged groups.
    group_years = groups["year"].to_dict()
    G = viz.build_graph(ratios, fingerprints, group_years)
    viz.export_graph(G, out / "stemma.graphml")
    viz.stemma_summary(ratios, fingerprints, coverage=coverage).to_csv(
        out / "stemma_summary.csv", index=False
    )

    # figure, if matplotlib ([viz] extra) is installed
    try:
        import matplotlib

        matplotlib.use("Agg")
        ax = viz.plot_stemma(G)
        ax.figure.savefig(out / "stemma.png", dpi=200)
    except ImportError:
        pass

    print(f"Wrote {len(edges)} edges -> {out/'edges.csv'}")
    print(f"Wrote stemma graph -> {out/'stemma.graphml'} and summary/fingerprints")


def _build_corpus(cfg: Config, maps_limit: int | None = None):
    """Stage A + B: build the :class:`georef.Map` objects, their padded token
    fields, and the per-locus clustering.

    Returns ``(maps, fields, grid_shape, base_grid_x, base_grid_y, clustering)``.
    """
    from .discretize import cluster_tokens, gather_tokens
    from .georef import (
        create_map_object,
        load_dataset,
        padded_token_fields,
        _OFFSETS,
    )
    from scipy.spatial.distance import pdist
    import numpy as np

    infos = load_dataset(cfg.paths["dataset"])
    if maps_limit:
        infos = infos[:maps_limit]
    maps = [create_map_object(info, cfg.grid["size_m"]) for info in infos]
    fields, grid_shape, base_gx, base_gy = padded_token_fields(maps)

    # delta uses the RAW undeformed reference vector.
    undeformed_vec = pdist(np.asarray(_OFFSETS), metric="euclidean")
    token_data = gather_tokens(fields, grid_shape, undeformed_vec)
    clustering = cluster_tokens(
        token_data,
        cluster_distance=cfg.clustering["max_distance"],
        distortion_threshold=cfg.distortion["undeformed_threshold"],
        alpha=cfg.clustering.get("alpha", 20.0),
        method=cfg.clustering.get("metric", "distortion"),
    )
    return maps, fields, grid_shape, base_gx, base_gy, clustering


def run_full(cfg: Config, maps_limit: int | None = None, pruning: str = "single"):
    """Full pipeline from a dataset of maps + GCPs to the stemma.

    Requires the geospatial stack (rasterio/pyproj/alphashape). Chains
    georef -> discretize -> collation -> stemma and writes the collation matrix
    and maps.csv, then runs the stemma stage. ``maps_limit`` restricts to the
    first N map folders (for quick smoke tests).

    NOTE: end-to-end, this path should be validated against the reference
    outputs on real data in the geospatial environment; the geometry stages
    cannot be exercised without that stack installed.
    """
    out = Path(cfg.paths["output"])
    out.mkdir(parents=True, exist_ok=True)

    maps, _fields, _grid_shape, base_gx, base_gy, clustering = _build_corpus(cfg, maps_limit)
    map_names = [m.name for m in maps]

    # collation matrix (the stable interchange)
    col_mapping = {locus: f"{locus[0]}_{locus[1]}" for locus in clustering}
    states = assign_states_from_clusters(clustering, col_mapping, map_names)
    states.to_parquet(out / "collation_matrix.parquet")
    pd.DataFrame(
        {
            "map_id": map_names,
            "year": [parse_year(n) for n in map_names],
            "image_path": [m.image_path or "" for m in maps],
        }
    ).to_csv(out / "maps.csv", index=False)
    # locus -> projected coordinates, so the stemma stage can restrict to a
    # region of interest (area selection)
    pd.DataFrame(
        {
            "locus": [col_mapping[t] for t in clustering],
            "x": [base_gx[i, j] for (i, j) in clustering],
            "y": [base_gy[i, j] for (i, j) in clustering],
        }
    ).to_csv(out / "loci.csv", index=False)
    print(f"Wrote collation matrix ({states.shape[0]} maps x {states.shape[1]} loci)")

    run_stemma(cfg, pruning=pruning)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cartostemma")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("stemma", "run"):
        p = sub.add_parser(name)
        p.add_argument("config", type=str, help="path to config.yaml")
        p.add_argument(
            "--pruning", choices=["single", "triplet"], default="single",
            help="edge-pruning rule: 'single' (default, the paper's main method) "
                 "or 'triplet' (the thesis's more conservative alternative)",
        )

    args = parser.parse_args(argv)
    cfg = Config.load(args.config)

    if args.command == "stemma":
        run_stemma(cfg, pruning=args.pruning)
    elif args.command == "run":
        run_full(cfg, pruning=args.pruning)
    return 0


if __name__ == "__main__":
    sys.exit(main())
