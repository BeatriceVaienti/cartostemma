"""Visualization and graph export for the reconstructed stemma.

Two layers:
  * graph construction + export (networkx, a core dependency) -- produces the
    ``stemma.graphml`` that opens in Gephi / Cytoscape and the per-map summary;
  * figures (matplotlib, the ``[viz]`` extra, lazily imported) -- the stemma
    forest, the cluster-membership heatmap, and per-map distortion maps.

Terminology: nodes use the paper's ``innovation`` role rather than ``archetype``
(the corpus has no single archetype; family roots are innovation sources).
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


# --------------------------------------------------------------------------- #
# Graph construction and export                                               #
# --------------------------------------------------------------------------- #
def build_graph(
    edges: pd.DataFrame,
    fingerprints: pd.DataFrame | None = None,
    years: dict | None = None,
) -> nx.DiGraph:
    """Assemble the stemma DiGraph from the edge table (``inheritance_ratios``
    output) and, optionally, per-map fingerprints and years.

    Edge attributes: ``inheritance_ratio``, ``n_shared``, ``tier``.
    Node attributes: ``innovation`` / ``derivation`` / ``hapax`` counts, ``role``
    (the dominant one), and ``year``.
    """
    G = nx.DiGraph()
    for _, e in edges.iterrows():
        G.add_edge(
            e["source"], e["target"],
            inheritance_ratio=float(e.get("inheritance_ratio", 0.0)),
            n_shared=int(e.get("n_shared", 0)),
            n_target_loci=int(e.get("n_target_loci", 0)),
            tier=e.get("tier", "primary"),
        )
    if fingerprints is not None:
        for m, row in fingerprints.iterrows():
            if m not in G:
                G.add_node(m)
            counts = {k: int(row.get(k, 0)) for k in ("innovation", "derivation", "hapax")}
            G.nodes[m].update(counts)
            G.nodes[m]["role"] = max(counts, key=counts.get) if any(counts.values()) else "none"
    if years:
        for m in G.nodes:
            G.nodes[m]["year"] = int(years.get(m, 0))
    return G


def assign_families(G: nx.DiGraph) -> dict:
    """Label each node with its family. A family is the **tree rooted at a head**:
    every node attaches to its single strongest incoming *primary* edge (its
    dominant ancestor), and the family is the root that this chain reaches. A node
    with no incoming primary edge is a head (its own root).

    This matches the thesis's family concept -- an innovation source together with
    the maps that primarily derive from it -- and, unlike weakly-connected
    components, does not fuse two lineages just because a single primary edge
    bridges them (e.g. an outgoing primary edge from a head into another family).
    Returns ``{node: family_id}``."""
    parent: dict = {}
    for n in G.nodes:
        best, best_r = None, -1.0
        for u in G.predecessors(n):
            d = G[u][n]
            if d.get("tier") == "primary":
                r = float(d.get("inheritance_ratio", 0) or 0)
                if r > best_r:
                    best_r, best = r, u
        parent[n] = best

    def root(n):
        seen: set = set()
        while parent.get(n) is not None and n not in seen:
            seen.add(n)
            n = parent[n]
        return n

    ids: dict = {}
    family: dict = {}
    for n in G.nodes:
        family[n] = ids.setdefault(root(n), len(ids))
    return family


def family_counts(G: nx.DiGraph, family: dict | None = None) -> tuple[int, int, int]:
    """Return ``(#families, #familyless_maps, #maps)``. A *family* is a primary
    tree with more than one member; single-node components are familyless (drawn
    grey). Maps that primarily derive within a family are counted in it."""
    family = family if family is not None else assign_families(G)
    from collections import Counter

    sizes = Counter(family.values())
    n_families = sum(1 for s in sizes.values() if s > 1)
    n_familyless = sum(1 for s in sizes.values() if s == 1)
    return n_families, n_familyless, G.number_of_nodes()


def export_graph(G: nx.DiGraph, path) -> None:
    """Write the stemma to GraphML (or GEXF, by extension) for Gephi/Cytoscape."""
    path = str(path)
    if path.endswith(".gexf"):
        nx.write_gexf(G, path)
    else:
        nx.write_graphml(G, path)


def stemma_summary(
    edges: pd.DataFrame,
    fingerprints: pd.DataFrame | None = None,
    groups: dict | None = None,
    coverage: dict | None = None,
) -> pd.DataFrame:
    """Per-map summary matching the reference ``stemma_summary.csv``:
    each map's best predecessors and successors (with shared-loci counts), plus
    distorted vs. total token counts. Used both as an output and for regression."""
    maps = set(edges["source"]) | set(edges["target"])
    if fingerprints is not None:
        maps |= set(fingerprints.index)  # include familyless maps (no surviving edges)
    rows = []
    for m in sorted(maps):
        preds = edges[edges["target"] == m].sort_values("n_shared", ascending=False)
        succs = edges[edges["source"] == m].sort_values("n_shared", ascending=False)
        distorted = int((fingerprints.loc[m, ["innovation", "derivation", "hapax"]].sum()
                         if fingerprints is not None and m in fingerprints.index else 0))
        rows.append({
            "map_name": m,
            "group_name": (groups or {}).get(m, "ungrouped"),
            "distorted_tokens": distorted,
            "total_tokens": int((coverage or {}).get(m, distorted)),
            "predecessors": [[s, int(n)] for s, n in zip(preds["source"], preds["n_shared"])],
            "successors": [[t, int(n)] for t, n in zip(succs["target"], succs["n_shared"])],
        })
    columns = ["map_name", "group_name", "distorted_tokens", "total_tokens",
               "predecessors", "successors"]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# Figures (matplotlib; requires the [viz] extra)                              #
# --------------------------------------------------------------------------- #
def _temporal_layout(G: nx.DiGraph, family: dict) -> dict:
    """Node positions: y = year, x grouped by family (each family a column band,
    spread over a couple of sub-columns to reduce vertical overlap). The y-axis
    is inverted in :func:`plot_stemma` so earliest maps sit at the top."""
    by_family: dict = {}
    for node, fid in family.items():
        by_family.setdefault(fid, []).append(node)
    # order families by their earliest map, so lineages read left-to-right in time
    fam_order = sorted(
        by_family, key=lambda fid: min(G.nodes[n].get("year", 0) for n in by_family[fid])
    )
    pos, x = {}, 0.0
    for fid in fam_order:
        nodes = sorted(by_family[fid], key=lambda n: G.nodes[n].get("year", 0))
        for k, node in enumerate(nodes):
            pos[node] = (x + (k % 2) * 0.7, G.nodes[node].get("year", 0))
        x += 2.2
    return pos


def plot_stemma(G: nx.DiGraph, family: dict | None = None, ax=None, labels: bool = True):
    """Draw the stemma: **chronological top-to-bottom** (earliest at the top),
    families coloured, innovation sources ringed, hapax-dominant nodes as
    diamonds; primary edges solid, secondary dashed. Map names are labelled next
    to each node when ``labels`` is true. Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    family = family if family is not None else assign_families(G)
    pos = _temporal_layout(G, family)

    if ax is None:
        n = G.number_of_nodes()
        width = max(10, 0.55 * len(set(family.values())) + (6 if labels else 0))
        height = max(8, 0.16 * n)
        ax = plt.subplots(figsize=(width, height))[1]

    cmap = plt.get_cmap("tab20")
    node_color = [cmap(family.get(n, 0) % 20) for n in G.nodes]

    for tier, style in (("primary", "solid"), ("secondary", "dashed")):
        es = [(u, v) for u, v, d in G.edges(data=True) if d.get("tier") == tier]
        nx.draw_networkx_edges(G, pos, edgelist=es, style=style, ax=ax,
                               edge_color="0.4", arrowsize=8,
                               alpha=1.0 if tier == "primary" else 0.5)

    innovation = [n for n in G.nodes if G.nodes[n].get("role") == "innovation"]
    hapax = [n for n in G.nodes if G.nodes[n].get("role") == "hapax"]
    other = [n for n in G.nodes if n not in innovation and n not in hapax]
    idx = {n: k for k, n in enumerate(G.nodes)}
    for nodes, shape, lw in ((other, "o", 0.5), (innovation, "o", 2.5), (hapax, "D", 1.5)):
        if nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes, node_shape=shape, ax=ax, node_size=140,
                node_color=[node_color[idx[n]] for n in nodes],
                edgecolors="black", linewidths=lw,
            )

    if labels:
        for name, (px, py) in pos.items():
            ax.annotate(name, (px, py), fontsize=5, va="center",
                        xytext=(4, 0), textcoords="offset points")

    ax.invert_yaxis()  # earliest year at the top
    ax.set_ylabel("year")
    ax.set_xticks([])
    ax.set_title("Planimetric stemma (chronological, top→bottom; families coloured, innovation sources ringed)")
    ax.figure.tight_layout()
    return ax


# --------------------------------------------------------------------------- #
# Map-level analyses on the deformed grid ("beyond the stemma")                #
#                                                                              #
# The pairwise comparison and the per-map innovation/hapax/derivation          #
# signature both overlay one dot per grid node on the map image at the node's  #
# deformed-grid pixel position (``deformed_grid_x/y`` are already image pixel  #
# coordinates, y increasing downward, so they are plotted directly), coloured  #
# by the analysis.                                                             #
# --------------------------------------------------------------------------- #

_NB_PALETTE = {
    "pink_dark": "#f75785", "pink_light": "#f8b0be",
    "aqua_dark": "#009da5", "aqua_light": "#3cc5be",
    "orange_dark": "#ffa631", "orange_light": "#ffd766",
    "red_dark": "#e84743", "red_light": "#ed8e83",
    "blue_dark": "#489fee", "blue_light": "#8fcfff",
    "dark_grey": "#413d3a", "light_grey": "#cac7c7",
}

_ANALYSIS_COLORS = {
    "match": _NB_PALETTE["aqua_dark"],        # same positive cluster across maps
    "differ": _NB_PALETTE["red_dark"],        # different positive clusters (or noise)
    "mixed": _NB_PALETTE["dark_grey"],        # absent/undistorted in one of the maps
    "undistorted": "#ffffff",                 # state 0
    "innovation": _NB_PALETTE["aqua_dark"],   # novelty first shown by this map
    "hapax": _NB_PALETTE["pink_dark"],        # one-off (state -1)
    "derivation": _NB_PALETTE["orange_dark"], # inherited from an earlier map
}

# Background treatment for the map underlay in the analyses: darken, then lift,
# reduce contrast and desaturate, so the coloured dots read clearly over the
# busy hi-res map scan.
_MAP_ENHANCE = {"darken": 0.65, "whiten": 0.3, "contrast": 0.2, "color": 0.2}

# Per-category dot opacity for the shared-distortion comparison. Kept fairly
# opaque so the colour reads true over any map background -- at low alpha a grey
# dot over dark ink looks whitish and a white dot over a light scan looks grey,
# which reads as if white/grey were swapped.
_COMPARE_ALPHA = {"match": 0.85, "differ": 0.85, "mixed": 0.85, "undistorted": 0.7}

def _map_index(name, maps) -> int:
    for i, mp in enumerate(maps):
        if mp.name == name:
            return i
    raise ValueError(f"map not found: {name!r}")


def _year_of(name) -> int:
    """Best-effort chronological key from a map name like ``1835_catherwood``."""
    head = str(name).split("_", 1)[0]
    return int(head) if head.isdigit() else 0


def _open_image(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(path).convert("RGB")


def _crop_to_points(img, xs, ys, margin_px=60, margin_ratio=0.05):
    """Crop ``img`` to the (xs, ys) bounding box; return (cropped, xs', ys')."""
    if not xs:
        return img, xs, ys
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    ox = max(margin_px, int((max_x - min_x) * margin_ratio))
    oy = max(margin_px, int((max_y - min_y) * margin_ratio))
    left, upper = max(int(min_x - ox), 0), max(int(min_y - oy), 0)
    right, lower = min(int(max_x + ox), img.width), min(int(max_y + oy), img.height)
    return (img.crop((left, upper, right, lower)),
            [x - left for x in xs], [y - upper for y in ys])


def _deformed_grids(mp):
    """Return the map's deformed grid in the *global* padded frame, so a locus
    key ``(i, j)`` (which spans the whole corpus lattice) indexes the right node.
    ``padded_token_fields`` attaches these; fall back to the un-padded per-map
    grid only if a map was built outside that path (indices then match locally).
    """
    gx = getattr(mp, "padded_deformed_grid_x", None)
    gy = getattr(mp, "padded_deformed_grid_y", None)
    if gx is None or gy is None:
        gx, gy = mp.grid.graph.deformed_grid_x, mp.grid.graph.deformed_grid_y
    return gx, gy


def _resolve_entry(name, maps, clone_groups=None):
    """Resolve a name to ``(representative_index, [member_indices])``. A clone-
    group id expands to its member maps (first member is the representative used
    for the panel image/grid); a plain map name is its own single member."""
    clone_groups = clone_groups or {}
    members = clone_groups[name] if name in clone_groups else [name]
    present = {mp.name for mp in maps}
    idxs = [_map_index(m, maps) for m in members if m in present]
    if not idxs:  # e.g. a maps_limit run that dropped this group's members
        raise ValueError(f"no maps found for {name!r} "
                         f"(members {members} not in the loaded corpus)")
    return idxs[0], idxs


def _enhance_image(img, enhance):
    """Tone down the map underlay (darken / lift / de-contrast / desaturate) so
    the coloured dots stand out over the scan."""
    if not enhance:
        return img
    from PIL import ImageEnhance
    if enhance.get("darken", 1.0) < 1.0:
        img = ImageEnhance.Brightness(img).enhance(enhance["darken"])
    if enhance.get("whiten", 0.0) > 0.0:
        img = ImageEnhance.Brightness(img).enhance(1.0 + enhance["whiten"])
    if enhance.get("contrast", 0.0) > 0.0:
        img = ImageEnhance.Contrast(img).enhance(1.0 - enhance["contrast"])
    if enhance.get("color", 0.0) > 0.0:
        img = ImageEnhance.Color(img).enhance(1.0 - enhance["color"])
    return img


def _draw_map_dots(mp, xs, ys, cols, ax, dot_size, title, alpha=0.9,
                   alphas=None, enhance=None):
    """Overlay coloured dots on a (hi-res) map image, cropped to the dots' extent.

    The underlay is optionally toned down via ``enhance`` (see
    :data:`_MAP_ENHANCE`); dots have no outline and use either a single ``alpha``
    or per-dot ``alphas``. The full-resolution scan is kept -- output sharpness is
    governed by the figure size/dpi set in the calling plot function.
    """
    from matplotlib.colors import to_rgba
    img = _open_image(mp.image_path)
    img, xs, ys = _crop_to_points(img, xs, ys)
    img = _enhance_image(img, enhance)
    ax.imshow(img)
    if alphas is not None:
        ax.scatter(xs, ys, s=dot_size, edgecolors="none",
                   c=[to_rgba(c, a) for c, a in zip(cols, alphas)])
    else:
        ax.scatter(xs, ys, s=dot_size, c=cols, alpha=alpha, edgecolors="none")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _entry_reading_sets(res, entries):
    """For each entry (a map or a clone group), the set of its *readings* at this
    locus: each positive cluster label, plus a per-entry unique marker for a
    one-off (``-1``) so a hapax counts as a distortion that matches no other
    entry -- it is a real (if unique) distortion, never treated as undistorted.
    State ``0`` (undistorted) contributes nothing."""
    mi = res["map_indices"]
    sets = []
    for k, (_name, member_idxs) in enumerate(entries):
        s = set()
        for o in member_idxs:
            if o in mi:
                v = int(res["clusters"][mi.index(o)])
                if v > 0:
                    s.add(v)
                elif v == -1:
                    s.add(("hapax", k))   # unique to entry k -> matches nobody
        sets.append(s)
    return sets


def _group_match_role(sets, k):
    """Role of entry ``k`` at a locus, from the per-entry reading ``sets`` (see
    the old ``compare_selected_map_lists``): ``match`` if every entry shares a
    common reading, ``undistorted`` if none is distorted or -- when presence is
    split -- for the undistorted entries, ``mixed`` for the distorted entries when
    presence is split, ``differ`` when all are distorted but share no reading (so
    a hapax vs a cluster, or two different clusters, both read as ``differ``)."""
    n_nonempty = sum(1 for s in sets if s)
    if n_nonempty == 0:
        return "undistorted"                          # nobody distorted here
    if n_nonempty == len(sets):                        # every entry present
        return "match" if set.intersection(*sets) else "differ"
    return "undistorted" if not sets[k] else "mixed"   # split presence


def plot_map_comparison(names, clustering_results, maps, colors=None,
                        dot_size=150, axes=None, clone_groups=None,
                        allowed_loci=None, enhance=None, cell_in=8.0, dpi=250):
    """Compare 2+ entries: colour every shared grid node by whether the entries
    carry the *same* distortion there.

    Each name is either a single map or a **clone-group id** (see
    ``clone_groups``). A group is treated as one entry whose label at a locus is
    the *set* of positive clusters its members carry; all member maps of the
    group are still drawn as panels, and they receive identical dot colours
    (the group-level match verdict).

    Teal = all entries share a common positive cluster; white = none is distorted
    (or, where presence is split, the entries that are undistorted here); grey =
    an entry is distorted where another is absent; red = all entries are
    distorted but disagree. Raw shared-distortion view -- no inheritance
    direction and no edge pruning.

    ``clustering_results`` is ``discretize.cluster_tokens`` output; ``maps`` the
    :class:`georef.Map` list aligned to its ``map_indices``; ``clone_groups`` a
    ``{group_id: [member_name, ...]}`` mapping (``cfg.clone_groups``).
    """
    import matplotlib.pyplot as plt
    C = {**_ANALYSIS_COLORS, **(colors or {})}
    clone_groups = clone_groups or {}
    enhance = _MAP_ENHANCE if enhance is None else enhance

    # resolve each name to (name, [member map indices]); a plain map is a
    # single-member entry, a clone group expands to all its members.
    entries = [(n, _resolve_entry(n, maps, clone_groups)[1]) for n in names]

    # one panel per member map, tagged with the entry it belongs to, so every
    # member of a group is shown but coloured by the group-level verdict.
    panels = [(mid, k) for k, (_n, midxs) in enumerate(entries) for mid in midxs]

    if axes is None:
        # at most two columns, so each map stays large (a 3-clone group vs one
        # map lays out 2x2 instead of a squeezed 1x4 row that is hard to read).
        ncol = 1 if len(panels) == 1 else 2
        nrow = (len(panels) + ncol - 1) // ncol
        _, axgrid = plt.subplots(nrow, ncol, figsize=(cell_in * ncol, cell_in * nrow),
                                 squeeze=False, dpi=dpi)
        axes = [axgrid[r][c] for r in range(nrow) for c in range(ncol)]
        for extra in range(len(panels), nrow * ncol):
            axes[extra].axis("off")

    for panel, (mid, k) in enumerate(panels):
        mp = maps[mid]
        gx, gy = _deformed_grids(mp)
        xs, ys, cols, alphas = [], [], [], []
        for (i, j), res in clustering_results.items():
            if allowed_loci is not None and (i, j) not in allowed_loci:
                continue
            if mid not in res["map_indices"]:
                continue
            role = _group_match_role(_entry_reading_sets(res, entries), k)
            xs.append(gx[i, j]); ys.append(gy[i, j])
            cols.append(C[role]); alphas.append(_COMPARE_ALPHA[role])
        _draw_map_dots(mp, xs, ys, cols, axes[panel], dot_size, mp.name,
                       alphas=alphas, enhance=enhance)
    plt.tight_layout()
    return axes[0].figure


def signature_labels(name, clustering_results, maps, clone_groups=None) -> dict:
    """Per grid node ``(i,j)``, the role of ``name``'s distortion there:
    ``innovation`` / ``derivation`` / ``hapax`` / ``undistorted``.

    Let ``R`` be the set of positive readings the witness carries at the locus
    (for a clone group, the union over its members). The locus is a **derivation**
    if *any* predecessor -- a map outside the witness with an earlier year --
    carries one of the readings in ``R``; only when no predecessor shares any of
    them is it an **innovation** (a genuinely new distortion here). If ``R`` is
    empty it is a **hapax** (some member carries the one-off ``-1``) or
    **undistorted** (state ``0``). ``name`` may be a map or a clone-group id; the
    representative member fixes the grid/position and the effective year is the
    earliest member's year.
    """
    rep, members = _resolve_entry(name, maps, clone_groups)
    memberset = set(members)
    entry_year = min(_year_of(maps[o].name) for o in members)
    out = {}
    for (i, j), res in clustering_results.items():
        mi = res["map_indices"]
        if rep not in mi:
            continue
        lbls = res["clusters"]
        states = [int(lbls[mi.index(o)]) for o in members if o in mi]
        readings = {s for s in states if s > 0}
        if not readings:
            out[(i, j)] = "hapax" if any(s == -1 for s in states) else "undistorted"
            continue
        # derivation if any earlier map outside this witness carries a matching
        # reading; otherwise this witness is the innovator at this locus.
        derivative = any(
            o not in memberset
            and int(lbls[k]) in readings
            and _year_of(maps[o].name) < entry_year
            for k, o in enumerate(mi)
        )
        out[(i, j)] = "derivation" if derivative else "innovation"
    return out


def plot_map_signature(name, clustering_results, maps, colors=None,
                       dot_size=150, ax=None, clone_groups=None,
                       allowed_loci=None, enhance=None, cell_in=8.0, dpi=250):
    """Innovation/hapax/derivation signature of a single map or clone group:
    each covered grid node is coloured by the role of its locus for this
    witness. ``clone_groups`` lets
    ``name`` be a clone-group id; ``allowed_loci`` restricts to a set of ``(i,j)``
    keys (e.g. an area selection).
    """
    import matplotlib.pyplot as plt
    C = {**_ANALYSIS_COLORS, **(colors or {})}
    enhance = _MAP_ENHANCE if enhance is None else enhance
    rep, _members = _resolve_entry(name, maps, clone_groups)
    mp = maps[rep]
    gx, gy = _deformed_grids(mp)
    xs, ys, cols = [], [], []
    for (i, j), role in signature_labels(name, clustering_results, maps, clone_groups).items():
        if allowed_loci is not None and (i, j) not in allowed_loci:
            continue
        xs.append(gx[i, j]); ys.append(gy[i, j]); cols.append(C[role])
    if ax is None:
        _, ax = plt.subplots(figsize=(cell_in, cell_in), dpi=dpi)
    _draw_map_dots(mp, xs, ys, cols, ax, dot_size, f"{name} — signature",
                   enhance=enhance)
    return ax.figure
