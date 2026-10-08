"""Build the cartographic collation matrix and classify its loci.

Pure functions operating on a pandas DataFrame (the collation matrix), which is
the stable interchange between the geospatial front half of the pipeline and
the stemma back half.

State codes:
    v > 0     valid distortion cluster (shared positive value == shared pattern)
    0         distortion below threshold (undistorted)
    -1        noise: distortion present but not groupable
    NaN       map does not cover this locus
"""
from __future__ import annotations

from typing import Hashable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Cluster results -> collation matrix                                          #
# --------------------------------------------------------------------------- #
def assign_states_from_clusters(
    clustering_results: dict,
    col_mapping: dict[Hashable, Hashable],
    index: list[str],
) -> pd.DataFrame:
    """Turn per-locus clustering output into the collation matrix.

    ``clustering_results`` maps a locus to a dict with ``map_indices`` and
    ``clusters``. The cluster labels are already final states -- ``0``
    undistorted, ``-1`` noise, positive integers for shared clusters (see
    ``discretize.cluster_tokens``) -- so they are placed directly. A (map, locus)
    cell is left ``NaN`` when the map does not cover the locus.
    """
    columns = [col_mapping[t] for t in clustering_results]
    df = pd.DataFrame(index=index, columns=columns, dtype=object)
    for token, data in clustering_results.items():
        col_idx = df.columns.get_loc(col_mapping[token])
        for map_idx, label in zip(data["map_indices"], data["clusters"]):
            df.iat[int(map_idx), col_idx] = int(label)
    return df


# --------------------------------------------------------------------------- #
# Clone-group merging                                                          #
# --------------------------------------------------------------------------- #
def _as_set(val) -> set:
    if isinstance(val, set):
        return val
    if pd.isna(val):
        return set()
    return {val}


def merge_clone_groups(
    states: pd.DataFrame,
    clone_groups: dict[str, list[str]],
    years: dict[str, int],
) -> pd.DataFrame:
    """Collapse clone groups into single representative witnesses.

    Maps not assigned to any clone group form their own singleton group. Within
    a group, each locus keeps the *union* of the members' non-NaN state values
    (so a later match succeeds if it overlaps any member's value). The group's
    effective year is the earliest member's year.

    Returns a DataFrame indexed by group id, with the locus columns plus two
    extra columns ``year`` (effective/earliest) and ``maps`` (member names).
    """
    # map name -> group id (own name if not a clone member)
    name_to_group: dict[str, str] = {}
    for gid, members in clone_groups.items():
        for name in members:
            name_to_group[name] = gid

    group_members: dict[str, list[str]] = {}
    for name in states.index:
        gid = name_to_group.get(name, name)
        group_members.setdefault(gid, []).append(name)

    loci = list(states.columns)
    df = pd.DataFrame(index=list(group_members), columns=loci, dtype=object)

    for gid, members in group_members.items():
        for locus in loci:
            combined: set = set()
            for name in members:
                combined |= _as_set(states.at[name, locus])
            df.at[gid, locus] = combined if combined else np.nan

    df["year"] = pd.Series(
        {gid: min(years[m] for m in members) for gid, members in group_members.items()}
    )
    df["maps"] = pd.Series(group_members)
    return df


# --------------------------------------------------------------------------- #
# Area selection: full extent vs. a region of interest                        #
# --------------------------------------------------------------------------- #
def select_loci(loci_xy: pd.DataFrame, area=None, buffer: float = 0.0) -> list:
    """Return the loci to include in the analysis.

    ``loci_xy`` is indexed by locus name with columns ``x``, ``y`` (projected
    coordinates of each grid node). ``area`` selects the scope:
        * ``None``                      -> the full map extent (all loci);
        * ``(xmin, ymin, xmax, ymax)``  -> a bounding box (pure numpy);
        * a shapely geometry            -> a polygon region of interest.

    This is what lets a single corpus yield either a global stemma (full extent)
    or a local stemma focused on a region. Loci are classified corpus-wide
    first; this filter then restricts which loci drive the stemma.
    """
    if area is None:
        return list(loci_xy.index)
    if isinstance(area, (tuple, list)) and len(area) == 4:
        xmin, ymin, xmax, ymax = area
        xmin, ymin, xmax, ymax = xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer
        m = (
            (loci_xy["x"] >= xmin) & (loci_xy["x"] <= xmax)
            & (loci_xy["y"] >= ymin) & (loci_xy["y"] <= ymax)
        )
        return list(loci_xy.index[m])
    # otherwise: a shapely geometry (point-in-polygon)
    from shapely.geometry import Point

    geom = area.buffer(buffer) if buffer else area
    return [name for name, row in loci_xy.iterrows() if geom.contains(Point(row["x"], row["y"]))]


# --------------------------------------------------------------------------- #
# Locus classification: innovation / derivation / hapax                       #
# --------------------------------------------------------------------------- #
def classify_loci(df_groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label every (group, locus) cluster as innovation / derivation / hapax.

    Groups are processed in chronological order. The first group to exhibit a
    given (locus, cluster) value is the **innovation** (the innovation source of
    that pattern); later groups sharing it are **derivation**; a noise value
    (-1) is a **hapax**. State 0 and other non-positive values carry no
    genealogical weight and are ignored.

    Returns:
        df_stemma : per cell, a dict {cluster_value: label} (or NaN)
        df_labels : per cell, a single collapsed label by priority
                    innovation > derivation > hapax (or NaN)
    """
    loci = [c for c in df_groups.columns if c not in ("year", "maps")]
    order = sorted(df_groups.index, key=lambda g: df_groups.at[g, "year"])

    earliest_mention: dict[tuple, str] = {}

    def interpret(cluster_val: int, group_id: str, locus) -> str | None:
        if cluster_val == -1:
            return "hapax"
        if cluster_val <= 0:
            return None
        key = (locus, cluster_val)
        if key not in earliest_mention:
            earliest_mention[key] = group_id
            return "innovation"
        return "derivation"

    df_stemma = pd.DataFrame(index=order, columns=loci, dtype=object)
    for g in order:
        for locus in loci:
            cell = df_groups.at[g, locus]
            if pd.isna(cell) if not isinstance(cell, set) else not cell:
                df_stemma.at[g, locus] = np.nan
                continue
            labels = {}
            for cval in _as_set(cell):
                lbl = interpret(cval, g, locus)
                if lbl:
                    labels[cval] = lbl
            df_stemma.at[g, locus] = labels or np.nan

    priority = ("innovation", "derivation", "hapax")
    df_labels = pd.DataFrame(index=order, columns=loci, dtype=object)
    for g in order:
        for locus in loci:
            cell = df_stemma.at[g, locus]
            if isinstance(cell, dict):
                vals = set(cell.values())
                df_labels.at[g, locus] = next((p for p in priority if p in vals), np.nan)
            else:
                df_labels.at[g, locus] = np.nan

    return df_stemma, df_labels
