"""Stage A -- georeferencing and distortion-token encoding.

For each map, a Thin-Plate-Spline RBF interpolator is fitted from the ground
control points (real-world -> pixel), a regular grid of loci is warped through it,
and at every grid node the **distortion token** is computed: the 36-value vector
of all-to-all Euclidean distances among the node's 3x3 neighbourhood (C(9,2)=36).
Nodes outside the map's coverage (alpha shape of the GCPs) are masked to NaN.
Rendering of these results belongs in ``viz.py``, not here.

Input GCP format: QGIS georeferencer ``.points`` CSV, i.e. columns
``mapX, mapY, sourceX, sourceY`` (plus optional ``enable, dX, dY, residual``).
"""
from __future__ import annotations

from dataclasses import dataclass

import alphashape
import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import pdist
from shapely.geometry import Point, Polygon

# dataset loading lives in io.py (no geospatial deps); re-exported for convenience
from .io import extract_epsg, load_dataset  # noqa: F401

# Grid origin (base point) in the projected CRS; the regular grid is anchored
# here so that every map's grid nodes fall on the same world coordinates.
DEFAULT_BASE_X = 172119.73
DEFAULT_BASE_Y = 1131710.35


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def create_grid(bounds, base_x, base_y, grid_size):
    """Regular grid covering ``bounds``, anchored on (base_x, base_y) so grids of
    different maps share node coordinates. Returns meshgrid (x_grid, y_grid)."""
    try:
        x_min, x_max = bounds.left, bounds.right
        y_min, y_max = bounds.bottom, bounds.top
    except AttributeError:
        x_min, x_max, y_min, y_max = bounds[0], bounds[1], bounds[2], bounds[3]

    left = int(np.ceil((base_x - x_min) / grid_size))
    right = int(np.ceil((x_max - base_x) / grid_size))
    bottom = int(np.ceil((base_y - y_min) / grid_size))
    top = int(np.ceil((y_max - base_y) / grid_size))

    return np.meshgrid(
        np.arange(base_x - left * grid_size, base_x + right * grid_size, grid_size),
        np.arange(base_y - bottom * grid_size, base_y + top * grid_size, grid_size),
    )


def find_alpha_shape(points, alpha=0.001):
    """Concave hull (alpha shape) of the GCP points; falls back to the convex
    hull if the alpha shape is not a single polygon covering all points."""
    shape = alphashape.alphashape(points, alpha)
    if isinstance(shape, Polygon) and all(
        shape.contains(Point(p)) or shape.touches(Point(p)) for p in points
    ):
        return shape
    return alphashape.alphashape(points, 0)


def process_grid(grid: "Grid") -> np.ndarray:
    """Stack a grid's base coordinates into an (H, W, 2) array."""
    coords = np.full((*grid.base_grid_x.shape, 2), np.nan)
    coords[:, :, 0] = grid.base_grid_x
    coords[:, :, 1] = grid.base_grid_y
    return coords


def find_bounding_box(grids: list["Grid"]):
    """Overall (min_x, max_x, min_y, max_y) covering a list of grids."""
    min_x = min_y = np.inf
    max_x = max_y = -np.inf
    for grid in grids:
        coords = process_grid(grid)
        valid = coords[~np.isnan(coords[:, :, 0])]
        if valid.size:
            min_x, max_x = min(min_x, valid[:, 0].min()), max(max_x, valid[:, 0].max())
            min_y, max_y = min(min_y, valid[:, 1].min()), max(max_y, valid[:, 1].max())
    return min_x, max_x, min_y, max_y


def pad_grid_and_deformed(grid, token_vectors, min_x, max_x, min_y, max_y):
    """Place a map's token grid into a common padded frame spanning the whole
    corpus, so every map shares the same ``grid_shape``. Returns the padded base
    grids, token vectors (NaN outside coverage), and deformed grids."""
    dx = grid.base_grid_x[0, 1] - grid.base_grid_x[0, 0]
    dy = grid.base_grid_y[1, 0] - grid.base_grid_y[0, 0]
    xs = np.arange(min_x, max_x + 1e-6, dx)
    ys = np.arange(min_y, max_y + 1e-6, dy)
    px, py = np.meshgrid(xs, ys)

    H, W = px.shape
    ptv = np.full((H, W, token_vectors.shape[2]), np.nan)
    pdx = np.full((H, W), np.nan)
    pdy = np.full((H, W), np.nan)

    ix0 = int(round((grid.base_grid_x.min() - min_x) / dx))
    iy0 = int(round((grid.base_grid_y.min() - min_y) / dy))
    h0, w0 = grid.base_grid_x.shape
    ptv[iy0 : iy0 + h0, ix0 : ix0 + w0, :] = token_vectors
    pdx[iy0 : iy0 + h0, ix0 : ix0 + w0] = grid.graph.deformed_grid_x
    pdy[iy0 : iy0 + h0, ix0 : ix0 + w0] = grid.graph.deformed_grid_y
    return px, py, ptv, pdx, pdy


# --------------------------------------------------------------------------- #
# Grid: undeformed reference grid + coverage mask                              #
# --------------------------------------------------------------------------- #
@dataclass
class Grid:
    """The undeformed regular grid of a map and its GCP-derived coverage mask."""

    gcp_df: pd.DataFrame
    grid_size: int
    base_x: float
    base_y: float
    bounds: object = None
    real_points: np.ndarray = None
    pixel_points: np.ndarray = None
    alpha_shape: Polygon = None
    alpha_shape_mask: np.ndarray = None
    base_grid_x: np.ndarray = None
    base_grid_y: np.ndarray = None
    graph: "Graph" = None

    def __post_init__(self):
        self.real_points = np.array(self.gcp_df[["mapX", "mapY"]])
        self.pixel_points = np.array(self.gcp_df[["sourceX", "sourceY"]])
        if self.bounds is None:
            xs, ys = self.real_points[:, 0], self.real_points[:, 1]
            self.bounds = rasterio.coords.BoundingBox(
                left=xs.min(), bottom=ys.min(), right=xs.max(), top=ys.max()
            )
        self.base_grid_x, self.base_grid_y = create_grid(
            self.bounds, self.base_x, self.base_y, self.grid_size
        )
        self.alpha_shape = find_alpha_shape(self.real_points)
        self._compute_mask()

    def _compute_mask(self):
        mask = np.zeros_like(self.base_grid_x, dtype=bool)
        for i in range(self.base_grid_x.shape[0]):
            for j in range(self.base_grid_x.shape[1]):
                if self.alpha_shape.contains(Point(self.base_grid_x[i, j], self.base_grid_y[i, j])):
                    mask[i, j] = True
        self.alpha_shape_mask = mask


# --------------------------------------------------------------------------- #
# Graph: distortion tokens                                                     #
# --------------------------------------------------------------------------- #
# 3x3 neighbourhood offsets (row-major); the token is the 36 all-to-all
# distances among these nine nodes.
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
# condensed index pairs (a < b), matching scipy pdist ordering, for the 36 dists
_PAIR_A = np.array([a for a in range(9) for b in range(a + 1, 9)])
_PAIR_B = np.array([b for a in range(9) for b in range(a + 1, 9)])


class Graph:
    """Computes and stores a map's planimetric distortion as a field of tokens."""

    def __init__(self, grid: Grid):
        # 36-value all-to-all distance vector of an undeformed 3x3 grid (reference)
        self.undeformed_vec = pdist(np.array(_OFFSETS), metric="euclidean")

        base_grid = np.stack((grid.base_grid_x, grid.base_grid_y), axis=-1)
        transformed = self._warp(grid.gcp_df, base_grid)
        self.deformed_grid_x = transformed[:, :, 0]
        self.deformed_grid_y = transformed[:, :, 1]

        tokens = self._compute_token_vectors(transformed)
        self.token_vectors = tokens
        masked = tokens.copy()
        masked[~grid.alpha_shape_mask] = np.nan
        self.alpha_masked_token_vectors = masked

    @staticmethod
    def _warp(gcp_df, grid):
        """Thin-plate-spline RBF from real-world coords to pixel coords.

        Vectorized: the whole grid is evaluated in a single ``rbf`` call
        (identical result to a per-point loop, ~15x+ faster on dense grids)."""
        real = np.column_stack((gcp_df["mapX"].values, gcp_df["mapY"].values))
        pixel = np.column_stack((gcp_df["sourceX"].values, -gcp_df["sourceY"].values))
        rbf = RBFInterpolator(real, pixel, kernel="thin_plate_spline", neighbors=25)
        h, w, _ = grid.shape
        return rbf(grid.reshape(-1, 2)).reshape(h, w, 2)

    @staticmethod
    def _compute_token_vectors(transformed):
        """36-value all-to-all distance token at every node.

        Vectorized over the grid: neighbourhoods are gathered by NaN-padded
        slicing and all 36 pairwise distances computed at once (identical result
        to the per-node ``pdist`` loop, out-of-bounds neighbours -> NaN)."""
        h, w, _ = transformed.shape
        padded = np.full((h + 2, w + 2, 2), np.nan)
        padded[1:h + 1, 1:w + 1] = transformed
        neigh = np.empty((h, w, 9, 2))
        for k, (dy, dx) in enumerate(_OFFSETS):
            neigh[:, :, k, :] = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w, :]
        diffs = neigh[:, :, _PAIR_A, :] - neigh[:, :, _PAIR_B, :]
        return np.linalg.norm(diffs, axis=-1)


# --------------------------------------------------------------------------- #
# Map: a georeferenced witness                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Map:
    """A georeferenced historical map: its GCPs, undeformed grid and token field."""

    name: str
    gcp_df: pd.DataFrame
    grid_size: int
    base_x: float = DEFAULT_BASE_X
    base_y: float = DEFAULT_BASE_Y
    metadata: dict = None
    folder_path: str = None
    image_path: str = None
    epsg: int = None
    grid: Grid = None

    def __post_init__(self):
        self.grid = Grid(self.gcp_df, self.grid_size, self.base_x, self.base_y)
        self.grid.graph = Graph(self.grid)

    @property
    def token_vectors(self) -> np.ndarray:
        """This map's alpha-masked token field (NaN outside coverage)."""
        return self.grid.graph.alpha_masked_token_vectors


def create_map_object(map_info: dict, grid_size: int,
                      base_x: float = DEFAULT_BASE_X, base_y: float = DEFAULT_BASE_Y) -> Map:
    """Build a :class:`Map` from a dataset entry (see :func:`load_dataset`)."""
    return Map(
        name=map_info["folder"],
        gcp_df=map_info["points"],
        grid_size=grid_size,
        base_x=base_x,
        base_y=base_y,
        metadata=map_info.get("metadata"),
        folder_path=map_info.get("folder_path"),
        image_path=map_info.get("image_path"),
        epsg=int(map_info["epsg"]) if map_info.get("epsg") else None,
    )


# --------------------------------------------------------------------------- #
# Corpus-level assembly                                                        #
# --------------------------------------------------------------------------- #
def padded_token_fields(maps: list[Map]):
    """Warp every map's token field into a shared padded frame.

    Returns ``(token_vectors_per_map, grid_shape, base_grid_x, base_grid_y)``.
    The list is aligned so that node ``(i, j)`` is the same world location across
    all maps, and ``base_grid_x/y`` give that node's projected coordinates (used
    to geolocate loci for area selection).

    Side effect: each map gets ``padded_deformed_grid_x/y`` attributes (the map's
    own deformed grid placed in the shared global frame, NaN outside coverage), so
    a global node ``(i, j)`` can be located on that map's image for the map-level
    analyses (:func:`viz.plot_map_signature` / :func:`viz.plot_map_comparison`).
    """
    grids = [m.grid for m in maps]
    min_x, max_x, min_y, max_y = find_bounding_box(grids)
    fields = []
    base_x = base_y = None
    for m in maps:
        px, py, ptv, pdx, pdy = pad_grid_and_deformed(
            m.grid, m.token_vectors, min_x, max_x, min_y, max_y
        )
        fields.append(ptv)
        m.padded_deformed_grid_x = pdx
        m.padded_deformed_grid_y = pdy
        base_x, base_y = px, py  # identical padded frame for every map
    grid_shape = fields[0].shape[:2]
    return fields, grid_shape, base_x, base_y


def load_area(path: str, target_epsg: int, buffer: float = 100.0):
    """Load a study-area polygon from a shapefile (or any geopandas-readable
    file), reproject to ``target_epsg`` and buffer it. Returns a shapely geometry
    for restricting the analysis to a region of interest."""
    import geopandas as gpd  # local import: only needed for polygon areas

    gdf = gpd.read_file(path).to_crs(epsg=target_epsg)
    geom = gdf.unary_union
    return geom.buffer(buffer) if buffer else geom
