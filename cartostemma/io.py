"""Dataset I/O -- scanning map folders and reading GCP files.

Kept free of the geospatial stack (pandas/os/json only) so a dataset can be
inspected and validated without rasterio/GDAL installed.

Input layout: one sub-folder per map, each containing a QGIS georeferencer
``.points`` file (first line = CRS header, then ``mapX, mapY, sourceX,
sourceY[, ...]``), a map image, and optionally a ``metadata.json``.
"""
from __future__ import annotations

import json
import os

import pandas as pd

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf")
# derived/auxiliary rasters to avoid when picking the original scan
_AUX_KEYWORDS = ("mask", "thin_plate", "helmert", "modified", "warped", "feature",
                 "geotiff", "aux", "corrected", "_geo", ".thumb")
_IMG_EXT_PREF = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf")


def pick_image(files: list[str]) -> str | None:
    """Pick a map folder's *original* scan: prefer a non-auxiliary raster
    (not a mask / thin_plate / helmert / geotiff derivative), and prefer
    jpg/jpeg/png over tif. E.g. ``clarke_1810.jpeg`` over
    ``clarke_1810_thin_plate.tif`` or ``clarke_1810_mask.png``."""
    imgs = [f for f in files if f.lower().endswith(_IMAGE_EXTS)]
    if not imgs:
        return None

    def rank(f: str):
        fl = f.lower()
        is_aux = any(k in fl for k in _AUX_KEYWORDS)
        ext = next((i for i, e in enumerate(_IMG_EXT_PREF) if fl.endswith(e)), len(_IMG_EXT_PREF))
        return (is_aux, ext, len(f))  # non-aux, preferred ext, then shortest name

    return min(imgs, key=rank)


def extract_epsg(crs_info_str: str, default: str = "28193") -> str:
    """Best-effort EPSG code extraction from a QGIS CRS info string."""
    try:
        start = crs_info_str.find('ID["EPSG".6')
        end = crs_info_str.find("]]", start)
        return crs_info_str[start:end].split("'")[2].strip()
    except Exception:
        return default


def pick_points_file(files: list[str], image: str | None) -> str | None:
    """Choose the GCP ``.points`` file for a map folder.

    Priority: the plain ``<image>.points`` **first**, then any
    ``*_corrected.points``, then the auto-generated ``guessed_gcp.points``, then
    any other ``.points``. Preferring the plain file over ``_corrected`` matters:
    a few folders carry both, and choosing differently georeferences the map from
    different control points, which perturbs its distortion tokens and clusters."""
    stem = os.path.splitext(image)[0] if image else None
    candidates = [f for f in files if f.endswith(".points")]

    def first(pred):
        return next((f for f in candidates if pred(f)), None)

    return (
        (first(lambda f: f == f"{stem}.points") if stem else None)
        or first(lambda f: f.endswith("_corrected.points"))
        or first(lambda f: f == "guessed_gcp.points")
        or (candidates[0] if candidates else None)
    )


def load_dataset(dataset_dir: str) -> list[dict]:
    """Read a dataset folder (one sub-folder per map) into ``map_info`` dicts
    with keys ``folder``, ``folder_path``, ``points`` (DataFrame), ``epsg`` and,
    when present, ``image_path`` and ``metadata``."""
    entries: list[dict] = []
    for folder in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        files = [f for f in os.listdir(folder_path) if f != ".DS_Store"]

        img = pick_image(files)
        points_name = pick_points_file(files, img)
        if points_name is None:
            continue
        points_file = os.path.join(folder_path, points_name)

        first_row = pd.read_csv(points_file, nrows=1, delimiter=",")
        info: dict = {
            "folder": folder,
            "folder_path": folder_path,
            "points": pd.read_csv(points_file, skiprows=1, delimiter=","),
            "epsg": extract_epsg(str(first_row.columns)),
            "points_file": points_name,
        }
        if img:
            info["image_path"] = os.path.join(folder_path, img)
        meta = next((f for f in files if f.endswith("metadata.json")), None)
        if meta:
            with open(os.path.join(folder_path, meta)) as fh:
                info["metadata"] = json.load(fh)
        entries.append(info)
    return entries
