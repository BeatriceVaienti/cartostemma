# cartostemma

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23239666.svg)](https://doi.org/10.5281/zenodo.23239666)

**Automated cartographic stemmatology.** 
Reconstructs genealogical relationships among historical maps of the same
geographic area from their planimetric distortions, adapting stemmatological
methods from textual criticism.

Companion code for *"An Automated Approach to Cartographic Stemmatology"*
(Vaienti, di Lenardo, Kaplan; submitted to *Digital Scholarship in the
Humanities*). See [`CITATION.cff`](CITATION.cff).

The paper's own results (the Jerusalem corpus's collation matrix, computed
stemma edges, and map bibliographic metadata) are included here as worked
example data -- [`examples/jerusalem/`](examples/jerusalem/) -- and are also
archived separately with a permanent DOI for citation: **[DOI link, to be
added]**. `collation_matrix.parquet` + `maps.csv` from either copy are valid
input to `cartostemma stemma` as-is -- the paper's stemma can be reproduced
without the map images, which neither this repository nor that dataset
redistribute (see [`examples/jerusalem/README.md`](examples/jerusalem/README.md)
for why).

## Installation

The geospatial stack (GDAL / rasterio / pyproj) installs far more reliably
from **conda-forge** than from PyPI. Recommended:

```bash
conda create -n cartostemma -c conda-forge python=3.11 rasterio pyproj shapely \
    scikit-image opencv gdal
conda activate cartostemma
pip install -e .            # this package
```

Or use the full conda environment: `conda env create -f environment.yml`.

The pure stemma engine (`cartostemma.collation`, `cartostemma.stemma`) only needs
`pandas`/`numpy` and can be used without the geospatial stack -- except for
polygon-based local-region selection (`stemma.area: regions/haram.shp`),
which needs `shapely`; a bounding-box area or the full-extent default don't.

## Data 
### Input: a dataset folder (with one sub-folder per map)
For example:

```
datasets/jerusalem/
  1841_kiepert/
    1841_kiepert.tif          # the map image (tif/png/jpg)
    1841_kiepert.points       # ground control points (QGIS georeferencer format)
    metadata.json             # optional: bibliographic fields (title, date, ...)
  1842_bartlett/
    ...
```

The GCP file is a QGIS georeferencer `.points` file: one CRS header line,
then rows of `mapX, mapY, sourceX, sourceY[, ...]` control-point pairs
(`mapX`/`mapY` the georeferenced coordinates, `sourceX`/`sourceY` the
corresponding pixel coordinates in the map image). Map dates are parsed
from the folder-name prefix (`1841_...`).

### Interchange — the collation matrix (the stable API boundary)

The geospatial front half and the stemma back half communicate through one
documented artefact, **not** pickled Python objects:

```
collation_matrix.parquet   rows = maps, cols = loci, cells = state codes
```

State codes: `v>0` distortion cluster · `0` undistorted · `-1` noise · `NaN` no
coverage. You can bring your **own** collation matrix and run only the stemma
stage.

### Output

```
results/jerusalem/
  collation_matrix.parquet
  maps.csv                # map_id, year, image_path -- written by `run` only
  loci.csv                # locus, x, y -- written by `run` only (area selection)
  edges.csv               # source, target, n_shared, inheritance_ratio, tier
  map_fingerprints.csv    # innovation / derivation / hapax counts per map
  locus_labels.parquet
  stemma.graphml           # open in Gephi / Cytoscape for interactive exploration
  stemma_summary.csv       # per-map predecessors/successors summary
  stemma.png                # static forest figure (needs the [viz] extra)
```

`maps.csv`/`loci.csv` are produced by `cartostemma run` (the full pipeline);
`cartostemma stemma` expects them to already exist as input (see Quickstart
below) and only writes the files from `edges.csv` down.

## The core scripts

The pipeline is a package, `cartostemma/`, driven by the `cartostemma` CLI
(`cartostemma/cli.py`). Each module implements one stage of the paper's
pipeline, operating on plain pandas DataFrames / a config object so each stage
can be used independently:

| Module | What it does |
|---|---|
| `io.py` | Scans a dataset folder, reads GCP `.points` files. No geospatial dependencies. |
| `georef.py` | Georeferences each map (thin-plate-spline warp from its GCPs) and computes the **distortion token** at every grid node — the 36-value vector of all-to-all distances among its 3×3 neighbourhood. |
| `metrics.py` | Distance metrics on distortion tokens (`ratio_distance`, the distortion-aware ratio distance used for clustering). |
| `discretize.py` | Clusters tokens into discrete states per locus: `0` undistorted, `-1` noise, `1, 2, ...` shared distortion clusters. |
| `collation.py` | Builds the collation matrix from the clustering, merges clone groups (reprints/copies of the same plate), selects a region of interest, and classifies each cell as innovation / derivation / hapax. |
| `stemma.py` | The core stemmatic inference: single-ancestor edge pruning (the paper's main method; a triplet-based alternative is also provided), the Inheritance Ratio, and per-map fingerprints. |
| `viz.py` | Graph export (GraphML for Gephi/Cytoscape) and the static stemma forest figure. Also provides the map-level "beyond the stemma" analyses (`plot_map_signature`, `plot_map_comparison` — innovation/derivation signature and pairwise comparison), callable directly in Python; they are not wired to a CLI command. |
| `config.py` | Loads and validates `config.yaml`. |
| `cli.py` | Wires the stages together into the `cartostemma` command (`stemma`, `run`). |

All tunable parameters live in `config.yaml` (no magic numbers); see
`config.example.yaml` for every field.

## Quickstart (stemma stage — works from a pre-built collation matrix)

Given a `collation_matrix.parquet` and a `maps.csv` (`map_id, year`) in the
output folder:

```bash
cp config.example.yaml config.yaml     # edit paths + parameters
cartostemma stemma config.yaml
```

For the full pipeline (georeferencing → discretization → collation →
stemma) from a dataset of maps + GCPs:

```bash
cartostemma run config.yaml
```

Both commands accept `--pruning {single,triplet}` (default `single`, the
paper's main method; `triplet` is the more conservative alternative also
discussed in the paper).

The map-level analyses (a single map's innovation/derivation signature, or a
pairwise comparison) are not wired to a CLI command; call them directly once
you have a clustering result (e.g. from `cartostemma.cli._build_corpus`):

```python
from cartostemma import viz
viz.plot_map_signature("1841_kiepert", clustering, maps, clone_groups=cfg.clone_groups)
viz.plot_map_comparison(["1841_kiepert", "1842_bartlett"], clustering, maps)
```

## Scope: global stemma vs. local region

The same corpus yields either a **global stemma** (full map extent) or a
**local stemma** focused on a region of interest, set in `config.yaml`:

```yaml
stemma:
  area: null                          # full extent (global stemma)
  # area: [xmin, ymin, xmax, ymax]    # a bounding box
  # area: regions/haram.shp           # a polygon region of interest (local stemma)
  area_buffer: 100.0
```

Loci are classified corpus-wide, then restricted to the area, so "innovation"
still means first-in-corpus. For a local stemma, pair an area with a finer grid
(`grid.size_m: 50`, vs. 150 for the global run) — as in the paper's case studies.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Unit tests exercise each engine stage (discretization, collation, area
selection, single-ancestor and triplet pruning / inheritance ratio, graph
construction) on small synthetic inputs — no geospatial stack or real data
required.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
