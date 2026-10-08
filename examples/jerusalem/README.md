# Example data: the Jerusalem corpus

This is the paper's own data ("An Automated Approach to Cartographic
Stemmatology", Vaienti, di Lenardo, Kaplan): the computational outputs of
the stemmatic analysis of 200 maps of Jerusalem (1810-1925), and the
bibliographic metadata for each map. It does **not** contain the map
images themselves; see "Map images" below. It's included here as worked
example input/output for this package (see "Reproducing the paper's
results" below); this repository as a whole, data included, is archived
on Zenodo with a permanent DOI -- see the badge/citation in the top-level
[`README.md`](../../README.md).

Produced with this package at the parameters listed in `parameters.yaml`.

## Files

### `collation_matrix.csv` / `collation_matrix.parquet`
The core input to the stemmatic analysis, **before** clone-group merging:
one row per individual map/edition (200 rows -- the same granularity as
`map_metadata.json`'s `map_id`), one column per locus (geospatial grid
point, keyed by `locus_id`, defined in `locus_geospatial_position.csv`).
Both files hold identical data; the `.parquet` is what `cartostemma stemma`
reads directly (see "Reproducing the paper's results" below), the `.csv` is
for inspection without pandas. Each cell is the distortion-cluster state of
that map at that locus:

| Value | Meaning |
|---|---|
| positive integer | valid distortion cluster; maps sharing the same value at a locus exhibit the same distortion pattern there |
| `0` | distortion below threshold; treated as undistorted |
| `-1` | noise: distortion present but not similar enough to any other map at that locus |
| empty / NaN | the map does not cover this locus |

This is the single source of truth for all pre-pruning information in this
release: clone-group merging (identical reprints/translations collapsed
into one witness, per `clone_groups` in `parameters.yaml`), the locus-level
matches between any two witnesses, and the innovation/derivation/hapax
classification of each witness are all functions of this
matrix plus `maps.csv`, computed by the `cartostemma` package.

### `maps.csv`
`map_id, year` for every map in `collation_matrix.csv` -- the minimal
interchange `cartostemma stemma` needs alongside the collation matrix (year
drives chronological ordering and edge direction). `map_id` matches
`map_metadata.json`'s `map_id`; see that file for full bibliographic detail.

### `locus_geospatial_position.csv`
`locus_id, x, y` -- the projected coordinates (EPSG:28193) of every locus
used as a column in `collation_matrix.csv`. Needed to know the geospatial position of each locus.

### `map_metadata.json`
Bibliographic entry for every individual map/edition in the corpus (200
records, same granularity and `map_id`s as `collation_matrix.csv`). A JSON
array of objects, one per map:

| Field | Meaning |
|---|---|
| `map_id` | unique identifier for this specific map/edition |
| `merged_witness_id` | the clone-group id this map is merged into for analysis (equal to `map_id` if not merged); matches the witness ids in `edges_full.csv`/`stemma.graphml` |
| `title` | as recorded in the source catalogue |
| `mapmakers`, `surveyors`, `engravers`, `publishers` | arrays of names, as recorded in the source catalogue |
| `publication_date`, `survey_date` | ISO 8601; precision to the decade or to the century (e.g. `188?` or `18??`) where the source only gives that |
| `language`, `paper_extent`, `scale` | as recorded in the source catalogue, where available |
| `nli_code` | the item's National Library of Israel catalogue identifier, where we have one on record. **Absence does not mean the map is not held by the NLI**: where Wikimedia Commons also held the map, that source was given precedence. |
| `url` | a link to view the source image.|

Empty arrays (`[]`) mean no names of that role are recorded for the map;
`null` means the field is unknown or, for `url`, that no externally
accessible link was available.


### `edges_full.csv`
Every candidate map-to-map connection the pruning procedure evaluated,
**unfiltered** -- including pairs below the display thresholds used in the
paper's figures. One row per ordered pair `(source, target)`:

| Column | Meaning |
|---|---|
| `source`, `target` | witness ids (chronological source -> later target) |
| `n_shared` | number of loci where source and target share a distortion cluster after pruning |
| `n_target_loci` | size of the target's distorted-loci set (the denominator of the inheritance ratio) |
| `inheritance_ratio` | `n_shared / n_target_loci` |
| `tier` | `primary` (ratio >= 0.5), `secondary` (0.15-0.5), or `hidden` (below 0.15, or fewer than 15 shared loci) -- see paper §5.4 |
| `pruning_method` | `single_ancestor` (the paper's primary method; see `parameters.yaml`) |

### `stemma.graphml`
The same edge data as `edges_full.csv`, but **filtered to `primary` and
`secondary` tier only** -- i.e. exactly the graph rendered in the paper's
stemma figures, in GraphML for direct loading into Gephi, Cytoscape, igraph,
or networkx. Node attributes include each witness's year and its
innovation/derivation/hapax role.

### `parameters.yaml`
The exact pipeline parameters used to produce this release (grid resolution,
clustering thresholds, pruning method, display thresholds, clone groups).

## Reproducing the paper's results

`collation_matrix.parquet` + `maps.csv` together are exactly the input
`cartostemma stemma` expects (see the
[`cartostemma` package](https://github.com/BeatriceVaienti/cartostemma)), so the paper's stemma
can be reproduced from this release alone -- no map images needed:

```bash
mkdir -p results
cp collation_matrix.parquet maps.csv results/
cartostemma stemma config.yaml --pruning single   # config.yaml: paths.output: ./results,
                                                   # stemma/clone_groups from parameters.yaml
```

This regenerates `edges_full.csv`'s 935 rows and `stemma.graphml` exactly
(verified: identical `n_shared`, `n_target_loci`, `tier`, and
`inheritance_ratio` for every edge).

## Map images

This release does not redistribute the map images. Most are drawn from
openly digitized library/archive collections; `map_metadata.json` gives each map's `url` for viewing the source scan directly at the holding institution.

## License

This dataset (all files above) is released under CC-BY 4.0. Please cite the
paper if you use it.
