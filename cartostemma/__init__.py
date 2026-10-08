"""cartostemma -- automated cartographic stemmatology.

Reconstructs genealogical relationships among historical maps of the same area
from their planimetric distortions, adapting stemmatological methods from
textual criticism (paper: "An Automated Approach to Cartographic
Stemmatology").
"""
from __future__ import annotations

__version__ = "0.1.0"

# Submodules are imported explicitly by the caller (e.g. ``from cartostemma
# import stemma``) so that the pure-pandas engine (collation, stemma) can be
# used without the geospatial stack (pyproj/rasterio) that georef needs.
__all__ = ["collation", "discretize", "io", "metrics", "stemma", "config"]
