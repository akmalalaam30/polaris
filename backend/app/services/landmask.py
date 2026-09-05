"""Antarctic land / ice-shelf mask.

Routing and risk both need to know which cells a ship can physically occupy.
The mask is derived from the **land and coast flags carried by the NSIDC
G02135 sea-ice concentration GeoTIFFs** (values 253 = coast, 254 = land on the
25 km polar-stereographic grid).  That is a real, published mask - not an
invented coastline.

The packed mask is cached as ``data/demo/nsidc_land_mask.npz`` so demo mode and
the test-suite work fully offline; :func:`build_land_mask_from_nsidc` rebuilds
it from a freshly downloaded GeoTIFF when networking is available.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.utils.geo import (
    NSIDC_SOUTH_SHAPE,
    GridSpec,
    haversine_km,
    nsidc_grid_latlon,
    regrid_nearest,
)
from app.utils.logging import get_logger

log = get_logger("services.landmask")

MASK_FILENAME = "nsidc_land_mask.npz"

# NSIDC G02135 GeoTIFF flag values (concentration product, scaled by 10).
FLAG_POLE_HOLE = 2510
FLAG_UNUSED = 2520
FLAG_COAST = 2530
FLAG_LAND = 2540
FLAG_MISSING = 2550


def mask_path() -> Path:
    return Path(get_settings().demo_data_dir) / MASK_FILENAME


def decode_nsidc_concentration(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a raw NSIDC concentration GeoTIFF into (conc, land, missing).

    ``conc`` is a fraction in [0, 1] with NaN over land / missing pixels.
    """
    arr = np.asarray(raw).astype(float)
    land = (arr == FLAG_LAND) | (arr == FLAG_COAST)
    missing = (arr == FLAG_MISSING) | (arr == FLAG_UNUSED)
    pole = arr == FLAG_POLE_HOLE
    conc = np.where(arr <= 1000, arr / 1000.0, np.nan)
    conc[land] = np.nan
    conc[missing] = np.nan
    conc[pole] = np.nan  # Southern grid has no pole hole, kept for symmetry.
    return conc, land, missing | pole


def save_land_mask(land: np.ndarray, path: Path | None = None) -> Path:
    """Persist a boolean NSIDC-grid land mask (packed, a few kB on disk)."""
    path = Path(path or mask_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        land=np.packbits(np.asarray(land, dtype=bool)),
        shape=np.asarray(land.shape, dtype=np.int32),
        source=np.array("NSIDC G02135 v4.0 land/coast flags", dtype=object),
    )
    log.info("Land mask written to %s", path)
    return path


@lru_cache(maxsize=2)
def load_nsidc_land_mask(path_str: str | None = None) -> np.ndarray:
    """Load the packed NSIDC-grid land mask. Raises if the asset is missing."""
    path = Path(path_str) if path_str else mask_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Land mask asset not found at {path}. Run "
            f"'python scripts/download_data.py --landmask' (needs network) or "
            f"'python scripts/seed_demo_data.py' to regenerate it."
        )
    with np.load(path, allow_pickle=True) as z:
        shape = tuple(int(v) for v in z["shape"])
        flat = np.unpackbits(z["land"])[: shape[0] * shape[1]]
    return flat.reshape(shape).astype(bool)


def build_land_mask_from_nsidc(geotiff_path: Path) -> np.ndarray:
    """Extract the land mask from a downloaded NSIDC concentration GeoTIFF."""
    from PIL import Image

    with Image.open(geotiff_path) as im:
        raw = np.array(im)
    if raw.shape != NSIDC_SOUTH_SHAPE:
        log.warning("Unexpected NSIDC grid shape %s (expected %s)", raw.shape, NSIDC_SOUTH_SHAPE)
    _, land, _ = decode_nsidc_concentration(raw)
    return land


@lru_cache(maxsize=8)
def _land_fraction_cached(key: tuple) -> np.ndarray:
    grid = GridSpec(*key)
    land = load_nsidc_land_mask()
    lat, lon = nsidc_grid_latlon(land.shape)
    frac = regrid_nearest(lat, lon, land.astype(float), grid)
    # Cells with no NSIDC coverage (outside the polar grid) are open ocean.
    return np.nan_to_num(frac, nan=0.0)


def land_fraction(grid: GridSpec) -> np.ndarray:
    """Fraction of each analysis cell covered by land/ice shelf, in [0, 1]."""
    return _land_fraction_cached(
        (grid.lat_min, grid.lat_max, grid.lon_min, grid.lon_max, grid.dlat, grid.dlon)
    )


def navigable_mask(grid: GridSpec, land_threshold: float = 0.5) -> np.ndarray:
    """Boolean grid: True where a vessel could physically be (open water)."""
    return land_fraction(grid) < land_threshold


@lru_cache(maxsize=8)
def _coast_distance_cached(key: tuple, land_threshold: float) -> np.ndarray:
    grid = GridSpec(*key)
    land = ~navigable_mask(grid, land_threshold)
    lat2d, lon2d = grid.meshgrid()
    if not land.any():
        return np.full(grid.shape, 9999.0)
    li, lj = np.where(land)
    land_lat = grid.lats[li]
    land_lon = grid.lons[lj]
    out = np.empty(grid.shape)
    # Chunked brute force: the analysis grid is only a few thousand cells.
    for i in range(grid.shape[0]):
        d = haversine_km(
            lat2d[i, :][:, None], lon2d[i, :][:, None], land_lat[None, :], land_lon[None, :]
        )
        out[i, :] = d.min(axis=1)
    return out


def distance_to_land_km(grid: GridSpec, land_threshold: float = 0.5) -> np.ndarray:
    """Great-circle distance from each cell centre to the nearest land cell."""
    return _coast_distance_cached(
        (grid.lat_min, grid.lat_max, grid.lon_min, grid.lon_max, grid.dlat, grid.dlon),
        land_threshold,
    )


def is_navigable_point(grid: GridSpec, lat: float, lon: float, land_threshold: float = 0.5) -> bool:
    if not grid.contains(lat, lon):
        return False
    i, j = grid.index_of(lat, lon)
    return bool(navigable_mask(grid, land_threshold)[i, j])


def clear_cache() -> None:
    """Drop cached masks (tests change the domain between cases)."""
    load_nsidc_land_mask.cache_clear()
    _land_fraction_cached.cache_clear()
    _coast_distance_cached.cache_clear()
