"""Geospatial helpers: geodesy, polar-stereographic projection, regular grids.

Everything here is pure NumPy/pyproj so it works without GDAL/rasterio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

EARTH_RADIUS_KM = 6371.0088
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0
OMEGA = 7.2921e-5  # Earth angular velocity, rad/s

# NSIDC Sea Ice Polar Stereographic South (EPSG:3412) grid definition for the
# 25 km southern hemisphere grid used by G02135.
NSIDC_SOUTH_EPSG = 3412
NSIDC_SOUTH_CELL_SIZE_M = 25000.0
NSIDC_SOUTH_ORIGIN_X = -3950000.0
NSIDC_SOUTH_ORIGIN_Y = 4350000.0
NSIDC_SOUTH_SHAPE = (332, 316)  # rows (y), cols (x)

_TRANSFORMER_CACHE: dict[tuple[int, int], object] = {}


def _transformer(src: int, dst: int):
    """Cached pyproj transformer (lazily imported so pyproj stays optional)."""
    key = (src, dst)
    if key not in _TRANSFORMER_CACHE:
        from pyproj import Transformer

        _TRANSFORMER_CACHE[key] = Transformer.from_crs(
            f"EPSG:{src}", f"EPSG:{dst}", always_xy=True
        )
    return _TRANSFORMER_CACHE[key]


# ---------------------------------------------------------------------------
# Geodesy
# ---------------------------------------------------------------------------
def normalize_longitude(lon: float | np.ndarray) -> float | np.ndarray:
    """Wrap longitude into [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0 if isinstance(
        lon, (np.ndarray, list, tuple)
    ) else ((float(lon) + 180.0) % 360.0) - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres. Scalar or array-safe."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    d = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    return float(d) if np.isscalar(d) or d.ndim == 0 else d


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, degrees from North."""
    phi1, phi2 = np.radians(np.asarray(lat1, float)), np.radians(np.asarray(lat2, float))
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    y = np.sin(dl) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dl)
    brg = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return float(brg) if np.isscalar(brg) or np.ndim(brg) == 0 else brg


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """Point reached travelling ``distance_km`` along ``bearing_deg``."""
    d = distance_km / EARTH_RADIUS_KM
    brg = math.radians(bearing_deg)
    phi1, lam1 = math.radians(lat), math.radians(lon)
    phi2 = math.asin(math.sin(phi1) * math.cos(d) + math.cos(phi1) * math.sin(d) * math.cos(brg))
    lam2 = lam1 + math.atan2(
        math.sin(brg) * math.sin(d) * math.cos(phi1),
        math.cos(d) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), float(normalize_longitude(math.degrees(lam2)))


def offset_meters(lat: float, lon: float, dx_m: float, dy_m: float) -> tuple[float, float]:
    """Displace a point by local east (dx) / north (dy) metres.

    Uses a local tangent-plane approximation, which is appropriate for the
    sub-degree displacements produced by one iceberg-drift timestep.
    """
    dlat = (dy_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    coslat = max(math.cos(math.radians(lat)), 1e-6)
    dlon = (dx_m / (EARTH_RADIUS_M * coslat)) * (180.0 / math.pi)
    new_lat = max(-89.9, min(89.9, lat + dlat))
    return new_lat, float(normalize_longitude(lon + dlon))


def coriolis_parameter(lat_deg: float) -> float:
    """Coriolis parameter f = 2*Omega*sin(phi) (negative in the south)."""
    return 2.0 * OMEGA * math.sin(math.radians(lat_deg))


def path_length_km(points: Sequence[tuple[float, float]]) -> float:
    """Cumulative great-circle length of a (lat, lon) polyline."""
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=float)
    return float(np.sum(haversine_km(arr[:-1, 0], arr[:-1, 1], arr[1:, 0], arr[1:, 1])))


def cross_track_distance_km(lat, lon, lat1, lon1, lat2, lon2) -> float:
    """Perpendicular distance from a point to the great-circle through 1->2."""
    d13 = haversine_km(lat1, lon1, lat, lon) / EARTH_RADIUS_KM
    th13 = math.radians(initial_bearing_deg(lat1, lon1, lat, lon))
    th12 = math.radians(initial_bearing_deg(lat1, lon1, lat2, lon2))
    return abs(math.asin(math.sin(d13) * math.sin(th13 - th12)) * EARTH_RADIUS_KM)


def point_segment_distance_km(lat, lon, lat1, lon1, lat2, lon2) -> float:
    """Distance from a point to a great-circle *segment* (clamped at the ends)."""
    seg = haversine_km(lat1, lon1, lat2, lon2)
    if seg < 1e-6:
        return haversine_km(lat, lon, lat1, lon1)
    d13 = haversine_km(lat1, lon1, lat, lon)
    th13 = math.radians(initial_bearing_deg(lat1, lon1, lat, lon))
    th12 = math.radians(initial_bearing_deg(lat1, lon1, lat2, lon2))
    along = math.acos(max(-1.0, min(1.0, math.cos(d13 / EARTH_RADIUS_KM) /
                                    max(1e-12, math.cos(cross_track_distance_km(lat, lon, lat1, lon1, lat2, lon2) / EARTH_RADIUS_KM))))) * EARTH_RADIUS_KM
    if math.cos(th13 - th12) < 0:
        return d13
    if along > seg:
        return haversine_km(lat, lon, lat2, lon2)
    return cross_track_distance_km(lat, lon, lat1, lon1, lat2, lon2)


def densify_great_circle(
    lat1: float, lon1: float, lat2: float, lon2: float, step_km: float = 25.0
) -> list[tuple[float, float]]:
    """Interpolate points along a great circle at roughly ``step_km`` spacing."""
    total = haversine_km(lat1, lon1, lat2, lon2)
    n = max(1, int(math.ceil(total / max(step_km, 1e-3))))
    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)
    d = total / EARTH_RADIUS_KM
    if d < 1e-9:
        return [(lat1, lon1)]
    out = []
    for i in range(n + 1):
        f = i / n
        a = math.sin((1 - f) * d) / math.sin(d)
        b = math.sin(f * d) / math.sin(d)
        x = a * math.cos(phi1) * math.cos(lam1) + b * math.cos(phi2) * math.cos(lam2)
        y = a * math.cos(phi1) * math.sin(lam1) + b * math.cos(phi2) * math.sin(lam2)
        z = a * math.sin(phi1) + b * math.sin(phi2)
        out.append((math.degrees(math.atan2(z, math.hypot(x, y))),
                    float(normalize_longitude(math.degrees(math.atan2(y, x))))))
    return out


# ---------------------------------------------------------------------------
# Polar stereographic (NSIDC EPSG:3412) helpers
# ---------------------------------------------------------------------------
def nsidc_grid_latlon(shape: tuple[int, int] = NSIDC_SOUTH_SHAPE) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) arrays for the NSIDC 25 km southern polar grid cells."""
    rows, cols = shape
    x = NSIDC_SOUTH_ORIGIN_X + (np.arange(cols) + 0.5) * NSIDC_SOUTH_CELL_SIZE_M
    y = NSIDC_SOUTH_ORIGIN_Y - (np.arange(rows) + 0.5) * NSIDC_SOUTH_CELL_SIZE_M
    xx, yy = np.meshgrid(x, y)
    tr = _transformer(NSIDC_SOUTH_EPSG, 4326)
    lon, lat = tr.transform(xx, yy)
    return lat, lon


def latlon_to_nsidc_index(lat, lon, shape: tuple[int, int] = NSIDC_SOUTH_SHAPE):
    """Map lat/lon to (row, col) indices on the NSIDC 25 km grid."""
    tr = _transformer(4326, NSIDC_SOUTH_EPSG)
    x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    col = np.floor((x - NSIDC_SOUTH_ORIGIN_X) / NSIDC_SOUTH_CELL_SIZE_M).astype(int)
    row = np.floor((NSIDC_SOUTH_ORIGIN_Y - y) / NSIDC_SOUTH_CELL_SIZE_M).astype(int)
    rows, cols = shape
    valid = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    return row, col, valid


# ---------------------------------------------------------------------------
# Regular lat/lon analysis grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridSpec:
    """A regular lat/lon grid used for forecasting, risk and routing."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    dlat: float
    dlon: float

    @property
    def lats(self) -> np.ndarray:
        n = int(round((self.lat_max - self.lat_min) / self.dlat)) + 1
        return self.lat_min + np.arange(n) * self.dlat

    @property
    def lons(self) -> np.ndarray:
        n = int(round((self.lon_max - self.lon_min) / self.dlon)) + 1
        return self.lon_min + np.arange(n) * self.dlon

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.lats), len(self.lons))

    @property
    def n_cells(self) -> int:
        r, c = self.shape
        return r * c

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lat2d, lon2d) with shape ``self.shape``."""
        return np.meshgrid(self.lats, self.lons, indexing="ij")

    def index_of(self, lat: float, lon: float) -> tuple[int, int]:
        """Nearest grid index for a coordinate, clamped to the domain."""
        i = int(round((lat - self.lat_min) / self.dlat))
        j = int(round((lon - self.lon_min) / self.dlon))
        r, c = self.shape
        return max(0, min(r - 1, i)), max(0, min(c - 1, j))

    def contains(self, lat: float, lon: float) -> bool:
        return (
            self.lat_min - self.dlat / 2 <= lat <= self.lat_max + self.dlat / 2
            and self.lon_min - self.dlon / 2 <= lon <= self.lon_max + self.dlon / 2
        )

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        return float(self.lats[i]), float(self.lons[j])

    def to_dict(self) -> dict:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "dlat": self.dlat,
            "dlon": self.dlon,
            "shape": list(self.shape),
        }


def grid_from_settings(settings) -> GridSpec:
    """Build the analysis :class:`GridSpec` from application settings."""
    return GridSpec(
        lat_min=settings.domain_lat_min,
        lat_max=settings.domain_lat_max,
        lon_min=settings.domain_lon_min,
        lon_max=settings.domain_lon_max,
        dlat=settings.grid_resolution_lat,
        dlon=settings.grid_resolution_lon,
    )


def bilinear_sample(grid: GridSpec, field: np.ndarray, lat: float, lon: float) -> float:
    """Bilinearly sample a 2-D field defined on ``grid`` (clamped at edges)."""
    lats, lons = grid.lats, grid.lons
    fi = (lat - grid.lat_min) / grid.dlat
    fj = (lon - grid.lon_min) / grid.dlon
    i0 = int(np.clip(np.floor(fi), 0, len(lats) - 1))
    j0 = int(np.clip(np.floor(fj), 0, len(lons) - 1))
    i1 = min(i0 + 1, len(lats) - 1)
    j1 = min(j0 + 1, len(lons) - 1)
    ti = float(np.clip(fi - i0, 0.0, 1.0))
    tj = float(np.clip(fj - j0, 0.0, 1.0))
    v = (
        field[i0, j0] * (1 - ti) * (1 - tj)
        + field[i1, j0] * ti * (1 - tj)
        + field[i0, j1] * (1 - ti) * tj
        + field[i1, j1] * ti * tj
    )
    return float(v)


def regrid_nearest(
    src_lat: np.ndarray, src_lon: np.ndarray, src_val: np.ndarray, grid: GridSpec
) -> np.ndarray:
    """Bin an irregular/projected source field onto a regular lat/lon grid.

    Cells with no contributing source point are returned as NaN.  Averaging
    (rather than nearest-neighbour picking) is used because the NSIDC 25 km
    grid is finer than the analysis grid over most of the domain.
    """
    lat = np.asarray(src_lat, float).ravel()
    lon = np.asarray(src_lon, float).ravel()
    val = np.asarray(src_val, float).ravel()
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(val)
    lat, lon, val = lat[ok], lon[ok], val[ok]

    lon = np.where(lon < 0, lon + 360.0, lon) if grid.lon_min >= 0 and grid.lon_max > 180 else lon

    i = np.round((lat - grid.lat_min) / grid.dlat).astype(int)
    j = np.round((lon - grid.lon_min) / grid.dlon).astype(int)
    rows, cols = grid.shape
    m = (i >= 0) & (i < rows) & (j >= 0) & (j < cols)
    i, j, val = i[m], j[m], val[m]

    acc = np.zeros((rows, cols), dtype=float)
    cnt = np.zeros((rows, cols), dtype=float)
    np.add.at(acc, (i, j), val)
    np.add.at(cnt, (i, j), 1.0)
    out = np.full((rows, cols), np.nan)
    nz = cnt > 0
    out[nz] = acc[nz] / cnt[nz]
    return out


def fill_nan_nearest(field: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Fill NaNs by iterative 4-neighbour averaging (simple diffusion fill)."""
    out = np.array(field, dtype=float, copy=True)
    for _ in range(max_iter):
        nan_mask = np.isnan(out)
        if not nan_mask.any():
            break
        padded = np.pad(out, 1, mode="edge")
        neigh = np.stack(
            [padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]]
        )
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(neigh)
            counts = valid.sum(axis=0)
            totals = np.where(valid, neigh, 0.0).sum(axis=0)
            mean = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
        out[nan_mask] = mean[nan_mask]
        if np.isnan(out).sum() == nan_mask.sum():
            break
    return out


def smooth_field(field: np.ndarray, passes: int = 1) -> np.ndarray:
    """Cheap 3x3 box smoother that preserves array shape and ignores NaNs."""
    out = np.array(field, dtype=float, copy=True)
    for _ in range(max(0, passes)):
        p = np.pad(out, 1, mode="edge")
        stack = np.stack(
            [
                p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:],
                p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:],
                p[2:, :-2], p[2:, 1:-1], p[2:, 2:],
            ]
        )
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(stack)
            counts = valid.sum(axis=0)
            totals = np.where(valid, stack, 0.0).sum(axis=0)
            out = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return out


def bbox_of(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    """(lat_min, lat_max, lon_min, lon_max) of a set of (lat, lon) points."""
    arr = np.asarray(list(points), dtype=float)
    return (
        float(arr[:, 0].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].min()),
        float(arr[:, 1].max()),
    )
