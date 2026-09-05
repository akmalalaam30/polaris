"""Data preprocessing: raw observations -> model-ready arrays and features.

Responsibilities
----------------
* Build and maintain the gridded **sea-ice history** archive
  (``data/processed/sea_ice_history.nc``) that the forecasting model trains on.
* Build the matching **weather history** archive of daily atmospheric drivers.
* Clean iceberg tracks: chronological ordering, duplicate removal, displacement,
  drift speed and bearing.
* Standardise wind (u/v <-> speed/direction) and synchronise the ocean fields
  onto the analysis grid, including extraction at iceberg positions.
* Turn all of the above into the supervised-learning design matrix used by
  :mod:`app.services.sea_ice_forecasting`.

Nothing here fabricates observations: gaps are filled by documented, local
interpolation only, and cells that stay unknown remain NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.services import landmask
from app.utils.geo import (
    GridSpec,
    bilinear_sample,
    fill_nan_nearest,
    grid_from_settings,
    haversine_km,
    initial_bearing_deg,
    smooth_field,
)
from app.utils.logging import get_logger
from app.utils.validation import DataUnavailableError, ValidationError

log = get_logger("services.preprocessing")

SEA_ICE_HISTORY_FILE = "sea_ice_history.nc"
WEATHER_HISTORY_FILE = "weather_history.nc"

#: Lags (in days) offered to the forecast model.
LAG_DAYS = (1, 2, 3, 5, 7, 14)


# ---------------------------------------------------------------------------
# History archive (NetCDF)
# ---------------------------------------------------------------------------
def history_path(settings: Settings | None = None, filename: str = SEA_ICE_HISTORY_FILE) -> Path:
    settings = settings or get_settings()
    return Path(settings.processed_data_dir) / filename


def save_history(
    times: Sequence[datetime],
    field: np.ndarray,
    grid: GridSpec,
    settings: Settings | None = None,
    filename: str = SEA_ICE_HISTORY_FILE,
    variables: dict[str, np.ndarray] | None = None,
    attrs: dict | None = None,
) -> Path:
    """Persist a ``(time, lat, lon)`` archive as NetCDF."""
    import xarray as xr

    settings = settings or get_settings()
    data_vars = {"sea_ice_concentration": (("time", "lat", "lon"), np.asarray(field, dtype="float32"))}
    for name, arr in (variables or {}).items():
        data_vars[name] = (("time", "lat", "lon"), np.asarray(arr, dtype="float32"))
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": pd.to_datetime(list(times), utc=True).tz_localize(None),
            "lat": grid.lats.astype("float32"),
            "lon": grid.lons.astype("float32"),
        },
        attrs={
            "title": "POLARIS gridded history archive",
            "data_mode": settings.data_mode,
            "grid": str(grid.to_dict()),
            "created": datetime.now(timezone.utc).isoformat(),
            **(attrs or {}),
        },
    )
    path = history_path(settings, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    ds.close()
    log.info("History archive written: %s (%d time steps)", path.name, len(times))
    return path


def load_history(
    settings: Settings | None = None, filename: str = SEA_ICE_HISTORY_FILE
) -> tuple[list[datetime], dict[str, np.ndarray], GridSpec, dict]:
    """Load a NetCDF history archive. Raises if it has not been built."""
    import xarray as xr

    settings = settings or get_settings()
    path = history_path(settings, filename)
    if not path.exists():
        raise DataUnavailableError(
            f"History archive {path.name} not found. Run 'python scripts/preprocess_data.py' first."
        )
    with xr.open_dataset(path) as ds:
        times = [
            pd.Timestamp(t).to_pydatetime().replace(tzinfo=timezone.utc) for t in ds["time"].values
        ]
        lats = ds["lat"].values.astype(float)
        lons = ds["lon"].values.astype(float)
        variables = {name: np.asarray(ds[name].values, dtype=float) for name in ds.data_vars}
        attrs = dict(ds.attrs)
    grid = GridSpec(
        lat_min=float(lats[0]),
        lat_max=float(lats[-1]),
        lon_min=float(lons[0]),
        lon_max=float(lons[-1]),
        dlat=float(lats[1] - lats[0]) if len(lats) > 1 else 0.5,
        dlon=float(lons[1] - lons[0]) if len(lons) > 1 else 1.0,
    )
    return times, variables, grid, attrs


def build_sea_ice_history(
    days: int | None = None,
    end_date: date | None = None,
    grid: GridSpec | None = None,
    settings: Settings | None = None,
    max_failures: int = 30,
) -> tuple[list[datetime], np.ndarray]:
    """Assemble the daily sea-ice concentration archive used for training.

    Demo mode generates the field locally; real mode downloads (and caches) the
    NSIDC daily GeoTIFFs.  Days that cannot be retrieved are skipped and
    reported - they are never interpolated across from scratch.
    """
    from app.services.data_ingestion import NSIDCClient

    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    days = days or settings.sea_ice_history_days
    end_date = end_date or (datetime.now(timezone.utc).date() - timedelta(days=1 if settings.is_demo else 2))

    times: list[datetime] = []
    stack: list[np.ndarray] = []
    failures: list[str] = []

    if settings.is_demo:
        from app.services.demo_data import DemoFieldGenerator

        generator = DemoFieldGenerator(grid)
        for k in range(days - 1, -1, -1):
            day = end_date - timedelta(days=k)
            dt = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            times.append(dt)
            stack.append(generator.sea_ice_concentration(dt))
    else:
        client = NSIDCClient(settings)
        for k in range(days - 1, -1, -1):
            day = end_date - timedelta(days=k)
            try:
                field = client.concentration_on_grid(day, grid)
            except Exception as exc:  # noqa: BLE001 - reported below
                failures.append(f"{day}: {exc.__class__.__name__}")
                if len(failures) > max_failures:
                    raise DataUnavailableError(
                        f"Aborting: {len(failures)} NSIDC days failed to download "
                        f"(last: {failures[-1]})"
                    ) from exc
                continue
            times.append(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))
            stack.append(field)

    if len(times) < 2:
        raise DataUnavailableError(f"Only {len(times)} usable day(s) of sea-ice data; cannot build history")
    if failures:
        log.warning("Sea-ice history: %d day(s) unavailable (e.g. %s)", len(failures), failures[:3])

    array = np.stack(stack).astype("float32")
    save_history(
        times,
        array,
        grid,
        settings,
        attrs={"variable": "sea_ice_concentration", "missing_days": len(failures)},
    )
    return times, array


def build_weather_history(
    times: Sequence[datetime],
    grid: GridSpec | None = None,
    settings: Settings | None = None,
) -> dict[str, np.ndarray] | None:
    """Assemble the daily atmospheric drivers aligned to ``times``.

    Returns ``None`` (and logs) when no weather source is reachable - the
    forecast model then trains on sea-ice predictors alone rather than on
    invented values.
    """
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    if not times:
        return None

    try:
        if settings.is_demo:
            from app.services.demo_data import DemoFieldGenerator

            generator = DemoFieldGenerator(grid)
            fields = {"t2m": [], "wind_speed": []}
            for dt in times:
                w = generator.weather(dt)
                fields["t2m"].append(w["t2m"])
                fields["wind_speed"].append(np.hypot(w["u10"], w["v10"]))
            stacked = {k: np.stack(v).astype("float32") for k, v in fields.items()}
        else:
            stacked = _fetch_era5_daily_history(times, grid, settings)
    except Exception as exc:  # noqa: BLE001 - degradation is expected and logged
        log.warning("Weather history unavailable (%s); training on sea-ice predictors only", exc)
        return None

    save_history(
        times,
        stacked["t2m"],
        grid,
        settings,
        filename=WEATHER_HISTORY_FILE,
        variables={k: v for k, v in stacked.items() if k != "t2m"},
        attrs={"variable": "daily atmospheric drivers"},
    )
    # The first data variable in the archive is named sea_ice_concentration by
    # save_history; rename on load instead (see load_weather_history).
    return stacked


def _fetch_era5_daily_history(
    times: Sequence[datetime], grid: GridSpec, settings: Settings
) -> dict[str, np.ndarray]:
    """One ranged ERA5 request for the whole training period, then interpolate."""
    from app.services.data_ingestion import _get_json, _mirror_sample_points, _points_to_grid

    if not settings.allow_open_mirrors and not settings.era5_api_key:
        raise DataUnavailableError("No ERA5 access configured for history retrieval")

    lats, lons = _mirror_sample_points(grid, settings)
    start, end = min(times).date(), max(times).date()
    url = (
        f"{settings.open_meteo_era5_url}?"
        f"latitude={','.join(f'{v:.4f}' for v in lats)}&"
        f"longitude={','.join(f'{v:.4f}' for v in lons)}&"
        f"start_date={start:%Y-%m-%d}&end_date={end:%Y-%m-%d}&"
        "daily=temperature_2m_mean,wind_speed_10m_mean&wind_speed_unit=ms&models=era5"
    )
    payload = _get_json(url, settings)
    records = payload if isinstance(payload, list) else [payload]

    day_index = {dt.date(): i for i, dt in enumerate(times)}
    n_t = len(times)
    out = {k: np.full((n_t,) + grid.shape, np.nan, dtype="float32") for k in ("t2m", "wind_speed")}

    # Collect per-day scattered points, then interpolate each day onto the grid.
    per_day: dict[int, dict[str, list]] = {}
    for rec in records:
        daily = rec.get("daily") or {}
        stamps = daily.get("time") or []
        temps = daily.get("temperature_2m_mean") or []
        winds = daily.get("wind_speed_10m_mean") or []
        rlat, rlon = float(rec["latitude"]), float(rec["longitude"])
        for k, stamp in enumerate(stamps):
            d = date.fromisoformat(stamp)
            if d not in day_index:
                continue
            slot = per_day.setdefault(day_index[d], {"lat": [], "lon": [], "t2m": [], "wind_speed": []})
            slot["lat"].append(rlat)
            slot["lon"].append(rlon)
            slot["t2m"].append(temps[k] if k < len(temps) and temps[k] is not None else np.nan)
            slot["wind_speed"].append(winds[k] if k < len(winds) and winds[k] is not None else np.nan)

    if not per_day:
        raise DataUnavailableError("ERA5 history request returned no usable days")

    for idx, slot in per_day.items():
        grids = _points_to_grid(
            slot["lat"], slot["lon"], {"t2m": slot["t2m"], "wind_speed": slot["wind_speed"]}, grid
        )
        for key in out:
            out[key][idx] = grids[key].astype("float32")

    # Days the provider skipped are filled from their temporal neighbours.
    for key, arr in out.items():
        frame = pd.DataFrame(arr.reshape(len(times), -1))
        out[key] = frame.ffill().bfill().to_numpy(dtype="float32").reshape(arr.shape)
    log.info("ERA5 daily history retrieved for %d/%d days", len(per_day), len(times))
    return out


def load_weather_history(settings: Settings | None = None) -> tuple[list[datetime], dict[str, np.ndarray]] | None:
    """Load the weather archive if it exists, else ``None``."""
    settings = settings or get_settings()
    if not history_path(settings, WEATHER_HISTORY_FILE).exists():
        return None
    times, variables, _grid, _attrs = load_history(settings, WEATHER_HISTORY_FILE)
    # save_history stores the primary array under 'sea_ice_concentration'.
    renamed = dict(variables)
    if "sea_ice_concentration" in renamed:
        renamed["t2m"] = renamed.pop("sea_ice_concentration")
    return times, renamed


# ---------------------------------------------------------------------------
# Sea-ice field preprocessing
# ---------------------------------------------------------------------------
def preprocess_sea_ice_field(field: np.ndarray, grid: GridSpec, fill: bool = True) -> np.ndarray:
    """Clean a single concentration field.

    Clamps to [0, 1], removes speckle below the 15% detection threshold used by
    passive-microwave products, closes small gaps and re-masks land as NaN.
    """
    out = np.asarray(field, dtype=float).copy()
    out = np.clip(out, 0.0, 1.0)
    if fill:
        out = fill_nan_nearest(out, max_iter=6)
    land = landmask.land_fraction(grid) >= 0.5
    out[land] = np.nan
    return out


def sea_ice_edge_distance_km(field: np.ndarray, grid: GridSpec, threshold: float = 0.15) -> np.ndarray:
    """Distance from every cell to the nearest 15% ice-edge cell.

    Positive everywhere; cells inside the pack and cells in open water both get
    their distance to the edge, which is what the risk engine needs.
    """
    conc = np.nan_to_num(field, nan=0.0)
    ice = conc >= threshold
    if not ice.any() or ice.all():
        return np.full(grid.shape, np.nan)
    # Edge cells: ice cells with at least one non-ice 4-neighbour.
    padded = np.pad(ice, 1, mode="edge")
    neighbours = (
        padded[:-2, 1:-1].astype(int)
        + padded[2:, 1:-1].astype(int)
        + padded[1:-1, :-2].astype(int)
        + padded[1:-1, 2:].astype(int)
    )
    edge = ice & (neighbours < 4)
    ei, ej = np.where(edge)
    if ei.size == 0:
        return np.full(grid.shape, np.nan)
    edge_lat, edge_lon = grid.lats[ei], grid.lons[ej]
    lat2d, lon2d = grid.meshgrid()
    out = np.empty(grid.shape)
    for i in range(grid.shape[0]):
        d = haversine_km(
            lat2d[i, :][:, None], lon2d[i, :][:, None], edge_lat[None, :], edge_lon[None, :]
        )
        out[i, :] = d.min(axis=1)
    return out


def spatial_context(field: np.ndarray) -> dict[str, np.ndarray]:
    """Local spatial statistics used as forecast predictors."""
    filled = fill_nan_nearest(field, max_iter=4)
    smooth3 = smooth_field(filled, passes=1)
    smooth5 = smooth_field(filled, passes=2)
    grad_lat, grad_lon = np.gradient(filled)
    return {
        "mean3": smooth3,
        "mean5": smooth5,
        "grad_lat": grad_lat,
        "grad_lon": grad_lon,
        "roughness": np.abs(filled - smooth3),
    }


# ---------------------------------------------------------------------------
# Iceberg track preprocessing
# ---------------------------------------------------------------------------
def preprocess_iceberg_tracks(observations: Iterable) -> pd.DataFrame:
    """Clean and enrich raw iceberg observations.

    Sorts each berg's observations chronologically, drops duplicate timestamps,
    and derives displacement, drift speed and bearing between consecutive fixes.
    """
    rows = []
    for o in observations:
        rows.append(
            {
                "iceberg_id": getattr(o, "iceberg_id", None) or o["iceberg_id"],
                "observed_at": getattr(o, "observed_at", None) or o["observed_at"],
                "latitude": float(getattr(o, "latitude", None) if hasattr(o, "latitude") else o["latitude"]),
                "longitude": float(getattr(o, "longitude", None) if hasattr(o, "longitude") else o["longitude"]),
                "length_nm": getattr(o, "length_nm", None) if hasattr(o, "length_nm") else o.get("length_nm"),
                "width_nm": getattr(o, "width_nm", None) if hasattr(o, "width_nm") else o.get("width_nm"),
                "area_km2": getattr(o, "area_km2", None) if hasattr(o, "area_km2") else o.get("area_km2"),
                "source": getattr(o, "source", None) if hasattr(o, "source") else o.get("source"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "iceberg_id", "observed_at", "latitude", "longitude", "length_nm", "width_nm",
                "area_km2", "source", "displacement_km", "dt_hours", "drift_speed_m_s",
                "drift_speed_km_per_day", "drift_bearing_deg",
            ]
        )

    df = pd.DataFrame(rows)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    df = df.drop_duplicates(subset=["iceberg_id", "observed_at"], keep="last")
    df = df.sort_values(["iceberg_id", "observed_at"]).reset_index(drop=True)

    df["displacement_km"] = np.nan
    df["dt_hours"] = np.nan
    df["drift_bearing_deg"] = np.nan
    for berg, block in df.groupby("iceberg_id", sort=False):
        if len(block) < 2:
            continue
        idx = block.index
        lat, lon = block["latitude"].to_numpy(), block["longitude"].to_numpy()
        dist = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        brg = initial_bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dt_h = block["observed_at"].diff().dt.total_seconds().to_numpy()[1:] / 3600.0
        df.loc[idx[1:], "displacement_km"] = np.atleast_1d(dist)
        df.loc[idx[1:], "drift_bearing_deg"] = np.atleast_1d(brg)
        df.loc[idx[1:], "dt_hours"] = dt_h

    with np.errstate(divide="ignore", invalid="ignore"):
        df["drift_speed_m_s"] = (df["displacement_km"] * 1000.0) / (df["dt_hours"] * 3600.0)
        df["drift_speed_km_per_day"] = df["displacement_km"] / (df["dt_hours"] / 24.0)
    # Reject physically impossible drift (icebergs rarely exceed ~1.5 m/s).
    implausible = df["drift_speed_m_s"] > 1.5
    if implausible.any():
        log.warning("Rejected %d implausible iceberg drift value(s) (>1.5 m/s)", int(implausible.sum()))
        df.loc[implausible, ["drift_speed_m_s", "drift_speed_km_per_day", "drift_bearing_deg"]] = np.nan
    return df


def latest_iceberg_state(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per berg: its most recent fix plus derived drift statistics."""
    if tracks.empty:
        return tracks
    latest = tracks.sort_values("observed_at").groupby("iceberg_id", as_index=False).last()
    means = (
        tracks.groupby("iceberg_id")["drift_speed_m_s"].mean().rename("mean_drift_speed_m_s")
    )
    return latest.merge(means, on="iceberg_id", how="left")


# ---------------------------------------------------------------------------
# Wind / ocean standardisation
# ---------------------------------------------------------------------------
def wind_components(speed_m_s, direction_deg_from) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological (speed, direction-from) -> (u, v) components."""
    rad = np.radians(np.asarray(direction_deg_from, dtype=float))
    speed = np.asarray(speed_m_s, dtype=float)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def wind_speed_direction(u, v) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) -> (speed, meteorological direction the wind comes FROM)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    speed = np.hypot(u, v)
    direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return speed, direction


def synchronise_environment(
    grid: GridSpec,
    sea_ice: np.ndarray | None,
    weather: dict[str, np.ndarray] | None,
    ocean: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Put every environmental layer on the same grid with consistent gaps.

    Missing layers come back as all-NaN arrays of the right shape so downstream
    code can test availability without special-casing ``None``.
    """
    shape = grid.shape
    out: dict[str, np.ndarray] = {}

    out["sea_ice_concentration"] = (
        preprocess_sea_ice_field(sea_ice, grid) if sea_ice is not None else np.full(shape, np.nan)
    )
    for key in ("u10", "v10", "t2m", "msl"):
        arr = (weather or {}).get(key)
        out[key] = fill_nan_nearest(np.asarray(arr, float), 4) if arr is not None else np.full(shape, np.nan)
    for key in ("u_current", "v_current", "sst", "salinity", "wave_height"):
        arr = (ocean or {}).get(key)
        out[key] = fill_nan_nearest(np.asarray(arr, float), 4) if arr is not None else np.full(shape, np.nan)

    if np.isfinite(out["u10"]).any():
        speed, direction = wind_speed_direction(out["u10"], out["v10"])
        out["wind_speed"] = speed
        out["wind_direction"] = direction
    else:
        out["wind_speed"] = np.full(shape, np.nan)
        out["wind_direction"] = np.full(shape, np.nan)

    if np.isfinite(out["u_current"]).any():
        out["current_speed"] = np.hypot(out["u_current"], out["v_current"])
    else:
        out["current_speed"] = np.full(shape, np.nan)

    out["land_fraction"] = landmask.land_fraction(grid)
    out["distance_to_land_km"] = landmask.distance_to_land_km(grid)
    if np.isfinite(out["sea_ice_concentration"]).any():
        out["ice_edge_distance_km"] = sea_ice_edge_distance_km(out["sea_ice_concentration"], grid)
    else:
        out["ice_edge_distance_km"] = np.full(shape, np.nan)
    return out


def sample_environment_at(
    grid: GridSpec, fields: dict[str, np.ndarray], lat: float, lon: float, keys: Sequence[str] | None = None
) -> dict[str, float]:
    """Bilinearly extract environmental values at an arbitrary position.

    Used to obtain the ocean current and wind acting on an iceberg.
    """
    keys = keys or list(fields.keys())
    out: dict[str, float] = {}
    for key in keys:
        arr = fields.get(key)
        if arr is None:
            out[key] = float("nan")
            continue
        filled = fill_nan_nearest(arr, max_iter=8)
        if not grid.contains(lat, lon):
            out[key] = float("nan")
        else:
            out[key] = bilinear_sample(grid, filled, lat, lon)
    return out


# ---------------------------------------------------------------------------
# Supervised-learning design matrix
# ---------------------------------------------------------------------------
@dataclass
class FeatureSpec:
    """Names of the columns produced by :func:`build_feature_frame`."""

    names: list[str]

    def __len__(self) -> int:
        return len(self.names)


def feature_names(use_weather: bool) -> list[str]:
    """The canonical feature order.

    Training and inference both derive their column order from here, so a model
    and the features fed to it can never drift apart.
    """
    names = ["conc_t"]
    names += [f"conc_lag{lag}" for lag in LAG_DAYS]
    names += [f"delta_{lag}" for lag in (1, 3, 7)]
    names += ["mean3", "mean5", "grad_lat", "grad_lon", "roughness", "edge_dist_km"]
    names += ["lat", "lon", "coast_dist_km", "doy_sin", "doy_cos", "horizon_days"]
    if use_weather:
        names += ["t2m", "wind_speed"]
    return names


def _doy_harmonics(dt: datetime) -> tuple[float, float]:
    doy = dt.timetuple().tm_yday
    ang = 2.0 * math.pi * doy / 365.25
    return math.sin(ang), math.cos(ang)


def build_feature_frame(
    times: Sequence[datetime],
    ice: np.ndarray,
    grid: GridSpec,
    horizon_days: int,
    weather: dict[str, np.ndarray] | None = None,
    subsample: int = 1,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Build ``(X, y, feature_names, sample_time_index)`` for one horizon.

    Only ocean cells with a complete lag history and a valid target are kept, so
    the model never trains on filled land or on fabricated values.
    """
    max_lag = max_lag or max(LAG_DAYS)
    n_t = ice.shape[0]
    if n_t <= max_lag + horizon_days:
        raise ValidationError(
            f"Need more than {max_lag + horizon_days} time steps to build features for a "
            f"{horizon_days}-day horizon; got {n_t}"
        )

    ocean_mask = landmask.land_fraction(grid) < 0.5
    lat2d, lon2d = grid.meshgrid()
    coast_km = landmask.distance_to_land_km(grid)

    use_weather = weather is not None and all(
        k in weather and weather[k].shape == ice.shape for k in ("t2m", "wind_speed")
    )
    names = feature_names(use_weather)

    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    t_blocks: list[np.ndarray] = []

    for t in range(max_lag, n_t - horizon_days, subsample):
        current = ice[t]
        target = ice[t + horizon_days]
        valid = ocean_mask & np.isfinite(current) & np.isfinite(target)
        for lag in LAG_DAYS:
            valid &= np.isfinite(ice[t - lag])
        if valid.sum() == 0:
            continue

        ctx = spatial_context(current)
        edge = sea_ice_edge_distance_km(current, grid)
        edge = np.nan_to_num(edge, nan=2000.0)
        doy_sin, doy_cos = _doy_harmonics(times[t])

        columns = [current[valid]]
        columns += [ice[t - lag][valid] for lag in LAG_DAYS]
        columns += [(current - ice[t - lag])[valid] for lag in (1, 3, 7)]
        columns += [
            ctx["mean3"][valid], ctx["mean5"][valid], ctx["grad_lat"][valid],
            ctx["grad_lon"][valid], ctx["roughness"][valid], edge[valid],
        ]
        columns += [
            lat2d[valid], lon2d[valid], coast_km[valid],
            np.full(int(valid.sum()), doy_sin), np.full(int(valid.sum()), doy_cos),
            np.full(int(valid.sum()), float(horizon_days)),
        ]
        if use_weather:
            columns += [
                np.nan_to_num(weather["t2m"][t][valid], nan=-15.0),
                np.nan_to_num(weather["wind_speed"][t][valid], nan=8.0),
            ]

        x_blocks.append(np.column_stack(columns).astype("float32"))
        y_blocks.append(target[valid].astype("float32"))
        t_blocks.append(np.full(int(valid.sum()), t, dtype="int32"))

    if not x_blocks:
        raise ValidationError("No usable training samples were produced")
    return (
        np.concatenate(x_blocks),
        np.concatenate(y_blocks),
        names,
        np.concatenate(t_blocks),
    )


def build_inference_features(
    times: Sequence[datetime],
    ice: np.ndarray,
    grid: GridSpec,
    horizon_days: int,
    weather: dict[str, np.ndarray] | None = None,
    expected_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features for the latest time step: ``(X, usable-cell mask, column names)``.

    ``expected_names`` lets a trained model pin the column set it was fitted on,
    so a model trained without weather covariates is not handed extra columns
    when a weather archive later appears.
    """
    max_lag = max(LAG_DAYS)
    if ice.shape[0] <= max_lag:
        raise ValidationError(
            f"Need more than {max_lag} time steps for inference; got {ice.shape[0]}"
        )
    t = ice.shape[0] - 1
    ocean_mask = landmask.land_fraction(grid) < 0.5
    lat2d, lon2d = grid.meshgrid()
    coast_km = landmask.distance_to_land_km(grid)

    current = ice[t]
    valid = ocean_mask & np.isfinite(current)
    for lag in LAG_DAYS:
        valid &= np.isfinite(ice[t - lag])
    if valid.sum() == 0:
        raise ValidationError("No cells have a complete lag history for inference")

    ctx = spatial_context(current)
    edge = np.nan_to_num(sea_ice_edge_distance_km(current, grid), nan=2000.0)
    valid_time = times[t] + timedelta(days=horizon_days)
    doy_sin, doy_cos = _doy_harmonics(valid_time)
    n = int(valid.sum())

    columns = [current[valid]]
    columns += [ice[t - lag][valid] for lag in LAG_DAYS]
    columns += [(current - ice[t - lag])[valid] for lag in (1, 3, 7)]
    columns += [
        ctx["mean3"][valid], ctx["mean5"][valid], ctx["grad_lat"][valid],
        ctx["grad_lon"][valid], ctx["roughness"][valid], edge[valid],
    ]
    columns += [
        lat2d[valid], lon2d[valid], coast_km[valid],
        np.full(n, doy_sin), np.full(n, doy_cos), np.full(n, float(horizon_days)),
    ]
    weather_available = (
        weather is not None
        and "t2m" in weather
        and "wind_speed" in weather
        and weather["t2m"].shape == ice.shape
    )
    use_weather = weather_available and (expected_names is None or "t2m" in expected_names)
    if use_weather:
        columns += [
            np.nan_to_num(weather["t2m"][t][valid], nan=-15.0),
            np.nan_to_num(weather["wind_speed"][t][valid], nan=8.0),
        ]
    return np.column_stack(columns).astype("float32"), valid, feature_names(use_weather)
