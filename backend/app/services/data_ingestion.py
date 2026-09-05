"""Data ingestion layer.

One module per concern would be tidier, but the required project layout puts
all provider clients here.  The structure is:

* ``_http`` / ``download_file``      - retrying HTTP with clear errors
* ``NSIDCClient``                    - sea-ice concentration GeoTIFFs + extent index
* ``USNICClient``                    - Antarctic iceberg bulletin (CSV)
* ``ERA5Client``                     - atmospheric reanalysis (CDS API, or the
                                       key-free ERA5 archive mirror)
* ``CopernicusMarineClient``         - ocean analysis (CMEMS toolbox, or the
                                       key-free marine mirror)
* ``ingest_*``                       - validate -> normalise -> upsert -> log

Provenance rules
----------------
* Every row records the *actual* provider it came from.
* Demo/synthetic rows always use ``source="POLARIS-DEMO"`` and
  ``data_mode="demo"`` and are never emitted while ``DATA_MODE=real``.
* When a provider is unavailable in real mode the ingestion is recorded with
  status ``skipped``/``failed`` - it is never silently replaced by demo data.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.database import repositories as repo
from app.services import landmask
from app.services.demo_data import (
    DEMO_ICEBERG_SEEDS,
    DEMO_SOURCE,
    DemoFieldGenerator,
    demo_extent_index,
    generate_demo_iceberg_observations,
)
from app.services.landmask import decode_nsidc_concentration
from app.utils.geo import (
    GridSpec,
    fill_nan_nearest,
    grid_from_settings,
    nsidc_grid_latlon,
    regrid_nearest,
)
from app.utils.logging import get_logger
from app.utils.validation import (
    DataUnavailableError,
    ValidationReport,
    validate_point_frame,
)

log = get_logger("services.ingestion")

SOURCE_NSIDC = "NSIDC/G02135"
SOURCE_USNIC = "USNIC"
SOURCE_ERA5_CDS = "ERA5/CDS"
SOURCE_ERA5_MIRROR = "ERA5/open-archive"
SOURCE_LIVE_IFS = "ECMWF-IFS/live-forecast"
SOURCE_LIVE_MARINE = "ECMWF-WAM/live-marine"
SOURCE_CMEMS = "CMEMS"
SOURCE_MARINE_MIRROR = "CMEMS/open-marine"

MONTH_DIRS = {
    1: "01_Jan", 2: "02_Feb", 3: "03_Mar", 4: "04_Apr", 5: "05_May", 6: "06_Jun",
    7: "07_Jul", 8: "08_Aug", 9: "09_Sep", 10: "10_Oct", 11: "11_Nov", 12: "12_Dec",
}
NM_TO_KM = 1.852


@dataclass
class IngestResult:
    """Outcome of one ingestion run (persisted to ``ingestion_log``)."""

    source: str
    dataset: str
    status: str = "ok"          # ok | partial | skipped | failed
    ingested: int = 0
    rejected: int = 0
    message: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "dataset": self.dataset,
            "status": self.status,
            "ingested": self.ingested,
            "rejected": self.rejected,
            "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _http_get(url: str, settings: Settings, stream_to: Path | None = None) -> bytes:
    """GET with retries and exponential backoff. Raises on final failure."""
    import time

    import requests

    last_exc: Exception | None = None
    for attempt in range(1, settings.download_max_retries + 1):
        try:
            resp = requests.get(
                url,
                timeout=settings.http_timeout_seconds,
                headers={"User-Agent": f"POLARIS/{settings.app_version} (SIH26059; research)"},
            )
            resp.raise_for_status()
            payload = resp.content
            if stream_to is not None:
                stream_to.parent.mkdir(parents=True, exist_ok=True)
                stream_to.write_bytes(payload)
            return payload
        except Exception as exc:  # noqa: BLE001 - retried and re-raised below
            last_exc = exc
            wait = min(2 ** attempt, 8)
            log.warning(
                "GET failed (attempt %d/%d): %s - retrying in %ds",
                attempt, settings.download_max_retries, exc.__class__.__name__, wait,
            )
            if attempt < settings.download_max_retries:
                time.sleep(wait)
    raise DataUnavailableError(f"Download failed after {settings.download_max_retries} attempts: {last_exc}")


def _get_json(url: str, settings: Settings) -> Any:
    import json

    return json.loads(_http_get(url, settings).decode("utf-8"))


# ---------------------------------------------------------------------------
# NSIDC - sea ice
# ---------------------------------------------------------------------------
class NSIDCClient:
    """Client for the NSIDC Sea Ice Index (G02135) Antarctic products."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = Path(self.settings.raw_data_dir) / "nsidc"

    def concentration_url(self, day: date) -> str:
        return (
            f"{self.settings.nsidc_base_url}/south/daily/geotiff/{day:%Y}/"
            f"{MONTH_DIRS[day.month]}/S_{day:%Y%m%d}_concentration_v4.0.tif"
        )

    def extent_index_url(self) -> str:
        return f"{self.settings.nsidc_base_url}/south/daily/data/S_seaice_extent_daily_v4.0.csv"

    def local_path(self, day: date) -> Path:
        return self.cache_dir / f"S_{day:%Y%m%d}_concentration_v4.0.tif"

    def fetch_concentration(self, day: date, use_cache: bool = True) -> Path:
        """Download (or reuse) one daily concentration GeoTIFF."""
        path = self.local_path(day)
        if use_cache and path.exists() and path.stat().st_size > 10_000:
            return path
        _http_get(self.concentration_url(day), self.settings, stream_to=path)
        log.info("Downloaded NSIDC concentration for %s (%d bytes)", day, path.stat().st_size)
        return path

    @staticmethod
    def read_concentration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read a GeoTIFF into ``(concentration, land, missing)`` arrays.

        Reading is done with Pillow rather than rasterio/GDAL: the NSIDC grid is
        a fixed, documented polar-stereographic grid, so georeferencing comes
        from :mod:`app.utils.geo` constants and no GDAL dependency is required.
        """
        from PIL import Image

        with Image.open(path) as im:
            raw = np.array(im)
        if raw.ndim != 2:
            raise DataUnavailableError(f"{path.name}: expected a single-band raster, got {raw.shape}")
        return decode_nsidc_concentration(raw)

    def concentration_on_grid(self, day: date, grid: GridSpec, use_cache: bool = True) -> np.ndarray:
        """Fetch one day and regrid it onto the POLARIS analysis grid.

        Near the northern edge of the domain an analysis cell can be larger
        than a 25 km source cell in longitude, leaving small binning gaps; a
        short diffusion fill closes those without inventing large-scale
        structure.  Land cells are returned as NaN.
        """
        path = self.fetch_concentration(day, use_cache=use_cache)
        conc, _land, _missing = self.read_concentration(path)
        lat, lon = nsidc_grid_latlon(conc.shape)
        field = regrid_nearest(lat, lon, conc, grid)
        field = fill_nan_nearest(field, max_iter=3)
        field[landmask.land_fraction(grid) >= 0.5] = np.nan
        return field

    def fetch_extent_index(self) -> pd.DataFrame:
        """Daily Antarctic sea-ice extent/area index as a tidy frame."""
        raw = _http_get(self.extent_index_url(), self.settings)
        df = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
        # The file carries a units row directly under the header.
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df[pd.to_numeric(df.get("year"), errors="coerce").notna()].copy()
        df["observed_at"] = pd.to_datetime(
            dict(
                year=df["year"].astype(int),
                month=df["month"].astype(int),
                day=df["day"].astype(int),
            ),
            utc=True,
        )
        out = pd.DataFrame(
            {
                "observed_at": df["observed_at"],
                "extent_million_km2": pd.to_numeric(df.get("extent"), errors="coerce"),
                "area_million_km2": pd.to_numeric(df.get("area"), errors="coerce"),
            }
        )
        return out[out["extent_million_km2"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# USNIC - icebergs
# ---------------------------------------------------------------------------
class USNICClient:
    """Client for the US National Ice Center Antarctic iceberg bulletin."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def fetch(self) -> pd.DataFrame:
        raw = _http_get(self.settings.usnic_iceberg_url, self.settings)
        text = raw.decode("utf-8-sig", errors="replace")
        df = pd.read_csv(io.StringIO(text))
        df.columns = [str(c).strip() for c in df.columns]
        rename = {
            "Iceberg": "iceberg_id",
            "Length (NM)": "length_nm",
            "Width (NM)": "width_nm",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Area (sqKM)": "area_km2",
            "Last Update": "last_update",
        }
        missing = [c for c in rename if c not in df.columns]
        if missing:
            raise DataUnavailableError(
                f"USNIC bulletin format changed; missing columns {missing} (got {list(df.columns)})"
            )
        out = df.rename(columns=rename)[list(rename.values())].copy()
        out["iceberg_id"] = out["iceberg_id"].astype(str).str.strip().str.upper()
        out["last_update"] = pd.to_datetime(out["last_update"], format="%m/%d/%Y", utc=True, errors="coerce")
        out["observed_at"] = out["last_update"]
        for c in ("length_nm", "width_nm", "latitude", "longitude", "area_km2"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        # Recompute area when the bulletin omits it.
        need_area = out["area_km2"].isna() & out["length_nm"].notna() & out["width_nm"].notna()
        out.loc[need_area, "area_km2"] = (
            out.loc[need_area, "length_nm"] * out.loc[need_area, "width_nm"] * NM_TO_KM**2
        )
        return out


# ---------------------------------------------------------------------------
# ERA5 - atmosphere
# ---------------------------------------------------------------------------
class ERA5Client:
    """ERA5 atmospheric reanalysis.

    Preferred path is the Copernicus Climate Data Store (``cdsapi``), which
    needs a free CDS account.  When no key is configured and open mirrors are
    permitted, the key-free ERA5 archive endpoint is used instead; it serves the
    same ERA5 product and is recorded under a distinct ``source`` string.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def has_cds_credentials(self) -> bool:
        return bool(self.settings.era5_api_key)

    def available(self) -> bool:
        return self.has_cds_credentials or self.settings.allow_open_mirrors

    # -- CDS path --------------------------------------------------------
    def fetch_via_cds(self, day: date, grid: GridSpec, out_dir: Path) -> tuple[dict[str, np.ndarray], str]:
        """Retrieve ERA5 single-level fields as NetCDF via the CDS API."""
        try:
            import cdsapi  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DataUnavailableError(
                "cdsapi is not installed; `pip install cdsapi` to use the credentialed ERA5 path"
            ) from exc
        if not self.has_cds_credentials:
            raise DataUnavailableError("ERA5_API_KEY is not configured")

        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"era5_{day:%Y%m%d}.nc"
        if not target.exists():
            client = cdsapi.Client(url=self.settings.era5_api_url, key=self.settings.era5_api_key)
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "2m_temperature",
                        "mean_sea_level_pressure",
                    ],
                    "year": f"{day:%Y}",
                    "month": f"{day:%m}",
                    "day": f"{day:%d}",
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "area": [grid.lat_max, grid.lon_min, grid.lat_min, grid.lon_max],
                },
                str(target),
            )
        return self._read_era5_netcdf(target, grid), SOURCE_ERA5_CDS

    @staticmethod
    def _read_era5_netcdf(path: Path, grid: GridSpec) -> dict[str, np.ndarray]:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            ds = ds.mean(dim=[d for d in ("time", "valid_time", "number") if d in ds.dims])
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            lat2d, lon2d = np.meshgrid(ds[lat_name].values, ds[lon_name].values, indexing="ij")
            out: dict[str, np.ndarray] = {}
            mapping = {"u10": "u10", "v10": "v10", "t2m": "t2m", "msl": "msl"}
            for key, var in mapping.items():
                if var not in ds:
                    continue
                values = np.asarray(ds[var].values, dtype=float)
                regridded = regrid_nearest(lat2d, lon2d, values, grid)
                if key == "t2m":
                    regridded = regridded - 273.15
                if key == "msl":
                    regridded = regridded / 100.0
                out[key] = regridded
        if not out:
            raise DataUnavailableError(f"{path.name}: no recognised ERA5 variables present")
        return out

    # -- open mirror path ------------------------------------------------
    def fetch_via_mirror(self, day: date, grid: GridSpec) -> tuple[dict[str, np.ndarray], str]:
        """Sample the key-free ERA5 archive on a coarse grid and interpolate."""
        if not self.settings.allow_open_mirrors:
            raise DataUnavailableError("Open mirrors disabled (ALLOW_OPEN_MIRRORS=false)")
        lats, lons = _mirror_sample_points(grid, self.settings)
        url = (
            f"{self.settings.open_meteo_era5_url}?"
            f"latitude={','.join(f'{v:.4f}' for v in lats)}&"
            f"longitude={','.join(f'{v:.4f}' for v in lons)}&"
            f"start_date={day:%Y-%m-%d}&end_date={day:%Y-%m-%d}&"
            "hourly=wind_speed_10m,wind_direction_10m,temperature_2m,pressure_msl&"
            "wind_speed_unit=ms&models=era5"
        )
        payload = _get_json(url, self.settings)
        records = payload if isinstance(payload, list) else [payload]

        pt_lat, pt_lon = [], []
        vals: dict[str, list[float]] = {"u10": [], "v10": [], "t2m": [], "msl": []}
        for rec in records:
            hourly = rec.get("hourly") or {}
            speed = _mean_of(hourly.get("wind_speed_10m"))
            direction = _mean_direction(hourly.get("wind_direction_10m"))
            temp = _mean_of(hourly.get("temperature_2m"))
            pres = _mean_of(hourly.get("pressure_msl"))
            if speed is None or direction is None:
                continue
            # Meteorological direction is where the wind comes FROM.
            rad = math.radians(direction)
            pt_lat.append(float(rec["latitude"]))
            pt_lon.append(float(rec["longitude"]))
            vals["u10"].append(-speed * math.sin(rad))
            vals["v10"].append(-speed * math.cos(rad))
            vals["t2m"].append(temp if temp is not None else float("nan"))
            vals["msl"].append(pres if pres is not None else float("nan"))

        if len(pt_lat) < 4:
            raise DataUnavailableError("ERA5 mirror returned too few valid points")
        return _points_to_grid(pt_lat, pt_lon, vals, grid), SOURCE_ERA5_MIRROR

    def fetch(self, day: date, grid: GridSpec, out_dir: Path) -> tuple[dict[str, np.ndarray], str]:
        if self.has_cds_credentials:
            try:
                return self.fetch_via_cds(day, grid, out_dir)
            except Exception as exc:
                log.warning("ERA5 CDS path failed (%s); trying open archive", exc.__class__.__name__)
        return self.fetch_via_mirror(day, grid)


# ---------------------------------------------------------------------------
# Live atmospheric and marine conditions
# ---------------------------------------------------------------------------
class LiveWeatherClient:
    """Current and short-range forecast conditions.

    ERA5 is a *reanalysis*: assimilated after the fact and published with about
    five days of latency.  That is the right product for training a model on a
    year of consistent history, and the wrong one for an operational display -
    a navigation system showing six-day-old wind is describing the past.

    This client serves the ECMWF IFS forecast instead, valid now and running
    several days ahead, recorded under its own ``source`` so it is never
    conflated with the reanalysis.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def available(self) -> bool:
        return bool(self.settings.allow_open_mirrors)

    def fetch_current_grid(self, grid: GridSpec) -> tuple[dict[str, np.ndarray], datetime, str]:
        """Current atmospheric conditions interpolated onto the analysis grid."""
        if not self.available():
            raise DataUnavailableError(
                "Live weather requires ALLOW_OPEN_MIRRORS=true (the forecast endpoint needs no key)"
            )
        lats, lons = _mirror_sample_points(grid, self.settings)
        url = (
            f"{self.settings.weather_live_url}?"
            f"latitude={','.join(f'{v:.4f}' for v in lats)}&"
            f"longitude={','.join(f'{v:.4f}' for v in lons)}&"
            "current=temperature_2m,wind_speed_10m,wind_direction_10m,pressure_msl&"
            f"wind_speed_unit=ms&models={self.settings.weather_live_model}"
        )
        payload = _get_json(url, self.settings)
        records = payload if isinstance(payload, list) else [payload]

        pt_lat: list[float] = []
        pt_lon: list[float] = []
        vals: dict[str, list[float]] = {"u10": [], "v10": [], "t2m": [], "msl": []}
        valid_at: datetime | None = None
        for rec in records:
            cur = rec.get("current") or {}
            speed, direction = cur.get("wind_speed_10m"), cur.get("wind_direction_10m")
            if speed is None or direction is None:
                continue
            if valid_at is None and cur.get("time"):
                valid_at = pd.to_datetime(cur["time"], utc=True).to_pydatetime()
            rad = math.radians(float(direction))
            pt_lat.append(float(rec["latitude"]))
            pt_lon.append(float(rec["longitude"]))
            vals["u10"].append(-float(speed) * math.sin(rad))
            vals["v10"].append(-float(speed) * math.cos(rad))
            vals["t2m"].append(float(cur.get("temperature_2m") or float("nan")))
            vals["msl"].append(float(cur.get("pressure_msl") or float("nan")))

        if len(pt_lat) < 4:
            raise DataUnavailableError("Live weather returned too few valid points")
        fields = _points_to_grid(pt_lat, pt_lon, vals, grid)
        return fields, (valid_at or datetime.now(timezone.utc)), SOURCE_LIVE_IFS

    def fetch_points(
        self, points: Sequence[tuple[str, float, float]], forecast_hours: int | None = None
    ) -> list[dict]:
        """Current conditions plus an hourly outlook at named positions."""
        if not points:
            return []
        if not self.available():
            # ALLOW_OPEN_MIRRORS=false must actually prevent the call, not just
            # the gridded one - otherwise the setting silently means nothing
            # here and an operator who disabled open endpoints still hits them.
            raise DataUnavailableError(
                "Live weather requires ALLOW_OPEN_MIRRORS=true (the forecast endpoint needs no key)"
            )
        forecast_hours = forecast_hours or self.settings.station_forecast_hours
        days = max(1, min(7, math.ceil(forecast_hours / 24)))
        lat = ",".join(f"{p[1]:.4f}" for p in points)
        lon = ",".join(f"{p[2]:.4f}" for p in points)

        atmos = _get_json(
            f"{self.settings.weather_live_url}?latitude={lat}&longitude={lon}&"
            "current=temperature_2m,wind_speed_10m,wind_direction_10m,pressure_msl,"
            "relative_humidity_2m,cloud_cover&"
            "hourly=temperature_2m,wind_speed_10m,wind_direction_10m,pressure_msl&"
            f"wind_speed_unit=ms&forecast_days={days}&models={self.settings.weather_live_model}",
            self.settings,
        )
        atmos_records = atmos if isinstance(atmos, list) else [atmos]

        marine_records: list[dict] = []
        try:
            marine = _get_json(
                f"{self.settings.marine_live_url}?latitude={lat}&longitude={lon}&"
                "current=wave_height,wave_period,ocean_current_velocity,"
                "ocean_current_direction,sea_surface_temperature&"
                f"forecast_days={days}",
                self.settings,
            )
            marine_records = marine if isinstance(marine, list) else [marine]
        except Exception as exc:  # noqa: BLE001 - marine is optional at a land station
            log.warning("Live marine conditions unavailable: %s", exc)

        out: list[dict] = []
        for k, (name, plat, plon) in enumerate(points):
            atm = atmos_records[k].get("current", {}) if k < len(atmos_records) else {}
            mar = marine_records[k].get("current", {}) if k < len(marine_records) else {}
            hourly = atmos_records[k].get("hourly", {}) if k < len(atmos_records) else {}
            speed = atm.get("wind_speed_10m")
            entry = {
                "name": name,
                "latitude": plat,
                "longitude": plon,
                "observed_at": (
                    pd.to_datetime(atm["time"], utc=True).to_pydatetime() if atm.get("time") else None
                ),
                "air_temperature_c": atm.get("temperature_2m"),
                "wind_speed_m_s": speed,
                "wind_direction_deg": atm.get("wind_direction_10m"),
                "beaufort_force": _beaufort(speed),
                "mean_sea_level_pressure_hpa": atm.get("pressure_msl"),
                "relative_humidity_pct": atm.get("relative_humidity_2m"),
                "cloud_cover_pct": atm.get("cloud_cover"),
                # Marine values are null at a point inside the pack: waves are
                # not defined under sea ice. That is physics, not a gap.
                "significant_wave_height_m": mar.get("wave_height"),
                "wave_period_s": mar.get("wave_period"),
                "ocean_current_speed_m_s": mar.get("ocean_current_velocity"),
                "ocean_current_direction_deg": mar.get("ocean_current_direction"),
                "sea_surface_temperature_c": mar.get("sea_surface_temperature"),
                "source": SOURCE_LIVE_IFS,
                "marine_source": SOURCE_LIVE_MARINE if mar else None,
                "forecast": _hourly_outlook(hourly, forecast_hours),
            }
            entry["freezing_spray_risk"] = _freezing_spray(entry)
            out.append(entry)
        return out


def _beaufort(speed_m_s: float | None) -> int | None:
    """Beaufort force from wind speed - the scale bridge crews actually use."""
    if speed_m_s is None:
        return None
    limits = [0.5, 1.5, 3.3, 5.5, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6]
    return next((i for i, lim in enumerate(limits) if speed_m_s < lim), 12)


def _freezing_spray(entry: dict) -> str | None:
    """Qualitative freezing-spray hazard.

    Superstructure icing needs cold air, open water and enough wind to generate
    spray; the classical predictors combine exactly those.  Reported
    qualitatively because POLARIS does not compute an icing accretion rate.
    """
    t = entry.get("air_temperature_c")
    w = entry.get("wind_speed_m_s")
    sst = entry.get("sea_surface_temperature_c")
    if t is None or w is None:
        return None
    if t > -2.0 or w < 9.0:
        return "none"
    if sst is not None and sst > 5.0:
        return "none"
    if t < -12.0 and w > 17.0:
        return "severe"
    if t < -7.0 and w > 13.0:
        return "moderate"
    return "light"


def _hourly_outlook(hourly: dict, hours: int) -> list[dict]:
    """Trim the hourly forecast to the requested window."""
    times = (hourly or {}).get("time") or []
    out = []
    for i, stamp in enumerate(times[:hours]):
        out.append(
            {
                "valid_at": pd.to_datetime(stamp, utc=True).to_pydatetime(),
                "air_temperature_c": _at(hourly.get("temperature_2m"), i),
                "wind_speed_m_s": _at(hourly.get("wind_speed_10m"), i),
                "wind_direction_deg": _at(hourly.get("wind_direction_10m"), i),
                "mean_sea_level_pressure_hpa": _at(hourly.get("pressure_msl"), i),
            }
        )
    return out


def _at(series, index):
    if not series or index >= len(series):
        return None
    return series[index]


# ---------------------------------------------------------------------------
# Copernicus Marine - ocean
# ---------------------------------------------------------------------------
class CopernicusMarineClient:
    """Copernicus Marine (CMEMS) ocean analysis.

    Preferred path uses the official ``copernicusmarine`` toolbox with the
    operator's CMEMS credentials.  Without credentials, the key-free marine
    endpoint (which redistributes CMEMS/ECMWF ocean and wave analyses) is used
    and recorded under its own ``source`` string.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def has_credentials(self) -> bool:
        return bool(self.settings.copernicus_username and self.settings.copernicus_password)

    def available(self) -> bool:
        return self.has_credentials or self.settings.allow_open_mirrors

    def dataset_plan(self) -> list[tuple[str, list[str], dict[str, str]]]:
        """Which CMEMS dataset supplies which variable.

        Copernicus Marine splits the global physics analysis-forecast product:
        currents, temperature and salinity each have their own dataset, and the
        merged one carries the sea-ice fields.  Verified against the live
        catalogue - requesting ``uo`` from the merged dataset fails outright.
        """
        cfg = self.settings
        return [
            (cfg.copernicus_dataset_currents, ["uo", "vo"],
             {"uo": "u_current", "vo": "v_current"}),
            (cfg.copernicus_dataset_temperature, ["thetao"], {"thetao": "sst"}),
            (cfg.copernicus_dataset_salinity, ["so"], {"so": "salinity"}),
            # Sea-ice drift straight from the ocean model. This is the same
            # quantity as the NSIDC Polar Pathfinder motion vectors, without an
            # Earthdata login, and it arrives as a forecast rather than an
            # analysis.
            (cfg.copernicus_dataset_seaice, ["siconc", "sithick", "usi", "vsi"],
             {"siconc": "cmems_sea_ice_concentration", "sithick": "sea_ice_thickness",
              "usi": "sea_ice_u", "vsi": "sea_ice_v"}),
            # Significant wave height lives in the separate wave product; without
            # it the ocean risk component would lose its dominant term.
            (cfg.copernicus_dataset_waves, ["VHM0"], {"VHM0": "wave_height"}),
        ]

    def fetch_via_toolbox(self, day: date, grid: GridSpec, out_dir: Path) -> tuple[dict[str, np.ndarray], str]:
        """Retrieve every ocean variable through the official CMEMS toolbox.

        Each dataset is subset separately and merged onto the analysis grid.  A
        dataset that fails is logged and skipped rather than aborting the run,
        so one retired dataset id cannot take the whole ingestion down.
        """
        try:
            import copernicusmarine  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DataUnavailableError(
                "copernicusmarine is not installed; `pip install copernicusmarine` for the "
                "credentialed CMEMS path"
            ) from exc
        if not self.has_credentials:
            raise DataUnavailableError("COPERNICUS_USERNAME / COPERNICUS_PASSWORD are not configured")

        out_dir.mkdir(parents=True, exist_ok=True)
        fields: dict[str, np.ndarray] = {}
        failures: list[str] = []

        for dataset_id, variables, rename in self.dataset_plan():
            target = out_dir / f"cmems_{dataset_id}_{day:%Y%m%d}.nc"
            try:
                if not target.exists():
                    copernicusmarine.subset(
                        dataset_id=dataset_id,
                        variables=variables,
                        minimum_longitude=grid.lon_min,
                        maximum_longitude=grid.lon_max,
                        minimum_latitude=grid.lat_min,
                        maximum_latitude=grid.lat_max,
                        start_datetime=f"{day:%Y-%m-%d}T00:00:00",
                        end_datetime=f"{day:%Y-%m-%d}T23:59:59",
                        output_filename=target.name,
                        output_directory=str(out_dir),
                        username=self.settings.copernicus_username,
                        password=self.settings.copernicus_password,
                    )
                fields.update(self._read_cmems_netcdf(target, grid, rename))
            except Exception as exc:  # noqa: BLE001 - one dataset must not stop the rest
                failures.append(f"{dataset_id}: {exc.__class__.__name__}: {exc}")
                log.warning("CMEMS dataset %s failed: %s", dataset_id, exc)

        if not fields:
            raise DataUnavailableError("No CMEMS dataset could be retrieved: " + "; ".join(failures))
        if failures:
            log.warning("CMEMS partial retrieval; %d dataset(s) failed", len(failures))
        return fields, SOURCE_CMEMS

    @staticmethod
    def _read_cmems_netcdf(path: Path, grid: GridSpec, rename: dict[str, str]) -> dict[str, np.ndarray]:
        """Read one CMEMS subset and regrid the requested variables."""
        import xarray as xr

        out: dict[str, np.ndarray] = {}
        with xr.open_dataset(path) as ds:
            squeeze = [d for d in ("time", "depth", "elevation") if d in ds.dims]
            if squeeze:
                ds = ds.isel({d: 0 for d in squeeze})
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            lat2d, lon2d = np.meshgrid(ds[lat_name].values, ds[lon_name].values, indexing="ij")
            for var, key in rename.items():
                if var in ds:
                    out[key] = regrid_nearest(lat2d, lon2d, np.asarray(ds[var].values, float), grid)
        if not out:
            raise DataUnavailableError(f"{path.name}: none of {list(rename)} present")
        return out

    def fetch_ice_drift_history(
        self,
        start: date,
        end: date,
        grid: GridSpec,
        variables: Sequence[str] = ("usi", "vsi", "siconc"),
    ) -> tuple[list[datetime], dict[str, np.ndarray]]:
        """Daily sea-ice drift history on the analysis grid.

        ``usi``/``vsi`` are the ocean model's sea-ice velocity - the same
        quantity as the NSIDC Polar Pathfinder motion vectors, without an
        Earthdata login.  Ice *advection* and *convergence* are the dominant
        dynamic drivers of concentration change, so this is the physics the
        forecast model was missing.

        The dataset is opened lazily and strided down to approximately the
        analysis grid before anything is transferred: the source is 0.083 deg,
        so a stride of 6 x 12 lands almost exactly on a 0.5 x 1.0 degree grid
        and cuts the download by about seventy times.  Striding subsamples
        rather than averages, which is acceptable for a smooth velocity field
        and is why the result is re-gridded properly below.
        """
        try:
            import copernicusmarine  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DataUnavailableError("copernicusmarine is not installed") from exc
        if not self.has_credentials:
            raise DataUnavailableError("COPERNICUS_USERNAME / COPERNICUS_PASSWORD are not configured")

        ds = copernicusmarine.open_dataset(
            dataset_id=self.settings.copernicus_dataset_seaice,
            username=self.settings.copernicus_username,
            password=self.settings.copernicus_password,
            variables=list(variables),
            minimum_longitude=grid.lon_min,
            maximum_longitude=grid.lon_max,
            minimum_latitude=grid.lat_min,
            maximum_latitude=grid.lat_max,
            start_datetime=f"{start:%Y-%m-%d}T00:00:00",
            end_datetime=f"{end:%Y-%m-%d}T23:59:59",
        )
        try:
            src_dlat = float(abs(ds.latitude.values[1] - ds.latitude.values[0]))
            src_dlon = float(abs(ds.longitude.values[1] - ds.longitude.values[0]))
            stride_lat = max(1, int(round(grid.dlat / src_dlat)))
            stride_lon = max(1, int(round(grid.dlon / src_dlon)))
            log.info(
                "CMEMS ice drift: source %.4f/%.4f deg, striding %dx%d before transfer",
                src_dlat, src_dlon, stride_lat, stride_lon,
            )
            sub = ds.isel(
                latitude=slice(None, None, stride_lat),
                longitude=slice(None, None, stride_lon),
            ).load()
            times = [
                pd.Timestamp(t).to_pydatetime().replace(tzinfo=timezone.utc)
                for t in sub["time"].values
            ]
            lat2d, lon2d = np.meshgrid(sub["latitude"].values, sub["longitude"].values, indexing="ij")
            out: dict[str, np.ndarray] = {}
            for var in variables:
                if var not in sub:
                    continue
                stack = [
                    regrid_nearest(lat2d, lon2d, np.asarray(sub[var].values[k], float), grid)
                    for k in range(len(times))
                ]
                out[var] = np.stack(stack).astype("float32")
        finally:
            ds.close()

        if not out:
            raise DataUnavailableError("CMEMS returned none of the requested ice-drift variables")
        log.info("CMEMS ice drift history: %d day(s), variables %s", len(times), sorted(out))
        return times, out

    def fetch_via_mirror(self, grid: GridSpec) -> tuple[dict[str, np.ndarray], str]:
        if not self.settings.allow_open_mirrors:
            raise DataUnavailableError("Open mirrors disabled (ALLOW_OPEN_MIRRORS=false)")
        lats, lons = _mirror_sample_points(grid, self.settings)
        url = (
            f"{self.settings.open_meteo_marine_url}?"
            f"latitude={','.join(f'{v:.4f}' for v in lats)}&"
            f"longitude={','.join(f'{v:.4f}' for v in lons)}&"
            "hourly=wave_height,ocean_current_velocity,ocean_current_direction,"
            "sea_surface_temperature&forecast_days=1"
        )
        payload = _get_json(url, self.settings)
        records = payload if isinstance(payload, list) else [payload]

        pt_lat, pt_lon = [], []
        vals: dict[str, list[float]] = {"u_current": [], "v_current": [], "sst": [], "wave_height": []}
        for rec in records:
            hourly = rec.get("hourly") or {}
            speed = _mean_of(hourly.get("ocean_current_velocity"))
            direction = _mean_direction(hourly.get("ocean_current_direction"))
            if speed is None or direction is None:
                continue
            # Marine convention: current direction is where the flow goes TO.
            speed_ms = speed / 100.0 if speed > 5 else speed  # cm/s guard
            rad = math.radians(direction)
            pt_lat.append(float(rec["latitude"]))
            pt_lon.append(float(rec["longitude"]))
            vals["u_current"].append(speed_ms * math.sin(rad))
            vals["v_current"].append(speed_ms * math.cos(rad))
            sst = _mean_of(hourly.get("sea_surface_temperature"))
            hs = _mean_of(hourly.get("wave_height"))
            vals["sst"].append(sst if sst is not None else float("nan"))
            vals["wave_height"].append(hs if hs is not None else float("nan"))

        if len(pt_lat) < 4:
            raise DataUnavailableError("Marine mirror returned too few valid points")
        return _points_to_grid(pt_lat, pt_lon, vals, grid), SOURCE_MARINE_MIRROR

    def fetch(self, day: date, grid: GridSpec, out_dir: Path) -> tuple[dict[str, np.ndarray], str]:
        if self.has_credentials:
            try:
                return self.fetch_via_toolbox(day, grid, out_dir)
            except Exception as exc:
                log.warning("CMEMS toolbox path failed (%s); trying open marine endpoint", exc.__class__.__name__)
        return self.fetch_via_mirror(grid)


# ---------------------------------------------------------------------------
# Shared helpers for the point-sampling providers
# ---------------------------------------------------------------------------
def _mirror_sample_points(grid: GridSpec, settings: Settings) -> tuple[list[float], list[float]]:
    lats = np.arange(grid.lat_min, grid.lat_max + 1e-9, settings.mirror_sample_dlat)
    lons = np.arange(grid.lon_min, grid.lon_max + 1e-9, settings.mirror_sample_dlon)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    return [float(v) for v in la.ravel()], [float(v) for v in lo.ravel()]


def _mean_of(series: Iterable | None) -> float | None:
    if not series:
        return None
    arr = np.array([v for v in series if v is not None], dtype=float)
    return float(arr.mean()) if arr.size else None


def _mean_direction(series: Iterable | None) -> float | None:
    """Circular mean of a direction series in degrees."""
    if not series:
        return None
    arr = np.array([v for v in series if v is not None], dtype=float)
    if not arr.size:
        return None
    rad = np.radians(arr)
    return float((math.degrees(math.atan2(np.sin(rad).mean(), np.cos(rad).mean())) + 360.0) % 360.0)


def _points_to_grid(
    pt_lat: Sequence[float], pt_lon: Sequence[float], values: dict[str, list[float]], grid: GridSpec
) -> dict[str, np.ndarray]:
    """Interpolate scattered points onto the analysis grid (linear + fill)."""
    from scipy.interpolate import griddata

    lat2d, lon2d = grid.meshgrid()
    pts = np.column_stack([np.asarray(pt_lat, float), np.asarray(pt_lon, float)])
    out: dict[str, np.ndarray] = {}
    for key, raw in values.items():
        v = np.asarray(raw, dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 4:
            out[key] = np.full(grid.shape, np.nan)
            continue
        interp = griddata(pts[ok], v[ok], (lat2d, lon2d), method="linear")
        nearest = griddata(pts[ok], v[ok], (lat2d, lon2d), method="nearest")
        out[key] = np.where(np.isfinite(interp), interp, nearest)
    return out


def _grid_rows(
    grid: GridSpec,
    observed_at: datetime,
    fields: dict[str, np.ndarray],
    column_map: dict[str, str],
    source: str,
    data_mode: str,
    dataset_version: str | None = None,
    extra: dict | None = None,
    skip_all_nan: bool = True,
) -> list[dict]:
    """Flatten gridded fields into per-cell DB rows, dropping empty cells."""
    lat2d, lon2d = grid.meshgrid()
    rows: list[dict] = []
    keys = [k for k in column_map if k in fields]
    stacked = {k: fields[k] for k in keys}
    rows_n, cols_n = grid.shape
    for i in range(rows_n):
        for j in range(cols_n):
            payload = {}
            any_value = False
            for k in keys:
                v = stacked[k][i, j]
                if v is None or (isinstance(v, float) and math.isnan(v)) or not np.isfinite(v):
                    payload[column_map[k]] = None
                else:
                    payload[column_map[k]] = float(v)
                    any_value = True
            if skip_all_nan and not any_value:
                continue
            row = {
                "observed_at": observed_at,
                "latitude": float(lat2d[i, j]),
                "longitude": float(lon2d[i, j]),
                "source": source,
                "data_mode": data_mode,
                "dataset_version": dataset_version,
            }
            row.update(payload)
            if extra:
                row.update({k: (v[i, j] if isinstance(v, np.ndarray) else v) for k, v in extra.items()})
            rows.append(row)
    return rows


def _midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Ingestion entry points
# ---------------------------------------------------------------------------
def ingest_sea_ice(
    session,
    grid: GridSpec | None = None,
    days: int = 5,
    end_date: date | None = None,
    settings: Settings | None = None,
) -> IngestResult:
    """Ingest daily gridded sea-ice concentration into the database.

    Real mode pulls NSIDC G02135 GeoTIFFs; demo mode generates the synthetic
    field.  Both write through the identical validation/upsert path.
    """
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    demo = settings.is_demo
    source = DEMO_SOURCE if demo else SOURCE_NSIDC
    result = IngestResult(source=source, dataset="sea_ice_concentration")
    entry = repo.start_ingestion(session, source, result.dataset, settings.data_mode)

    # NSIDC publishes the previous day's grid with roughly 24-48 h of latency.
    end_date = end_date or (datetime.now(timezone.utc).date() - timedelta(days=1 if demo else 2))
    generator = DemoFieldGenerator(grid) if demo else None
    client = None if demo else NSIDCClient(settings)

    total, failures, per_day = 0, [], []
    for k in range(days):
        day = end_date - timedelta(days=k)
        observed_at = _midnight(day)
        try:
            if demo:
                field = generator.sea_ice_concentration(observed_at)
            else:
                field = client.concentration_on_grid(day, grid)
                if np.isfinite(field).sum() < 0.05 * field.size:
                    raise DataUnavailableError(f"NSIDC grid for {day} is almost entirely empty")
        except Exception as exc:
            failures.append(f"{day}: {exc.__class__.__name__}: {exc}")
            log.warning("Sea-ice ingestion failed for %s: %s", day, exc)
            continue

        land = landmask.land_fraction(grid) >= 0.5
        rows = []
        lat2d, lon2d = grid.meshgrid()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = field[i, j]
                is_land = bool(land[i, j])
                if is_land:
                    concentration, flag = 0.0, "land"
                elif np.isfinite(v):
                    concentration, flag = float(np.clip(v, 0.0, 1.0)), "ok"
                else:
                    # The NSIDC polar-stereographic grid is a square and its
                    # corners fall short of the domain's northern edge.  Those
                    # cells sit well north of the ice edge; they are recorded as
                    # ice-free and explicitly flagged rather than dropped.
                    concentration, flag = 0.0, "outside_source_grid"
                rows.append(
                    {
                        "observed_at": observed_at,
                        "latitude": float(lat2d[i, j]),
                        "longitude": float(lon2d[i, j]),
                        "concentration": concentration,
                        "is_land": is_land,
                        "quality_flag": flag,
                        "source": source,
                        "data_mode": settings.data_mode,
                        "dataset_version": "demo-1.0" if demo else "G02135-v4.0",
                    }
                )
        n = repo.upsert_sea_ice_observations(session, rows)
        total += n
        per_day.append({"date": str(day), "cells": n})

    pruned = repo.prune_sea_ice_observations(session, settings.sea_ice_db_retention_days)
    result.ingested = total
    result.details = {"days": per_day, "pruned_rows": pruned, "failures": failures}
    if total == 0:
        result.status = "failed"
        result.message = "; ".join(failures) or "no days ingested"
    elif failures:
        result.status = "partial"
        result.message = f"{len(failures)} day(s) failed: " + "; ".join(failures[:3])
    repo.finish_ingestion(session, entry, result.status, result.ingested, len(failures), result.message, result.details)
    log.info("Sea-ice ingestion %s: %d rows over %d day(s)", result.status, total, len(per_day))
    return result


def ingest_extent_index(session, settings: Settings | None = None, days: int = 400) -> IngestResult:
    """Ingest the hemispheric daily sea-ice extent index."""
    settings = settings or get_settings()
    demo = settings.is_demo
    source = DEMO_SOURCE if demo else SOURCE_NSIDC
    result = IngestResult(source=source, dataset="sea_ice_extent_index")
    entry = repo.start_ingestion(session, source, result.dataset, settings.data_mode)
    try:
        if demo:
            rows = demo_extent_index(datetime.now(timezone.utc), days=days)
        else:
            df = NSIDCClient(settings).fetch_extent_index().tail(days)
            rows = [
                {
                    "observed_at": r.observed_at.to_pydatetime(),
                    "hemisphere": "south",
                    "extent_million_km2": float(r.extent_million_km2),
                    "area_million_km2": (
                        float(r.area_million_km2) if pd.notna(r.area_million_km2) else None
                    ),
                    "source": SOURCE_NSIDC,
                    "data_mode": "real",
                    "dataset_version": "G02135-v4.0",
                }
                for r in df.itertuples()
            ]
        result.ingested = repo.upsert_sea_ice_extent(session, rows)
    except Exception as exc:
        result.status = "failed"
        result.message = f"{exc.__class__.__name__}: {exc}"
        log.error("Extent index ingestion failed: %s", exc)
    repo.finish_ingestion(session, entry, result.status, result.ingested, 0, result.message, result.details)
    return result


def ingest_icebergs(session, settings: Settings | None = None) -> IngestResult:
    """Ingest tracked Antarctic iceberg positions (USNIC, or demo bergs)."""
    settings = settings or get_settings()
    demo = settings.is_demo
    source = DEMO_SOURCE if demo else SOURCE_USNIC
    result = IngestResult(source=source, dataset="antarctic_icebergs")
    entry = repo.start_ingestion(session, source, result.dataset, settings.data_mode)

    try:
        if demo:
            rows = generate_demo_iceberg_observations(
                datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            )
            report = ValidationReport(source=DEMO_SOURCE, n_input=len(rows), n_valid=len(rows))
        else:
            df = USNICClient(settings).fetch()
            clean, report = validate_point_frame(
                df,
                source=SOURCE_USNIC,
                dedupe_on=("iceberg_id", "observed_at"),
                value_ranges={"length_nm": (0.0, 200.0), "width_nm": (0.0, 200.0), "area_km2": (0.0, 1e6)},
            )
            rows = [
                {
                    "iceberg_id": str(r.iceberg_id),
                    "observed_at": r.observed_at.to_pydatetime(),
                    "latitude": float(r.latitude),
                    "longitude": float(r.longitude),
                    "length_nm": float(r.length_nm) if pd.notna(r.length_nm) else None,
                    "width_nm": float(r.width_nm) if pd.notna(r.width_nm) else None,
                    "area_km2": float(r.area_km2) if pd.notna(r.area_km2) else None,
                    "last_update": r.last_update.to_pydatetime() if pd.notna(r.last_update) else None,
                    "source": SOURCE_USNIC,
                    "data_mode": "real",
                    "dataset_version": "usnic-bulletin",
                }
                for r in clean.itertuples()
            ]
        result.ingested = repo.upsert_icebergs(session, rows)
        result.rejected = report.n_dropped_invalid + report.n_dropped_duplicate
        result.details = report.to_dict()
        if result.ingested == 0:
            result.status = "failed"
            result.message = "no iceberg records ingested"
    except Exception as exc:
        result.status = "failed"
        result.message = f"{exc.__class__.__name__}: {exc}"
        log.error("Iceberg ingestion failed: %s", exc)
    repo.finish_ingestion(session, entry, result.status, result.ingested, result.rejected, result.message, result.details)
    log.info("Iceberg ingestion %s: %d rows", result.status, result.ingested)
    return result


def ingest_weather(
    session,
    grid: GridSpec | None = None,
    day: date | None = None,
    settings: Settings | None = None,
) -> IngestResult:
    """Ingest an atmospheric analysis field (ERA5, or the demo atmosphere)."""
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    demo = settings.is_demo
    day = day or (datetime.now(timezone.utc).date() - timedelta(days=(1 if demo else 6)))
    observed_at = _midnight(day)
    source = DEMO_SOURCE if demo else SOURCE_ERA5_MIRROR
    result = IngestResult(source=source, dataset="atmosphere")
    entry = repo.start_ingestion(session, source, result.dataset, settings.data_mode)

    try:
        if demo:
            fields = DemoFieldGenerator(grid).weather(observed_at)
        else:
            fields = None
            # The operational layer wants conditions valid *now*. ERA5 is a
            # reanalysis with ~5 days of latency, so the live IFS forecast is
            # tried first and ERA5 is the fallback, not the other way round.
            if settings.prefer_live_weather:
                try:
                    live = LiveWeatherClient(settings)
                    if live.available():
                        fields, observed_at, source = live.fetch_current_grid(grid)
                        result.source = source
                except Exception as exc:  # noqa: BLE001 - fall back to reanalysis
                    log.warning("Live weather unavailable (%s); falling back to ERA5", exc)
            if fields is None:
                client = ERA5Client(settings)
                if not client.available():
                    raise DataUnavailableError(
                        "No weather source available: set ERA5_API_KEY, or enable ALLOW_OPEN_MIRRORS"
                    )
                fields, source = client.fetch(day, grid, Path(settings.raw_data_dir) / "era5")
                result.source = source

        speed = np.hypot(fields["u10"], fields["v10"])
        direction = (np.degrees(np.arctan2(-fields["u10"], -fields["v10"])) + 360.0) % 360.0
        fields = dict(fields, wind_speed=speed, wind_direction=direction)
        rows = _grid_rows(
            grid,
            observed_at,
            fields,
            {
                "u10": "u10_m_s",
                "v10": "v10_m_s",
                "wind_speed": "wind_speed_m_s",
                "wind_direction": "wind_direction_deg",
                "t2m": "air_temperature_c",
                "msl": "mean_sea_level_pressure_hpa",
            },
            source=result.source,
            data_mode=settings.data_mode,
            dataset_version="demo-1.0" if demo else (
                "ecmwf-ifs-forecast" if result.source == SOURCE_LIVE_IFS else "era5-single-levels"
            ),
        )
        result.ingested = repo.upsert_weather(session, rows)
        age_h = (datetime.now(timezone.utc) - observed_at).total_seconds() / 3600.0
        result.details = {
            "observed_at": observed_at.isoformat(),
            "age_hours": round(age_h, 1),
            "cells": len(rows),
        }
        if result.ingested == 0:
            result.status = "failed"
            result.message = "no weather cells ingested"
    except DataUnavailableError as exc:
        result.status = "skipped"
        result.message = str(exc)
        log.warning("Weather ingestion skipped: %s", exc)
    except Exception as exc:
        result.status = "failed"
        result.message = f"{exc.__class__.__name__}: {exc}"
        log.error("Weather ingestion failed: %s", exc)
    repo.finish_ingestion(session, entry, result.status, result.ingested, 0, result.message, result.details)
    log.info("Weather ingestion %s (%s): %d rows", result.status, result.source, result.ingested)
    return result


def ingest_ocean(
    session,
    grid: GridSpec | None = None,
    day: date | None = None,
    settings: Settings | None = None,
) -> IngestResult:
    """Ingest an ocean analysis field (CMEMS, or the demo ocean)."""
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    demo = settings.is_demo
    day = day or (datetime.now(timezone.utc).date() - timedelta(days=(1 if demo else 0)))
    observed_at = _midnight(day)
    source = DEMO_SOURCE if demo else SOURCE_MARINE_MIRROR
    result = IngestResult(source=source, dataset="ocean")
    entry = repo.start_ingestion(session, source, result.dataset, settings.data_mode)

    try:
        if demo:
            fields = DemoFieldGenerator(grid).ocean(observed_at)
        else:
            client = CopernicusMarineClient(settings)
            if not client.available():
                raise DataUnavailableError(
                    "No CMEMS access configured: set COPERNICUS_USERNAME/PASSWORD, or enable "
                    "ALLOW_OPEN_MIRRORS"
                )
            fields, source = client.fetch(day, grid, Path(settings.raw_data_dir) / "cmems")
            result.source = source

        rows = _grid_rows(
            grid,
            observed_at,
            fields,
            {
                "u_current": "u_current_m_s",
                "v_current": "v_current_m_s",
                "sst": "sea_surface_temperature_c",
                "salinity": "salinity_psu",
                "wave_height": "significant_wave_height_m",
            },
            source=result.source,
            data_mode=settings.data_mode,
            dataset_version="demo-1.0" if demo else settings.copernicus_dataset_id,
        )
        result.ingested = repo.upsert_ocean(session, rows)
        result.details = {"observed_at": observed_at.isoformat(), "cells": len(rows)}
        if result.ingested == 0:
            result.status = "failed"
            result.message = "no ocean cells ingested"
    except DataUnavailableError as exc:
        result.status = "skipped"
        result.message = str(exc)
        log.warning("Ocean ingestion skipped: %s", exc)
    except Exception as exc:
        result.status = "failed"
        result.message = f"{exc.__class__.__name__}: {exc}"
        log.error("Ocean ingestion failed: %s", exc)
    repo.finish_ingestion(session, entry, result.status, result.ingested, 0, result.message, result.details)
    log.info("Ocean ingestion %s (%s): %d rows", result.status, result.source, result.ingested)
    return result


def ingest_all(
    session,
    days: int = 5,
    settings: Settings | None = None,
    grid: GridSpec | None = None,
) -> dict[str, dict]:
    """Run every ingestion in dependency order and return their results."""
    settings = settings or get_settings()
    grid = grid or grid_from_settings(settings)
    results = {
        "sea_ice": ingest_sea_ice(session, grid, days=days, settings=settings),
        "extent_index": ingest_extent_index(session, settings=settings),
        "icebergs": ingest_icebergs(session, settings=settings),
        "weather": ingest_weather(session, grid, settings=settings),
        "ocean": ingest_ocean(session, grid, settings=settings),
    }
    return {k: v.to_dict() for k, v in results.items()}
