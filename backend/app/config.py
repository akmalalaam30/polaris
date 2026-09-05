"""POLARIS central configuration.

All runtime configuration is read from environment variables (optionally via a
``.env`` file).  Nothing here contains credentials; secrets must be supplied by
the operator.  See ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# app/config.py -> app -> backend -> POLARIS
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo"
MODELS_DIR = PROJECT_ROOT / "models"

APP_VERSION = "1.0.0"
APP_NAME = "POLARIS"
APP_TITLE = "POLARIS - Antarctic Sea-Ice, Iceberg Trajectory & Navigation Decision Support"


class Settings(BaseSettings):
    """Runtime settings for the POLARIS backend."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR / ".env", PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Application ------------------------------------------------------
    app_name: str = APP_NAME
    app_version: str = APP_VERSION
    log_level: str = "INFO"
    log_file: str | None = None

    # -- Data mode --------------------------------------------------------
    # "demo" -> use locally generated, clearly labelled synthetic data.
    # "real" -> ingest genuine observations from the upstream providers.
    data_mode: Literal["demo", "real"] = "demo"

    # -- Database ---------------------------------------------------------
    # PostgreSQL/PostGIS is the production target:
    #   postgresql+psycopg2://user:pass@localhost:5432/polaris
    # SQLite is the zero-dependency default so the backend runs anywhere.
    #: ``{mode}`` is replaced by DATA_MODE, so demo and real keep separate
    #: databases and can be switched between without rebuilding either.
    database_url: str = Field(
        default_factory=lambda: "sqlite:///" + (BACKEND_DIR / "polaris_{mode}.db").as_posix()
    )
    database_echo: bool = False

    # -- Filesystem -------------------------------------------------------
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    #: ``{mode}`` is substituted here too, so a demo archive can never be fed
    #: to a real-mode model or vice versa.
    processed_data_dir: Path = PROCESSED_DATA_DIR / "{mode}"
    #: Shared, not mode-specific: this holds the bundled NSIDC-derived land
    #: mask, which is real geography in either mode.
    demo_data_dir: Path = DEMO_DATA_DIR
    model_path: Path = MODELS_DIR / "{mode}"

    # -- External providers ----------------------------------------------
    nsidc_base_url: str = "https://noaadata.apps.nsidc.org/NOAA/G02135"
    usnic_iceberg_url: str = "https://usicecenter.gov/File/DownloadCurrent?pId=134"
    copernicus_username: str | None = None
    copernicus_password: str | None = None
    #: Copernicus Marine splits the global physics analysis-forecast product
    #: across several datasets; currents, temperature and salinity each live in
    #: their own, and the merged dataset carries the sea-ice fields.  Verified
    #: against the live catalogue - a single dataset id does not cover them all.
    copernicus_dataset_currents: str = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
    copernicus_dataset_temperature: str = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
    copernicus_dataset_salinity: str = "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"
    #: Sea-ice concentration, thickness and drift velocity (usi/vsi).
    copernicus_dataset_seaice: str = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
    #: Significant wave height (VHM0) from the global wave analysis-forecast.
    copernicus_dataset_waves: str = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
    #: Retained for backwards compatibility with older .env files.
    copernicus_dataset_id: str = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
    era5_api_key: str | None = None
    era5_api_url: str = "https://cds.climate.copernicus.eu/api"
    #: When the credentialed providers (CDS / Copernicus Marine) are not
    #: configured, real mode may fall back to the open, key-free mirrors that
    #: redistribute the same ERA5 / CMEMS products.  Provenance is recorded
    #: separately for the mirror so the two are never conflated.
    allow_open_mirrors: bool = True
    open_meteo_era5_url: str = "https://archive-api.open-meteo.com/v1/era5"
    #: Live atmospheric forecast for the *operational* layer.  ERA5 is a
    #: reanalysis with about five days of latency - the right product for
    #: training on a year of consistent history, and the wrong one for telling a
    #: master what the wind is doing tonight.  This endpoint serves the ECMWF
    #: IFS forecast, valid now and out to several days.
    weather_live_url: str = "https://api.open-meteo.com/v1/forecast"
    weather_live_model: str = "ecmwf_ifs025"
    marine_live_url: str = "https://marine-api.open-meteo.com/v1/marine"
    #: Prefer the live forecast over ERA5 reanalysis for current conditions.
    prefer_live_weather: bool = True
    #: Hours of forecast retrieved for the station outlook.
    station_forecast_hours: int = 72
    open_meteo_marine_url: str = "https://marine-api.open-meteo.com/v1/marine"
    #: Coarse sampling used when querying the point-based mirrors (degrees).
    mirror_sample_dlat: float = 2.0
    mirror_sample_dlon: float = 5.0
    http_timeout_seconds: float = 60.0
    download_max_retries: int = 3

    # -- Model domain (Antarctic / Southern Ocean study region) -----------
    # Default domain spans Maitri (11.7E) to Bharati (76.2E) with margin.
    domain_lat_min: float = -75.0
    domain_lat_max: float = -50.0
    domain_lon_min: float = 0.0
    domain_lon_max: float = 100.0
    grid_resolution_lat: float = 0.5
    grid_resolution_lon: float = 1.0

    # -- Sea-ice forecasting ---------------------------------------------
    forecast_horizons_hours: tuple[int, ...] = (24, 48, 72)
    sea_ice_model_name: str = "sea_ice_gbr"
    sea_ice_history_days: int = 400
    sea_ice_min_training_days: int = 45
    #: Gridded observations kept in the relational DB (full history lives as
    #: NetCDF under data/processed for model training).
    sea_ice_db_retention_days: int = 14

    # -- Iceberg trajectory ----------------------------------------------
    iceberg_timestep_seconds: int = 1800
    iceberg_ensemble_members: int = 24

    # -- Risk engine weights (documented + configurable) ------------------
    risk_weight_sea_ice: float = 0.40
    risk_weight_iceberg: float = 0.25
    risk_weight_weather: float = 0.20
    risk_weight_ocean: float = 0.10
    risk_weight_constraint: float = 0.05

    # Concentration (fraction 0-1) above which transit is treated as blocked
    # for a vessel with no ice-breaking capability.
    sea_ice_impassable_concentration: float = 0.90
    iceberg_critical_distance_km: float = 15.0
    iceberg_influence_distance_km: float = 120.0

    # -- Route optimisation ----------------------------------------------
    route_distance_weight: float = 1.0
    route_risk_weight: float = 1.0
    route_fuel_weight: float = 1.0
    # Risk above which an edge is removed from the navigation graph entirely.
    route_max_acceptable_risk: float = 0.85

    # -- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api"
    cors_origins: str = "*"
    #: Precompute the land mask, forecasts, trajectories and risk grid at
    #: startup so no user's first request pays the several-second build cost.
    warm_cache_on_startup: bool = True

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("data_mode", mode="before")
    @classmethod
    def _lower_mode(cls, v):
        return v.lower().strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _expand_mode_placeholder(self):
        """Substitute ``{mode}`` in the mode-specific paths.

        Demo and real must never share a database, a processed archive or a
        model directory - a demo-trained model fed real observations would
        produce plausible nonsense.  Writing ``{mode}`` into any of these three
        settings keeps them separate; writing a literal path opts out.
        """
        self.database_url = self.database_url.replace("{mode}", self.data_mode)
        self.processed_data_dir = Path(str(self.processed_data_dir).replace("{mode}", self.data_mode))
        self.model_path = Path(str(self.model_path).replace("{mode}", self.data_mode))
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_demo(self) -> bool:
        return self.data_mode == "demo"

    @property
    def risk_weights(self) -> dict[str, float]:
        return {
            "sea_ice": self.risk_weight_sea_ice,
            "iceberg": self.risk_weight_iceberg,
            "weather": self.risk_weight_weather,
            "ocean": self.risk_weight_ocean,
            "constraint": self.risk_weight_constraint,
        }

    def ensure_directories(self) -> None:
        for d in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.demo_data_dir,
            self.model_path,
        ):
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor (FastAPI dependency friendly)."""
    s = Settings()
    s.ensure_directories()
    return s


def reload_settings() -> Settings:
    """Clear the cache and re-read the environment (used by tests/scripts)."""
    get_settings.cache_clear()
    return get_settings()


# ---------------------------------------------------------------------------
# Reference stations (Indian Antarctic research bases)
# ---------------------------------------------------------------------------
# Published coordinates of the Indian Antarctic research stations operated by
# NCPOR.  These are *station* locations on land; the router snaps them to the
# nearest navigable ocean node (see services/route_optimizer.py).
STATIONS: dict[str, dict] = {
    "bharati": {
        "name": "Bharati Station",
        "operator": "NCPOR / India",
        "region": "Larsemann Hills, Prydz Bay, East Antarctica",
        "latitude": -69.4067,
        "longitude": 76.1894,
    },
    "maitri": {
        "name": "Maitri Station",
        "operator": "NCPOR / India",
        "region": "Schirmacher Oasis, Queen Maud Land",
        "latitude": -70.7660,
        "longitude": 11.7311,
    },
    "cape_town": {
        "name": "Cape Town (resupply port)",
        "operator": "South Africa",
        "region": "Southern Africa",
        "latitude": -33.9060,
        "longitude": 18.4230,
    },
}

DEMO_DATA_DISCLAIMER = (
    "DEMO / SYNTHETIC DATA - generated locally by POLARIS for demonstration and "
    "testing. These values are NOT real scientific observations and must not be "
    "used for navigation."
)

REAL_DATA_DISCLAIMER = (
    "Derived from third-party observational/reanalysis products. POLARIS output is "
    "decision support only and does not guarantee safe navigation."
)
