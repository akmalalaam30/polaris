"""Generation of the POLARIS DEMO dataset.

Everything produced here is **synthetic**.  It is generated so that the exact
same pipeline (ingest -> preprocess -> forecast -> trajectory -> risk -> route)
can be demonstrated without network access or provider credentials.

The fields are *physically shaped* rather than random: the sea-ice seasonal
cycle, the circumpolar trough, the Antarctic Circumpolar Current and the coastal
counter-current are all parameterised from published descriptions of Southern
Ocean climatology.  They are still **not observations** and every record written
to the database carries ``source="POLARIS-DEMO"`` and ``data_mode="demo"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from app.services.landmask import land_fraction
from app.utils.geo import GridSpec, smooth_field
from app.utils.logging import get_logger

log = get_logger("services.demo_data")

DEMO_SOURCE = "POLARIS-DEMO"

# Antarctic sea-ice seasonal cycle: minimum in late February (~day 51),
# maximum in late September (~day 265).
_DOY_MIN = 51.0
_DOY_MAX = 265.0


def _ramp(doy: float) -> float:
    """Piecewise-linear 0 -> 1 -> 0 seasonal ramp, peaking at ``_DOY_MAX``.

    0 corresponds to the annual sea-ice minimum (late February) and 1 to the
    maximum (late September).
    """
    period = 365.25
    up = (_DOY_MAX - _DOY_MIN) % period          # ~214 days of growth
    down = period - up                            # ~151 days of retreat
    d = (doy - _DOY_MIN) % period
    return d / up if d <= up else 1.0 - (d - up) / down


def seasonal_ice_edge_latitude(dt: datetime, lons: np.ndarray) -> np.ndarray:
    """Latitude of the nominal 15% ice edge for each longitude (degrees, <0)."""
    phase = 0.5 * (1.0 - np.cos(np.pi * _ramp(dt.timetuple().tm_yday)))
    # Summer edge ~ -65.5 S, winter edge ~ -57.5 S in the 0-100 E sector.
    base = -65.8 + 8.3 * phase
    # Longitudinal structure: the Weddell/Ross sectors bulge north; in the
    # 0-100 E sector the edge is comparatively zonal with a ~1.5 deg wave.
    wave = 1.5 * np.sin(np.radians(lons * 2.0 + 30.0)) + 0.8 * np.sin(np.radians(lons * 5.0))
    return base + wave


@dataclass
class DemoFieldGenerator:
    """Deterministic, seeded generator of demo environmental fields."""

    grid: GridSpec
    seed: int = 20260101

    def __post_init__(self) -> None:
        self._land = land_fraction(self.grid)
        self._lat2d, self._lon2d = self.grid.meshgrid()

    # -- helpers ---------------------------------------------------------
    def _rng(self, dt: datetime, salt: int = 0) -> np.random.Generator:
        """Reproducible per-day RNG so repeated runs give identical data."""
        day = int(dt.replace(tzinfo=timezone.utc).timestamp() // 86400)
        return np.random.default_rng(self.seed + salt * 7919 + day)

    def _correlated_noise(self, dt: datetime, salt: int, scale: float, passes: int = 4) -> np.ndarray:
        """Spatially smooth noise field, temporally correlated day to day."""
        prev = self._rng(dt - timedelta(days=1), salt).standard_normal(self.grid.shape)
        cur = self._rng(dt, salt).standard_normal(self.grid.shape)
        combined = 0.7 * smooth_field(prev, passes) + 0.7 * smooth_field(cur, passes)
        std = combined.std() or 1.0
        return combined / std * scale

    # -- sea ice ---------------------------------------------------------
    def sea_ice_concentration(self, dt: datetime) -> np.ndarray:
        """Synthetic sea-ice concentration field, fraction in [0, 1]."""
        edge = seasonal_ice_edge_latitude(dt, self.grid.lons)[None, :]
        # Transition width narrows in winter (sharper ice edge).
        phase = 0.5 * (1.0 - np.cos(np.pi * _ramp(dt.timetuple().tm_yday)))
        width = 2.4 - 0.9 * phase
        conc = 1.0 / (1.0 + np.exp((self._lat2d - edge) / width))
        # Interior pack is compact but not uniform.
        conc = np.clip(conc * (0.93 + 0.07 * np.cos(np.radians(self._lon2d * 3))), 0, 1)
        conc = np.clip(conc + self._correlated_noise(dt, 1, 0.06), 0.0, 1.0)
        conc[conc < 0.06] = 0.0
        # Coastal polynyas: persistent low-concentration bands next to land.
        polynya = np.exp(-((self._land - 0.35) ** 2) / 0.02) * 0.35
        conc = np.clip(conc - polynya * (self._land > 0.05), 0.0, 1.0)
        conc[self._land >= 0.5] = np.nan
        return conc

    # -- atmosphere ------------------------------------------------------
    def weather(self, dt: datetime) -> dict[str, np.ndarray]:
        """Synthetic 10 m wind, 2 m temperature and MSL pressure."""
        rng = self._rng(dt, 2)
        # Three eastward-propagating synoptic lows in the circumpolar trough.
        msl = np.full(self.grid.shape, 990.0)
        t_days = (dt - datetime(2020, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0
        for k in range(3):
            lon0 = (25.0 + 120.0 * k + 9.0 * t_days) % 360.0
            if lon0 > 180:
                lon0 -= 360
            lat0 = -62.0 + 3.0 * np.sin(0.3 * t_days + k)
            depth = 25.0 + 12.0 * rng.standard_normal()
            dlon = np.minimum(np.abs(self._lon2d - lon0), 360 - np.abs(self._lon2d - lon0))
            r2 = (dlon / 12.0) ** 2 + ((self._lat2d - lat0) / 5.0) ** 2
            msl -= depth * np.exp(-r2)
        # Continental high over the plateau, ridge to the north.
        msl += 22.0 * np.exp(-((self._lat2d + 74.0) / 6.0) ** 2)
        msl += 14.0 * np.exp(-((self._lat2d + 48.0) / 8.0) ** 2)
        msl += self._correlated_noise(dt, 3, 1.5)

        # Geostrophic wind from the pressure gradient (southern hemisphere:
        # flow is clockwise around lows, so the sign of f is negative).
        dpdy, dpdx = np.gradient(msl * 100.0)  # Pa per grid step
        dy_m = self.grid.dlat * 111_320.0
        dx_m = self.grid.dlon * 111_320.0 * np.cos(np.radians(self._lat2d))
        f = 2 * 7.2921e-5 * np.sin(np.radians(self._lat2d))
        f = np.where(np.abs(f) < 2e-5, np.sign(f) * 2e-5, f)
        rho = 1.3
        ug = -(1.0 / (rho * f)) * (dpdy / dy_m)
        vg = (1.0 / (rho * f)) * (dpdx / np.maximum(dx_m, 1.0))
        # 10 m wind is ~0.7 of geostrophic with ~20 deg cross-isobar inflow.
        ang = np.radians(20.0)
        u10 = 0.7 * (ug * np.cos(ang) - vg * np.sin(ang))
        v10 = 0.7 * (ug * np.sin(ang) + vg * np.cos(ang))
        speed = np.hypot(u10, v10)
        too_fast = speed > 35.0
        scale = np.where(too_fast, 35.0 / np.maximum(speed, 1e-6), 1.0)
        u10, v10 = u10 * scale, v10 * scale

        phase = 0.5 * (1.0 - np.cos(np.pi * _ramp(dt.timetuple().tm_yday)))
        t2m = (
            2.0
            + 0.62 * (self._lat2d + 50.0) * 1.0
            - 14.0 * phase
            - 10.0 * self._land
            + self._correlated_noise(dt, 4, 1.2)
        )
        return {"u10": u10, "v10": v10, "t2m": t2m, "msl": msl}

    # -- ocean -----------------------------------------------------------
    def ocean(self, dt: datetime) -> dict[str, np.ndarray]:
        """Synthetic surface currents, SST, salinity and wave height."""
        lat = self._lat2d
        # Antarctic Circumpolar Current: eastward jet centred near 55-58 S.
        acc = 0.32 * np.exp(-((lat + 56.5) / 4.5) ** 2)
        # Antarctic Coastal (East Wind Drift) Current: westward near the coast.
        coastal = -0.16 * np.exp(-((lat + 68.5) / 2.6) ** 2)
        u = acc + coastal
        v = np.zeros_like(u)
        # Mesoscale eddy field (streamfunction -> non-divergent velocities).
        psi = self._correlated_noise(dt, 5, 1.0, passes=6) * 1.6e4
        dpsi_dy, dpsi_dx = np.gradient(psi)
        dy_m = self.grid.dlat * 111_320.0
        dx_m = self.grid.dlon * 111_320.0 * np.cos(np.radians(lat))
        u = u - dpsi_dy / dy_m
        v = v + dpsi_dx / np.maximum(dx_m, 1.0)
        u = np.clip(u, -0.9, 0.9)
        v = np.clip(v, -0.9, 0.9)

        phase = 0.5 * (1.0 - np.cos(np.pi * _ramp(dt.timetuple().tm_yday)))
        sst = np.clip(
            8.5 + 0.42 * (lat + 50.0) - 1.6 * phase + self._correlated_noise(dt, 6, 0.5),
            -1.9,
            14.0,
        )
        salinity = np.clip(34.55 - 0.02 * (lat + 60.0) + self._correlated_noise(dt, 7, 0.08), 33.0, 35.2)

        wx = self.weather(dt)
        wind = np.hypot(wx["u10"], wx["v10"])
        # Simple fetch/duration-limited significant wave height, damped by ice.
        hs = 0.022 * wind**2
        conc = np.nan_to_num(self.sea_ice_concentration(dt), nan=1.0)
        hs = np.clip(hs * (1.0 - 0.92 * conc), 0.1, 12.0)

        for arr in (u, v, sst, salinity, hs):
            arr[self._land >= 0.5] = np.nan
        return {
            "u_current": u,
            "v_current": v,
            "sst": sst,
            "salinity": salinity,
            "wave_height": hs,
        }


# ---------------------------------------------------------------------------
# Demo icebergs
# ---------------------------------------------------------------------------
#: Synthetic bergs.  IDs are deliberately prefixed ``DEMO-`` so they can never
#: be confused with real USNIC designators such as A76C or B09B.
# All seeds are verified to sit in open water on the NSIDC-derived land mask.
DEMO_ICEBERG_SEEDS = [
    {"iceberg_id": "DEMO-01", "latitude": -66.2, "longitude": 22.4, "length_nm": 18.0, "width_nm": 9.0},
    {"iceberg_id": "DEMO-02", "latitude": -64.8, "longitude": 41.0, "length_nm": 11.0, "width_nm": 6.0},
    {"iceberg_id": "DEMO-03", "latitude": -66.4, "longitude": 58.7, "length_nm": 26.0, "width_nm": 14.0},
    {"iceberg_id": "DEMO-04", "latitude": -65.1, "longitude": 70.9, "length_nm": 8.0, "width_nm": 4.0},
    {"iceberg_id": "DEMO-05", "latitude": -62.3, "longitude": 33.5, "length_nm": 14.0, "width_nm": 7.0},
    {"iceberg_id": "DEMO-06", "latitude": -68.1, "longitude": 76.8, "length_nm": 21.0, "width_nm": 11.0},
    {"iceberg_id": "DEMO-07", "latitude": -63.4, "longitude": 12.9, "length_nm": 9.0, "width_nm": 5.0},
    {"iceberg_id": "DEMO-08", "latitude": -65.4, "longitude": 88.2, "length_nm": 16.0, "width_nm": 8.0},
    {"iceberg_id": "DEMO-09", "latitude": -61.2, "longitude": 62.0, "length_nm": 7.0, "width_nm": 4.0},
    {"iceberg_id": "DEMO-10", "latitude": -66.5, "longitude": 48.3, "length_nm": 30.0, "width_nm": 16.0},
    {"iceberg_id": "DEMO-11", "latitude": -67.8, "longitude": 5.6, "length_nm": 12.0, "width_nm": 6.0},
    {"iceberg_id": "DEMO-12", "latitude": -64.0, "longitude": 95.1, "length_nm": 10.0, "width_nm": 5.0},
]

NM_TO_KM = 1.852


def generate_demo_iceberg_observations(
    end_date: datetime, history_days: int = 30, step_days: int = 3
) -> list[dict]:
    """Build a short synthetic observation history for the demo bergs.

    Positions are advanced with the same coastal/ACC current pattern used by the
    ocean generator plus a small stochastic component, so the derived drift
    velocities are self-consistent with the demo ocean field.
    """
    rows: list[dict] = []
    rng = np.random.default_rng(4242)
    for berg in DEMO_ICEBERG_SEEDS:
        lat = float(berg["latitude"])
        lon = float(berg["longitude"])
        # Walk backwards to the start of the history, then forward emitting rows.
        n_steps = max(1, history_days // step_days)
        track: list[tuple[datetime, float, float]] = []
        cur_lat, cur_lon = lat, lon
        for k in range(n_steps + 1):
            dt = end_date - timedelta(days=step_days * k)
            track.append((dt, cur_lat, cur_lon))
            # Reverse-advect: coastal current is westward, ACC eastward.
            u = 0.32 * np.exp(-((cur_lat + 56.5) / 4.5) ** 2) - 0.16 * np.exp(
                -((cur_lat + 68.5) / 2.6) ** 2
            )
            v = 0.02 * np.sin(np.radians(cur_lon * 3.0))
            dt_s = step_days * 86400.0
            cur_lon -= (u * 0.75 * dt_s) / (111_320.0 * max(np.cos(np.radians(cur_lat)), 0.1))
            cur_lat -= (v * 0.75 * dt_s) / 111_320.0
            cur_lat += rng.normal(0, 0.03)
            cur_lon += rng.normal(0, 0.06)

        length_km = float(berg["length_nm"]) * NM_TO_KM
        width_km = float(berg["width_nm"]) * NM_TO_KM
        for dt, blat, blon in reversed(track):
            rows.append(
                {
                    "iceberg_id": berg["iceberg_id"],
                    "observed_at": dt.replace(tzinfo=timezone.utc),
                    "latitude": round(float(blat), 4),
                    "longitude": round(float(blon), 4),
                    "length_nm": float(berg["length_nm"]),
                    "width_nm": float(berg["width_nm"]),
                    "area_km2": round(length_km * width_km, 2),
                    "last_update": dt.replace(tzinfo=timezone.utc),
                    "source": DEMO_SOURCE,
                    "data_mode": "demo",
                    "dataset_version": "demo-1.0",
                }
            )
    return rows


def demo_extent_index(end_date: datetime, days: int = 365) -> list[dict]:
    """Synthetic hemispheric extent index consistent with the demo ice fields."""
    rows = []
    for k in range(days):
        dt = (end_date - timedelta(days=k)).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
        phase = 0.5 * (1.0 - np.cos(np.pi * _ramp(dt.timetuple().tm_yday)))
        extent = 2.9 + 15.4 * phase
        rows.append(
            {
                "observed_at": dt,
                "hemisphere": "south",
                "extent_million_km2": round(float(extent), 3),
                "area_million_km2": round(float(extent * 0.83), 3),
                "source": DEMO_SOURCE,
                "data_mode": "demo",
                "dataset_version": "demo-1.0",
            }
        )
    return list(reversed(rows))
