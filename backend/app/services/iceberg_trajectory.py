"""Physics-based iceberg trajectory prediction.

Model
-----
The iceberg is treated as a rigid body driven by a momentum balance of the form
used in the standard iceberg-drift literature (e.g. Bigg et al. 1997; Wagner,
Dell & Eisenman 2017):

    M(1+Cm) dv/dt = -M f k x v + F_air + F_water + F_ice + F_pressure

* ``F_air``      quadratic drag on the sail (freeboard) area
* ``F_water``    quadratic drag on the keel area, relative to the ocean current
* ``F_ice``      drag from the surrounding sea-ice pack, engaged only in compact
                 ice (concentration above ~0.85) where the pack can transmit
                 stress to the berg
* ``F_pressure`` sea-surface-tilt / pressure-gradient force, expressed through
                 the geostrophic ocean current as ``M f k x u_water``
* ``Cm``         added-mass coefficient (0.5), the entrained water an
                 accelerating berg must also move

The system is integrated with a classical RK4 scheme.  Positional uncertainty
comes from a Monte-Carlo ensemble that perturbs the forcing fields and the drag
coefficients within their documented uncertainty; the reported radius is the
ensemble spread, not an assumed confidence interval.

Known limitations (documented, not hidden)
------------------------------------------
* Environmental fields are held at their analysis values for the whole forecast
  (persistence forcing).  With 24-72 h horizons this is the dominant error term.
* Iceberg thickness is estimated from waterline length; it is rarely observed.
* Deterioration (melt, calving, roll) is not modelled, so bergs keep their mass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np

from app.config import Settings, get_settings
from app.services import landmask
from app.utils.geo import (
    EARTH_RADIUS_M,
    OMEGA,
    GridSpec,
    bilinear_sample,
    coriolis_parameter,
    fill_nan_nearest,
    haversine_km,
    initial_bearing_deg,
    offset_meters,
)
from app.utils.logging import get_logger
from app.utils.validation import ValidationError, validate_coordinate

log = get_logger("services.iceberg_trajectory")

MODEL_NAME = "iceberg_dynamics"
MODEL_VERSION = "1.0.0"

# -- physical constants -----------------------------------------------------
RHO_ICE = 850.0        # kg/m3, bulk density of glacial ice
RHO_AIR = 1.225        # kg/m3
RHO_WATER = 1027.0     # kg/m3, Southern Ocean surface
RHO_SEA_ICE = 910.0    # kg/m3
NM_TO_M = 1852.0


@dataclass
class IcebergState:
    """Geometry and kinematics of one iceberg."""

    iceberg_id: str
    latitude: float
    longitude: float
    length_m: float
    width_m: float
    thickness_m: float
    u_m_s: float = 0.0
    v_m_s: float = 0.0
    grounded: bool = False

    @property
    def freeboard_m(self) -> float:
        """Height above the waterline from hydrostatic balance."""
        return self.thickness_m * (1.0 - RHO_ICE / RHO_WATER)

    @property
    def draft_m(self) -> float:
        return self.thickness_m - self.freeboard_m

    @property
    def mass_kg(self) -> float:
        return RHO_ICE * self.length_m * self.width_m * self.thickness_m

    @property
    def sail_area_m2(self) -> float:
        """Cross-sectional area exposed to the wind (mean of the two faces)."""
        return 0.5 * (self.length_m + self.width_m) * self.freeboard_m

    @property
    def keel_area_m2(self) -> float:
        return 0.5 * (self.length_m + self.width_m) * self.draft_m

    @property
    def plan_area_m2(self) -> float:
        return self.length_m * self.width_m

    @property
    def speed_m_s(self) -> float:
        return float(math.hypot(self.u_m_s, self.v_m_s))


@dataclass
class DriftParameters:
    """Drag coefficients and integration settings (all configurable)."""

    air_drag: float = 1.3          # Ca, bluff-body form drag on the sail
    water_drag: float = 0.9        # Cw, form drag on the keel
    ice_drag: float = 1.0          # Ci, form drag against the pack
    added_mass: float = 0.5        # Cm
    timestep_s: float = 1800.0
    #: Representative Antarctic pack-ice thickness (m), used for the area the
    #: pack presses against.
    sea_ice_thickness_m: float = 1.0
    #: Concentration above which the pack transmits stress to the berg.
    ice_lock_concentration: float = 0.85
    #: Concentration above which a berg is treated as beset and relaxed towards
    #: the pack velocity (see ``besetment_timescale_s``).
    ice_full_lock_concentration: float = 0.95
    #: e-folding time for that relaxation; set to 0 to disable the regime.
    besetment_timescale_s: float = 43200.0
    max_speed_m_s: float = 1.5     # sanity clamp on integrated velocity


@dataclass
class TrajectoryPoint:
    """One predicted position on a trajectory."""

    valid_at: datetime
    horizon_hours: float
    latitude: float
    longitude: float
    speed_m_s: float
    bearing_deg: float
    uncertainty_radius_km: float
    grounded: bool = False

    def to_dict(self) -> dict:
        return {
            "valid_at": self.valid_at,
            "horizon_hours": self.horizon_hours,
            "latitude": round(self.latitude, 5),
            "longitude": round(self.longitude, 5),
            "speed_m_s": round(self.speed_m_s, 4),
            "bearing_deg": round(self.bearing_deg, 2),
            "uncertainty_radius_km": round(self.uncertainty_radius_km, 3),
            "grounded": self.grounded,
        }


@dataclass
class TrajectoryResult:
    """Complete forecast for one iceberg."""

    iceberg_id: str
    issued_at: datetime
    origin: tuple[float, float]
    points: list[TrajectoryPoint]
    state: IcebergState
    ensemble_size: int
    data_mode: str
    forcing: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total_displacement_km(self) -> float:
        if not self.points:
            return 0.0
        last = self.points[-1]
        return float(haversine_km(self.origin[0], self.origin[1], last.latitude, last.longitude))

    def point_at(self, horizon_hours: float) -> TrajectoryPoint | None:
        for p in self.points:
            if abs(p.horizon_hours - horizon_hours) < 1e-6:
                return p
        return None

    def to_dict(self) -> dict:
        return {
            "iceberg_id": self.iceberg_id,
            "issued_at": self.issued_at,
            "origin": {"latitude": self.origin[0], "longitude": self.origin[1]},
            "points": [p.to_dict() for p in self.points],
            "total_displacement_km": round(self.total_displacement_km, 2),
            "ensemble_size": self.ensemble_size,
            "data_mode": self.data_mode,
            "forcing": self.forcing,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Geometry estimation
# ---------------------------------------------------------------------------
def estimate_thickness_m(length_m: float, width_m: float | None = None) -> float:
    """Estimate iceberg thickness from waterline length.

    Small and mid-size bergs follow the widely used empirical draft-length
    relation ``draft = 3.78 * L^0.63`` (L in metres).  Large tabular Antarctic
    bergs calve directly from ice shelves, so their thickness is set by the
    shelf rather than by their planform: the estimate is therefore capped at
    350 m, close to the thickness of the major East Antarctic shelves.  The
    floor of 30 m keeps growlers/bergy bits physically sensible.
    """
    length_m = max(float(length_m), 1.0)
    draft = 3.78 * (length_m**0.63)
    thickness = draft / (RHO_ICE / RHO_WATER)  # convert draft to full thickness
    return float(np.clip(thickness, 30.0, 350.0))


def state_from_observation(
    iceberg_id: str,
    latitude: float,
    longitude: float,
    length_nm: float | None = None,
    width_nm: float | None = None,
    area_km2: float | None = None,
    thickness_m: float | None = None,
) -> IcebergState:
    """Build an :class:`IcebergState` from a bulletin row.

    Missing dimensions are reconstructed from whichever fields are present; when
    nothing is available a small default berg (5 x 3 NM) is assumed and the
    caller is expected to surface that assumption.
    """
    lat, lon = validate_coordinate(latitude, longitude, iceberg_id)
    if length_nm and length_nm > 0:
        length_m = float(length_nm) * NM_TO_M
        width_m = float(width_nm) * NM_TO_M if width_nm and width_nm > 0 else length_m * 0.5
    elif area_km2 and area_km2 > 0:
        # Assume a 2:1 tabular planform when only the area is reported.
        width_m = math.sqrt(float(area_km2) * 1e6 / 2.0)
        length_m = 2.0 * width_m
    else:
        length_m, width_m = 5.0 * NM_TO_M, 3.0 * NM_TO_M
    return IcebergState(
        iceberg_id=iceberg_id,
        latitude=lat,
        longitude=lon,
        length_m=length_m,
        width_m=width_m,
        thickness_m=float(thickness_m) if thickness_m else estimate_thickness_m(length_m, width_m),
    )


# ---------------------------------------------------------------------------
# Force balance
# ---------------------------------------------------------------------------
def _acceleration(
    state: IcebergState,
    u: float,
    v: float,
    env: dict[str, float],
    params: DriftParameters,
) -> tuple[float, float]:
    """Acceleration (m/s^2) of the berg at velocity (u, v) under ``env``."""
    mass = state.mass_kg * (1.0 + params.added_mass)
    f = coriolis_parameter(state.latitude)

    ua = env.get("u10", 0.0) or 0.0
    va = env.get("v10", 0.0) or 0.0
    uw = env.get("u_current", 0.0) or 0.0
    vw = env.get("v_current", 0.0) or 0.0
    conc = env.get("sea_ice_concentration", 0.0) or 0.0
    conc = float(np.clip(conc, 0.0, 1.0))

    # Air drag on the sail.
    dua, dva = ua - u, va - v
    wind_rel = math.hypot(dua, dva)
    ca = 0.5 * RHO_AIR * params.air_drag * state.sail_area_m2 * wind_rel
    fax, fay = ca * dua, ca * dva

    # Water drag on the keel.
    duw, dvw = uw - u, vw - v
    water_rel = math.hypot(duw, dvw)
    cw = 0.5 * RHO_WATER * params.water_drag * state.keel_area_m2 * water_rel
    fwx, fwy = cw * duw, cw * dvw

    # Sea-ice drag.  Only a compact pack transmits meaningful stress, and it
    # acts on the berg's *side* area presented to the ice (perimeter half-span
    # times pack thickness), not on its plan area.  The pack itself is taken to
    # be in free drift: the surface current plus 2% of the wind.
    fix = fiy = 0.0
    ui, vi = _pack_velocity(ua, va, uw, vw)
    if conc > params.ice_lock_concentration:
        engagement = (conc - params.ice_lock_concentration) / max(
            1e-6, 1.0 - params.ice_lock_concentration
        )
        dui, dvi = ui - u, vi - v
        ice_rel = math.hypot(dui, dvi)
        ice_area = 0.5 * (state.length_m + state.width_m) * params.sea_ice_thickness_m
        ci = 0.5 * RHO_SEA_ICE * params.ice_drag * ice_area * ice_rel * engagement
        fix, fiy = ci * dui, ci * dvi

    # Coriolis on the berg, and the sea-surface tilt expressed through the
    # geostrophic ocean current.
    fcx, fcy = state.mass_kg * f * v, -state.mass_kg * f * u
    fpx, fpy = -state.mass_kg * f * vw, state.mass_kg * f * uw

    ax = (fax + fwx + fix + fcx + fpx) / mass
    ay = (fay + fwy + fiy + fcy + fpy) / mass
    return ax, ay


def _rk4_step(
    state: IcebergState, env: dict[str, float], params: DriftParameters
) -> tuple[float, float, float, float]:
    """One RK4 step; returns ``(u_new, v_new, dx_m, dy_m)``."""
    dt = params.timestep_s
    u0, v0 = state.u_m_s, state.v_m_s

    k1u, k1v = _acceleration(state, u0, v0, env, params)
    k2u, k2v = _acceleration(state, u0 + 0.5 * dt * k1u, v0 + 0.5 * dt * k1v, env, params)
    k3u, k3v = _acceleration(state, u0 + 0.5 * dt * k2u, v0 + 0.5 * dt * k2v, env, params)
    k4u, k4v = _acceleration(state, u0 + dt * k3u, v0 + dt * k3v, env, params)

    du = (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
    dv = (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    u_new = float(np.clip(u0 + du, -params.max_speed_m_s, params.max_speed_m_s))
    v_new = float(np.clip(v0 + dv, -params.max_speed_m_s, params.max_speed_m_s))

    # Trapezoidal displacement over the step.
    dx = 0.5 * (u0 + u_new) * dt
    dy = 0.5 * (v0 + v_new) * dt
    return u_new, v_new, dx, dy


def _pack_velocity(ua: float, va: float, uw: float, vw: float) -> tuple[float, float]:
    """Free-drift velocity of the surrounding pack: current + 2% of the wind."""
    return uw + 0.02 * ua, vw + 0.02 * va


ENV_KEYS = ("u10", "v10", "u_current", "v_current", "sea_ice_concentration")


def prepare_forcing(grid: GridSpec, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gap-fill the forcing fields **once** so integration is a pure lookup.

    Filling inside the integration loop would repeat the same diffusion fill
    tens of thousands of times per ensemble; hoisting it out is the single
    biggest cost saving in this module.
    """
    prepared: dict[str, np.ndarray] = {}
    for key in ENV_KEYS:
        arr = fields.get(key)
        if arr is None:
            prepared[key] = np.zeros(grid.shape)
            continue
        filled = fill_nan_nearest(np.asarray(arr, dtype=float), max_iter=10)
        prepared[key] = np.nan_to_num(filled, nan=0.0)
    return prepared


def _sample(grid: GridSpec, prepared: dict[str, np.ndarray], lat: float, lon: float) -> dict[str, float]:
    if not grid.contains(lat, lon):
        return {k: 0.0 for k in ENV_KEYS}
    return {k: bilinear_sample(grid, prepared[k], lat, lon) for k in ENV_KEYS}


def _integrate(
    state: IcebergState,
    grid: GridSpec,
    fields: dict[str, np.ndarray],
    params: DriftParameters,
    horizon_hours: float,
    output_every_hours: float,
    perturbation: dict[str, float] | None = None,
) -> list[tuple[float, float, float, float, float, bool]]:
    """Integrate one member; returns ``(hours, lat, lon, u, v, grounded)`` samples."""
    work = IcebergState(**{**state.__dict__})
    ocean = landmask.navigable_mask(grid)
    prepared = fields if _is_prepared(fields) else prepare_forcing(grid, fields)
    n_steps = max(1, int(round(horizon_hours * 3600.0 / params.timestep_s)))
    out_every = max(1, int(round(output_every_hours * 3600.0 / params.timestep_s)))
    samples: list[tuple[float, float, float, float, float, bool]] = [
        (0.0, work.latitude, work.longitude, work.u_m_s, work.v_m_s, False)
    ]

    for step in range(1, n_steps + 1):
        env = _sample(grid, prepared, work.latitude, work.longitude)
        if perturbation:
            env["u10"] += perturbation["wind_u"]
            env["v10"] += perturbation["wind_v"]
            env["u_current"] += perturbation["cur_u"]
            env["v_current"] += perturbation["cur_v"]

        member_params = params
        if perturbation:
            member_params = DriftParameters(
                air_drag=params.air_drag * perturbation["air_scale"],
                water_drag=params.water_drag * perturbation["water_scale"],
                ice_drag=params.ice_drag,
                added_mass=params.added_mass,
                timestep_s=params.timestep_s,
                ice_lock_concentration=params.ice_lock_concentration,
                ice_full_lock_concentration=params.ice_full_lock_concentration,
                max_speed_m_s=params.max_speed_m_s,
            )

        u_new, v_new, dx, dy = _rk4_step(work, env, member_params)

        # Besetment: in a very compact pack the berg is carried with the ice
        # rather than drifting through it, so its velocity is relaxed towards
        # the pack velocity on a documented timescale.
        conc = float(np.clip(env.get("sea_ice_concentration", 0.0), 0.0, 1.0))
        tau = member_params.besetment_timescale_s
        if tau > 0 and conc >= member_params.ice_full_lock_concentration:
            ui, vi = _pack_velocity(env["u10"], env["v10"], env["u_current"], env["v_current"])
            alpha = 1.0 - math.exp(-member_params.timestep_s / tau)
            u_new += alpha * (ui - u_new)
            v_new += alpha * (vi - v_new)
        new_lat, new_lon = offset_meters(work.latitude, work.longitude, dx, dy)

        blocked = False
        if grid.contains(new_lat, new_lon):
            i, j = grid.index_of(new_lat, new_lon)
            blocked = not ocean[i, j]
        if blocked:
            # Grounding: the berg stops at the coast rather than crossing it.
            # It stays put for the rest of the forecast, and every remaining
            # output step reports that same position so the control run and the
            # ensemble members stay index-aligned.
            work.grounded = True
            work.u_m_s = work.v_m_s = 0.0
        else:
            work.latitude, work.longitude = new_lat, new_lon
            work.u_m_s, work.v_m_s = u_new, v_new

        if step % out_every == 0 or step == n_steps:
            samples.append(
                (
                    step * params.timestep_s / 3600.0,
                    work.latitude,
                    work.longitude,
                    work.u_m_s,
                    work.v_m_s,
                    work.grounded,
                )
            )
    return samples


def _sample_many(
    grid: GridSpec, prepared: dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray
) -> dict[str, np.ndarray]:
    """Bilinearly sample the forcing at many positions at once.

    Vectorised counterpart of :func:`_sample`: the ensemble integrates all
    members simultaneously, so the per-step cost stops scaling with member
    count.  Positions outside the domain sample as zero, matching ``_sample``.
    """
    lats, lons = grid.lats, grid.lons
    fi = (lat - grid.lat_min) / grid.dlat
    fj = (lon - grid.lon_min) / grid.dlon
    inside = (
        (lat >= grid.lat_min - grid.dlat / 2) & (lat <= grid.lat_max + grid.dlat / 2)
        & (lon >= grid.lon_min - grid.dlon / 2) & (lon <= grid.lon_max + grid.dlon / 2)
    )
    i0 = np.clip(np.floor(fi), 0, len(lats) - 1).astype(int)
    j0 = np.clip(np.floor(fj), 0, len(lons) - 1).astype(int)
    i1 = np.minimum(i0 + 1, len(lats) - 1)
    j1 = np.minimum(j0 + 1, len(lons) - 1)
    ti = np.clip(fi - i0, 0.0, 1.0)
    tj = np.clip(fj - j0, 0.0, 1.0)

    out: dict[str, np.ndarray] = {}
    for key in ENV_KEYS:
        f = prepared[key]
        v = (
            f[i0, j0] * (1 - ti) * (1 - tj)
            + f[i1, j0] * ti * (1 - tj)
            + f[i0, j1] * (1 - ti) * tj
            + f[i1, j1] * ti * tj
        )
        out[key] = np.where(inside, v, 0.0)
    return out


def _acceleration_many(
    state: IcebergState,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    env: dict[str, np.ndarray],
    air_drag: np.ndarray,
    water_drag: np.ndarray,
    params: DriftParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised form of :func:`_acceleration` over an ensemble of members."""
    mass = state.mass_kg * (1.0 + params.added_mass)
    f = 2.0 * OMEGA * np.sin(np.radians(lat))

    ua, va = env["u10"], env["v10"]
    uw, vw = env["u_current"], env["v_current"]
    conc = np.clip(env["sea_ice_concentration"], 0.0, 1.0)

    dua, dva = ua - u, va - v
    ca = 0.5 * RHO_AIR * air_drag * state.sail_area_m2 * np.hypot(dua, dva)
    fax, fay = ca * dua, ca * dva

    duw, dvw = uw - u, vw - v
    cw = 0.5 * RHO_WATER * water_drag * state.keel_area_m2 * np.hypot(duw, dvw)
    fwx, fwy = cw * duw, cw * dvw

    ui, vi = uw + 0.02 * ua, vw + 0.02 * va
    engaged = conc > params.ice_lock_concentration
    engagement = np.where(
        engaged,
        (conc - params.ice_lock_concentration) / max(1e-6, 1.0 - params.ice_lock_concentration),
        0.0,
    )
    dui, dvi = ui - u, vi - v
    ice_area = 0.5 * (state.length_m + state.width_m) * params.sea_ice_thickness_m
    ci = 0.5 * RHO_SEA_ICE * params.ice_drag * ice_area * np.hypot(dui, dvi) * engagement
    fix, fiy = ci * dui, ci * dvi

    fcx, fcy = state.mass_kg * f * v, -state.mass_kg * f * u
    fpx, fpy = -state.mass_kg * f * vw, state.mass_kg * f * uw

    return (fax + fwx + fix + fcx + fpx) / mass, (fay + fwy + fiy + fcy + fpy) / mass


def _integrate_ensemble(
    state: IcebergState,
    grid: GridSpec,
    prepared: dict[str, np.ndarray],
    params: DriftParameters,
    horizon_hours: float,
    output_every_hours: float,
    perturbations: Sequence[dict[str, float]],
) -> list[np.ndarray]:
    """Integrate the control run and every ensemble member together.

    Member 0 is the unperturbed control.  Returns one array per output step,
    shaped ``(n_members + 1, 5)`` holding lat, lon, u, v and a grounded flag.
    """
    n = len(perturbations) + 1
    lat = np.full(n, state.latitude, dtype=float)
    lon = np.full(n, state.longitude, dtype=float)
    u = np.full(n, state.u_m_s, dtype=float)
    v = np.full(n, state.v_m_s, dtype=float)
    grounded = np.zeros(n, dtype=bool)

    zeros = np.zeros(n)
    wind_u = np.array([0.0] + [p["wind_u"] for p in perturbations])
    wind_v = np.array([0.0] + [p["wind_v"] for p in perturbations])
    cur_u = np.array([0.0] + [p["cur_u"] for p in perturbations])
    cur_v = np.array([0.0] + [p["cur_v"] for p in perturbations])
    air_drag = params.air_drag * np.array([1.0] + [p["air_scale"] for p in perturbations])
    water_drag = params.water_drag * np.array([1.0] + [p["water_scale"] for p in perturbations])

    ocean = landmask.navigable_mask(grid)
    dt = params.timestep_s
    n_steps = max(1, int(round(horizon_hours * 3600.0 / dt)))
    out_every = max(1, int(round(output_every_hours * 3600.0 / dt)))
    samples: list[np.ndarray] = []

    def forcing(la: np.ndarray, lo: np.ndarray) -> dict[str, np.ndarray]:
        env = _sample_many(grid, prepared, la, lo)
        env["u10"] = env["u10"] + wind_u
        env["v10"] = env["v10"] + wind_v
        env["u_current"] = env["u_current"] + cur_u
        env["v_current"] = env["v_current"] + cur_v
        return env

    for step in range(1, n_steps + 1):
        env = forcing(lat, lon)
        acc = lambda uu, vv: _acceleration_many(  # noqa: E731 - local shorthand
            state, lat, uu, vv, env, air_drag, water_drag, params
        )
        k1u, k1v = acc(u, v)
        k2u, k2v = acc(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v)
        k3u, k3v = acc(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v)
        k4u, k4v = acc(u + dt * k3u, v + dt * k3v)
        du = (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        dv = (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        u_new = np.clip(u + du, -params.max_speed_m_s, params.max_speed_m_s)
        v_new = np.clip(v + dv, -params.max_speed_m_s, params.max_speed_m_s)

        conc = np.clip(env["sea_ice_concentration"], 0.0, 1.0)
        tau = params.besetment_timescale_s
        if tau > 0:
            beset = conc >= params.ice_full_lock_concentration
            alpha = (1.0 - math.exp(-dt / tau)) * beset
            ui = env["u_current"] + 0.02 * env["u10"]
            vi = env["v_current"] + 0.02 * env["v10"]
            u_new = u_new + alpha * (ui - u_new)
            v_new = v_new + alpha * (vi - v_new)

        dx = 0.5 * (u + u_new) * dt
        dy = 0.5 * (v + v_new) * dt
        new_lat = np.clip(lat + np.degrees(dy / EARTH_RADIUS_M), -89.9, 89.9)
        coslat = np.maximum(np.cos(np.radians(lat)), 1e-6)
        new_lon = lon + np.degrees(dx / (EARTH_RADIUS_M * coslat))
        new_lon = ((new_lon + 180.0) % 360.0) - 180.0

        i, j = _grid_index_many(grid, new_lat, new_lon)
        blocked = ~ocean[i, j] | grounded
        grounded = grounded | blocked
        lat = np.where(blocked, lat, new_lat)
        lon = np.where(blocked, lon, new_lon)
        u = np.where(blocked, 0.0, u_new)
        v = np.where(blocked, 0.0, v_new)

        if step % out_every == 0 or step == n_steps:
            samples.append(np.column_stack([lat, lon, u, v, grounded.astype(float)]))
    return samples


def _grid_index_many(grid: GridSpec, lat: np.ndarray, lon: np.ndarray):
    # A degenerate berg (zero mass) produces NaN accelerations; casting NaN to
    # int is undefined, so those members are pinned to cell 0 and will be
    # rejected by the caller rather than silently indexing at random.
    lat = np.nan_to_num(np.asarray(lat, float), nan=grid.lat_min)
    lon = np.nan_to_num(np.asarray(lon, float), nan=grid.lon_min)
    i = np.clip(np.round((lat - grid.lat_min) / grid.dlat), 0, grid.shape[0] - 1).astype(int)
    j = np.clip(np.round((lon - grid.lon_min) / grid.dlon), 0, grid.shape[1] - 1).astype(int)
    return i, j


def _is_prepared(fields: dict[str, np.ndarray]) -> bool:
    """True when ``fields`` already went through :func:`prepare_forcing`."""
    return all(k in fields for k in ENV_KEYS) and all(
        np.isfinite(fields[k]).all() for k in ENV_KEYS
    )


def _perturbations(n: int, seed: int = 11) -> list[dict[str, float]]:
    """Forcing perturbations for the Monte-Carlo ensemble.

    Scales reflect typical analysis errors: ~1.5 m/s on 10 m wind and ~0.05 m/s
    on surface currents, plus 15% uncertainty on each drag coefficient.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(
            {
                "wind_u": float(rng.normal(0.0, 1.5)),
                "wind_v": float(rng.normal(0.0, 1.5)),
                "cur_u": float(rng.normal(0.0, 0.05)),
                "cur_v": float(rng.normal(0.0, 0.05)),
                "air_scale": float(np.clip(rng.normal(1.0, 0.15), 0.5, 1.6)),
                "water_scale": float(np.clip(rng.normal(1.0, 0.15), 0.5, 1.6)),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_trajectory(
    state: IcebergState,
    grid: GridSpec,
    fields: dict[str, np.ndarray],
    issued_at: datetime,
    horizon_hours: float = 72.0,
    output_every_hours: float = 6.0,
    params: DriftParameters | None = None,
    ensemble_size: int | None = None,
    settings: Settings | None = None,
) -> TrajectoryResult:
    """Forecast one iceberg's drift with an uncertainty ensemble."""
    settings = settings or get_settings()
    params = params or DriftParameters(timestep_s=float(settings.iceberg_timestep_seconds))
    ensemble_size = settings.iceberg_ensemble_members if ensemble_size is None else ensemble_size
    if horizon_hours <= 0:
        raise ValidationError("horizon_hours must be positive")
    if state.mass_kg <= 0 or state.length_m <= 0 or state.width_m <= 0:
        raise ValidationError(
            f"Iceberg {state.iceberg_id} has non-physical dimensions "
            f"({state.length_m:.0f} x {state.width_m:.0f} x {state.thickness_m:.0f} m); "
            f"no trajectory can be integrated."
        )
    if not grid.contains(state.latitude, state.longitude):
        # Outside the analysis domain there is no wind, current or ice field to
        # integrate. Returning a stationary trajectory would be a confident
        # wrong answer, so refuse instead: many USNIC bergs sit in the Weddell
        # and Scotia seas, well outside a 0-100 E domain.
        raise ValidationError(
            f"Iceberg {state.iceberg_id} is at ({state.latitude:.2f}, {state.longitude:.2f}), "
            f"outside the POLARIS analysis domain (lat {grid.lat_min}..{grid.lat_max}, "
            f"lon {grid.lon_min}..{grid.lon_max}). No environmental forcing is available there, "
            f"so no trajectory can be computed. Widen the domain in the configuration to include it."
        )

    prepared = prepare_forcing(grid, fields) if not _is_prepared(fields) else fields
    if state.u_m_s == 0.0 and state.v_m_s == 0.0:
        # Spin-up matters: a berg started from rest needs many hours to reach
        # its balance velocity.  Without an observed drift vector, seed it with
        # the local water velocity, which is the leading-order balance.
        env0 = _sample(grid, prepared, state.latitude, state.longitude)
        state.u_m_s, state.v_m_s = env0["u_current"], env0["v_current"]

    perturbations = _perturbations(max(0, ensemble_size))
    samples = _integrate_ensemble(
        state, grid, prepared, params, horizon_hours, output_every_hours, perturbations
    )

    notes: list[str] = []
    env0 = _sample(grid, prepared, state.latitude, state.longitude)
    forcing = {k: round(float(v), 4) for k, v in env0.items()}
    forcing["initial_speed_m_s"] = round(float(math.hypot(state.u_m_s, state.v_m_s)), 4)
    if not np.isfinite(np.asarray(fields.get("u_current", np.nan), float)).any():
        notes.append("No ocean-current field available; drift is driven by wind and Coriolis only.")
    if not np.isfinite(np.asarray(fields.get("u10", np.nan), float)).any():
        notes.append("No wind field available; drift is driven by the ocean current only.")

    points: list[TrajectoryPoint] = []
    prev_lat, prev_lon = state.latitude, state.longitude
    for step in samples:
        # Row 0 is the control run; the remaining rows are ensemble members.
        lat, lon, u, v, grounded = step[0]
        hours = (len(points) + 1) * output_every_hours
        spread_km = 0.0
        if len(step) > 1:
            radii = haversine_km(lat, lon, step[1:, 0], step[1:, 1])
            spread_km = float(np.sqrt(np.mean(np.square(radii))))
        points.append(
            TrajectoryPoint(
                valid_at=issued_at + timedelta(hours=hours),
                horizon_hours=float(hours),
                latitude=float(lat),
                longitude=float(lon),
                speed_m_s=float(math.hypot(u, v)),
                bearing_deg=float(initial_bearing_deg(prev_lat, prev_lon, lat, lon)),
                uncertainty_radius_km=spread_km,
                grounded=bool(grounded),
            )
        )
        prev_lat, prev_lon = lat, lon

    if points and points[-1].grounded:
        notes.append("Iceberg grounds on the coast within the forecast window.")

    return TrajectoryResult(
        iceberg_id=state.iceberg_id,
        issued_at=issued_at,
        origin=(state.latitude, state.longitude),
        points=points,
        state=state,
        ensemble_size=len(perturbations),
        data_mode=settings.data_mode,
        forcing=forcing,
        notes=notes,
    )


def predict_many(
    states: Sequence[IcebergState],
    grid: GridSpec,
    fields: dict[str, np.ndarray],
    issued_at: datetime,
    horizon_hours: float = 72.0,
    output_every_hours: float = 6.0,
    settings: Settings | None = None,
    ensemble_size: int | None = None,
) -> dict[str, TrajectoryResult]:
    """Forecast a set of icebergs, skipping any that fail individually."""
    results: dict[str, TrajectoryResult] = {}
    fields = prepare_forcing(grid, fields)
    for state in states:
        try:
            results[state.iceberg_id] = predict_trajectory(
                state, grid, fields, issued_at, horizon_hours, output_every_hours,
                settings=settings, ensemble_size=ensemble_size,
            )
        except Exception as exc:  # noqa: BLE001 - one bad berg must not stop the batch
            log.error("Trajectory failed for %s: %s", state.iceberg_id, exc)
    return results


def trajectory_rows(result: TrajectoryResult) -> list[dict]:
    """Convert a trajectory into ``iceberg_trajectories`` rows."""
    return [
        {
            "iceberg_id": result.iceberg_id,
            "issued_at": result.issued_at,
            "valid_at": p.valid_at,
            "horizon_hours": p.horizon_hours,
            "latitude": p.latitude,
            "longitude": p.longitude,
            "speed_m_s": p.speed_m_s,
            "bearing_deg": p.bearing_deg,
            "uncertainty_radius_km": p.uncertainty_radius_km,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "data_mode": result.data_mode,
        }
        for p in result.points
    ]


def persist_trajectories(session, results: dict[str, TrajectoryResult]) -> int:
    from app.database import repositories as repo

    rows: list[dict] = []
    for result in results.values():
        rows.extend(trajectory_rows(result))
    n = repo.save_trajectory(session, rows)
    log.info("Persisted %d trajectory points for %d iceberg(s)", n, len(results))
    return n
