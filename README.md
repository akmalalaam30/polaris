# POLARIS

**AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory and Navigation Decision Support System**

| | |
|---|---|
| Hackathon | Smart India Hackathon 2026 |
| Problem statement | **SIH26059** |
| Organisation | Ministry of Earth Sciences (MoES) |
| Department | National Centre for Polar and Ocean Research (NCPOR) |
| Category / Theme | Software / Transportation & Logistics |
| Team | Brookies |

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [The problem being solved](#2-the-problem-being-solved)
3. [Architecture](#3-architecture)
4. [Folder structure](#4-folder-structure)
5. [Technologies](#5-technologies)
6. [Data sources](#6-data-sources)
7. [Dataset descriptions](#7-dataset-descriptions)
8. [Installation](#8-installation)
9. [Environment variables](#9-environment-variables)
10. [Database setup](#10-database-setup)
11. [Data ingestion](#11-data-ingestion)
12. [Model training](#12-model-training)
13. [Model evaluation](#13-model-evaluation)
14. [Running the backend](#14-running-the-backend)
15. [API endpoints](#15-api-endpoints)
16. [Demo mode](#16-demo-mode)
17. [Real-data mode](#17-real-data-mode)
18. [Running tests](#18-running-tests)
19. [End-to-end workflow](#19-end-to-end-workflow)
20. [Limitations](#20-limitations)
21. [Scientific assumptions](#21-scientific-assumptions)

---

## 1. Project overview

POLARIS is a backend decision-support system for Antarctic navigation. It
ingests real sea-ice, iceberg, atmospheric and ocean data, forecasts sea-ice
concentration with a trained machine-learning model, predicts iceberg drift with
a physics-based model, fuses both into a spatial navigation-risk grid, and
solves for optimal routes over that grid.

The **backend is the product**. A small browser client is included only to
demonstrate and verify it.

Everything the system reports is either measured, computed, or explicitly
labelled as an estimate:

* Forecast accuracy figures are **real held-out validation scores**, always
  reported against a persistence baseline — and where the learned model does not
  beat that baseline, POLARIS says so and ships the baseline instead (§13).
* Iceberg positional uncertainty is a **measured Monte-Carlo ensemble spread**,
  not an assumed confidence interval.
* Fuel and duration are **model estimates** from a documented speed model, and
  route comparisons are computed from those same numbers.
* Synthetic data is always labelled `data_mode="demo"` and
  `source="POLARIS-DEMO"`, and is never presented as observation.

### Verified status

Every claim below was executed and measured on this codebase.

| Capability | Status |
|---|---|
| Backend starts, Swagger served at `/docs` | Working |
| Database connects (SQLite default, PostgreSQL/PostGIS supported) | Working |
| Real-data ingestion (NSIDC, USNIC, ERA5, ocean/wave analysis) | Working against live services |
| Demo-data ingestion | Working |
| Preprocessing → NetCDF history archive | Working |
| Sea-ice forecast, 24/48/72 h, trained + saved + loaded | Working; trained on 371 days of real NSIDC data |
| Ice-dynamics + thermodynamic features (CMEMS drift, FDD) | Working; roughly tripled real-data skill (§13) |
| Model selection gated on measured skill vs persistence | Working (§13) |
| Iceberg trajectory prediction with uncertainty ensemble | Working |
| Risk engine and spatial risk grid | Working |
| Navigation graph, A* and Dijkstra | Working |
| Shortest / safest / POLARIS routes | Working |
| All REST endpoints, validation and error handling | Working |
| Automated tests | **204 passing** (`pytest`, ~77 s) |
| Same suite on PostgreSQL 16.4 + PostGIS 3.4 | **verified** — the Postgres path is tested, not assumed |
| Cold-start latency | Eliminated by a background warm-up: first click 0.86 s total across all endpoints, was 69 s |
| End-to-end integration test | Working (`tests/test_e2e.py`) |
| Browser client | Verified in Chromium, no console errors |

---

## 2. The problem being solved

Research and resupply vessels operating to Antarctic stations must transit a sea
whose hazards change daily: the sea-ice edge advances and retreats, giant
tabular icebergs drift for years across shipping tracks, and Southern Ocean
depressions bring storm-force winds and freezing spray. Route choices are
consequential — besetment costs weeks, and there is no salvage nearby.

The information needed to make those choices exists, but it is scattered across
providers, in different projections, resolutions and formats, and none of it is
expressed as *what this means for my ship on this passage*.

POLARIS closes that gap:

```
scattered observations  →  common grid  →  forecast + drift prediction
                        →  a single risk field for a specified vessel
                        →  ranked route options with their trade-offs stated
```

The demonstration case is the passage between the two Indian Antarctic stations,
**Bharati** (Larsemann Hills, Prydz Bay) and **Maitri** (Schirmacher Oasis,
Queen Maud Land) — about 2 700 km apart along the East Antarctic coast.

---

## 3. Architecture

```
                    ┌──────────────────────────────────────────────┐
   EXTERNAL DATA    │ NSIDC G02135 · USNIC · ERA5 · CMEMS          │
                    └───────────────────────┬──────────────────────┘
                                            │
                    ┌───────────────────────▼──────────────────────┐
   INGESTION        │ services/data_ingestion.py                   │
                    │ download · retry · decode · regrid · log     │
                    └───────────────────────┬──────────────────────┘
                                            │
                    ┌───────────────────────▼──────────────────────┐
   VALIDATION       │ utils/validation.py                          │
                    │ coordinates · timestamps · ranges · dedupe   │
                    └───────────────────────┬──────────────────────┘
                                            │
                    ┌───────────────────────▼──────────────────────┐
   PREPROCESSING    │ services/preprocessing.py                    │
                    │ gap fill · land mask · track kinematics ·    │
                    │ NetCDF history archive · feature matrix      │
                    └──────────┬──────────────────────┬────────────┘
                               │                      │
          ┌────────────────────▼─────┐   ┌────────────▼──────────────────┐
   AI     │ sea_ice_forecasting.py   │   │ iceberg_trajectory.py         │
          │ HistGradientBoosting     │   │ RK4 momentum balance +        │
          │ 24 / 48 / 72 h + sigma   │   │ Monte-Carlo ensemble          │
          └────────────────────┬─────┘   └────────────┬──────────────────┘
                               │                      │
                    ┌──────────▼──────────────────────▼────────────┐
   RISK             │ services/risk_engine.py                      │
                    │ sea-ice · iceberg · weather · ocean ·        │
                    │ constraint → normalised 0–1 grid             │
                    └───────────────────────┬──────────────────────┘
                                            │
                    ┌───────────────────────▼──────────────────────┐
   ROUTING          │ services/route_optimizer.py                  │
                    │ NetworkX graph · A* / Dijkstra ·             │
                    │ shortest | safest | POLARIS                  │
                    └───────────────────────┬──────────────────────┘
                                            │
        ┌───────────────────────────────────┼──────────────────────┐
        ▼                                   ▼                      ▼
 ┌─────────────┐                   ┌────────────────┐    ┌──────────────────┐
 │ PostgreSQL/ │◄──────────────────┤  FastAPI       ├───►│ Browser test     │
 │ SQLite      │  repositories.py  │  /api/* /docs  │    │ client  (/ui)    │
 └─────────────┘                   └────────────────┘    └──────────────────┘
```

### Design decisions worth knowing

**A single analysis grid.** Every layer is resampled onto one regular lat/lon
grid (default 0.5° × 1.0° over 75°S–50°S, 0°–100°E: 51 × 101 = 5 151 cells).
Forecasting, risk and routing all operate on it, so the layers can never
disagree about geometry.

**The land mask is real.** It is derived from the land and coast flags published
inside the NSIDC G02135 GeoTIFFs, not from an invented coastline. It is bundled
as a 1.7 kB packed array so demo mode and the tests have the true Antarctic
coastline offline. Routes are verified never to cross it.

**Raster archive on disk, derived products in the database.** The full gridded
history lives as NetCDF under `data/processed/` (what a model trains on); the
database holds the operational state — recent observations, forecasts,
trajectories, risk grids and routes. Storing 400 days × 5 151 cells as rows
would be 2 M rows of no operational value.

**One cache, keyed on the data.** `services/environment.py` caches the derived
products against the timestamps of the observations that produced them, so a
cache entry is invalidated the moment new data is ingested and two endpoints
answering the same question always give the same number.

**Degradation is visible, never silent.** If a risk layer has no data it is
named in `missing_layers` and the total is renormalised over the layers that
were available. If no route satisfies the navigability constraints, POLARIS
retries on a relaxed graph and marks the result `degraded` with the reason
attached, rather than returning a compliant-looking route.

---

## 4. Folder structure

```
POLARIS/
│
├── backend/
│   ├── app/
│   │   ├── main.py                     FastAPI app, error handlers, /ui mount
│   │   ├── config.py                   pydantic-settings; all tunables
│   │   │
│   │   ├── api/
│   │   │   ├── deps.py                 shared dependencies and helpers
│   │   │   ├── routes_health.py        /health, /health/live, /health/ready, /system
│   │   │   ├── routes_sea_ice.py       /sea-ice/current, /forecast, /extent
│   │   │   ├── routes_icebergs.py      /icebergs, /{id}/trajectory, /trajectories
│   │   │   ├── routes_risk.py          /risk/map, /risk/point
│   │   │   ├── routes_navigation.py    /route/optimize, /navigation/*
│   │   │   └── routes_dashboard.py     /dashboard/summary, /models, /ingestion
│   │   │
│   │   ├── models/
│   │   │   ├── database_models.py      11 SQLAlchemy tables
│   │   │   └── ml_models.py            forecaster classes, metrics, uncertainty
│   │   │
│   │   ├── schemas/
│   │   │   ├── common.py               provenance, grid, health, errors
│   │   │   ├── sea_ice.py  iceberg.py  risk.py  route.py
│   │   │
│   │   ├── services/
│   │   │   ├── data_ingestion.py       NSIDC / USNIC / ERA5 / CMEMS clients
│   │   │   ├── demo_data.py            synthetic field generator (labelled)
│   │   │   ├── landmask.py             NSIDC-derived land mask
│   │   │   ├── preprocessing.py        archives, kinematics, feature matrix
│   │   │   ├── sea_ice_forecasting.py  train / validate / save / load / predict
│   │   │   ├── iceberg_trajectory.py   RK4 drift physics + ensemble
│   │   │   ├── risk_engine.py          five risk components + grid assembly
│   │   │   ├── route_optimizer.py      navigation graph, A*/Dijkstra, profiles
│   │   │   ├── environment.py          data-keyed cache shared by all endpoints
│   │   │   └── dashboard_service.py    aggregation for /dashboard/summary
│   │   │
│   │   ├── database/
│   │   │   ├── connection.py           engine, sessions, health check
│   │   │   ├── repositories.py         all SQL; idempotent upserts
│   │   │   └── init_db.py              schema creation, PostGIS upgrade
│   │   │
│   │   └── utils/
│   │       ├── geo.py                  geodesy, EPSG:3412, GridSpec, regridding
│   │       ├── logging.py              logging with credential redaction
│   │       └── validation.py           input and dataset validation
│   │
│   ├── tests/                          185 tests
│   │   ├── conftest.py                 reduced-domain, demo-mode fixtures
│   │   ├── test_health.py              health, schema, OpenAPI, log redaction
│   │   ├── test_data.py                ingestion, validation, preprocessing, repos
│   │   ├── test_sea_ice.py             metrics, training, artifacts, inference
│   │   ├── test_icebergs.py            force balance, integration, endpoints
│   │   ├── test_risk.py                every component + grid + endpoints
│   │   ├── test_routes.py              graph, A*/Dijkstra, profiles, endpoints
│   │   └── test_e2e.py                 full pipeline integration
│   │
│   ├── scripts/
│   │   ├── download_data.py            step 1: ingest
│   │   ├── preprocess_data.py          step 2: build the history archives
│   │   ├── train_models.py             step 3: train and validate
│   │   └── seed_demo_data.py           all steps, demo mode, one command
│   │
│   ├── requirements.txt
│   ├── .env.example
│   ├── pytest.ini
│   └── run.py
│
├── frontend/
│   ├── src/  index.html · app.js · styles.css · vendor/ (Leaflet, offline)
│   ├── package.json                    no build step, no dependencies
│   └── README.md
│
├── data/
│   ├── raw/                            provider downloads (cached, gitignored)
│   ├── processed/                      NetCDF history archives (gitignored)
│   └── demo/nsidc_land_mask.npz        bundled NSIDC-derived land mask (1.7 kB)
│
├── models/{demo,real}/                 model cards (tracked); .joblib binaries
│                                       are gitignored - retrain in ~5 min
├── notebooks/
├── README.md
└── .gitignore
```

---

## 5. Technologies

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Pydantic v2 | typed request/response schemas, automatic OpenAPI |
| Server | Uvicorn | ASGI |
| ORM | SQLAlchemy 2.0 | same code on SQLite and PostgreSQL |
| Database | SQLite (default) / PostgreSQL + PostGIS | zero-setup default, production target supported |
| Numerics | NumPy, SciPy, pandas | fields, interpolation, tabular data |
| ML | scikit-learn `HistGradientBoostingRegressor` | strong on tabular spatiotemporal features, trains in seconds, no GPU |
| Gridded I/O | xarray + netCDF4 | the standard format for `(time, lat, lon)` archives |
| Projections | pyproj | NSIDC polar stereographic (EPSG:3412) ↔ WGS-84 |
| Raster | Pillow | reads the NSIDC GeoTIFFs without a GDAL dependency |
| Geometry | Shapely | geometric predicates |
| Graph | NetworkX | navigation graph, A* and Dijkstra |
| HTTP | requests / httpx | provider downloads |
| Testing | pytest | 185 tests |
| Frontend | Leaflet (vendored) + vanilla JS | no build step to fail on demo day |

### Deliberate omissions

**PyTorch / deep learning.** A ConvLSTM was considered and rejected. With ~1 year
of daily fields the training set is a few hundred samples per cell — far too few
to fit a spatiotemporal network without overfitting, and it would have had to be
compared against the same persistence baseline anyway. Gradient boosting over
engineered lag/spatial/seasonal features *does* beat that baseline, measurably
(§13), which a placeholder network would not have. The model interface
(`app/models/ml_models.py`) is designed so a PyTorch forecaster can be dropped in
behind the same `fit`/`predict`/`save`/`load` contract.

**GDAL / rasterio / GeoPandas.** The only raster POLARIS reads is the NSIDC
GeoTIFF on a fixed, documented grid, so Pillow plus pyproj covers it without a
notoriously heavy install.

---

## 6. Data sources

| # | Source | Product | Access | Used for |
|---|---|---|---|---|
| 1 | **NSIDC Sea Ice Index v4** ([G02135](https://nsidc.org/data/g02135/versions/4)) | Daily Antarctic sea-ice concentration GeoTIFF, 25 km polar stereographic (EPSG:3412) | Public, no credentials | Observed concentration, model training, **land mask** |
| 2 | **NSIDC Sea Ice Index v4** | `S_seaice_extent_daily_v4.0.csv` | Public | Hemispheric extent/area time series |
| 3 | **US National Ice Center** ([Antarctic Icebergs](https://usicecenter.gov/Products/AntarcIcebergs)) | Iceberg bulletin CSV: designator, position, dimensions, last update | Public | Tracked iceberg positions, drift derivation |
| 4 | **ERA5** ([Copernicus CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)) | 10 m wind, 2 m temperature, MSL pressure | Free account (`cdsapi`) — **verified working** | Model **training history** (see below) |
| 4b | **ECMWF IFS forecast** | Live 10 m wind, 2 m temperature, MSL pressure | Public, no credentials — **verified working** | **Operational** weather layer, weather risk, station conditions |
| 5 | **Copernicus Marine** ([CMEMS](https://data.marine.copernicus.eu/)) | Currents, temperature, salinity, **sea-ice drift & thickness**, significant wave height | Free account (`copernicusmarine`) — **verified working** | Ocean risk, iceberg current forcing, wave risk |
| 6 | **NSIDC Polar Pathfinder** ([nsidc-0116](https://nsidc.org/data/nsidc-0116/versions/4)) | 25 km sea-ice motion vectors | Earthdata login | *Not integrated* — see [Limitations](#20-limitations) |

### Reanalysis is not a forecast

ERA5 is a **reanalysis**: assimilated after the fact and published with roughly
five days of latency. That makes it the right product for training a model on a
year of consistent history, and the wrong one for an operational display — a
navigation system showing six-day-old wind is describing the past.

POLARIS therefore splits the two:

| Purpose | Product | Latency |
|---|---|---|
| Training history (371 days of covariates) | ERA5 reanalysis | ~5 days |
| Operational conditions, weather risk, station readings | ECMWF IFS forecast | **valid now** |

Measured on the running system, the layer ages are:

```
weather    live      (ECMWF IFS, valid within the hour)
ocean      13 h      (CMEMS analysis-forecast, runs ahead of today)
icebergs   37 h      (USNIC bulletin, issued daily)
sea ice    2.5 days  (NSIDC G02135 publication latency)
```

`GET /api/weather/current` reports exactly this, so the freshness of each layer
is visible rather than implied. Before this split the weather layer was 156
hours old.

### CMEMS is five datasets, not one

Copernicus Marine splits the global physics analysis-forecast product, and this
is not obvious from the portal. Requesting currents from the merged dataset
fails outright with `VariableDoesNotExistInTheDataset`. The verified mapping:

| Dataset | Variables |
|---|---|
| `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m` | `uo`, `vo` — currents |
| `cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m` | `thetao` — temperature |
| `cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m` | `so` — salinity |
| `cmems_mod_glo_phy_anfc_0.083deg_P1D-m` | `siconc`, `sithick`, `usi`, `vsi` — sea ice |
| `cmems_mod_glo_wav_anfc_0.083deg_PT3H-i` | `VHM0` — significant wave height |

POLARIS subsets each one separately and merges them onto the analysis grid. A
dataset that fails is logged and skipped rather than aborting the ingestion.

Note the fourth row: **CMEMS supplies sea-ice drift velocity (`usi`/`vsi`)** —
the same quantity as the NSIDC Polar Pathfinder motion vectors, with no
Earthdata login, and as a forecast rather than an analysis.

### On the open mirrors

ERA5 and CMEMS require free registration. To keep real mode fully functional
without credentials, POLARIS can fall back to the open, key-free archive
endpoints that redistribute the same ERA5 and CMEMS/ECMWF products
(`ALLOW_OPEN_MIRRORS=true`, the default).

This is **not** a silent substitution. A record retrieved that way is stored with
`source="ERA5/open-archive"` or `source="CMEMS/open-marine"`, distinct from
`"ERA5/CDS"` and `"CMEMS"`, and that string is returned in every API response's
`provenance`. Set `ALLOW_OPEN_MIRRORS=false` to require the official APIs; the
ingestion is then recorded as `skipped` with the reason, and the risk engine
reports the layer as missing.

---

## 7. Dataset descriptions

### NSIDC G02135 daily concentration GeoTIFF

316 × 332 uint16 raster on the NSIDC South Polar Stereographic grid (EPSG:3412,
25 km, origin −3 950 000 / 4 350 000 m). Values are concentration × 1000, with
flags 2510 (pole hole), 2530 (coast), 2540 (land), 2550 (missing).

POLARIS decodes it in `services/landmask.py`, converts to a fraction in [0, 1],
projects cell centres to WGS-84 with pyproj, and area-averages onto the analysis
grid. Publication latency is roughly 24–48 h, so ingestion defaults to
`today − 2` in real mode.

**The land mask comes from flags 2530 and 2540** — 22 005 of 104 912 source
cells. This is what makes routing geographically honest.

### USNIC Antarctic iceberg bulletin

A small CSV (currently ~33 tracked bergs) with designator, length/width in
nautical miles, latitude, longitude, area, and last update date. POLARIS
validates and de-duplicates on `(iceberg_id, observed_at)`, so accumulating
bulletins over days builds a position history, from which drift speed and
bearing are derived (rejecting anything above 1.5 m/s as implausible).

### ERA5 single levels

10 m u/v wind, 2 m temperature, mean sea-level pressure. Retrieved as NetCDF via
`cdsapi` when credentials are set, or as point samples on a coarse grid
(2° × 5°, 143 points) and interpolated when using the open archive.

### Ocean analysis

Surface currents (u, v), sea-surface temperature, salinity and significant wave
height. From the CMEMS global analysis-forecast product via the official
toolbox, or the open marine endpoint.

### POLARIS demo dataset

Generated locally by `services/demo_data.py`. **Synthetic, never observation.**
The fields are physically *shaped* rather than random:

* Sea ice: seasonal ice-edge migration between a summer edge near 65.8°S and a
  winter edge near 57.5°S, minimum late February and maximum late September,
  with a sharpening edge in winter, coastal polynyas and spatially correlated,
  day-to-day-persistent noise.
* Atmosphere: three eastward-propagating depressions in the circumpolar trough,
  a continental high over the plateau, and geostrophic wind derived from the
  resulting pressure field with 20° cross-isobar inflow.
* Ocean: an eastward Antarctic Circumpolar Current jet near 56.5°S, a westward
  coastal current near 68.5°S, and a non-divergent mesoscale eddy field from a
  random streamfunction.

Measured output: mean wind 8.6 m/s, MSL 969–1017 hPa, mean surface current
0.18 m/s (p95 0.39), significant wave height up to 12 m in storms. Iceberg IDs
are prefixed `DEMO-` so they can never be mistaken for USNIC designators.

---

## 8. Installation

**Requirements:** Python 3.10+ (developed on 3.12). Nothing else — no Node.js, no
database server, no GDAL.

```bash
git clone <repository-url>
cd POLARIS/backend

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # optional; every setting has a working default
```

Optional, only for the credentialed provider paths:

```bash
pip install cdsapi copernicusmarine
```

---

## 9. Environment variables

All configuration lives in `backend/.env` (see `.env.example` for the annotated
full list). Nothing is required to start. The most relevant:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_MODE` | `demo` | `demo` = synthetic, `real` = live providers |
| `DATABASE_URL` | `sqlite:///./polaris.db` | PostgreSQL: `postgresql+psycopg2://user:pass@host:5432/polaris` |
| `LOG_LEVEL` | `INFO` | |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | |
| `DOMAIN_LAT_MIN/MAX`, `DOMAIN_LON_MIN/MAX` | −75/−50, 0/100 | analysis domain |
| `GRID_RESOLUTION_LAT/LON` | 0.5 / 1.0 | grid cell size in degrees |
| `ERA5_API_KEY` | unset | Copernicus CDS key |
| `COPERNICUS_USERNAME` / `_PASSWORD` | unset | CMEMS credentials |
| `ALLOW_OPEN_MIRRORS` | `true` | permit the key-free ERA5/CMEMS mirrors |
| `SEA_ICE_HISTORY_DAYS` | 400 | history length built for training |
| `ICEBERG_ENSEMBLE_MEMBERS` | 24 | Monte-Carlo members for drift uncertainty |
| `RISK_WEIGHT_*` | .40/.25/.20/.10/.05 | risk component weights |
| `ROUTE_MAX_ACCEPTABLE_RISK` | 0.85 | risk ceiling for graph inclusion |

**Credentials are never hardcoded and never logged.** `utils/logging.py`
installs a filter that redacts anything matching a password, API key, token or
`user:pass@host` URL pattern before it reaches any handler — there is a test for
it (`test_health.py::test_log_filter_redacts_credentials`).

---

## 10. Database setup

### SQLite (default — nothing to do)

```bash
python -m app.database.init_db
```

WAL journaling and foreign keys are enabled automatically.

### PostgreSQL + PostGIS

**Verified**: all 190 tests pass against PostgreSQL 16.4 with PostGIS 3.4, not
only against SQLite. Run them yourself:

```bash
docker run -d --name polaris-postgres   -e POSTGRES_USER=polaris -e POSTGRES_PASSWORD=changeme -e POSTGRES_DB=polaris   -p 55432:5432 postgis/postgis:16-3.4

POLARIS_TEST_DATABASE_URL=postgresql+psycopg2://polaris:changeme@localhost:55432/polaris pytest
```

`POLARIS_TEST_DATABASE_URL` points the suite at any database; everything is
created and dropped inside it, and PostGIS is enabled automatically when the
target is PostgreSQL so the generated geography columns and GiST indexes are
exercised too.

For a normal (non-Docker) install:

```bash
createdb polaris
psql -d polaris -c "CREATE EXTENSION IF NOT EXISTS postgis;"
```

Set `DATABASE_URL` in `.env`, then:

```bash
python -m app.database.init_db          # add --reset to drop first
```

On PostgreSQL the initialiser additionally adds, to each point table, a
generated `geography(Point,4326)` column with a GiST index, so operators can run
`ST_DWithin` queries directly against the warehouse. **The application never
depends on PostGIS** — it is a query accelerator, and everything works
identically without it.

### Schema

| Table | Contents |
|---|---|
| `sea_ice_observations` | Gridded concentration per cell/day, with land and quality flags |
| `sea_ice_extent_index` | Hemispheric daily extent/area |
| `iceberg_observations` | Tracked berg positions, dimensions, derived drift |
| `weather_observations` | u10, v10, wind speed/direction, 2 m temperature, MSL |
| `ocean_observations` | Currents, SST, salinity, wave height |
| `forecasts` | Predicted concentration per cell/horizon, with uncertainty |
| `iceberg_trajectories` | Predicted positions, speed, bearing, ensemble spread |
| `risk_grid` | Per-cell component and total risk, navigability |
| `route_results` | Computed routes: geometry, metrics, parameters used |
| `ingestion_log` | Audit trail of every ingestion, including skips and failures |
| `model_registry` | What was trained, when, on how much, and how well |

Every observation row carries `source` and `data_mode`. Every write is an
idempotent upsert on the table's natural key, so re-running an ingestion updates
rather than duplicates — there is a test for that too.

---

## 11. Data ingestion

```bash
python scripts/download_data.py                  # honours DATA_MODE
python scripts/download_data.py --mode real      # force real providers
python scripts/download_data.py --days 7         # more days of sea-ice grids
python scripts/download_data.py --only icebergs  # a single dataset
python scripts/download_data.py --landmask       # rebuild the land-mask asset
python scripts/download_data.py --init-db --json
```

Measured real-mode run:

```
dataset        status    source                     rows  message
----------------------------------------------------------------------------
sea_ice        ok        NSIDC/G02135               5151
extent_index   ok        NSIDC/G02135                400
icebergs       ok        USNIC                        33
weather        ok        ERA5/open-archive          5151
ocean          ok        CMEMS/open-marine          5151
```

The ingestion layer downloads with retry and exponential backoff; validates
files, coordinates, timestamps and value ranges; normalises longitude to
[−180, 180); handles NetCDF, CSV and GeoTIFF; regrids from EPSG:3412; de-dupes
on natural keys; and writes an audit row for every attempt. A failure in one
dataset never aborts the others.

---

## 12. Model training

```bash
python scripts/preprocess_data.py --days 200     # build the history archive
python scripts/train_models.py                   # train 24/48/72 h
python scripts/train_models.py --horizons 24 --algorithm persistence
python scripts/train_models.py --rebuild-history --days 250
```

### The model

Two `HistGradientBoostingRegressor` candidates per horizon — one on the
concentration level, one on the change over the horizon — plus a persistence
baseline. All three are scored on the same held-out split and the best is kept
(§13). Each uses 24 features per ocean cell:

* concentration now, plus lags at 1, 2, 3, 5, 7 and 14 days
* changes over 1, 3 and 7 days
* local spatial mean (3×3, 5×5), meridional/zonal gradient, roughness
* distance to the 15% ice edge, distance to the coast
* latitude, longitude
* day-of-year harmonics, the horizon itself
* 2 m temperature and 10 m wind speed, when a weather archive is available

### What makes the numbers trustworthy

* The split is **chronological, never shuffled** — all training samples precede
  all validation samples. A test asserts this.
* Every candidate is scored against the **persistence baseline** on the same
  held-out data, and the baseline is itself a candidate: if it wins, it is what
  gets shipped.
* Uncertainty is calibrated from the **validation residuals**, binned by
  predicted concentration, so the reported σ is a measured spread.
* Feature importance is measured by permutation (RMSE increase when a column is
  shuffled), not by an impurity heuristic.
* Metrics are written to `models/<name>_h<hours>.json` and to `model_registry`.
* **Models are never retrained inside an API request.** Artifacts are loaded
  once and cached; there is a test asserting the cache returns the same object.

---

## 13. Model evaluation

All figures below are measured on a held-out chronological split of the data
actually used. Reproduce them with the commands in §19.

### Model selection is gated on measured skill

Sea-ice concentration is *strongly* persistent, so a learned model does not
automatically beat "tomorrow looks like today". Rather than assert that it does,
`train_horizon` fits two learned candidates and scores both against the
persistence baseline on the same split:

* `gbr_level` — gradient boosting on the concentration itself, using every
  available feature
* `gbr_lean` — the same, restricted to the sea-ice features alone. Extra
  predictors are not free; this candidate measures whether they pay for
  themselves rather than assuming it
* `gbr_delta` — gradient boosting on the **change** over the horizon, which makes
  persistence the zero-prediction

Whichever has the lowest validation RMSE is saved — **including persistence
itself if neither learned candidate beats it**. Every candidate's score and the
identity of the winner are recorded in the metrics, the model card and the API
response. A test (`test_selected_model_is_never_worse_than_persistence`) enforces
that the shipped model can never be worse than the baseline.

### Real data — 371 days of NSIDC G02135, 51 × 101 grid

| Horizon | Selected | Train | Validation | MAE | RMSE | R² | Persistence RMSE | **Skill** | Ice-edge accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 24 h | `gbr_delta` | 1 007 632 | 255 456 | 0.0175 | **0.0384** | 0.990 | 0.0403 | **+0.091** | 0.990 |
| 48 h | `gbr_level` | 1 007 632 | 251 908 | 0.0264 | **0.0586** | 0.978 | 0.0604 | **+0.059** | 0.984 |
| 72 h | `gbr_lean` | 1 004 084 | 251 908 | 0.0317 | **0.0705** | 0.968 | 0.0746 | **+0.106** | 0.980 |

All candidates:

```
  24h  persistence=0.0403/+0.000  gbr_level=0.0386/+0.081  gbr_delta=0.0384/+0.091  gbr_lean=0.0409/-0.033
  48h  persistence=0.0604/+0.000  gbr_level=0.0586/+0.059  gbr_delta=0.0600/+0.015  gbr_lean=0.0600/+0.015
  72h  persistence=0.0746/+0.000  gbr_level=0.0708/+0.099  gbr_delta=0.0711/+0.090  gbr_lean=0.0705/+0.106
```

### What the ice-dynamics features bought

Adding sea-ice drift and thermodynamic features changed the real-data result
substantially:

| Horizon | Sea-ice features only | With drift + thermodynamics | |
|---|---|---|---|
| 24 h | **−0.033** (worse than persistence) | **+0.091** | learned model now wins clearly |
| 48 h | 0.000 (baseline was selected) | **+0.059** | a learned model wins for the first time |
| 72 h | +0.085 | **+0.106** | modest further gain |

The 24 h row is the cleanest evidence: `gbr_lean` uses exactly the old feature
set and scores **−0.033** — worse than assuming no change — while the same
algorithm with drift and thermodynamics reaches **+0.091**. The features, not
the model, made the difference.

Permutation importance confirms they are genuinely used, not merely present:
`ice_advection_per_day` is the third-strongest predictor at 24 h, and `fdd_7`
(seven-day accumulated freezing-degree-days) is third at 48 h.

At 72 h the selector picks `gbr_lean` — the extra features stop paying for
themselves at that range, and the measurement says so rather than the
configuration assuming it.

### The physics behind those features

Sea-ice concentration evolves by

```
dc/dt = -u . grad(c)  -  c div(u)  +  thermodynamics
        (advection)      (convergence)
```

Both dynamic terms are computed per day from the CMEMS sea-ice velocity field
(`usi`/`vsi`), and the thermodynamic term is represented by accumulated
freezing-degree-days over 3 and 7 days (Stefan's law makes growth scale with
the square root of accumulated FDD).

One design note worth stating, because the obvious approach fails: Antarctic
pack drifts at roughly 0.05–0.15 m/s, so 24 h of motion is 4–13 km against grid
cells of about 55 km. A feature that tracks *where the ice came from* is
therefore **sub-grid and unresolvable** at this resolution. The divergence and
gradient of a smooth drift field are perfectly well resolved, which is why the
continuity terms are used instead of a displacement feature.

### Demo mode — 200-day synthetic history, same grid### Demo mode — 200-day synthetic history, same grid

| Horizon | Selected | Train | Validation | MAE | RMSE | R² | Persistence RMSE | **Skill** | Ice-edge accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 24 h | `gbr_lean` | 275 058 | 70 623 | 0.0385 | **0.0488** | 0.983 | 0.0585 | **+0.305** | 0.968 |
| 48 h | `gbr_lean` | 271 341 | 70 623 | 0.0477 | **0.0598** | 0.974 | 0.0784 | **+0.418** | 0.961 |
| 72 h | `gbr_level` | 271 341 | 70 623 | 0.0479 | **0.0595** | 0.974 | 0.0786 | **+0.427** | 0.961 |

Note that `gbr_lean` wins at 24 and 48 h here: the synthetic ice field is not
advected by the synthetic drift field, so the drift features carry no signal in
demo mode and the selector correctly discards them. That is the mechanism
working as intended in both directions.

> Demo skill is much higher than real skill because the synthetic field is
> smoother and more predictable than the real ice pack. **Demo metrics
> demonstrate that the pipeline works; they are not an estimate of operational
> accuracy.** The real-data table above is the one to judge the system by.

### Reading the numbers

* **RMSE** is in concentration units (0–1). 0.04 means the typical cell is wrong
  by 4 percentage points of ice cover.
* **Skill** is `1 − MSE(model)/MSE(persistence)`. 0 means no better than assuming
  no change; +0.10 means 10% of the baseline's squared error removed.
* **Ice-edge accuracy** is ice/no-ice agreement at the 15% threshold — the
  quantity that actually matters for navigation.
* **Top predictors** are measured by permutation (RMSE increase when a column is
  shuffled), not an impurity heuristic. On real data: current concentration
  dominates, then the 3×3 spatial mean, then 2 m temperature.

### What makes these numbers trustworthy

* The split is **chronological, never shuffled** — all training samples precede
  all validation samples. A test asserts it.
* Every model is scored against persistence on the **same** held-out data.
* Uncertainty is calibrated from the **validation residuals**, binned by
  predicted concentration, so the reported σ is a measured spread.
* Metrics are written to `models/<name>_h<hours>.json` and to `model_registry`,
  and returned by `/api/sea-ice/forecast` and `/api/dashboard/models`.

---

## 14. Running the backend

```bash
cd POLARIS/backend
python run.py                    # or: uvicorn app.main:app --reload
python run.py --init-db          # create the schema first
python run.py --port 8080 --reload
```

Then:

* Swagger UI — <http://localhost:8000/docs>
* OpenAPI JSON — <http://localhost:8000/openapi.json>
* Test client — <http://localhost:8000/ui/>
* Endpoint index — <http://localhost:8000/api>

The API starts even if the database is unreachable; data endpoints then return
503 with an actionable message rather than failing to boot.

---

## 15. API endpoints

### Health and system

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Overall status plus per-subsystem detail (database, ingested data, history archive, trained models). 503 if the database is down. |
| GET | `/api/health/live` | Liveness; does not touch the database |
| GET | `/api/health/ready` | Readiness; requires ingested data |
| GET | `/api/system` | Runtime configuration; credentials reported only as booleans |

### Sea ice

| Method | Path | Key parameters |
|---|---|---|
| GET | `/api/sea-ice/current` | `format=grid\|cells\|both`, `stride`, `bbox`, `limit` |
| GET | `/api/sea-ice/forecast` | `forecast_hours=24\|48\|72`, `region`, `format`, `include_uncertainty` |
| GET | `/api/sea-ice/extent` | `days` |

The forecast response carries `validation_metrics` — the model's real held-out
scores including its skill against persistence — and `algorithm`, which says
`persistence` if no artifact has been trained.

### Weather

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/weather/stations` | Live conditions and an hourly outlook at each station, with a freezing-spray hazard flag |
| GET | `/api/weather/point` | The same at any coordinate |
| GET | `/api/weather/current` | Age of every environmental layer |

Marine fields come back `null` at a point inside the pack — waves are not
defined under sea ice, so that is physics rather than a gap.

### Icebergs

| Method | Path | Key parameters |
|---|---|---|
| GET | `/api/icebergs` | `min_area_km2`, `bbox`, `in_domain_only`, `limit` |
| GET | `/api/icebergs/{iceberg_id}/trajectory` | `forecast_hours` (≤240), `persist` |
| GET | `/api/icebergs/trajectories` | batch form for map rendering |

An unknown iceberg returns 404 **with the list of known designators**.

### Risk

| Method | Path | Key parameters |
|---|---|---|
| GET | `/api/risk/map` | `forecast_hours`, `ice_class`, `draft_m`, `risk_tolerance`, `format`, `stride`, `min_risk`, `top_hotspots`, `persist` |
| GET | `/api/risk/point` | `latitude`, `longitude`, plus the vessel parameters |

### Navigation

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/route/optimize` | Compute shortest / safest / POLARIS routes |
| GET | `/api/navigation/stations` | Route endpoints usable by key, with snap distances |
| GET | `/api/navigation/landmask` | The land mask as GeoJSON (lets a client draw the coast with no tile server) |
| GET | `/api/navigation/routes` | Recently computed routes |
| GET | `/api/navigation/routes/{request_id}` | Retrieve a stored result |

### Dashboard

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/dashboard/summary` | Sea ice, forecasts, icebergs, risk hotspots, routing availability, ingestion, row counts |
| GET | `/api/dashboard/models` | Artifacts on disk and their registry rows |
| GET | `/api/dashboard/ingestion` | Audit trail, including skipped and failed runs |

### Example request

```bash
curl -X POST http://localhost:8000/api/route/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "start":       {"station": "bharati"},
    "destination": {"station": "maitri"},
    "vessel": {
      "ice_class": "icebreaker",
      "speed_knots": 14,
      "fuel_consumption_tpd": 45,
      "draft_m": 7.0,
      "risk_tolerance": 0.3
    },
    "preferences": {
      "forecast_hours": 24,
      "profiles": ["shortest", "safest", "polaris"]
    },
    "persist": true
  }'
```

Measured response (demo mode, September conditions, icebreaker):

| Profile | Distance | Duration | Fuel (est.) | Mean risk | vs shortest |
|---|---|---|---|---|---|
| shortest | 3 766 km | 943 h | 1 767 t | 0.459 | — |
| safest | 6 983 km | 841 h | 1 578 t | 0.193 | +85% distance, **−58% risk** |
| polaris | 6 135 km | 809 h | 1 517 t | 0.233 | +63% distance, −49% risk |

Worth reading carefully: the safest route is 85% longer yet **faster and
cheaper**, because avoiding the pack keeps the vessel near service speed while
the direct route crawls through ice. That is an output of the speed model, not
an assumption built into it — and exactly the kind of trade-off a router that
only minimised distance would hide.

### Response quality

Every endpoint validates its input, returns a typed Pydantic model, and uses
meaningful status codes. Errors share one envelope:

```json
{
  "error": "invalid_input",
  "detail": "forecast_hours must be one of [24, 48, 72], got 99",
  "status_code": 422,
  "path": "/api/sea-ice/forecast",
  "hint": "Check the request against the schema at /docs."
}
```

Handled: invalid coordinates, points outside the domain, unknown stations and
icebergs, malformed bboxes, unsupported horizons, all-zero risk weights, missing
datasets (503 with a hint naming the script to run), missing model artifacts,
database failures, and unreachable providers. Tracebacks are never returned to
the client.

Every data response carries a `provenance` block: `data_mode`, contributing
`sources`, `observed_at`, `generated_at`, and an explicit disclaimer.

---

## 16. Demo mode

For a demonstration where network access or credentials cannot be relied on:

```bash
cd POLARIS/backend
python scripts/seed_demo_data.py --reset
python run.py
```

That runs the **entire real pipeline** on synthetic input — schema, ingestion,
history archives, training, forecasting, trajectories, risk grid and the
Bharati ↔ Maitri routes — in about 50 seconds. There is no separate demo code
path: the same services, the same database, the same endpoints.

Safeguards against synthetic data being mistaken for observation:

* every row carries `source="POLARIS-DEMO"` and `data_mode="demo"`;
* iceberg IDs are prefixed `DEMO-`, so no USNIC designator can be imitated;
* every API response's `provenance.disclaimer` states the data is synthetic and
  must not be used for navigation;
* the browser client shows a `data mode: demo` badge at all times;
* real mode never emits demo rows — when a provider is unavailable the
  ingestion is recorded as `skipped`, and a test asserts this.

### A note on what the demo shows

Run in September (peak Antarctic sea-ice extent), the demonstration route
reports `degraded` for the default 1A Super vessel: no path to either station
satisfies the navigability constraints, because consolidated pack ice reaches
both stations. **That is the correct answer** — the Antarctic resupply season is
November to March. The seed script therefore also re-runs the same request for
an icebreaker-capable vessel, so both the constraint and its resolution are
visible. A system that returned a confident route there would be wrong.

---

## 17. Real-data mode

```bash
cd POLARIS/backend
export DATA_MODE=real                     # Windows: set DATA_MODE=real

python scripts/download_data.py --days 5
python scripts/preprocess_data.py --days 200   # downloads/caches NSIDC GeoTIFFs
python scripts/train_models.py
python run.py
```

What each provider needs:

| Provider | Credentials | Without them |
|---|---|---|
| NSIDC G02135 | none | works |
| USNIC icebergs | none | works |
| ERA5 | free CDS account (`ERA5_API_KEY` + `cdsapi`) | open archive mirror, distinctly labelled |
| Copernicus Marine | free CMEMS account (`copernicusmarine`) | open marine mirror, distinctly labelled |

With `ALLOW_OPEN_MIRRORS=false` and no credentials, weather and ocean ingestion
are recorded as `skipped`; POLARIS still runs on sea ice and icebergs alone and
reports `weather` and `ocean` in `missing_layers`, renormalising the risk total
over the layers it does have.

### What real mode actually returns

Measured on a live run (early September 2026):

```
dataset        status    source                     rows
--------------------------------------------------------
sea_ice        ok        NSIDC/G02135             10302   observed 2026-09-03
extent_index   ok        NSIDC/G02135               400   17.36 M km2 hemispheric
icebergs       ok        USNIC                       33   A76C, B09G, C18B, ...
weather        ok        ERA5/CDS                  5151   via cdsapi, credentialed
ocean          ok        CMEMS                     3809   via copernicusmarine
```

All four primary sources are first-party, with credentials — no mirrors in this
run. Measured values from it: ERA5 10 m wind −21 to +21 m/s, 2 m temperature
−53 to +5 °C, MSL 962–1016 hPa; CMEMS currents to 0.78 m/s, SST −2 to 8.3 °C,
salinity 32.9–36.4 PSU, sea-ice drift to 0.50 m/s, significant wave height to
9.7 m. ERA5 retrievals queue at CDS for 1–2 minutes.

* **33 real tracked icebergs**, of which **15 lie inside** the default
  0–100°E domain and 18 do not. Out-of-domain bergs are listed with
  `in_domain: false`; asking for their trajectory returns 422 explaining that no
  forcing field exists there, rather than a fabricated stationary track.
* A real in-domain berg (B09G) drifts **15.2 km over 72 h (0.058 m/s)** under
  ERA5 wind of (−12.6, −5.8) m/s and a surface current of (−0.03, +0.06) m/s,
  with an ensemble spread of 17.8 km — physically plausible for a berg in 83%
  concentration pack.
* The 48 h forecast reports `algorithm: persistence`, because that is what the
  skill gate selected on real data (§13). The API says so rather than implying a
  learned model is in use.

Building 400 days of real NSIDC history downloads about 240 MB of GeoTIFFs into
`data/raw/nsidc/` (roughly 15 minutes on a good connection) and is cached, so
subsequent runs are immediate.

The provider layer is modular: each client is a class in
`services/data_ingestion.py` with its own `fetch`, so adding a source or
swapping an endpoint touches one class and its configuration entry.

---

## 18. Running tests

```bash
cd POLARIS/backend
pytest                          # 204 tests, about 77 s
pytest -v
pytest tests/test_e2e.py        # the integration test alone
pytest -k "iceberg and physics"
```

Current result:

```
204 passed in 77.00s
```

The suite runs against a throwaway SQLite database and a **reduced analysis
domain** (1.0° × 2.0° cells), so the full pipeline — ingest, preprocess, train,
forecast, drift, risk, route — executes for real in under a minute. **Nothing is
mocked.**

| File | Tests | What it verifies |
|---|---|---|
| `test_health.py` | 9 | Connectivity, schema, health/readiness, OpenAPI completeness, credential redaction |
| `test_data.py` | 27 | Ingestion, idempotency, demo labelling, provider-unavailable handling, validation, geodesy, land mask, preprocessing, repositories |
| `test_sea_ice.py` | 26 | Metric correctness, uncertainty calibration, training, **skill over persistence**, chronological split, artifact round-trip, inference, baseline fallback, endpoints |
| `test_icebergs.py` | 28 | Hydrostatics, force balance, drag scaling, added mass, current-following, uncertainty growth, grounding, endpoints |
| `test_risk.py` | 43 | Every component's monotonicity and bounds, iceberg trajectory influence, union combination, missing-layer renormalisation, grid assembly, endpoints |
| `test_routes.py` | 40 | Speed/fuel model, graph structure, land avoidance, **A* == Dijkstra optimum**, profile ordering, metric consistency, endpoints |
| `test_e2e.py` | 16 | Stage-by-stage integration and API agreement |

These are behavioural assertions, not existence checks. Representative examples:

* `test_model_beats_the_persistence_baseline` — every horizon must have positive
  skill, or the suite fails.
* `test_coriolis_and_pressure_gradient_cancel_at_the_water_velocity` — a berg
  moving with the water must feel no net rotational force, to 1e-15.
* `test_drag_scales_with_the_square_of_relative_velocity` — doubling the wind
  must quadruple the force, exactly.
* `test_astar_and_dijkstra_agree_on_cost` — recomputes both path costs and
  requires them equal, proving the heuristic is admissible.
* `test_route_never_crosses_land` — densifies every leg at 30 km and checks each
  sample against the land mask.
* `test_api_reports_the_same_numbers_as_the_services` — compares the API's
  forecast field against the service's array element-by-element.

Five real bugs were found and fixed during development, four of them by these
tests and one by running against live data:

1. A feature-name mismatch that broke the untrained-baseline fallback.
2. `Literal[int]` query parameters that rejected *every* value, making
   `forecast_hours` on `/api/risk/map` unusable.
3. A coastal-risk ramp that silently evaluated to zero on coarse grids.
4. Grounded icebergs emitting output points off-cadence, which misaligned the
   control run from its ensemble members and corrupted the uncertainty radius.
5. **Found in real mode:** icebergs outside the analysis domain (18 of the 33
   USNIC bergs are in the Weddell/Scotia sector) were integrated against an
   all-zero forcing field and returned a confident 0 km drift with a non-zero
   uncertainty ring. They are now listed with `in_domain: false` and the
   trajectory endpoint returns 422 explaining why, rather than fabricating one.

---

## 19. End-to-end workflow

```bash
cd POLARIS/backend

python scripts/download_data.py       # 1. ingest
python scripts/preprocess_data.py     # 2. build the history archives
python scripts/train_models.py        # 3. train and validate
pytest                                # 4. verify
python run.py                         # 5. serve
```

Then exercise it:

```bash
curl http://localhost:8000/api/health
curl "http://localhost:8000/api/sea-ice/current?stride=2"
curl "http://localhost:8000/api/sea-ice/forecast?forecast_hours=72"
curl http://localhost:8000/api/icebergs
curl "http://localhost:8000/api/icebergs/DEMO-04/trajectory?forecast_hours=72"
curl "http://localhost:8000/api/risk/map?forecast_hours=24&stride=2"
curl http://localhost:8000/api/dashboard/summary
# POST /api/route/optimize — see the example in §15
```

Or, in one command for a demonstration:

```bash
python scripts/seed_demo_data.py --reset && python run.py
# then open http://localhost:8000/ui/
```

The data flow the integration test asserts:

```
observations → environment snapshot → forecast (+24 h)
             → iceberg trajectories (+72 h, 24-member ensemble)
             → risk grid (conditioned on that forecast, informed by those tracks)
             → navigation graph → 3 routes → persisted → returned by the API
```

---

## 20. Limitations

Stated plainly, because a decision-support tool that hides its limits is worse
than none.

**Forecasting**

1. **Skill over persistence is real but modest** — +0.091 at 24 h, +0.059 at
   48 h, +0.106 at 72 h on real data (§13). That is a genuine improvement, and
   the ice-dynamics features roughly tripled it, but persistence remains a
   strong competitor at these lead times: much of the absolute accuracy still
   comes from ice not moving far in a day. Treat the forecast as a measurably
   better-than-persistence refinement, not as a step change.
2. Trained on roughly one year of daily fields. That is enough for lag and
   spatial structure, not enough to learn multi-year variability, and the
   day-of-year features are fitted on a single cycle. A shorter archive covering
   only part of the seasonal cycle produces *negative* skill (§13) — history
   length is the parameter that matters most here.
3. Concentration only. Ice thickness, ridging, floe size and lead structure all
   matter operationally and are not modelled.
4. Sea-ice **drift** is used (CMEMS `usi`/`vsi`, via the advection and
   convergence terms), but **deformation** — ridging and rafting, which
   thicken ice without changing concentration — is not. The NSIDC Polar
   Pathfinder motion vectors (nsidc-0116) remain unintegrated; CMEMS supplies
   the same quantity without an Earthdata login, so they would add an
   independent estimate rather than new information.

**Iceberg trajectories**

5. **Forcing is held at its analysis value for the whole forecast.** At 24–72 h
   this is the dominant error term. Time-varying forcing needs an ocean/atmosphere
   forecast rather than an analysis.
6. Thickness is estimated from waterline length via an empirical draft relation,
   capped at 350 m for large tabular bergs. Thickness is rarely observed, and it
   scales the mass and both drag areas directly.
7. No deterioration: no melt, calving, or rolling, so bergs keep their mass.
8. Grounding is detected against the land mask, not bathymetry — a berg with a
   290 m keel would ground on a shoal long before reaching the coast.

**Risk engine**

9. Component thresholds are parameterised from published navigation and icing
   guidance, **not fitted to incident data**. They are explicit and configurable
   precisely so an operator can replace them with their own table.
10. Default weights (0.40/0.25/0.20/0.10/0.05) are a reasonable starting point,
   not a validated optimum.
11. **No bathymetry.** Proximity to the coast is the proxy for shoal water and
    uncharted ground. This cannot replace a chart, and POLARIS says so in the
    function, the API description and here.

**Routing**

12. Grid resolution (0.5° × 1.0°, roughly 55 × 40 km at 65°S) sets the finest
    manoeuvre that can be represented. Narrow leads and coastal passages are
    below it.
13. The vessel is a point with a speed model. No seakeeping, no manoeuvring
    limits, no ice-ramming or backing-and-filling behaviour.
14. Fuel is `time × consumption rate at service speed`. Real consumption varies
    with propulsion load, especially in ice. Treat the numbers as comparative
    between routes, not absolute.
15. Static optimisation: the environment does not evolve as the vessel moves
    along the route. Over a 30-day passage that is a real simplification.
16. Two stations, one region. The domain is configurable, but the default
    covers 0–100°E only.

**Data**

17. NSIDC latency is 24–48 h, so "current" means yesterday or the day before.
18. USNIC tracks only bergs above roughly 10 nautical miles on a longest axis.
    Smaller bergy bits and growlers — a genuine hazard — are not in any feed
    POLARIS consumes.
19. Iceberg drift history accumulates only from the bulletins POLARIS has
    ingested. On a first run there is one fix per berg and no derived velocity.

**What POLARIS does not do**

It does not guarantee safe navigation, replace ice charts or official ice
services, satisfy any regulatory requirement, or account for vessel condition,
crew, cargo or ice-management support. It ranks options using the information
available and reports what it did not know.

---

## 21. Scientific assumptions

### Terminology, used consistently

| Term | Meaning in POLARIS |
|---|---|
| **Observation** | A measured value from a provider (`data_mode="real"`) |
| **Forecast** | A model prediction of a future *field* (sea-ice concentration) |
| **Prediction** | A physics-model result for a specific object (iceberg position) |
| **Estimation** | A derived quantity from a documented model (fuel, duration, thickness) |
| **Recommendation** | A computational ranking of options (the POLARIS route) |

### Sea-ice forecasting

* The 15% concentration threshold defines the ice edge (the passive-microwave
  convention).
* Concentration is spatiotemporally autocorrelated at 1–3 days — the premise
  behind both the lag features and the persistence baseline.
* Cells are treated as conditionally independent given their features; spatial
  structure enters through the neighbourhood features, not a spatial prior.
* Land cells never carry a forecast.

### Iceberg drift

The momentum balance follows the standard formulation (Bigg et al. 1997;
Lichey & Hellmer 2001; Wagner, Dell & Eisenman 2017):

```
M(1+Cm) dv/dt = −M f k×v + F_air + F_water + F_ice + F_pressure
```

with quadratic drag on the sail and keel, sea-ice drag engaged above 0.85
concentration, and the sea-surface tilt expressed through the geostrophic
current. Integrated with RK4.

| Parameter | Value | Basis |
|---|---|---|
| ρ_ice / ρ_water / ρ_air / ρ_sea-ice | 850 / 1027 / 1.225 / 910 kg m⁻³ | standard |
| Air drag C_a | 1.3 | bluff-body form drag |
| Water drag C_w | 0.9 | keel form drag |
| Sea-ice drag C_i | 1.0, on the side area × 1 m pack thickness | Lichey & Hellmer form |
| Added mass C_m | 0.5 | entrained water |
| Draft/thickness | ρ_ice/ρ_water ≈ 0.83 | hydrostatic |
| Thickness | 3.78·L^0.63 (draft), capped at 350 m | empirical draft–length relation; the cap reflects East Antarctic ice-shelf thickness, since a tabular berg's thickness is set at calving, not by its planform |
| Besetment | above 0.95 concentration, velocity relaxes to the pack with a 12 h e-folding time | parameterisation of bergs drifting with compact pack |
| Ensemble | σ 1.5 m/s on wind, 0.05 m/s on current, 15% on each drag coefficient | typical analysis error |

Two consistency properties are asserted by tests: a berg moving with the water
feels no net rotational force (Coriolis and pressure gradient cancel exactly),
and drag scales exactly with the square of relative velocity.

### Risk

* Risk is normalised to **[0, 1]**: 0 negligible, 1 extreme. Used consistently
  everywhere — the grid, point queries, route means and maxima.
* The total is a weighted sum **renormalised over the components that had
  data**, so a missing layer lowers confidence rather than lowering the score.
* Individual iceberg risks combine as an independent union
  `1 − Π(1 − rᵢ)`, so many distant bergs cannot sum into a false alarm.
* Effective distance to a berg is its **closest approach over the forecast
  window minus that point's ensemble uncertainty** — a berg drifting towards a
  cell raises its risk before it arrives.
* Component thresholds: sea ice 0.15 → the vessel's ice capability; wind
  10 → 28 m/s (Beaufort 5 → 10); icing below −2 °C with wind above 10 m/s;
  waves 2.5 → 9 m; currents 0.6 → 1.5 m/s. All in `services/risk_engine.py`,
  all configurable.

### Vessel model

Ice capability by class — the concentration at which risk saturates and speed
collapses: `none` 0.30, `1C` 0.45, `1B` 0.55, `1A` 0.70, `1A Super` 0.82,
`PC5` 0.88, `icebreaker` 0.95. These are **model parameters for ranking routes**,
not certified capabilities of any real ship, and not a substitute for a Polar
Ship Certificate or POLARIS (IMO) risk-index assessment.

Speed model: unaffected in open water, falling towards 12% of service speed as
concentration approaches the vessel's capability, minus up to 25% for added wave
resistance by 8 m significant wave height.

### Routing

Edge cost, in equivalent kilometres:

```
cost = w_distance · distance
     + w_risk     · risk × distance × 4
     + w_fuel     · fuel_tonnes × 12
```

The exchange rates (4 km per unit of risk sustained over 1 km; 12 km per tonne)
are the tunable statement of how much detour safety and fuel are worth. A*'s
heuristic is `w_distance × great-circle distance to goal`, which is admissible
because the risk and fuel terms are non-negative — so A* returns a true optimum
for the stated cost, and a test verifies it matches Dijkstra exactly.

---

## Acknowledgements

Data from the **National Snow and Ice Data Center** (Sea Ice Index v4, G02135),
the **US National Ice Center** (Antarctic iceberg bulletin), and the
**Copernicus Programme** (ERA5 reanalysis; Copernicus Marine Service ocean
analysis). Station coordinates and operational context from the
**National Centre for Polar and Ocean Research**, Ministry of Earth Sciences.

---

## Disclaimer

POLARIS is a **decision-support prototype** built for Smart India Hackathon 2026.
Its forecasts, trajectory predictions and risk scores carry uncertainty and
**do not guarantee safe navigation**. It is not a substitute for official ice
charts, ice services, nautical charts, weather routing services, or the
judgement of a master and ice pilot.
