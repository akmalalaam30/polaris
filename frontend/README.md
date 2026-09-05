# POLARIS test client

A deliberately minimal browser client whose only job is to **prove the backend
works** and to make a demonstration easy to run. It is not a product UI.

## What it does

| Panel | Backend endpoint it calls |
|---|---|
| Backend health / data-mode badges | `GET /api/health` |
| Land & ice-shelf outline | `GET /api/navigation/landmask` |
| Sea-ice concentration (observed) | `GET /api/sea-ice/current` |
| Sea-ice concentration (+24/48/72 h) | `GET /api/sea-ice/forecast` |
| Iceberg markers, drift tracks, uncertainty rings | `GET /api/icebergs`, `GET /api/icebergs/trajectories` |
| Navigation risk grid + hotspots | `GET /api/risk/map` |
| Route endpoints | `GET /api/navigation/stations` |
| Shortest / safest / POLARIS routes | `POST /api/route/optimize` |
| Status panel | `GET /api/dashboard/summary` |

**There is no fixture data anywhere in this directory.** If the backend is not
running, the page shows errors instead of plausible-looking numbers — which is
the point of a test client.

## Running it

The backend already serves this directory, so nothing extra is needed:

```bash
cd POLARIS/backend
python run.py
```

then open <http://localhost:8000/ui/>.

To serve it separately (for example while editing it):

```bash
cd POLARIS/frontend
npm start          # or: python -m http.server 5500 --directory src
```

On port 5500 or 3000 the client targets `http://localhost:8000/api`, so the
backend must permit that origin (`CORS_ORIGINS` in `backend/.env`; the default
`*` already does).

## Design notes

* **No build step, no npm dependencies.** `npm install` is not required and
  `node_modules/` is never created. This removes a whole class of demo-day
  failures.
* **Leaflet 1.9.4 is vendored** under `src/vendor/`, so the map library loads
  with no internet connection.
* **Basemap tiles are opt-in.** They are the only feature that needs internet.
  With tiles off, the coastline is drawn from `/api/navigation/landmask` — the
  *same* land mask the router respected, which is more useful for verification
  than a generic basemap anyway.
* Numbers are shown to the precision the API returns them. Model metrics shown
  next to a forecast are the model's real held-out validation scores.

## Files

```
src/
├── index.html      layout and controls
├── app.js          all API calls and map rendering (~450 lines, no framework)
├── styles.css      minimal styling
└── vendor/         Leaflet 1.9.4 (vendored for offline use)
```
