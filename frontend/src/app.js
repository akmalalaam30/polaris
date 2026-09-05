/* POLARIS backend test client.
 *
 * This page has no data of its own. Every value it displays is fetched from the
 * POLARIS API at request time - there are no fixtures, no cached sample
 * responses and no client-side modelling. If the backend is not running, the
 * page shows errors rather than plausible-looking numbers.
 */

'use strict';

const API = (location.port === '5500' || location.port === '3000')
  ? 'http://localhost:8000/api'   // served by a separate static server
  : '/api';                       // served by the backend itself

const state = {
  map: null,
  layers: {},
  tiles: null,
  routes: {},
  routeLayers: {},
  selectedRoute: null,
  stations: [],
  grid: null,
};

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------
async function api(path, options) {
  const response = await fetch(API + path, options);
  const text = await response.text();
  let payload;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch (err) {
    throw new Error(`${response.status}: response was not JSON`);
  }
  if (!response.ok) {
    const detail = payload && (payload.detail || payload.error);
    throw new Error(
      typeof detail === 'string' ? detail : JSON.stringify(detail || payload)
    );
  }
  return payload;
}

const postJSON = (path, body) =>
  api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

// ---------------------------------------------------------------------------
// Colour scales
// ---------------------------------------------------------------------------
function iceColour(value) {
  // White ice on a dark ocean, the convention used by ice charts.
  const t = Math.max(0, Math.min(1, value));
  const r = Math.round(30 + 215 * t);
  const g = Math.round(70 + 180 * t);
  const b = Math.round(120 + 135 * t);
  return `rgb(${r},${g},${b})`;
}

function riskColour(value) {
  const t = Math.max(0, Math.min(1, value));
  const r = Math.round(40 + 210 * Math.min(1, t * 1.6));
  const g = Math.round(150 - 120 * t);
  const b = Math.round(90 - 60 * t);
  return `rgb(${r},${g},${b})`;
}

function gradientCss(fn) {
  const stops = [];
  for (let i = 0; i <= 10; i += 1) stops.push(fn(i / 10));
  return `linear-gradient(to right, ${stops.join(',')})`;
}

function showLegend(title, fn) {
  document.getElementById('legend').classList.remove('hidden');
  document.getElementById('legend-title').textContent = title;
  document.getElementById('legend-bar').style.background = gradientCss(fn);
}

// ---------------------------------------------------------------------------
// Map setup
// ---------------------------------------------------------------------------
function initMap() {
  state.map = L.map('map', { preferCanvas: true, worldCopyJump: false })
    .setView([-64, 45], 3);
  ['land', 'ice', 'risk', 'bergs', 'stations', 'routes'].forEach((name) => {
    state.layers[name] = L.layerGroup().addTo(state.map);
  });
  state.layers.risk.remove();
}

function toggleTiles(on) {
  if (on && !state.tiles) {
    state.tiles = L.tileLayer(
      'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      { maxZoom: 8, attribution: '&copy; OpenStreetMap contributors' }
    ).addTo(state.map);
    state.tiles.bringToBack();
  } else if (!on && state.tiles) {
    state.map.removeLayer(state.tiles);
    state.tiles = null;
  }
}

function cellRectangle(lat, lon, grid, style) {
  const halfLat = grid.dlat / 2;
  const halfLon = grid.dlon / 2;
  return L.rectangle(
    [[lat - halfLat, lon - halfLon], [lat + halfLat, lon + halfLon]],
    style
  );
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------
async function loadLandmask() {
  const geojson = await api('/navigation/landmask');
  state.layers.land.clearLayers();
  L.geoJSON(geojson, {
    style: { color: '#6b7c88', weight: 0.4, fillColor: '#b9c6ce', fillOpacity: 0.95 },
    interactive: false,
  }).addTo(state.layers.land);
}

async function loadSeaIce() {
  const choice = document.getElementById('ice-source').value;
  const meta = document.getElementById('ice-meta');
  state.layers.ice.clearLayers();
  meta.textContent = 'loading...';

  try {
    let field;
    let grid;
    if (choice === 'current') {
      const body = await api('/sea-ice/current?format=grid');
      field = body.field;
      grid = body.grid;
      state.grid = grid;
      meta.textContent =
        `observed ${body.observed_at ? body.observed_at.slice(0, 10) : 'n/a'}\n` +
        `mean ${fmt(body.statistics.mean_concentration, 3)}, ` +
        `ice-covered ${pct(body.statistics.ice_covered_fraction)}\n` +
        `source: ${body.provenance.sources.join(', ') || 'n/a'}`;
    } else {
      const body = await api(`/sea-ice/forecast?forecast_hours=${choice}&format=grid`);
      field = body.field;
      grid = body.grid;
      state.grid = grid;
      const m = body.validation_metrics || {};
      meta.textContent =
        `valid ${body.valid_at.slice(0, 16).replace('T', ' ')} UTC\n` +
        `model ${body.algorithm}\n` +
        (m.rmse
          ? `held-out RMSE ${fmt(m.rmse, 4)} vs persistence ` +
            `${fmt(m.persistence_baseline && m.persistence_baseline.rmse, 4)} ` +
            `(skill ${fmt(m.skill_vs_persistence, 3)})`
          : 'untrained baseline - no validation metrics');
    }

    field.values.forEach((row, i) => {
      row.forEach((value, j) => {
        if (value === null || value < 0.05) return;
        cellRectangle(field.lats[i], field.lons[j], grid, {
          stroke: false,
          fillColor: iceColour(value),
          fillOpacity: 0.25 + 0.55 * value,
          interactive: false,
        }).addTo(state.layers.ice);
      });
    });
    if (document.getElementById('layer-ice').checked) {
      showLegend('Sea-ice concentration', iceColour);
    }
  } catch (err) {
    meta.innerHTML = `<span class="error">${err.message}</span>`;
  }
}

async function loadRisk() {
  const horizon = document.getElementById('route-horizon').value;
  const iceClass = document.getElementById('ice-class').value;
  state.layers.risk.clearLayers();
  try {
    const body = await api(
      `/risk/map?format=grid&forecast_hours=${horizon}&ice_class=${iceClass}&top_hotspots=5`
    );
    const grid = body.summary.grid;
    body.total_risk_field.forEach((row, i) => {
      row.forEach((value, j) => {
        if (value === null) return;
        cellRectangle(body.lats[i], body.lons[j], grid, {
          stroke: false,
          fillColor: riskColour(value),
          fillOpacity: 0.15 + 0.6 * value,
        })
          .bindTooltip(
            `risk ${value.toFixed(3)} at ${body.lats[i].toFixed(1)}, ${body.lons[j].toFixed(1)}`
          )
          .addTo(state.layers.risk);
      });
    });
    body.hotspots.forEach((h) => {
      L.circleMarker([h.latitude, h.longitude], {
        radius: 6, color: '#7a1f1f', weight: 2, fillOpacity: 0,
      })
        .bindPopup(
          `<b>Risk hotspot</b><br>total ${h.total_risk.toFixed(3)}<br>` +
          `driven by ${h.dominant_component.replace('_', ' ')}`
        )
        .addTo(state.layers.risk);
    });
  } catch (err) {
    console.error(err);
  }
}

async function loadIcebergs() {
  state.layers.bergs.clearLayers();
  try {
    const [list, tracks] = await Promise.all([
      api('/icebergs'),
      api('/icebergs/trajectories?forecast_hours=72'),
    ]);
    const byId = {};
    tracks.trajectories.forEach((t) => { byId[t.iceberg_id] = t; });

    list.icebergs.forEach((berg) => {
      const radius = Math.max(4, Math.min(11, 3 + Math.log10((berg.area_km2 || 10) + 1) * 2.4));
      L.circleMarker([berg.latitude, berg.longitude], {
        radius, color: '#0d3b52', weight: 1.4, fillColor: '#7fd4f5', fillOpacity: 0.9,
      })
        .bindPopup(
          `<b>${berg.iceberg_id}</b><br>` +
          `observed ${berg.observed_at.slice(0, 10)}<br>` +
          `area ${fmt(berg.area_km2, 1)} km&sup2;<br>` +
          `drift ${fmt(berg.drift_speed_m_s, 3)} m/s` +
          (berg.drift_bearing_deg !== null ? ` bearing ${fmt(berg.drift_bearing_deg, 0)}&deg;` : '') +
          `<br><small>${berg.source}</small>`
        )
        .addTo(state.layers.bergs);

      const track = byId[berg.iceberg_id];
      if (!track || !track.points.length) return;
      const line = [[berg.latitude, berg.longitude]].concat(
        track.points.map((p) => [p.latitude, p.longitude])
      );
      L.polyline(line, { color: '#0d3b52', weight: 1.6, dashArray: '4,3', opacity: 0.8 })
        .bindTooltip(`${berg.iceberg_id}: predicted 72 h drift`)
        .addTo(state.layers.bergs);
      const last = track.points[track.points.length - 1];
      if (last.uncertainty_radius_km > 0) {
        L.circle([last.latitude, last.longitude], {
          radius: last.uncertainty_radius_km * 1000,
          color: '#0d3b52', weight: 1, opacity: 0.5, fillOpacity: 0.05, dashArray: '3,4',
        })
          .bindTooltip(`+72 h ensemble spread: ${last.uncertainty_radius_km.toFixed(1)} km`)
          .addTo(state.layers.bergs);
      }
    });
  } catch (err) {
    console.error(err);
  }
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------
const ROUTE_COLOURS = { shortest: '#1f6fb2', safest: '#1f8a4c', polaris: '#c8631a' };

async function optimise() {
  const button = document.getElementById('optimize');
  const status = document.getElementById('route-status');
  button.disabled = true;
  status.textContent = 'computing...';

  const body = {
    start: { station: document.getElementById('start').value },
    destination: { station: document.getElementById('destination').value },
    vessel: {
      ice_class: document.getElementById('ice-class').value,
      speed_knots: Number(document.getElementById('speed').value),
      fuel_consumption_tpd: Number(document.getElementById('fuel').value),
      risk_tolerance: Number(document.getElementById('tolerance').value),
    },
    preferences: {
      forecast_hours: Number(document.getElementById('route-horizon').value),
      profiles: ['shortest', 'safest', 'polaris'],
    },
    persist: true,
  };

  try {
    const result = await postJSON('/route/optimize', body);
    state.routes = result.routes;
    renderRoutes(result);
    status.textContent =
      `request ${result.request_id} - graph ${result.graph.nodes} nodes / ` +
      `${result.graph.edges} edges`;
  } catch (err) {
    status.innerHTML = `<span class="error">${err.message}</span>`;
    document.getElementById('route-results').innerHTML =
      `<span class="error">Route optimisation failed.</span>`;
  } finally {
    button.disabled = false;
  }
}

function renderRoutes(result) {
  state.layers.routes.clearLayers();
  state.routeLayers = {};
  const container = document.getElementById('route-results');
  container.innerHTML = '';

  const bounds = [];
  Object.entries(result.routes).forEach(([profile, route]) => {
    const colour = ROUTE_COLOURS[profile] || '#444';
    const coords = route.waypoints.map((w) => [w.latitude, w.longitude]);
    bounds.push(...coords);

    const line = L.polyline(coords, {
      color: colour,
      weight: profile === 'polaris' ? 4.5 : 3,
      opacity: profile === 'polaris' ? 0.95 : 0.75,
      dashArray: route.status === 'degraded' ? '8,5' : null,
    })
      .bindTooltip(
        `${profile}: ${route.distance_km.toFixed(0)} km, ` +
        `${route.duration_hours.toFixed(0)} h, risk ${route.mean_risk.toFixed(3)}`
      )
      .addTo(state.layers.routes);
    state.routeLayers[profile] = line;

    const comparison = (result.comparison && result.comparison[profile]) || {};
    const card = document.createElement('div');
    card.className = 'route-card';
    card.style.borderLeftColor = colour;
    card.innerHTML =
      `<h3>${profile}${profile === result.recommended_profile ? ' &middot; recommended' : ''}</h3>` +
      '<dl>' +
      row('distance', `${route.distance_km.toFixed(0)} km`) +
      row('duration', `${route.duration_hours.toFixed(0)} h`) +
      row('fuel (est.)', `${route.estimated_fuel_tonnes.toFixed(0)} t`) +
      row('mean risk', route.mean_risk.toFixed(3)) +
      row('max risk', route.max_risk.toFixed(3)) +
      row('max ice conc.', route.max_sea_ice_concentration.toFixed(2)) +
      (comparison.distance_vs_shortest_pct !== undefined
        ? row('vs shortest', `${signed(comparison.distance_vs_shortest_pct)}% dist, ` +
                             `${signed(comparison.mean_risk_vs_shortest_pct)}% risk`)
        : '') +
      (route.risk_assessment && route.risk_assessment.closest_iceberg_km != null
        ? row('closest berg', `${route.risk_assessment.closest_iceberg_km.toFixed(0)} km`)
        : '') +
      '</dl>' +
      (route.notes && route.notes.length
        ? `<div class="note">${route.notes.join('<br>')}</div>`
        : '');
    card.addEventListener('click', () => selectRoute(profile, card));
    container.appendChild(card);
  });

  [result.start, result.destination].forEach((point, index) => {
    L.marker([point.latitude, point.longitude])
      .bindPopup(`<b>${index === 0 ? 'Start' : 'Destination'}</b><br>${point.name || ''}`)
      .addTo(state.layers.routes);
    bounds.push([point.latitude, point.longitude]);
  });

  if (bounds.length) state.map.fitBounds(bounds, { padding: [30, 30] });
}

function selectRoute(profile, card) {
  document.querySelectorAll('.route-card').forEach((c) => c.classList.remove('active'));
  card.classList.add('active');
  Object.entries(state.routeLayers).forEach(([name, layer]) => {
    layer.setStyle({ opacity: name === profile ? 1 : 0.25, weight: name === profile ? 5.5 : 2.5 });
  });
  state.map.fitBounds(state.routeLayers[profile].getBounds(), { padding: [30, 30] });
}

const row = (label, value) => `<dt>${label}</dt><dd>${value}</dd>`;
const signed = (v) => (v === undefined || v === null ? 'n/a' : `${v > 0 ? '+' : ''}${v.toFixed(1)}`);
const fmt = (v, digits) => (v === null || v === undefined ? 'n/a' : Number(v).toFixed(digits));
const pct = (v) => (v === null || v === undefined ? 'n/a' : `${(v * 100).toFixed(0)}%`);

// ---------------------------------------------------------------------------
// Live station conditions
// ---------------------------------------------------------------------------
const HAZARD_LABEL = {
  none: 'no freezing spray',
  light: 'light freezing spray',
  moderate: 'moderate freezing spray',
  severe: 'SEVERE freezing spray',
};

function windArrow(deg) {
  // Meteorological direction is where the wind comes FROM; the arrow shows
  // where it is going, which is what a navigator reads off a chart.
  if (deg === null || deg === undefined) return '';
  const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  return dirs[Math.round(deg / 45) % 8];
}

async function loadStationWeather() {
  const el = document.getElementById('stations');
  state.layers.stations.clearLayers();
  try {
    const body = await api('/weather/stations?forecast_hours=24');
    el.className = '';
    el.innerHTML = body.stations.map((s) => {
      const marine = s.significant_wave_height_m !== null
        ? row('sea state', `${fmt(s.significant_wave_height_m, 1)} m, SST ${fmt(s.sea_surface_temperature_c, 1)}&deg;C`)
        : row('sea state', '<span title="waves are not defined under sea ice">under ice</span>');
      const hazard = s.freezing_spray_risk
        ? `<div class="hazard hz-${s.freezing_spray_risk}">${HAZARD_LABEL[s.freezing_spray_risk]}</div>`
        : '';
      return `<div class="stn">
        <h3>${s.display_name}</h3>
        <div class="when">valid ${String(s.observed_at).slice(0, 16).replace('T', ' ')} UTC</div>
        <dl>
          ${row('air temp', `${fmt(s.air_temperature_c, 1)} &deg;C`)}
          ${row('wind', `${fmt(s.wind_speed_m_s, 1)} m/s from ${windArrow(s.wind_direction_deg)} (F${s.beaufort_force === null ? '-' : s.beaufort_force})`)}
          ${row('pressure', `${fmt(s.mean_sea_level_pressure_hpa, 0)} hPa`)}
          ${marine}
        </dl>
        ${hazard}
      </div>`;
    }).join('');

    body.stations.forEach((s) => {
      const cold = s.air_temperature_c !== null && s.air_temperature_c < -10;
      L.circleMarker([s.latitude, s.longitude], {
        radius: 7, color: '#0f2b3d', weight: 2,
        fillColor: cold ? '#bcd4e6' : '#f0c98a', fillOpacity: 0.95,
      })
        .bindPopup(
          `<b>${s.display_name}</b><br>` +
          `<small>${s.operator || ''}</small><br>` +
          `valid ${String(s.observed_at).slice(0, 16).replace('T', ' ')} UTC<br>` +
          `${fmt(s.air_temperature_c, 1)} &deg;C, wind ${fmt(s.wind_speed_m_s, 1)} m/s ` +
          `from ${windArrow(s.wind_direction_deg)}<br>` +
          `${fmt(s.mean_sea_level_pressure_hpa, 0)} hPa` +
          (s.significant_wave_height_m !== null
            ? `<br>Hs ${fmt(s.significant_wave_height_m, 1)} m` : '') +
          `<br><small>${s.source}</small>`
        )
        .addTo(state.layers.stations);
    });
  } catch (err) {
    el.innerHTML = `<span class="error">${err.message}</span>`;
  }
}

async function loadFreshness() {
  const el = document.getElementById('freshness');
  try {
    const body = await api('/weather/current');
    const cls = (h) => (h === null ? '' : h < 3 ? 'age-live' : h < 48 ? 'age-recent' : 'age-stale');
    const label = (h) => {
      if (h === null || h === undefined) return 'no data';
      if (h < 1) return 'live';
      if (h < 48) return `${h.toFixed(0)} h ago`;
      return `${(h / 24).toFixed(1)} days ago`;
    };
    el.className = '';
    el.innerHTML = Object.entries(body.layers).map(([name, l]) =>
      `<div class="fresh"><span>${name.replace('_', ' ')}</span>` +
      `<span class="age ${cls(l.age_hours)}">${label(l.age_hours)}</span></div>`
    ).join('');
  } catch (err) {
    el.innerHTML = `<span class="error">${err.message}</span>`;
  }
}

// ---------------------------------------------------------------------------
// Status panels
// ---------------------------------------------------------------------------
async function loadHealth() {
  const badge = document.getElementById('health-badge');
  try {
    const body = await api('/health');
    badge.textContent = `backend: ${body.status} v${body.version}`;
    badge.className = `badge ${body.status === 'ok' ? 'ok' : 'warn'}`;
    const mode = document.getElementById('mode-badge');
    mode.textContent = `data mode: ${body.data_mode}`;
    mode.className = `badge ${body.data_mode}`;
  } catch (err) {
    badge.textContent = 'backend: unreachable';
    badge.className = 'badge bad';
  }
}

async function loadStations() {
  const body = await api('/navigation/stations');
  state.stations = body.stations;
  const start = document.getElementById('start');
  const destination = document.getElementById('destination');
  body.stations
    .filter((s) => s.in_domain)
    .forEach((station) => {
      [start, destination].forEach((select) => {
        const option = document.createElement('option');
        option.value = station.key;
        option.textContent = station.name;
        select.appendChild(option);
      });
    });
  start.value = 'bharati';
  destination.value = 'maitri';
}

async function loadDashboard() {
  const el = document.getElementById('dashboard');
  try {
    const body = await api('/dashboard/summary');
    const lines = [
      ['sea-ice observed', body.sea_ice.observed_at ? body.sea_ice.observed_at.slice(0, 10) : 'n/a'],
      ['ice-covered area', pct(body.sea_ice.ice_covered_fraction)],
      ['hemispheric extent', `${fmt(body.sea_ice.hemispheric_extent_million_km2, 2)} M km²`],
      ['icebergs tracked', body.icebergs.tracked_count],
      ['mean risk', fmt(body.risk.mean_risk, 3)],
      ['navigable cells', `${body.risk.cells_navigable} / ${body.risk.cells_navigable + body.risk.cells_blocked}`],
      ['graph nodes', body.routing.graph_nodes],
    ];
    Object.entries(body.forecast.horizons).forEach(([hours, info]) => {
      lines.push([
        `forecast +${hours} h`,
        info.status === 'ok' ? `skill ${fmt(info.skill_vs_persistence, 3)}` : info.status,
      ]);
    });
    if (body.risk.missing_layers && body.risk.missing_layers.length) {
      lines.push(['missing layers', body.risk.missing_layers.join(', ')]);
    }
    el.className = '';
    el.innerHTML = lines
      .map(([k, v]) => `<div class="dash-line"><span>${k}</span><span>${v}</span></div>`)
      .join('');
    document.getElementById('disclaimer').textContent = body.disclaimer;
  } catch (err) {
    el.innerHTML = `<span class="error">${err.message}</span>`;
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
function bindLayerToggle(id, name, onEnable) {
  document.getElementById(id).addEventListener('change', async (event) => {
    if (event.target.checked) {
      state.layers[name].addTo(state.map);
      if (onEnable) await onEnable();
    } else {
      state.map.removeLayer(state.layers[name]);
    }
    updateLegend();
  });
}

function updateLegend() {
  if (document.getElementById('layer-risk').checked) {
    showLegend('Navigation risk (0-1)', riskColour);
  } else if (document.getElementById('layer-ice').checked) {
    showLegend('Sea-ice concentration', iceColour);
  } else {
    document.getElementById('legend').classList.add('hidden');
  }
}

async function boot() {
  initMap();
  bindLayerToggle('layer-land', 'land');
  bindLayerToggle('layer-ice', 'ice');
  bindLayerToggle('layer-risk', 'risk', loadRisk);
  bindLayerToggle('layer-bergs', 'bergs');
  bindLayerToggle('layer-stations', 'stations');
  document.getElementById('layer-tiles')
    .addEventListener('change', (e) => toggleTiles(e.target.checked));
  document.getElementById('ice-source').addEventListener('change', loadSeaIce);
  document.getElementById('optimize').addEventListener('click', optimise);
  document.getElementById('route-horizon').addEventListener('change', () => {
    if (document.getElementById('layer-risk').checked) loadRisk();
  });
  document.getElementById('tolerance').addEventListener('input', (e) => {
    document.getElementById('tolerance-value').textContent = Number(e.target.value).toFixed(2);
  });

  await loadHealth();
  updateLegend();
  await Promise.all([
    loadLandmask().catch(console.error),
    loadStations().catch(console.error),
    loadSeaIce(),
    loadIcebergs(),
    loadStationWeather(),
    loadFreshness(),
    loadDashboard(),
  ]);

  // Live conditions age in real time; refresh them without reloading the page.
  setInterval(() => { loadStationWeather(); loadFreshness(); }, 10 * 60 * 1000);
}

document.addEventListener('DOMContentLoaded', boot);
