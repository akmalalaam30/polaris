# Notebooks

Exploratory space. Nothing in the POLARIS pipeline depends on anything here -
the backend is driven entirely by `backend/scripts/` and the API, so a notebook
can be deleted without affecting the system.

## Getting a session set up

```python
import sys, os
sys.path.insert(0, "../backend")
os.environ.setdefault("DATA_MODE", "demo")

from app.config import get_settings
from app.utils.geo import grid_from_settings
from app.database.connection import session_scope
from app.database import repositories as repo

settings = get_settings()
grid = grid_from_settings(settings)
```

## Useful entry points

```python
# The gridded history archive the forecast model trains on
from app.services.preprocessing import load_history
times, variables, grid, attrs = load_history(settings)
ice = variables["sea_ice_concentration"]          # (time, lat, lon)

# A forecast, with its real held-out validation metrics attached
from app.services.sea_ice_forecasting import forecast
result = forecast(24, settings, grid)
result.metrics["skill_vs_persistence"]

# Iceberg drift for one berg
from app.services import iceberg_trajectory as itraj

# The current risk grid
from app.services.environment import get_risk_grid
with session_scope() as session:
    risk = get_risk_grid(session, settings, grid, horizon_hours=24)
risk.summary()
```

`matplotlib` is listed (commented) in `backend/requirements.txt`; uncomment it
if you want to plot here.

## A caution

If you are working in demo mode, everything you plot is **synthetic**. Label any
figure you take out of a notebook accordingly - `settings.data_mode` tells you
which mode produced it.
