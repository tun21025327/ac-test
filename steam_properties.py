"""Shared physics constants/helpers for the steam-cycle metric scripts.

Saturated-steam density comes from IAPWS-IF97 (via the `iapws` package) rather
than an ad-hoc approximation, so every script that needs "how much steam mass
does a pressure change represent" agrees with every other one.
"""
import numpy as np
from iapws import IAPWS97

ATM_BAR = 1.01325  # bar — used to convert gauge pressure to absolute

# Autoclave free (steam-space) chamber volumes, m^3. AC8 is a larger unit than
# AC1-7. Carried over from prior analysis (see leakage kg/h chart footnote:
# "V_free = 85.09 m^3 (AC1-7), 154.84 m^3 (AC8)") — if these are wrong, every
# mass-based metric below scales linearly with them, so confirm against the
# actual vessel spec sheet if available.
V_FREE = {ac: 154.84 if ac == 8 else 85.09 for ac in range(1, 9)}


def rho_sat_vapor_single(p_gauge_bar):
    """Saturated water-vapor density (kg/m^3) at a given GAUGE pressure (bar).
    Below atmospheric (vacuum, gauge < 0) still resolves to a valid absolute
    pressure as long as it's above the triple point; clipped defensively."""
    p_abs_mpa = max((p_gauge_bar + ATM_BAR) / 10.0, 1e-4)  # bar -> MPa, floor near triple point
    return IAPWS97(P=p_abs_mpa, x=1.0).rho


_rho_cache = {}

def rho_sat_vapor(p_gauge_bar):
    """Vectorized, cached (IAPWS97 calls are not cheap) — accepts a scalar or
    array of gauge pressures in bar, returns kg/m^3 of the same shape."""
    scalar_input = np.isscalar(p_gauge_bar)
    arr = np.atleast_1d(np.asarray(p_gauge_bar, dtype=float))
    out = np.empty_like(arr)
    for i, p in enumerate(arr):
        key = round(float(p), 3)  # 1 mbar resolution is plenty for this use
        if key not in _rho_cache:
            _rho_cache[key] = rho_sat_vapor_single(key)
        out[i] = _rho_cache[key]
    return float(out[0]) if scalar_input else out
