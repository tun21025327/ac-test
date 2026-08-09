"""Shared helpers for the per-metric analysis scripts (leakage, condensate,
vacuum depth/speed, transfer efficiency, fresh-vs-transferred, ...).

Keeping this logic in one place is the whole point of the fix: previously
each chart was regenerated from a natural-language description in a fresh
chat, so segment-extraction/filtering rules (e.g. "ignore idle windows
shorter than N minutes before fitting a slope") silently differed between
charts and even between re-runs of the "same" chart. Import from here so
every script that says "per idle segment" or "per cycle" means the same
thing.
"""
import numpy as np
import stage_classifier as sc

AUTOCLAVE_LABELS = [f"AC{i}" for i in range(1, sc.N_AUTOCLAVES + 1)]


def stage_runs(stages, label, t=None, min_duration_h=None):
    """contiguous_runs() for one stage label, optionally dropping runs
    shorter than min_duration_h. Short segments dominate noisy per-segment
    stats (a 2-minute 'idle' sliver gives a wildly different slope/count
    than a 50-minute one) — filtering them out is what several of the old
    charts were missing, and is the direct cause of their huge cycle-to-
    cycle SD."""
    runs = sc.contiguous_runs(stages == label)
    if min_duration_h is not None and t is not None:
        runs = [(s, e) for (s, e) in runs if (t[e - 1] - t[s]) >= min_duration_h]
    return runs


def linear_fit_slope(t, p, s, e):
    """bar/h (or any p-unit per h) slope of a raw linear fit over [s, e)."""
    if e - s < 2:
        return np.nan
    coeffs = np.polyfit(t[s:e], p[s:e], 1)
    return coeffs[0]


def mean_sd(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, 0
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0), len(arr)


def vacuum_runs(stages, t=None, min_duration_h=None):
    """contiguous_runs() for 'vacuum', each extended by one sample past its
    end. classify_trace() assigns the 'vacuum' label to [vac_start, vac_min)
    — the true minimum-pressure sample AT vac_min belongs to the *next*
    stage (transfer_in1), so a plain contiguous_runs() slice would exclude
    the deepest point of the dip entirely. Extending by 1 sample restores it
    for depth/speed metrics without changing the stage boundaries themselves."""
    runs = sc.contiguous_runs(stages == 'vacuum')
    n = len(stages)
    runs = [(s, min(e + 1, n)) for (s, e) in runs]
    if min_duration_h is not None and t is not None:
        runs = [(s, e) for (s, e) in runs if (t[e - 1] - t[s]) >= min_duration_h]
    return runs


def idle_leakage_stats(result, min_duration_h=0.25, max_abs_bar_per_h=2.0):
    """Per-AC idle-phase leakage: bar/h (raw slope) and kg/h (real mass rate,
    duration-normalized). Shared by idle_leakage.py and any comparison script
    so 'idle leakage' means the same computation everywhere it's used — see
    idle_leakage.py's docstring for why segments shorter than min_duration_h
    are dropped and why kg/h divides by the segment's real duration.

    max_abs_bar_per_h drops segments whose slope exceeds this magnitude —
    real idle drift never gets close to it (the largest genuine value seen
    across two full datasets is ~0.5 bar/h), so a segment this steep means
    classify_trace() mislabeled part of a vacuum/fresh_steam transition as
    'idle' (seen on the Jul 13-17 dataset: a 46-minute 'idle' run that went
    0.00 -> 12.84 bar, i.e. an entire cycle, not a leak). These are counted
    and returned as n_flagged rather than silently dropped.

    Returns bar_per_h, kg_per_h: {ac1: (mean, sd, n, n_flagged)}.
    """
    import steam_properties as sp
    bar_per_h, kg_per_h = {}, {}
    for ac0 in range(sc.N_AUTOCLAVES):
        ac1 = ac0 + 1
        t = result.times[ac0]
        p = result.pressures[ac0]
        stages = result.stages[ac0]

        runs = stage_runs(stages, 'idle', t=t, min_duration_h=min_duration_h)

        slopes, kg_rates, n_flagged = [], [], 0
        for (s, e) in runs:
            slope = linear_fit_slope(t, p, s, e)
            if max_abs_bar_per_h is not None and abs(slope) > max_abs_bar_per_h:
                n_flagged += 1
                continue
            slopes.append(slope)
            dur_h = t[e - 1] - t[s]
            d_rho = sp.rho_sat_vapor(p[e - 1]) - sp.rho_sat_vapor(p[s])
            kg_rates.append(d_rho * sp.V_FREE[ac1] / dur_h)

        m, sdv, n = mean_sd(slopes)
        bar_per_h[ac1] = (m, sdv, n, n_flagged)
        m2, sdv2, n2 = mean_sd(kg_rates)
        kg_per_h[ac1] = (m2, sdv2, n2, n_flagged)
    return bar_per_h, kg_per_h


def idle_pressure_fluctuation_stats(result, min_duration_h=0.25, max_abs_slope_bar_per_h=2.0):
    """Per-AC pressure FLUCTUATION during idle — the noise/ripple riding on
    top of the leak trend, as opposed to idle_leakage_stats' slope (the
    trend itself). For each idle run, fit the same linear trend
    idle_leakage_stats fits, then take the std of the residual (raw
    pressure minus that fit) as the segment's fluctuation magnitude —
    subtracting the trend first means a segment that's both leaking AND
    noisy doesn't get its fluctuation inflated just by the leak.

    Uses the same run selection (>= min_duration_h) and the same
    slope-based sanity flag as idle_leakage_stats (a whole mislabeled
    vacuum/fresh_steam transition, not real idle) so the two metrics are
    always computed over the identical set of segments.

    Returns bar_fluct: {ac1: (mean, sd, n, n_flagged)}, in bar (peak
    variability of the residual, via its standard deviation).
    """
    bar_fluct = {}
    for ac0 in range(sc.N_AUTOCLAVES):
        ac1 = ac0 + 1
        t = result.times[ac0]
        p = result.pressures[ac0]
        stages = result.stages[ac0]

        runs = stage_runs(stages, 'idle', t=t, min_duration_h=min_duration_h)

        fluct_vals, n_flagged = [], 0
        for (s, e) in runs:
            slope = linear_fit_slope(t, p, s, e)
            if max_abs_slope_bar_per_h is not None and abs(slope) > max_abs_slope_bar_per_h:
                n_flagged += 1
                continue
            coeffs = np.polyfit(t[s:e], p[s:e], 1)
            residual = p[s:e] - np.polyval(coeffs, t[s:e])
            fluct_vals.append(float(residual.std()))

        m, sdv, n = mean_sd(fluct_vals)
        bar_fluct[ac1] = (m, sdv, n, n_flagged)
    return bar_fluct


def idle_total_leakage_stats(result, min_duration_h=0.25, max_abs_bar_per_h=2.0):
    """Per-AC idle-phase leakage TOTALED over the whole loaded date range —
    as opposed to idle_leakage_stats, which reports the average RATE
    (kg/h) per segment. This sums each segment's raw mass change (not
    divided by duration) across every qualifying idle segment, so it
    answers "how much steam did this AC actually lose to idle leakage over
    the period", not "how fast does it leak on a typical idle".

    Same run selection (>= min_duration_h) and the same slope-based sanity
    flag as idle_leakage_stats, so a segment excluded there (a whole
    mislabeled vacuum/fresh_steam transition, not real idle) is excluded
    here too.

    Returns {ac1: (total_kg, n, n_flagged, avg_duration_h)} — total_kg is
    signed (negative = net loss, matching idle_leakage_stats' convention).
    """
    import steam_properties as sp
    totals = {}
    for ac0 in range(sc.N_AUTOCLAVES):
        ac1 = ac0 + 1
        t = result.times[ac0]
        p = result.pressures[ac0]
        stages = result.stages[ac0]

        runs = stage_runs(stages, 'idle', t=t, min_duration_h=min_duration_h)

        total_kg, durations, n_flagged = 0.0, [], 0
        for (s, e) in runs:
            slope = linear_fit_slope(t, p, s, e)
            if max_abs_bar_per_h is not None and abs(slope) > max_abs_bar_per_h:
                n_flagged += 1
                continue
            d_rho = sp.rho_sat_vapor(p[e - 1]) - sp.rho_sat_vapor(p[s])
            total_kg += d_rho * sp.V_FREE[ac1]
            durations.append(t[e - 1] - t[s])

        avg_dur, _, n = mean_sd(durations)
        totals[ac1] = (total_kg, n, n_flagged, avg_dur)
    return totals


def cycle_count(stages):
    """Number of completed cycles for one AC = number of vacuum runs."""
    return len(sc.contiguous_runs(stages == 'vacuum'))


def bar_labels(ax, bars, values, sds=None, fmt="{:.2f}", sd_fmt="SD={:.2f}"):
    """Consistent value(+-SD) labels above/below bars, matching the style
    used across the existing chart set (bold mean, smaller SD line).

    Uses a fixed point offset (via annotate) rather than an offset scaled to
    the largest bar's magnitude — with a fixed offset, a label never
    collides with the axis just because some other bar in the same chart is
    much bigger (e.g. one outlier autoclave dwarfing the rest)."""
    ax.margins(y=0.15)  # headroom so labels don't clip the frame
    for bar, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        y = bar.get_height()
        va = 'bottom' if y >= 0 else 'top'
        pts = 4 if y >= 0 else -4
        ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, y),
                    xytext=(0, pts), textcoords='offset points',
                    ha='center', va=va, fontsize=11, fontweight='bold')
    if sds is not None:
        for bar, v, sdv in zip(bars, values, sds):
            if not np.isfinite(v) or not np.isfinite(sdv):
                continue
            y = bar.get_height()
            va = 'bottom' if y >= 0 else 'top'
            pts = 18 if y >= 0 else -18
            ax.annotate(sd_fmt.format(sdv), (bar.get_x() + bar.get_width() / 2, y),
                        xytext=(0, pts), textcoords='offset points',
                        ha='center', va=va, fontsize=8.5)
