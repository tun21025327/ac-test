"""Canonical steam-cycle stage classifier, shared by every analysis script.

Extracted from doom.py so that every downstream chart (leakage, condensate,
transfer efficiency, vacuum depth/speed, fresh-vs-transferred steam, etc.)
classifies stages exactly the same way. Import this module instead of
re-deriving the classification logic — that is what was causing the same
chart to give different numbers across sessions: each was regenerated from
a natural-language description instead of re-running the same code.

Usage:
    import stage_classifier as sc
    dfs = sc.load_csv_files(SCRIPT_DIR, ["Q2_20260626.CSV", ...])
    result = sc.classify_all_autoclaves(dfs)
    result.times[ac]      # 0-indexed autoclave -> concatenated time(h) array
    result.pressures[ac]  # cleaned pressure array
    result.stages[ac]     # per-sample stage label array (post rotation-match)
    result.match_log      # rotation-matched transfer_out claim log
"""
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass, field

# ── Stage colours / labels (pre-rotation-split stages) ─────────────────────────
STAGE_COLORS = {
    'vacuum':        '#a8d8f0',
    'transfer_in1':  '#f5c542',
    'interval':      '#d9d9d9',
    'transfer_in2':  '#ffd980',
    'keeping':       '#b5e8b0',
    'idle':          '#c9b8e8',
    'fresh_steam':   '#ffb380',
    'transfer_out_steam_off': '#f08080',
}
STAGE_LABELS = {
    'vacuum':        'Vacuum (P < 0)',
    'transfer_in1':  'Transfer In 1 (~30 min)',
    'interval':      'Interval (pause in rise)',
    'transfer_in2':  'Transfer In 2 (~30 min)',
    'keeping':       'Keeping (~4h plateau)',
    'idle':          'Idle (minor fluctuation, ~1h)',
    'fresh_steam':   'Fresh Steam (rise to 12 bar)',
    'transfer_out_steam_off': 'Transfer Out + Steam Off',
}
STAGE_ORDER = ['vacuum', 'transfer_in1', 'interval', 'transfer_in2',
               'fresh_steam', 'keeping', 'idle', 'transfer_out_steam_off']

# ── Stage colours / labels AFTER rotation-matched transfer-out split ───────────
# (every downstream script should use these post-split names/colors so charts
#  are visually and semantically consistent with each other)
STAGE_COLORS_FULL = dict(STAGE_COLORS, **{
    'transfer_out1': '#e06666',   # matches a downstream AC's transfer_in2
    'transfer_out2': '#f4a582',   # matches a downstream AC's transfer_in1
    'steam_off':     '#8b3a3a',   # unclaimed / vented to waste
})
STAGE_LABELS_FULL = dict(STAGE_LABELS, **{
    'transfer_out1': 'Transfer Out 1 (-> partner TI2)',
    'transfer_out2': 'Transfer Out 2 (-> partner TI1)',
    'steam_off':     'Steam Off (wasted, unclaimed)',
})
STAGE_ORDER_FULL = ['vacuum', 'transfer_in1', 'interval', 'transfer_in2',
                    'fresh_steam', 'keeping', 'idle',
                    'transfer_out1', 'transfer_out2', 'steam_off']

# ── Thresholds (tunable) ───────────────────────────────────────────────────────
VACUUM_THRESHOLD     = -0.1    # used only to locate the dip region (for finding the minimum)
SMOOTH_WINDOW        = 15
TI1_DURATION         = 0.5     # 30 min nominal duration for transfer_in1
TI2_DURATION         = 0.5     # 30 min nominal duration for transfer_in2
TI2_TARGET_BAR       = 4.0     # nominal pressure where TI2 ends / fresh_steam begins
KEEPING_TARGET_BAR   = 12.0    # pressure must reach this to enter keeping — strict minimum, never lowered
KEEPING_FLAT_GRAD    = 1.0     # bar/h — below this, rise has stopped (interval or keeping)
KEEPING_NOM_DURATION = 4.0     # nominal keeping duration (h) — informs idle search start
IDLE_FLAT_GRAD       = 1.0     # bar/h — idle fluctuates but stays roughly flat
STEAM_OFF_KEEP_MIN   = 8.5     # below this on the descent → transfer_out+steam_off
STEAM_OFF_FINAL_BAR  = 1.0     # steam_off proper is only the final ~1 bar -> 0 bar band right before vacuum
N_AUTOCLAVES         = 8

FLOW_MIN_THRESHOLD   = 0.05    # Main Meter steam flow below this = "no flow"
FRESH_STEAM_MAX_EXTENSION = 200  # samples (~3.3h) — safety cap on the extension past 12 bar
MAX_FRESH_STEAM_BAR  = 12.6    # hard ceiling — extension never pushes past this pressure

MIN_CYCLE_GAP_HOURS  = 3.0     # two vacuum dips closer than this are same cycle

# ── Data-quality filter ─────────────────────────────────────────────────────────
PRESSURE_FLOOR = -1.0  # bar — samples below this are ignored/interpolated

# ── Known faulty sensors: AC4's gauge reads ~0.15-0.2 bar LOW vs. true pressure,
#    so every absolute pressure threshold is lowered by this much for AC4 only
#    (autoclave number 1-indexed -> bar to subtract from every threshold).
SENSOR_OFFSETS = {
    4: 0.2,
}
VACUUM_TOLERANCE = {ac: 0.05 for ac in range(1, N_AUTOCLAVES + 1) if ac != 4}
# AC4 gets no separate vacuum tolerance — its -0.2 bar already comes from
# SENSOR_OFFSETS above (a real sensor calibration issue, not stacked with
# this). Everyone else gets a -0.05 bar tolerance on the vacuum-start
# zero-crossing check only (does NOT shift keep_target/fresh/steam_off).

# ── Known 6h-product cycles (everything else defaults to 4h) ───────────────────
# Automatic detection (single-AC pressure jump, cross-AC divergence, total
# hold duration — several variants of each) was tried and abandoned: ground-
# truth data confirmed keeping and idle are NOT distinguishable from pressure
# at all (a known 6h cycle's pressure trace showed the same flat-plus-noise
# behavior from keeping-start all the way to its real descent 7.75h later,
# with no detectable transition anywhere in between). So instead of guessing,
# this is a direct lookup from operator-confirmed cycle logs (Jun26-Jul7):
# key = (autoclave 1-indexed, cycle_idx) where cycle_idx is that AC's 0-indexed
# position in chronological vacuum-cycle order (same indexing CYCLE_OVERRIDES
# uses) — value = true keeping duration in hours.
KEEPING_DURATION_OVERRIDES = {
    (1, 8): 6.0,
    (1, 13): 6.0,
    (2, 8): 6.0,
    (3, 8): 6.0,
    (4, 11): 6.0,
    (4, 12): 6.0,
    (5, 9): 6.0,
    (5, 11): 6.0,
    (5, 12): 6.0,
    (6, 9): 6.0,
    (6, 12): 6.0,
    (7, 11): 6.0,
    (7, 12): 6.0,
    (7, 14): 6.0,
    (8, 8): 6.0,
    (8, 11): 6.0,
    (8, 13): 6.0,
}

# ── Rotation table (from autoclave_rotation.jpg) ────────────────────────────────
# ti1_src / ti2_src: which autoclave THIS one's Transfer In 1 / 2 comes FROM.
ROTATION = {
    1: {'ti1_src': 3, 'ti2_src': 4},
    2: {'ti1_src': 4, 'ti2_src': 5},
    3: {'ti1_src': 5, 'ti2_src': 6},
    4: {'ti1_src': 6, 'ti2_src': 7},
    5: {'ti1_src': 7, 'ti2_src': 8},
    6: {'ti1_src': 8, 'ti2_src': 1},
    7: {'ti1_src': 1, 'ti2_src': 2},
    8: {'ti1_src': 2, 'ti2_src': 3},
}

DROP_MIN_BAR = 0.3   # minimum pressure drop across the window to count as "decreasing"
CLAIMABLE_STAGES = ('transfer_out_steam_off', 'idle')


def load_csv_files(script_dir, filenames):
    """Read each CSV, building a time(h) column from Time/HH:MM:SS if the
    file doesn't already have a precomputed one (raw exports differ)."""
    dfs = []
    for fname in filenames:
        df = pd.read_csv(os.path.join(script_dir, fname))
        if 'time(h)' not in df.columns:
            tod = pd.to_datetime(df['Time'], format='%H:%M:%S')
            df['time(h)'] = tod.dt.hour + tod.dt.minute / 60 + tod.dt.second / 3600
        dfs.append(df)
    return dfs


def clean_pressure(p, floor=PRESSURE_FLOOR):
    """Implausibly deep negative pressure (sensor glitches) below `floor` is
    treated as invalid and linearly interpolated from nearest valid neighbours."""
    p = p.astype(float).copy()
    bad = p < floor
    n_bad = int(bad.sum())
    if n_bad:
        good_idx = np.where(~bad)[0]
        bad_idx = np.where(bad)[0]
        if len(good_idx) > 0:
            p[bad_idx] = np.interp(bad_idx, good_idx, p[good_idx])
    return p, n_bad


def find_vacuum_regions(p, t_arr=None, min_samples=3, merge_gap=5, vacuum_threshold=None):
    """Negative-pressure regions = vacuum onset.

    Rules:
    - Only samples where p < VACUUM_THRESHOLD are ever marked vacuum.
    - merge_gap bridges brief noise INSIDE a dip (positive samples sandwiched
      between two negative runs) — never pulls boundary into positive territory.
    - One vacuum per cycle: if two dips are < MIN_CYCLE_GAP_HOURS apart,
      they belong to the same cycle — merge them and keep the deeper minimum.
    """
    if vacuum_threshold is None:
        vacuum_threshold = VACUUM_THRESHOLD
    is_vac = p < vacuum_threshold
    padded = is_vac.copy()
    for i in range(1, len(padded) - 1):
        if not padded[i]:
            look_back = is_vac[max(i - merge_gap, 0):i]
            look_fwd  = is_vac[i + 1:i + 1 + merge_gap]
            if np.any(look_back) and np.any(look_fwd):
                padded[i] = True

    raw_regions, in_vac, start = [], False, 0
    for i, v in enumerate(padded):
        if v and not in_vac:
            in_vac, start = True, i
        elif not v and in_vac:
            in_vac = False
            if i - start >= min_samples:
                raw_regions.append((start, i))
    if in_vac and len(p) - start >= min_samples:
        raw_regions.append((start, len(p)))

    if not raw_regions:
        return raw_regions

    def gap_hours(r1, r2):
        if t_arr is not None:
            return float(t_arr[r2[0]] - t_arr[r1[1]])
        return (r2[0] - r1[1]) / 60.0  # rough: assume 1 sample/min

    merged = [raw_regions[0]]
    for r in raw_regions[1:]:
        if gap_hours(merged[-1], r) < MIN_CYCLE_GAP_HOURS:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], r[1]))
        else:
            merged.append(r)

    return merged


def find_index_at_time(t, start_idx, offset_hours):
    """Returns index where t[i] - t[start_idx] >= offset_hours."""
    t0 = t[start_idx]
    for i in range(start_idx, len(t)):
        if t[i] - t0 >= offset_hours:
            return i
    return len(t) - 1


def contiguous_runs(mask):
    """Return list of (start, end) index pairs for contiguous True runs in mask."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    gaps = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], gaps + 1])
    ends   = np.concatenate([gaps + 1, [len(idx)]])
    return [(idx[s], idx[e - 1] + 1) for s, e in zip(starts, ends)]


def find_pump_on_runs(pump, merge_gap=3, min_samples=5):
    """Contiguous runs where the shared 'Vacuum pump' column reads 1 (the
    machine has exactly one vacuum pump serving all 8 autoclaves in
    rotation). Bridges brief 0-flicker gaps the same way find_vacuum_regions
    bridges noise in the pressure signal, and drops sub-5-minute blips."""
    is_on = pump.astype(bool)
    padded = is_on.copy()
    for i in range(1, len(padded) - 1):
        if not padded[i]:
            look_back = is_on[max(i - merge_gap, 0):i]
            look_fwd = is_on[i + 1:i + 1 + merge_gap]
            if np.any(look_back) and np.any(look_fwd):
                padded[i] = True
    runs = contiguous_runs(padded)
    return [(s, e) for (s, e) in runs if e - s >= min_samples]


def assign_vacuum_windows_from_pump(pump_on_runs, ac_pressure, n_autoclaves=N_AUTOCLAVES,
                                     sensor_offsets=None, extend_samples=10,
                                     neg_threshold=-0.15, start_max_bar=2.0,
                                     min_candidate_separation=5):
    """For each pump-on window, identify which autoclave(s) are actually
    being evacuated (pressure dropping from ~0 toward vacuum, e.g. -0.5 bar)
    and assign the window to them. Only one AC can be in vacuum at a time
    (one shared pump) — but a single detected pump-on window can still
    contain TWO back-to-back pulls for two different ACs if the real gap
    between them is shorter than find_pump_on_runs()'s noise-bridging
    tolerance (a single missing sample between two dips looks identical to
    a sensor flicker within one dip). Confirmed on real data: AC6 and AC7
    both showed genuine dips (-0.65 and -0.81 bar) inside one nominally
    54-minute pump-on run, separated by exactly one 0-sample in the raw
    signal — picking only the deeper of the two silently dropped AC6's
    entire cycle. So: find ALL qualifying candidates in the window, not
    just the best one, and split the window between them in time order.

    Matching rule per candidate: pressure at the window's start is below
    start_max_bar (excludes anything mid-keeping/idle/fresh_steam, which
    never sits anywhere near 0 bar) AND the minimum pressure within the
    window (extended a bit past pump-off, in case the true minimum lags
    slightly) reaches at or below neg_threshold (a genuine vacuum dip, not
    idle noise). Candidates whose minima land within min_candidate_separation
    samples of each other are treated as the same event (keep the deeper
    one) rather than split, to avoid over-splitting on near-duplicate noise.
    Sensor offsets are applied so AC4's known low-reading bias doesn't bias
    the comparison.

    Returns {ac_1indexed: [(vac_start_idx, vac_min_idx), ...]} sorted by time.
    """
    sensor_offsets = sensor_offsets or {}
    n = len(next(iter(ac_pressure.values())))
    windows = {ac1: [] for ac1 in range(1, n_autoclaves + 1)}
    unmatched = 0
    for (ps, pe) in pump_on_runs:
        search_end = min(pe + extend_samples, n)
        candidates = []  # (min_idx, ac1)
        for ac1 in range(1, n_autoclaves + 1):
            offset = sensor_offsets.get(ac1, 0.0)
            p_true = ac_pressure[ac1 - 1][ps:search_end] + offset
            if len(p_true) == 0:
                continue
            start_val = p_true[0]
            if start_val > start_max_bar:
                continue
            local_min_idx = int(np.argmin(p_true))
            min_val = p_true[local_min_idx]
            if min_val > neg_threshold:
                continue
            candidates.append((ps + local_min_idx, ac1, min_val))
        if not candidates:
            unmatched += 1
            continue

        candidates.sort()  # time order (by min_idx)
        merged = [candidates[0]]
        for cand in candidates[1:]:
            if cand[0] - merged[-1][0] < min_candidate_separation:
                if cand[2] < merged[-1][2]:  # deeper minimum wins the near-duplicate
                    merged[-1] = cand
            else:
                merged.append(cand)

        seg_start = ps
        for (min_idx, ac1, _) in merged:
            windows[ac1].append((seg_start, min_idx))
            seg_start = min_idx  # next candidate's window picks up where this dip bottomed out
    for ac1 in windows:
        windows[ac1].sort(key=lambda w: w[1])
    return windows, unmatched


def find_flow_on_span(flow, start_idx, cycle_end, threshold=FLOW_MIN_THRESHOLD,
                       merge_gap=3, search_limit=400, min_duration_samples=10):
    """First SUBSTANTIAL contiguous run of Main Meter steam flow > threshold
    starting at or after start_idx (brief 0-flicker gaps bridged the same
    way vacuum-pump and vacuum-region detection bridge noise). Runs shorter
    than min_duration_samples are skipped rather than returned — a brief
    residual flow blip right at the TI2/interval boundary (a few minutes)
    would otherwise be mistaken for the real ~25-35 min fresh-steam
    admission that actually follows it. Returns (span_start, span_end) or
    None if no qualifying run begins within
    [start_idx, min(start_idx+search_limit, cycle_end))."""
    limit = min(start_idx + search_limit, cycle_end, len(flow))
    if start_idx >= limit:
        return None
    is_on = flow[start_idx:limit] > threshold
    padded = is_on.copy()
    for i in range(1, len(padded) - 1):
        if not padded[i]:
            look_back = is_on[max(i - merge_gap, 0):i]
            look_fwd = is_on[i + 1:i + 1 + merge_gap]
            if np.any(look_back) and np.any(look_fwd):
                padded[i] = True
    runs = contiguous_runs(padded)
    for (s_rel, e_rel) in runs:
        if e_rel - s_rel >= min_duration_samples:
            return start_idx + s_rel, start_idx + e_rel
    return None


def classify_trace(p, t, HEAD_KEEPING_END_HOURS=None, cycle_overrides=None, flow=None,
                    other_ac_fresh_starts=None, pressure_offset=0.0, vacuum_tolerance=0.0,
                    external_vac_windows=None, keeping_duration_overrides=None):
    n = len(p)
    p = p.astype(float)  # ensure float ops; this IS the raw data, used throughout

    keep_target    = KEEPING_TARGET_BAR  - pressure_offset
    max_fresh      = MAX_FRESH_STEAM_BAR - pressure_offset
    steam_off_min  = STEAM_OFF_KEEP_MIN  - pressure_offset
    vac_threshold  = VACUUM_THRESHOLD    - pressure_offset

    stages = np.full(n, 'keeping', dtype=object)  # default fallback

    def local_grad(i):
        lo, hi = max(i - 1, 0), min(i + 1, n - 1)
        dt = t[hi] - t[lo]
        return (p[hi] - p[lo]) / dt if dt > 1e-9 else 0.0

    def find_flat_interval_end(start_idx, search_limit, pressure_cap):
        start_idx = min(start_idx, n - 1)
        end_idx = start_idx
        if p[start_idx] >= pressure_cap:
            return end_idx
        for i in range(start_idx, min(start_idx + search_limit, n - 1)):
            if abs(local_grad(i)) < KEEPING_FLAT_GRAD:
                end_idx = i + 1
            else:
                break
        return min(end_idx, n - 1)

    STALL_WINDOW_SAMPLES = 45   # ~54 min — if pressure barely moves over this
    STALL_RANGE_BAR = 0.8       # span within this counts as "stalled," not still climbing

    def find_fresh_steam_end(start_idx, search_limit, cycle_end):
        limit = min(start_idx + search_limit, cycle_end)
        reach_idx = None
        for i in range(start_idx, limit):
            if p[i] >= keep_target:
                reach_idx = i
                break
            if i - start_idx >= STALL_WINDOW_SAMPLES:
                w0 = i - STALL_WINDOW_SAMPLES
                window = p[w0:i + 1]
                if window.max() - window.min() < STALL_RANGE_BAR:
                    reach_idx = w0
                    break
        if reach_idx is None:
            return min(start_idx + search_limit, cycle_end)

        if flow is None:
            return reach_idx

        other_stop_time = None
        if other_ac_fresh_starts is not None and len(other_ac_fresh_starts) > 0:
            later = other_ac_fresh_starts[other_ac_fresh_starts > t[reach_idx]]
            if len(later) > 0:
                other_stop_time = later.min()

        end_idx = reach_idx
        limit = min(reach_idx + FRESH_STEAM_MAX_EXTENSION, cycle_end, n - 1)
        for i in range(reach_idx, limit):
            if p[i] > max_fresh:
                break
            if other_stop_time is not None and t[i] >= other_stop_time:
                break
            still_flowing = flow[i] > FLOW_MIN_THRESHOLD
            still_rising  = local_grad(i) >= KEEPING_FLAT_GRAD
            if still_flowing or still_rising:
                end_idx = i + 1
            else:
                break
        return min(end_idx, cycle_end)

    FRESH_STEAM_CLOSE_TO_TARGET = 1.5  # bar — how near keep_target counts as "this run finished the job"

    def find_fresh_steam_span(start_idx, cycle_end):
        """Fresh steam = the Main Meter flow turning on (0 -> nonzero) and
        back off (nonzero -> 0) — this directly brackets the physical
        fresh-steam admission, replacing the old pressure-target/stall
        heuristic. If the operator tops up with more live steam later
        (pressure dropped more than expected, after the AC has already
        settled near keep_target), that's a SEPARATE, later flow-on episode
        and must not be classified as fresh_steam.

        Real admissions are sometimes split into 2+ flow-on bursts with a
        genuine pause between them (valve modulation) before pressure
        actually gets near target — e.g. a short ~15-min burst that only
        reaches ~1 bar, then a pause, then the real ~30-min admission that
        finishes the climb to 12 bar. Treating the first (partial) burst as
        "the" fresh_steam run would leave the real climb sitting
        unclassified. So: keep advancing to the NEXT flow-on run as long as
        the current candidate's end pressure isn't yet close to keep_target
        — once one gets there, that's fresh_steam's real end. This is
        different from the operator-topup case because there pressure has
        already BEEN at/near target and later dropped, whereas here it
        never reached target in the first place.

        Falls back to the pressure-based find_fresh_steam_end() when flow
        isn't available (pass 1 baseline classification, called with
        flow=None) or no flow-on run reaching target is found in range."""
        if flow is not None:
            search_from = start_idx
            for _ in range(6):  # small bounded loop over successive bursts
                span = find_flow_on_span(flow, search_from, cycle_end)
                if span is None:
                    break
                fs, fe = span
                if p[fs] >= max_fresh:  # sanity: shouldn't already be past target
                    break
                if p[fe - 1] >= keep_target - FRESH_STEAM_CLOSE_TO_TARGET:
                    return fs, fe
                search_from = fe  # this burst didn't finish the job — look for the next one
        fe = find_fresh_steam_end(start_idx, 400, cycle_end)
        return start_idx, fe

    zero_bar_equiv = 0.0 - pressure_offset - vacuum_tolerance
    STEEP_DROP_GRAD = 4.0
    KINK_WINDOW = 6
    MAX_LOOKBACK_HOURS = 3.0

    def find_vacuum_start(vac_min, lower_bound):
        search_start = lower_bound
        cutoff_time = t[vac_min] - MAX_LOOKBACK_HOURS
        while search_start < vac_min and t[search_start] < cutoff_time:
            search_start += 1

        def window_slope(i):
            j = min(i + KINK_WINDOW, vac_min)
            if j <= i:
                return 0.0
            dt = t[j] - t[i]
            return (p[j] - p[i]) / dt if dt > 1e-9 else 0.0

        vac_start_idx = None
        for i in range(vac_min - 1, search_start - 1, -1):
            if window_slope(i) <= -STEEP_DROP_GRAD:
                vac_start_idx = i
            else:
                break
        if vac_start_idx is not None:
            return vac_start_idx

        vac_start_idx = search_start
        for i in range(vac_min, max(search_start - 1, -1), -1):
            if p[i] >= zero_bar_equiv:
                vac_start_idx = i + 1
                break
        return vac_start_idx

    if external_vac_windows is not None:
        # vacuum windows already determined from the 'Vacuum pump' column
        # (see assign_vacuum_windows_from_pump) — skip the pressure-slope
        # heuristic entirely and use these directly.
        vac_windows = list(external_vac_windows)
    else:
        vac_regions = find_vacuum_regions(p.copy(), t_arr=t, vacuum_threshold=vac_threshold)
        vac_windows = []
        for idx, (vs, ve) in enumerate(vac_regions):
            vac_min = vs + int(np.argmin(p[vs:ve]))
            lower_bound = 0 if idx == 0 else vac_regions[idx - 1][1]
            vac_start_idx = find_vacuum_start(vac_min, lower_bound)
            vac_windows.append((vac_start_idx, vac_min))

    # ── Head: classify data BEFORE the first vacuum (data may start mid-cycle) ─
    if vac_windows:
        first_vs = vac_windows[0][0]
        if first_vs > 0:
            head_descent_start = 0
            for i in range(first_vs - 1, -1, -1):
                if p[i] >= steam_off_min:
                    head_descent_start = i + 1
                    break
            else:
                head_descent_start = 0

            if head_descent_start < first_vs:
                stages[head_descent_start:first_vs] = 'transfer_out_steam_off'

            if head_descent_start > 0:
                head_peak = p[:head_descent_start].max()
                if HEAD_KEEPING_END_HOURS is None and head_peak < keep_target:
                    stages[0:head_descent_start] = 'fresh_steam'
                else:
                    if HEAD_KEEPING_END_HOURS is not None:
                        keeping_nom_end = find_index_at_time(t, 0, HEAD_KEEPING_END_HOURS)
                    else:
                        keeping_nom_end = find_index_at_time(t, 0, KEEPING_NOM_DURATION)
                    keeping_nom_end = min(keeping_nom_end, head_descent_start)
                    stages[0:keeping_nom_end] = 'keeping'
                    if keeping_nom_end < head_descent_start:
                        stages[keeping_nom_end:head_descent_start] = 'idle'

    for idx in range(len(vac_windows)):
        vac_start_idx, vac_min = vac_windows[idx]

        if idx + 1 < len(vac_windows):
            cycle_end = vac_windows[idx + 1][0]
        else:
            cycle_end = n

        stages[vac_start_idx:vac_min] = 'vacuum'

        override = (cycle_overrides or {}).get(idx)

        if override == 'no_transfer':
            fresh_start, fresh_end = find_fresh_steam_span(vac_min, cycle_end)
            if vac_min < fresh_start:
                stages[vac_min:fresh_start] = 'interval'
            if fresh_start < fresh_end:
                stages[fresh_start:fresh_end] = 'fresh_steam'
        elif override == 'single_transfer':
            ti1_end = min(find_index_at_time(t, vac_min, TI1_DURATION), cycle_end - 1)
            if vac_min < ti1_end:
                stages[vac_min:ti1_end] = 'transfer_in1'
            fresh_start, fresh_end = find_fresh_steam_span(ti1_end, cycle_end)
            if ti1_end < fresh_start:
                stages[ti1_end:fresh_start] = 'interval'
            if fresh_start < fresh_end:
                stages[fresh_start:fresh_end] = 'fresh_steam'
        else:
            ti1_end = min(find_index_at_time(t, vac_min, TI1_DURATION), cycle_end - 1)
            if vac_min < ti1_end:
                stages[vac_min:ti1_end] = 'transfer_in1'

            interval1_end = find_flat_interval_end(ti1_end, 200, keep_target)
            if ti1_end < interval1_end:
                stages[ti1_end:interval1_end] = 'interval'

            ti2_end = min(find_index_at_time(t, interval1_end, TI2_DURATION), cycle_end - 1)
            if interval1_end < ti2_end:
                stages[interval1_end:ti2_end] = 'transfer_in2'

            interval2_end = find_flat_interval_end(ti2_end, 200, keep_target)
            if ti2_end < interval2_end:
                stages[ti2_end:interval2_end] = 'interval'

            fresh_start, fresh_end = find_fresh_steam_span(interval2_end, cycle_end)
            if interval2_end < fresh_start:
                stages[interval2_end:fresh_start] = 'interval'
            if fresh_start < fresh_end:
                stages[fresh_start:fresh_end] = 'fresh_steam'

        if fresh_end >= cycle_end:
            continue

        descent_start = cycle_end
        for i in range(cycle_end - 1, fresh_end, -1):
            if p[i] >= steam_off_min:
                descent_start = i + 1
                break
        if descent_start < cycle_end:
            stages[descent_start:cycle_end] = 'transfer_out_steam_off'

        keeping_duration = (keeping_duration_overrides or {}).get(idx, KEEPING_NOM_DURATION)
        keeping_nom_end = find_index_at_time(t, fresh_end, keeping_duration)
        keeping_nom_end = min(keeping_nom_end, descent_start)

        if fresh_end < keeping_nom_end:
            stages[fresh_end:keeping_nom_end] = 'keeping'
        if keeping_nom_end < descent_start:
            stages[keeping_nom_end:descent_start] = 'idle'

    return stages


def _vacuum_windows_for(stages_len, p, t, sensor_offset, vacuum_tolerance):
    """Recompute vacuum windows the same way classify_trace does internally
    (needed again in step 4 of rotation-matching to find cycle_ends)."""
    vac_regions = find_vacuum_regions(p.copy(), t_arr=t)
    zero_bar_equiv = 0.0 - sensor_offset - vacuum_tolerance
    vac_windows = []
    for idx, (vs, ve) in enumerate(vac_regions):
        vac_min = vs + int(np.argmin(p[vs:ve]))
        lower_bound = 0 if idx == 0 else vac_regions[idx - 1][1]

        search_start = lower_bound
        cutoff_time = t[vac_min] - 3.0
        while search_start < vac_min and t[search_start] < cutoff_time:
            search_start += 1

        def window_slope(i, vac_min=vac_min):
            j = min(i + 6, vac_min)
            if j <= i:
                return 0.0
            dt = t[j] - t[i]
            return (p[j] - p[i]) / dt if dt > 1e-9 else 0.0

        vac_start_idx = None
        for i in range(vac_min - 1, search_start - 1, -1):
            if window_slope(i) <= -4.0:
                vac_start_idx = i
            else:
                break
        if vac_start_idx is None:
            vac_start_idx = search_start
            for i in range(vac_min, max(search_start - 1, -1), -1):
                if p[i] >= zero_bar_equiv:
                    vac_start_idx = i + 1
                    break
        vac_windows.append((vac_start_idx, vac_min))
    return vac_windows


@dataclass
class ClassificationResult:
    times: dict = field(default_factory=dict)       # ac(0-idx) -> concatenated time(h) array
    pressures: dict = field(default_factory=dict)    # ac(0-idx) -> cleaned pressure array
    stages: dict = field(default_factory=dict)        # ac(0-idx) -> stage label array (post rotation-match)
    flow: np.ndarray = None                            # shared Main Meter steam flow, same time axis
    match_log: list = field(default_factory=list)
    n_bad_by_ac: dict = field(default_factory=dict)   # ac(0-idx) -> count of interpolated glitch samples


def concat_per_ac_column(dfs, column_fmt, n_autoclaves=N_AUTOCLAVES):
    """Concatenate a per-autoclave column (e.g. 'Temperature Autocalve {i}')
    across days using the EXACT same day-offset scheme classify_all_autoclaves
    uses for time/pressure (offset += that day's own last time(h) value), so
    the result lines up sample-for-sample with ClassificationResult.times /
    .stages. Use this instead of re-deriving your own concatenation loop —
    a slightly different offset would silently misalign a metric against the
    stage boundaries it's supposed to be measured within."""
    out = {}
    for ac in range(n_autoclaves):
        col = column_fmt.format(i=ac + 1)
        chunks = [dfs[day][col].values.astype(float) for day in range(len(dfs))]
        out[ac] = np.concatenate(chunks)
    return out


def classify_all_autoclaves(dfs, n_autoclaves=N_AUTOCLAVES,
                             head_keeping_end_overrides=None,
                             cycle_overrides=None,
                             sensor_offsets=None,
                             vacuum_tolerance=None,
                             keeping_duration_overrides=None,
                             verbose=True):
    """Full pipeline: per-AC classify_trace (2-pass, flow-aware) + rotation-
    matched transfer-out splitting. This is the single source of truth every
    downstream script should call — do not re-derive this logic per script.

    head_keeping_end_overrides: {ac_1indexed: hours or None}
    cycle_overrides: {(ac_1indexed, cycle_0indexed): 'single_transfer'|'no_transfer'}
    sensor_offsets / vacuum_tolerance: override the module defaults if needed
    keeping_duration_overrides: {(ac_1indexed, cycle_0indexed): hours} — defaults
        to the module's KEEPING_DURATION_OVERRIDES (known 6h-product cycles);
        pass {} to disable, or your own dict to override with different data
    """
    head_keeping_end_overrides = head_keeping_end_overrides or {}
    cycle_overrides = cycle_overrides or {}
    sensor_offsets = SENSOR_OFFSETS if sensor_offsets is None else sensor_offsets
    vacuum_tolerance = VACUUM_TOLERANCE if vacuum_tolerance is None else vacuum_tolerance
    keeping_duration_overrides = KEEPING_DURATION_OVERRIDES if keeping_duration_overrides is None else keeping_duration_overrides

    all_pressures, all_times, all_flows, all_pump = [], [], [], []
    has_pump_col = all('Vacuum pump' in df.columns for df in dfs)
    for df in dfs:
        pressures = [df[f'Pressure Autocalve {i}'].values for i in range(1, n_autoclaves + 1)]
        all_pressures.append(pressures)
        all_times.append(df['time(h)'].values)
        all_flows.append(df['Main Meter steam flow'].values)
        if has_pump_col:
            all_pump.append(df['Vacuum pump'].values)

    combined_flow = np.concatenate(all_flows)
    combined_pump = np.concatenate(all_pump) if has_pump_col else None

    ac_time, ac_pressure, ac_overrides, ac_head_override, ac_keeping_overrides = {}, {}, {}, {}, {}
    n_bad_by_ac = {}
    for ac in range(n_autoclaves):
        c_time, c_pressure = [], []
        offset = 0
        for day in range(len(dfs)):
            t = all_times[day]
            p = all_pressures[day][ac]
            c_time.append(t + offset)
            c_pressure.append(p)
            offset += t[-1]
        ac_time[ac] = np.concatenate(c_time)
        raw_pressure = np.concatenate(c_pressure)
        cleaned_pressure, n_bad = clean_pressure(raw_pressure)
        ac_pressure[ac] = cleaned_pressure
        n_bad_by_ac[ac] = n_bad
        if n_bad and verbose:
            print(f"AC{ac + 1}: ignored {n_bad} sample(s) below {PRESSURE_FLOOR} bar (interpolated)")
        ac_head_override[ac] = head_keeping_end_overrides.get(ac + 1)
        ac_overrides[ac] = {
            cycle_idx: mode
            for (ac1, cycle_idx), mode in cycle_overrides.items()
            if ac1 == ac + 1
        }
        ac_keeping_overrides[ac] = {
            cycle_idx: hours
            for (ac1, cycle_idx), hours in keeping_duration_overrides.items()
            if ac1 == ac + 1
        }

    # ── Vacuum windows: from the shared 'Vacuum pump' column when present,
    #    matched to whichever AC's pressure is actually dropping toward
    #    vacuum in each pump-on window (only one AC can be in vacuum at a
    #    time). Falls back to the pressure-slope heuristic per-AC inside
    #    classify_trace() if the column isn't in this dataset. ─────────────
    ac_vac_windows = None
    if has_pump_col:
        pump_on_runs = find_pump_on_runs(combined_pump)
        ac_vac_windows, n_unmatched = assign_vacuum_windows_from_pump(
            pump_on_runs, ac_pressure, n_autoclaves=n_autoclaves, sensor_offsets=sensor_offsets)
        if verbose:
            total_matched = sum(len(v) for v in ac_vac_windows.values())
            print(f"Vacuum pump column: {len(pump_on_runs)} pump-on runs -> "
                  f"{total_matched} matched to an AC, {n_unmatched} unmatched")
            for ac in range(n_autoclaves):
                if len(ac_vac_windows.get(ac + 1, [])) == 0:
                    print(f"  WARNING: AC{ac + 1} got 0 vacuum windows from the pump column")

    def external_windows_for(ac):
        return ac_vac_windows[ac + 1] if ac_vac_windows is not None else None

    # ── Pass 1: baseline classification (no flow-based extension) just to find
    #            each autoclave's own fresh_steam start times ─────────────────
    baseline_fresh_starts = {ac: [] for ac in range(n_autoclaves)}
    for ac in range(n_autoclaves):
        baseline_stage = classify_trace(ac_pressure[ac], ac_time[ac],
                                         HEAD_KEEPING_END_HOURS=ac_head_override[ac],
                                         cycle_overrides=ac_overrides[ac],
                                         flow=None,
                                         pressure_offset=sensor_offsets.get(ac + 1, 0.0),
                                         vacuum_tolerance=vacuum_tolerance.get(ac + 1, 0.0),
                                         external_vac_windows=external_windows_for(ac),
                                         keeping_duration_overrides=ac_keeping_overrides[ac])
        prev = None
        for i, st in enumerate(baseline_stage):
            if st == 'fresh_steam' and prev != 'fresh_steam':
                baseline_fresh_starts[ac].append(ac_time[ac][i])
            prev = st
        baseline_fresh_starts[ac] = np.array(baseline_fresh_starts[ac])

    # ── Pass 2: final classification, with flow-based extension that respects
    #            every OTHER autoclave's fresh_steam start times ──────────────
    combined_times, combined_pressures, combined_stages = [], [], []
    for ac in range(n_autoclaves):
        other_starts = np.concatenate([
            baseline_fresh_starts[other] for other in range(n_autoclaves) if other != ac
        ]) if n_autoclaves > 1 else np.array([])
        other_starts.sort()

        full_stage = classify_trace(ac_pressure[ac], ac_time[ac],
                                     HEAD_KEEPING_END_HOURS=ac_head_override[ac],
                                     cycle_overrides=ac_overrides[ac],
                                     flow=combined_flow,
                                     other_ac_fresh_starts=other_starts,
                                     pressure_offset=sensor_offsets.get(ac + 1, 0.0),
                                     vacuum_tolerance=vacuum_tolerance.get(ac + 1, 0.0),
                                     external_vac_windows=external_windows_for(ac),
                                     keeping_duration_overrides=ac_keeping_overrides[ac])

        combined_times.append(ac_time[ac])
        combined_pressures.append(ac_pressure[ac])
        combined_stages.append(full_stage)

    # ── Rotation-matched transfer-out splitting ─────────────────────────────
    # ROTATION gives the DEFAULT donor for each receiver's TI1/TI2 slot, but
    # operators sometimes swap that pairing on the fly to resolve a schedule
    # conflict (two ACs both wanting the same partner at once). So the
    # default donor is tried first; only when it fails the same
    # overlap/drop checks as before (poor overlap = schedule drift, or flat
    # = no real transfer happened there) do we search every OTHER AC's
    # pressure trace over that exact same time window for whoever actually
    # shows a matching decreasing trend, and use that AC instead.
    ti1_runs, ti2_runs = {}, {}
    for ac in range(n_autoclaves):
        ti1_runs[ac + 1] = contiguous_runs(combined_stages[ac] == 'transfer_in1')
        ti2_runs[ac + 1] = contiguous_runs(combined_stages[ac] == 'transfer_in2')

    # Slot mapping is physical, not arbitrary: transfer_in1 draws from a
    # donor's transfer_out2 (the donor's SECOND, smaller dregs-only
    # discharge), and transfer_in2 draws from transfer_out1 (the donor's
    # FIRST, bigger discharge from keeping level). That pairing itself
    # (which label to search for) is fixed below; a pressure-target check
    # on top of it was tried and reverted — real end pressures vary too
    # much cycle to cycle (confirmed on a visually-verified real match
    # ending at ~5 bar for what should nominally be a ~1 bar transfer_out2)
    # for a fixed target/tolerance to be a reliable filter. The largest
    # drop among overlap/drop-qualifying candidates remains the deciding
    # signal for a real transfer.
    def candidate_fit(y, s, e):
        """overlap/drop of donor y's own window [s,e) against the
        claimable-stage + decreasing-trend criteria."""
        stages_y = combined_stages[y - 1]
        p_y = combined_pressures[y - 1].astype(float)
        window = stages_y[s:e]
        claimable_mask = np.isin(window, CLAIMABLE_STAGES)
        overlap = np.mean(claimable_mask) if len(window) else 0.0
        drop = float(p_y[s] - p_y[e - 1]) if e > s else 0.0
        return overlap, drop, claimable_mask

    claims = {ac + 1: [] for ac in range(n_autoclaves)}
    match_log = []
    for x in range(1, n_autoclaves + 1):
        p_x = combined_pressures[x - 1].astype(float)
        for label, runs, rotation_key in (('transfer_out2', ti1_runs[x], 'ti1_src'),
                                           ('transfer_out1', ti2_runs[x], 'ti2_src')):
            primary_y = ROTATION[x][rotation_key]
            for (s, e) in runs:
                # Don't force ANY pairing (rotation-default or swapped)
                # unless the RECEIVER itself actually shows a matching
                # rise — classify_trace() labels a fixed ~30 min
                # transfer_in1/2 window every cycle by default regardless
                # of whether a real transfer happened (no_transfer /
                # single_transfer cycles need an explicit cycle_overrides
                # entry to suppress it), so without this check the
                # swapped-donor search could coincidentally "find" some
                # unrelated AC's real transfer happening at the same time
                # and wrongly attribute it here.
                recv_rise = float(p_x[e - 1] - p_x[s]) if e > s else 0.0
                if recv_rise < DROP_MIN_BAR:
                    match_log.append((primary_y, x, label, 0.0, recv_rise,
                                       'SKIPPED (receiver shows no rise - likely no real transfer this slot)'))
                    continue

                overlap, drop, claimable_mask = candidate_fit(primary_y, s, e)
                if overlap >= 0.5 and drop >= DROP_MIN_BAR:
                    claims[primary_y].append((s, e, label, x, claimable_mask, 'rotation', overlap, drop))
                    continue

                reason = 'poor overlap - schedule drift' if overlap < 0.5 else 'flat - no decreasing trend'
                best = None
                for y in range(1, n_autoclaves + 1):
                    if y in (x, primary_y):
                        continue
                    ov, dr, cm = candidate_fit(y, s, e)
                    if ov >= 0.5 and dr >= DROP_MIN_BAR:
                        if best is None or dr > best[2]:
                            best = (y, ov, dr, cm)
                if best:
                    y, ov, dr, cm = best
                    claims[y].append((s, e, label, x, cm, f'swapped (expected AC{primary_y}, {reason})', ov, dr))
                else:
                    match_log.append((primary_y, x, label, overlap, drop,
                                       f'SKIPPED ({reason}, no alternate donor found either)'))

    for y in range(1, n_autoclaves + 1):
        stages = combined_stages[y - 1]
        for (s, e, label, x, claimable_mask, source, overlap, drop) in claims[y]:
            window = stages[s:e]
            window[claimable_mask] = label
            match_log.append((y, x, label, overlap, drop, f'applied ({source})'))

    # ── Whatever's still 'transfer_out_steam_off' becomes 'idle'/'interval'/'steam_off' ─
    for ac in range(n_autoclaves):
        stages = combined_stages[ac]
        p = combined_pressures[ac].astype(float)
        t = combined_times[ac]
        if ac_vac_windows is not None:
            vac_windows = ac_vac_windows[ac + 1]
        else:
            vac_windows = _vacuum_windows_for(len(stages), p, t,
                                               sensor_offsets.get(ac + 1, 0.0),
                                               vacuum_tolerance.get(ac + 1, 0.0))

        # Adjacency rule (per cycle, i.e. between one vacuum start and the
        # next): idle only occurs before the first applied transfer_out
        # claim, steam_off is only the narrow final descent (~1 bar -> 0 bar)
        # right before vacuum, and everything else past the last applied
        # claim (or the whole post-keeping tail if this cycle has no
        # transfer at all) down to that point is interval — as is any gap
        # sandwiched between two applied claims (a TO1<->TO2 gap).
        # 'idle' is included as "unclaimed" too: classify_trace pre-labels
        # the region right after keeping (before descent_start) as 'idle'
        # ahead of rotation-matching, so a claim window can land entirely
        # inside that pre-label without ever touching 'transfer_out_steam_off'
        # — leftover 'idle' remnants need the same before/between/after
        # re-classification as leftover 'transfer_out_steam_off' remnants.
        #
        # Blocks are grouped by exact label value (not just "is one of the
        # relevant labels") — an applied claim sits with zero gap right next
        # to an unclaimed run, so grouping by relevance alone would merge
        # them into one block and let the unclaimed side's relabeling
        # clobber the real transfer_out1/transfer_out2 claim next to it.
        cycle_starts = sorted(vw[0] for vw in vac_windows)
        # Include the HEAD region (data before the very first captured
        # vacuum pull, when the file starts mid-cycle) as its own segment —
        # otherwise it's silently excluded from this relabeling entirely
        # (the loop below only walks from cycle_starts[0] onward), leaving
        # any unclaimed 'transfer_out_steam_off' there stuck with its raw
        # pre-split color, which looks like a second (bogus) transfer_out
        # block sitting right next to a real one.
        if cycle_starts and cycle_starts[0] > 0:
            cycle_starts = [0] + cycle_starts
        cycle_bounds = cycle_starts + [len(stages)]
        unclaimed = ('transfer_out_steam_off', 'idle')
        relevant = set(unclaimed) | {'transfer_out1', 'transfer_out2'}
        low_bar = STEAM_OFF_FINAL_BAR - sensor_offsets.get(ac + 1, 0.0)
        for i in range(len(cycle_starts)):
            cs, ce = cycle_starts[i], cycle_bounds[i + 1]
            blocks = []
            j = cs
            while j < ce:
                if stages[j] not in relevant:
                    j += 1
                    continue
                k = j
                while k < ce and stages[k] == stages[j]:
                    k += 1
                blocks.append((j, k))
                j = k
            is_claim = [stages[s] not in unclaimed for (s, e) in blocks]
            for bi, (s, e) in enumerate(blocks):
                if is_claim[bi]:
                    continue
                claim_before = any(is_claim[:bi])
                claim_after = any(is_claim[bi + 1:])
                if claim_after:
                    stages[s:e] = 'idle' if not claim_before else 'interval'
                    continue
                # past the last claim (or no claim at all this cycle): only
                # the final low-pressure band right before vacuum is real
                # steam_off — scan backward for the last sample still above
                # that band, so the split point is a stable crossing rather
                # than the first noisy dip.
                steam_off_start = e
                for m in range(e - 1, s - 1, -1):
                    if p[m] >= low_bar:
                        steam_off_start = m + 1
                        break
                else:
                    steam_off_start = s
                if s < steam_off_start:
                    stages[s:steam_off_start] = 'interval'
                if steam_off_start < e:
                    stages[steam_off_start:e] = 'steam_off'
                elif claim_before and claim_after:
                    stages[s:e] = 'interval'
                else:
                    stages[s:e] = 'steam_off'

    if verbose:
        n_applied = sum(1 for m in match_log if m[-1].startswith('applied'))
        n_skipped = sum(1 for m in match_log if 'SKIPPED' in m[-1])
        print(f"Rotation match log ({n_applied} applied, {n_skipped} skipped):")
        for y, x, label, overlap, drop, status in match_log:
            print(f"  AC{y} {label:15s} (serving AC{x})  overlap={overlap:.0%}  drop={drop:+.2f}bar  -> {status}")

    result = ClassificationResult(
        times={ac: combined_times[ac] for ac in range(n_autoclaves)},
        pressures={ac: combined_pressures[ac] for ac in range(n_autoclaves)},
        stages={ac: combined_stages[ac] for ac in range(n_autoclaves)},
        flow=combined_flow,
        match_log=match_log,
        n_bad_by_ac=n_bad_by_ac,
    )
    return result
