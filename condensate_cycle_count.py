"""Condensate cycle count proxy — temperature dip-recover events during
Keeping+Idle, per autoclave.

Fixes vs. the old (unsaved, chat-regenerated) version:
  - Reported as events PER HOUR, not raw counts per hold. classify_trace()'s
    keeping+idle window length is not fixed (idle absorbs "whatever's left"
    after a nominal 4h keeping duration, so real combined length varies
    cycle-to-cycle and AC-to-AC) — counting raw ripple events over windows of
    different lengths measures window length as much as it measures
    condensate cycling. That mismatch is almost certainly why the old chart
    had AC7 at 90.3 events and AC8 at 3.2 — a ~28x gap that's far more
    plausible as a window-duration artifact than a real physical difference.
  - Uses scipy.signal.find_peaks (prominence-based trough detection) instead
    of an ad-hoc dip/recover threshold walk, so "a dip that round-trips at
    least 0.15 C" is a precise, reproducible definition.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import stage_classifier as sc
import metrics_common as mc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'graph')

DIP_THRESHOLD_C = 0.15
MIN_HOLD_DURATION_H = 0.25  # 15 min — drop tiny/degenerate keeping+idle windows

JULY_FILES_ALL = [
    "Q2_20260626.csv", "Q2_20260627.csv", "Q2_20260628.csv", "Q2_20260629.csv", "Q2_20260630.csv",
    "Q2_20260701.csv", "Q2_20260702.csv", "Q2_20260703.csv", "Q2_20260704.csv",
    "Q2_20260705.csv", "Q2_20260706.csv", "Q2_20260707.csv",
]
DATE_RANGE_LABEL = "Jun 26 - Jul 7, 2026"

dfs = sc.load_csv_files(SCRIPT_DIR, JULY_FILES_ALL)
result = sc.classify_all_autoclaves(dfs, verbose=False)
temps = sc.concat_per_ac_column(dfs, 'Temperature Autocalve {i}')

rate_mean, rate_sd, rate_n = [], [], []
total_count = []
n_holds_list = []

for ac0 in range(sc.N_AUTOCLAVES):
    t = result.times[ac0]
    temp = temps[ac0]
    stages = result.stages[ac0]

    hold_mask = (stages == 'keeping') | (stages == 'idle')
    runs = sc.contiguous_runs(hold_mask)
    runs = [(s, e) for (s, e) in runs if (t[e - 1] - t[s]) >= MIN_HOLD_DURATION_H]

    rates, counts = [], []
    for (s, e) in runs:
        troughs, _ = find_peaks(-temp[s:e], prominence=DIP_THRESHOLD_C)
        dur_h = t[e - 1] - t[s]
        rates.append(len(troughs) / dur_h)
        counts.append(len(troughs))

    m, sdv, n = mc.mean_sd(rates)
    rate_mean.append(m); rate_sd.append(sdv); rate_n.append(n)
    total_count.append(sum(counts))
    n_holds_list.append(len(runs))

fig, ax = plt.subplots(figsize=(11, 7))
colors = plt.cm.tab10(np.linspace(0, 1, sc.N_AUTOCLAVES))
bars = ax.bar(mc.AUTOCLAVE_LABELS, rate_mean, color='#5b3fa0', edgecolor='black')
mc.bar_labels(ax, bars, rate_mean, rate_sd, fmt="{:.1f}", sd_fmt="SD={:.1f}")
ax.set_ylabel('condensate cycles per hour')
ax.set_xlabel('autoclaves')
ax.set_title(f'Condensate Cycle Rate — Temperature Dip-Recover Events\n{DATE_RANGE_LABEL} — '
             f'during Keeping+Idle, threshold = {DIP_THRESHOLD_C}°C, rate-normalized',
             fontsize=13, fontweight='bold')
fig.text(0.5, -0.02,
         f'Proxy signal only — inferred from single-sensor temperature ripple, not a direct condensate valve measurement  |  '
         f'rate = trough count (scipy find_peaks, prominence >= {DIP_THRESHOLD_C}°C) / hold duration, so different-length holds are comparable',
         ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'condensate_cycle_rate.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Chart 2: total cycle count (summed over the whole date range) ─────────────
# Note: unlike the rate chart above, this is NOT duration-normalized — it's
# the raw total number of condensate dip-recover events across all holds in
# the period. Fair to compare AC-to-AC only to the extent their total hold
# time is similar (each AC here goes through a similar ~22-23 cycles over
# the same 12 days, so total hold time is roughly comparable) — if you need
# a metric that's robust to holds of very different total length, use the
# rate chart (condensate_cycle_rate.png) instead.
fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.bar(mc.AUTOCLAVE_LABELS, total_count, color='#5b3fa0', edgecolor='black')
for bar, c in zip(bars, total_count):
    ax.annotate(str(c), (bar.get_x() + bar.get_width() / 2, c), xytext=(0, 4),
                textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('total condensate cycles (count)')
ax.set_xlabel('autoclaves')
ax.set_title(f'Condensate Cycle Count — Total Temperature Dip-Recover Events\n{DATE_RANGE_LABEL} — '
             f'during Keeping+Idle, threshold = {DIP_THRESHOLD_C}°C, summed over all holds',
             fontsize=13, fontweight='bold')
fig.text(0.5, -0.02,
         f'Proxy signal only — inferred from single-sensor temperature ripple, not a direct condensate valve measurement  |  '
         f'total = sum of trough counts (scipy find_peaks, prominence >= {DIP_THRESHOLD_C}°C) across all {DATE_RANGE_LABEL} holds — not rate-normalized',
         ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'condensate_cycle_total.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("Condensate cycles/hour (mean +/- SD, n holds):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {rate_mean[i]:.2f} +/- {rate_sd[i]:.2f}  (n={rate_n[i]})")
print("Total condensate cycles (summed over all holds):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {total_count[i]}  (n_holds={n_holds_list[i]})")
print("Saved condensate_cycle_rate.png and condensate_cycle_total.png")
