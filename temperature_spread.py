"""Temperature spread (delta-T) during hold, per autoclave — max minus min
single-sensor temperature during each Keeping+Idle hold.

Unlike the leakage-rate and condensate-count metrics, this one is a bounded
range (not a count/slope that scales with window length), so it doesn't need
the same duration-normalization fix — but degenerate near-zero-length holds
are still dropped (MIN_HOLD_DURATION_H) since a 1-sample "hold" trivially has
delta-T = 0 and would silently pull the mean down.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import stage_classifier as sc
import metrics_common as mc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'graph')

MIN_HOLD_DURATION_H = 0.25  # 15 min

JULY_FILES_ALL = [
    "Q2_20260626.csv", "Q2_20260627.csv", "Q2_20260628.csv", "Q2_20260629.csv", "Q2_20260630.csv",
    "Q2_20260701.csv", "Q2_20260702.csv", "Q2_20260703.csv", "Q2_20260704.csv",
    "Q2_20260705.csv", "Q2_20260706.csv", "Q2_20260707.csv",
]
DATE_RANGE_LABEL = "Jun 26 - Jul 7, 2026"

dfs = sc.load_csv_files(SCRIPT_DIR, JULY_FILES_ALL)
result = sc.classify_all_autoclaves(dfs, verbose=False)
temps = sc.concat_per_ac_column(dfs, 'Temperature Autocalve {i}')

dt_mean, dt_sd = [], []

for ac0 in range(sc.N_AUTOCLAVES):
    t = result.times[ac0]
    temp = temps[ac0]
    stages = result.stages[ac0]

    hold_mask = (stages == 'keeping') | (stages == 'idle')
    runs = sc.contiguous_runs(hold_mask)
    runs = [(s, e) for (s, e) in runs if (t[e - 1] - t[s]) >= MIN_HOLD_DURATION_H]

    spreads = [temp[s:e].max() - temp[s:e].min() for (s, e) in runs]
    m, sdv, n = mc.mean_sd(spreads)
    dt_mean.append(m); dt_sd.append(sdv)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.bar(mc.AUTOCLAVE_LABELS, dt_mean, color='#d9694f', edgecolor='black')
mc.bar_labels(ax, bars, dt_mean, dt_sd, fmt="{:.2f}", sd_fmt="SD={:.2f}")
ax.set_ylabel('ΔT = max - min temperature (°C)')
ax.set_xlabel('autoclaves')
ax.set_title(f'Temperature Spread (ΔT) During Hold\n{DATE_RANGE_LABEL} — during Keeping+Idle, per-hold max minus min',
             fontsize=13, fontweight='bold')
fig.text(0.5, -0.02,
         'Single sensor only — not a true top-vs-bottom ΔT (that needs two separate probes), just overall temperature spread within the hold  |  '
         f'holds shorter than {int(MIN_HOLD_DURATION_H*60)} min are excluded',
         ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'temperature_spread.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("Delta-T during hold (mean +/- SD):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {dt_mean[i]:.2f} +/- {dt_sd[i]:.2f}")
print("Saved temperature_spread.png")
