"""Post-transfer pressure vs target — raw pressure at the end of each
autoclave's Transfer In 1 / Transfer In 2 stage, per cycle.

Note on the (still large) error bars: Transfer In 1/2 are fixed 30-minute
nominal windows in stage_classifier.classify_trace() (TI1_DURATION /
TI2_DURATION), not detected from where the real physical transfer actually
finishes. Sampling pressure at a fixed clock offset will land at different
points of the real transfer curve depending on how fast that cycle's
transfer actually ran — so a big spread here is telling you the real
transfer duration varies cycle-to-cycle, not that this script is unreliable.
That's a genuine finding worth flagging, not something a downstream chart can
paper over without changing how TI1/TI2 boundaries are detected upstream.

AC4's sensor reads ~0.2 bar low (see stage_classifier.SENSOR_OFFSETS). Raw,
uncorrected values are plotted for every AC (consistent axis), and AC4's
true target is called out as 3.8 bar instead of adjusting its data.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import stage_classifier as sc
import metrics_common as mc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'graph')

TARGET_BAR = 4.0

JULY_FILES_ALL = [
    "Q2_20260626.csv", "Q2_20260627.csv", "Q2_20260628.csv", "Q2_20260629.csv", "Q2_20260630.csv",
    "Q2_20260701.csv", "Q2_20260702.csv", "Q2_20260703.csv", "Q2_20260704.csv",
    "Q2_20260705.csv", "Q2_20260706.csv", "Q2_20260707.csv",
]
DATE_RANGE_LABEL = "Jun 26 - Jul 7, 2026"

dfs = sc.load_csv_files(SCRIPT_DIR, JULY_FILES_ALL)
result = sc.classify_all_autoclaves(dfs, verbose=False)

ti1_mean, ti1_sd = [], []
ti2_mean, ti2_sd = [], []

for ac0 in range(sc.N_AUTOCLAVES):
    p = result.pressures[ac0]
    stages = result.stages[ac0]

    ti1_runs = sc.contiguous_runs(stages == 'transfer_in1')
    ti2_runs = sc.contiguous_runs(stages == 'transfer_in2')

    ti1_vals = [p[e - 1] for (s, e) in ti1_runs if e > s]
    ti2_vals = [p[e - 1] for (s, e) in ti2_runs if e > s]

    m1, sd1, _ = mc.mean_sd(ti1_vals)
    m2, sd2, _ = mc.mean_sd(ti2_vals)
    ti1_mean.append(m1); ti1_sd.append(sd1)
    ti2_mean.append(m2); ti2_sd.append(sd2)

x = np.arange(sc.N_AUTOCLAVES)
w = 0.36
fig, ax = plt.subplots(figsize=(12, 7))
b1 = ax.bar(x - w/2, ti1_mean, w, yerr=ti1_sd, capsize=4, color='#f0c419', edgecolor='black', label='After Transfer In 1')
b2 = ax.bar(x + w/2, ti2_mean, w, yerr=ti2_sd, capsize=4, color='#e07b1a', edgecolor='black', label='After Transfer In 2')

for bar, m, sdv in zip(b1, ti1_mean, ti1_sd):
    ax.annotate(f"{m:.2f}±{sdv:.2f}", (bar.get_x() + bar.get_width()/2, m + sdv),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
for bar, m, sdv in zip(b2, ti2_mean, ti2_sd):
    ax.annotate(f"{m:.2f}±{sdv:.2f}", (bar.get_x() + bar.get_width()/2, m + sdv),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

ax.axhline(TARGET_BAR, color='red', linestyle='--', linewidth=1.5)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax.text(sc.N_AUTOCLAVES - 0.5, 1.05, '1 bar', color='gray', fontsize=9, ha='right')
ax.set_xticks(x)
ax.set_xticklabels(mc.AUTOCLAVE_LABELS)
ax.set_ylabel('Mean pressure (bar)')
ax.set_title(f'Post-Transfer Pressure vs Target, per Autoclave\n{DATE_RANGE_LABEL}  |  '
             f'Target: {TARGET_BAR} bar after TI2 (3.8 for AC4)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
fig.text(0.5, -0.03,
         'Note: AC4 sensor reads ~0.2 bar LOW — raw uncorrected value shown, not adjusted (true target for AC4 is 3.8 bar, not 4.0)  |  '
         'bars = mean +/- SD across cycles; large SD reflects real cycle-to-cycle transfer-duration variability against the fixed 30-min TI1/TI2 window, see script docstring',
         ha='center', fontsize=8.5, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'post_transfer_summary.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("After TI1 (mean +/- SD):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {ti1_mean[i]:.2f} +/- {ti1_sd[i]:.2f}")
print("After TI2 (mean +/- SD):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {ti2_mean[i]:.2f} +/- {ti2_sd[i]:.2f}")
print("Saved post_transfer_summary.png")
