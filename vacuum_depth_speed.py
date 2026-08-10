"""Vacuum depth (ratio to target) and vacuum speed (time to reach target),
per autoclave.

Fixes vs. the old (unsaved, chat-regenerated) version:
  - AC4's known ~0.2 bar-low sensor bias (see stage_classifier.SENSOR_OFFSETS,
    the same correction classify_trace() already applies to every other
    threshold) is now applied here too. That's why the old chart showed AC4
    at a 1.34 depth ratio, way outside every other AC's 0.89-0.95 band —
    AC4's raw dip genuinely reads ~0.2 bar deeper than the true vacuum, and
    nothing was compensating for it in that chart.
  - The 'vacuum' stage run is extended by one sample: classify_trace() labels
    [vac_start, vac_min) as 'vacuum', so the single deepest sample (at
    vac_min itself) technically belongs to the *next* stage. Using the plain
    stage run for "how deep did the vacuum get" would silently miss the
    actual minimum. See metrics_common.vacuum_runs().
  - Pass/fail at an exact ratio-of-1.0 cutoff is kept (it's a real spec
    target, not an arbitrary line), but is now computed on the corrected
    pressure for every AC, so a near-0%/near-100% split reflects genuine
    vacuum-pump performance instead of an uncorrected sensor artifact.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import stage_classifier as sc
import metrics_common as mc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'graph')

VACUUM_TARGET_BAR = -0.6    # true (corrected) bar — depth target
SPEED_TARGET_BAR  = -0.55   # true (corrected) bar — speed checkpoint
SPEED_TARGET_MIN  = 11.5    # minutes — nominal target to reach SPEED_TARGET_BAR
SPEED_OUTLIER_CUTOFF_MIN = 35.0  # a vacuum pull taking longer than this to reach
                                   # SPEED_TARGET_BAR is treated as an outlier (e.g.
                                   # multiple ACs sharing the vacuum pump at once)
                                   # and excluded from the mean/SD, not just capped

JULY_FILES_ALL = [
    "Q2_20260626.csv", "Q2_20260627.csv", "Q2_20260628.csv", "Q2_20260629.csv", "Q2_20260630.csv",
    "Q2_20260701.csv", "Q2_20260702.csv", "Q2_20260703.csv", "Q2_20260704.csv",
    "Q2_20260705.csv", "Q2_20260706.csv", "Q2_20260707.csv",
]
DATE_RANGE_LABEL = "Jun 26 - Jul 7, 2026"

dfs = sc.load_csv_files(SCRIPT_DIR, JULY_FILES_ALL)
result = sc.classify_all_autoclaves(dfs, verbose=False)

depth_ratio_mean, depth_ratio_sd, pass_rate, depth_n = [], [], [], []
speed_mean, speed_sd, speed_n, speed_n_outliers = [], [], [], []

for ac0 in range(sc.N_AUTOCLAVES):
    ac1 = ac0 + 1
    t = result.times[ac0]
    p_raw = result.pressures[ac0]
    offset = sc.SENSOR_OFFSETS.get(ac1, 0.0)
    p_true = p_raw + offset  # correct this AC's known sensor bias to true bar

    runs = mc.vacuum_runs(result.stages[ac0], t=t)

    ratios, speeds_min, n_outliers = [], [], 0
    for (s, e) in runs:
        seg_min = p_true[s:e].min()
        ratios.append(seg_min / VACUUM_TARGET_BAR)

        # speed: minutes from run start to first crossing of SPEED_TARGET_BAR
        below = np.where(p_true[s:e] <= SPEED_TARGET_BAR)[0]
        if len(below):
            minutes = (t[s + below[0]] - t[s]) * 60.0
            if minutes > SPEED_OUTLIER_CUTOFF_MIN:
                n_outliers += 1  # e.g. shared-pump contention slowing this one pull
            else:
                speeds_min.append(minutes)

    m, sdv, n = mc.mean_sd(ratios)
    depth_ratio_mean.append(m); depth_ratio_sd.append(sdv); depth_n.append(n)
    pass_rate.append(np.mean(np.array(ratios) >= 1.0) if ratios else np.nan)

    m2, sdv2, n2 = mc.mean_sd(speeds_min)
    speed_mean.append(m2); speed_sd.append(sdv2); speed_n.append(n2)
    speed_n_outliers.append(n_outliers)

# ── Chart 1: vacuum depth ratio + pass rate ─────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
colors = plt.cm.tab10(np.linspace(0, 1, sc.N_AUTOCLAVES))
bars = ax.bar(mc.AUTOCLAVE_LABELS, depth_ratio_mean, color=colors, edgecolor='black')
mc.bar_labels(ax, bars, depth_ratio_mean, depth_ratio_sd, fmt="{:.2f}", sd_fmt="SD={:.2f}")
for bar, pr, n in zip(bars, pass_rate, depth_n):
    ax.text(bar.get_x() + bar.get_width() / 2, 0.06, f"{pr:.0%} pass\n(n={n})",
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')
ax.axhline(1.0, color='red', linewidth=1.5, label='Target = 1.00 (exact)')
ax.set_ylabel(f'Depth Ratio (true pressure / target {VACUUM_TARGET_BAR} bar)')
ax.set_title(f'Vacuum Depth — Mean per Autoclave, {DATE_RANGE_LABEL}\n'
             f'Ratio = deepest TRUE pressure in each vacuum run / target ({VACUUM_TARGET_BAR} bar)  |  '
             f'AC4 sensor-offset corrected', fontsize=12.5, fontweight='bold')
ax.legend(loc='upper right')
fig.text(0.5, -0.02,
         'AC4 raw pressure is corrected +0.2 bar (known low-reading sensor, same offset classify_trace() uses) before computing depth — '
         'this is the fix for the old chart\'s 1.34 outlier',
         ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'vacuum_depth_adjusted.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Chart 2: vacuum speed ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.bar(mc.AUTOCLAVE_LABELS, speed_mean, color=colors, edgecolor='black')
mc.bar_labels(ax, bars, speed_mean, speed_sd, fmt="{:.2f} min", sd_fmt="SD={:.2f} min")
for bar, n_out in zip(bars, speed_n_outliers):
    if n_out:
        ax.annotate(f"{n_out} outlier{'s' if n_out != 1 else ''}\nexcluded",
                    (bar.get_x() + bar.get_width() / 2, 0), xytext=(0, 6),
                    textcoords='offset points', ha='center', va='bottom', fontsize=8, color='#8a1f1f')
ax.axhline(SPEED_TARGET_MIN, color='red', linewidth=1.5, label=f'Target = {SPEED_TARGET_MIN} min')
ax.set_ylabel(f'Time to reach {SPEED_TARGET_BAR} bar (min)')
ax.set_title(f'Vacuum Speed — Mean per Autoclave, {DATE_RANGE_LABEL}\n'
             f'Time from vacuum start to first crossing {SPEED_TARGET_BAR} bar (true, AC4 corrected)',
             fontsize=12.5, fontweight='bold')
ax.legend(loc='upper right')
fig.text(0.5, -0.02,
         f'Pulls taking longer than {SPEED_OUTLIER_CUTOFF_MIN:.0f} min to reach {SPEED_TARGET_BAR} bar are treated as outliers '
         f'(e.g. multiple autoclaves sharing the vacuum pump at once) and excluded from mean/SD, not capped into it',
         ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'vacuum_speed_adjusted.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("Vacuum depth ratio (mean +/- SD, n, pass%):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {depth_ratio_mean[i]:.3f} +/- {depth_ratio_sd[i]:.3f}  (n={depth_n[i]}, pass={pass_rate[i]:.0%})")
print(f"Vacuum speed to target (mean +/- SD min, n, outliers excluded at >{SPEED_OUTLIER_CUTOFF_MIN:.0f} min):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {speed_mean[i]:.2f} +/- {speed_sd[i]:.2f}  (n={speed_n[i]}, outliers={speed_n_outliers[i]})")
print("Saved vacuum_depth_adjusted.png and vacuum_speed_adjusted.png")
