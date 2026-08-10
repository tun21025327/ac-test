"""Percentage contribution of transferred steam vs. fresh steam toward the
climb to 12 bar, and cycle count, per autoclave.

Earlier version of this script tried to convert both sides to mass (kg) —
fresh from the real Main Meter, transferred from a receiver-side density
mass-balance — and got a ~95/5 split. That approach is only as good as the
transferred-mass estimate, and there's no real meter to check it against
(the Main Meter only measures fresh steam), so it can't be validated the way
the fresh_steam attribution could be. Per your call: since we can't get a
trustworthy transferred TON figure, use PRESSURE contribution instead — it's
measured directly from the same pressure trace the whole classification is
built on, no density/volume model required:

  transferred_bar = pressure gained from end-of-vacuum to start-of-fresh_steam
                     (i.e. everything the TI1+interval+TI2+interval stages add)
  fresh_bar        = pressure gained during the fresh_steam stage itself

  transferred% = transferred_bar / (transferred_bar + fresh_bar) * 100
  fresh%        = fresh_bar / (transferred_bar + fresh_bar) * 100

Expected ratio is roughly 70/30 fresh/transferred.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import stage_classifier as sc
import metrics_common as mc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'graph')

JULY_FILES_ALL = [
    "Q2_20260626.csv", "Q2_20260627.csv", "Q2_20260628.csv", "Q2_20260629.csv", "Q2_20260630.csv",
    "Q2_20260701.csv", "Q2_20260702.csv", "Q2_20260703.csv", "Q2_20260704.csv",
    "Q2_20260705.csv", "Q2_20260706.csv", "Q2_20260707.csv",
]
DATE_RANGE_LABEL = "Jun 26 - Jul 7, 2026"

dfs = sc.load_csv_files(SCRIPT_DIR, JULY_FILES_ALL)
result = sc.classify_all_autoclaves(dfs, verbose=False)

fresh_pct_mean, fresh_pct_sd = [], []
transferred_pct_mean, transferred_pct_sd = [], []
n_cycles = []

for ac0 in range(sc.N_AUTOCLAVES):
    p = result.pressures[ac0]
    stages = result.stages[ac0]

    vac_runs = sc.contiguous_runs(stages == 'vacuum')      # (vs, vac_min) per cycle
    fresh_runs = sc.contiguous_runs(stages == 'fresh_steam')  # (fs, fe) per cycle

    fresh_pcts, transferred_pcts = [], []
    for (fs, fe) in fresh_runs:
        # most recent vacuum end (= transfer start) before this fresh_steam run
        prior_vac_ends = [ve for (vs, ve) in vac_runs if ve <= fs]
        if not prior_vac_ends:
            continue
        transfer_start = max(prior_vac_ends)

        transferred_bar = p[fs] - p[transfer_start]
        fresh_bar = p[fe - 1] - p[fs]
        total_bar = transferred_bar + fresh_bar
        if total_bar <= 0:
            continue
        fresh_pcts.append(fresh_bar / total_bar * 100)
        transferred_pcts.append(transferred_bar / total_bar * 100)

    fm, fsd, _ = mc.mean_sd(fresh_pcts)
    tm, tsd, _ = mc.mean_sd(transferred_pcts)
    fresh_pct_mean.append(fm); fresh_pct_sd.append(fsd)
    transferred_pct_mean.append(tm); transferred_pct_sd.append(tsd)
    n_cycles.append(mc.cycle_count(stages))

fig, ax1 = plt.subplots(figsize=(14, 8))
x = np.arange(sc.N_AUTOCLAVES)
w = 0.36
b1 = ax1.bar(x - w/2, fresh_pct_mean, w, yerr=fresh_pct_sd, capsize=4, color='#2f6faa', edgecolor='black', label='fresh steam [%]')
b2 = ax1.bar(x + w/2, transferred_pct_mean, w, yerr=transferred_pct_sd, capsize=4, color='#e8b830', edgecolor='black', label='transferred steam [%]')
for bar, v, sdv in zip(b1, fresh_pct_mean, fresh_pct_sd):
    ax1.annotate(f"{v:.1f}", (bar.get_x()+bar.get_width()/2, v+sdv), xytext=(0,14), textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color='#1a3f66')
    ax1.annotate(f"SD={sdv:.1f}", (bar.get_x()+bar.get_width()/2, v+sdv), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8, color='#1a3f66')
for bar, v, sdv in zip(b2, transferred_pct_mean, transferred_pct_sd):
    ax1.annotate(f"{v:.1f}", (bar.get_x()+bar.get_width()/2, v+sdv), xytext=(0,14), textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color='#8a6810')
    ax1.annotate(f"SD={sdv:.1f}", (bar.get_x()+bar.get_width()/2, v+sdv), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8, color='#8a6810')
ax1.set_ylim(0, 115)
ax1.set_yticks(range(0, 101, 20))
ax1.set_xticks(x); ax1.set_xticklabels(mc.AUTOCLAVE_LABELS)
ax1.set_ylabel('steam rate [%]')
ax1.set_xlabel('autoclave')

ax2 = ax1.twinx()
ax2.plot(x, n_cycles, color='black', marker='o', linewidth=2, label='number of autoclave cycles [-]')
for xi, n in zip(x, n_cycles):
    ax2.annotate(str(n), (xi, n), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
ax2.set_ylim(0, max(n_cycles) * 1.6)
ax2.set_ylabel('number of autoclave cycles [-]')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

ax1.set_title(f'Percentage Use of Fresh Steam vs. Transferred Steam, and Number of Autoclave Cycles\n{DATE_RANGE_LABEL} — '
              f'pressure-contribution split: transferred = TI1+interval+TI2 rise, fresh = fresh_steam-stage rise, per cycle',
              fontsize=13, fontweight='bold')
fig.text(0.5, -0.02,
         'transferred% = (p at fresh_steam start − p at end of vacuum) / total rise  |  fresh% = (p at fresh_steam end − p at fresh_steam start) / total rise, per cycle, then averaged',
         ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fresh_vs_transferred_steam.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("Fresh/Transferred % (pressure-contribution basis) and cycles:")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: fresh={fresh_pct_mean[i]:.1f}+/-{fresh_pct_sd[i]:.1f}%  "
          f"transferred={transferred_pct_mean[i]:.1f}+/-{transferred_pct_sd[i]:.1f}%  cycles={n_cycles[i]}")
print("Saved fresh_vs_transferred_steam.png")
