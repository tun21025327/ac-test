"""Savable steam — total leftover donor steam mass sitting ABOVE a 1 bar
floor after Transfer Out 2 completes, per autoclave, Jun 26 - Jul 7, 2026.

transfer_leakage_mass.py already reports the full leftover donor mass at
the end of Transfer Out 2 (all of it, as "wasted"). This chart narrows
that to the portion above 1 bar specifically — steam_off itself (see
stage_classifier.STEAM_OFF_FINAL_BAR) is only ever the narrow ~1 bar -> 0
bar band right before the next vacuum pull, so anything still sitting
above 1 bar right when Transfer Out 2 ends is pressure the transfer
stopped short of using, not steam that's inherently unrecoverable the way
the final ~1 bar->0 bar band is (that residual has to be pulled down to
vacuum regardless). Hence "savable": if the transfer had run a bit longer
or the receiving AC had more capacity, this portion could plausibly have
been delivered instead of vented.

TOTAL (not average rate) across the whole date range, in tons, plus the
number of Transfer Out 2 cycles counted and their average duration.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import stage_classifier as sc
import steam_properties as sp
import metrics_common as mc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'graph')

FLOOR_BAR = 1.0    # steam below this is treated as unrecoverable (matches
                    # stage_classifier.STEAM_OFF_FINAL_BAR's steam_off floor)
KG_PER_TON = 1000.0

JULY_FILES_ALL = [
    "Q2_20260626.csv", "Q2_20260627.csv", "Q2_20260628.csv", "Q2_20260629.csv", "Q2_20260630.csv",
    "Q2_20260701.csv", "Q2_20260702.csv", "Q2_20260703.csv", "Q2_20260704.csv",
    "Q2_20260705.csv", "Q2_20260706.csv", "Q2_20260707.csv",
]
DATE_RANGE_LABEL = "Jun 26 - Jul 7, 2026"

dfs = sc.load_csv_files(SCRIPT_DIR, JULY_FILES_ALL)
result = sc.classify_all_autoclaves(dfs, verbose=False)

floor_mass_by_ac = {ac1: sp.rho_sat_vapor(FLOOR_BAR) * sp.V_FREE[ac1] for ac1 in range(1, sc.N_AUTOCLAVES + 1)}

total_ton, n_cycles, avg_dur_min = [], [], []
for ac1 in range(1, sc.N_AUTOCLAVES + 1):
    ac0 = ac1 - 1
    p = result.pressures[ac0]
    t = result.times[ac0]
    stages = result.stages[ac0]
    offset = sc.SENSOR_OFFSETS.get(ac1, 0.0)

    to2_runs = sc.contiguous_runs(stages == 'transfer_out2')
    savable_kg, durations_h = [], []
    for (s, e) in to2_runs:
        if e <= s:
            continue
        p_end = p[e - 1] + offset
        leftover_kg = sp.rho_sat_vapor(p_end) * sp.V_FREE[ac1]
        savable = max(0.0, leftover_kg - floor_mass_by_ac[ac1])
        savable_kg.append(savable)
        durations_h.append(t[e - 1] - t[s])

    total_ton.append(sum(savable_kg) / KG_PER_TON)
    n_cycles.append(len(savable_kg))
    avg_h, _, _ = mc.mean_sd(durations_h)
    avg_dur_min.append(avg_h * 60 if np.isfinite(avg_h) else np.nan)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.bar(mc.AUTOCLAVE_LABELS, total_ton, color='#1f7a4c', edgecolor='black')
for bar, v, n, dmin in zip(bars, total_ton, n_cycles, avg_dur_min):
    ax.annotate(f"{v:.3f} ton", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 6),
                textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')
    dur_txt = f"{dmin:.0f} min" if np.isfinite(dmin) else "n/a"
    ax.annotate(f"n={n} cycles\navg TO2={dur_txt}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 26),
                textcoords='offset points', ha='center', va='bottom', fontsize=8.5)
ax.margins(y=0.22)
ax.set_ylabel('savable steam, total [ton]')
ax.set_xlabel('autoclaves')
ax.set_title(f'Savable Steam — Leftover Above {FLOOR_BAR:.0f} Bar After Transfer Out 2, {DATE_RANGE_LABEL}\n'
             f'total mass (IAPWS-97) across all cycles, not rate-averaged',
             fontsize=12.5, fontweight='bold')
fig.text(0.5, -0.02,
         f'savable = max(0, ρ(p_true at end of Transfer Out 2) − ρ({FLOOR_BAR:.0f} bar)) × V_free, summed over every Transfer Out 2 cycle  |  '
         f'V_free = {sp.V_FREE[1]:.2f} m³ (AC1-7), {sp.V_FREE[8]:.2f} m³ (AC8)  |  '
         f'{FLOOR_BAR:.0f} bar floor matches steam_off\'s own final band — pressure below it is never "savable" regardless',
         ha='center', fontsize=8.5, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'savable_steam_total_ton.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Savable steam (>{FLOOR_BAR:.0f} bar after Transfer Out 2), {DATE_RANGE_LABEL} (ton total, n cycles, avg TO2 duration):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {total_ton[i]:.4f} ton  (n={n_cycles[i]} cycles, avg TO2={avg_dur_min[i]:.1f} min)")
print("Saved savable_steam_total_ton.png")
