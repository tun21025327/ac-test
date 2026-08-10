"""Total idle-phase leakage per autoclave, Jun 26 - Jul 7, 2026 — TOTAL mass
lost (ton), not the average rate idle_leakage.py reports. Each bar is
labeled with the number of qualifying idle segments (~one per cycle) and
their average duration, so a big total can be read alongside whether it
came from many short idles or fewer long ones.

Uses the same segment selection as idle_leakage.py (idle runs >= 15 min,
segments with |linear-fit slope| > 2 bar/h excluded as classification
artifacts) via metrics_common.idle_total_leakage_stats, so the two charts'
numbers are directly reconcilable (total ≈ average rate × n × avg duration).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import stage_classifier as sc
import steam_properties as sp
import metrics_common as mc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'graph')

MIN_IDLE_DURATION_H = 0.25  # 15 min — same filter as idle_leakage.py
KG_PER_TON = 1000.0

JULY_FILES_ALL = [
    "Q2_20260626.csv", "Q2_20260627.csv", "Q2_20260628.csv", "Q2_20260629.csv", "Q2_20260630.csv",
    "Q2_20260701.csv", "Q2_20260702.csv", "Q2_20260703.csv", "Q2_20260704.csv",
    "Q2_20260705.csv", "Q2_20260706.csv", "Q2_20260707.csv",
]
DATE_RANGE_LABEL = "Jun 26 - Jul 7, 2026"

dfs = sc.load_csv_files(SCRIPT_DIR, JULY_FILES_ALL)
result = sc.classify_all_autoclaves(dfs, verbose=False)

totals = mc.idle_total_leakage_stats(result, min_duration_h=MIN_IDLE_DURATION_H)
acs = list(range(1, sc.N_AUTOCLAVES + 1))

total_ton = [totals[a][0] / KG_PER_TON for a in acs]
n_cycles  = [totals[a][1] for a in acs]
n_flagged = [totals[a][2] for a in acs]
avg_idle_h = [totals[a][3] for a in acs]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.bar(mc.AUTOCLAVE_LABELS, total_ton, color='#7a1f1f', edgecolor='black')
for bar, v, n, avg_h in zip(bars, total_ton, n_cycles, avg_idle_h):
    va, pts = ('top', -6) if v < 0 else ('bottom', 6)
    ax.annotate(f"{v:.3f} ton", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, pts),
                textcoords='offset points', ha='center', va=va, fontsize=11, fontweight='bold')
    label_y, va2, pts2 = (v, 'top', -22) if v < 0 else (v, 'bottom', 22)
    ax.annotate(f"n={n} cycles\navg idle={avg_h*60:.0f} min", (bar.get_x() + bar.get_width() / 2, label_y),
                xytext=(0, pts2), textcoords='offset points', ha='center', va=va2, fontsize=8.5)
ax.margins(y=0.22)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('total idle leakage [ton]')
ax.set_xlabel('autoclaves')
ax.set_title(f'Total Idle Leakage — {DATE_RANGE_LABEL}\n'
             f'summed real mass change (IAPWS-97) across all qualifying idle segments, not rate-averaged',
             fontsize=13, fontweight='bold')
fig.text(0.5, -0.02,
         f'total = Σ [ρ(p_end) − ρ(p_start)] × V_free over every idle run ≥ {int(MIN_IDLE_DURATION_H*60)} min '
         f'(segments with |leak slope| > 2 bar/h excluded as classification artifacts)  |  '
         f'V_free = {sp.V_FREE[1]:.2f} m³ (AC1-7), {sp.V_FREE[8]:.2f} m³ (AC8)',
         ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'idle_leakage_total_ton.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Total idle leakage, {DATE_RANGE_LABEL} (ton, n cycles, avg idle time, flagged):")
for i, a in enumerate(acs):
    print(f"  AC{a}: {total_ton[i]:+.4f} ton  (n={n_cycles[i]} cycles, avg idle={avg_idle_h[i]*60:.1f} min, flagged={n_flagged[i]})")
print("Saved idle_leakage_total_ton.png")
