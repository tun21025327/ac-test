"""Transfer leakage — leftover donor steam converted to mass (kg), per
autoclave.

"Leftover after both transfer periods" = the steam still sitting in the
donor's chamber once it has finished serving BOTH of its transfer-out slots
(transfer_out1 and transfer_out2 — each donor serves two different partner
autoclaves per the rotation table, see stage_classifier.ROTATION). That
final, post-TO2 pressure is effectively wasted/unclaimed steam (vented on
the next vacuum pull rather than delivered anywhere) — this is the real
"leakage" quantity, expressed in kg instead of bar via IAPWS-97 saturated-
vapor density (steam_properties.py) so it's comparable in real mass terms,
not just relative pressure.

Reuses stage_classifier's already rotation-matched transfer_out1/
transfer_out2 labels — no separate matching pass.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import stage_classifier as sc
import steam_properties as sp
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

to2_kg_mean, to2_kg_sd = [], []

for ac1 in range(1, sc.N_AUTOCLAVES + 1):
    ac0 = ac1 - 1
    p = result.pressures[ac0]
    stages = result.stages[ac0]
    offset = sc.SENSOR_OFFSETS.get(ac1, 0.0)

    to2_runs = sc.contiguous_runs(stages == 'transfer_out2')
    to2_kg = [sp.rho_sat_vapor(p[e - 1] + offset) * sp.V_FREE[ac1] for (s, e) in to2_runs if e > s]

    m2, sd2, _ = mc.mean_sd(to2_kg)
    to2_kg_mean.append(m2); to2_kg_sd.append(sd2)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(mc.AUTOCLAVE_LABELS, to2_kg_mean, yerr=to2_kg_sd, capsize=4, color='#7a1f1f', edgecolor='black')
mc.bar_labels(ax, bars, to2_kg_mean, to2_kg_sd, fmt="{:.1f}", sd_fmt="SD={:.1f}")
ax.set_ylabel('Leftover donor steam mass (kg)')
ax.set_xlabel('autoclave (donor)')
ax.set_title(f'Transfer Leakage — Leftover Donor Steam Mass After Both Transfers, {DATE_RANGE_LABEL}\n'
             f'IAPWS-97 saturated-vapor density × chamber volume, at end of Transfer Out 2',
             fontsize=12.5, fontweight='bold')
fig.text(0.5, -0.02,
         'kg = ρ(p_true at end of Transfer Out 2) × V_free  |  V_free = 85.09 m³ (AC1-7), 154.84 m³ (AC8)  |  '
         'steam still in the donor once both of its transfer-out slots are done, vented rather than delivered',
         ha='center', fontsize=8.5, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'transfer_leakage_mass.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("Transfer leakage mass, kg (mean +/- SD), after Transfer Out 2:")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {to2_kg_mean[i]:.2f}+/-{to2_kg_sd[i]:.2f}")
print("Saved transfer_leakage_mass.png")
