"""Idle-phase leakage rate, per autoclave — bar/h (raw slope) and kg/h (mass).

Fixes vs. the old (unsaved, chat-regenerated) version:
  - Idle segments shorter than MIN_IDLE_DURATION_H are dropped before fitting
    a slope. classify_trace()'s idle window is "whatever's left after a fixed
    4h nominal keeping duration", so its real length varies cycle-to-cycle —
    a 3-minute idle sliver produces a wildly noisy slope that isn't a real
    leakage-rate difference. This filtering is the main reason the old bar/h
    chart had AC1 at -0.4656 bar/h (2-3x every other AC).
  - kg/h is now an actual RATE: [rho(p_end) - rho(p_start)] * V_free / duration,
    not just the raw mass delta labeled "per h". Without dividing by the
    segment's real duration, two idle windows of different lengths aren't
    comparable even though they're both plotted as one bar.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import stage_classifier as sc
import steam_properties as sp
import metrics_common as mc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'graph')

MIN_IDLE_DURATION_H = 0.25  # 15 min — shorter idle runs are dropped (see docstring)

JULY_FILES_ALL = [
    "Q2_20260626.csv", "Q2_20260627.csv", "Q2_20260628.csv", "Q2_20260629.csv", "Q2_20260630.csv",
    "Q2_20260701.csv", "Q2_20260702.csv", "Q2_20260703.csv", "Q2_20260704.csv",
    "Q2_20260705.csv", "Q2_20260706.csv", "Q2_20260707.csv",
]
DATE_RANGE_LABEL = "Jun 26 - Jul 7, 2026"

dfs = sc.load_csv_files(SCRIPT_DIR, JULY_FILES_ALL)
result = sc.classify_all_autoclaves(dfs, verbose=False)

bar_per_h, kg_per_h = mc.idle_leakage_stats(result, min_duration_h=MIN_IDLE_DURATION_H)
bar_per_h_mean = [bar_per_h[ac1][0] for ac1 in range(1, sc.N_AUTOCLAVES + 1)]
bar_per_h_sd   = [bar_per_h[ac1][1] for ac1 in range(1, sc.N_AUTOCLAVES + 1)]
bar_per_h_n    = [bar_per_h[ac1][2] for ac1 in range(1, sc.N_AUTOCLAVES + 1)]
KG_PER_TON = 1000.0
ton_per_h_mean = [kg_per_h[ac1][0] / KG_PER_TON for ac1 in range(1, sc.N_AUTOCLAVES + 1)]
ton_per_h_sd   = [kg_per_h[ac1][1] / KG_PER_TON for ac1 in range(1, sc.N_AUTOCLAVES + 1)]
ton_per_h_n    = [kg_per_h[ac1][2] for ac1 in range(1, sc.N_AUTOCLAVES + 1)]

# ── Chart 1: bar/h ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(mc.AUTOCLAVE_LABELS, bar_per_h_mean, color='#1f5fa6', edgecolor='black')
mc.bar_labels(ax, bars, bar_per_h_mean, bar_per_h_sd, fmt="{:.4f}", sd_fmt="SD={:.4f}")
ax.set_ylabel('leakage rate [bar/h]')
ax.set_xlabel('autoclaves')
ax.set_title(f'Leakage / Slope Test During Idle (minor fluctuation)\n{DATE_RANGE_LABEL} — '
             f'linear-fit slope of raw pressure vs. time, idle runs ≥ {int(MIN_IDLE_DURATION_H*60)} min only',
             fontsize=13, fontweight='bold')
fig.text(0.5, -0.02,
         f'bar/h = linear-fit slope of raw pressure vs. time across each idle segment (n shown = usable segments after the {int(MIN_IDLE_DURATION_H*60)}-min filter)  |  '
         f'idle segments shorter than {int(MIN_IDLE_DURATION_H*60)} min are excluded to avoid slope noise from tiny windows',
         ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'idle_leakage_bar_per_h.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Chart 2: ton/h ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(mc.AUTOCLAVE_LABELS, ton_per_h_mean, color='#2e7d32', edgecolor='black')
mc.bar_labels(ax, bars, ton_per_h_mean, ton_per_h_sd, fmt="{:.4f}", sd_fmt="SD={:.4f}")
ax.set_ylabel('leakage rate [ton/h]')
ax.set_xlabel('autoclaves')
ax.set_title(f'Idle Leakage — Real Mass Rate (IAPWS-97 saturated-vapor density)\n{DATE_RANGE_LABEL} — '
             f'idle runs ≥ {int(MIN_IDLE_DURATION_H*60)} min only', fontsize=13, fontweight='bold')
fig.text(0.5, -0.02,
         f'ton/h = [ρ(p_end) − ρ(p_start)] × V_free / duration / 1000, per idle segment  |  '
         f'V_free = {sp.V_FREE[1]:.2f} m³ (AC1-7), {sp.V_FREE[8]:.2f} m³ (AC8)  |  '
         f'ρ = IAPWS-97 saturated-vapor density at gauge+{sp.ATM_BAR:.3f} bar abs',
         ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'idle_leakage_ton_per_h.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("bar/h  mean+/-SD (n, flagged):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {bar_per_h_mean[i]:+.4f} +/- {bar_per_h_sd[i]:.4f}  (n={bar_per_h_n[i]}, flagged={bar_per_h[ac1][3]})")
print("ton/h  mean+/-SD (n, flagged):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {ton_per_h_mean[i]:+.4f} +/- {ton_per_h_sd[i]:.4f}  (n={ton_per_h_n[i]}, flagged={kg_per_h[ac1][3]})")
print("Saved idle_leakage_bar_per_h.png and idle_leakage_ton_per_h.png")
