"""Transfer-out efficiency (donor leftover pressure) and donor-vs-receiver
pressure at the end of each rotation-matched transfer.

Uses the SAME rotation-matched transfer_out1/transfer_out2 labels that
stage_classifier.classify_all_autoclaves() already computed (only applied
where the overlap/drop checks passed — see stage_classifier.py's rotation-
matching step) instead of re-deriving a separate matching pass. Donor and
receiver runs for the same physical transfer are paired by time overlap on
the shared, common time axis (all autoclaves share day offsets).
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

# invert ROTATION: for a donor AC, which AC does it serve via transfer_out1 / transfer_out2
# (ti1_src / ti2_src values are each a permutation of 1..8, so this inversion is 1:1)
SERVES_VIA_OUT1 = {cfg['ti2_src']: x for x, cfg in sc.ROTATION.items()}
SERVES_VIA_OUT2 = {cfg['ti1_src']: x for x, cfg in sc.ROTATION.items()}


def overlapping_run(target_s, target_e, t_donor, t_recv, recv_runs):
    """Pick the receiver run whose time interval overlaps most with the
    donor run's [t_donor[target_s], t_donor[target_e-1]] interval."""
    d0, d1 = t_donor[target_s], t_donor[target_e - 1]
    best, best_overlap = None, 0.0
    for (s, e) in recv_runs:
        r0, r1 = t_recv[s], t_recv[e - 1]
        ov = min(d1, r1) - max(d0, r0)
        if ov > best_overlap:
            best_overlap, best = ov, (s, e)
    return best


# ── donor leftover pressure (efficiency) + paired donor/receiver pressure ──────
donor_after_to1 = {ac1: [] for ac1 in range(1, sc.N_AUTOCLAVES + 1)}
donor_after_to2 = {ac1: [] for ac1 in range(1, sc.N_AUTOCLAVES + 1)}
donor_to1_durations = {ac1: [] for ac1 in range(1, sc.N_AUTOCLAVES + 1)}
donor_to2_durations = {ac1: [] for ac1 in range(1, sc.N_AUTOCLAVES + 1)}
pair_values = {}  # (donor, receiver) -> {'donor': [...], 'receiver': [...]}

for donor in range(1, sc.N_AUTOCLAVES + 1):
    d0 = donor - 1
    p_donor = result.pressures[d0]
    t_donor = result.times[d0]
    stages_donor = result.stages[d0]

    to1_runs = sc.contiguous_runs(stages_donor == 'transfer_out1')
    to2_runs = sc.contiguous_runs(stages_donor == 'transfer_out2')

    donor_after_to1[donor] = [p_donor[e - 1] for (s, e) in to1_runs if e > s]
    donor_after_to2[donor] = [p_donor[e - 1] for (s, e) in to2_runs if e > s]
    donor_to1_durations[donor] = [t_donor[e - 1] - t_donor[s] for (s, e) in to1_runs if e > s]
    donor_to2_durations[donor] = [t_donor[e - 1] - t_donor[s] for (s, e) in to2_runs if e > s]

    # transfer_out1 -> receiver's transfer_in2 ; transfer_out2 -> receiver's transfer_in1
    if donor in SERVES_VIA_OUT1:
        recv = SERVES_VIA_OUT1[donor]
        r0 = recv - 1
        p_recv, t_recv = result.pressures[r0], result.times[r0]
        recv_runs = sc.contiguous_runs(result.stages[r0] == 'transfer_in2')
        key = (donor, recv)
        pair_values.setdefault(key, {'donor': [], 'receiver': []})
        for (s, e) in to1_runs:
            if e <= s:
                continue
            match = overlapping_run(s, e, t_donor, t_recv, recv_runs)
            if match:
                rs, re = match
                pair_values[key]['donor'].append(p_donor[e - 1])
                pair_values[key]['receiver'].append(p_recv[re - 1])

    if donor in SERVES_VIA_OUT2:
        recv = SERVES_VIA_OUT2[donor]
        r0 = recv - 1
        p_recv, t_recv = result.pressures[r0], result.times[r0]
        recv_runs = sc.contiguous_runs(result.stages[r0] == 'transfer_in1')
        key = (donor, recv)
        pair_values.setdefault(key, {'donor': [], 'receiver': []})
        for (s, e) in to2_runs:
            if e <= s:
                continue
            match = overlapping_run(s, e, t_donor, t_recv, recv_runs)
            if match:
                rs, re = match
                pair_values[key]['donor'].append(p_donor[e - 1])
                pair_values[key]['receiver'].append(p_recv[re - 1])

# ── Chart 1: Transfer-Out Efficiency ────────────────────────────────────────
to1_mean = [mc.mean_sd(donor_after_to1[ac1])[0] for ac1 in range(1, sc.N_AUTOCLAVES + 1)]
to1_sd   = [mc.mean_sd(donor_after_to1[ac1])[1] for ac1 in range(1, sc.N_AUTOCLAVES + 1)]
to2_mean = [mc.mean_sd(donor_after_to2[ac1])[0] for ac1 in range(1, sc.N_AUTOCLAVES + 1)]
to2_sd   = [mc.mean_sd(donor_after_to2[ac1])[1] for ac1 in range(1, sc.N_AUTOCLAVES + 1)]

x = np.arange(sc.N_AUTOCLAVES)
w = 0.36
fig, ax = plt.subplots(figsize=(12, 7))
b1 = ax.bar(x - w/2, to1_mean, w, yerr=to1_sd, capsize=4, color='#e8a598', edgecolor='black', label='After Transfer Out 1 (ideal ~4 bar)')
b2 = ax.bar(x + w/2, to2_mean, w, yerr=to2_sd, capsize=4, color='#7a1f1f', edgecolor='black', label='After Transfer Out 2 (ideal ~0 bar = fully drained)')
for bar, m, sdv in zip(b1, to1_mean, to1_sd):
    ax.annotate(f"{m:.2f}±{sdv:.2f}", (bar.get_x()+bar.get_width()/2, m+sdv), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
for bar, m, sdv in zip(b2, to2_mean, to2_sd):
    ax.annotate(f"{m:.2f}±{sdv:.2f}", (bar.get_x()+bar.get_width()/2, m+sdv), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
ax.axhline(4.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(sc.N_AUTOCLAVES - 0.5, 1.05, '1 bar', color='gray', fontsize=9, ha='right')
ax.set_xticks(x); ax.set_xticklabels(mc.AUTOCLAVE_LABELS)
ax.set_ylabel('Donor pressure (bar)')
ax.set_title(f'Transfer-Out Efficiency — Leftover Donor Pressure\n{DATE_RANGE_LABEL}  |  Higher "after TO2" = more unused steam left',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'transfer_out_efficiency.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Chart 2: Donor vs Receiver pressure per rotation pair ──────────────────────
pair_keys = sorted(pair_values.keys(), key=lambda k: (k[0], k[1]))
donor_m, donor_sd, recv_m, recv_sd, labels = [], [], [], [], []
for (d, r) in pair_keys:
    dm, dsd, _ = mc.mean_sd(pair_values[(d, r)]['donor'])
    rm, rsd, _ = mc.mean_sd(pair_values[(d, r)]['receiver'])
    donor_m.append(dm); donor_sd.append(dsd); recv_m.append(rm); recv_sd.append(rsd)
    labels.append(f"{d}→{r}")

x = np.arange(len(pair_keys))
w = 0.38
fig, ax = plt.subplots(figsize=(16, 8))
b1 = ax.bar(x - w/2, donor_m, w, yerr=donor_sd, capsize=3, color='#7a1f1f', edgecolor='black', label='Donor pressure (end of transfer)')
b2 = ax.bar(x + w/2, recv_m, w, yerr=recv_sd, capsize=3, color='#2f6faa', edgecolor='black', label='Receiver pressure (end of transfer)')
for bar, m, sdv in zip(b1, donor_m, donor_sd):
    ax.annotate(f"{m:.2f}±{sdv:.2f}", (bar.get_x()+bar.get_width()/2, m+sdv), xytext=(0,2), textcoords='offset points', ha='center', fontsize=7, rotation=90)
for bar, m, sdv in zip(b2, recv_m, recv_sd):
    ax.annotate(f"{m:.2f}±{sdv:.2f}", (bar.get_x()+bar.get_width()/2, m+sdv), xytext=(0,2), textcoords='offset points', ha='center', fontsize=7, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel('Donor → Receiver autoclave (rotation pair)')
ax.set_ylabel('Pressure at end of transfer (bar)')
ax.set_title(f'Donor vs. Receiver Pressure at End of Each Transfer\n{DATE_RANGE_LABEL} — gap between bars = pressure left undelivered',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'transfer_pair_delta.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Chart 3: Leftover steam above 1 bar, per donor AC (donor only, TO2 only) ──
# Donor-only quantity: how much steam (mass, not raw pressure) was left
# sitting above a 1 bar floor at the end of Transfer Out 2 specifically —
# the truly final leftover before steam_off (Transfer Out 1's own
# end-of-transfer pressure is excluded, since most of that continues to be
# drained further during TO2 right after and would otherwise be double
# counted). TOTAL across the whole date range (not an average rate), in
# tons — same definition and convention as savable_steam.py.
FLOOR_BAR = 1.0
KG_PER_TON = 1000.0

leftover_ton, leftover_n, leftover_avg_min = [], [], []
for ac1 in range(1, sc.N_AUTOCLAVES + 1):
    offset = sc.SENSOR_OFFSETS.get(ac1, 0.0)
    floor_mass = sp.rho_sat_vapor(FLOOR_BAR) * sp.V_FREE[ac1]
    pooled_p = donor_after_to2[ac1]
    pooled_dur = donor_to2_durations[ac1]

    savable_kg = [max(0.0, sp.rho_sat_vapor(p + offset) * sp.V_FREE[ac1] - floor_mass) for p in pooled_p]
    leftover_ton.append(sum(savable_kg) / KG_PER_TON)
    leftover_n.append(len(savable_kg))
    avg_h, _, _ = mc.mean_sd(pooled_dur)
    leftover_avg_min.append(avg_h * 60 if np.isfinite(avg_h) else np.nan)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.bar(mc.AUTOCLAVE_LABELS, leftover_ton, color='#1f7a4c', edgecolor='black')
for bar, v, n, dmin in zip(bars, leftover_ton, leftover_n, leftover_avg_min):
    ax.annotate(f"{v:.3f} ton", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 6),
                textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')
    dur_txt = f"{dmin:.0f} min" if np.isfinite(dmin) else "n/a"
    ax.annotate(f"n={n} cycles\navg TO2={dur_txt}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 26),
                textcoords='offset points', ha='center', va='bottom', fontsize=8.5)
ax.margins(y=0.22)
ax.set_ylabel('leftover steam above 1 bar, total [ton]')
ax.set_xlabel('donor autoclave')
ax.set_title(f'Leftover Steam Above 1 Bar, Per Donor AC — {DATE_RANGE_LABEL}\n'
             f'total mass (IAPWS-97), donor only, Transfer Out 2 only (excludes Transfer Out 1)',
             fontsize=12.5, fontweight='bold')
fig.text(0.5, -0.02,
         f'leftover = max(0, ρ(p_true at end of Transfer Out 2) − ρ({FLOOR_BAR:.0f} bar)) × V_free, summed over every Transfer Out 2 event  |  '
         f'V_free = {sp.V_FREE[1]:.2f} m³ (AC1-7), {sp.V_FREE[8]:.2f} m³ (AC8)',
         ha='center', fontsize=8.5, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'transfer_donor_leftover_above_1bar.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("Transfer-out efficiency (after TO1 / after TO2, mean +/- SD):")
for ac1 in range(1, sc.N_AUTOCLAVES + 1):
    print(f"  AC{ac1}: TO1={to1_mean[ac1-1]:.2f}+/-{to1_sd[ac1-1]:.2f}  TO2={to2_mean[ac1-1]:.2f}+/-{to2_sd[ac1-1]:.2f}")
print("Donor/receiver pairs:", len(pair_keys))
print("Leftover steam above 1 bar, per donor AC (ton total, n events, avg transfer duration):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: {leftover_ton[i]:.4f} ton  (n={leftover_n[i]}, avg={leftover_avg_min[i]:.1f} min)")
print("Saved transfer_out_efficiency.png, transfer_pair_delta.png, and transfer_donor_leftover_above_1bar.png")
