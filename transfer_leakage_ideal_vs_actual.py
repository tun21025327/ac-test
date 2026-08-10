"""Ideal-vs-actual transfer leakage: what if the donor/receiver pressures
had fully equalized after Transfer Out 2 / Transfer In 1, instead of
stopping wherever they actually stopped?

For each donor->receiver transfer event (paired the same way
transfer_efficiency.py pairs them — by time overlap on the shared clock):
  1. Take the mass in each chamber at the END of the transfer (not the
     start — see below):
     m_donor_end = rho(p_donor_end) * V_donor
     m_recv_end  = rho(p_recv_end)  * V_recv
  2. An IDEAL transfer would let the valve stay open a little longer, until
     both chambers reach the same pressure (same density, since steam
     density is a function of pressure only), so the combined end-state
     mass redistributes in proportion to volume:
         ideal_donor_leftover = (m_donor_end + m_recv_end) * V_donor / (V_donor + V_recv)
  3. Compare that to the actual donor mass at the end of the transfer
     (m_donor_end, the same "leakage" figure as transfer_leakage_mass.py).

  Why equalize from the END states and not the START states: an earlier
  version of this script mass-balanced from the start of the transfer
  (donor ~4-5 bar, receiver just out of vacuum near 0 bar) and got an
  "ideal" leftover BIGGER than the actual leftover — backwards, since a
  transfer that stops early should leave the donor with MORE than a fully
  equalized transfer would, not less. The cause: most of the steam that
  leaves the donor early on condenses against the still-cold receiver/load
  (that IS the point of the transfer — releasing latent heat to warm
  things up), so it vanishes from a vapor-density mass balance without
  ever showing up as receiver pressure. Checking one real cycle confirmed
  it directly: donor+receiver vapor mass dropped from 317 kg at the start
  of the window to 235 kg at the end — a real ~26% "loss" to condensate,
  not a bug. By the end of the transfer both chambers are much closer to
  thermal equilibrium, so a small further equalization step from the two
  actual end pressures isn't confounded by that same fresh condensation.
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

# donor's transfer_out2 <-> receiver's transfer_in1 (see stage_classifier.ROTATION docstring)
SERVES_VIA_OUT2 = {cfg['ti1_src']: x for x, cfg in sc.ROTATION.items()}


def overlapping_run(target_s, target_e, t_donor, t_recv, recv_runs):
    d0, d1 = t_donor[target_s], t_donor[target_e - 1]
    best, best_overlap = None, 0.0
    for (s, e) in recv_runs:
        r0, r1 = t_recv[s], t_recv[e - 1]
        ov = min(d1, r1) - max(d0, r0)
        if ov > best_overlap:
            best_overlap, best = ov, (s, e)
    return best


actual_mean, actual_sd = [], []
ideal_mean, ideal_sd = [], []
waste_mean, waste_sd = [], []
n_pairs_list = []

for donor in range(1, sc.N_AUTOCLAVES + 1):
    d0 = donor - 1
    p_donor = result.pressures[d0]
    t_donor = result.times[d0]
    offset_d = sc.SENSOR_OFFSETS.get(donor, 0.0)
    V_donor = sp.V_FREE[donor]

    to2_runs = sc.contiguous_runs(result.stages[d0] == 'transfer_out2')

    actual_vals, ideal_vals, waste_vals = [], [], []

    if donor in SERVES_VIA_OUT2:
        recv = SERVES_VIA_OUT2[donor]
        r0 = recv - 1
        p_recv = result.pressures[r0]
        t_recv = result.times[r0]
        offset_r = sc.SENSOR_OFFSETS.get(recv, 0.0)
        V_recv = sp.V_FREE[recv]
        recv_runs = sc.contiguous_runs(result.stages[r0] == 'transfer_in1')

        for (s, e) in to2_runs:
            if e <= s:
                continue
            match = overlapping_run(s, e, t_donor, t_recv, recv_runs)
            if not match:
                continue
            rs, re = match
            if re <= rs:
                continue

            p_donor_end = p_donor[e - 1] + offset_d
            p_recv_end = p_recv[re - 1] + offset_r

            # Ideal = equalize from the ACTUAL END states, not the start.
            # Equalizing from the start (donor ~4-5 bar, receiver just out of
            # vacuum near 0) was wrong: most of the steam that leaves the
            # donor early on condenses against the still-cold receiver/load
            # (that's the point of the transfer — releasing latent heat to
            # warm things up), so it vanishes from a vapor-density mass
            # balance without ever showing up as receiver pressure. That
            # made the "ideal" leftover look inflated. By the end of the
            # transfer both chambers are much closer to thermal equilibrium,
            # so a small further equalization step from there isn't
            # confounded by fresh condensation the same way.
            m_donor_end = sp.rho_sat_vapor(p_donor_end) * V_donor
            m_recv_end = sp.rho_sat_vapor(p_recv_end) * V_recv
            total_mass_end = m_donor_end + m_recv_end

            ideal_donor_leftover = total_mass_end * V_donor / (V_donor + V_recv)
            actual_donor_leftover = m_donor_end

            actual_vals.append(actual_donor_leftover)
            ideal_vals.append(ideal_donor_leftover)
            waste_vals.append(actual_donor_leftover - ideal_donor_leftover)

    am, asd, an = mc.mean_sd(actual_vals)
    im, isd, _ = mc.mean_sd(ideal_vals)
    wm, wsd, _ = mc.mean_sd(waste_vals)
    actual_mean.append(am); actual_sd.append(asd)
    ideal_mean.append(im); ideal_sd.append(isd)
    waste_mean.append(wm); waste_sd.append(wsd)
    n_pairs_list.append(an)

# ── Chart 1: ideal transfer leak vs actual transfer leak ────────────────────
# Both bars are the same quantity (donor steam mass wasted/unclaimed after
# Transfer Out 2 completes) — "ideal" is the unavoidable minimum leak if the
# donor/receiver pressures had fully equalized; "actual" is what really got
# left behind. Neither of these touches idle-phase leakage (idle_leakage.py)
# — this is strictly the transfer-out leak.
x = np.arange(sc.N_AUTOCLAVES)
w = 0.36
fig, ax = plt.subplots(figsize=(12, 7))
b1 = ax.bar(x - w/2, ideal_mean, w, yerr=ideal_sd, capsize=4, color='#8fbf8f', edgecolor='black', label='Ideal transfer leak (pressure-equalized)')
b2 = ax.bar(x + w/2, actual_mean, w, yerr=actual_sd, capsize=4, color='#7a1f1f', edgecolor='black', label='Actual transfer leak (observed)')
for bar, m, sdv in zip(b1, ideal_mean, ideal_sd):
    ax.annotate(f"{m:.1f}\nSD={sdv:.1f}", (bar.get_x()+bar.get_width()/2, m+sdv), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8)
for bar, m, sdv in zip(b2, actual_mean, actual_sd):
    ax.annotate(f"{m:.1f}\nSD={sdv:.1f}", (bar.get_x()+bar.get_width()/2, m+sdv), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
ax.margins(y=0.18)
ax.set_xticks(x); ax.set_xticklabels(mc.AUTOCLAVE_LABELS)
ax.set_ylabel('Transfer leak — donor steam mass after Transfer Out 2 (kg)')
ax.set_xlabel('autoclave (donor)')
ax.set_title(f'Ideal Transfer Leak vs. Actual Transfer Leak, {DATE_RANGE_LABEL}\n'
             f'Ideal = donor+receiver equalize further from their actual end pressures, split by chamber volume',
             fontsize=12.5, fontweight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'transfer_leak_ideal_vs_actual.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Chart 2: actual transfer leak minus ideal transfer leak ────────────────
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(mc.AUTOCLAVE_LABELS, waste_mean, yerr=waste_sd, capsize=4, color='#b5651d', edgecolor='black')
mc.bar_labels(ax, bars, waste_mean, waste_sd, fmt="{:.1f}", sd_fmt="SD={:.1f}")
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Actual transfer leak minus ideal transfer leak (kg)')
ax.set_xlabel('autoclave (donor)')
ax.set_title(f'Actual Transfer Leak vs. Ideal Transfer Leak — Difference, {DATE_RANGE_LABEL}\n'
             f'positive = donor kept more steam (leaked more) than a fully-equalized transfer would have',
             fontsize=12.5, fontweight='bold')
fig.text(0.5, -0.02,
         'difference = actual transfer leak − ideal transfer leak, per transfer event, then averaged',
         ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'transfer_leak_actual_minus_ideal.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("Ideal transfer leak vs actual transfer leak, after Transfer Out 2 (kg, mean +/- SD, n pairs):")
for i, ac1 in enumerate(range(1, sc.N_AUTOCLAVES + 1)):
    print(f"  AC{ac1}: ideal={ideal_mean[i]:.1f}+/-{ideal_sd[i]:.1f}  actual={actual_mean[i]:.1f}+/-{actual_sd[i]:.1f}  "
          f"diff={waste_mean[i]:+.1f}+/-{waste_sd[i]:.1f}  (n={n_pairs_list[i]})")
print("Saved transfer_leak_ideal_vs_actual.png and transfer_leak_actual_minus_ideal.png")
