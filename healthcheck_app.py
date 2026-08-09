"""Autoclave Health Check — web app.

Upload steam-log CSV file(s) (same format as Q2_20260626.csv etc. — Date,
Time, Pressure Autocalve 1-8, Temperature Autocalve 1-8, Vacuum pump,
Main Meter steam flow, ...) and get every health-check chart this project
produces, computed live from stage_classifier.classify_all_autoclaves() —
the same canonical pipeline every saved PNG in this project comes from.
No duplicated analysis logic: every chart function below is the plotting
half of the matching root-folder script (idle_leakage.py,
vacuum_depth_speed.py, etc.), just returning a Figure instead of saving one.

Run with:
    streamlit run healthcheck_app.py
"""
import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

import stage_classifier as sc
import steam_properties as sp
import metrics_common as mc

st.set_page_config(page_title="Autoclave Health Check", layout="wide")

MIN_IDLE_DURATION_H = 0.25
KG_PER_TON = 1000.0
FLOOR_BAR = 1.0
STEAM_PRICE_THB_PER_TON = 850.0  # matches the pptx's own baht conversion for savable-steam/idle-leakage loss estimates


# ── Data loading ──────────────────────────────────────────────────────────────
def load_uploaded_csvs(files):
    """Same time(h) construction as stage_classifier.load_csv_files, but
    reading from Streamlit's UploadedFile objects instead of paths."""
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if 'time(h)' not in df.columns:
            tod = pd.to_datetime(df['Time'], format='%H:%M:%S')
            df['time(h)'] = tod.dt.hour + tod.dt.minute / 60 + tod.dt.second / 3600
        dfs.append(df)
    return dfs


def detect_ac4_fix_date(dfs):
    """AC4's sensor was physically fixed after Jul 7, 2026 — auto-detect
    from the files' own Date column whether the correction should apply."""
    try:
        first_date = pd.to_datetime(dfs[0]['Date'].iloc[0], dayfirst=True)
        return first_date >= pd.Timestamp('2026-07-08')
    except Exception:
        return None  # unknown — let the user decide


# ── Chart builders (each = the plotting half of the matching script) ──────────
def fig_vacuum(result, vacuum_target_bar, speed_target_bar, speed_target_min, date_label):
    depth_ratio_mean, depth_ratio_sd, pass_rate, depth_n = [], [], [], []
    speed_mean, speed_sd, speed_n, speed_n_outliers = [], [], [], []
    SPEED_OUTLIER_CUTOFF_MIN = 35.0

    for ac0 in range(sc.N_AUTOCLAVES):
        ac1 = ac0 + 1
        t = result.times[ac0]
        p_raw = result.pressures[ac0]
        offset = sc.SENSOR_OFFSETS.get(ac1, 0.0)
        p_true = p_raw + offset
        runs = mc.vacuum_runs(result.stages[ac0], t=t)
        ratios, speeds_min, n_outliers = [], [], 0
        for (s, e) in runs:
            seg_min = p_true[s:e].min()
            ratios.append(seg_min / vacuum_target_bar)
            below = np.where(p_true[s:e] <= speed_target_bar)[0]
            if len(below):
                minutes = (t[s + below[0]] - t[s]) * 60.0
                if minutes > SPEED_OUTLIER_CUTOFF_MIN:
                    n_outliers += 1
                else:
                    speeds_min.append(minutes)
        m, sdv, n = mc.mean_sd(ratios)
        depth_ratio_mean.append(m); depth_ratio_sd.append(sdv); depth_n.append(n)
        pass_rate.append(np.mean(np.array(ratios) >= 1.0) if ratios else np.nan)
        m2, sdv2, n2 = mc.mean_sd(speeds_min)
        speed_mean.append(m2); speed_sd.append(sdv2); speed_n.append(n2)
        speed_n_outliers.append(n_outliers)

    fig1, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, sc.N_AUTOCLAVES))
    bars = ax.bar(mc.AUTOCLAVE_LABELS, depth_ratio_mean, color=colors, edgecolor='black')
    mc.bar_labels(ax, bars, depth_ratio_mean, depth_ratio_sd, fmt="{:.2f}", sd_fmt="SD={:.2f}")
    for bar, pr, n in zip(bars, pass_rate, depth_n):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.06, f"{pr:.0%} pass\n(n={n})",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')
    ax.axhline(1.0, color='red', linewidth=1.5, label='Target = 1.00 (exact)')
    ax.set_ylabel(f'Depth Ratio (true pressure / target {vacuum_target_bar} bar)')
    ax.set_title(f'Vacuum Depth — Mean per Autoclave, {date_label}', fontsize=12.5, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()

    fig2, ax = plt.subplots(figsize=(11, 7))
    bars = ax.bar(mc.AUTOCLAVE_LABELS, speed_mean, color=colors, edgecolor='black')
    mc.bar_labels(ax, bars, speed_mean, speed_sd, fmt="{:.2f} min", sd_fmt="SD={:.2f} min")
    for bar, n_out in zip(bars, speed_n_outliers):
        if n_out:
            ax.annotate(f"{n_out} outlier{'s' if n_out != 1 else ''}\nexcluded",
                        (bar.get_x() + bar.get_width() / 2, 0), xytext=(0, 6),
                        textcoords='offset points', ha='center', va='bottom', fontsize=8, color='#8a1f1f')
    ax.axhline(speed_target_min, color='red', linewidth=1.5, label=f'Target = {speed_target_min} min')
    ax.set_ylabel(f'Time to reach {speed_target_bar} bar (min)')
    ax.set_title(f'Vacuum Speed — Mean per Autoclave, {date_label}', fontsize=12.5, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig1, fig2, depth_ratio_mean, depth_ratio_sd, speed_mean, speed_sd


def fig_idle_leakage(result, date_label):
    bar_h, kg_h = mc.idle_leakage_stats(result, min_duration_h=MIN_IDLE_DURATION_H)
    ton_mean = [kg_h[a][0] / KG_PER_TON for a in range(1, sc.N_AUTOCLAVES + 1)]
    ton_sd = [kg_h[a][1] / KG_PER_TON for a in range(1, sc.N_AUTOCLAVES + 1)]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(mc.AUTOCLAVE_LABELS, ton_mean, color='#2e7d32', edgecolor='black')
    mc.bar_labels(ax, bars, ton_mean, ton_sd, fmt="{:.4f}", sd_fmt="SD={:.4f}")
    ax.set_ylabel('leakage rate [ton/h]')
    ax.set_title(f'Idle Leakage — Real Mass Rate, {date_label}', fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    return fig, bar_h, kg_h


def fig_idle_total(result, date_label):
    totals = mc.idle_total_leakage_stats(result, min_duration_h=MIN_IDLE_DURATION_H)
    acs = list(range(1, sc.N_AUTOCLAVES + 1))
    total_ton = [totals[a][0] / KG_PER_TON for a in acs]
    n_cycles = [totals[a][1] for a in acs]
    avg_idle_h = [totals[a][3] for a in acs]
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.bar(mc.AUTOCLAVE_LABELS, total_ton, color='#7a1f1f', edgecolor='black')
    for bar, v, n, avg_h in zip(bars, total_ton, n_cycles, avg_idle_h):
        va, pts = ('top', -6) if v < 0 else ('bottom', 6)
        ax.annotate(f"{v:.3f} ton", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, pts),
                    textcoords='offset points', ha='center', va=va, fontsize=11, fontweight='bold')
        ax.annotate(f"n={n}\navg={avg_h*60:.0f}min", (bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, pts + (-16 if v < 0 else 16)), textcoords='offset points',
                    ha='center', va=va, fontsize=8)
    ax.margins(y=0.22)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('total idle leakage [ton]')
    ax.set_title(f'Total Idle Leakage, {date_label}', fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    return fig, totals


def fig_condensate(result, date_label):
    from scipy.signal import find_peaks
    DIP_THRESHOLD_C = 0.15
    rate_mean, rate_sd, total_count = [], [], []
    for ac0 in range(sc.N_AUTOCLAVES):
        t = result.times[ac0]
        temp = st.session_state['temps'][ac0]
        stages = result.stages[ac0]
        hold_mask = (stages == 'keeping') | (stages == 'idle')
        runs = sc.contiguous_runs(hold_mask)
        runs = [(s, e) for (s, e) in runs if (t[e - 1] - t[s]) >= MIN_IDLE_DURATION_H]
        rates, counts = [], []
        for (s, e) in runs:
            troughs, _ = find_peaks(-temp[s:e], prominence=DIP_THRESHOLD_C)
            dur_h = t[e - 1] - t[s]
            rates.append(len(troughs) / dur_h)
            counts.append(len(troughs))
        m, sdv, n = mc.mean_sd(rates)
        rate_mean.append(m); rate_sd.append(sdv)
        total_count.append(sum(counts))

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.bar(mc.AUTOCLAVE_LABELS, total_count, color='#5b3fa0', edgecolor='black')
    for bar, c in zip(bars, total_count):
        ax.annotate(str(c), (bar.get_x() + bar.get_width() / 2, c), xytext=(0, 4),
                    textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('total condensate cycles (count)')
    ax.set_title(f'Condensate Cycle Count, {date_label}', fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    return fig, total_count


def fig_temp_spread(result, date_label):
    temp_deltas_mean, temp_deltas_sd = [], []
    for ac0 in range(sc.N_AUTOCLAVES):
        t = result.times[ac0]
        temp = st.session_state['temps'][ac0]
        stages = result.stages[ac0]
        hold_mask = (stages == 'keeping') | (stages == 'idle')
        runs = sc.contiguous_runs(hold_mask)
        runs = [(s, e) for (s, e) in runs if (t[e - 1] - t[s]) >= MIN_IDLE_DURATION_H]
        deltas = [temp[s:e].max() - temp[s:e].min() for (s, e) in runs]
        m, sdv, n = mc.mean_sd(deltas)
        temp_deltas_mean.append(m); temp_deltas_sd.append(sdv)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(mc.AUTOCLAVE_LABELS, temp_deltas_mean, color='#d1603d', edgecolor='black')
    mc.bar_labels(ax, bars, temp_deltas_mean, temp_deltas_sd, fmt="{:.2f}", sd_fmt="SD={:.2f}")
    ax.set_ylabel('ΔT = max - min temperature (°C)')
    ax.set_title(f'Temperature Spread During Hold, {date_label}', fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    return fig, temp_deltas_mean, temp_deltas_sd


def fig_transfer_pct(result, date_label):
    fresh_pct_mean, fresh_pct_sd = [], []
    transferred_pct_mean, transferred_pct_sd = [], []
    n_cycles = []
    for ac0 in range(sc.N_AUTOCLAVES):
        p = result.pressures[ac0]
        stages = result.stages[ac0]
        vac_runs = sc.contiguous_runs(stages == 'vacuum')
        fresh_runs = sc.contiguous_runs(stages == 'fresh_steam')
        fresh_pcts, transferred_pcts = [], []
        for (fs, fe) in fresh_runs:
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

    fig, ax1 = plt.subplots(figsize=(12, 7))
    x = np.arange(sc.N_AUTOCLAVES)
    w = 0.36
    b1 = ax1.bar(x - w/2, fresh_pct_mean, w, yerr=fresh_pct_sd, capsize=4, color='#2f6faa', edgecolor='black', label='fresh steam [%]')
    b2 = ax1.bar(x + w/2, transferred_pct_mean, w, yerr=transferred_pct_sd, capsize=4, color='#e8b830', edgecolor='black', label='transferred steam [%]')
    for bar, v, sdv in zip(b1, fresh_pct_mean, fresh_pct_sd):
        ax1.annotate(f"{v:.1f}", (bar.get_x()+bar.get_width()/2, v+sdv), xytext=(0,14), textcoords='offset points', ha='center', fontsize=9, fontweight='bold', color='#1a3f66')
        ax1.annotate(f"SD={sdv:.1f}", (bar.get_x()+bar.get_width()/2, v+sdv), xytext=(0,3), textcoords='offset points', ha='center', fontsize=7, color='#1a3f66')
    for bar, v, sdv in zip(b2, transferred_pct_mean, transferred_pct_sd):
        ax1.annotate(f"{v:.1f}", (bar.get_x()+bar.get_width()/2, v+sdv), xytext=(0,14), textcoords='offset points', ha='center', fontsize=9, fontweight='bold', color='#8a6810')
        ax1.annotate(f"SD={sdv:.1f}", (bar.get_x()+bar.get_width()/2, v+sdv), xytext=(0,3), textcoords='offset points', ha='center', fontsize=7, color='#8a6810')
    ax1.set_ylim(0, 115)
    ax1.set_xticks(x); ax1.set_xticklabels(mc.AUTOCLAVE_LABELS)
    ax1.set_ylabel('steam rate [%]')
    ax2 = ax1.twinx()
    ax2.plot(x, n_cycles, color='black', marker='o', linewidth=2, label='cycles')
    for xi, n in zip(x, n_cycles):
        ax2.annotate(str(n), (xi, n), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    ax2.set_ylim(0, max(n_cycles) * 1.6 if n_cycles and max(n_cycles) else 1)
    ax2.set_ylabel('number of cycles [-]')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.set_title(f'Fresh vs. Transferred Steam, {date_label}', fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    return fig, fresh_pct_mean, transferred_pct_mean, n_cycles


def fig_price_loss(idle_totals, savable_ton, date_label):
    """Combined transfer + idle steam loss, converted to THB at
    STEAM_PRICE_THB_PER_TON — same conversion and methodology the pptx used
    for its own savable-steam/idle-leakage baht estimates (1 ton fresh
    steam = 850 THB). Transfer loss here is the same 'savable steam' figure
    (leftover above 1 bar after Transfer Out 2) shown in the Savable Steam
    tab, not the full post-TO2 leftover — matching what the pptx itself
    based its money figure on."""
    plt.rcParams['font.family'] = 'Tahoma'  # renders Thai glyphs
    idle_ton = [abs(idle_totals[a][0]) / KG_PER_TON for a in range(1, sc.N_AUTOCLAVES + 1)]
    transfer_ton = list(savable_ton)
    combined_ton = [i + t for i, t in zip(idle_ton, transfer_ton)]
    combined_thb = [c * STEAM_PRICE_THB_PER_TON for c in combined_ton]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(sc.N_AUTOCLAVES)
    w = 0.36
    b1 = ax.bar(x - w/2, idle_ton, w, color='#8e6bb0', edgecolor='black', label='idle leakage (ตัน)')
    b2 = ax.bar(x + w/2, transfer_ton, w, color='#1f7a4c', edgecolor='black', label='transfer leakage / savable steam (ตัน)')
    for bar, v in zip(b1, idle_ton):
        ax.annotate(f"{v:.3f}", (bar.get_x()+bar.get_width()/2, v), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8)
    for bar, v in zip(b2, transfer_ton):
        ax.annotate(f"{v:.3f}", (bar.get_x()+bar.get_width()/2, v), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8)
    for xi, thb in zip(x, combined_thb):
        ax.annotate(f"{thb:,.0f} บาท", (xi, max(idle_ton[xi], transfer_ton[xi])), xytext=(0, 22),
                    textcoords='offset points', ha='center', fontsize=9, fontweight='bold', color='#7a1f1f')
    ax.set_xticks(x); ax.set_xticklabels(mc.AUTOCLAVE_LABELS)
    ax.set_ylabel('ไอน้ำที่สูญเสีย (ตัน)')
    ax.set_title(f'สรุปมูลค่าไอน้ำที่สูญเสีย — Transfer + Idle Leakage, {date_label}\n'
                 f'ราคาไอน้ำ {STEAM_PRICE_THB_PER_TON:.0f} บาท/ตัน — total = {sum(combined_thb):,.0f} บาท',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    fig.text(0.5, -0.02,
              'มูลค่า (บาท) = (idle leakage total + transfer savable steam total) × 850 บาท/ตัน, ต่อเตา',
              ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    return fig, idle_ton, transfer_ton, combined_ton, combined_thb


def fig_post_transfer(result, date_label):
    ti1_mean, ti1_sd, ti2_mean, ti2_sd = [], [], [], []
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
    ax.margins(y=0.15)
    ax.axhline(4.0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(mc.AUTOCLAVE_LABELS)
    ax.set_ylabel('Mean pressure (bar)')
    ax.set_title(f'Post-Transfer Pressure vs Target, {date_label}  |  Target 4.0 bar', fontsize=12.5, fontweight='bold')
    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig, ti1_mean, ti2_mean


def fig_transfer_out_efficiency(result, date_label):
    donor_after_to1 = {ac1: [] for ac1 in range(1, sc.N_AUTOCLAVES + 1)}
    donor_after_to2 = {ac1: [] for ac1 in range(1, sc.N_AUTOCLAVES + 1)}
    for ac1 in range(1, sc.N_AUTOCLAVES + 1):
        ac0 = ac1 - 1
        p = result.pressures[ac0]
        stages = result.stages[ac0]
        to1_runs = sc.contiguous_runs(stages == 'transfer_out1')
        to2_runs = sc.contiguous_runs(stages == 'transfer_out2')
        donor_after_to1[ac1] = [p[e - 1] for (s, e) in to1_runs if e > s]
        donor_after_to2[ac1] = [p[e - 1] for (s, e) in to2_runs if e > s]
    to1_mean = [mc.mean_sd(donor_after_to1[a])[0] for a in range(1, sc.N_AUTOCLAVES + 1)]
    to1_sd = [mc.mean_sd(donor_after_to1[a])[1] for a in range(1, sc.N_AUTOCLAVES + 1)]
    to2_mean = [mc.mean_sd(donor_after_to2[a])[0] for a in range(1, sc.N_AUTOCLAVES + 1)]
    to2_sd = [mc.mean_sd(donor_after_to2[a])[1] for a in range(1, sc.N_AUTOCLAVES + 1)]
    x = np.arange(sc.N_AUTOCLAVES)
    w = 0.36
    fig, ax = plt.subplots(figsize=(12, 7))
    b1 = ax.bar(x - w/2, to1_mean, w, yerr=to1_sd, capsize=4, color='#e8a598', edgecolor='black', label='After Transfer Out 1 (ideal ~4 bar)')
    b2 = ax.bar(x + w/2, to2_mean, w, yerr=to2_sd, capsize=4, color='#7a1f1f', edgecolor='black', label='After Transfer Out 2 (ideal ~0 bar)')
    for bar, m, sdv in zip(b1, to1_mean, to1_sd):
        ax.annotate(f"{m:.2f}±{sdv:.2f}", (bar.get_x()+bar.get_width()/2, m+sdv), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
    for bar, m, sdv in zip(b2, to2_mean, to2_sd):
        ax.annotate(f"{m:.2f}±{sdv:.2f}", (bar.get_x()+bar.get_width()/2, m+sdv), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
    ax.margins(y=0.15)
    ax.set_xticks(x); ax.set_xticklabels(mc.AUTOCLAVE_LABELS)
    ax.set_ylabel('Donor pressure (bar)')
    ax.set_title(f'Transfer-Out Efficiency, {date_label}', fontsize=12.5, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig, to1_mean, to2_mean, donor_after_to2


def fig_savable_steam(result, donor_after_to2, date_label):
    total_ton, n_cycles = [], []
    for ac1 in range(1, sc.N_AUTOCLAVES + 1):
        offset = sc.SENSOR_OFFSETS.get(ac1, 0.0)
        floor_mass = sp.rho_sat_vapor(FLOOR_BAR) * sp.V_FREE[ac1]
        savable_kg = [max(0.0, sp.rho_sat_vapor(p + offset) * sp.V_FREE[ac1] - floor_mass) for p in donor_after_to2[ac1]]
        total_ton.append(sum(savable_kg) / KG_PER_TON)
        n_cycles.append(len(savable_kg))
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.bar(mc.AUTOCLAVE_LABELS, total_ton, color='#1f7a4c', edgecolor='black')
    for bar, v, n in zip(bars, total_ton, n_cycles):
        ax.annotate(f"{v:.3f} ton\nn={n}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 6),
                    textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.margins(y=0.2)
    ax.set_ylabel('savable steam, total [ton]')
    ax.set_title(f'Savable Steam (>{FLOOR_BAR:.0f} bar after TO2), {date_label}', fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    return fig, total_ton


# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🧪 Autoclave Health Check")
st.caption("Upload steam-log CSV file(s) — same format as the project's Q2_YYYYMMDD.csv exports — "
           "and get every health-check chart, computed live via stage_classifier.classify_all_autoclaves().")

uploaded = st.file_uploader("Steam CSV file(s)", type=["csv"], accept_multiple_files=True)

with st.expander("Product type / keeping time", expanded=True):
    st.caption("Keeping duration can't be detected from pressure alone (confirmed against ground-truth "
               "cycle logs — a 6h cycle's pressure trace looks identical to a 4h one). Set the default "
               "product's keeping time below, and optionally list specific cycles that ran a different "
               "product (matched to the nearest actual vacuum start).")
    default_keeping_h = st.number_input("Default keeping duration for all cycles (hours)",
                                         value=4.0, min_value=0.5, max_value=24.0, step=0.5)
    st.caption("Exceptions (optional) — cycles known to be a different product:")
    exceptions_df = st.data_editor(
        pd.DataFrame({'Autoclave (1-8)': pd.Series(dtype='int'),
                      'Day # (as uploaded, 1st file = 1)': pd.Series(dtype='int'),
                      'Approx. vacuum-start time (HH:MM)': pd.Series(dtype='str'),
                      'Keeping hours': pd.Series(dtype='float')}),
        num_rows='dynamic', use_container_width=True, key='exceptions_editor',
    )

if uploaded:
    dfs = load_uploaded_csvs(uploaded)

    # AC4's sensor correction isn't offered as an option — it was a real,
    # known-faulty gauge that was physically fixed after Jul 7, 2026; any
    # data collected going forward (which is what this app is for) doesn't
    # need it, so AC4 is treated the same as every other AC.
    sensor_offsets = {}
    vacuum_tolerance = {ac: 0.05 for ac in range(1, sc.N_AUTOCLAVES + 1)}

    date_label = f"{len(dfs)} file(s) uploaded"

    day_offsets = []
    offset = 0.0
    for df in dfs:
        day_offsets.append(offset)
        offset += df['time(h)'].values[-1]

    sc.KEEPING_NOM_DURATION = float(default_keeping_h)  # module-level default read by classify_trace()

    with st.spinner("Classifying stages..."):
        # Pass 1: baseline classification (default keeping duration only) —
        # needed to find each exception's actual cycle_idx (nearest vacuum
        # start to the time the user gave), the same key
        # stage_classifier.KEEPING_DURATION_OVERRIDES uses.
        baseline = sc.classify_all_autoclaves(
            dfs, sensor_offsets=sensor_offsets, vacuum_tolerance=vacuum_tolerance, verbose=False,
        )

        keeping_overrides = {}
        if exceptions_df is not None and len(exceptions_df):
            for row in exceptions_df.itertuples():
                ac1 = row._1
                day_no = row._2
                time_str = row._3
                hours = row._4
                if pd.isna(ac1) or pd.isna(day_no) or not time_str or pd.isna(hours):
                    continue
                try:
                    ac1 = int(ac1)
                    day_no = int(day_no)
                    hh, mm = (time_str.split(':') + ['0'])[:2]
                    target_t = day_offsets[day_no - 1] + float(hh) + float(mm) / 60.0
                except (ValueError, IndexError):
                    st.warning(f"Couldn't parse exception row: AC{ac1}, day {day_no}, time {time_str!r} — skipped.")
                    continue
                ac0 = ac1 - 1
                vac_runs = sc.contiguous_runs(baseline.stages[ac0] == 'vacuum')
                if not vac_runs:
                    continue
                times = baseline.times[ac0]
                cycle_idx = min(range(len(vac_runs)), key=lambda i: abs(times[vac_runs[i][0]] - target_t))
                keeping_overrides[(ac1, cycle_idx)] = float(hours)

        result = sc.classify_all_autoclaves(
            dfs, sensor_offsets=sensor_offsets, vacuum_tolerance=vacuum_tolerance,
            keeping_duration_overrides=keeping_overrides, verbose=False,
        )
        if keeping_overrides:
            st.caption(f"Applied {len(keeping_overrides)} keeping-time exception(s): "
                       + ", ".join(f"AC{a} cycle {c}={h}h" for (a, c), h in keeping_overrides.items()))

        temp_cols_present = all(f'Temperature Autocalve {i}' in dfs[0].columns for i in range(1, sc.N_AUTOCLAVES + 1))
        if temp_cols_present:
            st.session_state['temps'] = sc.concat_per_ac_column(dfs, 'Temperature Autocalve {i}')

    st.success(f"Classified {sum(mc.cycle_count(result.stages[a]) for a in range(sc.N_AUTOCLAVES))} total cycles across 8 autoclaves.")

    tabs = st.tabs(["Vacuum", "Transfer In/Out", "Fresh vs Transferred", "Idle Leakage",
                     "Savable Steam", "Condensate & Temp", "มูลค่าที่สูญเสีย (THB)", "Summary Table"])

    with tabs[0]:
        st.subheader("Vacuum Depth & Speed")
        f1, f2, depth_m, depth_sd, speed_m, speed_sd = fig_vacuum(result, -0.6, -0.55, 11.5, date_label)
        c1, c2 = st.columns(2)
        c1.pyplot(f1); c2.pyplot(f2)

    with tabs[1]:
        st.subheader("Post-Transfer Pressure (Receiver side)")
        f3, ti1_m, ti2_m = fig_post_transfer(result, date_label)
        st.pyplot(f3)
        st.subheader("Transfer-Out Efficiency (Donor side)")
        f4, to1_m, to2_m, donor_to2 = fig_transfer_out_efficiency(result, date_label)
        st.pyplot(f4)

    with tabs[2]:
        st.subheader("Fresh vs. Transferred Steam")
        f5, fresh_m, transferred_m, n_cycles = fig_transfer_pct(result, date_label)
        st.pyplot(f5)

    with tabs[3]:
        st.subheader("Idle Leakage Rate")
        f6, bar_h, kg_h = fig_idle_leakage(result, date_label)
        st.pyplot(f6)
        st.subheader("Total Idle Leakage")
        f7, totals = fig_idle_total(result, date_label)
        st.pyplot(f7)

    with tabs[4]:
        st.subheader("Savable Steam (leftover above 1 bar after Transfer Out 2)")
        f8, savable_ton = fig_savable_steam(result, donor_to2, date_label)
        st.pyplot(f8)

    with tabs[5]:
        if temp_cols_present:
            st.subheader("Condensate Cycle Count")
            f9, cond_total = fig_condensate(result, date_label)
            st.pyplot(f9)
            st.subheader("Temperature Spread During Hold")
            f10, temp_m, temp_sd = fig_temp_spread(result, date_label)
            st.pyplot(f10)
        else:
            st.info("No Temperature Autocalve columns found in the uploaded file(s) — condensate/temperature charts need those.")
            cond_total = [np.nan] * sc.N_AUTOCLAVES
            temp_m = [np.nan] * sc.N_AUTOCLAVES

    with tabs[6]:
        st.subheader("สรุปมูลค่าไอน้ำที่สูญเสีย (Transfer + Idle, THB)")
        f11, idle_ton, transfer_ton, combined_ton, combined_thb = fig_price_loss(totals, savable_ton, date_label)
        st.pyplot(f11)
        st.metric("มูลค่ารวมทั้งหมด (Total, all ACs)", f"{sum(combined_thb):,.0f} บาท")

    with tabs[7]:
        st.subheader("Summary — same layout as health check main.xlsx")
        summary = pd.DataFrame({
            'AC': mc.AUTOCLAVE_LABELS,
            'vacuum efficiency': [round(v, 2) for v in depth_m],
            'vacuum rate (min)': [round(v, 2) for v in speed_m],
            'transfer %': [round(v, 1) for v in transferred_m],
            'receiver gain TI1 (bar)': [round(v, 2) for v in ti1_m],
            'receiver gain TI2 (bar)': [round(v, 2) for v in ti2_m],
            'donor leftover TO1 (bar)': [round(v, 2) for v in to1_m],
            'donor leftover TO2 (bar)': [round(v, 2) for v in to2_m],
            'idle leakage (kg/h)': [round(kg_h[a][0], 2) for a in range(1, sc.N_AUTOCLAVES + 1)],
            'savable steam (ton)': [round(v, 3) for v in savable_ton],
            'condensate cycles': cond_total,
            'temp spread (°C)': [round(v, 2) if np.isfinite(v) else np.nan for v in temp_m],
            'price loss (THB)': [round(v, 0) for v in combined_thb],
        })
        st.dataframe(summary, use_container_width=True)

        buf = io.BytesIO()
        summary.to_excel(buf, index=False, sheet_name='Sheet1')
        st.download_button("Download summary as Excel", data=buf.getvalue(),
                            file_name="healthcheck_summary.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload one or more CSV files to run the health check.")
