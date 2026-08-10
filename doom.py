import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import stage_classifier as sc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMIZE HERE
# ══════════════════════════════════════════════════════════════════════════════
# Pick specific day(s) by position (1 = Jun 26, 2 = Jun 27, ... 12 = Jul 7) —
# doesn't need to be a contiguous range, e.g. [4] for just day 4 (Jun 29),
# or [1, 4, 7] for a few specific days, or list(range(1, 13)) for all 12.
DAYS_TO_SHOW = list(range(1,13))
AUTOCLAVES_TO_SHOW = [1, 2, 3, 4, 5, 6, 7, 8]   # which autoclaves (1-indexed) to PLOT
                                                  # (all 8 are still loaded/classified internally
                                                  #  so the rotation-matched transfer detection
                                                  #  stays correct — this only limits the chart)
# ══════════════════════════════════════════════════════════════════════════════

# ── Load data ──────────────────────────────────────────────────────────────────
JULY_FILES_ALL = [
    "Q2_20260626.csv",
    "Q2_20260627.csv", "Q2_20260628.csv", "Q2_20260629.csv", "Q2_20260630.csv",
    "Q2_20260701.csv", "Q2_20260702.csv", "Q2_20260703.csv", "Q2_20260704.csv",
    "Q2_20260705.csv", "Q2_20260706.csv", "Q2_20260707.csv",
]
DAY_LABELS_ALL = ["Jun 26", "Jun 27", "Jun 28", "Jun 29", "Jun 30", "Jul 1", "Jul 2", "Jul 3", "Jul 4", "Jul 5", "Jul 6", "Jul 7"]

# keep chronological order regardless of how DAYS_TO_SHOW was listed, and
# ignore any out-of-range values
day_positions = sorted(set(d for d in DAYS_TO_SHOW if 1 <= d <= len(JULY_FILES_ALL)))
if not day_positions:
    day_positions = list(range(1, len(JULY_FILES_ALL) + 1))
JULY_FILES = [JULY_FILES_ALL[d - 1] for d in day_positions]
DAY_LABELS = [DAY_LABELS_ALL[d - 1] for d in day_positions]

dfs = sc.load_csv_files(SCRIPT_DIR, JULY_FILES)

# ── Per-cycle classification overrides ────────────────────────────────────────
# Key: (autoclave_1indexed, cycle_0indexed)
# Value: 'single_transfer' → only TI1 then straight to fresh_steam (no interval/TI2)
#        'no_transfer'     → vacuum goes straight to fresh_steam (no TI1/interval/TI2)
CYCLE_OVERRIDES = {}

# Known keeping-end times (in hours from t=0) for autoclaves whose first
# rotation has missing/incomplete data — set by direct observation.
# Index here is autoclave number (1-indexed) -> hours; None = use default rule.
HEAD_KEEPING_END_OVERRIDES = {}

# ── Classify (single source of truth — see stage_classifier.py) ────────────────
result = sc.classify_all_autoclaves(
    dfs,
    head_keeping_end_overrides=HEAD_KEEPING_END_OVERRIDES,
    cycle_overrides=CYCLE_OVERRIDES,
)
combined_times = [result.times[ac] for ac in range(sc.N_AUTOCLAVES)]
combined_pressures = [result.pressures[ac] for ac in range(sc.N_AUTOCLAVES)]
combined_stages = [result.stages[ac] for ac in range(sc.N_AUTOCLAVES)]

# ── Plot with the split stages ──────────────────────────────────────────────────
autoclave_colors = ['blue', 'green', 'gold', 'red', 'purple', 'orange', 'cyan', 'magenta']
STAGE_COLORS = sc.STAGE_COLORS_FULL
STAGE_LABELS = sc.STAGE_LABELS_FULL
STAGE_ORDER_NEW = sc.STAGE_ORDER_FULL

# ── Only plot the selected autoclaves (all 8 were still loaded/classified
#    above, so rotation-matched transfer detection stays correct either way) ──
display_acs = sorted(set(a for a in AUTOCLAVES_TO_SHOW if 1 <= a <= sc.N_AUTOCLAVES))
if not display_acs:
    display_acs = list(range(1, sc.N_AUTOCLAVES + 1))
n_display = len(display_acs)
n_cols = 2 if n_display > 1 else 1
n_rows = int(np.ceil(n_display / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 5 * n_rows), sharex=True, sharey=True, squeeze=False)
axes = axes.flatten()

for plot_idx, ac1 in enumerate(display_acs):
    ac = ac1 - 1  # back to 0-indexed for the internal arrays
    ax = axes[plot_idx]
    t, p, s = combined_times[ac], combined_pressures[ac], combined_stages[ac]

    added_labels = set()
    i = 0
    while i < len(s):
        stage = s[i]
        j = i
        while j < len(s) and s[j] == stage:
            j += 1
        label = STAGE_LABELS[stage] if stage not in added_labels else None
        ax.axvspan(t[i], t[j - 1], color=STAGE_COLORS[stage], alpha=0.45, label=label)
        added_labels.add(stage)
        i = j

    ax.plot(t, p, marker='.', ms=2, linestyle='None', color=autoclave_colors[ac])
    ax.axhline(12.0, color='red', linestyle='-', linewidth=1.1)
    ax.set_title(f'Autoclave {ac1}', fontsize=11, fontweight='bold')
    ax.set_xlabel('time (h)', fontsize=9)
    ax.set_ylabel('pressure (bar)', fontsize=9)

# hide any unused trailing subplot cells (e.g. an odd number of ACs selected)
for extra_idx in range(n_display, len(axes)):
    axes[extra_idx].axis('off')

legend_handles = [Patch(facecolor=STAGE_COLORS[k], alpha=0.7, label=STAGE_LABELS[k])
                  for k in STAGE_ORDER_NEW]
legend_handles.append(Line2D([0], [0], color='red', linewidth=1.1, label='12 bar'))
axes[0].legend(handles=legend_handles, fontsize=6.5, loc='upper right')

date_range = DAY_LABELS[0] + (f' - {DAY_LABELS[-1]}' if len(DAY_LABELS) > 1 else '')
fig.suptitle(f'Pressure Trend {date_range}, 2026 - Stage Classification with Rotation-Matched Transfer Out\n'
             f'Showing {n_display} of {sc.N_AUTOCLAVES} autoclaves, {len(DAY_LABELS)} of {len(JULY_FILES_ALL)} days',
             fontsize=14, fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'rotation_stage_classification_july.png'), dpi=150, bbox_inches='tight')
print("Saved rotation_stage_classification_july.png")
plt.show()
