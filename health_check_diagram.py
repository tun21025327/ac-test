"""Health-check overview diagram — a stylized (not data-accurate) pressure
trace through one autoclave cycle, with the stage-colored bands used
everywhere else in this project, annotated with the health-check points an
operator would look at. Rebuilt from scratch (the original PNG's source
wasn't saved anywhere in the project) per instruction: general trend +
clear labels only, accuracy against real data doesn't matter here.

Checkpoints 2 (Transfer-In Check) and 3 (Transfer-Out Check) from the
original are merged into one combined "Transfer In/Out Check" box, so
there are 4 checkpoints total instead of 5.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import stage_classifier as sc

plt.rcParams['font.family'] = 'Tahoma'  # renders Thai glyphs (also covers Latin/English text)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'graph')

# ── Build a stylized, general-trend cycle curve (not real data) ────────────
rng = np.random.default_rng(7)

def seg(t0, t1, y0, y1, n, noise=0.0, ease='linear'):
    t = np.linspace(t0, t1, n)
    frac = (t - t0) / (t1 - t0)
    if ease == 'ease_out':
        frac = 1 - (1 - frac) ** 2
    elif ease == 'ease_in':
        frac = frac ** 2
    y = y0 + (y1 - y0) * frac
    if noise:
        y = y + rng.normal(0, noise, n)
    return t, y

segments = [
    ('vacuum',        *seg(0.0, 2.0, 0.3, -0.6, 40, ease='ease_in')),
    ('transfer_in1',  *seg(2.0, 3.0, -0.6, 1.0, 30, ease='ease_out')),
    ('interval',      *seg(3.0, 3.5, 1.0, 1.05, 12, noise=0.02)),
    ('transfer_in2',  *seg(3.5, 4.5, 1.05, 4.0, 30, ease='ease_out')),
    ('interval',      *seg(4.5, 5.0, 4.0, 4.05, 12, noise=0.03)),
    ('fresh_steam',   *seg(5.0, 7.0, 4.0, 12.0, 40, ease='ease_out')),
    ('keeping',       *seg(7.0, 13.0, 12.0, 11.85, 90, noise=0.05)),
    ('idle',          *seg(13.0, 14.0, 11.85, 11.5, 20, noise=0.06)),
    ('transfer_out1', *seg(14.0, 15.5, 11.5, 4.0, 35, ease='ease_in')),
    ('transfer_out2', *seg(15.5, 16.5, 4.0, 1.0, 25, ease='ease_in')),
    ('steam_off',     *seg(16.5, 18.0, 1.0, -0.1, 30, ease='ease_out')),
]

all_t = np.concatenate([s[1] for s in segments])
all_p = np.concatenate([s[2] for s in segments])

STAGE_COLORS = sc.STAGE_COLORS_FULL
STAGE_LABELS = sc.STAGE_LABELS_FULL

fig, ax = plt.subplots(figsize=(16, 10))

# stage-colored background bands
for name, t, p in segments:
    ax.axvspan(t[0], t[-1], color=STAGE_COLORS.get(name, '#dddddd'), alpha=0.45)

ax.plot(all_t, all_p, color='#222222', linewidth=2.4)
ax.axhline(0, color='#999999', linewidth=0.8, linestyle='--')

ax.set_xlim(all_t.min(), all_t.max())
ax.set_ylim(-2.2, 14.5)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# ── Health-check markers (target points on the curve) ──────────────────────
def point_at(t_target):
    idx = int(np.argmin(np.abs(all_t - t_target)))
    return all_t[idx], all_p[idx]

markers = {
    'evac':   point_at(1.9),    # vacuum minimum
    'ti':     point_at(4.3),    # after transfer_in2
    'to1':    point_at(15.4),   # after transfer_out1
    'to2':    point_at(16.4),   # after transfer_out2
    'slope1': point_at(7.3),
    'slope2': point_at(12.7),
    'cond':   point_at(10.0),
}
for key, (mt, mp) in markers.items():
    ax.add_patch(Circle((mt, mp), 0.22, facecolor='none', edgecolor='#b0201f', linewidth=2.2, zorder=5))

def callout(text, xy_list, box_xy, ha='left'):
    """One annotation box with a leader line to each point in xy_list."""
    for xy in xy_list:
        ax.annotate('', xy=xy, xytext=box_xy,
                    arrowprops=dict(arrowstyle='-', color='#b0201f', linewidth=1.1),
                    zorder=4)
    ax.text(box_xy[0], box_xy[1], text, ha=ha, va='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#b0201f', linewidth=1.6),
             zorder=6, linespacing=1.6)

callout(
    "จุดที่ 1: ตรวจสอบสุญญากาศ (Evacuation Check)\n"
    "ความลึกสุญญากาศสุดท้าย และความเร็วในการลดความดัน\n"
    "(เทียบกับ -0.65 บาร์ในอุดมคติ; เวลาที่ถึง -0.55 บาร์)",
    [markers['evac']], (1.9, -1.9), ha='center')

callout(
    "จุดที่ 2: ตรวจสอบการรับ-ถ่ายไอน้ำ (Transfer In/Out Check)\n"
    "รวมจุดตรวจสอบการรับไอน้ำและการถ่ายไอน้ำออกเดิมเข้าด้วยกัน\n"
    "รับ: ความดันหลังรับไอน้ำรอบ 1 และ 2 (เป้าหมาย ~4.0 บาร์)\n"
    "ถ่าย: ความดันตกค้างหลังถ่ายออกรอบ 1 และ 2\n"
    "      (เป้าหมาย ~4 บาร์หลัง TO1, ~1 บาร์หลัง TO2 = ถ่ายหมด)",
    [markers['ti'], markers['to1'], markers['to2']], (9.0, -1.9), ha='center')

callout(
    "จุดที่ 3: ตรวจสอบการรั่วไหลและความชัน (Leakage & Slope Check)\n"
    "ความชันของความดันขณะวาล์วปิดสนิท (บาร์/ชม.)\n"
    "เทียบกับอัตราการสูญเสียความร้อนปกติ -> ไอน้ำรั่วเข้า/ออกผิดปกติ",
    [markers['slope1'], markers['slope2']], (7.0, 14.0), ha='center')

callout(
    "จุดที่ 4: ตรวจสอบน้ำทิ้ง (Condensate Check)\n"
    "จำนวนรอบเปิด/ปิดวาล์วน้ำทิ้ง และ dT\n"
    "ตลอดช่วงความดันสูง\n"
    "(วัดจากความดันอย่างเดียวไม่ได้ — ต้องใช้ข้อมูลน้ำทิ้งเพิ่ม)",
    [markers['cond']], (14.5, 14.0), ha='center')

ax.set_title('พิกัดการตรวจจับความผิดปกติของหม้ออบไอน้ำ (Health Check Points Overview)\n'
             'สร้างใหม่ — แสดงแนวโน้มทั่วไปเท่านั้น ไม่ใช่ข้อมูลจริง (rebuilt, stylized, not real logged data)',
             fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'health_check_diagram_edited.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved health_check_diagram_edited.png")
