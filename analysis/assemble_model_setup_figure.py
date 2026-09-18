#!/usr/bin/env python3
"""Assembles high-resolution publication Figure: Model setup, BCs, and load cases."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

TMP_DIR = Path('/tmp/model_setup_renders')
OUT_PDF = Path('analysis_outputs/plos_revision/figures/model_setup.pdf')
OUT_PNG = Path('analysis_outputs/plos_revision/figures/model_setup.png')

# Load cropped images to remove excess whitespace
def load_and_crop(path, pad=15):
    im = Image.open(path)
    # convert to RGBA
    im = im.convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    diff = np.array(im)
    # find non-white pixels
    mask = (diff[:, :, 0] < 250) | (diff[:, :, 1] < 250) | (diff[:, :, 2] < 250)
    if not np.any(mask):
        return np.array(im)
    y_idx, x_idx = np.where(mask)
    x_min = max(0, x_idx.min() - pad)
    x_max = min(im.width, x_idx.max() + pad)
    y_min = max(0, y_idx.min() - pad)
    y_max = min(im.height, y_idx.max() + pad)
    return np.array(im.crop((x_min, y_min, x_max, y_max)))

img_A = load_and_crop(TMP_DIR / 'panel_A.png')
img_B1 = load_and_crop(TMP_DIR / 'panel_B1.png')
img_B2 = load_and_crop(TMP_DIR / 'panel_B2.png')
img_C1 = load_and_crop(TMP_DIR / 'panel_C1.png')
img_C2 = load_and_crop(TMP_DIR / 'panel_C2.png')
img_D1 = load_and_crop(TMP_DIR / 'panel_D1.png')
img_D2 = load_and_crop(TMP_DIR / 'panel_D2.png')
img_D3 = load_and_crop(TMP_DIR / 'panel_D3.png')

fig = plt.figure(figsize=(10.5, 10.0), dpi=300)

# Layout: 2 primary rows
# Top: A (50%) + B1, B2 (50%)
# Bottom: C1, C2 (standing, 40%) + D1, D2, D3 (labor, 60%)

gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.15, 1.0], hspace=0.22)

# Top section
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0], width_ratios=[1.35, 1.0, 1.0], wspace=0.15)
ax_A = fig.add_subplot(gs_top[0])
ax_B1 = fig.add_subplot(gs_top[1])
ax_B2 = fig.add_subplot(gs_top[2])

# Bottom section: Standing (2 subplots) + Labor (3 subplots)
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_main[1], width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0], wspace=0.12)
ax_C1 = fig.add_subplot(gs_bot[0])
ax_C2 = fig.add_subplot(gs_bot[1])
ax_D1 = fig.add_subplot(gs_bot[2])
ax_D2 = fig.add_subplot(gs_bot[3])
ax_D3 = fig.add_subplot(gs_bot[4])

# --- Panel A ---
ax_A.imshow(img_A)
ax_A.axis('off')
ax_A.set_title(r'$\mathbf{A}$  Pelvic FE assembly & ligaments', loc='left', fontsize=11, pad=6, fontweight='bold')

# Annotations on Panel A
ax_A.annotate(r'Fixed boundary: $S_1$ facet ($\mathbf{u}=\mathbf{0}$)',
             xy=(0.50, 0.77), xycoords='axes fraction',
             xytext=(0.50, 0.94), textcoords='axes fraction',
             ha='center', fontsize=8.5, fontweight='bold', color='#1a9641',
             arrowprops=dict(arrowstyle='->', color='#1a9641', lw=1.5))

ax_A.annotate('Pubic symphysis\n(fibrocartilage)',
             xy=(0.50, 0.08), xycoords='axes fraction',
             xytext=(0.50, -0.05), textcoords='axes fraction',
             ha='center', fontsize=8, color='#7b3294', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#7b3294', lw=1.2))

ax_A.annotate('Sacroiliac joint\n(SIJ cartilage)',
             xy=(0.72, 0.65), xycoords='axes fraction',
             xytext=(0.85, 0.75), textcoords='axes fraction',
             ha='center', fontsize=8, color='#2b83ba', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#2b83ba', lw=1.2))

# --- Panel B1 ---
ax_B1.imshow(img_B1)
ax_B1.axis('off')
ax_B1.set_title(r'$\mathbf{B}_1$  Inlet plane diameters', loc='left', fontsize=10, pad=6, fontweight='bold')
ax_B1.text(0.5, 0.02, 'AP: Conjugata vera (123 mm)\nML: Transverse (131 mm)',
           transform=ax_B1.transAxes, ha='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#cccccc', lw=0.8))

# --- Panel B2 ---
ax_B2.imshow(img_B2)
ax_B2.axis('off')
ax_B2.set_title(r'$\mathbf{B}_2$  Outlet plane diameters', loc='left', fontsize=10, pad=6, fontweight='bold')
ax_B2.text(0.5, 0.02, 'BIS: Biischiadic (103 mm)\nBIT: Bituberous (132 mm)',
           transform=ax_B2.transAxes, ha='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#cccccc', lw=0.8))

# --- Panel C1 ---
ax_C1.imshow(img_C1)
ax_C1.axis('off')
ax_C1.set_title(r'$\mathbf{C}_1$  SP2leg stance', loc='left', fontsize=9.5, pad=5, fontweight='bold')
ax_C1.text(0.5, 0.02, 'Bilateral standing\n2 × 400 N vertical', transform=ax_C1.transAxes,
           ha='center', fontsize=7.5, color='#d95f02', fontweight='bold')

# --- Panel C2 ---
ax_C2.imshow(img_C2)
ax_C2.axis('off')
ax_C2.set_title(r'$\mathbf{C}_2$  SP1leg stance', loc='left', fontsize=9.5, pad=5, fontweight='bold')
ax_C2.text(0.5, 0.02, 'Unilateral standing\n1 × 800 N vertical', transform=ax_C2.transAxes,
           ha='center', fontsize=7.5, color='#d95f02', fontweight='bold')

# --- Panel D1 ---
ax_D1.imshow(img_D1)
ax_D1.axis('off')
ax_D1.set_title(r'$\mathbf{D}_1$  $\mathrm{LAB}_1$ (Inlet)', loc='left', fontsize=9.5, pad=5, fontweight='bold')
ax_D1.text(0.5, 0.02, 'Inlet expansion\n±400 N lateral', transform=ax_D1.transAxes,
           ha='center', fontsize=7.5, color='#e7298a', fontweight='bold')
# Draw 2D lateral expansion arrows on D1
ax_D1.annotate('', xy=(0.88, 0.52), xytext=(0.78, 0.52), xycoords='axes fraction',
               arrowprops=dict(arrowstyle='->', color='#e7298a', lw=2))
ax_D1.annotate('', xy=(0.12, 0.52), xytext=(0.22, 0.52), xycoords='axes fraction',
               arrowprops=dict(arrowstyle='->', color='#e7298a', lw=2))

# --- Panel D2 ---
ax_D2.imshow(img_D2)
ax_D2.axis('off')
ax_D2.set_title(r'$\mathbf{D}_2$  $\mathrm{LAB}_2$ (Midpelvis)', loc='left', fontsize=9.5, pad=5, fontweight='bold')
ax_D2.text(0.5, 0.02, 'Tuberosity distraction\n±400 N lateral', transform=ax_D2.transAxes,
           ha='center', fontsize=7.5, color='#7570b3', fontweight='bold')

# --- Panel D3 ---
ax_D3.imshow(img_D3)
ax_D3.axis('off')
ax_D3.set_title(r'$\mathbf{D}_3$  $\mathrm{LAB}_3$ (Outlet)', loc='left', fontsize=9.5, pad=5, fontweight='bold')
ax_D3.text(0.5, 0.02, 'AP outlet distraction\n±400 N anteroposterior', transform=ax_D3.transAxes,
           ha='center', fontsize=7.5, color='#e6ab02', fontweight='bold')

# Add legend for ligaments at bottom of panel A
leg_elements = [
    plt.Line2D([0], [0], color='#d7191c', lw=2.5, label='Anterior SIJ lig.'),
    plt.Line2D([0], [0], color='#fdae6b', lw=2.5, label='Posterior SIJ lig.'),
    plt.Line2D([0], [0], color='#2b83ba', lw=2.5, label='Interosseous SIJ lig.'),
    plt.Line2D([0], [0], color='#e66101', lw=2.5, label='Sacrospinous lig.'),
    plt.Line2D([0], [0], color='#5e3c99', lw=2.5, label='Sacrotuberous lig.'),
    plt.Line2D([0], [0], color='#c51b7d', lw=2.5, label='Pubic ligaments'),
]
fig.legend(handles=leg_elements, loc='lower left', bbox_to_anchor=(0.04, 0.495),
           ncol=3, fontsize=7.5, frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
fig.savefig(OUT_PNG, bbox_inches='tight', dpi=300)
plt.close(fig)
print('Figure saved to:', OUT_PDF, 'and', OUT_PNG)
