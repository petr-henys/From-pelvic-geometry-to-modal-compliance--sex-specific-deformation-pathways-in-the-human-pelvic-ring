#!/usr/bin/env python3
"""Generates publication-quality Figure 8: FE model setup, ligaments, boundary conditions, and load cases."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

OUT_PDF = Path('analysis_outputs/plos_revision/figures/model_setup.pdf')
OUT_PNG = Path('analysis_outputs/plos_revision/figures/model_setup.png')
TMP_DIR = Path('/tmp/model_setup_hq')
TMP_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load FE geometry
mesh = pv.read('results/ref_S1P_fixed_new2/paraview/mesh_complete.vtk')
ligs = pv.read('results/ref_S1P_fixed_new2/ligaments/ligaments.vtm')

# Extract full surface of entire FE model
surf = mesh.extract_surface()

# Extract intact pelvic bone surface (RegionName == 'PelvisBone') without carving out patches
bone_surf = surf.extract_cells(surf.cell_data['RegionName'] == 'PelvisBone').extract_surface()

# Articular cartilages
pub_sym = surf.extract_cells(surf.cell_data['RegionName'] == 'PubicSymphysis').extract_surface().compute_normals(point_normals=True)
sij_l = surf.extract_cells(surf.cell_data['RegionName'] == 'SIJCartilageLeft').extract_surface().compute_normals(point_normals=True)
sij_r = surf.extract_cells(surf.cell_data['RegionName'] == 'SIJCartilageRight').extract_surface().compute_normals(point_normals=True)

# Separate bone bodies cleanly for Panel A shading
bodies = bone_surf.split_bodies()
r_hemi = bodies[0].extract_surface().compute_normals(point_normals=True, feature_angle=60.0)
l_hemi = bodies[1].extract_surface().compute_normals(point_normals=True, feature_angle=60.0)

# Sacrum with seamless integrated S1 facet (no z-fighting, no internal edges)
sacrum = bodies[2].extract_surface()
sac_cols = np.full((sacrum.n_cells, 3), [232, 226, 216], dtype=np.uint8) # ivory bone #e8e2d8
sac_cols[sacrum.cell_data['DisplayName'] == 'S1_facet'] = [26, 150, 65]   # fixed S1 green #1a9641
sacrum.cell_data['RGB'] = sac_cols
sacrum = sacrum.compute_normals(point_normals=True, feature_angle=60.0)

# Helper function for smooth intact bone with arbitrary surface patches colored seamlessly
def get_colored_bone(highlight_dict=None):
    cols = np.full((bone_surf.n_cells, 3), [236, 231, 223], dtype=np.uint8) # bone ivory #ece7df
    if highlight_dict:
        for name, rgb in highlight_dict.items():
            mask = bone_surf.cell_data['DisplayName'] == name
            cols[mask] = rgb
    out = bone_surf.copy()
    out.cell_data['RGB'] = cols
    return out.compute_normals(point_normals=True, feature_angle=60.0)

bone_smooth = get_colored_bone()

# Colors
c_sij = '#2b83ba'       # SIJ cartilage blue
c_pub = '#7b3294'       # Pubic symphysis purple

lig_colors = {
    'symphisys': '#c51b7d',
    'right_anterior_SIJ_ligaments': '#d7191c',
    'left_anterior_SIJ_ligaments': '#d7191c',
    'right_posterior_SIJ_ligaments': '#fdae61',
    'left_posterior_SIJ_ligaments': '#fdae61',
    'right_INL': '#2b83ba',
    'left_INL': '#2b83ba',
    'right_SS': '#e66101',
    'left_SS': '#e66101',
    'right_ST': '#5e3c99',
    'left_ST': '#5e3c99',
}

def render_scene(path, actors, cam_pos, focal_point, view_up, size=(1200, 1200)):
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background('white')
    for mesh_obj, kwargs in actors:
        p.add_mesh(mesh_obj, smooth_shading=True, **kwargs)
    p.camera_position = [cam_pos, focal_point, view_up]
    p.screenshot(str(path))
    p.close()

center = bone_surf.center

# ==========================================
# 1. RENDER PANEL A: Full model + ligaments
# ==========================================
actors_A = [
    (sacrum, {'scalars': 'RGB', 'rgb': True, 'specular': 0.15}),
    (r_hemi, {'color': '#ece7df', 'specular': 0.15}),
    (l_hemi, {'color': '#ece7df', 'specular': 0.15}),
    (pub_sym, {'color': c_pub, 'specular': 0.2}),
    (sij_l, {'color': c_sij, 'specular': 0.2}),
    (sij_r, {'color': c_sij, 'specular': 0.2}),
]
for i in range(len(ligs)):
    blk = ligs[i]
    if blk.n_cells > 0:
        col = lig_colors.get(ligs.get_block_name(i), '#636363')
        actors_A.append((blk.tube(radius=1.35), {'color': col}))

cam_A = [center[0] + 15, center[1] - 390, center[2] + 240]
render_scene(TMP_DIR / 'hq_panel_A.png', actors_A, cam_A, center, (0, 0.4, 0.9), size=(1600, 1400))

# ==========================================
# 2. RENDER PANEL B1: Inlet diameters
# ==========================================
from analysis.spectral_data import load_inlet_landmarks, load_outlet_landmarks
inlet = load_inlet_landmarks()
ap = inlet['AP']
ml = inlet['ML']
inlet_center = (ap.mean(axis=0) + ml.mean(axis=0)) / 2.0
ap_vec = ap[1] - ap[0]; ml_vec = ml[1] - ml[0]
inlet_norm = np.cross(ml_vec, ap_vec); inlet_norm /= np.linalg.norm(inlet_norm)

actors_B1 = [
    (bone_smooth, {'scalars': 'RGB', 'rgb': True, 'specular': 0.15}),
    (pub_sym, {'color': c_pub}),
    (sij_l, {'color': c_sij}),
    (sij_r, {'color': c_sij}),
    (pv.lines_from_points(ap).tube(radius=2.2), {'color': '#d7191c'}),
    (pv.lines_from_points(ml).tube(radius=2.2), {'color': '#2b83ba'}),
    (pv.PolyData(ap), {'color': '#d7191c', 'point_size': 18, 'render_points_as_spheres': True}),
    (pv.PolyData(ml), {'color': '#2b83ba', 'point_size': 18, 'render_points_as_spheres': True}),
]
render_scene(TMP_DIR / 'hq_panel_B1.png', actors_B1, inlet_center - 380 * inlet_norm, inlet_center, (0, 0, 1), size=(1200, 1200))

# ==========================================
# 3. RENDER PANEL B2: Outlet diameters
# ==========================================
outlet = load_outlet_landmarks()
bis = outlet['BIS']
bit = outlet['BIT']
oap = outlet['OUTLETAP']
outlet_center = (bis.mean(axis=0) + bit.mean(axis=0)) / 2.0
bis_vec = bis[1] - bis[0]; oap_vec = oap[1] - oap[0]
outlet_norm = np.cross(bis_vec, oap_vec); outlet_norm /= np.linalg.norm(outlet_norm)

actors_B2 = [
    (bone_smooth, {'scalars': 'RGB', 'rgb': True, 'specular': 0.15}),
    (pub_sym, {'color': c_pub}),
    (pv.lines_from_points(bis).tube(radius=2.2), {'color': '#7b3294'}),
    (pv.lines_from_points(bit).tube(radius=2.2), {'color': '#1a9641'}),
    (pv.PolyData(bis), {'color': '#7b3294', 'point_size': 18, 'render_points_as_spheres': True}),
    (pv.PolyData(bit), {'color': '#1a9641', 'point_size': 18, 'render_points_as_spheres': True}),
]
# Inferior/perineal view with sacrum/posterior at top, matching B1 anatomical orientation
render_scene(TMP_DIR / 'hq_panel_B2.png', actors_B2, outlet_center + 380 * outlet_norm, outlet_center, (0, 1, 0), size=(1200, 1200))

# ==========================================
# 4. RENDER C1: SP2leg (Bilateral stance)
# ==========================================
c_al = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'left_AC_notch').center)
c_ar = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'right_AC_notch').center)
arr_al = pv.Arrow(start=c_al - [0, 0, 48], direction=[0, 0, 1], scale=48, tip_length=0.30, tip_radius=0.13, shaft_radius=0.06)
arr_ar = pv.Arrow(start=c_ar - [0, 0, 48], direction=[0, 0, 1], scale=48, tip_length=0.30, tip_radius=0.13, shaft_radius=0.06)

bone_c1 = get_colored_bone({
    'S1_facet': [26, 150, 65],
    'left_AC_notch': [217, 95, 2],
    'right_AC_notch': [217, 95, 2],
})

actors_C1 = [
    (bone_c1, {'scalars': 'RGB', 'rgb': True, 'specular': 0.15}),
    (arr_al, {'color': '#d95f02'}),
    (arr_ar, {'color': '#d95f02'}),
]
cam_C = [center[0], center[1] - 420, center[2] + 40]
render_scene(TMP_DIR / 'hq_panel_C1.png', actors_C1, cam_C, center, (0, 0, 1), size=(1000, 1000))

# ==========================================
# 5. RENDER C2: SP1leg (Unilateral stance)
# ==========================================
arr_ar_single = pv.Arrow(start=c_ar - [0, 0, 58], direction=[0, 0, 1], scale=58, tip_length=0.28, tip_radius=0.14, shaft_radius=0.07)

bone_c2 = get_colored_bone({
    'S1_facet': [26, 150, 65],
    'right_AC_notch': [217, 95, 2],
})

actors_C2 = [
    (bone_c2, {'scalars': 'RGB', 'rgb': True, 'specular': 0.15}),
    (arr_ar_single, {'color': '#d95f02'}),
]
render_scene(TMP_DIR / 'hq_panel_C2.png', actors_C2, cam_C, center, (0, 0, 1), size=(1000, 1000))

# ==========================================
# 6. RENDER D1: LAB1 (Inlet ring expansion)
# ==========================================
c_rl = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'ring_contact_left').center)
c_rr = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'ring_contact_right').center)
mid_ring = (c_rl + c_rr) / 2.0
dist_rl = 48.0; dist_rr = 48.0
arr_rl = pv.Arrow(start=mid_ring + [10, 0, 0], direction=[1, 0, 0], scale=dist_rl, tip_length=0.28, tip_radius=0.12, shaft_radius=0.06)
arr_rr = pv.Arrow(start=mid_ring - [10, 0, 0], direction=[-1, 0, 0], scale=dist_rr, tip_length=0.28, tip_radius=0.12, shaft_radius=0.06)

bone_d1 = get_colored_bone({
    'S1_facet': [26, 150, 65],
    'ring_contact_left': [231, 41, 138],
    'ring_contact_right': [231, 41, 138],
})

actors_D1 = [
    (bone_d1, {'scalars': 'RGB', 'rgb': True, 'specular': 0.15}),
    (arr_rl, {'color': '#e7298a'}),
    (arr_rr, {'color': '#e7298a'}),
]
render_scene(TMP_DIR / 'hq_panel_D1.png', actors_D1, inlet_center - 380 * inlet_norm, inlet_center, (0, 0, 1), size=(1000, 1000))

# ==========================================
# 7. RENDER D2: LAB2 (Ischial distraction)
# ==========================================
c_il = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'left_ischium_tuber').center)
c_ir = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'right_ischium_tuber').center)
mid_tuber = (c_il + c_ir) / 2.0
dist_il = 46.0; dist_ir = 46.0
arr_il = pv.Arrow(start=mid_tuber + [8, 0, 0], direction=[1, 0, 0], scale=dist_il, tip_length=0.30, tip_radius=0.13, shaft_radius=0.06)
arr_ir = pv.Arrow(start=mid_tuber - [8, 0, 0], direction=[-1, 0, 0], scale=dist_ir, tip_length=0.30, tip_radius=0.13, shaft_radius=0.06)

bone_d2 = get_colored_bone({
    'left_ischium_tuber': [117, 112, 179],
    'right_ischium_tuber': [117, 112, 179],
})

actors_D2 = [
    (bone_d2, {'scalars': 'RGB', 'rgb': True, 'specular': 0.15}),
    (arr_il, {'color': '#7570b3'}),
    (arr_ir, {'color': '#7570b3'}),
]
render_scene(TMP_DIR / 'hq_panel_D2.png', actors_D2, mid_tuber + 380 * outlet_norm, mid_tuber, (0, 1, 0), size=(1000, 1000))

# ==========================================
# 8. RENDER D3: LAB3 (AP outlet distraction)
# ==========================================
c_pub = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'pubis_ins').center)
c_scj = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'SCJ').center)
arr_pub = pv.Arrow(start=c_pub, direction=[0, -1, 0], scale=52, tip_length=0.30, tip_radius=0.13, shaft_radius=0.06)
arr_scj = pv.Arrow(start=c_scj, direction=[0, 1, 0], scale=52, tip_length=0.30, tip_radius=0.13, shaft_radius=0.06)

bone_d3 = get_colored_bone({
    'S1_facet': [26, 150, 65],
    'pubis_ins': [217, 119, 6],
    'SCJ': [217, 119, 6],
})

actors_D3 = [
    (bone_d3, {'scalars': 'RGB', 'rgb': True, 'specular': 0.15}),
    (arr_pub, {'color': '#d97706'}),
    (arr_scj, {'color': '#d97706'}),
]
render_scene(TMP_DIR / 'hq_panel_D3.png', actors_D3, [center[0] - 420, center[1] - 270, center[2] + 90], center, (0, 0, 1), size=(1000, 1000))

print('All 8 high-res renders completed.')

# ==========================================
# 9. ASSEMBLE COMPOSITE MATPLOTLIB FIGURE
# ==========================================
def load_and_crop(path, pad=18):
    im = Image.open(path).convert('RGBA')
    diff = np.array(im)
    mask = (diff[:, :, 0] < 250) | (diff[:, :, 1] < 250) | (diff[:, :, 2] < 250)
    if not np.any(mask):
        return np.array(im)
    y_idx, x_idx = np.where(mask)
    x_min = max(0, x_idx.min() - pad)
    x_max = min(im.width, x_idx.max() + pad)
    y_min = max(0, y_idx.min() - pad)
    y_max = min(im.height, y_idx.max() + pad)
    return np.array(im.crop((x_min, y_min, x_max, y_max)))

img_A = load_and_crop(TMP_DIR / 'hq_panel_A.png')
img_B1 = load_and_crop(TMP_DIR / 'hq_panel_B1.png')
img_B2 = load_and_crop(TMP_DIR / 'hq_panel_B2.png')
img_C1 = load_and_crop(TMP_DIR / 'hq_panel_C1.png')
img_C2 = load_and_crop(TMP_DIR / 'hq_panel_C2.png')
img_D1 = load_and_crop(TMP_DIR / 'hq_panel_D1.png')
img_D2 = load_and_crop(TMP_DIR / 'hq_panel_D2.png')
img_D3 = load_and_crop(TMP_DIR / 'hq_panel_D3.png')

fig = plt.figure(figsize=(10.5, 8.8), dpi=300)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

gs = gridspec.GridSpec(2, 1, height_ratios=[1.18, 0.92], hspace=0.22)

# Top section
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], width_ratios=[1.35, 1.0, 1.0], wspace=0.15)
ax_A = fig.add_subplot(gs_top[0])
ax_B1 = fig.add_subplot(gs_top[1])
ax_B2 = fig.add_subplot(gs_top[2])

# Bottom section: 5 subpanels
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1], width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0], wspace=0.12)
ax_C1 = fig.add_subplot(gs_bot[0])
ax_C2 = fig.add_subplot(gs_bot[1])
ax_D1 = fig.add_subplot(gs_bot[2])
ax_D2 = fig.add_subplot(gs_bot[3])
ax_D3 = fig.add_subplot(gs_bot[4])

# --- Panel A ---
ax_A.imshow(img_A)
ax_A.axis('off')
ax_A.set_title(r'$\mathbf{A}$  Pelvic FE assembly & ligaments', loc='left', fontsize=10.5, pad=5, fontweight='bold')

# Leader lines & annotations for Panel A
ax_A.annotate(r'Fixed boundary: $S_1$ facet ($\mathbf{u}=\mathbf{0}$)',
             xy=(0.50, 0.77), xycoords='axes fraction',
             xytext=(0.50, 0.96), textcoords='axes fraction',
             ha='center', fontsize=8.5, fontweight='bold', color='#1a9641',
             arrowprops=dict(arrowstyle='->', color='#1a9641', lw=1.5))

ax_A.annotate('Sacroiliac joint\n(articular cartilage)',
             xy=(0.72, 0.65), xycoords='axes fraction',
             xytext=(0.84, 0.76), textcoords='axes fraction',
             ha='center', fontsize=7.5, color='#2b83ba', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#2b83ba', lw=1.2))

ax_A.annotate('Pubic symphysis\n(fibrocartilage disc)',
             xy=(0.50, 0.08), xycoords='axes fraction',
             xytext=(0.50, 0.20), textcoords='axes fraction',
             ha='center', fontsize=7.5, color='#7b3294', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#7b3294', lw=1.2))

# Add legend for ligaments below panel A
leg_elements = [
    plt.Line2D([0], [0], color='#d7191c', lw=2.2, label='Anterior SIJ lig.'),
    plt.Line2D([0], [0], color='#fdae6b', lw=2.2, label='Posterior SIJ lig.'),
    plt.Line2D([0], [0], color='#2b83ba', lw=2.2, label='Interosseous SIJ lig.'),
    plt.Line2D([0], [0], color='#e66101', lw=2.2, label='Sacrospinous lig.'),
    plt.Line2D([0], [0], color='#5e3c99', lw=2.2, label='Sacrotuberous lig.'),
    plt.Line2D([0], [0], color='#c51b7d', lw=2.2, label='Pubic ligaments'),
]
ax_A.legend(handles=leg_elements, loc='upper center', bbox_to_anchor=(0.50, -0.05),
            ncol=3, fontsize=7.0, frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

# --- Panel B1 ---
ax_B1.imshow(img_B1)
ax_B1.axis('off')
ax_B1.set_title(r'$\mathbf{B}_1$  Inlet plane diameters', loc='left', fontsize=10, pad=5, fontweight='bold')
ax_B1.text(0.5, -0.08, 'AP: Conjugata vera (123 mm)\nML: Transverse diameter (131 mm)',
           transform=ax_B1.transAxes, ha='center', fontsize=7.5,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#cccccc', lw=0.7))

# --- Panel B2 ---
ax_B2.imshow(img_B2)
ax_B2.axis('off')
ax_B2.set_title(r'$\mathbf{B}_2$  Outlet plane diameters', loc='left', fontsize=10, pad=5, fontweight='bold')
ax_B2.text(0.5, -0.08, 'BIS: Biischiadic diameter (103 mm)\nBIT: Bituberous diameter (132 mm)',
           transform=ax_B2.transAxes, ha='center', fontsize=7.5,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#cccccc', lw=0.7))

# --- Panel C1 ---
ax_C1.imshow(img_C1)
ax_C1.axis('off')
ax_C1.set_title(r'$\mathbf{C}_1$  SP2leg stance', loc='left', fontsize=9.5, pad=4, fontweight='bold')
ax_C1.text(0.5, -0.10, 'Bilateral standing\n2 × 400 N vertical', transform=ax_C1.transAxes,
           ha='center', fontsize=7.2, color='#d95f02', fontweight='bold')

# --- Panel C2 ---
ax_C2.imshow(img_C2)
ax_C2.axis('off')
ax_C2.set_title(r'$\mathbf{C}_2$  SP1leg stance', loc='left', fontsize=9.5, pad=4, fontweight='bold')
ax_C2.text(0.5, -0.10, 'Unilateral standing\n1 × 800 N vertical', transform=ax_C2.transAxes,
           ha='center', fontsize=7.2, color='#d95f02', fontweight='bold')

# --- Panel D1 ---
ax_D1.imshow(img_D1)
ax_D1.axis('off')
ax_D1.set_title(r'$\mathbf{D}_1$  $\mathrm{LAB}_1$ (Inlet)', loc='left', fontsize=9.5, pad=4, fontweight='bold')
ax_D1.text(0.5, -0.10, 'Inlet expansion\n±400 N lateral', transform=ax_D1.transAxes,
           ha='center', fontsize=7.2, color='#e7298a', fontweight='bold')

# --- Panel D2 ---
ax_D2.imshow(img_D2)
ax_D2.axis('off')
ax_D2.set_title(r'$\mathbf{D}_2$  $\mathrm{LAB}_2$ (Midpelvis)', loc='left', fontsize=9.5, pad=4, fontweight='bold')
ax_D2.text(0.5, -0.10, 'Tuberosity distraction\n±400 N lateral', transform=ax_D2.transAxes,
           ha='center', fontsize=7.2, color='#7570b3', fontweight='bold')

# --- Panel D3 ---
ax_D3.imshow(img_D3)
ax_D3.axis('off')
ax_D3.set_title(r'$\mathbf{D}_3$  $\mathrm{LAB}_3$ (Outlet)', loc='left', fontsize=9.5, pad=4, fontweight='bold')
ax_D3.text(0.5, -0.10, 'AP outlet distraction\n±400 N anteroposterior', transform=ax_D3.transAxes,
           ha='center', fontsize=7.2, color='#b45309', fontweight='bold')

# Super-headers for bottom row to make groupings obvious
fig.text(0.275, 0.44, 'Habitual Locomotor Loads (Standing)', fontsize=9.5, fontweight='bold', ha='center', color='#222222')
fig.text(0.671, 0.44, 'Parturition-Motivated Proxy Loads (Labor Stages)', fontsize=9.5, fontweight='bold', ha='center', color='#222222')

# Draw subtle divider line between standing and labor
divider = plt.Line2D([0.433, 0.433], [0.05, 0.45], color='#cccccc', lw=1.0, linestyle='--', transform=fig.transFigure)
fig.lines.append(divider)

fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
fig.savefig(OUT_PNG, bbox_inches='tight', dpi=300)
plt.close(fig)
print('Composite figure generated successfully at:', OUT_PDF)
