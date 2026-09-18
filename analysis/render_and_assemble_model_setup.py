#!/usr/bin/env python3
"""Generates publication-quality Figure 8: FE model setup, ligaments, boundary conditions, and load cases.

Designed with card-based visual containers, zero geometry-label overlap,
and mathematically aligned baselines with balanced vertical rhythm.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUT_PDF = Path('analysis_outputs/plos_revision/figures/model_setup.pdf')
OUT_PNG = Path('analysis_outputs/plos_revision/figures/model_setup.png')
MANUSCRIPT_PDF = Path('manuscripts/natcomm/images/model_setup.pdf')
TMP_DIR = Path('/tmp/model_setup_hq')
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. 3D RENDERING (PyVista)
# ==========================================
mesh = pv.read('results/ref_S1P_fixed_new2/paraview/mesh_complete.vtk')
ligs = pv.read('results/ref_S1P_fixed_new2/ligaments/ligaments.vtm')
surf = mesh.extract_surface()

# Extract intact pelvic bone surface
bone_surf = surf.extract_cells(surf.cell_data['RegionName'] == 'PelvisBone').extract_surface()

# Articular cartilages
pub_sym = surf.extract_cells(surf.cell_data['RegionName'] == 'PubicSymphysis').extract_surface().compute_normals(point_normals=True)
sij_l = surf.extract_cells(surf.cell_data['RegionName'] == 'SIJCartilageLeft').extract_surface().compute_normals(point_normals=True)
sij_r = surf.extract_cells(surf.cell_data['RegionName'] == 'SIJCartilageRight').extract_surface().compute_normals(point_normals=True)

# Bone bodies
bodies = bone_surf.split_bodies()
r_hemi = bodies[0].extract_surface().compute_normals(point_normals=True, feature_angle=60.0)
l_hemi = bodies[1].extract_surface().compute_normals(point_normals=True, feature_angle=60.0)

# Sacrum with seamless integrated S1 facet
sacrum = bodies[2].extract_surface()
sac_cols = np.full((sacrum.n_cells, 3), [232, 226, 216], dtype=np.uint8) # ivory bone #e8e2d8
sac_cols[sacrum.cell_data['DisplayName'] == 'S1_facet'] = [26, 150, 65]   # fixed S1 green #1a9641
sacrum.cell_data['RGB'] = sac_cols
sacrum = sacrum.compute_normals(point_normals=True, feature_angle=60.0)

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

from analysis.publication_rendering import create_publication_plotter

def render_scene(path, actors, cam_pos, focal_point, view_up, size=(1200, 1200), ssao=True):
    p = create_publication_plotter(
        window_size=size,
        enable_ssaa=True,
        enable_ssao=ssao,
        ssao_radius=25.0,
        ssao_bias=0.005,
        ssao_kernel_size=256,
    )
    for mesh_obj, kwargs in actors:
        kw = dict(smooth_shading=True, ambient=0.25, diffuse=0.75, specular=0.12)
        kw.update(kwargs)
        p.add_mesh(mesh_obj, **kw)
    p.camera_position = [cam_pos, focal_point, view_up]
    p.screenshot(str(path))
    p.close()

center = bone_surf.center

# Panel A: Full model + ligaments
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

# Panel B1: Inlet diameters
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

# Panel B2: Outlet diameters
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
render_scene(TMP_DIR / 'hq_panel_B2.png', actors_B2, outlet_center + 380 * outlet_norm, outlet_center, (0, 1, 0), size=(1200, 1200))

# Panel C1: SP2leg (Bilateral stance) - arrows shifted slightly anteriorly so arrowhead is crisp and visible
c_al = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'left_AC_notch').center)
c_ar = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'right_AC_notch').center)
arr_al = pv.Arrow(start=c_al - [0, 18, 65], direction=[0, 0, 1], scale=52, tip_length=0.32, tip_radius=0.14, shaft_radius=0.065)
arr_ar = pv.Arrow(start=c_ar - [0, 18, 65], direction=[0, 0, 1], scale=52, tip_length=0.32, tip_radius=0.14, shaft_radius=0.065)

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

# Panel C2: SP1leg (Unilateral stance) - arrow shifted slightly anteriorly so arrowhead is crisp and visible
arr_ar_single = pv.Arrow(start=c_ar - [0, 18, 75], direction=[0, 0, 1], scale=62, tip_length=0.32, tip_radius=0.15, shaft_radius=0.075)

bone_c2 = get_colored_bone({
    'S1_facet': [26, 150, 65],
    'right_AC_notch': [217, 95, 2],
})

actors_C2 = [
    (bone_c2, {'scalars': 'RGB', 'rgb': True, 'specular': 0.15}),
    (arr_ar_single, {'color': '#d95f02'}),
]
render_scene(TMP_DIR / 'hq_panel_C2.png', actors_C2, cam_C, center, (0, 0, 1), size=(1000, 1000))

# Panel D1: LAB1 (Inlet ring expansion)
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

# Panel D2: LAB2 (Ischial distraction)
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

# Panel D3: LAB3 (AP outlet distraction)
c_pub = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'pubis_ins').center)
c_scj = np.array(bone_surf.extract_cells(bone_surf.cell_data['DisplayName'] == 'SCJ').center)
arr_pub = pv.Arrow(start=c_pub, direction=[0, -1, 0], scale=48, tip_length=0.30, tip_radius=0.13, shaft_radius=0.065)
arr_scj = pv.Arrow(start=c_scj, direction=[0, 1, 0], scale=48, tip_length=0.30, tip_radius=0.13, shaft_radius=0.065)

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
cam_D3 = [center[0] - 440, center[1] - 280, center[2] + 95]
render_scene(TMP_DIR / 'hq_panel_D3.png', actors_D3, cam_D3, center, (0, 0, 1), size=(1000, 1000))

print('All 8 high-res renders completed.')

# ==========================================
# 2. IMAGE PREPARATION & STANDARDIZED CROPPING
# ==========================================
def load_and_crop(path, pad=20):
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

def load_crop_and_pad_to_aspect(path, target_aspect=1.0, pad=25):
    """Crops non-white content and pads onto a white canvas of fixed aspect ratio."""
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
    cropped = im.crop((x_min, y_min, x_max, y_max))
    w, h = cropped.size
    
    if w / h > target_aspect:
        new_w = w
        new_h = int(round(w / target_aspect))
    else:
        new_h = h
        new_w = int(round(h * target_aspect))
    
    canvas = Image.new('RGBA', (new_w, new_h), (255, 255, 255, 255))
    paste_x = (new_w - w) // 2
    paste_y = (new_h - h) // 2
    canvas.paste(cropped, (paste_x, paste_y), cropped)
    return np.array(canvas)

img_A = load_and_crop(TMP_DIR / 'hq_panel_A.png', pad=25)
img_B1 = load_crop_and_pad_to_aspect(TMP_DIR / 'hq_panel_B1.png', target_aspect=1.12, pad=25)
img_B2 = load_crop_and_pad_to_aspect(TMP_DIR / 'hq_panel_B2.png', target_aspect=1.12, pad=25)

# Standardize bottom row panels to identical aspect ratio (1.05) so all 5 images have the EXACT SAME size & vertical center
img_C1 = load_crop_and_pad_to_aspect(TMP_DIR / 'hq_panel_C1.png', target_aspect=1.05, pad=22)
img_C2 = load_crop_and_pad_to_aspect(TMP_DIR / 'hq_panel_C2.png', target_aspect=1.05, pad=22)
img_D1 = load_crop_and_pad_to_aspect(TMP_DIR / 'hq_panel_D1.png', target_aspect=1.05, pad=22)
img_D2 = load_crop_and_pad_to_aspect(TMP_DIR / 'hq_panel_D2.png', target_aspect=1.05, pad=22)
img_D3 = load_crop_and_pad_to_aspect(TMP_DIR / 'hq_panel_D3.png', target_aspect=1.05, pad=22)

# ==========================================
# 3. ASSEMBLE COMPOSITE FIGURE
# ==========================================
fig = plt.figure(figsize=(11.5, 9.2), dpi=300)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 8.5

# Main layout GridSpec:
# Row 0: Top section (Panels A, B1, B2)
# Row 1: Category Header Banners for Load Cases
# Row 2: Bottom images (C1, C2, D1, D2, D3)
# Row 3: Bottom info cards (C1, C2, D1, D2, D3)
gs_main = gridspec.GridSpec(
    4, 1,
    height_ratios=[1.30, 0.08, 0.72, 0.22],
    hspace=0.10,
    top=0.96, bottom=0.03, left=0.04, right=0.96
)

# ------------------------------------------
# TOP SECTION: Sub-grid for A, B1, B2
# Row 0: Images
# Row 1: Info / Legend cards (sharing identical height & baseline)
# ------------------------------------------
gs_top = gridspec.GridSpecFromSubplotSpec(
    2, 3,
    subplot_spec=gs_main[0],
    height_ratios=[1.0, 0.24],
    width_ratios=[1.42, 1.0, 1.0],
    hspace=0.05, wspace=0.12
)

ax_A = fig.add_subplot(gs_top[0, 0])
ax_A_leg = fig.add_subplot(gs_top[1, 0])

ax_B1 = fig.add_subplot(gs_top[0, 1])
ax_B1_info = fig.add_subplot(gs_top[1, 1])

ax_B2 = fig.add_subplot(gs_top[0, 2])
ax_B2_info = fig.add_subplot(gs_top[1, 2])

# --- Panel A ---
ax_A.imshow(img_A)
ax_A.axis('off')
ax_A.set_title(r'$\mathbf{A}$   Pelvic finite-element assembly & soft tissues', loc='left', fontsize=10.5, pad=7, fontweight='bold', color='#0f172a')

# 1. S1 fixed boundary badge (top center, clearly clear of iliac crests)
ax_A.annotate(
    r'Fixed boundary: $S_1$ facet ($\mathbf{u} = \mathbf{0}$)',
    xy=(0.50, 0.77), xycoords='axes fraction',
    xytext=(0.50, 0.95), textcoords='axes fraction',
    ha='center', va='center', fontsize=8.2, fontweight='bold', color='#15803d',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#f0fdf4', edgecolor='#86efac', lw=1.0),
    arrowprops=dict(arrowstyle='->', color='#15803d', lw=1.5)
)

# 2. Sacroiliac joint badge (top right corner, 100% OUTSIDE bone geometry)
ax_A.annotate(
    'Sacroiliac joint\n(articular cartilage)',
    xy=(0.70, 0.65), xycoords='axes fraction',
    xytext=(0.86, 0.90), textcoords='axes fraction',
    ha='center', va='center', fontsize=7.8, fontweight='bold', color='#0369a1',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#f0f9ff', edgecolor='#7dd3fc', lw=1.0),
    arrowprops=dict(arrowstyle='->', color='#0369a1', lw=1.3, connectionstyle='arc3,rad=-0.12')
)

# 3. Pubic symphysis badge (bottom left corner, 100% OUTSIDE bone geometry, canal completely unobstructed)
ax_A.annotate(
    'Pubic symphysis\n(fibrocartilage disc)',
    xy=(0.50, 0.05), xycoords='axes fraction',
    xytext=(0.13, 0.07), textcoords='axes fraction',
    ha='center', va='center', fontsize=7.8, fontweight='bold', color='#7c3aed',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#faf5ff', edgecolor='#d8b4fe', lw=1.0),
    arrowprops=dict(arrowstyle='->', color='#7c3aed', lw=1.3, connectionstyle='arc3,rad=0.18')
)

# Panel A Ligament Legend Card (ax_A_leg)
ax_A_leg.axis('off')
rect_leg = FancyBboxPatch((0.01, 0.04), 0.98, 0.92, boxstyle='round,pad=0.03,rounding_size=0.05',
                          facecolor='#f8fafc', edgecolor='#e2e8f0', lw=1.0, transform=ax_A_leg.transAxes)
ax_A_leg.add_patch(rect_leg)

# 6 ligament items in 2 rows x 3 columns
lig_items = [
    ('#d7191c', 'Anterior SIJ lig.'),
    ('#fdae61', 'Posterior SIJ lig.'),
    ('#2b83ba', 'Interosseous SIJ lig.'),
    ('#e66101', 'Sacrospinous lig.'),
    ('#5e3c99', 'Sacrotuberous lig.'),
    ('#c51b7d', 'Pubic ligaments'),
]
xs = [0.025, 0.35, 0.67]
ys = [0.65, 0.25]
for idx, (col, lbl) in enumerate(lig_items):
    col_idx = idx % 3
    row_idx = idx // 3
    ax_A_leg.plot(xs[col_idx], ys[row_idx], marker='s', markersize=6.5, color=col, transform=ax_A_leg.transAxes)
    ax_A_leg.text(xs[col_idx] + 0.032, ys[row_idx], lbl, transform=ax_A_leg.transAxes,
                  va='center', ha='left', fontsize=7.2, color='#1e293b', fontweight='medium')

# --- Panel B1: Inlet diameters ---
ax_B1.imshow(img_B1)
ax_B1.axis('off')
ax_B1.set_title(r'$\mathbf{B}_1$   Inlet plane (superior view)', loc='left', fontsize=10.0, pad=7, fontweight='bold', color='#0f172a')

# Panel B1 Info Card (ax_B1_info)
ax_B1_info.axis('off')
rect_b1 = FancyBboxPatch((0.01, 0.04), 0.98, 0.92, boxstyle='round,pad=0.03,rounding_size=0.05',
                         facecolor='#f8fafc', edgecolor='#e2e8f0', lw=1.0, transform=ax_B1_info.transAxes)
ax_B1_info.add_patch(rect_b1)
ax_B1_info.text(0.50, 0.76, 'Pelvic Inlet Diameters', transform=ax_B1_info.transAxes,
               ha='center', va='center', fontsize=7.8, fontweight='bold', color='#0f172a')
# AP
ax_B1_info.plot(0.06, 0.44, marker='o', markersize=5.5, color='#d7191c', transform=ax_B1_info.transAxes)
ax_B1_info.text(0.12, 0.44, 'Conjugata vera (AP): 123 mm', transform=ax_B1_info.transAxes,
               ha='left', va='center', fontsize=7.2, color='#1e293b', fontweight='medium')
# ML
ax_B1_info.plot(0.06, 0.18, marker='o', markersize=5.5, color='#2b83ba', transform=ax_B1_info.transAxes)
ax_B1_info.text(0.12, 0.18, 'Transverse diameter (ML): 131 mm', transform=ax_B1_info.transAxes,
               ha='left', va='center', fontsize=7.2, color='#1e293b', fontweight='medium')

# --- Panel B2: Outlet diameters ---
ax_B2.imshow(img_B2)
ax_B2.axis('off')
ax_B2.set_title(r'$\mathbf{B}_2$   Outlet plane (inferior view)', loc='left', fontsize=10.0, pad=7, fontweight='bold', color='#0f172a')

# Panel B2 Info Card (ax_B2_info)
ax_B2_info.axis('off')
rect_b2 = FancyBboxPatch((0.01, 0.04), 0.98, 0.92, boxstyle='round,pad=0.03,rounding_size=0.05',
                         facecolor='#f8fafc', edgecolor='#e2e8f0', lw=1.0, transform=ax_B2_info.transAxes)
ax_B2_info.add_patch(rect_b2)
ax_B2_info.text(0.50, 0.76, 'Pelvic Outlet Diameters', transform=ax_B2_info.transAxes,
               ha='center', va='center', fontsize=7.8, fontweight='bold', color='#0f172a')
# BIS
ax_B2_info.plot(0.06, 0.44, marker='o', markersize=5.5, color='#7b3294', transform=ax_B2_info.transAxes)
ax_B2_info.text(0.12, 0.44, 'Biischiadic diameter (BIS): 103 mm', transform=ax_B2_info.transAxes,
               ha='left', va='center', fontsize=7.2, color='#1e293b', fontweight='medium')
# BIT
ax_B2_info.plot(0.06, 0.18, marker='o', markersize=5.5, color='#1a9641', transform=ax_B2_info.transAxes)
ax_B2_info.text(0.12, 0.18, 'Bituberous diameter (BIT): 132 mm', transform=ax_B2_info.transAxes,
               ha='left', va='center', fontsize=7.2, color='#1e293b', fontweight='medium')

# ------------------------------------------
# CATEGORY HEADER BANNERS (Row 1)
# Eliminates vast dead space and provides clear visual grouping
# ------------------------------------------
gs_hdr = gridspec.GridSpecFromSubplotSpec(
    1, 2,
    subplot_spec=gs_main[1],
    width_ratios=[0.40, 0.60],
    wspace=0.04
)

ax_hdr_stand = fig.add_subplot(gs_hdr[0])
ax_hdr_stand.axis('off')
rect_hs = FancyBboxPatch((0.00, 0.05), 0.98, 0.90, boxstyle='round,pad=0.02,rounding_size=0.08',
                         facecolor='#fffbeb', edgecolor='#fde68a', lw=1.0, transform=ax_hdr_stand.transAxes)
ax_hdr_stand.add_patch(rect_hs)
ax_hdr_stand.text(0.50, 0.50, 'Habitual Locomotor Loads (Standing)', transform=ax_hdr_stand.transAxes,
                 ha='center', va='center', fontsize=8.8, fontweight='bold', color='#92400e')

ax_hdr_labor = fig.add_subplot(gs_hdr[1])
ax_hdr_labor.axis('off')
rect_hl = FancyBboxPatch((0.00, 0.05), 1.00, 0.90, boxstyle='round,pad=0.02,rounding_size=0.08',
                         facecolor='#fdf2f8', edgecolor='#fbcfe8', lw=1.0, transform=ax_hdr_labor.transAxes)
ax_hdr_labor.add_patch(rect_hl)
ax_hdr_labor.text(0.50, 0.50, 'Parturition-Motivated Proxy Loads (Birth Canal Transit Stages)', transform=ax_hdr_labor.transAxes,
                 ha='center', va='center', fontsize=8.8, fontweight='bold', color='#9d174d')

# ------------------------------------------
# BOTTOM SECTION: 5 Load Cases (Images in Row 2, Info Cards in Row 3)
# ------------------------------------------
gs_bot_img = gridspec.GridSpecFromSubplotSpec(
    1, 5,
    subplot_spec=gs_main[2],
    width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
    wspace=0.10
)

gs_bot_card = gridspec.GridSpecFromSubplotSpec(
    1, 5,
    subplot_spec=gs_main[3],
    width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
    wspace=0.10
)

axes_img = [fig.add_subplot(gs_bot_img[i]) for i in range(5)]
axes_card = [fig.add_subplot(gs_bot_card[i]) for i in range(5)]

bot_imgs = [img_C1, img_C2, img_D1, img_D2, img_D3]
bot_titles = [
    r'$\mathbf{C}_1$   $\mathrm{SP2leg}$',
    r'$\mathbf{C}_2$   $\mathrm{SP1leg}$',
    r'$\mathbf{D}_1$   $\mathrm{LAB}_1$ (Inlet)',
    r'$\mathbf{D}_2$   $\mathrm{LAB}_2$ (Midpelvis)',
    r'$\mathbf{D}_3$   $\mathrm{LAB}_3$ (Outlet)',
]

card_data = [
    {
        'sub': 'Bilateral standing',
        'force': '2 × 400 N vertical',
        'target': 'Acetabular notches',
        'bg': '#fffbeb', 'border': '#fde68a', 'accent': '#b45309'
    },
    {
        'sub': 'Unilateral standing',
        'force': '1 × 800 N vertical',
        'target': 'Right acetabular notch',
        'bg': '#fffbeb', 'border': '#fde68a', 'accent': '#b45309'
    },
    {
        'sub': 'Inlet ring expansion',
        'force': '±400 N lateral',
        'target': 'Iliopectineal inner ring',
        'bg': '#fdf2f8', 'border': '#fbcfe8', 'accent': '#be185d'
    },
    {
        'sub': 'Tuberosity distraction',
        'force': '±400 N lateral',
        'target': 'Ischial tuberosities/spines',
        'bg': '#faf5ff', 'border': '#e9d5ff', 'accent': '#7c3aed'
    },
    {
        'sub': 'AP outlet distraction',
        'force': '±400 N anteroposterior',
        'target': 'Pubic crest & SCJ',
        'bg': '#fffbeb', 'border': '#fde68a', 'accent': '#b45309'
    },
]

for i in range(5):
    # Image
    ax_i = axes_img[i]
    ax_i.imshow(bot_imgs[i])
    ax_i.axis('off')
    ax_i.set_title(bot_titles[i], loc='center', fontsize=9.0, pad=5, fontweight='bold', color='#0f172a')
    
    # Card
    ax_c = axes_card[i]
    ax_c.axis('off')
    cd = card_data[i]
    
    rect_c = FancyBboxPatch((0.02, 0.05), 0.96, 0.90, boxstyle='round,pad=0.03,rounding_size=0.06',
                            facecolor=cd['bg'], edgecolor=cd['border'], lw=1.0, transform=ax_c.transAxes)
    ax_c.add_patch(rect_c)
    
    ax_c.text(0.50, 0.74, cd['sub'], transform=ax_c.transAxes,
              ha='center', va='center', fontsize=7.4, fontweight='bold', color='#0f172a')
    ax_c.text(0.50, 0.44, cd['force'], transform=ax_c.transAxes,
              ha='center', va='center', fontsize=7.2, fontweight='bold', color=cd['accent'])
    ax_c.text(0.50, 0.18, cd['target'], transform=ax_c.transAxes,
              ha='center', va='center', fontsize=6.6, color='#64748b', fontweight='medium')

# Export both to analysis_outputs and directly to manuscript directory
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
fig.savefig(OUT_PNG, bbox_inches='tight', dpi=300)
fig.savefig(MANUSCRIPT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)

print(f'Successfully generated Figure 8:')
print(f'  - PDF: {OUT_PDF}')
print(f'  - PNG: {OUT_PNG}')
print(f'  - Manuscript: {MANUSCRIPT_PDF}')
