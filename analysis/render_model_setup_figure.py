#!/usr/bin/env python3
"""High-resolution scientific rendering of pelvic FE model, ligaments, boundary conditions, and loads."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from analysis.spectral_data import load_inlet_landmarks, load_outlet_landmarks
from analysis.publication_rendering import create_publication_plotter

OUT_DIR = Path('analysis_outputs/plos_revision/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path('/tmp/model_setup_renders')
TMP_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load FE geometry
mesh = pv.read('results/ref_S1P_fixed_new2/paraview/mesh_complete.vtk')
ligs = pv.read('results/ref_S1P_fixed_new2/ligaments/ligaments.vtm')

# Separate bodies
bone = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'PelvisBone').extract_surface()
bodies = bone.split_bodies()
r_hemi = bodies[0]
l_hemi = bodies[1]
sacrum = bodies[2]

s1 = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'S1_facet').extract_surface()
pub_sym = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'PubicSymphysis').extract_surface()
sij_l = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'SIJCartilageLeft').extract_surface()
sij_r = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'SIJCartilageRight').extract_surface()

ac_l = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'left_AC_notch').extract_surface()
ac_r = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'right_AC_notch').extract_surface()
ring_l = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'ring_contact_left').extract_surface()
ring_r = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'ring_contact_right').extract_surface()
isch_l = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'left_ischium_tuber').extract_surface()
isch_r = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'right_ischium_tuber').extract_surface()
pub_ins = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'pubis_ins').extract_surface()
scj = mesh.extract_cells(mesh.cell_data['DisplayName'] == 'SCJ').extract_surface()

# Colors
c_sacrum = '#e8e2d8'
c_hemi = '#ece7df'
c_sij = '#2b83ba'       # SIJ cartilage blue
c_pub = '#7b3294'       # Pubic symphysis purple
c_s1 = '#1a9641'        # Fixed S1 green
c_load = '#d95f02'      # Load orange

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
    p = create_publication_plotter(
        window_size=size,
        background='white',
        enable_ssaa=True,
        enable_ssao=True,
        ssao_radius=25.0,
        ssao_bias=0.005,
        ssao_kernel_size=256,
    )
    for mesh_obj, kwargs in actors:
        clean_kwargs = dict(kwargs)
        if isinstance(mesh_obj, pv.PolyData) and "Normals" not in mesh_obj.point_data:
            mesh_obj = mesh_obj.compute_normals(point_normals=True, feature_angle=60.0)
        p.add_mesh(mesh_obj, smooth_shading=True, **clean_kwargs)
    p.camera_position = [cam_pos, focal_point, view_up]
    p.screenshot(str(path))
    p.close()

center = bone.center

# --- A. Perspective overview with ligaments & S1 facet ---
actors_A = [
    (sacrum, {'color': c_sacrum, 'specular': 0.15}),
    (r_hemi, {'color': c_hemi, 'specular': 0.15}),
    (l_hemi, {'color': c_hemi, 'specular': 0.15}),
    (s1, {'color': c_s1, 'specular': 0.35}),
    (pub_sym, {'color': c_pub, 'specular': 0.2}),
    (sij_l, {'color': c_sij, 'specular': 0.2}),
    (sij_r, {'color': c_sij, 'specular': 0.2}),
]
for i in range(len(ligs)):
    blk = ligs[i]
    if blk.n_cells > 0:
        col = lig_colors.get(ligs.get_block_name(i), '#636363')
        actors_A.append((blk.tube(radius=1.35), {'color': col}))

cam_A = [center[0] + 15, center[1] - 380, center[2] + 240]
render_scene(TMP_DIR / 'panel_A.png', actors_A, cam_A, center, (0, 0.4, 0.9), size=(1600, 1400))

# --- B1. Inlet diameters ---
inlet = load_inlet_landmarks()
ap = inlet['AP']
ml = inlet['ML']
inlet_center = (ap.mean(axis=0) + ml.mean(axis=0)) / 2.0
ap_vec = ap[1] - ap[0]; ml_vec = ml[1] - ml[0]
inlet_norm = np.cross(ml_vec, ap_vec); inlet_norm /= np.linalg.norm(inlet_norm)

actors_B1 = [
    (bone, {'color': '#ece7df', 'specular': 0.15}),
    (pub_sym, {'color': c_pub}),
    (sij_l, {'color': c_sij}),
    (sij_r, {'color': c_sij}),
    (pv.lines_from_points(ap).tube(radius=2.2), {'color': '#d7191c'}),
    (pv.lines_from_points(ml).tube(radius=2.2), {'color': '#2b83ba'}),
    (pv.PolyData(ap), {'color': '#d7191c', 'point_size': 16, 'render_points_as_spheres': True}),
    (pv.PolyData(ml), {'color': '#2b83ba', 'point_size': 16, 'render_points_as_spheres': True}),
]
render_scene(TMP_DIR / 'panel_B1.png', actors_B1, inlet_center - 380 * inlet_norm, inlet_center, (0, 0, 1), size=(1200, 1200))

# --- B2. Outlet diameters ---
outlet = load_outlet_landmarks()
bis = outlet['BIS']
bit = outlet['BIT']
oap = outlet['OUTLETAP']
outlet_center = (bis.mean(axis=0) + bit.mean(axis=0)) / 2.0
bis_vec = bis[1] - bis[0]; oap_vec = oap[1] - oap[0]
outlet_norm = np.cross(bis_vec, oap_vec); outlet_norm /= np.linalg.norm(outlet_norm)

actors_B2 = [
    (bone, {'color': '#ece7df', 'specular': 0.15}),
    (pub_sym, {'color': c_pub}),
    (pv.lines_from_points(bis).tube(radius=2.2), {'color': '#7b3294'}),
    (pv.lines_from_points(bit).tube(radius=2.2), {'color': '#1a9641'}),
    (pv.PolyData(bis), {'color': '#7b3294', 'point_size': 16, 'render_points_as_spheres': True}),
    (pv.PolyData(bit), {'color': '#1a9641', 'point_size': 16, 'render_points_as_spheres': True}),
]
render_scene(TMP_DIR / 'panel_B2.png', actors_B2, outlet_center - 380 * outlet_norm, outlet_center, (0, 0, 1), size=(1200, 1200))

# --- C. Standing loads (SP2leg / SP1leg) ---
c_acl = np.array(ac_l.center); c_acr = np.array(ac_r.center)
arr_l = pv.Arrow(start=c_acl - [0, 0, 52], direction=[0, 0, 1], scale=52, tip_length=0.32, tip_radius=0.12, shaft_radius=0.05)
arr_r = pv.Arrow(start=c_acr - [0, 0, 52], direction=[0, 0, 1], scale=52, tip_length=0.32, tip_radius=0.12, shaft_radius=0.05)

actors_C1 = [
    (bone, {'color': '#ece7df', 'specular': 0.15}),
    (s1, {'color': c_s1, 'specular': 0.35}),
    (ac_l, {'color': '#d95f02'}),
    (ac_r, {'color': '#d95f02'}),
    (arr_l, {'color': '#d95f02'}),
    (arr_r, {'color': '#d95f02'}),
]
render_scene(TMP_DIR / 'panel_C1.png', actors_C1, [center[0], center[1] - 420, center[2] - 40], center, (0, 0, 1), size=(1000, 1000))

# SP1leg (right only, 800 N)
arr_r_800 = pv.Arrow(start=c_acr - [0, 0, 65], direction=[0, 0, 1], scale=65, tip_length=0.32, tip_radius=0.14, shaft_radius=0.06)
actors_C2 = [
    (bone, {'color': '#ece7df', 'specular': 0.15}),
    (s1, {'color': c_s1, 'specular': 0.35}),
    (ac_r, {'color': '#d95f02'}),
    (arr_r_800, {'color': '#d95f02'}),
]
render_scene(TMP_DIR / 'panel_C2.png', actors_C2, [center[0], center[1] - 420, center[2] - 40], center, (0, 0, 1), size=(1000, 1000))

# --- D. Labor loads ---
# LAB1: Ring contact
actors_D1 = [
    (bone, {'color': '#ece7df', 'specular': 0.15}),
    (s1, {'color': c_s1, 'specular': 0.35}),
    (ring_l, {'color': '#e7298a'}),
    (ring_r, {'color': '#e7298a'}),
]
render_scene(TMP_DIR / 'panel_D1.png', actors_D1, [center[0], center[1] - 320, center[2] + 280], center, (0, 0.4, 0.9), size=(1000, 1000))

# LAB2: Ischial tuberosity
c_itl = np.array(isch_l.center); c_itr = np.array(isch_r.center)
arr_il = pv.Arrow(start=c_itl, direction=[1, 0, 0], scale=45, tip_length=0.32, tip_radius=0.12, shaft_radius=0.05)
arr_ir = pv.Arrow(start=c_itr, direction=[-1, 0, 0], scale=45, tip_length=0.32, tip_radius=0.12, shaft_radius=0.05)
actors_D2 = [
    (bone, {'color': '#ece7df', 'specular': 0.15}),
    (s1, {'color': c_s1, 'specular': 0.35}),
    (isch_l, {'color': '#7570b3'}),
    (isch_r, {'color': '#7570b3'}),
    (arr_il, {'color': '#7570b3'}),
    (arr_ir, {'color': '#7570b3'}),
]
render_scene(TMP_DIR / 'panel_D2.png', actors_D2, [center[0], center[1] - 250, center[2] - 320], center, (0, 0, 1), size=(1000, 1000))

# LAB3: Pubis & SCJ
c_pub = np.array(pub_ins.center); c_scj = np.array(scj.center)
arr_pub = pv.Arrow(start=c_pub, direction=[0, -1, 0], scale=45, tip_length=0.32, tip_radius=0.12, shaft_radius=0.05)
arr_scj = pv.Arrow(start=c_scj, direction=[0, 1, 0], scale=45, tip_length=0.32, tip_radius=0.12, shaft_radius=0.05)
actors_D3 = [
    (bone, {'color': '#ece7df', 'specular': 0.15}),
    (s1, {'color': c_s1, 'specular': 0.35}),
    (pub_ins, {'color': '#e6ab02'}),
    (scj, {'color': '#e6ab02'}),
    (arr_pub, {'color': '#e6ab02'}),
    (arr_scj, {'color': '#e6ab02'}),
]
render_scene(TMP_DIR / 'panel_D3.png', actors_D3, [center[0] - 350, center[1] - 250, center[2] + 100], center, (0, 0, 1), size=(1000, 1000))

print('All component renders successfully generated.')
