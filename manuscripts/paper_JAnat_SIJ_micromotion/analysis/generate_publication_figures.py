#!/usr/bin/env python3
"""Publication figure generator for the Journal of Anatomy SIJ micromotion manuscript.

Generates an anatomical overview, load profiles, paired component contrasts,
conditional sex estimates, variant variance ratios, adjusted volume slopes,
and supplementary raw observations with conditional geometric-mean curves.
"""

from __future__ import annotations

import sys
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from PIL import Image
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.formula.api as smf

PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "tables" / "generated"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling matching PLOS / J Anat
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "figure.titlesize": 11.5,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Okabe-Ito Colorblind-safe palette
COLOR_FEMALE = "#d55e00"   # Vermilion
COLOR_MALE = "#0072b2"     # Deep blue
COLOR_NEUTRAL = "#2b83ba"  # Slate blue
COLOR_AP = "#0072b2"       # Blue
COLOR_ML = "#009e73"       # Green
COLOR_CC = "#d55e00"       # Vermilion
COLOR_NUT = "#e69f00"      # Amber

LOAD_ORDER = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
LAB_LOADS = ["LAB_phase1", "LAB_phase2", "LAB_phase3"]
LOAD_LABELS = {
    "SP2leg": "SP2leg\n(2-leg stand)",
    "SP1leg": "SP1leg\n(1-leg stand)",
    "LAB_phase1": "LAB1\n(Ring pair)",
    "LAB_phase2": "LAB2\n(Ischial pair)",
    "LAB_phase3": "LAB3\n(AP pair)",
}
LOAD_SHORT = {
    "SP2leg": "SP2leg",
    "SP1leg": "SP1leg",
    "LAB_phase1": "LAB1",
    "LAB_phase2": "LAB2",
    "LAB_phase3": "LAB3",
}

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_df = pd.read_csv(TABLE_DIR / "subject_level.csv")
    delta_df = pd.read_csv(TABLE_DIR / "directional_rerouting.csv")
    variance_df = pd.read_csv(TABLE_DIR / "variance_channels.csv")
    return subject_df, delta_df, variance_df

# ==============================================================================
# FIGURE 1: 3D ANATOMY, COORDINATES, AND LOAD CASES
# ==============================================================================
def render_fig1_3d_anatomy(out_png: Path) -> None:
    """Render high-resolution 3D pelvic anatomy with SIJ highlighted using PyVista."""
    import pyvista as pv
    sys.path.insert(0, str(REPO_ROOT))
    from analysis.publication_rendering import create_publication_plotter

    mesh = pv.read(str(REPO_ROOT / "results" / "ref_S1P_fixed_new2" / "paraview" / "mesh_complete.vtk"))

    bone = mesh.extract_cells(mesh.cell_data["DisplayName"] == "PelvisBone").extract_surface()
    bodies = bone.split_bodies()
    r_hemi = bodies[0].extract_surface()
    l_hemi = bodies[1].extract_surface()
    sacrum = bodies[2].extract_surface()

    # Fit oriented elliptical cylinder plate to S1 facet to eliminate tetrahedral coplanar z-fighting
    s1_vol = mesh.extract_cells(mesh.cell_data["DisplayName"] == "S1_facet")
    pts = s1_vol.points
    c = pts.mean(0)
    u, s, vh = np.linalg.svd(pts - c)
    normal = vh[2]
    if normal[2] < 0:
        normal = -normal
    v_major = vh[0]
    v_minor = np.cross(normal, v_major)

    cyl = pv.Cylinder(center=[0, 0, 0], direction=[0, 0, 1], radius=1.0, height=3.0, resolution=120)
    scale_mat = np.diag([22.5, 14.5, 1.0, 1.0])
    rot_mat = np.eye(4)
    rot_mat[:3, 0] = v_major
    rot_mat[:3, 1] = v_minor
    rot_mat[:3, 2] = normal
    rot_mat[:3, 3] = c + normal * 2.2
    trans_mat = rot_mat @ scale_mat
    cyl.transform(trans_mat, inplace=True)

    pub_sym = mesh.extract_cells(mesh.cell_data["DisplayName"] == "PubicSymphysis").extract_surface()
    sij_l = mesh.extract_cells(mesh.cell_data["DisplayName"] == "SIJCartilageLeft").extract_surface()
    sij_r = mesh.extract_cells(mesh.cell_data["DisplayName"] == "SIJCartilageRight").extract_surface()

    p = create_publication_plotter(window_size=(1800, 1600), enable_ssaa=True, enable_ssao=True, ssao_radius=15.0)

    p.add_mesh(sacrum, color="#dfdcd4", smooth_shading=True, specular=0.15)
    p.add_mesh(r_hemi, color="#e8e5dc", smooth_shading=True, specular=0.15)
    p.add_mesh(l_hemi, color="#e8e5dc", smooth_shading=True, specular=0.15)
    p.add_mesh(pub_sym, color="#7b3294", smooth_shading=True, specular=0.25)
    p.add_mesh(cyl, color="#1a9641", smooth_shading=True, specular=0.35, ambient=0.2)

    # Highlight SIJ cartilage in vibrant high-contrast orange
    p.add_mesh(sij_l, color="#d95f02", smooth_shading=True, specular=0.5, ambient=0.25)
    p.add_mesh(sij_r, color="#d95f02", smooth_shading=True, specular=0.5, ambient=0.25)

    center = bone.center
    cam_pos = (center[0] + 110, center[1] - 440, center[2] + 290)
    focal = (center[0], center[1] + 15, center[2] - 10)
    view_up = (0, 0, 1)

    p.camera_position = [cam_pos, focal, view_up]
    p.camera.zoom(1.18)
    p.screenshot(str(out_png))
    p.close()

def render_sij_3d_articulation(out_png: Path) -> None:
    """Render high-resolution 3D exploded view of SIJ articulation with coordinate triad."""
    import pyvista as pv
    sys.path.insert(0, str(REPO_ROOT))
    from analysis.publication_rendering import create_publication_plotter

    mesh = pv.read(str(REPO_ROOT / "results" / "ref_S1P_fixed_new2" / "paraview" / "mesh_complete.vtk"))
    bone = mesh.extract_cells(mesh.cell_data["DisplayName"] == "PelvisBone").extract_surface()
    bodies = bone.split_bodies()
    l_hemi = bodies[0].extract_surface()
    sacrum = bodies[2].extract_surface().compute_normals(point_normals=True, feature_angle=60.0)
    sij_l = mesh.extract_cells(mesh.cell_data["DisplayName"] == "SIJCartilageLeft").extract_surface()

    c = np.array(sij_l.center)
    l_hemi_exploded = l_hemi.copy()
    l_hemi_exploded.points = l_hemi_exploded.points + [-40.0, 0, 0]
    l_hemi_exploded = l_hemi_exploded.compute_normals(point_normals=True, feature_angle=60.0)

    scale = 30.0
    arr_ml = pv.Arrow(start=c, direction=[-1, 0, 0], scale=scale, tip_length=0.32, tip_radius=0.14, shaft_radius=0.065)
    arr_ap = pv.Arrow(start=c, direction=[0, 1, 0], scale=scale, tip_length=0.32, tip_radius=0.14, shaft_radius=0.065)
    arr_cc = pv.Arrow(start=c, direction=[0, 0, 1], scale=scale, tip_length=0.32, tip_radius=0.14, shaft_radius=0.065)
    sphere = pv.Sphere(radius=2.5, center=c)

    p = create_publication_plotter(window_size=(1100, 1100), enable_ssaa=True, enable_ssao=True, ssao_radius=22.0)
    p.add_mesh(sacrum, color="#dfdcd4", smooth_shading=True, specular=0.15)
    p.add_mesh(l_hemi_exploded, color="#ece8e0", smooth_shading=True, specular=0.15)
    p.add_mesh(sij_l, color="#ea580c", smooth_shading=True, specular=0.4, ambient=0.25)
    p.add_mesh(sphere, color="#1e293b")
    p.add_mesh(arr_ml, color="#16a34a")
    p.add_mesh(arr_ap, color="#0284c7")
    p.add_mesh(arr_cc, color="#dc2626")

    cam = c + np.array([-130, -170, 95])
    p.camera_position = [cam, c, (0, 0, 1)]
    p.camera.zoom(1.35)
    p.screenshot(str(out_png))
    p.close()

def _load_crop_and_pad_to_aspect(path: Path | str, target_aspect: float = 1.0, pad: int = 22) -> np.ndarray:
    """Crops white border and pads onto a white canvas of fixed aspect ratio."""
    im = Image.open(str(path)).convert("RGBA")
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

    canvas = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 255))
    paste_x = (new_w - w) // 2
    paste_y = (new_h - h) // 2
    canvas.paste(cropped, (paste_x, paste_y), cropped)
    return np.array(canvas)

def build_figure_1() -> None:
    """Compact anatomical overview; definitions use the manuscript's reporting frame."""
    cache = FIG_DIR / '.cache'
    fig = plt.figure(figsize=(12, 7.0), layout='constrained')
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=.07)
    top = outer[0].subgridspec(1, 3, width_ratios=[1, .85, 1.15], wspace=.06)
    for j, filename, title in [(0,'sij_fig1_render.png','A. Pelvic assembly'),
                                (1,'test_sij_smooth.png','B. SIJ articulation')]:
        ax=fig.add_subplot(top[j])
        ax.imshow(_load_crop_and_pad_to_aspect(cache/filename, target_aspect=1.1, pad=12))
        ax.axis('off');ax.set_title(title,loc='left',fontweight='bold')
        if j==0:
            note='Green: S1 constraint   Orange: SIJ cartilage\nPurple: pubic symphysis'
        else:
            note='Exploded view; arrows illustrate axis directions\nGreen: ML   Blue: AP   Red: CC'
        ax.text(.5,-.02,note,transform=ax.transAxes,ha='center',va='top',fontsize=8)
    ax=fig.add_subplot(top[2]);ax.axis('off')
    ax.set_title('C. Joint-motion definitions',loc='left',fontweight='bold')
    lines=[('Reference','Unloaded subject anatomy'),
           ('Rotations',r'Fixed-axis $xyz$: $R_z(\gamma)R_y(\beta)R_x(\alpha)$'),
           ('Translations','Difference of fitted bone maps at the\nsacral region-of-interest centroid'),
           ('Reporting axes','Pelvis-aligned ML, AP and CC;\nabsolute components are compared'),
           ('Bilateral asymmetry',r'$A_d=\left|\|\mathbf{d}_L\|-\|\mathbf{d}_R\|\right|$')]
    for y,(heading,body) in zip([.94,.77,.60,.38,.16],lines):
        ax.text(.02,y,heading,transform=ax.transAxes,fontsize=9,fontweight='bold',va='top')
        ax.text(.02,y-.065,body,transform=ax.transAxes,fontsize=8.5,va='top',linespacing=1.4)
    bottom=outer[1].subgridspec(1,5,wspace=.04)
    names=['C1','C2','D1','D2','D3']
    titles=['D. SP2leg','E. SP1leg','F. LAB1','G. LAB2','H. LAB3']
    notes=['Bilateral acetabular load\n400 N per side; 800 N total',
           'Right acetabular load\n800 N total',
           'Internal ring force pair\nOpposing ML forces: 400 N each',
           'Ischial force pair\nOpposing ML forces: 400 N each',
           'Pubis–sacrococcygeal pair\nOpposing AP forces: 400 N each']
    for j,(name,title,note) in enumerate(zip(names,titles,notes)):
        ax=fig.add_subplot(bottom[j])
        ax.imshow(_load_crop_and_pad_to_aspect(cache/f'hq_panel_{name}.png',target_aspect=.95,pad=12))
        ax.axis('off');ax.set_title(title,loc='left',fontweight='bold')
        ax.text(.5,-.035,note,transform=ax.transAxes,ha='center',va='top',fontsize=7.8,linespacing=1.5)
    for ext in ['pdf','png']:fig.savefig(FIG_DIR/f'Fig1_anatomy_coordinates_loads.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# FIGURE 2: PRIMARY LOAD PROFILES, NUTATION, AND BILATERAL ASYMMETRY
# ==============================================================================
def build_figure_2(subject_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), sharex=True)

    panels = [
        (axes[0, 0], "rot_mag_deg", "A. Cardan rotation norm (|θ|)", "Rotation magnitude (deg)", "#0072b2"),
        (axes[0, 1], "trans_mag_mm", "B. 3D Translation magnitude (|d|)", "Translation magnitude (mm)", "#009e73"),
        (axes[1, 0], "nut_abs_deg", r"C. Absolute ML rotation ($|\theta_x|$)", "Absolute ML rotation (deg)", "#e69f00"),
        (axes[1, 1], "lr_trans_asym_mm", r"D. Bilateral asymmetry ($|\|d_L\|-\|d_R\||$)", "Left-right asymmetry (mm)", "#d55e00"),
    ]

    rng = np.random.default_rng(20260920)
    x_indices = np.arange(len(LOAD_ORDER))
    width = 0.32

    for ax, col, title, ylabel, main_color in panels:
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10.0)
        ax.set_ylabel(ylabel, fontsize=9.0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for i, load in enumerate(LOAD_ORDER):
            sub = subject_df[subject_df["load_case"] == load]
            female_vals = sub[sub["sex_F"] == 1.0][col].dropna().to_numpy()
            male_vals = sub[sub["sex_F"] == 0.0][col].dropna().to_numpy()

            # Boxplots for Females and Males side by side
            bp_f = ax.boxplot(female_vals, positions=[i - width/2], widths=width*0.85, patch_artist=True,
                              showfliers=False, medianprops=dict(color="black", lw=1.2))
            bp_m = ax.boxplot(male_vals, positions=[i + width/2], widths=width*0.85, patch_artist=True,
                              showfliers=False, medianprops=dict(color="black", lw=1.2))

            for patch in bp_f["boxes"]:
                patch.set(facecolor=COLOR_FEMALE, alpha=0.75, edgecolor="#903000")
            for patch in bp_m["boxes"]:
                patch.set(facecolor=COLOR_MALE, alpha=0.75, edgecolor="#004080")

            # Jittered scatter overlay
            jitter_f = rng.normal(i - width/2, 0.035, size=len(female_vals))
            jitter_m = rng.normal(i + width/2, 0.035, size=len(male_vals))
            ax.scatter(jitter_f, female_vals, color=COLOR_FEMALE, s=7, alpha=0.28, edgecolors="none")
            ax.scatter(jitter_m, male_vals, color=COLOR_MALE, s=7, alpha=0.28, edgecolors="none")

            # Annotate medians above
            med_f = np.median(female_vals)
            med_m = np.median(male_vals)

        # Legend on first panel
        if ax == axes[0, 0]:
            legend_elements = [
                patches.Patch(facecolor=COLOR_FEMALE, alpha=0.75, edgecolor="#903000", label="Female (N = 150)"),
                patches.Patch(facecolor=COLOR_MALE, alpha=0.75, edgecolor="#004080", label="Male (N = 128)"),
            ]
            ax.legend(handles=legend_elements, loc="upper left", frameon=True, framealpha=0.9, fontsize=8.0)

    for ax in axes[1, :]:
        ax.set_xticks(x_indices)
        ax.set_xticklabels([LOAD_LABELS[l] for l in LOAD_ORDER], fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig2_primary_load_profiles.pdf", dpi=300)
    fig.savefig(FIG_DIR / "Fig2_primary_load_profiles.png", dpi=300)
    plt.close(fig)
    print("Generated Fig 2.")

# ==============================================================================
# FIGURE 3: DIRECTIONAL RE-ROUTING (LOCOMOTOR SHEAR VS OBSTETRIC EXPANSION)
# ==============================================================================
def build_figure_3(delta_df: pd.DataFrame, subject_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Panel A: Component differences relative to SP2leg
    ax_a = axes[0]
    ax_a.set_title("A. Changes in absolute translation components", loc="left", fontweight="bold", fontsize=10.0)

    loads = ["SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
    metrics = ["ap_trans_abs_mm", "ml_trans_abs_mm", "cc_trans_abs_mm"]
    labels = ["Anteroposterior (AP)", "Mediolateral (ML)", "Craniocaudal (CC)"]
    colors = [COLOR_AP, COLOR_ML, COLOR_CC]

    x = np.arange(len(loads))
    bar_w = 0.24

    for i, (m, label, c) in enumerate(zip(metrics, labels, colors)):
        sub = delta_df[delta_df["metric"] == m].set_index("load_case").reindex(loads)
        vals = sub["median_delta"].to_numpy(dtype=float)
        yerr_low = vals - sub["ci95_low"].to_numpy(dtype=float)
        yerr_high = sub["ci95_high"].to_numpy(dtype=float) - vals

        bars = ax_a.bar(x + (i - 1) * bar_w, vals, width=bar_w, color=c, alpha=0.88,
                        edgecolor="black", linewidth=0.5, label=label)
        ax_a.errorbar(x + (i - 1) * bar_w, vals, yerr=[yerr_low, yerr_high],
                      fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5)

        # Value labels above/below bars
        for j, val in enumerate(vals):
            pos_y = (sub["ci95_high"].iloc[j] + 0.045 if val >= 0 else sub["ci95_low"].iloc[j] - 0.045)
            ax_a.text(x[j] + (i - 1) * bar_w, pos_y, f"{val:+.2f}",
                      ha="center", va="center", fontsize=6.5, fontweight="bold")

    ax_a.axhline(0, color="black", lw=0.8)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([LOAD_SHORT[l] for l in loads], fontsize=9.0)
    ax_a.set_ylabel("Median within-subject difference vs SP2leg (mm)", fontsize=9.0)
    ax_a.margins(y=0.22)
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.0)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel B: 2D State-Space Contrast of Directional Shifts (ML Expansion vs CC Compression)
    ax_b = axes[1]
    ax_b.set_title("B. Paired component-magnitude contrasts", loc="left", fontweight="bold", fontsize=10.0)

    # Compute within-subject medians for ML and CC deltas
    base_sp2 = subject_df[subject_df["load_case"] == "SP2leg"].sort_values("subject_idx").reset_index(drop=True)

    load_colors = {
        "SP1leg": "#e69f00",
        "LAB_phase1": "#009e73",
        "LAB_phase2": "#0072b2",
        "LAB_phase3": "#cc79a7",
    }

    ax_b.scatter(0, 0, marker="o", s=80, color="black", zorder=5, label="SP2leg (Baseline)")
    ax_b.text(0.01, -0.04, "SP2leg", fontsize=8.5, fontweight="bold")

    for load in loads:
        cur = subject_df[subject_df["load_case"] == load].sort_values("subject_idx").reset_index(drop=True)
        delta_ml = cur["ml_trans_abs_mm"] - base_sp2["ml_trans_abs_mm"]
        delta_cc = cur["cc_trans_abs_mm"] - base_sp2["cc_trans_abs_mm"]

        med_x = float(np.median(delta_ml))
        med_y = float(np.median(delta_cc))

        # Plot individual clouds
        ax_b.scatter(delta_ml, delta_cc, color=load_colors[load], s=8, alpha=0.15, edgecolors="none")

        # Arrow from origin to median
        ax_b.annotate("", xy=(med_x, med_y), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color=load_colors[load], lw=2.0, mutation_scale=14))
        ax_b.scatter(med_x, med_y, color=load_colors[load], s=70, edgecolors="black", lw=1.0, zorder=6, label=LOAD_SHORT[load])
        ax_b.text(med_x + (-0.14 if load == "LAB_phase1" else 0.03), med_y + (0.08 if load == "LAB_phase1" else (0.04 if med_y >= 0 else -0.05)), LOAD_SHORT[load],
                  fontsize=8.5, fontweight="bold", color=load_colors[load])

    ax_b.axhline(0, color="gray", lw=0.6, linestyle=":")
    ax_b.axvline(0, color="gray", lw=0.6, linestyle=":")
    ax_b.set_xlabel("Change in |ML| component (mm)", fontsize=9.0)
    ax_b.set_ylabel("Change in |CC| component (mm)", fontsize=9.0)
    ax_b.grid(True, linestyle="--", alpha=0.3)
    ax_b.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=8.0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig3_directional_rerouting.pdf", dpi=300)
    fig.savefig(FIG_DIR / "Fig3_directional_rerouting.png", dpi=300)
    plt.close(fig)
    print("Generated Fig 3.")

# ==============================================================================
# FIGURE 4: SEXUAL DIMORPHISM UNDER PARTURITION LOADS
# ==============================================================================
def build_figure_4(subject_df: pd.DataFrame) -> None:
    effects = pd.read_csv(TABLE_DIR / "sex_models_lab.csv")
    nested = pd.read_csv(TABLE_DIR / "nested_sex_models.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=True, gridspec_kw={"height_ratios": [.70, 1]})
    for ax, outcome, title, unit in zip(axes[0], ["trans_mag_mm", "rot_mag_deg"],
            ["A. Fully adjusted translation", "B. Fully adjusted rotation"], ["mm", "deg"]):
        sub = effects[effects.outcome == outcome].set_index("load_case").loc[LAB_LOADS]
        beta = sub.beta_female_minus_male.to_numpy()
        ax.errorbar(beta, np.arange(3), xerr=[beta-sub.ci95_low, sub.ci95_high-beta],
                    fmt="o", color=COLOR_ML if unit=="mm" else COLOR_FEMALE, capsize=4)
        ax.axvline(0, color="0.4", ls="--", lw=.8)
        ax.set_yticks(range(3), [LOAD_SHORT[l] for l in LAB_LOADS])
        ax.set_xlabel(f"Female − male ({unit}); M2 estimate and 95% CI")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.margins(x=.3, y=.3)
        for j,q in enumerate(sub.p_fdr_bh):
            ax.text(.98, j, f"q={q:.3g}", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
        ax.grid(axis="x", alpha=.2)
    ax=axes[1,0]
    for j,load in enumerate(LAB_LOADS):
        sub=nested[(nested.load_case==load)&(nested.outcome=='trans_mag_mm')].set_index('model').loc[['M0','M1','M2']]
        ax.plot(range(3),sub.beta,marker='o',label=LOAD_SHORT[load])
    ax.axhline(0,color='0.5',lw=.8)
    ax.set_xticks(range(3),['M0: age','M1: + volume','M2: + triad'])
    ax.set_ylabel('Female − male translation (mm)')
    ax.set_title('C. Nested conditional estimates',loc='left',fontweight='bold')
    ax.legend(frameon=False);ax.grid(alpha=.2)
    ax=axes[1,1];sub=subject_df[subject_df.load_case=='LAB_phase1']
    labels=[]
    for sex,color,name in [(0,COLOR_MALE,'Male'),(1,COLOR_FEMALE,'Female')]:
        ss=sub[sub.sex_F==sex]
        ax.scatter(ss.SubpubicAngle,ss.trans_mag_mm,s=12,alpha=.5,color=color,label=name)
        rho,p=stats.spearmanr(ss.SubpubicAngle,ss.trans_mag_mm)
        labels.append(f'{name}: ρ={rho:.2f}')
    rho,p=stats.spearmanr(sub.SubpubicAngle,sub.trans_mag_mm)
    ax.text(.03,.97,f'Pooled ρ={rho:.2f}\n'+ '\n'.join(labels),transform=ax.transAxes,va='top',fontsize=8)
    ax.set_xlabel('Subpubic angle (degrees)');ax.set_ylabel('LAB1 translation magnitude (mm)')
    ax.set_title('D. Descriptive morphometric association',loc='left',fontweight='bold')
    ax.legend(loc='lower right',frameon=False);ax.grid(alpha=.2)
    for ext in ['pdf','png']: fig.savefig(FIG_DIR/f'Fig4_sex_effects_lab.{ext}',dpi=300)
    plt.close(fig)

# ==============================================================================
# FIGURE 5: VARIANCE CHANNELS (SHAPE VS MATERIAL HETEROGENEITY)
# ==============================================================================
def build_figure_5(variance_df: pd.DataFrame) -> None:
    metrics = ["rot_mag_deg", "trans_mag_mm", "nut_abs_deg", "ap_trans_abs_mm", "ml_trans_abs_mm", "cc_trans_abs_mm"]
    labels = [
        "Rotation magnitude (|θ|)",
        "Translation magnitude (|d|)",
        "Absolute ML rotation (|θ_x|)",
        "AP translation component",
        "ML translation component",
        "CC translation component",
    ]

    # Filter available metrics
    avail_metrics = [m for m in metrics if m in variance_df["metric"].values]
    avail_labels = [labels[i] for i, m in enumerate(metrics) if m in avail_metrics]

    shape_mat = np.vstack([
        variance_df[variance_df["metric"] == m].set_index("load_case").reindex(LOAD_ORDER)["shape_over_full_pct"].to_numpy(dtype=float)
        for m in avail_metrics
    ])
    mat_mat = np.vstack([
        variance_df[variance_df["metric"] == m].set_index("load_case").reindex(LOAD_ORDER)["material_over_full_pct"].to_numpy(dtype=float)
        for m in avail_metrics
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)

    # Panel A: Shape-only / Full variance (%)
    im0 = axes[0].imshow(shape_mat, aspect="auto", cmap="Blues", vmin=min(100.0, float(shape_mat.min())), vmax=max(100.0, float(shape_mat.max())))
    axes[0].set_title("A. Shape-only / Full model variance (%)", loc="left", fontweight="bold", fontsize=10.5)
    axes[0].set_xticks(np.arange(len(LOAD_ORDER)))
    axes[0].set_xticklabels([LOAD_SHORT[l] for l in LOAD_ORDER], fontsize=9.0)
    axes[0].set_yticks(np.arange(len(avail_metrics)))
    axes[0].set_yticklabels(avail_labels, fontsize=8.5)

    # Print exact percentage in each cell
    for r in range(shape_mat.shape[0]):
        for c in range(shape_mat.shape[1]):
            val = shape_mat[r, c]
            axes[0].text(c, r, f"{val:.1f}%", ha="center", va="center",
                         fontsize=8.5, fontweight="bold",
                         color="white" if val > (shape_mat.min()+shape_mat.max())/2 else "black")

    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Variance ratio (%)", fontsize=8.5)

    # Panel B: Material-only / Full variance (%)
    im1 = axes[1].imshow(mat_mat, aspect="auto", cmap="OrRd", vmin=0.0, vmax=float(mat_mat.max()))
    axes[1].set_title("B. Material-only / Full model variance (%)", loc="left", fontweight="bold", fontsize=10.5)
    axes[1].set_xticks(np.arange(len(LOAD_ORDER)))
    axes[1].set_xticklabels([LOAD_SHORT[l] for l in LOAD_ORDER], fontsize=9.0)
    axes[1].set_yticks(np.arange(len(avail_metrics)))
    axes[1].set_yticklabels([])

    # Print exact percentage in each cell
    for r in range(mat_mat.shape[0]):
        for c in range(mat_mat.shape[1]):
            val = mat_mat[r, c]
            axes[1].text(c, r, f"{val:.3f}%", ha="center", va="center",
                         fontsize=8.5, fontweight="bold",
                         color="white" if val > mat_mat.max()/2 else "black")

    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Variance ratio (%)", fontsize=8.5)

    fig.savefig(FIG_DIR / "Fig5_variance_channels.pdf", dpi=300)
    fig.savefig(FIG_DIR / "Fig5_variance_channels.png", dpi=300)
    plt.close(fig)
    print("Generated Fig 5.")

# ==============================================================================
# FIGURE 6: ALLOMETRIC SCALING (LOG-LOG GRID WITH REGRESSION METRICS)
# ==============================================================================
def build_supplementary_scatter(subject_df: pd.DataFrame) -> None:
    allom_df = pd.read_csv(TABLE_DIR / "allometry_models.csv")

    fig, axes = plt.subplots(len(LOAD_ORDER), 2, figsize=(8.5, 11.5), sharex=True)

    outcomes = ["rot_mag_deg", "trans_mag_mm"]
    y_labels = ["Rotation magnitude (|θ|, deg)", "Translation magnitude (|d|, mm)"]

    for i_row, (outcome, ylabel) in enumerate(zip(outcomes, y_labels)):
        for i_col, load in enumerate(LOAD_ORDER):
            ax = axes[i_col, i_row]
            sub = subject_df[subject_df["load_case"] == load].copy()

            # Retrieve model statistics
            row_match = allom_df[(allom_df["load_case"] == load) & (allom_df["outcome"] == outcome)]
            beta_m = row_match["male_exponent"].values[0] if len(row_match) > 0 else 0.0
            ci_m_low = row_match["male_ci95_low"].values[0] if len(row_match) > 0 else 0.0
            ci_m_high = row_match["male_ci95_high"].values[0] if len(row_match) > 0 else 0.0

            beta_f = row_match["female_exponent"].values[0] if len(row_match) > 0 else 0.0
            ci_f_low = row_match["female_ci95_low"].values[0] if len(row_match) > 0 else 0.0
            ci_f_high = row_match["female_ci95_high"].values[0] if len(row_match) > 0 else 0.0

            r2_val = row_match["adj_r2"].values[0] if len(row_match) > 0 else 0.0

            model = smf.ols(f"np.log({outcome}) ~ log_total_volume * sex_F + age_z + AP_z + BiischiadicWidth_z + SubpubicAngle_z", data=sub).fit(cov_type="HC3")
            np.testing.assert_allclose([model.params['log_total_volume'], model.params['log_total_volume'] + model.params['log_total_volume:sex_F']], [beta_m, beta_f], atol=1e-9)

            # Scatter
            ax.scatter(sub[sub["sex_F"] == 0.0]["total_volume"], sub[sub["sex_F"] == 0.0][outcome],
                       color=COLOR_MALE, s=11, alpha=0.35, edgecolors="none", label="Male" if (i_row==0 and i_col==0) else None)
            ax.scatter(sub[sub["sex_F"] == 1.0]["total_volume"], sub[sub["sex_F"] == 1.0][outcome],
                       color=COLOR_FEMALE, s=11, alpha=0.35, edgecolors="none", label="Female" if (i_row==0 and i_col==0) else None)

            # Conditional geometric means from the same adjusted model as Table 4B.
            for sex, color in [(0, COLOR_MALE), (1, COLOR_FEMALE)]:
                ss=sub[sub.sex_F==sex]
                x_line=np.geomspace(ss.total_volume.min(),ss.total_volume.max(),100)
                design=pd.DataFrame(dict(log_total_volume=np.log(x_line),sex_F=sex,
                    age_z=0.,AP_z=0.,BiischiadicWidth_z=0.,SubpubicAngle_z=0.))
                pred=model.get_prediction(design).summary_frame()
                ax.plot(x_line,np.exp(pred['mean']),color=color,lw=1.6)
                ax.fill_between(x_line,np.exp(pred.mean_ci_lower),np.exp(pred.mean_ci_upper),color=color,alpha=.13)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.25)

            # Titles and labels
            ax.set_title(LOAD_SHORT[load] + (" — rotation" if i_row == 0 else " — translation"), fontweight="bold", fontsize=10.0)
            ax.set_ylabel(ylabel, fontsize=9.0)
            if i_col == len(LOAD_ORDER)-1:
                ax.set_xlabel("Pelvic bone volume (cm³)", fontsize=8.5)

            # Highlight SP1leg column
            if load == "SP1leg":
                ax.set_facecolor("#fffcf5")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=2, frameon=True, framealpha=0.9, fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "SuppFig1_allometry_scatter.pdf", dpi=300)
    fig.savefig(FIG_DIR / "SuppFig1_allometry_scatter.png", dpi=300)
    plt.close(fig)
    print("Generated Fig 6.")

def build_figure_6(subject_df):
    build_supplementary_scatter(subject_df)
    data=pd.read_csv(TABLE_DIR/'allometry_models.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),constrained_layout=True)
    for ax,outcome,title in zip(axes,['rot_mag_deg','trans_mag_mm'],['A. Rotation','B. Translation']):
        d=data[data.outcome==outcome].set_index('load_case').loc[LOAD_ORDER]
        for sex,color,offset in [('male',COLOR_MALE,-.12),('female',COLOR_FEMALE,.12)]:
            beta=d[sex+'_exponent'].to_numpy();lo=d[sex+'_ci95_low'].to_numpy();hi=d[sex+'_ci95_high'].to_numpy()
            y=np.arange(5)+offset
            ax.errorbar(beta,y,xerr=[beta-lo,hi-beta],fmt='o',capsize=3,color=color,label=sex.capitalize())
            for yi,q in zip(y,d[sex+'_q']):
                ax.text(1.01,yi,f'q={q:.3g}',transform=ax.get_yaxis_transform(),va='center',fontsize=8,color=color)
        ax.set_yticks(range(5),[LOAD_SHORT[l] for l in LOAD_ORDER]);ax.invert_yaxis()
        ax.axvline(0,color='.3',ls='--',lw=.8);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Conditional volume exponent (95% CI)')
        ax.set_title(title,loc='left',fontweight='bold');ax.legend(frameon=False,loc='best')
    for ext in ['pdf','png']:fig.savefig(FIG_DIR/f'Fig6_allometry_loglog.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def main():
    print("Loading data...")
    subject_df, delta_df, variance_df = load_data()

    print("Building Fig 1...")
    build_figure_1()

    print("Building Fig 2...")
    build_figure_2(subject_df)

    print("Building Fig 3...")
    build_figure_3(delta_df, subject_df)

    print("Building Fig 4...")
    build_figure_4(subject_df)

    print("Building Fig 5...")
    build_figure_5(variance_df)

    print("Building Fig 6...")
    build_figure_6(subject_df)

    print("All figures successfully generated.")

if __name__ == "__main__":
    main()
