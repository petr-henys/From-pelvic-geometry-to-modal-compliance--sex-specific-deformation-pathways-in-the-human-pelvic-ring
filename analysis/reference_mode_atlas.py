"""Reference-only whole-pelvis global deformation pattern atlas; no anatomical subdivision."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import pyvista as pv
import basix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
from scipy.spatial import cKDTree
from analysis.spectral_data import load_reference_space
from analysis.global_deformation_patterns import GlobalDeformationModel, PATTERNS
MECHANISMS = (*PATTERNS, "residual")
OUT = ROOT / 'analysis_outputs/mode_mechanisms/reference'
FIG = ROOT / 'analysis_outputs/plos_revision/figures'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    parser, V = load_reference_space()
    mesh = parser.mesh_dolfinx
    coords = mesh.geometry.x.copy()
    cells = mesh.geometry.dofmap.copy()
    if cells.shape[1] != 4:
        raise ValueError('Linear tetrahedra required')
    distance, mapping = cKDTree(V.tabulate_dof_coordinates()).query(coords)
    if distance.max() > 1e-9 or len(np.unique(mapping)) != len(coords):
        raise ValueError('Geometry-to-DOF mapping is not bijective')
    source = ROOT / 'results/ref_S1P_fixed_new2/reference_solution.npz'
    modes = np.load(source)['eigenvectors'][:, mapping]
    vertices = coords[cells]
    edges = np.stack([vertices[:, j] - vertices[:, 0] for j in (1, 2, 3)], axis=-1)
    volumes = np.abs(np.linalg.det(edges)) / 6
    if np.any(volumes <= 0):
        raise ValueError('Degenerate tetrahedra')

    # Positive degree-4 tetrahedral quadrature rule (14 points per tetrahedron)
    # Required for exact integration of u_h * p_2 (degree 3) and p_2 * p_2 (degree 4).
    pts, wts = basix.make_quadrature(basix.CellType.tetrahedron, 4)
    bary = np.column_stack([1.0 - pts.sum(axis=1), pts[:, 0], pts[:, 1], pts[:, 2]])
    qweights = 6.0 * wts  # normalized so sum(qweights) == 1.0 per unit tet volume
    points = np.einsum('qv,cvd->cqd', bary, vertices).reshape(-1, 3)
    weights = (volumes[:, None] * qweights).ravel()

    model = GlobalDeformationModel(points, weights)
    profiles = []
    records = []
    for k, u in enumerate(modes):
        uq = np.einsum('qv,cvd->cqd', bary, u[cells]).reshape(-1, 3)
        result = model.fit(uq)
        if not result['is_valid']:
            raise ValueError(f'No nonrigid deformation in mode {k+1}')
        row = dict(mode=k+1, **result)
        records.append(row)
        profiles.append(row)

    pd.DataFrame(records).to_csv(OUT / 'global_deformation_profiles.csv', index=False)

    grid = pv.UnstructuredGrid(
        np.column_stack([np.full(len(cells), 4), cells]).ravel(),
        np.full(len(cells), pv.CellType.TETRA),
        coords
    )
    grid.point_data['node_id'] = np.arange(len(coords))
    from analysis.publication_rendering import (
        create_publication_plotter,
        add_ghost_reference_mesh,
        add_deformed_scalar_mesh,
        configure_publication_camera,
        crop_image_whitespace,
    )

    surface = grid.extract_surface(algorithm='dataset_surface').compute_normals(point_normals=True, feature_angle=60.0)
    ids = surface.point_data['node_id']
    center = (coords.min(axis=0) + coords.max(axis=0)) / 2

    def render(u, view):
        p = create_publication_plotter(
            window_size=(1050, 900),
            enable_ssaa=True,
            enable_depth_peeling=True,
            enable_ssao=False,
        )
        norm = np.linalg.norm(u, axis=1)
        shape = surface.copy()
        shape.points = surface.points + 50.0 * u[ids] / norm.max()
        shape = shape.compute_normals(point_normals=True, feature_angle=60.0)
        
        add_ghost_reference_mesh(p, surface, color='#b8c0cc', opacity=0.28)
        add_deformed_scalar_mesh(p, shape, scalars=norm[ids] / norm.max(), cmap='viridis', clim=(0, 1))
        configure_publication_camera(p, center, view=view, distance=700.0, parallel_scale=195.0)
        img = p.screenshot(return_img=True)
        p.close()
        return crop_image_whitespace(img, pad=8)

    colors = {
        'axial': '#2b7bba',      # rich distinct blue
        'bending': '#e69f00',    # warm amber
        'shear': '#009e73',      # emerald green
        'torsion': '#cc79a7',    # rose purple
        'residual': '#dcdcdc'    # light neutral grey
    }
    color_contrast = {
        'axial': 'white',
        'bending': '#1a1a1a',
        'shear': 'white',
        'torsion': '#1a1a1a',
        'residual': '#333333'
    }

    # Figure dimensions matching journal full page (190 mm x 242 mm)
    fig = plt.figure(figsize=(190 / 25.4, 242 / 25.4), dpi=350)
    outer = fig.add_gridspec(5, 3, left=0.022, right=0.978, top=0.908, bottom=0.012, hspace=0.22, wspace=0.09)

    for k, (u, profile) in enumerate(zip(modes, profiles)):
        print(f'Rendering global mode {k+1}', flush=True)
        row = k // 3
        col = k % 3
        cell_spec = outer[row, col]

        # Crisp card boundary with seamless pure white background
        card_ax = fig.add_subplot(cell_spec)
        card_ax.axis('off')
        card = FancyBboxPatch(
            (0.00, 0.00), 1.00, 1.00,
            boxstyle="round,pad=0.010,rounding_size=0.025",
            facecolor='#ffffff',
            edgecolor='#d2d7df',
            linewidth=0.85,
            transform=card_ax.transAxes,
            zorder=0
        )
        card_ax.add_patch(card)

        # Subgridspec inside cell:
        # row 0: Header (Mode title + Fitted % badge)
        # row 1: Meshes (Anterior on left, Cranial on right)
        # row 2: Stacked bar chart + metrics
        sub_grid = cell_spec.subgridspec(3, 2, height_ratios=[0.32, 2.30, 0.72], hspace=0.05, wspace=0.03)

        # 1. Mode header
        header_ax = fig.add_subplot(sub_grid[0, :])
        header_ax.axis('off')
        header_ax.text(0.02, 0.5, f"Mode {k+1}", fontsize=10.5, fontweight='bold', color='#111827', va='center', ha='left')
        
        fitted_val = 100 * (1.0 - profile['f_residual'])
        header_ax.text(0.98, 0.5, f"Fitted: {fitted_val:.1f}%", fontsize=8.5, fontweight='bold', color='#1f2937', va='center', ha='right')

        # 2. Meshes (Anterior on left, Cranial on right)
        ax_ap = fig.add_subplot(sub_grid[1, 0])
        ax_ap.imshow(render(u, 'AP'))
        ax_ap.axis('off')

        ax_cc = fig.add_subplot(sub_grid[1, 1])
        ax_cc.imshow(render(u, 'CC'))
        ax_cc.axis('off')

        # 3. Stacked bar chart
        bar_ax = fig.add_subplot(sub_grid[2, :])
        bar_ax.axis('off')
        
        left = 0
        for mech in (*PATTERNS, 'residual'):
            val = 100 * profile['f_' + mech]
            color = colors[mech]
            bar_ax.barh(0, val, left=left, color=color, height=0.50, edgecolor='white', linewidth=1.0, zorder=2)
            if val >= 11:
                tc = color_contrast[mech]
                bar_ax.text(left + val / 2, 0, f"{val:.0f}%", ha='center', va='center', fontsize=7.8, fontweight='bold', color=tc, zorder=3)
            left += val
        
        bar_ax.set_xlim(0, 100)
        bar_ax.set_ylim(-0.55, 0.45)
        
        # Subtitle below bar: Residual & nonrigid norm fraction
        res_val = 100 * profile['f_residual']
        nonrigid_val = 100 * profile['nonrigid_fraction_of_total']
        summary_txt = f"Residual: {res_val:.1f}%  ·  Nonrigid norm: {nonrigid_val:.1f}%"
        bar_ax.text(50, -0.40, summary_txt, ha='center', va='center', fontsize=7.2, color='#4b5563', fontweight='normal')

    # Figure Header
    # 1. Main Title
    fig.text(0.5, 0.985, 'Whole-pelvis 3D canonical deformation patterns', fontsize=12.5, fontweight='bold', ha='center', va='top', color='#111827')

    # 2. Legend for deformation families
    legend_patches = [
        Patch(facecolor=colors['axial'], edgecolor='none', label='Axial'),
        Patch(facecolor=colors['bending'], edgecolor='none', label='Bending'),
        Patch(facecolor=colors['shear'], edgecolor='none', label='Shear'),
        Patch(facecolor=colors['torsion'], edgecolor='none', label='Torsion'),
        Patch(facecolor=colors['residual'], edgecolor='#b5b5b5', linewidth=0.5, label='Residual')
    ]
    fig.legend(
        handles=legend_patches,
        loc='upper center',
        bbox_to_anchor=(0.36, 0.963),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        handlelength=1.1,
        handleheight=0.7,
        columnspacing=1.0
    )

    # 3. Small horizontal viridis colorbar indicating displacement magnitude
    cb_ax = fig.add_axes([0.72, 0.945, 0.23, 0.009])
    norm_dummy = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb = matplotlib.colorbar.ColorbarBase(
        cb_ax,
        cmap='viridis',
        norm=norm_dummy,
        orientation='horizontal'
    )
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['0', '', 'max'])
    cb.ax.tick_params(labelsize=6.8, length=2, pad=1, color='#4b5563')
    cb_ax.set_title('Displacement amplitude |u|', fontsize=7.2, pad=2.5, color='#374151', fontweight='medium')

    # 4. View explanation subtitle
    fig.text(0.5, 0.925, 'Each panel: Anterior view (left)  ·  Cranial view (right)  ·  Bars: Shapley nonrigid shape-norm partition', fontsize=7.8, color='#4b5563', ha='center', va='center')

    for ext in ['pdf', 'png']:
        fig.savefig(FIG / f'reference_mode_atlas.{ext}', dpi=350)
    plt.close(fig)

    (OUT / 'global_deformation_provenance.json').write_text(json.dumps(dict(
        source=str(source.relative_to(ROOT)),
        integration='Positive 14-point (degree-4) tetrahedral quadrature, exact for degree-4 polynomials',
        measure='Squared nonrigid displacement after rigid projection; not strain or energy',
        allocation='Shapley contributions of four 3D canonical pattern families (axial, bending, shear, torsion)',
        n_modes=len(modes)
    ), indent=2))


if __name__ == '__main__':
    main()

