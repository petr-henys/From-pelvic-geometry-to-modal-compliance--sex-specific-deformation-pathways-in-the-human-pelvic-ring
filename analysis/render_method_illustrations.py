"""Render two reproducible Methods figures for BMMB using Basix and PyVista.

One figure shows the actual degree-4 tetrahedron rule in pelvic-mesh context;
the other shows representative degree-<=2 dictionary fields after exact
volume-weighted removal of their infinitesimal rigid-motion components.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import basix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pyvista as pv

from analysis.global_deformation_patterns import rigid_design
from analysis.publication_rendering import (
    add_deformed_scalar_mesh,
    add_ghost_reference_mesh,
    configure_publication_camera,
    create_publication_plotter,
    crop_image_whitespace,
)
from analysis.spectral_data import load_reference_space

OUT = ROOT / 'analysis_outputs/method_illustrations'
MANUSCRIPT = ROOT / 'manuscripts/bmmb'
BLUE = '#2b7bba'
ORANGE = '#e69f00'
GREEN = '#009e73'
ROSE = '#cc79a7'
DARK = '#172638'
MUTED = '#536274'


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=350, facecolor='white')
    fig.savefig(MANUSCRIPT / f'{stem}.pdf', dpi=350, facecolor='white')
    plt.close(fig)


def reference_mesh():
    parser, V = load_reference_space()
    mesh = parser.mesh_dolfinx
    points = mesh.geometry.x.copy()
    cells = mesh.geometry.dofmap.copy()
    if cells.shape[1] != 4:
        raise ValueError('Expected linear tetrahedra')
    if not np.allclose(V.tabulate_dof_coordinates(), points, atol=1e-9):
        raise ValueError('FE geometry and displacement DOFs have different order')
    grid = pv.UnstructuredGrid(
        np.column_stack([np.full(len(cells), 4), cells]).ravel(),
        np.full(len(cells), pv.CellType.TETRA), points,
    )
    grid.point_data['node_id'] = np.arange(len(points))
    surface = grid.extract_surface(algorithm='dataset_surface')
    vertices = points[cells]
    edges = np.stack([vertices[:, j] - vertices[:, 0] for j in (1, 2, 3)], axis=-1)
    volumes = np.abs(np.linalg.det(edges)) / 6.0
    center = np.einsum('n,nj->j', volumes, vertices.mean(axis=1)) / volumes.sum()
    mean_sq = np.einsum(
        'n,n->', volumes,
        (np.sum(vertices.sum(axis=1)**2, axis=1)
         + np.sum(vertices**2, axis=(1, 2))) / 20.0,
    ) / volumes.sum()
    radius = np.sqrt(mean_sq - center @ center)
    return points, cells, surface, volumes, center, radius


def rule():
    q, w = basix.make_quadrature(basix.CellType.tetrahedron, 4)
    if len(w) != 14 or np.any(w <= 0) or not np.isclose(w.sum(), 1/6):
        raise ValueError('Unexpected degree-4 tetrahedral rule')
    bary = np.column_stack([1.0 - q.sum(axis=1), q])
    return q, w, bary


def pelvis_quadrature_view(points, cells, surface, bary):
    # Show actual quadrature points in a systematic subset of reference cells.
    subset = cells[::80]
    samples = np.einsum('qv,cvd->cqd', bary, points[subset]).reshape(-1, 3)
    plotter = create_publication_plotter(
        window_size=(1000, 950), enable_ssaa=True, enable_depth_peeling=True,
    )
    plotter.add_mesh(surface, color='#c7d2df', opacity=0.20,
                     smooth_shading=True, show_edges=False)
    plotter.add_mesh(pv.PolyData(samples), color=BLUE, point_size=3.2,
                     render_points_as_spheres=True, opacity=0.75)
    center = (points.min(axis=0)+points.max(axis=0))/2
    configure_publication_camera(plotter, center, view='OBLIQUE', parallel_scale=195)
    image = crop_image_whitespace(plotter.screenshot(return_img=True), pad=12)
    plotter.close()
    return image, len(subset), len(samples)


def tetra_quadrature_view(q, w):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    tetra = pv.UnstructuredGrid(np.array([4, 0, 1, 2, 3]),
                                np.array([pv.CellType.TETRA]), vertices)
    plotter = create_publication_plotter(
        window_size=(1000, 950), enable_ssaa=True, enable_depth_peeling=True,
    )
    plotter.add_mesh(tetra.extract_surface(algorithm='dataset_surface'), color='#dde6f0', opacity=0.18,
                     show_edges=True, edge_color='#536579', line_width=3)
    groups = [np.where(np.isclose(w, value))[0] for value in np.unique(np.round(w, 12))]
    colors = (BLUE, ORANGE, GREEN)
    radii = (0.028, 0.039, 0.047)
    for ids, color, radius in zip(groups, colors, radii):
        glyph = pv.PolyData(q[ids]).glyph(geom=pv.Sphere(radius=radius,
                                                          theta_resolution=20,
                                                          phi_resolution=20),
                                        scale=False, orient=False)
        plotter.add_mesh(glyph, color=color, smooth_shading=True)
    plotter.camera_position = [(2.2, 1.9, 1.8), (0.28, 0.28, 0.28), (0, 0, 1)]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = 0.72
    image = crop_image_whitespace(plotter.screenshot(return_img=True), pad=12)
    plotter.close()
    return image, groups


def draw_quadrature(points, cells, surface):
    q, w, bary = rule()
    pelvis, n_cells_shown, n_points_shown = pelvis_quadrature_view(points, cells, surface, bary)
    tetra, groups = tetra_quadrature_view(q, w)
    fig = plt.figure(figsize=(7.48, 2.95), dpi=350)
    gs = fig.add_gridspec(1, 3, left=.025, right=.985, top=.82, bottom=.08,
                          wspace=.12, width_ratios=[1.0, 1.0, 1.12])
    ax0, ax1, ax2 = [fig.add_subplot(gs[0, j]) for j in range(3)]
    for ax, img in [(ax0, pelvis), (ax1, tetra)]:
        ax.imshow(img)
        ax.set_anchor('N')
        ax.set_axis_off()
    for ax, letter, title in [(ax0, 'A', 'Pelvic reference mesh'),
                              (ax1, 'B', 'One tetrahedral element'),
                              (ax2, 'C', 'Polynomial products')]:
        ax.text(.01, 1.025, letter, transform=ax.transAxes, fontsize=11,
                fontweight='bold', color=DARK, va='bottom')
        ax.text(.13, 1.025, title, transform=ax.transAxes, fontsize=8.6,
                fontweight='bold', color=DARK, va='bottom')
    ax0.text(.5, -.015, f'{n_cells_shown:,} of {len(cells):,} cells shown',
             transform=ax0.transAxes, ha='center', va='top', fontsize=7.8, color=MUTED)
    handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=5.5,
                          color=c, label=f'{len(ids)} points')
               for ids, c in zip(groups, (BLUE, ORANGE, GREEN))]
    ax1.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.115),
               frameon=False, ncol=3, fontsize=7.3, handletextpad=.2,
               columnspacing=.5)
    ax2.set_box_aspect(.82)
    ax2.set_anchor('N')
    ax2.set_xlim(0, 4.65)
    ax2.set_ylim(-.65, 3.45)
    ax2.set_yticks([2.6, 1.5, .4],
                   [r'$\mathbf{u}_h\cdot\mathbf{u}_h$',
                    r'$\mathbf{u}_h\cdot\mathbf{p}_2$',
                    r'$\mathbf{p}_2\cdot\mathbf{p}_2$'], fontsize=8.4)
    ax2.set_xticks([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], fontsize=8)
    ax2.set_xlabel('Polynomial degree on each element', fontsize=8.5, color=DARK)
    ax2.set_axisbelow(True)
    ax2.grid(axis='x', color='#e5eaf0', linewidth=.7)
    ax2.barh([2.6, 1.5, .4], [2, 3, 4], height=.36,
             color=['#aebcca', BLUE, GREEN], edgecolor='none')
    for y, d in [(2.6, 2), (1.5, 3), (.4, 4)]:
        ax2.text(d-.12, y, rf'$\mathcal{{P}}_{d}$', ha='right', va='center',
                 fontsize=8.5, color='white', fontweight='bold')
    ax2.axvline(4.12, color=ORANGE, lw=1.5, ls='--')
    ax2.text(4.1, 3.31, '14-point rule\nexact through degree 4',
             color='#9b6200', ha='right', va='top', fontsize=7.5)
    ax2.spines[['top','right','left']].set_visible(False)
    ax2.tick_params(axis='y', length=0)
    ax2.tick_params(axis='x', length=0, colors=MUTED)
    fig.text(.5, .975, 'Volume integration of modal and polynomial fields',
             ha='center', va='top', fontsize=11.5, weight='bold', color=DARK)
    save(fig, 'quadrature_method')
    return dict(n_elements=len(cells), n_quadrature_points=len(cells)*len(w),
                displayed_elements=n_cells_shown, displayed_points=n_points_shown,
                normalized_rule_weights=(6*w).tolist())


def raw_field(name, x):
    xx, yy, zz = x.T
    if name == 'axial':
        return np.column_stack([xx, np.zeros_like(xx), np.zeros_like(xx)])
    if name == 'bending':
        return np.column_stack([-xx*yy, .5*xx**2, np.zeros_like(xx)])
    if name == 'shear':
        return np.column_stack([yy, xx, np.zeros_like(xx)])
    if name == 'torsion':
        return np.column_stack([np.zeros_like(xx), -xx*zz, xx*yy])
    raise ValueError(name)


def remove_rigid(name, xnode, xquad, weights):
    gram = np.zeros((6, 6))
    rhs = np.zeros(6)
    for start in range(0, len(xquad), 200_000):
        xx=xquad[start:start+200_000]
        ww=weights[start:start+200_000]
        R=rigid_design(xx)
        wr=np.repeat(ww,3)
        gram += R.T @ (R*wr[:,None])
        rhs += R.T @ (raw_field(name,xx).ravel()*wr)
    coeff=np.linalg.solve(gram,rhs)
    if np.linalg.norm(rhs-gram@coeff)>1e-10*max(1.0,np.linalg.norm(rhs)):
        raise RuntimeError(f'Rigid projection failed for {name}')
    field=raw_field(name,xnode)- (rigid_design(xnode)@coeff).reshape(-1,3)
    return field, coeff


def rendered_family(surface, points, u, view):
    plotter=create_publication_plotter(window_size=(850,750),
                                       enable_ssaa=True,enable_depth_peeling=True)
    ids=surface.point_data['node_id']
    amplitude=np.linalg.norm(u,axis=1)
    displayed=50*u/amplitude.max()
    shape=surface.copy()
    shape.points=surface.points+displayed[ids]
    shape=shape.compute_normals(point_normals=True,feature_angle=60.)
    add_ghost_reference_mesh(plotter,surface,color='#b8c0cc',opacity=.28)
    add_deformed_scalar_mesh(plotter,shape,scalars=amplitude[ids]/amplitude.max(),
                             cmap='viridis',clim=(0,1))
    center=(points.min(axis=0)+points.max(axis=0))/2
    configure_publication_camera(plotter,center,view=view,distance=700.,
                                 parallel_scale=195.)
    image=crop_image_whitespace(plotter.screenshot(return_img=True),pad=8)
    plotter.close()
    return image


def draw_families(points,cells,surface,volumes,center,radius):
    _,w,bary=rule()
    vertices=points[cells]
    qpoints=np.einsum('qv,cvd->cqd',bary,vertices).reshape(-1,3)
    qweights=(volumes[:,None]*(6*w)).ravel()
    if not np.isclose(qweights.sum(),volumes.sum(),rtol=1e-12):
        raise RuntimeError('Quadrature weights do not reproduce mesh volume')
    xnode=(points-center)/radius
    xquad=(qpoints-center)/radius
    fig=plt.figure(figsize=(7.48,5.25),dpi=350)
    gs=fig.add_gridspec(2,2,left=.018,right=.982,top=.925,bottom=.067,
                        wspace=.075,hspace=.07)
    data=[('axial','Axial',r'$(\tilde{x},0,0)$',BLUE,'Normal extension'),
          ('bending','Bending',r'$(-\tilde{x}\tilde{y},\;\frac{1}{2}\tilde{x}^2,\;0)$',ORANGE,'Shear-free flexure'),
          ('shear','Shear',r'$(\tilde{y},\tilde{x},0)$',GREEN,'Transverse shear'),
          ('torsion','Torsion',r'$(0,-\tilde{x}\tilde{z},\tilde{x}\tilde{y})$',ROSE,'Twist gradient')]
    coeffs={}
    for j,(key,title,formula,color,signature) in enumerate(data):
        print('Rendering dictionary family',title,flush=True)
        u,coeff=remove_rigid(key,xnode,xquad,qweights)
        coeffs[key]=coeff.tolist()
        container=fig.add_subplot(gs[j//2,j%2])
        container.set_axis_off()
        card=FancyBboxPatch((0,0),1,1,boxstyle='round,pad=.005,rounding_size=.018',
                            transform=container.transAxes,facecolor='white',
                            edgecolor='#d1d9e3',lw=.8)
        container.add_patch(card)
        container.text(.035,.95,chr(ord('A')+j),fontsize=10.5,weight='bold',
                       color=color,va='top',transform=container.transAxes)
        container.text(.11,.95,f'{title}   ·   {signature}',fontsize=9.1,
                       weight='bold',color=DARK,va='top',transform=container.transAxes)
        inner=gs[j//2,j%2].subgridspec(3,2,height_ratios=[.36,3.2,.42],
                                       hspace=.00,wspace=.0)
        for col,view in enumerate(('AP','CC')):
            ax=fig.add_subplot(inner[1,col])
            ax.imshow(rendered_family(surface,points,u,view))
            ax.set_axis_off()
            ax.text(.04,.04,'Anterior' if view=='AP' else 'Cranial',
                    fontsize=7.4,color=DARK,transform=ax.transAxes,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.82,'pad':2})
        label=fig.add_subplot(inner[2,:])
        label.set_axis_off()
        label.text(.5,.52,formula,ha='center',va='center',fontsize=9.7,color=DARK,
                   transform=label.transAxes)
    fig.text(.5,.988,'Representative fields in the 3D polynomial dictionary',
             ha='center',va='top',fontsize=11.5,weight='bold',color=DARK)
    fig.text(.5,.015,'Ghost: undeformed reference  ·  Colour: normalized displacement magnitude  ·  Each field scaled to 50 mm for display',
             ha='center',va='bottom',fontsize=7.5,color=MUTED)
    save(fig,'deformation_families')
    return dict(rigid_projection_coefficients=coeffs,
                display_max_displacement_mm=50,
                example_fields=[x[0] for x in data])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    points,cells,surface,volumes,center,radius=reference_mesh()
    quadrature=draw_quadrature(points,cells,surface)
    families=draw_families(points,cells,surface,volumes,center,radius)
    (OUT/'provenance.json').write_text(json.dumps(dict(
        source='analysis/render_method_illustrations.py',
        mesh='results/ref_S1P_fixed_new2/simulation_metadata.json',
        quadrature=quadrature,
        families=families,
        figure_note='Displayed point cloud is a systematic subset of cells; all integrations use every cell.'
    ),indent=2)+'\n')
    print('Saved method illustrations',flush=True)

if __name__=='__main__':
    main()
