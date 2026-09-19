"""Export LaTeX and CSV tables for the global deformation pattern decomposition appendix.

Outputs:
1. analysis_outputs/tables/global_deformation_modes_table.tex (and .csv)
   Complete 15-mode quantitative profile table with nonrigid fraction, Shapley shares,
   residual, total fitted %, and standalone R^2 values.
2. analysis_outputs/tables/global_deformation_synthetic_table.tex (and .csv)
   Synthetic benchmark table validating 100% recovery on canonical fields,
   rigid-body invariance, and backwards identification on both symmetric domains
   and irregular pelvic geometry.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import basix
import pandas as pd
from analysis.spectral_data import load_reference_space
from analysis.global_deformation_patterns import GlobalDeformationModel

TABLES_DIR = ROOT / 'analysis_outputs/tables'


def generate_modes_table():
    profiles_csv = ROOT / 'analysis_outputs/mode_mechanisms/reference/global_deformation_profiles.csv'
    if not profiles_csv.exists():
        raise FileNotFoundError(f'Missing {profiles_csv}')

    df = pd.read_csv(profiles_csv)
    
    # Format CSV for export
    out_df = pd.DataFrame({
        'Mode': df['mode'].astype(int),
        'Nonrigid (%)': (df['nonrigid_fraction_of_total'] * 100).round(1),
        'Fitted (%)': ((1.0 - df['f_residual']) * 100).round(1),
        'Axial (%)': (df['f_axial'] * 100).round(1),
        'Bending (%)': (df['f_bending'] * 100).round(1),
        'Shear (%)': (df['f_shear'] * 100).round(1),
        'Torsion (%)': (df['f_torsion'] * 100).round(1),
        'Residual (%)': (df['f_residual'] * 100).round(1),
        'R2_Axial (%)': (df['standalone_axial'] * 100).round(1),
        'R2_Bending (%)': (df['standalone_bending'] * 100).round(1),
        'R2_Shear (%)': (df['standalone_shear'] * 100).round(1),
        'R2_Torsion (%)': (df['standalone_torsion'] * 100).round(1),
    })
    out_df.to_csv(TABLES_DIR / 'global_deformation_modes_table.csv', index=False)

    # Format LaTeX table
    lines = [
        r'\begin{tabular}{rrrrrrrrrrrr}',
        r'\toprule',
        r' & & & \multicolumn{5}{c}{Shapley norm partition (\%)} & \multicolumn{4}{c}{Standalone $R^2$ (\%)} \\',
        r'\cmidrule(lr){4-8} \cmidrule(lr){9-12}',
        r'Mode & Nonrigid & Fitted & Axial & Bending & Shear & Torsion & Residual & Axial & Bending & Shear & Torsion \\',
        r'\midrule'
    ]

    for _, r in out_df.iterrows():
        mode = int(r['Mode'])
        nonrigid = f"{r['Nonrigid (%)']:.1f}"
        fitted = f"{r['Fitted (%)']:.1f}"
        ax = f"{r['Axial (%)']:.1f}"
        bend = f"{r['Bending (%)']:.1f}"
        shear = f"{r['Shear (%)']:.1f}"
        tor = f"{r['Torsion (%)']:.1f}"
        res = f"{r['Residual (%)']:.1f}"
        r2_ax = f"{r['R2_Axial (%)']:.1f}"
        r2_bend = f"{r['R2_Bending (%)']:.1f}"
        r2_sh = f"{r['R2_Shear (%)']:.1f}"
        r2_tor = f"{r['R2_Torsion (%)']:.1f}"
        lines.append(f"{mode} & {nonrigid} & {fitted} & {ax} & {bend} & {shear} & {tor} & {res} & {r2_ax} & {r2_bend} & {r2_sh} & {r2_tor} \\\\")

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        ''
    ])

    tex_path = TABLES_DIR / 'global_deformation_modes_table.tex'
    tex_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Generated {tex_path}")


def generate_synthetic_table():
    # 1. Symmetric domain tests
    x_1d = np.linspace(-1, 1, 15)
    X, Y, Z = np.meshgrid(x_1d, x_1d, x_1d, indexing='ij')
    pts_sym = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    w_sym = np.ones(len(pts_sym))
    mod_sym = GlobalDeformationModel(pts_sym, w_sym)
    xs, ys, zs = pts_sym.T

    # 2. Reference pelvis tests
    parser, V = load_reference_space()
    mesh = parser.mesh_dolfinx
    coords = mesh.geometry.x.copy()
    cells = mesh.geometry.dofmap.copy()
    vertices = coords[cells]
    edges = np.stack([vertices[:, j] - vertices[:, 0] for j in (1, 2, 3)], axis=-1)
    volumes = np.abs(np.linalg.det(edges)) / 6
    pts, wts = basix.make_quadrature(basix.CellType.tetrahedron, 4)
    bary = np.column_stack([1.0 - pts.sum(axis=1), pts[:, 0], pts[:, 1], pts[:, 2]])
    qweights = 6.0 * wts
    pts_pelv = np.einsum('qv,cvd->cqd', bary, vertices).reshape(-1, 3)
    w_pelv = (volumes[:, None] * qweights).ravel()
    mod_pelv = GlobalDeformationModel(pts_pelv, w_pelv)
    c = mod_pelv.center
    rg = mod_pelv.length
    xp, yp, zp = ((pts_pelv - c) / rg).T

    cases = [
        # (Domain, Name, Field, Expected primary)
        ('Symmetric cube', 'Pure axial stretch ($x\\mathbf{e}_x$)', np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)]), mod_sym),
        ('Symmetric cube', 'Pure bending ($-xy\\mathbf{e}_x + \\frac{1}{2}x^2\\mathbf{e}_y$)', np.column_stack([-xs*ys, 0.5*xs**2, np.zeros_like(xs)]), mod_sym),
        ('Symmetric cube', 'Pure shear ($y\\mathbf{e}_x + x\\mathbf{e}_y$)', np.column_stack([ys, xs, np.zeros_like(xs)]), mod_sym),
        ('Symmetric cube', 'Pure torsion ($yz\\mathbf{e}_x - xy\\mathbf{e}_z$)', np.column_stack([ys*zs, np.zeros_like(xs), -xs*ys]), mod_sym),
        ('Pelvis mesh', 'Pure axial stretch ($x\\mathbf{e}_x$)', np.column_stack([xp, np.zeros_like(xp), np.zeros_like(xp)]), mod_pelv),
        ('Pelvis mesh', 'Pure triaxial ($x\\mathbf{e}_x - \\frac{1}{2}y\\mathbf{e}_y - \\frac{1}{2}z\\mathbf{e}_z$)', np.column_stack([xp, -0.5*yp, -0.5*zp]), mod_pelv),
        ('Pelvis mesh', 'Pure bending ($-xy\\mathbf{e}_x + \\frac{1}{2}x^2\\mathbf{e}_y$)', np.column_stack([-xp*yp, 0.5*xp**2, np.zeros_like(xp)]), mod_pelv),
        ('Pelvis mesh', 'Pure shear ($y\\mathbf{e}_x + x\\mathbf{e}_y$)', np.column_stack([yp, xp, np.zeros_like(xp)]), mod_pelv),
        ('Pelvis mesh', 'Pure torsion ($yz\\mathbf{e}_x - xy\\mathbf{e}_z$)', np.column_stack([yp*zp, np.zeros_like(xp), -xp*yp]), mod_pelv),
        ('Pelvis mesh', 'Infinitesimal rigid translation', np.tile([1.0, 2.0, -1.0], (len(pts_pelv), 1)), mod_pelv),
        ('Pelvis mesh', 'Infinitesimal rigid rotation', np.cross([0.2, -0.5, 0.3], pts_pelv - c), mod_pelv),
        ('Pelvis mesh', 'Axial stretch + rigid perturbation', np.column_stack([xp, np.zeros_like(xp), np.zeros_like(xp)]) + np.tile([1.0, 2.0, -1.0], (len(pts_pelv), 1)) + np.cross([0.2, -0.5, 0.3], pts_pelv - c), mod_pelv),
    ]

    records = []
    for domain, name, field, model in cases:
        res = model.fit(field)
        if not res['is_valid']:
            records.append({
                'Domain': domain,
                'Synthetic field': name,
                'Axial (%)': 0.0,
                'Bending (%)': 0.0,
                'Shear (%)': 0.0,
                'Torsion (%)': 0.0,
                'Residual (%)': 0.0,
                'Fitted (%)': 0.0,
                'Status': 'Rigid: rejected ($f_{\\mathrm{nonrigid}} < 10^{-20}$)'
            })
        else:
            fit = (1.0 - res['f_residual']) * 100
            status = r'Exact closure (0.0\% residual)' if res['f_residual'] < 1e-10 else f'{fit:.1f}\\% fit'
            records.append({
                'Domain': domain,
                'Synthetic field': name,
                'Axial (%)': max(0.0, round(res['f_axial'] * 100, 1)),
                'Bending (%)': max(0.0, round(res['f_bending'] * 100, 1)),
                'Shear (%)': max(0.0, round(res['f_shear'] * 100, 1)),
                'Torsion (%)': max(0.0, round(res['f_torsion'] * 100, 1)),
                'Residual (%)': max(0.0, round(res['f_residual'] * 100, 1)),
                'Fitted (%)': max(0.0, round(fit, 1)),
                'Status': status
            })

    df_synth = pd.DataFrame(records)
    df_synth.to_csv(TABLES_DIR / 'global_deformation_synthetic_table.csv', index=False)

    # Format LaTeX
    lines = [
        r'\begin{tabular}{llrrrrrl}',
        r'\toprule',
        r'Domain & Synthetic deformation field & Axial & Bending & Shear & Torsion & Resid. & Outcome \\',
        r' & & (\%) & (\%) & (\%) & (\%) & (\%) & \\',
        r'\midrule',
    ]

    prev_domain = None
    for _, r in df_synth.iterrows():
        dom = r['Domain']
        if prev_domain is not None and dom != prev_domain:
            lines.append(r'\midrule')
        prev_domain = dom
        field = r['Synthetic field']
        ax = f"{r['Axial (%)']:.1f}"
        bend = f"{r['Bending (%)']:.1f}"
        sh = f"{r['Shear (%)']:.1f}"
        tor = f"{r['Torsion (%)']:.1f}"
        res = f"{r['Residual (%)']:.1f}"
        status = r['Status']
        lines.append(f"{dom} & {field} & {ax} & {bend} & {sh} & {tor} & {res} & {status} \\\\")

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        ''
    ])

    tex_path = TABLES_DIR / 'global_deformation_synthetic_table.tex'
    tex_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Generated {tex_path}")


if __name__ == '__main__':
    generate_modes_table()
    generate_synthetic_table()
