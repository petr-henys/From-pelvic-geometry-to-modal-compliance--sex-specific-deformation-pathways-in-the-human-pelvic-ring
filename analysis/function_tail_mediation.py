#!/usr/bin/env python3
"""Labour output metric + within-sex routing mediation dashboard.

Implements the manuscript TODO:
  - define an explicit labour ``output`` metric from static displacement fields
    (e.g. inlet transverse widening under LAB),
  - test whether within-subspace routing variables (E9/(E9+E10)) are associated
    with the function–tail relationship within each sex,
  - generate a publication dashboard figure + a small LaTeX summary table.

Design principles
-----------------
* Reuses existing validated loaders/metrics (tail-risk CSVs, routing Zarr, landmark
  markups, patient-specific functional directions).
* Deterministic (no stochastic fitting or random plotting steps).
* I/O-light: reads displacement Zarr in subject-chunks and never caches the whole
  field in RAM.

Outputs (default paths)
-----------------------
analysis_outputs/figures/fig6_function_tail_mediation_bone.pdf
analysis_outputs/tables/function_tail_mediation_lab_phase1_bone.tex
analysis_outputs/function_tail_mediation/function_outputs_lab_phase1.csv
analysis_outputs/function_tail_mediation/mediation_models_lab_phase1.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.fig_utils import build_routing_df, ols_robust  # noqa: E402
from analysis.spectral_data import load_fe_mesh_coords, load_inlet_landmarks, load_outlet_landmarks  # noqa: E402
from analysis.spectral_metrics import _idw_weights, compute_patient_directions  # noqa: E402
from utils.plot_utils import (  # noqa: E402
    ANNOT_SIZE,
    DEFAULT_DPI,
    FULL_WIDTH,
    ROW_H,
    setup_plot_style,
)

DEFAULT_DATA_DIR = PROJECT_ROOT / "results" / "ref_S1P_fixed_new2" / "data"
DEFAULT_TAIL_CSV = PROJECT_ROOT / "analysis_outputs" / "tail_risk" / "metrics_tail.csv"
DEFAULT_FIG_DIR = PROJECT_ROOT / "analysis_outputs" / "figures"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "analysis_outputs" / "tables"
DEFAULT_OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "function_tail_mediation"


SUPPORTED_FUNCTIONALS: tuple[str, ...] = ("AP", "ML", "BIS", "BIT", "OUTLETAP")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--tail-csv", type=Path, default=DEFAULT_TAIL_CSV)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--loadcase", default="LAB_phase1", help="Labour loadcase used for function/routing/tail alignment.")
    p.add_argument("--domain", default="bone", help="Tissue group in tail-risk CSV (default: bone).")
    p.add_argument(
        "--functional",
        default="ML",
        choices=list(SUPPORTED_FUNCTIONALS),
        help="Which diameter functional defines labour output (default: ML inlet transverse).",
    )
    p.add_argument("--k", type=int, default=32, help="IDW neighbors (must match canonical functional settings).")
    p.add_argument("--power", type=float, default=2.0, help="IDW power (must match canonical functional settings).")
    p.add_argument("--smoothing", type=float, default=50.0, help="RBF smoothing for patient-specific directions.")
    p.add_argument("--neighbors", type=int, default=10, help="RBF neighbors for patient-specific directions.")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return p.parse_args(argv)


def _load_template_points() -> np.ndarray:
    import pyvista as pv

    tpl = pv.read(str(PROJECT_ROOT / "data" / "pelvic.vtk"))
    return np.asarray(tpl.points, dtype=float)


def _load_shapes_mmap() -> np.ndarray:
    return np.load(str(PROJECT_ROOT / "data" / "X.npy"), mmap_mode="r")


def _load_landmark_pair(functional: str) -> tuple[np.ndarray, np.ndarray]:
    functional = str(functional).upper()
    if functional in ("AP", "ML"):
        lm = load_inlet_landmarks()
    else:
        lm = load_outlet_landmarks()
    if functional not in lm:
        raise KeyError(f"Missing landmarks for '{functional}'. Available: {sorted(lm)}")
    pts = np.asarray(lm[functional], dtype=float)
    if pts.shape != (2, 3):
        raise ValueError(f"Landmark '{functional}' has shape {pts.shape}, expected (2, 3)")
    return pts[0], pts[1]


def _compute_function_output(
    *,
    data_dir: Path,
    loadcase: str,
    functional: str,
    k: int,
    power: float,
    smoothing: float,
    neighbors: int,
    cache_dir: Path,
) -> pd.DataFrame:
    """Compute diameter change and max displacement magnitude per subject."""
    import zarr
    from scipy.spatial import cKDTree

    disp_path = data_dir / f"displacements_{loadcase}.zarr"
    if not disp_path.exists():
        raise FileNotFoundError(f"Missing displacement Zarr: {disp_path}")
    disp_z = zarr.open_group(str(disp_path), mode="r")["data"]  # (N, n_nodes, 3)
    n_subj, n_nodes, gdim = disp_z.shape
    if gdim != 3:
        raise ValueError(f"Expected displacement last-dim=3, got {disp_z.shape}")

    # ── patient-specific direction e_s (cached) ──────────────────────────────
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{functional}_smooth{str(smoothing).replace('.', 'p')}_nbr{neighbors}"
    dirs_path = cache_dir / f"patient_dirs_{tag}.npy"

    if dirs_path.exists():
        dirs = np.load(str(dirs_path))
        if dirs.shape != (n_subj, 3):
            raise ValueError(f"Cached dirs shape {dirs.shape} mismatch expected {(n_subj, 3)}: {dirs_path}")
        logger.info("Loaded cached patient directions: %s", dirs_path)
    else:
        template_points = _load_template_points()
        shapes = _load_shapes_mmap()
        p1, p2 = _load_landmark_pair(functional)
        logger.info("Computing patient directions (%s) via RBF (this is the slow step) …", functional)
        dirs = compute_patient_directions(
            template_points,
            shapes,
            p1,
            p2,
            smoothing=float(smoothing),
            neighbors=int(neighbors),
        )
        np.save(str(dirs_path), dirs.astype(np.float64, copy=False))
        logger.info("Cached patient directions: %s", dirs_path)

    # ── IDW neighbor weights on FE mesh (template) ───────────────────────────
    coords = load_fe_mesh_coords()
    if coords is None:
        raise RuntimeError("Cannot load FE mesh coordinates from FEB; required for IDW evaluation.")
    if coords.shape != (n_nodes, 3):
        raise ValueError(f"FE mesh coords shape {coords.shape} mismatch displacements n_nodes={n_nodes}")
    tree = cKDTree(coords)
    p1, p2 = _load_landmark_pair(functional)
    idx_pos, w_pos = _idw_weights(tree, p1, k=k, power=power)
    idx_neg, w_neg = _idw_weights(tree, p2, k=k, power=power)

    # ── streaming output computation (subject-chunks) ───────────────────────
    out = np.full(n_subj, np.nan, dtype=float)
    max_u = np.full(n_subj, np.nan, dtype=float)
    chunk = int(getattr(disp_z, "chunks", (1,))[0] or 1)

    for s0 in range(0, n_subj, chunk):
        s1 = min(n_subj, s0 + chunk)
        block = np.asarray(disp_z[s0:s1], dtype=float)  # (nb, n_nodes, 3)
        if block.ndim != 3 or block.shape[2] != 3:
            raise RuntimeError(f"Unexpected displacement block shape {block.shape} for {loadcase}")
        # max displacement magnitude per subject (mm)
        max_u[s0:s1] = np.linalg.norm(block, axis=2).max(axis=1)

        # IDW displacement at landmarks
        u_pos = (block[:, idx_pos, :] * w_pos[None, :, None]).sum(axis=1)  # (nb, 3)
        u_neg = (block[:, idx_neg, :] * w_neg[None, :, None]).sum(axis=1)
        delta = u_pos - u_neg

        # ΔD = e_s · (u_pos − u_neg)
        out[s0:s1] = np.einsum("ij,ij->i", dirs[s0:s1], delta, optimize=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        out_per_mm = out / max_u

    return pd.DataFrame(
        {
            "subject_idx": np.arange(n_subj, dtype=int),
            f"delta_{functional}_mm": out,
            "max_u_mm": max_u,
            f"delta_{functional}_per_mm": out_per_mm,
        }
    )


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3f}"


def _write_latex_table(path: Path, rows: list[dict], *, functional: str, loadcase: str, domain: str) -> None:
    """Write a minimal LaTeX table for SI inclusion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    func_tex = rf"\ensuremath{{\Delta D_{{\mathrm{{{functional}}}}}}}"
    _lc_map = {
        "SP2leg": r"SP2leg",
        "SP1leg": r"SP1leg",
        "LAB_phase1": r"LAB$_1$",
        "LAB_phase2": r"LAB$_2$",
        "LAB_phase3": r"LAB$_3$",
    }
    lc_tex = _lc_map.get(str(loadcase))
    if lc_tex is None:
        lc_tex = rf"\texttt{{{str(loadcase).replace('_', r'\_')}}}"
    dom_tex = str(domain).replace("_", r"\_")
    lines: list[str] = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Within-sex function--tail models and routing term.} "
                 rf"Robust OLS (HC3) for {dom_tex} upper-tail localization under {lc_tex}, with a parturition-motivated output proxy "
                 rf"{func_tex} defined from landmark diameter change. "
                 r"Coefficients are reported for the normalized output model "
                 rf"$\log_{{10}}P99(w)\sim {func_tex}/\max\|u\| + E_9/(E_9+E_{{10}}) + \mathrm{{age}}$ "
                 r"within each sex.}")
    lines.append(r"\label{tab:function-tail-mediation}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(rf"Sex & $N$ & $\beta_{{{func_tex}/\max\|u\|}}$ & $p$ & $\beta_{{\mathrm{{ratio}}}}$ & $p$ \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f"{r['sex']} & {r['N']} & {r['beta_out']:.3g} & {r['p_out']} & {r['beta_ratio']:.3g} & {r['p_ratio']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _make_dashboard(
    df: pd.DataFrame,
    *,
    functional: str,
    loadcase: str,
    domain: str,
    out_path: Path,
    dpi: int,
    model_rows: list[dict],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mpl_colors
    from matplotlib.lines import Line2D
    from scipy import stats as sp_stats

    setup_plot_style(dpi=dpi)

    func_mm = f"delta_{functional}_mm"
    func_norm = f"delta_{functional}_per_mm"
    y_col = "logP99"

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FULL_WIDTH, 3 * ROW_H),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.05, 0.8]},
    )

    cmap = plt.get_cmap("viridis")
    norm = mpl_colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Common axis limits for comparability (all sexes together)
    x_abs_all = pd.to_numeric(df[func_mm], errors="coerce").to_numpy(float)
    x_norm_all = pd.to_numeric(df[func_norm], errors="coerce").to_numpy(float)
    y_all = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)

    def _lims(v: np.ndarray, pad_frac: float = 0.06) -> tuple[float, float]:
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (-1.0, 1.0)
        lo, hi = float(np.min(v)), float(np.max(v))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return (lo - 1.0, hi + 1.0)
        pad = pad_frac * (hi - lo)
        return (lo - pad, hi + pad)

    x_abs_lim = _lims(x_abs_all)
    x_norm_lim = _lims(x_norm_all)
    y_lim = _lims(y_all, pad_frac=0.05)

    # For coefficient panels, unify y-range across sexes
    all_betas = []
    all_cis = []
    for mr in model_rows:
        all_betas.extend([float(mr.get("beta_out", np.nan)), float(mr.get("beta_ratio", np.nan))])
        all_cis.extend([float(mr.get("ci_out", np.nan)), float(mr.get("ci_ratio", np.nan))])
    betas = np.asarray(all_betas, dtype=float)
    cis = np.asarray(all_cis, dtype=float)
    if np.isfinite(betas).any() and np.isfinite(cis).any():
        lo = float(np.nanmin(betas - cis))
        hi = float(np.nanmax(betas + cis))
        pad = 0.12 * (hi - lo if hi > lo else 1.0)
        coef_lim = (lo - pad, hi + pad)
    else:
        coef_lim = (-0.3, 0.3)

    for ri, sex in enumerate(["M", "F"]):
        sub = df[df["sex"] == sex].copy()
        for col, xcol, xlabel in [
            (0, func_mm, r"$\Delta D$ (mm)"),
            (1, func_norm, r"$\Delta D/\max\|u\|$ (mm/mm)"),
        ]:
            ax = axes[ri, col]
            x = pd.to_numeric(sub[xcol], errors="coerce").to_numpy(float)
            y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(float)
            ratio = pd.to_numeric(sub["ratio"], errors="coerce").to_numpy(float)
            swap = pd.to_numeric(sub["swap_9_10"], errors="coerce").to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(ratio) & np.isfinite(swap)
            x, y, ratio, swap = x[ok], y[ok], ratio[ok], swap[ok].astype(int, copy=False)

            # No jitter by default (publication cleanliness). Keep RNG for future toggles.
            x_j = x

            # swap markers, ratio colors
            for sv, mk in [(0, "o"), (1, "^")]:
                sel = swap == sv
                if sel.sum() == 0:
                    continue
                ax.scatter(
                    x_j[sel],
                    y[sel],
                    c=ratio[sel],
                    cmap=cmap,
                    norm=norm,
                    s=24,
                    alpha=1.0,
                    marker=mk,
                    edgecolors="0.15",
                    linewidths=0.25,
                    rasterized=False,
                )

            if x.size >= 5:
                r, p = sp_stats.spearmanr(x, y)
                ax.text(
                    0.02,
                    0.97,
                    rf"$\rho_s={r:.2f}$, $p={_format_p(float(p))}$",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=ANNOT_SIZE,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.85", "alpha": 1.0},
                )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\log_{10}P99(w)$" if col == 0 else "")
            ax.grid(True, linewidth=0.3, alpha=0.25)
            ax.set_ylim(*y_lim)
            if col == 0:
                ax.set_xlim(*x_abs_lim)
            else:
                ax.set_xlim(*x_norm_lim)

        # rightmost column: coefficient panel for normalized model
        ax_t = axes[ri, 2]
        ax_t.grid(True, axis="y", linewidth=0.3, alpha=0.25)
        ax_t.axhline(0.0, color="0.2", linewidth=0.6, alpha=0.8)
        mr = next((r for r in model_rows if r["sex"] == sex), None)
        ax_t.set_xticks([0, 1])
        ax_t.set_xticklabels([r"$\Delta D/\max\|u\|$", r"$E_9/(E_9+E_{10})$"], fontsize=ANNOT_SIZE)
        ax_t.set_ylabel(r"$\beta$ (robust OLS, HC3)" if ri == 0 else "")
        ax_t.set_title("")
        ax_t.set_ylim(*coef_lim)
        if mr is None:
            ax_t.text(
                0.5,
                0.5,
                "Model unavailable\n(insufficient N)",
                transform=ax_t.transAxes,
                ha="center",
                va="center",
                fontsize=ANNOT_SIZE,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.85", "alpha": 1.0},
            )
        else:
            x_pos = np.array([0, 1], dtype=float)
            bet = np.array([mr["beta_out"], mr["beta_ratio"]], dtype=float)
            ci = np.array([mr.get("ci_out", np.nan), mr.get("ci_ratio", np.nan)], dtype=float)
            ax_t.errorbar(
                x_pos,
                bet,
                yerr=ci,
                fmt="o",
                color="black",
                ecolor="0.2",
                elinewidth=0.8,
                capsize=3,
                capthick=0.8,
            )
            for xi, yi, pv in zip(x_pos, bet, [mr["p_out"], mr["p_ratio"]]):
                ax_t.text(
                    xi,
                    yi + 0.03 * (coef_lim[1] - coef_lim[0]),
                    f"p={pv}",
                    ha="center",
                    va="bottom",
                    fontsize=ANNOT_SIZE,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "0.85", "alpha": 1.0},
                )

    # Column headers (two-line to avoid overlaps)
    func_tex = rf"\Delta D_{{\mathrm{{{str(functional).upper()}}}}}"
    axes[0, 0].set_title("Output" + "\n" + rf"${func_tex}$ (mm)", fontweight="bold", pad=9)
    axes[0, 1].set_title(
        "Normalized output" + "\n" + rf"${func_tex}/\max\|u\|$ (mm/mm)",
        fontweight="bold",
        pad=9,
    )
    axes[0, 2].set_title("Model terms\n(normalized model)", fontweight="bold", pad=9)

    # Colorbar + swap legend
    cb = fig.colorbar(mappable, ax=axes[:, :2], shrink=0.92, pad=0.01, fraction=0.04)
    cb.set_label(r"$E_9/(E_9+E_{10})$", fontsize=ANNOT_SIZE)
    cb.ax.tick_params(labelsize=ANNOT_SIZE)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="0.15", markersize=5, label="swap=0"),
        Line2D([0], [0], marker="^", linestyle="None", color="0.15", markersize=5, label="swap=1"),
    ]
    # Swap legend (in coeff panel)
    axes[0, 0].legend(handles=legend_handles, loc="lower right", fontsize=ANNOT_SIZE, frameon=False)

    # Row labels (inside left panels, top-right; avoids collisions with legend/corr box)
    axes[0, 0].text(
        0.98,
        0.98,
        "Males",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.85", "alpha": 1.0},
    )
    axes[1, 0].text(
        0.98,
        0.98,
        "Females",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="top",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.85", "alpha": 1.0},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote dashboard: %s", out_path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "cache"

    logger.info("=== Function–tail mediation dashboard ===")
    logger.info("data-dir : %s", args.data_dir)
    logger.info("tail-csv : %s", args.tail_csv)
    logger.info("loadcase : %s", args.loadcase)
    logger.info("domain   : %s", args.domain)
    logger.info("functional: %s", args.functional)

    # ── 1) Tail (aligned loadcase) ───────────────────────────────────────────
    tail = pd.read_csv(args.tail_csv)
    df = tail[(tail["group"] == args.domain) & (tail["loadcase"] == args.loadcase)].copy()
    if df.empty:
        raise ValueError(f"No tail rows for group={args.domain} loadcase={args.loadcase} in {args.tail_csv}")
    if df["subject_idx"].duplicated().any():
        raise ValueError("Tail CSV has repeated subject_idx for this (group, loadcase).")

    df = df[df["P99_SED"] > 0].copy()
    df["logP99"] = np.log10(df["P99_SED"].to_numpy(float))

    # ── 2) Routing (same loadcase) ───────────────────────────────────────────
    routing = build_routing_df(labour_case=args.loadcase, data_dir=args.data_dir)
    df = df.merge(routing[["subject_idx", "ratio", "theta"]], on="subject_idx", how="left")

    # ── 3) Labour output metric from displacements ───────────────────────────
    func_df = _compute_function_output(
        data_dir=args.data_dir,
        loadcase=args.loadcase,
        functional=args.functional,
        k=int(args.k),
        power=float(args.power),
        smoothing=float(args.smoothing),
        neighbors=int(args.neighbors),
        cache_dir=cache_dir,
    )
    df = df.merge(func_df, on="subject_idx", how="left")

    # Save per-subject outputs for provenance / reuse
    suffix = "" if str(args.functional).upper() == "ML" else f"_{str(args.functional).lower()}"
    out_outputs_csv = args.out_dir / f"function_outputs_{args.loadcase}_{args.domain}{suffix}.csv"
    df.sort_values("subject_idx").to_csv(out_outputs_csv, index=False)
    logger.info("Wrote per-subject merged dataset: %s", out_outputs_csv)

    # ── 4) Within-sex models (mediation-style association) ───────────────────
    func_mm = f"delta_{args.functional}_mm"
    func_norm = f"delta_{args.functional}_per_mm"

    model_records: list[dict] = []
    key_rows_for_table: list[dict] = []

    for sex in ["M", "F"]:
        sub = df[df["sex"] == sex].copy()
        sub = sub[np.isfinite(sub[["logP99", func_mm, func_norm, "ratio", "age"]]).all(axis=1)].copy()
        if len(sub) < 20:
            logger.warning("Sex=%s has only n=%d usable rows; skipping models.", sex, len(sub))
            continue

        # Total (absolute output)
        m_abs = ols_robust(sub, f"logP99 ~ {func_mm} + age")
        m_abs_r = ols_robust(sub, f"logP99 ~ {func_mm} + ratio + age")

        # Normalized output: separates deformation scale from directional output
        m_norm_r = ols_robust(sub, f"logP99 ~ {func_norm} + ratio + age")

        # Mediator association
        m_med = ols_robust(sub, f"{func_norm} ~ ratio + age")

        def add_terms(model, model_name: str, terms: list[str]):
            for t in terms:
                model_records.append(
                    {
                        "sex": sex,
                        "model": model_name,
                        "term": t,
                        "beta": float(model.params.get(t, np.nan)),
                        "se": float(model.bse.get(t, np.nan)),
                        "p": float(model.pvalues.get(t, np.nan)),
                        "N": int(model.nobs),
                    }
                )

        add_terms(m_abs, "tail~abs+age", [func_mm, "age"])
        add_terms(m_abs_r, "tail~abs+ratio+age", [func_mm, "ratio", "age"])
        add_terms(m_norm_r, "tail~norm+ratio+age", [func_norm, "ratio", "age"])
        add_terms(m_med, "norm~ratio+age", ["ratio", "age"])

        key_rows_for_table.append(
            {
                "sex": sex,
                "N": int(m_norm_r.nobs),
                "beta_out": float(m_norm_r.params[func_norm]),
                "se_out": float(m_norm_r.bse[func_norm]),
                "ci_out": 1.96 * float(m_norm_r.bse[func_norm]),
                "p_out": _format_p(float(m_norm_r.pvalues[func_norm])),
                "beta_ratio": float(m_norm_r.params["ratio"]),
                "se_ratio": float(m_norm_r.bse["ratio"]),
                "ci_ratio": 1.96 * float(m_norm_r.bse["ratio"]),
                "p_ratio": _format_p(float(m_norm_r.pvalues["ratio"])),
            }
        )

    models_df = pd.DataFrame(model_records)
    out_models_csv = args.out_dir / f"mediation_models_{args.loadcase}_{args.domain}{suffix}.csv"
    models_df.to_csv(out_models_csv, index=False)
    logger.info("Wrote model summary: %s", out_models_csv)

    # ── 5) Figure + LaTeX table ──────────────────────────────────────────────
    fig_path = args.fig_dir / f"fig6_function_tail_mediation_{args.domain}{suffix}.pdf"
    _make_dashboard(
        df,
        functional=args.functional,
        loadcase=args.loadcase,
        domain=args.domain,
        out_path=fig_path,
        dpi=int(args.dpi),
        model_rows=key_rows_for_table,
    )

    table_path = args.table_dir / f"function_tail_mediation_{args.loadcase}_{args.domain}{suffix}.tex"
    _write_latex_table(
        table_path,
        key_rows_for_table,
        functional=str(args.functional).upper(),
        loadcase=str(args.loadcase),
        domain=str(args.domain),
    )
    logger.info("Wrote LaTeX table: %s", table_path)


if __name__ == "__main__":
    main()
