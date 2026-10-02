"""Format the new anatomical prediction and load-mixture analyses."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "tables/prediction"
FIG = PAPER / "figures"
LOADS = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
LABEL = {a: b for a, b in zip(LOADS, ["SP2leg", "SP1leg", "LAB1", "LAB2", "LAB3"])}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42})


def table(path, columns, rows):
    lines = [r"\begin{tabular}{"+"l"*len(columns)+"}", r"\toprule",
             " & ".join(columns)+r" \\", r"\midrule"]
    lines.extend(" & ".join(row)+r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines)+"\n")


def prediction():
    metrics = pd.read_csv(DATA / "prediction_metrics.csv")
    gains = pd.read_csv(DATA / "paired_prediction_gains.csv")
    rows = []
    for load in LOADS:
        for outcome, label in [("rot_mag_deg", "Rotation (deg)"), ("trans_mag_mm", "Translation (mm)")]:
            a = metrics.loc[(metrics.load_case == load) & (metrics.outcome == outcome)].set_index("model")
            g = gains.loc[(gains.load_case == load) & (gains.outcome == outcome) & (gains.comparison == "size_to_triad")].iloc[0]
            rows.append([LABEL[load], label, f"{a.loc['size','cv_r2']:.3f}", f"{a.loc['triad','cv_r2']:.3f}",
                         f"{a.loc['size','rmse']:.4f}", f"{a.loc['triad','rmse']:.4f}",
                         f"{g.reduction_pct:.1f}"])
    table(DATA / "table_prediction.tex", ["Load", "Outcome", "$R^2_{size}$", "$R^2_{triad}$",
                                           "RMSE size", "RMSE triad", "Reduction (\\%)"], rows)
    rows = []
    for load in LOADS:
        for outcome, label in [("rot_mag_deg", "Rotation"), ("trans_mag_mm", "Translation")]:
            a = metrics.loc[(metrics.load_case == load) & (metrics.outcome == outcome)].set_index("model")
            selected = gains.loc[(gains.load_case == load) & (gains.outcome == outcome)].set_index("comparison")
            g = selected.loc["size_to_triad"]
            rows.append([LABEL[load], label, f"{g.delta_rmse:.4f} [{g.ci95_low:.4f}, {g.ci95_high:.4f}]",
                         f"{a.loc['triad_sex','cv_r2']:.3f}", f"{a.loc['all_dimensions','cv_r2']:.3f}",
                         f"{selected.loc['triad_to_triad_sex','reduction_pct']:.1f}".replace("-0.0", "0.0"),
                         f"{selected.loc['triad_to_all_dimensions','reduction_pct']:.1f}"])
    table(DATA / "supp_prediction.tex", ["Load", "Outcome", "$\\Delta$RMSE size--triad [95\\% CI]",
                                          "$R^2_{triad+sex}$", r"$R^2_{8\ measures}$",
                                          "Sex gain (\\%)", "8-measure gain (\\%)"], rows)
    rows = []
    for load in LOADS:
        for outcome, label in [("rot_mag_deg", "Rotation"), ("trans_mag_mm", "Translation")]:
            a = metrics.loc[(metrics.load_case == load) & (metrics.outcome == outcome)].set_index("model")
            selected = gains.loc[(gains.load_case == load) & (gains.outcome == outcome)].set_index("comparison")
            intervals = []
            for comparison in ["triad_to_triad_sex", "triad_to_all_dimensions"]:
                g = selected.loc[comparison]
                intervals.append(f"{g.delta_rmse:.5f} [{g.ci95_low:.5f}, {g.ci95_high:.5f}]")
            rows.append([LABEL[load], label, *intervals, f"{a.loc['triad','mae']:.4f}",
                         f"{a.loc['triad','p95_absolute_error']:.4f}"])
    table(DATA / "supp_prediction_benchmarks.tex", ["Load", "Outcome", "Sex $\\Delta$RMSE [95\\% CI]",
          "8-measure $\\Delta$RMSE [95\\% CI]", "Triad MAE", "Triad P95 error"], rows)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    positions = np.arange(5)
    for ax, outcome, title in zip(axes[:2], ["rot_mag_deg", "trans_mag_mm"], ["A. Rotation", "B. Translation"]):
        for offset, model, color, label in [(-.09, "size", "#888888", "Size + age"),
                                            (.09, "triad", "#0072b2", "+ anatomical triad")]:
            sub = metrics.loc[(metrics.outcome == outcome) & (metrics.model == model)].set_index("load_case").loc[LOADS]
            ax.barh(positions+offset, sub.cv_r2, height=.17, color=color, label=label)
        ax.set(yticks=positions, yticklabels=[LABEL[x] for x in LOADS], xlim=(0, .82), xlabel="Held-out prediction $R^2$", title=title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.06, 0), ncol=2, fontsize=8)
    archive = np.load(DATA / "held_out_predictions.npz")
    # Scatter uses repeat-mean predictions only for display; metrics pool errors
    # over repeats, and must not be inferred from the scatter's ensemble values.
    predicted = archive["predictions"][1, :, :, :, 1].mean(axis=0)
    observed = archive["target"][:, :, 1]
    for j, load in enumerate(LOADS):
        axes[2].scatter(observed[:, j], predicted[:, j], s=6, alpha=.35, label=LABEL[load])
    bounds = [0, max(observed.max(), predicted.max())*1.02]
    axes[2].plot(bounds, bounds, "k--", lw=.8)
    axes[2].set(xlabel="FE translation (mm)", ylabel="Held-out prediction (mm)", title="C. Triad-model translations", xlim=bounds, ylim=bounds)
    axes[2].legend(fontsize=7)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(FIG / "Fig_prediction_subject_heldout.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Fig_prediction_subject_heldout.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def supporting():
    agreement = pd.read_csv(DATA / "agreement_audit.csv")
    rows = []
    for r in agreement.loc[agreement.metric.isin(["rot_mag_deg", "trans_mag_mm"])].itertuples():
        rows.append([LABEL[r.load_case], "Rotation (deg)" if r.metric == "rot_mag_deg" else "Translation (mm)",
                     f"{r.rmse:.4f}", f"{r.mean_bias:.4f}", f"{r.p95_absolute_error:.4f}",
                     f"{r.maximum_absolute_error:.4f}"])
    table(DATA / "supp_agreement_tails.tex", ["Load", "Outcome", "RMSE", "Mean bias", "P95 abs. error", "Max abs. error"], rows)
    sites = pd.read_csv(DATA / "exploratory_site_contrasts.csv")
    rows = [[{"ml_trans_abs_mm": "ML", "ap_trans_abs_mm": "AP", "cc_trans_abs_mm": "CC"}[r.metric],
             f"{r.median_delta_mm:.4f}", f"[{r.low:.4f}, {r.high:.4f}]", f"{r.q:.3g}"]
            for r in sites.itertuples()]
    table(DATA / "supp_site_contrasts.tex", ["Component", "LAB2--LAB1 (mm)", "95\\% CI (mm)", "$q$"], rows)
    slopes = pd.read_csv(DATA / "volume_model_audit.csv")
    rows = []
    for load in LOADS:
        for outcome, label in [("rot_mag_deg", "Rotation"), ("trans_mag_mm", "Translation")]:
            for sex in ["male", "female"]:
                a = slopes.loc[(slopes.load_case == load) & (slopes.outcome == outcome) & (slopes.sex == sex)].set_index("specification")
                intervals = []
                for spec in ["without_triad", "published"]:
                    r = a.loc[spec]
                    intervals.append(f"{r.slope:.3f} [{r.low:.3f}, {r.high:.3f}]")
                rows.append([LABEL[load], label, sex.capitalize(), *intervals,
                             f"{a.loc['without_triad','q']:.3g}"])
    table(DATA / "supp_volume_adjustment.tex", ["Load", "Outcome", "Sex", "Without triad [95\\% CI]",
                                                "With triad [95\\% CI]", "$q$ without triad"], rows)


def mixture():
    sub = pd.read_csv(DATA / "mixture_subject_endpoints.csv")
    summary = pd.read_csv(DATA / "mixture_summary.csv")
    rows = []
    for r in summary.itertuples():
        rows.append([f"{r.weight:.2f}", f"{r.median_translation_mm:.4f}",
                     f"{r.median_directional_reduction_pct:.1f} [{r.reduction_ci95_low:.1f}, {r.reduction_ci95_high:.1f}]",
                     f"{r.scalar_oracle_rmse_mm:.5f}", f"{r.vector_oracle_rmse_mm:.5f}"])
    table(DATA / "table_mixture.tex", ["Ischial fraction $w$", "Translation (mm)",
                                       "Directional reduction (\\%) [95\\% CI]",
                                       "Scalar RMSE (mm)", "Vector RMSE (mm)"], rows)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    for i, a in sub.groupby("subject_idx"):
        axes[0].plot(a.weight, a.translation_mm, color="#0072b2", lw=.4, alpha=.07)
    for col, color, label, style in [("translation_mm", "#0072b2", "Re-extracted mixed field", "-"),
                                     ("scalar_oracle_mm", "#d55e00", "Scalar endpoint mixture", "--"),
                                     ("vector_oracle_mm", "#009e73", "Signed-vector mixture", ":")]:
        medians = sub.groupby("weight")[col].median()
        axes[0].plot(medians.index, medians, color=color, lw=2, label=label, ls=style)
    axes[0].set(xlabel="Ischial force fraction $w$", ylabel="Bilateral translation (mm)", title="A. Controlled force redistribution")
    axes[0].legend(fontsize=7)
    midpoint = sub.loc[sub.weight == .5]
    axes[1].scatter(midpoint.mean_endpoint_cosine, midpoint.directional_reduction_pct, s=9, alpha=.45, color="#0072b2")
    axes[1].set(xlabel="Mean cosine of endpoint vectors", ylabel="Reduction from scalar mixture (%)", title="B. Directional cancellation", xlim=(-1.05, 1.05))
    interior = summary.loc[summary.weight.between(.01, .99)]
    axes[2].plot(interior.weight, interior.scalar_oracle_rmse_mm, "o-", color="#d55e00", label="Scalar endpoint mixture")
    axes[2].plot(interior.weight, interior.vector_oracle_rmse_mm, "o-", color="#009e73", label="Signed-vector mixture")
    axes[2].set(xlabel="Ischial force fraction $w$", ylabel="RMSE against mixed field (mm)", title="C. Endpoint-information test")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "Fig_load_mixture_mechanism.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Fig_load_mixture_mechanism.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    prediction()
    supporting()
    if (DATA / "mixture_summary.csv").exists():
        mixture()
