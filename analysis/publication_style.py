"""Central publication plotting system for PLOS Computational Biology.

Provides a unified visual grammar:
- Restrained, data-first scientific typography and layout
- Okabe-Ito colorblind-accessible semantic palette
- Standardized physical dimensions (single, 1.5, and double column)
- Unified distribution visualization (deterministic raw points + median + IQR)
- Clean mathematical typesetting for statistical annotations
- True editable vector exports (PDF/SVG) with selective rasterization
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

logger = logging.getLogger("publication_style")

# ── Physical publication dimensions (inches) ───────────────────
# PLOS Computational Biology guidelines:
# - Single column: 83–87 mm (~3.27–3.42 in)
# - 1.5 column: ~130–140 mm (~5.1–5.5 in)
# - Full width (max page): 174–182 mm (~6.85–7.16 in)
COL1_WIDTH = 3.42
COL15_WIDTH = 5.25
COL2_WIDTH = 6.85
FULL_WIDTH = 7.16

# Standard row heights
ROW_H = 1.8
ROW_H_STANDARD = 2.0
ROW_H_COMPACT = 1.6
ROW_H_TALL = 2.6

# ── Global Semantic Palette (Okabe-Ito & Restrained Neutrals) ───
# Sex stratification (only used when sex is part of the scientific question)
MALE_COLOR = "#0072B2"      # Okabe-Ito blue
FEMALE_COLOR = "#D55E00"    # Okabe-Ito vermilion

# Functional modal categories
BACKBONE_COLOR = "#4B5563"      # Neutral slate gray: modes 1–3 (habitual stance)
MIDRANK_COLOR = "#E69F00"       # Okabe-Ito amber/orange: modes 4–8
INLET_SWAP_COLOR = "#009E73"    # Okabe-Ito bluish green: modes 9–10 (key functional pair)
HIGHER_RESERVE_COLOR = "#CC79A7"# Okabe-Ito reddish purple: modes 11–15

# Exchange status
SWAP_COLOR = "#0072B2"          # Reassigned / exchanged
NOSWAP_COLOR = "#6B7280"        # Unchanged / no exchange
UNPAIRED_COLOR = "#9CA3AF"      # Unpaired / failure

# Neutral references
NEUTRAL_COLOR = "#374151"       # Charcoal
LIGHT_NEUTRAL = "#F3F4F6"       # Soft background gray
BORDER_COLOR = "#E5E7EB"        # Subtle grid/border line

# Prevalent block map
BLOCK_COLORS = {
    "1-3": BACKBONE_COLOR,
    "4-8": MIDRANK_COLOR,
    "9-10": INLET_SWAP_COLOR,
    "11-15": HIGHER_RESERVE_COLOR,
    "5-6": MIDRANK_COLOR,
    "12-13": HIGHER_RESERVE_COLOR,
    "12-15": HIGHER_RESERVE_COLOR,
    "14-15": HIGHER_RESERVE_COLOR,
}

# Accessible cycle for general series
OKABE_ITO_CYCLE = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#009E73",  # Bluish green
    "#E69F00",  # Amber
    "#56B4E9",  # Sky blue
    "#CC79A7",  # Reddish purple
    "#4B5563",  # Slate
]

STYLE_FILE = Path(__file__).resolve().parent / "publication.mplstyle"


def apply_publication_style(extra_rc: dict[str, Any] | None = None) -> None:
    """Apply the repository-level publication style globally."""
    if STYLE_FILE.is_file():
        plt.style.use(str(STYLE_FILE))
    else:
        logger.warning("Stylesheet %s not found; using fallback rcParams", STYLE_FILE)
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })
    if extra_rc:
        plt.rcParams.update(extra_rc)


@contextlib.contextmanager
def publication_context(extra_rc: dict[str, Any] | None = None):
    """Context manager for temporary application of publication style."""
    if STYLE_FILE.is_file():
        with plt.style.context(str(STYLE_FILE)):
            if extra_rc:
                with mpl.rc_context(extra_rc):
                    yield
            else:
                yield
    else:
        with mpl.rc_context(extra_rc):
            yield


def panel_label(
    ax: plt.Axes,
    label: str,
    offset: tuple[float, float] = (-0.12, 1.04),
    fontsize: float = 10.0,
    fontweight: str = "bold",
    ha: str = "left",
    va: str = "bottom",
) -> None:
    """Draw a standardized bold panel identifier (e.g. 'A', 'B', 'C').

    Places the label in axes coordinate space consistently across all figures.
    """
    ax.text(
        offset[0],
        offset[1],
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        ha=ha,
        va=va,
        color="#111827",
    )


def style_axis(
    ax: plt.Axes,
    spines: tuple[str, ...] = ("left", "bottom"),
    y_grid: bool = False,
    x_grid: bool = False,
    grid_color: str = "#e5e7eb",
    spine_color: str = "#374151",
    spine_width: float = 0.75,
) -> None:
    """Ensure clean, restrained axis spines and subtle grid."""
    for s in ["top", "right", "left", "bottom"]:
        if s in spines:
            ax.spines[s].set_visible(True)
            ax.spines[s].set_color(spine_color)
            ax.spines[s].set_linewidth(spine_width)
        else:
            ax.spines[s].set_visible(False)

    if y_grid or x_grid:
        ax.set_axisbelow(True)
    if y_grid:
        ax.grid(axis="y", color=grid_color, linestyle="-", linewidth=0.5, alpha=0.8)
    else:
        ax.grid(axis="y", visible=False)
    if x_grid:
        ax.grid(axis="x", color=grid_color, linestyle="-", linewidth=0.5, alpha=0.8)
    else:
        ax.grid(axis="x", visible=False)


def style_distribution(
    ax: plt.Axes,
    data_list: Sequence[Sequence[float] | np.ndarray],
    positions: Sequence[float] | np.ndarray,
    labels: Sequence[str] | None = None,
    color: str | Sequence[str] = NEUTRAL_COLOR,
    width: float = 0.45,
    show_points: bool = True,
    pt_alpha: float = 0.35,
    pt_size: float = 10.0,
    point_colors: Sequence[Sequence[str]] | None = None,
    jitter_width: float = 0.12,
    rng_seed: int = 42,
) -> None:
    """Plot distribution following the data-first grammar: raw points + median + IQR.

    Raw observations provide density and sample transparency; the central estimate
    (median) and interquartile range (IQR) remain prominent.
    """
    rng = np.random.default_rng(rng_seed)
    n_groups = len(data_list)
    colors = [color] * n_groups if isinstance(color, str) else list(color)

    for i, (vals, pos, col) in enumerate(zip(data_list, positions, colors)):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue

        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        iqr = q75 - q25
        whisker_lo = max(arr.min(), q25 - 1.5 * iqr)
        whisker_hi = min(arr.max(), q75 + 1.5 * iqr)

        # 1. Raw observations (deterministic jitter)
        if show_points:
            jitter = rng.uniform(-jitter_width, jitter_width, size=len(arr))
            p_cols = point_colors[i] if point_colors is not None else col
            ax.scatter(
                pos + jitter,
                arr,
                s=pt_size,
                c=p_cols,
                alpha=pt_alpha,
                edgecolors="none",
                linewidths=0,
                zorder=2,
                rasterized=(len(arr) > 500),
            )

        # 2. IQR Box (crisp outline, transparent interior)
        rect = plt.Rectangle(
            (pos - width / 2.0, q25),
            width,
            q75 - q25,
            facecolor="none",
            edgecolor=col,
            linewidth=1.1,
            zorder=3,
        )
        ax.add_patch(rect)

        # 3. Whiskers and caps
        ax.plot([pos, pos], [whisker_lo, q25], color=col, linewidth=0.9, zorder=3)
        ax.plot([pos, pos], [q75, whisker_hi], color=col, linewidth=0.9, zorder=3)
        cap_w = width * 0.35
        ax.plot([pos - cap_w, pos + cap_w], [whisker_lo, whisker_lo], color=col, linewidth=0.9, zorder=3)
        ax.plot([pos - cap_w, pos + cap_w], [whisker_hi, whisker_hi], color=col, linewidth=0.9, zorder=3)

        # 4. Median bar (strongest visual element)
        ax.plot(
            [pos - width / 2.0, pos + width / 2.0],
            [med, med],
            color="#111827",
            linewidth=1.8,
            solid_capstyle="butt",
            zorder=4,
        )

    if labels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)


def style_scatter(
    ax: plt.Axes,
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    c: str | Sequence[str] = MALE_COLOR,
    s: float = 12.0,
    alpha: float = 0.45,
    marker: str = "o",
    label: str | None = None,
    rasterized: bool = False,
) -> Any:
    """Plot publication-quality scatter points with restrained alpha."""
    return ax.scatter(
        x,
        y,
        c=c,
        s=s,
        alpha=alpha,
        marker=marker,
        edgecolors="none",
        linewidths=0,
        label=label,
        rasterized=rasterized,
        zorder=2,
    )


def style_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    cmap: str = "viridis",
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    annot: bool = False,
    annot_fmt: str = "{:.2f}",
    annot_size: float = 7.5,
    annot_color_threshold: float | None = None,
) -> Any:
    """Display a matrix with nearest interpolation, clean borders, and crisp annotations."""
    im = ax.imshow(
        matrix,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    if annot:
        thresh = annot_color_threshold if annot_color_threshold is not None else 0.55
        nr, nc = matrix.shape
        for r in range(nr):
            for c in range(nc):
                val = matrix[r, c]
                if np.isnan(val):
                    continue
                txt = annot_fmt.format(val)
                # Auto high-contrast font color
                tc = "white" if abs(val) > thresh else "#111827"
                ax.text(
                    c,
                    r,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=annot_size,
                    color=tc,
                )

    style_axis(ax, spines=("left", "bottom", "top", "right"), y_grid=False, x_grid=False)
    return im


def style_colorbar(
    fig: plt.Figure,
    mappable: Any,
    ax: Any,
    label: str = "",
    orientation: str = "vertical",
    shrink: float = 0.8,
    pad: float = 0.03,
    fraction: float = 0.04,
) -> Any:
    """Add a restrained, publication-proportioned colorbar."""
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation=orientation,
        shrink=shrink,
        pad=pad,
        fraction=fraction,
    )
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("#374151")
    if label:
        cbar.set_label(label, fontsize=8.5, color="#1f2937")
    cbar.ax.tick_params(labelsize=8, color="#374151", labelcolor="#374151")
    return cbar


def format_pvalue(p_val: float, prefix: str = "p = ") -> str:
    """Format p-value with proper mathematical notation (LaTeX mathtext)."""
    if not np.isfinite(p_val):
        return f"{prefix}--"
    if p_val < 0.001:
        exponent = int(np.floor(np.log10(p_val)))
        coeff = p_val / (10 ** exponent)
        return rf"${prefix}{coeff:.1f} \times 10^{{{exponent}}}$"
    if p_val < 0.05:
        return rf"${prefix}{p_val:.3f}$"
    return rf"${prefix}{p_val:.2f}$"


def save_publication_figure(
    fig: plt.Figure,
    path_stem: str | Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 400,
    close: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """Save figure to vector and raster formats with standardized parameters."""
    p_stem = Path(path_stem)
    p_stem.parent.mkdir(parents=True, exist_ok=True)
    out_paths = []

    for fmt in formats:
        out_file = p_stem.with_suffix(f".{fmt}")
        fig.savefig(
            out_file,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
            edgecolor="none",
        )
        out_paths.append(out_file)

    if verbose:
        logger.info("Saved publication figure: %s (%s)", p_stem.name, ", ".join(formats))
    if close:
        plt.close(fig)
    return out_paths
