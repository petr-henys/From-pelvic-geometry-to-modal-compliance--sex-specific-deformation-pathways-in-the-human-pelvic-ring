"""Plotting utilities for publication-quality figures.

Matplotlib configuration and Okabe-Ito color scheme for consistent,
journal-ready figures.  Designed for double-column width (7.2 in) at
11 pt body text; at ~6.5 in text width the 0.9x scale keeps all labels
above 6 pt at final printed size.

Colors (Okabe-Ito palette):
- Male: #0173B2 (blue)
- Female: #DE8F05 (orange)
- Scale effect: #CC78BC (purple)
- Sex effect / Age: #029E73 (teal-green)
- Shape effect: #E69F00 (orange)
- Material effect: #56B4E9 (sky blue)
- Neutral bars: #4C72B0 (steel blue)
- Swap line: #D55E00 (vermillion)
- Veering verdict: #029E73 (green)
- Mixing verdict: #CC78BC (purple)
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cycler

# ── Okabe-Ito colorblind-friendly palette ─────────────────────
MALE_COLOR = '#0173B2'
FEMALE_COLOR = '#DE8F05'

# Effect colors (consistent across all analysis plots)
SCALE_EFFECT_COLOR = '#CC78BC'
SEX_EFFECT_COLOR = '#029E73'
SHAPE_EFFECT_COLOR = '#E69F00'
MATERIAL_EFFECT_COLOR = '#56B4E9'

# Additional semantic colors
NEUTRAL_COLOR = '#4C72B0'       # steel blue — non-sex-stratified bars/boxes
AGE_COLOR = '#029E73'           # teal — age effects (same hue as SEX_EFFECT)
VEERING_COLOR = '#029E73'       # green — veering verdict annotation
MIXING_COLOR = '#CC78BC'        # purple — mixing verdict annotation
SWAP_LINE_COLOR = '#D55E00'     # vermillion — running swap-rate curves

# Permutation colors (for mode pairing analysis)
PERMUTATION_COLORS = {
    'aligned': '#B6ACFF',
    'reassigned': '#FF8C00',
    'unpaired': '#6E7881',
}
PERMUTATION_DISPLAY = {
    'aligned': 'Matches ref',
    'reassigned': 'Mapped elsewhere',
    'unpaired': 'Unpaired',
}

# ── Plotting constants ────────────────────────────────────────
DEFAULT_DPI = 600              # publication-quality raster resolution
DEFAULT_FONT_SIZE = 8          # 8 pt base → ~7.2 pt at \linewidth

# Publication figure dimensions (inches) — A4 journal column widths
FULL_WIDTH = 7.2               # double-column / full text width
HALF_WIDTH = 3.5               # single-column figure
COL15_WIDTH = 5.5              # 1.5-column figure
GOLDEN = (1 + 5**0.5) / 2     # golden ratio ≈ 1.618

# Standard subplot row heights (inches)
ROW_H = 1.8                   # standard row (scatter, bar, error-bar)
ROW_H_SMALL = 1.4             # compact row (stacked axes)
PANEL_PAD = 0.25              # constrained_layout w_pad / h_pad

# Font sizes derived from base — use ONLY these for hardcoded fontsize=
SUPTITLE_SIZE = DEFAULT_FONT_SIZE + 1    # 9 pt — figure suptitle
ANNOT_SIZE = DEFAULT_FONT_SIZE - 1       # 7 pt — in-plot annotations
SMALL_ANNOT_SIZE = DEFAULT_FONT_SIZE - 2 # 6 pt — dense multi-panel labels

# Standard element styling (import these to keep plots uniform)
FILL_ALPHA = 0.55
SCATTER_ALPHA = 0.40
LINE_WIDTH = 0.8
SPINE_WIDTH = 0.5
TICK_WIDTH = 0.5
BOX_LW = 0.6                   # box plot whiskers/caps
MEDIAN_LW = 1.0                # box plot median line
BAR_EDGE_LW = 0.4
BAR_EDGE_COLOR = '0.3'

# ── Neutral background / text gray tones ──────────────────────
_TEXT_COLOR = '0.15'            # near-black for body text
_LABEL_COLOR = '0.20'          # axis labels
_TICK_COLOR = '0.35'           # tick labels — slightly muted
_SPINE_COLOR = '0.30'          # axis spines
_GRID_COLOR = '0.90'           # subtle y-grid

_COLOR_CYCLE = [
    NEUTRAL_COLOR,
    MALE_COLOR,
    FEMALE_COLOR,
    SCALE_EFFECT_COLOR,
    SEX_EFFECT_COLOR,
    SHAPE_EFFECT_COLOR,
    MATERIAL_EFFECT_COLOR,
    SWAP_LINE_COLOR,
]


def setup_plot_style(dpi: int = DEFAULT_DPI, font_size: int = DEFAULT_FONT_SIZE) -> None:
    """Configure matplotlib for modern, publication-ready figures.

    Designed for double-column journal figures (7.2 in wide) with 11 pt
    body text.  At ``\\linewidth`` ~ 6.5 in the scale factor is ~0.9,
    so 8 pt base renders at ~7.2 pt on the printed page.

    Key style choices:
    - Light horizontal grid on y-axis for readability
    - Minimal spines (left + bottom only)
    - Muted tick/label colors to keep data prominent
    - Tight constrained-layout defaults for multi-panel figures

    Parameters
    ----------
    dpi : int
        Figure save resolution.
    font_size : int
        Base font size in points.
    """
    plt.rcParams.update({
        # ── Font family ────────────────────────────────────────
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica',
                            'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'mathtext.default': 'regular',
        'axes.unicode_minus': False,
        # ── Font sizes ─────────────────────────────────────────
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'figure.titlesize': font_size + 1,     # suptitle
        'figure.titleweight': 'bold',
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size - 1,
        'legend.title_fontsize': font_size,
        # ── Axes ───────────────────────────────────────────────
        'axes.linewidth': SPINE_WIDTH,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': _SPINE_COLOR,
        'axes.labelcolor': _LABEL_COLOR,
        'axes.facecolor': '#ffffff',
        'axes.titleweight': 'bold',
        'axes.titlepad': 4,
        'axes.prop_cycle': cycler(color=_COLOR_CYCLE),
        'axes.xmargin': 0.02,
        'axes.ymargin': 0.05,
        # ── Ticks ──────────────────────────────────────────────
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.color': _TICK_COLOR,
        'ytick.color': _TICK_COLOR,
        'xtick.labelcolor': _TICK_COLOR,
        'ytick.labelcolor': _TICK_COLOR,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.major.width': TICK_WIDTH,
        'ytick.major.width': TICK_WIDTH,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        # ── Grid — subtle horizontal lines only ────────────────
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': _GRID_COLOR,
        'grid.linewidth': 0.4,
        'grid.linestyle': '-',
        'grid.alpha': 0.8,
        # ── Figure ─────────────────────────────────────────────
        'figure.facecolor': '#ffffff',
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.w_pad': 0.04,
        'figure.constrained_layout.h_pad': 0.04,
        'figure.constrained_layout.wspace': 0.03,
        'figure.constrained_layout.hspace': 0.06,
        # ── Saving ─────────────────────────────────────────────
        'savefig.dpi': dpi,
        'figure.dpi': 150,
        'savefig.transparent': False,
        'savefig.facecolor': '#ffffff',
        'savefig.edgecolor': '#ffffff',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        # ── Embedded fonts (PDF/PS) ────────────────────────────
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        # ── Legend ──────────────────────────────────────────────
        'legend.frameon': True,
        'legend.framealpha': 0.85,
        'legend.edgecolor': '0.85',
        'legend.fancybox': True,
        'legend.handlelength': 1.2,
        'legend.handletextpad': 0.4,
        'legend.columnspacing': 0.8,
        'legend.borderaxespad': 0.3,
        'legend.borderpad': 0.35,
        # ── Lines ──────────────────────────────────────────────
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': 4,
        'errorbar.capsize': 2.0,
        'patch.linewidth': BAR_EDGE_LW,
        # ── Boxplots ──────────────────────────────────────────
        'boxplot.boxprops.linewidth': BOX_LW,
        'boxplot.whiskerprops.linewidth': BOX_LW,
        'boxplot.capprops.linewidth': BOX_LW,
        'boxplot.medianprops.linewidth': MEDIAN_LW,
        'boxplot.medianprops.color': 'black',
        'boxplot.flierprops.markersize': 3,
    })


def set_suptitle(fig: plt.Figure, title: str, **kwargs) -> None:
    """Set a figure suptitle with controlled font size and wrapping.

    Long titles are soft-wrapped at ~60 chars to avoid layout
    deformation in constrained-layout figures.

    Parameters
    ----------
    fig : Figure
        Target figure.
    title : str
        Suptitle text.
    **kwargs
        Forwarded to ``fig.suptitle()``.
    """
    defaults = dict(
        fontsize=SUPTITLE_SIZE,
        fontweight='bold',
        y=0.98,
    )
    defaults.update(kwargs)
    wrapped = '\n'.join(textwrap.wrap(title, width=60))
    fig.suptitle(wrapped, **defaults)


def panel_label(ax: plt.Axes, label: str, *, loc: str = 'left') -> None:
    """Set a bold panel label as axes title.

    Parameters
    ----------
    ax : Axes
        Target axes.
    label : str
        Full label text, e.g. "(a) Gap distribution".
    loc : str
        Title location ('left', 'center', 'right').
    """
    ax.set_title(label, loc=loc, fontweight='bold', fontsize=DEFAULT_FONT_SIZE)


def format_mode_label(mode_idx: int) -> str:
    """Convert 0-based mode index to 1-based label ("Mode 1")."""
    return f"Mode {mode_idx + 1}"


def format_mode_label_short(mode_idx: int, *, one_based: bool = False) -> str:
    """Short mode label, e.g. "M1".

    Parameters
    ----------
    mode_idx : int
        Mode index.
    one_based : bool
        If True, *mode_idx* is already 1-based.
    """
    m = mode_idx if one_based else mode_idx + 1
    return f"M{m}"
