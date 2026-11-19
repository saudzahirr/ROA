"""Figure style for print.

* figure widths: 88 mm (one column) or 180 mm (two columns); figures are drawn
  at their final size, so the text in the saved file is the text in print;
* lettering: sans-serif (Arial / Helvetica), 5-7 pt; panel labels 8 pt bold,
  lower case;
* line weights 0.5-1 pt;
* colours: the Okabe-Ito colour-blind-safe palette (no red-green contrasts) and
  a perceptually uniform colour map instead of a rainbow scale;
* output: vector PDF with embedded TrueType fonts, plus a 600 dpi PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1.0 / 25.4
SINGLE = 88 * MM  # one-column width in inches
DOUBLE = 180 * MM  # two-column width in inches
FONT = 7  # axis labels, legends
SMALL = 6  # tick labels
LABEL = 8  # panel labels

# sequential colour map for iteration counts and voltages (set to "jet" to
# reproduce the rainbow scale of the earlier figures)
CMAP = "viridis"

# Okabe-Ito palette
BLUE, SKY, GREEN, YELLOW, ORANGE, VERMILLION, PURPLE, GREY, BLACK = (
    "#0072B2", "#56B4E9", "#009E73", "#F0E442", "#E69F00", "#D55E00", "#CC79A7", "#7F7F7F", "#000000")

# one colour / marker / label per method, used in every figure
METHODS = {
    "NR": dict(label="NR", color=BLUE, marker="o"),
    "NR-SGD": dict(label="NR + SGD", color=PURPLE, marker="X"),
    "SGD": dict(label="SGD", color=SKY, marker="v"),
    "NR-LS": dict(label="NR + line search", color=SKY, marker="v"),
    "CN-RK4A": dict(label="Continuous Newton (RK4)", color=GREY, marker="h"),
    "LM": dict(label="Levenberg-Marquardt", color=GREEN, marker="D"),
    "LI": dict(label="LI", color=GREEN, marker="<"),
    "ALI": dict(label="Accelerated Landweber", color=YELLOW, marker=">"),
    "NR-LI": dict(label="NR + LI", color=ORANGE, marker="s"),
    "NR-ALI": dict(label="NR-ALI", color=VERMILLION, marker="^"),
    "NR-ALI-W": dict(label="NR-ALI + look-ahead trial", color=GREY, marker="P"),
    "NR-ALI-G": dict(label="Guarded, no angle cap", color=VERMILLION, marker="o"),
    "NR-ALI-GC": dict(label="Guarded NR + LI (proposed)", color=BLACK, marker="*"),
}


def apply() -> None:
    """Set the Matplotlib defaults used by every figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT,
        "axes.labelsize": FONT,
        "axes.titlesize": FONT,
        "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL,
        "legend.fontsize": SMALL,
        "legend.frameon": False,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "grid.linewidth": 0.3,
        "grid.color": "#D0D0D0",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,  # embed TrueType fonts (editable text)
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.04) -> None:
    """Bold lower-case panel label at the top left of an axes, outside the data area."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=LABEL, fontweight="bold", va="bottom", ha="right")


def legend_above(fig, handles, labels, ncol: int, y: float = 1.0) -> None:
    """One legend for the whole figure, above the panels, so that it never covers a curve."""
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=2.2,
               columnspacing=1.4, borderaxespad=0.2)


def save(fig, path: str | Path) -> Path:
    """Save as vector PDF and as 600 dpi PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path.with_suffix(".pdf")
