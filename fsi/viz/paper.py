"""Shared figure style for the paper. Every figure imports from here so the set reads as one system.

Colour is assigned by the ROLE a series plays, not by its rank in a list -- there are ~12 localizers but they are
not 12 peer categories: there is our method, our training-free variant, the baselines, the reference floor, and
the oracle ceiling. Encoding that structure is what makes the figures readable; cycling 12 hues would not be.

Palette validated with the dataviz validator at the strictest setting (`--pairs all`, light surface #fcfcfb):
lightness band PASS, chroma floor PASS, CVD separation PASS (worst all-pairs dE 9.2 deutan), normal-vision floor
PASS (worst 24.0). AQUA carries a contrast WARN (2.74 vs the 3:1 target) which obligates relief -- so every bar
chart here ships visible value labels. Do not substitute colours without re-running the validator."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- role palette ------------------------------------------------------------------------------------------
OURS      = "#2a78d6"   # FSI -- the proposed method
OURS_ALT  = "#1baf7a"   # FSI-RBC -- our training-free geometric scorer
ACCENT    = "#eb6834"   # oracle / ceiling / the thing being highlighted
BASELINE  = "#6b6a66"   # every competing method: one neutral, they are not the story
FLOOR     = "#b8b7b1"   # zz_* degenerate reference scorers
INK       = "#0b0b0b"
INK2      = "#52514e"
GRID      = "#e2e1dc"

ROLE = {"fsi": OURS, "fsi_rbc": OURS_ALT}
def color_for(name: str) -> str:
    """Role lookup: ours, floor, or baseline."""
    if name in ROLE:
        return ROLE[name]
    return FLOOR if name.startswith("zz_") else BASELINE

def is_ref(name: str) -> bool:
    return name.startswith("zz_")

PRETTY = {"fsi": "FSI (ours)", "fsi_rbc": "FSI-RBC (ours)", "gdn": "GDN", "tranad": "TranAD",
          "aerca": "AERCA", "rcd": "RCD", "graphsl": "GraphSL", "direct_l1": r"direct-$\ell_1$",
          "largest_residual": "largest-residual", "zz_constant": "constant (floor)",
          "zz_random": "random (floor)", "zz_anti_energy": "anti-energy (floor)"}
def pretty(name: str) -> str:
    return PRETTY.get(name, name)

# --- rcParams ----------------------------------------------------------------------------------------------
def use_paper_style(scale: float = 1.0):
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8 * scale, "axes.titlesize": 9 * scale, "axes.labelsize": 8 * scale,
        "xtick.labelsize": 7.5 * scale, "ytick.labelsize": 7.5 * scale, "legend.fontsize": 7.5 * scale,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": False,
        "lines.linewidth": 1.6, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.columnspacing": 1.2,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

# Column widths for a two-column paper, in inches.
COL, WIDE = 3.4, 7.0

def ygrid(ax):
    """Recessive horizontal grid behind the marks -- the only grid these figures use."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6)

def save(fig, path, also_png: bool = True):
    """Vector PDF for the manuscript plus a PNG for slides/preview."""
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"))
    if also_png:
        fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return p

def label_bars(ax, bars, fmt="{:.3f}", pad=2, fontsize=6.5, rotation=0):
    """Visible value labels. Required, not decorative: the aqua slot sits below 3:1 contrast on white, and the
    validator's relief rule makes direct labels the condition for using it."""
    for b in bars:
        w, h = b.get_width(), b.get_height()
        if ax.patches and abs(w) < abs(h):      # vertical bars
            ax.annotate(fmt.format(h), (b.get_x() + w / 2, h), textcoords="offset points",
                        xytext=(0, pad), ha="center", va="bottom", fontsize=fontsize, color=INK2,
                        rotation=rotation)
        else:                                   # horizontal bars
            ax.annotate(fmt.format(w), (w, b.get_y() + h / 2), textcoords="offset points",
                        xytext=(pad, 0), ha="left", va="center", fontsize=fontsize, color=INK2)

def floor_band(ax, value, label="no-information floor", horizontal=True):
    """Mark the degenerate-scorer floor. Every accuracy figure carries this: the metric it replaced (CvV) went
    unchallenged for months precisely because no table showed what a scorer with no information achieved."""
    fn = ax.axhline if horizontal else ax.axvline
    fn(value, color=FLOOR, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    (ax.text if horizontal else ax.text)(
        *( (0.995, value) if horizontal else (value, 0.995) ), f" {label} ",
        transform=ax.get_yaxis_transform() if horizontal else ax.get_xaxis_transform(),
        ha="right" if horizontal else "center", va="bottom", fontsize=6.5, color=INK2)
