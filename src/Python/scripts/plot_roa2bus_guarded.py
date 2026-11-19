"""Two-bus region-of-attraction map of the proposed method, drawn in the style of
the MATLAB maps of the ROA repository (stand-in for ROA/Guarded.m when MATLAB is
not available).  Output: figures/roa/Guarded_58mm.png and _88mm.png"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
from run_roa2bus_guarded import run

np.seterr(all="ignore")
n, max_iter = 256, 25
vs, ds = np.linspace(0, 5, n), np.linspace(-math.pi / 2, math.pi / 2, n)
cache = Path(__file__).resolve().parents[1] / "results" / "journal" / "roa2bus_capped_map.npy"
if cache.exists():
    it = np.load(cache)
else:
    it = np.full((n, n), -1.0)
    for i, v in enumerate(vs):
        for j, d in enumerate(ds):
            try:
                k, x = run(np.array([d, v]), "capped", max_iter, cap=0.5)
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
                k = None
            if k is not None and abs(x[1]) > 0.5:
                it[i, j] = k
    np.save(cache, it)
print(f"operable solution reached from {100 * np.mean(it > 0):.1f}% of the grid, mean {it[it > 0].mean():.2f} iterations")
out = Path(__file__).resolve().parents[1] / "figures" / "roa"
cmap = ListedColormap(np.vstack([[0, 0, 0, 1], plt.get_cmap("jet")(np.linspace(0, 1, max_iter))]))
for width, height, fs in ((58, 44, 6), (88, 60, 7)):
    small = width < 60
    plt.rcParams.update({"font.family": "Arial", "font.size": fs, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                         "mathtext.fontset": "custom", "mathtext.it": "Arial:italic", "mathtext.rm": "Arial"})
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    ax = fig.add_axes([0.2, 0.2, 0.52, 0.74] if small else [0.13, 0.15, 0.68, 0.81])
    cax = fig.add_axes([0.76, 0.2, 0.045, 0.74] if small else [0.84, 0.15, 0.035, 0.81])
    m = ax.pcolormesh(ds, vs, it, cmap=cmap, vmin=-1, vmax=max_iter, shading="nearest", rasterized=True)
    cs = ax.contour(ds, vs, np.where(it > 0, it, np.nan), levels=[5, 6, 7, 8], colors="k", linewidths=0.4)
    ax.clabel(cs, fmt="%d", fontsize=fs - 1, inline=True)
    ax.set_xlabel(r"$\theta_2$ [rad]", labelpad=1); ax.set_ylabel(r"v$_2$ [pu]", labelpad=1)
    ax.set_xlim(ds[0], ds[-1]); ax.set_ylim(0, 5)
    ax.set_xticks([-1, 0, 1] if small else np.arange(-1.5, 1.6, 0.5)); ax.set_yticks(range(6) if small else np.arange(0, 5.1, 0.5))
    ax.tick_params(direction="out", pad=1.5)
    cb = fig.colorbar(m, cax=cax, ticks=[0, 5, 10, 15, 20, 25])
    cb.set_label("Iterations to converge", labelpad=1); cb.outline.set_linewidth(0.5)
    fig.savefig(out / f"Guarded_{width}mm.png", dpi=600)
    plt.close(fig)
