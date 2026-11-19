"""Two-bus region-of-attraction maps in the style of the MATLAB maps of the ROA
repository, for the methods that have no MATLAB script there (or while MATLAB
is unavailable): stochastic gradient descent alone, Newton-Raphson with
stochastic gradient descent, and the proposed guarded method.

As in the MATLAB scripts a run counts as converged when the mismatch falls
below 1e-5 p.u. within 25 iterations, whichever solution it reaches.

    uv run python scripts/roa2bus_oldstyle.py

Output: figures/roa/{SGD,NRSGD,Guarded}_58mm.png and _88mm.png,
        results/journal/roa2bus_oldstyle.csv
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
from roa2bus_four import MAX_ITER, TOL, lw, nr_sgd, proposed  # noqa: E402
from run_roa2bus_guarded import jac, mis, run as run_basic  # noqa: E402


def sgd(x, max_iter=MAX_ITER, eta=0.01, b1=0.9, b2=0.999):
    """Gradient steps with the Adam rule alone (the single load bus is drawn in every step)."""
    mom, sec = np.zeros(2), np.zeros(2)
    for it in range(1, max_iter + 1):
        F = mis(x)
        if np.max(np.abs(F)) < TOL:
            return it, x
        g = -(jac(x).T @ F)
        mom = b1 * mom + (1 - b1) * g
        sec = b2 * sec + (1 - b2) * g * g
        x = x - eta * (mom / (1 - b1**it)) / (np.sqrt(sec / (1 - b2**it)) + 1e-8)
    return None, x


def fixed_step_gd(x, max_iter=MAX_ITER, eta=0.01):
    """Gradient descent with a fixed step: what SGD is on a system with a single load bus."""
    for it in range(1, max_iter + 1):
        F = mis(x)
        if np.max(np.abs(F)) < TOL:
            return it, x
        x = x + eta * (jac(x).T @ F)
    return None, x


def landweber(x, max_iter=MAX_ITER):
    for it in range(1, max_iter + 1):
        if np.max(np.abs(mis(x))) < TOL:
            return it, x
        x = lw(x)
    return None, x


# all six panels of the two-bus figure, drawn by one renderer so that fonts, layout and colours are identical
METHODS = {"NewtonRaphson": lambda x: run_basic(x, "nr", MAX_ITER), "LandweberIteration": landweber,
           "SGD": fixed_step_gd, "NRSGD": nr_sgd, "Hybrid": lambda x: run_basic(x, "hybrid", MAX_ITER),
           "Guarded": proposed}
LEVELS = {"NewtonRaphson": [5, 6, 7, 8], "LandweberIteration": [11, 13, 15, 16], "SGD": [13, 16, 19, 22],
          "NRSGD": [5, 6, 7, 8], "Hybrid": [5, 6, 7, 8], "Guarded": [5, 6, 7, 8]}

np.seterr(all="ignore")
n = 256
root = Path(__file__).resolve().parents[1]
out = root / "figures" / "roa" / "py"
out.mkdir(parents=True, exist_ok=True)
vs, ds = np.linspace(0, 5, n), np.linspace(-math.pi / 2, math.pi / 2, n)
def matlab_jet(m):
    """MATLAB's jet(m) colour map."""
    k = math.ceil(m / 4)
    u = np.concatenate([np.arange(1, k + 1) / k, np.ones(k - 1), np.arange(k, 0, -1) / k])
    g = math.ceil(k / 2) - (m % 4 == 1) + np.arange(1, u.size + 1)
    r, b = g + k, g - k
    J = np.zeros((m, 3))
    J[r[r <= m] - 1, 0] = u[:np.sum(r <= m)]
    J[g[g <= m] - 1, 1] = u[:np.sum(g <= m)]
    J[b[b >= 1] - 1, 2] = u[u.size - np.sum(b >= 1):]
    return J


cmap = ListedColormap(np.vstack([[0, 0, 0], matlab_jet(MAX_ITER)]))
rows = []
for name, fn in METHODS.items():
    cache = root / "results" / "journal" / f"roa2bus_oldstyle_{name}{'_fixed' if name == 'SGD' else ''}.npy"
    if cache.exists():
        it = np.load(cache)
    else:
        it = np.full((n, n), -1.0)
        for i, v in enumerate(vs):
            for j, d in enumerate(ds):
                try:
                    k, x = fn(np.array([d, v]))
                except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError, ZeroDivisionError):
                    k = None  # the iterate diverged to infinity
                if k is not None and np.all(np.isfinite(x)):
                    it[i, j] = k
        np.save(cache, it)
    ok = it > 0
    rows.append([name, f"{100 * ok.mean():.1f}", f"{it[ok].mean():.2f}" if ok.any() else "nan"])
    print(rows[-1], flush=True)
    for width, height, fs in ((58, 44, 6), (88, 60, 6.5)):
        small = width < 60
        plt.rcParams.update({"font.family": "Arial", "font.size": fs, "axes.linewidth": 0.5,
                             "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
                             "ytick.major.size": 2, "mathtext.fontset": "custom", "mathtext.it": "Arial:italic",
                             "mathtext.rm": "Arial", "axes.unicode_minus": False, "axes.labelsize": fs + 1})
        fig = plt.figure(figsize=(width / 25.4, height / 25.4))
        ax = fig.add_axes([0.17, 0.19, 0.555, 0.75] if small else [0.12, 0.15, 0.68, 0.81])
        cax = fig.add_axes([0.75, 0.19, 0.07, 0.75] if small else [0.83, 0.15, 0.045, 0.81])
        m = ax.pcolormesh(ds, vs, it, cmap=cmap, vmin=-1, vmax=MAX_ITER, shading="nearest", rasterized=True)
        if ok.mean() > 0.01:
            from scipy.ndimage import median_filter
            field = median_filter(np.where(ok, it, MAX_ITER + 5), size=5)
            cs = ax.contour(ds, vs, field, levels=[lv - 0.5 for lv in LEVELS[name]], colors="k", linewidths=0.4)
            ax.clabel(cs, fmt=lambda lv: f"{lv + 0.5:.0f}", fontsize=fs - 1, inline=True, inline_spacing=1)
        ax.set_xlabel(r"$\theta_2$ [rad]", labelpad=1)
        ax.set_ylabel(r"v$_2$ [pu]", labelpad=1)
        ax.set_xlim(ds[0], ds[-1]); ax.set_ylim(0, 5)
        ax.set_xticks([-1, 0, 1] if small else np.arange(-1.5, 1.6, 0.5))
        ax.set_yticks(range(6) if small else np.arange(0, 5.1, 0.5))
        ax.tick_params(direction="out", pad=1.5)
        cb = fig.colorbar(m, cax=cax, ticks=[0, 5, 10, 15, 20, 25])
        cb.set_label("Iterations to converge", labelpad=1)
        cb.outline.set_linewidth(0.5)
        fig.savefig(out / f"{name}_{width}mm.png", dpi=600)
        plt.close(fig)
with (root / "results" / "journal" / "roa2bus_oldstyle.csv").open("w", newline="") as fh:
    csv.writer(fh).writerows([["method", "converged_percent", "mean_iterations"], *rows])
