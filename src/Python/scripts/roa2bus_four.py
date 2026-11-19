"""Two-bus regions of attraction of the four methods compared in the paper:
Newton-Raphson, Newton-Raphson with Landweber fallback, Newton-Raphson with
stochastic gradient descent (re-implementation) and the proposed guarded method.

Every run is classified by where it ends: the operable (high-voltage) solution,
the low-voltage solution, or no convergence within 25 iterations.

    uv run python scripts/roa2bus_four.py

Output: figures/Fig2_roa.pdf/.png, results/journal/roa2bus_four.csv
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
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from run_roa2bus_guarded import jac, mis, run as run_basic  # noqa: E402

B, TOL, MAX_ITER, GUARD = 10.0, 1e-5, 25, 0.5
FLAT = np.array([0.0, 1.0])


def lw(z):
    J = jac(z)
    return z + (J.T @ mis(z)) / (np.linalg.norm(J, 2) ** 2 + 2.2e-16)


def proposed(x, max_iter=MAX_ITER, guard=GUARD, cap=0.5, m=2, rho=0.25, restore=0.5):
    """Guarded Newton-Landweber method: voltage guard, angle cap, look-ahead trial,
    accelerated Landweber step and restoration step (as roapf.solvers, w-nr-ali)."""
    trial, phi, xprev, k = False, math.inf, None, 0
    for it in range(1, max_iter + 1):
        F = mis(x)
        if not np.all(np.isfinite(F)):
            return None, x
        if np.max(np.abs(F)) < TOL:
            return it, x
        f0 = np.linalg.norm(F)
        J = jac(x)
        ok = abs(np.linalg.det(J)) > 1e-12 * max(np.linalg.norm(J, 1) ** 2, 1e-300)
        dx = np.linalg.solve(J, F) if ok else None
        adm = dx is not None and np.all(np.isfinite(dx)) and x[1] + dx[1] >= guard
        if adm and abs(dx[0]) > cap:
            dx = dx * (cap / abs(dx[0]))
        fn = np.linalg.norm(mis(x + dx)) if adm else math.inf

        def fallback():
            nonlocal xprev, k
            k += 1
            z = x if xprev is None else x + (k - 1) / (k + 2) * (x - xprev)
            cand = lw(z)
            if xprev is not None and not np.linalg.norm(mis(cand)) < f0:
                k, cand = 1, lw(x)
            if not cand[1] >= guard:  # restoration step
                xprev, k = None, 0
                return x + restore * (FLAT - x)
            xprev = x
            return cand

        if trial:
            if adm and fn < fbar:
                x, trial, xprev, k = x + dx, False, None, 0
            elif adm and count < m and math.isfinite(fn):
                x, count = x + dx, count + 1
            else:
                x, phi, trial = xbar, fbar, False
        elif adm and fn < f0:
            x, xprev, k = x + dx, None, 0
        elif adm and f0 <= rho * phi:
            xbar, fbar, count, trial = fallback(), f0, 1, True
            x = x + dx
        else:
            x = fallback()
    return None, x


def nr_sgd(x, max_iter=MAX_ITER, nr_iters=10, sgd_steps=50, eta=0.01, b1=0.9, b2=0.999):
    """Newton-Raphson with Adam gradient phases (as roapf.solvers.solve_nr_sgd;
    the single load bus is drawn in every stochastic step)."""
    mom, sec, t, it = np.zeros(2), np.zeros(2), 0, 0
    if np.max(np.abs(mis(x))) < TOL:
        return 1, x
    while it < max_iter:
        x_start, ok, norm = x.copy(), True, math.inf
        for _ in range(nr_iters):
            if it >= max_iter:
                break
            it += 1
            J = jac(x)
            if not abs(np.linalg.det(J)) > 0:
                ok = False
                break
            x = x + np.linalg.solve(J, mis(x))
            norm = np.max(np.abs(mis(x)))
            if not np.isfinite(norm) or norm > 1e8:
                ok = False
                break
            if norm < TOL:
                return it + 1, x
        if it >= max_iter:
            break
        if not ok or norm > 1e3:
            x = x_start
        for _ in range(sgd_steps):
            g = -(jac(x).T @ mis(x))
            t += 1
            mom = b1 * mom + (1 - b1) * g
            sec = b2 * sec + (1 - b2) * g * g
            x = x - eta * (mom / (1 - b1**t)) / (np.sqrt(sec / (1 - b2**t)) + 1e-8)
    return None, x


METHODS = [
    ("nr", "Newton–Raphson", lambda x: run_basic(x, "nr", MAX_ITER)),
    ("nr-li", "Newton–Raphson + Landweber", lambda x: run_basic(x, "hybrid", MAX_ITER)),
    ("nr-sgd", "Newton–Raphson + stochastic gradient descent", nr_sgd),
    ("proposed", "Guarded Newton–Landweber (proposed)", proposed),
]


def main(n=256):
    np.seterr(all="ignore")
    root = Path(__file__).resolve().parents[1]
    vs, ds = np.linspace(0, 5, n), np.linspace(-math.pi / 2, math.pi / 2, n)
    maps, rows = {}, []
    for key, title, fn in METHODS:
        cache = root / "results" / "journal" / f"roa2bus_four_{key}.npy"
        if cache.exists():
            it = np.load(cache)
        else:
            it = np.zeros((n, n))  # > 0 iterations to the operable solution, -1 low-voltage solution, 0 none
            for i, v in enumerate(vs):
                for j, d in enumerate(ds):
                    try:
                        k, x = fn(np.array([d, v]))
                    except (FloatingPointError, OverflowError, np.linalg.LinAlgError, ZeroDivisionError):
                        k = None
                    if k is not None and np.all(np.isfinite(x)):
                        it[i, j] = k if abs(x[1]) > GUARD else -1
            np.save(cache, it)
        maps[key] = it
        op = it > 0
        rows.append([key, f"{100 * op.mean():.1f}", f"{100 * (it < 0).mean():.1f}", f"{100 * (it == 0).mean():.1f}",
                     f"{it[op].mean():.2f}", int(it[op].max())])
        print(rows[-1], flush=True)
    with (root / "results" / "journal" / "roa2bus_four.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows([["method", "operable_percent", "low_voltage_percent", "not_converged_percent",
                                   "mean_iterations_operable", "max_iterations_operable"], *rows])

    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "mathtext.fontset": "custom", "mathtext.it": "Arial:italic", "mathtext.rm": "Arial",
                         "pdf.fonttype": 42})
    grey = (0.62, 0.62, 0.62, 1.0)
    cmap = ListedColormap(plt.get_cmap("jet")(np.linspace(0, 1, MAX_ITER)))
    cmap.set_under("black")
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 128 / 25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.845, bottom=0.085, top=0.955, wspace=0.07, hspace=0.17)
    mesh = None
    for ax, (key, title, _), letter in zip(axes.flat, METHODS, "abcd"):
        it = maps[key]
        mesh = ax.pcolormesh(ds, vs, np.ma.masked_where(it < 0, it), cmap=cmap, vmin=0.5, vmax=MAX_ITER + 0.5,
                             shading="nearest", rasterized=True)
        ax.pcolormesh(ds, vs, np.ma.masked_where(it >= 0, it), cmap=ListedColormap([grey]), shading="nearest",
                      rasterized=True)
        # labelled iteration contours, as in the maps of the ROA repository (smooth basins only)
        from scipy.ndimage import median_filter
        field = median_filter(np.where(it > 0, it, MAX_ITER + 5), size=5)
        cs = ax.contour(ds, vs, field, levels=[4.5, 5.5, 6.5, 7.5], colors="k", linewidths=0.45)
        ax.clabel(cs, fmt=lambda lv: f"{lv + 0.5:.0f}", fontsize=6, inline=True, inline_spacing=2)
        ax.set_xlim(ds[0], ds[-1]); ax.set_ylim(0, 5)
        ax.set_xticks(np.arange(-1.5, 1.6, 0.5)); ax.set_yticks(range(6))
        ax.tick_params(direction="out", pad=2)
        ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        ax.text(0.05, 1.03, f"{title}  ({100 * (it > 0).mean():.0f}%)",
                transform=ax.transAxes, va="bottom")
    for ax in axes[-1]:
        ax.set_xlabel(r"Starting angle $\theta_2$ (rad)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Starting voltage $v_2$ (p.u.)")
    cax = fig.add_axes([0.862, 0.36, 0.016, 0.595])
    cb = fig.colorbar(mesh, cax=cax, ticks=[1, 5, 10, 15, 20, 25])
    cb.set_label("Iterations to the operable solution")
    cb.outline.set_linewidth(0.5)
    fig.legend(handles=[Patch(facecolor=grey, edgecolor="k", linewidth=0.4, label="low-voltage\nsolution"),
                        Patch(facecolor="black", edgecolor="k", linewidth=0.4, label="no convergence")],
               loc="lower left", bbox_to_anchor=(0.856, 0.09), frameon=False, handlelength=1.2, handleheight=1.0,
               borderaxespad=0, labelspacing=0.8)
    out = root / "figures" / "Fig2_roa"
    fig.savefig(out.with_suffix(".pdf"), dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(out.with_suffix(".pdf"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 256)
