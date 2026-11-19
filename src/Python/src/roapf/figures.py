"""Publication figures, drawn at final print size (see style.py)."""

from __future__ import annotations

import csv
import tempfile
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from . import cases as c
from . import roa2bus, style
from .largescale import matches, perturbed_guess, reference
from .network import Network
from .plots import VOLTAGE_CMAP, _bus_kinds, _layout_for, _voltage_field, plot_voltage_map_3d
from .solvers import solve

# label -> (solver name, keyword arguments)
SOLVERS = {
    "NR": ("nr", {}), "NR-SGD": ("nr-sgd", {}), "NR-LS": ("nr-ls", {}), "LM": ("lm", {}),
    "LI": ("li", {}), "SGD": ("sgd", {}), "ALI": ("ali", {}), "NR-LI": ("nr-li", {}), "NR-ALI": ("nr-ali", {}),
    "NR-ALI-W": ("w-nr-ali", {}), "NR-ALI-G": ("w-nr-ali", {"v_guard": 0.5}),
    "NR-ALI-GC": ("w-nr-ali", {"v_guard": 0.5, "dtheta_cap": 0.5, "restore": 0.5}),
    "CN-RK4A": ("cn-rk4a", {}),
}
LETTERS = "abcdefghijklmnopqrstuvwxyz"
# Milano's temperature-map style: red branches, blue generators, green loads, red transit buses
BRANCH = "#C00000"
KIND = {"Generator": "blue", "Load": "green", "Transit": "red"}


def run(net: Network, label: str, **kw):
    method, base = SOLVERS[label]
    return solve(net, method, **{"tol": 1e-8, **base, **kw})


def _curve(ax, x, y, label, ok=True, **kw):
    st = style.METHODS[label]
    return ax.plot(x, y, color=st["color"], marker=st["marker"], linestyle="-" if ok else ":",
                   markeredgewidth=0, **kw)[0]


def _handles(labels):
    return [Line2D([], [], color=style.METHODS[k]["color"], marker=style.METHODS[k]["marker"],
                   markeredgewidth=0, label=style.METHODS[k]["label"]) for k in labels]


# --------------------------------------------------------------------------- region of attraction
def roa_maps(keys, titles, path, ncols: int = 2, n: int = 256, cache: dict | None = None) -> Path:
    """Iteration-count maps of the 2-bus system; black = no convergence within the limit."""
    style.apply()
    cache = {} if cache is None else cache
    nrows = -(-len(keys) // ncols)
    width = style.DOUBLE if ncols > 1 else style.SINGLE
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 0.36 * width * nrows * (2 / ncols) + 0.25),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    cmap = ListedColormap(plt.get_cmap(style.CMAP)(np.linspace(1.0, 0.12, 256)))
    cmap.set_bad("black")
    mesh = None
    for k, (ax, key, title) in enumerate(zip(axes.flat, keys, titles)):
        sw = cache.setdefault(key, roa2bus.sweep(key, n=n))
        mesh = ax.pcolormesh(sw.delta, sw.v, np.ma.masked_invalid(sw.iterations), cmap=cmap, vmin=1, vmax=25,
                             shading="nearest", rasterized=True)
        ax.set_ylim(0, 3)
        ax.set_title(f"{title}  ({100 * sw.converged_fraction:.1f}% converged)", loc="left", pad=3)
        style.panel_label(ax, LETTERS[k], x=-0.08)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r"Initial angle $\theta_2$ (rad)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Initial voltage $v_2$ (p.u.)")
    cb = fig.colorbar(mesh, ax=axes, extend="max", fraction=0.025, pad=0.015, aspect=35)
    cb.set_label("Iterations to converge (black: not converged)")
    cb.outline.set_linewidth(0.5)
    return style.save(fig, path)


# --------------------------------------------------------------------------- mechanism
def difficult_start(net: Network, level: float, limit: int = 40):
    """First random initial guess from which NR fails and the guarded hybrid succeeds."""
    v_ref = reference(net, 1e-8)
    for seed in range(200):
        v0 = perturbed_guess(net, level, np.random.default_rng(seed))
        if (not matches(run(net, "NR", v0=v0, max_iter=limit), v_ref)
                and matches(run(net, "NR-ALI-GC", v0=v0, max_iter=limit), v_ref)):
            return v0, v_ref
    raise RuntimeError("no such initial guess found")


def mechanism(systems, path, labels=("NR", "LI", "SGD", "NR-SGD", "NR-LI", "NR-ALI-GC"), limit: int = 40,
              first: int = 0, legend: bool = True) -> Path:
    """Mismatch, cond(J) and lowest voltage along the iterations; one row per system.

    ``systems`` is a list of (network, title, perturbation level).
    """
    style.apply()
    rows = len(systems)
    fig, axes = plt.subplots(rows, 3, figsize=(style.DOUBLE, 1.75 * rows + 0.2), squeeze=False, sharex="row",
                             constrained_layout=True)
    for r, (net, title, level) in enumerate(systems):
        v0, v_ref = difficult_start(net, level, limit)
        cond_sol = run(net, "NR", v0=v_ref, track=True, max_iter=1, tol=0.0).cond[0]
        for label in labels:
            res = run(net, label, v0=v0, max_iter=limit, track=True)
            ok = matches(res, v_ref)
            k = np.arange(len(res.cond))
            _curve(axes[r, 0], np.arange(len(res.history)), np.clip(res.history, 1e-13, 1e12), label, ok)
            _curve(axes[r, 1], k, np.clip(res.cond, 1, 1e16), label, ok)
            _curve(axes[r, 2], k, np.clip(res.vmin, 0, 1.5), label, ok)
        axes[r, 0].set_yscale("log")
        axes[r, 1].set_yscale("log")
        axes[r, 1].axhline(cond_sol, color="k", linestyle="--", linewidth=0.5)
        axes[r, 2].axhline(0.5, color="k", linestyle="--", linewidth=0.5)
        axes[r, 0].set_ylabel("Largest mismatch (p.u.)")
        axes[r, 1].set_ylabel("Condition number of $J$")
        axes[r, 2].set_ylabel("Lowest bus voltage (p.u.)")
        for col in range(3):
            axes[r, col].grid(True)
            axes[r, col].set_xlabel("Iteration")
            style.panel_label(axes[r, col], LETTERS[first + 3 * r + col], x=-0.16)
        if rows > 1:
            axes[r, 0].set_title(f"{title}, $s$ = {level:g}", loc="left", pad=3)
    if legend:
        style.legend_above(fig, _handles(labels), [style.METHODS[k]["label"] for k in labels], ncol=min(len(labels), 6))
    return style.save(fig, path)


# --------------------------------------------------------------------------- sampled region of attraction
def read_robustness(results: Path, names: dict) -> OrderedDict:
    """case -> method -> {level: (success %, mean iterations)} from results/journal/robust_*.csv."""
    out = OrderedDict()
    for case in names:
        for f in sorted(results.glob(f"robust_{case}*.csv")):
            for r in csv.DictReader(f.open()):
                if r["method"] == "NR-SGD" and "ACTIVSg" in case and "NR-SGD" not in f.name:
                    continue  # placeholder rows; the real run is in the *_NR-SGD.csv file
                if r["method"] == "LM" and "ACTIVSg2000" in case:
                    continue  # not run on the 2000-bus system
                out.setdefault(case, OrderedDict()).setdefault(r["method"], {})[float(r["level"])] = (
                    float(r["converged_percent"]), float(r["mean_iterations"]))
    return out


def sampled_roa(rob: OrderedDict, names: dict, path,
                labels=("NR", "LI", "SGD", "NR-SGD", "NR-LI", "NR-ALI-GC"), first: int = 0) -> Path:
    style.apply()
    fig, axes = plt.subplots(2, 3, figsize=(style.DOUBLE, 3.9), sharey=True, constrained_layout=True)
    for k, (ax, (case, methods)) in enumerate(zip(axes.flat, rob.items())):
        for label in labels:
            if label in methods:
                lv = sorted(methods[label])
                _curve(ax, lv, [methods[label][x][0] for x in lv], label, clip_on=False, zorder=3)
        ax.set_title(names[case], loc="left", pad=3)
        ax.set_ylim(0, 100)
        ax.grid(True)
        ax.set_xlabel("Perturbation of the initial guess, $s$")
        style.panel_label(ax, LETTERS[first + k], x=-0.1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Initial guesses that\\nreach the solution (%)".replace("\\n", "\n"))
    legend_ax = axes.flat[len(rob)]
    legend_ax.axis("off")
    legend_ax.legend(handles=_handles(labels), loc="center left", handlelength=2.2, labelspacing=0.9)
    for ax in axes.flat[len(rob) + 1:]:
        ax.set_visible(False)
    return style.save(fig, path)


# --------------------------------------------------------------------------- ill-conditioned systems
def ill_conditioned(cases, path, labels=("NR", "LI", "SGD", "NR-SGD", "NR-LI", "NR-ALI-GC"), first: int = 0) -> Path:
    """Mismatch histories from a flat start on ill-conditioned systems with a known solution.

    ``cases`` is a list of (IllCase, title).
    """
    style.apply()
    limits = {"LI": 3000, "SGD": 3000, "ALI": 3000, "NR": 100, "NR-SGD": 100, "LM": 100}
    fig, axes = plt.subplots(1, len(cases), figsize=(style.DOUBLE, 2.5), sharey=True, squeeze=False,
                             constrained_layout=True)
    for k, (ax, (ic, title)) in enumerate(zip(axes[0], cases)):
        for label in labels:
            res = run(ic.net, label, max_iter=limits.get(label, 300))
            ok = matches(res, ic.v_solution)
            h = np.clip(np.array(res.history[1:]), 1e-13, 1e6)
            every = max(1, h.size // 12) if h.size > 40 else 1
            _curve(ax, np.arange(1, h.size + 1), h, label, ok, markevery=every)
            print(f"{ic.name} {label}: reach={ok} converged={res.converged} iterations={res.iterations}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 3000)
        ax.set_ylim(1e-12, 1e6)
        ax.axhline(1e-8, color="k", linestyle="--", linewidth=0.5)
        ax.grid(True)
        ax.set_xlabel("Iteration")
        ax.set_title(title, loc="left", pad=3)
        style.panel_label(ax, LETTERS[first + k], x=-0.07)
    axes[0, 0].set_ylabel("Largest mismatch (p.u.)")
    style.legend_above(fig, _handles(labels), [style.METHODS[k]["label"] for k in labels], ncol=3)
    return style.save(fig, path)


# --------------------------------------------------------------------------- voltage maps
def _draw_map(ax, net: Network, v: np.ndarray, grid: int = 500, smooth: float = 0.02):
    lay = _layout_for(net)
    x0, y0, x1, y1 = lay.frame
    _, _, z, lo, hi, _ = _voltage_field(net, v, lay, grid, smooth)  # the colour field fills the whole frame
    big = net.case.n_bus > 100
    huge = net.case.n_bus > 1000

    def up(y):
        return y0 + y1 - y

    vm = np.abs(v)
    if VOLTAGE_CMAP != "jet":
        lo, hi = float(vm.min()), float(vm.max())
    im = ax.imshow(z, extent=(x0, x1, y0, y1), origin="lower", cmap=VOLTAGE_CMAP, vmin=lo, vmax=hi,
                   interpolation="bilinear", aspect="auto")
    for line in lay.lines.values():
        px, py = zip(*line)
        ax.plot(px, [up(y) for y in py], color=BRANCH, linewidth=0.15 if huge else 0.3 if big else 0.7)
    if lay.outline:
        ox, oy = zip(*lay.outline)
        ax.plot(ox, [up(q) for q in oy], color="black", linewidth=0.7)
    for xa, ya, xb, yb in ([] if big else lay.bars.values()):  # no bus bars on large systems
        ax.plot([xa, xb], [up(ya), up(yb)], color="black", linewidth=0.4 if huge else 0.7 if big else 0.9)
    ids, kinds = _bus_kinds(net)
    for kind, mask, marker, color in kinds:
        color = KIND[kind]
        if mask.any():
            ax.plot([lay.markers[b][0] for b in ids[mask]], [up(lay.markers[b][1]) for b in ids[mask]],
                    linestyle="none", marker=marker, markersize=1.0 if huge else 1.8 if big else 3.5,
                    markerfacecolor=color, markeredgecolor="black", markeredgewidth=0.1 if big else 0.4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    return im


def voltage_maps(systems, path) -> Path:
    """One row per system: 2-D temperature map and 3-D voltage surface. ``systems``: (net, v, title)."""
    style.apply()
    rows = len(systems)
    fig, axes = plt.subplots(rows, 2, figsize=(style.DOUBLE, 2.75 * rows + 0.3), squeeze=False,
                             constrained_layout=True, gridspec_kw={"width_ratios": [1.0, 1.05]})
    with tempfile.TemporaryDirectory() as tmp:
        for r, (net, v, title) in enumerate(systems):
            im = _draw_map(axes[r, 0], net, v)
            cb = fig.colorbar(im, ax=axes[r, 0], fraction=0.045, pad=0.02)
            cb.set_label("Voltage magnitude (p.u.)")
            cb.outline.set_linewidth(0.5)
            axes[r, 0].set_title(title, loc="left", pad=3)
            axes[r, 1].imshow(plt.imread(plot_voltage_map_3d(net, v, Path(tmp))))
            axes[r, 1].axis("off")
            style.panel_label(axes[r, 0], LETTERS[2 * r], x=-0.02)
            style.panel_label(axes[r, 1], LETTERS[2 * r + 1], x=0.02)
    handles = [Line2D([], [], linestyle="none", marker=m, markerfacecolor=col, markeredgecolor="black",
                      markeredgewidth=0.4, markersize=4, label=lab)
               for lab, m, col in (("Generator", "s", KIND["Generator"]), ("Load", "v", KIND["Load"]),
                                   ("Transit", "o", KIND["Transit"]))]
    handles += [Line2D([], [], color=BRANCH, linewidth=0.8, label="Branch"),
                Line2D([], [], color="black", linewidth=0.9, label="Bus bar")]
    if all(net.case.n_bus > 100 for net, _, _ in systems):
        handles = handles[:-1]  # bus bars are drawn for small systems only
    style.legend_above(fig, handles, [h.get_label() for h in handles], ncol=5)
    return style.save(fig, path)


def is_pq_gen(net: Network) -> np.ndarray:  # kept for scripts that want the generator mask
    return np.isin(np.arange(net.case.n_bus), np.concatenate([net.ref, net.pv])) & (net.case.bus[:, c.PD] == 0)


def solvability_map(path, n: int = 400) -> Path:
    """Loads of the two-bus system for which Newton-Raphson converges from a flat start,
    with the analytical solvability boundary Q = B/4 - P^2/B."""
    style.apply()
    sv = roa2bus.solvability(n=n)
    fig, ax = plt.subplots(figsize=(style.SINGLE * 0.98, 2.35), constrained_layout=True)
    ax.pcolormesh(sv.p * 100, sv.q * 100, sv.solvable.T.astype(float), shading="nearest", rasterized=True,
                  cmap=ListedColormap(["#F3D9CC", "#CFE5F2"]), vmin=0, vmax=1)
    ax.plot(sv.p_boundary * 100, sv.q_boundary * 100, color="black", linewidth=1.0)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 300)
    ax.text(150, 95, "Solvable\nNewton-Raphson converges\nfrom a flat start", ha="center", va="center",
            color=style.BLUE)
    ax.text(455, 200, "Unsolvable\nno power flow solution", ha="center", va="center", color=style.VERMILLION)
    # only the arrow: the label "Boundary Sigma" is typeset by LaTeX in Fig1_solvability.tex, so that
    # the bold maths Sigma is identical in both panels
    ax.annotate("", xy=(300, 160), xytext=(345, 252), ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", linewidth=0.6, color="black", shrinkA=1, shrinkB=1))
    ax.set_xlabel("Active power load (MW)")
    ax.set_ylabel("Reactive power load (MVAr)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    return style.save(fig, path)
