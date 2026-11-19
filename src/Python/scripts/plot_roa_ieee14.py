"""Draw the IEEE 14-bus region-of-attraction maps computed by roa_ieee14.py,
one row per swept load bus.

    uv run python scripts/plot_roa_ieee14.py <n> <bus> [<bus> ...]

Output: figures/Fig_roa14.pdf/.png
"""
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import median_filter

n, buses = int(sys.argv[1]), [int(b) for b in sys.argv[2:]]
root = Path(__file__).resolve().parents[1]
MAX_ITER = 25
vs, ds = np.linspace(0, 5, n), np.linspace(-math.pi / 2, math.pi / 2, n)
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "mathtext.fontset": "custom", "mathtext.it": "Arial:italic", "mathtext.rm": "Arial",
                     "pdf.fonttype": 42})
grey = (0.62, 0.62, 0.62, 1.0)
cmap = ListedColormap(plt.get_cmap("jet")(np.linspace(0, 1, MAX_ITER)))
cmap.set_under("black")
rows = len(buses)
fig, axes = plt.subplots(rows, 2, figsize=(180 / 25.4, (62 * rows + 6) / 25.4), sharey=True, squeeze=False)
fig.subplots_adjust(left=0.06, right=0.845, bottom=0.16 / rows + 0.01, top=1 - 0.08 / rows, wspace=0.07, hspace=0.42)
titles = (("nr", "Newton\u2013Raphson"), ("proposed", "Guarded NR + LI (proposed)"))
letters = iter("abcdefgh")
for r, bus in enumerate(buses):
    for ax, (key, title) in zip(axes[r], titles):
        it = np.load(root / "results" / "journal" / "roa14" / f"bus{bus}_{key}_{n}.npy")
        mesh = ax.pcolormesh(ds, vs, np.ma.masked_where(it < 0, it), cmap=cmap, vmin=0.5, vmax=MAX_ITER + 0.5,
                             shading="nearest", rasterized=True)
        ax.pcolormesh(ds, vs, np.ma.masked_where(it >= 0, it), cmap=ListedColormap([grey]), shading="nearest",
                      rasterized=True)
        field = median_filter(np.where(it > 0, it, MAX_ITER + 5), size=5)
        cs = ax.contour(ds, vs, field, levels=[4.5, 5.5, 6.5, 7.5], colors="k", linewidths=0.45)
        ax.clabel(cs, fmt=lambda lv: f"{lv + 0.5:.0f}", fontsize=6, inline=True, inline_spacing=2)
        ax.set_xlim(ds[0], ds[-1]); ax.set_ylim(0, 5)
        ax.set_xticks(np.arange(-1.5, 1.6, 0.5)); ax.set_yticks(range(6))
        ax.tick_params(direction="out", pad=2)
        ax.set_xlabel(rf"Starting angle $\theta_{{{bus}}}$ (rad)")
        ax.text(0.0, 1.03, next(letters), transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        ax.text(0.05, 1.03, f"{title}, bus {bus}  ({100 * (it > 0).mean():.0f}%)", transform=ax.transAxes,
                va="bottom")
        print(bus, key, f"{100 * (it > 0).mean():.1f}% operating point, {100 * (it < 0).mean():.1f}% other, "
                        f"{100 * (it == 0).mean():.1f}% none, mean {it[it > 0].mean():.2f}, max {it.max():.0f}")
    axes[r, 0].set_ylabel(rf"Starting voltage $v_{{{bus}}}$ (p.u.)")
top = axes[0, 1].get_position()
cax = fig.add_axes([0.862, top.y0, 0.016, top.height])
cb = fig.colorbar(mesh, cax=cax, ticks=[1, 5, 10, 15, 20, 25])
cb.set_label("Iterations to the operating point")
cb.outline.set_linewidth(0.5)
bot = axes[-1, 1].get_position()
fig.legend(handles=[Patch(facecolor=grey, edgecolor="k", linewidth=0.4, label="other solution"),
                    Patch(facecolor="black", edgecolor="k", linewidth=0.4, label="no convergence")],
           loc="lower left", bbox_to_anchor=(0.856, bot.y0 + (0.5 * bot.height if rows > 1 else 0)), frameon=False,
           handlelength=1.2, handleheight=1.0, borderaxespad=0, labelspacing=0.8)
out = root / "figures" / "Fig_roa14"
fig.savefig(out.with_suffix(".pdf"), dpi=600)
fig.savefig(out.with_suffix(".png"), dpi=300)
print(out.with_suffix(".pdf"))
