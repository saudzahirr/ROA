"""Figures for the manuscript (Times New Roman, 12 pt, as in the MATLAB scripts)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, ListedColormap  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from . import cases as c  # noqa: E402
from .layouts import LAYOUTS, Layout  # noqa: E402
from .network import Network  # noqa: E402
from .roa2bus import Solvability, Sweep  # noqa: E402
from .solvers import LABELS, Result  # noqa: E402

plt.rcParams.update({
    "font.family": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "mathtext.fontset": "stix",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# colour map of the voltage maps; "jet" reproduces the rainbow scale of Milano's temperature maps
VOLTAGE_CMAP = "jet"
LINE_COLOR = "#FFFFFF" if VOLTAGE_CMAP != "jet" else "#cc1010"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def plot_roa(sw: Sweep, out_dir: Path, title: bool = True) -> Path:
    """Iteration-count map over the initial guess; black = no convergence."""
    m = sw.method
    data = np.where(np.isnan(sw.iterations), -1.0, sw.iterations)
    vmax = m.max_iter
    if m.scale_to_data and np.any(data > 0):
        vmax = int(data.max())
    cmap = ListedColormap(np.vstack([[0, 0, 0, 1], plt.get_cmap("jet")(np.linspace(0, 1, m.max_iter))]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    mesh = ax.pcolormesh(sw.delta, sw.v, data, cmap=cmap, vmin=-1, vmax=vmax, shading="nearest",
                         rasterized=True)
    levels = [lv for lv in m.levels if lv <= vmax and data.min() < lv < data.max()]
    if levels:
        cs = ax.contour(sw.delta, sw.v, data, levels=sorted(set(levels)), colors="k", linewidths=1.2)
        ax.clabel(cs, fmt="%d", fontsize=12)
    fig.colorbar(mesh, ax=ax)
    ax.set_xlabel(r"$\theta_2$ [rad]")
    ax.set_ylabel(r"$v_2$ [pu]")
    if title:
        ax.set_title(f"Region of Attraction of {m.title}")
    return _save(fig, out_dir / f"{m.filename}.png")


def plot_solvability(s: Solvability, out_dir: Path, title: bool = True, s_base: float = 100.0) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5.25))
    ax.pcolormesh(s.p * s_base, s.q * s_base, s.solvable.T.astype(float),
                  cmap=ListedColormap([[1, 0, 0], [0, 1, 0]]), vmin=0, vmax=1, shading="nearest",
                  rasterized=True)
    ax.plot(s.p_boundary * s_base, s.q_boundary * s_base, "k-", linewidth=2)
    ax.text(230, 210, r"$\mathbf{\Sigma}$", fontsize=18)
    ax.set_xlim(s.p[0] * s_base, s.p[-1] * s_base)
    ax.set_ylim(s.q[0] * s_base, s.q[-1] * s_base)
    ax.set_xlabel("Real power load P (MW)")
    ax.set_ylabel("Reactive power load Q (MVar)")
    if title:
        ax.set_title("PQ Solvability Region (2-Bus)")
    return _save(fig, out_dir / "SolvabilityBoundary.png")


# --------------------------------------------------------------------------
_STYLE = {
    "nr": dict(color="#0072BD", marker="o", markerfacecolor="blue"),
    "nr-gd": dict(color="#D95319", marker="s", markerfacecolor="red"),
    "nr-li": dict(color="#EDB120", marker="d", markerfacecolor="lime"),
}


def plot_convergence(results: list[Result], case_name: str, out_dir: Path, tol: float,
                     log: bool = True, title: bool = False) -> Path:
    """Maximum power mismatch after each iteration for the compared methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    longest = 1
    for r in results:
        h = np.asarray(r.history, dtype=float)  # h[k] = mismatch after k iterations
        h = h[np.isfinite(h)]
        label = LABELS[r.method] + ("" if r.converged else " (did not converge)")
        ax.plot(np.arange(h.size), h, linewidth=2, markersize=7, label=label, **_STYLE[r.method])
        longest = max(longest, h.size - 1)
    ax.axhline(tol, color="k", linestyle="--", linewidth=0.8)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(1e-15, 1e6)
        ax.text(0.98 * longest, tol * 0.75, ha="right", s= f"Tolerance = $10^{{{int(np.log10(tol))}}}$", va="top")
    ax.set_xticks(np.arange(0, longest + 1, max(1, longest // 15)))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Maximum Power Mismatch (p.u.)")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.6)
    ax.legend(loc="lower left")
    if title:
        ax.set_title(f"Convergence on {case_name}")
    return _save(fig, out_dir / f"Convergence{'Log' if log else ''}-{case_name}.png")


# --------------------------------------------------------------------------
_MAP_RC = {"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
           "axes.formatter.use_mathtext": True, "font.size": 12}


def _auto_positions(graph: nx.Graph) -> dict:
    """Bus positions that keep connected buses close and spread the network evenly.

    Each connected component is laid out with the stress-minimising Kamada-Kawai
    method (distances on the drawing follow the number of branches between two
    buses), started from a force-directed layout. Above 3,000 buses the
    force-directed layout alone is used.
    """
    n = graph.number_of_nodes()
    pos = nx.spring_layout(graph, seed=1, iterations=300 if n <= 800 else 120)
    if n <= 3000:
        pos = nx.kamada_kawai_layout(graph, pos=pos)
    return pos


COORD_DIR = Path(__file__).resolve().parents[2] / "cases" / "coordinates"


def _closed_outline(lay) -> np.ndarray | None:
    """The outline as an array of points if it is a complete, closed boundary."""
    if not lay.outline:
        return None
    poly = np.array(lay.outline, dtype=float)
    return None if np.isnan(poly).any() or len(poly) < 4 else poly


def _outside_mask(lay, X, Y) -> np.ndarray | None:
    """True where the grid point (X, Y) of the voltage field lies outside the outline."""
    poly = _closed_outline(lay)
    if poly is None:
        return None
    _, y0, _, y1 = lay.frame
    path = MplPath(np.column_stack([poly[:, 0], y0 + y1 - poly[:, 1]]))  # the field has y pointing up
    return ~path.contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)


def _route_inside(poly: np.ndarray, markers: dict, pairs: list) -> dict:
    """Branches whose straight segment would leave the outlined area are routed inside it.

    The route is the shortest path that stays inside the outline: a polyline from
    one bus to the other through (slightly inset) corner points of the boundary.
    """
    path = MplPath(poly)
    t = np.linspace(0.03, 0.97, 24)[None, :, None]

    def visible(a, pts):
        seg = a[None, None, :] + t * (pts[:, None, :] - a[None, None, :])
        return path.contains_points(seg.reshape(-1, 2)).reshape(len(pts), -1).all(axis=1)

    need = []
    for f, to in pairs:
        if f in markers and to in markers:
            a, b = np.array(markers[f]), np.array(markers[to])
            if path.contains_point(a) and path.contains_point(b) and not visible(a, b[None, :])[0]:
                need.append((f, to))
    if not need:
        return {}
    step = poly.mean(axis=0) - poly
    inset = poly + 4.0 * step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-9)
    inset = inset[path.contains_points(inset)]
    graph = nx.Graph()
    for i in range(len(inset) - 1):
        for j in np.flatnonzero(visible(inset[i], inset[i + 1:])):
            graph.add_edge(i, i + 1 + int(j), w=float(np.linalg.norm(inset[i] - inset[i + 1 + j])))
    routes = {}
    for f, to in need:
        a, b = np.array(markers[f]), np.array(markers[to])
        for name, pt in (("a", a), ("b", b)):
            for j in np.flatnonzero(visible(pt, inset)):
                graph.add_edge(name, int(j), w=float(np.linalg.norm(pt - inset[j])))
        try:
            nodes = nx.shortest_path(graph, "a", "b", weight="w")
            routes[(f, to)] = [tuple(a)] + [tuple(inset[k]) for k in nodes[1:-1]] + [tuple(b)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        graph.remove_nodes_from([k for k in ("a", "b") if k in graph])
    return routes


def _geographic_layout(case, path: Path) -> Layout:
    """Diagram on the geographic footprint of a synthetic grid.

    ``path`` is a CSV file with the columns bus, substation, latitude, longitude.
    Buses of one substation are drawn in a small ring around its location, and
    every branch is a straight segment.
    """
    import csv

    rows = list(csv.DictReader(path.open()))
    lat = np.array([float(r["latitude"]) for r in rows])
    lon = np.array([float(r["longitude"]) for r in rows])
    border = path.with_name(path.stem + "_outline.csv")  # optional boundary of the footprint
    o_lat, o_lon = np.empty(0), np.empty(0)
    if border.exists():
        pts = list(csv.DictReader(border.open()))
        o_lat = np.array([float(r["latitude"]) for r in pts])
        o_lon = np.array([float(r["longitude"]) for r in pts])
    lat0, lon0, k = lat.mean(), lon.mean(), np.cos(np.deg2rad(lat.mean()))
    x, y = (lon - lon0) * k, lat - lat0  # equirectangular projection
    ox, oy = (o_lon - lon0) * k, o_lat - lat0
    ax_, ay_ = np.concatenate([x, ox]), np.concatenate([y, oy])
    if ox.size and max(np.ptp(x), np.ptp(y)) < 0.6 * max(np.ptp(ax_), np.ptp(ay_)):
        ax_, ay_ = x, y  # the grid covers only part of the outlined area: zoom in on the grid
    scale = 820.0 / max(np.ptp(ax_), np.ptp(ay_))
    cx, cy = 0.5 * (ax_.max() + ax_.min()), 0.5 * (ay_.max() + ay_.min())
    x, ox = 500.0 + (x - cx) * scale, 500.0 + (ox - cx) * scale
    y, oy = 500.0 - (y - cy) * scale, 500.0 - (oy - cy) * scale  # layout coordinates have y pointing down
    outside = (ox < 5) | (ox > 995) | (oy < 5) | (oy > 995)  # the part of the outline beyond the frame is not drawn
    ox, oy = np.where(outside, np.nan, ox), np.where(outside, np.nan, oy)
    outline = list(zip(ox.tolist(), oy.tolist()))
    groups = {}
    for k, r in enumerate(rows):
        groups.setdefault(r["substation"], []).append(k)
    markers = {}
    for members in groups.values():
        m = len(members)
        for q, k in enumerate(members):
            ang = 2.0 * np.pi * q / m
            r0 = 0.0 if m == 1 else 3.2
            markers[int(rows[k]["bus"])] = (float(x[k] + r0 * np.cos(ang)), float(y[k] + r0 * np.sin(ang)))
    br = case.branch[case.branch[:, c.BR_STATUS] > 0]
    pairs = list(zip(br[:, c.F_BUS].astype(int).tolist(), br[:, c.T_BUS].astype(int).tolist()))
    # buses that are not in the coordinate file are placed next to the buses they are connected to
    missing = [int(b) for b in case.bus[:, c.BUS_I] if int(b) not in markers]
    for _ in range(5):
        for b in list(missing):
            near = [markers[o] for f, t in pairs for o in ((t,) if f == b else (f,) if t == b else ()) if o in markers]
            if near:
                markers[b] = (float(np.mean([q[0] for q in near])) + 3.2, float(np.mean([q[1] for q in near])) + 3.2)
                missing.remove(b)
    for b in missing:
        markers[b] = (500.0, 500.0)
    in_case = {int(b) for b in case.bus[:, c.BUS_I]}  # the coordinate file may list buses the case lacks
    markers = {b: xy for b, xy in markers.items() if b in in_case}
    lines = {}
    for f, t in pairs:
        if f in markers and t in markers:
            lines[(f, t)] = [markers[f], markers[t]]
    if outline and not np.isnan(np.array(outline)).any():  # keep every branch inside a complete boundary
        lines.update(_route_inside(np.array(outline, dtype=float), markers, pairs))
    bars = {b: (px - 2.2, py, px + 2.2, py) for b, (px, py) in markers.items()}
    return Layout(frame=(0, 0, 1000, 1000), bars=bars, markers=markers, lines=lines, outline=outline)


def _layout_for(net: Network) -> Layout:
    """Digitised single-line diagram if there is one, otherwise an automatic diagram.

    Rules of the automatic diagram, chosen so that the connections of a large
    network can be followed by eye:

    1. buses that are connected are placed next to each other (stress layout),
       so that almost all branches are short;
    2. positions are snapped to a regular lattice with one bus per cell, which
       aligns the bus bars in rows and prevents overlapping buses;
    3. every bus is a horizontal bar whose length grows with the number of
       branches connected to it;
    4. branches leave a bar on the side facing the other bus, and their
       connection points are ordered along the bar by the direction of the
       other bus, so that branches of one bus do not cross near the bar;
    5. a branch is a short vertical stub at each end joined by one straight
       (horizontal, vertical or diagonal) segment, as in hand-drawn diagrams.
    """
    case = net.case
    if case.name in LAYOUTS:
        return LAYOUTS[case.name]
    if (COORD_DIR / f"{case.name}.csv").exists():  # synthetic grids with published coordinates
        return _geographic_layout(case, COORD_DIR / f"{case.name}.csv")
    ids = case.bus[:, c.BUS_I].astype(int)
    br = case.branch[case.branch[:, c.BR_STATUS] > 0]
    graph = nx.Graph()
    graph.add_nodes_from(ids.tolist())
    graph.add_edges_from(zip(br[:, c.F_BUS].astype(int).tolist(), br[:, c.T_BUS].astype(int).tolist()))
    pos = _auto_positions(graph)
    nodes = list(pos)
    xy = np.array([pos[b] for b in nodes], dtype=float)
    xy = (xy - xy.min(axis=0)) / np.ptp(xy, axis=0)
    # spread dense clusters over the frame: blend each coordinate with its rank, which keeps
    # the left-right and up-down order of the buses (and so the radial structure)
    rank = np.argsort(np.argsort(xy, axis=0), axis=0) / max(len(nodes) - 1, 1)
    xy = 0.5 * xy + 0.5 * rank

    # rule 2: snap to a lattice, most connected buses first, nearest free cell
    n = len(nodes)
    size = int(np.ceil(np.sqrt(n) * 2.2))
    taken, cell = set(), {}
    for k in sorted(range(n), key=lambda q: -graph.degree(nodes[q])):
        cx, cy = int(round(xy[k, 0] * (size - 1))), int(round(xy[k, 1] * (size - 1)))
        radius = 0
        while True:
            ring = [(cx + dx, cy + dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
                    if max(abs(dx), abs(dy)) == radius and 0 <= cx + dx < size and 0 <= cy + dy < size
                    and (cx + dx, cy + dy) not in taken]
            if ring:
                best = min(ring, key=lambda t: (t[0] - xy[k, 0] * (size - 1)) ** 2 + (t[1] - xy[k, 1] * (size - 1)) ** 2)
                taken.add(best)
                cell[nodes[k]] = best
                break
            radius += 1
    step = 900.0 / (size - 1)
    markers = {b: (50.0 + col * step, 50.0 + (size - 1 - row) * step) for b, (col, row) in cell.items()}

    # rule 3: bar length grows with the number of connected branches
    half = {b: step * min(0.42, 0.14 + 0.05 * graph.degree(b)) for b in nodes}
    bars = {b: (x - half[b], y, x + half[b], y) for b, (x, y) in markers.items()}

    # rule 4: connection points along each bar, ordered by the direction of the other bus
    attach = {}
    for b in nodes:
        x0, y0 = markers[b]
        nbrs = sorted(graph.neighbors(b), key=lambda o: (markers[o][0] - x0, markers[o][1]))
        m = len(nbrs)
        for q, o in enumerate(nbrs):
            frac = 0.0 if m == 1 else -0.8 + 1.6 * q / (m - 1)
            attach[(b, o)] = x0 + frac * half[b]

    # rule 5: stub - straight segment - stub
    stub = 0.22 * step
    lines = {}
    for f, t in graph.edges:
        (_, ya), (_, yb) = markers[f], markers[t]
        xa, xb = attach[(f, t)], attach[(t, f)]
        if ya == yb:  # same row: leave both bars upwards and join above them
            lines[(f, t)] = [(xa, ya), (xa, ya - stub), (xb, yb - stub), (xb, yb)]
        else:
            sa = stub if yb > ya else -stub
            lines[(f, t)] = [(xa, ya), (xa, ya + sa), (xb, yb - sa), (xb, yb)]
    return Layout(frame=(0, 0, 1000, 1000), bars=bars, markers=markers, lines=lines)


def _voltage_field(net: Network, v: np.ndarray, lay: Layout, grid: int, smooth: float,
                   tent: bool = False):
    """Interpolate the bus voltage magnitudes over the layout frame.

    Bus coordinates and voltages are collected as (x, y, z) triplets,
    interpolated onto a uniform ``meshgrid`` with ``griddata`` and smoothed
    with a Gaussian filter. The frame is held just below the lowest voltage so
    that the colours fade towards the border.
    """
    ids = net.case.bus[:, c.BUS_I].astype(int)
    vm = dict(zip(ids.tolist(), np.abs(v).tolist()))
    x0, y0, x1, y1 = lay.frame
    span = max(vm.values()) - min(vm.values())
    step = 0.01 if span <= 0.12 else 0.05
    lo = np.floor(round((min(vm.values()) - 0.5 * step) / step, 6)) * step
    hi = np.ceil(round(max(vm.values()) / step, 6)) * step
    pts = [(lay.markers[b][0], y0 + y1 - lay.markers[b][1]) for b in lay.markers]
    val = [vm[b] for b in lay.markers]
    bus_xy, bus_v = np.array(pts), np.array(val)

    def ring(inset, n):
        dx, dy = inset * (x1 - x0), inset * (y1 - y0)
        t = np.linspace(0, 1, n)
        xs, ys = x0 + dx + t * (x1 - x0 - 2 * dx), y0 + dy + t * (y1 - y0 - 2 * dy)
        return ([(x, y0 + dy) for x in xs] + [(x, y1 - dy) for x in xs]
                + [(x0 + dx, y) for y in ys] + [(x1 - dx, y) for y in ys])

    if tent:  # 3-D view: the surface rises from a low frame straight to the buses
        lo = min(vm.values()) - 0.6 * span
        hi = max(vm.values()) + 0.30 * span
    else:
        for px, py in ring(0.06, 14):  # inner ring: distance-weighted mean of the bus voltages
            w = 1.0 / ((bus_xy[:, 0] - px) ** 2 + (bus_xy[:, 1] - py) ** 2 + 1e-9)
            pts.append((px, py))
            val.append(float(w @ bus_v / w.sum()))
    for px, py in ring(0.0, 2 if tent else 14):  # frame (corners only for the 3-D tent)
        pts.append((px, py))
        val.append(lo)
    X, Y = np.meshgrid(np.linspace(x0, x1, grid), np.linspace(y0, y1, grid))
    Z = griddata(np.array(pts), np.array(val), (X, Y), method="linear")
    Z = np.nan_to_num(Z, nan=lo)
    if smooth > 0:
        Z = gaussian_filter(Z, sigma=smooth * grid, mode="nearest")
    return X, Y, np.clip(Z, lo, hi), lo, hi, step


def _bus_kinds(net: Network):
    """Generator / Load / Transit classification used in the legend."""
    case = net.case
    ids = case.bus[:, c.BUS_I].astype(int)
    order = np.argsort(ids)
    pg = np.zeros(case.n_bus)
    on = case.gen[case.gen[:, c.GEN_STATUS] > 0]
    np.add.at(pg, order[np.searchsorted(ids[order], on[:, c.GEN_BUS].astype(int))], on[:, c.PG])
    is_load = (case.bus[:, c.PD] != 0) | (case.bus[:, c.QD] != 0)
    is_gen = ~is_load & ((pg > 0) | np.isin(np.arange(case.n_bus), net.ref))
    return ids, (("Generator", is_gen, "s", "blue"), ("Load", is_load, "v", "green"),
                 ("Transit", ~is_load & ~is_gen, "o", "red"))


def plot_voltage_map(net: Network, v: np.ndarray, out_dir: Path, grid: int = 500,
                     smooth: float = 0.02) -> Path:
    """2-D voltage temperature map (after Milano, Fig. 22.1)."""
    lay = _layout_for(net)
    x0, y0, x1, y1 = lay.frame
    X, Y, Z, lo, hi, step = _voltage_field(net, v, lay, grid, smooth)
    big = net.case.n_bus > 100

    def up(y):
        return y0 + y1 - y

    with plt.rc_context(_MAP_RC):
        fig, ax = plt.subplots(figsize=(8.2, 6.5))
        im = ax.imshow(Z, extent=(x0, x1, y0, y1), origin="lower", cmap="jet", vmin=lo - 0.12 * (hi - lo), vmax=hi,
                       interpolation="bilinear", aspect="auto")
        for path in lay.lines.values():
            px, py = zip(*path)
            ax.plot(px, [up(y) for y in py], color="red", linewidth=0.4 if big else 1.2, zorder=2)
        for xa, ya, xb, yb in lay.bars.values():
            ax.plot([xa, xb], [up(ya), up(yb)], color="black", linewidth=1.2, zorder=3)
        ids, kinds = _bus_kinds(net)
        for label, mask, marker, color in kinds:
            if mask.any():
                mx = [lay.markers[b][0] for b in ids[mask]]
                my = [up(lay.markers[b][1]) for b in ids[mask]]
                ax.plot(mx, my, linestyle="none", marker=marker, markersize=2.5 if big else 7,
                        markerfacecolor=color, markeredgecolor="black",
                        markeredgewidth=0.2 if big else 0.8, zorder=4, label=label)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower left", numpoints=2, framealpha=1, edgecolor="black", fancybox=False,
                  fontsize=10, borderaxespad=0.8, markerscale=2.5 if big else 1)
        cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03, aspect=20)
        cb.set_ticks(np.arange(lo, hi + 1e-9, step))
        cb.ax.yaxis.set_major_formatter("{x:.2f}")
        return _save(fig, out_dir / f"VoltageMap2D-{net.case.name}.png")


def plot_voltage_map_3d(net: Network, v: np.ndarray, out_dir: Path, grid: int = 520,
                        smooth: float = 0.003, elev: float = 40, azim: float = -50) -> Path:
    """3-D voltage-level surface with the network drawn above it (after Milano, Fig. 22.3)."""
    lay = _layout_for(net)
    x0, y0, x1, y1 = lay.frame
    X, Y, Z, lo, hi, _ = _voltage_field(net, v, lay, grid, smooth, tent=True)
    big = net.case.n_bus > 100
    top = float(np.abs(v).max()) + 0.12 * (hi - lo)  # height of the single-line diagram

    def up(y):
        return y0 + y1 - y

    with plt.rc_context(_MAP_RC):
        fig = plt.figure(figsize=(8.5, 6.5))
        ax = fig.add_subplot(projection="3d", computed_zorder=False)
        # Colour scale of Milano's Fig. 22.3: the frame is blue, the bulk of the buses
        # green, and only the highest voltages run through yellow and orange to red.
        v_hi = float(np.abs(v).max())
        if VOLTAGE_CMAP == "jet":  # the scale of Milano's Fig. 22.3
            t = 0.14 + (Z - lo) / max(v_hi - lo, 1e-9) * (0.72 - 0.14)
            t = np.clip(t + 0.9 * np.clip(t - 0.60, 0.0, None), 0.0, 1.0)
        else:  # perceptually uniform scale over the range of the bus voltages
            v_lo = float(np.abs(v).min())
            t = np.clip((Z - v_lo) / max(v_hi - v_lo, 1e-9), 0.0, 1.0)
        rgb = plt.get_cmap(VOLTAGE_CMAP)(t)[..., :3]
        shade = LightSource(azdeg=300, altdeg=55).shade_rgb(rgb, Z, blend_mode="soft", vert_exag=500)
        ax.plot_surface(X, Y, Z, facecolors=shade, rstride=1, cstride=1, linewidth=0, antialiased=False,
                        shade=False, zorder=1)
        for path in lay.lines.values():
            px, py = zip(*path)
            ax.plot(px, [up(y) for y in py], top, color=LINE_COLOR, linewidth=0.4 if big else 1.4, zorder=3)
        if lay.outline:
            ox, oy = zip(*lay.outline)
            ax.plot(ox, [up(q) for q in oy], top, color="black", linewidth=0.9, zorder=3)
        for xa, ya, xb, yb in ([] if big else lay.bars.values()):  # no bus bars on large systems
            ax.plot([xa, xb], [up(ya), up(yb)], top, color="black", linewidth=0.7 if big else 1.6, zorder=4)
        ids = net.case.bus[:, c.BUS_I].astype(int)
        gen = np.isin(np.arange(net.case.n_bus), np.concatenate([net.ref, net.pv]))
        for mask, size, color in ((~gen, 18, "#18c018"), (gen, 55, "#1010d0")):
            if mask.any():
                mx = [lay.markers[b][0] for b in ids[mask]]
                my = [up(lay.markers[b][1]) for b in ids[mask]]
                ax.scatter(mx, my, top, s=size * (0.12 if big else 1), color=color, depthshade=False,
                           zorder=5)
        ax.set_zlim(lo, top)
        ax.set_box_aspect((x1 - x0, y1 - y0, 0.22 * (x1 - x0)), zoom=1.12)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        path = _save(fig, out_dir / f"VoltageMap3D-{net.case.name}.png")
    img = plt.imread(path)
    ink = np.argwhere(img[..., :3].min(axis=-1) < 0.98)
    (r0, c0), (r1, c1) = ink.min(axis=0), ink.max(axis=0)
    pad = 20
    plt.imsave(path, img[max(r0 - pad, 0):r1 + pad, max(c0 - pad, 0):c1 + pad])
    return path


_ROBUST_STYLE = {
    "NR": dict(color="#0072BD", marker="o", markerfacecolor="blue"),
    "NR-GD": dict(color="#D95319", marker="s", markerfacecolor="red"),
    "NR-LI (safeguard)": dict(color="#EDB120", marker="d", markerfacecolor="lime"),
    "NR-LI (alternate)": dict(color="#7E2F8E", marker="^", markerfacecolor="violet"),
}
_ROBUST_LABEL = {"NR": "Newton-Raphson (NR)", "NR-LI (safeguard)": "Hybrid NR-LI",
                 "NR-LI (alternate)": "Alternating NR-LI", "NR-GD": "NR + Gradient Descent"}


def plot_robustness(levels, rates: dict[str, list[float]], case_name: str, out_dir: Path) -> Path:
    """Share of random initial guesses that reach the operating point, per perturbation level."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for label, values in rates.items():
        ax.plot(levels, values, linewidth=2, markersize=7, label=_ROBUST_LABEL[label],
                **_ROBUST_STYLE[label])
    ax.set_xlabel(r"Perturbation level $s$ (p.u. / rad)")
    ax.set_ylabel("Converged initial guesses (%)")
    ax.set_ylim(-3, 103)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.legend(loc="upper right")
    return _save(fig, out_dir / f"Robustness-{case_name}.png")
