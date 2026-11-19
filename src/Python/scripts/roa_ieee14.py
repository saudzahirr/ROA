"""Region of attraction on the IEEE 14-bus system in the plane of one load bus.

The starting voltage magnitude and angle of one PQ bus are swept over the grid
of the two-bus maps (v in [0, 5] p.u., theta in [-pi/2, pi/2]); all other buses
start from the flat start.  A run is classified as reaching the operating point
(all voltages within 1e-4 p.u. of the reference), another solution, or none
within 25 iterations (tolerance 1e-5 p.u., as for the two-bus maps).

    uv run python scripts/roa_ieee14.py <bus number> [n]

Output: results/journal/roa14/bus<k>_<method>_<n>.npy
        (> 0 iterations to the operating point, -1 other solution, 0 none)
"""
import math
import sys
from pathlib import Path

import numpy as np

from roapf.cases import load_case
from roapf.largescale import reference
from roapf.network import build_network
from roapf.solvers import solve

METHODS = {"nr": ("nr", {}), "proposed": ("w-nr-ali", {"v_guard": 0.5, "dtheta_cap": 0.5, "restore": 0.5})}


def sweep(bus, n, case="case14"):
    net = build_network(load_case(case, str(Path(__file__).resolve().parents[1] / "cases" / "matpower")))
    v_ref = reference(net, 1e-8)
    k = int(np.where(net.case.bus[:, 0] == bus)[0][0])
    assert k in net.pq, f"bus {bus} is not a load bus"
    out = Path(__file__).resolve().parents[1] / "results" / "journal" / "roa14"
    out.mkdir(parents=True, exist_ok=True)
    vs, ds = np.linspace(0, 5, n), np.linspace(-math.pi / 2, math.pi / 2, n)
    for key, (method, kw) in METHODS.items():
        it = np.zeros((n, n))
        for i, vm in enumerate(vs):
            for j, d in enumerate(ds):
                v0 = net.v_flat.copy()
                v0[k] = max(vm, 1e-6) * np.exp(1j * d)
                r = solve(net, method, tol=1e-5, max_iter=25, v0=v0, **kw)
                if r.converged:
                    it[i, j] = r.iterations if np.max(np.abs(r.v - v_ref)) < 1e-4 else -1
        np.save(out / f"bus{bus}_{key}_{n}.npy", it)
        print(bus, key, f"operating point {100 * (it > 0).mean():.1f}%  other {100 * (it < 0).mean():.1f}%  "
              f"none {100 * (it == 0).mean():.1f}%  mean it {it[it > 0].mean():.2f} max {it.max():.0f}", flush=True)


if __name__ == "__main__":
    sweep(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 128)
