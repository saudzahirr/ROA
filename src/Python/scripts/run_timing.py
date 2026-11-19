"""Wall-clock cost of the iterations (single thread of this machine; indicative only).

    uv run python scripts/run_timing.py

For each system: time of one Newton iteration (Jacobian, sparse LU, mismatch),
of one Landweber iteration (Jacobian, power iteration for sigma_1, mismatch),
the number of power iterations, and the total time of NR and of the proposed
method from a flat start and from one difficult random start.
Results: results/journal/timing.csv
"""

import csv
import os
import time
from pathlib import Path

import numpy as np

from roapf.cases import load_case
from roapf.largescale import matches, perturbed_guess, reference
from roapf.network import build_network
from roapf.solvers import newton_step, sigma_max_sq, solve

CASE_DIR = os.environ.get("ROAPF_CASES", str(Path(__file__).resolve().parents[1] / "cases" / "matpower"))
SYSTEMS = [("case14", 0.35), ("case39", 0.2), ("case118", 0.25), ("case_ACTIVSg500", 0.1),
           ("case_ACTIVSg2000", 0.03), ("case9241pegase", None)]


def best(fn, repeat=5):
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return min(t)


def power_iterations(jac, tol=1e-6, limit=300):
    if jac.shape[1] <= 100:
        return 0  # exact singular value decomposition
    x = np.random.default_rng(0).standard_normal(jac.shape[1])
    x /= np.linalg.norm(x)
    lam = 0.0
    for k in range(1, limit + 1):
        y = jac.T @ (jac @ x)
        new = float(np.linalg.norm(y))
        x = y / new
        if abs(new - lam) <= tol * new:
            return k
        lam = new
    return limit


rows = []
for name, level in SYSTEMS:
    net = build_network(load_case(name, CASE_DIR))
    v = net.v_flat
    jac, dy = net.jacobian(v), net.mismatch(v)
    t_jac = best(lambda: net.jacobian(v))
    t_lu = best(lambda: newton_step(jac, dy))
    t_sig = best(lambda: sigma_max_sq(jac))
    t_mis = best(lambda: net.mismatch(v))
    n_pow = power_iterations(jac)
    t_newton, t_landweber = t_jac + t_lu + 2 * t_mis, t_jac + t_sig + t_mis
    flat = {m: solve(net, m, tol=1e-8, max_iter=50, **kw) for m, kw in (("nr", {}), ("w-nr-ali", {"v_guard": 0.5, "dtheta_cap": 0.5, "restore": 0.5}))}
    hard_nr = hard_g = None
    if level is not None:
        v_ref = reference(net, 1e-8)
        for seed in range(200):
            v0 = perturbed_guess(net, level, np.random.default_rng(seed))
            a = solve(net, "nr", tol=1e-8, max_iter=40, v0=v0)
            if matches(a, v_ref):
                continue
            b = solve(net, "w-nr-ali", tol=1e-8, max_iter=40, v0=v0, v_guard=0.5, dtheta_cap=0.5, restore=0.5)
            if matches(b, v_ref):
                hard_nr, hard_g = a, b
                break
    rows.append([name, net.case.n_bus, f"{1e3 * t_newton:.2f}", f"{1e3 * t_landweber:.2f}", n_pow,
                 flat["nr"].iterations if flat["nr"].converged else "", f"{1e3 * flat['nr'].seconds:.1f}",
                 flat["w-nr-ali"].iterations if flat["w-nr-ali"].converged else "",
                 f"{1e3 * flat['w-nr-ali'].seconds:.1f}",
                 hard_g.iterations if hard_g else "", hard_g.n_correction if hard_g else "",
                 f"{1e3 * hard_g.seconds:.1f}" if hard_g else "", f"{1e3 * hard_nr.seconds:.1f}" if hard_nr else ""])
    print(rows[-1], flush=True)
out = Path("results/journal/timing.csv")
with out.open("w", newline="") as fh:
    csv.writer(fh).writerows([["case", "n_bus", "newton_iteration_ms", "landweber_iteration_ms", "power_iterations",
                               "flat_nr_iterations", "flat_nr_ms", "flat_proposed_iterations", "flat_proposed_ms",
                               "hard_proposed_iterations", "hard_landweber_steps", "hard_proposed_ms",
                               "hard_nr_ms_until_failure"], *rows])
