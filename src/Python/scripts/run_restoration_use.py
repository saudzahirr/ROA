"""How often the restoration step of the proposed method is used.

Share of runs with at least one restoration step, and mean number of restoration
steps in those runs, on the random starting points of the sampled
region-of-attraction test and of the protocol of Costilla-Enriquez et al.
Output: results/journal/restoration_use.csv
"""
import csv
import os
from pathlib import Path

import numpy as np

from roapf import cases as c
from roapf.cases import load_case
from roapf.largescale import perturbed_guess, reference
from roapf.network import build_network
from roapf.solvers import solve

CASE_DIR = os.environ.get("ROAPF_CASES", "cases/matpower")
KW = dict(tol=1e-8, v_guard=0.5, dtheta_cap=0.5, restore=0.5)
rows = []


def record(test, case, level, runs):
    used = [r.n_restore for r in runs if r.n_restore > 0]
    rows.append([test, case, level, len(runs), f"{100 * len(used) / len(runs):.0f}",
                 f"{np.mean(used):.1f}" if used else "0"])
    print(rows[-1], flush=True)


ROBUST = {"case14": (50, [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]), "case39": (50, [0.1, 0.15, 0.2, 0.25, 0.3]),
          "case118": (50, [0.15, 0.2, 0.25, 0.3, 0.35]), "case_ACTIVSg500": (20, [0.03, 0.04, 0.06, 0.08, 0.1, 0.125]),
          "case_ACTIVSg2000": (15, [0.02, 0.03, 0.04])}
for case, (n, levels) in ROBUST.items():
    net = build_network(load_case(case, CASE_DIR))
    rng = np.random.default_rng(0)
    for s in levels:
        record("random", case, s, [solve(net, "w-nr-ali", max_iter=40, v0=perturbed_guess(net, s, rng), **KW)
                                   for _ in range(n)])

for case in ("case14", "case39", "case57", "case118"):
    net = build_network(load_case(case, CASE_DIR))
    s0 = net.sbus.copy()
    pd = net.case.bus[:, c.PD] / net.case.base_mva

    def nr_from(v0):
        r = solve(net, "nr", tol=1e-8, max_iter=30, v0=v0)
        return (r.converged and np.abs(r.v).min() > 0.5), r.v

    alpha, v, step = 1.0, reference(net, 1e-8), 0.1
    while step > 1e-3:
        net.sbus = s0 - (alpha + step - 1.0) * pd
        ok, v_new = nr_from(v)
        if ok:
            alpha, v = alpha + step, v_new
        else:
            step /= 2
    for load, factor in (("normal", 1.0), ("heavy", 1.0 + 0.9 * (alpha - 1.0))):
        net.sbus = s0.copy()
        v_ref = reference(net, 1e-8)
        for mu in np.linspace(1.0, factor, 30)[1:]:
            net.sbus = s0 - (mu - 1.0) * pd
            _, v_ref = nr_from(v_ref)
        net.sbus = s0 - (factor - 1.0) * pd
        vm, va = np.abs(v_ref), np.angle(v_ref)
        rng = np.random.default_rng(1)
        runs = []
        for _ in range(50):
            m0, a0 = np.abs(net.v_flat), np.angle(net.v_flat)
            m0[net.pq] = rng.uniform(vm.min(), vm.max(), net.pq.size)
            a0[net.pvpq] = rng.uniform(va.min(), va.max(), net.pvpq.size)
            runs.append(solve(net, "w-nr-ali", max_iter=100, v0=m0 * np.exp(1j * a0), **KW))
        record("protocol of ref. 16", case, load, runs)
        flat = solve(net, "nr", tol=1e-8, max_iter=30)
        print("   flat-start NR:", flat.converged, flat.iterations, flush=True)
with Path("results/journal/restoration_use.csv").open("w", newline="") as fh:
    csv.writer(fh).writerows([["test", "case", "level", "runs", "runs_with_restoration_percent",
                               "mean_restoration_steps_in_those_runs"], *rows])
