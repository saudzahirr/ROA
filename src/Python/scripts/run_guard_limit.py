"""Voltage guard against operable solutions with low voltages.

    uv run python scripts/run_guard_limit.py

Heavily loaded systems (generators controlling voltage, 99% of the maximum
loading) have operable solutions with voltages near or below typical guard
values. The proposed method is run from a flat start with several guards.
Results: results/journal/guard_limit.csv
"""

import csv
import os
from pathlib import Path

import numpy as np

from roapf.cases import load_case
from roapf.illcond import make_ill_conditioned
from roapf.largescale import matches
from roapf.solvers import solve

CASE_DIR = os.environ.get("ROAPF_CASES", str(Path(__file__).resolve().parents[1] / "cases" / "matpower"))
rows = []
for name in ("case14", "case39", "case57", "case118", "case300", "case_ACTIVSg500"):
    for rx in (1.0, 2.0):
        for ic in make_ill_conditioned(load_case(name, CASE_DIR), False, rx, fractions=(0.99,)):
            vmin = float(np.abs(ic.v_solution).min())
            cells = []
            for guard in (0.0, 0.3, 0.5, 0.7, 0.8):
                r = solve(ic.net, "w-nr-ali", tol=1e-8, max_iter=300, v_guard=guard, dtheta_cap=0.5, restore=0.5)
                ok = matches(r, ic.v_solution)
                cells.append(r.iterations if ok else ("other" if r.converged else "fail"))
            rows.append([ic.name, f"{vmin:.3f}", *cells])
            print(rows[-1], flush=True)
with Path("results/journal/guard_limit.csv").open("w", newline="") as fh:
    csv.writer(fh).writerows([["case", "lowest_solution_voltage", "no_guard", "guard_0.3", "guard_0.5", "guard_0.7",
                               "guard_0.8"], *rows])
