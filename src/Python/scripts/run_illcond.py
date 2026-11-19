"""Build ill-conditioned cases with a known solution and solve them from a flat start.

    uv run python scripts/run_illcond.py [case ...]

Set ROAPF_CASES to the folder with the MATPOWER .m files (only needed for the
ACTIVSg systems). The generated cases are written to cases/illconditioned/ and
the results to results/journal/illcond.csv.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

from roapf.cases import load_case
from roapf.illcond import make_ill_conditioned, write_matpower
from roapf.largescale import matches
from roapf.solvers import solve

CASE_DIR = os.environ.get("ROAPF_CASES")
OUT = Path("results/journal")
CASES = Path("cases/illconditioned")
# label -> (method, keyword arguments, iteration limit)
METHODS = {
    "NR": ("nr", {}, 100),
    "NR-SGD": ("nr-sgd", {}, 100),
    "SGD": ("sgd", {}, 3000),
    "CN-RK4": ("cn-rk4", {}, 100),
    "CN-RK4A": ("cn-rk4a", {}, 100),
    "NR-LS": ("nr-ls", {}, 300),
    "LM": ("lm", {}, 100),
    "LI": ("li", {}, 3000),
    "ALI": ("ali", {}, 3000),
    "NR-LI": ("nr-li", {}, 300),
    "NR-ALI": ("nr-ali", {}, 300),
    "NR-ALI-G": ("w-nr-ali", {"v_guard": 0.5}, 300),
    "NR-ALI-GC": ("w-nr-ali", {"v_guard": 0.5, "dtheta_cap": 0.5, "restore": 0.5}, 300),
}
if os.environ.get("ROAPF_ONLY"):
    METHODS = {k: METHODS[k] for k in os.environ["ROAPF_ONLY"].split(",")}
names = sys.argv[1:] or ["case14", "case39", "case57", "case118", "case300", "case_ACTIVSg500",
                         "case_ACTIVSg2000"]
out_file = OUT / (("illcond.csv" if len(sys.argv) == 1 else f"illcond_{'_'.join(names)}.csv") if not os.environ.get("ROAPF_ONLY") else f"illcond_{'_'.join(names)}_{'_'.join(METHODS)}.csv")
rows = []
for name in names:
    case = load_case(name, CASE_DIR)
    big = case.n_bus > 400
    for pv_to_pq in (False, True):
        for rx in (1.0, 2.0):
            for ic in make_ill_conditioned(case, pv_to_pq, rx, fractions=(0.9, 0.99)):
                net, v = ic.net, ic.v_solution
                write_matpower(ic.case, CASES / f"{ic.name}.m",
                               f"Ill-conditioned variant of {name}: PV buses "
                               f"{'converted to PQ' if pv_to_pq else 'kept'}, R x {rx:g}, loading "
                               f"{ic.fraction:g} of the maximum ({ic.lambda_max:.3f}).\n"
                               f"VM / VA hold the known solution; cond(J) there is {ic.cond:.3e}.")
                line = []
                for label, (method, kw, limit) in METHODS.items():
                    if big and label in ("NR-SGD", "LI", "ALI") or (case.n_bus > 600 and label == "LM"):
                        rows.append([ic.name, name, case.n_bus, pv_to_pq, rx, ic.fraction, f"{ic.lambda_max:.4f}",
                                     f"{ic.cond:.3e}", label, "skipped", "", "", ""])
                        continue
                    r = solve(net, method, tol=1e-8, max_iter=limit, **kw)
                    status = "ok" if matches(r, v) else ("other" if r.converged else "fail")
                    rows.append([ic.name, name, case.n_bus, pv_to_pq, rx, ic.fraction, f"{ic.lambda_max:.4f}",
                                 f"{ic.cond:.3e}", label, status, r.iterations, f"{r.history[-1]:.2e}",
                                 f"{r.seconds:.2f}"])
                    line.append(f"{label}:{r.iterations if status == 'ok' else status}")
                print(f"{ic.name:36s} cond={ic.cond:.1e} Vmin={np.abs(v).min():.2f}  " + "  ".join(line), flush=True)
                OUT.mkdir(parents=True, exist_ok=True)
                with out_file.open("w", newline="") as fh:
                    csv.writer(fh).writerows([["case", "base", "n_bus", "pv_to_pq", "rx", "fraction", "lambda_max",
                                               "cond_J", "method", "status", "iterations", "final_mismatch",
                                               "seconds"], *rows])
