"""Two-bus system: share of the 256 x 256 starting points from which each of the
six methods of the paper reaches the operable solution, the low-voltage solution
or neither within 25 iterations.  Output: results/journal/roa2bus_split.csv"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from roa2bus_four import MAX_ITER, TOL, lw, nr_sgd, proposed  # noqa: E402
from run_roa2bus_guarded import jac, mis, run as run_basic  # noqa: E402


def landweber(x):
    for it in range(1, MAX_ITER + 1):
        if np.max(np.abs(mis(x))) < TOL:
            return it, x
        x = lw(x)
    return None, x


def sgd(x, eta=0.01, b1=0.9, b2=0.999):
    mom, sec = np.zeros(2), np.zeros(2)
    for it in range(1, MAX_ITER + 1):
        F = mis(x)
        if np.max(np.abs(F)) < TOL:
            return it, x
        g = -(jac(x).T @ F)
        mom, sec = b1 * mom + (1 - b1) * g, b2 * sec + (1 - b2) * g * g
        x = x - eta * (mom / (1 - b1**it)) / (np.sqrt(sec / (1 - b2**it)) + 1e-8)
    return None, x


METHODS = {"NR": lambda x: run_basic(x, "nr", MAX_ITER), "LI": landweber, "SGD": sgd, "NR + SGD": nr_sgd,
           "NR + LI": lambda x: run_basic(x, "hybrid", MAX_ITER), "Proposed": proposed}
np.seterr(all="ignore")
n = 256
vs, ds = np.linspace(0, 5, n), np.linspace(-math.pi / 2, math.pi / 2, n)
rows = []
for name, fn in METHODS.items():
    hi = lo = 0
    its = []
    for v in vs:
        for d in ds:
            try:
                k, x = fn(np.array([d, v]))
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError, ZeroDivisionError):
                k = None
            if k is not None and np.all(np.isfinite(x)):
                if abs(x[1]) > 0.5:
                    hi += 1
                    its.append(k)
                else:
                    lo += 1
    rows.append([name, f"{100 * (hi + lo) / n**2:.1f}", f"{100 * hi / n**2:.1f}", f"{100 * lo / n**2:.1f}",
                 f"{np.mean(its):.1f}" if its else "--"])
    print(rows[-1], flush=True)
out = Path(__file__).resolve().parents[1] / "results" / "journal" / "roa2bus_split.csv"
with out.open("w", newline="") as fh:
    csv.writer(fh).writerows([["method", "converged_percent", "operable_percent", "low_voltage_percent",
                               "mean_iterations_operable"], *rows])
