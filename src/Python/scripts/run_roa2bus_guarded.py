"""Two-bus region of attraction, split by the solution reached.

Newton-Raphson, the plain Newton-Landweber hybrid (Hybrid.m) and the guarded
method (Guarded.m) on the 256 x 256 grid of starting points of the ROA
repository.  The two-bus system has two solutions; a run counts for the
operable one if it ends with v2 > 0.5 p.u.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

B = 10.0
PS, QS = -0.9, -0.6
TOL = 1e-5


def mis(x):
    d, v = x
    p = B * v * math.sin(d)
    q = B * v * v - B * v * math.cos(d)
    return np.array([PS - p, QS - q])


def jac(x):
    d, v = x
    return np.array([[B * v * math.cos(d), B * math.sin(d)],
                     [B * v * math.sin(d), 2 * B * v - B * math.cos(d)]])


def lw(z):
    J = jac(z)
    return z + (J.T @ mis(z)) / (np.linalg.norm(J, 2) ** 2 + 2.2e-16)


def run(x, method, max_iter, guard=0.5, m=2, rho=0.25, cap=0.0):
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
        if method == "nr":
            if dx is None:
                return None, x
            x = x + dx
            continue
        if method == "hybrid":
            if dx is not None and np.linalg.norm(mis(x + dx)) < f0:
                x = x + dx
            else:
                x = lw(x)
            continue
        adm = dx is not None and np.all(np.isfinite(dx)) and x[1] + dx[1] >= guard
        if adm and cap > 0 and abs(dx[0]) > cap:
            dx = dx * (cap / abs(dx[0]))
        fn = np.linalg.norm(mis(x + dx)) if adm else math.inf
        take, start = False, False
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
            take, start = True, True
        else:
            take = True
        if take:
            k += 1
            z = x if xprev is None else x + (k - 1) / (k + 2) * (x - xprev)
            cand = lw(z)
            if xprev is not None and not np.linalg.norm(mis(cand)) < f0:
                k, cand = 1, lw(x)
            xprev = x
            if start:
                xbar, fbar, count, trial = cand, f0, 1, True
                x = x + dx
            else:
                x = cand
    return None, x


def main(n=256):
    out = Path(__file__).resolve().parents[1] / "results" / "journal" / "roa2bus_guarded.csv"
    vs, ds = np.linspace(0, 5, n), np.linspace(-math.pi / 2, math.pi / 2, n)
    rows = []
    for method, max_iter in [("nr", 25), ("hybrid", 25), ("guarded", 25), ("hybrid", 100), ("guarded", 100),
                             ("guarded", 300), ("capped", 25), ("capped", 100)]:
        hi = lo = 0
        its = []
        for v in vs:
            for d in ds:
                try:
                    it, x = run(np.array([d, v]), method, max_iter, cap=0.5 if method == "capped" else 0.0)
                except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
                    it = None
                if it is not None:
                    if abs(x[1]) > 0.5:
                        hi += 1; its.append(it)
                    else:
                        lo += 1
        rows.append([method, max_iter, 100 * hi / n**2, 100 * lo / n**2, float(np.mean(its)) if its else float("nan")])
        print(*rows[-1], flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "max_iter", "operable_pct", "low_voltage_pct", "mean_iterations_operable"])
        w.writerows(rows)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 256)
