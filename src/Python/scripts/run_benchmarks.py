"""Benchmarks behind the journal paper. Usage (from the roapf project):

    uv run python scripts/run_benchmarks.py robust <case> <samples> <level> [<level> ...]
    uv run python scripts/run_benchmarks.py starts
    uv run python scripts/run_benchmarks.py stress

Set ROAPF_CASES to the folder with the MATPOWER .m files. Results go to results/journal/.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

from roapf.cases import load_case
from roapf.largescale import LARGE_CASES, matches, perturbed_guess, reference
from roapf.network import build_network
from roapf.solvers import dc_start, solve

CASE_DIR = Path(os.environ.get("ROAPF_CASES", "."))
OUT = Path("results/journal")
OUT.mkdir(parents=True, exist_ok=True)
TOL = 1e-8

# label -> (method, keyword arguments)
VARIANTS = {
    "NR": ("nr", {}),
    "NR-SGD": ("nr-sgd", {}),
    "LI": ("li", {}),
    "SGD": ("sgd", {}),
    "CN-RK4": ("cn-rk4", {}),
    "CN-RK4A": ("cn-rk4a", {}),
    "NR-LS": ("nr-ls", {}),
    "LM": ("lm", {}),
    "NR-LI": ("nr-li", {}),
    "NR-ALI": ("nr-ali", {}),
    "NR-ALI-W": ("w-nr-ali", {}),
    "NR-ALI-G": ("w-nr-ali", {"v_guard": 0.5}),
    # sensitivity of the proposed method to the voltage guard (run with ROAPF_ONLY)
    # guarded method with the angle cap (and its sensitivity)
    "NR-ALI-GC": ("w-nr-ali", {"v_guard": 0.5, "dtheta_cap": 0.5, "restore": 0.5}),
    "C0.3": ("w-nr-ali", {"v_guard": 0.5, "dtheta_cap": 0.3, "restore": 0.5}),
    "C0.7": ("w-nr-ali", {"v_guard": 0.5, "dtheta_cap": 0.7, "restore": 0.5}),
    "C1.0": ("w-nr-ali", {"v_guard": 0.5, "dtheta_cap": 1.0, "restore": 0.5}),
    "GC0.0": ("w-nr-ali", {"v_guard": 0.0, "dtheta_cap": 0.5, "restore": 0.5}),
    "GC0.3": ("w-nr-ali", {"v_guard": 0.3, "dtheta_cap": 0.5, "restore": 0.5}),
    "GC0.7": ("w-nr-ali", {"v_guard": 0.7, "dtheta_cap": 0.5, "restore": 0.5}),
    "GC0.8": ("w-nr-ali", {"v_guard": 0.8, "dtheta_cap": 0.5, "restore": 0.5}),
    "R0": ("w-nr-ali", {"v_guard": 0.5, "dtheta_cap": 0.5}),
    "R0.25": ("w-nr-ali", {"v_guard": 0.5, "dtheta_cap": 0.5, "restore": 0.25}),
    "R1": ("w-nr-ali", {"v_guard": 0.5, "dtheta_cap": 0.5, "restore": 1.0}),
    "C0": ("w-nr-ali", {"v_guard": 0.5, "restore": 0.5}),
    "NR-C": ("w-nr-ali", {"v_guard": 0.0, "dtheta_cap": 0.5, "watch": 0}),
    "G0.3": ("w-nr-ali", {"v_guard": 0.3}),
    "G0.7": ("w-nr-ali", {"v_guard": 0.7}),
    "G0.8": ("w-nr-ali", {"v_guard": 0.8}),
}
LM_LIMIT = int(os.environ.get("ROAPF_LM_LIMIT", "3000"))  # LM is skipped on larger systems
# ROAPF_ONLY="NR-LI-W" runs a subset; the random initial guesses do not depend on the subset
SUFFIX = ""
if not os.environ.get("ROAPF_ONLY"):
    VARIANTS = {k: v for k, v in VARIANTS.items() if not k.startswith(("G0.", "C0.", "C1.", "GC0.", "NR-C", "NR-ALI-GC", "CN-", "R0", "R1", "C0", "LI", "SGD"))}
if os.environ.get("ROAPF_ONLY"):
    VARIANTS = {k: VARIANTS[k] for k in os.environ["ROAPF_ONLY"].split(",")}
    SUFFIX = "_" + "_".join(VARIANTS)


class Skipped:
    converged, iterations, n_correction, seconds, v = False, 0, 0, 0.0, None


def run(net, label, v0=None, max_iter=50):
    method, kw = VARIANTS[label]
    if (label == "LM" and net.case.n_bus > LM_LIMIT) or (label == "NR-SGD" and net.case.n_bus > int(os.environ.get("ROAPF_SGD_LIMIT", "400"))):
        return Skipped()
    return solve(net, method, tol=TOL, max_iter=max_iter, v0=v0, **kw)


def save(name, header, rows):
    with (OUT / name).open("w", newline="") as fh:
        csv.writer(fh).writerows([header, *rows])


def robust(case, samples, levels):
    net = build_network(load_case(case, CASE_DIR))
    v_ref = reference(net, TOL)
    rng = np.random.default_rng(0)
    rows = []
    for s in levels:
        wins = dict.fromkeys(VARIANTS, 0)
        iters = {k: [] for k in VARIANTS}
        for _ in range(samples):
            v0 = perturbed_guess(net, s, rng)
            for label in VARIANTS:
                r = run(net, label, v0, max_iter=40)
                if matches(r, v_ref):
                    wins[label] += 1
                    iters[label].append(r.iterations)
        for label in VARIANTS:
            mean_it = float(np.mean(iters[label])) if iters[label] else float("nan")
            rows.append([case, label, s, f"{100 * wins[label] / samples:.1f}", f"{mean_it:.2f}"])
        print(case, s, {k: f"{100 * wins[k] / samples:.0f}%" for k in VARIANTS}, flush=True)
        save(f"robust_{case}{SUFFIX}.csv", ["case", "method", "level", "converged_percent", "mean_iterations"], rows)


def starts():
    names = ["case14", "case39", "case118", "case300", "case_ACTIVSg500", *LARGE_CASES]
    rows = []
    for name in names:
        net = build_network(load_case(name, CASE_DIR))
        v_ref = reference(net, TOL)
        for start, v0 in (("flat", None), ("dc", dc_start(net))):
            for label in VARIANTS:
                r = run(net, label, v0)
                if isinstance(r, Skipped):
                    rows.append([name, net.case.n_bus, start, label, "skipped", "", "", ""])
                    continue
                status = "ok" if matches(r, v_ref) else ("other" if r.converged else "fail")
                rows.append([name, net.case.n_bus, start, label, status, r.iterations, r.n_correction,
                             f"{r.seconds:.2f}"])
            print(name, start, [(x[3], x[4], x[5]) for x in rows[-len(VARIANTS):]], flush=True)
        save(f"starts{SUFFIX}.csv", ["case", "n_bus", "start", "method", "status", "iterations", "fallback_steps",
                            "seconds"], rows)


def stress():
    """Flat-start solves close to the loadability limit (ill-conditioned Jacobian)."""
    rows = []
    for name in os.environ.get("ROAPF_STRESS", "case14 case39 case118 case300 case_ACTIVSg500 case_ACTIVSg2000").split():
        net = build_network(load_case(name, CASE_DIR))
        s0 = net.sbus.copy()

        def solved(lam, v0):
            net.sbus = lam * s0
            r = solve(net, "nr", tol=TOL, max_iter=30, v0=v0)
            ok = r.converged and np.abs(r.v).min() > 0.5
            return ok, r.v

        # continuation in the loading factor, then bisection on the last interval
        lam, v, step = 1.0, reference(net, TOL), 0.1
        while step > 1e-4:
            ok, v_new = solved(lam + step, v)
            if ok:
                lam, v = lam + step, v_new
            else:
                step /= 2
        v_base = reference(net, TOL)
        for frac in (0.90, 0.95, 0.99, 0.999):
            # reference = operable solution followed by continuation from the base case
            target, v_ref = frac * lam, v_base
            for mu in np.linspace(1.0, target, 40)[1:]:
                ok, v_ref = solved(mu, v_ref)
            net.sbus = target * s0
            cond = float(np.linalg.cond(net.jacobian(v_ref).toarray())) if net.n_state <= 4000 else float("nan")
            for label in VARIANTS:
                r = run(net, label)
                status = "ok" if matches(r, v_ref) else ("other" if r.converged else "fail")
                rows.append([name, f"{lam:.4f}", frac, f"{cond:.3e}", label, status, r.iterations])
            print(name, f"lam_max={lam:.3f}", frac, f"cond={cond:.2e}",
                  [(x[4], x[5], x[6]) for x in rows[-len(VARIANTS):]], flush=True)
        net.sbus = s0
        save(f"stress2{SUFFIX}.csv", ["case", "lambda_max", "fraction", "jacobian_cond", "method", "status",
                            "iterations"], rows)


def ce(case, samples=50):
    """Protocol of Costilla-Enriquez et al.: random initial guesses drawn uniformly between the
    smallest and largest voltage magnitude / angle of the solution, at normal and at heavy load."""
    from roapf import cases as c

    net = build_network(load_case(case, CASE_DIR))
    s0 = net.sbus.copy()
    pd = net.case.bus[:, c.PD] / net.case.base_mva
    rows = []

    def nr_from(v0):
        r = solve(net, "nr", tol=TOL, max_iter=30, v0=v0)
        return (r.converged and np.abs(r.v).min() > 0.5), r.v

    # largest factor on the active load for which the operable solution can be followed
    alpha, v, step = 1.0, reference(net, TOL), 0.1
    while step > 1e-3:
        net.sbus = s0 - (alpha + step - 1.0) * pd
        ok, v_new = nr_from(v)
        if ok:
            alpha, v = alpha + step, v_new
        else:
            step /= 2
    for load, factor in (("normal", 1.0), ("heavy", 1.0 + 0.9 * (alpha - 1.0))):
        net.sbus = s0.copy()  # reset before the reference is computed
        v_ref = reference(net, TOL)
        for mu in np.linspace(1.0, factor, 30)[1:]:
            net.sbus = s0 - (mu - 1.0) * pd
            _, v_ref = nr_from(v_ref)
        net.sbus = s0 - (factor - 1.0) * pd
        vm, va = np.abs(v_ref), np.angle(v_ref)
        rng = np.random.default_rng(1)
        hit = dict.fromkeys(VARIANTS, 0)
        conv = dict.fromkeys(VARIANTS, 0)
        iters = {k: [] for k in VARIANTS}
        for _ in range(samples):
            m0, a0 = np.abs(net.v_flat), np.angle(net.v_flat)
            m0[net.pq] = rng.uniform(vm.min(), vm.max(), net.pq.size)
            a0[net.pvpq] = rng.uniform(va.min(), va.max(), net.pvpq.size)
            v0 = m0 * np.exp(1j * a0)
            for label in VARIANTS:
                r = run(net, label, v0, max_iter=100)
                conv[label] += bool(r.converged)
                if matches(r, v_ref):
                    hit[label] += 1
                    iters[label].append(r.iterations)
        for label in VARIANTS:
            mean_it = float(np.mean(iters[label])) if iters[label] else float("nan")
            rows.append([case, load, f"{factor:.3f}", label, f"{100 * hit[label] / samples:.0f}",
                         f"{100 * conv[label] / samples:.0f}", f"{mean_it:.1f}"])
        print(case, load, f"factor={factor:.2f}", {k: f"{100 * hit[k] / samples:.0f}%" for k in VARIANTS},
              flush=True)
        save(f"ce_{case}{SUFFIX}.csv", ["case", "load", "load_factor", "method", "reach_solution_percent",
                                        "converged_percent", "mean_iterations"], rows)
    net.sbus = s0


if __name__ == "__main__":
    task = sys.argv[1]
    if task == "robust":
        robust(sys.argv[2], int(sys.argv[3]), [float(x) for x in sys.argv[4:]])
    elif task == "starts":
        starts()
    elif task == "stress":
        stress()
    elif task == "ce":
        ce(sys.argv[2])
