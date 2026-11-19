"""Command line interface: ``uv run roapf --help``."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from . import largescale, plots, roa2bus
from .cases import AVAILABLE, load_case
from .network import build_network
from .solvers import LABELS, METHODS, MODES, solve


def cmd_roa(args) -> None:
    keys = args.methods or list(roa2bus.METHODS)
    sys = roa2bus.TwoBus(z=complex(args.r, args.x))
    rows = []
    for key in keys:
        t0 = time.perf_counter()
        sw = roa2bus.sweep(key, n=args.n, sys=sys)
        path = plots.plot_roa(sw, args.out, title=not args.no_title)
        done = sw.iterations[~np.isnan(sw.iterations)]
        mean_it = done.mean() if done.size else float("nan")
        rows.append([key, f"{100 * sw.converged_fraction:.1f}", f"{mean_it:.2f}"])
        print(f"{key:20s} converged from {100 * sw.converged_fraction:5.1f}% of the grid, "
              f"mean {mean_it:5.2f} iterations  [{time.perf_counter() - t0:4.1f}s]  -> {path}")
    _write_csv(args.results / "roa_summary.csv", ["method", "converged_percent", "mean_iterations"], rows)


def cmd_solvability(args) -> None:
    s = roa2bus.solvability(n=args.n)
    print("->", plots.plot_solvability(s, args.out, title=not args.no_title))


def cmd_convergence(args) -> None:
    rows, traces = [], []
    for name in args.cases:
        net = build_network(load_case(name, args.case_dir))
        v0, tag = None, ""
        if args.perturb > 0:
            found = largescale.find_difficult_start(net, args.perturb, args.tol, args.max_iter)
            if found is None:
                print(f"{net.case.name}: no initial guess found where NR fails and NR-LI converges")
                continue
            seed, v0 = found
            tag = f"-perturbed{args.perturb:g}"
            print(f"{net.case.name}: perturbation level {args.perturb:g}, seed {seed}")
        results = [solve(net, m, args.mode, args.tol, args.max_iter, args.gd_step, v0) for m in METHODS]
        for log in (True, False):
            plots.plot_convergence(results, net.case.name + tag, args.out, args.tol, log=log)
        for r in results:
            rows.append([net.case.name, LABELS[r.method], r.iterations, r.converged, f"{r.history[-1]:.3e}"])
            traces += [[net.case.name, r.method, k, f"{h:.6e}"] for k, h in enumerate(r.history)]
        print(f"{net.case.name:8s} " + "   ".join(
            f"{r.method}: {r.iterations if r.converged else 'FAIL'}" for r in results))
        solved = next((r for r in results if r.converged), None)
        if solved is not None and not args.no_maps and not tag:
            plots.plot_voltage_map(net, solved.v, args.out)
            plots.plot_voltage_map_3d(net, solved.v, args.out)
    _write_csv(args.results / "convergence_summary.csv",
               ["case", "method", "iterations", "converged", "final_mismatch_pu"], rows)
    _write_csv(args.results / "convergence_history.csv",
               ["case", "method", "iteration", "mismatch_pu"], traces)
    print(f"mode = {args.mode}, tolerance = {args.tol:g}; tables in {args.results}")


def cmd_robustness(args) -> None:
    variants = [v for v in largescale.VARIANTS if v[0] != "nr-gd"]
    rows = []
    for name in args.cases:
        rates = largescale.perturbed_start_study(name, args.case_dir, args.levels, args.samples, args.tol,
                                                 args.max_iter, variants)
        plots.plot_robustness(args.levels, rates, name, args.out)
        rows += [[name, label, level, f"{rate:.1f}"]
                 for label, values in rates.items() for level, rate in zip(args.levels, values)]
        _write_csv(args.results / "robustness.csv", ["case", "method", "level", "converged_percent"], rows)


def cmd_flat(args) -> None:
    rows = largescale.flat_start_study(args.cases or largescale.LARGE_CASES, args.case_dir, args.tol,
                                       args.max_iter, args.gd_step)
    _write_csv(args.results / "largescale_flat_start.csv",
               ["case", "n_bus", "method", "converged", "same_solution", "iterations", "correction_steps",
                "final_mismatch", "seconds"],
               [[r.case, r.n_bus, r.label, r.result.converged, r.same_solution, r.result.iterations,
                 r.result.n_correction, f"{r.result.history[-1]:.3e}", f"{r.result.seconds:.2f}"]
                for r in rows])


def cmd_all(args) -> None:
    cmd_solvability(args)
    cmd_roa(args)
    cmd_convergence(args)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows([header, *rows])


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, default=Path("figures"), help="figure directory")
    common.add_argument("--results", type=Path, default=Path("results"), help="CSV directory")
    common.add_argument("--no-title", action="store_true", help="omit figure titles")

    grid = argparse.ArgumentParser(add_help=False)
    grid.add_argument("-n", type=int, default=256, help="grid points per axis (default 256)")
    grid.add_argument("--methods", nargs="+", choices=list(roa2bus.METHODS), metavar="METHOD",
                      help=f"subset of: {', '.join(roa2bus.METHODS)}")
    grid.add_argument("-r", type=float, default=0.0, help="2-bus line resistance, pu (default 0)")
    grid.add_argument("-x", type=float, default=0.1, help="2-bus line reactance, pu (default 0.1)")

    conv = argparse.ArgumentParser(add_help=False)
    conv.add_argument("--cases", nargs="+", default=["case14", "case39", "case118", "case300"],
                      help=f"built in: {', '.join(AVAILABLE)}; or names of .m files in --case-dir")
    conv.add_argument("--case-dir", type=Path, default=None, help="folder with MATPOWER .m case files")
    conv.add_argument("--no-maps", action="store_true", help="skip the 2-D / 3-D voltage maps")
    conv.add_argument("--perturb", type=float, default=0.0,
                      help="start from a random guess at this perturbation level for which NR fails "
                           "and NR-LI converges (default 0: flat start)")
    conv.add_argument("--mode", choices=MODES, default="safeguard",
                      help="how NR and the correction step are combined (default: safeguard)")
    conv.add_argument("--tol", type=float, default=1e-8)
    conv.add_argument("--max-iter", type=int, default=50)
    conv.add_argument("--gd-step", type=float, default=1e-4, help="fixed gradient-descent step size")

    rob = argparse.ArgumentParser(add_help=False)
    rob.add_argument("--cases", nargs="+", default=["case14", "case39", "case118", "case300"])
    rob.add_argument("--case-dir", type=Path, default=None, help="folder with MATPOWER .m case files")
    rob.add_argument("--levels", nargs="+", type=float, default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    rob.add_argument("--samples", type=int, default=100)
    rob.add_argument("--tol", type=float, default=1e-8)
    rob.add_argument("--max-iter", type=int, default=50)

    flat = argparse.ArgumentParser(add_help=False)
    flat.add_argument("--cases", nargs="+", default=None, help="default: the large MATPOWER cases")
    flat.add_argument("--case-dir", type=Path, required=True, help="folder with MATPOWER .m case files")
    flat.add_argument("--tol", type=float, default=1e-8)
    flat.add_argument("--max-iter", type=int, default=50)
    flat.add_argument("--gd-step", type=float, default=1e-4)

    parser = argparse.ArgumentParser(prog="roapf", description=__doc__)
    sub = parser.add_subparsers(required=True)
    for name, func, parents, text in [
        ("roa", cmd_roa, [common, grid], "2-bus region-of-attraction maps"),
        ("solvability", cmd_solvability, [common, grid], "2-bus PQ solvability region"),
        ("convergence", cmd_convergence, [common, conv], "NR vs hybrid convergence on the IEEE cases"),
        ("robustness", cmd_robustness, [common, rob], "convergence rate from random initial guesses"),
        ("flat-start", cmd_flat, [common, flat], "flat-start comparison on large MATPOWER cases"),
        ("all", cmd_all, [common, grid, conv], "2-bus maps and convergence study"),
    ]:
        sub.add_parser(name, parents=parents, help=text).set_defaults(func=func)
    args = parser.parse_args(argv)
    args.func(args)
