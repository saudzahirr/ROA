"""Flat-start and perturbed-start experiments on large MATPOWER cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cases import load_case
from .network import Network, build_network
from .solvers import Result, solve

# (method, mode) pairs compared on every case
VARIANTS = (
    ("nr", "safeguard", "NR"),
    ("nr-gd", "alternate", "NR-GD"),
    ("nr-li", "safeguard", "NR-LI (safeguard)"),
    ("nr-li", "alternate", "NR-LI (alternate)"),
)

LARGE_CASES = (
    "case1354pegase", "case1888rte", "case1951rte", "case_ACTIVSg2000", "case2383wp", "case2736sp",
    "case2737sop", "case2746wop", "case2746wp", "case2848rte", "case2868rte", "case2869pegase",
    "case3012wp", "case3120sp", "case3375wp", "case6468rte", "case6470rte", "case6495rte",
    "case6515rte", "case8387pegase", "case9241pegase", "case_ACTIVSg10k", "case13659pegase",
    "case_ACTIVSg25k", "case_ACTIVSg70k",
)


@dataclass
class Row:
    case: str
    n_bus: int
    label: str
    result: Result
    same_solution: bool  # converged to the operating point stored in the case file


def reference(net: Network, tol: float) -> np.ndarray | None:
    """Operating point reached by NR from the voltages stored in the case file."""
    r = solve(net, "nr", tol=tol, max_iter=30, v0=net.v_case)
    return r.v if r.converged else None


def matches(r: Result, v_ref: np.ndarray | None) -> bool:
    return bool(r.converged and v_ref is not None and np.max(np.abs(r.v - v_ref)) < 1e-4)


def flat_start_study(names, case_dir: Path, tol: float, max_iter: int, gd_step: float,
                     log=print) -> list[Row]:
    rows = []
    for name in names:
        net = build_network(load_case(name, case_dir))
        v_ref = reference(net, tol)
        parts = []
        for method, mode, label in VARIANTS:
            r = solve(net, method, mode, tol, max_iter, gd_step)
            rows.append(Row(name, net.case.n_bus, label, r, matches(r, v_ref)))
            parts.append(f"{label}: {r.iterations if r.converged else 'FAIL'}"
                         + ("" if not r.converged or rows[-1].same_solution else "*"))
        log(f"{name:18s} {net.case.n_bus:6d} buses   " + "   ".join(parts))
    return rows


def perturbed_start_study(name: str, case_dir: Path, levels, samples: int, tol: float, max_iter: int,
                          variants=VARIANTS, seed: int = 0, log=print) -> dict[str, list[float]]:
    """Share of random initial guesses from which each method reaches the stored solution.

    Level ``s`` draws PQ-bus magnitudes uniformly in [1 - s, 1 + s] pu and all
    non-slack angles uniformly in [-s, s] rad around the flat start.
    """
    net = build_network(load_case(name, case_dir))
    v_ref = reference(net, tol)
    rng = np.random.default_rng(seed)
    out = {label: [] for _, _, label in variants}
    for s in levels:
        wins = dict.fromkeys(out, 0)
        for _ in range(samples):
            v0 = perturbed_guess(net, s, rng)
            for method, mode, label in variants:
                wins[label] += matches(solve(net, method, mode, tol, max_iter, v0=v0), v_ref)
        for label in out:
            out[label].append(100.0 * wins[label] / samples)
        log(f"{name} level {s:.2f}: " + "  ".join(f"{k} {v[-1]:5.1f}%" for k, v in out.items()))
    return out


def perturbed_guess(net: Network, s: float, rng: np.random.Generator) -> np.ndarray:
    """Flat start with PQ magnitudes moved by U(-s, s) pu and non-slack angles by U(-s, s) rad."""
    vm, va = np.abs(net.v_flat), np.angle(net.v_flat)
    vm[net.pq] = 1.0 + rng.uniform(-s, s, net.pq.size)
    va[net.pvpq] = rng.uniform(-s, s, net.pvpq.size)
    return vm * np.exp(1j * va)


def find_difficult_start(net: Network, s: float, tol: float, max_iter: int, tries: int = 200):
    """First random initial guess (seed 0, 1, ...) from which NR fails but the hybrid NR-LI
    reaches the stored operating point. Returns ``(seed, v0)`` or ``None``."""
    v_ref = reference(net, tol)
    for seed in range(tries):
        v0 = perturbed_guess(net, s, np.random.default_rng(seed))
        if matches(solve(net, "nr", tol=tol, max_iter=max_iter, v0=v0), v_ref):
            continue
        if matches(solve(net, "nr-li", "safeguard", tol, max_iter, v0=v0), v_ref):
            return seed, v0
    return None
