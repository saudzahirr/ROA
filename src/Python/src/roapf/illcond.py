"""Construction of ill-conditioned power-flow cases with a known solution.

Starting from a standard case, three modifications that are known to make the
power flow hard are combined:

* ``pv_to_pq``  generators lose voltage control: every PV bus becomes a PQ bus
                whose reactive injection is frozen at its base-case value
                (generators at their reactive limits);
* ``rx``        all branch resistances are multiplied by ``rx`` (high R/X ratio,
                as in weak or distribution-like networks);
* ``loading``   all injections are scaled by a common factor up to a chosen
                fraction of the largest factor for which the operable solution
                can be followed by continuation.

Because the loading is increased by continuation from the base case, the
solution of every generated case is known to exist and is stored in the case.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import cases as c
from .cases import Case
from .network import Network, build_network
from .solvers import condition_number, solve


@dataclass
class IllCase:
    case: Case  # modified case; bus VM / VA hold the known solution
    net: Network
    v_solution: np.ndarray
    base: str
    pv_to_pq: bool
    rx: float
    lambda_max: float
    fraction: float
    cond: float  # cond_2 of the Jacobian at the solution

    @property
    def name(self) -> str:
        tag = self.base.removeprefix("case").lstrip("_")
        return (f"ill_{tag}_{'pq' if self.pv_to_pq else 'pv'}_rx{self.rx:g}"
                f"_load{round(1000 * self.fraction):04d}")


def _solve_from(net: Network, v0, max_iter=40):
    r = solve(net, "nr", tol=1e-9, max_iter=max_iter, v0=v0)
    ok = r.converged and float(np.abs(r.v).min()) > 0.3
    return ok, r.v


def _base_variant(case: Case, pv_to_pq: bool, rx: float) -> tuple[Case, np.ndarray] | None:
    """Apply the structural modifications and return the case with its solution."""
    net0 = build_network(case)
    ok, v = _solve_from(net0, net0.v_case)
    if not ok:
        return None
    bus, gen, branch = case.bus.copy(), case.gen.copy(), case.branch.copy()
    if pv_to_pq:
        s = net0.injections(v)
        ids = bus[:, c.BUS_I].astype(int)
        for i in net0.pv:
            bus[i, c.BUS_TYPE] = c.PQ
            rows = np.flatnonzero((gen[:, c.GEN_BUS].astype(int) == ids[i]) & (gen[:, c.GEN_STATUS] > 0))
            gen[rows, c.QG] = 0.0
            gen[rows[0], c.QG] = s[i].imag * case.base_mva + bus[i, c.QD]
    new = replace(case, bus=bus, gen=gen, branch=branch)
    # raise the resistances gradually so that the solution can be followed
    for k in np.linspace(1.0, rx, max(2, int(np.ceil((rx - 1.0) / 0.25)) + 1)):
        step = replace(new, branch=branch.copy())
        step.branch[:, c.BR_R] = case.branch[:, c.BR_R] * k
        ok, v = _solve_from(build_network(step), v)
        if not ok:
            return None
    new.branch[:, c.BR_R] = case.branch[:, c.BR_R] * rx
    return new, v


def make_ill_conditioned(case: Case, pv_to_pq: bool = True, rx: float = 1.0,
                         fractions=(0.9, 0.95, 0.99, 0.999)) -> list[IllCase]:
    """Ill-conditioned variants of ``case`` at the given fractions of the maximum loading."""
    made = _base_variant(case, pv_to_pq, rx)
    if made is None:
        return []
    variant, v_base = made
    net = build_network(variant)
    s0 = net.sbus.copy()

    lam, v, step = 1.0, v_base, 0.1
    while step > 1e-4:  # continuation, then bisection on the last interval
        net.sbus = (lam + step) * s0
        ok, v_new = _solve_from(net, v)
        if ok:
            lam, v = lam + step, v_new
        else:
            step /= 2

    out = []
    for frac in fractions:
        target, v = frac * lam, v_base
        for mu in np.linspace(1.0, target, 60)[1:]:
            net.sbus = mu * s0
            ok, v = _solve_from(net, v)
        if not ok:
            continue
        bus, gen = variant.bus.copy(), variant.gen.copy()
        bus[:, [c.PD, c.QD]] *= target
        gen[:, [c.PG, c.QG]] *= target
        bus[:, c.VM], bus[:, c.VA] = np.abs(v), np.degrees(np.angle(v))
        stressed = replace(variant, bus=bus, gen=gen, name=variant.name)
        snet = build_network(stressed)
        ok, v_sol = _solve_from(snet, v)
        if not ok:
            continue
        cond = condition_number(snet.jacobian(v_sol))
        out.append(IllCase(stressed, snet, v_sol, case.name, pv_to_pq, rx, lam, frac, cond))
    return out


def write_matpower(case: Case, path: str | Path, comment: str = "") -> None:
    """Write a case as a MATPOWER version-2 ``.m`` file (bus, gen and branch matrices)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def block(name, a):
        rows = "\n".join("\t" + "\t".join(f"{x:.10g}" for x in row) + ";" for row in a)
        return f"mpc.{name} = [\n{rows}\n];\n"

    text = [f"function mpc = {path.stem}", *(f"% {line}" for line in comment.splitlines()),
            "mpc.version = '2';", f"mpc.baseMVA = {case.base_mva:g};", "",
            block("bus", case.bus), block("gen", case.gen), block("branch", case.branch)]
    path.write_text("\n".join(text), encoding="utf-8")
