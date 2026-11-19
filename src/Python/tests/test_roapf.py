import math

import numpy as np
import pytest

from roapf import roa2bus
from roapf.cases import AVAILABLE, load_case
from roapf.network import build_network
from roapf.solvers import METHODS, solve


def test_two_bus_closed_form():
    """NR from a flat start reaches the high-voltage root of the 2-bus system."""
    sys = roa2bus.TwoBus()
    x, p, q = 0.1, sys.p_load, sys.q_load
    # V^4 + (2 Q X - 1) V^2 + X^2 (P^2 + Q^2) = 0
    b = 2 * q * x - 1
    v_exact = math.sqrt((-b + math.sqrt(b * b - 4 * x * x * (p * p + q * q))) / 2)
    v, d = 1.0, 0.0
    for _ in range(10):
        dp, dq, pc, qc = roa2bus.mismatch(v, d, sys)
        dd, dv = roa2bus.solve2(*roa2bus.jacobian(v, pc, qc, sys), dp, dq)
        d, v = d + dd, v + dv
    assert v == pytest.approx(v_exact, abs=1e-10)


@pytest.mark.parametrize("key", list(roa2bus.METHODS))
def test_flat_start_converges(key):
    m = roa2bus.METHODS[key]
    assert m.func(1.0, 0.0, roa2bus.TwoBus(), m.tol, m.max_iter) is not None


def test_jacobian_matches_finite_differences():
    net = build_network(load_case("case14"))
    v = net.v_flat * np.exp(1j * np.linspace(0, -0.2, net.case.n_bus))
    x, jac, h = net.pack(v), net.jacobian(v).toarray(), 1e-6
    for k in range(x.size):
        xp = x.copy()
        xp[k] += h
        col = -(net.mismatch(net.unpack(xp, v)) - net.mismatch(v)) / h
        assert np.allclose(col, jac[:, k], atol=1e-4)


@pytest.mark.parametrize("name", AVAILABLE)
def test_methods_agree(name):
    net = build_network(load_case(name))
    ref = solve(net, "nr")
    assert ref.converged
    for method in METHODS:
        r = solve(net, method, mode="safeguard")
        assert r.converged and np.allclose(r.v, ref.v, atol=1e-8)
    r = solve(net, "nr-li", mode="alternate")
    assert r.converged and np.allclose(r.v, ref.v, atol=1e-8)


def test_case14_matches_published_solution():
    net = build_network(load_case("case14"))
    r = solve(net, "nr-li", tol=1e-10)
    # MATPOWER case14 stores the solved voltages
    assert np.allclose(np.abs(r.v), net.case.bus[:, 7], atol=2e-3)
