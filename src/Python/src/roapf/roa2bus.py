"""Region of attraction (ROA) of 13 power-flow methods on the 2-bus system.

Python port of the MATLAB scripts in https://github.com/saudzahirr/ROA.

    Bus 1: slack, V1 = 1.0 /_ 0 pu
    Bus 2: PQ load, S2 = 0.9 + j0.6 pu
    Line : Z = j0.1 pu

Every method is a function ``(v0, d0, sys, tol, max_iter) -> iterations | None``
taking the initial guess |V2| = v0, theta2 = d0 and returning the number of
iterations needed to converge, or ``None`` if it did not. The iteration logic
follows the MATLAB code line by line (including its convergence tests and
safeguards) so that the published figures are reproduced.

Plain Python floats are used instead of NumPy: for a 2x2 problem this is far
faster, and a 256 x 256 sweep takes a few seconds per method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

EPS = 2.220446049250313e-16


@dataclass(frozen=True)
class TwoBus:
    z: complex = 0.1j  # line impedance, pu
    p_load: float = 0.9  # pu
    q_load: float = 0.6  # pu

    @property
    def y_off(self) -> complex:  # Y12 = Y21
        return -1.0 / self.z

    @property
    def y_diag(self) -> complex:  # Y22
        return 1.0 / self.z

    @property
    def p_spec(self) -> float:
        return -self.p_load

    @property
    def q_spec(self) -> float:
        return -self.q_load

    @property
    def s_spec(self) -> complex:
        return complex(self.p_spec, self.q_spec)


# --------------------------------------------------------------------------
# shared building blocks
# --------------------------------------------------------------------------
def injection(v: float, d: float, sys: TwoBus) -> tuple[float, float]:
    """P2, Q2 injected at bus 2 for |V2| = v, theta2 = d."""
    vc = v * complex(math.cos(d), math.sin(d))
    s = vc * (sys.y_off * 1.0 + sys.y_diag * vc).conjugate()
    return s.real, s.imag


def mismatch(v: float, d: float, sys: TwoBus) -> tuple[float, float, float, float]:
    """dP, dQ (specified - calculated) and the calculated P2, Q2."""
    p, q = injection(v, d, sys)
    return sys.p_spec - p, sys.q_spec - q, p, q


def jacobian(v: float, p: float, q: float, sys: TwoBus) -> tuple[float, float, float, float]:
    """[dP/dtheta, dP/dV; dQ/dtheta, dQ/dV] at bus 2, returned as (a, b, c, d)."""
    g, b = sys.y_diag.real, sys.y_diag.imag
    return (-q - b * v * v, p / v + g * v, p - g * v * v, q / v - b * v)


def solve2(a, b, c, d, r1, r2) -> tuple[float, float]:
    """Solve [a b; c d] x = r (ZeroDivisionError if singular)."""
    det = a * d - b * c
    return (d * r1 - b * r2) / det, (a * r2 - c * r1) / det


def singular_values_sq(a, b, c, d) -> tuple[float, float]:
    """Squared largest / smallest singular values of [a b; c d]."""
    t = a * a + b * b + c * c + d * d
    det = a * d - b * c
    root = math.sqrt(max(t * t - 4.0 * det * det, 0.0))
    return 0.5 * (t + root), max(0.5 * (t - root), 0.0)


def rcond1(a, b, c, d) -> float:
    """Reciprocal 1-norm condition number of [a b; c d] (0 if singular)."""
    det = a * d - b * c
    if det == 0.0 or not math.isfinite(det):
        return 0.0
    n1 = max(abs(a) + abs(c), abs(b) + abs(d))
    ninv = max(abs(d) + abs(c), abs(b) + abs(a)) / abs(det)
    return 1.0 / (n1 * ninv)


def landweber_dx(jac, dp, dq) -> tuple[float, float]:
    """dx = alpha J^T dy with alpha = 1 / (sigma_max^2 + eps)."""
    a, b, c, d = jac
    alpha = 1.0 / (singular_values_sq(a, b, c, d)[0] + EPS)
    return alpha * (a * dp + c * dq), alpha * (b * dp + d * dq)


# --------------------------------------------------------------------------
# Jacobian-based methods
# --------------------------------------------------------------------------
def newton_raphson(v, d, sys, tol, max_iter):
    for it in range(1, max_iter + 1):
        dp, dq, p, q = mismatch(v, d, sys)
        if max(abs(dp), abs(dq)) < tol:
            return it
        dd, dv = solve2(*jacobian(v, p, q, sys), dp, dq)
        d, v = d + dd, v + dv
    return None


def gradient_descent(v, d, sys, tol, max_iter, alpha=0.01):
    for it in range(1, max_iter + 1):
        dp, dq, p, q = mismatch(v, d, sys)
        if max(abs(dp), abs(dq)) < tol:
            return it
        a, b, c, e = jacobian(v, p, q, sys)
        d, v = d + alpha * (a * dp + c * dq), v + alpha * (b * dp + e * dq)
    return None


def landweber(v, d, sys, tol, max_iter):
    for it in range(1, max_iter + 1):
        dp, dq, p, q = mismatch(v, d, sys)
        if max(abs(dp), abs(dq)) < tol:
            return it
        dd, dv = landweber_dx(jacobian(v, p, q, sys), dp, dq)
        d, v = d + dd, v + dv
    return None


def hybrid(v, d, sys, tol, max_iter):
    """Hybrid NR-LI: take the NR step if it reduces ||dy||_2, else a Landweber step."""
    for it in range(1, max_iter + 1):
        dp, dq, p, q = mismatch(v, d, sys)
        if max(abs(dp), abs(dq)) < tol:
            return it
        jac = jacobian(v, p, q, sys)
        use_li = not rcond1(*jac) > 1e-12
        if not use_li:
            dd, dv = solve2(*jac, dp, dq)
            if not (math.isfinite(dd) and math.isfinite(dv)):
                use_li = True
            else:
                try:
                    dp_t, dq_t, _, _ = mismatch(v + dv, d + dd, sys)
                    improved = math.hypot(dp_t, dq_t) < math.hypot(dp, dq)
                except (ValueError, OverflowError):
                    improved = False
                if improved:
                    d, v = d + dd, v + dv
                else:
                    use_li = True
        if use_li:
            dd, dv = landweber_dx(jac, dp, dq)
            d, v = d + dd, v + dv
    return None


def levenberg_marquardt(v, d, sys, tol, max_iter):
    lam = 0.01
    for it in range(1, max_iter + 1):
        dp, dq, p, q = mismatch(v, d, sys)
        if max(abs(dp), abs(dq)) < tol:
            return it
        a, b, c, e = jacobian(v, p, q, sys)
        # (J^T J + lambda I) dx = J^T F
        m11, m12, m22 = a * a + c * c + lam, a * b + c * e, b * b + e * e + lam
        g1, g2 = a * dp + c * dq, b * dp + e * dq
        if rcond1(m11, m12, m12, m22) < 1e-14:
            dd, dv = np.linalg.pinv(np.array([[m11, m12], [m12, m22]])) @ np.array([g1, g2])
        else:
            dd, dv = solve2(m11, m12, m12, m22, g1, g2)
        try:
            dp_t, dq_t, _, _ = mismatch(v + dv, d + dd, sys)
            improved = math.hypot(dp_t, dq_t) < math.hypot(dp, dq)
        except (ValueError, OverflowError):
            improved = False
        if improved:
            d, v, lam = d + dd, v + dv, lam / 10.0
        else:
            lam *= 10.0
    return None


def fast_decoupled(v, d, sys, tol, max_iter):
    b_p = sys.y_off.imag  # B'  (dP/dtheta approximation)
    b_pp = -sys.y_diag.imag  # B'' (dQ/dV approximation)
    for it in range(1, max_iter + 1):
        dp, dq, _, _ = mismatch(v, d, sys)
        d, v = d + dp / b_p, v + dq / b_pp
        if max(abs(dp), abs(dq)) < tol:
            return it
    return None


def iwamoto(v, d, sys, tol, max_iter):
    """NR with an optimal-multiplier step length (finite-difference curvature)."""
    eps_fd = 1e-4
    for it in range(1, max_iter + 1):
        dp, dq, p, q = mismatch(v, d, sys)
        if max(abs(dp), abs(dq)) < tol:
            return it
        a, b, c, e = jac = jacobian(v, p, q, sys)
        s_max, s_min = singular_values_sq(*jac)
        if not s_max < 1e16 * s_min:  # cond(J) > 1e8
            return None
        dd, dv = solve2(*jac, dp, dq)
        g1, g2 = a * dd + b * dv, c * dd + e * dv
        p_p, q_p = injection(v + eps_fd * dv, d + eps_fd * dd, sys)
        scale = 0.5 * eps_fd**2 + EPS
        q1 = (p_p - p - eps_fd * g1) / scale
        q2 = (q_p - q - eps_fd * g2) / scale
        denom = g1 * g1 + g2 * g2 + dp * q1 + dq * q2
        mult = -(dp * g1 + dq * g2) / denom
        if not math.isfinite(mult) or mult <= 0 or mult > 2:
            mult = 1.0
        d, v = d + mult * dd, v + mult * dv
    return None


# --------------------------------------------------------------------------
# derivative-free / quasi-Newton methods
# --------------------------------------------------------------------------
def secant(v, d, sys, tol, max_iter, perturbation=0.01):
    """Component-wise secant updates (theta from dP, |V| from dQ)."""
    v0, d0, v1, d1 = v, d, v + perturbation, d + perturbation
    f0 = mismatch(v0, d0, sys)[:2]
    f1 = mismatch(v1, d1, sys)[:2]
    for it in range(1, max_iter + 1):
        if math.hypot(*f1) < tol:
            return it
        if math.hypot(f1[0] - f0[0], f1[1] - f0[1]) < 1e-14:
            return None
        df_t, df_v = f1[0] - f0[0], f1[1] - f0[1]
        d_new = d1 - f1[0] * (d1 - d0) / df_t if abs(df_t) > 1e-14 else d1
        v_new = v1 - f1[1] * (v1 - v0) / df_v if abs(df_v) > 1e-14 else v1
        v0, d0, f0 = v1, d1, f1
        v1, d1 = v_new, d_new
        f1 = mismatch(v1, d1, sys)[:2]
        if v1 < 0.1 or v1 > 3.0 or abs(d1) > math.pi:
            return None
    return None


def broyden(v, d, sys, tol, max_iter):
    """Broyden's (good) method, started from the analytic Jacobian as in Broyden.m."""
    dp, dq, p, q = mismatch(v, d, sys)
    b11, b12, b21, b22 = jacobian(v, p, q, sys)
    for it in range(1, max_iter + 1):
        if max(abs(dp), abs(dq)) < tol:
            return it
        if not abs(b11 * b22 - b12 * b21) > 1e-12:
            return None
        s1, s2 = solve2(b11, b12, b21, b22, -dp, -dq)
        d_new, v_new = d + s1, v + s2
        if v_new < 0.1 or v_new > 3.0 or abs(d_new) > math.pi:
            return None
        d, v = d_new, v_new
        dp_old, dq_old = dp, dq
        dp, dq, _, _ = mismatch(v, d, sys)
        ss = s1 * s1 + s2 * s2
        if math.sqrt(ss) > 1e-12:
            r1 = (dp - dp_old) - (b11 * s1 + b12 * s2)
            r2 = (dq - dq_old) - (b21 * s1 + b22 * s2)
            b11, b12 = b11 + r1 * s1 / ss, b12 + r1 * s2 / ss
            b21, b22 = b21 + r2 * s1 / ss, b22 + r2 * s2 / ss
    return None


# --------------------------------------------------------------------------
# fixed-point methods on the complex voltage
# --------------------------------------------------------------------------
def _gs_update(vc: complex, sys: TwoBus) -> complex:
    """V2 = (conj(S2) / conj(V2) - Y21 V1) / Y22."""
    return (sys.s_spec.conjugate() / vc.conjugate() - sys.y_off * 1.0) / sys.y_diag


def _sor(v, d, sys, tol, max_iter, omega):
    vc = v * complex(math.cos(d), math.sin(d))
    for it in range(1, max_iter + 1):
        new = vc + omega * (_gs_update(vc, sys) - vc)
        if abs(new - vc) < tol:
            return it
        vc = new
    return None


def gauss_seidel(v, d, sys, tol, max_iter):
    return _sor(v, d, sys, tol, max_iter, 1.0)


def jacobi(v, d, sys, tol, max_iter):
    # with a single unknown bus Jacobi and Gauss-Seidel coincide
    return _sor(v, d, sys, tol, max_iter, 1.0)


def sor(v, d, sys, tol, max_iter, omega=1.2):
    return _sor(v, d, sys, tol, max_iter, omega)


def fixed_point(v, d, sys, tol, max_iter, relaxation=0.7):
    """Under-relaxed fixed-point iteration in polar form (FixedPoint.m)."""

    def clamp(x, lo, hi):
        return max(lo, min(x, hi))

    for it in range(1, max_iter + 1):
        v_old, d_old = v, d
        vc = v * complex(math.cos(d), math.sin(d))
        if not abs(vc) > 1e-10:
            return None
        new = (1 - relaxation) * vc + relaxation * _gs_update(vc, sys)
        v_new, d_new = abs(new), math.atan2(new.imag, new.real)
        if it > 50:  # fall back to a scaled-mismatch update if still not converged
            dp, dq, _, _ = mismatch(v_old, d_old, sys)
            d_new = d_old + 0.1 * dp / (v_old**2 + 1e-6)
            v_new = v_old + 0.1 * dq / (v_old + 1e-6)
        v = clamp((1 - relaxation) * v_old + relaxation * v_new, 0.1, 3.0)
        d = clamp((1 - relaxation) * d_old + relaxation * d_new, -math.pi, math.pi)
        if max(abs(v - v_old), abs(d - d_old)) < tol:
            dp, dq, _, _ = mismatch(v, d, sys)
            if max(abs(dp), abs(dq)) < tol * 10:
                return it
    return None


# --------------------------------------------------------------------------
# registry and sweep
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Method:
    key: str
    title: str
    filename: str
    func: Callable
    levels: tuple[int, ...]  # labelled iteration contours
    tol: float = 1e-5
    max_iter: int = 25
    v_range: tuple[float, float] = (0.0, 5.0)
    scale_to_data: bool = False  # colour axis ends at the largest count found


METHODS: dict[str, Method] = {
    m.key: m
    for m in [
        Method("jacobi", "Jacobi Method", "Jacobi-RegionOfAttraction", jacobi, tuple(range(5, 11))),
        Method("gauss-seidel", "Gauss-Seidel Method", "GaussSeidel-RegionOfAttraction", gauss_seidel,
               tuple(range(5, 11))),
        Method("newton-raphson", "Newton-Raphson Method", "NewtonRaphson-RegionOfAttraction",
               newton_raphson, (4, 5, 6, 7, 8, 9)),
        Method("secant", "Secant Method", "SecantMethod-RegionOfAttraction", secant, (5, 10, 15, 20),
               scale_to_data=True),
        Method("levenberg-marquardt", "Levenberg-Marquardt Method",
               "LevenbergMarquardt-RegionOfAttraction", levenberg_marquardt, (4, 5, 6, 7, 8, 9)),
        Method("gradient-descent", "Gradient Descent Method", "GradientDescent-RegionOfAttraction",
               gradient_descent, tuple(range(12, 26, 3))),
        Method("landweber", "Landweber Iteration", "LandweberIteration-RegionOfAttraction", landweber,
               (11, 13, *range(15, 26))),
        Method("fast-decoupled", "Fast Decoupled Method", "FastDecoupled-RegionOfAttraction",
               fast_decoupled, (5, 7, 9, *range(10, 16), *range(16, 26, 2))),
        Method("broyden", "Broyden's Method", "Broyden-RegionOfAttraction", broyden,
               tuple(range(1, 16, 2)), scale_to_data=True),
        Method("fixed-point", "Fixed Point Iteration", "FixedPoint-RegionOfAttraction", fixed_point,
               (14, *range(17, 26)), max_iter=100, scale_to_data=True),
        Method("iwamoto", "Iwamoto Method", "Iwamoto-RegionOfAttraction", iwamoto, (5, 6, 7, 8, 25),
               tol=1e-6, v_range=(0.1, 3.0)),
        Method("sor", "SOR Method", "SOR-RegionOfAttraction", sor, (11, 13, 14, 15, 16)),
        Method("hybrid", "Hybrid Method", "Hybrid-RegionOfAttraction", hybrid, (5, 6, 7, 8, 12)),
    ]
}


@dataclass
class Sweep:
    method: Method
    v: np.ndarray
    delta: np.ndarray
    iterations: np.ndarray  # (len(v), len(delta)); NaN where the method did not converge

    @property
    def converged_fraction(self) -> float:
        return float(np.mean(~np.isnan(self.iterations)))


def sweep(method: Method | str, n: int = 256, sys: TwoBus | None = None) -> Sweep:
    """Iteration count over an n x n grid of initial guesses (|V2|, theta2)."""
    method = METHODS[method] if isinstance(method, str) else method
    sys = sys or TwoBus()
    v = np.linspace(*method.v_range, n)
    delta = np.linspace(-math.pi / 2, math.pi / 2, n)
    out = np.full((n, n), np.nan)
    for i, v0 in enumerate(v.tolist()):
        for j, d0 in enumerate(delta.tolist()):
            try:
                it = method.func(v0, d0, sys, method.tol, method.max_iter)
            except (ZeroDivisionError, OverflowError, ValueError):
                it = None  # inf / NaN in MATLAB: the iteration never converges
            if it is not None:
                out[i, j] = it
    return Sweep(method, v, delta, out)


# --------------------------------------------------------------------------
# solvability region in the (P, Q) load plane
# --------------------------------------------------------------------------
@dataclass
class Solvability:
    p: np.ndarray  # pu
    q: np.ndarray  # pu
    solvable: np.ndarray  # (len(p), len(q)) bool: NR converged from a flat start
    p_boundary: np.ndarray
    q_boundary: np.ndarray  # analytic boundary Q = B/4 - P^2/B


def solvability(n: int = 256, x12: float = 0.1, tol: float = 1e-5, max_iter: int = 25) -> Solvability:
    p = np.linspace(0.01, 6.0, n)
    q = np.linspace(0.01, 3.0, n)
    ok = np.zeros((n, n), dtype=bool)
    for i, pl in enumerate(p.tolist()):
        for j, ql in enumerate(q.tolist()):
            try:
                ok[i, j] = newton_raphson(1.0, 0.0, TwoBus(1j * x12, pl, ql), tol, max_iter) is not None
            except (ZeroDivisionError, OverflowError, ValueError):
                pass
    b = 1.0 / x12
    pb = np.linspace(0.0, 6.0, 500)
    return Solvability(p, q, ok, pb, b / 4.0 - pb**2 / b)
