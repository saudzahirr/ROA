"""Newton-Raphson, gradient-type and hybrid power-flow solvers.

All methods work on the mismatch dy = y - f(x) and the Jacobian J = df/dx.

``nr``        Newton-Raphson,                         dx = J^-1 dy
``nr-ls``     NR with a backtracking line search on ||dy||_2 (step halving)
``lm``        Levenberg-Marquardt,                    (J^T J + lam D) dx = J^T dy, D = diag(J^T J)
``nr-li``     hybrid NR + Landweber Iteration (LI),   dx = J^T dy / sigma_max(J)^2
``nr-gd``     hybrid NR + fixed-step gradient descent
``nr-ls-li``  NR -> damped NR -> Landweber cascade
``nr-ls-lm``  NR -> damped NR -> Levenberg-Marquardt cascade

For ``nr-li`` / ``nr-gd`` two ways of combining the steps exist:

``safeguard``  (``Hybrid.m``) the NR step is accepted only if it reduces
               ||dy||_2; otherwise a correction step is taken from the same point.
``alternate``  every iteration takes an NR step and, if that did not converge,
               a correction step.

In the cascades a full NR step is tried first; if it does not reduce the
mismatch the step is halved (the NR direction is a descent direction of
0.5 ||dy||^2 whenever J is non-singular); only if no step length down to
1/64 helps, or J is singular, is the fallback step (LI or LM) taken.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .network import Network

METHODS = ("nr", "nr-gd", "nr-li")  # the three methods of the original manuscript
ALL_METHODS = ("nr", "nr-ls", "lm", "nr-gd", "nr-li", "nr-sli", "nr-ls-li", "nr-ls-sli", "nr-ls-lm",
               "tr-nr-li", "b-nr-li", "w-nr-li", "nr-sgd", "c-nr-li", "nr-ali", "w-nr-ali", "li", "ali", "cn-rk4", "cn-rk4a", "sgd")
MODES = ("safeguard", "alternate")
LABELS = {
    "nr": "Newton-Raphson (NR)",
    "nr-ls": "NR + line search",
    "lm": "Levenberg-Marquardt",
    "nr-gd": "NR + Gradient Descent",
    "nr-li": "NR + Landweber Iteration",
    "nr-ls-li": "NR + line search + Landweber",
    "nr-sli": "NR + scaled Landweber",
    "nr-ls-sli": "NR + line search + scaled Landweber",
    "nr-ls-lm": "NR + line search + LM",
    "tr-nr-li": "Trust-region NR-Landweber",
    "b-nr-li": "Bounded-step NR-Landweber",
    "w-nr-li": "Watchdog NR-Landweber",
    "nr-sgd": "NR + SGD (Adam)",
    "c-nr-li": "Contraction-certified NR-Landweber",
    "nr-ali": "NR + accelerated Landweber",
    "w-nr-ali": "Watchdog NR + accelerated Landweber",
    "li": "Landweber Iteration",
    "ali": "Accelerated Landweber Iteration",
    "cn-rk4": "Continuous Newton (RK4, h = 1)",
    "cn-rk4a": "Continuous Newton (RK4, adaptive h)",
}
MIN_STEP = 1.0 / 64.0


@dataclass
class Result:
    method: str
    converged: bool
    iterations: int
    v: np.ndarray  # complex bus voltages
    history: list[float]  # ||dy||_inf; history[k] is the mismatch after k iterations
    steps: list[str] = field(default_factory=list)  # step types taken, per iteration
    seconds: float = 0.0
    cond: list[float] = field(default_factory=list)  # cond_2(J) at the start of each iteration
    vmin: list[float] = field(default_factory=list)  # smallest |V| at the start of each iteration

    @property
    def n_correction(self) -> int:
        """Number of Landweber / gradient / LM steps taken."""
        return sum(s.count("LI") + s.count("GD") + s.count("LM") + s.count("DL") for s in self.steps)


def newton_step(jac: sp.spmatrix, dy: np.ndarray) -> np.ndarray | None:
    """Solve J dx = dy by sparse LU factorisation; ``None`` if J is singular."""
    try:
        with np.errstate(all="ignore"):
            dx = splu(sp.csc_matrix(jac)).solve(dy)
    except (RuntimeError, ValueError):
        return None
    return dx if np.all(np.isfinite(dx)) else None


def sigma_max_sq(jac: sp.spmatrix, tol: float = 1e-6, max_iter: int = 300) -> float:
    """Largest singular value squared of J.

    Exact for small systems; power iteration on J^T J for large sparse ones.
    """
    n = jac.shape[1]
    if n <= 100:
        return float(np.linalg.norm(jac.toarray(), 2) ** 2)
    x = np.random.default_rng(0).standard_normal(n)
    x /= np.linalg.norm(x)
    lam = 0.0
    for _ in range(max_iter):
        y = jac.T @ (jac @ x)
        new = float(np.linalg.norm(y))
        if new == 0.0 or not np.isfinite(new):
            return new
        x = y / new
        if abs(new - lam) <= tol * new:
            return new
        lam = new
    return lam


def condition_number(jac: sp.spmatrix) -> float:
    """2-norm condition number sigma_max / sigma_min of J.

    Exact (dense SVD) up to 1500 unknowns; for larger systems sigma_max comes from
    power iteration on J^T J and sigma_min from inverse power iteration with one
    sparse LU factorisation.
    """
    n = jac.shape[1]
    with np.errstate(all="ignore"):
        if n <= 1500:
            sv = np.linalg.svd(jac.toarray(), compute_uv=False)
            return float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        try:
            lu = splu(sp.csc_matrix(jac))
        except (RuntimeError, ValueError):
            return float("inf")
        x = np.random.default_rng(1).standard_normal(n)
        x /= np.linalg.norm(x)
        mu = 0.0
        for _ in range(40):  # largest eigenvalue of (J^T J)^-1 = 1 / sigma_min^2
            y = lu.solve(lu.solve(x, trans="T"))
            new = float(np.linalg.norm(y))
            if not np.isfinite(new) or new == 0.0:
                return float("inf")
            x = y / new
            if abs(new - mu) <= 1e-4 * new:
                mu = new
                break
            mu = new
        return float(np.sqrt(sigma_max_sq(jac) * mu))


def landweber_step(jac: sp.spmatrix, dy: np.ndarray) -> np.ndarray:
    """dx = alpha J^T dy with alpha = 1 / sigma_max(J)^2."""
    alpha = 1.0 / (sigma_max_sq(jac) + np.finfo(float).eps)
    return alpha * (jac.T @ dy)


def scaled_landweber_step(jac: sp.spmatrix, dy: np.ndarray) -> np.ndarray:
    """Landweber step in column-scaled variables.

    With D = diag(J^T J) and J_s = J D^-1/2 (unit column norms), the step is
    dx = D^-1/2 J_s^T dy / sigma_max(J_s)^2. Scaling removes the large spread
    between angle and magnitude sensitivities that makes the plain step short.
    """
    d = np.asarray(jac.multiply(jac).sum(axis=0)).ravel()
    w = 1.0 / np.sqrt(np.where(d > 0, d, 1.0))
    js = (jac @ sp.diags(w)).tocsc()
    alpha = 1.0 / (sigma_max_sq(js) + np.finfo(float).eps)
    return w * (alpha * (js.T @ dy))


def dogleg_step(p_newton, p_cauchy, radius: float) -> tuple[np.ndarray, str]:
    """Powell's dogleg: Newton step inside the trust region, otherwise the point of
    the path  0 -> Cauchy (gradient) point -> Newton point  that lies on its boundary."""
    if p_newton is not None and np.linalg.norm(p_newton) <= radius:
        return p_newton, "NR"
    nc = float(np.linalg.norm(p_cauchy))
    if p_newton is None or nc >= radius:
        return (radius / nc) * p_cauchy if nc > radius else p_cauchy, "LI"
    d = p_newton - p_cauchy
    a, b, c = float(d @ d), 2.0 * float(p_cauchy @ d), nc * nc - radius * radius
    tau = (-b + np.sqrt(max(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)
    return p_cauchy + tau * d, "DL"


def gradient_step(jac: sp.spmatrix, dy: np.ndarray, alpha: float) -> np.ndarray:
    """Plain steepest descent on 0.5 ||dy||^2 with a fixed step size."""
    return alpha * (jac.T @ dy)


def lm_step(jac: sp.spmatrix, dy: np.ndarray, lam: float) -> np.ndarray | None:
    """Levenberg-Marquardt step with Marquardt scaling: (J^T J + lam diag(J^T J)) dx = J^T dy."""
    a = (jac.T @ jac).tocsc()
    d = a.diagonal()
    d = np.where(d > 0, d, 1.0)
    try:
        with np.errstate(all="ignore"):
            dx = splu((a + lam * sp.diags(d)).tocsc()).solve(jac.T @ dy)
    except (RuntimeError, ValueError):
        return None
    return dx if np.all(np.isfinite(dx)) else None


def solve(
    net: Network,
    method: str = "nr-li",
    mode: str = "safeguard",
    tol: float = 1e-8,
    max_iter: int = 50,
    gd_step: float = 1e-4,
    v0: np.ndarray | None = None,
    min_step: float = MIN_STEP,
    armijo: float = 0.0,
    max_dtheta: float = 0.3,
    max_dv: float = 0.1,
    li_step: str = "landweber",
    watch: int = 2,
    retry: float = 0.25,
    theta_max: float = 1.0,
    track: bool = False,
    v_guard: float = 0.0,
    dtheta_cap: float = 0.0,
    restore: float = 0.0,
) -> Result:
    if method == "nr-sgd":
        return solve_nr_sgd(net, tol=tol, max_iter=max_iter, v0=v0, track=track)
    if method == "sgd":  # stochastic gradient descent alone: no Newton phase, one Adam step per iteration
        return solve_nr_sgd(net, tol=tol, max_iter=max_iter, v0=v0, nr_iters=0, sgd_steps=1, track=track)
    if method in ("cn-rk4", "cn-rk4a"):
        return solve_cn_rk4(net, tol=tol, max_iter=max_iter, v0=v0, adaptive=method == "cn-rk4a")
    if method not in ALL_METHODS:
        raise ValueError(f"method must be one of {ALL_METHODS}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    t0 = time.perf_counter()
    v = (net.v_flat if v0 is None else np.asarray(v0, dtype=complex)).copy()
    x = net.pack(v)
    dy = net.mismatch(v)
    history = [float(np.max(np.abs(dy)))]
    steps: list[str] = []
    converged = history[0] < tol
    lam = 1e-3  # Levenberg-Marquardt damping, adapted across iterations
    cond_log: list[float] = []
    vmin_log: list[float] = []
    radius = None  # trust-region radius of tr-nr-li, adapted across iterations
    wd = None  # watchdog state of w-nr-li: saved point while trial NR steps are running
    f_fail = np.inf  # mismatch at which the last watchdog trial failed
    n_restore = [0]  # restoration steps taken
    x_prev, k_acc = None, 0  # previous iterate and counter of the accelerated Landweber phase
    it = 0

    def norm_at(x_trial):
        return float(np.linalg.norm(net.mismatch(net.unpack(x_trial, v))))

    def correction(jac, dy):
        if method == "nr-gd":
            return gradient_step(jac, dy, gd_step), "GD"
        if method in ("nr-sli", "nr-ls-sli"):
            return scaled_landweber_step(jac, dy), "LI"
        return landweber_step(jac, dy), "LI"

    def lm_update(jac, dy, f0):
        """Raise the damping until the step reduces the mismatch (at most 12 times)."""
        nonlocal lam
        best = None
        for _ in range(12):
            step = lm_step(jac, dy, lam)
            if step is not None:
                best = step
                if norm_at(x + step) < f0:
                    lam = max(lam / 5.0, 1e-12)
                    return step
            lam *= 5.0
        return best

    def line_search(dx, f0):
        """Largest step length 1, 1/2, ... >= MIN_STEP that reduces the mismatch."""
        alpha = 1.0
        while alpha >= min_step:
            if norm_at(x + alpha * dx) < (1.0 - armijo * alpha) * f0:
                return alpha
            alpha *= 0.5
        return None

    with np.errstate(all="ignore"):
        while not converged and it < max_iter:
            it += 1
            jac = net.jacobian(v)
            f0 = float(np.linalg.norm(dy))
            taken = []
            if track:
                cond_log.append(condition_number(jac))
                vmin_log.append(float(np.abs(v).min()))

            if method == "li":  # pure Landweber iteration, no Newton steps
                x = x + landweber_step(jac, dy)
                taken.append("LI")
            elif method == "ali":  # pure Nesterov-accelerated Landweber with restart
                k_acc += 1
                z = x if x_prev is None else x + (k_acc - 1.0) / (k_acc + 2.0) * (x - x_prev)
                vz = net.unpack(z, v)
                cand = z + landweber_step(net.jacobian(vz), net.mismatch(vz))
                if x_prev is not None and not norm_at(cand) < f0:
                    k_acc = 1
                    cand = x + landweber_step(jac, dy)
                x_prev, x = x, cand
                taken.append("LI")
            elif method == "nr-ali":
                # NR with the acceptance test of Hybrid.m; the fallback is a
                # Nesterov-accelerated Landweber step:
                #     z = x_k + (k - 1)/(k + 2) (x_k - x_{k-1}),  x_{k+1} = z + omega J(z)^T dy(z).
                # The momentum is restarted whenever an NR step is accepted or the
                # accelerated step fails to reduce the mismatch.
                dx = newton_step(jac, dy)
                if dx is not None and norm_at(x + dx) < f0:
                    x, x_prev, k_acc = x + dx, None, 0
                    taken.append("NR")
                else:
                    k_acc += 1
                    z = x if x_prev is None else x + (k_acc - 1.0) / (k_acc + 2.0) * (x - x_prev)
                    vz = net.unpack(z, v)
                    cand = z + landweber_step(net.jacobian(vz), net.mismatch(vz))
                    if x_prev is not None and not norm_at(cand) < f0:  # restart
                        k_acc = 1
                        cand = x + landweber_step(jac, dy)
                    x_prev, x = x, cand
                    taken.append("LI")
            elif method == "c-nr-li":
                # Contraction-certified hybrid. The Newton step dx is accepted when
                # the natural (affine-invariant) monotonicity test holds:
                #     theta = ||J(x)^-1 F(x + dx)|| / ||dx|| < theta_max,
                # i.e. when the next simplified Newton correction is shorter than
                # the current one. theta estimates the local contraction factor of
                # Newton's method (Newton-Kantorovich), so the test asks whether x
                # lies in Newton's region of attraction instead of whether the
                # mismatch happens to decrease. Both solves reuse one LU factorisation.
                step = None
                try:
                    lu = splu(sp.csc_matrix(jac))
                    dx = lu.solve(dy)
                    if np.all(np.isfinite(dx)):
                        dx_bar = lu.solve(net.mismatch(net.unpack(x + dx, v)))
                        theta = float(np.linalg.norm(dx_bar)) / max(float(np.linalg.norm(dx)), 1e-300)
                        if np.isfinite(theta) and theta < theta_max:
                            step = dx
                except (RuntimeError, ValueError):
                    step = None
                if step is not None:
                    x = x + step
                    taken.append("NR")
                else:
                    x = x + landweber_step(jac, dy)
                    taken.append("LI")
            elif method in ("w-nr-li", "w-nr-ali"):
                # Watchdog hybrid. A full NR step that reduces ||dy||_2 is accepted
                # as in Hybrid.m. If it does not, NR is allowed up to `watch` further
                # trial steps; if the mismatch then falls below its value at the
                # saved point the trial is kept, otherwise the iterate is rolled
                # back and a (plain or accelerated) Landweber step is taken from the
                # saved point. A new trial is started only once the mismatch has
                # dropped to `retry` times the level of the last failed trial.
                def fallback():
                    nonlocal x_prev, k_acc
                    if method == "w-nr-li":
                        return x + landweber_step(jac, dy)
                    k_acc += 1
                    z = x if x_prev is None else x + (k_acc - 1.0) / (k_acc + 2.0) * (x - x_prev)
                    vz = net.unpack(z, v)
                    cand = z + landweber_step(net.jacobian(vz), net.mismatch(vz))
                    if x_prev is not None and not norm_at(cand) < f0:  # restart the momentum
                        k_acc = 1
                        cand = x + landweber_step(jac, dy)
                    # restoration: if the Landweber step would also take a PQ-bus
                    # magnitude below the guard, the iterate lies on the low-voltage
                    # side and descent leads away from the operable solution; move
                    # the fraction `restore` of the way back to the flat start instead
                    if restore > 0.0 and v_guard > 0.0 and cand.size > net.pvpq.size                             and not float(np.min(cand[net.pvpq.size:])) >= v_guard:
                        x_prev, k_acc = None, 0
                        n_restore[0] += 1
                        return x + restore * (net.pack(net.v_flat) - x)
                    x_prev = x
                    return cand

                p_n = newton_step(jac, dy)
                # voltage guard: a Newton step is only considered if the full step
                # leaves every PQ-bus magnitude above v_guard (a collapsing voltage is
                # the signature of a step that leaves the region of the operable solution)
                n_ang = net.pvpq.size
                phys = (p_n is not None and (v_guard <= 0.0 or p_n.size == n_ang
                                             or float(np.min((x + p_n)[n_ang:])) >= v_guard))
                # angle cap: an admissible Newton step is shortened as a whole so that
                # no angle changes by more than dtheta_cap rad in one iteration.
                if phys and dtheta_cap > 0.0:
                    worst = float(np.max(np.abs(p_n[:n_ang]), initial=0.0))
                    if worst > dtheta_cap:
                        p_n = p_n * (dtheta_cap / worst)
                if wd is not None:  # inside a trial
                    f_trial = norm_at(x + p_n) if phys else np.inf
                    if f_trial < wd["f"]:
                        x, wd, x_prev, k_acc = x + p_n, None, None, 0
                        taken.append("NR")
                    elif wd["left"] > 1 and np.isfinite(f_trial):
                        x = x + p_n
                        wd["left"] -= 1
                        taken.append("NR?")
                    else:
                        x, f_fail, wd = wd["next"], wd["f"], None
                        taken.append("LI")
                elif phys and norm_at(x + p_n) < f0:
                    x, x_prev, k_acc = x + p_n, None, 0
                    taken.append("NR")
                elif phys and watch > 0 and f0 <= retry * f_fail:
                    wd = {"f": f0, "next": fallback(), "left": watch}
                    x = x + p_n
                    taken.append("NR?")
                else:
                    x = fallback()
                    taken.append("LI")
            elif method == "b-nr-li":
                # Bounded-step hybrid: no update may move an angle by more than
                # max_dtheta rad or a magnitude by more than max_dv pu.
                n_ang = net.pvpq.size

                def bound(step):
                    worst = max(float(np.max(np.abs(step[:n_ang]), initial=0.0)) / max_dtheta,
                                float(np.max(np.abs(step[n_ang:]), initial=0.0)) / max_dv)
                    return step / worst if worst > 1.0 else step

                p_n = newton_step(jac, dy)
                done = False
                if p_n is not None:
                    full = bound(p_n)
                    scale = 1.0
                    for _ in range(3):  # bounded Newton step, then 1/2 and 1/4 of it
                        if norm_at(x + scale * full) < f0:
                            x = x + scale * full
                            limited = scale < 1.0 or full is not p_n
                            taken.append("NR*" if limited else "NR")
                            done = True
                            break
                        scale *= 0.5
                if not done:
                    g = jac.T @ dy
                    if li_step == "cauchy":
                        jg = jac @ g
                        step = (float(g @ g) / max(float(jg @ jg), 1e-300)) * g
                    else:
                        step = landweber_step(jac, dy)
                    x = x + bound(step)
                    taken.append("LI")
            elif method == "tr-nr-li":
                # Trust-region blend of the Newton step and the optimal-length
                # Landweber (Cauchy) step; one LU factorisation per iteration.
                p_n = newton_step(jac, dy)
                g = jac.T @ dy
                jg = jac @ g
                p_c = (float(g @ g) / max(float(jg @ jg), 1e-300)) * g
                if radius is None:
                    radius = float(np.linalg.norm(p_n if p_n is not None else p_c))
                accepted = False
                for _ in range(30):
                    step, tag = dogleg_step(p_n, p_c, radius)
                    predicted = f0**2 - float(np.linalg.norm(dy - jac @ step)) ** 2
                    f1 = norm_at(x + step)
                    rho = (f0**2 - f1**2) / predicted if predicted > 0 and np.isfinite(f1) else -1.0
                    size = float(np.linalg.norm(step))
                    if rho < 0.25:
                        radius = 0.25 * size
                    elif rho > 0.75 and size >= 0.99 * radius:
                        radius = 2.0 * radius
                    if rho > 1e-4:
                        accepted = True
                        break
                if not accepted:
                    break  # stationary point of ||dy||: no step reduces the mismatch
                x = x + step
                taken.append(tag)
            elif method == "lm":
                step = lm_update(jac, dy, f0)
                if step is None:
                    break
                x = x + step
                taken.append("LM")
            else:
                dx = newton_step(jac, dy)
                if method == "nr":
                    if dx is None:
                        break  # singular Jacobian: plain NR cannot continue
                    x = x + dx
                    taken.append("NR")
                elif method == "nr-ls":
                    if dx is None:
                        break
                    alpha = line_search(dx, f0) or min_step
                    x = x + alpha * dx
                    taken.append("NR" if alpha == 1.0 else f"NR*{alpha:g}")
                elif method in ("nr-ls-li", "nr-ls-sli", "nr-ls-lm"):
                    alpha = line_search(dx, f0) if dx is not None else None
                    if alpha is not None:
                        x = x + alpha * dx
                        taken.append("NR" if alpha == 1.0 else f"NR*{alpha:g}")
                    elif method != "nr-ls-lm":
                        x = x + correction(jac, dy)[0]
                        taken.append("LI")
                    else:
                        step = lm_update(jac, dy, f0)
                        if step is None:
                            break
                        x = x + step
                        taken.append("LM")
                elif mode == "safeguard":  # nr-li / nr-gd as in Hybrid.m
                    if dx is not None and norm_at(x + dx) < f0:
                        x = x + dx
                        taken.append("NR")
                    else:
                        step, tag = correction(jac, dy)
                        x = x + step
                        taken.append(tag)
                else:  # alternate
                    if dx is not None:
                        x = x + dx
                        v = net.unpack(x, v)
                        dy = net.mismatch(v)
                        taken.append("NR")
                    if dx is None or not np.max(np.abs(dy)) < tol:
                        jac = net.jacobian(v) if dx is not None else jac
                        step, tag = correction(jac, dy)
                        x = x + step
                        taken.append(tag)

            v = net.unpack(x, v)
            dy = net.mismatch(v)
            norm = float(np.max(np.abs(dy)))
            history.append(norm)
            steps.append("+".join(taken))
            if not np.isfinite(norm) or norm > 1e12:
                break  # diverged
            converged = norm < tol

    res = Result(method, bool(converged), it, v, history, steps, time.perf_counter() - t0, cond_log, vmin_log)
    res.n_restore = n_restore[0]
    return res


def solve_cn_rk4(net: Network, tol: float = 1e-8, max_iter: int = 50, v0: np.ndarray | None = None,
                 adaptive: bool = False, h_min: float = MIN_STEP) -> Result:
    """Continuous Newton method (Milano, IEEE Trans. Power Syst., 2009).

    The Newton iteration is read as the explicit Euler discretisation of
    dx/dt = -J(x)^-1 F(x); here the flow is integrated with the classical
    fourth-order Runge-Kutta scheme, x_{k+1} = x_k + h (k1 + 2 k2 + 2 k3 + k4) / 6,
    with one Jacobian factorisation per stage (four per iteration).
    ``adaptive=False`` uses the fixed step h = 1. ``adaptive=True`` halves h
    (down to ``h_min``) until the step reduces ||F||_2 and doubles it (up to 1)
    after an accepted step; this step-size rule is our choice.
    """
    t0 = time.perf_counter()
    v = (net.v_flat if v0 is None else np.asarray(v0, dtype=complex)).copy()
    x = net.pack(v)
    dy = net.mismatch(v)
    history = [float(np.max(np.abs(dy)))]
    steps: list[str] = []
    converged = history[0] < tol
    h, it = 1.0, 0

    def slope(x_at):
        v_at = net.unpack(x_at, v)
        return newton_step(net.jacobian(v_at), net.mismatch(v_at))

    def rk4(h):
        k1 = slope(x)
        k2 = slope(x + 0.5 * h * k1) if k1 is not None else None
        k3 = slope(x + 0.5 * h * k2) if k2 is not None else None
        k4 = slope(x + h * k3) if k3 is not None else None
        return None if k4 is None else x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    with np.errstate(all="ignore"):
        while not converged and it < max_iter:
            it += 1
            f0 = float(np.linalg.norm(dy))
            cand = rk4(h)
            if adaptive:
                while (cand is None or not float(np.linalg.norm(net.mismatch(net.unpack(cand, v)))) < f0)                         and h > h_min:
                    h *= 0.5
                    cand = rk4(h)
            if cand is None:
                break
            x = cand
            steps.append("RK4" if h == 1.0 else f"RK4*{h:g}")
            if adaptive:
                h = min(1.0, 2.0 * h)
            v = net.unpack(x, v)
            dy = net.mismatch(v)
            norm = float(np.max(np.abs(dy)))
            history.append(norm)
            if not np.isfinite(norm) or norm > 1e12:
                break
            converged = norm < tol
    return Result("cn-rk4a" if adaptive else "cn-rk4", bool(converged), it, v, history, steps,
                  time.perf_counter() - t0)


def dc_start(net: Network) -> np.ndarray:
    """Initial guess with DC power-flow angles (as MATPOWER's ``makeBdc``).

    Branch susceptances are 1/(x tap); phase shifters and bus shunt conductances
    enter the injections. Magnitudes are those of the flat start (1 pu at PQ
    buses, set-points at generators).
    """
    from . import cases as c

    case = net.case
    bus = case.bus
    n = bus.shape[0]
    ids = bus[:, c.BUS_I].astype(int)
    look = np.full(ids.max() + 1, -1)
    look[ids] = np.arange(n)
    br = case.branch[case.branch[:, c.BR_STATUS] > 0]
    f, t = look[br[:, c.F_BUS].astype(int)], look[br[:, c.T_BUS].astype(int)]
    b = 1.0 / br[:, c.BR_X] / np.where(br[:, c.TAP] == 0, 1.0, br[:, c.TAP])
    bbus = sp.coo_matrix((np.r_[b, b, -b, -b], (np.r_[f, t, f, t], np.r_[f, t, t, f])), shape=(n, n)).tocsc()
    shift = -np.deg2rad(br[:, c.SHIFT]) * b
    p_shift = np.zeros(n)
    np.add.at(p_shift, f, shift)
    np.add.at(p_shift, t, -shift)
    p = net.sbus.real - p_shift - bus[:, c.GS] / case.base_mva
    pvpq = net.pvpq
    theta = np.angle(net.v_flat).copy()
    try:
        theta[pvpq] = splu(bbus[pvpq][:, pvpq].tocsc()).solve(p[pvpq] - bbus[pvpq][:, net.ref] @ theta[net.ref])
    except (RuntimeError, ValueError):
        return net.v_flat.copy()
    if not np.all(np.isfinite(theta)):
        return net.v_flat.copy()
    return np.abs(net.v_flat) * np.exp(1j * theta)


def solve_nr_sgd(net: Network, tol: float = 1e-8, max_iter: int = 100, v0: np.ndarray | None = None,
                 sgd_steps: int = 50, nr_iters: int = 10, eta: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, seed: int = 0, track: bool = False) -> Result:
    """Re-implementation of the NR + stochastic gradient descent hybrid of
    Costilla-Enriquez, Weng and Zhang (IEEE Trans. Power Syst., 2021).

    NR runs first. If it has not converged after ``nr_iters`` iterations, or it
    diverges, the iterate is reset to the point at which that NR phase started,
    ``sgd_steps`` stochastic gradient steps are taken (a random bus is drawn in
    every step and only its mismatch terms are differentiated; Adam with the
    step size and decay rates of the reference), and NR is tried again. The
    switching rule and the phase lengths are not specified in the reference and
    are our choice. ``iterations`` counts NR iterations; SGD steps are counted
    as fallback steps.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    v = (net.v_flat if v0 is None else np.asarray(v0, dtype=complex)).copy()
    x = net.pack(v)
    history = [float(np.max(np.abs(net.mismatch(v))))]
    steps: list[str] = []
    npvpq, npq = net.pvpq.size, net.pq.size
    cond_log: list[float] = []
    vmin_log: list[float] = []
    # rows of the mismatch vector that belong to each non-slack bus
    q_row = {int(b): npvpq + k for k, b in enumerate(net.pq)}
    rows_of = [[k] + ([q_row[int(b)]] if int(b) in q_row else []) for k, b in enumerate(net.pvpq)]
    m = np.zeros(npvpq + npq)
    u = np.zeros(npvpq + npq)
    adam_t = 0
    it = 0
    converged = history[0] < tol
    with np.errstate(all="ignore"):
        while not converged and it < max_iter:
            x_start = x.copy()
            ok = True
            for _ in range(nr_iters):  # ---- Newton-Raphson phase
                if it >= max_iter:
                    break
                it += 1
                v = net.unpack(x, v)
                if track:
                    cond_log.append(condition_number(net.jacobian(v)))
                    vmin_log.append(float(np.abs(v).min()))
                dx = newton_step(net.jacobian(v), net.mismatch(v))
                if dx is None:
                    ok = False
                    break
                x = x + dx
                norm = float(np.max(np.abs(net.mismatch(net.unpack(x, v)))))
                history.append(norm)
                steps.append("NR")
                if not np.isfinite(norm) or norm > 1e8:
                    ok = False
                    break
                if norm < tol:
                    converged = True
                    break
            if converged or it >= max_iter:
                break
            x = x_start if not ok or history[-1] > 1e3 else x
            if track:
                v_now = net.unpack(x, v)
                cond_log.append(condition_number(net.jacobian(v_now)))
                vmin_log.append(float(np.abs(v_now).min()))
            for _ in range(sgd_steps):  # ---- stochastic gradient phase (Adam)
                v = net.unpack(x, v)
                rows = rows_of[int(rng.integers(len(rows_of)))]
                jac = net.jacobian(v).tocsr()[rows]
                g = -(jac.T @ net.mismatch(v)[rows])  # gradient of 0.5 (f_i - s_i)^2
                adam_t += 1
                m = beta1 * m + (1 - beta1) * g
                u = beta2 * u + (1 - beta2) * g * g
                x = x - eta * (m / (1 - beta1**adam_t)) / (np.sqrt(u / (1 - beta2**adam_t)) + 1e-8)
            steps.append(f"GD*{sgd_steps}")
            history.append(float(np.max(np.abs(net.mismatch(net.unpack(x, v))))))
            if nr_iters == 0:  # gradient steps only: every Adam step counts as an iteration
                it += 1
                converged = history[-1] < tol
    v = net.unpack(x, v)
    return Result("nr-sgd" if nr_iters else "sgd", bool(converged), it, v, history, steps, time.perf_counter() - t0,
                  cond_log, vmin_log)


class LogVoltage:
    """The same network in the coordinates x = [theta ; u] with |V| = exp(u).

    Voltage magnitudes stay positive for every update, and a Newton or gradient
    step in u is a relative change of the magnitude. All solvers accept this
    wrapper in place of a :class:`Network`.
    """

    def __init__(self, net: Network):
        self._net = net

    def __getattr__(self, name):
        return getattr(self._net, name)

    def pack(self, v):
        return np.concatenate([np.angle(v[self._net.pvpq]), np.log(np.abs(v[self._net.pq]))])

    def unpack(self, x, v_ref):
        n = self._net.pvpq.size
        va, vm = np.angle(v_ref), np.abs(v_ref)
        va[self._net.pvpq] = x[:n]
        vm[self._net.pq] = np.exp(np.clip(x[n:], -50.0, 50.0))
        return vm * np.exp(1j * va)

    def jacobian(self, v):
        n = self._net.pvpq.size
        scale = np.concatenate([np.ones(n), np.abs(v[self._net.pq])])  # d/du = |V| d/d|V|
        return (self._net.jacobian(v) @ sp.diags(scale)).tocsc()
