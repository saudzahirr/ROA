"""Power-flow equations for an N-bus network (manuscript Eqs. 1-3).

State vector      x  = [theta(PV, PQ) ; |V|(PQ)]
Mismatch vector   dy = [P_spec - P(x) (PV, PQ) ; Q_spec - Q(x) (PQ)]
Jacobian          J  = d[P; Q] / dx

All matrices are sparse, so systems with tens of thousands of buses fit in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import cases as c
from .cases import Case


@dataclass
class Network:
    """Everything the solvers need, with buses renumbered 0..n-1."""

    case: Case
    ybus: sp.csr_matrix  # (n, n) complex bus admittance matrix
    sbus: np.ndarray  # (n,) complex specified injection, pu
    ref: np.ndarray
    pv: np.ndarray
    pq: np.ndarray
    v_flat: np.ndarray  # flat start: 1 pu / 0 rad at PQ buses, set-points at generators
    v_case: np.ndarray  # voltages stored in the case file (usually a solved operating point)

    @property
    def pvpq(self) -> np.ndarray:
        return np.concatenate([self.pv, self.pq])

    @property
    def n_state(self) -> int:
        return self.pv.size + 2 * self.pq.size

    # ---- state vector <-> complex voltages --------------------------------
    def pack(self, v: np.ndarray) -> np.ndarray:
        return np.concatenate([np.angle(v[self.pvpq]), np.abs(v[self.pq])])

    def unpack(self, x: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
        """Insert state ``x`` into a copy of ``v_ref`` (which fixes slack and PV magnitudes)."""
        npvpq = self.pvpq.size
        va, vm = np.angle(v_ref), np.abs(v_ref)
        va[self.pvpq] = x[:npvpq]
        vm[self.pq] = x[npvpq:]
        return vm * np.exp(1j * va)

    # ---- equations ----------------------------------------------------------
    def injections(self, v: np.ndarray) -> np.ndarray:
        """Complex power injected at every bus, S = V conj(Y V)."""
        return v * np.conj(self.ybus @ v)

    def mismatch(self, v: np.ndarray) -> np.ndarray:
        ds = self.sbus - self.injections(v)
        return np.concatenate([ds[self.pvpq].real, ds[self.pq].imag])

    def jacobian(self, v: np.ndarray) -> sp.csc_matrix:
        y = self.ybus
        ibus = y @ v
        diag_v = sp.diags(v)
        diag_i = sp.diags(ibus)
        diag_vn = sp.diags(v / np.abs(v))
        ds_dva = (1j * diag_v @ (diag_i - y @ diag_v).conj()).tocsr()
        ds_dvm = (diag_v @ (y @ diag_vn).conj() + diag_i.conj() @ diag_vn).tocsr()
        pvpq, pq = self.pvpq, self.pq
        return sp.bmat([
            [ds_dva[pvpq][:, pvpq].real, ds_dvm[pvpq][:, pq].real],
            [ds_dva[pq][:, pvpq].imag, ds_dvm[pq][:, pq].imag],
        ], format="csc")


def build_network(case: Case) -> Network:
    bus, gen, branch = case.bus, case.gen, case.branch
    n = bus.shape[0]
    ids = bus[:, c.BUS_I].astype(int)
    lookup = np.full(ids.max() + 1, -1)
    lookup[ids] = np.arange(n)

    # ---- Ybus (pi-model branches with off-nominal taps and phase shifters)
    br = branch[branch[:, c.BR_STATUS] > 0]
    f, t = lookup[br[:, c.F_BUS].astype(int)], lookup[br[:, c.T_BUS].astype(int)]
    ys = 1.0 / (br[:, c.BR_R] + 1j * br[:, c.BR_X])
    tap = np.where(br[:, c.TAP] == 0, 1.0, br[:, c.TAP]) * np.exp(1j * np.deg2rad(br[:, c.SHIFT]))
    ytt = ys + 0.5j * br[:, c.BR_B]
    yff = ytt / (tap * np.conj(tap))
    yft, ytf = -ys / np.conj(tap), -ys / tap
    ysh = (bus[:, c.GS] + 1j * bus[:, c.BS]) / case.base_mva
    rows = np.concatenate([f, f, t, t, np.arange(n)])
    cols = np.concatenate([f, t, f, t, np.arange(n)])
    ybus = sp.coo_matrix((np.concatenate([yff, yft, ytf, ytt, ysh]), (rows, cols)), shape=(n, n)).tocsr()

    # ---- specified injections and generator voltage set-points
    on = gen[gen[:, c.GEN_STATUS] > 0]
    gbus = lookup[on[:, c.GEN_BUS].astype(int)]
    sbus = -(bus[:, c.PD] + 1j * bus[:, c.QD]) / case.base_mva
    np.add.at(sbus, gbus, (on[:, c.PG] + 1j * on[:, c.QG]) / case.base_mva)
    vm0 = np.ones(n)
    vm0[gbus[::-1]] = on[::-1, c.VG]  # the first in-service generator at a bus sets its voltage
    has_gen = np.zeros(n, dtype=bool)
    has_gen[gbus] = True

    # ---- bus types (a PV bus without an in-service generator is a PQ bus)
    btype = bus[:, c.BUS_TYPE].astype(int)
    ref = np.flatnonzero(btype == c.REF)
    pv = np.flatnonzero((btype == c.PV) & has_gen)
    pq = np.flatnonzero((btype == c.PQ) | ((btype == c.PV) & ~has_gen))
    vm0[pq] = 1.0

    va0 = np.zeros(n)
    va0[ref] = np.deg2rad(bus[ref, c.VA])
    v_case = bus[:, c.VM] * np.exp(1j * np.deg2rad(bus[:, c.VA]))
    fixed = np.concatenate([ref, pv])
    v_case[fixed] *= vm0[fixed] / np.abs(v_case[fixed])
    return Network(case, ybus, sbus, ref, pv, pq, vm0 * np.exp(1j * va0), v_case)
