"""Test systems in MATPOWER case format.

* ``case5``  - the ill-conditioned 5-bus system of the manuscript (Tables 1-2).
* ``case14``, ``case30``, ``case57``, ``case118``, ``case300`` - IEEE systems
  shipped with PYPOWER (identical to the MATPOWER case files).

Only the columns needed for a basic power flow are used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

import numpy as np

# MATPOWER column indices (0-based)
BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, VA = 0, 1, 2, 3, 4, 5, 7, 8
GEN_BUS, PG, QG, VG, GEN_STATUS = 0, 1, 2, 5, 7
F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS = 0, 1, 2, 3, 4, 8, 9, 10
PQ, PV, REF, ISOLATED = 1, 2, 3, 4


@dataclass
class Case:
    name: str
    base_mva: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    # optional {bus number: (x, y)} layout used by the voltage temperature map
    coords: dict[int, tuple[float, float]] = field(default_factory=dict)

    @property
    def n_bus(self) -> int:
        return self.bus.shape[0]


def case5() -> Case:
    """Ill-conditioned 5-bus system (manuscript Tables 1 and 2), 100 MVA base.

    Bus 1 slack, bus 3 PV (|V| = 1.05), bus 2 carries an 8 + j2.8 pu load, which
    pulls its voltage down to about 0.83 pu.
    """
    #        bus type  Pd    Qd   Gs Bs area Vm   Va baseKV zone Vmax Vmin
    bus = np.array([
        [1, REF, 0.0,   0.0,   0, 0, 1, 1.00, 0, 345, 1, 1.5, 0.5],
        [2, PQ,  800.0, 280.0, 0, 0, 1, 1.00, 0, 345, 1, 1.5, 0.5],
        [3, PV,  80.0,  40.0,  0, 0, 1, 1.05, 0, 345, 1, 1.5, 0.5],
        [4, PQ,  0.0,   0.0,   0, 0, 1, 1.00, 0, 345, 1, 1.5, 0.5],
        [5, PQ,  0.0,   0.0,   0, 0, 1, 1.00, 0, 345, 1, 1.5, 0.5],
    ], dtype=float)
    #        bus Pg    Qg Qmax  Qmin  Vg   mBase status Pmax Pmin
    gen = np.array([
        [1, 0.0,   0, 9999, -9999, 1.00, 100, 1, 9999, 0],
        [3, 520.0, 0, 400,  -280,  1.05, 100, 1, 9999, 0],
    ], dtype=float)
    #        from to r     x     b     rateA B C tap shift status
    branch = np.array([
        [1, 5, 0.00, 0.02, 0.00, 0, 0, 0, 0, 0, 1],
        [2, 4, 0.01, 0.10, 1.72, 0, 0, 0, 0, 0, 1],
        [2, 5, 0.01, 0.05, 0.88, 0, 0, 0, 0, 0, 1],
        [3, 4, 0.00, 0.01, 0.00, 0, 0, 0, 0, 0, 1],
        [4, 5, 0.00, 0.03, 0.44, 0, 0, 0, 0, 0, 1],
    ], dtype=float)
    return Case("case5", 100.0, bus, gen, branch, coords=dict(_COORDS_5))


def _from_pypower(name: str) -> Case:
    ppc = getattr(import_module(f"pypower.{name}"), name)()
    return Case(
        name,
        float(ppc["baseMVA"]),
        np.asarray(ppc["bus"], dtype=float),
        np.asarray(ppc["gen"], dtype=float),
        np.asarray(ppc["branch"], dtype=float),
        coords=dict(_COORDS_14) if name == "case14" else {},
    )


AVAILABLE = ("case5", "case14", "case30", "case39", "case57", "case118", "case300")


def load_case(name: str | int | Path, case_dir: str | Path | None = None) -> Case:
    """Load a case by name (``"case14"``), size (``14``) or MATPOWER ``.m`` file.

    Names that are not built in are looked up as ``<case_dir>/<name>.m``.
    """
    key = f"case{name}" if str(name).isdigit() else str(name)
    if key == "case5":
        return case5()
    if key in AVAILABLE:
        return _from_pypower(key)
    path = Path(key)
    if path.suffix != ".m" and case_dir is not None:
        path = Path(case_dir) / f"{key}.m"
    if path.suffix == ".m" and path.exists():
        return load_matpower(path)
    raise ValueError(f"unknown case {name!r}; built in: {', '.join(AVAILABLE)}")


def _matrix(text: str, field_name: str) -> np.ndarray:
    m = re.search(rf"mpc\.{field_name}\s*=\s*\[(.*?)\];", text, flags=re.S)
    if m is None:
        raise ValueError(f"mpc.{field_name} not found")
    rows = []
    for line in m.group(1).replace(";", "\n").splitlines():
        line = line.split("%")[0].strip()
        if line:
            rows.append(np.array(line.replace(",", " ").split(), dtype=float))
    return np.vstack(rows)


def load_matpower(path: str | Path) -> Case:
    """Read a MATPOWER version-2 case file (bus, gen and branch matrices only).

    Isolated buses are dropped together with their branches and generators.
    Files that rescale their data after the matrices, or that contain DC lines,
    are rejected because this reader would silently ignore those parts.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"^\s*mpc\.(bus|branch)\(.*\)\s*=", text, flags=re.M):
        raise ValueError(f"{path.name}: data are post-processed inside the file")
    if re.search(r"^\s*mpc\.dcline\s*=", text, flags=re.M):
        raise ValueError(f"{path.name}: DC lines are not supported")
    base = float(re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)", text).group(1))
    bus, gen, branch = _matrix(text, "bus"), _matrix(text, "gen"), _matrix(text, "branch")
    live = bus[:, BUS_TYPE] != ISOLATED
    ids = bus[live, BUS_I]
    bus = bus[live]
    gen = gen[np.isin(gen[:, GEN_BUS], ids)]
    branch = branch[np.isin(branch[:, F_BUS], ids) & np.isin(branch[:, T_BUS], ids)]
    return Case(path.stem, base, bus, gen, branch)


# Single-line-diagram layouts (arbitrary units, y pointing up) matching the
# manuscript's temperature-map figures.
_COORDS_5 = {1: (385, -847), 2: (915, -847), 3: (990, -393), 4: (1295, -695), 5: (840, -1150)}
_COORDS_14 = {
    1: (165, -700), 2: (585, -1197), 3: (1308, -1350), 4: (1315, -730), 5: (890, -857),
    6: (920, -633), 7: (1383, -588), 8: (1525, -545), 9: (1330, -500), 10: (1155, -430),
    11: (993, -400), 12: (580, -290), 13: (890, -203), 14: (1147, -288),
}
