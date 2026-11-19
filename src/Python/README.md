# roapf

Python implementation of the power flow methods and experiments: an N-bus
power flow in polar coordinates with sparse matrices, the guarded
Newton-Landweber method, the comparison methods, the benchmarks and the
figure scripts.

## Setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## The guarded Newton-Landweber method

```python
from roapf.cases import load_case
from roapf.network import build_network
from roapf.solvers import solve

net = build_network(load_case("case118"))
result = solve(net, "w-nr-ali", tol=1e-8, v_guard=0.5, dtheta_cap=0.5, restore=0.5)
print(result.converged, result.iterations)
```

| Argument | Meaning |
| --- | --- |
| `v_guard` | a Newton step is admissible only if the full step leaves every load-bus voltage above this value (p.u.) |
| `dtheta_cap` | an admissible Newton step is shortened so that no angle changes by more than this value (rad) |
| `restore` | if the Landweber step would also break the guard, the iterate moves this fraction of the way back to the flat start |
| `watch`, `retry` | length of the look-ahead trial and the factor that limits repeated trials |

Other methods of `solve`: `"nr"` (Newton-Raphson), `"li"` (Landweber iteration),
`"sgd"` (stochastic gradient descent), `"nr-sgd"` (Newton-Raphson with stochastic
gradient descent phases), `"nr-li"` (Newton-Raphson with Landweber fallback),
and several further variants listed in `roapf.solvers.ALL_METHODS`.

## Layout

| Path | Contents |
| --- | --- |
| `src/roapf/network.py` | bus admittance matrix, mismatch vector and Jacobian |
| `src/roapf/solvers.py` | all iterative methods |
| `src/roapf/cases.py` | test systems (IEEE cases from PYPOWER, MATPOWER case files) |
| `src/roapf/illcond.py` | construction of ill-conditioned systems with a known solution |
| `src/roapf/roa2bus.py` | two-bus system and the region-of-attraction sweep |
| `src/roapf/figures.py`, `plots.py`, `layouts.py`, `style.py` | figures |
| `scripts/run_benchmarks.py` | random starting points, flat and d.c. starts, loading stress, protocol of the NR + SGD study |
| `scripts/run_illcond.py`, `run_guard_limit.py`, `run_timing.py`, `run_restoration_use.py` | further experiments |
| `scripts/roa_ieee14.py`, `roa2bus_*.py`, `plot_*.py`, `make_figures.py` | region-of-attraction maps and figures |
| `scripts/make_matlab_ieee14.py` | writes `../MATLAB/ROA_IEEE14.m` with the IEEE 14-bus data embedded |
| `cases/coordinates` | substation coordinates of the synthetic ACTIVSg systems, for the voltage maps |

## Test systems

The IEEE 14-, 30-, 39-, 57-, 118- and 300-bus systems come with PYPOWER and
need no files. The larger systems are MATPOWER case files (PEGASE, RTE, Polish
and ACTIVSg cases); they are not part of this repository. Put the `.m` files in
`cases/matpower`, or set the environment variable `ROAPF_CASES` to their folder.

Results are written to `results/` and figures to `figures/`; both folders are
ignored by git.
