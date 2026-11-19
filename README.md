# ROA
Power Flow Region of Attraction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24382341.svg)](https://zenodo.org/records/24382341)

Region-of-attraction analysis of power flow methods, and a guarded
Newton-Landweber method that is built on it.

---

## Repository layout

```
src/
  MATLAB/    two-bus region-of-attraction maps of the individual methods,
             IEEE 14-bus maps, export of the maps at print size
  Python/    roapf: N-bus power flow, the guarded Newton-Landweber method,
             comparison methods, benchmarks and figure scripts
```

### `src/MATLAB`

One script per method. Each sweeps the starting voltage magnitude and angle of
the load bus of a two-bus system over a 256 x 256 grid and draws the number of
iterations to convergence.

| Script | Method |
| --- | --- |
| `NewtonRaphson.m` | Newton-Raphson |
| `LandweberIteration.m` | Landweber iteration |
| `SGD.m` | gradient descent with a fixed step (stochastic gradient descent on a system with one load bus) |
| `NRSGD.m` | Newton-Raphson with stochastic gradient descent phases |
| `Hybrid.m` | Newton-Raphson with Landweber fallback |
| `Guarded.m` | guarded Newton-Landweber method (voltage guard, angle cap, look-ahead trial, accelerated Landweber step, restoration step) |
| `LevenbergMarquardt.m`, `Iwamoto.m`, `GaussSeidel.m`, `Jacobi.m`, `SOR.m`, `FixedPoint.m`, `FDPF.m`, `SecantMethod.m`, `Broyden.m`, `GradientDescent.m` | further methods |
| `solvability.m` | solvable region of the two-bus system in the load plane |
| `ROA_IEEE14.m` | IEEE 14-bus maps of Newton-Raphson and the guarded method (self-contained: the network data are embedded) |
| `export_figures.m`, `printStyle.m` | run the method scripts off-screen and export the maps at print size |
| `main.m` | runs the method scripts |

Run from `src/MATLAB`:

```matlab
NewtonRaphson            % one map
ROA_IEEE14               % IEEE 14-bus maps
setenv('ROA_ONLY', 'NewtonRaphson,LandweberIteration,SGD,NRSGD,Hybrid,Guarded'); export_figures
```

The exported maps go to `figures/roa` at the top of the repository, or to the
folder named by the environment variable `ROA_FIG_DIR`. No toolbox is needed.

### `src/Python`

See [`src/Python/README.md`](src/Python/README.md).

---

## How to Cite

If you use this software in your research, please cite it as follows:

> **Zahir, Muhammad Saud.** (2025). *saudzahirr/ROA: v1.0.0 (Version 1.0.0)* [Computer software]. **Zenodo.** [https://doi.org/10.5281/zenodo.17477506](https://doi.org/10.5281/zenodo.17477506)

### BibTeX

```bibtex
@software{zahir_roa_2025,
  author       = {Muhammad Saud Zahir},
  title        = {saudzahirr/ROA: v1.0.0 (Version 1.0.0)},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17477506},
  url          = {https://doi.org/10.5281/zenodo.17477506}
}
```
