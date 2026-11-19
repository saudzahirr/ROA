"""Draw every figure of the paper at final size (Nature Portfolio style).

    uv run python scripts/make_figures.py [fig1 fig2 fig3 fig5 fig6 ed1 ed2 ed3 ed4]

Output: figures/*.pdf (vector) and *.png (600 dpi).
Set ROAPF_CASES to a folder with MATPOWER .m files to override cases/matpower.
"""

import os
import sys
from pathlib import Path

from roapf import figures
from roapf.cases import load_case
from roapf.illcond import make_ill_conditioned
from roapf.network import build_network

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ["ROAPF_FIG_OUT"]) if os.environ.get("ROAPF_FIG_OUT") else ROOT / "figures"
FLAT = not os.environ.get("ROAPF_MERGED")  # default: every figure starts at panel a (ROAPF_MERGED=1: lettered for stacking)
CASE_DIR = os.environ.get("ROAPF_CASES", str(ROOT / "cases" / "matpower"))
RESULTS = ROOT / "results" / "journal"
NAMES = {"case14": "IEEE 14-bus", "case39": "IEEE 39-bus", "case118": "IEEE 118-bus",
         "case_ACTIVSg500": "ACTIVSg 500-bus", "case_ACTIVSg2000": "ACTIVSg 2000-bus"}
ROA_TITLES = {
    "newton-raphson": "Newton-Raphson", "levenberg-marquardt": "Levenberg-Marquardt", "landweber": "Landweber",
    "hybrid": "NR-Landweber hybrid", "iwamoto": "Iwamoto", "gauss-seidel": "Gauss-Seidel", "sor": "SOR",
    "fixed-point": "Fixed point", "fast-decoupled": "Fast decoupled", "secant": "Secant", "broyden": "Broyden",
    "gradient-descent": "Gradient descent", "jacobi": "Jacobi",
}
todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig5", "fig6", "fig4", "fig8", "fig9"]
cache = {}

if "fig1" in todo:  # panel b of Fig. 1; panel a and Fig. 4 are TikZ drawings (not part of this repository)
    print(figures.solvability_map(OUT / "Fig1b_solvability_map"))


def net(name):
    return build_network(load_case(name, CASE_DIR))


def solved(name):
    n = net(name)
    return n, figures.run(n, "NR-ALI-GC").v, NAMES[name]


if "fig2" in todo:
    keys = ["newton-raphson", "levenberg-marquardt", "landweber", "hybrid"]
    print(figures.roa_maps(keys, [ROA_TITLES[k] for k in keys], OUT / "Fig2_region_of_attraction", cache=cache))
if "fig3" in todo:
    print(figures.mechanism([(net("case39"), NAMES["case39"], 0.2)], OUT / "Fig3_mechanism"))
if "fig5" in todo:
    rob = figures.read_robustness(RESULTS, NAMES)
    print(figures.sampled_roa(rob, NAMES, OUT / "Fig5_sampled_region_of_attraction", first=0 if FLAT else 6))
if "fig6" in todo:
    cases = []
    for name, title in (("case39", "IEEE 39-bus without voltage control"),
                        ("case118", "IEEE 118-bus without voltage control")):
        ic = make_ill_conditioned(load_case(name, CASE_DIR), True, 1.0, fractions=(0.99,))[0]
        cases.append((ic, f"{title} (99% of maximum loading)"))
    print(figures.ill_conditioned(cases, OUT / "Fig6_ill_conditioned", first=0 if FLAT else 4))
if "fig8" in todo:
    print(figures.voltage_maps([solved("case14"), solved("case39")], OUT / "Fig8_voltage_maps_ieee"))
if "ed1" in todo:
    keys = ["iwamoto", "gauss-seidel", "sor", "fixed-point", "fast-decoupled", "secant", "broyden",
            "gradient-descent", "jacobi"]
    print(figures.roa_maps(keys, [ROA_TITLES[k] for k in keys], OUT / "ED_Fig1_region_of_attraction_other", ncols=3,
                           cache=cache))
if "fig4" in todo:
    print(figures.mechanism([(net("case14"), NAMES["case14"], 0.35), (net("case118"), NAMES["case118"], 0.25),
                             (net("case_ACTIVSg500"), NAMES["case_ACTIVSg500"], 0.1)], OUT / "Fig4_mechanism_other",
                            first=0 if FLAT else 3, legend=FLAT))
if "fig9" in todo:
    print(figures.voltage_maps([solved("case_ACTIVSg500"), solved("case_ACTIVSg2000")],
                               OUT / "Fig9_voltage_maps_activsg"))
