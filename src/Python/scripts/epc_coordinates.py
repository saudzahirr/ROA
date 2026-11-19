"""Extract bus coordinates from a GE PSLF .epc case file.

    uv run python scripts/epc_coordinates.py <file.epc> <case name>

Writes cases/coordinates/<case name>.csv with the columns bus, substation,
latitude, longitude. The .epc file lists the location of every substation and,
for every bus, the substation it belongs to.
"""

import csv
import re
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines()
subs, buses, mode = {}, [], None
for line in lines:
    if line.startswith("substation data"):
        mode = "sub"
    elif line.startswith("bus data"):
        mode = "bus"
    elif re.match(r"^[a-z].* data\s*\[", line):
        mode = None
    elif mode == "sub":
        m = re.match(r'\s*(\d+)\s+"[^"]*"\s*:\s*(-?[\d.]+)\s+(-?[\d.]+)', line)
        if m:
            subs[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    elif mode == "bus":
        m = re.match(r'\s*(\d+)\s+".*\s(\d+)\s+"[^"]*"\s+-?\d+\s+-?\d+\s+"[^"]*"\s*$', line)
        if m:
            buses.append((int(m.group(1)), int(m.group(2))))

out = Path(__file__).resolve().parents[1] / "cases" / "coordinates" / f"{sys.argv[2]}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["bus", "substation", "latitude", "longitude"])
    w.writerows([[b, s, *subs[s]] for b, s in buses])
print(f"{len(subs)} substations, {len(buses)} buses -> {out}")
