import json
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/edward/Downloads")
hits = sorted(ROOT.glob("**/audit_baseline.json"))
if not hits:
    raise SystemExit("audit_baseline.json not found anywhere under ~/Downloads")
print("baseline:", hits[0], "\n")
base = json.load(open(hits[0]))
data = ROOT / "dataset" / "inertial_data"

print(f"{'dataset':<12}{'clips b/n':>16}{'frames b/n':>22}{'max_kmh b/n':>26}")
for ds, b in base.items():
    files = sorted((data / ds).glob("*.parquet"))
    if not files:
        print(f"{ds:<12}  MISSING")
        continue
    n_fr, mx = 0, float("-inf")
    for f in files:
        d = pd.read_parquet(f, columns=["speed_kmh"])
        n_fr += len(d)
        m = d["speed_kmh"].max()
        if pd.notna(m):
            mx = max(mx, m)
    if len(files) != b["clips"] or n_fr != b["frames"]:
        flag = "  <-- STRUCTURE MOVED"
    elif b["max_kmh"] == b["max_kmh"] and abs(mx - b["max_kmh"]) > 1e-6:
        flag = "  <-- VALUES MOVED"
    else:
        flag = ""
    print(f"{ds:<12}{b['clips']:>8}/{len(files):<7}"
          f"{b['frames']:>11,}/{n_fr:<10,}"
          f"{b['max_kmh']:>13.4f}/{mx:<12.4f}{flag}")