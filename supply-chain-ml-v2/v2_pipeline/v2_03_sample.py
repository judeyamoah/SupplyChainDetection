"""
Stage 3 - Draw the random sampling frame.
  positives : random sample of packages with >=1 non-malware OSV advisory (all advisory IDs kept)
  controls  : uniform random sample of registry names that appear NOWHERE in OSV (not even malware-only)
  supplement: optional --supplement-file of popular package names (added to the control pool, tagged 'supp')
              Needed because a uniform sample contains almost no high-download packages, while vulnerable
              packages are mostly popular. Disclose this in the paper. Source suggestions: `npm i download-counts`
              (top packages by downloads) or any top-N list you can cite.
Output: data_v2/sample_frame.csv (package_name, role)
"""

import argparse, json, random
import pandas as pd
from v2_common import D, SEED

ap = argparse.ArgumentParser()
ap.add_argument("--n-pos", type=int, default=1500)
ap.add_argument(
    "--pool",
    type=int,
    default=15000,
    help="size of uniform random control candidate pool",
)
ap.add_argument("--supplement-file")
a = ap.parse_args()
rng = random.Random(SEED)
osv = pd.read_csv(
    D / "osv_index.csv", dtype={"package_name": str}, keep_default_na=False
)
names = (D / "names.txt").read_text(encoding="utf-8").splitlines()
in_osv = set(osv.package_name)
vuln = sorted(osv.loc[osv.n_advisories > 0, "package_name"])
name_set = set(names)
vuln = [v for v in vuln if v in name_set]  # still published
pos = rng.sample(vuln, min(a.n_pos, len(vuln)))
cand = [n for n in names if n not in in_osv]
ctrl = rng.sample(cand, min(a.pool, len(cand)))
rows = [(p, "pos") for p in pos] + [(c, "ctrl") for c in ctrl]
if a.supplement_file:
    t = open(a.supplement_file, encoding="utf-8").read().strip()
    sup = json.loads(t) if t.startswith("[") else t.splitlines()
    have = set(ctrl)
    rows += [
        (s, "supp")
        for s in dict.fromkeys(sup)
        if s not in in_osv and s not in have and s in name_set
    ]
pd.DataFrame(rows, columns=["package_name", "role"]).to_csv(
    D / "sample_frame.csv", index=False
)
print(
    f"positives {len(pos)} of {len(vuln)} vulnerable published packages; controls {len(ctrl)}; total {len(rows)}"
)
