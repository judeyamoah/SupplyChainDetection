"""
Stage 5 - Match controls to positives on download bin x age bin, then write the final labelled set.
For each (download bin, age bin) cell: keep all positives; sample up to RATIO x n_pos controls from the
uniform+supplement pool in the same cell. Cells where controls run short are reported (match_report.csv);
do NOT hide them in the paper.
Output: data_v2/final_set.csv, data_v2/match_report.csv
"""

import argparse
import numpy as np, pandas as pd
from datetime import datetime, timezone
from v2_common import D, SEED, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--ratio", type=float, default=2.0)
ap.add_argument(
    "--fine",
    action="store_true",
    help="half-decade download bins (tighter matching on popularity)",
)
ap.add_argument(
    "--suffix",
    default="",
    help="e.g. _fine: writes final_set_fine.csv / match_report_fine.csv, leaving the originals untouched",
)
a = ap.parse_args()
rng = np.random.default_rng(SEED)
frame = pd.read_csv(
    D / "sample_frame.csv", dtype={"package_name": str}, keep_default_na=False
)
osv = pd.read_csv(
    D / "osv_index.csv", dtype={"package_name": str}, keep_default_na=False
).set_index("package_name")
meta = {r["name"]: r for r in read_jsonl(D / "metadata.jsonl")}
dl = {r["package_name"]: r for r in read_jsonl(D / "downloads.jsonl")}
now = datetime.now(timezone.utc)
rows = []
for n, role in zip(frame.package_name, frame.role):
    if n not in meta or n not in dl:
        continue
    c = pd.to_datetime(meta[n]["time"]["created"], utc=True)
    rows.append(
        {
            "package_name": n,
            "role": role,
            "label": int(role == "pos"),
            "monthly_downloads": dl[n]["monthly_downloads"],
            "age_days": (now - c).days,
            "created": c.isoformat(),
        }
    )
df = pd.DataFrame(rows)
COARSE = [-1, 9, 99, 999, 9999, 99999, 999999, np.inf]
FINE = [
    -1,
    9,
    31,
    99,
    316,
    999,
    3162,
    9999,
    31622,
    99999,
    316227,
    999999,
    3162277,
    np.inf,
]
DL_EDGES = FINE if a.fine else COARSE
AGE_EDGES = [-1, 365, 730, 1460, 2920, np.inf]
df["dl_bin"] = pd.cut(df.monthly_downloads, DL_EDGES, labels=False)
df["dl_tier"] = pd.cut(
    df.monthly_downloads, COARSE, labels=False
)  # always decade tiers, used for reporting/audits
df["age_bin"] = pd.cut(df.age_days, AGE_EDGES, labels=False)
keep, rep = [], []
for (d, g), cell in df.groupby(["dl_bin", "age_bin"]):
    pos = cell[cell.label == 1]
    ctl = cell[cell.label == 0]
    want = int(round(a.ratio * len(pos))) if len(pos) else 0
    take = (
        ctl.sample(n=min(want, len(ctl)), random_state=SEED) if want else ctl.iloc[0:0]
    )
    keep += [pos, take]
    rep.append(
        {
            "dl_bin": d,
            "age_bin": g,
            "positives": len(pos),
            "controls_available": len(ctl),
            "controls_wanted": want,
            "controls_taken": len(take),
            "shortfall": want - len(take),
        }
    )
final = pd.concat(keep)
final = final.join(
    osv[["n_advisories", "advisory_ids", "has_cve", "has_ghsa", "first_published"]],
    on="package_name",
)
final.to_csv(D / f"final_set{a.suffix}.csv", index=False)
r = pd.DataFrame(rep)
r.to_csv(D / f"match_report{a.suffix}.csv", index=False)
print(r.to_string(index=False))
print(
    f"\nfinal: {int(final.label.sum())} positives, {int((final.label == 0).sum())} controls; "
    f"control shortfall total {int(r.shortfall.clip(lower=0).sum())}"
)
print("Edges: downloads", DL_EDGES, "age(days)", AGE_EDGES)
