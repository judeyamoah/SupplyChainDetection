"""
Stage 6 - Build the 80-feature table by reusing 04_feature_engineering.extract_features unchanged.
Labels come only from the OSV index (non-malware advisories). Output: data_v2/features_v2.csv
Usage: python v2_06_features.py [--fe path/to/04_feature_engineering.py]
"""

import argparse, importlib.util, sys
from pathlib import Path
import pandas as pd
from v2_common import D, read_jsonl


def load_fe(path):
    spec = importlib.util.spec_from_file_location("fe04", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def expand(rec):
    r = dict(rec)
    r["readme"] = "x" * int(r.get("_readme_length", 0))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fe", default=None)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    cands = (
        [a.fe]
        if a.fe
        else [
            here / "04_feature_engineering.py",
            here.parent / "scripts" / "04_feature_engineering.py",
            here.parent / "04_feature_engineering.py",
        ]
    )
    fe_path = next((c for c in cands if c and Path(c).exists()), None)
    if not fe_path:
        sys.exit("Cannot find 04_feature_engineering.py; pass --fe")
    fe = load_fe(fe_path)
    final = pd.read_csv(
        D / f"final_set{a.suffix}.csv",
        dtype={"package_name": str},
        keep_default_na=False,
    )
    meta = {r["name"]: r for r in read_jsonl(D / "metadata.jsonl")}
    dl = {r["package_name"]: r for r in read_jsonl(D / "downloads.jsonl")}
    osv = {
        r.package_name: {
            "label": 1,
            "vulnerability_count": int(float(r.n_advisories)),
            "aliases": str(r.advisory_ids).split(";"),
            "severity": [],
        }
        for r in final[final.label == 1].itertuples()
    }
    rows = []
    for r in final.itertuples():
        row = fe.extract_features(expand(meta[r.package_name]), osv, dl)
        row["label"] = r.label
        for c in ["role", "dl_bin", "dl_tier", "age_bin", "created", "first_published"]:
            row[c] = getattr(r, c, "")
        rows.append(row)
    out = fe.clean_dataframe(pd.DataFrame(rows))
    out.to_csv(D / f"features_v2{a.suffix}.csv", index=False)
    print(out.shape, "positives:", int(out.label.sum()))
