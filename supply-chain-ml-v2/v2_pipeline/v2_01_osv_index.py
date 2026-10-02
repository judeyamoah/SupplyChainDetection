"""
Stage 1 - Index every npm advisory in the OSV bulk export.
Output: data_v2/osv_index.csv  (one row per package)
  package_name, n_advisories, advisory_ids (;), has_cve, has_ghsa, first_published, is_malware_only
Positives later = packages with n_non_malware > 0. Malware-only packages are excluded from BOTH classes
(they are a different phenomenon: intentionally malicious, not vulnerable).
"""
import argparse, datetime, json, zipfile, csv
from collections import defaultdict
from pathlib import Path
from v2_common import D, BASE

ap = argparse.ArgumentParser()
ap.add_argument("--zip", help="path to the local OSV npm zip (default: search for osv_npm_all.zip / all.zip)")
ap.add_argument("--download", action="store_true", help="download from OSV instead of using a local file")
a = ap.parse_args()

URL = "https://storage.googleapis.com/osv-vulnerabilities/npm/all.zip"
if a.download:
    import requests
    from v2_common import HEADERS
    zpath = D / "osv_npm_all.zip"
    r = requests.get(URL, headers=HEADERS, timeout=300); r.raise_for_status(); zpath.write_bytes(r.content)
else:
    names = ["osv_npm_all.zip", "all.zip"]
    cands = [Path(a.zip)] if a.zip else [d / n for d in (D, BASE, BASE / "data", BASE / "data" / "raw", Path.cwd(), Path.home() / "Downloads") for n in names]
    zpath = next((c for c in cands if c.exists()), None)
    if zpath is None:
        raise SystemExit("Local OSV zip not found. Pass --zip /path/to/osv_npm_all.zip (or --download).")
print(f"using {zpath}")
z = zipfile.ZipFile(zpath)
pk = defaultdict(lambda: {"ids": set(), "mal": set(), "aliases": set(), "pub": []})
for n in z.namelist():
    if not n.endswith(".json"):
        continue
    a = json.loads(z.read(n))
    if a.get("withdrawn"):
        continue
    aid = a.get("id", "")
    for aff in a.get("affected", []):
        p = aff.get("package", {})
        if p.get("ecosystem") != "npm" or not p.get("name"):
            continue
        d = pk[p["name"]]
        (d["mal"] if aid.startswith("MAL-") else d["ids"]).add(aid)
        d["aliases"].update(a.get("aliases", []))
        if a.get("published"):
            d["pub"].append(a["published"])
with open(D / "osv_index.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["package_name", "n_advisories", "advisory_ids", "n_malware", "has_cve", "has_ghsa",
                "first_published", "malware_only"])
    for name, d in sorted(pk.items()):
        allids = d["ids"]
        w.writerow([name, len(allids), ";".join(sorted(allids)), len(d["mal"]),
                    int(any(x.startswith("CVE-") for x in d["aliases"] | allids)),
                    int(any(x.startswith("GHSA-") for x in d["aliases"] | allids)),
                    min(d["pub"]) if d["pub"] else "", int(len(allids) == 0 and len(d["mal"]) > 0)])
pos = sum(1 for d in pk.values() if d["ids"]); mal = sum(1 for d in pk.values() if not d["ids"])
print(f"packages with >=1 non-malware advisory: {pos}; malware-only: {mal}")
(D / "osv_snapshot.txt").write_text(datetime.datetime.fromtimestamp(zpath.stat().st_mtime).date().isoformat())
print("snapshot date (file date of the zip) saved to osv_snapshot.txt - verify it is when you downloaded it, and cite it")
