"""
Stage 2 - Full list of npm package names (the sampling frame).
Default: page replicate.npmjs.com/_all_docs (ids only, ~3-4M names, a few minutes).
Alternative: --names-file FILE (JSON list or one name per line), e.g. from `npm i all-the-package-names`.
Output: data_v2/names.txt
"""
import argparse, json
from v2_common import D, session, get_json

ap = argparse.ArgumentParser(); ap.add_argument("--names-file"); a = ap.parse_args()
names = []
if a.names_file:
    t = open(a.names_file, encoding="utf-8").read().strip()
    names = json.loads(t) if t.startswith("[") else t.splitlines()
else:
    s = session(); start = None
    while True:
        params = {"limit": 5001}
        if start is not None:
            params["startkey"] = json.dumps(start)
        st, j = get_json(s, "https://replicate.npmjs.com/_all_docs", params)
        if not j:
            raise SystemExit(f"_all_docs failed at {start!r}; use --names-file instead")
        rows = [r["id"] for r in j["rows"] if not r["id"].startswith("_design/")]
        if start is not None and rows and rows[0] == start:
            rows = rows[1:]
        names += rows
        print(len(names), end="\r")
        if len(j["rows"]) < 5001:
            break
        start = j["rows"][-1]["id"]
names = sorted(set(n.strip() for n in names if n.strip()))
(D / "names.txt").write_text("\n".join(names), encoding="utf-8")
print(f"{len(names)} names written")
