"""
Stage 4 - Fetch trimmed metadata and downloads for every package in the frame. Resumable (re-run to continue).
Outputs: data_v2/metadata.jsonl   (trimmed packuments)
         data_v2/downloads.jsonl  ({package_name, weekly_downloads, monthly_downloads})
         data_v2/failed.txt       (unpublished / no versions / repeated errors)
"""

import argparse, json, shutil, threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import pandas as pd
from tqdm import tqdm
from v2_common import D, session, get_json, trim_packument, read_jsonl

WORKERS = 16
ap = argparse.ArgumentParser()
ap.add_argument(
    "--retry-failed",
    action="store_true",
    help="retry every name in failed.txt (old copy kept as failed_prev.txt)",
)
args = ap.parse_args()
frame = pd.read_csv(
    D / "sample_frame.csv", dtype={"package_name": str}, keep_default_na=False
)
names = [str(n) for n in frame.package_name]
lock = threading.Lock()
S = session()

# ---- metadata ----
done_meta = {r["name"] for r in read_jsonl(D / "metadata.jsonl")}
if args.retry_failed and (D / "failed.txt").exists():
    shutil.move(D / "failed.txt", D / "failed_prev.txt")
failed = (
    {l.split("\t")[0] for l in (D / "failed.txt").read_text().splitlines()}
    if (D / "failed.txt").exists()
    else set()
)
todo = [n for n in names if n not in done_meta and n not in failed]
fm = open(D / "metadata.jsonl", "a", encoding="utf-8")
ff = open(D / "failed.txt", "a", encoding="utf-8")


def meta(n):
    try:
        _meta(n)
    except Exception as e:  # never let one bad name kill the run
        with lock:
            ff.write(f"{n}\terror_after_retries\n")


def _meta(n):
    st, j = get_json(S, "https://registry.npmjs.org/" + quote(n, safe="@"))
    with lock:
        if j and j.get("versions") and (j.get("time") or {}).get("created"):
            fm.write(json.dumps(trim_packument(j)) + "\n")
        else:
            reason = (
                "404_unpublished"
                if st == 404
                else (
                    "error_after_retries"
                    if st is None
                    else (
                        "no_versions"
                        if not (j or {}).get("versions")
                        else "no_created_time"
                    )
                )
            )
            ff.write(f"{n}\t{reason}\n")
        if len(done_meta) % 200 == 0:
            fm.flush()
            ff.flush()
        done_meta.add(n)


with ThreadPoolExecutor(WORKERS) as ex:
    list(tqdm(ex.map(meta, todo), total=len(todo), desc="metadata"))
fm.close()
ff.close()

# ---- downloads ----
have = {r["package_name"] for r in read_jsonl(D / "downloads.jsonl")}
ok = {r["name"] for r in read_jsonl(D / "metadata.jsonl")}
need = [n for n in names if n in ok and n not in have]
unscoped = [n for n in need if not n.startswith("@")]
scoped = [n for n in need if n.startswith("@")]
fd = open(D / "downloads.jsonl", "a", encoding="utf-8")


def bulk(period, group):
    if len(group) == 1:
        st, j = get_json(
            S,
            f"https://api.npmjs.org/downloads/point/{period}/{quote(group[0], safe='@/')}",
        )
        return (
            {group[0]: (j or {}).get("downloads")}
            if st is not None
            else {group[0]: None}
        )
    st, j = get_json(
        S, f"https://api.npmjs.org/downloads/point/{period}/" + ",".join(group)
    )
    if j is None:
        return {n: None for n in group}
    return {
        n: (j.get(n) or {}).get("downloads", 0) if j.get(n) is not None else 0
        for n in group
    }


def dl_group(group):
    w, m = bulk("last-week", group), bulk("last-month", group)
    with lock:
        for n in group:
            if w[n] is None or m[n] is None:
                continue  # left for the next run
            fd.write(
                json.dumps(
                    {
                        "package_name": n,
                        "weekly_downloads": w[n],
                        "monthly_downloads": m[n],
                    }
                )
                + "\n"
            )
        fd.flush()


def dl_one(n):
    dl_group([n])


groups = [unscoped[i : i + 128] for i in range(0, len(unscoped), 128)]
with ThreadPoolExecutor(4) as ex:  # keep modest: api.npmjs.org rate-limits
    list(tqdm(ex.map(dl_group, groups), total=len(groups), desc="downloads (bulk)"))
with ThreadPoolExecutor(4) as ex:
    list(tqdm(ex.map(dl_one, scoped), total=len(scoped), desc="downloads (scoped)"))
fd.close()
print(
    "failed.txt = skipped packages, reason in 2nd column; only error_after_retries is worth retrying: python v2_04_fetch.py --retry-failed"
)
print(
    "Re-run this script until 'metadata' and both 'downloads' bars have nothing left to do."
)
