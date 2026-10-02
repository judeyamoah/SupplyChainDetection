"""Shared helpers for the v2 collection pipeline (paths, HTTP with retry, trimming)."""
import json, time, random
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter

BASE = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "v2" else Path(__file__).resolve().parent
D = BASE / "data_v2"
D.mkdir(parents=True, exist_ok=True)
SEED = 42
HEADERS = {"User-Agent": "ICAST-vuln-prediction-research/2.0", "Accept": "application/json"}

def session(pool=32):
    s = requests.Session()
    s.mount("https://", HTTPAdapter(pool_connections=pool, pool_maxsize=pool))
    s.headers.update(HEADERS)
    return s

def get_json(s, url, params=None, tries=6, timeout=60):
    """GET -> (status, json|None). Retries 429/5xx/network errors with backoff. 404 -> (404, None)."""
    for i in range(tries):
        try:
            r = s.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                return 404, None
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 2 ** i + random.random()))
                continue
            r.raise_for_status()
            return r.status_code, r.json()
        except (requests.RequestException, ValueError):
            time.sleep(min(60, 2 ** i + random.random()))
    return None, None

# Fields of the latest version that 04_feature_engineering.extract_features reads.
LATEST_KEYS = ["dependencies", "devDependencies", "optionalDependencies", "peerDependencies",
               "bundledDependencies", "scripts", "repository", "homepage", "bugs", "license",
               "author", "deprecated"]
TOP_KEYS = ["name", "description", "maintainers", "keywords", "repository", "homepage", "bugs",
            "license", "author", "dist-tags"]

def trim_packument(p):
    """Shrink a full packument to what feature extraction needs. readme is replaced by its length."""
    vers = p.get("versions") or {}
    latest = (p.get("dist-tags") or {}).get("latest")
    keep = latest if latest in vers else (list(vers)[-1] if vers else None)
    tv = {v: {} for v in vers}
    if keep is not None:
        full = vers[keep] or {}
        tv[keep] = {k: full[k] for k in LATEST_KEYS if k in full}
    t = p.get("time") or {}
    out = {k: p[k] for k in TOP_KEYS if k in p}
    out["versions"] = tv
    out["time"] = {"created": t.get("created"), "modified": t.get("modified")}
    # first-publish time of every version is kept (small) for optional temporal analyses
    out["_version_times"] = {v: t.get(v) for v in vers if t.get(v)}
    out["_readme_length"] = len(p.get("readme") or "")
    return out

def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
