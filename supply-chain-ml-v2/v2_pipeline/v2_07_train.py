"""
Stage 7 - Train/evaluate on the v2 matched set. Reuses get_models() from 05_train_models.py.
Adds PR-AUC (average precision), MCC, precision@k, bootstrap 95% CIs, and a temporal hold-out.
Defaults: SMOTE OFF (matched set is only ~1:2; class_weight='balanced' already compensates) - use --smote to compare.
  --drop-downloads : remove download features (the matching variable) to test signal beyond popularity
  --models "Random Forest,XGBoost"
Outputs: results_v2/*.csv
"""

import argparse, importlib.util, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
here = Path(__file__).resolve().parent
BASE = here.parent if (here.parent / "data_v2").exists() else here
D = BASE / "data_v2"
SEED = 42

ap = argparse.ArgumentParser()
ap.add_argument("--tm", default=None, help="path to 05_train_models.py")
ap.add_argument("--suffix", default="", help="use features_v2<suffix>.csv, e.g. _fine")
ap.add_argument("--smote", action="store_true")
ap.add_argument("--drop-downloads", action="store_true")
ap.add_argument(
    "--drop-names",
    action="store_true",
    help="remove name-derived features (length, tokens, entropy, dash/dot/digit, scope)",
)
ap.add_argument(
    "--drop-lifecycle",
    action="store_true",
    help="remove features sensitive to WHEN they are measured (version counts, update recency, release cadence); features are measured today, after disclosure",
)
ap.add_argument("--models", default=None)
ap.add_argument(
    "--cutoff",
    default="2021-01-01",
    help="temporal hold-out: train before this date, test on/after (vulnerable: first advisory date; benign: creation date)",
)
a = ap.parse_args()
cands = (
    [a.tm]
    if a.tm
    else [
        here / "05_train_models.py",
        BASE / "../scripts" / "05_train_models.py",
        BASE / "05_train_models.py",
    ]
)
tm_path = next((c for c in cands if c and Path(c).exists()), None)
if not tm_path:
    sys.exit("Cannot find 05_train_models.py; pass --tm")
spec = importlib.util.spec_from_file_location("tm05", tm_path)
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

df = pd.read_csv(D / f"features_v2{a.suffix}.csv")
meta_cols = [
    c
    for c in ["role", "dl_bin", "dl_tier", "age_bin", "created", "first_published"]
    if c in df.columns
]
TIER = "dl_tier" if "dl_tier" in df.columns else "dl_bin"
X, y = tm.prepare_features(df.drop(columns=meta_cols))
NAME_COLS = [
    "is_scoped_package",
    "scope_length",
    "package_name_token_count",
    "contains_dash",
    "contains_dot",
    "contains_digit",
    "contains_underscore",
    "name_length",
    "name_entropy_score",
    "short_name_flag",
]
LIFECYCLE_COLS = [
    "version_count",
    "latest_version_major",
    "latest_version_minor",
    "latest_version_patch",
    "version_major_jump_flag",
    "days_since_last_update",
    "update_frequency",
    "staleness_ratio",
    "recent_update_flag",
    "stale_abandoned_flag",
    "maintainer_per_version_ratio",
    "dependency_density",
    "total_dependency_density",
    "dependency_growth_proxy",
    "rapid_release_flag",
    "version_burst_flag",
]
tags = []
if a.drop_downloads:
    X = X.drop(columns=[c for c in X.columns if "download" in c])
    tags.append("nodownloads")
if a.drop_names:
    X = X.drop(columns=[c for c in NAME_COLS if c in X.columns])
    tags.append("nonames")
if a.drop_lifecycle:
    X = X.drop(columns=[c for c in LIFECYCLE_COLS if c in X.columns])
    tags.append("nolifecycle")
if a.smote:
    tags.append("smote")
R = BASE / "results_v2" / (("_".join(tags) or "baseline") + a.suffix)
R.mkdir(parents=True, exist_ok=True)
print("results ->", R)
print(f"X={X.shape}, positives={int(y.sum())}, prevalence={y.mean():.3f}")


def models():
    ms = tm.get_models()
    if not a.smote:
        ms = {
            k: ImbPipeline([s for s in v.steps if s[0] != "smote"])
            for k, v in ms.items()
        }
    if a.models:
        keep = [m.strip() for m in a.models.split(",")]
        ms = {k: v for k, v in ms.items() if k in keep}
    return ms


def p_at_k(y_true, score, k):
    k = min(k, len(y_true))
    idx = np.argsort(-score)[:k]
    return float(np.mean(np.asarray(y_true)[idx]))


def metrics(yt, score, thr=0.5):
    yt = np.asarray(yt)
    pred = (score >= thr).astype(int)
    return {
        "accuracy": accuracy_score(yt, pred),
        "precision": precision_score(yt, pred, zero_division=0),
        "recall": recall_score(yt, pred),
        "f1": f1_score(yt, pred),
        "roc_auc": roc_auc_score(yt, score),
        "pr_auc": average_precision_score(yt, score),
        "mcc": matthews_corrcoef(yt, pred),
        "p_at_50": p_at_k(yt, score, 50),
        "p_at_100": p_at_k(yt, score, 100),
        "p_at_10pct": p_at_k(yt, score, max(1, int(0.1 * len(yt)))),
    }


def boot_ci(yt, score, n=1000):
    rng = np.random.default_rng(SEED)
    yt = np.asarray(yt)
    out = {"roc_auc": [], "pr_auc": []}
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].min() == yt[i].max():
            continue
        out["roc_auc"].append(roc_auc_score(yt[i], score[i]))
        out["pr_auc"].append(average_precision_score(yt[i], score[i]))
    return {
        f"{k}_ci": f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
        for k, v in out.items()
    }


# 1) 10-fold stratified CV, out-of-fold predictions (one pooled metric set per model)
cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
rows = []
oof_store = {}
for name, m in models().items():
    oof = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
    oof_store[name] = oof
    rows.append({"model": name, **metrics(y, oof), **boot_ci(y, oof)})
    print(name, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)})
pd.DataFrame(rows).to_csv(R / "cv_oof_metrics.csv", index=False)

# 1b) Leakage / shortcut audit (reviewer: "explicit leakage audit")
#  (i) per-download-tier AUC from the out-of-fold predictions: if popularity were the shortcut, AUC inside a tier would collapse
rows_t = []
for name, oof in oof_store.items():
    for b, idx in df.groupby(TIER).groups.items():
        ii = df.index.get_indexer(idx)
        yy = y.values[ii]
        if len(ii) >= 20 and yy.min() != yy.max():
            rows_t.append(
                {
                    "model": name,
                    "dl_tier": b,
                    "n": len(ii),
                    "vulnerable": int(yy.sum()),
                    "benign_from_popular_list": int(
                        ((df["role"].values[ii] == "supp")).sum()
                    ),
                    "roc_auc": roc_auc_score(yy, oof[ii]),
                    "pr_auc": average_precision_score(yy, oof[ii]),
                }
            )
pd.DataFrame(rows_t).to_csv(R / "per_download_tier_auc.csv", index=False)
print("per-tier AUC written (results_v2/per_download_tier_auc.csv)")
#  (ii) uniform-only analysis: drop controls that came from the popular-list supplement, keep only download tiers where
#       uniform controls exist in reasonable numbers (tiers 0-3). This subset has NO supplement involvement.
keep = (df["role"] != "supp") & (df[TIER] <= 3)
if keep.sum() > 100 and y[keep].nunique() == 2:
    Xu, yu = X[keep.values], y[keep.values]
    rows_u = []
    for name, m in models().items():
        o = cross_val_predict(
            m,
            Xu,
            yu,
            cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
            method="predict_proba",
        )[:, 1]
        rows_u.append(
            {"model": name, "n": len(yu), "vulnerable": int(yu.sum()), **metrics(yu, o)}
        )
    pd.DataFrame(rows_u).to_csv(R / "uniform_only_cv_metrics.csv", index=False)
    print("uniform-only subset (no popular-list controls, tiers 0-3):")
    print(
        pd.DataFrame(rows_u)[["model", "n", "vulnerable", "roc_auc", "pr_auc", "mcc"]]
        .round(3)
        .to_string(index=False)
    )
#  (iii) columns removed from the feature set and why (goes into the paper as the leakage-audit table)
pd.DataFrame(
    [
        ("package_name", "identifier, not a feature"),
        ("label", "target"),
        (
            "vulnerability_count",
            "derived directly from the OSV record that defines the label",
        ),
        (
            "has_cve_alias",
            "derived from OSV aliases; exists only for labelled-vulnerable packages",
        ),
        (
            "has_ghsa_alias",
            "derived from OSV aliases; exists only for labelled-vulnerable packages",
        ),
        (
            "severity_count",
            "derived from OSV severity field; exists only for labelled-vulnerable packages",
        ),
        ("seed_label", "copy of the label"),
        ("osv_label", "copy of the label"),
        (
            "role, dl_bin, dl_tier, age_bin, created, first_published",
            "sampling/matching bookkeeping; used only to split data and audit, never as inputs",
        ),
    ],
    columns=["removed_column", "reason"],
).to_csv(R / "leakage_audit_removed_columns.csv", index=False)

# 2) temporal hold-out (reviewer's request): train on < cutoff, test on >= cutoff.
#    Event date: vulnerable package = date of its FIRST advisory (first_published); benign package = creation date
#    (it has no advisory). If a positive has no advisory date, its creation date is used.
created = pd.to_datetime(df["created"], utc=True, errors="coerce")
fp = pd.to_datetime(df["first_published"], utc=True, errors="coerce")
event = fp.where((df["label"] == 1) & fp.notna(), created)
cut = pd.Timestamp(a.cutoff, tz="UTC")
tr = (event < cut).values
te = ~tr
cnt = pd.DataFrame(
    [
        {
            "split": n,
            "n": int(m.sum()),
            "vulnerable": int(y[m].sum()),
            "benign": int((1 - y[m]).sum()),
        }
        for n, m in [("train (< %s)" % a.cutoff, tr), ("test (>= %s)" % a.cutoff, te)]
    ]
)
cnt.to_csv(R / "temporal_split_counts.csv", index=False)
print(cnt.to_string(index=False))
rows = []
if min(y[tr].sum(), y[te].sum()) < 30:
    print(
        "WARNING: fewer than 30 vulnerable packages on one side of the split; results will be unstable - report counts, not just metrics."
    )
if y[tr].nunique() == 2 and y[te].nunique() == 2:
    for name, m in models().items():
        m.fit(X[tr], y[tr])
        s_ = m.predict_proba(X[te])[:, 1]
        rows.append(
            {
                "model": name,
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "test_vulnerable": int(y[te].sum()),
                **metrics(y[te], s_),
                **boot_ci(y[te], s_),
            }
        )
    pd.DataFrame(rows).to_csv(R / "temporal_holdout_metrics.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
else:
    print("temporal split skipped: one side lacks both classes")

# 3) single-feature AUC (shortcut check)
sf = sorted(
    (
        (c, max(roc_auc_score(y, X[c]), 1 - roc_auc_score(y, X[c])))
        for c in X.columns
        if X[c].nunique() > 1
    ),
    key=lambda z: -z[1],
)
pd.DataFrame(sf, columns=["feature", "auc_sym"]).to_csv(
    R / "single_feature_auc.csv", index=False
)
print("top single-feature AUCs:", [(c, round(v, 3)) for c, v in sf[:8]])
