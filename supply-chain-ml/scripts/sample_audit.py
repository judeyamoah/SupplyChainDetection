#!/usr/bin/env python3
"""
sampling_audit.py  --  checks the ICAST 2026 npm vulnerability-prediction results against the way the classes were collected.

Run from the folder that contains npm_features.csv (or pass --csv). Writes CSV tables next to the script.

  1. reproduction  : hold-out metrics for all 80 features (compare with results/model_performance.csv)
  2. ablation      : drop / keep one feature group at a time (LR and RF)
  3. single feature: ROC-AUC of individual features on their own
  4. natural test  : train on the OSV-seeded positives (seed_label=1) + benign; test on the positives found in
                     the background sample (osv_label=1, seed_label=0) + held-out benign
  5. temporal      : train on packages created before CUTOFF, test on the rest (creation date recovered from
                     package_age_days and the collection date, see COLLECTION_DATE)
  6. class profile : per-class summary of the features that differ most (Table II of the paper)

Uses imbalanced-learn SMOTE and XGBoost when installed (that is what 05_train_models.py used); otherwise falls back
to a small SMOTE re-implementation and sklearn's HistGradientBoosting (marked GBM*, NOT XGBoost).
The scaler and SMOTE are fitted on the training portion only.
"""

import argparse, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="npm_features.csv")
args = ap.parse_args()
SEED = 42
COLLECTION_DATE = "2026-07-02"  # date 04_feature_engineering.py ran (package_age_days = that date - created)
CUTOFF = "2021-01-01"
DROP = [
    "package_name",
    "label",
    "vulnerability_count",
    "has_cve_alias",
    "has_ghsa_alias",
    "severity_count",
    "seed_label",
    "osv_label",
]

try:
    from imblearn.over_sampling import SMOTE as _SMOTE

    def smote(X, y):
        return _SMOTE(random_state=SEED).fit_resample(X, y)

    SMOTE_NAME = "imblearn"
except Exception:
    SMOTE_NAME = "fallback"

    def smote(X, y, k=5):
        rng = np.random.default_rng(SEED)
        Xm = X[y == 1]
        n_new = int((y == 0).sum() - (y == 1).sum())
        idx = (
            NearestNeighbors(n_neighbors=min(k + 1, len(Xm)))
            .fit(Xm)
            .kneighbors(Xm, return_distance=False)[:, 1:]
        )
        i = rng.integers(0, len(Xm), n_new)
        j = idx[i, rng.integers(0, idx.shape[1], n_new)]
        return np.vstack(
            [X, Xm[i] + rng.random((n_new, 1)) * (Xm[j] - Xm[i])]
        ), np.concatenate([y, np.ones(n_new, int)])


try:
    from xgboost import XGBClassifier

    BOOST = (
        "XGBoost",
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
    )
except Exception:
    BOOST = (
        "GBM*",
        HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.05, max_iter=300, random_state=SEED
        ),
    )


def models():  # same settings as 05_train_models.py
    return {
        "LR": (
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=SEED
            ),
            True,
        ),
        "SVM": (
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=SEED,
            ),
            True,
        ),
        "RF": (
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
            ),
            False,
        ),
        "MLP": (
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED),
            True,
        ),
        BOOST[0]: (BOOST[1], False),
    }


def fit_predict(est, scale, Xtr, ytr, Xte):
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Xs, ys = smote(Xtr, ytr)
    m = est.fit(Xs, ys)
    return m.predict(Xte), m.predict_proba(Xte)[:, 1]


def score(yt, p, s):
    return dict(
        prec=precision_score(yt, p, zero_division=0),
        rec=recall_score(yt, p),
        f1=f1_score(yt, p),
        auc=roc_auc_score(yt, s),
        mcc=matthews_corrcoef(yt, p),
    )


df = pd.read_csv(args.csv)
X = (
    df.drop(columns=[c for c in DROP if c in df.columns])
    .select_dtypes(include=[np.number])
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)
y = df.label.astype(int).values
print(
    f"{X.shape[1]} features, {len(y)} rows, {y.sum()} vulnerable | SMOTE: {SMOTE_NAME} | boosting: {BOOST[0]}"
)

# 6) class profile ----------------------------------------------------------------------------------
b, v = df[df.label == 0], df[df.label == 1]
prof = {
    "name starts with digit or '-'": (
        b.package_name.str[0].isin(list("-0123456789")).mean(),
        v.package_name.str[0].isin(list("-0123456789")).mean(),
    ),
    "scoped package (@scope/name)": (
        b.is_scoped_package.mean(),
        v.is_scoped_package.mean(),
    ),
    "contains a digit": (b.contains_digit.mean(), v.contains_digit.mean()),
    "has repository": (b.has_repository.mean(), v.has_repository.mean()),
    "median monthly downloads": (
        b.monthly_downloads.median(),
        v.monthly_downloads.median(),
    ),
    "median version count": (b.version_count.median(), v.version_count.median()),
    "median age (days)": (b.package_age_days.median(), v.package_age_days.median()),
    "no CVE/GHSA alias": (
        np.nan,
        ((v.has_cve_alias == 0) & (v.has_ghsa_alias == 0)).mean(),
    ),
}
pd.DataFrame(prof, index=["benign", "vulnerable"]).T.round(3).to_csv(
    "class_profile.csv"
)
print(pd.DataFrame(prof, index=["benign", "vulnerable"]).T.round(3))

# 1) reproduction -----------------------------------------------------------------------------------
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
rows = [
    dict(model=n, **score(yte, *fit_predict(e, s, Xtr.values, ytr, Xte.values)))
    for n, (e, s) in models().items()
]
pd.DataFrame(rows).round(3).to_csv("reproduction.csv", index=False)
print("\nreproduction\n", pd.DataFrame(rows).round(3).to_string(index=False))

# 2) ablation ----------------------------------------------------------------------------------------
G = {
    "naming": [
        "contains_digit",
        "contains_dash",
        "contains_dot",
        "contains_underscore",
        "name_length",
        "name_entropy_score",
        "is_scoped_package",
        "scope_length",
        "package_name_token_count",
        "short_name_flag",
    ],
    "downloads": [c for c in X.columns if "download" in c],
    "repository/transparency": [
        "has_repository",
        "has_github_repository",
        "has_homepage",
        "has_bugs_url",
        "has_license",
        "has_author",
        "has_keywords",
        "keyword_count",
        "description_length",
        "readme_length",
        "description_missing_flag",
        "readme_missing_flag",
        "transparency_score",
        "metadata_risk_score",
    ],
    "version/lifecycle": [
        "version_count",
        "latest_version_major",
        "latest_version_minor",
        "latest_version_patch",
        "version_major_jump_flag",
        "package_age_days",
        "days_since_last_update",
        "update_frequency",
        "staleness_ratio",
        "recent_update_flag",
        "very_new_package_flag",
        "stale_abandoned_flag",
        "dependency_growth_proxy",
        "dependency_risk_score",
    ],
    "maintainers": [
        "maintainer_count",
        "no_maintainer_flag",
        "single_maintainer_flag",
        "low_maintainer_flag",
        "maintainer_per_version_ratio",
        "maintainer_to_dependency_ratio",
        "maintainer_density",
    ],
}
G = {k: [c for c in cs if c in X.columns] for k, cs in G.items()}
allc = list(X.columns)
rows = []


def run(label, cols):
    for n in ("LR", "RF"):
        e, s = models()[n]
        rows.append(
            dict(
                setting=label,
                model=n,
                n_feat=len(cols),
                **score(
                    yte, *fit_predict(e, s, Xtr[cols].values, ytr, Xte[cols].values)
                ),
            )
        )


run("all features", allc)
for g, cs in G.items():
    run(f"drop {g}", [c for c in allc if c not in cs])
run("drop naming+downloads", [c for c in allc if c not in G["naming"] + G["downloads"]])
run(
    "drop naming+downloads+repo",
    [
        c
        for c in allc
        if c not in G["naming"] + G["downloads"] + G["repository/transparency"]
    ],
)
for g, cs in G.items():
    run(f"ONLY {g}", cs)
pd.DataFrame(rows).round(3).to_csv("ablation.csv", index=False)

# 3) single-feature AUC --------------------------------------------------------------------------------
sf = {
    c: max(roc_auc_score(y, X[c]), 1 - roc_auc_score(y, X[c]))
    for c in X.columns
    if X[c].nunique() > 1
}
pd.Series(sf).sort_values(ascending=False).round(3).to_csv(
    "single_feature_auc.csv", header=["auc"]
)
print(
    "\ntop single-feature AUC\n",
    pd.Series(sf).sort_values(ascending=False).head(6).round(3),
)

# 4) natural-sample test -------------------------------------------------------------------------------
nat = ((df.osv_label == 1) & (df.seed_label == 0)).values
seed = (df.seed_label == 1).values
ben = np.where(df.label.values == 0)[0]
rng = np.random.default_rng(SEED)
rng.shuffle(ben)
nb = len(ben) // 5
tr = np.concatenate([np.where(seed)[0], ben[nb:]])
te = np.concatenate([np.where(nat)[0], ben[:nb]])
rows = []
for n, (e, s) in models().items():
    p, sc_ = fit_predict(e, s, X.values[tr], y[tr], X.values[te])
    rows.append(
        dict(
            model=n,
            positives=int(y[te].sum()),
            flagged=int(p[y[te] == 1].sum()),
            benign_test=int((y[te] == 0).sum()),
            auc=roc_auc_score(y[te], sc_),
        )
    )
pd.DataFrame(rows).round(3).to_csv("natural_sample_test.csv", index=False)
print("\nnatural-sample test\n", pd.DataFrame(rows).round(3).to_string(index=False))

# 5) temporal split -------------------------------------------------------------------------------------
created = (np.datetime64(COLLECTION_DATE) - np.datetime64("1970-01-01")).astype(
    int
) - df.package_age_days.values
trm = created < (np.datetime64(CUTOFF) - np.datetime64("1970-01-01")).astype(int)
tem = ~trm
print(
    f"\ntemporal: train {trm.sum()} ({y[trm].sum()} vuln) / test {tem.sum()} ({y[tem].sum()} vuln)"
)
rows = [
    dict(
        model=n,
        **score(y[tem], *fit_predict(e, s, X.values[trm], y[trm], X.values[tem])),
    )
    for n, (e, s) in models().items()
]
pd.DataFrame(rows).round(3).to_csv("temporal_split.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
