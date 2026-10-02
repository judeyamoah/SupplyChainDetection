"""Checks that the environment can run scripts 01 to 06 and prints the versions to cite in the paper."""

import importlib, platform, sys

print(
    f"Python {platform.python_version()} on {platform.platform()} ({platform.machine()})"
)
missing = []
for name in [
    "requests",
    "tqdm",
    "numpy",
    "pandas",
    "sklearn",
    "imblearn",
    "xgboost",
    "matplotlib",
    "joblib",
]:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:<12} {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        missing.append(name)
        print(f"  {name:<12} FAILED: {exc}")
        if "libomp" in str(exc).lower() or "openmp" in str(exc).lower():
            print(
                "      -> XGBoost needs the OpenMP runtime on macOS. Fix Homebrew, then run: brew install libomp"
            )
if missing:
    sys.exit(f"Missing or broken: {', '.join(missing)}")

# Tiny end-to-end test: SMOTE inside a pipeline, then an XGBoost fit (same building blocks as 05_train_models.py).
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 6))
y = (rng.random(300) < 0.15).astype(int)
X[y == 1] += 1.5
for est in (
    LogisticRegression(max_iter=500),
    XGBClassifier(n_estimators=20, eval_metric="logloss"),
):
    Pipeline(
        [("scale", StandardScaler()), ("smote", SMOTE(random_state=42)), ("clf", est)]
    ).fit(X, y)
print("OK: SMOTE + LogisticRegression + XGBoost run correctly.")
