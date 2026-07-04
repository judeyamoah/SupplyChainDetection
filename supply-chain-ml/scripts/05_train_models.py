"""
05_train_models.py

Purpose:
Train and evaluate machine learning models for detecting vulnerable or
malicious npm packages using engineered metadata, dependency, temporal,
download, and script-risk features.

Input:
data/processed/npm_features.csv

Outputs:
results/model_performance.csv
results/confusion_matrix_values.csv
figures/confusion_matrix_<model>.png
figures/roc_curve_all_models.png
figures/feature_importance_random_forest.png
models/<model>.joblib
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_FILE = BASE_DIR / "data" / "processed" / "npm_features.csv"

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

PERFORMANCE_FILE = RESULTS_DIR / "model_performance.csv"
CONFUSION_VALUES_FILE = RESULTS_DIR / "confusion_matrix_values.csv"
FEATURE_IMPORTANCE_FILE = RESULTS_DIR / "feature_importance_random_forest.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10


# -----------------------------
# Utility Functions
# -----------------------------

def ensure_directories():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_FILE}")

    df = pd.read_csv(FEATURES_FILE)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    return df


def prepare_features(df):
    drop_columns = [
        "package_name",
        "label",
        "vulnerability_count",
        "has_cve_alias",
        "has_ghsa_alias",
        "severity_count",
        "seed_label",
        "osv_label"
    ]

    X = df.drop(columns=[col for col in drop_columns if col in df.columns])
    y = df["label"].astype(int)

    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y


def get_models():
    models = {
        "Logistic Regression": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Support Vector Machine": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "Neural Network": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=500,
                random_state=RANDOM_STATE
            ))
        ])
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

    return models


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    

    start_predict = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_predict

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "training_time_seconds": train_time,
        "prediction_time_seconds": predict_time
    }

    return metrics, y_pred, y_score


def plot_confusion_matrix(name, y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix)

    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Vulnerable"])
    ax.set_yticklabels(["Benign", "Vulnerable"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center")

    plt.tight_layout()

    filename = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def plot_roc_curves(roc_data):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, data in roc_data.items():
        ax.plot(
            data["fpr"],
            data["tpr"],
            label=f"{name} (AUC = {data['auc']:.3f})"
        )

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

    ax.set_title("ROC Curves for Machine Learning Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_all_models.png", dpi=300)
    plt.close()


def save_feature_importance(model, feature_names):
    try:
        classifier = model.named_steps["classifier"]

        if not hasattr(classifier, "feature_importances_"):
            return

        importances = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

        top_features = importance_df.head(20)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(top_features["feature"][::-1], top_features["importance"][::-1])
        ax.set_title("Top 20 Feature Importances - Random Forest")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance_random_forest.png", dpi=300)
        plt.close()

    except Exception as error:
        print(f"Feature importance generation failed: {error}")


# -----------------------------
# Main
# -----------------------------

def main():
    ensure_directories()

    print("=" * 60)
    print("MODEL TRAINING STARTED")
    print("=" * 60)

    df = load_dataset()
    X, y = prepare_features(df)

    print(f"Dataset shape: {df.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print("Class distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    models = get_models()

    performance_records = []
    confusion_records = []
    roc_data = {}

    for name, model in models.items():
        print("-" * 60)
        print(f"Training model: {name}")

        metrics, y_pred, y_score = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        performance_records.append(metrics)

        confusion_records.append({
            "model": name,
            "TP": metrics["true_positive"],
            "TN": metrics["true_negative"],
            "FP": metrics["false_positive"],
            "FN": metrics["false_negative"]
        })

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_data[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc(fpr, tpr)
        }

        plot_confusion_matrix(name, y_test, y_pred)

        model_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODELS_DIR / model_filename)

        if name == "Random Forest":
            save_feature_importance(model, X.columns)

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"MCC:       {metrics['mcc']:.4f}")
        
        cv = StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

        metrics["cv_f1_mean"] = cv_scores.mean()
        metrics["cv_f1_std"] = cv_scores.std()

        print(f"10-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    plot_roc_curves(roc_data)

    performance_df = pd.DataFrame(performance_records)
    confusion_df = pd.DataFrame(confusion_records)

    performance_df.to_csv(PERFORMANCE_FILE, index=False)
    confusion_df.to_csv(CONFUSION_VALUES_FILE, index=False)

    summary = {
        "dataset_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "vulnerable_count": int(y.sum()),
        "benign_count": int(len(y) - y.sum()),
        "test_size": TEST_SIZE,
        "cross_validation_folds": CV_FOLDS,
        "models_trained": list(models.keys())
    }

    with open(RESULTS_DIR / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("=" * 60)
    print("MODEL TRAINING COMPLETE")
    print(f"Performance results: {PERFORMANCE_FILE}")
    print(f"Confusion values: {CONFUSION_VALUES_FILE}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()