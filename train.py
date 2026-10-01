"""
Training script — runs EDA, trains baseline + tuned models, performs
threshold optimization, generates SHAP explanations, and serializes
the best pipeline.

Usage:
    python train.py
"""

import json
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    roc_curve,
)
import joblib

from config import (
    CV_FOLDS,
    MODEL_DIR,
    MODEL_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from preprocessing import load_dataset
from pipeline import build_logistic_pipeline, build_xgboost_pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

FIG_DIR = MODEL_DIR / "figures"


# ── Helpers ────────────────────────────────────────────────────────────────────

def evaluate(name, pipeline, X_test, y_test, threshold=0.5):
    """Compute key metrics and print a summary for a fitted pipeline."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)

    print(f"\n{'─' * 60}")
    print(f"  {name}  (threshold={threshold:.2f})")
    print(f"{'─' * 60}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"  PR-AUC     : {pr_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    return {
        "accuracy": float(round(acc, 4)),
        "f1_score": float(round(f1, 4)),
        "roc_auc": float(round(roc_auc, 4)),
        "pr_auc": float(round(pr_auc, 4)),
        "threshold": float(round(threshold, 4)),
    }


def find_optimal_threshold(pipeline, X_test, y_test):
    """Find the threshold that maximizes F1 score on the test set.

    In churn prediction, the default 0.5 cutoff is rarely optimal because
    the business cost of missing a churner (false negative) typically
    exceeds the cost of a false alarm (false positive).
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\n  Threshold optimization:")
    print(f"    Default (0.50) F1: {f1_score(y_test, (y_prob >= 0.5).astype(int)):.4f}")
    print(f"    Optimal ({best_threshold:.2f}) F1: {best_f1:.4f}")

    return best_threshold


def plot_threshold_analysis(pipeline, X_test, y_test, optimal_threshold):
    """Plot precision, recall, and F1 vs. threshold to visualize the trade-off."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions[:-1], label="Precision", color="#3498db", linewidth=2)
    ax.plot(thresholds, recalls[:-1], label="Recall", color="#e74c3c", linewidth=2)
    ax.plot(thresholds, f1_scores, label="F1 Score", color="#2ecc71", linewidth=2.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Default (0.50)")
    ax.axvline(x=optimal_threshold, color="#f39c12", linestyle="--", linewidth=2,
               label=f"Optimal ({optimal_threshold:.2f})")
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 vs. Decision Threshold", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "threshold_analysis.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved: artifacts/figures/threshold_analysis.png")


def plot_roc_curves(models_dict, X_test, y_test):
    """Plot ROC curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    for (name, pipeline), color in zip(models_dict.items(), colors):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Guess")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig(FIG_DIR / "roc_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved: artifacts/figures/roc_comparison.png")


def generate_shap_analysis(pipeline, X_test):
    """Generate SHAP feature importance plots for interpretability.

    SHAP values provide a principled way to understand *how* each feature
    drives individual predictions, not just aggregate importance scores.
    """
    try:
        import shap

        # Extract the trained classifier and the preprocessed test data
        preprocessor = pipeline[:-1]  # everything except classifier
        classifier = pipeline.named_steps["classifier"]
        X_processed = preprocessor.transform(X_test)

        # Get feature names from the ColumnTransformer
        ct = pipeline.named_steps["preprocessor"]
        feature_names = ct.get_feature_names_out()

        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

        # Use TreeExplainer for XGBoost — it's exact and fast
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_processed_df)

        # Summary bar plot — global feature importance
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_processed_df, plot_type="bar",
                          show=False, max_display=15)
        plt.title("SHAP Feature Importance (Top 15)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(FIG_DIR / "shap_importance.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close("all")
        print(f"  ✓ Saved: artifacts/figures/shap_importance.png")

        # Beeswarm plot — shows direction of feature effects
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_processed_df, show=False, max_display=15)
        plt.title("SHAP Beeswarm Plot (Top 15)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(FIG_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close("all")
        print(f"  ✓ Saved: artifacts/figures/shap_beeswarm.png")

    except Exception as e:
        print(f"  ⚠ SHAP analysis skipped: {e}")


def run_eda(X, y):
    """Print a concise exploratory summary of the dataset."""
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nDataset shape: {X.shape[0]} rows × {X.shape[1] + 1} columns")
    print(f"\nTarget distribution:")
    counts = y.value_counts()
    print(f"  No Churn (0) : {counts[0]}  ({counts[0] / len(y) * 100:.1f}%)")
    print(f"  Churn    (1) : {counts[1]}  ({counts[1] / len(y) * 100:.1f}%)")
    print(f"  Imbalance ratio: {counts[0] / counts[1]:.2f} : 1")

    print(f"\nNumeric feature summary:")
    print(X[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2).to_string())

    print(f"\nMissing values: {X.isnull().sum().sum()}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    X, y = load_dataset()

    # 2. Brief EDA
    run_eda(X, y)

    # 3. Stratified train/test split — stratification preserves the class
    #    ratio in both sets so evaluation reflects real-world distribution.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

    # Compute class weight ratio for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Baseline: Logistic Regression ──────────────────────────────────────
    print("\n▶ Training Logistic Regression baseline ...")
    lr_pipeline = build_logistic_pipeline()
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate("Logistic Regression", lr_pipeline, X_test, y_test)

    # ── Primary: XGBoost with cross-validated hyperparameter tuning ────────
    print("\n▶ Tuning XGBoost with cross-validation ...")
    xgb_pipeline = build_xgboost_pipeline(scale_pos_weight=scale_pos_weight)

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
        "classifier__min_child_weight": [1, 3],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        xgb_pipeline,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    print(f"\n  Best CV ROC-AUC : {grid_search.best_score_:.4f}")
    print(f"  Best params     : {grid_search.best_params_}")

    xgb_metrics_default = evaluate("XGBoost (default threshold)", best_xgb, X_test, y_test)

    # ── Threshold optimization ─────────────────────────────────────────────
    print("\n▶ Optimizing decision threshold ...")
    optimal_threshold = find_optimal_threshold(best_xgb, X_test, y_test)
    xgb_metrics_tuned = evaluate("XGBoost (optimized threshold)", best_xgb, X_test, y_test,
                                  threshold=optimal_threshold)
    plot_threshold_analysis(best_xgb, X_test, y_test, optimal_threshold)

    # ── ROC curve comparison ───────────────────────────────────────────────
    print("\n▶ Generating ROC curve comparison ...")
    plot_roc_curves(
        {"Logistic Regression": lr_pipeline, "XGBoost (Tuned)": best_xgb},
        X_test, y_test,
    )

    # ── SHAP analysis ─────────────────────────────────────────────────────
    print("\n▶ Running SHAP feature importance analysis ...")
    generate_shap_analysis(best_xgb, X_test)

    # ── Select and persist the best model ──────────────────────────────────
    best_pipeline = best_xgb
    best_name = "XGBoost (Tuned)"
    best_metrics = xgb_metrics_tuned

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\n✓ Best model ({best_name}) saved → {MODEL_PATH}")

    # Save optimal threshold alongside the model
    threshold_path = MODEL_DIR / "optimal_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({"optimal_threshold": round(float(optimal_threshold), 4)}, f, indent=2)
    print(f"✓ Optimal threshold saved → {threshold_path}")

    # Save metrics for the README / CI
    all_metrics = {
        "logistic_regression": lr_metrics,
        "xgboost_default_threshold": xgb_metrics_default,
        "xgboost_optimized_threshold": xgb_metrics_tuned,
        "best_model": best_name,
        "optimal_threshold": round(float(optimal_threshold), 4),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Metrics saved → {METRICS_PATH}")

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  {'Model':<35s} {'ROC-AUC':>8s} {'F1':>8s} {'PR-AUC':>8s}")
    print(f"  {'─' * 59}")
    print(f"  {'Logistic Regression':<35s} {lr_metrics['roc_auc']:>8.4f} {lr_metrics['f1_score']:>8.4f} {lr_metrics['pr_auc']:>8.4f}")
    print(f"  {'XGBoost (threshold=0.50)':<35s} {xgb_metrics_default['roc_auc']:>8.4f} {xgb_metrics_default['f1_score']:>8.4f} {xgb_metrics_default['pr_auc']:>8.4f}")
    print(f"  {'XGBoost (threshold=' + str(round(optimal_threshold,2)) + ')':<35s} {xgb_metrics_tuned['roc_auc']:>8.4f} {xgb_metrics_tuned['f1_score']:>8.4f} {xgb_metrics_tuned['pr_auc']:>8.4f}")
    print()


if __name__ == "__main__":
    main()
