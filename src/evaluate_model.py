"""
Model Evaluation and Diagnostic Reporting Module for Credit Card Fraud Detection.
Computes Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC, Confusion Matrix,
and produces high-resolution diagnostic charts.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any, Tuple, Optional


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on unseen test data using probability predictions.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-scores))
    else:
        y_proba = model.predict(X_test)

    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix elements: tn, fp, fn, tp
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except Exception:
        pr_auc = 0.0

    metrics = {
        "accuracy": round(float(accuracy), 5),
        "precision": round(float(precision), 5),
        "recall": round(float(recall), 5),
        "f1_score": round(float(f1), 5),
        "specificity": round(float(specificity), 5),
        "roc_auc": round(float(roc_auc), 5),
        "pr_auc": round(float(pr_auc), 5),
        "decision_threshold": round(float(threshold), 3),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "sample_counts": {
            "total_test_records": int(len(y_test)),
            "actual_fraud_records": int(y_test.sum()),
            "actual_legit_records": int((y_test == 0).sum())
        }
    }
    return metrics, y_proba, y_pred


def plot_confusion_matrix(cm_dict: Dict[str, int], output_path: str = "reports/figures/confusion_matrix.png") -> None:
    """Renders formatted confusion matrix heatmap with TP, TN, FP, FN annotations."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tn = cm_dict["true_negatives"]
    fp = cm_dict["false_positives"]
    fn = cm_dict["false_negatives"]
    tp = cm_dict["true_positives"]

    cm_matrix = np.array([[tn, fp], [fn, tp]])
    labels = [
        [f"TN\n{tn:,}\n(Legit Correct)", f"FP\n{fp:,}\n(False Alarm)"],
        [f"FN\n{fn:,}\n(Fraud Missed)", f"TP\n{tp:,}\n(Fraud Caught)"]
    ]

    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(cm_matrix, annot=labels, fmt="", cmap="Blues", cbar=True,
                xticklabels=["Predicted Legit (0)", "Predicted Fraud (1)"],
                yticklabels=["Actual Legit (0)", "Actual Fraud (1)"],
                annot_kws={"fontsize": 11, "fontweight": "bold"})
    plt.title("Random Forest Confusion Matrix on Test Set", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Ground Truth Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix to '{output_path}'")


def plot_evaluation_curves(
    y_test: pd.Series,
    y_proba_rf: np.ndarray,
    y_proba_baseline: Optional[np.ndarray] = None,
    output_dir: str = "reports/figures"
) -> None:
    """Generates ROC and Precision-Recall Curves comparing RF to baseline."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. ROC Curve
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)

    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(fpr_rf, tpr_rf, color="#2563eb", lw=2.2, label=f"Random Forest (AUC = {auc_rf:.4f})")

    if y_proba_baseline is not None:
        fpr_bl, tpr_bl, _ = roc_curve(y_test, y_proba_baseline)
        auc_bl = roc_auc_score(y_test, y_proba_baseline)
        plt.plot(fpr_bl, tpr_bl, color="#9ca3af", lw=1.8, linestyle="--", label=f"Baseline Logistic Regression (AUC = {auc_bl:.4f})")

    plt.plot([0, 1], [0, 1], color="#dc2626", linestyle=":", lw=1.5, label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    # 2. Precision-Recall Curve
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_proba_rf)
    ap_rf = average_precision_score(y_test, y_proba_rf)

    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(rec_rf, prec_rf, color="#16a34a", lw=2.2, label=f"Random Forest (PR-AUC = {ap_rf:.4f})")

    if y_proba_baseline is not None:
        prec_bl, rec_bl, _ = precision_recall_curve(y_test, y_proba_baseline)
        ap_bl = average_precision_score(y_test, y_proba_baseline)
        plt.plot(rec_bl, prec_bl, color="#9ca3af", lw=1.8, linestyle="--", label=f"Baseline Logistic Regression (PR-AUC = {ap_bl:.4f})")

    baseline_rate = y_test.mean()
    plt.plot([0, 1], [baseline_rate, baseline_rate], color="#dc2626", linestyle=":", lw=1.5, label=f"No-Skill Rate ({baseline_rate*100:.3f}%)")
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall (PR) Curve for Imbalanced Data", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()

    print(f"[Evaluation] Saved ROC and PR Curves to '{output_dir}'")


def plot_feature_importance(
    model: Any,
    feature_names: list,
    top_n: int = 15,
    output_path: str = "reports/figures/feature_importance.png"
) -> Dict[str, float]:
    """Generates feature importance ranking bar chart."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not hasattr(model, "feature_importances_"):
        print("[Evaluation] Model does not have feature_importances_ attribute.")
        return {}

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(9, 6), dpi=300)
    top_feats = feat_imp.head(top_n)
    sns.barplot(x=top_feats.values, y=top_feats.index, palette="viridis")
    plt.title(f"Top {top_n} Most Predictive Features (Random Forest Importance)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Gini Feature Importance Score", fontsize=11)
    plt.ylabel("Transaction Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Evaluation] Saved feature importance chart to '{output_path}'")

    return {k: round(float(v), 5) for k, v in feat_imp.items()}


if __name__ == "__main__":
    from src.data_loader import load_data, get_feature_target_split
    from src.preprocessing import split_data, DataPreprocessor

    if os.path.exists("models/random_forest_model.joblib") and os.path.exists("data/raw/creditcard.csv"):
        df = load_data("data/raw/creditcard.csv")
        X, y = get_feature_target_split(df)
        _, X_test, _, y_test = split_data(X, y)

        preprocessor = DataPreprocessor.load("models/scaler.joblib")
        X_test_scaled = preprocessor.transform(X_test)

        rf = joblib.load("models/random_forest_model.joblib")
        lr = joblib.load("models/baseline_logistic_regression.joblib") if os.path.exists("models/baseline_logistic_regression.joblib") else None

        metrics_rf, y_proba_rf, _ = evaluate_classifier(rf, X_test_scaled, y_test)
        y_proba_lr = None
        if lr is not None:
            metrics_lr, y_proba_lr, _ = evaluate_classifier(lr, X_test_scaled, y_test)
            metrics_rf["baseline_comparison"] = metrics_lr

        print("\n--- Model Evaluation Results ---")
        for k, v in metrics_rf.items():
            if k != "baseline_comparison":
                print(f"  {k}: {v}")

        # Save metrics to JSON
        os.makedirs("reports/metrics", exist_ok=True)
        with open("reports/metrics/evaluation_metrics.json", "w") as f:
            json.dump(metrics_rf, f, indent=4)

        with open("reports/metrics/confusion_matrix.json", "w") as f:
            json.dump(metrics_rf["confusion_matrix"], f, indent=4)

        plot_confusion_matrix(metrics_rf["confusion_matrix"])
        plot_evaluation_curves(y_test, y_proba_rf, y_proba_lr)
        plot_feature_importance(rf, list(X_test.columns))
