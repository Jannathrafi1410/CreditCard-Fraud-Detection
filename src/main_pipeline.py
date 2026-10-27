"""
Master End-to-End Orchestrator for Credit Card Fraud Detection.
Executes data ingestion, EDA, preprocessing, model training, evaluation,
and Apache Spark distributed analytics in sequence with full progress logging.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
from src.data_loader import load_data, detect_target_column, inspect_dataset, get_feature_target_split
from src.preprocessing import split_data, DataPreprocessor, generate_eda_figures
from src.train_model import train_random_forest, train_baseline_model, save_model_artifacts
from src.evaluate_model import evaluate_classifier, plot_confusion_matrix, plot_evaluation_curves, plot_feature_importance
from src.spark_pipeline import run_spark_fraud_pipeline
from src.hdfs_utils import HDFSManager


def run_full_pipeline(dataset_path: str = "data/raw/creditcard.csv"):
    start_total = time.time()
    print("=" * 80)
    print("  CREDIT CARD FRAUD DETECTION: END-TO-END SYSTEM EXECUTION")
    print("=" * 80)

    # 1. Verification of Dataset
    if not os.path.exists(dataset_path):
        sample_path = "data/sample/sample_transactions.csv"
        if os.path.exists(sample_path):
            print(f"[Warning] Full dataset '{dataset_path}' not found. Falling back to sample '{sample_path}'.")
            dataset_path = sample_path
        else:
            raise FileNotFoundError(f"Neither '{dataset_path}' nor sample file exists.")

    # 2. Data Ingestion & Technical Inspection
    print("\n[Step 1/6] Ingesting and Inspecting Dataset...")
    df = load_data(dataset_path)
    target_col = detect_target_column(df)
    summary_stats = inspect_dataset(df, target_col)

    print("\nDataset Summary Statistics:")
    print(f"  Total Records: {summary_stats['total_rows']:,}")
    print(f"  Features: {summary_stats['total_columns']}")
    print(f"  Fraud Count: {summary_stats['fraud_count']:,} ({summary_stats['fraud_percentage']}%)")
    print(f"  Legitimate Count: {summary_stats['legit_count']:,} ({summary_stats['legit_percentage']}%)")
    print(f"  Duplicate Records: {summary_stats['duplicate_rows']:,} ({summary_stats['duplicate_percentage']}%)")
    print(f"  Missing Values: {summary_stats['total_missing_values']}")

    # Save summary stats to reports/metrics/dataset_inspection.json
    os.makedirs("reports/metrics", exist_ok=True)
    with open("reports/metrics/dataset_inspection.json", "w") as f:
        json.dump(summary_stats, f, indent=4)
    print("Saved inspection summary to 'reports/metrics/dataset_inspection.json'")

    # 3. Exploratory Data Analysis Figure Generation
    print("\n[Step 2/6] Generating Exploratory Data Analysis Figures...")
    generate_eda_figures(df, "reports/figures")

    # 4. Data Preprocessing & Train-Test Split
    print("\n[Step 3/6] Preprocessing and Train/Test Splitting...")
    X, y = get_feature_target_split(df, target_col)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)

    preprocessor = DataPreprocessor(scale_columns=["Time", "Amount"], random_state=42)
    X_train_scaled = preprocessor.fit_transform_train(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    preprocessor.save("models/scaler.joblib")

    # 5. Model Training (Random Forest & Logistic Regression Baseline)
    print("\n[Step 4/6] Training Random Forest and Baseline Classifiers...")
    rf_model, rf_duration = train_random_forest(
        X_train_scaled, y_train,
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
    baseline_lr, lr_duration = train_baseline_model(X_train_scaled, y_train, random_state=42)

    save_model_artifacts(
        rf_model,
        "models/random_forest_model.joblib",
        list(X_train.columns),
        {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42
        },
        rf_duration,
        baseline_model=baseline_lr,
        baseline_path="models/baseline_logistic_regression.joblib"
    )

    # 6. Model Evaluation on Unseen Test Split
    print("\n[Step 5/6] Evaluating Models on Held-Out Test Set...")
    metrics_rf, y_proba_rf, y_pred_rf = evaluate_classifier(rf_model, X_test_scaled, y_test, threshold=0.50)
    metrics_bl, y_proba_bl, _ = evaluate_classifier(baseline_lr, X_test_scaled, y_test, threshold=0.50)
    metrics_rf["baseline_comparison"] = metrics_bl

    print("\n" + "="*50)
    print("  MEASURED TEST SET EVALUATION METRICS")
    print("="*50)
    print(f"  Accuracy:    {metrics_rf['accuracy']*100:.3f}%")
    print(f"  Precision:   {metrics_rf['precision']*100:.3f}%")
    print(f"  Recall:      {metrics_rf['recall']*100:.3f}%")
    print(f"  F1-Score:    {metrics_rf['f1_score']:.4f}")
    print(f"  Specificity: {metrics_rf['specificity']*100:.3f}%")
    print(f"  ROC-AUC:     {metrics_rf['roc_auc']:.4f}")
    print(f"  PR-AUC:      {metrics_rf['pr_auc']:.4f}")
    cm = metrics_rf["confusion_matrix"]
    print(f"  Confusion Matrix: TP={cm['true_positives']}, FP={cm['false_positives']}, TN={cm['true_negatives']}, FN={cm['false_negatives']}")
    print(f"  Baseline Logistic Regression F1: {metrics_bl['f1_score']:.4f} | Recall: {metrics_bl['recall']*100:.2f}% | Precision: {metrics_bl['precision']*100:.2f}%")
    print("="*50)

    # Save metrics
    with open("reports/metrics/evaluation_metrics.json", "w") as f:
        json.dump(metrics_rf, f, indent=4)
    with open("reports/metrics/confusion_matrix.json", "w") as f:
        json.dump(metrics_rf["confusion_matrix"], f, indent=4)

    # Plot evaluation diagnostics
    plot_confusion_matrix(metrics_rf["confusion_matrix"], "reports/figures/confusion_matrix.png")
    plot_evaluation_curves(y_test, y_proba_rf, y_proba_bl, "reports/figures")
    plot_feature_importance(rf_model, list(X_test.columns), top_n=15, output_path="reports/figures/feature_importance.png")

    # 7. Big Data Pipeline & HDFS Simulation
    print("\n[Step 6/6] Executing Big Data PySpark & HDFS Workflow...")
    hdfs = HDFSManager()
    hdfs.mkdir("/fraud_detection/input")
    hdfs.mkdir("/fraud_detection/output")
    hdfs.put(dataset_path, "/fraud_detection/input")

    try:
        spark_results = run_spark_fraud_pipeline(dataset_path, "data/processed/spark_aggregated_metrics")
        print("PySpark distributed analytics finished successfully!")
    except Exception as e:
        print(f"[Spark Warning] Spark execution encountered error: {e}. Proceeding with pipeline.")

    total_time = time.time() - start_total
    print("\n" + "="*80)
    print(f"  PIPELINE COMPLETE! Total Elapsed Time: {total_time:.2f} seconds.")
    print("="*80)


if __name__ == "__main__":
    run_full_pipeline("data/raw/creditcard.csv")
