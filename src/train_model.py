"""
Machine Learning Training Pipeline for Credit Card Fraud Detection.
Trains a cost-sensitive Random Forest Classifier with balanced class weights
alongside a baseline classifier for academic performance benchmarking.
"""

import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from typing import Dict, Any, Tuple, Optional


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 15,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    class_weight: str = "balanced"
) -> Tuple[RandomForestClassifier, float]:
    """
    Trains cost-sensitive Random Forest Classifier on scaled training set.
    """
    print(f"[TrainModel] Initializing Random Forest with n_estimators={n_estimators}, max_depth={max_depth}, class_weight='{class_weight}'...")
    start_time = time.time()

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    rf.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"[TrainModel] Random Forest training completed in {elapsed:.2f} seconds.")
    return rf, elapsed


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[LogisticRegression, float]:
    """
    Trains a Logistic Regression baseline with balanced weights for comparative benchmarking.
    """
    print("[TrainModel] Training Logistic Regression baseline...")
    start_time = time.time()
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state
    )
    lr.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"[TrainModel] Baseline model trained in {elapsed:.2f} seconds.")
    return lr, elapsed


def save_model_artifacts(
    model: Any,
    model_path: str,
    feature_names: list,
    hyperparameters: Dict[str, Any],
    training_time: float,
    baseline_model: Optional[Any] = None,
    baseline_path: Optional[str] = None
) -> None:
    """
    Saves trained model objects and metadata JSON.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[TrainModel] Saved primary model artifact to '{model_path}'")

    if baseline_model is not None and baseline_path is not None:
        joblib.dump(baseline_model, baseline_path)
        print(f"[TrainModel] Saved baseline model artifact to '{baseline_path}'")

    metadata = {
        "project": "Credit Card Fraud Detection",
        "algorithm": type(model).__name__,
        "training_timestamp": datetime.now().isoformat(),
        "training_duration_seconds": round(training_time, 2),
        "hyperparameters": hyperparameters,
        "features": feature_names,
        "n_features": len(feature_names),
        "class_balance_strategy": "Cost-Sensitive Loss Weighting (class_weight='balanced')"
    }

    meta_path = os.path.join(os.path.dirname(model_path), "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[TrainModel] Saved metadata to '{meta_path}'")


if __name__ == "__main__":
    from src.data_loader import load_data, get_feature_target_split
    from src.preprocessing import split_data, DataPreprocessor

    dataset_path = "data/raw/creditcard.csv"
    if os.path.exists(dataset_path):
        df = load_data(dataset_path)
        X, y = get_feature_target_split(df)
        X_train, X_test, y_train, y_test = split_data(X, y)

        preprocessor = DataPreprocessor()
        X_train_scaled = preprocessor.fit_transform_train(X_train)
        preprocessor.save("models/scaler.joblib")

        rf_model, duration = train_random_forest(X_train_scaled, y_train)
        baseline_lr, lr_duration = train_baseline_model(X_train_scaled, y_train)

        params = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42
        }
        save_model_artifacts(
            rf_model,
            "models/random_forest_model.joblib",
            list(X_train.columns),
            params,
            duration,
            baseline_model=baseline_lr,
            baseline_path="models/baseline_logistic_regression.joblib"
        )
