"""
Inference and Transaction Scoring Engine for Credit Card Fraud Detection.
Supports single-record risk evaluation and high-throughput batch scoring.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.preprocessing import DataPreprocessor


class FraudPredictor:
    """
    Production-grade inference pipeline wrapping trained ensemble and preprocessor.
    """
    def __init__(
        self,
        model_path: str = "models/random_forest_model.joblib",
        preprocessor_path: str = "models/scaler.joblib",
        default_threshold: float = 0.50
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")

        self.model = joblib.load(model_path)
        self.preprocessor = DataPreprocessor.load(preprocessor_path)
        self.default_threshold = default_threshold
        self.expected_features = self.preprocessor.feature_names

    def predict_single(
        self,
        transaction_data: Dict[str, Any],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a single transaction record and produces an actionable risk assessment.
        """
        thresh = threshold if threshold is not None else self.default_threshold
        df_single = pd.DataFrame([transaction_data])

        # Validate feature completeness
        missing_feats = [f for f in self.expected_features if f not in df_single.columns]
        if missing_feats:
            raise ValueError(f"Transaction record is missing required features: {missing_feats}")

        # Scale features
        X_scaled = self.preprocessor.transform(df_single)

        # Predict probability
        proba = float(self.model.predict_proba(X_scaled)[0, 1])
        prediction = 1 if proba >= thresh else 0

        # Determine Risk Tier and Operational Guidance
        if proba >= 0.80:
            risk_tier = "Critical Risk"
            verdict = "Flagged for Immediate Human Review & Temporary Hold"
            color_badge = "red"
        elif proba >= thresh:
            risk_tier = "High Risk"
            verdict = "Flagged for Secondary Verification"
            color_badge = "orange"
        elif proba >= 0.20:
            risk_tier = "Moderate Risk"
            verdict = "Transaction Cleared - Continuous Behavioral Monitoring"
            color_badge = "yellow"
        else:
            risk_tier = "Low Risk"
            verdict = "Classified as Legitimate - Standard Settlement"
            color_badge = "green"

        result = {
            "prediction": prediction,
            "prediction_label": "Suspicious / Potential Fraud" if prediction == 1 else "Legitimate",
            "fraud_probability": round(proba, 5),
            "applied_threshold": round(thresh, 3),
            "risk_tier": risk_tier,
            "operational_verdict": verdict,
            "color_badge": color_badge,
            "disclaimer": (
                "Notice: This prediction is a probabilistic machine learning risk score. "
                "It represents a statistical recommendation and does not constitute a "
                "conclusive legal or financial fraud determination."
            )
        }
        return result

    def predict_batch(
        self,
        df_batch: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Scores a batch of transactions and appends probability and risk columns.
        """
        thresh = threshold if threshold is not None else self.default_threshold
        df_work = df_batch.copy()

        # If Class is present in batch, keep aside for verification
        actual_class = df_work["Class"] if "Class" in df_work.columns else None
        if "Class" in df_work.columns:
            df_features = df_work.drop(columns=["Class"])
        else:
            df_features = df_work

        # Validate feature presence
        missing_feats = [f for f in self.expected_features if f not in df_features.columns]
        if missing_feats:
            raise ValueError(f"Batch dataset missing required features: {missing_feats}")

        # Ensure correct column ordering
        df_features = df_features[self.expected_features]

        # Transform using preprocessor
        X_scaled = self.preprocessor.transform(df_features)

        # Predict probabilities and classes
        probas = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probas >= thresh).astype(int)

        results_df = df_batch.copy()
        results_df["Fraud_Probability"] = np.round(probas, 4)
        results_df["Predicted_Class"] = preds
        results_df["Prediction_Label"] = np.where(preds == 1, "Flagged Fraud", "Legitimate")
        
        # Risk tier assignment
        conditions = [
            results_df["Fraud_Probability"] >= 0.80,
            results_df["Fraud_Probability"] >= thresh,
            results_df["Fraud_Probability"] >= 0.20
        ]
        choices = ["Critical Risk", "High Risk", "Moderate Risk"]
        results_df["Risk_Tier"] = np.select(conditions, choices, default="Low Risk")

        return results_df


if __name__ == "__main__":
    test_batch_file = "data/sample/test_batch_transactions.csv"
    if os.path.exists("models/random_forest_model.joblib") and os.path.exists(test_batch_file):
        predictor = FraudPredictor()
        batch_data = pd.read_csv(test_batch_file)
        scored = predictor.predict_batch(batch_data)
        print("\n--- Batch Scoring Preview ---")
        print(scored[["Fraud_Probability", "Predicted_Class", "Risk_Tier"]].head(10))
