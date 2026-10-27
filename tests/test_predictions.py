"""
Automated Unit Tests for Model Inference and Prediction Engine.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.predict import FraudPredictor


@pytest.fixture
def predictor_instance():
    model_path = "models/random_forest_model.joblib"
    scaler_path = "models/scaler.joblib"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        pytest.skip("Trained model or scaler artifact not available for testing.")
    return FraudPredictor(model_path=model_path, preprocessor_path=scaler_path)


@pytest.fixture
def sample_transaction_record():
    record = {"Time": 50000.0, "Amount": 120.0}
    for i in range(1, 29):
        record[f"V{i}"] = 0.05
    return record


def test_predict_single_structure(predictor_instance, sample_transaction_record):
    res = predictor_instance.predict_single(sample_transaction_record)
    assert "prediction" in res
    assert res["prediction"] in (0, 1)
    assert "fraud_probability" in res
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert "risk_tier" in res
    assert "operational_verdict" in res
    assert "disclaimer" in res


def test_predict_single_threshold_tuning(predictor_instance, sample_transaction_record):
    # With a high threshold, low-risk transaction should be 0
    res_high = predictor_instance.predict_single(sample_transaction_record, threshold=0.99)
    assert res_high["prediction"] == 0


def test_predict_batch_structure(predictor_instance, sample_transaction_record):
    # Create 5 identical records
    batch_df = pd.DataFrame([sample_transaction_record for _ in range(5)])
    scored = predictor_instance.predict_batch(batch_df)

    assert len(scored) == 5
    assert "Fraud_Probability" in scored.columns
    assert "Predicted_Class" in scored.columns
    assert "Risk_Tier" in scored.columns
    assert "Prediction_Label" in scored.columns
    assert set(scored["Predicted_Class"].unique()).issubset({0, 1})


def test_predict_batch_missing_column(predictor_instance, sample_transaction_record):
    bad_record = sample_transaction_record.copy()
    del bad_record["V1"]
    bad_batch = pd.DataFrame([bad_record])

    with pytest.raises(ValueError, match="Batch dataset missing required features"):
        predictor_instance.predict_batch(bad_batch)
