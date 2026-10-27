"""
Automated Unit Tests for Data Ingestion and Target Detection.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_data, detect_target_column, inspect_dataset, get_feature_target_split


@pytest.fixture
def synthetic_transaction_data():
    """Provides small controlled synthetic fixture for testing data ingestion."""
    np.random.seed(42)
    n_samples = 50
    data = {
        "Time": np.linspace(0, 1000, n_samples),
        "Amount": np.random.uniform(5, 500, n_samples),
        "Class": [1 if i % 10 == 0 else 0 for i in range(n_samples)]
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n_samples)
    return pd.DataFrame(data)


def test_detect_target_column_standard(synthetic_transaction_data):
    target = detect_target_column(synthetic_transaction_data)
    assert target == "Class"


def test_detect_target_column_alternative():
    df_alt = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0],
        "is_fraud": [0, 1, 0]
    })
    target = detect_target_column(df_alt)
    assert target == "is_fraud"


def test_detect_target_column_invalid():
    df_invalid = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0],
        "feature_2": [10.0, 20.0, 30.0]
    })
    with pytest.raises(ValueError):
        detect_target_column(df_invalid)


def test_inspect_dataset_metrics(synthetic_transaction_data):
    stats = inspect_dataset(synthetic_transaction_data)
    assert stats["total_rows"] == 50
    assert stats["total_columns"] == 31
    assert stats["fraud_count"] == 5
    assert stats["legit_count"] == 45
    assert stats["total_missing_values"] == 0
    assert stats["has_timestamp"] is True
    assert stats["has_amount"] is True


def test_get_feature_target_split(synthetic_transaction_data):
    X, y = get_feature_target_split(synthetic_transaction_data)
    assert "Class" not in X.columns
    assert len(X.columns) == 30
    assert len(y) == 50
    assert set(y.unique()) == {0, 1}


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_directory/non_existent_file.csv")
