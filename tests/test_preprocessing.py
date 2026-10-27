"""
Automated Unit Tests for Preprocessing and Scaler Logic.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor, split_data


@pytest.fixture
def sample_feature_matrix():
    np.random.seed(42)
    n = 100
    data = {
        "Time": np.random.uniform(0, 10000, n),
        "Amount": np.random.exponential(50, n)
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n)
    return pd.DataFrame(data)


def test_preprocessor_fit_transform(sample_feature_matrix):
    preprocessor = DataPreprocessor(scale_columns=["Time", "Amount"])
    assert not preprocessor.is_fitted

    scaled_train = preprocessor.fit_transform_train(sample_feature_matrix)
    assert preprocessor.is_fitted
    assert scaled_train.shape == sample_feature_matrix.shape

    # RobustScaler centers around median (scaled median should be ~0)
    assert np.isclose(scaled_train["Amount"].median(), 0.0, atol=1e-5)
    assert np.isclose(scaled_train["Time"].median(), 0.0, atol=1e-5)


def test_preprocessor_transform_unseen(sample_feature_matrix):
    preprocessor = DataPreprocessor(scale_columns=["Time", "Amount"])
    preprocessor.fit_transform_train(sample_feature_matrix)

    new_data = sample_feature_matrix.iloc[:10].copy()
    scaled_new = preprocessor.transform(new_data)
    assert scaled_new.shape == (10, 30)


def test_preprocessor_missing_feature_error(sample_feature_matrix):
    preprocessor = DataPreprocessor(scale_columns=["Time", "Amount"])
    preprocessor.fit_transform_train(sample_feature_matrix)

    incomplete_data = sample_feature_matrix.drop(columns=["V14"])
    with pytest.raises(ValueError, match="Missing required feature"):
        preprocessor.transform(incomplete_data)


def test_split_data_preserves_class_ratio():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(np.random.randn(n, 5), columns=[f"F{i}" for i in range(5)])
    # 10% fraud
    y = pd.Series([1 if i < 20 else 0 for i in range(n)])

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
    assert len(X_train) == 160
    assert len(X_test) == 40
    # Stratification check
    assert y_train.mean() == pytest.approx(0.10, abs=0.02)
    assert y_test.mean() == pytest.approx(0.10, abs=0.02)
