"""
Data Ingestion and Schema Inspection Module for Credit Card Fraud Detection.
Supports full and chunked loading, automated target detection, and schema validation.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List


def load_data(filepath: str, chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    Loads transaction dataset from CSV. If dataset exceeds memory, supports chunked loading.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[DataLoader] Ingesting dataset from '{filepath}' ({file_size_mb:.2f} MB)...")

    if chunksize is not None and chunksize > 0:
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(filepath)

    print(f"[DataLoader] Successfully loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Detects the ground-truth fraud target column without making hardcoded assumptions.
    Inspects candidate names such as 'Class', 'is_fraud', 'fraud', 'isFraud', 'target'.
    """
    candidates = ["Class", "is_fraud", "fraud", "isFraud", "target", "label"]
    for col in df.columns:
        if col in candidates or col.lower() in [c.lower() for c in candidates]:
            # Verify binary values
            unique_vals = sorted(df[col].dropna().unique().tolist())
            if unique_vals in ([0, 1], [0.0, 1.0], ["0", "1"], [False, True]):
                print(f"[DataLoader] Detected binary target column: '{col}' with classes: {unique_vals}")
                return col

    # Fallback search if standard candidates not found
    for col in df.columns:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            print(f"[DataLoader] Inferred binary target column: '{col}'")
            return col

    raise ValueError("Could not automatically identify a binary fraud target column. Columns present: " + str(list(df.columns)))


def inspect_dataset(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs comprehensive technical inspection of the transaction dataset.
    """
    if target_col is None:
        target_col = detect_target_column(df)

    total_rows = len(df)
    total_cols = len(df.columns)
    column_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_values = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    val_counts = df[target_col].value_counts().to_dict()
    fraud_count = int(val_counts.get(1, 0))
    legit_count = int(val_counts.get(0, 0))
    fraud_pct = (fraud_count / total_rows * 100) if total_rows > 0 else 0.0
    legit_pct = (legit_count / total_rows * 100) if total_rows > 0 else 0.0

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    has_time = "Time" in df.columns
    has_amount = "Amount" in df.columns
    pca_features = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]

    stats = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "target_column": target_col,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_percentage": round(fraud_pct, 4),
        "legit_percentage": round(legit_pct, 4),
        "total_missing_values": total_missing,
        "missing_by_column": missing_values,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round((duplicate_rows / total_rows * 100), 3) if total_rows > 0 else 0,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "has_timestamp": has_time,
        "has_amount": has_amount,
        "pca_feature_count": len(pca_features),
        "column_types": column_dtypes
    }
    return stats


def get_feature_target_split(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates feature matrix X and target vector y.
    """
    if target_col is None:
        target_col = detect_target_column(df)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


if __name__ == "__main__":
    test_path = "data/raw/creditcard.csv"
    if os.path.exists(test_path):
        data = load_data(test_path)
        summary = inspect_dataset(data)
        print("\n--- Technical Inspection Summary ---")
        for k, v in summary.items():
            if k not in ["missing_by_column", "column_types", "numerical_columns"]:
                print(f"{k}: {v}")
