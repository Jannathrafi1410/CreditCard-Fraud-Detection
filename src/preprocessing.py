"""
Data Preprocessing and Feature Engineering Pipeline for Credit Card Fraud Detection.
Implements leak-free scaling (RobustScaler), stratified splitting, and EDA figure generation.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict, Any, Optional, List


class DataPreprocessor:
    """
    Manages robust scaling, feature alignment, and leak-free transformations.
    """
    def __init__(self, scale_columns: Optional[List[str]] = None, random_state: int = 42):
        self.scale_columns = scale_columns or ["Time", "Amount"]
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []

    def fit_transform_train(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fits scaler strictly on training split and transforms numerical columns.
        """
        self.feature_names = list(X_train.columns)
        X_scaled = X_train.copy()

        cols_to_scale = [col for col in self.scale_columns if col in X_scaled.columns]
        if cols_to_scale:
            X_scaled[cols_to_scale] = self.scaler.fit_transform(X_scaled[cols_to_scale])
        
        self.is_fitted = True
        return X_scaled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms test or new inference records using the fitted scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit_transform_train first or load_scaler.")

        X_scaled = X.copy()
        # Ensure column presence and order
        for col in self.feature_names:
            if col not in X_scaled.columns:
                raise ValueError(f"Missing required feature: '{col}' in input record.")

        cols_to_scale = [col for col in self.scale_columns if col in X_scaled.columns]
        if cols_to_scale:
            X_scaled[cols_to_scale] = self.scaler.transform(X_scaled[cols_to_scale])

        # Return aligned columns
        return X_scaled[self.feature_names]

    def save(self, filepath: str) -> None:
        """Saves fitted preprocessor artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[Preprocessor] Saved preprocessor artifact to '{filepath}'")

    @classmethod
    def load(cls, filepath: str) -> "DataPreprocessor":
        """Loads fitted preprocessor artifact from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor artifact not found at: {filepath}")
        return joblib.load(filepath)


def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train-test splitting to preserve the minority fraud proportion.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Preprocessing] Split data into Train ({len(X_train):,} rows) and Test ({len(X_test):,} rows).")
    print(f"[Preprocessing] Train Fraud Rate: {y_train.mean()*100:.4f}% | Test Fraud Rate: {y_test.mean()*100:.4f}%")
    return X_train, X_test, y_train, y_test


def generate_eda_figures(df: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    """
    Generates and saves exploratory data analysis charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(7, 5), dpi=300)
    class_counts = df["Class"].value_counts()
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette=["#3b82f6", "#ef4444"])
    plt.title("Credit Card Transaction Class Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Transaction Class (0: Legitimate, 1: Fraudulent)", fontsize=11)
    plt.ylabel("Number of Transactions (Log Scale)", fontsize=11)
    plt.yscale("log")
    for i, count in enumerate(class_counts.values):
        pct = (count / len(df)) * 100
        ax.text(i, count * 1.2, f"{count:,}\n({pct:.3f}%)", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()

    # 2. Transaction Amount Distribution
    plt.figure(figsize=(8, 5), dpi=300)
    sns.histplot(df["Amount"], bins=50, color="#2563eb", kde=True)
    plt.title("Distribution of Transaction Amounts", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Transaction Amount ($/€)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "amount_distribution.png"))
    plt.close()

    # 3. Transaction Amount by Class Boxplot
    plt.figure(figsize=(7, 5), dpi=300)
    sns.boxplot(x="Class", y="Amount", data=df, palette=["#60a5fa", "#f87171"], showfliers=False)
    plt.title("Transaction Amount Distribution by Class (Excl. Outliers)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks([0, 1], ["Legitimate (0)", "Fraudulent (1)"])
    plt.xlabel("Transaction Class", fontsize=11)
    plt.ylabel("Transaction Amount ($/€)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "amount_by_class_boxplot.png"))
    plt.close()

    # 4. Transaction Time Density
    plt.figure(figsize=(9, 5), dpi=300)
    sns.kdeplot(data=df[df["Class"] == 0]["Time"], label="Legitimate (0)", color="#2563eb", fill=True, alpha=0.3)
    sns.kdeplot(data=df[df["Class"] == 1]["Time"], label="Fraudulent (1)", color="#dc2626", fill=True, alpha=0.4)
    plt.title("Transaction Density Across Elapsed Time (Seconds)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Time Elapsed (seconds since first transaction)", fontsize=11)
    plt.ylabel("Kernel Density", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transaction_time_density.png"))
    plt.close()

    # 5. Correlation Heatmap for Top Features
    plt.figure(figsize=(10, 8), dpi=300)
    # Correlation with Class
    corr = df.corr()
    top_corr_features = corr["Class"].abs().sort_values(ascending=False).head(15).index
    sub_corr = df[top_corr_features].corr()
    sns.heatmap(sub_corr, cmap="coolwarm", annot=True, fmt=".2f", vmin=-1, vmax=1, cbar_kws={"label": "Pearson Correlation"})
    plt.title("Correlation Heatmap: Top 15 Features Correlated with Fraud Class", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    print(f"[Preprocessing] Generated 5 EDA figures in '{output_dir}'.")


if __name__ == "__main__":
    from src.data_loader import load_data, get_feature_target_split
    test_path = "data/raw/creditcard.csv"
    if os.path.exists(test_path):
        data = load_data(test_path)
        generate_eda_figures(data)
        X, y = get_feature_target_split(data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        preprocessor = DataPreprocessor()
        X_train_scaled = preprocessor.fit_transform_train(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        preprocessor.save("models/scaler.joblib")
